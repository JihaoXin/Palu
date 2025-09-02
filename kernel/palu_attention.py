from typing import Optional, Tuple
from typing import Callable
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

import torch
from torch import nn

from transformers.models.llama.modeling_llama import (
    Cache, apply_rotary_pos_emb, 
    LlamaAttention, LlamaConfig,
)

# Import HeadwiseLowRankModule from svd_linear.py instead of defining it here
from palu.model.modules.svd_linear import HeadwiseLowRankModule

from .abx_rope import abx as recompute_k_gemv


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]

    # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    if is_causal is None:
        is_causal = causal_mask is None and query.shape[2] > 1

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None


class LlamaPaluAttention(LlamaAttention):
    """
    Llama Attention with Low-Rank KV-Cache with Palu. This module inherits from
    `LlamaAttention` but change linear layer and add custom Triton kernel.
    """
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        # Initialize all base attributes
        super().__init__(config, layer_idx)
        
        # === 基础配置参数 from config ===
        self.num_heads = config.num_attention_heads # (32, Q heads)
        self.num_key_value_heads = config.num_key_value_heads # (8, K/V heads)
        self.hidden_size = config.hidden_size          # 模型隐藏维度，如 4096
        self.group_size = config.group_size            # 每组的头数，如 4 (head_group_size)
        self.num_groups = config.num_groups            # Palu的G-LDR每个Attention layer的组数，如 2 (8 KV heads // 4 = 2)

        # Get layer-specific ranks from head_wise_ranks
        head_wise_ranks = getattr(config, "head_wise_ranks", {})
        layer_prefix = f"model.layers.{layer_idx}.self_attn"
        k_proj_key = f"{layer_prefix}.k_proj"
        v_proj_key = f"{layer_prefix}.v_proj"
        self.rank_k_list = head_wise_ranks[k_proj_key]
        self.rank_v_list = head_wise_ranks[v_proj_key]
        self.fused_hidden_dim_o = sum(self.rank_v_list) * self.group_size # Calculate fused output dimension for O projection

        # Q: Standard linear (not compressed)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        
        # K/V: Low-rank modules
        self.k_proj = HeadwiseLowRankModule(
            self.rank_k_list, 
            self.hidden_size, 
            self.num_key_value_heads * self.head_dim, 
            bias=config.attention_bias,
        )
        self.v_proj = HeadwiseLowRankModule(
            self.rank_v_list, 
            self.hidden_size, 
            self.num_key_value_heads * self.head_dim, 
            bias=config.attention_bias,
        )
        
        # O: Output projection with fused dimension
        if config.v_fusion:
            self.o_proj = nn.Linear(self.fused_hidden_dim_o, self.hidden_size, bias=config.attention_bias)
        else:
            self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        
        # RoPE latent flag from config
        self.rope_latent = getattr(config, "rope_latent", False)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward with conditional RoPE in latent space support
        """
        if not hasattr(self, '_printed_once'):
            print("😄😄😄😄😄😄😄😄😄😄😄layer {}: Using PaluAttention. RoPE Latent: {}".format(self.layer_idx, self.rope_latent))
            self._printed_once = True
        if not hasattr(self, 'rotary_emb'):
            from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
            self.rotary_emb = LlamaRotaryEmbedding(config=self.config).to(hidden_states.device)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        # Q projection
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = self.rotary_emb(query_states, position_ids=position_ids)
        # K projection
        if self.rope_latent: # RoPE in latent space: RoPE(x@U)@V
            key_latents = self.k_proj.project_to_latent(hidden_states)  # x@U
            key_latents_rope = self._apply_rope_to_latents(key_latents, cos, sin)  # RoPE(x@U)
            query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin)
            key_states = self.k_proj.reconstruct(key_latents_rope).view(hidden_shape).transpose(1, 2)  # RoPE(x@U)@V
        else: # Standard RoPE(x@U@V)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # V projection (standard for now)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # Cache latent space instead of full states for memory efficiency
            key_latents = self.k_proj.project_to_latent(hidden_states)  # x@U: [batch, seq, total_latent_k]
            value_latents = self.v_proj.project_to_latent(hidden_states)  # x@U: [batch, seq, total_latent_v]
            # Reshape latents to 4D format for caching: [batch, groups, seq, group_rank]
            batch_size, seq_len = key_latents.shape[:2]
            key_latents_4d = key_latents.view(batch_size, seq_len, self.num_groups, -1).transpose(1, 2)  # [batch, groups, seq, group_rank_k]
            value_latents_4d = value_latents.view(batch_size, seq_len, self.num_groups, -1).transpose(1, 2)    # [batch, groups, seq, group_rank_v]
            # Cache the latent states - much more memory efficient!
            cached_key_latents_4d, cached_value_latents_4d = past_key_value.update(
                key_latents_4d, value_latents_4d, self.layer_idx, cache_kwargs
            )
            # Reshape back to 3D for reconstruction: [batch, seq, total_latent]
            cached_seq_len = cached_key_latents_4d.shape[2]
            cached_key_latents = cached_key_latents_4d.transpose(1, 2).reshape(batch_size, cached_seq_len, -1)
            cached_value_latents = cached_value_latents_4d.transpose(1, 2).reshape(batch_size, cached_seq_len, -1)
            # Reconstruct full states for attention computation
            key_states = self.k_proj.reconstruct(cached_key_latents).view(batch_size, cached_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = self.v_proj.reconstruct(cached_value_latents).view(batch_size, cached_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            # Apply RoPE to Q (standard)
            _ , key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids) 

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                import logging
                logger = logging.getLogger(__name__)
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    
    def _apply_rope_to_latents(self, latents: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to grouped latent states - simplified version
        
        Args:
            latents: [batch, seq_len, total_latent_dim] 
            cos, sin: RoPE embeddings [batch, seq_len, head_dim]
            
        Returns:
            latents_rope: [batch, seq_len, total_latent_dim]
        """
        batch_size, seq_len, total_latent_dim = latents.shape
        
        # For simplicity, apply RoPE only to the first head_dim dimensions of the latents
        # This is a reasonable approximation for the RoPE(x@U)@V approach
        rope_dim = min(total_latent_dim, self.head_dim)
        
        if rope_dim == total_latent_dim:
            # All dimensions get RoPE
            # Reshape to [batch, 1, seq_len, total_latent_dim] for apply_rotary_pos_emb
            latents_reshaped = latents.unsqueeze(1)  # [batch, 1, seq_len, total_latent_dim]
            cos_truncated = cos[:, :, :rope_dim]  # [batch, seq_len, rope_dim]
            sin_truncated = sin[:, :, :rope_dim]  # [batch, seq_len, rope_dim]
            
            latents_rope, _ = apply_rotary_pos_emb(latents_reshaped, latents_reshaped, cos_truncated, sin_truncated)
            return latents_rope.squeeze(1)
        else:
            # Split into RoPE part and non-RoPE part
            rope_part = latents[:, :, :rope_dim]  # [batch, seq_len, rope_dim]
            non_rope_part = latents[:, :, rope_dim:]  # [batch, seq_len, remaining_dim]
            
            # Apply RoPE to rope_part
            rope_part_reshaped = rope_part.unsqueeze(1)  # [batch, 1, seq_len, rope_dim]
            cos_truncated = cos[:, :, :rope_dim]  # [batch, seq_len, rope_dim]
            sin_truncated = sin[:, :, :rope_dim]  # [batch, seq_len, rope_dim]
            
            rope_part_rotated, _ = apply_rotary_pos_emb(rope_part_reshaped, rope_part_reshaped, cos_truncated, sin_truncated)
            rope_part_rotated = rope_part_rotated.squeeze(1)  # [batch, seq_len, rope_dim]
            
            # Concatenate back
            latents_rope = torch.cat([rope_part_rotated, non_rope_part], dim=-1)
            return latents_rope

    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_value: Optional[Cache] = None,
    #     output_attentions: bool = False,
    #     golden_kernel: bool = False,
    #     **kwargs,
    # ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    #     if "padding_mask" in kwargs:
    #         warnings.warn(
    #             "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
    #         )

    #     bsz, q_len, _ = hidden_states.size()

    #     query_states = self.q_proj(hidden_states)
    #     # key_states = self.k_proj(hidden_states)
    #     # value_states = self.v_proj(hidden_states)
    #     key_h_states = self.k_proj.project_to_latent(hidden_states)
    #     value_h_states = self.v_proj.project_to_latent(hidden_states)

    #     query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    #     # key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    #     # value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    #     key_h_states = key_h_states.view(bsz, q_len, self.num_groups, self.group_rank_k).transpose(1, 2)
    #     value_h_states = value_h_states.view(bsz, q_len, self.num_groups, self.group_rank_v).transpose(1, 2)

    #     # kv_seq_len = key_states.shape[-2]
    #     kv_seq_len = key_h_states.shape[-2]
    #     if past_key_value is not None:
    #         if self.layer_idx is None:
    #             raise ValueError(
    #                 f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
    #                 "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
    #                 "with a layer index."
    #             )
    #         kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        
    #     # cos, sin = self.rotary_emb(query_states, seq_len=kv_seq_len)
    #     # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    #     if past_key_value is not None:
    #         # cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
    #         # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    #         key_h_states, value_h_states = past_key_value.update(key_h_states, value_h_states, self.layer_idx)


    #     if q_len > 1:
    #         # Prompting
    #         # Recompute the key states
    #         key_h_states = key_h_states.transpose(1, 2).reshape(bsz, kv_seq_len, self.total_rank_k)
    #         key_states = self.k_proj.reconstruct(key_h_states)
    #         key_states = key_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    #         # Apply RoPE after recomputing the key states
    #         cos, sin = self.rotary_emb(query_states, seq_len=kv_seq_len)
    #         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    #         attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    #     else:
    #         # Generating (Apply our reconsturction kernel)
    #         # A: (num_heads, 1, head_dim)
    #         # B: (num_heads, rank_per_groups, head_dim)
    #         # X: (num_head_groups, seq_len, rank_per_groups)
    #         # TODO: Optimize RoPE & sqrt(head_dim) into kernel
    #         # TODO: Check if sin & cos are share among different blocks
    #         cos, sin = self.rotary_emb(query_states, seq_len=kv_seq_len)
    #         query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin, position_ids)
    #         A = query_states.squeeze(0)
    #         B = self.k_proj.B
    #         X = key_h_states.squeeze(0)
    #         attn_weights = recompute_k_gemv(A, B, X).unsqueeze(0) / math.sqrt(self.head_dim)

    #     # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    #     if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
    #         raise ValueError(
    #             f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
    #             f" {attn_weights.size()}"
    #         )

    #     if attention_mask is not None:
    #         if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
    #             raise ValueError(
    #                 f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
    #             )
    #         attn_weights = attn_weights + attention_mask


    #     # Upcast attention to fp32
    #     attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    #     attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

    #     # Original version
    #     # value_states = self.v_proj.reconstruct(value_h_states)
    #     # value_states = value_states.reshape(1, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    #     # attn_output = torch.matmul(attn_weights, value_states)

    #     # Fusion version
    #     # attn_weights: (bsz, num_groups, q_len * group_size, kv_seq_len)
    #     attn_h_weights = attn_weights.reshape(1, self.num_groups, q_len * self.group_size, kv_seq_len)
    #     attn_h_output = torch.matmul(attn_h_weights, value_h_states)
    #     # attn_h_output: (bsz, num_heads, q_len * group_size, group_rank)
    #     attn_output = attn_h_output.reshape(1, self.num_heads, q_len, self.group_rank_v)


    #     attn_output = attn_output.transpose(1, 2).contiguous()
    #     attn_output = attn_output.reshape(bsz, q_len, -1)

    #     attn_output = self.o_proj(attn_output)
        
        
    #     if not output_attentions:
    #         attn_weights = None

    #     return attn_output, attn_weights, past_key_value
    
    @staticmethod
    def from_attention(
        module: LlamaAttention,
        config: LlamaConfig,
        whiten: bool = False,
    ):
        """
        从标准的 LlamaAttention 创建 LlamaPaluAttention。
        Args:
            module: 原始的 LlamaAttention 模块
            config: 配置对象，包含 head_wise_ranks 等信息
        """
        # 创建新的 LlamaPaluAttention 实例 (会根据 config.head_wise_ranks 自动初始化)
        new_module = LlamaPaluAttention(config, module.layer_idx)
        
        # Q 投影始终复用原始的 Linear 层（不压缩）
        new_module.q_proj = module.q_proj
        
        # K/V 投影处理
        if whiten:
            new_module.k_proj = HeadwiseLowRankModule.from_linear_whiten(
                module.k_proj, 
                new_module.rank_k_list, 
            )
            new_module.v_proj = HeadwiseLowRankModule.from_linear_whiten(
                module.v_proj, 
                new_module.rank_v_list, 
            )
        else:
            new_module.k_proj = HeadwiseLowRankModule.from_linear(
                module.k_proj, 
                new_module.rank_k_list, 
            )
            new_module.v_proj = HeadwiseLowRankModule.from_linear(
                module.v_proj, 
                new_module.rank_v_list, 
            )

        # No fusion version
        if not config.v_fusion:
            new_module.o_proj = module.o_proj
            return new_module

        # Fusion version
        # new_module.v_proj = new_v_proj.VT

        # fuse v_proj.U into o_proj
        new_o_weight = torch.zeros(new_module.o_proj.weight.size())

        head_dim = module.head_dim
        num_groups = config.num_groups
        group_size = config.group_size
        group_rank = new_module.group_rank_v 

        total_dims_2, total_ranks, total_fused_dims = 0, 0, 0
        for i in range(num_groups):
            total_dims = 0
            for _ in range(group_size):
                new_o_weight[:, total_fused_dims:total_fused_dims + group_rank] = \
                    module.o_proj.weight[:, total_dims_2:total_dims_2 + head_dim] @ \
                    new_module.v_proj.U_list[i].weight[total_dims:total_dims + head_dim, :]
                total_dims += head_dim
                total_dims_2 += head_dim
                total_fused_dims += group_rank

            total_ranks += group_rank

        with torch.no_grad():
            new_module.o_proj.weight.copy_(new_o_weight)

        return new_module