import math
import warnings
from typing import Optional, Tuple

import torch
from torch import nn
from loguru import logger

from transformers.models.llama.modeling_llama import (
    Cache, apply_rotary_pos_emb, 
    LlamaAttention, LlamaConfig,
)

from .abx_rope import abx as recompute_k_gemv


class HeadwiseLowRankModule(nn.Module):
    """ Headwise Low-Rank module """
    def __init__(self, ranks, in_features, out_features, bias):
        super().__init__()

        self.ranks = ranks
        self.num_groups = len(ranks)
        self.in_features = in_features
        self.out_features = out_features
        self.group_dim = out_features // self.num_groups

        if (self.group_dim * self.num_groups) != self.out_features:
            raise ValueError(
                f"out_features must be divisible by num_groups (got `out_features`: {self.out_features}"
                f" and `num_groups`: {self.num_groups})."
            )

        self.VT = nn.Linear(in_features, sum(ranks), bias=False)

        # Create the list of linear layers first
        Us = []
        for r in ranks:
            linear_layer = nn.Linear(r, self.group_dim, bias=bias)
            nn.init.normal_(linear_layer.weight)
            Us.append(linear_layer)

        self.U_list = nn.ModuleList(Us)
    
    def forward(self, hidden_states: torch.Tensor):
        """ hidden_states: Tensor of shape (batch_size, seq_len, in_features) """
        assert hidden_states.dim() == 3, f"hidden_states should have 3 dimensions, got {hidden_states.dim()}"
        
        hidden_states = self.VT(hidden_states)

        # hidden_states: Tensor of shape (batch_size, seq_len, r1 + r2 + ... )
        outputs = []
        total_ranks = 0
        for i in range(self.num_groups):
            outputs.append(self.U_list[i](hidden_states[:, :, total_ranks: total_ranks+self.ranks[i]]))
            total_ranks += self.ranks[i]

        return torch.cat(outputs, dim=-1)

    def project_to_latent(self, hidden_states: torch.Tensor):
        """ hidden_states: Tensor of shape (batch_size, seq_len, in_features) """
        assert hidden_states.dim() == 3, f"hidden_states should have 3 dimensions, got {hidden_states.dim()}"

        hidden_states = self.VT(hidden_states)

        return hidden_states
    
    def reconstruct(self, hidden_states: torch.Tensor):
        """ hidden_states: Tensor of shape (batch_size, seq_len, sum(ranks)) """
        assert hidden_states.dim() == 3, f"hidden_states should have 3 dimensions, got {hidden_states.dim()}"

        outputs = []
        total_ranks = 0
        for i in range(self.num_groups):
            outputs.append(self.U_list[i](hidden_states[:, :, total_ranks: total_ranks+self.ranks[i]]))
            total_ranks += self.ranks[i]

        return torch.cat(outputs, dim=-1)
    
    @staticmethod
    def from_linear(
        old_module: nn.Linear,
        ranks: list,
        attn_module: LlamaAttention = None,
    ):   
        new_module = HeadwiseLowRankModule(ranks, old_module.in_features, old_module.out_features, bias=old_module.bias is not None)
        w = old_module.weight.data.reshape(len(ranks), -1, old_module.in_features).float()

        wl = []
        wr = []
        for i in range(len(ranks)):
            l, s, r = torch.linalg.svd(w[i], full_matrices=False)
            l = l[:, 0:ranks[i]]
            s = s[0:ranks[i]]
            r = r[0:ranks[i], :]
            l = l.mul(s)

            # l: (head_dim, rank), r: (rank, hidden_size)
            wl.append(l)
            wr.append(r)

        # load to U
        for i in range(len(ranks)):
            if new_module.U_list[i].weight.data.shape != wl[i].shape:
                raise ValueError(f"{new_module.U_list[i].weight.data.shape} != {wl[i].shape}")
            new_module.U_list[i].weight.data = wl[i].contiguous()
        
        # Create B matrix for kernel (num_heads, group_rank_k, head_dim)
        if attn_module is not None:
            # Expect ranks per group; require uniform ranks for kernel path
            if len(set(new_module.ranks)) != 1:
                # Fallback: skip kernel precompute when non-uniform
                pass
            else:
                group_rank_k = new_module.ranks[0]
                # Stack U^T for each group. U_i: (group_dim, rank)
                U_list_T = [x.weight.data.T for x in new_module.U_list]  # (rank, group_dim)
                b = torch.stack(U_list_T)  # (num_groups, rank, group_dim)
                # Compute kv_group_size from new_module.group_dim
                kv_group_size = new_module.group_dim // attn_module.head_dim
                # Reshape to (num_groups, rank, kv_group_size, head_dim)
                b = b.reshape(new_module.num_groups, group_rank_k, kv_group_size, attn_module.head_dim)
                # If attention has more heads than kv heads, repeat along group dimension
                repeat_factor = attn_module.group_size // max(1, kv_group_size)
                if repeat_factor > 1:
                    b = b.repeat_interleave(repeat_factor, dim=2)
                # Now b shape: (num_groups, rank, group_size, head_dim)
                b = b.transpose(1, 2)  # (num_groups, group_size, rank, head_dim)
                b = b.reshape(attn_module.num_heads, group_rank_k, attn_module.head_dim)
                new_module.B = nn.Parameter(b)

        # load to VT
        # shape (sum(ranks), hidden_size)
        VT_weight = torch.cat(wr, dim=0).contiguous()
        assert new_module.VT.weight.data.shape == VT_weight.shape
        new_module.VT.weight.data = VT_weight
        
        return new_module

class LlamaPaluAttention(LlamaAttention):
    """
    Llama Attention with Low-Rank KV-Cache with Palu. This module inherits from
    `LlamaAttention` but change linear layer and add custom Triton kernel.
    """
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        
        # Fallbacks for HF version differences
        self.hidden_size = getattr(self, "hidden_size", getattr(config, "hidden_size"))
        self.num_heads = getattr(self, "num_heads", getattr(config, "num_attention_heads"))
        self.num_key_value_heads = getattr(
            self,
            "num_key_value_heads",
            getattr(config, "num_key_value_heads", self.num_heads),
        )
        self.head_dim = getattr(self, "head_dim", self.hidden_size // self.num_heads)

        self.group_size = config.group_size
        self.num_groups = config.num_groups
        # Optional ranks on config (fallback to uniform 1 per group if missing)
        self.total_rank_k = getattr(config, "total_rank_k", self.num_groups)
        self.total_rank_v = getattr(config, "total_rank_v", self.num_groups)
        self.group_rank_k = max(1, self.total_rank_k // self.num_groups)
        self.group_rank_v = max(1, self.total_rank_v // self.num_groups)
        self.fused_hidden_dim_o = self.group_rank_v * self.num_heads
        self.rank_k_list = [self.group_rank_k for _ in range(self.num_groups)]
        self.rank_v_list = [self.group_rank_v for _ in range(self.num_groups)]

        # Behavior flags
        self.rope_in_latent: bool = getattr(config, "rope_in_latent", False)
        self.use_fusion: bool = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = HeadwiseLowRankModule(self.rank_k_list, self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = HeadwiseLowRankModule(self.rank_v_list, self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.fused_hidden_dim_o, self.hidden_size, bias=config.attention_bias)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        golden_kernel: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Two modes: standard vs latent-RoPE
        if not self.rope_in_latent:
            # Standard low-rank linear replacement path (behaves like original attention)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            kv_seq_len = key_states.shape[-2]
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            attn_output = self.o_proj(attn_output)
            if not output_attentions:
                attn_weights = None
            return attn_output, attn_weights, past_key_value

        # Latent RoPE mode
        key_h_states = self.k_proj.project_to_latent(hidden_states)
        value_h_states = self.v_proj.project_to_latent(hidden_states)
        key_h_states = key_h_states.view(bsz, q_len, self.num_groups, self.group_rank_k).transpose(1, 2)
        value_h_states = value_h_states.view(bsz, q_len, self.num_groups, self.group_rank_v).transpose(1, 2)
        kv_seq_len = key_h_states.shape[-2]

        # Debug log once to confirm latent RoPE path
        if not getattr(self, "_debug_rope_logged", False):
            try:
                logger.info(
                    f"[rope_svd] layer={self.layer_idx} rope_in_latent=True mode={'prompt' if q_len>1 else 'generate'} "
                    f"Q={tuple(query_states.shape)} latentK={tuple(key_h_states.shape)} latentV={tuple(value_h_states.shape)} "
                    f"num_heads={self.num_heads} num_kv_heads={self.num_key_value_heads} group_size={self.group_size} num_groups={self.num_groups} "
                    f"group_rank_k={self.group_rank_k} group_rank_v={self.group_rank_v} head_dim={self.head_dim} use_fusion={self.use_fusion}"
                )
            except Exception:
                pass
            self._debug_rope_logged = True
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        
        # cos, sin = self.rotary_emb(query_states, seq_len=kv_seq_len)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # Cache latent K/V to reduce memory
            key_h_states, value_h_states = past_key_value.update(key_h_states, value_h_states, self.layer_idx)


        if q_len > 1:
            # Prompting: reconstruct then apply standard RoPE(Q,K)
            key_h_flat = key_h_states.transpose(1, 2).reshape(bsz, kv_seq_len, -1)
            key_states = self.k_proj.reconstruct(key_h_flat)
            key_states = key_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            cos, sin = self.rotary_emb(query_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        else:
            # Generating (Apply our reconsturction kernel)
            # A: (num_heads, 1, head_dim)
            # B: (num_heads, rank_per_groups, head_dim)
            # X: (num_head_groups, seq_len, rank_per_groups)
            # TODO: Optimize RoPE & sqrt(head_dim) into kernel
            # TODO: Check if sin & cos are share among different blocks
            cos, sin = self.rotary_emb(query_states, seq_len=kv_seq_len)
            query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin, position_ids)
            A = query_states.squeeze(0)
            B = self.k_proj.B
            X = key_h_states.squeeze(0)
            attn_weights = recompute_k_gemv(A, B, X).unsqueeze(0) / math.sqrt(self.head_dim)

        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask


        # Upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Original version
        # value_states = self.v_proj.reconstruct(value_h_states)
        # value_states = value_states.reshape(1, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # attn_output = torch.matmul(attn_weights, value_states)

        # Produce attention output
        if self.use_fusion:
            # Fusion version: keep latent V, project in o_proj
            attn_h_weights = attn_weights.reshape(1, self.num_groups, q_len * self.group_size, kv_seq_len)
            attn_h_output = torch.matmul(attn_h_weights, value_h_states)
            attn_output = attn_h_output.reshape(1, self.num_heads, q_len, self.group_rank_v)
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
            attn_output = self.o_proj(attn_output)
        else:
            # Non-fusion: reconstruct V then standard combine
            value_h_flat = value_h_states.transpose(1, 2).reshape(bsz, kv_seq_len, -1)
            value_states = self.v_proj.reconstruct(value_h_flat)
            value_states = value_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            attn_output = self.o_proj(attn_output)
        
        
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
    @staticmethod
    def from_attention(
        module: LlamaAttention,
        config: LlamaConfig,
        no_fusion: bool = False,
        rank_k_list: Optional[list] = None,
        rank_v_list: Optional[list] = None,
        rope_in_latent: Optional[bool] = None,
    ):
        new_module = LlamaPaluAttention(config, module.layer_idx)
        new_module.q_proj = module.q_proj
        # Override ranks if provided; otherwise use module defaults
        k_ranks = rank_k_list if rank_k_list is not None else new_module.rank_k_list
        v_ranks = rank_v_list if rank_v_list is not None else new_module.rank_v_list
        new_module.rank_k_list = k_ranks
        new_module.rank_v_list = v_ranks
        # Update grouping to match provided ranks
        new_module.num_groups = len(k_ranks)
        if new_module.num_heads % new_module.num_groups != 0:
            raise ValueError(
                f"num_heads {new_module.num_heads} not divisible by num_groups {new_module.num_groups}"
            )
        new_module.group_size = new_module.num_heads // new_module.num_groups
        # Derive totals and group ranks (assume uniform ranks across groups)
        new_module.total_rank_k = sum(k_ranks)
        new_module.total_rank_v = sum(v_ranks)
        if len(set(k_ranks)) == 1:
            new_module.group_rank_k = k_ranks[0]
        if len(set(v_ranks)) == 1:
            new_module.group_rank_v = v_ranks[0]
        new_module.k_proj = HeadwiseLowRankModule.from_linear(module.k_proj, k_ranks, new_module)
        new_module.v_proj = HeadwiseLowRankModule.from_linear(module.v_proj, v_ranks)

        # No fusion version
        if no_fusion:
            new_module.o_proj = module.o_proj
            new_module.use_fusion = False
            if rope_in_latent is not None:
                new_module.rope_in_latent = rope_in_latent
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

        if rope_in_latent is not None:
            new_module.rope_in_latent = rope_in_latent

        return new_module
