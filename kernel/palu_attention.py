import math
import warnings
from typing import Optional, Tuple

import torch
from torch import nn

from transformers.models.llama.modeling_llama import (
    Cache, apply_rotary_pos_emb, 
    LlamaAttention, LlamaConfig,
)

# Import HeadwiseLowRankModule from svd_linear.py instead of defining it here
from palu.model.modules.svd_linear import HeadwiseLowRankModule

from .abx_rope import abx as recompute_k_gemv

class LlamaPaluAttention(LlamaAttention):
    """
    Llama Attention with Low-Rank KV-Cache with Palu. This module inherits from
    `LlamaAttention` but change linear layer and add custom Triton kernel.
    """
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        
        # === 基础配置参数 ===
        self.hidden_size = config.hidden_size          # 模型隐藏维度，如 4096
        self.group_size = config.group_size            # 每组的头数，如 4 (head_group_size)
        self.num_groups = config.num_groups            # 组数，如 2 (8 KV heads // 4 = 2)
        self.total_rank_k = config.total_rank_k        # K 投影的全局总 rank，如 832
        self.total_rank_v = config.total_rank_v        # V 投影的全局总 rank，如 1024
        
        # === 从父类继承的关键维度 ===
        self.num_heads = config.num_attention_heads # (32, Q heads)
        self.num_key_value_heads = config.num_key_value_heads # (8, K/V heads)
        self.head_dim = config.hidden_size // config.num_attention_heads # (128)
        
        # === 计算派生维度 ===
        # 每个组分配的 rank (总 rank 平均分配给各组)
        self.group_rank_k = self.total_rank_k // self.num_groups  # 832 // 2 = 416
        self.group_rank_v = self.total_rank_v // self.num_groups  # 1024 // 2 = 512
        
        # O 投影的输入维度：每组的 V rank * 组内头数
        # 这是因为 V 投影输出会被 concat，然后送入 O 投影
        self.fused_hidden_dim_o = self.group_rank_v * self.group_size  # 512 * 4 = 2048
        
        # === 生成每组的 rank 列表 ===
        # 为每个组分配相同的 rank (简化版本，实际可以每组不同)
        self.rank_k_list = [self.group_rank_k for _ in range(self.num_groups)]  # [416, 416]
        self.rank_v_list = [self.group_rank_v for _ in range(self.num_groups)]  # [512, 512]

        # === 创建投影层 ===
        # Q 投影：标准线性层，不进行低秩分解
        # 输入: hidden_size (4096) -> 输出: num_attention_heads * head_dim (32 * 128 = 4096)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        
        # K 投影：使用低秩分解，HeadwiseLowRankModule
        # 输入: hidden_size (4096) -> 输出: num_key_value_heads * head_dim (8 * 128 = 1024)
        # 内部结构: 4096 -> VT(832) -> U[416->512, 416->512] -> 1024
        self.k_proj = HeadwiseLowRankModule(self.rank_k_list, self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        
        # V 投影：使用低秩分解，HeadwiseLowRankModule  
        # 输入: hidden_size (4096) -> 输出: num_key_value_heads * head_dim (8 * 128 = 1024)
        # 内部结构: 4096 -> VT(1024) -> U[512->512, 512->512] -> 1024
        self.v_proj = HeadwiseLowRankModule(self.rank_v_list, self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        
        # O 投影：标准线性层，但输入维度是压缩后的融合维度
        # 输入: fused_hidden_dim_o (2048) -> 输出: hidden_size (4096)
        # 注意：输入维度不是标准的 num_attention_heads * head_dim，而是压缩后的维度
        self.o_proj = nn.Linear(self.fused_hidden_dim_o, self.hidden_size, bias=config.attention_bias)
        
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
        no_fusion: bool = False,
    ):
        """
        从标准的 LlamaAttention 创建 LlamaPaluAttention。
        
        支持两种场景：
        1. 压缩场景：module.k_proj/v_proj 是 nn.Linear，需要转换为 HeadwiseLowRankModule
        2. 加载场景：module.k_proj/v_proj 已经是 HeadwiseLowRankModule，直接复用
        
        Args:
            module: 原始的 LlamaAttention 模块
            config: 配置对象，包含 head_wise_ranks 等信息
            no_fusion: 是否禁用融合优化
        """
        # 创建新的 LlamaPaluAttention 实例
        new_module = LlamaPaluAttention(config, module.layer_idx)
        
        # Q 投影始终复用原始的 Linear 层（不压缩）
        new_module.q_proj = module.q_proj
        
        # === K/V 投影处理：根据当前状态决定策略 ===
        if isinstance(module.k_proj, HeadwiseLowRankModule):
            # 场景 2：加载已压缩的模型
            # k_proj/v_proj 已经是 HeadwiseLowRankModule，权重已正确加载，直接复用
            new_module.k_proj = module.k_proj
            new_module.v_proj = module.v_proj
        else:
            # 场景 1：压缩原始模型
            # k_proj/v_proj 是原始的 nn.Linear，需要转换为 HeadwiseLowRankModule
            
            # 优先使用 config.head_wise_ranks 中保存的层特定 ranks
            head_wise_ranks = getattr(config, "head_wise_ranks", {})
            layer_prefix = f"model.layers.{module.layer_idx}.self_attn"
            k_proj_key = f"{layer_prefix}.k_proj"
            v_proj_key = f"{layer_prefix}.v_proj"
            
            if k_proj_key in head_wise_ranks and v_proj_key in head_wise_ranks:
                # 使用保存的层特定 ranks（如 [160, 160], [512, 512]）
                rank_k_list = head_wise_ranks[k_proj_key]
                rank_v_list = head_wise_ranks[v_proj_key]
            else:
                # 回退到 __init__ 中计算的全局 ranks
                rank_k_list = new_module.rank_k_list
                rank_v_list = new_module.rank_v_list
            
            # 获取 RoPE 配置
            rope_latent = getattr(config, "rope_latent", False)
            
            # 执行 SVD 分解，将 nn.Linear 转换为 HeadwiseLowRankModule
            new_module.k_proj = HeadwiseLowRankModule.from_linear(module.k_proj, rank_k_list, rope_in_latent=rope_latent)
            new_module.v_proj = HeadwiseLowRankModule.from_linear(module.v_proj, rank_v_list, rope_in_latent=rope_latent)

        # No fusion version
        if no_fusion:
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