#!/usr/bin/env python3
"""
同时优化VT和U来对齐RoPE(x@VT@U)和RoPE(x@VT)@U
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaRotaryEmbedding
from kernel.palu_attention import LlamaPaluAttention
import matplotlib.pyplot as plt
import json
import copy


def create_test_attention(rank_k=128, rank_v=384, num_groups=8, device='cuda'):
    """创建一个测试用的Palu Attention层"""
    config = LlamaConfig()
    config.hidden_size = 4096
    config.num_attention_heads = 32
    config.num_key_value_heads = 8
    config.group_size = 4
    config.num_groups = num_groups
    
    # 设置ranks
    group_rank_k = rank_k // num_groups
    group_rank_v = rank_v // num_groups
    config.head_wise_ranks = {
        "model.layers.0.self_attn.k_proj": [group_rank_k] * num_groups,
        "model.layers.0.self_attn.v_proj": [group_rank_v] * num_groups,
    }
    
    config.rope_latent = False
    config.v_fusion = False
    
    attention = LlamaPaluAttention(config, layer_idx=0).to(device)
    return attention, config


def test_vt_u_optimization(attention, batch_size=4, seq_len=64, num_steps=3000, device='cuda'):
    """同时优化VT和U"""
    hidden_dim = attention.hidden_size
    
    # 生成测试数据
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    # 生成RoPE
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    rotary_emb = LlamaRotaryEmbedding(config=attention.config).to(device)
    cos, sin = rotary_emb(hidden_states, position_ids)
    
    # 获取K投影
    k_proj = attention.k_proj
    
    # 保存原始权重
    original_U_weights = [U.weight.data.clone() for U in k_proj.U]
    original_VT_weight = k_proj.VT.weight.data.clone()
    
    # 创建目标（使用原始权重）
    with torch.no_grad():
        # PALU: x@VT@U with RoPE
        key_latents = k_proj.project_to_latent(hidden_states)
        key_states_palu = k_proj.reconstruct(key_latents)
        key_states_palu_4d = key_states_palu.view(batch_size, seq_len, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)
        _, key_states_palu_rope = apply_rotary_pos_emb_helper(None, key_states_palu_4d, cos, sin)
        target = key_states_palu_rope.detach()
    
    # 重新初始化参数用于优化
    # 策略1：从原始权重开始
    for i, U in enumerate(k_proj.U):
        U.weight.data = original_U_weights[i].clone()
    k_proj.VT.weight.data = original_VT_weight.clone()
    
    # 设置所有参数为可训练
    for param in attention.parameters():
        param.requires_grad = False
    
    # 只优化K投影的VT和U
    k_proj.VT.weight.requires_grad = True
    for U in k_proj.U:
        U.weight.requires_grad = True
    
    # 收集可优化参数
    params = [k_proj.VT.weight] + [U.weight for U in k_proj.U]
    
    # 优化器
    optimizer = optim.AdamW(params, lr=0.001, weight_decay=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=2)
    
    loss_history = []
    relative_errors = []
    vt_changes = []
    u_changes = []
    
    print("\n开始同时优化VT和U...")
    print(f"输入形状: {hidden_states.shape}")
    print(f"K投影: {len(k_proj.U)} 组, ranks = {k_proj.ranks}")
    print(f"可优化参数数量: VT={k_proj.VT.weight.numel():,}, U={sum(U.weight.numel() for U in k_proj.U):,}")
    
    best_loss = float('inf')
    best_weights = None
    patience = 0
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # HACK模式计算
        # Step 1: x@VT
        key_latents = k_proj.project_to_latent(hidden_states)
        
        # Step 2: Reshape and apply RoPE to latents
        latent_per_head = key_latents.shape[-1] // attention.num_key_value_heads
        key_latents_4d = key_latents.view(batch_size, seq_len, attention.num_key_value_heads, latent_per_head).transpose(1, 2)
        
        cos_latent = cos[..., :latent_per_head]
        sin_latent = sin[..., :latent_per_head]
        _, key_latents_rope = apply_rotary_pos_emb_helper(None, key_latents_4d, cos_latent, sin_latent)
        
        # Step 3: Reconstruct with U
        key_latents_rope_3d = key_latents_rope.transpose(1, 2).reshape(batch_size, seq_len, -1)
        key_states_hack = k_proj.reconstruct(key_latents_rope_3d)
        key_states_hack_4d = key_states_hack.view(batch_size, seq_len, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)
        
        # 计算损失
        mse_loss = nn.functional.mse_loss(key_states_hack_4d, target)
        
        # 正则化项
        reg_loss = 0
        # VT正则化
        reg_loss += 0.0001 * torch.norm(k_proj.VT.weight - original_VT_weight) ** 2
        # U正则化
        for i, U in enumerate(k_proj.U):
            reg_loss += 0.0001 * torch.norm(U.weight - original_U_weights[i]) ** 2
        
        # 保持VT@U ≈ 原始重构的约束
        with torch.no_grad():
            # 随机采样一些latents
            sample_latents = torch.randn(1, 100, sum(k_proj.ranks), device=device)
            original_recon = reconstruct_with_weights(sample_latents, original_U_weights, k_proj.ranks, k_proj.group_dim)
        current_recon = k_proj.reconstruct(sample_latents)
        consistency_loss = 0.001 * nn.functional.mse_loss(current_recon, original_recon)
        
        total_loss = mse_loss + reg_loss + consistency_loss
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # 记录
        loss_value = mse_loss.item()
        loss_history.append(loss_value)
        relative_error = (mse_loss / torch.var(target)).item()
        relative_errors.append(relative_error)
        
        # 记录权重变化
        vt_change = torch.norm(k_proj.VT.weight - original_VT_weight).item() / torch.norm(original_VT_weight).item()
        u_change_avg = sum(torch.norm(U.weight - original_U_weights[i]).item() / torch.norm(original_U_weights[i]).item() 
                           for i, U in enumerate(k_proj.U)) / len(k_proj.U)
        vt_changes.append(vt_change)
        u_changes.append(u_change_avg)
        
        # 保存最佳权重
        if loss_value < best_loss:
            best_loss = loss_value
            best_weights = {
                'VT': k_proj.VT.weight.data.clone(),
                'U': [U.weight.data.clone() for U in k_proj.U]
            }
            patience = 0
        else:
            patience += 1
        
        if step % 200 == 0:
            print(f"Step {step}: Loss = {loss_value:.8f}, RelErr = {relative_error:.6f}, "
                  f"VT_change = {vt_change:.4f}, U_change = {u_change_avg:.4f}")
        
        # 早停
        if patience > 500 and step > 1000:
            print(f"早停于步数 {step}")
            break
    
    # 使用最佳权重进行最终评估
    k_proj.VT.weight.data = best_weights['VT']
    for i, U in enumerate(k_proj.U):
        U.weight.data = best_weights['U'][i]
    
    with torch.no_grad():
        # 重新计算
        key_latents = k_proj.project_to_latent(hidden_states)
        latent_per_head = key_latents.shape[-1] // attention.num_key_value_heads
        key_latents_4d = key_latents.view(batch_size, seq_len, attention.num_key_value_heads, latent_per_head).transpose(1, 2)
        _, key_latents_rope = apply_rotary_pos_emb_helper(None, key_latents_4d, cos_latent, sin_latent)
        key_latents_rope_3d = key_latents_rope.transpose(1, 2).reshape(batch_size, seq_len, -1)
        key_states_final_hack = k_proj.reconstruct(key_latents_rope_3d)
        key_states_final_hack_4d = key_states_final_hack.view(batch_size, seq_len, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)
        
        final_loss = nn.functional.mse_loss(key_states_final_hack_4d, target)
        final_relative_error = (final_loss / torch.var(target)).item()
    
    # 分析权重变化
    final_vt_change = torch.norm(best_weights['VT'] - original_VT_weight).item() / torch.norm(original_VT_weight).item()
    final_u_changes = []
    for i, U_weight in enumerate(best_weights['U']):
        change = torch.norm(U_weight - original_U_weights[i]).item() / torch.norm(original_U_weights[i]).item()
        final_u_changes.append(change)
    
    print(f"\n最终VT相对变化: {final_vt_change:.4f}")
    for i, change in enumerate(final_u_changes):
        print(f"最终U[{i}]相对变化: {change:.4f}")
    
    # 恢复原始权重
    k_proj.VT.weight.data = original_VT_weight
    for i, U in enumerate(k_proj.U):
        U.weight.data = original_U_weights[i]
    
    return {
        'final_loss': final_loss.item(),
        'relative_error': final_relative_error,
        'best_loss': best_loss,
        'loss_history': loss_history,
        'relative_errors': relative_errors,
        'vt_changes': vt_changes,
        'u_changes': u_changes,
        'final_vt_change': final_vt_change,
        'final_u_changes': final_u_changes,
        'converged': final_relative_error < 0.01
    }


def reconstruct_with_weights(latents, U_weights, ranks, group_dim):
    """使用给定权重重建"""
    outputs = []
    total_ranks = 0
    for i, U_weight in enumerate(U_weights):
        latent = latents[:, :, total_ranks: total_ranks + ranks[i]]
        output = nn.functional.linear(latent, U_weight)
        outputs.append(output)
        total_ranks += ranks[i]
    return torch.cat(outputs, dim=-1)


def apply_rotary_pos_emb_helper(q, k, cos, sin):
    """辅助函数：应用RoPE"""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    
    if q is not None:
        q_embed = (q * cos) + (rotate_half(q) * sin)
    else:
        q_embed = None
        
    if k is not None:
        k_embed = (k * cos) + (rotate_half(k) * sin)
    else:
        k_embed = None
        
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def visualize_vt_u_results(results, save_path='rope_alignment_vt_u.png'):
    """可视化VT+U优化结果"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 损失曲线
    ax1.plot(results['loss_history'], alpha=0.7)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Loss History')
    ax1.set_yscale('log')
    ax1.grid(True)
    
    # 2. 相对误差曲线
    ax2.plot(results['relative_errors'], color='orange', alpha=0.7)
    ax2.axhline(y=0.01, color='g', linestyle='--', label='1% Target')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Relative Error')
    ax2.set_title('Relative Error History')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True)
    
    # 3. VT权重变化
    ax3.plot(results['vt_changes'], label='VT', color='blue', linewidth=2)
    ax3.plot(results['u_changes'], label='U (avg)', color='red', linewidth=2)
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Relative Change')
    ax3.set_title('Weight Changes During Optimization')
    ax3.legend()
    ax3.grid(True)
    
    # 4. 最终U变化
    ax4.bar(range(len(results['final_u_changes'])), results['final_u_changes'])
    ax4.set_xlabel('U Matrix Index')
    ax4.set_ylabel('Final Relative Change')
    ax4.set_title('Final U Matrix Changes')
    ax4.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n可视化结果已保存到: {save_path}")


def main(args):
    print("=" * 80)
    print("VT+U 同时优化测试")
    print("=" * 80)
    
    # 创建测试attention层
    attention, config = create_test_attention(
        rank_k=args.rank_k,
        rank_v=args.rank_v,
        num_groups=args.num_groups,
        device=args.device
    )
    
    # 运行测试
    results = test_vt_u_optimization(
        attention,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_steps=args.num_steps,
        device=args.device
    )
    
    # 输出结果
    print("\n" + "=" * 60)
    print("测试结果:")
    print("=" * 60)
    print(f"最终损失: {results['final_loss']:.8f}")
    print(f"相对误差: {results['relative_error']:.6f} ({results['relative_error']*100:.3f}%)")
    print(f"最佳损失: {results['best_loss']:.8f}")
    print(f"是否收敛: {'✓ 是' if results['converged'] else '✗ 否'} (阈值 < 1%)")
    print(f"VT相对变化: {results['final_vt_change']:.4f}")
    print(f"U平均变化: {sum(results['final_u_changes'])/len(results['final_u_changes']):.4f}")
    
    # 保存结果
    if args.save_results:
        with open('rope_alignment_vt_u_results.json', 'w') as f:
            save_data = {k: v for k, v in results.items() if k not in ['loss_history', 'relative_errors', 'vt_changes', 'u_changes']}
            save_data['final_stats'] = {
                'loss': results['final_loss'],
                'relative_error': results['relative_error'],
                'vt_change': results['final_vt_change'],
                'u_changes': results['final_u_changes']
            }
            json.dump(save_data, f, indent=2)
        print(f"\n详细结果已保存到: rope_alignment_vt_u_results.json")
    
    # 可视化
    if args.visualize:
        visualize_vt_u_results(results)
    
    # 结论
    print("\n" + "=" * 60)
    print("结论:")
    print("=" * 60)
    if results['converged']:
        print("✓ 同时优化VT和U可以实现 RoPE(x@VT)@U 对齐 RoPE(x@VT@U)")
        print("✓ 这证明了HACK模式在数学上是可行的，只需要调整投影矩阵")
        print("✓ 建议：使用这种方法fine-tune模型")
    else:
        print(f"△ 相对误差为 {results['relative_error']*100:.3f}%")
        if results['relative_error'] < 0.05:
            print("△ 虽未达到1%，但已经相当接近，可能足够实用")
        else:
            print("△ 可能需要更复杂的优化策略或更多步数")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VT+U同时优化测试")
    parser.add_argument("--rank_k", type=int, default=1024, help="K投影的总秩")
    parser.add_argument("--rank_v", type=int, default=3072, help="V投影的总秩")
    parser.add_argument("--num_groups", type=int, default=8, help="组数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--seq_len", type=int, default=64, help="序列长度")
    parser.add_argument("--num_steps", type=int, default=3000, help="优化步数")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    parser.add_argument("--visualize", action="store_true", help="是否生成图表")
    parser.add_argument("--save_results", action="store_true", help="是否保存结果")
    
    args = parser.parse_args()
    main(args)
