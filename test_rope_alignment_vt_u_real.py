#!/usr/bin/env python3
"""
测试（基于真实模型权重）是否可以通过同时优化 VT 和 U，使 RoPE(x@VT)@U 对齐 RoPE(x@VT@U)。
流程：
1) 加载真实模型权重（Palu模型目录）
2) 用一小批 wikitext2 文本跑一次前向，hook 捕获指定层的 hidden_states（self_attn 的输入）
3) 从该层的 k_proj 提取 VT 和 U（HeadwiseLowRankModule）
4) 计算目标：PALU 路径 RoPE(x@VT@U)
5) 优化：HACK 路径 RoPE(x@VT)@U，训练 VT+U
6) 报告相对误差和权重变化；可选将改后的 VT/U 保存到文件
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
import matplotlib.pyplot as plt

from utils import load_model_and_tokenizer
from kernel.palu_attention import LlamaPaluAttention


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_helper(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin) if q is not None else None
    k_embed = (k * cos) + (rotate_half(k) * sin) if k is not None else None
    return q_embed, k_embed


def capture_hidden_states(model, tokenizer, device: str, layer_idx: int, batch_size: int, seq_len: int, split: str = "test") -> torch.Tensor:
    """Run one forward pass on wikitext2 to capture the input hidden_states of the target self_attn layer.
    split: "train" or "test"
    """
    # Prepare a small batch from wikitext2
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(ds[:200]["text"])  # small slice
    toks = tokenizer(text, return_tensors="pt")
    flat_ids = toks.input_ids.squeeze(0)
    total_needed = batch_size * seq_len
    if flat_ids.shape[0] < total_needed:
        pad = total_needed - flat_ids.shape[0]
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        flat_ids = torch.cat([flat_ids, torch.full((pad,), pad_id, dtype=flat_ids.dtype)], dim=0)
    flat_ids = flat_ids[:total_needed]
    input_ids = flat_ids.view(batch_size, seq_len).to(device)

    # We'll capture the normalized hidden_states right before self_attn in the target decoder layer
    target_layer = model.model.layers[layer_idx]
    captured = {}

    class _CaptureAbort(Exception):
        pass

    def hook_pre_attn(mod, args, kwargs):
        hs = kwargs.get("hidden_states", None)
        if hs is None and len(args) >= 1:
            hs = args[0]
        if hs is not None and "hidden_states" not in captured:
            captured["hidden_states"] = hs.detach()
            # Abort the forward to avoid dtype issues inside attention
            raise _CaptureAbort()

    handle = target_layer.self_attn.register_forward_pre_hook(hook_pre_attn, with_kwargs=True)

    try:
        with torch.no_grad():
            model.config.use_cache = False
            _ = model(input_ids=input_ids)
    except _CaptureAbort:
        pass
    finally:
        handle.remove()

    hs = captured.get("hidden_states")
    if hs is None:
        raise RuntimeError("Failed to capture hidden_states. Hook did not trigger.")

    if hs.dim() != 3:
        raise RuntimeError(f"Captured hidden_states has unexpected shape: {hs.shape}")
    if hs.shape[0] != batch_size or hs.shape[1] != seq_len:
        hs = hs[:batch_size, :seq_len, :]
    return hs.to(device)


def evaluate_on_hidden_states(model, layer_idx: int, hidden_states: torch.Tensor) -> float:
    """Compute relative error on provided hidden_states without further updates."""
    device = hidden_states.device
    attn: LlamaPaluAttention = model.model.layers[layer_idx].self_attn  # type: ignore
    k_proj = attn.k_proj

    hidden_states = hidden_states.to(torch.float32)
    seq_len = hidden_states.shape[1]

    rotary_emb = LlamaRotaryEmbedding(config=attn.config).to(device)
    cos, sin = rotary_emb(hidden_states, torch.arange(seq_len, device=device).unsqueeze(0))
    cos = cos.to(torch.float32)
    sin = sin.to(torch.float32)

    with torch.no_grad():
        # Target
        latents = k_proj.project_to_latent(hidden_states)
        full = k_proj.reconstruct(latents)
        full_4d = full.view(hidden_states.shape[0], seq_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        _, target = apply_rotary_pos_emb_helper(None, full_4d, cos, sin)
        target = target.to(torch.float32)
        var_target = torch.var(target).clamp_min(1e-12)

        # Hack
        lat = k_proj.project_to_latent(hidden_states)
        per_head = lat.shape[-1] // attn.num_key_value_heads
        lat_4d = lat.view(hidden_states.shape[0], seq_len, attn.num_key_value_heads, per_head).transpose(1, 2)
        cos_lat, sin_lat = cos[..., :per_head], sin[..., :per_head]
        _, lat_rope = apply_rotary_pos_emb_helper(None, lat_4d, cos_lat, sin_lat)
        lat_rope_3d = lat_rope.transpose(1, 2).reshape(hidden_states.shape[0], seq_len, -1)
        hack = k_proj.reconstruct(lat_rope_3d)
        hack_4d = hack.view(hidden_states.shape[0], seq_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)

        mse = nn.functional.mse_loss(hack_4d, target)
        rel = (mse / var_target).item()
        return float(rel)


def dump_decomposed_weights(k_proj_module, out_dir: Path, layer_idx: int):
    """Save current VT and U weights to disk for inspection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vt_path = out_dir / f"layer{layer_idx}_kproj_VT.pt"
    torch.save(k_proj_module.VT.weight.detach().cpu(), vt_path)
    for i, U in enumerate(k_proj_module.U):
        torch.save(U.weight.detach().cpu(), out_dir / f"layer{layer_idx}_kproj_U_{i}.pt")


def reconstruct_with_weights(latents, U_weights, ranks, group_dim):
    """使用给定权重重建 - 用于consistency loss计算"""
    outputs = []
    total_ranks = 0
    for i, U_weight in enumerate(U_weights):
        latent = latents[:, :, total_ranks: total_ranks + ranks[i]]
        output = nn.functional.linear(latent, U_weight)
        outputs.append(output)
        total_ranks += ranks[i]
    return torch.cat(outputs, dim=-1)


def visualize_vt_u_results(results, save_path='rope_alignment_vt_u_real.png'):
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
    ax2.plot(results['relative_history'], color='orange', alpha=0.7)
    ax2.axhline(y=0.01, color='g', linestyle='--', label='1% Target')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Relative Error')
    ax2.set_title('Relative Error History')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True)

    # 3. 权重变化历史 (如果有的话)
    if 'vt_changes' in results and 'u_changes' in results:
        ax3.plot(results['vt_changes'], label='VT', color='blue', linewidth=2)
        ax3.plot(results['u_changes'], label='U (avg)', color='red', linewidth=2)
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Relative Change')
        ax3.set_title('Weight Changes During Optimization')
        ax3.legend()
        ax3.grid(True)
    else:
        ax3.text(0.5, 0.5, 'Weight change data not available',
                transform=ax3.transAxes, ha='center', va='center')
        ax3.set_title('Weight Changes (N/A)')

    # 4. 最终U变化
    if 'u_changes' in results:
        ax4.bar(range(len(results['u_changes'])), results['u_changes'])
        ax4.set_xlabel('U Matrix Index')
        ax4.set_ylabel('Final Relative Change')
        ax4.set_title('Final U Matrix Changes')
        ax4.grid(True, axis='y')
    else:
        ax4.text(0.5, 0.5, 'U change data not available',
                transform=ax4.transAxes, ha='center', va='center')
        ax4.set_title('Final U Changes (N/A)')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n可视化结果已保存到: {save_path}")


def vt_u_alignment_on_real(model, layer_idx: int, hidden_states: torch.Tensor, num_steps: int,
                            save_dir: Path = None, use_consistency_loss: bool = True) -> Dict[str, Any]:
    device = hidden_states.device
    attn: LlamaPaluAttention = model.model.layers[layer_idx].self_attn  # type: ignore
    k_proj = attn.k_proj

    # Work in float32 to avoid fp16 overflows/NaNs
    hidden_states = hidden_states.to(torch.float32)
    with torch.no_grad():
        k_proj.VT.weight.data = k_proj.VT.weight.data.to(torch.float32)
        for U in k_proj.U:
            U.weight.data = U.weight.data.to(torch.float32)

    # Rotary embeddings in float32
    seq_len = hidden_states.shape[1]
    rotary_emb = LlamaRotaryEmbedding(config=attn.config).to(device)
    cos, sin = rotary_emb(hidden_states, torch.arange(seq_len, device=device).unsqueeze(0))
    cos = cos.to(torch.float32)
    sin = sin.to(torch.float32)

    # Backup original weights
    original_VT = k_proj.VT.weight.detach().clone()
    original_U = [U.weight.detach().clone() for U in k_proj.U]

    # Target: PALU path RoPE(x@VT@U)
    with torch.no_grad():
        latents = k_proj.project_to_latent(hidden_states)  # x@VT
        full = k_proj.reconstruct(latents)                 # x@VT@U
        full_4d = full.view(hidden_states.shape[0], seq_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        _, target = apply_rotary_pos_emb_helper(None, full_4d, cos, sin)
        target = target.to(torch.float32)
        var_target = torch.var(target).clamp_min(1e-12)

    # Make VT, U trainable
    for p in model.parameters():
        p.requires_grad_(False)
    k_proj.VT.weight.requires_grad_(True)
    for U in k_proj.U:
        U.weight.requires_grad_(True)

    params = [k_proj.VT.weight] + [U.weight for U in k_proj.U]
    opt = optim.AdamW(params, lr=1e-4, weight_decay=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=max(200, num_steps // 5), T_mult=2)

    best = {"loss": float("inf"), "VT": None, "U": None}
    loss_hist, rel_hist, vt_changes, u_changes = [], [], [], []

    for step in range(num_steps):
        opt.zero_grad()
        # HACK path: RoPE(x@VT)@U
        lat = k_proj.project_to_latent(hidden_states)
        per_head = lat.shape[-1] // attn.num_key_value_heads
        lat_4d = lat.view(hidden_states.shape[0], seq_len, attn.num_key_value_heads, per_head).transpose(1, 2)
        cos_lat, sin_lat = cos[..., :per_head], sin[..., :per_head]
        _, lat_rope = apply_rotary_pos_emb_helper(None, lat_4d, cos_lat, sin_lat)
        lat_rope_3d = lat_rope.transpose(1, 2).reshape(hidden_states.shape[0], seq_len, -1)
        hack = k_proj.reconstruct(lat_rope_3d)
        hack_4d = hack.view(hidden_states.shape[0], seq_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)

        mse = nn.functional.mse_loss(hack_4d, target)
        if not torch.isfinite(mse):
            print(f"Encountered non-finite MSE at step {step}. Stopping.")
            break

        # Basic regularization to keep close to original
        reg = 1e-4 * (torch.norm(k_proj.VT.weight - original_VT) ** 2)
        for i, U in enumerate(k_proj.U):
            reg = reg + 1e-4 * (torch.norm(U.weight - original_U[i]) ** 2)

        loss = mse + reg

        # Add consistency loss if enabled (keep VT@U ≈ original reconstruction)
        if use_consistency_loss:
            with torch.no_grad():
                # Sample some latents for consistency check
                sample_latents = torch.randn(1, 100, sum(k_proj.ranks), device=device)
                original_recon = reconstruct_with_weights(sample_latents, original_U, k_proj.ranks, k_proj.group_dim)
            current_recon = k_proj.reconstruct(sample_latents)
            consistency_loss = 0.001 * nn.functional.mse_loss(current_recon, original_recon)
            loss = loss + consistency_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()

        with torch.no_grad():
            rel = (mse / var_target).item()
            loss_hist.append(mse.item())
            rel_hist.append(rel)

            # Track weight changes
            vt_change = torch.norm(k_proj.VT.weight - original_VT).item() / torch.norm(original_VT).item()
            u_change_avg = sum(torch.norm(U.weight - original_U[i]).item() / torch.norm(original_U[i]).item()
                              for i, U in enumerate(k_proj.U)) / len(k_proj.U)
            vt_changes.append(vt_change)
            u_changes.append(u_change_avg)

            if mse.item() < best["loss"]:
                best["loss"] = mse.item()
                best["VT"] = k_proj.VT.weight.detach().clone()
                best["U"] = [U.weight.detach().clone() for U in k_proj.U]

        if step % max(1, num_steps // 10) == 0:
            print(f"Step {step}: Loss={mse.item():.6f}, RelErr={rel:.6f}")

    # Restore best
    if best["VT"] is not None:
        with torch.no_grad():
            k_proj.VT.weight.copy_(best["VT"])
            for i, U in enumerate(k_proj.U):
                U.weight.copy_(best["U"][i])

    # Final eval
    with torch.no_grad():
        lat = k_proj.project_to_latent(hidden_states)
        per_head = lat.shape[-1] // attn.num_key_value_heads
        lat_4d = lat.view(hidden_states.shape[0], seq_len, attn.num_key_value_heads, per_head).transpose(1, 2)
        cos_lat, sin_lat = cos[..., :per_head], sin[..., :per_head]
        _, lat_rope = apply_rotary_pos_emb_helper(None, lat_4d, cos_lat, sin_lat)
        lat_rope_3d = lat_rope.transpose(1, 2).reshape(hidden_states.shape[0], seq_len, -1)
        hack = k_proj.reconstruct(lat_rope_3d)
        hack_4d = hack.view(hidden_states.shape[0], seq_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        final_mse = nn.functional.mse_loss(hack_4d, target)
        final_rel = (final_mse / var_target).item()

    # Changes
    vt_change = torch.norm(k_proj.VT.weight - original_VT).item() / torch.norm(original_VT).item()
    u_changes_final = [
        torch.norm(k_proj.U[i].weight - original_U[i]).item() / torch.norm(original_U[i]).item()
        for i in range(len(k_proj.U))
    ]

    # Optionally save updated weights
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(k_proj.VT.weight.detach().cpu(), save_dir / f"layer{layer_idx}_kproj_VT_aligned.pt")
        for i, U in enumerate(k_proj.U):
            torch.save(U.weight.detach().cpu(), save_dir / f"layer{layer_idx}_kproj_U_{i}_aligned.pt")
        with open(save_dir / f"layer{layer_idx}_summary.json", "w") as f:
            json.dump({
                "final_mse": float(final_mse.item()),
                "final_relative_error": float(final_rel),
                "vt_change": float(vt_change),
                "u_changes": [float(v) for v in u_changes_final],
            }, f, indent=2)

    return {
        "final_mse": float(final_mse.item()),
        "final_relative_error": float(final_rel),
        "vt_change": float(vt_change),
        "u_changes": [float(v) for v in u_changes_final],
        "loss_history": loss_hist,
        "relative_history": rel_hist,
        "vt_changes": vt_changes,
        "u_changes": u_changes,
    }


def main():
    parser = argparse.ArgumentParser(description="VT+U alignment on real model weights")
    parser.add_argument("--model_path", type=str,
                        default="Meta-Llama-3-8B-Instruct_ratio-0.7_gs-4-fisher_uniform-svd",
                        help="Path to Palu model directory")
    parser.add_argument("--layer_idx", type=int, default=0, help="Target layer index to test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dump_dir", type=str, default="vt_u_dumps", help="Where to dump VT/U weights")
    parser.add_argument("--save_aligned", action="store_true", help="Save aligned VT/U weights")
    parser.add_argument("--train_split", type=str, default="train", choices=["train", "validation", "test"], help="Split for alignment data")
    parser.add_argument("--eval_split", type=str, default="test", choices=["train", "validation", "test"], help="Split for evaluation data")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization plots")
    parser.add_argument("--no_consistency_loss", action="store_true", help="Disable consistency loss (VT@U ≈ original reconstruction)")
    parser.add_argument("--save_results", action="store_true", help="Save detailed results to JSON file")

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    # Sanity check target layer is PaluAttention
    attn = model.model.layers[args.layer_idx].self_attn
    if not isinstance(attn, LlamaPaluAttention):
        raise TypeError(f"Layer {args.layer_idx} self_attn is not LlamaPaluAttention: {type(attn)}")

    # Capture train hidden_states for alignment
    train_hidden_states = capture_hidden_states(model, tokenizer, args.device, args.layer_idx, args.batch_size, args.seq_len, split=args.train_split)

    # Dump current VT/U
    dump_dir = Path(args.dump_dir)
    dump_decomposed_weights(attn.k_proj, dump_dir, args.layer_idx)

    # Run alignment on train
    save_dir = dump_dir if args.save_aligned else None
    use_consistency = not args.no_consistency_loss
    summary = vt_u_alignment_on_real(model, args.layer_idx, train_hidden_states, args.num_steps, save_dir, use_consistency)

    # Evaluate on eval split (test)
    eval_hidden_states = capture_hidden_states(model, tokenizer, args.device, args.layer_idx, args.batch_size, args.seq_len, split=args.eval_split)
    eval_rel = evaluate_on_hidden_states(model, args.layer_idx, eval_hidden_states)

    print("\n=== Alignment Summary (Real Weights) ===")
    print(f"Layer: {args.layer_idx}")
    print(f"Consistency Loss: {'Enabled' if use_consistency else 'Disabled'}")
    print(f"Train Final MSE: {summary['final_mse']:.6f}")
    print(f"Train Final Relative Error: {summary['final_relative_error']*100:.3f}%")
    print(f"Eval Relative Error: {eval_rel*100:.3f}%")
    print(f"VT change: {summary['vt_change']:.4f}")
    print(f"U changes (avg): {sum(summary['u_changes'])/len(summary['u_changes']):.4f}")

    # Generate visualization if requested
    if args.visualize:
        vis_path = f"rope_alignment_real_layer{args.layer_idx}_{'consistency' if use_consistency else 'no_consistency'}.png"
        visualize_vt_u_results(summary, vis_path)

    # Save detailed results if requested
    if args.save_results:
        results_file = f"vt_u_alignment_real_results_layer{args.layer_idx}.json"
        with open(results_file, 'w') as f:
            # Remove large arrays that would make JSON too big
            save_data = {k: v for k, v in summary.items() if k not in ['loss_history', 'relative_history', 'vt_changes', 'u_changes']}
            save_data['config'] = {
                'layer_idx': args.layer_idx,
                'num_steps': args.num_steps,
                'batch_size': args.batch_size,
                'seq_len': args.seq_len,
                'use_consistency_loss': use_consistency,
                'eval_relative_error': float(eval_rel)
            }
            json.dump(save_data, f, indent=2)
        print(f"\nDetailed results saved to: {results_file}")


if __name__ == "__main__":
    main()
