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


def vt_u_alignment_on_real(model, layer_idx: int, hidden_states: torch.Tensor, num_steps: int,
                            save_dir: Path = None) -> Dict[str, Any]:
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
    loss_hist, rel_hist = [], []

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
        # small regularization to keep close to original
        reg = 1e-4 * (torch.norm(k_proj.VT.weight - original_VT) ** 2)
        for i, U in enumerate(k_proj.U):
            reg = reg + 1e-4 * (torch.norm(U.weight - original_U[i]) ** 2)
        loss = mse + reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()

        with torch.no_grad():
            rel = (mse / var_target).item()
            loss_hist.append(mse.item())
            rel_hist.append(rel)
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
    u_changes = [
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
                "u_changes": [float(v) for v in u_changes],
            }, f, indent=2)

    return {
        "final_mse": float(final_mse.item()),
        "final_relative_error": float(final_rel),
        "vt_change": float(vt_change),
        "u_changes": [float(v) for v in u_changes],
        "loss_history": loss_hist,
        "relative_history": rel_hist,
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
    summary = vt_u_alignment_on_real(model, args.layer_idx, train_hidden_states, args.num_steps, save_dir)

    # Evaluate on eval split (test)
    eval_hidden_states = capture_hidden_states(model, tokenizer, args.device, args.layer_idx, args.batch_size, args.seq_len, split=args.eval_split)
    eval_rel = evaluate_on_hidden_states(model, args.layer_idx, eval_hidden_states)

    print("\n=== Alignment Summary (Real Weights) ===")
    print(f"Layer: {args.layer_idx}")
    print(f"Train Final MSE: {summary['final_mse']:.6f}")
    print(f"Train Final Relative Error: {summary['final_relative_error']*100:.3f}%")
    print(f"Eval Relative Error: {eval_rel*100:.3f}%")
    print(f"VT change: {summary['vt_change']:.4f}")
    print(f"U changes (avg): {sum(summary['u_changes'])/len(summary['u_changes']):.4f}")


if __name__ == "__main__":
    main()
