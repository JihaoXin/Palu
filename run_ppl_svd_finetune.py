#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single-layer SVD + VT/U finetune + PPL evaluation.

Usage example:
  python run_ppl_svd_finetune.py \
    --model_path Meta-Llama-3-8B-Instruct_ratio-0.7_gs-4-fisher_uniform-svd \
    --layer_idx 0 --group_size 4 --rank_ratio 0.7 \
    --num_steps 1000 --batch_size 8 --seq_len 128 --lr 1e-4 \
    --dataset wikitext2 --seqlen 2048
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Ensure local imports work
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)

from kernel.palu_attention import LlamaPaluAttention as KernelPaluAttention
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


def evaluate_ppl(model, tokenizer, dataset_name="wikitext2", split="test", seqlen=2048, device="cuda"):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = [ex["text"] for ex in dataset if ex["text"].strip()]
    text = "\n\n".join(texts)

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]

    model.eval()
    nlls = []
    with torch.no_grad():
        max_end = input_ids.shape[0] - seqlen - 1
        if max_end <= 0:
            return float("nan")
        for i in tqdm(range(0, max_end, seqlen), desc="Evaluating PPL"):
            batch = input_ids[i:i+seqlen].unsqueeze(0).to(device)
            target = input_ids[i+1:i+seqlen+1].unsqueeze(0).to(device)
            outputs = model(batch)
            logits = outputs.logits
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
                reduction='mean'
            )
            nlls.append(loss.item())
    ppl = float(np.exp(np.mean(nlls))) if nlls else float("nan")
    return ppl


def svd_decompose_layer(model, layer_idx, group_size=4, rank_ratio=0.7):
    layer = model.model.layers[layer_idx]
    original_attn = layer.self_attn

    # If already Palu attention, return as-is
    if isinstance(original_attn, KernelPaluAttention):
        print(f"Layer {layer_idx} is already PaluAttention; skip SVD and reuse it.")
        return original_attn, False

    # Otherwise, convert from standard LlamaAttention using factory (supports optional whitening)
    palu_attn = KernelPaluAttention.from_attention(
        module=original_attn,
        config=model.config,
        whiten=True,
    )
    layer.self_attn = palu_attn
    return palu_attn, True


def capture_hidden_states(model, tokenizer, device, layer_idx, batch_size=8, seq_len=128, num_batches=1):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    all_hidden_states = []
    all_position_ids = []

    def hook_fn(module, args, kwargs):
        hs = args[0] if args else kwargs.get('hidden_states')
        if hs is not None:
            all_hidden_states.append(hs.detach().float())
            pos = torch.arange(hs.shape[1], device=hs.device).unsqueeze(0).expand(hs.shape[0], -1)
            all_position_ids.append(pos.detach())
        return args, kwargs

    handle = model.model.layers[layer_idx].self_attn.register_forward_pre_hook(hook_fn, with_kwargs=True)
    model.eval()
    with torch.no_grad():
        for _ in range(num_batches):
            texts = []
            while len(texts) < batch_size:
                idx = np.random.randint(len(dataset))
                t = dataset[idx]["text"]
                if len(t.strip()) > 10:
                    texts.append(t)
            inputs = tokenizer(texts, max_length=seq_len, truncation=True, padding="max_length", return_tensors="pt").to(device)
            _ = model(input_ids=inputs.input_ids)
    handle.remove()
    if not all_hidden_states:
        raise RuntimeError("Failed to capture hidden states")
    hidden_states = torch.cat(all_hidden_states, dim=0)
    position_ids = torch.cat(all_position_ids, dim=0)
    return hidden_states, position_ids


def finetune_layer(model, tokenizer, layer_idx, num_steps=1000, batch_size=8, seq_len=128, lr=1e-4, device="cuda", num_batches_per_step=2):
    palu_attn = model.model.layers[layer_idx].self_attn
    original_vt_k = palu_attn.k_proj.VT.weight.data.clone()
    original_u_k = [u.weight.data.clone() for u in palu_attn.k_proj.U]

    params = [palu_attn.k_proj.VT.weight] + [u.weight for u in palu_attn.k_proj.U]
    optimizer = torch.optim.AdamW(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps)

    model.train()
    losses = []
    for step in tqdm(range(num_steps), desc="Finetuning"):
        hs_list = []
        pos_list = []
        for _ in range(num_batches_per_step):
            hs, pos = capture_hidden_states(model, tokenizer, device, layer_idx, batch_size, seq_len, num_batches=1)
            hs_list.append(hs)
            pos_list.append(pos)
        hidden_states = torch.cat(hs_list, dim=0).to(device).float()
        position_ids = torch.cat(pos_list, dim=0).to(device)

        # Build RoPE embeddings once (ensure dtype matches attention weights)
        attn_dtype = palu_attn.q_proj.weight.dtype
        rotary = LlamaRotaryEmbedding(config=model.config).to(device)
        bsz = hidden_states.shape[0]
        kv_heads = palu_attn.num_key_value_heads
        head_dim = palu_attn.head_dim
        dummy_key = torch.empty(bsz, kv_heads, seq_len, head_dim, device=device, dtype=attn_dtype)
        cos, sin = rotary(dummy_key, position_ids)
        cos = cos.to(attn_dtype)
        sin = sin.to(attn_dtype)

        hidden_states_cast = hidden_states.to(attn_dtype)

        palu_attn.rope_latent = False
        with torch.no_grad():
            target = palu_attn(hidden_states_cast, position_ids=position_ids, position_embeddings=(cos, sin), use_cache=False)[0].float()

        palu_attn.rope_latent = True
        pred = palu_attn(hidden_states_cast, position_ids=position_ids, position_embeddings=(cos, sin), use_cache=False)[0].float()

        mse = nn.functional.mse_loss(pred, target)
        vt_reg = 1e-3 * (palu_attn.k_proj.VT.weight - original_vt_k).pow(2).mean()
        u_reg = 5e-4 * sum((palu_attn.k_proj.U[i].weight - original_u_k[i]).pow(2).mean() for i in range(len(palu_attn.k_proj.U)))
        loss = mse + vt_reg + u_reg

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        if step % 200 == 0:
            rel = (torch.norm(pred - target) / (torch.norm(target) + 1e-12)).item()
            print(f"Step {step}: Loss={loss.item():.6f}, RelErr={rel:.6f}")

    palu_attn.rope_latent = False
    model.eval()
    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--layer_idx", type=int, default=0)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--rank_ratio", type=float, default=0.7)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dataset", type=str, default="wikitext2")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--save_model", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Single-layer SVD + finetune + PPL eval")
    print("=" * 80)

    print("\n1) Load base model ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n2) Evaluate baseline PPL ...")
    base_ppl = evaluate_ppl(model, tokenizer, args.dataset, "test", args.seqlen, device)
    print(f"Baseline PPL: {base_ppl:.4f}")

    print(f"\n3) SVD decompose layer {args.layer_idx} ...")
    palu_attn, replaced = svd_decompose_layer(model, args.layer_idx, args.group_size, args.rank_ratio)

    if replaced:
        print("\n4) PPL after SVD (HACK mode) ...")
        ppl_hack = evaluate_ppl(model, tokenizer, args.dataset, "test", args.seqlen, device)
        print(f"SVD(HACK) PPL: {ppl_hack:.4f} (Δ {(ppl_hack/base_ppl - 1)*100:+.2f}%)")
    else:
        ppl_hack = base_ppl
        print("\n4) Skip PPL after SVD: layer already Palu; reuse baseline.")

    print(f"\n5) Finetune layer {args.layer_idx} (opt VT+U) ...")
    _ = finetune_layer(model, tokenizer, args.layer_idx, args.num_steps, args.batch_size, args.seq_len, args.lr, device)

    print("\n6) PPL after finetune (PALU mode) ...")
    ppl_ft = evaluate_ppl(model, tokenizer, args.dataset, "test", args.seqlen, device)
    print(f"Finetuned PPL: {ppl_ft:.4f} (Δ {(ppl_ft/base_ppl - 1)*100:+.2f}%)")

    print("\n" + "=" * 80)
    print("Summary")
    print(f"Baseline PPL: {base_ppl:.4f}")
    print(f"After SVD(HACK) PPL: {ppl_hack:.4f}")
    print(f"After Finetune(PALU) PPL: {ppl_ft:.4f}")
    print("=" * 80)

    results = {
        "layer_idx": args.layer_idx,
        "group_size": args.group_size,
        "rank_ratio": args.rank_ratio,
        "baseline_ppl": float(base_ppl),
        "svd_hack_ppl": float(ppl_hack),
        "finetuned_ppl": float(ppl_ft),
        "delta_baseline_percent": float((ppl_ft/base_ppl - 1) * 100),
    }
    out_path = os.path.join(CUR_DIR, f"ppl_svd_finetune_layer{args.layer_idx}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out_path}")

    if args.save_model:
        save_dir = os.path.join(CUR_DIR, f"{Path(args.model_path).name}_svdL{args.layer_idx}_finetuned")
        print(f"Saving finetuned model to: {save_dir}")
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()


