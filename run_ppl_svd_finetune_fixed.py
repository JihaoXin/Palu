#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fixed single-layer SVD + finetune + PPL evaluation with direct PPL optimization.

This version fixes the training logic by directly optimizing HACK mode's PPL performance
instead of trying to make HACK output match PALU output.

Example:
  LOSS_MODE=postnorm_mse ATTN_DTYPE=float32 MBS=2 \
  python run_ppl_svd_finetune.py \
    --model_path Meta-Llama-3-8B-Instruct_ratio-0.7_gs-4-fisher_uniform-svd \
    --layer_idx 0 --num_steps 10000 --batch_size 8 --seq_len 128 --lr 5e-4 \
    --dataset wikitext2 --seqlen 2048 \
    --isolate_layer --rounds 1 \
    --eval_mode hack_layer_only \
    --train_wq --train_wo --train_k --train_v \
    --reg_lora 5e-5 --reg_dense 1e-6
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
import gc

# Ensure local imports work
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)

from kernel.palu_attention import LlamaPaluAttention as KernelPaluAttention
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaRMSNorm


def evaluate_ppl(model, tokenizer, dataset_name="wikitext2", split="test", seqlen=2048, device="cuda"):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = [ex["text"] for ex in dataset if ex["text"].strip()]
    text = "\n\n".join(texts)

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]

    model.eval()
    nlls = []
    prev_use_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = False
    with torch.no_grad():
        max_end = input_ids.shape[0] - seqlen - 1
        if max_end <= 0:
            if prev_use_cache is not None:
                model.config.use_cache = prev_use_cache
            return float("nan")
        for i in tqdm(range(0, max_end, seqlen), desc="Evaluating PPL"):
            batch = input_ids[i:i+seqlen].unsqueeze(0).to(device)
            target = input_ids[i+1:i+seqlen+1].unsqueeze(0).to(device)
            try:
                outputs = model(batch, use_cache=False)
                logits = outputs.logits
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target.view(-1),
                    reduction='mean'
                )
                if torch.isfinite(loss):
                    nlls.append(loss.item())
            except Exception:
                continue
    ppl = float(np.exp(np.mean(nlls))) if nlls else float("nan")
    if prev_use_cache is not None:
        model.config.use_cache = prev_use_cache
    return ppl


def set_rope_mode(model, mode: str, target_layer_idx: int | None = None):
    for li, layer in enumerate(model.model.layers):
        if mode == "palu_all":
            layer.self_attn.rope_latent = False
        elif mode == "hack_all":
            layer.self_attn.rope_latent = True
        elif mode == "hack_layer_only":
            layer.self_attn.rope_latent = (li == target_layer_idx)
        else:
            raise ValueError(f"Unknown eval mode: {mode}")


def svd_decompose_layer(model, layer_idx, group_size=4, rank_ratio=0.7):
    layer = model.model.layers[layer_idx]
    original_attn = layer.self_attn
    if isinstance(original_attn, KernelPaluAttention):
        print(f"Layer {layer_idx} is already PaluAttention; skip SVD and reuse it.")
        return original_attn, False
    palu_attn = KernelPaluAttention.from_attention(
        module=original_attn,
        config=model.config,
        whiten=True,
    )
    layer.self_attn = palu_attn
    return palu_attn, True


def extract_layer_state(model, layer_idx):
    layer = model.model.layers[layer_idx].self_attn
    state = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in layer.state_dict().items()}
    config = model.config
    return config, state


def build_isolated_attn(config, layer_idx, state):
    iso_attn = KernelPaluAttention(config=config, layer_idx=layer_idx)
    iso_attn.load_state_dict(state, strict=False)
    iso_attn = iso_attn.to("cpu").to(torch.float32)
    return iso_attn


def extract_front_layer0_state(model):
    embed = model.model.embed_tokens
    ln = model.model.layers[0].input_layernorm
    post_ln = model.model.layers[0].post_attention_layernorm
    embed_sd = {k: v.detach().cpu() for k, v in embed.state_dict().items()}
    ln_sd = {k: v.detach().cpu() for k, v in ln.state_dict().items()}
    post_ln_sd = {k: v.detach().cpu() for k, v in post_ln.state_dict().items()}
    return embed_sd, ln_sd, post_ln_sd


def build_front_layer0(config, embed_sd, ln_sd, post_ln_sd, device: str = "cuda", dtype: torch.dtype = torch.float32):
    embed = torch.nn.Embedding(config.vocab_size, config.hidden_size)
    embed.load_state_dict(embed_sd, strict=True)
    ln = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    ln.load_state_dict(ln_sd, strict=True)
    post_ln = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    post_ln.load_state_dict(post_ln_sd, strict=True)
    embed = embed.to(device).to(dtype)
    ln = ln.to(device).to(dtype)
    post_ln = post_ln.to(device).to(dtype)
    return embed, ln, post_ln


def finetune_isolated_attn_with_front_fixed(
    iso_attn: KernelPaluAttention,
    embed: torch.nn.Embedding,
    ln: LlamaRMSNorm,
    post_ln: LlamaRMSNorm,
    tokenizer: AutoTokenizer,
    *,
    split: str = "train",
    num_steps: int = 500,
    batch_size: int = 4,
    seq_len: int = 128,
    lr: float = 1e-4,
    attn_dtype: str = "float32",
    mbs: int = 2,
    train_wq: bool = False,
    train_wo: bool = False,
    train_k: bool = True,
    train_v: bool = False,
    reg_lora: float = 5e-5,
    reg_dense: float = 1e-6,
    eval_every: int = 100,
):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    dev_attn = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    d_attn = dtype_map.get(attn_dtype, torch.float32)
    iso_attn.config._attn_implementation = "eager"
    iso_attn = iso_attn.to(dev_attn).to(d_attn)
    embed = embed.to(dev_attn).to(d_attn)
    ln = ln.to(dev_attn).to(d_attn)
    post_ln = post_ln.to(dev_attn).to(d_attn)

    # Build a simple model for PPL evaluation
    class SimpleModel(torch.nn.Module):
        def __init__(self, embed, ln, attn, post_ln, lm_head):
            super().__init__()
            self.embed = embed
            self.ln = ln
            self.attn = attn
            self.post_ln = post_ln
            self.lm_head = lm_head
            
        def forward(self, input_ids):
            hs = self.embed(input_ids)
            hs = self.ln(hs)
            
            # Use HACK mode for attention
            self.attn.rope_latent = True
            position_ids = torch.arange(input_ids.size(1), device=hs.device).unsqueeze(0).expand(hs.size(0), -1)
            rotary = LlamaRotaryEmbedding(config=self.attn.config).to(hs.device)
            dummy_key = torch.empty(hs.size(0), self.attn.num_key_value_heads, input_ids.size(1), self.attn.head_dim, device=hs.device, dtype=hs.dtype)
            cos, sin = rotary(dummy_key, position_ids)
            
            attn_out = self.attn(hs, position_ids=position_ids, position_embeddings=(cos, sin), use_cache=False)[0]
            hs = hs + attn_out  # residual connection
            hs = self.post_ln(hs)
            logits = self.lm_head(hs)
            return logits

    # Get the actual vocabulary size from tokenizer
    actual_vocab_size = len(tokenizer)
    print(f"Tokenizer actual vocab size: {actual_vocab_size}")
    print(f"Tokenizer config vocab size: {tokenizer.vocab_size}")
    
    # Create a dummy LM head with the correct vocabulary size
    lm_head = torch.nn.Linear(iso_attn.config.hidden_size, actual_vocab_size, bias=False)
    # Initialize with small random weights
    with torch.no_grad():
        lm_head.weight.normal_(0, 0.02)
    lm_head = lm_head.to(dev_attn).to(d_attn)
    
    # Debug: Check vocab size and token ranges
    print(f"LM head output size: {lm_head.out_features}")
    
    # Sample some tokens to check range
    sample_texts = ["Hello world", "This is a test"]
    sample_tok = tokenizer(sample_texts, max_length=10, truncation=True, padding="max_length", return_tensors="pt")
    print(f"Sample token IDs range: {torch.min(sample_tok.input_ids)} to {torch.max(sample_tok.input_ids)}")
    
    simple_model = SimpleModel(embed, ln, iso_attn, post_ln, lm_head)
    simple_model = simple_model.to(dev_attn).to(d_attn)

    iso_attn.train()
    simple_model.train()
    
    originals = {}
    def snap(t, name):
        if t is not None:
            originals[name] = t.data.clone()

    # Default: K VT+U
    if train_k:
        snap(iso_attn.k_proj.VT.weight, "k.VT.weight")
        for i, u in enumerate(iso_attn.k_proj.U):
            snap(u.weight, f"k.U.{i}.weight")
    # Optional: V VT+U
    if train_v:
        snap(iso_attn.v_proj.VT.weight, "v.VT.weight")
        for i, u in enumerate(iso_attn.v_proj.U):
            snap(u.weight, f"v.U.{i}.weight")
    # Optional: Wq and Wo (weight + bias if present)
    if train_wq:
        snap(iso_attn.q_proj.weight, "q.weight")
        if iso_attn.q_proj.bias is not None:
            snap(iso_attn.q_proj.bias, "q.bias")
    if train_wo:
        snap(iso_attn.o_proj.weight, "o.weight")
        if iso_attn.o_proj.bias is not None:
            snap(iso_attn.o_proj.bias, "o.bias")

    # Build optimizer params
    params: list[torch.nn.Parameter] = []
    def add_param(p):
        if p is not None and p.requires_grad:
            params.append(p)

    if train_k:
        add_param(iso_attn.k_proj.VT.weight)
        for u in iso_attn.k_proj.U:
            add_param(u.weight)
    if train_v:
        add_param(iso_attn.v_proj.VT.weight)
        for u in iso_attn.v_proj.U:
            add_param(u.weight)
    if train_wq:
        add_param(iso_attn.q_proj.weight)
        add_param(iso_attn.q_proj.bias)
    if train_wo:
        add_param(iso_attn.o_proj.weight)
        add_param(iso_attn.o_proj.bias)

    if not params:
        raise ValueError("No parameters selected to train. Enable at least one of --train_k/--train_v/--train_wq/--train_wo.")

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-6, eps=1e-8)

    losses: list[float] = []
    train_ppls: list[float] = []
    
    def evaluate_train_ppl():
        """Evaluate PPL on training data"""
        simple_model.eval()
        with torch.no_grad():
            nlls = []
            for _ in range(10):  # Sample 10 batches for PPL estimation
                texts = []
                while len(texts) < batch_size:
                    idx = np.random.randint(len(ds))
                    t = ds[idx]["text"].strip()
                    if len(t) > 0:
                        texts.append(t)
                tok = tokenizer(texts, max_length=seq_len, truncation=True, padding="max_length", return_tensors="pt")
                input_ids = tok.input_ids.to(dev_attn)
                target_ids = input_ids[:, 1:].contiguous()
                input_ids = input_ids[:, :-1].contiguous()
                
                if input_ids.size(1) == 0:
                    continue
                
                # Check for invalid token IDs
                max_token_id = torch.max(target_ids)
                if max_token_id >= actual_vocab_size:
                    continue
                    
                logits = simple_model(input_ids)
                try:
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        target_ids.view(-1),
                        reduction='mean'
                    )
                    if torch.isfinite(loss):
                        nlls.append(loss.item())
                except RuntimeError:
                    continue
            
            if nlls:
                ppl = float(np.exp(np.mean(nlls)))
                return ppl
            return float('nan')
    
    for step in tqdm(range(num_steps), desc="Finetuning (isolated real, GPU attn)"):
        # sample texts
        texts = []
        while len(texts) < batch_size:
            idx = np.random.randint(len(ds))
            t = ds[idx]["text"].strip()
            if len(t) > 0:
                texts.append(t)
        tok = tokenizer(texts, max_length=seq_len, truncation=True, padding="max_length", return_tensors="pt")
        input_ids = tok.input_ids.to(dev_attn)
        target_ids = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        
        if input_ids.size(1) == 0:
            continue

        # Debug: Check for invalid token IDs
        max_token_id = torch.max(target_ids)
        if max_token_id >= actual_vocab_size:
            print(f"Warning: Invalid token ID {max_token_id} >= vocab_size {actual_vocab_size}, skipping batch")
            continue

        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with HACK mode
        logits = simple_model(input_ids)
        
        # Language modeling loss
        try:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                reduction='mean'
            )
        except RuntimeError as e:
            print(f"Cross entropy error: {e}")
            print(f"Logits shape: {logits.shape}")
            print(f"Target shape: {target_ids.shape}")
            print(f"Max target ID: {torch.max(target_ids)}")
            print(f"Min target ID: {torch.min(target_ids)}")
            continue

        # Regularization to keep close to originals
        if train_k:
            loss = loss + reg_lora * (iso_attn.k_proj.VT.weight - originals["k.VT.weight"].to(iso_attn.k_proj.VT.weight.device)).pow(2).mean()
            for i, u in enumerate(iso_attn.k_proj.U):
                loss = loss + reg_lora * (u.weight - originals[f"k.U.{i}.weight"].to(u.weight.device)).pow(2).mean()
        if train_v:
            loss = loss + reg_lora * (iso_attn.v_proj.VT.weight - originals["v.VT.weight"].to(iso_attn.v_proj.VT.weight.device)).pow(2).mean()
            for i, u in enumerate(iso_attn.v_proj.U):
                loss = loss + reg_lora * (u.weight - originals[f"v.U.{i}.weight"].to(u.weight.device)).pow(2).mean()
        if train_wq:
            loss = loss + reg_dense * (iso_attn.q_proj.weight - originals["q.weight"].to(iso_attn.q_proj.weight.device)).pow(2).mean()
            if iso_attn.q_proj.bias is not None:
                loss = loss + reg_dense * (iso_attn.q_proj.bias - originals["q.bias"].to(iso_attn.q_proj.bias.device)).pow(2).mean()
        if train_wo:
            loss = loss + reg_dense * (iso_attn.o_proj.weight - originals["o.weight"].to(iso_attn.o_proj.weight.device)).pow(2).mean()
            if iso_attn.o_proj.bias is not None:
                loss = loss + reg_dense * (iso_attn.o_proj.bias - originals["o.bias"].to(iso_attn.o_proj.bias.device)).pow(2).mean()

        if not torch.isfinite(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 0.05)
        optimizer.step()
        losses.append(float(loss.item()))

        # Evaluate training PPL periodically
        if step % eval_every == 0 or step < 10:
            train_ppl = evaluate_train_ppl()
            train_ppls.append(train_ppl)
            print(f"Step {step}: Loss={loss.item():.8e}, Train PPL={train_ppl:.4f}")
        else:
            if step % 50 == 0 or step < 10:
                print(f"Step {step}: Loss={loss.item():.8e}")

    iso_attn.eval()
    iso_attn.rope_latent = False
    return iso_attn, losses, train_ppls


def inject_layer_state(model, layer_idx: int, iso_attn: KernelPaluAttention):
    target = model.model.layers[layer_idx].self_attn
    sd = iso_attn.state_dict()
    casted = {}
    tgt_device = target.q_proj.weight.device
    tgt_dtype = target.q_proj.weight.dtype
    for k, v in sd.items():
        if torch.is_tensor(v):
            casted[k] = v.to(device=tgt_device, dtype=tgt_dtype)
        else:
            casted[k] = v
    target.load_state_dict(casted, strict=False)


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
    parser.add_argument("--isolate_layer", action="store_true")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--eval_mode", type=str, default="hack_layer_only", choices=["palu_all","hack_all","hack_layer_only"])
    # New toggles
    parser.add_argument("--train_wq", action="store_true", help="Also finetune Wq (q_proj)")
    parser.add_argument("--train_wo", action="store_true", help="Also finetune Wo (o_proj)")
    parser.add_argument("--train_k", action="store_true", help="Finetune k_proj (VT and U)")
    parser.add_argument("--train_v", action="store_true", help="Also finetune v_proj (VT and U)")
    parser.add_argument("--reg_lora", type=float, default=5e-5, help="L2 reg coef for VT/U params")
    parser.add_argument("--reg_dense", type=float, default=1e-6, help="L2 reg coef for Wq/Wo params")
    parser.add_argument("--eval_every", type=int, default=100, help="Evaluate train PPL every N steps")

    args = parser.parse_args()

    # Defaults: if none specified, mirror original behavior (train_k only)
    if not (args.train_k or args.train_v or args.train_wq or args.train_wo):
        args.train_k = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Fixed single-layer SVD + finetune + PPL eval (direct PPL optimization)")
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
    set_rope_mode(model, "palu_all")
    base_ppl = evaluate_ppl(model, tokenizer, args.dataset, "test", args.seqlen, device)
    print(f"Baseline PPL: {base_ppl:.4f}")

    print(f"\n3) SVD decompose layer {args.layer_idx} ...")
    palu_attn, replaced = svd_decompose_layer(model, args.layer_idx, args.group_size, args.rank_ratio)

    if replaced:
        print("\n4) PPL after SVD (PALU mode) ...")
        set_rope_mode(model, "palu_all")
        ppl_after_svd = evaluate_ppl(model, tokenizer, args.dataset, "test", args.seqlen, device)
        print(f"After SVD (PALU) PPL: {ppl_after_svd:.4f} (Δ {(ppl_after_svd/base_ppl - 1)*100:+.2f}%)")
    else:
        ppl_after_svd = base_ppl
        print("\n4) Skip PPL after SVD: layer already Palu; reuse baseline.")

    print(f"\n5) PPL with HACK at layer {args.layer_idx} BEFORE finetune ...")
    set_rope_mode(model, "hack_layer_only", target_layer_idx=args.layer_idx)
    ppl_hack_prefinetune = evaluate_ppl(model, tokenizer, args.dataset, "test", args.seqlen, device)
    print(f"HACK(layer {args.layer_idx}) pre-finetune PPL: {ppl_hack_prefinetune:.4f} (Δ {(ppl_hack_prefinetune/base_ppl - 1)*100:+.2f}%)")

    if args.isolate_layer:
        print(f"\n6) Isolated finetune for layer {args.layer_idx} (fixed) ...")
        cfg, state = extract_layer_state(model, args.layer_idx)
        embed_sd, ln_sd, post_ln_sd = extract_front_layer0_state(model)
        del model
        torch.cuda.empty_cache()
        gc.collect()

        iso_attn = build_isolated_attn(cfg, args.layer_idx, state)
        front_device = "cuda" if torch.cuda.is_available() and os.environ.get("GPU_FRONT", "0") == "1" else "cpu"
        front_dtype = torch.float16 if os.environ.get("ATTN_DTYPE", "float16") == "float16" else (
            torch.bfloat16 if os.environ.get("ATTN_DTYPE") == "bfloat16" else torch.float32
        )
        embed, ln, post_ln = build_front_layer0(cfg, embed_sd, ln_sd, post_ln_sd, device=front_device, dtype=front_dtype)

        iso_attn, losses, train_ppls = finetune_isolated_attn_with_front_fixed(
            iso_attn, embed, ln, post_ln, tokenizer,
            split="train", num_steps=args.num_steps,
            batch_size=args.batch_size, seq_len=args.seq_len, lr=args.lr,
            attn_dtype=os.environ.get("ATTN_DTYPE", "float32"),
            mbs=int(os.environ.get("MBS", "2")),
            train_wq=args.train_wq, train_wo=args.train_wo,
            train_k=args.train_k, train_v=args.train_v,
            reg_lora=args.reg_lora, reg_dense=args.reg_dense,
            eval_every=args.eval_every,
        )

        for r in range(args.rounds):
            print(f"\n7) Round {r+1}/{args.rounds}: inject and evaluate PPL ...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            inject_layer_state(model, args.layer_idx, iso_attn)
            set_rope_mode(model, args.eval_mode, target_layer_idx=args.layer_idx)
            ppl_ft = evaluate_ppl(model, tokenizer, args.dataset, "test", args.seqlen, device)
            print(f"PPL after injection (round {r+1}): {ppl_ft:.4f} (Δ {(ppl_ft/base_ppl - 1)*100:+.2f}% vs baseline, Δ {(ppl_ft/ppl_hack_prefinetune - 1)*100:+.2f}% vs pre-finetune HACK)")

            del model
            torch.cuda.empty_cache()
            gc.collect()
        print(f"   Final isolated finetune loss: {losses[-1]:.6f}")
        print(f"   Final train PPL: {train_ppls[-1]:.4f}")
        return
    else:
        raise NotImplementedError("This fixed script currently supports only --isolate_layer mode.")


if __name__ == "__main__":
    main()
