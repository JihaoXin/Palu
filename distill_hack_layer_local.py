#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本地蒸馏：仅第 k 层切换到 HACK，其余 PALU，使用 logits KL（prefill-only）+ 可选 postnorm MSE 进行端到端蒸馏。

做法：
- Teacher: 全 PALU（冻结）
- Student: 仅第 k 层 HACK（仅训该层 K 路 VT/U），其余与 Teacher 同权重、PALU
- 数据：wikitext2 短序列，prefill-only，无 cache
- Loss: KL(student_logits || teacher_logits) * τ^2 [+ λ·postnorm_mse]
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)

from kernel.palu_attention import LlamaPaluAttention as KernelPaluAttention
from transformers.models.llama.modeling_llama import LlamaRMSNorm


def set_rope_mode(model, mode: str, target_layer_idx: int | None = None):
    for li, layer in enumerate(model.model.layers):
        if mode == "palu_all":
            layer.self_attn.rope_latent = False
        elif mode == "hack_all":
            layer.self_attn.rope_latent = True
        elif mode == "hack_layer_only":
            layer.self_attn.rope_latent = (li == (target_layer_idx or -1))
        else:
            raise ValueError(f"Unknown eval mode: {mode}")


def build_student_from_teacher(teacher: AutoModelForCausalLM):
    """Clone a student from teacher (same weights)."""
    from copy import deepcopy
    student = deepcopy(teacher)
    return student


def kl_divergence(student_logits, teacher_logits, temperature: float = 2.0):
    T = temperature
    s = nn.functional.log_softmax(student_logits / T, dim=-1)
    t = nn.functional.softmax(teacher_logits / T, dim=-1)
    return nn.functional.kl_div(s, t, reduction='batchmean') * (T * T)


def maybe_postnorm_mse(model, hidden_states, layer_idx: int, y_palu, y_hack):
    post_ln: LlamaRMSNorm = model.model.layers[layer_idx].post_attention_layernorm
    z_palu = post_ln(hidden_states + y_palu)
    z_hack = post_ln(hidden_states + y_hack)
    return nn.functional.mse_loss(z_hack.float(), z_palu.float())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--layer_idx", type=int, default=0)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--lambda_postnorm", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load teacher
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Teacher: palu_all, no grad
    set_rope_mode(teacher, "palu_all")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Student: clone, then hack only at layer_idx; optimize only that layer's K VT/U
    student = build_student_from_teacher(teacher)
    set_rope_mode(student, "hack_layer_only", target_layer_idx=args.layer_idx)
    student.train()
    # float32 for stability at the layer
    attn: KernelPaluAttention = student.model.layers[args.layer_idx].self_attn  # type: ignore
    attn = attn.to(torch.float32)
    student.model.layers[args.layer_idx].self_attn = attn

    params = [attn.k_proj.VT.weight] + [u.weight for u in attn.k_proj.U]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-6)

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    def sample_batch(bs, sl):
        texts = []
        while len(texts) < bs:
            t = ds[int(torch.randint(0, len(ds), (1,)).item())]["text"].strip()
            if len(t) > 0:
                texts.append(t)
        tok = tokenizer(texts, max_length=sl, truncation=True, padding="max_length", return_tensors="pt")
        return tok.input_ids.to(device)

    # prefill-only eval (no cache)
    teacher.config.use_cache = False
    student.config.use_cache = False

    for step in range(args.steps):
        input_ids = sample_batch(args.batch_size, args.seq_len)

        with torch.no_grad():
            t_out = teacher(input_ids=input_ids, use_cache=False)
            t_logits = t_out.logits.to(torch.float32)

        s_out = student(input_ids=input_ids, use_cache=False)
        s_logits = s_out.logits.to(torch.float32)

        kd = kl_divergence(s_logits, t_logits, temperature=args.temperature)

        # Optional auxiliary postnorm loss at target layer (on current batch first token window)
        aux = torch.tensor(0.0, device=s_logits.device)
        if args.lambda_postnorm > 0:
            # 取 student 的该层输入（用 Hook 更精确；此处用近似：将 embeddings + ln 作为 hidden_states）
            with torch.no_grad():
                emb = student.model.embed_tokens(input_ids).to(torch.float32)
                ln0: LlamaRMSNorm = student.model.layers[args.layer_idx].input_layernorm
                hs0 = ln0(emb)
                # Teacher/Student 当层输出（PALU/HACK）
                t_layer = teacher.model.layers[args.layer_idx].self_attn
                s_layer = student.model.layers[args.layer_idx].self_attn
                # Build RoPE on-the-fly
                from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
                rotary = LlamaRotaryEmbedding(config=t_layer.config).to(hs0.device)
                pos = torch.arange(args.seq_len, device=hs0.device).unsqueeze(0).expand(hs0.size(0), -1)
                cos, sin = rotary(hs0, pos)
            # 前向（注意 student 的该层为 float32）
            with torch.no_grad():
                t_layer.rope_latent = False
                y_palu = t_layer(hs0, position_ids=pos, position_embeddings=(cos, sin), use_cache=False)[0].to(torch.float32)
            s_layer.rope_latent = True
            y_hack = s_layer(hs0, position_ids=pos, position_embeddings=(cos, sin), use_cache=False)[0].to(torch.float32)
            aux = maybe_postnorm_mse(student, hs0, args.layer_idx, y_palu, y_hack)

        loss = kd + args.lambda_postnorm * aux
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 0.05)
        optimizer.step()

        if step % max(1, args.steps // 20) == 0:
            print(f"Step {step}: KD={kd.item():.6f}, AUX={aux.item():.6f}, LOSS={loss.item():.6f}")

    # 保存学生模型（可选）
    out_dir = os.path.join(CUR_DIR, f"student_hack_layer{args.layer_idx}")
    os.makedirs(out_dir, exist_ok=True)
    student.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved student to: {out_dir}")


if __name__ == "__main__":
    main()


