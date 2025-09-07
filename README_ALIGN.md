# Palu/Hack 单层对齐与评测说明

本文档介绍三个脚本及其使用方式：
- run_ppl_svd_finetune.py：单层隔离微调并注入整模型评测 PPL（直接PPL优化）
- test_rope_alignment_vt_u_real.py：在真实模型上验证 VT+U 同时优化的可行性（层内对齐实验）
- test_rope_alignment_vt_u.py：合成数据上验证 VT+U 同时优化（数学可行性实验）

## 一、背景与术语
- PALU Attention：RoPE(x@U@V)。
- HACK Attention：RoPE(x@U)@V。
- 目标：仅将第 k 层切换到 HACK，其余层保留 PALU，通过微调该层 K 路的 VT 和 U，使 HACK 与 PALU 在该层的行为尽可能一致，从而将端到端 PPL 损失降到最小。

## 二、脚本说明

### 1) run_ppl_svd_finetune.py
功能：
- 评测 baseline PPL（全 PALU，prefill-only，无 cache）。
- 隔离第 k 层进行直接PPL优化训练：构建简化模型（embed + ln + attn + post_ln + lm_head），直接优化HACK模式的PPL性能。
- 支持训练多个组件：Wq、Wo、K、V等，通过参数控制。
- 将该层权重注入回整模型，设置评测模式（仅第 k 层 HACK，其余 PALU），再评测 PPL（prefill-only）。

关键参数：
- `--layer_idx`：目标层索引
- `--isolate_layer`：启用隔离微调（必需）
- `--eval_mode`：`palu_all|hack_all|hack_layer_only`（常用：hack_layer_only）
- 训练组件选择：
  - `--train_wq`：训练 q_proj
  - `--train_wo`：训练 o_proj  
  - `--train_k`：训练 k_proj（VT和U）
  - `--train_v`：训练 v_proj（VT和U）
- 正则化参数：
  - `--reg_lora`：LoRA参数正则化系数（默认5e-5）
  - `--reg_dense`：密集参数正则化系数（默认1e-6）
- 环境变量：
  - `ATTN_DTYPE=float32|bfloat16|float16`（默认 float32 更稳）
  - `MBS=1|2...`（micro-batch，用于控制显存）

推荐命令：
```bash
ATTN_DTYPE=float32 MBS=2 \
python run_ppl_svd_finetune.py \
  --model_path Meta-Llama-3-8B-Instruct_ratio-0.7_gs-4-fisher_uniform-svd \
  --layer_idx 0 --num_steps 10000 --batch_size 8 --seq_len 128 --lr 5e-4 \
  --dataset wikitext2 --seqlen 2048 \
  --isolate_layer --rounds 1 \
  --eval_mode hack_layer_only \
  --train_wq --train_wo --train_k --train_v \
  --reg_lora 5e-5 --reg_dense 1e-6
```

说明：
- baseline：全 PALU（palu_all），prefill-only，无 KV cache。
- 训练：直接优化HACK模式的PPL性能，使用简化模型进行端到端训练。
- 注入评测：仅第 `layer_idx` 层 HACK，其余 PALU，prefill-only。

### 2) test_rope_alignment_vt_u_real.py
- 从真实模型 hook 指定层的 `hidden_states`，在该层上进行 VT+U 同时优化（PALU 作为目标，拟合 HACK），
- 仅层内实验（不注入全模型），用于观察相对误差和参数变化。

示例：
```bash
python test_rope_alignment_vt_u_real.py \
  --model_path Meta-Llama-3-8B-Instruct_ratio-0.7_gs-4-fisher_uniform-svd \
  --layer_idx 0 --batch_size 8 --seq_len 128 --num_steps 2000 \
  --save_aligned
```

### 3) test_rope_alignment_vt_u.py
- 合成数据下，验证 VT+U 同时优化能否使 HACK 逼近 PALU（数学可行性）。

示例：
```bash
python test_rope_alignment_vt_u.py --batch_size 8 --seq_len 128 --num_steps 3000 --device cuda
```

## 三、当前状态概述
- 已支持：
  - 直接PPL优化：构建简化模型直接优化HACK模式的PPL性能，避免中间对齐的语义差异。
  - 灵活的训练组件选择：支持训练Wq、Wo、K、V等多个组件。
  - 隔离训练模式：在GPU上进行端到端训练，避免内存问题。
  - 评测时可设置仅第 k 层 HACK，其余 PALU；prefill-only 评测。
  - 稳定的数据类型（float32）、finite 检查、梯度裁剪、微批累加、正则化。
- 优势：
  - 直接优化目标指标（PPL），避免中间对齐的语义损失。
  - 支持多组件联合训练，提高优化效果。
  - 简化的训练流程，减少调试复杂度。
