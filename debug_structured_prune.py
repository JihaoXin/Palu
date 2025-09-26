# %% [markdown]
#  # 配置与导入

# %%
import copy
import json
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable
import logging


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

import lm_eval
from kernel.palu_attention import apply_rotary_pos_emb
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from lm_eval.utils import make_table




# %% [markdown]
#  ## 结构化线性层定义

# %%
class StructuredPrunedLinear(nn.Module):
    """
    线性层裁剪后保留核心投影 + 可训练的后处理矩阵（初始为单位阵）。
    core: 核心投影层U, 保持不变
    post: 后处理矩阵V, 可训练
    keep_indices: 保留的列索引
    """

    def __init__(self, core: nn.Linear, post: nn.Linear, keep_indices: torch.Tensor):
        super().__init__()
        self.core = core
        self.post = post
        self.register_buffer("keep_indices", keep_indices, persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        latent = self.core(hidden_states)
        return self.post(latent)

    def full_weight(self) -> torch.Tensor:
        return self.post.weight @ self.core.weight

    def full_bias(self) -> torch.Tensor | None:
        if self.core.bias is None:
            return None
        return self.post.weight @ self.core.bias

    def frozen_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.core.parameters())

    def finetune_parameters(self) -> list[torch.nn.Parameter]:
        return [self.post.weight]

    @classmethod
    def from_dumped(cls, original_linear: nn.Linear, state_dict: dict[str, torch.Tensor], prefix: str):
        """Reconstruct a StructuredPrunedLinear from a saved state_dict entry."""
        weight_key = f"{prefix}core.weight"
        bias_key = f"{prefix}core.bias"
        if weight_key not in state_dict:
            raise KeyError(f"state_dict 中缺少 {weight_key}，无法恢复结构化线性层")

        latent_dim = state_dict[weight_key].shape[0]
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        dtype = original_linear.weight.dtype
        device = original_linear.weight.device
        has_bias = bias_key in state_dict

        dummy_keep = torch.arange(latent_dim, device=device, dtype=torch.long)
        core = nn.Linear(in_features, latent_dim, bias=has_bias, dtype=dtype, device=device)
        post = nn.Linear(latent_dim, out_features, bias=False, dtype=dtype, device=device)
        return cls(core, post, dummy_keep)



# %% [markdown]
#  ## 超参与路径配置

# %%
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logging.getLogger("lm-eval").setLevel(logging.CRITICAL)
logging.getLogger("lm-eval").propagate = False

MODEL_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"
DATASET_NAME = "wikitext-2-raw-v1"
KEEP_RATIO = 0.7
IMPORTANCE_BATCHES = 24
IMPORTANCE_BATCH_SIZE = 8
IMPORTANCE_SEQ_LEN = 2048
MASK_FINETUNE_STEPS = 100
EVAL_EVERY_MASK = 100
MASK_FINETUNE_LR = 2e-4
MASK_LAMBDA_REG = 1e-5
SEQ_LEN = 2048
BATCH_SIZE = 8
MAX_TEST_WINDOWS = 10
PALU_MODEL_DIR = Path(os.environ.get( "PALU_MODEL_DIR", "Meta-Llama-3-8B-Instruct_ratio-0.7_gs-4-fisher_uniform-whiten",)).resolve()
PALU_CONFIG_PATH = PALU_MODEL_DIR / "config.json"
if not PALU_CONFIG_PATH.is_file():
    raise FileNotFoundError(f"未找到 PaLU 模型配置文件: {PALU_CONFIG_PATH}")
with PALU_CONFIG_PATH.open("r", encoding="utf-8") as _cfg_fh:
    PALU_CONFIG = json.load(_cfg_fh)
PALU_HEADWISE_RANKS: dict[str, list[int]] = PALU_CONFIG.get("head_wise_ranks", {})


@dataclass(frozen=True)
class PruneHyperParams:
    keep_ratio: float = KEEP_RATIO
    importance_batches: int = IMPORTANCE_BATCHES
    importance_batch_size: int = IMPORTANCE_BATCH_SIZE
    importance_seq_len: int = IMPORTANCE_SEQ_LEN
    mask_finetune_steps: int = MASK_FINETUNE_STEPS
    eval_every_mask: int = EVAL_EVERY_MASK
    mask_finetune_lr: float = MASK_FINETUNE_LR
    mask_lambda_reg: float = MASK_LAMBDA_REG
    seq_len: int = SEQ_LEN
    batch_size: int = BATCH_SIZE
    max_test_windows: int = MAX_TEST_WINDOWS

def print_meta_info(hparams: PruneHyperParams, extra: dict | None = None):
    print("\n===== Meta Info & Hyperparameters =====")
    meta_fields = { "MODEL_PATH": MODEL_PATH, "DATASET_NAME": DATASET_NAME, **asdict(hparams), "PALU_MODEL_DIR": str(PALU_MODEL_DIR)}
    if extra:
        meta_fields.update(extra)
    for key, value in meta_fields.items():
        print(f"{key}: {value}")

# %% [markdown]
#  ## Dump 目录管理

# %%
def ensure_dump_dir(path: str) -> Path | None:
    """确保 dump 目录存在，失败时返回 None。"""
    dump_path = Path(path)
    try:
        dump_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"创建 {path} 目录失败: {exc}")
        return None
    return dump_path


def get_layer_dump_path(dump_dir: Path | None, layer_id: int) -> Path | None:
    """根据层号给出持久化文件路径。"""
    if dump_dir is None:
        return None
    suffix = f"{MODEL_PATH.split('/')[-1]}_layer{layer_id}_structured.pt"
    return dump_dir / suffix



# %% [markdown]
#  ## PaLU 权重加载工具

# %%
_palu_weight_map: dict[str, str] | None = None


def _get_palu_weight_map() -> dict[str, str]:
    """读取 PaLU safetensor 索引，只加载一次。"""
    global _palu_weight_map
    if _palu_weight_map is None:
        index_path = PALU_MODEL_DIR / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(
                f"未找到 PaLU 模型索引文件: {index_path}. 请确认 PALU_MODEL_DIR 设置正确。"
            )
        with index_path.open("r", encoding="utf-8") as fh:
            index_json = json.load(fh)
        weight_map = index_json.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError("PaLU 模型索引格式异常，缺少 weight_map 字段。")
        _palu_weight_map = weight_map
    return _palu_weight_map


def _load_palu_tensor(param_key: str) -> torch.Tensor:
    """从磁盘按键加载单个张量，避免一次性读入全部权重。"""
    weight_map = _get_palu_weight_map()
    shard = weight_map.get(param_key)
    if shard is None:
        raise KeyError(f"PaLU 模型中未找到参数 {param_key}")
    shard_path = PALU_MODEL_DIR / shard
    with safe_open(str(shard_path), framework="pt", device="cpu") as fh:
        return fh.get_tensor(param_key)


@lru_cache(maxsize=None)
def _get_palu_v_tensors(layer_id: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    prefix = f"model.layers.{layer_id}.self_attn.v_proj"
    ranks = PALU_HEADWISE_RANKS.get(prefix)
    if ranks is None:
        raise KeyError(f"PaLU 配置中缺少 {prefix} 的 head_wise_ranks 信息")

    vt_key = f"{prefix}.VT.weight"
    vt = _load_palu_tensor(vt_key)
    vt_dtype = vt.dtype

    blocks = []
    offset = 0
    for idx, rank in enumerate(ranks):
        u_key = f"{prefix}.U.{idx}.weight"
        u_weight = _load_palu_tensor(u_key)
        vt_chunk = vt[offset:offset + rank, :]
        offset += rank

        block = u_weight.to(torch.float32) @ vt_chunk.to(torch.float32)
        blocks.append(block.to(vt_dtype))

    weight = torch.cat(blocks, dim=0).contiguous()
    return weight, None


def build_palu_v_linear(layer_id: int, *, dtype: torch.dtype, device: torch.device) -> nn.Linear:
    """构造冻结的 V 投影层，保持与 PaLU 重建结果一致。"""
    weight_cpu, bias_cpu = _get_palu_v_tensors(layer_id)
    out_features, in_features = weight_cpu.shape
    has_bias = bias_cpu is not None
    linear = nn.Linear(in_features, out_features, bias=has_bias, dtype=dtype, device=device)
    linear.weight.data.copy_(weight_cpu.to(device=device, dtype=dtype))
    if has_bias and bias_cpu is not None:
        linear.bias.data.copy_(bias_cpu.to(device=device, dtype=dtype))
    for param in linear.parameters():
        param.requires_grad_(False)
    return linear

# %% [markdown]
#  ## Dump 恢复

# %%
def restore_layer_from_dump(
    layer_id: int,
    hack_attn: nn.Module,
    original_attn: nn.Module,
    dump_dir: Path | None,
) -> bool:
    """尝试从本地缓存恢复裁剪结果，节省重复计算。"""
    dump_path = get_layer_dump_path(dump_dir, layer_id)
    if dump_path is None or not dump_path.is_file():
        return False

    try:
        saved_state = torch.load(str(dump_path), map_location="cpu")
    except OSError as exc:
        print(f"加载第 {layer_id} 层裁剪结果失败，将重新计算。原因: {exc}")
        return False

    try:
        k_proj_restored = StructuredPrunedLinear.from_dumped(original_attn.k_proj, saved_state, "k_proj.")
    except KeyError as exc:
        print(f"第 {layer_id} 层保存的权重缺少必要的 k_proj 信息: {exc}，将重新计算。")
        return False

    dtype = hack_attn.v_proj.weight.dtype
    device = hack_attn.v_proj.weight.device
    v_proj_restored = build_palu_v_linear(layer_id, dtype=dtype, device=device)

    hack_attn.k_proj = k_proj_restored
    hack_attn.v_proj = v_proj_restored

    filtered_state = {k: v for k, v in saved_state.items() if not k.startswith("v_proj.")}
    hack_attn.load_state_dict(filtered_state, strict=False)

    hack_attn.to(dtype=torch.float16)
    for param in hack_attn.v_proj.parameters():
        param.requires_grad_(False)

    print(f"检测到第 {layer_id} 层已有裁剪结果，已从 {dump_path} 恢复（V 来自 PaLU 模型）。")
    return True



# %% [markdown]
#  ## 评估与生成函数

# %%
def evaluate_ppl(model, seqlen=2048, device="cuda", nsamples=None, input_ids=None):
    """基于给定 token 序列计算模型的困惑度。"""
    if input_ids is None:
        raise ValueError("evaluate_ppl 需要预先提供 input_ids。")
    assert input_ids.dim() == 2, "input_ids 必须是二维张量"

    if isinstance(device, str):
        device = torch.device(device)

    nsamples = input_ids.numel() // seqlen if nsamples is None else nsamples
    model.eval()

    nlls = []
    loss_fct = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i in tqdm(range(nsamples), disable=True):
            batch = input_ids[:, (i * seqlen):((i + 1) * seqlen)].to(device)
            outputs = model(batch)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, (i * seqlen):((i + 1) * seqlen)][:, 1:].to(device)
            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1)
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seqlen)).item()
    return ppl


def example_generation(model, tokenizer, device):
    """给出一个固定提示，方便直观检查生成效果。"""
    prompt = "Why research is so hard?"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=False,
        )

    gen_text = tokenizer.decode(gen_ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

    print("=== Example Prompt ===")
    print(prompt)
    print(gen_text)


def zero_shot_eval(model, tokenizer, tasks, *,
                   batch_size: int = 8,
                   max_length: int = 4096,
                   limit: int | None = None,
                   return_full: bool = False):
    """调用 lm-eval 对指定任务做 zero-shot 评估。"""
    task_list = [t.strip() for t in tasks.split(",")] if isinstance(tasks, str) else list(tasks)

    model.seqlen = max_length
    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, add_bos_token=False, batch_size=batch_size)
    task_manager = TaskManager()

    with torch.no_grad():
        results = lm_eval.simple_evaluate(
            model=lm_obj,
            tasks=task_list,
            task_manager=task_manager,
            log_samples=False,
            limit=limit,
        )

    print(make_table(results))
    return results if return_full else results["results"]



# %% [markdown]
#  ## 数据与模型加载

# %%
def load_model_and_tokenizer(model_path: str):
    """加载基础模型与分词器，默认让 pad token 回退到 eos。"""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def prepare_datasets(tokenizer, dataset_name: str):
    """取 wikiText 训练/测试集，并将测试集拼接成一段长文本用于 PPL。"""
    train_dataset = load_dataset("wikitext", dataset_name, split="train")
    test_dataset = load_dataset("wikitext", dataset_name, split="test")
    test_ids_all = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt").input_ids
    return train_dataset, test_ids_all


def build_rotary_embedding(model, device: torch.device) -> LlamaRotaryEmbedding:
    """复用 transformers 内部配置，构造全精度 RoPE 对象。"""
    rotary = LlamaRotaryEmbedding(config=model.model.layers[0].self_attn.config)
    return rotary.to(device=device, dtype=torch.float16)



# %% [markdown]
#  ## 剪枝与对齐函数

# %%
def reset_model(model, original_layers, layer_ids):
    if isinstance(layer_ids, int): layer_ids = [layer_ids]
    for layer_id in layer_ids:
        model.model.layers[layer_id] = copy.deepcopy(original_layers[layer_id])
    print(f"已将第 {layer_ids} 层注意力恢复为原始权重")
    return


def sample_batch(dataset, tokenizer, *, batch_size=8, seq_len=128, device="cuda"):
    # 随机抽取若干段文本，并统一截断/补齐长度，避免每次前向时 pad 规则不一致
    texts = []
    while len(texts) < batch_size:
        t = dataset[np.random.randint(len(dataset))]["text"].strip()
        if t:
            texts.append(t)
    tok = tokenizer(
        texts,
        max_length=seq_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return tok.input_ids.to(device)


@torch.no_grad()
def capture_layer_input_hidden_states(model, input_ids, layer_id, device):
    captured = {}

    class _StopForward(Exception):
        pass

    def _pre_hook(module, args):
        captured[layer_id] = args[0].detach()
        raise _StopForward()

    handle = model.model.layers[layer_id].register_forward_pre_hook(_pre_hook)
    assert model.config.use_cache is False, "use_cache 必须为 False"
    model.eval()
    try:
        _ = model(input_ids.to(device))
    except _StopForward:
        pass
    finally:
        handle.remove()

    if layer_id not in captured:
        raise RuntimeError(f"未捕获到第 {layer_id} 层的 hidden states")

    return captured[layer_id]


def rope_pair_to_indices(head_dim: int, pair_idx: int) -> tuple[int, int]:
    half = head_dim // 2
    return pair_idx, pair_idx + half


def compute_pair_scores(
    model,
    tokenizer,
    train_dataset,
    rotary_emb,
    layer_id,
    attn_module,
    *,
    batches=8,
    batch_size=8,
    seq_len=512,
    device="cuda",
):
    """计算给定注意力层的 RoPE 二元组能量，用于筛选重要频率维度。"""
    head_dim = attn_module.head_dim
    num_kv = attn_module.k_proj.weight.shape[0] // head_dim
    pair_scores = torch.zeros(num_kv, head_dim // 2, device=device, dtype=torch.float32)
    total = 0

    for _ in tqdm(range(batches), desc=f"RoPE pair scoring @layer{layer_id}", disable=True):
        # 按批采样语料，并记录该层输入的 hidden states，估计频率能量
        input_ids = sample_batch(
            train_dataset,
            tokenizer,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
        )
        hs = capture_layer_input_hidden_states(model, input_ids, layer_id, device)
        hs = model.model.layers[layer_id].input_layernorm(hs)
        B, T, _ = hs.shape
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        dummy = torch.empty(B, num_kv, T, head_dim, device=device, dtype=hs.dtype)
        cos, sin = rotary_emb(dummy, pos_ids)

        weight = attn_module.k_proj.weight.to(device=device, dtype=hs.dtype)
        bias = attn_module.k_proj.bias
        bias = bias.to(device=device, dtype=hs.dtype) if bias is not None else None
        k_full = F.linear(hs, weight, bias)
        k_full = k_full.view(B, T, num_kv, head_dim).transpose(1, 2)
        _, k_rope = apply_rotary_pos_emb(None, k_full, cos, sin)
        k_rope_fp32 = k_rope.to(torch.float32)
        first_half, second_half = torch.chunk(k_rope_fp32, 2, dim=-1)
        energy = (first_half.pow(2) + second_half.pow(2)).mean(dim=(0, 2))
        pair_scores += energy
        total += 1

    pair_scores /= max(total, 1)
    return pair_scores.detach().cpu()


def select_top_pairs(pair_scores, keep_ratio):
    keep_pairs = []
    num_heads, num_pairs = pair_scores.shape
    keep = max(1, int(round(num_pairs * keep_ratio)))
    for head_idx in range(num_heads):
        _, indices = torch.topk(pair_scores[head_idx], k=keep, largest=True, sorted=False)
        keep_pairs.append(sorted(indices.tolist()))
    return keep_pairs


def get_linear_full_params(linear_module: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(linear_module, StructuredPrunedLinear):
        return linear_module.full_weight(), linear_module.full_bias()
    return linear_module.weight, linear_module.bias


def create_structured_pruned_linear(original_linear: nn.Linear, head_dim: int, keep_pairs: list[list[int]]):
    weight = original_linear.weight.data
    device = weight.device
    dtype = weight.dtype
    num_heads = weight.shape[0] // head_dim
    # 收集需要保留的 RoPE 维度行索引（每对频率对应两个维度）
    keep_rows = []
    for head_idx in range(num_heads):
        base = head_idx * head_dim
        for pair_idx in keep_pairs[head_idx]:
            idx_a, idx_b = rope_pair_to_indices(head_dim, pair_idx)
            keep_rows.extend([base + idx_a, base + idx_b])
    keep_rows = sorted(set(keep_rows))
    if len(keep_rows) == 0:
        raise ValueError("keep_rows 为空，无法构建裁剪后的线性层")
    keep_idx_tensor = torch.tensor(keep_rows, device=device, dtype=torch.long)

    latent_dim = len(keep_rows)
    core = nn.Linear(
        original_linear.in_features,
        latent_dim,
        bias=original_linear.bias is not None,
        dtype=dtype,
        device=device,
    )
    core.weight.data.copy_(weight[keep_idx_tensor])
    if original_linear.bias is not None:
        core.bias.data.copy_(original_linear.bias.data[keep_idx_tensor])
    for param in core.parameters():
        param.requires_grad_(False)

    post = nn.Linear(latent_dim, original_linear.out_features, bias=False, dtype=dtype, device=device)
    post.weight.data.zero_()
    for col, orig_row in enumerate(keep_rows):
        post.weight.data[orig_row, col] = 1.0

    pruned_linear = StructuredPrunedLinear(core, post, keep_idx_tensor.cpu())
    return pruned_linear


def alignment_loss(model, input_ids, layer_id, original_attn, rotary_emb, device):
    """以原模型为 teacher，对齐剪枝后 RoPE key 以降低语义漂移。"""
    B, T = input_ids.shape
    pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

    hs = capture_layer_input_hidden_states(model, input_ids, layer_id, device)
    hs = model.model.layers[layer_id].input_layernorm(hs)
    hack_attn = model.model.layers[layer_id].self_attn
    head_dim = hack_attn.head_dim

    weight_ref, bias_ref = get_linear_full_params(original_attn.k_proj)
    weight_ref = weight_ref.to(device=device, dtype=hs.dtype)
    if bias_ref is not None:
        bias_ref = bias_ref.to(device=device, dtype=hs.dtype)
    num_kv = weight_ref.shape[0] // head_dim
    dummy = torch.empty(B, num_kv, T, head_dim, device=device, dtype=hs.dtype)
    cos, sin = rotary_emb(dummy, pos_ids)
    cos = cos.to(torch.float32)
    sin = sin.to(torch.float32)
    k_ref = F.linear(hs, weight_ref, bias_ref).view(B, T, num_kv, head_dim).transpose(1, 2)
    _, k_ref_rope = apply_rotary_pos_emb(None, k_ref.to(torch.float32), cos, sin)

    proj_dtype = (
        hack_attn.k_proj.core.weight.dtype
        if isinstance(hack_attn.k_proj, StructuredPrunedLinear)
        else hack_attn.k_proj.weight.dtype
    )
    hs_cast = hs.to(dtype=proj_dtype)
    k_new = hack_attn.k_proj(hs_cast).view(B, T, num_kv, head_dim).transpose(1, 2)
    _, k_new_rope = apply_rotary_pos_emb(None, k_new.to(torch.float32), cos, sin)

    return F.mse_loss(k_new_rope, k_ref_rope)


def run_masked_finetune(
    model,
    layer_id,
    hack_attn,
    original_attn,
    tokenizer,
    train_dataset,
    test_ids_all,
    rotary_emb,
    device,
    hparams: PruneHyperParams,
    no_finetune_ppl: float,
):
    loss_hist = []
    ppl_hist = []
    best_ppl = no_finetune_ppl
    best_attn = copy.deepcopy(hack_attn).to(torch.float16)

    if hparams.mask_finetune_steps > 0:
        print("开始 Masked Finetune……")
        train_params: list[torch.nn.Parameter] = []

        def _append_param(param: torch.nn.Parameter):
            # 防止重复加入同一参数，避免 optimizer 更新错位
            if not any(existing is param for existing in train_params):
                train_params.append(param)

        finetune_targets = [hack_attn.k_proj]
        for linear_module in finetune_targets:
            if isinstance(linear_module, StructuredPrunedLinear):
                for p in linear_module.frozen_parameters():
                    p.requires_grad_(False)
                for p in linear_module.finetune_parameters():
                    _append_param(p)
            else:
                _append_param(linear_module.weight)
                if linear_module.bias is not None:
                    _append_param(linear_module.bias)

        for param in hack_attn.v_proj.parameters():
            param.requires_grad_(False)
        for p in hack_attn.parameters():
            p.requires_grad_(False)
        for p in train_params:
            p.requires_grad_(True)

        init_params = [p.detach().clone() for p in train_params]
        optimizer = torch.optim.AdamW(train_params, lr=hparams.mask_finetune_lr, weight_decay=1e-6, eps=1e-8)

        hack_attn.to(dtype=torch.float32)

        for step in tqdm( range(1, hparams.mask_finetune_steps + 1), desc=f"Masked finetune @layer{layer_id}", disable=True):
            input_ids = sample_batch(
                train_dataset,
                tokenizer,
                batch_size=hparams.batch_size,
                seq_len=hparams.seq_len,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            align = alignment_loss(model, input_ids, layer_id, original_attn, rotary_emb, device)
            reg = sum(torch.sum((p - p_init).pow(2)) for p, p_init in zip(train_params, init_params))
            loss = align + hparams.mask_lambda_reg * reg
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(train_params, 0.05)
                optimizer.step()
                loss_hist.append(float(loss.item()))

            if step % hparams.eval_every_mask == 0:
                hack_attn_eval = copy.deepcopy(hack_attn).to(torch.float16)
                hack_attn_eval.eval()
                model.model.layers[layer_id].self_attn = hack_attn_eval
                ppl = evaluate_ppl(model, hparams.seq_len, device=device, nsamples=10, input_ids=test_ids_all)
                ppl_hist.append(ppl)
                print(f"Step {step}: masked_align={loss.item():.6e}, Quick PPL={ppl:.4f}")
                if ppl < best_ppl:
                    best_ppl = ppl
                    best_attn = copy.deepcopy(model.model.layers[layer_id].self_attn)
                model.model.layers[layer_id].self_attn = hack_attn

        hack_attn.to(dtype=torch.float16)

    model.model.layers[layer_id].self_attn = best_attn
    return best_attn, loss_hist, ppl_hist, best_ppl, no_finetune_ppl


def prune_single_layer(
    model,
    tokenizer,
    layer_id,
    original_attn,
    train_dataset,
    test_ids_all,
    rotary_emb,
    device,
    dump_dir,
    hparams: PruneHyperParams,
    resumed_layers: Iterable[int] | None,
):
    print("================================================================================================================================")
    print(f"\n>>> 处理第 {layer_id} 层")
    hack_attn = model.model.layers[layer_id].self_attn

    resumed_from_dump = False
    if resumed_layers is not None and layer_id in resumed_layers:
        resumed_from_dump = restore_layer_from_dump(layer_id, hack_attn, original_attn, dump_dir)
        if resumed_from_dump:
            pruned_snapshot = copy.deepcopy(hack_attn).to(torch.float16)
            best_snapshot = copy.deepcopy(hack_attn).to(torch.float16)
            print("该层已存在微调结果，跳过 Masked Finetune。")
            return {
                "pruned_attn": pruned_snapshot,
                "best_attn": best_snapshot,
                "keep_pairs": None,
                "pair_scores": None,
                "resumed": True,
            }

    print("计算 RoPE pair 重要性……")
    pair_scores = compute_pair_scores(
        model,
        tokenizer,
        train_dataset,
        rotary_emb,
        layer_id,
        attn_module=original_attn,
        batches=hparams.importance_batches,
        batch_size=hparams.importance_batch_size,
        seq_len=hparams.importance_seq_len,
        device=device,
    )
    keep_pairs = select_top_pairs(pair_scores, hparams.keep_ratio)
    print(f"每个头保留 {len(keep_pairs[0])} 对（{len(keep_pairs[0]) * 2} 个维度）")

    # 用选择到的频率对构造结构化裁剪后的 K 投影，同时将 V 投影替换为 PaLU 方案
    pruned_linear_k = create_structured_pruned_linear(hack_attn.k_proj, hack_attn.head_dim, keep_pairs)
    hack_attn.k_proj = pruned_linear_k
    hack_attn.v_proj = build_palu_v_linear(
        layer_id,
        dtype=hack_attn.v_proj.weight.dtype,
        device=hack_attn.v_proj.weight.device,
    )
    for param in hack_attn.v_proj.parameters():
        param.requires_grad_(False)

    pruned_snapshot = copy.deepcopy(hack_attn).to(torch.float16)
    print(f"👉🏻👉🏻👉🏻👉🏻Prune第 {layer_id} 层 Zero-shot ：")
    zero_shot_eval(model, tokenizer, tasks=["openbookqa"])

    best_attn, loss_hist, ppl_hist, best_ppl, current_ppl = run_masked_finetune(model, layer_id, hack_attn, original_attn, tokenizer, train_dataset, test_ids_all, rotary_emb, device, hparams)

    layer_final_ppl = evaluate_ppl(model, hparams.seq_len, device=device, input_ids=test_ids_all)
    print(f"👉🏻👉🏻👉🏻👉🏻微调第 {layer_id} 层后 PPL: {layer_final_ppl:.4f}")
    print(f"👉🏻👉🏻👉🏻👉🏻微调第 {layer_id} 层后Zero-shot：")
    zero_shot_eval(model, tokenizer, tasks=["openbookqa"])

    dump_path = get_layer_dump_path(dump_dir, layer_id)
    if dump_path is not None:
        try:
            torch.save(best_attn.state_dict(), str(dump_path))
            print(f"已保存第 {layer_id} 层最优权重到 {dump_path}")
        except OSError as exc:
            print(f"保存第 {layer_id} 层权重失败: {exc}")

    example_generation(model, tokenizer, device)

    return {
        "pruned_attn": pruned_snapshot,
        "best_attn": copy.deepcopy(model.model.layers[layer_id].self_attn),
        "keep_pairs": keep_pairs,
        "pair_scores": pair_scores,
        "resumed": False,
        "loss_hist": loss_hist,
        "ppl_hist": ppl_hist,
        "best_ppl": best_ppl,
        "current_ppl": current_ppl,
    }

# %% [markdown]
#  ## 主流程

# %% [markdown]
# ### 阶段1：加载模型和数据集

# %%
print("===== 加载模型和数据集 =====")
model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
device = model.device
print(f"已加载模型: {MODEL_PATH}到{model.device}")
train_dataset, test_ids_all = prepare_datasets(tokenizer, DATASET_NAME)
print(f"已加载数据集: {DATASET_NAME}")
original_layers = {lid: copy.deepcopy(model.model.layers[lid]) for lid in range(len(model.model.layers))}
print(f"已存档原始Decoder Layers: {list(original_layers.keys())} 于 original_layers")
rotary_full = LlamaRotaryEmbedding(config=model.model.layers[0].self_attn.config).to(device=device, dtype=torch.float16)

# %%
print("\n===== 原始模型评估 =====")
# 记录剪枝前的基线性能，后续方便对比
hparams = PruneHyperParams()
baseline_ppl = evaluate_ppl(model, hparams.seq_len, device=model.device, input_ids=test_ids_all)
print(f"原始模型整体 PPL: {baseline_ppl:.4f}")
print("原始模型 Zero-shot:")
zero_shot_eval(model, tokenizer, tasks=["openbookqa"])
example_generation(model, tokenizer, device)

# %% [markdown]
# ### 阶段2：顺序层级剪枝 + Masked Finetune

# %%
# 超参数
pruned_hack_layer_ids = []
hack_layer_ids = list(range(0, 1))
active_hack_layer_id = hack_layer_ids[0] if hack_layer_ids else 0
print_meta_info(hparams, {"device": device, "hack_layer_ids": hack_layer_ids, "active_hack_layer_id": active_hack_layer_id, "pruned_hack_layer_ids": pruned_hack_layer_ids})
# reset_model(model, original_layers, list(range(0, 32)))

# %%
print("\n===== 顺序层级剪枝 + Masked Finetune =====")
dump_dir = ensure_dump_dir("RAPDump")
pruned_layers: dict[int, nn.Module] = {}
best_layers: dict[int, nn.Module] = {}
pair_scores_map: dict[int, torch.Tensor | None] = {}

for layer_id in hack_layer_ids:
    # 逐层执行结构化剪枝 + （可选）细调，必要时读取已有 dump
    print(f"\n>>> ✂️Prune第 {layer_id} 层")
    hack_attn = model.model.layers[layer_id].self_attn
    original_attn = original_layers[layer_id].self_attn

    if layer_id in pruned_hack_layer_ids:
        restored = restore_layer_from_dump(layer_id, hack_attn, original_attn, dump_dir)
        if not restored:
            print("该层已存在微调结果，跳过 Masked Finetune。")
            continue

    print("计算 RoPE pair 重要性……")
    pair_scores = compute_pair_scores( model, tokenizer, train_dataset, rotary_full, layer_id, attn_module=original_attn, batches=hparams.importance_batches, batch_size=hparams.importance_batch_size, seq_len=hparams.importance_seq_len, device=device)
    keep_pairs = select_top_pairs(pair_scores, hparams.keep_ratio)
    print(f"每个头保留 {len(keep_pairs[0])} 个RoPE pair: {keep_pairs}")

    # 用选择到的频率对构造结构化裁剪后的 K 投影，同时将 V 投影替换为 PaLU 方案
    pruned_linear_k = create_structured_pruned_linear(hack_attn.k_proj, hack_attn.head_dim, keep_pairs)
    hack_attn.k_proj = pruned_linear_k
    hack_attn.v_proj = build_palu_v_linear( layer_id,dtype=hack_attn.v_proj.weight.dtype, device=hack_attn.v_proj.weight.device)
        
    # zero_shot_eval(model, tokenizer, tasks=["openbookqa"])
    # print(f"👉🏻👉🏻👉🏻👉🏻Prune第 {layer_id} 层后的无微调 Zero-shot ↑")
    no_finetune_ppl = evaluate_ppl(model, hparams.seq_len, device=device, input_ids=test_ids_all)
    print(f"👉🏻👉🏻👉🏻👉🏻Prune第 {layer_id} 层 PPL: {no_finetune_ppl:.4f}")
    best_attn, loss_hist, ppl_hist, best_ppl, no_finetune_ppl = run_masked_finetune( model, layer_id, hack_attn, original_attn, tokenizer, train_dataset, test_ids_all, rotary_full, device, hparams, no_finetune_ppl)

    layer_final_ppl = evaluate_ppl(model, hparams.seq_len, device=device, input_ids=test_ids_all)
    print(f"👉🏻👉🏻👉🏻👉🏻微调第 {layer_id} 层后 PPL: {layer_final_ppl:.4f}")
    # print(f"👉🏻👉🏻👉🏻👉🏻微调第 {layer_id} 层后Zero-shot：")
    # zero_shot_eval(model, tokenizer, tasks=["openbookqa"])
    # print(f"👉🏻👉🏻👉🏻👉🏻微调第 {layer_id} 层后的example生成：")
    # example_generation(model, tokenizer, device)

    dump_path = get_layer_dump_path(dump_dir, layer_id)
    torch.save(best_attn.state_dict(), str(dump_path))
    print(f"已保存第 {layer_id} 层最优权重到 {dump_path}")


# %% [markdown]
# ###  最终模型评估

# %%
print("\n===== 最终模型评估 =====")
# 剪枝完成后再次评估，确认整体指标及零样本表现
final_ppl = evaluate_ppl(model, hparams.seq_len, device=device, input_ids=test_ids_all)
print(f"最终模型整体 PPL: {final_ppl:.4f}")
final_zero_shot = zero_shot_eval(model, tokenizer, tasks=["openbookqa"])
print("最终模型 Zero-shot 结果:", final_zero_shot)
example_generation(model, tokenizer, device)


