# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# 《从零构建推理模型》来源：https://mng.bz/lZ5B
# 代码仓库：https://github.com/rasbt/reasoning-from-scratch

import os
import torch
import pytest
from pathlib import Path

from reasoning_from_scratch.ch02 import (
    get_device,
    generate_text_basic_cache,
)
from reasoning_from_scratch.qwen3 import (
    apply_rope,
    download_qwen3_small,
    GroupedQueryAttention,
    Qwen3Tokenizer,
    Qwen3Model,
    QWEN_CONFIG_06_B,
)
from reasoning_from_scratch.qwen3_batched import (
    generate_text_basic_batched_cache,
    generate_text_basic_batched_stream_cache,
    Qwen3Model as Qwen3ModelBatched,
)

skip_expensive = os.environ.get("SKIP_EXPENSIVE", "0") == "1"

# 提高 CI 的可复现性和稳健性
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
torch.backends.mkldnn.enabled = False
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


TORCH_VERSION_PARTS = torch.__version__.split("+")[0].split(".")
TORCH_MAJOR_MINOR = tuple(int(part) for part in TORCH_VERSION_PARTS[:2])


def _assert_generation_matches(text_single, text_batch, idx):
    if TORCH_MAJOR_MINOR >= (2, 11):
        assert text_single[:60] == text_batch[:60], (
            f"Mismatch within first 60 chars at prompt {idx}:\n"
            f"single={text_single}\n"
            f"batched={text_batch}"
        )
    else:
        assert text_single == text_batch, (
            f"Mismatch at prompt {idx}:\n"
            f"single={text_single}\n"
            f"batched={text_batch}"
        )


@pytest.mark.skipif(skip_expensive, reason="Skipping expensive test on CI")
@pytest.mark.parametrize("reasoning", [False, True])
def test_batched_vs_nonbatched_equivalence_with_batched_model(reasoning):

    device = get_device()

    # 下载并初始化分词器
    kind = "reasoning" if reasoning else "base"
    download_qwen3_small(kind=kind, tokenizer_only=False, out_dir="qwen3")
    tokenizer_path = Path("qwen3") / (
        "tokenizer-reasoning.json" if reasoning else "tokenizer-base.json"
    )
    model_path = Path("qwen3") / (
        "qwen3-0.6B-reasoning.pth" if reasoning else "qwen3-0.6B-base.pth"
    )
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_path,
        apply_chat_template=True if reasoning else False,
        add_generation_prompt=True if reasoning else False,
        add_thinking=True if reasoning else False,
    )

    # 模型
    model_batched = Qwen3ModelBatched(QWEN_CONFIG_06_B)
    model_batched.load_state_dict(torch.load(model_path, map_location=device))
    model_batched.to(device)
    model_batched.eval()

    # 提示词
    prompts = [
        "Explain large language models in two sentences.",
        "Explain large language models in one sentence.",
        "1+1?"
    ]

    # 非批处理输入
    single_inputs = [
        torch.tensor(tokenizer.encode(p), device=device).unsqueeze(0)
        for p in prompts
    ]

    # 批处理输入（左侧填充）
    tokenized = [tokenizer.encode(p) for p in prompts]
    max_len = max(len(t) for t in tokenized)
    pad_id = tokenizer.pad_token_id
    left_padded = [[pad_id] * (max_len - len(t)) + t for t in tokenized]
    attn_mask = [
        [0] * (max_len - len(t)) + [1] * len(t) for t in tokenized
    ]
    input_ids_batched = torch.tensor(left_padded, device=device)
    attn_mask_batched = torch.tensor(attn_mask, device=device, dtype=torch.bool)

    # 生成
    max_new_tokens = 12  # 成本低，但足以检查一致性
    outputs_single = []
    for input_ids in single_inputs:
        out = generate_text_basic_cache(
            model=model_batched,
            token_ids=input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
        outputs_single.append(out[0].tolist())

    outputs_batched = generate_text_basic_batched_cache(
        model=model_batched,
        token_ids=input_ids_batched,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        attn_mask=attn_mask_batched,
        pad_id=pad_id,
    )

    # MPS 上的 Torch 2.11 可能在后续近似平局后产生差异，因此只
    # 比较共享前缀。旧版本仍检查完全一致的文本。
    for idx, out_single in enumerate(outputs_single):
        out_batch = outputs_batched[idx].tolist()

        text_single = tokenizer.decode(out_single)
        text_batch = tokenizer.decode(out_batch)

        _assert_generation_matches(text_single, text_batch, idx)


# 替换 GroupedQueryAttention.forward
# 新的 forward 使用对填充稳定的 softmax 计算
# 简单 Qwen3 模型也使用该计算。
# 这样可以在数值上与批处理版本完全一致；
# 否则，微小的浮点差异可能会在生成几个词元后
# 导致轻微分歧。


def new_forward(self, x, mask, cos, sin, start_pos=0, cache=None):
    b, num_tokens, _ = x.shape

    # 应用投影
    queries = self.W_query(x)  # （批量, 词元数, 注意力头数 * 头维度）
    keys = self.W_key(x)       # （批量, 词元数, KV 组数 * 头维度）
    values = self.W_value(x)   # （批量, 词元数, KV 组数 * 头维度）

    # 重塑为注意力头/KV 组
    queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
    keys_new = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
    values_new = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

    # 可选归一化
    if self.q_norm:
        queries = self.q_norm(queries)
    if self.k_norm:
        keys_new = self.k_norm(keys_new)

    # 应用 RoPE
    queries = apply_rope(queries, cos, sin, offset=start_pos)
    keys_new = apply_rope(keys_new, cos, sin, offset=start_pos)

    if cache is not None:
        prev_k, prev_v = cache
        keys = torch.cat([prev_k, keys_new], dim=2)
        values = torch.cat([prev_v, values_new], dim=2)
    else:
        start_pos = 0  # 重置 RoPE
        keys, values = keys_new, values_new
    next_cache = (keys, values)

    # 扩展 K 和 V 以匹配注意力头数
    keys = keys.repeat_interleave(self.group_size, dim=1)
    values = values.repeat_interleave(self.group_size, dim=1)

    attn_scores = torch.matmul(queries.to(torch.float32), keys.transpose(2, 3).to(torch.float32))
    attn_scores = attn_scores / self.head_dim**0.5

    # 使用 -inf 应用掩码，使被屏蔽项在 softmax 后严格为零
    attn_scores = attn_scores.masked_fill(mask, -torch.inf)

    # 在未屏蔽集合上计算稳定的 log-sum-exp
    row_max = attn_scores.amax(dim=-1, keepdim=True)
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    exp_scores = torch.exp(attn_scores - row_max)
    exp_scores = exp_scores.masked_fill(mask, 0.0)

    denom = exp_scores.sum(dim=-1, keepdim=True)
    attn_weights = exp_scores / denom.clamp(min=torch.finfo(exp_scores.dtype).tiny)

    # 转回模型 dtype 并继续
    attn_weights = attn_weights.to(values.dtype)
    context = torch.matmul(attn_weights, values)
    context = context.transpose(1, 2).reshape(b, num_tokens, self.d_out)
    return self.out_proj(context), next_cache


@pytest.fixture(autouse=True)
def patch_attention(monkeypatch):
    monkeypatch.setattr(GroupedQueryAttention, "forward", new_forward)
    yield


@pytest.mark.skipif(skip_expensive, reason="Skipping expensive test on CI")
@pytest.mark.parametrize("reasoning", [False, True])
def test_batched_vs_nonbatched_equivalence_with_single_versus_batched_model(reasoning):

    device = get_device()

    # 下载并初始化分词器
    kind = "reasoning" if reasoning else "base"
    download_qwen3_small(kind=kind, tokenizer_only=False, out_dir="qwen3")
    tokenizer_path = Path("qwen3") / (
        "tokenizer-reasoning.json" if reasoning else "tokenizer-base.json"
    )
    model_path = Path("qwen3") / (
        "qwen3-0.6B-reasoning.pth" if reasoning else "qwen3-0.6B-base.pth"
    )
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_path,
        apply_chat_template=True if reasoning else False,
        add_generation_prompt=True if reasoning else False,
        add_thinking=True if reasoning else False,
    )

    # 模型
    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    model_batched = Qwen3ModelBatched(QWEN_CONFIG_06_B)
    model_batched.load_state_dict(torch.load(model_path, map_location=device))
    model_batched.to(device)
    model_batched.eval()

    # 提示词
    prompts = [
        "Explain large language models in two sentences.",
        "Explain large language models in one sentence.",
        "1+1?"
    ]

    # 非批处理输入
    single_inputs = [
        torch.tensor(tokenizer.encode(p), device=device).unsqueeze(0)
        for p in prompts
    ]

    # 批处理输入（左侧填充）
    tokenized = [tokenizer.encode(p) for p in prompts]
    max_len = max(len(t) for t in tokenized)
    pad_id = tokenizer.pad_token_id
    left_padded = [[pad_id] * (max_len - len(t)) + t for t in tokenized]
    attn_mask = [
        [0] * (max_len - len(t)) + [1] * len(t) for t in tokenized
    ]
    input_ids_batched = torch.tensor(left_padded, device=device)
    attn_mask_batched = torch.tensor(attn_mask, device=device, dtype=torch.bool)

    # 生成
    max_new_tokens = 12  # 成本低，但足以检查一致性
    outputs_single = []
    for input_ids in single_inputs:
        out = generate_text_basic_cache(
            model=model,
            token_ids=input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
        outputs_single.append(out[0].tolist())

    outputs_batched = generate_text_basic_batched_cache(
        model=model_batched,
        token_ids=input_ids_batched,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        attn_mask=attn_mask_batched,
        pad_id=pad_id,
    )

    for idx, out_single in enumerate(outputs_single):
        out_batch = outputs_batched[idx].tolist()

        text_single = tokenizer.decode(out_single)
        text_batch = tokenizer.decode(out_batch)

        _assert_generation_matches(text_single, text_batch, idx)


@pytest.mark.skipif(skip_expensive, reason="Skipping expensive test on CI")
@pytest.mark.parametrize("reasoning", [False, True])
def test_plain_vs_streaming_generation(reasoning):

    device = get_device()

    # 下载并初始化分词器
    kind = "reasoning" if reasoning else "base"
    download_qwen3_small(kind=kind, tokenizer_only=False, out_dir="qwen3")
    tokenizer_path = Path("qwen3") / (
        "tokenizer-reasoning.json" if reasoning else "tokenizer-base.json"
    )
    model_path = Path("qwen3") / (
        "qwen3-0.6B-reasoning.pth" if reasoning else "qwen3-0.6B-base.pth"
    )
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_path,
        apply_chat_template=True if reasoning else False,
        add_generation_prompt=True if reasoning else False,
        add_thinking=True if reasoning else False,
    )

    # 模型
    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    model_batched = Qwen3ModelBatched(QWEN_CONFIG_06_B)
    model_batched.load_state_dict(torch.load(model_path, map_location=device))
    model_batched.to(device)
    model_batched.eval()

    # 提示词
    prompts = [
        "Explain large language models in two sentences.",
        "Explain large language models in one sentence.",
        "1+1?"
    ]

    # 批处理输入（左侧填充）
    tokenized = [tokenizer.encode(p) for p in prompts]
    max_len = max(len(t) for t in tokenized)
    pad_id = tokenizer.pad_token_id
    left_padded = [[pad_id] * (max_len - len(t)) + t for t in tokenized]
    attn_mask = [
        [0] * (max_len - len(t)) + [1] * len(t) for t in tokenized
    ]
    input_ids_batched = torch.tensor(left_padded, device=device)
    attn_mask_batched = torch.tensor(attn_mask, device=device, dtype=torch.bool)

    # 生成
    max_new_tokens = 12  # 成本低，但足以检查一致性

    # 常规批处理
    outputs_batched_reg = generate_text_basic_batched_cache(
        model=model_batched,
        token_ids=input_ids_batched,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        attn_mask=attn_mask_batched,
        pad_id=pad_id,
    )
    outputs_batched_reg_tokens = outputs_batched_reg.tolist()

    # 批处理流式生成
    B = input_ids_batched.size(0)
    stream_tokens = [[] for _ in range(B)]
    for step_tokens in generate_text_basic_batched_stream_cache(
        model=model_batched,            # 与上面相同的模型
        token_ids=input_ids_batched,
        max_new_tokens=max_new_tokens,  # 相同的长度预算
        eos_token_id=tokenizer.eos_token_id,
        attn_mask=attn_mask_batched,
        pad_id=pad_id,
    ):
        # step_tokens：[B, 1]
        step_tokens = step_tokens.squeeze(1)
        for b in range(B):
            stream_tokens[b].append(int(step_tokens[b].item()))

    # 检查等价性
    for idx in range(B):
        reg_toks = outputs_batched_reg_tokens[idx]
        str_toks = stream_tokens[idx]

        if reg_toks != str_toks:
            text_single = tokenizer.decode(reg_toks)
            text_stream = tokenizer.decode(str_toks)
            assert reg_toks == str_toks, (
                f"Token mismatch at prompt {idx}:\n"
                f"regular_tokens={reg_toks}\n"
                f"stream_tokens ={str_toks}\n"
                f"regular_text={text_single}\n"
                f"stream_text ={text_stream}"
            )
