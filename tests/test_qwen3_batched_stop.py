# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# 《从零构建推理模型》来源：https://mng.bz/lZ5B
# 代码仓库：https://github.com/rasbt/reasoning-from-scratch

import os
import torch
import pytest
from pathlib import Path

from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.qwen3 import (
    download_qwen3_small,
    Qwen3Tokenizer,
    QWEN_CONFIG_06_B,
)
from reasoning_from_scratch.qwen3_batched import (
    generate_text_basic_batched_cache,
    generate_text_basic_batched_cache_stop,
    generate_text_basic_batched_stream_cache,
    generate_text_basic_batched_stream_cache_stop,
    Qwen3Model as Qwen3ModelBatched,
)

skip_expensive = os.environ.get("SKIP_EXPENSIVE", "0") == "1"

# 提高 CI 的可复现性和稳健性
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
torch.backends.mkldnn.enabled = False
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


@pytest.mark.skipif(skip_expensive, reason="Skipping expensive test on CI")
@pytest.mark.parametrize("reasoning", [False, True])
def test_batched_vs_batched_stop_equivalence(reasoning):

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
        apply_chat_template=reasoning,
        add_generation_prompt=reasoning,
        add_thinking=reasoning,
    )

    # 模型
    model_batched = Qwen3ModelBatched(QWEN_CONFIG_06_B)
    model_batched.load_state_dict(torch.load(model_path, map_location=device))
    model_batched.to(device).eval()

    # 提示词
    prompts = [
        "Explain large language models in two sentences.",
        "Explain large language models in one sentence.",
        "1+1?",
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
    max_new_tokens = 12
    outputs_reg = generate_text_basic_batched_cache(
        model=model_batched,
        token_ids=input_ids_batched,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        attn_mask=attn_mask_batched,
        pad_id=pad_id,
    )
    outputs_stop = generate_text_basic_batched_cache_stop(
        model=model_batched,
        token_ids=input_ids_batched,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        attn_mask=attn_mask_batched,
        pad_id=pad_id,
    )

    # 检查等价性
    for idx in range(len(prompts)):
        reg_toks = outputs_reg[idx].tolist()
        stop_toks = outputs_stop[idx].tolist()
        assert reg_toks == stop_toks, (
            f"Token mismatch at prompt {idx}:\n"
            f"regular_tokens={reg_toks}\n"
            f"stop_tokens   ={stop_toks}\n"
            f"regular_text={tokenizer.decode(reg_toks)}\n"
            f"stop_text   ={tokenizer.decode(stop_toks)}"
        )


@pytest.mark.skipif(skip_expensive, reason="Skipping expensive test on CI")
@pytest.mark.parametrize("reasoning", [False, True])
def test_stream_vs_stream_stop_equivalence(reasoning):

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
        apply_chat_template=reasoning,
        add_generation_prompt=reasoning,
        add_thinking=reasoning,
    )

    # 模型
    model_batched = Qwen3ModelBatched(QWEN_CONFIG_06_B)
    model_batched.load_state_dict(torch.load(model_path, map_location=device))
    model_batched.to(device).eval()

    # 提示词
    prompts = [
        "Explain large language models in two sentences.",
        "Explain large language models in one sentence.",
        "1+1?",
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
    max_new_tokens = 12
    B = input_ids_batched.size(0)

    # 常规流式生成
    reg_stream_tokens = [[] for _ in range(B)]
    for step_tokens in generate_text_basic_batched_stream_cache(
        model=model_batched,
        token_ids=input_ids_batched,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        attn_mask=attn_mask_batched,
        pad_id=pad_id,
    ):
        step_tokens = step_tokens.squeeze(1)
        for b in range(B):
            reg_stream_tokens[b].append(int(step_tokens[b].item()))

    # 停止流式生成
    stop_stream_tokens = [[] for _ in range(B)]
    for step_tokens in generate_text_basic_batched_stream_cache_stop(
        model=model_batched,
        token_ids=input_ids_batched,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        attn_mask=attn_mask_batched,
        pad_id=pad_id,
    ):
        step_tokens = step_tokens.squeeze(1)
        for b in range(B):
            stop_stream_tokens[b].append(int(step_tokens[b].item()))

    # 检查等价性
    for idx in range(B):
        assert reg_stream_tokens[idx] == stop_stream_tokens[idx], (
            f"Token mismatch at prompt {idx}:\n"
            f"regular_tokens={reg_stream_tokens[idx]}\n"
            f"stop_tokens   ={stop_stream_tokens[idx]}\n"
            f"regular_text={tokenizer.decode(reg_stream_tokens[idx])}\n"
            f"stop_text   ={tokenizer.decode(stop_stream_tokens[idx])}"
        )
