# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# 《从零构建推理模型》来源：https://mng.bz/lZ5B
# 代码仓库：https://github.com/rasbt/reasoning-from-scratch

import argparse
from pathlib import Path

import torch

from reasoning_from_scratch.ch02 import (
    generate_text_basic_stream_cache,
    get_device,
)
from reasoning_from_scratch.ch03 import render_prompt
from reasoning_from_scratch.qwen3 import (
    download_qwen3_distill_checkpoints,
    download_qwen3_small,
    Qwen3Model,
    Qwen3Tokenizer,
    QWEN_CONFIG_06_B,
)


def build_tokenizer(local_dir):
    download_qwen3_small(kind="reasoning", tokenizer_only=True, out_dir=local_dir)
    return Qwen3Tokenizer(
        tokenizer_file_path=Path(local_dir) / "tokenizer-reasoning.json",
        apply_chat_template=True,
        add_generation_prompt=True,
        add_thinking=True,
    )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="下载第 8 章蒸馏检查点并以流式方式生成回复。",
    )
    parser.add_argument(
        "--distill_type",
        type=str,
        default="deepseek_r1",
        choices=("deepseek_r1", "qwen3_235b_a22b"),
        help="要下载的蒸馏检查点系列。",
    )
    parser.add_argument(
        "--step",
        type=str,
        default="06682",
        help="要下载的训练步骤。",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Solve: If x + 7 = 19, what is x?",
        help="使用标准章节提示词模板呈现的数学问题。",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="要生成的最大新词元数。",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="qwen3",
        help="用于下载检查点和分词器的本地目录。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="运行设备（例如 cpu、cuda、mps）。省略时自动检测。",
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    checkpoint_path = download_qwen3_distill_checkpoints(
        distill_type=args.distill_type,
        step=args.step,
        out_dir=args.local_dir,
    )
    tokenizer = build_tokenizer(args.local_dir)

    model = Qwen3Model(QWEN_CONFIG_06_B)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    prompt = render_prompt(args.prompt)
    input_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)

    print()
    print("=" * 60)
    print(f"torch        : {torch.__version__}")
    print(f"device       : {device}")
    print(f"distill_type : {args.distill_type}")
    print(f"step         : {args.step}")
    print("=" * 60)
    print()

    for token in generate_text_basic_stream_cache(
        model=model,
        token_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    ):
        token_id = token.squeeze(0).item()
        print(tokenizer.decode([token_id]), end="", flush=True)

    print("\n")


if __name__ == "__main__":
    main()
