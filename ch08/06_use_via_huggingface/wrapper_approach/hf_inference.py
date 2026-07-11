# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# 《从零构建推理模型》来源：https://mng.bz/lZ5B
# 代码仓库：https://github.com/rasbt/reasoning-from-scratch

import argparse

import torch

from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import render_prompt

from hf_wrapper import build_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "通过轻量包装器运行本地 Hugging Face 风格生成，"
            "该包装器围绕从零实现的 Qwen3Model，无需导出模型文件夹。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tokenizer_kind",
        type=str,
        choices=("base", "reasoning"),
        default="base",
        help="原始 .pth 权重使用的分词器系列。",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="可选的本地 .pth 文件。省略时下载默认模型。",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="可选的分词器 JSON 文件。省略时下载分词器。",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="qwen3",
        help="模型和分词器的下载/缓存目录。",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Solve: If x + 7 = 19, what is x?",
        help="使用第 3 章提示词模板呈现的数学问题。",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="要生成的最大词元数。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="运行设备（cpu、cuda、mps）。默认自动检测。",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device) if args.device else get_device()
    model, tokenizer = build_model_and_tokenizer(
        tokenizer_kind=args.tokenizer_kind,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        local_dir=args.local_dir,
    )
    model.to(device)
    model.eval()

    prompt = render_prompt(args.prompt)
    input_ids = torch.tensor(
        tokenizer.encode(prompt),
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = output_ids[0, input_ids.shape[1]:]

    print()
    print("=" * 60)
    print(f"device         : {device}")
    print(f"tokenizer_kind : {args.tokenizer_kind}")
    print("=" * 60)
    print()
    print(tokenizer.decode(generated_ids.tolist()))
    print()


if __name__ == "__main__":
    main()
