# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# 《从零构建推理模型》来源：https://mng.bz/lZ5B
# 代码仓库：https://github.com/rasbt/reasoning-from-scratch

import argparse
import torch

from reasoning_from_scratch.qwen3 import (
    Qwen3Model,
    QWEN_CONFIG_06_B
)
from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import (
    load_math500_test,
    evaluate_math500_stream,
    load_model_and_tokenizer,
    load_tokenizer_only
)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="使用的设备：'auto'，或 'cpu'、'cuda'、'cuda:0'、'mps' 等 torch 设备字符串。",
    )
    parser.add_argument(
        "--which_model",
        type=str,
        default="base",
        choices=["base", "reasoning", "instruct"],
        help="要加载的模型变体",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=10,
        help="要评估的 MATH-500 样本数",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="生成的最大新词元数",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="为模型启用 torch.compile。",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="用于加载模型权重的可选 .pth 检查点路径。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="评估时输出每个样本是否正确。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)

    which_model = args.which_model
    dataset_size = args.dataset_size
    max_new_tokens = args.max_new_tokens
    use_compile = args.compile

    print("Model:", which_model)
    print("Device:", device)
    dev_name = str(device).replace(":", "-")

    math_data = load_math500_test()

    if args.which_model == "instruct":
        which_model = "reasoning"
    else:
        which_model = args.which_model

    if args.checkpoint_path:
        # 加载第 6 章保存的强化学习检查点文件
        tokenizer = load_tokenizer_only(which_model=which_model)
        model = Qwen3Model(QWEN_CONFIG_06_B)
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(device)
        if args.compile:
            torch._dynamo.config.allow_unspec_int_on_nn_module = True
            model = torch.compile(model)
    else:
        model, tokenizer = load_model_and_tokenizer(
            which_model=which_model,
            device=device,
            use_compile=args.compile
        )

    if args.which_model == "instruct":
        tokenizer.add_thinking = False

    model.eval()
    torch.set_float32_matmul_precision("high")

    num_correct, num_examples, acc = evaluate_math500_stream(
        model=model,
        out_path=f"math500_{which_model}-{dev_name}-evaluate-script.jsonl",
        tokenizer=tokenizer,
        device=device,
        math_data=math_data[:dataset_size],
        max_new_tokens=max_new_tokens,
        verbose=args.verbose,
    )
