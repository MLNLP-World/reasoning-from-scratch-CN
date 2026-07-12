# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# 《从零构建推理模型》来源：https://mng.bz/lZ5B
# 代码仓库：https://github.com/rasbt/reasoning-from-scratch

# 以类似第 2、3 章的方式，用最精简的流式模式运行模型
# 默认使用 KV 缓存，不加入额外复杂功能。
# 与 generate_simple.py 类似，但使用不带记忆的交互式 REPL。

import argparse
from pathlib import Path
import time
import torch

from reasoning_from_scratch.ch02 import (
    get_device,
    generate_stats
)
from reasoning_from_scratch.ch02 import (
    generate_text_basic_stream_cache
)
from reasoning_from_scratch.qwen3 import (
    download_qwen3_small,
    Qwen3Model,
    Qwen3Tokenizer,
    QWEN_CONFIG_06_B
)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="运行 Qwen3 文本生成（交互式 REPL）")
parser.add_argument(
    "--device",
    type=str,
    default=None,
    help="运行设备（例如 'cpu'、'cuda'、'mps'）。"
         "未提供时使用 get_device() 自动检测。"
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=2048,
    help="要生成的最大新词元数。"
)
parser.add_argument(
    "--compile",
    action="store_true",
    help="编译 PyTorch 模型。"
)
parser.add_argument(
    "--reasoning",
    action="store_true",
    help="使用推理模型变体。"
)

args = parser.parse_args()
device = torch.device(args.device) if args.device else get_device()

if args.reasoning:
    download_qwen3_small(kind="reasoning", tokenizer_only=False, out_dir="qwen3")
    tokenizer_path = Path("qwen3") / "tokenizer-reasoning.json"
    model_path = Path("qwen3") / "qwen3-0.6B-reasoning.pth"
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_path,
        apply_chat_template=True,
        add_generation_prompt=True,
        add_thinking=True
    )
else:
    download_qwen3_small(kind="base", tokenizer_only=False, out_dir="qwen3")
    tokenizer_path = Path("qwen3") / "tokenizer-base.json"
    model_path = Path("qwen3") / "qwen3-0.6B-base.pth"
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_path,
        apply_chat_template=False,
        add_generation_prompt=False,
        add_thinking=False
    )

model = Qwen3Model(QWEN_CONFIG_06_B)
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

if args.compile:
    model = torch.compile(model)

print()
print("=" * 60)
print(f"torch     : {torch.__version__}")
print(f"device    : {device}")
print("cache     : True")
print(f"compile   : {args.compile}")
print(f"reasoning : {args.reasoning}")
print("memory    : False")
print("=" * 60)
print()
print("交互式 REPL（无记忆）。输入 '\\exit' 或 '\\quit' 退出。\n")


def run_once(prompt: str):
    input_ids = tokenizer.encode(prompt)
    input_token_ids_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    start_time = time.time()
    all_token_ids = []

    print("[Model]\n", end="", flush=True)
    for token in generate_text_basic_stream_cache(
        model=model,
        token_ids=input_token_ids_tensor,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id
    ):
        token_id = token.squeeze(0).item()
        print(tokenizer.decode([token_id]), end="", flush=True)

        all_token_ids.append(token_id)

    end_time = time.time()
    print("\n")

    print("[Stats]")
    generate_stats(
        torch.tensor(all_token_ids),
        tokenizer,
        start_time,
        end_time
    )
    print("-" * 60)


# REPL 循环
try:
    while True:
        try:
            user_in = input(">> ").strip()
        except EOFError:
            print("")
            break
        if user_in.lower() in {r"\exit", r"\quit"}:
            break
        if not user_in:
            continue

        print("\n" + "-" * 60)
        print("[User]")
        print(user_in + "\n")
        run_once(user_in)
except KeyboardInterrupt:
    print("\n已由用户中断。")
