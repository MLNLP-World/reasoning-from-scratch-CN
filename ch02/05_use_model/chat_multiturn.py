# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# 《从零构建推理模型》来源：https://mng.bz/lZ5B
# 代码仓库：https://github.com/rasbt/reasoning-from-scratch

# 以类似第 2、3 章的方式，用最精简的流式模式运行模型
# 默认使用 KV 缓存，不加入额外复杂功能。
# 带多轮记忆的交互式 REPL（读取、求值、输出、循环）。

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
    help="每轮要生成的最大新词元数。"
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
else:
    download_qwen3_small(kind="base", tokenizer_only=False, out_dir="qwen3")
    tokenizer_path = Path("qwen3") / "tokenizer-base.json"
    model_path = Path("qwen3") / "qwen3-0.6B-base.pth"

# 稍后手动应用聊天模板
tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=tokenizer_path,
    apply_chat_template=False
)

model = Qwen3Model(QWEN_CONFIG_06_B)
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

if args.compile:
    model = torch.compile(model)

# 推理模型可能输出 <|im_end|>；基础模型可能输出 <|endoftext|>。
EOS_TOKEN_IDS = (
    tokenizer.encode("<|im_end|>")[0],
    tokenizer.encode("<|endoftext|>")[0]
)

print()
print("=" * 60)
print(f"torch     : {torch.__version__}")
print(f"device    : {device}")
print("cache     : True")
print(f"compile   : {args.compile}")
print(f"reasoning : {args.reasoning}")
print("memory    : True")
print(f"max_new_tokens (per turn): {args.max_new_tokens}")
print(f"context_length: {model.cfg['context_length']}")
print("=" * 60)
print()
print("带记忆的交互式 REPL。输入 '\\exit' 或 '\\quit' 退出。")
print("命令：\\clear（清除记忆），\\history（显示轮次数）\n")

# 以角色-内容字典列表保存多轮记忆
# 示例：{"role": "system"|"user"|"assistant", "content": str}
history = [
    {"role": "system", "content": "You are a helpful assistant."}
]


def build_prompt_from_history(history, add_assistant_header=True):
    """
    history: [{"role": "system"|"user"|"assistant", "content": str}, ...]
    """
    parts = []
    for m in history:
        role = m["role"]
        content = m["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

    if add_assistant_header:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def trim_input_tensor(input_ids_tensor, context_len, max_new_tokens):
    assert max_new_tokens < context_len
    keep_len = max(1, context_len - max_new_tokens)

    # 如果提示词过长，则从左侧截断到 keep_len
    if input_ids_tensor.shape[1] > keep_len:
        input_ids_tensor = input_ids_tensor[:, -keep_len:]

    return input_ids_tensor


def run_generate(user_text):
    # 将用户提示词加入历史记录
    history.append({"role": "user", "content": user_text})

    # 编码完整历史记录
    prompt = build_prompt_from_history(history, add_assistant_header=True)
    input_ids = tokenizer.encode(prompt)
    input_token_ids_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    # 从左侧截断（为生成内容留出空间）
    input_token_ids_tensor = trim_input_tensor(
        input_ids_tensor=input_token_ids_tensor,
        context_len=model.cfg["context_length"],
        max_new_tokens=args.max_new_tokens
    )

    start_time = time.time()
    all_token_ids = []

    print("[Model]\n", end="", flush=True)
    for tok in generate_text_basic_stream_cache(
        model=model,
        token_ids=input_token_ids_tensor,
        max_new_tokens=args.max_new_tokens,
        # eos_token_id=TOKENIZER.eos_token_id
    ):
        token_id = tok.squeeze(0)
        if token_id in EOS_TOKEN_IDS:  # 遇到停止词元时手动终止
            break
        piece = tokenizer.decode(token_id.tolist())
        print(piece, end="", flush=True)
        all_token_ids.append(token_id.item())

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

    # 将模型回复加入历史记录
    assistant_text = tokenizer.decode(all_token_ids)
    history.append({"role": "assistant", "content": assistant_text})
    return assistant_text


# 交互式 REPL（读取、求值、输出、循环）
try:
    while True:
        try:
            user_in = input(">> ").strip()
        except EOFError:
            print("")
            break

        low = user_in.lower()
        if low in {r"\exit", r"\quit"}:
            break
        if low == r"\clear":
            # 重置历史记录，但保留系统提示词
            system_entries = [m for m in history if m["role"] == "system"]
            history.clear()
            if system_entries:
                history.extend(system_entries)
            else:
                history.append({"role": "system", "content": "You are a helpful assistant."})
            print("（记忆已清除）\n")
            continue
        if low == r"\history":
            # 将助手轮次数记为目前的模型回复数
            assistant_turns = sum(1 for m in history if m["role"] == "assistant")
            print(f"（已保存轮次： {assistant_turns})\n")
            continue
        if not user_in:
            continue

        print("\n" + "-" * 60)
        print("[User]")
        print(user_in + "\n")
        run_generate(user_in)

except KeyboardInterrupt:
    print("\n已由用户中断。")
