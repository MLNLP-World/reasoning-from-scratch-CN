# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# 《从零构建大型语言模型》来源
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码：https://github.com/rasbt/LLMs-from-scratch


import os
from pathlib import Path

import torch
import chainlit

from reasoning_from_scratch.ch02 import (
    get_device,
)
from reasoning_from_scratch.ch02 import generate_text_basic_stream_cache
from reasoning_from_scratch.ch03 import load_model_and_tokenizer, load_tokenizer_only
from reasoning_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B


# ============================================================
# 可编辑：简单配置
# ============================================================
WHICH_MODEL = "reasoning"  # 基础模型使用 "base"
MAX_NEW_TOKENS = 38912
LOCAL_DIR = "qwen3"
# 设置 CHECKPOINT_PATH 可加载自定义 .pth 检查点，而不是
# LOCAL_DIR 中的默认权重。应使 WHICH_MODEL 与分词器保持一致
# 检查点预期的类型；第 8 章蒸馏检查点使用 "reasoning"。
# 终端示例：
#   CHECKPOINT_PATH=/absolute/path/to/model.pth \
#   uv run chainlit run qwen3_chat_interface.py
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH")
COMPILE = False
# ============================================================


DEVICE = get_device()

def load_app_model_and_tokenizer():
    if CHECKPOINT_PATH is None:
        return load_model_and_tokenizer(
            which_model=WHICH_MODEL,
            device=DEVICE,
            use_compile=COMPILE,
            local_dir=LOCAL_DIR,
        )

    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到检查点文件： {checkpoint_path}")

    tokenizer = load_tokenizer_only(which_model=WHICH_MODEL, local_dir=LOCAL_DIR)
    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.to(DEVICE)

    if COMPILE:
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        model = torch.compile(model)

    return model, tokenizer


MODEL, TOKENIZER = load_app_model_and_tokenizer()

# 虽然官方 TOKENIZER.eos_token_id 为 <|im_end|>（推理模型）
# 或 <|endoftext|>（基础模型），但某些自定义检查点两者都会输出。
EOS_TOKEN_IDS = (
    TOKENIZER.encode("<|im_end|>")[0],
    TOKENIZER.encode("<|endoftext|>")[0]
)


@chainlit.on_chat_start
async def on_start():
    chainlit.user_session.set("history", [])
    chainlit.user_session.get("history").append(
        {"role": "system", "content": "You are a helpful assistant."}
    )


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    Chainlit 主函数。
    """
    # 1）编码输入
    input_ids = TOKENIZER.encode(message.content)
    input_ids_tensor = torch.tensor(input_ids, device=DEVICE).unsqueeze(0)

    # 2）创建可接收流式内容的发出消息
    out_msg = chainlit.Message(content="")
    await out_msg.send()

    # 3）流式生成
    for tok in generate_text_basic_stream_cache(
        model=MODEL,
        token_ids=input_ids_tensor,
        max_new_tokens=MAX_NEW_TOKENS,
    ):
        token_id = tok.squeeze(0).item()
        if token_id in EOS_TOKEN_IDS:
            break
        piece = TOKENIZER.decode([token_id])
        await out_msg.stream_token(piece)

    # 4）完成流式消息
    await out_msg.update()
