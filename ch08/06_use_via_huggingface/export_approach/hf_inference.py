# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# 《从零构建推理模型》来源：https://mng.bz/lZ5B
# 代码仓库：https://github.com/rasbt/reasoning-from-scratch

import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import render_prompt


def wrap_prompt(prompt, tokenizer, tokenizer_kind):
    if tokenizer_kind == "reasoning":
        messages = [{"role": "user", "content": prompt}]
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    return prompt


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 model.generate(...) 运行转换后的 Qwen3 Hugging Face 模型。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="导出的 Hugging Face 模型目录路径。",
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
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
    )
    config = AutoConfig.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    prompt = render_prompt(args.prompt)
    prompt = wrap_prompt(
        prompt=prompt,
        tokenizer=tokenizer,
        tokenizer_kind=config.tokenizer_kind,
    )

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, prompt_len:]

    print()
    print("=" * 60)
    print(f"device         : {device}")
    print(f"tokenizer_kind : {config.tokenizer_kind}")
    print("=" * 60)
    print()
    print(tokenizer.decode(generated_ids, skip_special_tokens=False))
    print()


if __name__ == "__main__":
    main()
