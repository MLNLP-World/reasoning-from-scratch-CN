# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# 《从零构建推理模型》来源：https://mng.bz/lZ5B
# 代码仓库：https://github.com/rasbt/reasoning-from-scratch

import argparse
import json
from pathlib import Path
import time

import torch

import reasoning_from_scratch.bonus as bonus
from reasoning_from_scratch.qwen3 import (
    Qwen3Model,
    QWEN_CONFIG_06_B
)
from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import (
    eta_progress_message,
    extract_final_candidate,
    load_math500_test,
    evaluate_math500_stream,
    generate_text_stream_concat,
    load_model_and_tokenizer,
    load_tokenizer_only,
    render_prompt,
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
    parser.add_argument(
        "--hybrid_parser",
        action="store_true",
        help=(
            "使用高级混合解析器而不是"
            "本章默认解析器进行答案评分。"
        ),
    )
    return parser.parse_args()


def evaluate_math500_stream_hybrid(
    model,
    tokenizer,
    device,
    math_data,
    out_path=None,
    max_new_tokens=512,
    verbose=False,
):
    if out_path is None:
        dev_name = str(device).replace(":", "-")
        out_path = Path(f"math500-{dev_name}.jsonl")

    num_examples = len(math_data)
    num_correct = 0
    total_len = 0
    start_time = time.time()

    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(math_data, start=1):
            prompt = render_prompt(row["problem"])
            gen_text = generate_text_stream_concat(
                model,
                tokenizer,
                prompt,
                device,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
            )
            total_len += len(tokenizer.encode(gen_text))

            extracted = extract_final_candidate(gen_text)
            pred = bonus.normalize_text_hybrid(extracted)
            gold = bonus.normalize_text_hybrid(row["answer"])
            is_correct = pred == gold
            num_correct += int(is_correct)

            record = {
                "index": i,
                "problem": row["problem"],
                "gtruth_answer": row["answer"],
                "generated_text": gen_text,
                "extracted": extracted,
                "pred_normalized_hybrid": pred,
                "gold_normalized_hybrid": gold,
                "correct": bool(is_correct),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            progress_msg = eta_progress_message(
                processed=i,
                total=num_examples,
                start_time=start_time,
                show_eta=True,
                label="MATH-500",
            )
            print(progress_msg, end="\r", flush=True)
            if verbose:
                print(
                    f"\n\n{'='*50}\n{progress_msg}\n"
                    f"{'='*50}\nExtracted: {extracted}\n"
                    f"Expected:  {row['answer']}\n"
                    f"当前正确数： {num_correct}\n{'-'*50}"
                )

    seconds_elapsed = time.time() - start_time
    acc = num_correct / num_examples if num_examples else 0.0
    print(f"\nAccuracy: {acc*100:.1f}% ({num_correct}/{num_examples})")
    print(f"总耗时： {seconds_elapsed/60:.1f} min")
    avg_len = total_len / num_examples
    print(f"平均回复长度： {avg_len:.2f} tokens")
    print(f"日志已写入： {out_path}")
    return num_correct, num_examples, acc


if __name__ == "__main__":
    args = parse_args()

    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)

    which_model = args.which_model
    dataset_size = args.dataset_size
    max_new_tokens = args.max_new_tokens

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

    if args.hybrid_parser:
        backend_ok = bonus.normalize_text_hybrid(r"\frac{1}{2}") == "1/2"
        print("LaTeX 后端已就绪：", backend_ok)
        if not backend_ok:
            print('Suggestion: uv pip install "antlr4-python3-runtime==4.11.*"')

        num_correct, num_examples, acc = evaluate_math500_stream_hybrid(
            model=model,
            out_path=f"math500_{which_model}-{dev_name}-evaluate-script.jsonl",
            tokenizer=tokenizer,
            device=device,
            math_data=math_data[:dataset_size],
            max_new_tokens=max_new_tokens,
            verbose=args.verbose,
        )
    else:
        num_correct, num_examples, acc = evaluate_math500_stream(
            model=model,
            out_path=f"math500_{which_model}-{dev_name}-evaluate-script.jsonl",
            tokenizer=tokenizer,
            device=device,
            math_data=math_data[:dataset_size],
            max_new_tokens=max_new_tokens,
            verbose=args.verbose,
        )
