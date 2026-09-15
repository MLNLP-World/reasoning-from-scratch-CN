# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# 《从零构建推理模型》来源：https://mng.bz/lZ5B
# 代码仓库：https://github.com/rasbt/reasoning-from-scratch

import argparse
import json
from pathlib import Path
import time

import torch
from collections import Counter

from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import (
    load_math500_test,
    eta_progress_message,
    render_prompt,
    grade_answer,
    load_model_and_tokenizer,
    extract_final_candidate,
)
from reasoning_from_scratch.ch04 import (
    generate_text_stream_concat_flex,
    generate_text_top_p_stream_cache,
)


def self_consistency_vote(
    model,
    tokenizer,
    prompt,
    device,
    num_samples=10,
    temperature=0.8,
    top_p=0.9,
    max_new_tokens=2048,
    show_progress=True,
    show_long_answer=False,
    seed=None,
    early_stop=True,   # 新增
):
    full_answers, short_answers = [], []
    counts = Counter()
    groups = {}
    majority_winners, final_answer = [], None

    for i in range(num_samples):
        if seed is not None:
            torch.manual_seed(seed + i + 1)

        answer = generate_text_stream_concat_flex(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            verbose=show_long_answer,
            generate_func=generate_text_top_p_stream_cache,
            temperature=temperature,
            top_p=top_p,
        )

        short = extract_final_candidate(answer, fallback="number_then_full")
        full_answers.append(answer)
        short_answers.append(short)
        counts[short] += 1
        groups.setdefault(short, []).append(i)

        if show_progress:
            print(f"[Sample {i+1}/{num_samples}] → {short!r}")

        #########################################################
        # 新增
        # 如果某个答案已达到不低于 50% 的多数票，则提前停止
        if early_stop and counts[short] > num_samples / 2:
            majority_winners = [short]
            final_answer = short
            break
        #########################################################

    if final_answer is None:
        mc = counts.most_common()
        if mc:
            top_freq = mc[0][1]
            majority_winners = [s for s, f in mc if f == top_freq]
            final_answer = mc[0][0] if len(majority_winners) == 1 else None

    return {
        "full_answers": full_answers,
        "short_answers": short_answers,
        "counts": dict(counts),
        "groups": groups,
        "majority_winners": majority_winners,
        "final_answer": final_answer,
    }


def evaluate_math500_stream(
    model,
    tokenizer,
    device,
    math_data,
    out_path=None,
    max_new_tokens=2048,
    verbose=False,
    prompt_suffix="",    # 新增
    temperature=1.0,     # 新增
    top_p=1.0,           # 新增
    seed=None,           # 新增
    num_samples=10,      # 新增
    early_stop=False,    # 新增
):

    if out_path is None:
        dev_name = str(device).replace(":", "-")
        out_path = Path(f"math500-{dev_name}.jsonl")

    num_examples = len(math_data)
    num_correct = 0
    start_time = time.time()

    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(math_data, start=1):
            prompt = render_prompt(row["problem"])

            ###################################################################
            # 新增
            prompt += prompt_suffix
            results = self_consistency_vote(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                num_samples=num_samples,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                show_progress=False,
                show_long_answer=False,
                seed=seed,
                early_stop=early_stop,
            )

            # 如果尚未确定 final_answer（平票），
            # 按首次出现顺序解决平票
            if results["final_answer"] is None:
                extracted = results["majority_winners"][0]
            else:
                extracted = results["final_answer"]

            # extracted = extract_final_candidate(
            #     gen_text
            # )

            # 可选：获取长答案
            if extracted is not None:
                for idx, s in enumerate(results["short_answers"]):
                    if s == extracted:
                        long_answer = results["full_answers"][idx]
                        break
            gen_text = long_answer
            ###################################################################

            is_correct = grade_answer(
                extracted, row["answer"]
            )
            num_correct += int(is_correct)

            record = {
                "index": i,
                "problem": row["problem"],
                "gtruth_answer": row["answer"],
                "generated_text": gen_text,
                "extracted": extracted,
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
    print(f"日志已写入： {out_path}")
    return num_correct, num_examples, acc


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
        "--verbose",
        action="store_true",
        help="评估时输出每个样本是否正确。",
    )
    parser.add_argument(
        "--prompt_suffix",
        type=str,
        default="\n\nExplain step by step.",
        help="添加思维链提示词",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="自洽采样的随机种子",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="温度缩放设置",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="top-p 过滤（核采样）的阈值",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="自洽采样的样本数",
    )
    parser.add_argument(
        "--early_stop",
        action="store_true",
        help="达到严格多数票时启用提前停止",
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
        prompt_suffix=args.prompt_suffix,  # 新增
        temperature=args.temperature,      # 新增
        top_p=args.top_p,                  # 新增
        seed=args.seed,                    # 新增
        num_samples=args.num_samples,      # 新增
        early_stop=args.early_stop         # 新增
    )
