# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# 《从零构建推理模型》配套源码：https://mng.bz/lZ5B
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
from reasoning_from_scratch.ch05 import (  # 新增（Best-of-N）
    heuristic_score,
    avg_logprob_answer
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
    scoring="heuristic",  # 新增（Best-of-N）
):
    full_answers, short_answers = [], []
    counts = Counter()
    groups = {}
    majority_winners, final_answer = [], None
    best_score, best_idx = float("-inf"), None  # 新增（Best-of-N）

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
        if short in groups:  # 新增（Best-of-N）
            groups[short].append(i)  # 新增（Best-of-N）
        else:  # 新增（Best-of-N）
            groups[short] = [i]  # 新增（Best-of-N）
        if scoring == "heuristic":  # 新增（Best-of-N）
            score = heuristic_score(answer, prompt=prompt)  # 新增（Best-of-N）
        elif scoring == "logprob":  # 新增（Best-of-N）
            score = avg_logprob_answer(  # 新增（Best-of-N）
                model=model, tokenizer=tokenizer, prompt=prompt, answer=answer, device=device  # 新增（Best-of-N）
            )  # 新增（Best-of-N）
        else:  # 新增（Best-of-N）
            raise ValueError(f"未知的评分方法：{scoring}")  # 新增（Best-of-N）
        if score > best_score:  # 新增（Best-of-N）
            best_score, best_idx = score, i  # 新增（Best-of-N）

        if show_progress:
            print(f"[样本 {i+1}/{num_samples}] → {short!r}")

        #########################################################
        # 跟踪最佳候选答案；不提前停止，以便完成 Best-of-N 评分
        #########################################################

    if best_idx is not None:  # 新增（Best-of-N）
        final_answer = short_answers[best_idx]  # 新增（Best-of-N）
        majority_winners = [final_answer]  # 新增（Best-of-N）

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
    prompt_suffix="",
    temperature=1.0,
    top_p=1.0,
    seed=None,
    num_samples=10,
    scoring="heuristic",  # 新增（Best-of-N）
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
                scoring=scoring,  # 新增（Best-of-N）
            )

            extracted = results["final_answer"]  # 新增（Best-of-N）

            # extracted = extract_final_candidate(
            #     gen_text
            # )

            # 可选：获取完整答案
            long_answer = None  # 新增（Best-of-N）
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
                    f"{'='*50}\n提取结果：{extracted}\n"
                    f"预期答案：{row['answer']}\n"
                    f"当前正确数：{num_correct}\n{'-'*50}"
                )

    seconds_elapsed = time.time() - start_time
    acc = num_correct / num_examples if num_examples else 0.0
    print(f"\n准确率：{acc*100:.1f}%（{num_correct}/{num_examples}）")
    print(f"总耗时：{seconds_elapsed/60:.1f} 分钟")
    print(f"日志已写入：{out_path}")
    return num_correct, num_examples, acc


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="要使用的设备：'auto' 或任意 PyTorch 设备字符串，例如 'cpu'、'cuda'、'cuda:0'、'mps'。",
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
        help="要评估的 MATH-500 样本数量",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="生成时允许的最大新词元数",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="为模型启用 torch.compile。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="评估时打印每个样本的正确性。",
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
        help="自洽性采样的随机种子",
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
        help="自洽性采样的样本数",
    )
    parser.add_argument(  # 新增（Best-of-N）
        "--scoring",  # 新增（Best-of-N）
        type=str,  # 新增（Best-of-N）
        default="heuristic",  # 新增（Best-of-N）
        choices=["heuristic", "logprob"],  # 新增（Best-of-N）
        help="Best-of-N 评分方法。",  # 新增（Best-of-N）
    )  # 新增（Best-of-N）
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
        prompt_suffix=args.prompt_suffix,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        num_samples=args.num_samples,
        scoring=args.scoring,  # 新增（Best-of-N）
    )
