# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# 《从零构建推理模型》来源：https://mng.bz/lZ5B
# 代码仓库：https://github.com/rasbt/reasoning-from-scratch

import argparse
import time

import torch
from datasets import load_dataset, get_dataset_config_names
from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import load_model_and_tokenizer


# 与之前相同
def format_prompt(example):
    return (
        f"{example['question']}\n"
        f"A. {example['choices'][0]}\n"
        f"B. {example['choices'][1]}\n"
        f"C. {example['choices'][2]}\n"
        f"D. {example['choices'][3]}\n"
        "Answer: "  # 末尾空格鼓励模型生成单字母的下一词元
    )


def common_prefix_len(a, b):
    i = 0
    n = min(len(a), len(b))
    while i < n and a[i] == b[i]:
        i += 1
    return i


def avg_logprob_teacher_forced(model, tokenizer, prompt_fmt, prompt, prompt_ids, letter, choice_text):
    # 构建完整答案文本，然后只提取续写部分的词元 ID
    answer_text = f"{letter}. {choice_text}"
    ids_full = tokenizer.encode(prompt + answer_text)
    j = common_prefix_len(ids_full, prompt_ids)
    if j >= len(ids_full):
        raise ValueError("续写未生成新词元。")
    answer_ids = ids_full[j:]  # 答案续写的词元

    # 模型输入为“提示词 + 除最后一个答案词元外的所有内容”（用于逐个预测后续词元）
    device = prompt_fmt.device
    if len(answer_ids) == 0:
        return float("-inf")

    answer_prefix = torch.tensor(answer_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    combined = torch.cat([prompt_fmt, answer_prefix], dim=1)

    with torch.no_grad():
        # `combined` 中每个位置的 logits
        scores = model(combined).squeeze(0)  # 形状：[num_tokens, vocab_size]
        logp = torch.log_softmax(scores, dim=-1)

    prompt_len = prompt_fmt.shape[1]
    answer_len = len(answer_ids)

    # 精确切取模型预测答案词元所对应的行
    steps = logp[prompt_len-1:prompt_len-1+answer_len, :]  # [答案长度, 词表大小]

    # 收集真实答案词元的对数概率
    targets = torch.tensor(answer_ids, dtype=torch.long, device=device).unsqueeze(1)  # [答案长度, 1]
    avg_logp = steps.gather(dim=1, index=targets).mean().item()
    return avg_logp


def predict_choice_teacher_forced(model, tokenizer, prompt_fmt, prompt, prompt_ids, example):
    scores = {}
    for letter in "ABCD":
        idx = ord(letter) - ord("A")
        choice_text = example["choices"][idx]
        scores[letter] = avg_logprob_teacher_forced(
            model, tokenizer, prompt_fmt, prompt, prompt_ids, letter, choice_text
        )
    pred = max(scores, key=scores.get)
    return pred, scores


def evaluate_mmlu_teacher_forced(
    model,
    tokenizer,
    device,
    subsets="high_school_mathematics",  # 字符串、字符串列表或 "all"
    split="test",
    verbose_every=50,
):
    if subsets == "all":
        subset_list = get_dataset_config_names("cais/mmlu")
    elif isinstance(subsets, str):
        subset_list = [s.strip() for s in subsets.split(",")] if "," in subsets else [subsets]
    else:
        subset_list = list(subsets)

    total = 0
    correct = 0
    start = time.time()

    for subset in subset_list:
        ds = load_dataset("cais/mmlu", subset, split=split)
        for ex in ds:
            prompt = format_prompt(ex)
            prompt_ids = tokenizer.encode(prompt)
            prompt_fmt = torch.tensor(prompt_ids, device=device).unsqueeze(0)

            pred, _scores = predict_choice_teacher_forced(
                model, tokenizer, prompt_fmt, prompt, prompt_ids, ex
            )

            ans = ex["answer"]
            gold = "ABCD"[ans] if isinstance(ans, int) else str(ans).strip().upper()

            total += 1
            correct += int(pred == gold)

            if verbose_every and total % verbose_every == 0:
                print(f"MMLU {total} acc={correct/total:.3f} [{subset}]")

    acc = correct / max(1, total)
    print(
        f"\nMMLU 字母准确率（教师强制）： {correct}/{total} = {acc:.2%} "
        f"in {time.time()-start:.1f}s"
    )
    return {"accuracy": acc, "num_examples": total, "subsets": subset_list, "split": split}


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="通过对 'A. <choice>' 进行教师强制对数概率计算的零样本 MMLU。"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="使用的设备：'auto'，或以下 torch 设备字符串："
             "'cpu'、'cuda'、'cuda:0'、'mps'。",
    )
    parser.add_argument(
        "--which_model",
        type=str,
        default="base",
        choices=["base", "reasoning"],
        help="要加载的模型变体",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        default="high_school_mathematics",
        help="用逗号分隔的子集名称，或 'all'。",
    )
    args = parser.parse_args()

    device = get_device() if args.device == "auto" else torch.device(args.device)
    print(f"正在使用设备： {device}")

    model, tokenizer = load_model_and_tokenizer(args.which_model, device, use_compile=False)
    model.eval()
    torch.set_float32_matmul_precision("high")

    metrics = evaluate_mmlu_teacher_forced(
        model=model,
        tokenizer=tokenizer,
        device=device,
        subsets=args.subsets,
    )
    print(metrics)


if __name__ == "__main__":
    main()
