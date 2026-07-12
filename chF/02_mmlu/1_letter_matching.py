import argparse
import time

import torch
from datasets import load_dataset, get_dataset_config_names
from reasoning_from_scratch.ch02 import get_device, generate_text_basic_stream_cache
from reasoning_from_scratch.ch03 import load_model_and_tokenizer


# 与主 notebook 相同
def format_prompt(example):
    return (
        f"{example['question']}\n"
        f"A. {example['choices'][0]}\n"
        f"B. {example['choices'][1]}\n"
        f"C. {example['choices'][2]}\n"
        f"D. {example['choices'][3]}\n"
        "Answer: "  # 末尾空格鼓励模型生成单字母的下一词元
    )


# 与主 notebook 相同
def predict_choice(
    model, tokenizer, prompt_fmt, max_new_tokens=8
):
    pred = None
    for t in generate_text_basic_stream_cache(
        model=model,
        token_ids=prompt_fmt,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    ):
        answer = tokenizer.decode(t.squeeze(0).tolist())
        for letter in answer:
            letter = letter.upper()
            if letter in "ABCD":
                pred = letter
                break
        if pred:  # 一出现字母就停止
            break
    return pred


def evaluate_mmlu_letter(
    model,
    tokenizer,
    device,
    subsets="high_school_mathematics",  # 字符串、字符串列表或 "all"
    split="test",
    max_new_tokens=8,
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
            tok = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
            pred = predict_choice(model, tokenizer, tok, max_new_tokens)

            ans = ex["answer"]
            # “Gold”是 MMLU 中表示正确答案（真实答案）的术语
            gold = "ABCD"[ans] if isinstance(ans, int) else str(ans).strip().upper()

            total += 1
            correct += int(pred == gold)

            if verbose_every and total % verbose_every == 0:
                print(f"MMLU {total} acc={correct/total:.3f} [{subset}]")

    acc = correct / max(1, total)
    print(
        f"\nMMLU 字母准确率： {correct}/{total} = {acc:.2%} "
        f"in {time.time()-start:.1f}s"
    )
    return {"accuracy": acc, "num_examples": total, "subsets": subset_list, "split": split}


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="零样本 MMLU 字母评估器（A/B/C/D 匹配）。"
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

    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    print(f"正在使用设备： {device}")

    model, tokenizer = load_model_and_tokenizer(args.which_model, device, use_compile=False)
    model.eval()
    torch.set_float32_matmul_precision("high")

    metrics = evaluate_mmlu_letter(
        model=model,
        tokenizer=tokenizer,
        device=device,
        subsets=args.subsets,
    )
    print(metrics)


if __name__ == "__main__":
    main()
