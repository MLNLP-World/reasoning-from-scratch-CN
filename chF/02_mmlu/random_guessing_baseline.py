# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# 《从零构建推理模型》来源：https://mng.bz/lZ5B
# 代码仓库：https://github.com/rasbt/reasoning-from-scratch

import argparse
import random
import statistics as stats
from collections import Counter

from datasets import load_dataset


# Gold letter 是 MMLU 中表示正确答案字母的术语
def gold_letter(ans):
    if isinstance(ans, int):
        return "ABCD"[ans]
    s = str(ans).strip().upper()
    return s if s in {"A", "B", "C", "D"} else s[:1]


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="显示 MMLU 子集的真实答案分布和随机猜测基线。"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="high_school_mathematics",
        help="MMLU 子集名称。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机猜测基线的随机种子。",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10_000,
        help="随机猜测试验次数。",
    )
    args = parser.parse_args()

    ds = load_dataset("cais/mmlu", args.subset, split="test")

    labels = [gold_letter(ex["answer"]) for ex in ds]
    n = len(labels)
    counts = Counter(labels)

    print(f"Subset: {args.subset} | split: test | n={n}")
    print("数据集中提供的真实答案分布：")
    for letter in "ABCD":
        c = counts.get(letter, 0)
        pct = (c / n) if n else 0.0
        print(f"  {letter}: {c} ({pct:.2%})")

    if n == 0:
        print("\n没有样本，无法定义基线。")
        return

    # 重复随机猜测
    rng = random.Random(args.seed)
    accs = []
    for _ in range(args.trials):
        guesses = [rng.choice("ABCD") for _ in range(n)]
        correct = sum(1 for g, y in zip(guesses, labels) if g == y)
        accs.append(correct / n)

    mean_acc = stats.mean(accs)
    sd_acc = stats.stdev(accs) if len(accs) > 1 else 0.0

    print(f"\n随机猜测试验次数： {args.trials:,} trials (uniform A/B/C/D, seed={args.seed}):")
    print(f"  平均准确率： {mean_acc:.2%}")
    print(f"  各次试验的标准差： {sd_acc:.2%}")

    # 分位数
    qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    accs_sorted = sorted(accs)
    print("\n选定的准确率分位数：")
    for q in qs:
        idx = int(q * len(accs_sorted))
        print(f"  {q:.0%} quantile: {accs_sorted[idx]:.3%}")

    # 频数表（四舍五入）
    acc_counts = Counter(round(a, 2) for a in accs)
    print("\n完整的准确率频数表（四舍五入）：")
    for acc_val in sorted(acc_counts):
        freq = acc_counts[acc_val]
        pct = freq / args.trials
        print(f"  {acc_val:.3f}: {freq} times ({pct:.2%})")


if __name__ == "__main__":
    main()
