# Copyright (c) Sebastian Raschka under Apache License 2.0
# 《从零构建推理模型》来源：https://mng.bz/lZ5B
# 代码仓库：https://github.com/rasbt/reasoning-from-scratch

import json
import math
import argparse
import torch
from reasoning_from_scratch.ch02 import get_device


def bradley_terry_torch(vote_pairs, device):

    # 收集所有不重复的模型名称
    models = sorted({m for winner, loser in vote_pairs for m in (winner, loser)})
    n = len(models)
    idx = {m: i for i, m in enumerate(models)}

    # 转换为索引张量
    winners = torch.tensor([idx[winner] for winner, _ in vote_pairs], dtype=torch.long)
    losers = torch.tensor([idx[loser] for _, loser in vote_pairs], dtype=torch.long)

    # 可学习参数
    theta = torch.nn.Parameter(torch.zeros(n - 1, device=device))
    optimizer = torch.optim.Adam([theta], lr=0.01, weight_decay=1e-4)

    def scores():
        return torch.cat([theta, torch.zeros(1, device=device)])

    for epoch in range(500):
        s = scores()
        delta = s[winners] - s[losers]       # 分数差
        loss = -torch.nn.functional.logsigmoid(delta).mean()   # 负对数似然
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # 将潜在分数转换为类似 Elo 的尺度
    with torch.no_grad():
        s = scores()
        scale = 400.0 / math.log(10.0)
        R = s * scale
        R -= R.mean()
        R += 1000.0  # 以 1000 为中心

    return {m: float(r) for m, r in zip(models, R.cpu().tolist())}


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Bradley-Terry 排行榜。")
    parser.add_argument("--path", type=str, help="投票 JSON 的路径")
    args = parser.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        votes = json.load(f)

    device = get_device()
    ratings = bradley_terry_torch(votes, device)

    leaderboard = sorted(ratings.items(),
                         key=lambda x: -x[1])
    print("\n排行榜（Bradley-Terry）")
    print("-----------------------------")
    for i, (model, score) in enumerate(leaderboard, 1):
        print(f"{i:>2}. {model:<10} {score:7.1f}")
    print()


if __name__ == "__main__":
    main()
