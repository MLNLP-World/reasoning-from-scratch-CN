# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# 《从零构建推理模型》配套源码：https://mng.bz/lZ5B
# 代码仓库：https://github.com/rasbt/reasoning-from-scratch

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import requests
import torch


THINK_TOKEN_ID = 151667
END_THINK_TOKEN_ID = 151668


def download_from_github(rel_path, out=None):
    github_raw_base = (  # 基础 URL
        "https://raw.githubusercontent.com/rasbt/"
        "reasoning-from-scratch/refs/heads/main/"
    )

    rel_path = Path(rel_path)
    # 使用 URL 中的文件名作为默认输出文件名
    out = Path(out) if out is not None else Path(rel_path.name)

    # 如果本地文件已存在，则跳过下载
    if out.exists():
        size_kb = out.stat().st_size / 1e3
        print(f"{out}：{size_kb:.1f} KB（已缓存）")
        return out

    # 下载文件
    r = requests.get(github_raw_base + rel_path.as_posix())
    r.raise_for_status()

    out.write_bytes(r.content)
    size_kb = out.stat().st_size / 1e3
    print(f"{out}: {size_kb:.1f} KB")


def moving_average(values, window_fraction=0.25):
    # 平滑含噪训练信号，以呈现训练中的长期趋势
    window_size = max(1, int(window_fraction * len(values)))
    smoothed = []

    for i in range(len(values)):
        start_idx = max(0, i - window_size + 1)
        window_mean = sum(values[start_idx : i + 1]) / (i - start_idx + 1)
        smoothed.append(window_mean)

    return smoothed


def plot_grpo_metrics(csv_path, columns, save_as=None):
    data = {name: {"steps": [], "values": []} for name in columns}

    # 打开并读取 CSV 日志文件
    with Path(csv_path).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row or not row.get("step"):
                continue

            # 将训练步数作为所有指标共用的 x 轴
            step = int(row["step"])

            for name in columns:
                value_str = row.get(name)
                if value_str:
                    data[name]["steps"].append(step)
                    data[name]["values"].append(float(value_str))

    # 创建固定网格，以便并排展示损失、奖励、回复长度等指标
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(6, 4))
    axes = axes.ravel()

    for i, name in enumerate(columns):
        steps = data[name]["steps"]
        values = data[name]["values"]

        # 跳过不存在的指标
        if not values:
            fig.delaxes(axes[i])
            continue

        # 评估准确率使用条形图，因为并非每一步都有数据
        if name == "eval_acc":
            axes[i].bar(steps, values, width=20)
        else:
            axes[i].plot(steps, values, alpha=0.4)
            axes[i].plot(steps, moving_average(values))

        axes[i].set_ylabel(name)

    for j in (2, 3):
        if axes[j] in fig.axes:
            axes[j].set_xlabel("Step")

    plt.tight_layout()
    if save_as is not None:
        plt.savefig(save_as)
    plt.show()


def compute_advantage_stats(rewards_list):
    # 这是 GRPO 中已经计算的内容：
    rewards = torch.tensor(rewards_list)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

    # 下面是新增的统计量：
    adv_avg = advantages.mean().item()
    adv_std = advantages.std().item()

    return advantages, adv_avg, adv_std


def sequence_logprob_and_entropy(model, token_ids, prompt_len):
    # 原有代码：与第 5 章相同
    logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)

    targets = token_ids[1:]
    selected = logprobs[:-1].gather(1, targets.unsqueeze(-1)).squeeze(-1)

    # 生成答案词元的对数概率（对答案步骤求和）
    selected_answer_logprobs = selected[prompt_len - 1:]
    logp_all_steps = torch.sum(selected_answer_logprobs)

    # 新增：计算熵
    all_answer_logprobs = logprobs[:-1][prompt_len - 1:]
    if all_answer_logprobs.numel() == 0:  # 防止模型立即返回 EOS 词元
        entropy_all_steps = logp_all_steps.new_tensor(0.0)
    else:
        all_answer_probs = torch.exp(all_answer_logprobs)  # 将对数概率转换为概率
        plogp = all_answer_probs * all_answer_logprobs     # 逐元素计算 p * log p
        step_entropy = -torch.sum(plogp, dim=-1)           # 对词表维度求和，得到每一步的熵
        entropy_all_steps = torch.mean(step_entropy)       # 对答案步骤取平均值

    return logp_all_steps, entropy_all_steps


def reward_format(
    token_ids,
    prompt_len,
    start_think_id=151667,
    end_think_id=151668,
):
    try:
        gen = token_ids[prompt_len:].tolist()
        return float(
            gen.index(start_think_id) < gen.index(end_think_id)
        )
    except ValueError:
        return 0.0
