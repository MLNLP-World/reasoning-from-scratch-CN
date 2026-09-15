# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# 《从零构建推理模型》配套源码：https://mng.bz/lZ5B
# 代码仓库：https://github.com/rasbt/reasoning-from-scratch

from .qwen3 import KVCache
from .ch03 import (
    extract_final_candidate, grade_answer, render_prompt
)
from .ch04 import top_p_filter

import json
import time
from pathlib import Path

import requests
import torch


def load_math_train(local_path="math_train.json", save_copy=True):
    local_path = Path(local_path)

    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "math_full_minus_math500/refs/heads/main/"
        "math_full_minus_math500.json"
    )
    backup_url = (
        "https://f001.backblazeb2.com/file/reasoning-from-scratch/"
        "MATH/math_full_minus_math500.json"
    )

    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
        except requests.RequestException:
            print("正在使用备用 URL。")
            r = requests.get(backup_url, timeout=30)
            r.raise_for_status()

        data = r.json()

        if save_copy:
            with local_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    return data


@torch.no_grad()
def sample_response(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
):
    input_ids = torch.tensor(
        tokenizer.encode(prompt),
        device=device
        )

    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    logits = model(input_ids.unsqueeze(0), cache=cache)[:, -1]

    generated = []
    for _ in range(max_new_tokens):
        if temperature and temperature != 1.0:
            logits = logits / temperature

        probas = torch.softmax(logits, dim=-1)
        probas = top_p_filter(probas, top_p)
        next_token = torch.multinomial(
            probas.cpu(), num_samples=1
        ).to(device)

        token_id = next_token.item()
        generated.append(token_id)

        if (
            tokenizer.eos_token_id is not None
            and token_id == tokenizer.eos_token_id
        ):
            break
        logits = model(next_token, cache=cache)[:, -1]

    full_token_ids = torch.cat(
        [input_ids,
         torch.tensor(generated, device=device, dtype=input_ids.dtype),]
    )
    return full_token_ids, input_ids.numel(), tokenizer.decode(generated)


def reward_rlvr(answer_text, ground_truth):
    extracted = extract_final_candidate(
        answer_text, fallback=None  # 要求使用 \boxed{}
    )
    if not extracted:
        return 0.0
    correct = grade_answer(extracted, ground_truth)
    return float(correct)


def sequence_logprob(model, token_ids, prompt_len):
    logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)
    selected = logprobs[:-1].gather(
        1, token_ids[1:].unsqueeze(-1)
    ).squeeze(-1)
    return torch.sum(selected[prompt_len - 1:])


def compute_grpo_loss(
    model,
    tokenizer,
    example,
    device,
    num_rollouts=2,
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.9,
):
    assert num_rollouts >= 2
    roll_logps, roll_rewards, samples = [], [], []
    prompt = render_prompt(example["problem"])

    was_training = model.training
    model.eval()

    for _ in range(num_rollouts):
        # 阶段 1：生成 rollout
        token_ids, prompt_len, text = sample_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        # 阶段 2：计算奖励
        reward = reward_rlvr(text, example["answer"])

        # 阶段 4：计算对数概率
        logp = sequence_logprob(model, token_ids, prompt_len)

        roll_logps.append(logp)
        roll_rewards.append(reward)
        samples.append(
            {
                "text": text,
                "reward": reward,
                "gen_len": token_ids.numel() - prompt_len,
            }
        )

    if was_training:
        model.train()

    # 阶段 2：收集所有奖励
    rewards = torch.tensor(roll_rewards, device=device)

    # 阶段 3：计算优势值
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

    # 阶段 4：收集所有对数概率
    logps = torch.stack(roll_logps)

    # 阶段 5：计算策略梯度损失
    pg_loss = -(advantages.detach() * logps).mean()
    loss = pg_loss  # 下一章将在这里加入 KL 项

    return {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "rewards": roll_rewards,
        "advantages": advantages.detach().cpu().tolist(),
        "samples": samples,
        "loss_tensor": loss,
    }


def save_checkpoint(model, checkpoint_dir, step, suffix=""):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"-{suffix}" if suffix else ""
    ckpt_path = (
        checkpoint_dir /
        f"qwen3-0.6B-rlvr-grpo-step{step:05d}{suffix}.pth"
    )
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def append_csv_metrics(
    csv_log_path,
    step_idx,
    total_steps,
    loss,
    reward_avg,
    avg_response_len,
):
    if not csv_log_path.exists():
        csv_log_path.write_text(
            "step,total_steps,loss,reward_avg,avg_response_len\n",
            encoding="utf-8",
        )
    with csv_log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"{step_idx},{total_steps},{loss:.6f},{reward_avg:.6f},"
            f"{avg_response_len:.6f}\n"
        )


def train_rlvr_grpo(
    model,
    tokenizer,
    math_data,
    device,
    steps=None,
    num_rollouts=2,
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.9,
    lr=1e-5,
    checkpoint_every=50,
    checkpoint_dir=".",
    csv_log_path=None,

):
    if steps is None:
        steps = len(math_data)

    # 阶段 1：初始化优化过程
    # （模型已在函数外部初始化）
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    current_step = 0

    if csv_log_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_log_path = f"train_rlvr_grpo_metrics_{timestamp}.csv"
    csv_log_path = Path(csv_log_path)
    try:
        # 阶段 2：迭代各训练步骤
        for step in range(steps):

            # 阶段 3：重置损失梯度
            # （最佳实践是在每个步骤开始时执行此操作）
            optimizer.zero_grad()

            current_step = step + 1
            example = math_data[step % len(math_data)]

            # 阶段 4：计算 GRPO 损失
            stats = compute_grpo_loss(
                model=model,
                tokenizer=tokenizer,
                example=example,
                device=device,
                num_rollouts=num_rollouts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            # 阶段 5：通过反向传播计算损失梯度
            stats["loss_tensor"].backward()

            # 裁剪过大的梯度以提升训练稳定性
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 阶段 6：使用损失梯度更新模型权重
            optimizer.step()

            # 阶段 7：收集奖励、损失和回复长度
            reward_avg = torch.tensor(stats["rewards"]).mean().item()
            step_tokens = sum(sample["gen_len"] for sample in stats["samples"])
            avg_response_len = (
                step_tokens / len(stats["samples"]) if stats["samples"] else 0.0
            )
            append_csv_metrics(
                csv_log_path,
                current_step,
                steps,
                stats["loss"],
                reward_avg,
                avg_response_len,
            )

            # 打印每一步的指标
            print(
                f"[步骤 {current_step}/{steps}] "
                f"损失={stats['loss']:.4f} "
                f"平均奖励={reward_avg:.3f} "
                f"平均回复长度={avg_response_len:.1f}"
            )

            # 每 10 步采样一次输出，以检查模型是否
            # 能够生成连贯文本
            if current_step % 10 == 0:
                print(f"[步骤 {current_step}] 输出样本")
                for i, sample in enumerate(stats["samples"][:3]):
                    text = sample["text"].replace("\n", "\\n")
                    print(
                        f"  {i+1}）奖励={sample['reward']:.3f} "
                        f"长度={sample['gen_len']}：{text}"
                    )
                print()

            # 阶段 8：保存模型检查点
            if checkpoint_every and current_step % checkpoint_every == 0:
                ckpt_path = save_checkpoint(
                    model=model,
                    checkpoint_dir=checkpoint_dir,
                    step=current_step,
                )
                print(f"检查点已保存至 {ckpt_path}")

    # 如果提前中断训练，则保存模型检查点
    except KeyboardInterrupt:
        ckpt_path = save_checkpoint(
            model=model,
            checkpoint_dir=checkpoint_dir,
            step=max(1, current_step),
            suffix="interrupt",
        )
        print(f"\n收到键盘中断。检查点已保存至 {ckpt_path}")
        return model

    return model
