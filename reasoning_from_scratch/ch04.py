# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# 《从零构建推理模型》配套源码：https://mng.bz/lZ5B
# 代码仓库：https://github.com/rasbt/reasoning-from-scratch

from .ch02 import generate_text_basic_stream_cache
from .ch03 import extract_final_candidate
from .qwen3 import KVCache

from collections import Counter
import torch


def generate_text_stream_concat_flex(
    model, tokenizer, prompt, device, max_new_tokens,
    verbose=False,
    generate_func=None,  # 新增
    **generate_kwargs  # 新增
):

    if generate_func is None:  # 新增
        generate_func = generate_text_basic_stream_cache

    input_ids = torch.tensor(
        tokenizer.encode(prompt), device=device
        ).unsqueeze(0)

    generated_ids = []
    for token in generate_func(  # 新增
        model=model,
        token_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        **generate_kwargs,  # 新增
    ):
        next_token_id = token.squeeze(0)
        generated_ids.append(next_token_id.item())

        if verbose:
            print(
                tokenizer.decode(next_token_id.tolist()),
                end="",
                flush=True
            )
    return tokenizer.decode(generated_ids)


def plot_scores_bar(
    next_token_logits, start=19_800, end=19_900,
    arrow=True, ylabel="Logit 值"
):

    import matplotlib.pyplot as plt

    # 选取词表子区间
    x = torch.arange(start, end)

    # .cpu() 是 to(torch.device("cpu")) 的简写
    logits_section = next_token_logits[0, start:end].float().cpu()

    # 绘制 logits
    plt.bar(x, logits_section)
    plt.xlabel("词表索引")
    plt.ylabel(ylabel)

    # 突出显示最大的 logit
    if arrow:
        max_idx = torch.argmax(logits_section)
        plt.annotate(
            "Berlin",
            xy=(x[max_idx], logits_section[max_idx]),
            xytext=(x[max_idx] - 25, logits_section[max_idx] - 2),
            arrowprops={
                "facecolor": "black", "arrowstyle": "->", "lw": 1.5
            },
            fontsize=10,
        )

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def scale_logits_by_temperature(logits, temperature):
    if temperature <= 0:
        raise ValueError("温度必须为正数")
    return logits / temperature


def plot_logits_with_temperature(
    next_token_logits, start=19_800, end=19_900,
    temps=(0.5, 5.0),
):

    import matplotlib.pyplot as plt

    x = torch.arange(start, end)
    logits_orig = next_token_logits[0, start:end].float().cpu()

    # 应用温度缩放
    logits_scaled = [
        scale_logits_by_temperature(logits_orig, T) for T in temps
    ]
    # 绘制 logits
    plt.plot(x, logits_orig, label="原始 logits", lw=2)
    plt.plot(
        x, logits_scaled[0],
        label=f"T={temps[0]}（更尖锐）", ls="--", lw=1
    )
    plt.plot(
        x, logits_scaled[1],
        label=f"T={temps[1]}（更平坦）", ls=":", lw=3
    )

    # 突出显示最大的 logit
    max_idx = torch.argmax(logits_orig)
    plt.annotate(
        "Berlin",
        xy=(x[max_idx], logits_orig[max_idx]),
        xytext=(x[max_idx] - 25, logits_orig[max_idx] + 2),
        arrowprops={"facecolor": "black", "arrowstyle": "->", "lw": 1.5},
        fontsize=12,
    )

    plt.xlabel("词表索引")
    plt.ylabel("Logit 值")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def count_samples(probas, num_samples=1000, threshold=1, tokenizer=None):
    # 根据概率进行采样
    samples = torch.multinomial(
        probas.cpu(), num_samples=num_samples, replacement=True
    )

    # 统计每个索引被选中的次数
    counts = torch.bincount(samples.squeeze(0), minlength=1)

    # 打印结果
    for i, c in enumerate(counts):
        if c > threshold:
            if tokenizer is None:
                print(f"词表索引 {i}：{c.item()} 次")
            else:
                print(f"'{tokenizer.decode([i])}': {c.item()}x")


@torch.inference_mode()
def generate_text_temp_stream_cache(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None,
    temperature=0.
):
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()

    # 步骤 3.1：获取 logits
    out = model(token_ids, cache=cache)[:, -1]
    for _ in range(max_new_tokens):

        ########################################
        # 新增：
        orig_device = token_ids.device

        if temperature is None or temperature == 0.0:
            next_token = torch.argmax(out, dim=-1, keepdim=True)

        else:
            # 步骤 3.2：对 logits 应用温度缩放
            logits = scale_logits_by_temperature(out, temperature)

            # 步骤 3.3：转换为概率
            probas = torch.softmax(logits, dim=-1)

            # 步骤 3.4：根据概率采样词元
            next_token = torch.multinomial(probas.cpu(), num_samples=1)
            next_token = next_token.to(orig_device)

        #########################################
        if (eos_token_id is not None
                and torch.all(next_token == eos_token_id)):
            break

        yield next_token
        out = model(next_token, cache=cache)[:, -1]


def top_p_filter(probas, top_p):
    if top_p is None or top_p >= 1.0:
        return probas

    # 步骤 4.1：按概率降序排序
    sorted_probas, sorted_idx = torch.sort(probas, dim=1, descending=True)

    # 步骤 4.2：计算累积和
    cumprobas = torch.cumsum(sorted_probas, dim=1)

    # 步骤 4.3.1：保留词元之前的前缀累积概率小于 top_ps 的词元
    # 示例：[0.5, 0.41, 0.09] 在 top_p=0.9 时应保留前两个词元
    prefix = cumprobas - sorted_probas   # 每个词元之前的累积概率
    keep = prefix < top_p
    # 始终至少保留一个词元（当 top_p 很小或非正时的回退策略）
    keep[:, 0] = True

    # 步骤 4.3.2：将截断点之后的值置零
    kept_sorted = torch.where(
        keep, sorted_probas,
        torch.zeros_like(sorted_probas)
    )
    # 步骤 4.3.3：映射回原始顺序
    filtered = torch.zeros_like(probas).scatter(1, sorted_idx, kept_sorted)

    # 步骤 4.4：重新归一化，使总和为 1
    denom = torch.sum(filtered, dim=1, keepdim=True).clamp_min(1e-12)
    # 严格来说不必设置 keepdim=True，但这样代码也能用于批处理场景
    return filtered / denom


@torch.inference_mode()
def generate_text_top_p_stream_cache(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None,
    temperature=0.,
    top_p=None
):
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()

    # 步骤 3.1：获取 logits
    out = model(token_ids, cache=cache)[:, -1]
    for _ in range(max_new_tokens):

        orig_device = token_ids.device

        if temperature is None or temperature == 0.0:
            next_token = torch.argmax(out, dim=-1, keepdim=True)

        else:
            # 步骤 3.2：对 logits 应用温度缩放
            logits = scale_logits_by_temperature(out, temperature)

            # 步骤 3.3：转换为概率
            probas = torch.softmax(logits, dim=-1)

            # （新增）步骤 4：对概率应用 top-p 过滤
            probas = top_p_filter(probas, top_p)

            # 步骤 3.4：根据概率采样词元
            next_token = torch.multinomial(probas.cpu(), num_samples=1)
            next_token = next_token.to(orig_device)

        if (eos_token_id is not None
                and torch.all(next_token == eos_token_id)):
            break

        yield next_token
        out = model(next_token, cache=cache)[:, -1]


def self_consistency_vote(
    model, tokenizer, prompt, device,
    num_samples=10, temperature=0.8, top_p=0.9, max_new_tokens=2048,
    show_progress=True, show_long_answer=False, seed=None,
):
    full_answers, short_answers = [], []

    # 1）采样多个答案
    for i in range(num_samples):
        if seed is not None:
            torch.manual_seed(seed + i + 1)

        answer = generate_text_stream_concat_flex(
            model=model, tokenizer=tokenizer, prompt=prompt, device=device,
            max_new_tokens=max_new_tokens, verbose=show_long_answer,
            generate_func=generate_text_top_p_stream_cache,
            temperature=temperature, top_p=top_p,
        )

        # 2）从每个答案中提取最终的简短答案
        short = extract_final_candidate(
            answer, fallback="number_then_full"
        )
        full_answers.append(answer)
        short_answers.append(short)
        if show_progress:
            print(f"[样本 {i+1}/{num_samples}] → {short!r}")

    # 3）选择出现次数最多的最终答案（自洽性投票）
    counts = Counter(short_answers)
    groups = {s: [] for s in counts}
    for idx, s in enumerate(short_answers):
        groups[s].append(idx)

    mc = counts.most_common()
    if not mc:
        majority_winners, final_answer = [], None
    else:
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
