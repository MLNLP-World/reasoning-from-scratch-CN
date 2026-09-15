# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# 《从零构建推理模型》配套源码：https://mng.bz/lZ5B
# 代码仓库：https://github.com/rasbt/reasoning-from-scratch

from .ch03 import extract_final_candidate, render_prompt
from .ch04 import (
    generate_text_stream_concat_flex,
    generate_text_top_p_stream_cache
)
import math
import torch


def heuristic_score(
    answer,
    prompt=None,  # 会被忽略的占位参数
    brevity_bonus=500.0,
    boxed_bonus=2.0,
    extract_bonus=1.0,
    fulltext_bonus=0.0,
):
    score = 0.0

    # 奖励含有最终 boxed 值的答案
    cand = extract_final_candidate(answer, fallback="none")
    if cand:
        score += boxed_bonus

    # 如果答案没有 boxed 值，则给予较低奖励
    else:
        cand = extract_final_candidate(answer, fallback="number_only")
        if cand:
            score += extract_bonus
        else:
            cand = extract_final_candidate(
                answer, fallback="number_then_full"
            )
            if cand:
                score += fulltext_bonus

    # 加入随文本长度增加而衰减的简洁性奖励
    score += 1.5 * math.exp(-len(answer) / brevity_bonus)
    return score


@torch.inference_mode()
def calc_next_token_probas(model, tokenizer, prompt, device):

    token_ids = torch.tensor(tokenizer.encode(prompt), device=device)

    # 以类似文本生成函数的方式获取 logits 和概率
    logits = model(token_ids.unsqueeze(0)).squeeze(0)
    all_probas = torch.softmax(logits, dim=-1)

    # 需要评分的位置（这里为全部位置）
    t_idx = torch.arange(0, token_ids.shape[0] - 1, device=device)

    # 由于已有文本，因此知道真实的下一个词元
    next_ids = token_ids[1:]

    # 获取每个下一词元的概率
    next_token_probas = all_probas[t_idx, next_ids]

    print(
        "下一词元概率：",
        [p.item() for p in next_token_probas]
    )

    # 序列的似然是各概率分数的乘积
    print(
        "联合概率：",
        torch.prod(next_token_probas)
    )


@torch.inference_mode()
def calc_next_token_logprobas(model, tokenizer, prompt, device, show=True):

    token_ids = torch.tensor(tokenizer.encode(prompt), device=device)

    logits = model(token_ids.unsqueeze(0)).squeeze(0)
    # 现在改用 log_softmax
    all_logprobas = torch.log_softmax(logits, dim=-1)

    t_idx = torch.arange(0, token_ids.shape[0] - 1, device=device)
    next_ids = token_ids[1:]
    next_token_logprobas = all_logprobas[t_idx, next_ids]

    # 将乘积替换为求和
    sum_next_token_logprobas = torch.sum(next_token_logprobas)

    if show:
        print("下一词元对数概率：", next_token_logprobas)
        print("联合对数概率：", sum_next_token_logprobas)
    else:
        return next_token_logprobas, sum_next_token_logprobas


@torch.inference_mode()
def avg_logprob_answer(model, tokenizer, prompt, answer, device="cpu"):

    # 分别编码提示词和答案词元，以便稍后获得提示词长度
    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(answer)
    full_ids = torch.tensor(prompt_ids + answer_ids, device=device)

    # 与前面的 calc_next_token_logprobas 相同
    logits = model(full_ids.unsqueeze(0)).squeeze(0)
    logprobs = torch.log_softmax(logits, dim=-1)

    # 答案词元所对应位置的索引范围
    start = len(prompt_ids) - 1
    end = full_ids.shape[0] - 1

    # 与之前相同，只是改用 start 和 end
    t_idx = torch.arange(start, end, device=device)
    next_tokens = full_ids[start + 1 : end + 1]
    next_token_logps = logprobs[t_idx, next_tokens]

    # 对答案词元分数取平均值
    return torch.mean(next_token_logps).item()


def make_critique_prompt(raw_prompt, draft):
    return (
        "You are a meticulous reviewer. Identify logical errors, missing "
        "steps, or arithmetic mistakes. If the answer seems correct, "
        "say so briefly. Then propose a concise plan to fix issues.\n\n"
        f"Question:\n{raw_prompt}\n\n"
        f"Draft answer:\n{draft}\n\n"
        "Write a short critique and bullet-point fix plan "
        "(under ~120 words).\n"
        "Critique:"
    )


def make_refine_prompt(raw_prompt, draft, critique):
    return (
        "Revise the answer using the critique. Keep it concise and "
        "end with a final boxed result: \\boxed{ANSWER}\n\n"
        f"Question:\n{raw_prompt}\n\n"
        f"Previous answer:\n{draft}\n\n"
        f"Critique:\n{critique}\n\n"
        "Revised answer:"
    )


def self_refinement_loop(
    model,
    tokenizer,
    raw_prompt,
    device,
    iterations=2,
    max_response_tokens=2048,
    max_critique_tokens=256,
    score_fn=None,
    prompt_renderer=render_prompt,
    prompt_suffix="",
    verbose=False,
    temperature=0.7,
    top_p=0.9,
):
    steps = []

    # 初始回复（草稿）
    prompt = prompt_renderer(raw_prompt) + prompt_suffix
    current_full = generate_text_stream_concat_flex(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=max_response_tokens,
        verbose=False,
        generate_func=generate_text_top_p_stream_cache,
        temperature=temperature,
        top_p=top_p,
    )

    current_extracted = extract_final_candidate(
        current_full, fallback="number_then_full"
    )
    if score_fn:
        current_score = score_fn(answer=current_full, prompt=prompt)
    else:
        current_score = 0.0

    # 执行一次或多次迭代
    for it in range(iterations):
        draft_before_full = current_full
        draft_before_extracted = current_extracted
        score_before = current_score

        # 批判性分析回复
        critique_prompt = make_critique_prompt(
            raw_prompt, draft_before_full
        )
        critique_full = generate_text_stream_concat_flex(
            model=model,
            tokenizer=tokenizer,
            prompt=critique_prompt,
            device=device,
            max_new_tokens=max_critique_tokens,
            verbose=False,
            generate_func=generate_text_top_p_stream_cache,
            temperature=temperature,
            top_p=top_p,
        )

        # 改进回复
        refine_prompt = make_refine_prompt(
            raw_prompt, draft_before_full, critique_full
        )
        revised_full = generate_text_stream_concat_flex(
            model=model,
            tokenizer=tokenizer,
            prompt=refine_prompt,
            device=device,
            max_new_tokens=max_response_tokens,
            verbose=False,
            generate_func=generate_text_top_p_stream_cache,
            temperature=temperature,
            top_p=top_p,
        )

        revised_extracted = extract_final_candidate(
            revised_full, fallback="number_then_full"
        )
        if score_fn:
            revised_score = score_fn(
                answer=revised_full, prompt=prompt  # 这里仍使用原始提示词
            )
        else:
            revised_score = 0.0

        # 记录结果
        step = {
            "iteration": it + 1,
            "draft_full": draft_before_full,
            "draft_extracted": draft_before_extracted,
            "critique": critique_full,
            "revised_full": revised_full,
            "revised_extracted": revised_extracted,
            "score_before": score_before,
            "score_after": revised_score,
        }
        steps.append(step)

        if verbose:
            print(
                f"[改进 {it+1}/{iterations}]"
                f"\n当前答案：{draft_before_extracted}"
                f"\n修改后答案：{revised_extracted}"
                f"\n修改前得分：{score_before:.3f}"
                f"\n修改后得分：{revised_score:.3f}"
                f"\n{'=' * 25}"
            )

        # 如果修改后的回复不差于原回复，则接受它
        if revised_score >= current_score:
            current_full = revised_full
            current_extracted = revised_extracted
            current_score = revised_score

    return {
        "final_full": current_full,
        "final_extracted": current_extracted,
        "steps": steps,
    }
