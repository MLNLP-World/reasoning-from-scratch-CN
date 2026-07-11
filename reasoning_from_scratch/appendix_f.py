
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# 《从零构建推理模型》配套源码：https://mng.bz/lZ5B
# 代码仓库：https://github.com/rasbt/reasoning-from-scratch

from reasoning_from_scratch.ch02 import generate_text_basic_stream_cache


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
        if pred:  # 一旦出现字母就停止
            break
    return pred


def elo_ratings(vote_pairs, k_factor=32, initial_rating=1000):
    # 用相同的初始等级分初始化所有模型
    ratings = {
        model: initial_rating
        for pair in vote_pairs
        for model in pair
    }

    # 每场对局后更新等级分
    for winner, loser in vote_pairs:
        rating_winner, rating_loser = ratings[winner], ratings[loser]

        # 根据等级分计算当前胜者的预期得分
        expected_winner = 1.0 / (
            1.0 + 10 ** ((rating_loser - rating_winner) / 400.0)
        )

        # k_factor 决定等级分更新的敏感程度
        ratings[winner] = (
            rating_winner + k_factor * (1 - expected_winner)
        )
        ratings[loser] = (
            rating_loser + k_factor * (0 - (1 - expected_winner))
        )

    return ratings
