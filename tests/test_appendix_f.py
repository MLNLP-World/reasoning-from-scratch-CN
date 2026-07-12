# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# 《从零构建推理模型》来源：https://mng.bz/lZ5B
# 代码仓库：https://github.com/rasbt/reasoning-from-scratch

from reasoning_from_scratch.appendix_f import elo_ratings
import math


def test_elo_single_match():
    r = elo_ratings([("A", "B")], k_factor=32, initial_rating=1000)
    assert math.isclose(r["A"], 1016)
    assert math.isclose(r["B"], 984)


def test_elo_total_points_constant():
    votes = [("A", "B"), ("B", "C"), ("A", "C")]
    r = elo_ratings(votes)
    assert math.isclose(sum(r.values()), 3000)