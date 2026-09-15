# 第 7 章：改进强化学习中的策略优化

本节包含高级 GRPO 脚本，在第 6 章实现的基础上加入更多指标跟踪、稳定化方法和奖励建模变体。


&nbsp;
## 脚本概览

&nbsp;
### 主体脚本

- `7_3_plus_tracking.py` （*7.3 跟踪更高级的 GRPO 性能指标*）：跟踪更多性能指标（优势值统计和熵）
- `7_4_plus_clip_ratio.py` （*7.4 使用裁剪后的策略比率稳定序列级 GRPO*）：与上面类似，但使用裁剪后的策略比率计算策略梯度损失
- `7_5_plus_kl.py` （*7.5 使用 KL 项控制模型变化幅度*）：与上面类似，但加入 KL 损失项
- `7_6_plus_format_reward.py` （*7.6 加入显式格式奖励*）：与上面类似，但为 `<think>` 词元加入额外的格式奖励（与其他脚本的关键区别是：它应用于推理模型而非基础模型，因为正如主体章节所述，推理模型已经熟悉这些词元）

<br>

&nbsp;
### GRPO 技巧附加脚本

GRPO 于 2024 年 4 月首次发表（[DeepSeekMath](https://arxiv.org/abs/2402.03300)），并在 2025 年 1 月因 [DeepSeek-R1](https://arxiv.org/abs/2501.12948) 而流行起来。此后文献提出了许多改进，其中较有代表性的包括：

1. 零梯度信号过滤 ([DAPO by Yu et al., 2025](https://arxiv.org/abs/2503.14476))
2. 主动采样 (DAPO)
3. 词元级损失 (DAPO)
4. 不使用 KL 损失 (DAPO and [Dr. GRPO by Liu et al., 2025](https://arxiv.org/abs/2503.20783))
5. 提高裁剪上界 (DAPO)
6. 截断重要性采样 ([Yao et al., 2025](https://fengyao.notion.site/off-policy-rl))
7. 不进行标准差归一化 (Dr. GRPO)
8. 使用领域特定的 KL 强度进行 KL 调优；数学任务设为零 ([DeepSeek V3.2](https://arxiv.org/abs/2512.02556)
9. 重加权 KL (DeepSeek V3.2)
10. 离策略序列掩码 (DeepSeek V3.2)
11. 为 top-p / top-k 保留采样掩码 (DeepSeek V3.2)
12. 保留原始 GRPO 优势值归一化 (DeepSeek V3.2)
13. 聚合前对各项奖励分别进行组内归一化 ([GDPO by Liu et al., 2026](https://arxiv.org/abs/2601.05242))
14. 序列级重要性采样与裁剪 ([GSPO by Zheng et al., 2025](https://arxiv.org/abs/2507.18071))
15. 裁剪重要性采样权重，而不是词元更新 ([CISPO by MiniMax et al., 2025](https://arxiv.org/abs/2506.13585))

（完成主体内容后，我计划再写一篇更详细的介绍。）

<br>

下面的脚本实现了其中部分改进：

- `7_7_improvements/olmo3_style.py`：该脚本在 [7_5_plus_kl.py](7_5_plus_kl.py) 的基础上实现了类似 [Olmo 3](https://arxiv.org/abs/2512.13961) 的第 1–7 项改进

- `7_7_improvements/deepseek_v32_style.py`：该脚本在 [7_5_plus_kl.py](7_5_plus_kl.py) 的基础上实现了类似 [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) 的第 8–12 项改进

- `7_7_improvements/gdpo.py`：在 [7_6_plus_format_reward.py](7_6_plus_format_reward.py) 的基础上实现 [GDPO](https://arxiv.org/abs/2601.05242)（因为 GDPO 是针对多奖励的改进）

---

**注意**：如果不使用 `uv`，请将下面示例中的 `uv run ...py` 替换为 `python ...py`。

---


&nbsp;
