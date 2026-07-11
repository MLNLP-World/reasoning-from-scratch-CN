# 第 6 章：使用强化学习训练推理模型

&nbsp;

&nbsp;
## 附加材料s

- [rlvr_grpo_original_no_kl.py](rlvr_grpo_original_no_kl.py)：实现原始 GRPO 算法的脚本，使用可验证奖励强化学习（RLVR）训练推理模型。该算法由 [DeepSeek R1](https://arxiv.org/abs/2501.12948) 使用，最初在 [DeepSeekMath](https://arxiv.org/abs/2402.03300) 论文中提出。本脚本省略了 KL 散度项（与 [DAPO](https://arxiv.org/abs/2503.14476)、[Dr. GRPO](https://arxiv.org/abs/2503.20783)、[Olmo 3](https://arxiv.org/abs/2512.13961) 等工作的建议一致）
  - KL 散度项用于确保训练后的模型不会过度偏离原始模型，但它也可能损害性能（尤其是在数学任务上）
  - 该脚本在概念上实现了与第 6 章相同的代码，但包含两项小的性能优化：
    1. 移除 `torch.multinomial` 采样器中的 `.cpu()` 转换，使吞吐量提升 20%；实现详情请参阅 [PR #178](https://github.com/rasbt/reasoning-from-scratch/pull/178)
    2. 使用 `--skip-zero-advantage-updates` 标志时，如果所有奖励均相同，则跳过模型更新；这会进一步加快训练并降低内存需求（超过 `--max_new_tokens` 的长序列计算成本最高，而且常因在生成正确答案前达到词元上限而获得零奖励）；实现详情请参阅 [PR #186](https://github.com/rasbt/reasoning-from-scratch/pull/186)
    - 如果想查看不含上述两项改进的脚本，可以在[这里](https://github.com/rasbt/reasoning-from-scratch/blob/da009e41aacb17a433968cf84a4a6cf2a0fa4655/ch06/02_rlvr_grpo_scripts_intro/rlvr_grpo_original_no_kl.py)阅读原始代码
- [rlvr_grpo_original_no_kl_batched.py](rlvr_grpo_original_no_kl_batched.py)：与上述脚本相同，但支持批量训练。请注意，这会增加内存需求，因此可能需要减少 rollout 数量和长度。用法与上述脚本相同，只是增加了 `--num_batches`。
  - 与第 3 章的 [evaluate_math500_batched.py](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch03/02_math500-verifier-scripts/evaluate_math500_batched.py) 不同，本代码不需要从 [qwen3_batched.py](https://github.com/rasbt/reasoning-from-scratch/blob/main/reasoning_from_scratch/qwen3_batched.py) 导入 `Qwen3Model`；详情见 [PR #179](https://github.com/rasbt/reasoning-from-scratch/pull/179)

- [rlvr_grpo_original_no_kl_batched_fsdp.py](rlvr_grpo_original_no_kl_batched_fsdp.py)：与上述脚本相同，但使用 PyTorch FSDP 支持多 GPU 训练。如果有多块 GPU，推荐使用此脚本。用法与上述脚本相同，只是增加了 `--num_gpus`。

这些脚本从 [`reasoning_from_scratch`](../../reasoning_from_scratch) 包导入部分功能以避免重复代码。（安装详情请参阅[第 2 章环境配置说明](../../ch02/02_setup-tips/python-instructions.md)。）同时，脚本也重新实现了本章的核心函数，以便检查和修改。



<br>

---

**注意**：如果不使用 `uv`，请将下面示例中的 `uv run ...py` 替换为 `python ...py`。

---


&nbsp;

|      | 方法                                 | 步数 | 最大词元数 | Rollout 数 | MATH-500 准确率 | 平均词元数 |
| ---- | -------------------------------------- | ---- | ---------- | ------------ | ------------ | --------------- |
| 1    | 基础模型（第 3 章）                       | -    |            |              | 15.2%        | 78.85           |
| 2    | 推理模型（第 3 章）                  | -    |            |              | 48.2%        | 1369.79         |
| 3    | 原始 GRPO (chapter 7)              | 50   | 512        | 8            | 33.4%        | 910.33          |
| 4    | 原始 GRPO (chapter 7)              | 100  | 512        | 8            | 0.4%         | 1168.05         |
| 5    | 原始 GRPO 但无 KL（本章） | 50   | 512        | 8            | 47.4%        | 586.11          |
| 6    | 原始 GRPO 但无 KL（本章） | 100  | 512        | 8            | 44.0%        | 555.95          |
| 7    | GRPO Olmo 3 改进版（第 7 章）           | 50   | 512        | 8            | 46.4%        | 601.61          |
| 8    | GRPO Olmo 3 改进版（第 7 章）           | 100  | 512        | 8            | 45.4%        | 589.51          |
| 9    | GRPO DeepSeek V3.2 改进版（第 7 章）    | 50   | 512        | 8            | 44.2%        | 618.49          |
| 10   | GRPO DeepSeek V3.2 改进版（第 7 章）    | 100  | 512        | 8            | 45.2%        | 676.96          |

每 50 步保存一次检查点。如果使用 KeyboardInterrupt 中断脚本，也会把最后一步保存为检查点。

为降低所需计算内存，训练最多只允许生成 512 个词元（即上表中的最大词元数）。

不过，评估脚本（与第 3 章方法相同）最多允许生成 2048 个词元；上表的“平均词元数”列表示在 MATH-500 测试集上平均使用的词元数。（训练使用 MATH 数据集中与 MATH-500 测试集不重叠的 12,000 个样本。详情见 [https://github.com/rasbt/math_full_minus_math500](https://github.com/rasbt/math_full_minus_math500)。）

**第 1 行**

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model base
```

- 提示：可以在上面的执行命令中加入 `--show_eta`，显示针对当前计算机估算的脚本总运行时间

**第 2 行**

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model reasoning
```

**第 3、4 行**

```bash
uv run ../../ch07/02_rlvr_grpo_scripts_advanced/rlvr_grpo_original.py \
--num_rollouts 8 \
--max_new_tokens 512 
```

随后，在生成的检查点上运行 `evaluate_math500.py` 脚本以评估模型。例如：

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model base \
--checkpoint_path checkpoints/rlvr_grpo_original/qwen3-0.6B-rlvr-grpo-step00050.pth
```

**第 5、6 行**

```bash
uv run rlvr_grpo_original_no_kl.py \
--num_rollouts 8 \
--steps 100 \
--max_new_tokens 512
```

**第 7、8 行**

```bash
uv run ../../ch07/02_rlvr_grpo_scripts_original/rlvr_grpo_olmo3.py \
--num_rollouts 8 \
--max_new_tokens 512 
```

**第 9、10 行**

```bash
uv run ../../ch07/02_rlvr_grpo_scripts_original/rlvr_grpo_deepseek_v32.py \
--num_rollouts 8 \
--max_new_tokens 512 
```


<br>

如果内存不足，可以考虑减少 rollout 数量（`--num_rollouts`）或回复长度（`--max_new_tokens`）。下表列出了一些资源需求供参考。



| num_rollouts | max_new_tokens | 所需内存（GB） |
| ------------ | -------------- | ----------------- |
| 8            | 1024           | 30.50 GB          |
| 8            | 512            | 20.31 GB          |
| 8            | 256            | 15.60 GB          |
| 4            | 1024           | 12.80 GB          |
| 4            | 512            | 14.60 GB          |
| 4            | 256            | 10.59 GB          |


请注意，减少词元数或 rollout 数量很可能会降低性能。使用较少 rollout 时，可以把 `--accum_steps` 从 1 增加到 2 或 4（梯度累积），以一定程度提升训练稳定性；但这会增加计算时间。

请注意，使用这些设置时，原始（vanilla）GRPO 方法在训练超过 50 步后并不稳定；如果要训练 50 步以上，建议考虑第 7 章的改进版本。


<br>

原始 GRPO 算法可以通过多种方式改进，以稳定训练并提升效果；这正是[下一章](../../ch07)的主题。



&nbsp;
## 绘制训练过程

[plot_metrics.py](plot_metrics.py) 可用于绘制 CSV 格式的训练过程。`logs` 文件夹包含一次 200 步的示例运行（除将 `--max_new_tokens` 提高到 2048 外，日志使用默认设置生成）：

```bash
uv run plot_metrics.py \
--csv logs/rlvr_grpo_original_no_kl_metrics.csv \
--moving_average 20
```

（`--moving_average 20` 设置会对过去 20% 的步骤取平均，从而得到更平滑的趋势线。）

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/ch06/other/plot.webp?1" width="600px">
