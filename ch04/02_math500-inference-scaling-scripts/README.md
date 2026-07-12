# 第4章：通过推理时扩展提升推理能力


&nbsp;
## 更多资料

- [cot_prompting_math500.py](cot_prompting_math500.py)：可在 MATH-500 数据集上运行链式思维（CoT）提示的独立脚本
- [self_consistency_math500.py](self_consistency_math500.py)：可在 MATH-500 数据集上运行自洽采样的独立脚本
- [run_all_experiments_math500.sh](run_all_experiments_math500.sh)：运行下方 README 表格中全部实验（第 4–12 行）的便捷 Bash 脚本

这两个评估脚本都会复用 [`reasoning_from_scratch`](../../reasoning_from_scratch) 包中的功能，避免重复造轮子。（安装细节见[第2章环境配置说明](../../ch02/02_setup-tips/python-instructions.md)。）

<br>

---

**提示**：如果你不使用 `uv`，将下方示例里的 `uv run ...py` 改成 `python ...py` 即可。

---

&nbsp;

## 链式思维提示（Chain-of-Thought Prompting）

[cot_prompting_math500.py](cot_prompting_math500.py) 实现第 4 章介绍的思维链提示方法。

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/ch04/CH04_F04_raschka.webp" width=600>

下表（第 3 行）将 CoT 与第 3 章的基线模型作对比：

|    | 方法                                     | 模型  | 准确率 | 用时     |
|----|------------------------------------------|-------|--------|----------|
| 1  | 基线（第3章），贪心解码                  | Base  | 15.2%  | 10.1 分钟 |
| 2  | 基线（第3章），贪心解码                  | Reasoning | 48.2% | 182.1 分钟 |
| 3  | 链式思维提示（“CoT”）                    | Base  | 40.6%  | 84.5 分钟 |

表中的准确率和运行时间使用 "cuda" GPU（DGX Spark）在 MATH-500 测试集全部 500 个样本上计算。

要复现第 1 行实验：

```bash
python cot_prompting_math500.py \
--which_model "base" \
--dataset_size 500
```

或使用 `uv`：

```bash
uv run cot_prompting_math500.py \
--which_model "base" \
--dataset_size 500
```

如需更多参数，请附带 `--help`。

&nbsp;
## 自洽采样（Self-Consistency Sampling）

[self_consistency_math500.py](self_consistency_math500.py) 实现第 4 章介绍的自洽采样方法。（可选地，也提供一个 [self_consistency_math500_batched.py](self_consistency_math500_batched.py) 批处理版本，能把全部 `--num_samples` 当作一个批次执行，以加速运算，不过会消耗更多显存/内存。）

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/ch04/CH04_F17_raschka.webp" width=600>

下表（第 4–12 行）对比了该方法与第3章基线（第 1–2 行）：

| 序号 | 方法                                   | 模型  | 准确率 | 用时      |
|-----|----------------------------------------|-------|--------|-----------|
| 1   | 基线（第3章），贪心解码                | Base  | 15.2%  | 10.1 分钟 |
| 2   | 基线（第3章），贪心解码                | Reasoning | 48.2% | 182.1 分钟 |
| 3   | 链式思维（“CoT”）                      | Base  | 40.6%  | 84.5 分钟 |
| 4   | 温度 + Top-p                           | Base  | 17.8%  | 30.7 分钟 |
| 5   | Top-p + 自洽（n=3）                    | Base  | 29.6%  | 97.6 分钟 |
| 6   | Top-p + 自洽（n=5）                    | Base  | 27.8%  | 116.8 分钟 |
| 7   | Top-p + 自洽（n=10）                   | Base  | 31.6%  | 300.4 分钟 |
| 8   | Top-p + CoT                            | Base  | 33.4%  | 129.2 分钟 |
| 9   | 自洽（n=3） + Top-p + CoT              | Base  | 42.2%  | 211.6 分钟 |
| 10  | 自洽（n=5） + Top-p + CoT              | Base  | 48.0%  | 452.9 分钟 |
| 11  | 自洽（n=10） + Top-p + CoT             | Base  | 52.0%  | 862.6 分钟 |
| 12  | 自洽（n=3） + Top-p + CoT              | Reasoning | 55.2% | 544.4 分钟 |

表中的准确率和运行时间使用 "cuda" GPU（DGX Spark）在 MATH-500 测试集全部 500 个样本上计算。

以下命令可复现各行实验（若不用 `uv`，将 `uv run` 换成 `python`）。

**第 4 行：**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 1 \
    --dataset_size 500
```

**第 5 行：**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500
```

**第 6 行：**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 5 \
    --dataset_size 500
```

**第 7 行：**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 10 \
    --dataset_size 500
```

**第 8 行：**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 1 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
```

**第 9 行：**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
```

**第 10 行：**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 5 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
```

**第 11 行：**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 10 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
```

**第 12 行：**

```bash
uv run self_consistency_math500.py \
    --which_model "reasoning" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
```

更多参数可通过 `--help` 查看。
