# 第 5 章：通过自我改进实现推理时扩展


&nbsp;
## 附加材料

- [self_refinement_math500.py](self_refinement_math500.py)：使用自我改进方法在 MATH-500 数据集上评估模型的独立脚本

该脚本从 [`reasoning_from_scratch`](../../reasoning_from_scratch) 包导入功能以避免重复代码。（安装详情请参阅[第 2 章环境配置说明](../../ch02/02_setup-tips/python-instructions.md)。）



<br>

---

**注意**：如果不使用 `uv`，请将下面示例中的 `uv run ...py` 替换为 `python ...py`。

---



&nbsp;

## 自我改进

[`self_refinement_math500.py`](self_refinement_math500.py) 脚本实现了第 5 章介绍的自我改进方法。


&nbsp;

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/ch05/CH05_F21_raschka.webp" width=600>

&nbsp;



| #  | 方法            | 评分器     | 迭代次数 | 模型      | 准确率   | 时间      |
|----|-----------------|-----------|------------|-----------|----------|-----------|
| 1  | 基线（第 3 章） | -         | -          | 基础      | 15.2%    | 10.1 分钟  |
| 2  | 自我改进 | 无         | 1          | 基础      | 25.0%    | 84.8 分钟  |
| 3  | 自我改进 | 无         | 2          | 基础      | 22.0%    | 165.4 分钟 |
|    |                 |           |            |           |          |           |
| 4  | 自我改进 | 启发式评分 | 1          | 基础      | 21.6%    | 84.7 分钟  |
| 5  | 自我改进 | 启发式评分 | 2          | 基础      | 20.8%    | 151.4 分钟 |
|    |                 |           |            |           |          |           |
| 6  | 自我改进 | 对数概率   | 1          | 基础      | 21.4%    | 85.3 分钟  |
| 7  | 自我改进 | 对数概率   | 2          | 基础      | 22.0%    | 165.3 分钟 |
|    |                 |           |            |           |          |           |
| 8  | 自我改进 | 提取后对数概率   | 1          | 基础      | 20.4%    | 85.0 分钟  |
| 9  | 自我改进 | 提取后对数概率   | 2          | 基础      | 21.2%    | 160.2 分钟 |
|    |                 |           |            |           |          |           |
| 10 | 基线（第 3 章） | -         | -          | 推理      | 48.2%    | 182.1 分钟 |
| 11 | 自我改进 | 无         | 1          | 推理      | 56.6%    | 498.8 分钟 |
| 12 | 自我改进 | 无         | 2          | 推理      | 56.6%    | 713.9 分钟 |
|    |                 |           |            |           |          |           |
| 13 | 自我改进 | 启发式评分 | 1          | 推理      | 57.8%    | 498.6 分钟 |
| 14 | 自我改进 | 启发式评分 | 2          | 推理      | 57.8%    | 713.9 分钟 |
|    |                 |           |            |           |          |           |
| 15 | 自我改进 | 对数概率   | 1          | 推理      | 48.4%    | 499.7 分钟 |
| 16 | 自我改进 | 对数概率   | 2          | 推理      | 48.6%    | 753.0 分钟 |

表中的准确率和运行时间是在 CUDA GPU（DGX Spark）上使用 MATH-500 测试集全部 500 个样本计算得到的。

下面给出了运行第 4–12 行自洽性实验的命令（如果不使用 `uv`，请将 `uv run` 替换为 `python`）。

**第 2 行：**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "none"
```

**第 3 行：**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "none"
```

**第 4 行：**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "heuristic"
```

**第 5 行：**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "heuristic"
```

**第 6 行：**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "logprob"
```

**第 7 行：**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "logprob"
```

**第 8 行：**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "logprob_extract"
```

**第 9 行：**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "logprob_extract"
```

**第 11 行：**

```bash
uv run self_refinement_math500.py \
    --which_model "reasoning" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "none"
```

**第 12 行：**

```bash
uv run self_refinement_math500.py \
    --which_model "reasoning" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "none"
```

**第 13 行：**

```bash
uv run self_refinement_math500.py \
    --which_model "reasoning" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "heuristic"
```

**第 14 行：**

```bash
uv run self_refinement_math500.py \
    --which_model "reasoning" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "heuristic"
```

**第 15 行：**

```bash
uv run self_refinement_math500.py \
    --which_model "reasoning" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "logprob"
```

**第 16 行：**

```bash
uv run self_refinement_math500.py \
    --which_model "reasoning" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "logprob"
```




&nbsp;

## 使用评分器打破平局的自洽性方法

[`self_consistency_scorer_math500.py`](self_consistency_scorer_math500.py) 在自洽性方法的基础上，使用第 5 章实现的评分器来打破平局。


&nbsp;

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/appendix-b/majority-vote.webp" width=600>

&nbsp;



|   | 方法                                     | 模型  | 准确率   | 时间      |
|---|------------------------------------------|-------|----------|-----------|
| 1 | 第 4 章的思维链提示基线    | 基础  | 33.4%    | 129.2 分钟 |
| 2 | 自洽性 (n=3) + 多数投票   | 基础  | 43.2%    | 328.2 分钟 |
| 3 | 自洽性 (n=3) + 启发式评分       | 基础  | 43.4%    | 326.5 分钟 |
| 4 | 自洽性 (n=3) + 平均对数概率    | 基础  | 44.8%    | 327.7 分钟 |


表中的准确率和运行时间是在 CUDA GPU（DGX Spark）上使用 MATH-500 测试集全部 500 个样本计算得到的。

下面给出了运行第 2–4 行自洽性实验的命令（如果不使用 `uv`，请将 `uv run` 替换为 `python`）。

**第 2 行：**

```bash
uv run self_consistency_scorer_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step." \
    --scoring "none"
```

**第 3 行：**

```bash
uv run self_consistency_scorer_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step." \
    --scoring "heuristic"
```

**第 4 行：**

```bash
uv run self_consistency_scorer_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step." \
    --scoring "logprob"
```

&nbsp;

## Best-of-N

[`self_consistency_scorer_math500.py`](self_consistency_scorer_math500.py) 实现了 Best-of-N 推理时扩展方法。

Best-of-N 与自洽性方法类似，都会生成多个答案；不同之处在于，它不通过多数投票选择最终答案，而是使用评分函数对所有生成答案进行评分。

[`best_of_n_math500.py`](best_of_n_math500.py) 使用第 5 章实现的评分器扩展了自洽性方法的平局处理机制。


&nbsp;

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/appendix-b/best-of-n.webp" width=600>

&nbsp;

|   | 方法                                     | 模型  | 准确率   | 时间      |
|---|------------------------------------------|-------|----------|-----------|
| 1 | 思维链提示基线 | 基础  | 33.4%    | 129.2 分钟 |
| 2 | Best-of-N (n=3) + 启发式评分              | 基础  | 40.6%    | 327.7 分钟 |
| 3 | Best-of-N (n=3) + 平均对数概率           | 基础  | 43.2%    | 330.2 分钟 |


表中的准确率和运行时间是在 CUDA GPU（DGX Spark）上使用 MATH-500 测试集全部 500 个样本计算得到的。

下面给出了运行第 2、3 行实验的命令（如果不使用 `uv`，请将 `uv run` 替换为 `python`）。

**第 2 行：**

```bash
uv run best_of_n_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
    --scoring "heuristic"
)
```

**第 3 行：**

```bash
uv run best_of_n_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
    --scoring "logprob"
)
```
