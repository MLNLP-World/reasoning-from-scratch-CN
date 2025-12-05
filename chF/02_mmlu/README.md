# MMLU 基准评测

这份增补资料实现了三种在 MMLU 上评估模型的方法。
- 方法 1 用作直观的入门示例
- 方法 2 是实践中最常用的方案
- 方法 3 则更稳健，更适合推理模型

- 请注意，代码会从 Hugging Face 模型库中加载 [MMLU 数据集](https://huggingface.co/datasets/cais/mmlu)，因此在运行前需要先安装 `datasets` Python 库：

```python
pip install datasets
```

或

```python
uv add datasets
```

- 在下文中，我们会将这些 MMLU 评估方法应用到 (`"high_school_mathematics"`) 子集
- 还有许多其他有趣的子集；此处出于简洁和高效选择该子集；例如你可以
  - 使用 `--subsets list` 列出其他可用子集 
  - 例如使用 `--subsets "astronomy,high_school_mathematics"` 来选择多个子集
  - 使用 `--subsets "all"` 在所有子集上评估

（为简化流程并保持代码可读性，这里专注于零样本设置，而非 5-shot。）

<br>

---

**Note**: 如果你不是 `uv` 用户，请将下面示例中的 `uv run ...py` 替换为 `python ...py`。

---

&nbsp;

## 方法 1：MMLU 字母匹配

- 让模型输出完整答案
- 提取模型生成的第一个 A/B/C/D 字母并与正确答案比较
- 这是最直观的方法，但缺点是模型可能不会回复 A/B/C/D 字母

<br>

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/mmlu/method_1.webp" width=700>

<br>

```bash
➜  02_mmlu git:(main) ✗ uv run 1_letter_matching.py --which_model base     
Using Apple Silicon GPU (MPS)
Using device: mps
✓ qwen3/qwen3-0.6B-base.pth already up-to-date
✓ qwen3/tokenizer-base.json already up-to-date
MMLU 50 acc=0.240 [high_school_mathematics]
MMLU 100 acc=0.200 [high_school_mathematics]
MMLU 150 acc=0.193 [high_school_mathematics]
MMLU 200 acc=0.235 [high_school_mathematics]
MMLU 250 acc=0.224 [high_school_mathematics]

MMLU letter accuracy: 58/270 = 21.48% in 69.1s
{'accuracy': 0.21481481481481482, 'num_examples': 270, 'subsets': ['high_school_mathematics'], 'split': 'test'}
```

```bash
➜  02_mmlu git:(main) ✗ uv run 1_letter_matching.py --which_model reasoning
Using Apple Silicon GPU (MPS)
Using device: mps
qwen3-0.6B-reasoning.pth: 100% (1433 MiB / 1433 MiB)
tokenizer-reasoning.json: 100% (10 MiB / 10 MiB)
MMLU 50 acc=0.220 [high_school_mathematics]
MMLU 100 acc=0.230 [high_school_mathematics]
MMLU 150 acc=0.220 [high_school_mathematics]
MMLU 200 acc=0.210 [high_school_mathematics]
MMLU 250 acc=0.216 [high_school_mathematics]

MMLU letter accuracy: 57/270 = 21.11% in 43.6s
{'accuracy': 0.2111111111111111, 'num_examples': 270, 'subsets': ['high_school_mathematics'], 'split': 'test'}
```



&nbsp;

## 方法 2：对数概率评分

- 将提示传入模型并获取下一个 token 的对数概率（关于 log-prob 的讨论可参见第 4 章）
- 对每个答案选项，计算如果拼接该字母，最先出现的 token ID
- 比较这四个对数概率并选择最大的那个

<br>

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/mmlu/method_2.webp" width=700>

<br>

```bash
➜  02_mmlu git:(main) ✗ uv run 2_logprob.py --which_model base 
Using Apple Silicon GPU (MPS)
Using device: mps
✓ qwen3/qwen3-0.6B-base.pth already up-to-date
✓ qwen3/tokenizer-base.json already up-to-date
MMLU 50 acc=0.360 [high_school_mathematics]
MMLU 100 acc=0.420 [high_school_mathematics]
MMLU 150 acc=0.400 [high_school_mathematics]
MMLU 200 acc=0.370 [high_school_mathematics]
MMLU 250 acc=0.344 [high_school_mathematics]

MMLU letter accuracy (log-prob): 93/270 = 34.44% in 22.5s
{'accuracy': 0.34444444444444444, 'num_examples': 270, 'subsets': ['high_school_mathematics'], 'split': 'test'}
```

```bash
➜  02_mmlu git:(main) ✗ uv run 2_logprob.py --which_model reasoning
Using Apple Silicon GPU (MPS)
Using device: mps
✓ qwen3/qwen3-0.6B-reasoning.pth already up-to-date
✓ qwen3/tokenizer-reasoning.json already up-to-date
MMLU 50 acc=0.220 [high_school_mathematics]
MMLU 100 acc=0.230 [high_school_mathematics]
MMLU 150 acc=0.220 [high_school_mathematics]
MMLU 200 acc=0.210 [high_school_mathematics]
MMLU 250 acc=0.216 [high_school_mathematics]

MMLU letter accuracy (log-prob): 57/270 = 21.11% in 22.4s
{'accuracy': 0.2111111111111111, 'num_examples': 270, 'subsets': ['high_school_mathematics'], 'split': 'test'}
```



&nbsp;

## 方法 3：Teacher forcing

- 相较于仅查看 A/B/C/D 的对数概率，更稳健的评分方式（尤其适用于推理模型）是把答案字母连同完整答案字符串一起输入模型
- 在该例中，答案字符串分别为 "A. 7"、"B. 11"、"C. 16"、"D. 8"
- 这种方法在文献中被称为 “teacher forcing”
- 它是最可靠的方案，但缺点是相较方法 2 的对数概率方式需要 4 倍时间（因为要把 4 个答案变体都输入模型）

<br>

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/mmlu/method_3.webp" width=700>

<br>

```bash
➜  02_mmlu git:(main) ✗ uv run 3_teacher_forcing.py --which_model base 
Using Apple Silicon GPU (MPS)
Using device: mps
✓ qwen3/qwen3-0.6B-base.pth already up-to-date
✓ qwen3/tokenizer-base.json already up-to-date
MMLU 50 acc=0.360 [high_school_mathematics]
MMLU 100 acc=0.310 [high_school_mathematics]
MMLU 150 acc=0.307 [high_school_mathematics]
MMLU 200 acc=0.315 [high_school_mathematics]
MMLU 250 acc=0.312 [high_school_mathematics]

MMLU letter accuracy (teacher-forced): 86/270 = 31.85% in 67.9s
{'accuracy': 0.31851851851851853, 'num_examples': 270, 'subsets': ['high_school_mathematics'], 'split': 'test'}
```

```bash
➜  02_mmlu git:(main) ✗ uv run 3_teacher_forcing.py --which_model reasoning
Using Apple Silicon GPU (MPS)
Using device: mps
✓ qwen3/qwen3-0.6B-reasoning.pth already up-to-date
✓ qwen3/tokenizer-reasoning.json already up-to-date
MMLU 50 acc=0.240 [high_school_mathematics]
MMLU 100 acc=0.250 [high_school_mathematics]
MMLU 150 acc=0.267 [high_school_mathematics]
MMLU 200 acc=0.255 [high_school_mathematics]
MMLU 250 acc=0.280 [high_school_mathematics]

MMLU letter accuracy (teacher-forced): 78/270 = 28.89% in 68.8s
{'accuracy': 0.28888888888888886, 'num_examples': 270, 'subsets': ['high_school_mathematics'], 'split': 'test'}
```



## 随机猜测基线

- 该随机猜测基线用于帮助理解上面的结果
  
- 一个在所有答案上等概率随机猜测的模型，期望精度为 $25\%$
  
- 但对于随机猜测器而言，准确率会围绕 $25\%$ 波动（取决于样本量）

- 例如，可以把一次评估建模为在 $n$ 道题中命中 $K$ 次的二项分布：

  - $K \sim \mathrm{Binomial}(n,p)$，其中 $p=\tfrac14$，$n$ 为题目数量。
  - 准确率 $A = K/n$。

- 下面以 *high_school_mathematics* 子集（$n=270$）为例

- 一般地，二项分布的性质为：

  - 均值：$\mathbb{E}[K] = np$
  - 标准差：$\sigma_K = \sqrt{np(1-p)}$

- 对准确率 $A=K/n$：

  - 均值：$\mathbb{E}[A] = p = 0.25$
  - 标准差：$\sigma_A = \sqrt{\tfrac{p(1-p)}{n}}$

- 代入 $n=270$：

  - $\mathbb{E}[A] = 25\%$  
  - $\sigma_A = \sqrt{\tfrac{0.25\cdot 0.75}{270}} \approx 2.64\%$

- 将一个标准差（$\pm 1\sigma$）的准确率区间换算为答对题数：

  - 下限：$K \le \lfloor 270\,(0.25-0.02636)\rfloor = 60$
  - 上限：$K \ge \lceil 270\,(0.25+0.02636)\rceil = 75$
  - （区间内部为 $K=61,\dots,74$，等价于 $A\in[22.36\%,\,27.64\%]$）

- 因此落在该区间之外的概率为：

  $$
  z = \pm\,\frac{75-67.5}{\sqrt{270\cdot 0.25\cdot 0.75}} \approx \pm 1.054, \qquad
  \Pr(|A-0.25|>0.02636) \approx 2\bigl(1-\Phi(1.054)\bigr) \approx 0.292.
  $$

- 即，当模型均匀随机猜测时（假设等概率），约有 $29.2\%$ 的情况会得到低于 $22.36\%$ 或高于 $27.64\%$ 的准确率
- 下面为经验性结果：


```bash
➜  02_mmlu git:(main) ✗ uv run 0_random_guessing_baseline.py --subset "high_school_mathematics"
Subset: high_school_mathematics | split: test | n=270
Gold distribution provided in the dataset:
  A: 57 (21.11%)
  B: 71 (26.30%)
  C: 71 (26.30%)
  D: 71 (26.30%)

Random guessing over 10,000 trials (uniform A/B/C/D, seed=42):
  Mean accuracy: 24.98%
  Std dev across trials: 2.65%

Selected quantiles of accuracy:
  1% quantile: 18.889%
  5% quantile: 20.741%
  25% quantile: 23.333%
  50% quantile: 24.815%
  75% quantile: 26.667%
  95% quantile: 29.259%
  99% quantile: 31.111%

Full frequency table of accuracies (rounded):
  0.160: 1 times (0.01%)
  0.170: 11 times (0.11%)
  0.180: 38 times (0.38%)
  0.190: 124 times (1.24%)
  0.200: 302 times (3.02%)
  0.210: 562 times (5.62%)
  0.220: 612 times (6.12%)
  0.230: 1254 times (12.54%)
  0.240: 1619 times (16.19%)
  0.250: 1096 times (10.96%)
  0.260: 1525 times (15.25%)
  0.270: 1248 times (12.48%)
  0.280: 572 times (5.72%)
  0.290: 565 times (5.65%)
  0.300: 281 times (2.81%)
  0.310: 132 times (1.32%)
  0.320: 28 times (0.28%)
  0.330: 24 times (0.24%)
  0.340: 5 times (0.05%)
  0.360: 1 times (0.01%)
```

