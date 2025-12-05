# 第3章：评估推理模型


&nbsp;
## 更多资料

- [evaluate_math500.py](evaluate_math500.py)：命令行独立脚本，可在 MATH-500 数据集上评估模型
- [evaluate_math500_batched.py](evaluate_math500_batched.py)：与上类似，但会在生成阶段并行处理多条样本以提高吞吐

两个脚本都会复用 [`reasoning_from_scratch`](../../reasoning_from_scratch) 包中的功能以避免重复代码。（安装方式见[第2章环境说明](../../ch02/02_setup-tips/python-instructions.md)。）

<br>

---

**提示**：如果不使用 `uv`，请将下面示例中的 `uv run ...py` 替换为 `python ...py`。

---

&nbsp;

## `evaluate_math500.py` 用法

运行：

```bash
python evaluate_math500.py
```

或使用 `uv`：

```bash
uv run evaluate_math500.py
```

参数：

```bash
uv run evaluate_math500.py --help

options:
  -h, --help            show this help message and exit
  --device DEVICE       Device to use: "auto" (default) or any torch device string
                        (e.g., "cpu", "cuda", "cuda:0", "mps").
  --which_model {base,reasoning}
                        Model variant to load (default: "base").
  --dataset_size DATASET_SIZE
                        Number of MATH-500 examples to evaluate (default: 10).
  --max_new_tokens MAX_NEW_TOKENS
                        Max new tokens to generate (default: 2048).
  --compile             Enable torch.compile.
  --verbose             Print per-sample correctness while evaluating.
```

&nbsp;
## `evaluate_math500_batched.py` 用法

批处理版本会在生成阶段并行解码多条样本：

```bash
uv run evaluate_math500_batched.py --help
```

额外参数：

```bash
  --batch_size BATCH_SIZE
                        Number of examples to generate in parallel (default: 4).
  --disable_efficient_mode
                        Use a simpler batched inference method. Slower and more
                        memory-intensive, but easier to debug.
```

&nbsp;

**实现说明：** 默认情况下，批处理生成在某条序列输出停用 token 时就会停止该序列；若加 `--disable_efficient_mode`，所有序列都会解码到最久的一条才结束。两者只影响计算效率，不影响结果，因为超出停用 token 的 token 会被丢弃。

&nbsp;

**MPS 设备提示：**

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run evaluate_math500_batched.py
```

高效批处理推理用到的部分 PyTorch 运算尚未完全支持 MPS。如仍有问题，可改用 `--disable_efficient_mode`。

&nbsp;

- `evaluate_math500.py --dataset_size 500`

| Device / Dataset size                              | Base model | Reasoning model |
| -------------------------------------------------- | ---------- | --------------- |
| **Mac Mini M4 CPU**（500 条，顺序）                 | 43.6 min   | 未运行（发热严重） |
| **Mac Mini M4 GPU**（500 条，顺序）                 | 37.5 min   | 未运行（发热严重） |
| **DGX Spark**（500 条，顺序）                      | 10.0 min   | 182.2 min       |
| **H100 GPU**（500 条，顺序）                       | 13.3 min   | 185.4 min       |

<br><br>

- `evaluate_math500_batched.py --dataset_size 500 --batch_size 128`

| Device / Dataset size                                        | Base model | Reasoning model |
| ------------------------------------------------------------ | ---------- | --------------- |
| **Mac Mini M4 CPU**（500 条，批处理，`--batch_size 128`）       | 167.2 min  | 未运行（发热严重） |
| **Mac Mini M4 GPU**（500 条，批处理，`--batch_size 128`）       | Error*     | Error           |
| **DGX Spark**（500 条，批处理，`--batch_size 128`）            | 16.3 min   | 119.3 min       |
| **H100 GPU**（500 条，批处理，`--batch_size 128`）             | 3.3 min    | 14.6 min        |

- 基础模型准确率 15.6%（78/500）；推理模型准确率 50.8%（254/500）。
