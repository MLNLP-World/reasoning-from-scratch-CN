# 优化版 Qwen3

本书提供的 Qwen3 from-scratch 实现兼顾效率（CPU/GPU 皆可）与简洁性，同时保持易于阅读。

作为可选方案，你也可以使用即插即用的 `Qwen3Model` 版本，它在 GPU 上稍更高效。附录 C 中的 [`qwen3_optimized.py`](../../reasoning_from_scratch/qwen3_optimized.py) 相比基线 [`qwen3.py`](../../reasoning_from_scratch/qwen3.py) 主要有两点不同：

- 使用 PyTorch 内置的 `torch.nn.functional.scaled_dot_product` 实现注意力，而不是自定义实现。
- 引入改造过的 `KVCache`，会为 key/value 张量预先分配内存。这样会增加内存占用，但避免了运行时反复申请存储。

如果想更直观地比较差异，建议将 [`qwen3.py`](../../reasoning_from_scratch/qwen3.py) 与 [`qwen3_optimized.py`](../../reasoning_from_scratch/qwen3_optimized.py) 并排查看，或直接做 diff。

<br>

![](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/optimized-LLM/vscode.webp)

<br>

&nbsp;
## 使用方法

优化版代码可直接替换主章节中的导入方式，例如：

**替换前：**

```python
from reasoning_from_scratch.qwen3 import Qwen3Model
from reasoning_from_scratch.ch02 import generate_text_basic_cache
```

**替换后：**

```python
from reasoning_from_scratch.qwen3_optimized import Qwen3Model
from reasoning_from_scratch.qwen3_optimized import generate_text_basic_cache
```

&nbsp;
## 如何比较性能

要在本机评估性能，可以运行本目录下的 [`compare_inference.py`](compare_inference.py)：

```python
python compare_inference.py
```

或

```python
uv run compare_inference.py
```

可搭配以下参数：

- `--device`：选择运行设备，例如 `cpu`、`mps`、`cuda`
- `--cache`：启用 KV 缓存
- `--compile`：启用 `torch.compile`
- `--reasoning`：使用 Qwen3 推理（reasoning）版本，而非基础版。基础版针对给定提示约生成 50 个 token，reasoning 版约 2000 个 token
- `--optimize`：使用 `qwen3_optimized.py` 中的优化模型取代 `qwen3.py`

<br>

&nbsp;
### 标准模型

| 模型     | 模式              | 命令                            | 硬件            | 词元/秒       | GPU 内存（VRAM） |
| -------- | ----------------- | ------------------------------- | --------------- | ------------- | ----------------- |
| qwen3.py | Regular           | --device cpu                    | Mac Mini M4 CPU | 6             | -                 |
| qwen3.py | Regular compiled  | --device cpu --compile          | Mac Mini M4 CPU | 6             | -                 |
| qwen3.py | KV cache          | --device cpu --cache            | Mac Mini M4 CPU | 28            | -                 |
| qwen3.py | KV cache compiled | --device cpu --compile --cache  | Mac Mini M4 CPU | 68            | -                 |
|          |                   |                                 |                 |               |                   |
| qwen3.py | Regular           | --device mps                    | Mac Mini M4 GPU | 17            | -                 |
| qwen3.py | Regular compiled  | --device mps --compile          | Mac Mini M4 GPU | InductorError | -                 |
| qwen3.py | KV cache          | --device mps --cache            | Mac Mini M4 GPU | 18            | -                 |
| qwen3.py | KV cache compiled | --device mps --compile --cache  | Mac Mini M4 GPU | InductorError | -                 |
|          |                   |                                 |                 |               |                   |
| qwen3.py | Regular           | --device cuda                   | NVIDIA H100 GPU | 51            | 1.55 GB           |
| qwen3.py | Regular compiled  | --device cuda --compile         | NVIDIA H100 GPU | 164           | 1.81 GB           |
| qwen3.py | KV cache          | --device cuda --cache           | NVIDIA H100 GPU | 48            | 1.52 GB           |
| qwen3.py | KV cache compiled | --device cuda --compile --cache | NVIDIA H100 GPU | 141           | 1.81 GB           |

<br>

&nbsp;
### 优化模型

| 模型               | 模式              | 命令                                        | 硬件            | 词元/秒    | GPU 内存（VRAM） |
| ------------------ | ----------------- | ------------------------------------------- | --------------- | ---------- | ----------------- |
| qwen3_optimized.py | Regular           | --optimized --device cpu                    | Mac Mini M4 CPU | 5          | -                 |
| qwen3_optimized.py | Regular compiled  | --optimized --device cpu --compile          | Mac Mini M4 CPU | 7          | -                 |
| qwen3_optimized.py | KV cache          | --optimized --device cpu --cache            | Mac Mini M4 CPU | 49         | -                 |
| qwen3_optimized.py | KV cache compiled | --optimized --device cpu --compile --cache  | Mac Mini M4 CPU | 51         | -                 |
|                    |                   |                                             |                 |            |                   |
| qwen3_optimized.py | Regular           | --optimized --device mps                    | Mac Mini M4 GPU | 21         | -                 |
| qwen3_optimized.py | Regular compiled  | --optimized --device mps --compile          | Mac Mini M4 GPU | NameError  | -                 |
| qwen3_optimized.py | KV cache          | --optimized --device mps --cache            | Mac Mini M4 GPU | 29         | -                 |
| qwen3_optimized.py | KV cache compiled | --optimized --device mps --compile --cache  | Mac Mini M4 GPU | 38         | -                 |
|                    |                   |                                             |                 |            |                   |
| qwen3_optimized.py | Regular           | --optimized --device cuda                   | NVIDIA H100 GPU | 55         | 1.50 GB           |
| qwen3_optimized.py | Regular compiled  | --optimized --device cuda --compile         | NVIDIA H100 GPU | 173        | 1.81 GB           |
| qwen3_optimized.py | KV cache          | --optimized --device cuda --cache           | NVIDIA H100 GPU | 56         | 5.85 GB           |
| qwen3_optimized.py | KV cache compiled | --optimized --device cuda --compile --cache | NVIDIA H100 GPU | 177        | 5.85 GB           |

<br>

对比两张表可见：大多数情况下，优化版的 tokens/sec 更高。

不过要注意，在“编译 + KV 缓存”场景下，未优化版（68 tok/s）仍快于优化版（51 tok/s）。

此外，优化版在使用 KV 缓存时占用的基础内存更大（约 5.85 GB），而未优化版约 1.5 GB。这是因为优化版会为最大支持的上下文长度预先分配 KV 张量。（若未优化版也以 41k 上下文长度运行，RAM 占用会与之相近。）

**实用建议：在 CPU 上使用未优化版（搭配 `--cache` 与 `--compile`）；在 GPU 上使用优化版（同样搭配 `--cache` 与 `--compile`）。**
