# 故障排除指南

本页汇总了学习本书过程中常见的问题和设置建议。

&nbsp;
## JupyterLab 滚动问题

如果你在 JupyterLab 而不是 VSCode 中查看 notebook 代码，请注意，JupyterLab 的默认设置在近期版本中出现过滚动问题。建议打开 Settings -> Settings Editor，将“Windowing mode”改为“none”（如下图所示），这似乎可以解决该问题。

![Jupyter 问题 1](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_1.webp)

<br>

![Jupyter 问题 2](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_2.webp)

&nbsp;
## 第 2 章

&nbsp;
### 文件下载问题

如果文件下载出现问题，请使用[此讨论页面](https://github.com/rasbt/reasoning-from-scratch/discussions/145)。

代码会从以下 Hugging Face 位置下载文件。也可以在浏览器中手动打开这些链接，检查计算机或网络是否将其屏蔽：

- 第 2 章模型和分词器文件：[rasbt/qwen3-from-scratch](https://huggingface.co/rasbt/qwen3-from-scratch/tree/main)
- 基础模型文件：[qwen3-0.6B-base.pth](https://huggingface.co/rasbt/qwen3-from-scratch/resolve/main/qwen3-0.6B-base.pth)
- 基础分词器文件：[tokenizer-base.json](https://huggingface.co/rasbt/qwen3-from-scratch/resolve/main/tokenizer-base.json)
- 推理模型文件：[qwen3-0.6B-reasoning.pth](https://huggingface.co/rasbt/qwen3-from-scratch/resolve/main/qwen3-0.6B-reasoning.pth)
- 推理分词器文件：[tokenizer-reasoning.json](https://huggingface.co/rasbt/qwen3-from-scratch/resolve/main/tokenizer-reasoning.json)
- 第 7 章 GRPO 检查点：[rasbt/qwen3-from-scratch-grpo-checkpoints](https://huggingface.co/rasbt/qwen3-from-scratch-grpo-checkpoints/tree/main)
- 第 8 章蒸馏检查点：[rasbt/qwen3-from-scratch-distill-checkpoints](https://huggingface.co/rasbt/qwen3-from-scratch-distill-checkpoints/tree/main)

&nbsp;
#### SSL / 代理 / 证书错误

如果模型下载失败，且错误中提到 `SSL`、`CERTIFICATE_VERIFY_FAILED` 或 `ProxyError`，通常是环境问题，而不是文件缺失。

总体而言这种情况并不常见，但在公司或学校计算机上可能发生，因为 VPN、代理、防火墙或防病毒软件会拦截 HTTPS 流量。此时可以尝试：

- 检查上面列出的相关 Hugging Face URL 能否在浏览器中打开。
- 如果分词器可以下载，但 `.pth` 模型文件不能下载，代理可能屏蔽了较大的文件或 `.pth` 扩展名。
- 请求 IT 团队允许该下载，或让 Python 信任代理证书。
- 一些受管计算机的读者报告称，安装 `pip install pip-system-certs` 后问题得到解决；它会让 Python 使用操作系统证书存储。

&nbsp;
### `InductorError: CppCompileError`

如果 Linux 用户执行包含以下代码的 `torch.compile` 时看到 `InductorError: CppCompileError: C++ compile error`：

```python
Python.h: No such file or directory
81 | #include <Python.h>
| ^~~~~~~~~~
compilation terminated.
```

这表示 Python 运行时可能缺少为 CPU 编译模型所需的一些 C++ 头文件。

例如，可以检查该文件是否存在：`ls -l /usr/include/python3.12/Python.h`。

如果不存在，可以尝试安装其他 Python 运行时：

```bash
sudo apt-get install -y python3.12-dev build-essential
```

也可以在调用 `torch.compile` 前禁用 PyTorch 的 C++ 要求：

```python
import torch
import torch._inductor.config as inductor_config

inductor_config.cpp_wrapper = False

compiled_model = torch.compile(model)
```

更多背景信息另见 [#192](https://github.com/rasbt/reasoning-from-scratch/issues/192)。

&nbsp;
### Windows CPU：涉及 `algorithm` 或 `omp.h` 的 `fatal error C1083`

如果 Windows 上的 `torch.compile()` 失败，并显示：

```text
fatal error C1083: Cannot open include file: 'algorithm': No such file or directory
```

或者：

```text
fatal error C1083: Cannot open include file: 'omp.h': No such file or directory
```

问题通常出在 TorchInductor 使用的本地 Windows 编译器/OpenMP 设置，而不是本书或本仓库的代码。

一位读者在[论坛](https://livebook.manning.com/forum?product=raschka2&comment=583365)中报告了以下针对纯 CPU Intel 系统的建议：

- 升级 PyTorch 解决了缺少 `algorithm` 头文件的问题。
- 但缺少 `omp.h` 头文件的问题仍然存在。
- 使用 `"eager"` 或 `"aot_eager"` 等回退后端可让代码运行。

例如：

```python
compiled_model = torch.compile(model, backend="eager")

# or

compiled_model = torch.compile(model, backend="aot_eager")
```

请注意，这只是变通方案，并非完整修复。它可能有所帮助，但不会使用完整的 TorchInductor 编译路径，因此加速幅度可能小于正常工作的 `torch.compile()`。

**还请记住，torch.compile 并非学习本书所必需，完全可以跳过该节。**

如果仍想让它正常工作，建议在花费大量时间调试前先运行一个最小健全性检查：

```python
import torch

device = "cpu"  # or "xpu"

def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b


opt_foo = torch.compile(foo)
out = opt_foo(torch.randn(10, 10).to(device), torch.randn(10, 10).to(device))
print(out.shape)
```

如果这个小示例也失败，那么问题很可能出在 PyTorch/编译器设置，而不是本书的模型代码。

其他设置建议另见[在 Windows 上使用 `torch.compile()`](ch02/04_torch-compile-windows/README.md)和 PyTorch [Windows CPU/XPU 指南](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html)。如前所述，如果 `torch.compile()` 在你的系统上仍不稳定，可以在本书示例中跳过它。

&nbsp;
## 第 6 章

&nbsp;
### 损坏的检查点

在 `train_rlvr_grpo`（第 6 章）中，按下 `Ctrl+C` 会触发 `KeyboardInterrupt` 处理程序，保存一个带 `-interrupt` 后缀的检查点。如果在保存完成前再次按下 `Ctrl+C`，可能会在 `torch.save` 写入过程中将其打断，留下被截断的 `.pth` 文件。请等待出现 `-interrupt` 检查点消息后再退出。

损坏的模型检查点通常会在加载时引发错误，或在评估期间失败；另一个明显迹象是文件远小于预期的约 1.5 GB。

&nbsp;
## 其他问题

对于其他问题，欢迎新建 GitHub [Issue](https://github.com/rasbt/reasoning-from-scratch/issues)。
