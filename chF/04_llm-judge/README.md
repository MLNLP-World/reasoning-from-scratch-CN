# LLM 评审器

这份增补资料实现了 LLM-as-a-judge 方法，即通过开源 Ollama 库调用 gpt-oss:20b 来评估 Qwen3 0.6B base 与 reasoning 变体在 MATH-500 数据集上的表现。

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/appendix-f/Appendix_F_F06_raschka.webp" width="500px">

- Ollama 是一个高效运行 LLM 的开源应用
- 它封装了 llama.cpp ([https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp))，该项目用纯 C/C++ 实现 LLM 以最大化效率
- 请注意，它是用于 LLM 推理（文本生成）的工具，而非训练或微调 LLM
- 在运行下方代码之前，请访问 [https://ollama.com](https://ollama.com) 并按照说明（例如单击 “Download” 按钮并下载与你的操作系统对应的应用）完成安装
- macOS 与 Windows 用户需运行下载好的 ollama 应用，若提示安装命令行用法，请选择 “yes”
- Linux 用户可以使用 ollama 官网上提供的安装命令
- 在本机上运行 ollama 的方式有 3 种：

**1. `ollama serve`**

- 以服务器形式运行 ollama 后端，通常监听 `http://localhost:11434`。只有当我们通过 API 调用时才会加载模型；若要在 Python 中使用 ollama，就需要这种方式。

**2. `ollama run gpt-oss:20b`**

- 这是一个便捷封装。如果服务器尚未运行，它会先启动，再（首次）下载模型，并进入可与模型交互的终端。其背后仍使用同一服务器 API。

**3. Ollama desktop app**

- 自动运行相同的后端并提供图形界面（如上图所示）。
它还会应用默认设置（系统提示、温度、停止序列），因此输出可能与直接调用 API 不同。

## 用法

下方列出可选参数及其默认值。

<br>

---

**Note**: 如果你不是 `uv` 用户，请将示例中的 `uv run ...py` 替换为 `python ...py`。

---

```bash
uv run ollama-judge.py --help
usage: ollama-judge.py [-h] [--device DEVICE]
                       [--which_model {base,reasoning}]
                       [--dataset_size DATASET_SIZE]
                       [--max_new_tokens MAX_NEW_TOKENS]
                       [--url URL]
                       [--judge_model JUDGE_MODEL]

options:
  -h, --help            show this help message and
                        exit
  --device DEVICE       Device e.g., "cpu",
                        "cuda", "cuda:0", "mps".
  --which_model {base,reasoning}
                        Candidate variant to use.
                        Defaults to "base".
  --dataset_size DATASET_SIZE
                        Number of MATH-500
                        examples to evaluate.
                        Default: 10
  --max_new_tokens MAX_NEW_TOKENS
                        Max new tokens for
                        candidate generation.
                        Default: 2048
  --url URL             Ollama chat endpoint for
                        the judge. Default: "http:
                        //localhost:11434/api/chat
                        "
  --judge_model JUDGE_MODEL
                        Judge model name (Ollama).
                        Used only for scoring.
                        Default: "gpt-oss:20b"
```

**基础模型**

```bash
➜  uv run ollama-judge.py
Using Apple Silicon GPU (MPS)
Model: base
Device: mps
✓ qwen3/qwen3-0.6B-base.pth already up-to-date
✓ qwen3/tokenizer-base.json already up-to-date
Ollama running: True
[1/10] score=5
[2/10] score=1
[3/10] score=5
[4/10] score=5
[5/10] score=3
[6/10] score=5
[7/10] score=5
[8/10] score=3
[9/10] score=5
[10/10] score=1

Summary
-------
Average score: 3.800 over 10 example(s)
Counts: 1:2 2:0 3:2 4:0 5:6
```

**推理模型**

```bash
➜  uv run ollama-judge.py --which_model reasoning
Using Apple Silicon GPU (MPS)
Model: reasoning
Device: mps
✓ qwen3/qwen3-0.6B-reasoning.pth already up-to-date
✓ qwen3/tokenizer-reasoning.json already up-to-date
Ollama running: True
[1/10] score=5
[2/10] score=5
[3/10] score=5
[4/10] score=5
[5/10] score=4
[6/10] score=5
[7/10] score=5
[8/10] score=1
[9/10] score=5
[10/10] score=3

Summary
-------
Average score: 4.300 over 10 example(s)
Counts: 1:1 2:0 3:1 4:1 5:7
```

