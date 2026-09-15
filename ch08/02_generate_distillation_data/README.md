# 第 8 章附加材料：生成蒸馏数据

此文件夹包含为数学问题生成教师模型输出的脚本；按照第 8 章所述，这些输出可作为蒸馏数据，用于训练更小的推理模型。

&nbsp;
**目录：**

- [文件](#files)
- [输入数据格式](#input-data-format)
- [输出格式](#output-format)
- [1. 使用 Ollama 在本地生成](#1-local-generation-with-ollama)
  - [1.1 Ollama 设置](#11-ollama-setup)
  - [1.2 使用 Ollama 在本地生成数据](#12-local-data-generation-with-ollama)
  - [1.3 Ollama 故障排除](#13-ollama-troubleshooting)
    - [1.3.1 Ollama 未运行](#131-ollama-not-running)
    - [1.3.2 尚未下载 Ollama 模型](#132-ollama-model-not-downloaded)
- [2. 使用 OpenRouter 在云端生成](#2-hosted-generation-with-openrouter)
  - [2.1 OpenRouter 设置](#21-openrouter-setup)
  - [2.2 使用 OpenRouter 生成数据](#22-data-generation-with-openrouter)
- [蒸馏数据集](#datasets-for-distillation)
- [数据集统计](#dataset-statistics)
- [教师模型准确率](#teacher-accuracy)
- [生成 MATH-500 蒸馏数据集](#generating-a-math-500-distillation-dataset)
- [生成包含 12,000 个 MATH 样本的蒸馏数据集](#generating-a-distillation-dataset-of-12000-math-samples)


&nbsp;
## 文件

- [average_field_lengths_json.py](average_field_lengths_json.py)：输出所生成数据集基本统计信息的工具脚本。
- [generate_with_ollama.py](generate_with_ollama.py)：使用 Ollama 生成用于蒸馏的模型答案。如果希望从可在本地运行的较小模型（如 Qwen3 4B、gpt-oss 20B、DeepSeek R1 32B 等）进行蒸馏，建议使用此脚本。
- [generate_with_openrouter.py](generate_with_openrouter.py)：通过 OpenRouter API 使用模型生成蒸馏答案。对于 DeepSeek R1（671B）或 Kimi K2.5（1T）等无法在本地运行的大型模型，建议使用此脚本。
- [math_train_sample.json](math_train_sample.json)：用于快速健全性检查的小型样本数据集。

&nbsp;
## 输入数据格式

两个脚本都通过 `--math_json` 接收 JSON 文件。每个对象至少应包含：

- `problem`（字符串）：数学问题。
- `answer`（字符串）：真实答案。

`level`、`type` 和 `unique_id` 等额外字段会被忽略。示例结构参见 [math_train_sample.json](math_train_sample.json)，该文件基于第 6、7、8 章使用的 [math_full_minus_math500.json](https://github.com/rasbt/math_full_minus_math500/blob/main/math_full_minus_math500.json)。

要处理完整的 12,000 个样本，只需下载 [math_full_minus_math500.json](https://github.com/rasbt/math_full_minus_math500/blob/main/math_full_minus_math500.json)，并通过 `--math_json math_full_minus_math500.json` 传给脚本。请注意，这会耗费很长时间，因此建议先将文件截断为几百或一千个样本。


&nbsp;
## 输出格式

两个脚本都会写入一个 JSON 数组，其中每行类似：

```
{
  "problem": "...",             # The original "problem"
  "gtruth_answer": "...",       # The original "answer"
  "message_thinking": "...",    # The model's thinking stream
  "message_content": "..."      # The model's final answer
}
```

说明：

- 输入 JSON 文件中原有的 `"answer"` 被重命名为 `"gtruth_answer"` 以避免歧义（因为“answer”是通用术语，也可能指模型答案）。
- 每处理一个样本都会增量写入文件，因此可以使用中间文件，也可以中断运行。
- 脚本提供 `--resume` 选项，可继续被中断的运行。

&nbsp;
## 1. 使用 Ollama 在本地生成


- Ollama 是用于高效运行 LLM 的开源应用程序。
- 它是 llama.cpp（[https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)）的包装器；llama.cpp 使用纯 C/C++ 实现 LLM，以最大限度提高效率。
- 请注意，它用于让 LLM 生成文本（推理），而不是训练或微调 LLM。

&nbsp;
### 1.1 Ollama 设置


- 运行以下代码前，请访问 [https://ollama.com](https://ollama.com)，按照说明安装 Ollama（例如单击“Download”按钮，下载适用于操作系统的 Ollama 应用程序）。
- macOS 和 Windows 用户请打开下载的 Ollama 应用程序；如果提示安装命令行工具，请选择“yes”。
- Linux 用户可以使用 Ollama 网站提供的安装命令。
- 可以通过三种方式在计算机上运行 Ollama：


&nbsp;
**1. `ollama serve`**

- 这会把 Ollama 后端作为服务器运行，通常地址为 `http://localhost:11434`。在通过 API 调用之前不会加载模型。如果要通过 Python 使用 Ollama，应采用此方式。

&nbsp;
**2. `ollama run deepseek-r1:8b`**

- 这是一个便捷包装命令。如果服务器尚未运行，它会先启动服务器，首次使用时下载模型，然后进入可与模型对话的交互式终端。其底层使用相同的服务器 API。
- 在 `--max_new_tokens 8192` 设置下，`deepseek-r1:8b` 模型约需 30 GB 内存。
  - 如果内存更大，建议尝试更大的模型以获得质量更高的答案，例如约需 60 GB 内存的 `deepseek-r1:32b`
  - 如果内存较少，请选择更小的模型；较小的 R1 模型列表见[此处](https://ollama.com/library/deepseek-r1)。也可以不使用 DeepSeek 模型，而是在 [Ollama 网站](https://ollama.com/)的“Search model”字段中选择其他感兴趣的模型。
  - 也可以将 `--max_new_tokens 8192` 降为 `--max_new_tokens 2048` 以减少内存用量，但这可能会过早截断某些答案。

&nbsp;
**3. Ollama 桌面应用**

- 这会自动运行同一个后端，并在其上提供图形界面（如上图所示）。
  它还会应用默认设置（系统提示词、温度和停止序列），因此输出可能与直接使用 API 时不同。

&nbsp;
### 1.2 使用 Ollama 在本地生成数据

```bash
uv run generate_with_ollama.py \
  --math_json math_train_sample.json \
  --dataset_size 5 \
  --model deepseek-r1:8b \
  --max_new_tokens 8192 \
  --out_file sample_ollama_outputs.json
```

如果不使用 `uv`，请将 `uv run` 替换为 `python`。

预期输出如下：

```
Loading model: deepseek-r1:8b
Using CUDA:0
Model ready
5/5 | MATH-500: 5/5 | ETA: 00s        
Total time: 3.2 min

Wrote 5 rows to: /home/rasbt/reasoning-from-scratch-codedev/ch08/sample_ollama_outputs.json
```

生成的 [sample_ollama_outputs.json](sample_ollama_outputs.json) 文件包含以下条目：

```json
  {
    "problem": "A rectangular band formation...",
    "gtruth_answer": "98",
    "message_thinking": "I need to find the largest number of...",
    "message_content": "The function is continuous..."
  },
```

`"message_thinking"` 字段包含思维链解释，`"message_content"` 包含最终答案。例如，可以按如下方式连接：

```python
complete_answer = f"<think>{data['message_thinking']}</think>\n\n{data['message_content']}"
```

即：

```
"<think>I need to find the largest number of...</think>

The function is continuous..."
```

&nbsp;
### 1.3 Ollama 故障排除

以下是运行 Ollama 数据生成脚本时的一些常见问题。

&nbsp;
#### 1.3.1 Ollama 未运行

如果看到类似以下的错误：

```
Loading model: deepseek-r1:32b
Using CUDA:0
Traceback (most recent call last):
  File "/home/rasbt/reasoning-from-scratch-codedev/ch08/generate_with_ollama.py", line 379, in <module>
    query_ollama_chat(
  File "/home/rasbt//reasoning-from-scratch-codedev/ch08/generate_with_ollama.py", line 235, in query_ollama_chat
    raise RuntimeError(
RuntimeError: Failed to query Ollama after 3 attempt(s). Last error: <urlopen error [Errno 111] Connection refused>
```

请确保 `ollama serve` 正在运行（可在另一个终端标签页中运行）。

&nbsp;
#### 1.3.2 尚未下载 Ollama 模型

如果看到以下错误：

```
Loading model: deepseek-r1:8b
Using CUDA:0
Traceback (most recent call last):
  File "/home/rasbt/reasoning-from-scratch-codedev/ch08/generate_with_ollama.py", line 379, in <module>
    query_ollama_chat(
  File "/home/rasbt/reasoning-from-scratch-codedev/ch08/generate_with_ollama.py", line 235, in query_ollama_chat
    raise RuntimeError(
RuntimeError: Failed to query Ollama after 3 attempt(s). Last error: HTTP 404 from Ollama at http://localhost:11434/api/chat: {"error":"model 'deepseek-r1:8b' not found"}
```

这表示尚未下载模型。此时请在另一个终端中运行 `ollama run deepseek-r1:8b`，该命令会下载模型并启动聊天。可以在聊天中试用模型，然后通过 `\bye` 退出。


&nbsp;
## 2. 使用 OpenRouter 在云端生成

Ollama 很适合在本地运行模型。不过，一些大型模型（如具有 671B 参数的 DeepSeek R1）过于庞大，无法在本地硬件上运行。对于这些情况，建议使用 [OpenRouter](https://openrouter.ai)：它通过类似 ChatGPT 的 API 提供大量托管在云端的开放权重和专有 LLM。

撰写本文时，[DeepSeek R1](https://openrouter.ai/deepseek/deepseek-r1) 每 100 万个输入词元收费 \$0.70，每 100 万个输出词元收费 \$2.50。OpenRouter 上还有许多更便宜（也更快）的模型；即便是更新的 [DeepSeek V3.2](https://openrouter.ai/deepseek/deepseek-v3.2)，每 100 万个输出词元也只需 $0.40。

下面进行一个简单的成本估算。假设输入提示词平均长 11 个词元，回复平均长 1524 个词元，为 1000 个 MATH 问题生成答案约需 $3.82。

具体计算如下：

- 输入词元总数：11 × 1000 = 11,000
- 输出词元总数：1524 × 1000 = 1,524,000
- 输入成本：`(11,000 / 1,000,000) × $0.70 = $0.0077`
- 输出成本：`(1,524,000 / 1,000,000) × $2.50 = $3.81`
- 总成本：`$3.81 + $0.0077 ≈ $3.82`

&nbsp;
### 2.1 OpenRouter 设置

设置过程很简单：在 [OpenRouter](https://openrouter.ai/) 创建账户，在 [https://openrouter.ai/settings/keys](https://openrouter.ai/settings/keys) 生成 API 密钥，并将其保存在安全位置（例如密码管理器）。


&nbsp;
### 2.2 使用 OpenRouter 生成数据

OpenRouter 脚本的工作方式与 Ollama 脚本类似，只需在命令前将 API 密钥设为环境变量：

```bash
OPENROUTER_API_KEY="YOUR_API_KEY" uv run generate_with_openrouter.py \
  --math_json math_train_sample.json \
  --dataset_size 5 \
  --model deepseek/deepseek-r1 \
  --num_processes 1 \
  --out_file sample_openrouter_outputs.json
```

如果不使用 `uv`，请将 `uv run` 替换为 `python`。

输出如下：

```
Loading model: deepseek/deepseek-r1
Using OpenRouter API: https://openrouter.ai/api/v1/chat/completions
Model ready
5/5 | MATH-500: 5/5 | ETA: 00s        
Total time: 2.2 min

Wrote 5 rows to: /Users/sebastian/Developer/reasoning-from-scratch/ch08/02_generate_distillation_data/sample_openrouter_outputs.json
```

[sample_openrouter_outputs.json](sample_openrouter_outputs.json) 输出文件的结构与 Ollama 脚本生成的文件相同。

**提示：** 如果要生成大量数据，按顺序执行蒸馏过程可能非常慢（例如使用 DeepSeek R1 生成 12,000 个答案约需 100 小时）。此时建议通过 `--num_processes` 并行运行多个数据生成进程。例如，对 DeepSeek R1 使用 `--num_processes 50`，可将运行时间从 100 小时缩短到约 2 小时。


&nbsp;
## 蒸馏数据集

通过上述 OpenRouter 方法生成的数据集集合见：[https://huggingface.co/datasets/rasbt/math_distill](https://huggingface.co/datasets/rasbt/math_distill)。

&nbsp;
## 数据集统计

要查看数据集统计信息，请使用 [average_field_lengths_json.py] 脚本：

```bash
uv run average_field_lengths_json.py \
--json_path sample_openrouter_outputs.json
```

```
tokenizer-reasoning.json: 100% (10 MiB / 10 MiB)
Records: 5
Tokenizer: reasoning
Field             AvgTokens  MinTokens  MaxToken  Count
gtruth_answer          9.40          9        10      5
message_content      196.00        166       259      5
message_thinking     933.20        449      1676      5
problem               77.80         30       121      5
```

&nbsp;
## 教师模型准确率

要计算生成该数据集的模型（即教师模型）的准确率，请使用 [../../ch03/02_math500-verifier-scripts/evaluate_json.py](../../ch03/02_math500-verifier-scripts/evaluate_json.py) 脚本：

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_json.py \
--json_path sample_openrouter_outputs.json \
--gtruth_answer gtruth_answer \
--generated_text message_content
```

```
Accuracy: 100.0% (5/5)
```

&nbsp;
## 生成 MATH-500 蒸馏数据集

要为包含 500 个样本的 MATH-500 数据集生成教师答案，可以省略 `--math_json`；两个脚本都会自动加载 `math500_test.json`（首次使用时保存本地副本）。

**Ollama**

```bash
uv run generate_with_ollama.py \
  --dataset_size 500 \
  --model deepseek-r1:8b \
  --max_new_tokens 8192 \
  --out_file math500_ollama_distill.json
```

**OpenRouter**

```bash
OPENROUTER_API_KEY="YOUR_API_KEY" uv run generate_with_openrouter.py \
  --dataset_size 500 \
  --model deepseek/deepseek-r1 \
  --num_processes 1 \
  --out_file math500_openrouter_distill.json
```

&nbsp;
## 生成包含 12,000 个 MATH 样本的蒸馏数据集

这里使用第 6、7、8 章中同一份不重叠的 12,000 样本训练集。如果尚未获取，请先下载：

```bash
curl -fL -o math_full_minus_math500.json \
https://raw.githubusercontent.com/rasbt/math_full_minus_math500/refs/heads/main/math_full_minus_math500.json
```

**Ollama**

```bash
uv run generate_with_ollama.py \
  --math_json math_full_minus_math500.json \
  --dataset_size 12000 \
  --model deepseek-r1:8b \
  --max_new_tokens 8192 \
  --resume \
  --out_file math12000_ollama_distill.json
```

**OpenRouter**

```bash
OPENROUTER_API_KEY="YOUR_API_KEY" uv run generate_with_openrouter.py \
  --math_json math_full_minus_math500.json \
  --dataset_size 12000 \
  --model deepseek/deepseek-r1 \
  --num_processes 50 \
  --resume \
  --out_file math12000_openrouter_distill.json
```

对于大规模 OpenRouter 运行，请根据账户限制和期望吞吐量调整 `--num_processes`。
