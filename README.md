# 从零构建推理模型

本仓库包含开发 LLM 推理模型所需的代码，也是图书 [*Build a Reasoning Model (From Scratch)*](https://mng.bz/lZ5B) 的官方代码仓库。

<br>
<br>

<a href="https://mng.bz/lZ5B"><img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/cover.webp?123" width="250px"></a>

（彩色印刷。）

<br>

在 [*Build a Reasoning Model (From Scratch)*](https://mng.bz/lZ5B) 一书中，你将学习并理解推理型大语言模型（LLM）的工作原理。

推理是近年来改进 LLM 最令人兴奋也最重要的进展之一，但如果只听到“推理”这个术语并从理论层面阅读相关内容，它也极易被误解。因此，本书采用动手实践的方法：从一个预训练基础 LLM 开始，再逐步通过代码自行加入推理能力，让你清楚看到它的具体工作方式。

本书介绍的方法会引导你开发一个用于教学、小而可用的推理模型。其流程与 DeepSeek R1、GPT-5 Thinking 等大规模推理模型的构建方法相呼应。此外，本书还包含加载现有预训练模型权重的代码。

- 官方[源代码仓库](https://github.com/rasbt/reasoning-from-scratch)
- [Manning 图书页面](https://mng.bz/lZ5B)（出版社网站）
- Amazon.com 图书页面（待定）
- ISBN 9781633434677

<br>
<br>

要下载本仓库副本，请单击 [Download ZIP](https://github.com/rasbt/reasoning-from-scratch/archive/refs/heads/main.zip)，或在终端中执行以下命令：

```bash
git clone --depth 1 https://github.com/rasbt/reasoning-from-scratch.git
```

<br>

> **提示：**
> 第 2 章提供了有关安装 Python、管理 Python 软件包和设置编码环境的更多建议。

<br>
<br>

## 目录（更新中）

[![Code tests Linux](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-linux.yml/badge.svg)](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-linux.yml)
[![Code tests macOS](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-macos.yml/badge.svg)](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-macos.yml)
[![Code tests Windows](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-windows.yml/badge.svg)](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-windows.yml)

- [故障排除指南](./troubleshooting.md)

| 章节标题 | 主要代码 |
| --- | --- |
| 第 1 章：理解推理模型 | 无代码 |
| 第 2 章：使用预训练 LLM 生成文本 | - [ch02_main.ipynb](ch02/01_main-chapter-code/ch02_main.ipynb)<br/>- [ch02_exercise-solutions.ipynb](ch02/01_main-chapter-code/ch02_exercise-solutions.ipynb) |
| 第 3 章：评估推理模型 | - [ch03_main.ipynb](ch03/01_main-chapter-code/ch03_main.ipynb)<br/>- [ch03_exercise-solutions.ipynb](ch03/01_main-chapter-code/ch03_exercise-solutions.ipynb) |
| 第 4 章：通过推理时扩展改进推理能力 | - [ch04_main.ipynb](ch04/01_main-chapter-code/ch04_main.ipynb)<br/>- [ch04_exercise-solutions.ipynb](ch04/01_main-chapter-code/ch04_exercise-solutions.ipynb) |
| 第 5 章：通过自我改进实现推理时扩展 | - [ch05_main.ipynb](ch05/01_main-chapter-code/ch05_main.ipynb)<br/>- [ch05_exercise-solutions.ipynb](ch05/01_main-chapter-code/ch05_exercise-solutions.ipynb) |
| 第 6 章：用强化学习训练推理模型 | - [ch06_main.ipynb](ch06/01_main-chapter-code/ch06_main.ipynb)<br/>- [ch06_exercise-solutions.ipynb](ch06/01_main-chapter-code/ch06_exercise-solutions.ipynb) |
| 第 7 章：改进用于强化学习的 GRPO | - [ch07_main.ipynb](ch07/01_main-chapter-code/ch07_main.ipynb)<br/>- [ch07_exercise-solutions.ipynb](ch07/01_main-chapter-code/ch07_exercise-solutions.ipynb) |
| 第 8 章：蒸馏推理模型以实现高效推理 | - [ch08_main.ipynb](ch08/01_main-chapter-code/ch08_main.ipynb)<br/>- [ch08_exercise-solutions.ipynb](ch08/01_main-chapter-code/ch08_exercise-solutions.ipynb) |
| 附录 A：参考资料与延伸阅读 | 无代码 |
| 附录 B：练习解答 | 代码和解答位于各章子文件夹中 |
| 附录 C：Qwen3 LLM 源代码 | - [chC_main.ipynb](chC/01_main-chapter-code/chC_main.ipynb) |
| 附录 D：使用更大的 LLM | - [chD_main.ipynb](chD/chD_main.ipynb) |
| 附录 E：批处理与面向吞吐量的执行 | - [chE_main.ipynb](chE/chE_main.ipynb) |
| 附录 F：LLM 评估的常用方法 | - [chF_main.ipynb](chF/01_main-chapter-code/chF_main.ipynb) |
| 附录 G：构建聊天界面 | - [chG](chG) |

<br>
&nbsp;

下图总结了本书介绍的主要技术。

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/mental-model.webp" width="650px">

<br>

&nbsp;
## 配套图书

请注意，*Build A Reasoning Model (From Scratch)* 是一本独立图书，专注于改进 LLM 推理的方法。

本书使用预训练的开源基础 LLM（Qwen3），并在其上从零编写代码来应用推理方法，包括推理时扩展、强化学习和蒸馏。

不过，如果你想了解传统基础 LLM 是如何实现的，可能会喜欢我的上一本书 [*Build a Large Language Model (From Scratch)*](https://amzn.to/4fqvn0D)。

<a href="https://amzn.to/4fqvn0D"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover.jpg?123" width="120px"></a>

- [Amazon 链接](https://amzn.to/4fqvn0D)
- [Manning 链接](http://mng.bz/orYv)
- [GitHub 仓库](https://github.com/rasbt/LLMs-from-scratch)

<br>
&nbsp;

## 硬件要求

本书正文各章的代码大多设计为可在消费级硬件上于合理时间内运行，不需要专用服务器硬件，让更多读者能够动手学习。此外，代码会在 GPU 可用时自动使用 GPU。第 2–4 章可在 CPU 和 GPU 上良好运行；如果要复现第 5、6 章的结果，建议使用 GPU。

（更多建议请参阅[设置提示](ch02/02_setup-tips/python-instructions.md)文档。）

&nbsp;
## 练习

本书每章都包含若干练习。解答汇总在附录 B 中，对应的代码 notebook 位于本仓库各章的主要代码文件夹中（例如 [`ch02/01_main-chapter-code/ch02_exercise-solutions.ipynb`](ch02/01_main-chapter-code/ch02_exercise-solutions.ipynb)）。

&nbsp;
## 附加材料

多个文件夹为感兴趣的读者提供了可选附加材料：

- **第 2 章：使用预训练 LLM 生成文本**
  - [可选 Python 设置和云 GPU 建议](ch02/02_setup-tips)
  - [使用针对 GPU 优化的 LLM 版本](ch02/03_optimized-LLM)
  - [在 Windows 上使用 `torch.compile()`](ch02/04_torch-compile-windows)
  - [运行推理并与模型聊天](ch02/05_use_model)
- **第 3 章：评估 LLM**
  - [MATH-500 验证器脚本](ch03/02_math500-verifier-scripts)
  - [高级解析器](ch03/03_advanced-parser)（混合 LaTeX 解析器）
- **第 4 章：通过推理时扩展改进推理能力**
  - [在 MATH-500 上进行推理扩展](ch04/02_math500-inference-scaling-scripts)（思维链提示、自洽方法）
- **第 5 章：通过自我改进实现推理时扩展**
  - [更多 MATH-500 推理扩展方法](ch05/02_math500-more-inference-scaling-scripts)（Best-of-N、自我改进）
- **第 6 章：用强化学习训练推理模型**
  - [GRPO 脚本](ch06/02_rlvr_grpo_scripts_intro)，包含批处理模式
- **第 7 章：改进用于强化学习的 GRPO**
  - [高级 GRPO 脚本](ch07/03_rlvr_grpo_scripts_advanced)（包括 DeepSeek-V3.2、Olmo3 和 GDPO 风格训练）
  - [下载训练检查点](ch07/04_download_trainining_checkpoints)（如何下载并使用第 6、7 章的 GRPO 检查点）
- **第 8 章：蒸馏推理模型以实现高效推理**
  - [生成蒸馏数据](ch08/02_generate_distillation_data)（通过 Ollama 或 OpenRouter 生成教师模型输出）
  - [使用蒸馏进行训练](ch08/04_train_with_distillation)（包括单样本和批处理蒸馏脚本）
  - [下载训练检查点](ch08/05_download_training_checkpoints)（如何下载并使用第 8 章蒸馏检查点）
  - [通过 Hugging Face 使用 Qwen3](ch08/06_use_via_huggingface)（如何通过 `transformers` 使用基础模型以及第 6–8 章检查点）
- **附录 F：LLM 评估的常用方法**
  - [MMLU 评估方法](chF/02_mmlu)
  - [LLM 排行榜](chF/03_leaderboards)
  - [LLM 作为评判者](chF/04_llm-judge)
- **附录 G：构建聊天界面**
  - [聊天界面代码](chG/01_main-chapter-code)

&nbsp;
## 问题、反馈与参与贡献

常见问题请参阅[故障排除指南](./troubleshooting.md)。

欢迎各种反馈，最好通过 [Manning 讨论论坛](https://livebook.manning.com/forum?product=raschka2&page=1)或 [GitHub Discussions](https://github.com/rasbt/reasoning-from-scratch/discussions) 分享。如果有任何问题，或只是想与他人交流想法，也欢迎在论坛中发帖。

请注意，由于本仓库包含与纸质图书对应的代码，目前无法接受扩展正文代码内容的贡献，因为这会导致代码与实体书产生偏差。保持一致有助于确保所有读者获得顺畅的学习体验。

&nbsp;
## 引用

如果本书或代码对你的研究有帮助，请考虑引用。

Chicago 格式引用：

> Raschka, Sebastian. *Build A Reasoning Model (From Scratch)*. Manning, 2025. ISBN: 9781633434677.

BibTeX 条目：

```
@book{build-llms-from-scratch-book,
  author       = {Sebastian Raschka},
  title        = {Build A Reasoning Model (From Scratch)},
  publisher    = {Manning},
  year         = {2025},
  isbn         = {9781633434677},
  url          = {https://mng.bz/lZ5B},
  github       = {https://github.com/rasbt/reasoning-from-scratch}
}
```
