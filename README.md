<h1 align="center">
  <img src="./imgs/icon/ai.png" width="30" />《从零构建推理大模型》
</h1>

<p align="center">
  <img alt="version" src="https://img.shields.io/badge/version-v0.1.0-brightgreen" />
  <img alt="Status-building" src="https://img.shields.io/badge/Status-building-blue" />
  <img alt="PRs-Welcome" src="https://img.shields.io/badge/PRs-Welcome-red" />
  <a href="https://github.com/MLNLP-World/reasoning-from-scratch-CN/stargazers">
    <img alt="Stars" src="https://img.shields.io/github/stars/MLNLP-World/reasoning-from-scratch-CN?style=social&amp;label=Stars" />
  </a>
  <a href="https://github.com/MLNLP-World/reasoning-from-scratch-CN/network/members">
    <img alt="Forks" src="https://img.shields.io/github/forks/MLNLP-World/reasoning-from-scratch-CN?style=social&amp;label=Forks" />
  </a>
  <a href="https://github.com/MLNLP-World/reasoning-from-scratch-CN/issues">
    <img alt="Issues" src="https://img.shields.io/github/issues/MLNLP-World/reasoning-from-scratch-CN?style=social&amp;label=Issues" />
  </a>
</p>

<p align="center">
  <a href="#项目动机">项目动机</a> /
  <a href="#课程简介">课程简介</a> /
  <a href="#课程资源">课程资源</a> /
  <a href="#原书readme">原书Readme</a> /
  <a href="#组织者列表">组织者列表</a> /
  <a href="#贡献者列表">贡献者列表</a>
</p>

---

<h2 id="项目动机">
  <img src="./imgs/icon/motivation.png" width="25" /> 项目动机
</h2>

原项目与地址:[《reasoning-from-scratch》](https://github.com/rasbt/reasoning-from-scratch.git)

本项目是对GitHub项目[《reasoning-from-scratch》](https://github.com/rasbt/reasoning-from-scratch.git)内容的中文翻译，包括详细的markdown 笔记和相关的jupyter 代码。翻译过程中，我们尽可能保持原意的准确性，同时对部分内容进行了语序和表达的优化，以更贴合中文学习者的阅读习惯。需要特别说明的是，原作者为该项目的主要贡献者，本汉化版本仅作为学习辅助资料，不对原内容进行修改或延伸。

由于个人能力有限，翻译中可能存在不完善之处，欢迎提出宝贵意见并多多包涵 希望通过这一翻译项目，更多中文学习者能够从中受益，也希望为国内社区的 LLM 学习和研究贡献一份力量。

> 本项目的特色： **jupyter代码**均有详细中文注释，帮助大家更快上手实践。 **诸多的附加材料**可以拓展知识

本项目所用徽章来自互联网，如侵犯了您的图片版权请联系我们删除，谢谢。

<h2 id="课程简介">
  <img src="./imgs/icon/intro.png" width="25" /> 课程简介
</h2>

谈到推理型大语言模型（Reasoning LLMs），人们往往把“推理”视作某种遥不可及的黑盒特性。但在《Reasoning-from-scratch》中，我们强调的是：推理依旧扎根于常规 LLM 技术，只是通过更精细的思路管理、链式思考、推理时扩展等方法，将模型的思考过程显性化、结构化。也就是说，推理型 LLM 并非脱离机器学习体系的“独立分支”，而是在深度学习基础上进一步增强逻辑推导、假设验证与反思能力的实战路径。

原项目与地址:[《reasoning-from-scratch》](https://github.com/rasbt/reasoning-from-scratch.git)  
[https://github.com/rasbt/reasoning-from-scratch.git](https://github.com/rasbt/reasoning-from-scratch.git)

[《reasoning-from-scratch》](https://github.com/rasbt/reasoning-from-scratch.git)项目致力于在传统 `从零实现 LLM` 的基础上，进一步探索 `推理`这一关键能力：课程依旧提供扎实的 Transformer/序列建模底层实现，但重点转向如何评估推理模型，让现有 LLM 获得链式思考、推理时扩展、强化学习微调等能力。每个环节都配有完整代码与讲解，帮助学习者亲手打造具备可解释推理能力的小型模型，并理解更大推理模型的核心思路。

<h2 id="课程资源">
  <img src="./imgs/icon/resource.png" width="25" /> 课程资源
</h2>

- 英文原版地址：[原版地址](https://github.com/rasbt/reasoning-from-scratch.git)
- 教材网址：[原版教材](https://mng.bz/lZ5B)
- 汉化地址：[汉化地址](https://github.com/MLNLP-World/reasoning-from-scratch-CN.git)

本项目提供配套代码、Jupyter 笔记本和 Markdown 文档，可在仓库中获取。

<h2 id="原书readme">
  <img src="./imgs/icon/notes.png" width="25" /> 原书Readme
</h2>

# 从零构建推理大模型

英文原项目收录了用于开发 LLM 推理模型的代码，也是图书 [*Build a Reasoning Model (From Scratch)*](https://mng.bz/lZ5B) 的官方代码库。本仓库提供其中文翻译。

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
- [Amazon.com 图书页面](https://amzn.to/4aAKiFY)
- ISBN 9781633434677

<br>
<br>

要下载原项目的英文仓库副本，请单击[Download ZIP](https://github.com/rasbt/reasoning-from-scratch/archive/refs/heads/main.zip)，或在终端中执行：

```bash
git clone --depth 1 https://github.com/rasbt/reasoning-from-scratch.git
cd reasoning-from-scratch
```

<br>

> **提示：**
> 第 2 章提供了有关安装 Python、管理 Python 软件包和设置编码环境的更多建议。

<br>
<br>

## 目录


[![Code tests Linux](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-linux.yml/badge.svg)](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-linux.yml)
[![Code tests macOS](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-macos.yml/badge.svg)](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-macos.yml)
[![Code tests Windows](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-windows.yml/badge.svg)](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-windows.yml)

- [故障排除指南](./troubleshooting.md)

| 章节名称 | 原项目主体代码 | 中文翻译代码 |
| --- | --- | --- |
| 第 1 章：理解推理模型 | 无代码 | 无代码 |
| 第 2 章：使用预训练 LLM 生成文本 | - [ch02_main.ipynb](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch02/01_main-chapter-code/ch02_main.ipynb)<br>- [ch02_exercise-solutions.ipynb](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch02/01_main-chapter-code/ch02_exercise-solutions.ipynb) | - [ch02_main.ipynb](ch02/01_main-chapter-code/ch02_main.ipynb)<br>- [ch02_exercise-solutions.ipynb](ch02/01_main-chapter-code/ch02_exercise-solutions.ipynb) |
| 第 3 章：评估推理模型 | - [ch03_main.ipynb](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch03/01_main-chapter-code/ch03_main.ipynb)<br>- [ch03_exercise-solutions.ipynb](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch03/01_main-chapter-code/ch03_exercise-solutions.ipynb) | - [ch03_main.ipynb](ch03/01_main-chapter-code/ch03_main.ipynb)<br>- [ch03_exercise-solutions.ipynb](ch03/01_main-chapter-code/ch03_exercise-solutions.ipynb) |
| 第 4 章：通过推理时扩展改进推理能力 | - [ch04_main.ipynb](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch04/01_main-chapter-code/ch04_main.ipynb)<br>- [ch04_exercise-solutions.ipynb](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch04/01_main-chapter-code/ch04_exercise-solutions.ipynb) | - [ch04_main.ipynb](ch04/01_main-chapter-code/ch04_main.ipynb)<br>- [ch04_exercise-solutions.ipynb](ch04/01_main-chapter-code/ch04_exercise-solutions.ipynb) |
| 第 5 章：通过自我改进实现推理时扩展 | - [ch05_main.ipynb](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch05/01_main-chapter-code/ch05_main.ipynb)<br>- [ch05_exercise-solutions.ipynb](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch05/01_main-chapter-code/ch05_exercise-solutions.ipynb) | - [ch05_main.ipynb](ch05/01_main-chapter-code/ch05_main.ipynb)<br>- [ch05_exercise-solutions.ipynb](ch05/01_main-chapter-code/ch05_exercise-solutions.ipynb) |
| 第 6 章：用强化学习训练推理模型 | - [ch06_main.ipynb](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch06/01_main-chapter-code/ch06_main.ipynb)<br>- [ch06_exercise-solutions.ipynb](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch06/01_main-chapter-code/ch06_exercise-solutions.ipynb) | - [ch06_main.ipynb](ch06/01_main-chapter-code/ch06_main.ipynb)<br>- [ch06_exercise-solutions.ipynb](ch06/01_main-chapter-code/ch06_exercise-solutions.ipynb) |
| 第 7 章：改进用于强化学习的 GRPO | - [ch07_main.ipynb](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch07/01_main-chapter-code/ch07_main.ipynb)<br>- [ch07_exercise-solutions.ipynb](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch07/01_main-chapter-code/ch07_exercise-solutions.ipynb) | - [ch07_main.ipynb](ch07/01_main-chapter-code/ch07_main.ipynb)<br>- [ch07_exercise-solutions.ipynb](ch07/01_main-chapter-code/ch07_exercise-solutions.ipynb) |
| 第 8 章：蒸馏推理模型以实现高效推理 | - [ch08_main.ipynb](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch08/01_main-chapter-code/ch08_main.ipynb)<br>- [ch08_exercise-solutions.ipynb](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch08/01_main-chapter-code/ch08_exercise-solutions.ipynb) | - [ch08_main.ipynb](ch08/01_main-chapter-code/ch08_main.ipynb)<br>- [ch08_exercise-solutions.ipynb](ch08/01_main-chapter-code/ch08_exercise-solutions.ipynb) |
| 附录 A：参考资料与延伸阅读 | 无代码 | 无代码 |
| 附录 B：练习解答 | 代码和解答位于[原项目各章子文件夹](https://github.com/rasbt/reasoning-from-scratch#table-of-contents)中 | 代码和解答位于各章子文件夹中 |
| 附录 C：Qwen3 LLM 源代码 | - [chC_main.ipynb](https://github.com/rasbt/reasoning-from-scratch/blob/main/chC/01_main-chapter-code/chC_main.ipynb) | - [chC_main.ipynb](chC/01_main-chapter-code/chC_main.ipynb) |
| 附录 D：使用更大的 LLM | - [chD_main.ipynb](https://github.com/rasbt/reasoning-from-scratch/blob/main/chD/chD_main.ipynb) | - [chD_main.ipynb](chD/chD_main.ipynb) |
| 附录 E：批处理与面向吞吐量的执行 | - [chE_main.ipynb](https://github.com/rasbt/reasoning-from-scratch/blob/main/chE/chE_main.ipynb) | - [chE_main.ipynb](chE/chE_main.ipynb) |
| 附录 F：LLM 评估的常用方法 | - [chF_main.ipynb](https://github.com/rasbt/reasoning-from-scratch/blob/main/chF/01_main-chapter-code/chF_main.ipynb)<br>- [chF](https://github.com/rasbt/reasoning-from-scratch/tree/main/chF) | - [chF_main.ipynb](chF/01_main-chapter-code/chF_main.ipynb)<br>- [chF](chF) |
| 附录 G：构建聊天界面 | - [chG](https://github.com/rasbt/reasoning-from-scratch/tree/main/chG) | - [chG](chG) |

<br>
&nbsp;

下图总结了本书介绍的主要技术。

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/mental-model.webp" width="650px">

<br>

&nbsp;

## 配套图书

请注意，*Build A Reasoning Model (From Scratch)* 是一本独立图书，专注于改进 LLM 推理的方法。

本书使用预训练的开源基础 LLM（Qwen3），并在其上从零编写代码来应用推理方法，包括推理时扩展、强化学习和蒸馏。

不过，如果你想了解传统基础 LLM 是如何实现的，可以参考作者的上一本书 [*Build a Large Language Model (From Scratch)*](https://amzn.to/4fqvn0D)。

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

## 额外材料

多个文件夹为感兴趣的读者提供了可选附加材料：

| 章节 | 原项目资料 | 中文翻译资料 |
| --- | --- | --- |
| **第 2 章：使用预训练 LLM 生成文本** | - [Optional Python Setup and Cloud GPU Recommendations](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch02/02_setup-tips)<br>- [Using a GPU-optimized version of the LLM](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch02/03_optimized-LLM)<br>- [Using `torch.compile()` on Windows](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch02/04_torch-compile-windows)<br>- [Run inference and chat with the model](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch02/05_use_model) | - [可选 Python 设置和云 GPU 建议](ch02/02_setup-tips)<br>- [使用针对 GPU 优化的 LLM 版本](ch02/03_optimized-LLM)<br>- [在 Windows 上使用 torch.compile()](ch02/04_torch-compile-windows)<br>- [运行推理并与模型聊天](ch02/05_use_model) |
| **第 3 章：评估 LLM** | - [MATH-500 Verifier Scripts](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch03/02_math500-verifier-scripts)<br>- [Advanced Parser](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch03/03_advanced-parser) (hybrid LaTeX parser) | - [MATH-500 验证器脚本](ch03/02_math500-verifier-scripts)<br>- [高级解析器](ch03/03_advanced-parser)（混合 LaTeX 解析器） |
| **第 4 章：通过推理时扩展改进推理能力** | - [Inference Scaling on MATH-500](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch04/02_math500-inference-scaling-scripts) (CoT prompting, self-consistency) | - [在 MATH-500 上进行推理扩展](ch04/02_math500-inference-scaling-scripts)（思维链提示、自洽方法） |
| **第 5 章：通过自我改进实现推理时扩展** | - [More Inference Scaling on MATH-500](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch05/02_math500-more-inference-scaling-scripts) (Best-of-N, self-refinement) | - [更多 MATH-500 推理扩展方法](ch05/02_math500-more-inference-scaling-scripts)（Best-of-N、自我改进） |
| **第 6 章：用强化学习训练推理模型** | - [GRPO scripts](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch06/02_rlvr_grpo_scripts_intro) with a batched mode | - [GRPO 脚本](ch06/02_rlvr_grpo_scripts_intro)，包含批处理模式 |
| **第 7 章：改进用于强化学习的 GRPO** | - [Advanced GRPO scripts](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch07/03_rlvr_grpo_scripts_advanced) (including DeepSeek-V3.2-, Olmo3-, and GDPO-style training)<br>- [Download training checkpoints](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch07/04_download_trainining_checkpoints) (how to download and use the chapter 6 and 7 GRPO checkpoints) | - [高级 GRPO 脚本](ch07/03_rlvr_grpo_scripts_advanced)（包括 DeepSeek-V3.2、Olmo3 和 GDPO 风格训练）<br>- [下载训练检查点](ch07/04_download_trainining_checkpoints)（如何下载并使用第 6、7 章的 GRPO 检查点） |
| **第 8 章：蒸馏推理模型以实现高效推理** | - [Generate distillation data](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch08/02_generate_distillation_data) (teacher-output generation via Ollama or OpenRouter)<br>- [Train with distillation](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch08/04_train_with_distillation) (including single-example and batched distillation scripts)<br>- [Download training checkpoints](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch08/05_download_training_checkpoints) (how to download and use the chapter 8 distillation checkpoints)<br>- [Use Qwen3 with Hugging Face](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch08/06_use_via_huggingface) (how to use the base model and chapter 6-8 checkpoints with `transformers`) | - [生成蒸馏数据](ch08/02_generate_distillation_data)（通过 Ollama 或 OpenRouter 生成教师模型输出）<br>- [使用蒸馏进行训练](ch08/04_train_with_distillation)（包括单样本和批处理蒸馏脚本）<br>- [下载训练检查点](ch08/05_download_training_checkpoints)（如何下载并使用第 8 章蒸馏检查点）<br>- [通过 Hugging Face 使用 Qwen3](ch08/06_use_via_huggingface)（如何通过 transformers 使用基础模型以及第 6–8 章检查点） |
| **附录 F：LLM 评估的常用方法** | - [MMLU Evaluation Methods](https://github.com/rasbt/reasoning-from-scratch/tree/main/chF/02_mmlu)<br>- [LLM leaderboards](https://github.com/rasbt/reasoning-from-scratch/tree/main/chF/03_leaderboards)<br>- [LLM-as-a-judge](https://github.com/rasbt/reasoning-from-scratch/tree/main/chF/04_llm-judge) | - [MMLU 评估方法](chF/02_mmlu)<br>- [LLM 排行榜](chF/03_leaderboards)<br>- [LLM 作为评判者](chF/04_llm-judge) |
| **附录 G：构建聊天界面** | - [Chat interface code](https://github.com/rasbt/reasoning-from-scratch/tree/main/chG/01_main-chapter-code) | - [聊天界面代码](chG/01_main-chapter-code) |

&nbsp;

## 问题、反馈与参与贡献

常见问题请参阅[故障排除指南](./troubleshooting.md)。

欢迎各种形式的反馈，最好通过 [Manning 讨论论坛](https://livebook.manning.com/forum?product=raschka2&page=1)或 [GitHub Discussions](https://github.com/rasbt/reasoning-from-scratch/discussions) 分享。如果你有任何问题，或只是想与他人交流想法，也欢迎在论坛中发帖。

请注意，作者的英文仓库包含与印刷书籍相对应的代码，因此目前无法接受扩展主要章节代码内容的贡献，因为这可能会导致与实体书籍的内容不一致。保持一致性有助于确保每个人的顺畅体验。

&nbsp;

## 引用

如果本书或代码对你的研究有帮助，请考虑引用。

Chicago 格式引用：

> Raschka, Sebastian. *Build A Reasoning Model (From Scratch)*. Manning, 2025. ISBN: 9781633434677.

BibTeX 条目：

```bibtex
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

<h2 id="组织者列表">
  <img src="./imgs/icon/organizer.png" width="25" />组织者列表
</h2>

感谢以下同学对本项目进行组织与指导

<p>
  <a href="https://github.com/airyuanshen" title="airyuanshen">
    <img src="./imgs/profile/airyuanshen.png" width="80" alt="airyuanshen" />
  </a>
  <a href="https://github.com/qiuqionglin" title="qiuqionglin">
    <img src="./imgs/profile/qiuqionglin.png" width="80" alt="qiuqionglin" />
  </a>
  <a href="https://github.com/allen-shaw" title="allen-shaw">
    <img src="./imgs/profile/allen-shaw.png" width="80" alt="allen-shaw" />
  </a>
  <a href="https://github.com/qinlibo-hit" title="qinlibo-hit">
    <img src="./imgs/profile/qinlibo-hit.png" width="80" alt="qinlibo-hit" />
  </a>
</p>

<h2 id="贡献者列表">
  <img src="./imgs/icon/heart.png" width="25" />贡献者列表
</h2>

感谢以下同学对本项目的支持与贡献

<p>
  <a href="https://github.com/airyuanshen" title="airyuanshen">
    <img src="./imgs/profile/airyuanshen.png" width="80" alt="airyuanshen" />
  </a>
  <a href="https://github.com/qiuqionglin" title="qiuqionglin">
    <img src="./imgs/profile/qiuqionglin.png" width="80" alt="qiuqionglin" />
  </a>
  <a href="https://github.com/allen-shaw" title="allen-shaw">
    <img src="./imgs/profile/allen-shaw.png" width="80" alt="allen-shaw" />
  </a>
  <a href="https://github.com/qinlibo-hit" title="qinlibo-hit">
    <img src="./imgs/profile/qinlibo-hit.png" width="80" alt="qinlibo-hit" />
  </a>
</p>

感谢原作者 Sebastian Raschka 与[英文上游贡献者](https://github.com/rasbt/reasoning-from-scratch/graphs/contributors)。
