<p align="center">
<h1 align="center"> <img src="./imgs/icon/ai.png" width="30" />《从零构建推理模型》</h1>
</p>
<p align="center">
    <a href="https://img.shields.io/badge/version-v0.1.0-blue">
      <img alt="version" src="https://img.shields.io/badge/version-v0.1.0-blue?color=FF8000?color=009922" />
    </a>
  <a >
       <img alt="Status-building" src="https://img.shields.io/badge/Status-building-blue" />
    </a>
  <a >
       <img alt="PRs-Welcome" src="https://img.shields.io/badge/PRs-Welcome-red" />
    </a>
  <a href="https://github.com/MLNLP-World/reasoning-from-scratch-CN/stargazers">
  <img alt="Stars" src="https://img.shields.io/github/stars/MLNLP-World/reasoning-from-scratch-CN?style=social" />
</a>
<a href="https://github.com/MLNLP-World/reasoning-from-scratch-CN/network/members">
  <img alt="Forks" src="https://img.shields.io/github/forks/MLNLP-World/reasoning-from-scratch-CN?style=social" />
</a>
<a href="https://github.com/MLNLP-World/reasoning-from-scratch-CN/issues">
  <img alt="Issues" src="https://img.shields.io/github/issues/MLNLP-World/reasoning-from-scratch-CN?style=social" />
</a>
    <br />
</p>

<div align="center">
<p align="center">
  <a href="#项目动机">项目动机</a>/
  <a href="#课程简介">课程简介</a>/
  <a href="#课程资源">课程资源</a>/
  <a href="#原书Readme">原书Readme</a>/
  <a href="#贡献者">贡献者</a>
</p>
</div>

---

<h2 id="项目动机">
  <img src="./imgs/icon/motivation.png" width="25" /> 项目动机
</h2>

原项目与地址:[《reasoning-from-scratch》](https://github.com/rasbt/reasoning-from-scratch.git)

本项目是对GitHub项目[《reasoning-from-scratch》](https://github.com/rasbt/reasoning-from-scratch.git)内容的中文翻译，包括详细的markdown 笔记和相关的jupyter 代码。翻译过程中，我们尽可能保持原意的准确性，同时对部分内容进行了语序和表达的优化，以更贴合中文学习者的阅读习惯。需要特别说明的是，原作者为该项目的主要贡献者，本汉化版本仅作为学习辅助资料，不对原内容进行修改或延伸。

由于个人能力有限，翻译中可能存在不完善之处，欢迎提出宝贵意见并多多包涵 希望通过这一翻译项目，更多中文学习者能够从中受益，也希望为国内社区的 LLM 学习和研究贡献一份力量。

> 本项目的特色：
> **jupyter代码**均有详细中文注释，帮助大家更快上手实践。
> **诸多的附加材料**可以拓展知识

本项目所用徽章来自互联网，如侵犯了您的图片版权请联系我们删除，谢谢。

<h2 id="课程简介">
  <img src="./imgs/icon/intro.png" width="25" /> 课程简介
</h2>

谈到推理型大语言模型（Reasoning LLMs），人们往往把“推理”视作某种遥不可及的黑盒特性。但在《Reasoning-from-scratch》中，我们强调的是：推理依旧扎根于常规 LLM 技术，只是通过更精细的思路管理、链式思考、推理时扩展等方法，将模型的思考过程显性化、结构化。也就是说，推理型 LLM 并非脱离机器学习体系的“独立分支”，而是在深度学习基础上进一步增强逻辑推导、假设验证与反思能力的实战路径。*

原项目与地址:[《reasoning-from-scratch》](https://github.com/rasbt/reasoning-from-scratch.git)
[https://github.com/rasbt/reasoning-from-scratch.git](https://github.com/rasbt/reasoning-from-scratch.git)

在 [《reasoning-from-scratch》](https://github.com/rasbt/reasoning-from-scratch.git)项目致力于在传统 `从零实现 LLM` 的基础上，进一步探索 `推理`这一关键能力：课程依旧提供扎实的 Transformer/序列建模底层实现，但重点转向如何评估推理模型，让现有 LLM 获得链式思考、推理时扩展、强化学习微调等能力。每个环节都配有完整代码与讲解，帮助学习者亲手打造具备可解释推理能力的小型模型，并理解更大推理模型的核心思路。

<h2 id="课程资源">
  <img src="./imgs/icon/resource.png" width="25" /> 课程资源
</h2>

- 英文原版地址：[原版地址](https://github.com/rasbt/reasoning-from-scratch.git)
- 教材网址：[原版教材](https://mng.bz/lZ5B)
- 汉化地址：[汉化地址](https://github.com/MLNLP-World/reasoning-from-scratch-CN.git)

此外，本门课程还有相应的代码实现。**每章都有相应的jupyter记事本，提供模型的完整python代码**，所有的资源都可在网上免费获取。

<h2 id="原书Readme">
  <img src="./imgs/icon/notes.png" width="25" /> 原书Readme
</h2>

# 《从零构建推理模型》配套代码

本仓库收录了用于开发 LLM 推理模型的全部代码，也是图书 [*Build a Reasoning Model (From Scratch)*](https://mng.bz/lZ5B) 的官方代码库。

<br>
<br>

<a href="https://mng.bz/lZ5B"><img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/cover.webp?123" width="250px"></a>

（彩色印刷。）

<br>

在 [*Build a Reasoning Model (From Scratch)*](https://mng.bz/lZ5B) 一书中，你将系统学习并理解推理型大语言模型（LLM）的工作方式。

“推理”是近年来改进 LLM 的最重要、也最容易被误解的能力之一。如果只是在纸面上阅读相关理论，很难真正掌握；因此，本书采用动手实践的方式。我们会从一个预训练的基础 LLM 起步，然后亲自一步步在代码中加入推理能力，确保看清每一步发生了什么。

本书介绍的方法会引导你完成一个“麻雀虽小、但足够用”的推理模型，主要用于教学演示。整个流程与 DeepSeek R1、GPT-5 Thinking 等大规模推理模型的实现思路相呼应。此外，本书还包含加载现有预训练模型权重的示例代码。

- 官方 [源代码仓库](https://github.com/rasbt/reasoning-from-scratch)
- [Manning 官网图书链接](https://mng.bz/lZ5B)
- Amazon 图书页（待更新）
- ISBN 9781633434677

<br>
<br>

如果想下载代码仓库，可点击 [Download ZIP](https://github.com/rasbt/reasoning-from-scratch/archive/refs/heads/main.zip)，或在终端执行以下命令：

```bash
git clone --depth 1 https://github.com/rasbt/reasoning-from-scratch.git
```

<br>

> **提示：**
> 第 2 章提供了安装 Python、管理依赖与配置开发环境的更多建议。

<br>
<br>

## 目录（更新中）

[![Code tests Linux](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-linux.yml/badge.svg)](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-linux.yml)
[![Code tests macOS](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-macos.yml/badge.svg)](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-macos.yml)
[![Code tests Windows](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-windows.yml/badge.svg)](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-windows.yml)

| 章节标题                            | 主体代码                                                                                                                                                     |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 第 1 章：理解推理模型               | 无代码                                                                                                                                                       |
| 第 2 章：使用预训练 LLM 生成文本    | -[ch02_main.ipynb](ch02/01_main-chapter-code/ch02_main.ipynb)<br/>- [ch02_exercise-solutions.ipynb](ch02/01_main-chapter-code/ch02_exercise-solutions.ipynb) |
| 第 3 章：评估推理模型               | -[ch03_main.ipynb](ch03/01_main-chapter-code/ch03_main.ipynb)<br/>- [ch03_exercise-solutions.ipynb](ch03/01_main-chapter-code/ch03_exercise-solutions.ipynb) |
| 第 4 章：利用推理时扩展提升推理能力 | -[ch04_main.ipynb](ch04/01_main-chapter-code/ch04_main.ipynb)<br/>- [ch04_exercise-solutions.ipynb](ch04/01_main-chapter-code/ch04_exercise-solutions.ipynb) |
| 第 5 章：通过自我改进实现推理时扩展 | 待补充                                                                                                                                                       |
| 第 6 章：用强化学习训练推理模型     | 待补充                                                                                                                                                       |
| 第 7 章：蒸馏推理模型以实现高效推理 | 待补充                                                                                                                                                       |
| 第 8 章：完善推理流水线与未来方向   | 待补充                                                                                                                                                       |
| 附录 A：参考资料与延伸阅读          | 无代码                                                                                                                                                       |
| 附录 B：习题解答                    | 代码与解答分布在各章节子目录                                                                                                                                 |
| 附录 C：Qwen3 LLM 源码              | -[chC_main.ipynb](chC/01_main-chapter-code/chC_main.ipynb)                                                                                                   |
| 附录 D                              | 待补充                                                                                                                                                       |
| 附录 E                              | 待补充                                                                                                                                                       |
| 附录 F：LLM 评估的常见方法          | -[chF_main.ipynb](chF/01_main-chapter-code/chF_main.ipynb)                                                                                                   |

<br>

下图给出了本书主要技术路线的心智图总结。

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/mental-model.webp" width="650px">

<br>

&nbsp;

## 配套图书

请注意，《Build A Reasoning Model (From Scratch)》是一部专注于 LLM 推理方法的独立著作。

在书中，我们基于一个开源预训练基础模型（Qwen3），在其之上从头实现多种推理技术，包括推理时扩展、强化学习与蒸馏。

如果你还想了解常规基础 LLM 的内部实现，可以参考我前一本书 [*Build a Large Language Model (From Scratch)*](https://amzn.to/4fqvn0D)。

<a href="https://amzn.to/4fqvn0D"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover.jpg?123" width="120px"></a>

- [Amazon 链接](https://amzn.to/4fqvn0D)
- [Manning 链接](http://mng.bz/orYv)
- [GitHub 仓库](https://github.com/rasbt/LLMs-from-scratch)

<br>

## 硬件需求

本书主要章节的代码设计为可在消费级硬件上于合理时间内运行，无需专门的服务器配置，也能自动检测并利用可用的 GPU。第 2-4 章在 CPU 或 GPU 上都能顺畅执行；若要复现实验 5、6 章的结果，推荐使用 GPU。

（更多建议可参见 [setup_tips](ch02/https://github.com/rasbt/reasoning-from-scratch/blob/main/ch02/01_main-chapter-code/python-instructions.md) 文档。）

&nbsp;

## 习题

每章都会安排若干练习。解答概要集中在附录 B，对应的代码笔记位于各章主目录中（例如 [`ch02/01_main-chapter-code/ch02_exercise-solutions.ipynb`](ch02/01_main-chapter-code/ch02_exercise-solutions.ipynb)）。

&nbsp;

## 额外资料

下列目录提供了额外的可选内容，供感兴趣的读者拓展：

- **第 2 章：使用预训练 LLM 生成文本**
  - [可选的 Python 环境与云 GPU 建议](ch02/02_setup-tips)
  - [使用 GPU 优化版 LLM](ch02/03_optimized-LLM)
  - [在 Windows 上使用 `torch.compile()`](ch02/04_torch-compile-windows)
  - [运行推理并与模型对话](ch02/05_use_model)
- **第 3 章：评估 LLM**
  - [MATH-500 验证脚本](ch03/02_math500-verifier-scripts)
- **第 4 章：利用推理时扩展改进推理**
  - [MATH-500 推理扩展脚本](ch04/02_math500-inference-scaling-scripts)
- **附录 F：LLM 评估的常见方法**
  - [MMLU 评估方法](chF/02_mmlu)
  - [LLM 排行榜](chF/03_leaderboards)
  - [LLM-as-a-judge](chF/04_llm-judge)

&nbsp;

## 问题、反馈与贡献

欢迎通过 [Manning 论坛](https://livebook.manning.com/forum?product=raschka2&page=1) 或 [GitHub Discussions](https://github.com/rasbt/reasoning-from-scratch/discussions) 提交任何反馈、疑问或想法。

请注意：由于本仓库对应用纸质书的内容进行配套，因此目前无法接受会扩展章节主体内容的贡献，以免与实体书产生不一致。保持同步可让所有读者都获得一致体验。

&nbsp;

## 引用方式

如果这本书或代码对你的研究有所帮助，欢迎引用。

Chicago 引用格式：

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

<h2 id="贡献者">
  <img src="./imgs/icon/heart.png" width="25" /> 贡献者
</h2>

<a href="https://github.com/qiuqionglin" title="qiuqionglin">
  <img src="https://github.com/qiuqionglin.png?size=80" width="80" alt="qiuqionglin" style="border-radius:50%;" />
</a>

</a>
