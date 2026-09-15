# 第 8 章附加材料：通过 Hugging Face 使用 Qwen3

此文件夹介绍两种通过 Hugging Face `transformers` 使用本仓库从零实现的 [`Qwen3Model`](../../reasoning_from_scratch/qwen3.py) 及兼容 `.pth` 检查点的方法。

两种方法都支持 Hugging Face 风格的推理和训练；区别在于需要可复用的 Hugging Face 模型目录，还是围绕现有 PyTorch 模型的轻量本地包装器。

&nbsp;
## 使用方式


&nbsp;
### 1) `wrapper_approach`

[./wrapper_approach](./wrapper_approach) 将模型保留为本地 `.pth` 文件，并用轻量的本地 `PreTrainedModel` 包装 `Qwen3Model`，使其能够使用 Hugging Face API 的部分功能。

如果有以下需求，请使用此方法：

- 尽可能少的额外代码
- 在本仓库内进行本地实验
- 无需导出即可使用 `model.generate(...)` 和 `transformers.Trainer`
- 直接从 `.pth` 加载基础模型或第 6–8 章的检查点


&nbsp;
### 2) `export_approach`

[./export_approach](./export_approach) 将从零实现的 Qwen3 权重或兼容检查点转换为与 Hugging Face 兼容的模型文件夹。

如果有以下需求，请使用此方法：

- 包含 `config.json`、分词器文件和权重的已保存模型目录
- 使用 `AutoConfig`、`AutoTokenizer` 和 `AutoModelForCausalLM`
- 更接近 Hugging Face 模型常规打包方式的工作流



&nbsp;
## 应选择哪一种？

- 如果目的是学习，或希望与 `transformers` 进行更轻量的本地集成，请选择 [wrapper_approach](wrapper_approach)。
- 如果目标是创建 Hugging Face 模型包并优化计算性能，请选择 [export_approach](export_approach)。