# 第 8 章附加材料：通过本地 Hugging Face 包装器使用 Qwen3

此文件夹展示如何用轻量的本地 `PreTrainedModel` 类包装从零实现的 [`Qwen3Model`](../../../reasoning_from_scratch/qwen3.py)，使其与 Hugging Face `transformers` 库兼容。

这样可以使用：

- `model.generate(...)`
- `transformers.Trainer`

并直接配合本仓库中的本地 `.pth` 模型文件，包括 Qwen3 基础权重以及第 6–8 章的兼容检查点。

&nbsp;
## 文件

- [hf_wrapper.py](hf_wrapper.py)：本书所用从零实现 `Qwen3Model` 的本地 `PreTrainedModel` 包装器
- [hf_inference.py](hf_inference.py)：使用包装器和仓库分词器生成文本
- [hf_trainer.py](hf_trainer.py)：使用包装器和第 8 章蒸馏 JSON 格式的 `Trainer` 示例

---

**注意**：如果不使用 `uv`，请在以下示例中将 `uv run ...py` 替换为 `python ...py`。

---

&nbsp;
## 此包装器的作用

该包装器将模型保留在本仓库中，并使其适配 Hugging Face API。

具体来说，它会：

- 将本地 `.pth` 模型文件直接加载到 `Qwen3Model`
- 用 `PreTrainedModel` 接口包装该模型
- 提供与 `Trainer` 兼容的 `forward(...)` 方法
- 启用 `model.generate(...)`
- 继续使用仓库中的 `Qwen3Tokenizer`

之所以提供此方法，是因为一些读者希望在功能比本仓库从零实现代码更丰富的 `transformers` 中进一步探索这些模型。

&nbsp;
## 局限性

这是一个围绕从零实现模型的小型本地包装器。

需要注意：

- 它适用于已安装 `reasoning_from_scratch` 的环境
- 它不提供 `AutoTokenizer.from_pretrained(...)` 工作流
- 它不会创建包含 `config.json` 和分词器文件的可复用模型目录
- 为保持实现简单，生成时会重新计算完整前缀，而不把从零实现的 KV 缓存适配到 Hugging Face 缓存类；如果需要完整支持，请改用 [../export_approach](../export_approach)

这些限制使代码保持简短，并专注于本仓库内的本地使用场景。

&nbsp;
## 第 1 步：安装依赖

除仓库依赖外，本指南还使用 Hugging Face Transformers。

```bash
pip install transformers accelerate
```

如果使用 `uv`，则运行：

```bash
uv add --dev transformers accelerate
```

&nbsp;
## 第 2 步：运行本地包装推理

要通过包装器运行基础模型，请使用：

```bash
  uv run hf_inference.py \
    --tokenizer_kind base \
    --prompt "If x + 7 = 19, what is x?"
```

要运行推理变体，请使用：

```bash
  uv run hf_inference.py \
    --tokenizer_kind reasoning \
    --prompt "If x + 7 = 19, what is x?"
```

要改为运行本地检查点，请使用：

```bash
uv run hf_inference.py \
  --tokenizer_kind reasoning \
  --model_path ../../04_train_with_distillation/checkpoints/distill/qwen3-0.6B-distill-step00004-epoch1.pth \
  --prompt "If x + 7 = 19, what is x?"
```

如果省略 `--model_path`，脚本会根据所选 `--tokenizer_kind` 下载默认基础模型或推理模型。如果提供 `--model_path`，它可以指向 Qwen3 基础 `.pth` 文件，或第 6–8 章生成的任意兼容检查点。

推理脚本内部会：

1. 构建本地包装模型
2. 将所选 `.pth` 模型文件加载到包装后的 `Qwen3Model`
3. 使用仓库分词器对提示词进行分词
4. 调用 `model.generate(...)`

&nbsp;
## 第 3 步：使用 `Trainer` 继续训练

同一个包装器也可以配合 `transformers.Trainer` 使用：

```bash
uv run hf_trainer.py \
  --tokenizer_kind reasoning \
  --model_path ../../04_train_with_distillation/checkpoints/distill/qwen3-0.6B-distill-step00004-epoch1.pth \
  --data_path ../../02_generate_distillation_data/sample_openrouter_outputs.json \
  --dataset_size 5 \
  --validation_size 1 \
  --epochs 1 \
  --logging_steps 1
```

与推理相同，`--model_path` 可以指向 Qwen3 基础权重或第 6–8 章的兼容检查点。

该训练器沿用第 8 章其他部分只对答案计算损失的训练目标：

- 屏蔽提示词词元
- 只有答案词元参与损失计算
- 推理模式用 `<think>...</think>` 包裹教师推理轨迹

输入 JSON 格式与 [../../02_generate_distillation_data](../../02_generate_distillation_data) 中生成的蒸馏数据一致。

&nbsp;
