# 下载和使用训练检查点

此文件夹介绍如何从 Hugging Face 模型中心 [https://huggingface.co/rasbt/qwen3-from-scratch-distill-checkpoints](https://huggingface.co/rasbt/qwen3-from-scratch-distill-checkpoints) 下载并使用第 8 章的蒸馏检查点。

这些检查点是供 `reasoning_from_scratch` 包使用的普通 PyTorch `state_dict` 文件，并非 Hugging Face Transformers 检查点。

---

**注意**：如果不使用 `uv`，请在以下示例中将 `uv run ...py` 替换为 `python ...py`。

---

&nbsp;
## 可用的检查点文件夹

- `ch08_distill_deepseek_r1`：[`ch08_main.ipynb`](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch08/01_main-chapter-code/ch08_main.ipynb) 表格第 3–5 行使用的 3 个 DeepSeek-R1 蒸馏检查点
- `ch08_distill_qwen3_235b_a22b`：[`ch08_main.ipynb`](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch08/01_main-chapter-code/ch08_main.ipynb) 表格第 6–8 行使用的 3 个 Qwen3 235B A22B 蒸馏检查点

检查点托管在：

- [rasbt/qwen3-from-scratch-distill-checkpoints](https://huggingface.co/rasbt/qwen3-from-scratch-distill-checkpoints)

&nbsp;
## 下载检查点

使用 [`reasoning_from_scratch.qwen3`](https://github.com/rasbt/reasoning-from-scratch/blob/main/reasoning_from_scratch/qwen3.py) 中的 `download_qwen3_distill_checkpoints(...)`：

```python
from reasoning_from_scratch.qwen3 import download_qwen3_distill_checkpoints

checkpoint_path = download_qwen3_distill_checkpoints(
    distill_type="deepseek_r1",
    step="06682",
    out_dir="qwen3",
)
```

&nbsp;
## 应使用哪种分词器

以下检查点使用推理分词器：

- `ch08_distill_deepseek_r1`
- `ch08_distill_qwen3_235b_a22b`

&nbsp;

## 使用示例

以下示例会下载检查点及匹配的分词器，加载模型，并使用第 2 章的 `generate_text_basic_stream_cache` 生成文本：

```python
from pathlib import Path
import torch

from reasoning_from_scratch.ch02 import (
    get_device,
    generate_text_basic_stream_cache,
)
from reasoning_from_scratch.ch03 import render_prompt
from reasoning_from_scratch.qwen3 import (
    download_qwen3_distill_checkpoints,
    download_qwen3_small,
    Qwen3Model,
    Qwen3Tokenizer,
    QWEN_CONFIG_06_B,
)

device = get_device()
local_dir = Path("qwen3")

checkpoint_path = download_qwen3_distill_checkpoints(
    distill_type="deepseek_r1",
    step="06682",
    out_dir=local_dir,
)
download_qwen3_small(kind="reasoning", tokenizer_only=True, out_dir=local_dir)

tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=local_dir / "tokenizer-reasoning.json",
    apply_chat_template=True,
    add_generation_prompt=True,
    add_thinking=True,
)
model = Qwen3Model(QWEN_CONFIG_06_B)
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

prompt = render_prompt("Solve: If x + 7 = 19, what is x?")
input_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)

for token in generate_text_basic_stream_cache(
    model=model,
    token_ids=input_ids,
    max_new_tokens=256,
    eos_token_id=tokenizer.eos_token_id,
):
    token_id = token.squeeze(0).item()
    print(tokenizer.decode([token_id]), end="", flush=True)
```

&nbsp;
## Qwen3 示例

对于 `ch08_distill_qwen3_235b_a22b`，使用同一个辅助函数并指定另一种 `distill_type`：

```python
from reasoning_from_scratch.qwen3 import download_qwen3_distill_checkpoints

download_qwen3_distill_checkpoints(
    distill_type="qwen3_235b_a22b",
    step="05746",
    out_dir="qwen3",
)
```

&nbsp;
## 可用步骤

`deepseek_r1` 可用的已保存步骤：

- `06682`
- `13364`
- `20046`

`qwen3_235b_a22b` 可用的已保存步骤：

- `05746`
- `11492`
- `17238`
