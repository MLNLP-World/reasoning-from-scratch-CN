# 下载和使用训练检查点

本文件夹介绍如何从 Hugging Face 的 [https://huggingface.co/rasbt/qwen3-from-scratch-grpo-checkpoints](https://huggingface.co/rasbt/qwen3-from-scratch-grpo-checkpoints) 下载并使用第 7 章训练检查点。

这些检查点是供 `reasoning_from_scratch` 包使用的普通 PyTorch `state_dict` 文件，并不是 Hugging Face Transformers 检查点。

---

**注意**：如果不使用 `uv`，请将下面示例中的 `uv run ...py` 替换为 `python ...py`。

---

&nbsp;
## 可用的检查点文件夹

- `7_3_plus_tracking`：带有额外指标跟踪的 GRPO 检查点
- `7_4_plus_clip_ratio`：使用裁剪策略比率的 GRPO 检查点
- `7_5_plus_kl`：包含 KL 项的 GRPO 检查点
- `7_6_plus_format_reward`：为 `<think>` 标签加入显式格式奖励的 GRPO 检查点

检查点托管于：

- [rasbt/qwen3-from-scratch-grpo-checkpoints](https://huggingface.co/rasbt/qwen3-from-scratch-grpo-checkpoints)

&nbsp;
## 下载检查点

使用以下模块中的 `download_qwen3_grpo_checkpoints(...)`： [`reasoning_from_scratch.qwen3`](https://github.com/rasbt/reasoning-from-scratch/blob/main/reasoning_from_scratch/qwen3.py):

```python
from reasoning_from_scratch.qwen3 import download_qwen3_grpo_checkpoints

checkpoint_path = download_qwen3_grpo_checkpoints(
    grpo_type="clip_ratio",
    step="00050",
    out_dir="qwen3",
)
```

&nbsp;
## 应使用哪个分词器

以下检查点使用基础分词器：

- `7_3_plus_tracking`
- `7_4_plus_clip_ratio`
- `7_5_plus_kl`

以下检查点使用推理分词器：

- `7_6_plus_format_reward`

原因是 `7_6_plus_format_reward` 从推理模型开始训练，并且要求推理模型的对话格式。

&nbsp;
## 使用示例

下面的示例会下载检查点及匹配的分词器、加载模型，并使用第 2 章的 `generate_text_basic_stream_cache` 生成文本：

```python
from pathlib import Path
import torch

from reasoning_from_scratch.ch02 import (
    get_device,
    generate_text_basic_stream_cache,
)
from reasoning_from_scratch.ch03 import render_prompt
from reasoning_from_scratch.qwen3 import (
    download_qwen3_grpo_checkpoints,
    download_qwen3_small,
    Qwen3Model,
    Qwen3Tokenizer,
    QWEN_CONFIG_06_B,
)

device = get_device()
local_dir = Path("qwen3")

checkpoint_path = download_qwen3_grpo_checkpoints(
    grpo_type="clip_ratio",
    step="00050",
    out_dir=local_dir,
)
download_qwen3_small(kind="base", tokenizer_only=True, out_dir=local_dir)

tokenizer = Qwen3Tokenizer(tokenizer_file_path=local_dir / "tokenizer-base.json")
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
## 格式奖励示例

对于 `7_6_plus_format_reward`，请切换到推理分词器：

```python
from pathlib import Path

from reasoning_from_scratch.qwen3 import (
    download_qwen3_small,
    Qwen3Tokenizer,
)

local_dir = Path("qwen3")
download_qwen3_small(kind="reasoning", tokenizer_only=True, out_dir=local_dir)

tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=local_dir / "tokenizer-reasoning.json",
    apply_chat_template=True,
    add_generation_prompt=True,
    add_thinking=True,
)
```

&nbsp;
## 第 6 章示例

同一辅助函数也支持第 6 章原始的不含 KL 检查点：

```python
from reasoning_from_scratch.qwen3 import download_qwen3_grpo_checkpoints

download_qwen3_grpo_checkpoints(grpo_type="no_kl", step="00050", out_dir="qwen3")
```

&nbsp;
## 可用检查点

章节映射：

- `no_kl`：第 6 章原始无 KL GRPO 设置的基线
- `tracking`：主体章节的第 7.3 节
- `clip_ratio`：主体章节的第 7.4 节
- `kl`：主体章节的第 7.5 节
- `format_reward`：主体章节的第 7.6 节

可用的已保存步数：

- `no_kl`: `00050`, `00100`, `00500`, `01000`, `01500`, `03000`, `05000`, `09000`
- `tracking`: `00050`, `00100`, `00150`, `00200`, `00250`, `00300`, `00350`, `00400`, `00450`, `00500`
- `clip_ratio`: `00050`, `00100`, `00150`, `00200`, `00250`, `00300`, `00350`, `00400`, `00450`, `00500`
- `kl`: `00050`, `00100`, `00150`, `00200`, `00250`, `00300`, `00350`, `00400`, `00450`, `00500`
- `format_reward`: `00050`, `00100`, `00150`, `00200`, `00250`, `00300`, `00350`, `00400`, `00450`, `00500`
