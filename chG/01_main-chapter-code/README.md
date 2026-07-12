# 附录 G：构建聊天界面

此文件夹包含用于运行类似 ChatGPT 的用户界面代码，以便与本书使用或开发的 LLM 交互，如下图所示。

![Chainlit 界面示例](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen3-chainlit.gif)

该用户界面使用开源的 [Chainlit Python 包](https://github.com/Chainlit/chainlit)实现。

&nbsp;
## 第 1 步：安装依赖

首先安装 `chainlit` 包及其依赖：

```bash
pip install chainlit
```

如果使用 `uv`，则运行：

```bash
uv add chainlit
```

&nbsp;

## 第 2 步：运行 `app` 代码

此文件夹包含两个文件：

1. [`qwen3_chat_interface.py`](qwen3_chat_interface.py)：加载 Qwen3 0.6B 模型并以思考模式使用。
2. [`qwen3_chat_interface_multiturn.py`](qwen3_chat_interface_multiturn.py)：与上面相同，但会记住消息历史。

（可打开并查看这些文件以了解详情。）

在终端中运行以下命令之一来启动 UI 服务器：

```bash
chainlit run qwen3_chat_interface.py
```

如果使用 `uv`，则运行：

```bash
uv run chainlit run qwen3_chat_interface.py
```

执行上述任一命令后，应会打开新的浏览器标签页，以便与模型交互。如果浏览器标签页没有自动打开，请查看终端输出，并将本地地址复制到浏览器地址栏（通常为 `http://localhost:8000`）。

## 使用自定义检查点

由于 `chainlit run ...` 会接管命令行参数，这些脚本通过 `CHECKPOINT_PATH` 环境变量读取自定义检查点路径，而不使用 `argparse`。

终端示例：

```bash
CHECKPOINT_PATH=/absolute/path/to/qwen3-0.6B-distill-step06682-epoch1.pth \
uv run chainlit run qwen3_chat_interface.py
```

说明：

- 请使脚本中的 `WHICH_MODEL` 与检查点预期的分词器保持一致。
- [`ch08/05_download_training_checkpoints`](../../ch08/05_download_training_checkpoints) 中的第 8 章检查点使用推理分词器，因此请设置 `WHICH_MODEL = "reasoning"`。
- 设置 `CHECKPOINT_PATH` 后，脚本只会把分词器下载到 `LOCAL_DIR`，不会重新下载默认模型权重。
