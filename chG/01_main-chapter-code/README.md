# 附录 G：聊天界面

本文件夹包含用来运行类似 ChatGPT 的用户界面，以便与本书中使用或开发的 LLM 交互，如下图所示。

![Chainlit 界面示例](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen3-chainlit.gif)

为实现该界面，我们使用开源的 [Chainlit Python 包](https://github.com/Chainlit/chainlit)。

&nbsp;
## 步骤 1：安装依赖

首先安装 `chainlit` 包及其依赖：

```bash
pip install chainlit
```

如果使用 `uv`，则运行：

```bash
uv add chainlit
```

&nbsp;

## 步骤 2：运行 `app` 代码

此文件夹包含 2 个文件：

1. [`qwen3_chat_interface.py`](qwen3_chat_interface.py)：加载并以 thinking 模式使用 Qwen3 0.6B 模型。
2. [`qwen3_chat_interface_multiturn.py`](qwen3_chat_interface_multiturn.py)：与上面相同，但会记住消息历史。

（可打开并查看这些文件以了解详情。）

在终端中运行下列命令之一以启动 UI 服务器：

```bash
chainlit run qwen3_chat_interface.py
```

如果使用 `uv`，则运行：

```bash
uv run chainlit run qwen3_chat_interface.py
```

执行上述任一命令后，应会自动打开一个浏览器标签页，供你与模型交互。如果未自动打开，请查看终端输出并将其中的本地地址复制到浏览器地址栏（通常为 `http://localhost:8000`）。

