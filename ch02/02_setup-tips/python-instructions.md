# Python环境配置建议

本书代码基本是自包含的，并尽量减少外部依赖。但为了让内容易读、篇幅控制在2000页以内，仍需要安装少量Python包。

本节介绍两种入门友好的方式，帮助你安装运行示例所需的依赖。

当然还有许多其他Python包管理方式；如果你已经有成熟的环境或偏好，可直接跳过本节。

若下面两个选项都不适用，欢迎在 [Discussion](https://github.com/rasbt/reasoning-from-scratch/discussions) 中求助。

&nbsp;
## 方案1：使用`pip`（内置，通用）

如果你已经安装了较新的Python版本，可以直接用内置的`pip`来安装包。

本书使用 Python 3.12。不过，只要 PyTorch 支持，较新的 Python 3.13、3.14 以及较旧的 3.11、3.10 也能正常工作。可以通过以下命令检查 Python 版本：

```bash
python --version
```

运行[附录 G 聊天界面](../../chG/01_main-chapter-code)时请使用 Python 3.13，因为 Chainlit 目前不支持 Python 3.14。

如果你还在用Python 3.9或更低版本，建议从 [python.org](https://www.python.org/downloads/) 安装最新版，或用 [`pyenv`](https://github.com/pyenv/pyenv) 管理多版本。安装新版本时请确认PyTorch是否支持，可参考 [PyTorch官网](https://pytorch.org/get-started/locally/) 的建议。PyTorch通常会落后Python新版本几个月，因此最新Python版本不一定立即受支持。

需要安装新包（如PyTorch、Jupyter Lab）时：

```bash
pip install torch jupyterlab
```

也可以通过 [`requirements.txt`](https://github.com/rasbt/reasoning-from-scratch/blob/main/requirements.txt) 一次性安装本书所需全部依赖：

```bash
pip install -r https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/refs/heads/main/requirements.txt
```


&nbsp;
## 方案2：使用`uv`（更快且广受推荐）

`pip` 仍是官方且经典的安装方式，但 [`uv`](https://github.com/astral-sh/uv) 是现代化的Python包管理器，具备以下优势：

- 自动创建并管理虚拟环境
- 安装速度快
- 维护锁定文件，便于复现
- 支持`pip`风格的命令

&nbsp;
### 安装`uv`及Python包

安装`uv`可以使用以下命令（最新建议请参考官方 [Installation](https://docs.astral.sh/uv/getting-started/installation/) 页面）。

&nbsp;
**macOS / Linux：**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

&nbsp;
**Windows（PowerShell）：**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

安装完成后，你可以像上一节那样安装包，只需把`pip`换成`uv pip`，例如：

```bash
uv pip install torch jupyterlab
```

不过既然使用`uv`，我更推荐用其原生语法，而不是 `uv pip`，详见下文。

&nbsp;
### 推荐的`uv`工作流

与其使用`uv pip`，我更建议采用原生的`uv`操作。

首先，将GitHub仓库克隆到本地：

```bash
git clone https://github.com/rasbt/reasoning-from-scratch.git
```

然后进入该目录（以Linux或macOS为例）：

```bash
cd reasoning-from-scratch
```

`pyproject.toml` 文件列出了所需软件包。首次运行脚本或打开 Jupyter Lab 时，`uv` 会自动为 `reasoning-from-scratch` 项目创建一个默认隐藏的虚拟环境文件夹（`.venv`），并在其中安装全部依赖。

仓库未包含 `.python-version` 文件。若要为本地 `uv` 环境指定 Python 3.13，可运行以下命令：

```bash
uv python pin 3.13
uv sync
```

通常不需要额外安装软件包；但一般来说，可以通过 `uv add` 安装 `pyproject.toml` 中尚未列出的额外包：

```bash
uv add llms_from_scratch
```

该命令会把新包添加到虚拟环境和 `pyproject.toml` 中。

&nbsp;
### 使用`uv`运行代码

下面是通过`uv`启动Jupyter Lab和运行脚本的命令：

打开Jupyter Lab：

```bash
uv run jupyter lab
```

运行Python脚本：

```bash
uv run python script.py
```

仓库随附的 `uv.lock` 使用本书原始代码的 PyTorch 2.10.0。因此，常规 `uv run` 默认采用该锁定版本；`uv` 会为支持的操作系统选择对应的软件包。

`pyproject.toml` 允许的其他 PyTorch 版本可以通过更新本地锁文件使用：

```bash
uv lock --upgrade-package torch
```

更新一次后，后续 `uv run` 会使用新的锁定版本。该命令会修改本地 `uv.lock`。若要恢复仓库提交的版本，可运行：

```bash
git restore uv.lock
uv sync
```

> **进阶用法：** 以上只是给`pip`用户参考的简化流程。若想深入了解如何用`uv`管理虚拟环境，可阅读[这份文档](https://github.com/rasbt/LLMs-from-scratch/tree/main/setup/01_optional-python-setup-preferences)。  
> 如果你使用macOS或Linux并偏好`uv`原生命令，请参考[该教程](https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/01_optional-python-setup-preferences/native-uv.md)。同时建议查阅[官方uv文档](https://docs.astral.sh/uv/)以获取更多信息。



&nbsp;
### JupyterLab 使用提示

如果在 JupyterLab 而不是 VSCode 中查看 notebook 代码，请注意，JupyterLab 的默认设置在近期版本中出现过滚动问题。建议打开 Settings -> Settings Editor，将“Windowing mode”改为“none”（如下图所示），这似乎可以解决该问题。

![Jupyter 问题 1](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_1.webp)

<br>

![Jupyter 问题 2](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_2.webp)

&nbsp;
## 有问题？

如有疑问，欢迎在本仓库的 [Discussions](https://github.com/rasbt/reasoning-from-scratch/discussions) 论坛反馈。
