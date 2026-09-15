# 测试

此目录包含仓库的 Python 测试套件。

## 本地运行

首先安装开发环境：

```bash
uv sync --group dev
```

### 1. 常规测试套件，忽略高成本测试（推荐）

建议使用此方式快速测试和开发新功能。

```bash
SKIP_EXPENSIVE=1 RUN_REAL_DOWNLOAD_TESTS=0 uv run pytest tests
```

运行单个测试文件：

```bash
SKIP_EXPENSIVE=1 RUN_REAL_DOWNLOAD_TESTS=0 uv run pytest tests/test_ch03.py
```

这是最接近默认 GitHub 测试矩阵的本地运行方式。

### 2. 常规测试套件加高成本测试

一些代码测试默认被忽略，因为运行成本较高。完成基础调试后，建议运行这些测试。

```bash
SKIP_EXPENSIVE=0 RUN_REAL_DOWNLOAD_TESTS=0 uv run pytest tests
```

请注意，这会运行测试文件中受 `SKIP_EXPENSIVE` 控制的测试，但仍会排除下载大型模型检查点的真实网络/下载集成测试。

### 3. 仅运行下载测试

一些测试用于检查模型检查点文件能否下载，以及服务器是否仍正常工作。无需经常在本地运行，它们主要用于偶尔检查。

使用以下命令运行下载测试：

```bash
SKIP_EXPENSIVE=0 RUN_REAL_DOWNLOAD_TESTS=1 uv run pytest tests -k real_download
```

工作方式：

- `pytest tests` 收集 `tests/` 目录中的测试
- `-k real_download` 只保留名称中包含 `real_download` 的测试

更有针对性的示例：要直接运行附录 D 的真实快照测试，请使用：

```bash
SKIP_EXPENSIVE=0 RUN_REAL_DOWNLOAD_TESTS=1 uv run pytest tests/test_appendix_d.py -k real_download_1_7b
```

当前需显式启用的真实下载测试包括：

- `tests/test_ch03.py`：真实下载 `math500_test.json` 和分词器
- `tests/test_ch06.py`：真实下载数学训练集
- `tests/test_ch07.py`：真实下载 GitHub 原始文件
- `tests/test_ch08.py`：真实下载蒸馏数据集和分词器
- `tests/test_appendix_d.py`：真实下载 `Qwen/Qwen3-1.7B-Base` 快照
- `tests/test_qwen3.py`：真实比较 `Qwen/Qwen3-0.6B` 分词器

### 4. 运行全部测试（不推荐）

这会运行测试套件中的所有内容，包括计算成本较高的测试和下载测试。

```bash
SKIP_EXPENSIVE=0 RUN_REAL_DOWNLOAD_TESTS=1 uv run pytest tests
```

不建议在常规代码修改期间使用，因为文件下载成本很高，也没有必要频繁运行。

## GitHub CI 中运行的内容

默认 GitHub 测试矩阵运行常规测试套件，并省略较重的测试：

- `.github/workflows/tests-linux.yml`
- `.github/workflows/tests-macos.yml`
- `.github/workflows/tests-windows.yml`
- `.github/workflows/basic-tests-pip.yml`

这些工作流设置 `SKIP_EXPENSIVE=1`，因此会跳过高成本测试。原因是 GitHub CI 没有运行这些测试所需的计算资源（例如 GPU）。

真实网络/下载集成测试在单独的工作流中运行：

- `.github/workflows/real-download-tests.yml`

该工作流设置 `RUN_REAL_DOWNLOAD_TESTS=1`，并只运行通过 `-k real_download` 选出的测试。它不属于默认 PR/push 矩阵，而是每周定时运行，也可以通过 `workflow_dispatch` 手动启动。
