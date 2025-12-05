# 在 Windows 上使用 `torch.compile()`

`torch.compile()` 依赖 *TorchInductor*，它会 JIT 编译算子，需要可用的 C/C++ 编译工具链。

因此，在 Windows 上要让 `torch.compile` 正常工作，整体准备会比 Linux 或 macOS 更麻烦一些（后者通常只需安装 PyTorch）。

如果你是 Windows 用户，并觉得配置 `torch.compile` 过于复杂也不用担心：本仓库的所有示例在不编译的情况下依然可以正常运行。

下面这些提示来自 [Daniel Kleine](https://github.com/d-kleine) 的建议以及 [PyTorch 官方指南](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html)。

&nbsp;
## 1 基础配置（CPU 或 CUDA 通用）

&nbsp;
### 1.1 安装 Visual Studio 2022

- 勾选 **“Desktop development with C++”** 工作负载。
- 记得安装 **英文语言包**（否则可能遇到 UTF-8 编码错误）。

&nbsp;
### 1.2 打开正确的命令行

请从以下任一终端启动 Python：

- **“x64 Native Tools Command Prompt for VS 2022”**
- **“Visual Studio 2022 Developer Command Prompt”**

或手动初始化环境：

```bash
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

&nbsp;
### 1.3 确认编译器可用

运行：

```bash
cl.exe
```

若能看到版本信息，就说明编译器已就绪。

&nbsp;
## 2 常见错误排查

&nbsp;
### 2.1 报错：`cl not found`

安装 **Visual Studio Build Tools**，选择 “C++ build tools” 工作负载，并从开发者命令行运行 Python。（更多细节参见微软[官方指南](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170)）

&nbsp;
### 2.2 报错：`triton not found`（CUDA 场景）

手动安装 Windows 版本的 Triton：

```bash
pip install "triton-windows<3.4"
```

若使用 `uv`：

```bash
uv pip install "triton-windows<3.4"
```

（前面提到，TorchInductor 在编译 CUDA kernel 时需要 Triton。）

&nbsp;
## 3 其他注意事项

在 Windows 上，`cl.exe` 只能在 Visual Studio Developer 环境中使用。这意味着除非从开发者命令行启动，否则在 Jupyter 等 notebook 中调用 `torch.compile()` 可能会失败。

文章开头提到的 [PyTorch 指南](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html) 对部分用户在 Windows CPU 构建上启用 `torch.compile()` 很有帮助。但请注意它针对的是 PyTorch 的 unstable 分支，阅读时仅作参考。

**如果编译始终存在问题，大可以跳过。`torch.compile()` 只是锦上添花，并非学习本书所必须。**
