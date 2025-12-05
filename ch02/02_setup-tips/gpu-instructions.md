
# GPU云端资源

本节介绍在云端运行本书代码的替代方案。

虽然这些代码在普通笔记本或台式机（没有独立GPU）上也能运行，但配备NVIDIA GPU的云平台能够显著缩短运行时间，尤其是第5~7章。

&nbsp;

## 使用Lightning Studio

若想在云端获得顺畅的开发体验，我推荐 [Lightning AI Studio](https://lightning.ai/) 平台。它允许创建持久化环境，并且可以在云端CPU或GPU上使用VSCode与Jupyter Lab。

启动新的Studio之后，打开终端并执行以下步骤克隆仓库、安装依赖：

```bash
git clone https://github.com/rasbt/reasoning-from-scratch.git
cd reasoning-from-scratch
pip install -r requirements.txt
```

（与Google Colab不同，Lightning AI Studio的环境会持久化，即便在CPU和GPU之间切换，以上步骤也只需执行一次。）

然后进入你想运行的Python脚本或Jupyter Notebook。需要更高性能时，也可以很方便地挂载GPU，比如在第5章预训练LLM或第6、7章做微调时。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/studio.webp" alt="1" width="700">

&nbsp;

## 使用Google Colab

如需使用Google Colab云端环境，访问 [https://colab.research.google.com/](https://colab.research.google.com/)，从GitHub菜单打开对应章节的Notebook，或像下图所示把Notebook文件拖放到 *Upload* 区域。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_1.webp" alt="1" width="700">


同时别忘了把相关文件（数据集、Notebook引用的.py文件等）一并上传到Colab环境，如下图所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_2.webp" alt="2" width="700">


如需使用GPU，可按照下图所示修改 *Runtime* 设置。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_3.webp" alt="3" width="700">


&nbsp;
## 有问题？

若有任何疑问，欢迎在该GitHub仓库的 [Discussions](https://github.com/rasbt/reasoning-from-scratch/discussions) 论坛交流。
