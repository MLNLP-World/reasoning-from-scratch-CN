# 排行榜排名

这份额外材料介绍了两种依据成对比较构建 LM Arena（原 Chatbot Arena）风格排行榜的方法。

这两个实现都通过 `--path` 参数，从 JSON 文件读取成对偏好列表（左：胜者，右：败者）。下面是提供的 [votes.json](votes.json) 文件节选：

```json
[
  ["GPT-5", "Claude-3"],
  ["GPT-5", "Llama-4"],
  ["Claude-3", "Llama-3"],
  ["Llama-4", "Llama-3"],
  ...
]
```

<br>

---

**注意**：如果你不是 `uv` 用户，请在下面的示例中将 `uv run ...py` 替换为 `python ...py`。

---

&nbsp;
## 方法 1：Elo 评分

- 实现了 LM Arena 最初使用的、受国际象棋排名启发的流行 Elo 评分方法
- 详见[主笔记本](../01_main-chapter-code/chF_main.ipynb)

```bash
➜  03_leaderboards git:(main) ✗ uv run 1_elo_leaderboard.py --path votes.json

Leaderboard (Elo) 
-----------------------
 1. GPT-5       1095.9
 2. Claude-3    1058.7
 3. Llama-4      958.2
 4. Llama-3      887.2
```

&nbsp;
## 方法 2：Bradley-Terry 模型

- 实现了一个 [Bradley-Terry 模型](https://en.wikipedia.org/wiki/Bradley–Terry_model)，与官方论文（[Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132)）描述的新 LM Arena 排行榜类似
- 与 LM Arena 排行榜一样，得分会重新缩放，使其接近原始 Elo 分数
- 此处示例使用 PyTorch 的 Adam 优化器来拟合模型（便于熟悉 PyTorch 代码并保持良好可读性）

```bash
➜  03_leaderboards git:(main) ✗ uv run 2_bradley_terry_leaderboard.py --path votes.json

Leaderboard (Bradley-Terry)
-----------------------------
 1. GPT-5       1140.6
 2. Claude-3    1058.7
 3. Llama-4      950.3
 4. Llama-3      850.4
```
