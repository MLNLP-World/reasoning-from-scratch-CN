# 第 3 章：高级解析器（附加材料）

此文件夹包含议题 [#133](https://github.com/rasbt/reasoning-from-scratch/issues/133) 中的解析器实验；该议题提出了一种混合 LaTeX 解析器，用于处理本章当前解析器可能遗漏的边界情况。

&nbsp;

## 文件

- [compare_with_current_parser.ipynb](compare_with_current_parser.ipynb)：包含使用示例的 notebook
- [math500_gpt_answers.json](math500_gpt_answers.json)：包含 LLM 答案的 MATH-500 样本，用于上述 notebook 的一个小节
- [gen_llm_answers.py](gen_llm_answers.py)：以 JSON 格式获取 Qwen3 模型方框答案的便捷脚本
- [evaluate_math500_advanced.py](evaluate_math500_advanced.py)：与第 3 章 LLM 评估脚本 [evaluate_math500.py](../02_math500-verifier-scripts/evaluate_math500.py) 相同，但额外支持 `--hybrid_parser` 参数以使用另一种混合解析器，例如：

```python
uv run evaluate_math500_advanced.py --dataset_size 500 --hybrid_parser
```

&nbsp;
## 与第 3 章解析器的区别

[reasoning_from_scratch/ch03.py](../../reasoning_from_scratch/ch03.py) 中的本章解析器力求保持简洁且便于教学：

- 侧重轻量归一化和符号等价性检查
- 主要将答案视为算术/符号表达式

此文件夹中的混合解析器（`latex_normalizer_hybrid.py`）优先匹配模式，覆盖范围更广：

- 在回退解析之前识别答案格式
- 增加对区间、并集、方程、矩阵、集合表示法、成员关系（`\in`）和 `\pm` 的支持
- 能更好地保留重要边界情况，例如带进制下标的答案（`52_8`）和文本大小写（`\text{Evelyn}`）

行为不同的示例：

- `52_8` -> 本章解析路径通常得到 `528`；混合解析器保留 `52_8`
- `11,\! 111,\! 111,\! 100` -> 本章解析路径可能得到元组；混合解析器归一化为 `11111111100`
- `(0,9) \cup (9,36)` -> 本章解析路径通常保留为文本；混合解析器返回符号并集

权衡：

- 本章解析器：更简单、更快且更容易解释
- 混合解析器：对 LaTeX 边界情况覆盖更好，但规则和复杂度也更高；此外还增加了 SymPy LaTeX 后端依赖

&nbsp;
## 使用方法

可以直接从包中导入混合解析器：

```python
from reasoning_from_scratch.bonus.parser import normalize_text_hybrid, sympy_parser_hybrid
```

更详细的使用示例参见 [compare_with_current_parser.ipynb](compare_with_current_parser.ipynb)。
