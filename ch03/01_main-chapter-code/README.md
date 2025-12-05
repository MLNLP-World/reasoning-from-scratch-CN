# 第3章：评估推理模型


&nbsp;
## 本章主要代码

- [ch03_main.ipynb](ch03_main.ipynb)：本章主代码
- [ch03_exercise-solutions.ipynb](ch03_exercise-solutions.ipynb)：习题解答


&nbsp;
## 更多资料

- [../02_math500-verifier-scripts/evaluate_math500.py](../02_math500-verifier-scripts/evaluate_math500.py)：在命令行独立评估 MATH-500 数据集的脚本
- [../02_math500-verifier-scripts/evaluate_math500_batched.py](../02_math500-verifier-scripts/evaluate_math500_batched.py)：与上类似，但在生成阶段并行处理多条样本以提升吞吐

上述评估脚本会从 [`reasoning_from_scratch`](../../reasoning_from_scratch) 包中导入功能以避免重复造轮子。（安装细节可参考[第2章的环境说明](../../ch02/02_setup-tips/python-instructions.md)。）
