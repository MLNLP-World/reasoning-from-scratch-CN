# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# 《从零构建推理模型》配套源码：https://mng.bz/lZ5B
# 代码仓库：https://github.com/rasbt/reasoning-from-scratch

# 向后兼容模块：流式生成函数已移至 ch02.py。
from .ch02 import (
    generate_text_basic_stream,
    generate_text_basic_stream_cache,
)

__all__ = [
    "generate_text_basic_stream",
    "generate_text_basic_stream_cache",
]
