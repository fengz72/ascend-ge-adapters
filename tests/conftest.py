"""pytest 共享 fixture — 把仓库根加入 sys.path (测试用 `from core... import`)。

约定: `tests/test_*.py` = 纯 CPU 单测 (CI 可跑, 不 import torch/torch_npu);
      `tests/{tiny_e2e,smoke,...}.py` = 需要 NPU 的脚本 (pytest 不收集, 手动跑)。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
