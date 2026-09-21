"""core — Ascend GE 适配框架 (通用脚手架)。

契约层: config (ModelConfig) / graph (IoSpec, Graph)。
适配层: adapter (GeModelAdapter) / exporter (GeExporter)。
管线层: source / passes / backend / verify / pipeline。

__init__ 只 re-export **契约层** (无 torch/torch_npu/torchair 依赖) — 这样
`import core.backend` / `core.verify` 不会把 NPU 栈拖进来, 离线单测与 CI 才能在
没装 torch_npu 的机器上跑 (docs §14 回归门)。适配层/管线层按子模块直接导入:
`from core.adapter import GeModelAdapter`、`from core.exporter import GeExporter`。
"""

from .config import ModelConfig, load_config, load_adapter, write_manifest
from .graph import IoNode, IoSpec, Graph
