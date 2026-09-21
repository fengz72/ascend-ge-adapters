"""
GE 导出基类 — PyTorch → AIR 的通用管线

管线: adapter.load/adapt → build_inputs → golden → trace (mark_dynamic
    → dynamo_export, frozen_parameter + dynamic=True) → 校验
图的边界由模型文件的 patched forward 决定 (适配后 model(...) 即图接口)。
子类钩子 (模型相关):
    build_inputs(model, **kw)    dummy 输入 (与 patched forward 签名一致, NPU 张量)
    mark_dynamic(inputs, **kw)   标记动态维度 (默认 no-op)

ATC 编译 (AIR/ONNX → OM) 不在此处 — 归 core/backend.compile_graph (委托 tools.atc_utils)。
"""

import os

import torch
import torch_npu  # noqa: F401  torch.npu 依赖
from torch_npu.dynamo.torchair import dynamo_export, CompilerConfig


class GeExporter:
    """通用 AIR 导出器 — 无模型专属逻辑。

    输入生成/动态标记/io_spec 节点声明全部委托给 adapter (模型专属知识归 adapter,
    见 GeModelAdapter.build_inputs/mark_dynamic/io_input_nodes)。本类只负责
    mark_dynamic 后的 dynamo_export 通用流程。

    用法:
        air = GeExporter(adapter, export_dir, export_name).trace(model, inputs, **input_kwargs)
    """

    def __init__(self, adapter, export_dir, export_name, frozen_parameter=1):
        self.adapter = adapter
        self.export_dir = export_dir
        self.export_name = export_name
        self.frozen_parameter = frozen_parameter

    # ---- 管线 ----

    def trace(self, model, inputs, **input_kwargs):
        """adapter.mark_dynamic + dynamo_export → air_path (model/inputs 已就绪)。

        单独拆出以便 pipeline 在 trace 前插入 golden (eager) —— docs §10。
        导出前启用 `_source_name` 回移补丁 (core/_torchair_source_name.py), 让图的 Data
        节点带上 forward 入参名 — graph.from_air 据此自动完成 node↔logical 配对。
        """
        from core import _torchair_source_name
        _torchair_source_name.enable()

        inputs = self.adapter.mark_dynamic(inputs, **input_kwargs)

        config = CompilerConfig()
        config.experimental_config.frozen_parameter = self.frozen_parameter

        os.makedirs(self.export_dir, exist_ok=True)
        air_path = os.path.join(self.export_dir, f"{self.export_name}.air")

        print(f"=== 导出 AIR (动态): {air_path} ===")
        for i, t in enumerate(inputs):
            print(f"  input[{i}]: shape={tuple(t.shape)} dtype={t.dtype}")

        dynamo_export(
            *inputs,
            model=model,
            export_path=self.export_dir,
            export_name=self.export_name,
            dynamic=True,
            config=config,
        )
        torch.npu.synchronize()

        if os.path.exists(air_path):
            size = os.path.getsize(air_path) / 1024 / 1024
            print(f"=== AIR 导出完成: {air_path} ({size:.1f} MB) ===")
        else:
            print(f"=== [WARN] AIR 文件未生成: {air_path} "
                  f"(请检查上方日志中的 'export error!') ===")
        return air_path
