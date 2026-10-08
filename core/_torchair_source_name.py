"""导出 AIR 时给图的 Data 节点写 `_source_name` (= forward 入参名)。

Data 节点只有 `argN_1` + `index`, 无语义名, 且 index 序 ≠ dynamo_export 入参序
(qwen2.5-0.5b 实测图序为 asl, input_ids, position_ids) — 按位置配 node↔logical 会
静默喂错张量。

补 torchair 两处: `_npu_backend` 用 `_try_get_metadata_from_dynamo` 取各入参的
LocalSource/GlobalSource 存进 `_STATE`; `parse_input` 按 `self.graph.num_inputs` 查表,
写进 `data.node.attr["_source_name"]`。下标能对齐是因为 `arg_pos_to_source` =
[params..., buffers..., placeholders...], 与 parse_input 的调用序一致。

上游 torchair (截至 master) 无此能力; 原生支持时 (ge.Data 带 source_name 形参) 自动 no-op。
异常只 WARN 不阻断导出 → 节点没有 `_source_name` → 由 graph.from_air 硬失败兜底。
"""

_STATE = {}
_PATCHED = None


def native_support() -> bool:
    """已装 torchair 是否原生支持 ge.Data(source_name=...)。"""
    try:
        import inspect

        from torchair.ge import _ge_graph as ge_graph
        return "source_name" in inspect.signature(ge_graph.Data).parameters
    except Exception:
        return False


def enable() -> bool:
    """打补丁, 返回是否生效 (原生已支持 → False, 无需补丁)。幂等。"""
    global _PATCHED
    if _PATCHED is not None:
        return _PATCHED
    _PATCHED = _install()
    return _PATCHED


def _install() -> bool:
    if native_support():
        return False
    try:
        import inspect

        from torch._dynamo.source import GlobalSource, LocalSource
        from torch._functorch.aot_autograd import _try_get_metadata_from_dynamo
        from torch._functorch._aot_autograd.descriptors import (
            BufferAOTInput, ParamAOTInput, PlainAOTInput)

        import torchair.npu_fx_compiler as fx_compiler
        from torchair._ge_concrete_graph import fx2ge_converter
        from torchair.ge import _ge_graph as ge_graph
    except Exception as e:                                   # torch/torchair 结构变了 → 放弃补丁
        print(f"[source_name][WARN] 无法导入 torch/torchair 内部模块 ({e}); "
              f"Data 节点将无 _source_name, graph.from_air 会硬失败")
        return False

    # torch 2.10 给 _try_get_metadata_from_dynamo 加了必填的 full_args_descs (2.9 是 3 参);
    # dynamo 图分支里不用它, 只有 export 分支遍历, 按签名探测决定要不要补第 4 个实参。
    _needs_descs = "full_args_descs" in inspect.signature(
        _try_get_metadata_from_dynamo).parameters

    def _parse_input(self, target, args, kwargs, meta_outputs):
        index = self.graph.num_inputs
        data = _orig_parse_input(self, target, args, kwargs, meta_outputs)
        sources = _STATE.get("sources") or ()
        source = sources[index] if index < len(sources) else None
        if isinstance(source, LocalSource):
            name = f"local:{source.local_name}"
        elif isinstance(source, GlobalSource):
            name = f"global:{source.global_name}"
        else:
            return data
        data.node.attr["_source_name"].s = ge_graph.compat_as_bytes(name)
        return data

    def _backend(gm, example_inputs, *args, **kwargs):
        try:
            params = dict(gm.named_parameters(remove_duplicate=False))
            buffers = dict(gm.named_buffers(remove_duplicate=False))
            call = [gm, {**params, **buffers}.keys(),
                    len(params) + len(buffers) + len(example_inputs)]
            if _needs_descs:
                call.append([ParamAOTInput(k) for k in params]
                            + [BufferAOTInput(k) for k in buffers]
                            + [PlainAOTInput(i) for i in range(len(example_inputs))])
            _STATE["sources"] = _try_get_metadata_from_dynamo(*call)[0] or []
        except Exception:
            import traceback
            print(f"[source_name][WARN] dynamo source 元数据采集失败:\n{traceback.format_exc()}")
            _STATE["sources"] = []
        return _orig_backend(gm, example_inputs, *args, **kwargs)

    try:
        _orig_parse_input = fx2ge_converter.GeConcreteGraph.parse_input
        _orig_backend = fx_compiler._npu_backend
        fx2ge_converter.GeConcreteGraph.parse_input = _parse_input
        fx_compiler._npu_backend = _backend
    except Exception as e:
        print(f"[source_name][WARN] 打补丁失败 ({e}); Data 节点将无 _source_name, "
              f"graph.from_air 会硬失败")
        return False
    return True
