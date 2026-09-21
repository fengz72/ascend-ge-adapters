"""回移 torchair PR#3675 的 Data 节点 `_source_name` 能力 — 让导出图自带 forward 入参名。

问题: GE 图的 Data 节点只有 `argN_1` + `index` + 符号 shape, **没有语义名**, 而 index 序
也不等于 dynamo_export 入参序 (qwen2.5-0.5b 实测图序为 asl, input_ids, position_ids) —
io_spec 的 node↔logical 因此无法从图侧确定。

上游解法 (Ascend/torchair PR#3675, 已合入 master): 用 dynamo 的 source 元数据
(`_try_get_metadata_from_dynamo` → LocalSource/GlobalSource) 把入参名写进 Data 节点的
`_source_name` 属性 (`local:<name>` / `global:<name>`)。本机 torch_npu 内置的 torchair
还没有这段, 故按同一机制在此回移。

对齐关系 (实测确认): `GeConcreteGraph.parse_input` 的 `data_index = self.graph.num_inputs`
与 `arg_pos_to_source` 下标一一对应 (参数/buffer/符号 shape 也各占一位, 后续被冻结成 Const
或在图里重新编号, 但 parse_input 时刻的下标是对齐的)。

设计: 上游已原生支持时自动 no-op; 任何异常只 WARN 不阻断导出 (退化为"按 forward 序假定",
由 compare 阶段兜底)。
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
        from torch._dynamo.source import GlobalSource, LocalSource
        from torch._functorch.aot_autograd import _try_get_metadata_from_dynamo

        import torchair._ge_concrete_graph.ge_apis as ge_apis
        import torchair.npu_fx_compiler as fx_compiler
        from torchair._ge_concrete_graph import fx2ge_converter
        from torchair.ge import _ge_graph as ge_graph
    except Exception as e:                                   # torchair 结构变了 → 放弃补丁
        print(f"[source_name][WARN] 无法导入 torchair 内部模块 ({e}); io_spec 退回按 forward 序假定")
        return False

    def _data(*args, **kwargs):
        name = kwargs.pop("source_name", None) or _STATE.get("current")
        tensor = _orig_data(*args, **kwargs)
        if name:
            ge_graph.get_default_ge_graph().op[-1].attr["_source_name"].s = \
                ge_graph.compat_as_bytes(name)
        return tensor

    def _parse_input(self, target, args, kwargs, meta_outputs):
        sources = _STATE.get("sources") or ()
        index = self.graph.num_inputs
        name = None
        if index < len(sources):
            source = sources[index]
            if isinstance(source, LocalSource):
                name = f"local:{source.local_name}"
            elif isinstance(source, GlobalSource):
                name = f"global:{source.global_name}"
        _STATE["current"] = name
        try:
            return _orig_parse_input(self, target, args, kwargs, meta_outputs)
        finally:
            _STATE["current"] = None

    def _backend(gm, example_inputs, *args, **kwargs):
        try:
            params = dict(gm.named_parameters(remove_duplicate=False))
            buffers = dict(gm.named_buffers(remove_duplicate=False))
            full_args_num = len(params) + len(buffers) + len(example_inputs)
            sources = _try_get_metadata_from_dynamo(
                gm, {**params, **buffers}.keys(), full_args_num)[0]
            _STATE["sources"] = sources or []
        except Exception as e:
            print(f"[source_name][WARN] dynamo source 元数据采集失败 ({e})")
            _STATE["sources"] = []
        return _orig_backend(gm, example_inputs, *args, **kwargs)

    try:
        _orig_data = ge_apis.Data
        _orig_parse_input = fx2ge_converter.GeConcreteGraph.parse_input
        _orig_backend = fx_compiler._npu_backend
        ge_apis.Data = _data
        fx2ge_converter.GeConcreteGraph.parse_input = _parse_input
        fx_compiler._npu_backend = _backend
    except Exception as e:
        print(f"[source_name][WARN] 打补丁失败 ({e}); io_spec 退回按 forward 序假定")
        return False
    return True
