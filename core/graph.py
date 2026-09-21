"""Graph 抽象 + io_spec (图接口契约)。

契约见 docs/architecture.md §4/§5.3/§6:
    io_spec 记**动态维声明** (shape 中 -1), 不含 file、不含具体 shape;
    具体 shape 由 bundle.json 承载 (verify.py)。

io_spec 来源:
    AIR  — exporter 提供 logical/shape/dtype/dynamic (它知道 forward 签名与 dummy 张量),
           from_air 只从 dynamo.pbtxt 解析 Data 节点真名 (arg1_1...) 按位置填入。
    ONNX — from_onnx 直接从 onnx 图 I/O 解析。
"""

import json
import os
import re
from dataclasses import dataclass, field, asdict

# onnx TensorProto elem_type → dtype 字符串
_ONNX_DTYPE = {
    1: "float32", 2: "uint8", 3: "int8", 5: "int16", 6: "int32", 7: "int64",
    9: "bool", 10: "float16", 11: "double", 12: "uint32", 13: "uint64",
    16: "bfloat16",
}


@dataclass
class IoNode:
    node: str = ""                 # 图里真名 (AIR Data 节点 / onnx I/O 名)
    logical: str = ""              # 语义名 (exporter 声明 / onnx 名)
    dtype: str = ""
    format: str = "ND"
    shape: list = field(default_factory=list)        # -1 = 动态维
    dynamic_dims: list = field(default_factory=list)  # 动态轴下标


@dataclass
class IoSpec:
    inputs: list = field(default_factory=list)       # list[IoNode]
    outputs: list = field(default_factory=list)

    def to_json(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"inputs": [asdict(n) for n in self.inputs],
                       "outputs": [asdict(n) for n in self.outputs]},
                      f, indent=2, ensure_ascii=False)
        return path

    @staticmethod
    def from_json(path) -> "IoSpec":
        with open(path) as f:
            d = json.load(f)
        return IoSpec(inputs=[IoNode(**n) for n in d.get("inputs", [])],
                      outputs=[IoNode(**n) for n in d.get("outputs", [])])


@dataclass
class Graph:
    kind: str                      # air | onnx
    path: str
    io_spec: IoSpec = field(default_factory=IoSpec)

    @staticmethod
    def from_air(air_path, inputs: list, outputs: list) -> "Graph":
        """inputs/outputs: exporter 提供的 IoNode (logical/shape/dtype/dynamic 已填,
        **按 forward/build_inputs 入参序**)。从 dynamo.pbtxt 解析 Data 节点真名填入 node。

        node↔logical 配对靠 Data 节点的 `_source_name` 属性 (= forward 入参名, 由
        torchair PR#3675 / core._torchair_source_name 回移补丁写入), **不靠位置**:
        图 Data 序 ≠ dynamo_export 入参序 (qwen2.5-0.5b 实测图序为 actual_seq_lengths,
        input_ids, position_ids), 按位置映射会静默喂错张量。

        io_spec.inputs 按 **index 序 (= C++/ATC 的喂入序)** 落盘; bundle 仍按 forward 序,
        两侧靠 logical 名配对 (docs §5.3/§5.4)。

        配对不全 (pbtxt 无 `_source_name`, 或 logical 名与 forward 入参名不一致) 时
        **默认硬失败** — 静默按位置映射等于喂错张量。仅在 GE_ALLOW_POSITIONAL_IO_SPEC=1
        时退回位置映射并 WARN (适用于确知图序 == forward 序的场景)。
        """
        entries = _parse_air_data_nodes(air_path)
        nodes, unmatched = _pair_by_source(entries, inputs)
        if len(nodes) == len(inputs) and not unmatched:
            return Graph(kind="air", path=air_path,
                         io_spec=IoSpec(inputs=nodes, outputs=outputs))

        detail = (f"配对 {len(nodes)}/{len(inputs)}, 未匹配 Data 节点 {unmatched or '无'}, "
                  f"adapter 声明 logical={[n.logical for n in inputs]}")
        if os.environ.get("GE_ALLOW_POSITIONAL_IO_SPEC") != "1":
            raise RuntimeError(
                f"无法从 {air_path} 的 Data 节点确定 node↔logical 配对 ({detail})。\n"
                f"图 Data 序 ≠ forward 入参序, 按位置硬配会**静默喂错张量** → 默认硬失败。\n"
                f"排查: ① 导出是否启用了 _source_name (GeExporter.trace 会自动调 "
                f"core._torchair_source_name.enable(); 直接调 dynamo_export 则需自己调) "
                f"② io_input_nodes 的 logical 名是否与 forward 入参名逐字一致。\n"
                f"确知图序 == forward 序时, 可 GE_ALLOW_POSITIONAL_IO_SPEC=1 放行退化路径。")

        print(f"[graph][WARN] GE_ALLOW_POSITIONAL_IO_SPEC=1 → 退回按 forward 序位置映射 ({detail})")
        fallback = list(inputs)
        for i, io in enumerate(fallback):
            if i < len(entries):
                io.node = entries[i][1]
            else:
                io.node = io.node or io.logical
        return Graph(kind="air", path=air_path,
                     io_spec=IoSpec(inputs=fallback, outputs=outputs))

    @staticmethod
    def from_onnx(onnx_path) -> "Graph":
        """从 onnx 图 I/O 解析 io_spec (node=logical=onnx 名)。"""
        import onnx
        m = onnx.load(onnx_path)
        g = m.graph

        def parse(vi):
            t = vi.type.tensor_type
            dtype = _ONNX_DTYPE.get(t.elem_type, f"onnx_{t.elem_type}")
            shape, dyn = [], []
            for i, d in enumerate(t.shape.dim):
                if d.HasField("dim_value"):
                    shape.append(d.dim_value)
                else:
                    shape.append(-1)
                    dyn.append(i)
            return IoNode(node=vi.name, logical=vi.name, dtype=dtype,
                          format="ND", shape=shape, dynamic_dims=dyn)

        io_spec = IoSpec(inputs=[parse(x) for x in g.input],
                         outputs=[parse(x) for x in g.output])
        return Graph(kind="onnx", path=onnx_path, io_spec=io_spec)


def _pair_by_source(entries: list, inputs: list):
    """按 Data 节点的 `_source_name` 把 logical 配到 node 名, 返回 (图序 IoNode, 未匹配 node)。

    entries: [(index, node_name, source_name)] 已按 index 升序 → 输出顺序即**图喂入序**。
    名字对不上 (source 缺失/与 adapter 声明的 logical 不一致) 的项进 unmatched, 由调用方
    决定退化策略 — 绝不静默按位置硬配 (错序会让 C++ 喂错张量, 表现为 tiling 崩或精度全错)。
    """
    by_logical = {n.logical: n for n in inputs}
    ordered, unmatched = [], []
    for _, node_name, source in entries:
        io = by_logical.get(_logical_of(source))
        if io is None:
            unmatched.append(node_name)
            continue
        io.node = node_name
        ordered.append(io)
    return ordered, unmatched


def _logical_of(source):
    """`local:input_ids` / `global:x` / `input_ids` → logical 名; 取不到返回 None。"""
    if not source:
        return None
    for prefix in ("local:", "global:"):
        if source.startswith(prefix):
            source = source[len(prefix):]
            break
    return source or None


def _parse_air_data_nodes(air_path) -> list:
    """从 dynamo.pbtxt 提取 op=Data 的 (index, 节点名, _source_name), 按 index 升序返回。

    index 序 = **图侧喂入序** (C++/ATC 按位置喂入); `_source_name` = forward 入参名
    (`local:<name>`, 由 torchair PR#3675 或 core._torchair_source_name 回移补丁写入),
    是 node↔logical 配对的唯一可靠依据 —— 实测图序 ≠ dynamo_export 入参序
    (qwen2.5-0.5b 图序为 actual_seq_lengths, input_ids, position_ids)。

    pbtxt 因 frozen_parameter 内嵌权重可达 GB 级, 用 grep 流式提取避免全量读入。
    格式注意: 是 `op: "Data"` (非 op_type); name 在 op 前一行, index/_source_name 在其后 attr 内。
    """
    if os.path.isdir(air_path):
        pbtxt = os.path.join(air_path, "dynamo.pbtxt")
    else:
        pbtxt = os.path.join(os.path.dirname(air_path), "dynamo.pbtxt")
    if not os.path.exists(pbtxt):
        return []

    import subprocess
    try:
        out = subprocess.run(["grep", "-a", "-B1", "-A12", 'op: "Data"', pbtxt],
                             capture_output=True, text=True, timeout=600)
        text = out.stdout
    except Exception as e:
        print(f"[graph][WARN] pbtxt Data 节点提取失败 ({e})")
        return []

    entries = []
    for group in text.split("\n--\n"):
        name = re.search(r'name:\s*"([^"]+)"', group)
        idx = re.search(r'key:\s*"index".*?i:\s*(\d+)', group, re.S)
        src = re.search(r'key:\s*"_source_name"[\s\S]{0,80}?s:\s*"([^"]+)"', group)
        if name and idx:
            entries.append((int(idx.group(1)), name.group(1),
                            src.group(1) if src else None))
    entries.sort()
    return entries
