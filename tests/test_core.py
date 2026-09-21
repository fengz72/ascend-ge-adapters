"""core/ 纯函数单测 — 纯 CPU, 不 import torch/torch_npu, CI 可跑。

专打**静默失败点**(错了不崩、只是结果不对):
  graph  node↔logical 配对与图序 (错配 = C++ 按位置喂错张量) / 配对不全默认硬失败
  config yaml→dataclass (缺字段默认、class→class_name 改名) / load_adapter 构造约定
  verify compare_bundle 的 shape 门禁与按 logical 匹配
  manifest 相对路径 (可移植性)

需要 NPU 的链路见 tests/tiny_e2e.py、tests/smoke.py (脚本, pytest 不收集)。
"""

import json
import os

import numpy as np
import pytest

from core.adapter import GeModelAdapter
from core.config import load_adapter, load_config, write_manifest, AdaptCfg, ModelConfig, ModelMeta, SourceCfg
from core.graph import Graph, IoNode, IoSpec, _logical_of, _pair_by_source, _parse_air_data_nodes
from core.verify import Verifier, bundle_has_golden

# ---------------------------------------------------------------- 工具

def write_pbtxt(dirpath, data_nodes, extra_const=True):
    """按真实 dynamo.pbtxt 排版造 fixture。data_nodes: [(name, index, source|None)]。"""
    blocks = []
    for name, index, source in data_nodes:
        attrs = ["    attr {\n        key: \"index\"\n        value {\n"
                 f"            s: 'i: {index}\\n'\n        }}\n    }}"]
        if source:
            attrs.append("    attr {\n        key: \"_source_name\"\n        value {\n"
                         f"            s: 's: \"{source}\"\\n'\n        }}\n    }}")
        attrs.append("    attr {\n        key: \"_input_name_key\"\n        value {\n"
                     "            s: 'list {\\n  s: \"x\"\\n}\\n'\n        }\n    }")
        blocks.append(f'node {{\n    name: "{name}"\n    op: "Data"\n' + "\n".join(attrs) + "\n}")
        if extra_const:                      # 真图里 Data 之间夹着 Const, 确认不会被误当 Data
            blocks.append(f'node {{\n    name: "const_{name}"\n    op: "Const"\n'
                          "    attr {\n        key: \"index\"\n        value {\n"
                          "            s: 'i: 99\\n'\n        }\n    }\n}")
    (dirpath / "dynamo.pbtxt").write_text("\n".join(blocks) + "\n")
    return str(dirpath / "m.air")


def io_inputs():
    """forward 入参序 (与 qwen2.5-0.5b 的 adapter 声明一致)。"""
    return [IoNode(logical="input_ids", dtype="int64", shape=[-1], dynamic_dims=[0]),
            IoNode(logical="position_ids", dtype="int64", shape=[-1], dynamic_dims=[0]),
            IoNode(logical="actual_seq_lengths", dtype="int64", shape=[-1], dynamic_dims=[0])]


def io_outputs():
    return [IoNode(logical="logits", dtype="float16", shape=[-1, 8], dynamic_dims=[0])]


# ---------------------------------------------------------------- graph: 名字解析

@pytest.mark.parametrize("raw,expect", [
    ("local:input_ids", "input_ids"),
    ("global:table", "table"),
    ("input_ids", "input_ids"),          # 裸名 (未来上游若不带前缀)
    ("local:", None),
    ("", None),
    (None, None),
])
def test_logical_of(raw, expect):
    assert _logical_of(raw) == expect


def test_parse_data_nodes_sorted_and_source(tmp_path):
    """index 乱序写入也要按 index 升序返回; _source_name 的嵌套转义要能解析。"""
    write_pbtxt(tmp_path, [("arg7_1", 2, "local:position_ids"),
                           ("arg1_1", 0, "local:actual_seq_lengths"),
                           ("arg4_1", 1, "local:input_ids")])
    entries = _parse_air_data_nodes(str(tmp_path / "m.air"))
    assert entries == [(0, "arg1_1", "local:actual_seq_lengths"),
                       (1, "arg4_1", "local:input_ids"),
                       (2, "arg7_1", "local:position_ids")]


def test_parse_data_nodes_without_source(tmp_path):
    write_pbtxt(tmp_path, [("arg1_1", 0, None)])
    assert _parse_air_data_nodes(str(tmp_path / "m.air")) == [(0, "arg1_1", None)]


def test_parse_data_nodes_missing_pbtxt(tmp_path):
    assert _parse_air_data_nodes(str(tmp_path / "nope.air")) == []


# ---------------------------------------------------------------- graph: 配对与图序

def test_pair_by_source_emits_graph_order():
    """关键: 输出顺序 = 图 index 序, 不是 forward 序 (qwen 实测两者不同)。"""
    entries = [(0, "arg1_1", "local:actual_seq_lengths"),
               (1, "arg4_1", "local:input_ids"),
               (2, "arg7_1", "local:position_ids")]
    nodes, unmatched = _pair_by_source(entries, io_inputs())
    assert [(n.node, n.logical) for n in nodes] == [
        ("arg1_1", "actual_seq_lengths"), ("arg4_1", "input_ids"), ("arg7_1", "position_ids")]
    assert unmatched == []


def test_pair_by_source_reports_unmatched():
    entries = [(0, "arg1_1", None), (1, "arg4_1", "local:不存在")]
    nodes, unmatched = _pair_by_source(entries, [IoNode(logical="x")])
    assert nodes == []
    assert unmatched == ["arg1_1", "arg4_1"]


def test_from_air_pairs_by_source(tmp_path):
    air = write_pbtxt(tmp_path, [("arg1_1", 0, "local:actual_seq_lengths"),
                                 ("arg4_1", 1, "local:input_ids"),
                                 ("arg7_1", 2, "local:position_ids")])
    graph = Graph.from_air(air, io_inputs(), io_outputs())
    assert [n.logical for n in graph.io_spec.inputs] == \
        ["actual_seq_lengths", "input_ids", "position_ids"]
    assert [n.node for n in graph.io_spec.inputs] == ["arg1_1", "arg4_1", "arg7_1"]
    # 非输入字段原样保留
    assert graph.io_spec.inputs[1].dtype == "int64"
    assert graph.io_spec.outputs[0].shape == [-1, 8]


def test_from_air_hard_fails_without_source(tmp_path, monkeypatch):
    """配对不全默认硬失败 — 静默按位置映射等于喂错张量。"""
    monkeypatch.delenv("GE_ALLOW_POSITIONAL_IO_SPEC", raising=False)
    air = write_pbtxt(tmp_path, [("arg1_1", 0, None)])
    with pytest.raises(RuntimeError, match="node↔logical"):
        Graph.from_air(air, [IoNode(logical="x", dtype="int64", shape=[-1], dynamic_dims=[0])],
                       io_outputs())


def test_from_air_positional_fallback_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("GE_ALLOW_POSITIONAL_IO_SPEC", "1")
    air = write_pbtxt(tmp_path, [("arg1_1", 0, None)])
    graph = Graph.from_air(air, [IoNode(logical="x", dtype="int64", shape=[-1], dynamic_dims=[0])],
                           io_outputs())
    assert [(n.node, n.logical) for n in graph.io_spec.inputs] == [("arg1_1", "x")]


# ---------------------------------------------------------------- config

FULL_YAML = """
model: {name: m1, soc: Ascend910_9382}
source: {type: torch, ref: /w, class: MyModel, module: my.py}
adapt: {adapter_class: Minimal, params: {prefix: false}}
inputs: {batch_size: 4, seq_len: 32, prefix_len: 0, seed: 7}
graph: {format: air, dynamic: {max_seq_len: 1024}}
passes: [WeightNzAndMatMulV3Pass]
backend: {type: ge_session, aicore_num: 12}
runtime: {device: 6}
verify: {enabled: false}
"""

MINIMAL_YAML = """
model: {name: m2}
source: {type: name, ref: Qwen/Qwen2.5-0.5B}
"""


def _cfg_dir(tmp_path, text, name="m"):
    root = tmp_path / name
    (root / "config").mkdir(parents=True)
    path = root / "config" / "model.yaml"
    path.write_text(text)
    return str(path), str(root)


def test_load_config_full(tmp_path):
    path, root = _cfg_dir(tmp_path, FULL_YAML)
    cfg = load_config(path)
    assert cfg.model_dir == root
    assert cfg.model.name == "m1" and cfg.model.soc == "Ascend910_9382"
    assert cfg.source.class_name == "MyModel"          # yaml 的 `class` → 字段 class_name
    assert cfg.source.module == "my.py"
    assert cfg.adapt.params == {"prefix": False}
    assert (cfg.inputs.batch_size, cfg.inputs.seq_len, cfg.inputs.seed) == (4, 32, 7)
    assert cfg.graph.format == "air" and cfg.graph.dynamic.max_seq_len == 1024
    assert cfg.passes == ["WeightNzAndMatMulV3Pass"]
    assert cfg.backend.type == "ge_session" and cfg.backend.aicore_num == 12
    assert cfg.runtime.device == 6 and cfg.verify.enabled is False


def test_load_config_defaults_and_unknown_keys(tmp_path):
    path, _ = _cfg_dir(tmp_path, MINIMAL_YAML + "\nunknown_section: {a: 1}\n")
    cfg = load_config(path)
    assert cfg.model.soc == "Ascend910_9382"           # 缺字段用默认
    assert (cfg.inputs.batch_size, cfg.inputs.seq_len) == (10, 208)
    assert cfg.graph.dynamic.max_seq_len == 2048
    assert cfg.backend.type == "om_acl" and cfg.backend.aicore_num is None
    assert cfg.runtime.device == 0 and cfg.verify.enabled is True
    assert cfg.passes == []


# ---------------------------------------------------------------- load_adapter (A1 回归)

MODEL_PY = '''
from core.adapter import GeModelAdapter


class Minimal(GeModelAdapter):
    """不写 __init__ — 基类收 **params, 必须能被 load_adapter 直接实例化。"""


class NoKwargs(GeModelAdapter):
    def __init__(self):
        super().__init__()


class Pruner(GeModelAdapter):
    def __init__(self, prune_tokens=None, **params):
        super().__init__(**params)
        self.prune_tokens = prune_tokens
'''


def _model_dir(tmp_path, adapter_class, params=None):
    root = tmp_path / "mdl"
    (root / "config").mkdir(parents=True)
    (root / "model.py").write_text(MODEL_PY)
    (root / "config" / "tokens.json").write_text(json.dumps({"token_ids": [1, 2, 3]}))
    cfg = ModelConfig(model=ModelMeta(name="mdl"), source=SourceCfg(type="torch"),
                      adapt=AdaptCfg(adapter_class=adapter_class, params=params or {}),
                      model_dir=str(root))
    return cfg, root


def test_load_adapter_minimal_no_init(tmp_path):
    cfg, _ = _model_dir(tmp_path, "Minimal", {"prefix": False})
    adapter = load_adapter(cfg)
    assert isinstance(adapter, GeModelAdapter)
    assert adapter.params == {"prefix": False}


def test_load_adapter_no_kwargs_constructor(tmp_path):
    """A1 回归: 适配器构造函数不收 kwargs 时, 不得被塞 prune_tokens=None。"""
    cfg, _ = _model_dir(tmp_path, "NoKwargs")
    assert isinstance(load_adapter(cfg), GeModelAdapter)


def test_load_adapter_prune_token_file(tmp_path):
    cfg, root = _model_dir(tmp_path, "Pruner", {"prune_token_file": "config/tokens.json"})
    assert load_adapter(cfg).prune_tokens == [1, 2, 3]


def test_load_adapter_prune_not_declared(tmp_path):
    """未声明 prune_token_file → 不传 prune_tokens (保持 None)。"""
    cfg, _ = _model_dir(tmp_path, "Pruner", {"prefix": True})
    adapter = load_adapter(cfg)
    assert adapter.prune_tokens is None and adapter.params == {"prefix": True}


# ---------------------------------------------------------------- manifest

def test_write_manifest_relative_paths(tmp_path):
    base = tmp_path / "work"
    cfg = ModelConfig(model=ModelMeta(name="m1"), source=SourceCfg(type="torch"))
    path = write_manifest(cfg, str(base / "air" / "m1.air"), str(base / "om" / "m1.om"),
                          str(base / "air" / "m1.io_spec.json"),
                          str(base / "verification" / "bundle.json"),
                          base_dir=str(base), device=6)
    m = json.load(open(path))
    assert m == {"backend": "om_acl", "graph_path": "air/m1.air", "om_path": "om/m1.om",
                 "io_spec": "air/m1.io_spec.json", "device": 6, "passes_vendor": "m1",
                 "bundle": "verification/bundle.json"}


def test_write_manifest_null_om(tmp_path):
    """ge_session 无离线产物 → om_path 为 null (C++ 侧须容忍 null, 见 test_runtime_contract)。"""
    base = tmp_path / "work"
    cfg = ModelConfig(model=ModelMeta(name="m1"), source=SourceCfg(type="torch"))
    path = write_manifest(cfg, str(base / "air" / "m1.air"), None,
                          str(base / "air" / "s.json"), None, base_dir=str(base))
    m = json.load(open(path))
    assert m["om_path"] is None and m["bundle"] is None


# ---------------------------------------------------------------- verify.compare_bundle (D4 回归)

def _bundle_tree(tmp_path, golden_shape, out_shape, out_data=None):
    root = tmp_path / "v"
    (root / "inputs").mkdir(parents=True)
    (root / "out").mkdir(parents=True)
    golden = np.arange(int(np.prod(golden_shape)), dtype=np.float16).reshape(golden_shape)
    golden.tofile(root / "golden_logits.bin")
    np.arange(4, dtype=np.int64).tofile(root / "inputs" / "input_ids.bin")
    json.dump({"inputs": [{"logical": "input_ids", "shape": [4], "file": "inputs/input_ids.bin"}],
               "golden": {"logical": "logits", "shape": list(golden_shape),
                          "file": "golden_logits.bin"},
               "provenance": {"seed": 0}}, open(root / "bundle.json", "w"))
    # 默认输出与 golden 同值 (PASS 用例); out_data 非空则用它 (FAIL 用例)
    flat = (np.asarray(out_data, dtype=np.float16).ravel() if out_data is not None
            else np.resize(golden.ravel(), int(np.prod(out_shape))))
    data = flat.astype(np.float16).reshape(out_shape)
    data.tofile(root / "out" / "output_0.bin")
    json.dump({"outputs": [{"logical": "logits", "dtype": "float16",
                            "shape": list(out_shape), "bytes": int(data.nbytes),
                            "file": "output_0.bin"}]}, open(root / "out" / "outputs.json", "w"))
    return str(root / "bundle.json"), str(root / "out")


def test_compare_bundle_pass_and_fail(tmp_path):
    bundle, out = _bundle_tree(tmp_path / "pass", (2, 8), (2, 8))
    rep = Verifier().compare_bundle(bundle, out, dtype="float16", verbose=False)
    assert rep["pass_overall"] and rep["cosine_similarity"] > 0.9999

    bad = np.random.RandomState(0).randn(2, 8).astype(np.float16)
    bundle, out = _bundle_tree(tmp_path / "fail", (2, 8), (2, 8), out_data=bad)
    rep = Verifier().compare_bundle(bundle, out, dtype="float16", verbose=False)
    assert not rep["pass_overall"]


def test_compare_bundle_rejects_shape_mismatch(tmp_path):
    """D4 回归: 门禁不得 flatten/截断兜底 (否则错序/错 shape 也能 cosine PASS)。"""
    bundle, out = _bundle_tree(tmp_path, (2, 8), (4, 8))
    with pytest.raises(ValueError, match="形状不一致"):
        Verifier().compare_bundle(bundle, out, dtype="float16", verbose=False)


def test_compare_bundle_matches_by_logical(tmp_path):
    """输出项按 logical 与 golden 配对, 不按位置 (多输出时位置会错)。"""
    bundle, out = _bundle_tree(tmp_path, (2, 8), (2, 8))
    index = json.load(open(os.path.join(out, "outputs.json")))
    index["outputs"].insert(0, {"logical": "hidden", "dtype": "float16", "shape": [2, 4],
                                "bytes": 16, "file": "output_0.bin"})
    json.dump(index, open(os.path.join(out, "outputs.json"), "w"))
    rep = Verifier().compare_bundle(bundle, out, dtype="float16", verbose=False)
    assert rep["pass_overall"]


def test_compare_bundle_missing_outputs(tmp_path):
    bundle, out = _bundle_tree(tmp_path, (2, 8), (2, 8))
    os.remove(os.path.join(out, "outputs.json"))
    with pytest.raises(FileNotFoundError, match="outputs.json"):
        Verifier().compare_bundle(bundle, out, dtype="float16", verbose=False)


# ---------------------------------------------------------------- passes 路径解析

def test_resolve_pass_dir(tmp_path):
    """yaml 的 passes 条目: 路径 (绝对/相对仓库根/相对 CWD) 与 只写名字 都要能解析。"""
    from core.passes import resolve_pass_dir

    tp = tmp_path / "third_party"
    src = tp / "custom_development_code" / "fusion_pass" / "FooPass"
    src.mkdir(parents=True)

    rel = "third_party/custom_development_code/fusion_pass/FooPass"
    assert resolve_pass_dir(rel, str(tp)) == ("FooPass", str(src))       # 相对仓库根
    assert resolve_pass_dir(str(src), str(tp)) == ("FooPass", str(src))  # 绝对路径
    assert resolve_pass_dir(str(src) + "/", str(tp)) == ("FooPass", str(src))  # 尾斜杠
    assert resolve_pass_dir("FooPass", str(tp)) == ("FooPass", str(src))  # 只写名字 (向后兼容)
    assert resolve_pass_dir("Nope", str(tp)) is None
    assert resolve_pass_dir("", str(tp)) is None
    assert resolve_pass_dir(None, str(tp)) is None


def test_pass_manager_skips_missing(tmp_path, capsys):
    """找不到源码只 WARN 不抛 (管线不因 pass 缺失而崩)。"""
    from core.passes import PassManager

    tp = tmp_path / "third_party"
    tp.mkdir()
    PassManager("m", ["Nope"], str(tp)).prepare()
    assert "找不到 pass 源码" in capsys.readouterr().out


# ---------------------------------------------------------------- backend 组合校验 (B1)

def test_ge_session_rejects_onnx():
    """形态③ ONNX 走不了在线后端 — 必须在编译/配置期报错, 而不是等 C++ 运行期。"""
    from core.backend import compile_graph
    from core.config import BackendCfg

    cfg = ModelConfig(model=ModelMeta(name="m"), source=SourceCfg(type="onnx", ref="m.onnx"),
                      backend=BackendCfg(type="ge_session"))
    with pytest.raises(ValueError, match="ge_session 不支持 ONNX"):
        compile_graph(cfg, Graph(kind="onnx", path="m.onnx"))


def test_ge_session_accepts_air():
    from core.backend import compile_graph
    from core.config import BackendCfg

    cfg = ModelConfig(model=ModelMeta(name="m"), source=SourceCfg(type="torch"),
                      backend=BackendCfg(type="ge_session"))
    assert compile_graph(cfg, Graph(kind="air", path="m.air")) is None


def test_unknown_backend_type():
    from core.backend import compile_graph
    from core.config import BackendCfg

    cfg = ModelConfig(model=ModelMeta(name="m"), source=SourceCfg(type="torch"),
                      backend=BackendCfg(type="nope"))
    with pytest.raises(ValueError, match="未知 backend.type"):
        compile_graph(cfg, Graph(kind="air", path="m.air"))


# ---------------------------------------------------------------- verify.save_bundle (A3a)

def _io_spec():
    return IoSpec(inputs=[IoNode(node="arg1_1", logical="actual_seq_lengths", dtype="int64",
                                 shape=[-1], dynamic_dims=[0]),
                          IoNode(node="arg4_1", logical="input_ids", dtype="int64",
                                 shape=[-1], dynamic_dims=[0])],
                  outputs=[IoNode(logical="logits", dtype="float16", shape=[-1, 8],
                                  dynamic_dims=[0])])


def test_save_bundle_labels_by_forward_order(tmp_path):
    """io_spec 是图序 (asl, input_ids), inputs 是 forward 序 (input_ids, asl)
    → 必须靠 logical_order 贴标签, 否则 .bin 与 logical 名错位。"""
    inputs = [np.arange(4, dtype=np.int64), np.array([2, 4], dtype=np.int64)]
    golden = np.zeros((2, 8), dtype=np.float16)
    path = Verifier().save_bundle(str(tmp_path), inputs, golden, _io_spec(),
                                  provenance={"seed": 0},
                                  logical_order=["input_ids", "actual_seq_lengths"])
    b = json.load(open(path))
    assert [(e["logical"], e["shape"]) for e in b["inputs"]] == \
        [("input_ids", [4]), ("actual_seq_lengths", [2])]
    assert b["golden"] == {"logical": "logits", "shape": [2, 8], "file": "golden_logits.bin"}
    assert os.path.getsize(os.path.join(str(tmp_path), "inputs", "input_ids.bin")) == 32
    assert os.path.getsize(os.path.join(str(tmp_path), "inputs", "actual_seq_lengths.bin")) == 16
    assert bundle_has_golden(path) is True


def test_save_bundle_without_golden(tmp_path):
    """A3a: verify.enabled=false → golden=None 也要能落 bundle (只写 inputs)。"""
    inputs = [np.arange(4, dtype=np.int64)]
    path = Verifier().save_bundle(str(tmp_path), inputs, None, _io_spec(), provenance={},
                                  logical_order=["input_ids"])
    b = json.load(open(path))
    assert b["golden"] is None
    assert [e["logical"] for e in b["inputs"]] == ["input_ids"]
    assert not os.path.exists(os.path.join(str(tmp_path), "golden_logits.bin"))
    assert bundle_has_golden(path) is False
    with pytest.raises(ValueError, match="无 golden"):
        Verifier().compare_bundle(path, str(tmp_path), dtype="float16", verbose=False)
