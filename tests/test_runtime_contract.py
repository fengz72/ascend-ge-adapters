"""runtime/ (C++) 契约测试 — 需已构建 ge_runtime, 但**不需 NPU**(用非法 device 99 短路)。

覆盖 C++ 侧的契约解析与错误路径:
  manifest  base_dir 推导 (deploy/ 的父目录) / om_path 为 null 不得炸 JSON 解析 / 未知 backend
  io_spec + bundle  按 logical 配对 / .bin 字节数与具体 shape 不符须清晰报错
  部署态  无 bundle 时须给出可操作的报错 (A3b 后为 --input 用法)

二进制不存在时整体 skip (先 `bash runtime/build.sh`)。
"""

import json
import os
import subprocess

import numpy as np
import pytest

from core.backend import RUNTIME_BIN, default_output_dir

pytestmark = pytest.mark.skipif(
    not os.path.exists(RUNTIME_BIN),
    reason=f"C++ runtime 未构建: {RUNTIME_BIN} (先执行 bash runtime/build.sh)")

VOCAB = 8
# io_spec 输入 (图序): 与 qwen2.5-0.5b 同构 — 两个 int64 动态输入
SPEC_INPUTS = [{"node": "arg1_1", "logical": "input_ids", "dtype": "int64",
                "format": "ND", "shape": [-1], "dynamic_dims": [0]},
               {"node": "arg4_1", "logical": "position_ids", "dtype": "int64",
                "format": "ND", "shape": [-1], "dynamic_dims": [0]}]


def build_tree(root, backend="om_acl", elems=(4, 4), bundle=True, golden_shape=(2, VOCAB),
               n_inputs=1):
    """造一套最小契约树: deploy/manifest + air/io_spec + verification/{bundle,inputs}。

    elems: 各输入的元素数 (决定 .bin 字节数与 bundle 的具体 shape)。
    n_inputs: 用 SPEC_INPUTS 的前 n 个 (1 或 2)。
    """
    os.makedirs(f"{root}/deploy", exist_ok=True)
    os.makedirs(f"{root}/air", exist_ok=True)
    os.makedirs(f"{root}/verification/inputs", exist_ok=True)
    spec_inputs = SPEC_INPUTS[:n_inputs]

    json.dump({"backend": backend, "graph_path": "air/x.air",
               "om_path": "om/x.om" if backend == "om_acl" else None,
               "io_spec": "air/x.io_spec.json", "device": 0,
               "bundle": "verification/bundle.json" if bundle else None},
              open(f"{root}/deploy/manifest.json", "w"), indent=2)
    json.dump({"inputs": spec_inputs,
               "outputs": [{"node": "logits", "logical": "logits", "dtype": "float16",
                            "format": "ND", "shape": [-1, VOCAB], "dynamic_dims": [0]}]},
              open(f"{root}/air/x.io_spec.json", "w"), indent=2)

    entries = []
    for i, node in enumerate(spec_inputs):
        n = elems[i]
        rel = f"inputs/{node['logical']}.bin"
        np.arange(n, dtype=np.int64).tofile(f"{root}/verification/{rel}")
        entries.append({"logical": node["logical"], "shape": [n], "file": rel})
    if bundle:
        json.dump({"inputs": entries,
                   "golden": {"logical": "logits", "shape": list(golden_shape),
                              "file": "golden_logits.bin"},
                   "provenance": {"seed": 0}},
                  open(f"{root}/verification/bundle.json", "w"), indent=2)
    return f"{root}/deploy/manifest.json"


def input_arg(root, logical, shape):
    """部署态 --input 规格: logical:shape:file (.bin 已由 build_tree 落在 verification/inputs/)。"""
    return f"{logical}:{shape}:{root}/verification/inputs/{logical}.bin"


def run(manifest, *args):
    return subprocess.run([RUNTIME_BIN, manifest, *args], capture_output=True, text=True)


def test_default_output_dir_convention(tmp_path):
    manifest = build_tree(str(tmp_path))
    assert default_output_dir(manifest) == f"{tmp_path}/verification/outputs"


def test_base_dir_and_byte_check(tmp_path):
    """base_dir = deploy/ 的父目录; .bin 字节数与 bundle 具体 shape 不符须清晰报错。"""
    root = str(tmp_path)
    manifest = build_tree(root)
    np.arange(3, dtype=np.int64).tofile(f"{root}/verification/inputs/input_ids.bin")  # 故意短

    p = run(manifest, "--device", "99")
    assert p.returncode != 0
    assert "size 24 != shape bytes 32" in p.stderr, p.stderr
    assert f"base={root}" in p.stdout, p.stdout          # manifest 相对路径解析正确


def test_null_om_path_tolerated(tmp_path):
    """ge_session 的 manifest om_path=null — nlohmann 的 value() 对 null 会抛, 须容忍。"""
    manifest = build_tree(str(tmp_path), backend="ge_session")
    p = run(manifest, "--device", "99")
    assert "json.exception" not in p.stderr, p.stderr
    assert "backend=ge_session" in p.stdout, p.stdout


def test_unknown_backend(tmp_path):
    manifest = build_tree(str(tmp_path), backend="nope")
    p = run(manifest, "--device", "99")
    assert p.returncode != 0 and "unknown backend" in p.stderr


def test_missing_bundle_actionable_error(tmp_path):
    """既无 bundle 又无 --input → 报错必须同时指明两条出路 (docs §5.5 部署态)。"""
    manifest = build_tree(str(tmp_path), bundle=False)
    p = run(manifest, "--device", "99")
    assert p.returncode != 0
    assert "bundle" in p.stderr and "--input" in p.stderr, p.stderr


def test_equals_form_option_parsing(tmp_path):
    """--key=value 形式 (Python 侧 --runtime-opt 透传用) 要被正确拆分。"""
    root = str(tmp_path)
    manifest = build_tree(root)
    np.arange(3, dtype=np.int64).tofile(f"{root}/verification/inputs/input_ids.bin")
    p = run(manifest, "--device=99", "--bench=2")
    assert "size 24 != shape bytes 32" in p.stderr, p.stderr   # 说明 --device= 被吃掉而非当位置参数


# ---------------------------------------------------------------- 部署态 (A3b: 无 bundle)

def test_deploy_mode_reaches_backend(tmp_path):
    """--input 齐活时应越过契约层, 直到 device 99 才失败 (证明 plan 合成成功)。"""
    root = str(tmp_path)
    manifest = build_tree(root, bundle=False, n_inputs=2)
    p = run(manifest, "--device", "99",
            "--input", input_arg(root, "input_ids", 4),
            "--input", input_arg(root, "position_ids", 4))
    assert "部署态" in p.stdout, p.stdout
    assert "aclrtSetDevice(99) failed" in p.stderr, p.stderr   # 走到了 backend


def test_deploy_mode_order_follows_io_spec(tmp_path):
    """CLI 给的顺序无关 — plan 必须按 io_spec 的图序产出 (日志顺序即喂入序)。"""
    root = str(tmp_path)
    manifest = build_tree(root, bundle=False, n_inputs=2)
    p = run(manifest, "--device", "99",
            "--input", input_arg(root, "position_ids", 4),
            "--input", input_arg(root, "input_ids", 4))
    assert "部署态" in p.stdout
    assert "aclrtSetDevice(99) failed" in p.stderr, p.stderr


def test_deploy_mode_incomplete_inputs(tmp_path):
    root = str(tmp_path)
    manifest = build_tree(root, bundle=False, n_inputs=2)
    p = run(manifest, "--device", "99", "--input", input_arg(root, "input_ids", 4))
    assert p.returncode != 0
    assert "缺少 --input" in p.stderr and "position_ids" in p.stderr, p.stderr


def test_deploy_mode_unknown_logical(tmp_path):
    """logical 名打错必须报错并列出 io_spec 的合法名 (否则会静默少喂一个输入)。"""
    root = str(tmp_path)
    manifest = build_tree(root, bundle=False, n_inputs=2)
    p = run(manifest, "--device", "99",
            "--input", input_arg(root, "input_ids", 4),
            "--input", f"pos_ids:4:{root}/verification/inputs/position_ids.bin")
    assert p.returncode != 0
    assert "不在 io_spec 里" in p.stderr and "position_ids" in p.stderr, p.stderr


def test_deploy_mode_bad_shape_bytes(tmp_path):
    """shape 与 .bin 字节数不符 → 与 bundle 路径同一套校验。"""
    root = str(tmp_path)
    manifest = build_tree(root, bundle=False, n_inputs=1)
    p = run(manifest, "--device", "99", "--input", input_arg(root, "input_ids", 8))
    assert p.returncode != 0
    assert "size 32 != shape bytes 64" in p.stderr, p.stderr


def test_deploy_mode_malformed_spec(tmp_path):
    root = str(tmp_path)
    manifest = build_tree(root, bundle=False, n_inputs=1)
    p = run(manifest, "--device", "99", "--input", "input_ids:4")     # 缺 file 段
    assert p.returncode != 0 and "--input 格式应为" in p.stderr, p.stderr


# ---------------------------------------------------------------- dump / profiling (acl.json)

def test_acl_json_generated_for_dump_and_profiling(tmp_path):
    """om_acl + --dump/--profiling → 在 output_dir 生成 acl.json (dump 段 + profiler 段)。"""
    root = str(tmp_path)
    manifest = build_tree(root)
    out_dir = f"{root}/out"
    p = run(manifest, "--device", "99", "--output_dir", out_dir,
            "--dump", "--dump_path", f"{root}/dump", "--dump_mode", "all", "--dump_level", "kernel",
            "--profiling", "--profiling_output", f"{root}/prof",
            "--profiling_aic_metrics", "PipeUtilization")
    assert p.returncode != 0                      # device 99 必然失败, 但 acl.json 应先落地
    cfg = json.load(open(f"{out_dir}/acl.json"))
    assert cfg["dump"]["dump_mode"] == "all" and cfg["dump"]["dump_level"] == "kernel"
    assert cfg["dump"]["dump_path"] == f"{root}/dump"
    assert cfg["dump"]["dump_list"] == [{}]       # 未指定 model_name/layer 时是空对象
    assert cfg["profiler"]["switch"] == "on" and cfg["profiler"]["output"] == f"{root}/prof"
    assert cfg["profiler"]["aic_metrics"] == "PipeUtilization"
    assert cfg["profiler"]["task_time"] == "on" and cfg["profiler"]["ascendcl"] == "on"
    assert "dump 与 profiling 同开" in p.stdout, p.stdout     # 两套机制同开须告警


def test_profiling_no_flags(tmp_path):
    root = str(tmp_path)
    manifest = build_tree(root)
    out_dir = f"{root}/out"
    run(manifest, "--device", "99", "--output_dir", out_dir, "--profiling",
        "--profiling_no_task_time", "--profiling_no_ascendcl")
    cfg = json.load(open(f"{out_dir}/acl.json"))
    assert cfg["profiler"]["task_time"] == "off" and cfg["profiler"]["ascendcl"] == "off"
    assert "aic_metrics" not in cfg["profiler"]               # 未指定就不写该键
    assert "dump" not in cfg


def test_dump_ignored_on_ge_session(tmp_path):
    """dump 是 OM/ACL 专属 (aclmdl*Dump API) — 在线后端须 WARN 忽略而不是静默丢弃。"""
    manifest = build_tree(str(tmp_path), backend="ge_session")
    p = run(manifest, "--device", "99", "--dump")
    assert "dump 是 OM/ACL 专属能力" in p.stdout, p.stdout
