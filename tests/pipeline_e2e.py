"""pipeline 级 e2e — 走 `core.pipeline.run()` 的编排全链路, 补 tiny_e2e 的覆盖缺口。

与 tiny_e2e 的分工 (两道门, 不重复):
    tiny_e2e.py     直接调 Graph/Verifier/run_atc/ge_runtime — 验**契约层与运行时**
                    (io_spec 图序配对、bundle/manifest 解析、两后端执行)
    pipeline_e2e.py 调 pipeline.run() — 验**编排层与配置层**, 即 tiny_e2e 手写绕过的那些:
        load_config + platforms 强 schema + resolve_platform (按真机 soc 探测并摊平)
        adapt.form 声明化 (tag 进产物名 / load 把文件载成 list / flags 增删脚本参数)
        load_adapter (importlib 载入 model.py + 通用 load)
        run_scripts (custom_ops/passes/inputs 三类用户脚本 + $GE_ENV_FILE 回传)
        _generate_inputs (子进程产激励 → 读回 → 与 io_input_nodes 的 logical 序对账)
        GeExporter.trace (含 _source_name 补丁) → Graph.from_air → save_bundle
        write_build_record / check_build_record (构建指纹: 改配置后拒绝复用旧产物)
        adapter 三组契约 (patch_specs/setup、unpack_requests/reference_columns、
                          mark_dynamic/io_input_nodes/io_output_nodes)
        两道精度门 (reference 验适配 / compare 验编译) 与 FAIL 时的非 0 退出

模型是**极小的 HF 形态假模型** (Embedding+ReLU+Linear, 测试文件里定义), 不加载真实权重 ——
`core.source.load_source` 被替换成造这个假模型 (它写死 AutoModelForCausalLM.from_pretrained,
真实权重归 models/<m>/run.sh 那道门)。适配手法与 qwen2.5-0.5b 同构, 只是小:
    L0 (原版)  forward(input_ids=[1,T], position_ids, use_cache) → .logits [1,T,out]
    L1 (适配)  forward(input_ids=[T], actual_seq_lengths=[N])    → [N,out'] (packed, 取末 token)
    patch      torch.nn.ReLU.forward → 数值等价的 clamp (patch **torch 的类**而非本模型模块,
               故不受 importlib 每次 exec 新建模块对象的影响; setup 里断言补丁确实生效)
    setup      按 keep_tokens 剪裁 lin 的输出宽 (= qwen 的 lm_head 剪裁)
    门① 靠 unpack_requests (packed → 逐请求) + reference_columns (只比保留的列)

用法 (需先 bash runtime/build.sh):
    PYTHONPATH=.:$PYTHONPATH python3 tests/pipeline_e2e.py --device 1
    PYTHONPATH=.:$PYTHONPATH python3 tests/pipeline_e2e.py --device 1 --work-dir /tmp/keep
"""

import argparse
import copy
import json
import os
import shutil
import stat
import sys
import tempfile
from types import SimpleNamespace

import torch
import torch_npu
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import pipeline as pipeline_mod                      # noqa: E402
from core.backend import RUNTIME_BIN                            # noqa: E402
from core.config import (export_name, load_adapter, load_config,               # noqa: E402
                         resolve_platform)

VOCAB, HIDDEN, OUT_FULL = 32, 16, 12
KEEP_TOKENS = [3, 5, 7, 11]          # lin 剪裁后保留的列 (= 门① 的 reference_columns)
BATCH, SEQ = 6, 16

MODEL_PY = '''"""极小模型的 adapter — 与 models/qwen2.5-0.5b/model.py 同构, 只是小。"""

import torch
import torch.nn as nn

from core.adapter import GeModelAdapter
from core.graph import IoNode

_ORIG_RELU_FORWARD = torch.nn.ReLU.forward


def _relu_forward(self, x):
    """与 nn.ReLU **数值等价** (clamp), 但走的是 patch 进来的实现。

    等价是必需的: 门① 比对 L0(原版) 与 L1(适配后), patch 改了数值就会 FAIL ——
    真实适配 (融合算子替换) 同理, 换实现不换语义。
    """
    return torch.clamp(x, min=0.0)


class TinyAdapter(GeModelAdapter):
    """patch ReLU + 剪裁输出宽; keep_tokens 由 adapt.form.load 从 json 载入后传入。"""

    MODELING = torch.nn

    def __init__(self, keep_tokens=None, **params):
        super().__init__(**params)
        self.keep_tokens = keep_tokens

    def patch_specs(self):
        # patch **torch 的类** (不是本模块的类): load_adapter 每次 importlib exec 都新建一个
        # 模块对象, 而测试里的假模型是另一处定义的 —— 打在 torch.nn.ReLU 上两边都生效
        return [(torch.nn.ReLU, "forward", _relu_forward)]

    def setup(self, model, **kwargs):
        # 断言 apply_patches 真的生效了 (adapt 先 patch 后 setup, 见 core/adapter.py)
        assert torch.nn.ReLU.forward is _relu_forward, "apply_patches 未生效"
        assert torch.nn.ReLU.forward is not _ORIG_RELU_FORWARD, "patch 目标没被替换"
        if not self.keep_tokens:
            return
        device, dtype = model.lin.weight.device, model.lin.weight.dtype
        idx = torch.as_tensor(list(self.keep_tokens), dtype=torch.long, device=device)
        pruned = nn.Linear(model.lin.in_features, len(self.keep_tokens), bias=False)
        pruned.weight = nn.Parameter(model.lin.weight.data.index_select(0, idx).clone())
        model.lin = pruned.to(device, dtype)
        model.config.vocab_size = len(self.keep_tokens)

    def mark_dynamic(self, inputs, **kwargs):
        for t in inputs:
            torch._dynamo.mark_dynamic(t, 0)        # T 与 N 都动态
        return inputs

    def io_input_nodes(self, inputs, **kwargs):
        # logical 名须 == forward 入参名 (graph.from_air 靠 Data 节点的 _source_name 配对),
        # 顺序须 == 输入脚本落盘的 FORWARD_ORDER (pipeline 对账, 不一致硬失败)
        nodes = []
        for t, logical in zip(inputs, ["input_ids", "actual_seq_lengths"]):
            nodes.append(IoNode(logical=logical, dtype=str(t.dtype).replace("torch.", ""),
                                shape=[-1] + list(t.shape)[1:], dynamic_dims=[0]))
        return nodes

    # io_output_nodes 不覆盖 → 用基类默认 (单输出 "logits", dim0 动态)

    def unpack_requests(self, inputs):
        """L1 packed → L0 逐请求 (顺序 = 输出行序); position 用 arange 重新生成。"""
        ids, asl = inputs
        bounds, requests, prev = [int(x) for x in asl.tolist()], [], 0
        for end in bounds:
            tok = ids[prev:end]
            requests.append((tok, torch.arange(tok.numel(), dtype=torch.long,
                                               device=ids.device)))
            prev = end
        return requests

    def reference_columns(self):
        return self.keep_tokens
'''

GEN_INPUTS_PY = '''"""极小模型的输入生成器 — 与 models/<m>/scripts/gen_inputs.py 同一套接口。"""

import argparse
import os
import sys

import numpy as np
import torch                                # main() 里 check_varlen / from_numpy 收张量

sys.path.insert(0, os.environ["GE_TEST_REPO_ROOT"])
from tools.varlen import check_varlen, generate_varlen_inputs, write_inputs  # noqa: E402

FORWARD_ORDER = ["input_ids", "actual_seq_lengths"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--vocab", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prefix-len", type=int, default=0)   # 本模型不用; 验 form.flags 的增删不报错
    a = p.parse_args()

    ids, pos, _lens, cum = generate_varlen_inputs(a.batch_size, a.seq_len,
                                                  vocab_size=a.vocab, seed=a.seed)
    tensors = {"input_ids": ids.squeeze(0).numpy().astype(np.int64),
               "actual_seq_lengths": np.array(cum, dtype=np.int64)}
    segments = check_varlen(torch.from_numpy(tensors["input_ids"]),
                            pos.squeeze(0), tensors["actual_seq_lengths"])
    assert segments == a.batch_size, (segments, a.batch_size)
    prov = {k: v for k, v in vars(a).items() if k != "out"}
    prov["total_tokens"] = int(tensors["input_ids"].size)
    prov["segments"] = segments
    os.makedirs(a.out, exist_ok=True)
    path = write_inputs(a.out, tensors, FORWARD_ORDER, prov)
    print(f"[gen_inputs] N={a.batch_size} T={prov['total_tokens']} vocab={a.vocab} → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

NOOP_SH = '''#!/bin/bash
# 验 run_scripts: 执行到了 + $GE_SRC_DIR 传进来了 + $GE_ENV_FILE 能把 env 回传给框架
set -euo pipefail
echo "[noop] custom_ops 阶段执行 (GE_SRC_DIR=${GE_SRC_DIR:-<未声明>})"
[ -n "${GE_ENV_FILE:-}" ] && echo "GE_TINY_MARKER=ok" >> "${GE_ENV_FILE}"
'''


class TinyOut:
    """假 HF 输出容器 (verify.reference 取 .logits)。"""

    def __init__(self, logits):
        self.logits = logits


class Tiny(torch.nn.Module):
    """极小假模型: 一个 forward 同时服务 L0 (逐请求 HF 形态) 与 L1 (packed varlen)。

    靠 `actual_seq_lengths is None` 分流 —— 该分支在 trace 期就是 Python 常量, 不会进图。
    emb/relu/lin 都是**逐位置**的, 故 L0 取末 token 与 L1 的 index_select(asl-1) 数值恒等,
    门① 才有意义 (它验的是"适配没改语义")。
    """

    def __init__(self, dtype):
        super().__init__()
        self.config = SimpleNamespace(use_cache=True, vocab_size=VOCAB)   # adapt() 要写 use_cache
        self.emb = torch.nn.Embedding(VOCAB, HIDDEN)
        self.act = torch.nn.ReLU()
        self.lin = torch.nn.Linear(HIDDEN, OUT_FULL, bias=False)
        self.to(dtype)

    def get_input_embeddings(self):
        # pipeline._generate_inputs 从这里读 vocab (**不是** config.vocab_size — 剪裁会改写它)
        return self.emb

    def forward(self, input_ids, actual_seq_lengths=None, position_ids=None, use_cache=False):
        o = self.lin(self.act(self.emb(input_ids)))
        if actual_seq_lengths is None:                 # L0: [1,T,out] 带 .logits
            return TinyOut(o)
        return o.index_select(0, actual_seq_lengths - 1)   # L1: [N,out] 每请求末 token


def fake_load_source(cfg, model_dir, dtype=torch.float16, device=None):
    """替换 core.source.load_source: 不加载真实权重, 造 Tiny (已上卡、eval)。"""
    if device is None:
        raise ValueError("device 必填")
    torch_npu.npu.set_device(device)
    return Tiny(dtype).npu().eval()


def write_model_dir(root, soc, backend, repo_root):
    """生成一个自洽的 models/<name>/ 目录 (model.py + config/model.yaml + scripts/)。"""
    cfg_dir = os.path.join(root, "config")
    scripts = os.path.join(root, "scripts")
    os.makedirs(cfg_dir, exist_ok=True)
    os.makedirs(scripts, exist_ok=True)

    with open(os.path.join(root, "model.py"), "w") as f:
        f.write(MODEL_PY)
    gen = os.path.join(scripts, "gen_inputs.py")
    with open(gen, "w") as f:
        f.write(GEN_INPUTS_PY.replace("os.environ[\"GE_TEST_REPO_ROOT\"]", repr(repo_root)))
    noop = os.path.join(scripts, "noop_env.sh")
    with open(noop, "w") as f:
        f.write(NOOP_SH)
    os.chmod(noop, os.stat(noop).st_mode | stat.S_IXUSR)

    keep = os.path.join(cfg_dir, "keep_tokens.json")
    with open(keep, "w") as f:
        json.dump({"token_ids": KEEP_TOKENS}, f)      # load.kind=token_ids 走 json.load

    # 路径一律写**绝对**: resolve_path 只认绝对或相对仓库根, 而临时目录在仓库外
    doc = {
        "model": {"name": "tiny-pipeline"},
        "source": {"ref": "unused-load_source-被测试替换"},
        "platforms": {
            # custom_ops 挂一个 no-op 脚本: 验 run_scripts 的执行 + $GE_SRC_DIR + $GE_ENV_FILE 回传
            "probe": {"soc": soc, "passes": [],
                      "custom_ops": [{"script": noop, "path": scripts}]},
            "other": {"soc": "Ascend310P1", "passes": [], "custom_ops": []},
        },
        "adapt": {
            "adapter_class": "TinyAdapter",
            "params": {"keep_token_file": keep},
            "form": {"keep_token_file": {"tag": "keep",
                                         "load": {"kind": "token_ids", "as": "keep_tokens"}}},
        },
        "inputs": {"script": gen, "batch_size": BATCH, "seq_len": SEQ, "seed": 0},
        "graph": {"dynamic": {"max_seq_len": 64}},
        "backend": {"type": backend},
        "verify": {"enabled": True},
    }

    path = os.path.join(cfg_dir, "model.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)
    return path


def check_gate(report, label):
    assert report is not None, f"{label} 门没跑 (report=None)"
    assert report.get("pass_overall"), f"{label} FAIL: {report}"
    print(f"  [gate] {label}: PASS cosine={report['cosine_similarity']:.8f} "
          f"rel_l2={report['relative_l2_error']:.3e}")


def run_backend(backend, root, soc, device, repo_root):
    """跑一遍完整管线, 断言产物/两道门/env 回传。返回 (config_path, 产物 dict)。"""
    model_dir = os.path.join(root, backend)
    os.makedirs(model_dir, exist_ok=True)
    config_path = write_model_dir(model_dir, soc, backend, repo_root)

    print(f"\n{'=' * 70}\n=== pipeline.run() — backend={backend} ===\n{'=' * 70}")
    os.environ.pop("GE_TINY_MARKER", None)
    out = pipeline_mod.run(config_path, device=device, dtype=torch.float16,
                           warmup=1, bench=2)

    # 编排层断言: 产物齐全 + 名字带上了 platform 与 adapt.form 的 tag
    cfg = resolve_platform(load_config(config_path), device=device)
    want = "tiny-pipeline-probe-keep"
    assert export_name(cfg) == want, (export_name(cfg), want)
    assert os.path.basename(out["air"]) == f"{want}.air", out["air"]
    for key in ("air", "io_spec", "bundle", "manifest"):
        assert out[key] and os.path.exists(out[key]), f"{key} 缺产物: {out[key]}"
    if backend == "om_acl":
        assert out["om"] and os.path.exists(out["om"]), out["om"]
    else:
        assert out["om"] is None, "ge_session 不该有 OM"
    assert os.path.exists(os.path.join(os.path.dirname(out["air"]), f"{want}.build.json")), \
        "构建指纹未落盘"

    # run_scripts 的 env 回传 (custom_ops 阶段的 no-op 脚本写了 GE_TINY_MARKER)
    assert os.environ.get("GE_TINY_MARKER") == "ok", \
        "$GE_ENV_FILE 回传未生效 (core/setup_scripts._apply_env_file)"

    # 两道精度门: reference 验适配 (L0 原版 vs L1 适配后), compare 验编译 (运行时 vs golden)
    check_gate(out["reference"], "reference (原版 vs 适配后 eager)")
    check_gate(out["report"], f"compare  (运行时 vs golden, {backend})")

    # io_spec 的图序配对结果: logical 名齐、node 名是图里的真名
    spec = json.load(open(out["io_spec"]))
    logicals = [n["logical"] for n in spec["inputs"]]
    assert sorted(logicals) == ["actual_seq_lengths", "input_ids"], logicals
    assert all(n["node"].startswith("arg") for n in spec["inputs"]), spec["inputs"]
    assert [n["logical"] for n in spec["outputs"]] == ["logits"], spec["outputs"]
    print(f"  io_spec 输入 (图序): {[(n['node'], n['logical']) for n in spec['inputs']]}")
    return config_path, out


def check_build_fingerprint(config_path, device, root):
    """改配置后 --skip export 必须**硬失败**; GE_ALLOW_STALE_ARTIFACT=1 则放行。"""
    print(f"\n{'=' * 70}\n=== 构建指纹: 改配置后拒绝复用旧产物 ===\n{'=' * 70}")
    raw = yaml.safe_load(open(config_path))
    original = raw["inputs"]["seq_len"]
    raw["inputs"]["seq_len"] = original + 8
    with open(config_path, "w") as f:
        yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)

    skip_all = ("ops", "export", "passes", "compile", "run", "compare", "reference")
    try:
        pipeline_mod.run(config_path, skip=skip_all, device=device, dtype=torch.float16)
        raise AssertionError("改了 inputs.seq_len 还复用旧 AIR — check_build_record 没拦住")
    except RuntimeError as e:
        assert "构建指纹" in str(e), e
        print(f"  ✓ 硬失败: {str(e).splitlines()[0]}")
        assert any("seq_len" in line for line in str(e).splitlines()), \
            f"报错没指出是哪个键不一致:\n{e}"

    os.environ["GE_ALLOW_STALE_ARTIFACT"] = "1"
    try:
        pipeline_mod.run(config_path, skip=skip_all, device=device, dtype=torch.float16)
        print("  ✓ GE_ALLOW_STALE_ARTIFACT=1 → WARN 放行")
    finally:
        os.environ.pop("GE_ALLOW_STALE_ARTIFACT", None)
        raw["inputs"]["seq_len"] = original
        with open(config_path, "w") as f:
            yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)


def check_config_guards(config_path):
    """配置层的强 schema 守卫 (不需要跑图, 只解析)。"""
    print(f"\n{'=' * 70}\n=== 配置强 schema 守卫 ===\n{'=' * 70}")

    base = yaml.safe_load(open(config_path))

    def rejects(mutate, label, want):
        raw = copy.deepcopy(base)
        mutate(raw)
        tmp = config_path + ".guard"
        with open(tmp, "w") as f:
            yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)
        try:
            load_config(tmp)
            raise AssertionError(f"{label}: 应当硬失败却通过了")
        except ValueError as e:
            assert want in str(e), f"{label}: 报错没提到 {want!r}\n{e}"
            print(f"  ✓ {label}")
        finally:
            os.unlink(tmp)

    rejects(lambda r: r["adapt"]["params"].__setitem__("new_switch", True),
            "adapt.params 有键未在 adapt.form 分类", "没在 adapt.form 里分类")
    rejects(lambda r: r["platforms"]["probe"].__setitem__("prefix", True),
            "platforms 段写了形态开关", "未知键")
    rejects(lambda r: r["platforms"]["probe"].pop("soc"), "platforms 缺 soc", "缺 soc")
    rejects(lambda r: r.__setitem__("passes", []), "顶层写 passes (已移进 platforms)", "已移进")
    rejects(lambda r: r["adapt"]["form"]["keep_token_file"]["load"].__setitem__("kind", "nope"),
            "adapt.form.load.kind 不认", "不认")
    rejects(lambda r: r["adapt"]["form"]["keep_token_file"]["load"].__setitem__("ass", "x"),
            "adapt.form.load 有未知键", "未知键")


def check_null_param_adapter(config_path):
    """adapt.params 的 load 键置 null (文档的"关"写法) → 路径键不得漏进 adapter kwargs。

    回归守卫: 曾经 pop 只在真值时做, null 时 cls(**params) 会收到 keep_token_file=None
    而 TypeError — adapter 的 __init__ 只收内容 (keep_tokens), 不收路径键。
    """
    print(f"\n{'=' * 70}\n=== adapt.params 置 null: load_adapter 不泄漏路径键 ===\n{'=' * 70}")
    raw = yaml.safe_load(open(config_path))
    raw["adapt"]["params"]["keep_token_file"] = None
    tmp = config_path + ".nullparam"
    with open(tmp, "w") as f:
        yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)
    try:
        adapter = load_adapter(load_config(tmp))
        assert adapter.keep_tokens is None, adapter.keep_tokens
        print("  ✓ null param: 路径键被移除, adapter 只收到 keep_tokens=None")
    finally:
        os.unlink(tmp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--work-dir", default=None, help="产物目录 (默认临时目录, 结束即删)")
    args = p.parse_args()

    if not os.path.exists(RUNTIME_BIN):
        raise SystemExit(f"C++ 运行时未构建: {RUNTIME_BIN} (先执行 bash runtime/build.sh)")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    torch_npu.npu.set_device(args.device)
    # 平台按**真机**探测 (不写死 soc): 顺带验 resolve_platform 的探测路径在任何卡上都成立
    soc = torch.npu.get_device_properties(args.device).name
    print(f"[pipeline_e2e] device={args.device} soc={soc} (探测所得, 写进临时 model.yaml)")

    # load_source 写死 AutoModelForCausalLM.from_pretrained → 换掉 (真实权重归 run.sh 那道门)
    pipeline_mod.load_source = fake_load_source

    root = args.work_dir or tempfile.mkdtemp(prefix="ge_pipe_")
    os.makedirs(root, exist_ok=True)
    try:
        config_path = None
        for backend in ("om_acl", "ge_session"):
            config_path, _ = run_backend(backend, root, soc, args.device, repo_root)
        # 两道负向门用 om_acl 那份配置 (它有 OM, --skip 全开时只走指纹校验)
        check_build_fingerprint(config_path, args.device, root)
        om_cfg = os.path.join(root, "om_acl", "config", "model.yaml")
        check_config_guards(om_cfg)
        check_null_param_adapter(om_cfg)
        print(f"\n{'=' * 70}\n=== PIPELINE E2E PASS "
              f"(om_acl + ge_session + 构建指纹 + 配置守卫) ===\n{'=' * 70}")
    finally:
        if not args.work_dir:
            shutil.rmtree(root, ignore_errors=True)
        else:
            print(f"[pipeline_e2e] 产物保留在 {root}")


if __name__ == "__main__":
    main()
