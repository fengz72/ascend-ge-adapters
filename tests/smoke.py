"""端到端冒烟测试 — 真实权重跑通管线, 作为 core/ + runtime/ 改动的回归门。

默认覆盖 (Python 侧): source 加载 → adapter 适配 → build_inputs → golden(eager)
    → trace(→AIR) → from_air(io_spec, 真实 pbtxt) → save_bundle → write_manifest。
默认跳过: compile(ATC, 慢且需 pass) / run / compare。

全链路 (含 C++ runtime 闭环, 需先 bash runtime/build.sh):
    PYTHONPATH=.:$PYTHONPATH python3 tests/smoke.py \
        --weights /export/home/models/Qwen2.5-0.5B --device 6 --skip passes
    # --skip passes: 脏环境 (opp/vendors 已装同名 pass) 下避免重复注册冲突, 见 docs §7

需要: 真实权重 + 空闲 NPU。耗时数分钟 (含 dynamo_export, AIR ~1GB)。

用法:
    PYTHONPATH=.:$PYTHONPATH python3 tests/smoke.py \
        --weights /export/home/models/Qwen2.5-0.5B --device 6
"""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.pipeline import run as run_pipeline   # noqa: E402

CONFIG = "models/qwen2.5-0.5b/config/model.yaml"


def _vocab_size(weights_dir):
    """从权重目录的 config.json 读 vocab_size (模型事实来源, 不在测试里写字面量)。"""
    try:
        with open(os.path.join(weights_dir, "config.json")) as f:
            return int(json.load(f)["vocab_size"])
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _prod(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="/export/home/models/Qwen2.5-0.5B")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--config", default=CONFIG)
    p.add_argument("--skip", default="compile,run,compare",
                   help="逗号分隔: export,passes,compile,run,compare")
    args = p.parse_args()

    skip = tuple(s for s in args.skip.split(",") if s)

    # 用真实权重路径临时改写 config (不改仓库内的 model.yaml)
    tmp_cfg_dir = tempfile.mkdtemp()
    work = tempfile.mkdtemp(prefix="ge_smoke_")
    try:
        import yaml
        cfg = yaml.safe_load(open(args.config))
        cfg["source"]["ref"] = args.weights
        cfg["runtime"]["device"] = args.device
        cfg["inputs"]["batch_size"] = 2
        cfg["inputs"]["seq_len"] = 16
        # config 须放在 <model_dir>/config/ 下 (load_config 据此推 model_dir)
        model_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.config)))
        cfg_path = os.path.join(model_dir, "config", "_smoke_model.yaml")
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True)

        out = run_pipeline(cfg_path, skip=skip,
                           device=args.device, batch_size=2, seq_len=16, work_dir=work)

        # ---- 断言产物 ----
        # 分两类, 避免 smoke 退化成"版本快照"(升级即红, 噪声掩盖真回归):
        #   结构不变量 — 与 torchair/CANN 版本无关的形态约束
        #   模型事实   — 从入参 (batch/seq) 与权重 config.json 推导, 不写字面量
        assert os.path.exists(out["air"]) and os.path.getsize(out["air"]) > 0, "AIR 未产出"

        batch, seq = 2, 16
        tokens, n_seq = batch * seq, batch
        vocab = _vocab_size(args.weights)

        io = json.load(open(out["io_spec"]))
        logical = [n["logical"] for n in io["inputs"]]
        nodes = [n["node"] for n in io["inputs"]]
        assert len(io["inputs"]) == 3, io["inputs"]
        assert all(re.fullmatch(r"arg\d+_\d+", n) for n in nodes), f"node 名形态异常: {nodes}"
        assert len(set(nodes)) == len(nodes), f"node 名重复: {nodes}"
        assert set(logical) == {"input_ids", "position_ids", "actual_seq_lengths"}, logical
        assert all(n["shape"] == [-1] and n["dynamic_dims"] == [0] for n in io["inputs"])
        # io_spec 是**图序**, 不假定等于 forward 序 — 顺序对错由 run+compare 端到端兜底
        assert len(io["outputs"]) == 1 and io["outputs"][0]["dynamic_dims"] == [0]
        if vocab:
            assert io["outputs"][0]["shape"] == [-1, vocab], io["outputs"][0]["shape"]

        bj = json.load(open(out["bundle"]))
        # bundle 按 forward 序, logical 集合与 io_spec 一致, shape 与 .bin 字节数自洽
        assert [i["logical"] for i in bj["inputs"]] == \
            ["input_ids", "position_ids", "actual_seq_lengths"], bj["inputs"]
        assert {i["logical"] for i in bj["inputs"]} == set(logical)
        shapes = {i["logical"]: i["shape"] for i in bj["inputs"]}
        assert shapes["input_ids"] == [tokens] and shapes["position_ids"] == [tokens], shapes
        assert shapes["actual_seq_lengths"] == [n_seq], shapes
        for e in bj["inputs"]:
            size = os.path.getsize(os.path.join(work, "verification", e["file"]))
            assert size == _prod(e["shape"]) * 8, f"{e['logical']}.bin 字节数与 shape 不符: {size}"
        assert bj["golden"]["shape"] == [n_seq, vocab], bj["golden"]["shape"]
        assert bj["provenance"]["seed"] == 0 and "git_commit" in bj["provenance"]
        gsize = os.path.getsize(os.path.join(work, "verification", bj["golden"]["file"]))
        assert gsize == n_seq * vocab * 2, f"golden .bin 字节数异常: {gsize}"

        mf = json.load(open(out["manifest"]))
        assert mf["device"] == args.device and mf["passes_vendor"] == "qwen2.5-0.5b"
        assert not mf["graph_path"].startswith(".."), f"manifest 路径错位: {mf['graph_path']}"

        if "run" not in skip:
            idx = os.path.join(work, "verification", "outputs", "outputs.json")
            assert os.path.exists(idx), f"C++ runtime 未产出输出索引: {idx}"
            entries = json.load(open(idx))["outputs"]
            assert len(entries) == 1, entries
            # 运行时实测 shape/logical 须与 golden 一致 (动态维特化后的真实形状)
            assert entries[0]["shape"] == bj["golden"]["shape"], entries
            assert entries[0]["logical"] == bj["golden"]["logical"], entries
        if "run" not in skip and "compare" not in skip:
            rep = out.get("report")
            assert rep and rep.get("pass_overall"), f"精度未通过: {rep}"

        print("\n=== SMOKE PASS ===")
        print(f"  AIR:     {out['air']} ({os.path.getsize(out['air'])/1e6:.0f} MB)")
        print(f"  io_spec: {nodes} ← {logical}")
        print(f"  golden:  {bj['golden']['shape']} ({gsize} B)")
        print(f"  manifest: backend={mf['backend']} device={mf['device']}")
        if out.get("report"):
            print(f"  compare: PASS (cosine={out['report']['cosine_similarity']:.8f}, "
                  f"rel_l2={out['report']['relative_l2_error']:.3e})")
    finally:
        shutil.rmtree(work, ignore_errors=True)
        smoke_cfg = os.path.join(os.path.dirname(os.path.abspath(args.config)), "_smoke_model.yaml")
        if os.path.exists(smoke_cfg):
            os.unlink(smoke_cfg)
        shutil.rmtree(tmp_cfg_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
