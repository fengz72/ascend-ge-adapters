"""极小模型 e2e — 低显存跑通阶段二闭环 (两后端), 作为 core/ + runtime/ 的回归门。

流程: eager golden → dynamo_export(AIR, dynamic) → io_spec/bundle/manifest (走 core/)
     → ATC → OM → ge_runtime(om_acl) → compare
     → ge_runtime(ge_session, 复用同一 AIR/bundle) → compare

不加载真实权重 (显存 ~几百 MB, 分钟级), 只验**运行时链路**; 真实模型的适配正确性
(原版 HF 参考比对) 与精度/性能验收归 models/<model>/ 的 run.sh 与 docs/。

用法 (需先 bash runtime/build.sh):
    PYTHONPATH=.:$PYTHONPATH python3 tests/tiny_e2e.py --device 1
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import torch
import torch_npu
from torch_npu.dynamo.torchair import CompilerConfig, dynamo_export

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import _torchair_source_name                  # noqa: E402
from core.backend import RUNTIME_BIN, default_output_dir       # noqa: E402
from core.config import (BackendCfg, ModelConfig, ModelMeta,   # noqa: E402
                         SourceCfg, write_manifest)
from core.graph import Graph, IoNode                           # noqa: E402
from core.verify import Verifier, collect_provenance           # noqa: E402
from tools.atc_utils import run_atc                            # noqa: E402

IN_DIM, OUT_DIM, TOKENS = 64, 32, 4


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(IN_DIM, OUT_DIM, bias=False)

    def forward(self, x):
        return torch.relu(self.lin(x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--soc", default="Ascend910_9382")
    p.add_argument("--work-dir", default=None, help="产物目录 (默认临时目录, 结束即删)")
    p.add_argument("--bench", type=int, default=3)
    args = p.parse_args()

    if not os.path.exists(RUNTIME_BIN):
        raise SystemExit(f"C++ 运行时未构建: {RUNTIME_BIN} (先执行 bash runtime/build.sh)")

    work = args.work_dir or tempfile.mkdtemp(prefix="ge_tiny_")
    os.makedirs(work, exist_ok=True)
    torch_npu.npu.set_device(args.device)
    try:
        model = Tiny().to(torch.float16).npu().eval()
        x = torch.randn(TOKENS, IN_DIM, dtype=torch.float16, device="npu")
        torch._dynamo.mark_dynamic(x, 0)
        with torch.no_grad():
            golden = model(x)
        print(f"[tiny] golden {tuple(golden.shape)} {golden.dtype} (device {args.device})")

        air_dir = os.path.join(work, "air")
        os.makedirs(air_dir, exist_ok=True)
        assert _torchair_source_name.enable() or _torchair_source_name.native_support(), \
            "source_name 补丁未生效 — io_spec 会退回按位置映射"
        cfg = CompilerConfig()
        cfg.experimental_config.frozen_parameter = 1
        dynamo_export(x, model=model, export_path=air_dir, export_name="tiny",
                      dynamic=True, config=cfg)
        torch.npu.synchronize()
        air_path = os.path.join(air_dir, "tiny.air")
        assert os.path.exists(air_path) and os.path.getsize(air_path) > 0, "AIR 未产出"
        pbtxt = os.path.join(air_dir, "dynamo.pbtxt")
        hit = subprocess.run(["grep", "-a", "-c", "-m1", "_source_name", pbtxt],
                             capture_output=True, text=True).stdout.strip()
        assert hit not in ("", "0"), "导出图的 Data 节点没有 _source_name (补丁未生效?)"

        graph = Graph.from_air(
            air_path,
            [IoNode(logical="x", dtype="float16", shape=[-1, IN_DIM], dynamic_dims=[0])],
            [IoNode(logical="out", dtype="float16", shape=[-1, OUT_DIM], dynamic_dims=[0])])
        io_spec_path = os.path.join(air_dir, "tiny.io_spec.json")
        graph.io_spec.to_json(io_spec_path)
        assert graph.io_spec.inputs[0].node.startswith("arg"), graph.io_spec.inputs[0].node

        verify = Verifier()
        bundle_path = verify.save_bundle(
            os.path.join(work, "io"), [x], golden, graph.io_spec,
            collect_provenance(seed=0, model="tiny", soc=args.soc))

        om = run_atc(air_path, os.path.join(work, "om"), args.soc)
        assert om, "ATC 编译失败"

        for backend in ("om_acl", "ge_session"):
            report = _run_backend(backend, args, work, graph, io_spec_path,
                                  bundle_path, om, verify)
            assert report["pass_overall"], f"{backend} 精度未通过: {report}"
            print(f"[tiny] {backend}: PASS cosine={report['cosine_similarity']:.8f} "
                  f"rel_l2={report['relative_l2_error']:.3e}")

        print("\n=== TINY E2E PASS (om_acl + ge_session) ===")
    finally:
        if not args.work_dir:
            shutil.rmtree(work, ignore_errors=True)


def _run_backend(backend, args, work, graph, io_spec_path, bundle_path, om, verify):
    """写 manifest (backend 二选一) → 跑 ge_runtime → compare, 返回指标 dict。"""
    cfg = ModelConfig(model=ModelMeta(name="tiny", soc=args.soc),
                      source=SourceCfg(type="torch"),
                      backend=BackendCfg(type=backend))
    manifest = write_manifest(cfg, graph.path, om if backend == "om_acl" else None,
                              io_spec_path, bundle_path, base_dir=work, device=args.device)
    out_dir = default_output_dir(manifest)
    subprocess.run([RUNTIME_BIN, manifest, "--output_dir", out_dir,
                    "--device", str(args.device), "--warmup", "1", "--bench", str(args.bench)],
                   check=True)
    report = verify.compare_bundle(bundle_path, out_dir, dtype="float16", verbose=False)
    kept = os.path.join(work, "io", f"outputs_{backend}")
    shutil.rmtree(kept, ignore_errors=True)
    os.rename(out_dir, kept)
    return report


if __name__ == "__main__":
    main()
