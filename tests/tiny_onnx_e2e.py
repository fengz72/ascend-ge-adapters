"""形态③ ONNX e2e — 客户直接给 .onnx, 走 pipeline 的 onnx 分支到 OM 执行。

链路: 用 onnx helper 造一个极小 ONNX (MatMul+Add+Relu, 静态 shape)
     → pipeline(source.type=onnx): Graph.from_onnx(io_spec) → ATC(--framework=5) → OM
     → manifest(无 bundle) → ge_runtime **部署态** (--input logical:shape:file)
     → outputs.json / output_0.bin
     → 与 onnxruntime 的 CPU 参考输出比对 (形态③ 无 golden, 参考输出只在测试里用)

不依赖 torch.onnx (它要 onnxscript); 只用 onnx + numpy + onnxruntime。

用法 (需先 bash runtime/build.sh):
    PYTHONPATH=.:$PYTHONPATH python3 tests/tiny_onnx_e2e.py --device 6
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.backend import RUNTIME_BIN                       # noqa: E402
from core.pipeline import run as run_pipeline              # noqa: E402

IN_DIM, OUT_DIM, BATCH = 64, 32, 4


def make_onnx(path, seed=0):
    """造 tiny.onnx (x[BATCH,IN_DIM] → out[BATCH,OUT_DIM]), 返回 (W, B) 供参考计算。"""
    rng = np.random.RandomState(seed)
    w = (rng.randn(IN_DIM, OUT_DIM) * 0.1).astype(np.float32)
    b = rng.randn(OUT_DIM).astype(np.float32) * 0.01
    nodes = [helper.make_node("MatMul", ["x", "W"], ["mm"]),
             helper.make_node("Add", ["mm", "B"], ["add"]),
             helper.make_node("Relu", ["add"], ["out"])]
    graph = helper.make_graph(
        nodes, "tiny",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [BATCH, IN_DIM])],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, [BATCH, OUT_DIM])],
        initializer=[numpy_helper.from_array(w, "W"), numpy_helper.from_array(b, "B")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8                     # 保守版本, ATC 的 onnx parser 覆盖面最广
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return w, b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--soc", default="Ascend910_9382")
    p.add_argument("--work-dir", default=None)
    args = p.parse_args()

    if not os.path.exists(RUNTIME_BIN):
        raise SystemExit(f"C++ 运行时未构建: {RUNTIME_BIN} (先执行 bash runtime/build.sh)")

    work = args.work_dir or tempfile.mkdtemp(prefix="ge_onnx_")
    model_dir = os.path.join(work, "model")
    os.makedirs(os.path.join(model_dir, "config"), exist_ok=True)
    try:
        # ---- 1. 造 ONNX + 输入 .bin (静态 shape → io_spec 无动态维 → ATC 传 --input_shape) ----
        onnx_path = os.path.join(work, "tiny.onnx")
        make_onnx(onnx_path)
        x = np.random.RandomState(1).randn(BATCH, IN_DIM).astype(np.float32)
        bin_path = os.path.join(work, "x.bin")
        x.tofile(bin_path)

        import yaml
        yaml.safe_dump({
            "model": {"name": "tiny-onnx", "soc": args.soc},
            "source": {"type": "onnx", "ref": onnx_path},
            "graph": {"format": "onnx"},
            "passes": [],
            "backend": {"type": "om_acl"},
            "verify": {"enabled": False},
        }, open(os.path.join(model_dir, "config", "model.yaml"), "w"), allow_unicode=True)

        # ---- 2. 跑管线 (部署态: 无 bundle → --input 给具体 shape + .bin) ----
        out = run_pipeline(os.path.join(model_dir, "config", "model.yaml"),
                           skip=("passes",), device=args.device, work_dir=work,
                           runtime_inputs=(f"x:{BATCH},{IN_DIM}:{bin_path}",))

        # ---- 3. io_spec 由 ONNX 图 I/O 派生 ----
        io = json.load(open(out["io_spec"]))
        assert [n["logical"] for n in io["inputs"]] == ["x"], io["inputs"]
        assert io["inputs"][0]["dtype"] == "float32", io["inputs"][0]
        assert io["inputs"][0]["shape"] == [BATCH, IN_DIM], io["inputs"][0]
        assert io["inputs"][0]["dynamic_dims"] == [], io["inputs"][0]
        assert [n["logical"] for n in io["outputs"]] == ["out"], io["outputs"]
        assert out["om"] and os.path.getsize(out["om"]) > 0, "OM 未产出"

        # ---- 4. 运行时输出 vs onnxruntime CPU 参考 ----
        entries = json.load(open(os.path.join(out["outputs"], "outputs.json")))["outputs"]
        assert len(entries) == 1 and entries[0]["logical"] == "out", entries
        assert entries[0]["shape"] == [BATCH, OUT_DIM], entries
        assert entries[0]["dtype"] == "float32", entries

        got = np.fromfile(os.path.join(out["outputs"], entries[0]["file"]),
                          dtype=np.float32).reshape(entries[0]["shape"])
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ref = sess.run(None, {"x": x})[0]

        cos = float(np.dot(got.ravel(), ref.ravel()) /
                    (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-8))
        rel_l2 = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-8))
        print(f"[onnx] vs onnxruntime(CPU): cosine={cos:.8f} rel_l2={rel_l2:.3e} "
              f"max_abs={np.max(np.abs(got - ref)):.3e}")
        assert cos > 0.9999 and rel_l2 < 0.01, f"ONNX 路径精度异常: cosine={cos}, rel_l2={rel_l2}"

        print("\n=== ONNX E2E PASS (form ③: onnx → ATC(fw=5) → OM → 部署态 run) ===")
        print(f"  io_spec: {out['io_spec']}")
        print(f"  om:      {out['om']} ({os.path.getsize(out['om'])/1e6:.1f} MB)")
        print(f"  outputs: {out['outputs']}")
    finally:
        if not args.work_dir:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
