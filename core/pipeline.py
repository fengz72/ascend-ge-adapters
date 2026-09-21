"""GE 适配管线编排 — 读 model.yaml, 串起 source→adapt→export→io_spec→golden→bundle
→passes→compile→manifest。docs/architecture.md §3/§10/§11。

逐模型只剩 model.py(Adapter) + config/model.yaml; 本管线通用、配置驱动。

用法:
    python -m core.pipeline --config models/qwen2.5-0.5b/config/model.yaml
    python -m core.pipeline --config <yaml> --skip compile,run,compare
    # 冒烟/调试可覆盖: --device 6 --batch-size 2 --seq-len 16 --work-dir /tmp/x
"""

import argparse
import os

import torch
import torch_npu

from core.config import load_config, load_adapter, write_manifest
from core.source import load_source
from core.graph import Graph, IoNode, IoSpec
from core.exporter import GeExporter
from core.passes import PassManager
from core.backend import Backend, default_output_dir, run_runtime
from core.verify import Verifier, bundle_has_golden, collect_provenance

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_THIRD_PARTY = os.path.join(_REPO_ROOT, "third_party")


def _export_name(cfg):
    return f"{cfg.model.name}-prefix" if cfg.adapt.params.get("prefix") else cfg.model.name


def run(config_path, skip=(), dtype=torch.float16, device=None,
        batch_size=None, seq_len=None, work_dir=None,
        warmup=0, bench=1, runtime_extra=(), runtime_inputs=()):
    """跑管线, 返回产物路径 dict。skip ⊂ {export,passes,compile,run,compare}。

    run/compare 为阶段二闭环: run 调 C++ runtime (manifest 驱动, OM/ACL 或 GeSession),
    compare 用 bundle golden 比对运行时输出 (docs §10)。

    runtime_inputs: 部署态输入规格 ("logical:d0,d1,...:file.bin", 可多条) — 仅 ONNX 形态需要
            (它没有 adapter/build_inputs, 因而没有 bundle), torch 形态由 bundle 提供输入。
    """
    cfg = load_config(config_path)
    skip = set(skip)
    md = cfg.model_dir
    eff_device = cfg.runtime.device if device is None else device
    cfg.runtime.device = eff_device      # CLI --device 须对 source 加载/manifest 一致生效
    torch_npu.npu.set_device(eff_device)

    name = _export_name(cfg)
    base = work_dir or md
    air_dir = os.path.join(base, "air")
    om_dir = os.path.join(base, "om")
    bundle_dir = os.path.join(base, "verification")
    air_path = os.path.join(air_dir, f"{name}.air")
    io_spec_path = os.path.join(air_dir, f"{name}.io_spec.json")

    input_kwargs = dict(
        batch_size=cfg.inputs.batch_size if batch_size is None else batch_size,
        seq_len=cfg.inputs.seq_len if seq_len is None else seq_len,
        prefix_len=cfg.inputs.prefix_len,
    )

    # ---- ONNX 源: 原样, 无适配/golden ----
    if cfg.source.type == "onnx":
        # 形态③ 只能走离线后端: GeSession 的 Graph::LoadFromFile 不解析 ONNX。
        # 在读图/编译之前就拦 (配置期), 而不是等 C++ 运行期报错。
        if cfg.backend.type != "om_acl":
            raise ValueError(
                f"source.type=onnx 只支持 backend.type=om_acl (当前 {cfg.backend.type!r}): "
                "GeSession 在线后端只加载 GE 图 (.air/.pbtxt)。ONNX 请经 ATC "
                "(--framework=5) 编译为 OM 后执行 — docs §1/§9")
        graph = load_source(cfg, md)
        graph.io_spec.to_json(io_spec_path)
        env = PassManager(cfg.model.name, cfg.passes, _THIRD_PARTY).prepare() if "passes" not in skip else {}
        om = _compile(cfg, graph, env, base, skip)
        mp = write_manifest(cfg, graph.path, om, io_spec_path, None, base_dir=base, device=eff_device)

        # 形态③ 无 adapter/build_inputs → 没有 bundle (无 golden, 也无 .bin)。
        # run 走**部署态**: 由调用方用 --input logical:shape:file 给具体输入 (docs §5.5)。
        outputs_dir = default_output_dir(mp)
        if "run" not in skip:
            if not runtime_inputs:
                raise ValueError(
                    "ONNX 形态无 bundle, run 阶段需要具体输入: "
                    "--input logical:d0,d1,...:file.bin (逐输入, 可重复; "
                    f"io_spec 需要 {[n.logical for n in graph.io_spec.inputs]})")
            run_runtime(mp, output_dir=outputs_dir, device=eff_device, warmup=warmup,
                        bench=bench, extra=runtime_extra, inputs=runtime_inputs)
            print("[pipeline] ONNX 形态无 golden → 不做 compare (需自行比对, 如 onnxruntime 参考输出)")
        return {"graph": graph.path, "io_spec": io_spec_path, "om": om, "manifest": mp,
                "outputs": outputs_dir if "run" not in skip else None}

    # ---- torch 源 (name / 源码): source 加载 + adapter 适配 ----
    adapter = load_adapter(cfg)
    raw = load_source(cfg, md, dtype=dtype)
    # graph.dynamic.max_seq_len 是人工声明的唯一事实源 → 透传给 setup (RoPE 表/因果 mask
    # 等图常量据此定长; 不传则 adapter 用默认值, 与 yaml 声明脱节 → 长序列 Gather 越界)
    model = adapter.adapt(raw, max_seq_len=cfg.graph.dynamic.max_seq_len)
    exporter = GeExporter(adapter, air_dir, name)
    verify = Verifier()

    if "export" in skip:
        graph = Graph(kind="air", path=air_path, io_spec=IoSpec.from_json(io_spec_path))
        bundle_path = os.path.join(bundle_dir, "bundle.json")
        bundle_path = bundle_path if os.path.exists(bundle_path) else None
    else:
        inputs = adapter.build_inputs(model, **input_kwargs)
        want_golden = cfg.verify.enabled and "verify" not in skip
        golden = verify.golden(model, inputs) if want_golden else None   # trace 前算 (docs §10)
        air_path = exporter.trace(model, inputs, **input_kwargs)

        in_nodes = adapter.io_input_nodes(inputs, **input_kwargs)
        out_nodes = [_output_node(golden)]
        graph = Graph.from_air(air_path, in_nodes, out_nodes)
        graph.io_spec.to_json(io_spec_path)

        # bundle 总要落 (run 阶段需要具体 shape + .bin); golden 为 None (verify.enabled=false)
        # 时只写 inputs, bundle.golden=null → compare 阶段自动跳过
        prov = collect_provenance(seed=cfg.inputs.seed, model=cfg.model.name,
                                  soc=cfg.model.soc, dtype=str(dtype), **input_kwargs)
        bundle_path = verify.save_bundle(bundle_dir, inputs, golden, graph.io_spec, prov,
                                         logical_order=[n.logical for n in in_nodes])

    env = PassManager(cfg.model.name, cfg.passes, _THIRD_PARTY).prepare() if "passes" not in skip else {}
    om = _compile(cfg, graph, env, base, skip)
    mp = write_manifest(cfg, graph.path, om, io_spec_path, bundle_path, base_dir=base, device=eff_device)

    outputs_dir = default_output_dir(mp)
    if "run" not in skip:
        run_runtime(mp, output_dir=outputs_dir, device=eff_device,
                    warmup=warmup, bench=bench, extra=runtime_extra)

    report = None
    if "compare" not in skip and bundle_path:
        if not bundle_has_golden(bundle_path):
            print("[pipeline] bundle 无 golden (verify.enabled=false) → 跳过 compare")
        elif not os.path.exists(os.path.join(outputs_dir, "outputs.json")):
            print(f"[pipeline] 无运行时输出 ({outputs_dir}) → 跳过 compare")
        else:
            out_dtype = graph.io_spec.outputs[0].dtype if graph.io_spec.outputs else "float16"
            report = verify.compare_bundle(bundle_path, outputs_dir, dtype=out_dtype)

    return {"air": air_path, "io_spec": io_spec_path, "om": om,
            "bundle": bundle_path, "manifest": mp,
            "outputs": outputs_dir if "run" not in skip else None, "report": report}


def _output_node(golden):
    """从 golden 张量派生输出 IoNode (dim0=N 动态)。"""
    if golden is None:
        return IoNode(logical="logits", dtype="float16", shape=[-1], dynamic_dims=[0])
    return IoNode(logical="logits", dtype=str(golden.dtype).replace("torch.", ""),
                  shape=[-1] + list(golden.shape[1:]), dynamic_dims=[0])


def _compile(cfg, graph, env, base_dir, skip):
    """om_acl → ATC 编译出 om_path; ge_session → None (但仍校验 图形态×后端 组合)。"""
    if "compile" in skip:
        return None
    return Backend.from_config(cfg, base_dir=base_dir).compile(graph, env)


def main():
    p = argparse.ArgumentParser(description="GE 适配管线 (配置驱动)")
    p.add_argument("--config", required=True, help="model.yaml 路径")
    p.add_argument("--skip", default="", help="逗号分隔: export,passes,compile,run,compare")
    p.add_argument("--device", type=int, default=None, help="覆盖 cfg.runtime.device")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--work-dir", default=None, help="覆盖产物根目录 (默认 model_dir)")
    p.add_argument("--dtype", default="float16", help="float16|float32|bfloat16")
    p.add_argument("--warmup", type=int, default=0, help="C++ runtime 预热次数")
    p.add_argument("--bench", type=int, default=1, help="C++ runtime 计时执行次数")
    p.add_argument("--runtime-opt", action="append", default=[],
                   help="透传 C++ runtime 的选项 (可重复), 如 --runtime-opt --aicore_num=12")
    p.add_argument("--input", action="append", default=[], dest="inputs",
                   help="部署态输入 (可重复): logical:d0,d1,...:file.bin — ONNX 形态 run 阶段必需")
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)
    skip = tuple(s for s in args.skip.split(",") if s)
    out = run(args.config, skip=skip, dtype=dtype, device=args.device,
              batch_size=args.batch_size, seq_len=args.seq_len, work_dir=args.work_dir,
              warmup=args.warmup, bench=args.bench, runtime_extra=tuple(args.runtime_opt),
              runtime_inputs=tuple(args.inputs))
    print("=== 管线产物 ===")
    for k, v in out.items():
        if k != "report":
            print(f"  {k}: {v}")

    report = out.get("report")
    if report is None:
        return
    verdict = "PASS" if report.get("pass_overall") else "FAIL"
    print(f"  compare: {verdict} (cosine={report['cosine_similarity']:.8f}, "
          f"rel_l2={report['relative_l2_error']:.3e})")
    if verdict == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
