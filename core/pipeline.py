"""GE 适配管线编排 — 读 model.yaml, 串起 source→adapt→export→io_spec→golden→bundle
→passes→compile→manifest。docs/architecture.md §3/§10/§11。

逐模型只剩 model.py(Adapter) + config/model.yaml; 本管线通用、配置驱动。

用法:
    python -m core.pipeline --config models/qwen2.5-0.5b/config/model.yaml
    python -m core.pipeline --config <yaml> --skip compile,run,compare
    # 冒烟/调试可覆盖: --device 6 (必填) --batch-size 2 --seq-len 16 --work-dir /tmp/x
    # ops/passes 是**用户脚本**接口 (model.yaml 的 custom_ops/passes 填脚本路径, 见 core/setup_scripts.py)
"""

import argparse
import glob
import os

import torch
import torch_npu

from core.config import export_name, load_config, load_adapter, write_manifest
from core.source import load_source
from core.graph import Graph, IoNode, IoSpec
from core.exporter import GeExporter
from core.setup_scripts import run_scripts
from core.backend import backend_extra, compile_graph, default_output_dir, run_runtime
from core.verify import Verifier, bundle_has_golden, collect_provenance


def run(config_path, skip=(), dtype=torch.float16, device=None,
        batch_size=None, seq_len=None, work_dir=None,
        warmup=0, bench=1, runtime_extra=()):
    """跑管线, 返回产物路径 dict。skip ⊂ {ops,export,passes,compile,run,compare,reference}。

    run/compare 为阶段二闭环: run 调 C++ runtime (manifest 驱动, OM/ACL 或 GeSession),
    compare 用 bundle golden 比对运行时输出 (docs §10)。

    reference (只在 export 阶段, 与 golden 同时算): **原版未 patch 的 HF** 逐请求前向,
    与 patched-eager golden 比对 — 隔离出"适配"这一个变量 (compare 隔离的是"编译")。
    adapter 未实现 unpack_requests 时自动跳过并 WARN。
    """
    cfg = load_config(config_path)
    skip = set(skip)
    md = cfg.model_dir
    if device is None:
        raise ValueError("必须指定 device (CLI --device): 用哪张卡是运行期事实, 不进 model.yaml")
    torch_npu.npu.set_device(device)
    print(f"[pipeline] device={device}")

    name = export_name(cfg)
    base = work_dir or md
    air_dir = os.path.join(base, "air")
    om_dir = os.path.join(base, "om")
    bundle_dir = os.path.join(base, "io")
    air_path = os.path.join(air_dir, f"{name}.air")
    io_spec_path = os.path.join(air_dir, f"{name}.io_spec.json")

    input_kwargs = dict(
        batch_size=cfg.inputs.batch_size if batch_size is None else batch_size,
        seq_len=cfg.inputs.seq_len if seq_len is None else seq_len,
        prefix_len=cfg.inputs.prefix_len,
        seed=cfg.inputs.seed,
    )

    # ---- 自定义算子安装脚本: 必须在 load_adapter 之前 (model.py 可能 import 算子绑定) ----
    if "ops" not in skip:
        run_scripts(cfg.custom_ops, "custom_ops")

    # ---- torch 源: source 加载 + adapter 适配 ----
    adapter = load_adapter(cfg)
    exporter = GeExporter(adapter, air_dir, name)
    # verify 段的门限覆盖透传给 Verifier; None (未覆盖) 的项过滤掉, 让框架默认生效
    # (直接传 None 会覆盖掉 Verifier 的常量默认 → 比对门限变成 None 而崩)
    verify = Verifier(**{k: v for k, v in (
        ("ref_cosine_min", cfg.verify.ref_cosine_min),
        ("ref_rel_l2_max", cfg.verify.ref_rel_l2_max),
        ("cmp_cosine_min", cfg.verify.cmp_cosine_min),
        ("cmp_rel_l2_max", cfg.verify.cmp_rel_l2_max),
    ) if v is not None})
    ref_report = None

    if "export" in skip:
        graph = Graph(path=air_path, io_spec=IoSpec.from_json(io_spec_path))
        bundle_path = os.path.join(bundle_dir, "bundle.json")
        bundle_path = bundle_path if os.path.exists(bundle_path) else None
    else:
        raw = load_source(cfg, md, dtype=dtype, device=device)
        inputs = adapter.build_inputs(raw, **input_kwargs)
        want_golden = cfg.verify.enabled and "verify" not in skip

        # 原版参考必须在 adapt **之前**算: patch 是类级 monkey-patch (进程全局), adapt 之后
        # 同进程里任何同架构实例都走 patched forward, "原版"就名存实亡 (docs §10)
        reference = (verify.reference(raw, adapter, inputs)
                     if want_golden and "reference" not in skip else None)

        # graph.dynamic.max_seq_len 是人工声明的唯一事实源 → 透传给 setup (RoPE 表/因果 mask
        # 等图常量据此定长; 不传则 adapter 用默认值, 与 yaml 声明脱节 → 长序列 Gather 越界)
        model = adapter.adapt(raw, max_seq_len=cfg.graph.dynamic.max_seq_len)
        golden = verify.golden(model, inputs) if want_golden else None   # trace 前算 (docs §10)
        if reference is not None:
            ref_report = verify.compare_reference(
                golden, reference, path=os.path.join(bundle_dir, "reference.json"))
        air_path = exporter.trace(model, inputs, **input_kwargs)

        in_nodes = adapter.io_input_nodes(inputs, **input_kwargs)
        out_nodes = [_output_node(golden)]
        graph = Graph.from_air(air_path, in_nodes, out_nodes)
        graph.io_spec.to_json(io_spec_path)

        # bundle 总要落 (run 阶段需要具体 shape + .bin); golden 为 None (verify.enabled=false)
        # 时只写 inputs, bundle.golden=null → compare 阶段自动跳过
        # adapt.params = 形态事实源 (prefix/prune…): 记进 provenance, 报告才能自证形态
        prov = collect_provenance(model=cfg.model.name,
                                  soc=cfg.model.soc, dtype=str(dtype),
                                  adapt_params=dict(cfg.adapt.params), **input_kwargs)
        bundle_path = verify.save_bundle(bundle_dir, inputs, golden, graph.io_spec, prov,
                                         logical_order=[n.logical for n in in_nodes])

    if "passes" not in skip:
        run_scripts(cfg.passes, "passes")
    om = _compile(cfg, graph, base, skip)
    mp = write_manifest(cfg, graph.path, om, io_spec_path, bundle_path, base_dir=base, device=device)

    outputs_dir = default_output_dir(mp)
    if "run" not in skip:
        run_runtime(mp, output_dir=outputs_dir, device=device,
                    warmup=warmup, bench=bench,
                    extra=backend_extra(cfg) + tuple(runtime_extra))

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
            "outputs": outputs_dir if "run" not in skip else None,
            "reference": ref_report, "report": report}


def _output_node(golden):
    """从 golden 张量派生输出 IoNode (dim0=N 动态)。"""
    if golden is None:
        return IoNode(logical="logits", dtype="float16", shape=[-1], dynamic_dims=[0])
    return IoNode(logical="logits", dtype=str(golden.dtype).replace("torch.", ""),
                  shape=[-1] + list(golden.shape[1:]), dynamic_dims=[0])


def _compile(cfg, graph, base_dir, skip):
    """om_acl → ATC 编译出 om_path; ge_session → None (但仍校验 图形态×后端 组合)。

    --skip compile 时**复用磁盘上已有的 OM** (否则 manifest.om_path 会是 null, run 阶段
    直接失败 — 而"复用 AIR/OM 只跑 runtime"正是 --skip compile 的用途)。
    """
    if "compile" in skip:
        if cfg.backend.type != "om_acl":
            return None
        om = _existing_om(base_dir, graph, cfg.backend.aicore_num)
        if om is None:
            print(f"[pipeline][WARN] --skip compile 但 {os.path.join(base_dir, 'om')} 下没有匹配的 OM "
                  f"→ manifest.om_path 为空, run 阶段会失败 (去掉 --skip compile 先编译)")
        else:
            print(f"[pipeline] --skip compile: 复用已有 OM {om}")
        return om
    return compile_graph(cfg, graph, base_dir=base_dir)


def _existing_om(base_dir, graph, aicore_num):
    """找 <base_dir>/om/<图名>[限核后缀]*.om (ATC 会追加 _linux_aarch64 之类后缀)。"""
    from tools.atc_utils import normalize_aicore

    stem = os.path.splitext(os.path.basename(graph.path))[0]
    _, suffix = normalize_aicore(aicore_num)
    pattern = os.path.join(base_dir, "om", stem + suffix + "*.om")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def main():
    p = argparse.ArgumentParser(description="GE 适配管线 (配置驱动)")
    p.add_argument("--config", required=True, help="model.yaml 路径")
    p.add_argument("--skip", default="",
                   help="逗号分隔: ops,export,passes,compile,run,compare,reference")
    p.add_argument("--device", type=int, required=True,
                   help="NPU 设备号 (必填; 运行期事实, 不进 model.yaml)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--work-dir", default=None, help="覆盖产物根目录 (默认 model_dir)")
    p.add_argument("--dtype", default="float16", help="float16|float32|bfloat16")
    p.add_argument("--warmup", type=int, default=0, help="C++ runtime 预热次数")
    p.add_argument("--bench", type=int, default=1, help="C++ runtime 计时执行次数")
    p.add_argument("--runtime-opt", action="append", default=[],
                   help="透传 C++ runtime 的选项 (可重复), 如 --runtime-opt --aicore_num=12")
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)
    skip = tuple(s for s in args.skip.split(",") if s)
    out = run(args.config, skip=skip, dtype=dtype, device=args.device,
              batch_size=args.batch_size, seq_len=args.seq_len, work_dir=args.work_dir,
              warmup=args.warmup, bench=args.bench, runtime_extra=tuple(args.runtime_opt))
    print("=== 管线产物 ===")
    for k, v in out.items():
        if k not in ("report", "reference"):
            print(f"  {k}: {v}")

    # 两道门, 各隔离一个变量 (docs §10): reference = 适配是否正确, compare = 编译是否正确
    failed = []
    for key, label in (("reference", "reference (原版 HF vs 适配后 eager)"),
                       ("report", "compare  (运行时输出 vs golden)")):
        rep = out.get(key)
        if rep is None:
            continue
        verdict = "PASS" if rep.get("pass_overall") else "FAIL"
        print(f"  {label}: {verdict} (cosine={rep['cosine_similarity']:.8f}, "
              f"rel_l2={rep['relative_l2_error']:.3e})")
        if verdict == "FAIL":
            failed.append(label)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
