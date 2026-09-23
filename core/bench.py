"""性能测试编排 — scenario yaml → bench plan → ge_runtime → 报告落盘归档。

分工 (docs §3/§10): **C++ 只测量并出数据** (perf.json / perf_requests.csv), Python 负责
编排 (生成请求池 → 建 plan → 起运行时)、provenance、排版 (md) 与归档索引。

一次 run 的产物 (docs §10):
    <report.dir>/<run_id>/
        run.json          # 快照: git/CANN/torch_npu 版本、device、soc、scenario 全文、plan
        perf.json         # C++ 出的性能数据 (聚合 + 每实例 + 阶段耗时)
        perf.md           # 人读表
        perf_requests.csv # 逐请求明细 (gitignore)
        accuracy.json/md  # 精度 (单请求 + compare_bundle; 与性能分开跑, 避免 D2H 污染延迟)
        raw/              # 精度跑的输出 .bin 等 (gitignore)
    <report.dir>/index.json   # 历次 run 一行摘要 (趋势/归档)

用法:
    python3 -m core.bench --scenario models/qwen2.5-0.5b/bench/varlen.yaml --device 8
    python3 -m core.bench --scenario <yaml> --device 8 --instances 4 --requests 2000
"""

import argparse
import datetime
import json
import os
import subprocess
from dataclasses import dataclass, field

from core.backend import RUNTIME_BIN, run_runtime
from core.config import _setup_entries
from core.setup_scripts import run_scripts
from core.verify import Verifier, collect_provenance

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _jsonable(obj):
    """json.dump 的 default: compare 报告里有 numpy 标量/数组 (如 max_diff_position)。"""
    import numpy as np

    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return [_jsonable(x) for x in obj]
    return str(obj)


@dataclass
class Scenario:
    name: str
    manifest: str
    instances: int = 1
    requests: int = 100
    warmup: int = 10
    seed: int = 0
    inputs: dict = field(default_factory=dict)          # {mode: pool, dir: ...}
    generate: list = field(default_factory=list)        # list[SetupEntry] 生成请求池的脚本
    report_dir: str = ""
    accuracy: dict = field(default_factory=dict)        # {enabled, bundle, dtype}
    backend_options: dict = field(default_factory=dict)
    device: int = None
    soc: str = ""
    path: str = ""
    raw: dict = field(default_factory=dict)


def _resolve(path, base_dir=None):
    """绝对路径原样; 否则依次试 仓库根 / scenario 所在目录 / CWD。"""
    if not path or os.path.isabs(path):
        return path
    for root in (_REPO_ROOT, base_dir, os.getcwd()):
        if not root:
            continue
        cand = os.path.join(root, path)
        if os.path.exists(cand):
            return os.path.normpath(cand)
    return os.path.normpath(os.path.join(_REPO_ROOT, path))


def load_scenario(path, device=None, instances=None, requests=None, warmup=None,
                  manifest=None) -> Scenario:
    """读 scenario yaml → Scenario (CLI 覆盖 device/instances/requests)。"""
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    base_dir = os.path.dirname(os.path.abspath(path))

    inputs = dict(raw.get("inputs") or {})
    if inputs.get("dir"):
        inputs["dir"] = _resolve(inputs["dir"], base_dir)
    report = dict(raw.get("report") or {})
    accuracy = dict(raw.get("accuracy") or {})
    if accuracy.get("bundle"):
        accuracy["bundle"] = _resolve(accuracy["bundle"], base_dir)

    sc = Scenario(
        name=str(raw.get("scenario") or os.path.splitext(os.path.basename(path))[0]),
        manifest=_resolve(raw.get("manifest") or "", base_dir),
        instances=int(raw.get("instances", 1)),
        requests=int(raw.get("requests", 100)),
        warmup=int(raw.get("warmup", 10)),
        seed=int(raw.get("seed", 0)),
        inputs=inputs,
        generate=_setup_entries(raw.get("generate")),
        report_dir=_resolve(report.get("dir") or "results", base_dir),
        accuracy=accuracy,
        backend_options=dict(raw.get("backend_options") or {}),
        device=device if device is not None else raw.get("device"),
        soc=str(raw.get("soc") or ""),
        path=os.path.abspath(path),
        raw=raw,
    )
    if instances is not None:
        sc.instances = instances
    if requests is not None:
        sc.requests = requests
    if warmup is not None:
        sc.warmup = warmup
    if manifest:
        sc.manifest = _resolve(manifest, base_dir)

    if sc.device is None:
        raise ValueError("scenario 未指定 device 且 CLI 未传 --device (device 是运行期事实, 必填)")
    if not os.path.exists(sc.manifest):
        raise ValueError(f"manifest 不存在: {sc.manifest}")
    if sc.inputs.get("mode", "pool") != "pool":
        raise ValueError(f"暂只支持 inputs.mode=pool (收到 {sc.inputs.get('mode')!r}); "
                         f"固定输入直接用 ge_runtime <manifest> --bench N")
    if not sc.inputs.get("dir"):
        raise ValueError("scenario 缺 inputs.dir (请求池目录)")
    return sc


def make_run_id(scenario) -> str:
    """run_id = 时间戳-git短sha-scenario名 (归档目录名, 可按时间排序)。"""
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT,
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        sha = "nogit"
    return f"{ts}-{sha}-{scenario.name}"


def build_plan(scenario, run_dir, run_id) -> str:
    """生成 C++ 的 bench plan (JSON) — C++ 不读 yaml, 只吃这个 plan。"""
    plan = {
        "schema": "ge-bench-plan/1",
        "manifest": scenario.manifest,
        "device": scenario.device,
        "instances": scenario.instances,
        "requests": scenario.requests,
        "warmup": scenario.warmup,
        "seed": scenario.seed,
        "inputs": {"mode": "pool", "dir": scenario.inputs["dir"]},
        "backend_options": scenario.backend_options,
        "report": {
            "perf_json": os.path.join(run_dir, "perf.json"),
            "requests_csv": os.path.join(run_dir, "perf_requests.csv"),
            "run_id": run_id,
            "scenario": scenario.name,
        },
    }
    os.makedirs(run_dir, exist_ok=True)
    plan_path = os.path.join(run_dir, "plan.json")
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    return plan_path


def _form_str(form) -> str:
    """形态 dict → 一行可读串 (adapt_params 展开, 其余 key=value)。"""
    if not form:
        return "未知 (bundle 无 provenance)"
    parts = []
    for key, value in (form.get("adapt_params") or {}).items():
        parts.append(f"{key}={value}")
    parts += [f"{k}={v}" for k, v in form.items() if k != "adapt_params"]
    return " · ".join(parts)


def render_perf_md(perf, scenario, run_id, provenance) -> str:
    """perf.json → 人读 markdown (聚合 + 每实例 + 口径说明)。"""
    e2e = perf.get("e2e_ms") or {}
    hbm = perf.get("hbm_mb") or {}
    pool = perf.get("pool") or {}
    load = perf.get("load") or {}
    warm = perf.get("warmup") or {}
    lines = [
        f"# 性能报告 — {scenario.name}",
        "",
        f"- run_id: `{run_id}`",
        f"- backend: **{perf.get('backend')}** · device: {perf.get('device')} · "
        f"soc: {scenario.soc or '-'}",
        f"- manifest: `{perf.get('manifest')}`",
        f"- 负载: instances={load.get('instances')} · requests={load.get('requests')} · "
        f"warmup={load.get('warmup')} · seed={load.get('seed')}",
        f"- 请求池: `{pool.get('dir')}` — {pool.get('size')} 套 / "
        f"{pool.get('distinct_shapes')} 种 shape",
        f"- 模型形态: {_form_str(provenance.get('model_form'))}",
        f"- 环境: git `{provenance.get('git_commit', '-')}` · CANN `{provenance.get('cann', '-')}` · "
        f"torch_npu `{provenance.get('torch_npu', '-')}`",
        "",
        "## 聚合",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| wall | {perf.get('wall_ms', 0):.3f} ms |",
        f"| **QPS** | **{perf.get('qps', 0):.2f} req/s** |",
        f"| e2e avg / p50 / p99 / max | {e2e.get('avg', 0):.3f} / {e2e.get('p50', 0):.3f} / "
        f"{e2e.get('p99', 0):.3f} / {e2e.get('max', 0):.3f} ms |",
        f"| e2e min | {e2e.get('min', 0):.3f} ms |",
        f"| HBM (建实例前 → 后) | {hbm.get('base', -1):.0f} → {hbm.get('peak', -1):.0f} MB |",
        f"| warmup | {warm.get('runs')} 次 / {warm.get('ms', 0):.1f} ms |",
        f"| errors | {perf.get('errors', 0)} |",
        "",
        "## 每实例",
        "",
        "| instance | req | QPS | e2e avg | e2e p99 | exec avg | exec p99 | h2d avg | desc avg "
        "| load ms | shapes | 特化 ms | err |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for ip in perf.get("instances") or []:
        ie2e, iexec = ip.get("e2e_ms") or {}, ip.get("exec_ms") or {}
        lines.append(
            f"| {ip.get('name')} | {ip.get('requests')} | {ip.get('qps', 0):.2f} "
            f"| {ie2e.get('avg', 0):.3f} | {ie2e.get('p99', 0):.3f} "
            f"| {iexec.get('avg', 0):.3f} | {iexec.get('p99', 0):.3f} "
            f"| {(ip.get('h2d_ms') or {}).get('avg', 0):.3f} "
            f"| {(ip.get('desc_ms') or {}).get('avg', 0):.3f} "
            f"| {ip.get('load_ms', 0):.1f} | {ip.get('distinct_shapes')} "
            f"| {ip.get('specialize_ms', 0):.1f} | {ip.get('errors', 0)} |")
    lines += [
        "",
        "## 口径",
        "",
        "- `e2e` = h2d + desc(重设该请求 shape) + execute+sync；`exec` 只含 execute+sync。",
        "- 每实例 = 一份独立加载的模型（ACL 每实例独立 `aclmdlLoadFromFile`；GeSession 单 "
        "Session 多图，每实例一份 `CompileGraph`+`LoadGraph`），1 worker ↔ 1 实例，无锁。",
        "- `特化` = 该实例在 **warmup 段**首次命中各 shape 的 exec 耗时之和（ACL 是 tiling 缓存"
        "建立，GeSession 还含图特化，两者都显著）；warmup 覆盖 每实例 × 每 shape，故测量段"
        "不再付这笔 —— 这也意味着**请求池的 distinct shape 数直接决定 warmup 成本**。",
        "- 测量段若出现 warmup 未覆盖的 shape，csv 里 `first_hit=1` 且 stderr 会 WARN"
        "（说明延迟被特化污染）。",
        "- `load ms` = 该实例的加载耗时；**GeSession 口径含全部图实例的 `CompileGraph`**"
        "（串行、一次性，~10s/份），ACL 口径是该实例的 `aclmdlLoadFromFile`+缓冲分配。",
        "- 请求池由模型侧脚本按分布生成（语义自洽由模型侧保证），各实例用 `seed + instance_id` "
        "独立随机抽样，可复现。",
        "- 精度不在本报告内（见 `accuracy.md`）：性能跑不落盘输出，避免 D2H 污染延迟。",
        "",
    ]
    return "\n".join(lines)


def render_accuracy_md(report, case) -> str:
    gate = case.get("gate") or {}
    lines = [
        "# 精度报告",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 输出 | {case.get('logical')} |",
        f"| shape | {case.get('shape')} |",
        f"| dtype | {case.get('dtype')} |",
        f"| golden | `{case.get('golden')}` |",
        f"| cosine | {report.get('cosine_similarity', 0):.8f} |",
        f"| relative_l2 | {report.get('relative_l2_error', 0):.6e} |",
        f"| max_abs | {report.get('max_abs_error', 0):.6e} |",
        f"| 门限 | cosine > {gate.get('cosine_min')} 且 rel_l2 < {gate.get('rel_l2_max')} |",
        f"| **判定** | **{'PASS' if report.get('pass_overall') else 'FAIL'}** |",
        "",
        "> 单请求、与性能跑分开（性能跑不落盘输出）。golden = NPU-eager，比对口径见 "
        "`tools/compare.py`；shape 不一致直接判失败，不做 flatten/截断。",
        "",
    ]
    return "\n".join(lines)


FORM_KEYS = ("adapt_params", "seed", "batch_size", "seq_len", "prefix_len", "dtype")


def form_from_manifest(manifest_path) -> dict:
    """从 manifest 指向的 bundle.provenance 取**形态**信息, 让性能报告也能自证是哪种形态。

    形态的事实源是 bundle 的 provenance (pipeline 写入, 含 adapt_params = prefix/prune 等开关);
    bench 只有 manifest, 顺着它读回来即可, 不必再解析 model.yaml。取不到返回 {}。
    """
    try:
        manifest = json.load(open(manifest_path))
        if not manifest.get("bundle"):
            return {}
        base = os.path.dirname(os.path.dirname(os.path.abspath(manifest_path)))
        prov = json.load(open(os.path.join(base, manifest["bundle"]))).get("provenance") or {}
        return {k: prov[k] for k in FORM_KEYS if k in prov}
    except (OSError, ValueError):
        return {}


def _io_spec_dtype(manifest_path, manifest):
    try:
        base = os.path.dirname(os.path.dirname(os.path.abspath(manifest_path)))
        spec = json.load(open(os.path.join(base, manifest["io_spec"])))
        return (spec.get("outputs") or [{}])[0].get("dtype") or "float16"
    except (OSError, ValueError, KeyError, IndexError):
        return "float16"


def run_accuracy(scenario, run_dir) -> dict:
    """单请求跑一次 + compare_bundle → {case, report} (未启用/无 golden 返回 {})。"""
    if not scenario.accuracy.get("enabled", True):
        return {}
    manifest = json.load(open(scenario.manifest))
    base = os.path.dirname(os.path.dirname(os.path.abspath(scenario.manifest)))
    bundle = scenario.accuracy.get("bundle") or (
        os.path.join(base, manifest["bundle"]) if manifest.get("bundle") else None)
    if not bundle or not os.path.exists(bundle):
        print(f"[bench] 无 golden bundle ({bundle}) → 跳过精度")
        return {}

    out_dir = os.path.join(run_dir, "raw", "accuracy_outputs")
    run_runtime(scenario.manifest, output_dir=out_dir, device=scenario.device, warmup=0, bench=1)
    report = Verifier().compare_bundle(
        bundle, out_dir, dtype=scenario.accuracy.get("dtype")
        or _io_spec_dtype(scenario.manifest, manifest), verbose=False)

    entries = json.load(open(os.path.join(out_dir, "outputs.json")))["outputs"]
    case = {"logical": entries[0].get("logical"), "shape": entries[0].get("shape"),
            "dtype": entries[0].get("dtype"), "golden": bundle,
            "gate": {"cosine_min": 0.9999, "rel_l2_max": 0.01}}
    verdict = "PASS" if report.get("pass_overall") else "FAIL"
    print(f"[bench] 精度 {verdict}: cosine={report['cosine_similarity']:.8f} "
          f"rel_l2={report['relative_l2_error']:.3e}")
    return {"case": case, "report": report}


def update_index(report_dir, entry) -> str:
    """results/index.json 追加一行摘要 (趋势/归档索引)。"""
    os.makedirs(report_dir, exist_ok=True)
    index_path = os.path.join(report_dir, "index.json")
    runs = []
    if os.path.exists(index_path):
        try:
            runs = json.load(open(index_path)).get("runs") or []
        except (OSError, ValueError):
            runs = []
    runs.append(entry)
    with open(index_path, "w") as f:
        json.dump({"schema": "ge-bench-index/1", "runs": runs}, f, indent=2, ensure_ascii=False,
                  default=_jsonable)
    return index_path


def run(scenario_path, device=None, instances=None, requests=None, warmup=None, manifest=None,
        skip_accuracy=False) -> str:
    """跑一次性能测试, 返回 run 目录。"""
    sc = load_scenario(scenario_path, device=device, instances=instances, requests=requests,
                       warmup=warmup, manifest=manifest)
    if not os.path.exists(RUNTIME_BIN):
        raise FileNotFoundError(f"C++ 运行时未构建: {RUNTIME_BIN} (先执行 bash runtime/build.sh)")

    # 1. 请求池 (可选: 先跑生成脚本; 与 pass/算子同一套 {path, script, args} 接口)
    if sc.generate:
        pool_dir = sc.inputs["dir"]
        os.makedirs(pool_dir, exist_ok=True)
        entries = []
        for g in sc.generate:
            args = list(g.args)
            if "--out" not in args:
                args += ["--out", pool_dir]
            entries.append(type(g)(script=g.script, path=g.path, args=args))
        run_scripts(entries, "generate", model_dir=os.path.dirname(sc.path))
    if not os.path.isdir(sc.inputs["dir"]):
        raise FileNotFoundError(f"请求池目录不存在: {sc.inputs['dir']} (配 generate 或手动生成)")

    # 2. plan → ge_runtime (C++ 只测量并出数据)
    run_id = make_run_id(sc)
    run_dir = os.path.join(sc.report_dir, run_id)
    plan_path = build_plan(sc, run_dir, run_id)
    print(f"=== 性能测试 {run_id} ===\n  plan: {plan_path}\n")
    subprocess.run([RUNTIME_BIN, "--bench-plan", plan_path], check=True)

    perf = json.load(open(os.path.join(run_dir, "perf.json")))

    # 3. provenance + 报告
    provenance = collect_provenance(model=sc.name, soc=sc.soc or None)
    provenance["cann"] = os.path.basename(os.environ.get("ASCEND_HOME_PATH", "")) or None
    model_form = form_from_manifest(sc.manifest)
    if model_form:
        provenance["model_form"] = model_form      # 形态自证: prefix/prune/seed/shape 等
    else:
        print("[bench][WARN] 未能从 bundle.provenance 取到形态信息 (manifest 无 bundle?) — "
              "本报告无法自证是哪种形态跑出来的")
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        json.dump({"schema": "ge-bench-run/1", "run_id": run_id, "scenario": sc.raw,
                   "scenario_path": sc.path, "provenance": provenance,
                   "perf_summary": {k: perf.get(k) for k in
                                    ("backend", "device", "qps", "wall_ms", "e2e_ms",
                                     "requests", "errors", "distinct_shapes")}},
                  f, indent=2, ensure_ascii=False, default=_jsonable)
    with open(os.path.join(run_dir, "perf.md"), "w") as f:
        f.write(render_perf_md(perf, sc, run_id, provenance))

    # 4. 精度 (与性能分开跑: 性能跑不落盘输出, 避免 D2H 污染延迟)
    accuracy = {} if skip_accuracy else run_accuracy(sc, run_dir)
    if accuracy:
        with open(os.path.join(run_dir, "accuracy.json"), "w") as f:
            json.dump({"schema": "ge-accuracy/1", "run_id": run_id, **accuracy},
                      f, indent=2, ensure_ascii=False, default=_jsonable)
        with open(os.path.join(run_dir, "accuracy.md"), "w") as f:
            f.write(render_accuracy_md(accuracy["report"], accuracy["case"]))

    # 5. 归档索引
    e2e = perf.get("e2e_ms") or {}
    update_index(sc.report_dir, {
        "run_id": run_id, "ts": provenance.get("timestamp"), "scenario": sc.name,
        "backend": perf.get("backend"), "device": perf.get("device"),
        "instances": sc.instances, "requests": perf.get("requests"),
        "qps": perf.get("qps"), "e2e_avg_ms": e2e.get("avg"), "e2e_p99_ms": e2e.get("p99"),
        "errors": perf.get("errors"), "distinct_shapes": perf.get("distinct_shapes"),
        "model_form": model_form or None,
        "accuracy": (accuracy.get("report") or {}).get("pass_overall") if accuracy else None,
        "git_commit": provenance.get("git_commit"), "dir": run_dir,
    })

    print(f"\n=== 报告已归档: {run_dir} ===")
    print("  perf.json / perf.md / perf_requests.csv / run.json"
          + (" / accuracy.json+md" if accuracy else ""))
    if perf.get("errors", 0):
        raise SystemExit(f"[bench] 有 {perf['errors']} 个请求失败, 见 {run_dir}")
    return run_dir


def main():
    p = argparse.ArgumentParser(description="GE 性能测试 (多实例 + 请求池回放)")
    p.add_argument("--scenario", required=True, help="scenario yaml 路径")
    p.add_argument("--device", type=int, default=None, help="NPU 设备号 (必填, 除非 yaml 里写了)")
    p.add_argument("--instances", type=int, default=None, help="覆盖 scenario.instances")
    p.add_argument("--requests", type=int, default=None, help="覆盖 scenario.requests")
    p.add_argument("--warmup", type=int, default=None, help="覆盖 scenario.warmup")
    p.add_argument("--manifest", default=None,
                   help="覆盖 scenario.manifest (如产物在 --work-dir 下而非模型目录)")
    p.add_argument("--skip-accuracy", action="store_true", help="只测性能, 不跑精度")
    args = p.parse_args()
    run(args.scenario, device=args.device, instances=args.instances,
        requests=args.requests, warmup=args.warmup, manifest=args.manifest,
        skip_accuracy=args.skip_accuracy)


if __name__ == "__main__":
    main()
