"""性能测试编排 — model.yaml 的 bench 段 → bench plan → ge_runtime → 报告落盘。

分工 (docs §3/§10): **C++ 只测量并出数据** (perf.json / perf_requests.csv), Python 负责
编排 (生成请求池 → 建 plan → 起运行时)、provenance、排版 (md) 与本地索引。

配置只有**一份** model.yaml (docs §5.2): bench 段放"这次压测怎么压"(并发/请求数/池套数),
模型侧事实 (soc / manifest / 报告目录 / 限核 / 形态参数) 一律从同一份配置取, 不在两处重复。
派生路径按约定: manifest=<model_dir>/io/manifest.json, 池=<model_dir>/io/pool,
报告=<model_dir>/results。

一次 run 的产物 (docs §10) — **全是本地过程产物, 整个 results/<run_id>/ 与 index.json 都
gitignored、可重跑再生**; 要长期保留的性能数字由人工 curate 进 results/README.md (唯一入库项):
    <model_dir>/results/<run_id>/
        run.json          # 快照: git/CANN/torch_npu 版本、device、soc、model.yaml 全文、plan
        perf.json         # C++ 出的性能数据 (聚合 + 每实例 + 阶段耗时)
        perf.md           # 人读表 (curate 基线时的素材)
        perf_requests.csv # 逐请求明细
        plan.json         # 传给 C++ 的 bench plan
    <model_dir>/results/index.json   # 历次 run 一行摘要 (本地趋势)

**不含精度**: 性能跑不落盘输出 (D2H 会污染延迟), 精度由 `run.sh` 的两道门负责
(`io/reference.json` 门① + compare 门②, docs §10) — 一个变量只由一处度量。

用法:
    python3 -m core.bench --config models/qwen2.5-0.5b/config/model.yaml --device 8
    python3 -m core.bench --config <model.yaml> --device 8 --instances 4 --requests 2000
"""

import argparse
import datetime
import json
import os
import subprocess
from dataclasses import dataclass, field

from core.backend import RUNTIME_BIN
from core.config import _setup_entries, load_config
from core.setup_scripts import resolve_path, run_scripts
from core.verify import collect_provenance

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Scenario:
    """一次性能测试的**有效配置** = model.yaml 的 bench 段 + 派生约定 + CLI 覆盖。"""
    name: str
    manifest: str
    model_dir: str = ""
    instances: int = 1
    requests: int = 100
    warmup: int = 10
    seed: int = 0
    inputs: dict = field(default_factory=dict)          # {mode: pool, dir: ...}
    generate: list = field(default_factory=list)        # list[SetupEntry] 生成请求池的脚本
    report_dir: str = ""
    backend_options: dict = field(default_factory=dict)
    device: int = None
    soc: str = ""
    path: str = ""                                      # model.yaml 路径 (run.json 快照溯源)
    raw: dict = field(default_factory=dict)             # model.yaml 全文 (run.json 快照)


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


def _strip_flag(args, flag):
    """从 argv 列表移除一个 flag 及其值 (兼容 `--flag X` 与 `--flag=X` 两种写法), 返回新列表。

    纯函数。用于 _form_args 在非 prefix 形态下移除 yaml 里手写的 --prefix —— 范围值属
    **负载口径**(留在 args), 但 flag 的**存在与否**由 prefix 标志管辖 (见 _form_args)。
    `--flag=` 前缀匹配不会误伤 `--flag-other` (后者不以 `--flag=` 开头, 也不 == `--flag`)。
    """
    out, skip = [], False
    for a in args:
        if skip:                        # 上一个是被删的 `--flag`, 这个是它的值 → 一并删
            skip = False
            continue
        if a == flag:                   # `--flag X`: 删 flag, 下一个 (值) 也删
            skip = True
            continue
        if a.startswith(flag + "="):    # `--flag=X`: 整个删
            continue
        out.append(a)
    return out


def _form_args(cfg, args):
    """把**形态事实**拼进请求池生成脚本的 args (负载口径留在 yaml / 脚本默认值里)。

    这三个量若在 yaml 里再抄一遍就会静默错配 (池与图接口对不上, 或报告口径与声明不符),
    所以由框架从 model.yaml 注入:
      --batch         prefix 形态下 act 是**静态** [batch+1] (io_spec 无动态维), 池里每套
                      请求的条数被图烙死, 必须 = inputs.batch_size
      --prune-tokens  lm_head 剪裁宽 = 图输出宽; 抄漏会让 golden shape 与图输出对不上
      --prefix        prefix 形态必须是 packed 布局; **范围**属负载口径 (yaml 可写 "20-25"),
                      没写就用 inputs.prefix_len 保底
    yaml 的 args 里已显式写过的不覆盖 (便于临时实验)。

    --prefix 的**存在与否**由 prefix 标志管辖 (范围值仍留在 args): prefix=true 注入/保留,
    prefix=false 则**移除** yaml 里手写的 --prefix —— 这样翻 adapt.params.prefix 一个开关
    即可无缝切形态, 不必同时手改 bench.pool.args (否则非 prefix 形态会生成 prefix-packed 池,
    与基线图接口对不上)。
    """
    out = list(args)

    def add(flag, value):
        if value is not None and flag not in out:
            out.extend([flag, str(value)])      # extend 而非 +=: 闭包里 += 会让 out 变成局部名

    add("--batch", cfg.inputs.batch_size)
    prune = cfg.adapt.params.get("prune_token_file")
    if prune:
        # 路径规则与 load_adapter 一致 (绝对 / 相对仓库根); 传给脚本的是绝对路径, 免依赖 CWD
        resolved = resolve_path(prune, is_file=True)
        if resolved is None:
            raise FileNotFoundError(
                f"adapt.params.prune_token_file 找不到: {prune!r} — 只认**绝对路径**或"
                f"**相对仓库根**的写法, 如 models/{cfg.model.name}/config/target_tokens.json")
        add("--prune-tokens", resolved)
    if cfg.adapt.params.get("prefix"):
        add("--prefix", cfg.inputs.prefix_len or None)
    else:
        out = _strip_flag(out, "--prefix")
        assert not any(a == "--prefix" or a.startswith("--prefix=") for a in out), \
            f"_strip_flag 未彻底移除 --prefix: {out}"
    return out


def load_bench(config_path, device=None, instances=None, requests=None, warmup=None,
               manifest=None) -> Scenario:
    """读 model.yaml → Scenario (bench 段 + 派生约定; CLI 覆盖 device/instances/requests/warmup)。

    派生路径按约定 (与 core/config.write_manifest、core/backend.default_output_dir 同一套):
        manifest  <model_dir>/io/manifest.json
        请求池     <model_dir>/io/pool          (跨 run 复用的缓存, 幂等靠脚本的 pool_meta.json)
        报告       <model_dir>/results
    """
    import yaml

    cfg = load_config(config_path)
    b = cfg.bench
    md = cfg.model_dir
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    generate = []
    if b.pool:
        generate = [_setup_entries([{"script": b.pool.script, "path": b.pool.path,
                                     "args": _form_args(cfg, b.pool.args)}])[0]]

    # 限核的单一事实源是 backend.aicore_num: om_acl 在 ATC 编译期 (已烙进 OM), ge_session
    # 在运行期 → 经 plan.json 的 backend_options 传给 C++ (与 pipeline 的 backend_extra 同源)。
    # 输出缓冲不在此声明: 池模式恒按池内最大 golden shape 分配 (acl_backend.cpp), C++ 侧的
    # output_reserve_mb 默认值与 --output_reserve CLI (部署态) 都还在, 需要时再加回这个键。
    backend_options = {}
    if cfg.backend.type == "ge_session" and cfg.backend.aicore_num:
        backend_options["aicore_num"] = str(cfg.backend.aicore_num)

    sc = Scenario(
        name=f"{cfg.model.name}-bench",   # 归档名/run_id 后缀; 一个模型一个场景, 不必再声明
        manifest=os.path.join(md, "io", "manifest.json"),
        model_dir=md,
        instances=b.instances,
        requests=b.requests,
        warmup=b.warmup,
        seed=b.sample_seed,
        inputs={"mode": "pool", "dir": os.path.join(md, "io", "pool")},
        generate=generate,
        report_dir=os.path.join(md, "results"),
        backend_options=backend_options,
        device=device,
        soc=cfg.model.soc,
        path=os.path.abspath(config_path),
        raw=raw,
    )
    if instances is not None:
        sc.instances = instances
    if requests is not None:
        sc.requests = requests
    if warmup is not None:
        sc.warmup = warmup
    if manifest:
        sc.manifest = _resolve(manifest, md)

    if sc.device is None:
        raise ValueError("未指定 device (CLI --device): 用哪张卡是运行期事实, 不进 model.yaml")
    if not os.path.exists(sc.manifest):
        raise ValueError(f"manifest 不存在: {sc.manifest} — 先跑管线产出 "
                         f"(./run.sh --device {sc.device})")
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
    execm = perf.get("exec_ms") or {}
    toks = perf.get("tokens") or {}
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
        f"| exec avg / p99 | {execm.get('avg', 0):.3f} / {execm.get('p99', 0):.3f} ms "
        f"(execute+sync, 不含 h2d/desc) |",
        f"| tokens avg / p50 / p99 / max | {toks.get('avg', 0):.1f} / {toks.get('p50', 0):.0f} / "
        f"{toks.get('p99', 0):.0f} / {toks.get('max', 0):.0f} (每请求 packed T) |",
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
        "- `tokens` = 每请求的 packed 总 token 数 T（一个请求 = N 条序列拼成的 varlen 输入，"
        "T = input_ids 长度，非单条序列长）；avg/p50/p99/max 是运行时**实际抽样到**的分布"
        "（池级 min/median/max 见 `io/pool/pool_meta.json`，那是生成期全集、非抽样）。",
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
        "- 精度不在本报告内：性能跑不落盘输出（避免 D2H 污染延迟），精度由 `run.sh` 的两道门"
        "负责（`io/reference.json` + 门② compare，docs §10）。",
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
        json.dump({"schema": "ge-bench-index/1", "runs": runs}, f, indent=2, ensure_ascii=False)
    return index_path


def run(config_path, device=None, instances=None, requests=None, warmup=None,
        manifest=None) -> str:
    """跑一次性能测试 (只出性能数据; 精度归 run.sh 的两道门), 返回 run 目录。"""
    sc = load_bench(config_path, device=device, instances=instances, requests=requests,
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
        run_scripts(entries, "generate")
    if not os.path.isdir(sc.inputs["dir"]):
        raise FileNotFoundError(f"请求池目录不存在: {sc.inputs['dir']} (配 bench.pool 或手动生成)")

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
        json.dump({"schema": "ge-bench-run/1", "run_id": run_id, "config": sc.raw,
                   "config_path": sc.path, "provenance": provenance,
                   "perf_summary": {k: perf.get(k) for k in
                                    ("backend", "device", "qps", "wall_ms", "e2e_ms", "exec_ms",
                                     "tokens", "requests", "errors", "distinct_shapes")}},
                   f, indent=2, ensure_ascii=False)
    with open(os.path.join(run_dir, "perf.md"), "w") as f:
        f.write(render_perf_md(perf, sc, run_id, provenance))

    # 4. 归档索引 (精度不在此: 性能跑不落盘输出, 精度由 run.sh 的两道门负责 — docs §10)
    e2e = perf.get("e2e_ms") or {}
    update_index(sc.report_dir, {
        "run_id": run_id, "ts": provenance.get("timestamp"), "scenario": sc.name,
        "backend": perf.get("backend"), "device": perf.get("device"),
        "instances": sc.instances, "requests": perf.get("requests"),
        "qps": perf.get("qps"), "e2e_avg_ms": e2e.get("avg"), "e2e_p99_ms": e2e.get("p99"),
        "errors": perf.get("errors"), "distinct_shapes": perf.get("distinct_shapes"),
        "model_form": model_form or None,
        "git_commit": provenance.get("git_commit"), "dir": run_dir,
    })

    print(f"\n=== 报告已落盘 (本地过程产物, 不入库): {run_dir} ===")
    print("  perf.json / perf.md / perf_requests.csv / run.json")
    print("  满意后把 perf.md 的数字 + provenance curate 进 results/README.md 的「当前基线」(唯一入库项)")
    if perf.get("errors", 0):
        raise SystemExit(f"[bench] 有 {perf['errors']} 个请求失败, 见 {run_dir}")
    return run_dir


def main():
    p = argparse.ArgumentParser(description="GE 性能测试 (多实例 + 请求池回放)")
    p.add_argument("--config", required=True, help="model.yaml 路径 (bench 段 = 压测口径)")
    p.add_argument("--device", type=int, required=True,
                   help="NPU 设备号 (必填; 运行期事实, 不进 model.yaml)")
    p.add_argument("--instances", type=int, default=None, help="覆盖 bench.instances")
    p.add_argument("--requests", type=int, default=None, help="覆盖 bench.requests")
    p.add_argument("--warmup", type=int, default=None, help="覆盖 bench.warmup")
    p.add_argument("--manifest", default=None,
                   help="覆盖 manifest 路径 (默认 <model_dir>/io/manifest.json; "
                        "产物在 --work-dir 下时用)")
    args = p.parse_args()
    run(args.config, device=args.device, instances=args.instances,
        requests=args.requests, warmup=args.warmup, manifest=args.manifest)


if __name__ == "__main__":
    main()
