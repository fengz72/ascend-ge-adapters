"""并发档位扫描 — 同一负载逐档**起进程**, 汇总成一张 scaling 表。

为什么不在 C++ 里做: 扫描 = "同一件事跑 N 遍", 而每档都要独立建/销资源 (ACL 每实例
`aclmdlLoadFromFile`; GeSession 每实例 `AddGraph`+`CompileGraph` ~10s)。进程级隔离最干净 ——
一档一个进程, 崩了不连累其它档, HBM 也彻底归还。C++ 只负责**测准一档**
(`BenchPool` / `BenchThroughput`), 档位循环归这里。

每档就是一次完整的 `core.bench` run (各自归档 `results/<run_id>/`), 本脚本只多做两件事:
逐档换 `--instances`, 以及把各档的 perf.json 汇成一张表 (`<model_dir>/results/sweep-<ts>.md`)。

用法:
    python3 tools/sweep.py --config models/qwen2.5-0.5b/config/model.yaml --device 8 \
        --levels 1,2,4,8
    # 每档请求数/预热覆盖 bench 段里的值:
    python3 tools/sweep.py --config <model.yaml> --device 8 --levels 1,2,4,8 --requests 4000

固定输入 (同一组输入反复跑) 的吞吐扫描: 把请求池目录做成**只含一个 bundle.json** 的目录
(runtime/pool.cpp 支持 `<dir>/bundle.json` 单套), 再走同一条路 —— 不必另开一个口径。
"""

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import bench                                  # noqa: E402


def parse_levels(spec):
    levels = [int(x) for x in str(spec).split(",") if x.strip()]
    if not levels or any(n < 1 for n in levels):
        raise ValueError(f"--levels 须是逗号分隔的正整数 (如 1,2,4,8): {spec!r}")
    return levels


def collect(run_dir):
    """读一档的 perf.json → 表格要用的字段 (读不到返回 None)。"""
    try:
        perf = json.load(open(os.path.join(run_dir, "perf.json")))
    except (OSError, ValueError):
        return None
    e2e = perf.get("e2e_ms") or {}
    hbm = perf.get("hbm_mb") or {}
    execs = [ip.get("exec_ms") or {} for ip in (perf.get("instances") or [])]
    return {
        "instances": len(perf.get("instances") or []),
        "requests": perf.get("requests", 0),
        "qps": perf.get("qps", 0.0),
        "e2e_avg": e2e.get("avg", 0.0),
        "e2e_p99": e2e.get("p99", 0.0),
        "exec_avg": (sum(e.get("avg", 0.0) for e in execs) / len(execs)) if execs else 0.0,
        "hbm_peak": hbm.get("peak", -1),
        "errors": perf.get("errors", 0),
        "run_dir": run_dir,
    }


def render_md(rows, failed, scenario, levels, device):
    base = next((r["qps"] for r in rows if r), 0.0)
    lines = [
        f"# 并发档位扫描 — {scenario}",
        "",
        f"- device: {device} · 档位: {', '.join(str(n) for n in levels)}",
        f"- 时间: {datetime.datetime.now().isoformat(timespec='seconds')}",
        "- 每档 = 一次独立的 `core.bench` run (独立进程, 明细见各自 run 目录)",
        "",
        "| instances | requests | QPS | 加速比 | 并行效率 | e2e avg | e2e p99 | exec avg "
        "| HBM peak | err | run |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for level, row in zip(levels, rows):
        if row is None:
            lines.append(f"| {level} | - | **失败** | - | - | - | - | - | - | - | - |")
            continue
        speedup = row["qps"] / base if base else 0.0
        lines.append(
            f"| {row['instances']} | {row['requests']} | {row['qps']:.2f} "
            f"| {speedup:.2f}x | {speedup / row['instances'] * 100:.0f}% "
            f"| {row['e2e_avg']:.3f} | {row['e2e_p99']:.3f} | {row['exec_avg']:.3f} "
            f"| {row['hbm_peak']:.0f} MB | {row['errors']} | `{os.path.basename(row['run_dir'])}` |")
    lines += [
        "",
        "> 加速比以**最低成功档**为基准; 并行效率 = 加速比 / instances。"
        "单卡上效率随档位下降是正常的 (算力争抢), 关注的是 QPS 是否还在涨、p99 是否可接受。",
    ]
    if failed:
        lines += ["", f"> 失败档位: {', '.join(str(n) for n in failed)}"]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description="并发档位扫描 (逐档起进程 + 汇总 scaling 表)")
    p.add_argument("--config", required=True, help="model.yaml 路径 (bench 段 = 压测口径)")
    p.add_argument("--device", type=int, required=True,
                   help="NPU 设备号 (必填; 运行期事实, 不进 model.yaml)")
    p.add_argument("--levels", default="1,2,4,8", help="并发档位, 逗号分隔 (默认 1,2,4,8)")
    p.add_argument("--requests", type=int, default=None, help="覆盖每档的总请求数")
    p.add_argument("--warmup", type=int, default=None, help="覆盖 warmup 下限")
    p.add_argument("--manifest", default=None,
                   help="覆盖 manifest 路径 (默认 <model_dir>/io/manifest.json)")
    p.add_argument("--out", default=None,
                   help="汇总表输出路径 (默认 <model_dir>/results/sweep-<ts>.md)")
    args = p.parse_args()

    levels = parse_levels(args.levels)
    rows, failed = [], []
    report_dir = None
    for level in levels:
        print(f"\n{'=' * 70}\n=== 档位 {level}/{levels[-1]} (instances={level}) ===\n{'=' * 70}")
        try:
            run_dir = bench.run(args.config, device=args.device, instances=level,
                                requests=args.requests, warmup=args.warmup,
                                manifest=args.manifest)
            rows.append(collect(run_dir))
            report_dir = report_dir or os.path.dirname(run_dir)
        except (SystemExit, RuntimeError, OSError) as e:
            print(f"[sweep] 档位 {level} 失败: {e}", file=sys.stderr)
            rows.append(None)
            failed.append(level)

    if rows and all(r is None for r in rows):
        raise SystemExit("[sweep] 全部档位失败, 无汇总可出")

    sc = bench.load_bench(args.config, device=args.device, manifest=args.manifest)
    report_dir = report_dir or sc.report_dir
    out = args.out or os.path.join(
        report_dir, f"sweep-{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    md = render_md(rows, failed, sc.name, levels, sc.device)
    with open(out, "w") as f:
        f.write(md)
    print("\n" + md)
    print(f"[sweep] 汇总表: {out}")
    if failed:
        raise SystemExit(f"[sweep] 有档位失败: {failed}")


if __name__ == "__main__":
    main()
