"""并发档位扫描 — 同一负载逐档跑, 汇总成一张 scaling 表。

为什么不在 C++ 里做: 扫描 = "同一件事跑 N 遍", 而每档都要独立建/销资源 (ACL 每实例
`aclmdlLoadFromFile`; GeSession 每实例 `AddGraph`+`CompileGraph` ~10s)。C++ 只负责**测准一档**
(`BenchPool` / `BenchThroughput`), 档位循环归这里。

隔离粒度: 每档的**测量**都跑在 `core.bench` 起的一个全新 `ge_runtime` 子进程里 —— NPU 资源
逐档建/销, HBM 彻底归还; 一档崩了 (段错误 / OOM) 也只损失那一档 (下面按 Exception 兜住,
跑完剩余档位再汇总)。本脚本自身是进程内循环: Python 侧只解析配置与排版报告, 不持有 NPU
资源, 没必要再套一层进程。

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
    """读一档的 perf.json → 表格要用的字段。

    读不到/版本不匹配就**让它抛** —— 调用方的 except 会把这一档记成失败并打出原因;
    在这里吞成 None 会让"perf.json 损坏"与"这档压根没跑起来"在汇总表里长得一样。
    """
    perf = bench.load_perf(run_dir)
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
    # 基准取**实例数最少**的成功档, 不是 rows 里的第一个 —— --levels 可以乱序给 (8,4,2,1),
    # 取第一个会让表尾那行"加速比以最低成功档为基准"变成假话
    ok = [r for r in rows if r]
    base = min(ok, key=lambda r: r["instances"])["qps"] if ok else 0.0
    lines = [
        f"# 并发档位扫描 — {scenario}",
        "",
        f"- device: {device} · 档位: {', '.join(str(n) for n in levels)}",
        f"- 时间: {datetime.datetime.now().isoformat(timespec='seconds')}",
        "- 每档 = 一次独立的 `core.bench` run (测量各在一个全新 `ge_runtime` 子进程里, "
        "明细见各自 run 目录)",
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

    # 先解析一次配置: ① manifest 缺失/平台探测失败就**当场**报错, 不必等跑完几档才发现
    # ② 汇总要用的 name/report_dir/device 从这里取 —— 原来在循环后又调一次 load_bench,
    #    白白多做一遍 resolve_platform (import torch_npu + 查设备)
    sc = bench.load_bench(args.config, device=args.device, manifest=args.manifest)

    rows, failed = [], []
    for level in levels:
        print(f"\n{'=' * 70}\n=== 档位 {level}/{levels[-1]} (instances={level}) ===\n{'=' * 70}")
        try:
            rows.append(collect(bench.run(args.config, device=args.device, instances=level,
                                          requests=args.requests, warmup=args.warmup,
                                          manifest=args.manifest)))
        except (Exception, SystemExit) as e:
            # 一档失败不连累其它档。**必须**兜住 Exception 而不只是 RuntimeError/OSError:
            # ge_runtime 段错误/OOM 时 subprocess.run(check=True) 抛的是 CalledProcessError,
            # bench 检出错误请求数抛 SystemExit, perf.json 损坏/版本不符抛 ValueError ——
            # 漏掉任何一个都会让剩余档位白跑 (连汇总表都出不来)。
            # KeyboardInterrupt 不在内: Ctrl-C 要能真的中断整个扫描。
            print(f"[sweep] 档位 {level} 失败: {type(e).__name__}: {e}", file=sys.stderr)
            rows.append(None)
            failed.append(level)

    if rows and all(r is None for r in rows):
        raise SystemExit("[sweep] 全部档位失败, 无汇总可出")

    out = args.out or os.path.join(
        sc.report_dir, f"sweep-{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}.md")
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
