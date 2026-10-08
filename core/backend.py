"""执行后端: 把 Graph 编译为可执行产物, 并构造/启动 C++ 运行时。

契约见 docs/architecture.md §1/§6/§9/§11/§15:
    两后端 — OM/ACL (离线: ATC 编译 Graph→OM, C++ ACL 执行)
             GeSession (在线: C++ 直接加载 AIR, 无离线产物)。
    两后端都消费 AIR (GE 原生图); compile_graph(cfg, graph) 按 cfg.backend.type 分发,
    返回 om_path 或 None; runtime_argv/run_runtime 在 manifest 写好后构造并启动 C++ 运行时。

**不抽 Backend 基类** (与 C++ 侧同一标准, docs §9): 两后端的差异只是"要不要 ATC",
一个分支足够。ATC 编译留 Python (§9), 包装 tools.atc_utils.run_atc; fusion pass 装进
opp/vendors 后由 CANN 自动扫描加载, 不需要 env 注入 (core/setup_scripts.py, docs §7)。
"""

import os
import subprocess

from tools.atc_utils import run_atc
from tools.parse_profiling import cmd_parse_and_export, cmd_summary, find_msprof

# C++ 运行时二进制 (§9/§14): runtime/ 构建产物 (bash runtime/build.sh)。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME_BIN = os.path.join(_REPO_ROOT, "runtime", "build", "ge_runtime")


def _input_shape_arg(io_spec):
    """把 io_spec.inputs 整理成 ATC --input_shape 串: name:d0,d1;name2:...

    仅**静态图**用 (compile 里检测到动态维就传 None, 由 GE 运行期特化 — docs §6②:
    实测同一动态 OM 可跨 shape 复用, 不需要 ATC 分档)。name 用图节点真名 (argN_1),
    顺序即 io_spec 的图序。`--dynamic_batch_size`/`--dynamic_dims` 分档属**未实现的
    可选优化** (减少运行期特化开销), 不是必需环节。
    """
    if not io_spec or not io_spec.inputs:
        return None
    parts = []
    for n in io_spec.inputs:
        if not n.shape:
            continue
        name = n.node or n.logical
        parts.append(f"{name}:{','.join(str(d) for d in n.shape)}")
    return ";".join(parts) or None


def compile_graph(cfg, graph, base_dir=None):
    """按 cfg.backend.type 把 graph 编译为后端产物: om_acl → OM 路径; ge_session → None。

    两后端都消费 AIR。om_acl: 动态图 (io_spec 含 -1 维) **不传 --input_shape** — GE 运行期
            自行特化, 同一 OM 可跨 shape 复用 (docs §6②); 仅静态图传具体 shape。ATC 恒用
            --framework=1 (AIR)。
    ge_session: 无离线产物 (C++ 直接加载 AIR), 返回 None。
    """
    if cfg.backend.type == "ge_session":
        return None

    if cfg.backend.type != "om_acl":
        raise ValueError(f"未知 backend.type: {cfg.backend.type!r} (期望 om_acl | ge_session)")

    is_dynamic = any(n.dynamic_dims for n in graph.io_spec.inputs)
    input_shape = None if is_dynamic else _input_shape_arg(graph.io_spec)
    om_dir = os.path.join(base_dir or cfg.model_dir, "om")
    return run_atc(graph.path, om_dir, cfg.model.soc, input_shape=input_shape,
                   aicore_num=cfg.backend.aicore_num)


def backend_extra(cfg) -> tuple:
    """model.yaml 的 backend 声明 → C++ 运行时选项 (只有在线后端有**运行期**选项)。

    限核在两个后端的生效点不同: om_acl 是 ATC 编译期 `--aicore_num` (compile_graph 已传),
    ge_session 是 GEInitialize 运行期选项 — 不转发就等于静默全核跑, yaml 里的声明成了装饰。
    CLI `--runtime-opt` 的同名选项排在其后, 覆盖 yaml (C++ 侧后到的赋值生效)。
    """
    if cfg.backend.type == "ge_session" and cfg.backend.aicore_num:
        return ("--aicore_num", str(cfg.backend.aicore_num))
    return ()


def default_output_dir(manifest_path: str) -> str:
    """C++ 运行时的默认输出目录: <manifest 根>/io/outputs (与 main.cpp 一致)。"""
    base = os.path.dirname(os.path.dirname(os.path.abspath(manifest_path)))
    return os.path.join(base, "io", "outputs")


def runtime_argv(manifest_path: str, output_dir=None, device=None,
                 warmup=0, bench=1, extra=()):
    """构造调 C++ 运行时的命令: [ge_runtime, manifest, --output_dir ... ] (§9/§11)。

    manifest 是唯一必需参数 (backend/路径/io_spec/device 全在里面); 其余为运行期覆盖。
    extra 透传后端专属选项 (如 --aicore_num / --precision_mode / --graph_run_mode)。
    """
    argv = [RUNTIME_BIN, manifest_path]
    if output_dir:
        argv += ["--output_dir", output_dir]
    if device is not None:
        argv += ["--device", str(device)]
    if warmup:
        argv += ["--warmup", str(warmup)]
    if bench != 1:
        argv += ["--bench", str(bench)]
    argv += list(extra)
    return argv


def run_runtime(manifest_path: str, output_dir=None, device=None,
                warmup=0, bench=1, extra=()):
    """跑 C++ 运行时 (子进程, 继承当前 CANN 环境), 返回输出目录。

    输出: <output_dir>/output_<i>.bin + outputs.json (logical/dtype/shape/file) —
    compare 阶段据此加载 (verify.Verifier.compare_bundle)。
    """
    if not os.path.exists(RUNTIME_BIN):
        raise FileNotFoundError(
            f"C++ 运行时未构建: {RUNTIME_BIN} (先执行 bash runtime/build.sh)")

    output_dir = output_dir or default_output_dir(manifest_path)
    argv = runtime_argv(manifest_path, output_dir, device, warmup, bench, extra)
    print(f"=== 运行 C++ runtime ===\n  命令: {' '.join(argv)}\n")
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(argv, check=True)
    return output_dir


def parse_profiling(prof_dir: str):
    """PROF_* → msprof parse+export (出 mindstudio_profiler_output/*.csv) → 打印算子摘要。

    `--profiling-parse` 的实现体: 采集在 C++ 侧, 解析在 Python/msprof 侧, pipeline 与 bench
    两个入口共用故落在这里 (与 run_runtime 同一层: 都是"起外部工具")。msprof 缺失只 WARN —
    数据已落盘, 不该让跑完的管线/性能测试因解析环节非 0 退出。
    """
    if find_msprof() is None:
        print(f"[profiling][WARN] 找不到 msprof (ASCEND_HOME_PATH="
              f"{os.environ.get('ASCEND_HOME_PATH', '-')}) — 数据已在 {prof_dir}, 手动解析: "
              f"python3 tools/parse_profiling.py parse-and-export --profiling_dir {prof_dir}")
        return
    print(f"=== 解析 profiling: {prof_dir} ===")
    cmd_parse_and_export(prof_dir)
    cmd_summary(prof_dir)
