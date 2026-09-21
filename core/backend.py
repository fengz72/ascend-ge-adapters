"""执行后端: 把 Graph 编译为可执行产物, 并构造/启动 C++ 运行时。

契约见 docs/architecture.md §1/§6/§9/§11/§15:
    两后端 — OM/ACL (离线: ATC 编译 Graph→OM, C++ ACL 执行)
             GeSession (在线: C++ 直接加载 AIR, 无离线产物)。
    compile_graph(cfg, graph) 按 cfg.backend.type 分发, 返回 om_path 或 None;
    runtime_argv/run_runtime 在 manifest 写好后构造并启动 C++ 运行时。

**不抽 Backend 基类** (与 C++ 侧同一标准, docs §9): 两后端的差异只是"要不要 ATC",
一个分支足够; 在线后端仍要校验 图形态×后端 组合 (ONNX 不支持 GeSession)。
ATC 编译留 Python (§9), 包装 tools.atc_utils.run_atc; fusion pass 装进 opp/vendors 后
由 CANN 自动扫描加载, 不需要 env 注入 (core/passes.py, docs §7)。
"""

import os
import subprocess

from tools.atc_utils import FRAMEWORK_AIR, FRAMEWORK_ONNX, run_atc

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

    om_acl: 动态图 (io_spec 含 -1 维) **不传 --input_shape** — GE 运行期自行特化,
            同一 OM 可跨 shape 复用 (docs §6②); 仅静态图传具体 shape。
            framework 按 graph.kind 取 (air=1 / onnx=5)。
    ge_session: 无离线产物; 但仍在此校验 图形态×后端 组合 — 形态③ ONNX 走不了在线后端
            (ge::Graph::LoadFromFile 只解析 GE 图), 在**配置期**报错而不是等 C++ 运行期。
    """
    if cfg.backend.type == "ge_session":
        if graph.kind == "onnx":
            raise ValueError(
                "backend.type=ge_session 不支持 ONNX 图 (GeSession 只加载 GE 图 .air/.pbtxt)。"
                "形态③ ONNX 请用 backend.type=om_acl — ATC 以 --framework=5 编译为 OM 后执行 "
                "(docs §1/§9)")
        return None

    if cfg.backend.type != "om_acl":
        raise ValueError(f"未知 backend.type: {cfg.backend.type!r} (期望 om_acl | ge_session)")

    is_dynamic = any(n.dynamic_dims for n in graph.io_spec.inputs)
    input_shape = None if is_dynamic else _input_shape_arg(graph.io_spec)
    om_dir = os.path.join(base_dir or cfg.model_dir, "om")
    return run_atc(graph.path, om_dir, cfg.model.soc, input_shape=input_shape,
                   aicore_num=cfg.backend.aicore_num,
                   framework=FRAMEWORK_ONNX if graph.kind == "onnx" else FRAMEWORK_AIR)


def default_output_dir(manifest_path: str) -> str:
    """C++ 运行时的默认输出目录: <manifest 根>/verification/outputs (与 main.cpp 一致)。"""
    base = os.path.dirname(os.path.dirname(os.path.abspath(manifest_path)))
    return os.path.join(base, "verification", "outputs")


def runtime_argv(manifest_path: str, output_dir=None, device=None,
                 warmup=0, bench=1, extra=(), inputs=()):
    """构造调 C++ 运行时的命令: [ge_runtime, manifest, --output_dir ... ] (§9/§11)。

    manifest 是唯一必需参数 (backend/路径/io_spec/device 全在里面); 其余为运行期覆盖。
    inputs: 部署态 (manifest 无 bundle) 的输入规格, 每项 "logical:d0,d1,...:file.bin"
            — dtype/format/node 仍由 io_spec 提供, 不在此重复声明。
    extra 透传后端专属选项 (如 --aicore_num / --precision_mode / --graph_run_mode)。
    """
    argv = [RUNTIME_BIN, manifest_path]
    if output_dir:
        argv += ["--output_dir", output_dir]
    for spec in inputs:
        argv += ["--input", spec]
    if device is not None:
        argv += ["--device", str(device)]
    if warmup:
        argv += ["--warmup", str(warmup)]
    if bench != 1:
        argv += ["--bench", str(bench)]
    argv += list(extra)
    return argv


def run_runtime(manifest_path: str, output_dir=None, device=None,
                warmup=0, bench=1, extra=(), inputs=()):
    """跑 C++ 运行时 (子进程, 继承当前 CANN 环境), 返回输出目录。

    输出: <output_dir>/output_<i>.bin + outputs.json (logical/dtype/shape/file) —
    compare 阶段据此加载 (verify.Verifier.compare_bundle)。
    """
    if not os.path.exists(RUNTIME_BIN):
        raise FileNotFoundError(
            f"C++ 运行时未构建: {RUNTIME_BIN} (先执行 bash runtime/build.sh)")

    output_dir = output_dir or default_output_dir(manifest_path)
    argv = runtime_argv(manifest_path, output_dir, device, warmup, bench, extra, inputs)
    print(f"=== 运行 C++ runtime ===\n  命令: {' '.join(argv)}\n")
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(argv, check=True)
    return output_dir
