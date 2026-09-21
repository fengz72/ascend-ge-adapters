"""执行后端: 把 Graph 编译为可执行产物, 并构造 C++ 运行时命令。

契约见 docs/architecture.md §1/§6/§9/§11/§15:
    两后端 — OM/ACL (离线: ATC 编译 Graph→OM, C++ ACL 执行)
             GeSession (在线: C++ 直接加载 AIR/ONNX, 无离线产物)。
    Backend.from_config(cfg) 按 cfg.backend.type 分发 (只需 cfg, 不需 manifest);
    compile(graph, env) 返回 om_path (om_acl) 或 None (ge_session);
    runtime_argv(manifest) 在 manifest 写好后构造 C++ 运行时命令。

    ATC 编译留 Python (§9), 包装 tools.atc_utils.run_atc; env 为 passes.prepare()
    返回的激活环境, 编译时临时注入 ATC 子进程。
"""

import contextlib
import os
import subprocess
from abc import ABC, abstractmethod

from tools.atc_utils import FRAMEWORK_AIR, FRAMEWORK_ONNX, run_atc

# C++ 运行时二进制 (§9/§14): runtime/ 构建产物 (bash runtime/build.sh)。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME_BIN = os.path.join(_REPO_ROOT, "runtime", "build", "ge_runtime")


@contextlib.contextmanager
def _injected_env(env):
    """临时把 env (pass 激活变量) 注入 os.environ, finally 还原。

    run_atc 不收 env 参数 (内部 os.environ.copy() 传给 ATC 子进程),
    故在此更新 os.environ, 让子进程继承 pass env (§7)。
    """
    if not env:
        yield
        return
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


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


class Backend(ABC):
    """执行后端基类。from_config 按 cfg.backend.type 分发到具体后端。"""

    @staticmethod
    def from_config(cfg, base_dir=None) -> "Backend":
        btype = cfg.backend.type
        if btype == "om_acl":
            return OmAclBackend(cfg, base_dir)
        if btype == "ge_session":
            return GeSessionBackend(cfg)
        raise ValueError(f"未知 backend.type: {btype!r}")

    @abstractmethod
    def compile(self, graph, env: dict):
        """编译 graph 为后端产物。返回 om_path 或 None。"""
        raise NotImplementedError


class OmAclBackend(Backend):
    """OM/ACL 离线后端: ATC 编译 Graph(AIR/ONNX) → OM (§1/§9)。"""

    def __init__(self, cfg, base_dir=None):
        self.cfg = cfg
        self.soc = cfg.model.soc
        self.aicore_num = cfg.backend.aicore_num
        self.om_dir = os.path.join(base_dir or cfg.model_dir, "om")

    def compile(self, graph, env: dict):
        """调 run_atc 把 graph 编译为 OM, 返回 om_path (失败 None)。

        env (pass 激活变量) 临时注入 ATC 子进程。
        动态图 (torchair dynamic 导出, io_spec 含 -1 维) **不传 --input_shape** —
        由 GE 处理动态维, ACL 运行时经 aclmdlSetDynamicInputTensorDesc 设实际 shape
        (旧验证流即如此); 仅静态图才传具体 input_shape。
        """
        # graph.kind → ATC framework: air=1 (GE 原生图), onnx=5 (docs §15 已定)
        is_dynamic = any(n.dynamic_dims for n in graph.io_spec.inputs)
        input_shape = None if is_dynamic else _input_shape_arg(graph.io_spec)
        with _injected_env(env):
            return run_atc(graph.path, self.om_dir, self.soc,
                           input_shape=input_shape, aicore_num=self.aicore_num,
                           framework=FRAMEWORK_ONNX if graph.kind == "onnx" else FRAMEWORK_AIR)


class GeSessionBackend(Backend):
    """GeSession 在线后端: C++ 直接加载 AIR/ONNX 在线执行, 无离线编译 (§1/§9)。"""

    def __init__(self, cfg):
        self.cfg = cfg

    def compile(self, graph, env: dict):
        # 形态③ ONNX 走不了在线后端: ge::Graph::LoadFromFile 只解析 GE 图 (.air/.pbtxt)。
        # 在**配置期**就拦下来 (而不是等 C++ 运行期报错), 附可操作的出路。
        if graph.kind == "onnx":
            raise ValueError(
                "backend.type=ge_session 不支持 ONNX 图 (GeSession 只加载 GE 图 .air/.pbtxt)。"
                "形态③ ONNX 请用 backend.type=om_acl — ATC 以 --framework=5 编译为 OM 后执行 "
                "(docs §1/§9)")
        # 在线后端无离线编译产物 (C++ 直接加载 AIR, API 序列见 docs §15), 返回 None。
        return None


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
