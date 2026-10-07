"""ATC 编译工具 — 把 AIR (GE 原生图) 编译为 OM (--framework=1)。

命令行构造与执行分离: `build_atc_argv` 是纯函数 (argv 可直接打印复核),
`run_atc` 只负责建目录、起子进程、找产物。

两个要点:
  - **不用 shell**: argv 列表直传 subprocess (路径含空格安全, 也没有命令注入面);
    `--input_shape=a:1,2;b:3` 作为单个 argv 元素, 不需要 shell 引号。
  - **framework**: 恒为 AIR/GE 原生图 = 1 (`--framework=1`)。
    动态图 (io_spec 含 -1 维) 由调用方决定**不传** `--input_shape`, GE 运行期自行特化
    (docs §6②: 同一 OM 可跨 shape 复用, 无需 ATC 分档)。

fusion pass 不在此处激活: 装到 `$ASCEND_HOME_PATH/opp/vendors/<vendor>/custom_fusion_passes/`
(pass 装进它自己的 vendor 目录、全局一份, 非 per-model — 见 docs §7.1)
后由 CANN 自动扫描加载 (core/setup_scripts.py, docs §7)。
"""

import glob
import os
import shlex
import subprocess

import numpy

FRAMEWORK_AIR = 1        # GE 原生图 (.air / .pbtxt)


def normalize_aicore(aicore_num):
    """限核规格归一 → (ATC 参数值, OM 文件名后缀)。

    None/"" → (None, "")            不传, ATC 用全核
    12 或 "12" → ("12|24", "_c12_24")   整数视为 AIC 核数, AIV = 2×AIC (c:v = 1:2)
    "12|24" → ("12|24", "_c12_24")   已带 AIV 的原样透传
    后缀用于区分不同核数配置的 OM, 避免互相覆盖。
    """
    if aicore_num is None or aicore_num == "":
        return None, ""
    spec = str(aicore_num)
    if spec.isdigit():
        aic = spec
        aiv = str(int(spec) * 2)
    else:
        parts = spec.split("|")
        aic = parts[0]
        aiv = parts[1] if len(parts) > 1 else aic
    return f"{aic}|{aiv}", f"_c{aic}_{aiv}"


def build_atc_argv(graph_path, om_output, soc, input_shape=None, is_debug=False,
                   aicore_num=None):
    """构造 atc 命令行 (argv 列表, 不含 shell 引号)。om_output 不带 .om 后缀。"""
    aicore_str, _ = normalize_aicore(aicore_num)
    argv = ["atc",
            f"--framework={FRAMEWORK_AIR}",
            f"--model={graph_path}",
            f"--output={om_output}",
            f"--soc_version={soc}"]
    if input_shape:
        argv.append(f"--input_shape={input_shape}")
    if is_debug:
        argv.append("--log=debug")
    if aicore_str:
        argv.append(f"--aicore_num={aicore_str}")
    return argv


def run_atc(graph_path, om_dir, soc, input_shape=None, is_debug=False,
            aicore_num=None):
    """执行 ATC 把 graph 编译为 OM, 返回 .om 路径 (失败返回 None)。

    OM 输出到 om_dir, 文件名 = 图名 (+ 限核后缀)。ATC 子进程的 PYTHONPATH 注入本机
    numpy site-packages — CANN 的 tbe pywrapper 用内嵌 python3, 缺 numpy 会编译失败。
    """
    os.makedirs(om_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(graph_path))[0]
    _, name_suffix = normalize_aicore(aicore_num)
    om_output = os.path.join(om_dir, stem + name_suffix)

    argv = build_atc_argv(graph_path, om_output, soc, input_shape=input_shape,
                          is_debug=is_debug, aicore_num=aicore_num)
    print("=== 执行 ATC 编译 ===")
    print(f"  命令: {shlex.join(argv)}\n")

    env = os.environ.copy()
    numpy_site = os.path.dirname(os.path.dirname(numpy.__file__))
    env["PYTHONPATH"] = f"{numpy_site}:{env.get('PYTHONPATH', '')}"
    result = subprocess.run(argv, capture_output=True, text=True, env=env)
    print(result.stdout)
    if result.returncode != 0:
        print("[ERROR] ATC 编译失败:")
        print(result.stderr[-3000:])
        return None

    om_file = om_output + ".om"
    if not os.path.exists(om_file):
        candidates = glob.glob(f"{om_output}*.om")
        if not candidates:
            print(f"[ERROR] OM 文件未生成: {om_file}")
            return None
        om_file = candidates[0]

    print(f"=== OM 编译完成: {om_file} ({os.path.getsize(om_file) / 1024 / 1024:.1f} MB) ===\n")
    return om_file
