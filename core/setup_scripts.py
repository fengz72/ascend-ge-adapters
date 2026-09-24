"""环境准备脚本接口 — fusion pass / 自定义算子的构建安装由**用户脚本**负责, 框架只按序执行。

为什么不做自动化: 每个 pass / 算子源的安装方式都不一样 (fusion pass 是 cmake 出 .so 再拷到
`opp/vendors/<v>/custom_fusion_passes/`; AscendC 自定义算子是 `build.sh` 产 `.run` 再
`--install-path=$ASCEND_HOME_PATH/opp`, 还要 pip 装 torch 绑定 wheel; 有的需要 source
`set_env.bash`)。框架内置任何一种都会对不上号, 还得跟着三方源改版。所以只提供一个稳定
接口: **yaml 里填脚本路径**, 框架负责解析路径 → 按序执行 → 失败即停 → 回收脚本导出的环境变量。

两类脚本 (model.yaml, 见 docs §7):
    custom_ops:  在**加载 adapter 之前**执行 — model.py 可能 import 算子的 torch 绑定
    passes:      在 **ATC 编译之前**执行 — fusion pass 装进 opp/vendors 后由 CANN 自动扫描

条目形状 (core.config.SetupEntry):
    - path: third_party/custom_development_code/fusion_pass/WeightNzAndMatMulV3Pass
      script: models/qwen2.5-0.5b/scripts/install_nz_pass.sh
  `path` 是**三方源目录**, 只用于 ① 配置里一眼看出 pass/算子从哪来 (溯源) ② 以
  `$GE_SRC_DIR` 传给脚本 (脚本不必硬编码源码位置); 框架**不拿它构建** — 构建方式
  归脚本。也允许只写脚本路径 (纯字符串条目)。

约定:
  - 路径解析 (script 与 path 同规则): **只认绝对路径或相对仓库根** (如 models/m/scripts/x.sh);
    相对 model_dir / CWD 一律硬失败 — 多基准会让同一份配置混两种写法, 且 CWD 相对意味着
    "换个目录跑就找不到脚本"
  - 执行方式: `.py` → 当前解释器; 其余 → `bash <script>` (不要求 +x 与 shebang)
  - 环境: 继承当前进程 env (先 source CANN 的 set_env.sh 与 models/<model>/env.sh);
    脚本若要回传环境变量, 把 `KEY=VALUE` 行写进 `$GE_ENV_FILE` — 框架读进 os.environ,
    从而传给后续 ATC / ge_runtime 子进程 (进程内 export 是拿不到的)
  - 输出: 直接继承 stdout/stderr (构建动辄几分钟, 要能看到进度)
  - 幂等: 由脚本自己判断 (如"已安装则 exit 0"), 框架不缓存
  - 失败: 非 0 退出即抛 RuntimeError — 静默继续只会把问题推到更难查的下游
    (算子没装上 → 导出/ATC 报一堆看不懂的错)
"""

import importlib
import os
import subprocess
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_path(entry, is_file=False):
    """**配置里声明的路径**→ 绝对路径; 不存在返回 None。

    只认两种写法: 绝对路径, 或相对**仓库根** (如 `models/m/scripts/x.sh`、
    `models/m/config/target_tokens.json`)。yaml 里所有路径声明 (script / path /
    prune_token_file …) 都走这一条规则 —— 多基准会让同一份配置混两种写法 (读的人得先问
    "这条相对谁"), 且 CWD 相对意味着"换个目录跑就找不到"。写错就硬失败, 不猜。

    is_file=True 时要求是普通文件 (脚本), 否则目录/文件皆可 (三方源)。
    """
    if not entry:
        return None
    entry = str(entry).strip()
    if not entry:
        return None
    path = entry if os.path.isabs(entry) else os.path.join(_REPO_ROOT, entry)
    found = os.path.isfile(path) if is_file else os.path.exists(path)
    return os.path.normpath(path) if found else None


def resolve_script(entry):
    """脚本条目 → 绝对路径 (规则见 resolve_path); 找不到返回 None。"""
    return resolve_path(entry, is_file=True)


def run_scripts(entries, stage):
    """按序执行脚本, 返回实际执行过的脚本路径列表。

    entries: SetupEntry / dict{path,script} / str(脚本路径) 皆可。
    stage 只用于日志/报错前缀 (如 "custom_ops" / "passes")。
    任一脚本非 0 退出 → RuntimeError (带脚本路径与退出码)。
    """
    ran = []
    for entry in entries or []:
        script_spec, src_spec, extra_args = _entry_fields(entry)
        script = resolve_script(script_spec)
        if script is None:
            raise FileNotFoundError(
                f"[{stage}] 找不到脚本: {script_spec!r} — 只认**绝对路径**或**相对仓库根** "
                f"({_REPO_ROOT}) 的写法, 如 models/<m>/scripts/x.sh; "
                f"不支持相对 model_dir / CWD")

        env = os.environ.copy()
        fd, env_file = tempfile.mkstemp(prefix="ge_env_")
        os.close(fd)
        env["GE_ENV_FILE"] = env_file

        # 三方源目录: 只溯源 + 传给脚本 ($GE_SRC_DIR), 框架不拿它构建
        src_dir = None
        if src_spec:
            src_dir = _resolve_dir(src_spec)
            if src_dir is None:
                print(f"[{stage}][WARN] 声明的源路径不存在: {src_spec!r} "
                      f"(submodule 未克隆? git submodule update --init --recursive) — "
                      f"仍执行脚本, 由脚本决定跳过还是失败")
            else:
                env["GE_SRC_DIR"] = src_dir

        cmd = [sys.executable, script] if script.endswith(".py") else ["bash", script]
        cmd += list(extra_args)
        label = os.path.basename(src_dir) if src_dir else os.path.basename(script)
        print(f"[{stage}] {label}" + (f" ← {src_dir}" if src_dir else "") + f" | {' '.join(cmd)}")
        try:
            proc = subprocess.run(cmd, env=env)
        finally:
            applied = _apply_env_file(env_file)
            os.unlink(env_file)
        if applied:
            print(f"[{stage}] 脚本回传环境变量: {', '.join(applied)}")
        # 脚本可能刚装了 Python 包 (如算子的 torch 绑定) → 清导入缓存, 否则本进程 import 不到
        importlib.invalidate_caches()

        if proc.returncode != 0:
            raise RuntimeError(
                f"[{stage}] 脚本失败 (exit={proc.returncode}): {script}\n"
                f"        框架不猜安装方式, 请检查脚本本身 (构建日志见上方输出)")
        ran.append(script)
    return ran


def _entry_fields(entry):
    """SetupEntry / dict / str → (script, path, args)。"""
    if isinstance(entry, str):
        return entry, "", []
    if isinstance(entry, dict):
        return (str(entry.get("script") or ""), str(entry.get("path") or ""),
                [str(a) for a in (entry.get("args") or [])])
    return (str(getattr(entry, "script", "") or ""), str(getattr(entry, "path", "") or ""),
            [str(a) for a in (getattr(entry, "args", None) or [])])


def _resolve_dir(entry):
    """三方源路径 → 绝对路径 (规则见 resolve_path, 目录或文件皆可); 找不到返回 None。"""
    return resolve_path(entry)


def _apply_env_file(path):
    """读脚本写下的 KEY=VALUE 行 → os.environ, 返回被设置的键名列表。

    PYTHONPATH 特殊处理: 除写 os.environ (给后续子进程) 外, 同时把新增路径插进本进程的
    sys.path —— 否则脚本刚装的绑定包在当前进程里 import 不到 (model.py 随后就要 import)。
    """
    applied = []
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except OSError:
        return applied
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if not key:
            continue
        os.environ[key] = value
        applied.append(key)
        if key == "PYTHONPATH":
            for entry in value.split(os.pathsep):
                entry = entry.strip()
                if entry and entry not in sys.path:
                    sys.path.insert(0, entry)
    return applied
