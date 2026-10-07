"""模型配置 (YAML 人工声明) 解析 + 运行时 manifest (JSON) 生成。

契约见 docs/architecture.md §5:
    model.yaml   人工声明 (source/adapt/inputs/graph/passes/custom_ops/backend/verify/bench; 不含 device)
    manifest.json 生成的 C++ 运行时契约 (io/manifest.json)
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class ModelMeta:
    name: str
    soc: str = "Ascend910_9382"


@dataclass
class SourceCfg:
    """源为 torch-only: 来路① ref → from_pretrained (来路② 客户源码按 YAGNI 推迟, 未实现, 见 docs §10)。"""
    ref: str = ""                   # hub id 或本地权重目录


@dataclass
class AdaptCfg:
    adapter_class: str = ""         # 同目录 model.py 里的类名
    params: dict = field(default_factory=dict)   # 仅适配行为开关 (prefix/prune)


@dataclass
class InputsCfg:
    batch_size: int = 10
    seq_len: int = 208
    prefix_len: int = 0
    seed: int = 0


@dataclass
class DynamicCfg:
    max_seq_len: int = 2048
    # 分档/range 字段随 ATC 机制定 (docs §15 待定)


@dataclass
class GraphCfg:
    dynamic: DynamicCfg = field(default_factory=DynamicCfg)


@dataclass
class BackendCfg:
    type: str = "om_acl"            # om_acl | ge_session
    aicore_num: Optional[str] = None


@dataclass
class SetupEntry:
    """一条环境准备声明 (fusion pass / 自定义算子): 源在哪 + 用哪个脚本装。

    script  安装脚本路径 (框架只负责执行, 不假设构建方式 — 见 core/setup_scripts.py)
    path    三方源目录 (如 third_party/custom_development_code/fusion_pass/XxxPass)。
            作用是**溯源**(配置里一眼看出 pass/算子从哪来) + 传给脚本 ($GE_SRC_DIR),
            框架自己不拿它构建。
    args    传给脚本的额外参数 (如请求池生成器的 --count/--dist)
    """
    script: str
    path: str = ""
    args: list = field(default_factory=list)


@dataclass
class VerifyCfg:
    """验证开关 + 两道精度门的门限覆盖。

    四个门限默认 None = 用 core/verify.py 的规范默认 (门① reference 0.999/0.02 跨实现,
    门② compare 0.9999/0.01 同源); 填显式值即覆盖。换 dtype(bf16)/更大词表/更长序列时
    噪声量级会变, 改这里即可, 不动框架 (docs §10/§13.8)。pipeline 构造 Verifier 时透传。
    """
    enabled: bool = True
    ref_cosine_min: Optional[float] = None    # 门① 原版 HF vs 适配后 eager (跨实现)
    ref_rel_l2_max: Optional[float] = None
    cmp_cosine_min: Optional[float] = None    # 门② 运行时输出 vs golden (同源)
    cmp_rel_l2_max: Optional[float] = None


@dataclass
class BenchCfg:
    """性能测试口径 (core.bench 用) — **一个模型一个场景**。

    只放"这次压测怎么压"; 模型侧事实一律不在此重复, 由 core.bench 从同一份 ModelConfig 取:
        soc            ← model.soc
        manifest       ← <model_dir>/io/manifest.json  (write_manifest 的固定约定)
        report_dir     ← <model_dir>/results
        请求池目录      ← <model_dir>/io/pool
        aicore_num     ← backend.aicore_num (ge_session 的运行期限核)
        --batch/--prune-tokens/--prefix ← inputs.batch_size / adapt.params (bench._form_args 注入)
    负载分布 (长度分布/词表上界/套数) 属**模型专属脚本**的口径, 写死在 pool.script 的
    argparse 默认值里, 配置只在要覆盖时写 pool.args。

    **不含精度开关**: 性能跑不落盘输出 (D2H 污染延迟), 精度只由 verify.enabled 驱动的那两道门
    度量 (run.sh 的 reference + compare) — 一个变量只由一处度量, 不设第二个开关。

    多场景并存 (同一形态要随机负载 + 固定 shape + 长序列压测三份报告) 目前不支持 —
    真出现该需求时再拆回独立 scenario 文件 (形状可从 git 历史的 bench/varlen.yaml 取),
    届时场景名从文件名取即可, 故这里**不设 scenario 字段**: 归档名一律 <model.name>-bench。
    """
    instances: int = 1              # = 并发 worker (1:1 绑实例, 无锁)
    requests: int = 100             # 总请求 (闭环, 均分到实例)
    warmup: int = 10                # 覆盖 每实例 × 每 shape 档 (GE 首次执行要特化)
    sample_seed: int = 0            # 请求池**抽样**种子 (每实例 seed+instance_id); ≠ inputs.seed
    pool: Optional["SetupEntry"] = None   # 请求池生成脚本 (负载口径在脚本默认值里)


@dataclass
class ModelConfig:
    model: ModelMeta
    source: SourceCfg
    adapt: AdaptCfg = field(default_factory=AdaptCfg)
    inputs: InputsCfg = field(default_factory=InputsCfg)
    graph: GraphCfg = field(default_factory=GraphCfg)
    passes: list = field(default_factory=list)        # list[SetupEntry] fusion pass (ATC 前执行)
    custom_ops: list = field(default_factory=list)    # list[SetupEntry] 自定义算子 (加载 adapter 前)
    backend: BackendCfg = field(default_factory=BackendCfg)
    verify: VerifyCfg = field(default_factory=VerifyCfg)
    bench: BenchCfg = field(default_factory=BenchCfg)
    model_dir: str = ""             # 配置文件所在模型目录 (load 时填入)


def _sub(cls, d: dict, **renames):
    """从 dict 构造 dataclass, 缺失字段用默认值; renames 映射 yaml 键→字段名。"""
    d = dict(d or {})
    for yaml_key, field_name in renames.items():
        if yaml_key in d:
            d[field_name] = d.pop(yaml_key)
    valid = {f for f in cls.__dataclass_fields__}
    return cls(**{k: v for k, v in d.items() if k in valid})


def _setup_entries(raw) -> list:
    """yaml 的 passes/custom_ops → list[SetupEntry]。

    两种写法:
        - path: third_party/.../XxxPass        # 源 (溯源 + $GE_SRC_DIR)
          script: models/m/scripts/install.sh  # 安装脚本
        - models/m/scripts/install.sh          # 只给脚本 (无源路径)
    """
    entries = []
    for item in raw or []:
        if isinstance(item, str):
            entries.append(SetupEntry(script=item))
        elif isinstance(item, dict):
            script = item.get("script") or item.get("install") or ""
            if not script:
                raise ValueError(f"passes/custom_ops 条目缺 script 字段: {item}")
            entries.append(SetupEntry(script=str(script), path=str(item.get("path") or ""),
                                      args=[str(a) for a in (item.get("args") or [])]))
        else:
            raise ValueError(f"passes/custom_ops 条目须是字符串或 {{path, script}} 映射: {item!r}")
    return entries


def load_config(path) -> ModelConfig:
    """解析 model.yaml → ModelConfig。model_dir = config/ 的父目录。

    **不含 device**: 用哪张卡是运行期事实 (每次运行/每台机器都可能不同), 由 CLI `--device`
    必填传入 (pipeline → load_source/write_manifest → manifest.device → C++ 运行时)。
    yaml 里多余的 `runtime:` 段会被忽略。
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    model_dir = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    dynamic = _sub(DynamicCfg, (raw.get("graph") or {}).get("dynamic", {}))
    bench_d = dict(raw.get("bench") or {})
    pool = _setup_entries([bench_d.pop("pool")] if bench_d.get("pool") else [])
    bench = _sub(BenchCfg, bench_d)
    bench.pool = pool[0] if pool else None      # SetupEntry 不走 _sub (它只做 yaml 键→字段名映射)

    return ModelConfig(
        model=_sub(ModelMeta, raw.get("model")),
        source=_sub(SourceCfg, raw.get("source")),
        adapt=_sub(AdaptCfg, raw.get("adapt")),
        inputs=_sub(InputsCfg, raw.get("inputs")),
        graph=GraphCfg(dynamic=dynamic),
        passes=_setup_entries(raw.get("passes")),
        custom_ops=_setup_entries(raw.get("custom_ops")),
        backend=_sub(BackendCfg, raw.get("backend")),
        verify=_sub(VerifyCfg, raw.get("verify")),
        bench=bench,
        model_dir=model_dir,
    )


def load_target_tokens(json_path):
    """加载 target token JSON ({"token_ids": [...]}), 返回 token_ids 列表。"""
    with open(json_path) as f:
        data = json.load(f)
    token_ids = data["token_ids"]
    assert len(token_ids) > 0, "token_ids 不能为空"
    return token_ids


def export_name(cfg: ModelConfig) -> str:
    """产物名 = 模型名 + **形态后缀** (AIR/OM/bundle/manifest 都以此命名)。

    凡是影响图结构的 adapt.params 开关都要编进名字, 否则切换形态时会**静默覆盖**上一种
    形态的产物 (你以为在做 A/B, 其实基线已经被冲掉)。新增形态开关时记得在这里加后缀。
    """
    name = cfg.model.name
    if cfg.adapt.params.get("prefix"):
        name += "-prefix"
    if cfg.adapt.params.get("prune_token_file"):
        name += "-prune"
    return name


def load_adapter(cfg: ModelConfig):
    """importlib 从 <model_dir>/model.py 加载 adapt.adapter_class 并按 params 实例化。

    params 直接作为 adapter 构造 kwargs (基类 __init__ 收 **params, 故最小 adapter
    无需自定义构造); 特殊键 prune_token_file **仅在声明时**才被载入成 prune_tokens 列表传入
    (文件 I/O 在配置层, adapter 只收 list)。它的路径规则与 script/path 一致 ——
    绝对 或 相对**仓库根** (core/setup_scripts.resolve_path), 找不到即硬失败。
    """
    import importlib.util

    from core.setup_scripts import resolve_path

    model_py = os.path.join(cfg.model_dir, "model.py")
    spec = importlib.util.spec_from_file_location(f"_ge_adapter_{cfg.model.name}", model_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, cfg.adapt.adapter_class)

    params = dict(cfg.adapt.params)
    ptf = params.pop("prune_token_file", None)
    if ptf:
        resolved = resolve_path(ptf, is_file=True)
        if resolved is None:
            raise FileNotFoundError(
                f"adapt.params.prune_token_file 找不到: {ptf!r} — 只认**绝对路径**或"
                f"**相对仓库根**的写法, 如 models/{cfg.model.name}/config/target_tokens.json")
        params["prune_tokens"] = load_target_tokens(resolved)
    return cls(**params)


def write_manifest(cfg: ModelConfig, graph_path, om_path, io_spec_path,
                   bundle_path=None, base_dir=None, device=None) -> str:
    """生成 <base_dir>/io/manifest.json (C++ 运行时契约)。

    路径相对 base_dir (默认 model_dir), 保证可移植; --work-dir 调试时 base_dir
    传 work_dir, 产物与 manifest 同根。C++ 侧的基准是 **manifest 的祖父目录**
    (runtime/io_spec.cpp: DirName(DirName(path))), 故 manifest 必须落在 base_dir 的
    一级子目录下 (io/), 换目录名要两侧同步。

    device **必填** (来自 CLI --device): manifest 是 C++ 运行时的唯一入口, 缺 device
    它无从知道跑哪张卡; 不设默认值是因为"默认 0 号卡"通常正是被占满的那张。
    """
    if device is None:
        raise ValueError("write_manifest 需要 device (由 pipeline 的 --device 传入)")
    base = base_dir or cfg.model_dir

    def rel(p):
        return os.path.relpath(p, base) if p else None

    manifest = {
        "backend": cfg.backend.type,
        "graph_path": rel(graph_path),
        "om_path": rel(om_path),
        "io_spec": rel(io_spec_path),
        "device": device,
        "bundle": rel(bundle_path),
    }
    io_dir = os.path.join(base, "io")
    os.makedirs(io_dir, exist_ok=True)
    manifest_path = os.path.join(io_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest_path
