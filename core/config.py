"""模型配置 (YAML 人工声明) 解析 + 运行时 manifest (JSON) 生成。

契约见 docs/architecture.md §5:
    model.yaml   人工声明 (source/adapt/inputs/graph/platforms/backend/verify/bench; 不含 device)
    manifest.json 生成的 C++ 运行时契约 (io/manifest.json)

**平台**是运行期事实 (由机器上的卡决定), 但"这个模型支持哪些平台、各平台的事实是什么"
是模型声明 → `platforms:` 表进 yaml, 由 resolve_platform 按 soc 探测选定 (CLI --platform
可覆盖), 再把选中的 profile **摊平**回 cfg.model.soc / cfg.passes / cfg.custom_ops /
cfg.backend.aicore_num — 下游 (backend/write_manifest/bench) 不需要知道"平台"这一层。
"""

import json
import os
from dataclasses import dataclass, field, replace
from typing import Optional

import yaml


@dataclass
class ModelMeta:
    name: str
    soc: str = ""       # 由 resolve_platform 从选中的 profile 填入; yaml 不再顶层声明


@dataclass
class PlatformProfile:
    """一个平台的**全部**平台相关事实。**强 schema**: yaml 写未知键 → 硬失败。

    只收平台相关键。形态开关 (prefix/prune) 属 adapt.params —— 形态是**串行演进**
    (最终收敛到 prefix), 平台是**长期并存且都进回归**, 两轴性质不同故不共用机制:
    形态靠改 adapt.params + git 记历史, 平台靠本表 (docs §5.2)。

    soc 有双重身份: ① ATC 的 --soc ② 平台探测的匹配键 (须与
    torch.npu.get_device_properties(device).name 返回值逐字一致)。
    """
    soc: str
    aicore_num: Optional[str] = None           # om_acl → ATC --aicore_num; ge_session → 运行期限核
    custom_ops: list = field(default_factory=list)   # list[SetupEntry] 自定义算子 (加载 adapter 前)
    passes: list = field(default_factory=list)       # list[SetupEntry] fusion pass (ATC 编译前)
    max_seq_len: Optional[int] = None          # 覆盖 graph.dynamic.max_seq_len (UB/L2 尺寸随平台变)


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
    type: str = "om_acl"            # om_acl | ge_session   (平台无关, 留在顶层)
    aicore_num: Optional[str] = None   # 由 resolve_platform 从 profile 填入; yaml 不再顶层声明


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
        soc            ← model.soc            (resolve_platform 从选中的 profile 摊平)
        manifest       ← <model_dir>/io/manifest.json  (write_manifest 的固定约定)
        report_dir     ← <model_dir>/results
        请求池目录      ← <model_dir>/io/pool
        aicore_num     ← backend.aicore_num   (同上, 来自 profile; ge_session 的运行期限核)
        --batch/--prune-tokens/--prefix ← inputs.batch_size / adapt.params (bench._form_args 注入)
    负载分布 (长度分布/词表上界/套数) 属**模型专属脚本**的口径, 写死在 pool.script 的
    argparse 默认值里, 配置只在要覆盖时写 pool.args。

    **不含精度开关**: 性能跑不落盘输出 (D2H 污染延迟), 精度只由 verify.enabled 驱动的那两道门
    度量 (run.sh 的 reference + compare) — 一个变量只由一处度量, 不设第二个开关。

    多场景并存 (同一形态要随机负载 + 固定 shape + 长序列压测三份报告) 目前不支持 —
    真出现该需求时再拆回独立 scenario 文件 (形状可从 git 历史的 bench/varlen.yaml 取)。
    归档名 = export_name(cfg) + "-bench" (含平台与形态后缀), 故**不设 scenario 字段**:
    平台/形态的区分由产物名承担, 两平台两形态的归档互不撞名。
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
    passes: list = field(default_factory=list)        # list[SetupEntry] — resolve_platform 摊平填入
    custom_ops: list = field(default_factory=list)    # list[SetupEntry] — resolve_platform 摊平填入
    backend: BackendCfg = field(default_factory=BackendCfg)
    verify: VerifyCfg = field(default_factory=VerifyCfg)
    bench: BenchCfg = field(default_factory=BenchCfg)
    platforms: dict = field(default_factory=dict)     # 短名 → PlatformProfile (yaml 声明)
    platform: str = ""              # resolve_platform 选中的 profile 短名 (进产物名/provenance)
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


def _platforms(raw) -> dict:
    """yaml 的 platforms: 段 → {短名: PlatformProfile}。**强 schema**: 未知键硬失败。

    为什么这里严格而 _sub 宽松: 平台 profile 是新引入的段, 没有历史包袱; 且它装的是
    **编译目标**(soc)与**要装哪些算子/pass** —— 拼错键静默忽略的后果是"以为限了核其实
    全核跑"、"以为装了 pass 其实没装", 都属于查不出来的静默错配。
    """
    valid = set(PlatformProfile.__dataclass_fields__)
    out = {}
    for name, body in (raw or {}).items():
        body = dict(body or {})
        unknown = set(body) - valid
        if unknown:
            raise ValueError(
                f"platforms.{name} 有未知键 {sorted(unknown)} — 平台 profile 是强 schema, "
                f"只认 {sorted(valid)}。形态开关 (prefix/prune_token_file) 属 adapt.params, "
                f"不放这里 (形态串行演进, 平台长期并存, 两轴不共用机制)")
        if not body.get("soc"):
            raise ValueError(
                f"platforms.{name} 缺 soc — 它既是 ATC 的 --soc, 也是平台探测的匹配键 "
                f"(须与 torch.npu.get_device_properties(device).name 逐字一致)")
        aicore = body.get("aicore_num")
        out[str(name)] = PlatformProfile(
            soc=str(body["soc"]),
            aicore_num=None if aicore is None else str(aicore),
            custom_ops=_setup_entries(body.get("custom_ops")),
            passes=_setup_entries(body.get("passes")),
            max_seq_len=body.get("max_seq_len"),
        )
    if not out:
        raise ValueError("model.yaml 缺 platforms: 段 — 至少声明一个平台 (soc 必填); "
                         "平台相关事实 (soc/custom_ops/passes/aicore_num) 已从顶层移进该段")
    return out


# 已移进 platforms.<name> 的顶层键 → 出现在 yaml 里即硬失败 (静默忽略 = 以为生效其实没有)
_MOVED_TO_PLATFORM = {
    ("model", "soc"): "platforms.<name>.soc",
    ("backend", "aicore_num"): "platforms.<name>.aicore_num",
}
_MOVED_TOP_LEVEL = {"custom_ops": "platforms.<name>.custom_ops",
                    "passes": "platforms.<name>.passes"}


def _reject_moved_keys(raw):
    for section, key in _MOVED_TO_PLATFORM:
        if key in (raw.get(section) or {}):
            raise ValueError(
                f"model.yaml 的 {section}.{key} 已移进 {_MOVED_TO_PLATFORM[(section, key)]} "
                f"— 平台相关事实按平台声明, 顶层写会被静默忽略故直接硬失败")
    for key, dest in _MOVED_TOP_LEVEL.items():
        if raw.get(key) is not None:
            raise ValueError(f"model.yaml 顶层的 {key}: 已移进 {dest} (平台相关)")


def load_config(path) -> ModelConfig:
    """解析 model.yaml → ModelConfig。model_dir = config/ 的父目录。

    **不含 device**: 用哪张卡是运行期事实 (每次运行/每台机器都可能不同), 由 CLI `--device`
    必填传入 (pipeline → load_source/write_manifest → manifest.device → C++ 运行时)。
    **不含平台选择**: platforms 表在此解析, 但选哪个由 resolve_platform 在拿到 device 后定
    (cfg.passes/custom_ops/model.soc/backend.aicore_num 在那之前是空的)。
    yaml 里多余的 `runtime:` 段会被忽略; 已移进 platforms 的顶层键则硬失败。
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    _reject_moved_keys(raw)

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
        backend=_sub(BackendCfg, raw.get("backend")),
        verify=_sub(VerifyCfg, raw.get("verify")),
        bench=bench,
        platforms=_platforms(raw.get("platforms")),
        model_dir=model_dir,
    )


def resolve_platform(cfg: ModelConfig, device=None, platform=None) -> ModelConfig:
    """选定平台 → 返回把该 profile **摊平**进 cfg 的新 ModelConfig (原 cfg 不改)。

    平台由谁定:
        platform (CLI --platform) 优先 — 交叉编译 / 无卡机器上解析配置的逃生口
        否则探测: torch.npu.get_device_properties(device).name 去 platforms 表按 soc 匹配
    匹配不到 → 硬失败并列出可选项 (静默用错 soc 会让 ATC 产出跑不起来的 OM)。

    摊平后下游 (backend.compile_graph / write_manifest / bench) 仍读 cfg.model.soc、
    cfg.backend.aicore_num、cfg.passes、cfg.custom_ops, 不需要知道"平台"这一层。
    """
    if platform is not None:
        name = platform
        if name not in cfg.platforms:
            raise ValueError(f"--platform {name!r} 不在 model.yaml 的 platforms 表里 "
                             f"(可选: {sorted(cfg.platforms)})")
    else:
        if device is None:
            raise ValueError("resolve_platform 需要 device (探测平台) 或 platform (显式指定)")
        import torch
        import torch_npu  # noqa: F401  torch.npu 依赖

        soc = torch.npu.get_device_properties(device).name
        hits = [n for n, p in cfg.platforms.items() if p.soc == soc]
        if not hits:
            raise ValueError(
                f"device {device} 的 soc {soc!r} 不在 model.yaml 的 platforms 表里 "
                f"(已声明: {[(n, p.soc) for n, p in cfg.platforms.items()]}) — "
                f"补一个 profile, 或用 --platform 显式指定")
        if len(hits) > 1:
            raise ValueError(f"platforms 表里 soc={soc!r} 对应多个 profile {hits} — soc 须唯一")
        name = hits[0]

    p = cfg.platforms[name]
    graph = cfg.graph
    if p.max_seq_len is not None:
        graph = GraphCfg(dynamic=replace(cfg.graph.dynamic, max_seq_len=p.max_seq_len))
    return replace(cfg,
                   platform=name,
                   model=replace(cfg.model, soc=p.soc),
                   custom_ops=list(p.custom_ops),
                   passes=list(p.passes),
                   backend=replace(cfg.backend, aicore_num=p.aicore_num),
                   graph=graph)


def load_target_tokens(json_path):
    """加载 target token JSON ({"token_ids": [...]}), 返回 token_ids 列表。"""
    with open(json_path) as f:
        data = json.load(f)
    token_ids = data["token_ids"]
    assert len(token_ids) > 0, "token_ids 不能为空"
    return token_ids


def export_name(cfg: ModelConfig) -> str:
    """产物名 = 模型名 + **平台** + **形态后缀** (AIR/OM/bundle/manifest 都以此命名)。

    凡是影响图结构或编译目标的开关都要编进名字, 否则切换时会**静默覆盖**上一种的产物
    (你以为在做 A/B, 其实基线已经被冲掉)。新增开关时记得在这里加后缀。

    平台后缀 = resolve_platform 选中的 profile 短名。平台长期并存 (910_9382 与 A5 都进
    回归), 故两平台产物必须互不覆盖; 未解析 (cfg.platform 为空) 则不加 —— 那意味着 cfg
    还没过 resolve_platform, 而 pipeline/bench 一定先调它。
    """
    name = cfg.model.name
    if cfg.platform:
        name += f"-{cfg.platform}"
    if cfg.adapt.params.get("prefix"):
        name += "-prefix"
    if cfg.adapt.params.get("prune_token_file"):
        name += "-prune"
    return name


# ==================== 构建指纹 (复用磁盘产物前的一致性校验) ====================
#
# --skip export / --skip compile 会复用磁盘上的 AIR/io_spec/bundle/OM。产物名 (export_name)
# 已编码 platform+prefix+prune, 故**换平台或换形态不会误复用** —— 但还有一批量不进名字,
# 改了它们而复用旧产物就是静默错配:
#     max_seq_len   图常量长度 (RoPE 表 / 因果 mask) → 改它等于换了张图
#     inputs.*      bundle 的具体 shape 与 .bin 数据
#     soc/aicore_num/backend.type   OM 的编译目标与限核
# 指纹按**产物**存 (<air>/<name>.build.json, 与 AIR 同名同目录), 不是存进 io/manifest.json ——
# manifest 是单份且每次运行覆盖, A/B 交替时会拿 B 的指纹去校验 A 的产物, 误拒。

def build_fingerprint(cfg: ModelConfig, dtype=None) -> dict:
    """决定磁盘产物能否复用的全部配置事实 (export_name 已编码的部分也一并记, 便于报错时对照)。"""
    return {
        "export_name": export_name(cfg),
        "platform": cfg.platform,
        "soc": cfg.model.soc,
        "adapt_params": dict(cfg.adapt.params),
        "max_seq_len": cfg.graph.dynamic.max_seq_len,
        "inputs": {"batch_size": cfg.inputs.batch_size, "seq_len": cfg.inputs.seq_len,
                   "prefix_len": cfg.inputs.prefix_len, "seed": cfg.inputs.seed},
        "backend": {"type": cfg.backend.type, "aicore_num": cfg.backend.aicore_num},
        "dtype": str(dtype) if dtype is not None else None,
    }


def build_record_path(air_path) -> str:
    """<air>/<name>.air → <air>/<name>.build.json (与产物同名同目录, 故天然按产物隔离)。"""
    return os.path.splitext(air_path)[0] + ".build.json"


def write_build_record(air_path, cfg: ModelConfig, dtype=None) -> str:
    """导出时落指纹 (+ git commit / 时间戳, 便于事后归因), 返回记录路径。"""
    rec = build_fingerprint(cfg, dtype)
    try:
        import datetime
        import subprocess
        rec["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        rec["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
    except Exception:
        pass                                        # 归因信息收集不到就跳过, 不影响校验
    path = build_record_path(air_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    return path


def check_build_record(air_path, cfg: ModelConfig, dtype=None, what="AIR/io_spec"):
    """复用磁盘产物前校验指纹; 不一致 → **硬失败** (GE_ALLOW_STALE_ARTIFACT=1 放行并 WARN)。

    为什么不 WARN 了事: 复用错产物表现为 tiling 崩或精度全错, 且**不指向真因** —— 与
    graph.from_air 拒绝按位置硬配 io_spec 是同一类静默错误, 宁可当场失败。
    记录缺失 (产物早于本机制) 只 WARN: 无从校验, 但不应堵死既有产物。
    """
    path = build_record_path(air_path)
    if not os.path.exists(path):
        print(f"[config][WARN] {what} 无构建指纹 ({os.path.basename(path)} 不存在) → 无从校验"
              f"是否仍与当前配置一致 (产物早于指纹机制? 重跑导出即可补上)")
        return
    with open(path) as f:
        old = json.load(f)
    new = build_fingerprint(cfg, dtype)
    diff = {k: (old.get(k), v) for k, v in new.items() if old.get(k) != v}
    if not diff:
        return

    detail = "\n".join(f"    {k}: 盘上={a!r} → 当前={b!r}" for k, (a, b) in diff.items())
    msg = (f"{what} 的构建指纹与当前配置不一致 — 复用它会静默错配:\n{detail}\n"
           f"  产物: {air_path}\n"
           f"  指纹: {path}\n"
           f"→ 去掉相应的 --skip 重新构建; 确知无影响可 GE_ALLOW_STALE_ARTIFACT=1 放行")
    if os.environ.get("GE_ALLOW_STALE_ARTIFACT") != "1":
        raise RuntimeError(msg)
    print(f"[config][WARN] GE_ALLOW_STALE_ARTIFACT=1 → 放行陈旧产物\n{msg}")


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
        # 自证平台 (C++ 侧 Manifest::Load 逐键读, 忽略未知字段 → 加键不破坏运行时)。
        # 权威指纹在 <air>/<name>.build.json (check_build_record), 这里只为报告/人工归因。
        "platform": cfg.platform,
        "soc": cfg.model.soc,
    }
    io_dir = os.path.join(base, "io")
    os.makedirs(io_dir, exist_ok=True)
    manifest_path = os.path.join(io_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest_path
