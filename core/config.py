"""模型配置 (YAML 人工声明) 解析 + 运行时 manifest (JSON) 生成。

契约见 docs/architecture.md §5:
    model.yaml   人工声明 (source/adapt/inputs/graph/passes/backend/verify; 不含 device)
    manifest.json 生成的 C++ 运行时契约 (deploy/manifest.json)
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
    type: str                       # name | torch | onnx
    ref: str = ""                   # name=hub id; torch=权重目录; onnx=.onnx 路径
    module: str = ""                # torch 源码形态: 模块文件
    class_name: str = ""            # torch 源码形态: 模型类名
    weights: str = ""               # torch 源码形态: 权重路径


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
    format: str = "air"             # air | onnx
    dynamic: DynamicCfg = field(default_factory=DynamicCfg)


@dataclass
class BackendCfg:
    type: str = "om_acl"            # om_acl | ge_session
    aicore_num: Optional[str] = None


@dataclass
class VerifyCfg:
    enabled: bool = True


@dataclass
class ModelConfig:
    model: ModelMeta
    source: SourceCfg
    adapt: AdaptCfg = field(default_factory=AdaptCfg)
    inputs: InputsCfg = field(default_factory=InputsCfg)
    graph: GraphCfg = field(default_factory=GraphCfg)
    passes: list = field(default_factory=list)
    backend: BackendCfg = field(default_factory=BackendCfg)
    verify: VerifyCfg = field(default_factory=VerifyCfg)
    model_dir: str = ""             # 配置文件所在模型目录 (load 时填入)


def _sub(cls, d: dict, **renames):
    """从 dict 构造 dataclass, 缺失字段用默认值; renames 映射 yaml 键→字段名。"""
    d = dict(d or {})
    for yaml_key, field_name in renames.items():
        if yaml_key in d:
            d[field_name] = d.pop(yaml_key)
    valid = {f for f in cls.__dataclass_fields__}
    return cls(**{k: v for k, v in d.items() if k in valid})


def load_config(path) -> ModelConfig:
    """解析 model.yaml → ModelConfig。model_dir = config/ 的父目录。

    **不含 device**: 用哪张卡是运行期事实 (每次运行/每台机器都可能不同), 由 CLI `--device`
    必填传入 (pipeline → load_source/write_manifest → manifest.device → C++ 运行时)。
    yaml 里多余的 `runtime:` 段会被忽略。
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    model_dir = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    graph_d = dict(raw.get("graph") or {})
    dynamic = _sub(DynamicCfg, graph_d.pop("dynamic", {}))

    return ModelConfig(
        model=_sub(ModelMeta, raw.get("model")),
        source=_sub(SourceCfg, raw.get("source"), **{"class": "class_name"}),
        adapt=_sub(AdaptCfg, raw.get("adapt")),
        inputs=_sub(InputsCfg, raw.get("inputs")),
        graph=GraphCfg(format=graph_d.get("format", "air"), dynamic=dynamic),
        passes=list(raw.get("passes") or []),
        backend=_sub(BackendCfg, raw.get("backend")),
        verify=_sub(VerifyCfg, raw.get("verify")),
        model_dir=model_dir,
    )


def load_target_tokens(json_path):
    """加载 target token JSON ({"token_ids": [...]}), 返回 token_ids 列表。"""
    with open(json_path) as f:
        data = json.load(f)
    token_ids = data["token_ids"]
    assert len(token_ids) > 0, "token_ids 不能为空"
    return token_ids


def load_adapter(cfg: ModelConfig):
    """importlib 从 <model_dir>/model.py 加载 adapt.adapter_class 并按 params 实例化。

    params 直接作为 adapter 构造 kwargs (基类 __init__ 收 **params, 故最小 adapter
    无需自定义构造); 特殊键 prune_token_file (相对 model_dir) **仅在声明时**才被载入成
    prune_tokens 列表传入 (文件 I/O 在配置层, adapter 只收 list)。
    """
    import importlib.util
    model_py = os.path.join(cfg.model_dir, "model.py")
    spec = importlib.util.spec_from_file_location(f"_ge_adapter_{cfg.model.name}", model_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, cfg.adapt.adapter_class)

    params = dict(cfg.adapt.params)
    ptf = params.pop("prune_token_file", None)
    if ptf:
        params["prune_tokens"] = load_target_tokens(os.path.join(cfg.model_dir, ptf))
    return cls(**params)


def write_manifest(cfg: ModelConfig, graph_path, om_path, io_spec_path,
                   bundle_path=None, base_dir=None, device=None) -> str:
    """生成 <base_dir>/deploy/manifest.json (C++ 运行时契约)。

    路径相对 base_dir (默认 model_dir), 保证可移植; --work-dir 调试时 base_dir
    传 work_dir, 产物与 manifest 同根。passes_vendor 存逻辑名 (model.name),
    运行时展开为 $ASCEND_HOME_PATH/opp/vendors/<name> — 不内联绝对路径。

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
        "passes_vendor": cfg.model.name,
        "bundle": rel(bundle_path),
    }
    deploy_dir = os.path.join(base, "deploy")
    os.makedirs(deploy_dir, exist_ok=True)
    manifest_path = os.path.join(deploy_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest_path
