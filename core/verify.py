"""验证模块 — golden 生成 + bundle 落盘 + 精度比对 (docs §10/§11)。

验证流三段 (docs §10):
    Python: golden = NPU-eager 前向 (patched 模型 is_compiling()=False, 走 torch_npu 算子)
            save_bundle 把 inputs/golden .bin + 具体 shape + provenance 落盘 → bundle.json
    C++:    backend 跑 OM/GeSession on inputs → outputs (.bin)
    Python: compare_bundle(bundle, outputs_dir) → report (复用 tools/compare.py, 不重写比对数学)

两层 shape (docs §6): io_spec 记动态维声明 (-1), bundle 记**具体 shape** (驱动 .bin 加载)。
bundle.json schema 严格按 docs §5.4: {inputs:[{logical,shape,file}], golden:{...}, provenance:{...}}。
golden 在 trace 之前算 (docs §10); seed 等 provenance 由调用方塞入, 本模块只负责原样写盘。
"""

import json
import os

import numpy as np


class Verifier:
    """golden 生成 / bundle 落盘 / 精度比对 (docs §11 verify.py 接口)。

    用法:
        v = Verifier()
        golden = v.golden(model, inputs)                          # NPU-eager logits (在 NPU)
        bundle = v.save_bundle(dir, inputs, golden, io_spec, prov)  # → bundle.json 路径
        report = v.compare_bundle(bundle, outputs_dir, dtype)      # 复用 tools/compare.py
    """

    def __init__(self, rtol=1e-3, atol=1e-5):
        # 比对默认容忍度 (与 tools/compare.py 一致); compare 可逐次覆盖
        self.rtol = rtol
        self.atol = atol

    # ---- golden ----

    def golden(self, model, inputs):
        """eager 前向取 NPU-eager golden (返回 logits 张量, 在 NPU 上)。

        inputs 顺序 = 适配模型 forward 入参 (input_ids, position_ids, actual_seq_lengths)。
        patched 模型 eager 下 is_compiling()=False, 走 torch_npu 算子, 与 OM/GeSession 的
        graph 路径同源 — 对比隔离出"编译"这一个变量 (docs §10)。设备由调用方管理,
        模型已在 NPU; golden 应在 trace 之前算。
        """
        import torch                      # 延迟导入: 本模块其余部分 (bundle/compare) 不依赖 torch

        with torch.no_grad():
            return model(*inputs)

    # ---- bundle ----

    def save_bundle(self, dir, inputs, golden, io_spec, provenance, logical_order=None):
        """落盘 inputs/golden .bin + bundle.json (schema 见 docs §5.4), 返回 bundle.json 路径。

        每个输入张量 → <dir>/inputs/<logical>.bin; golden → <dir>/golden_logits.bin。
        logical 名取自 logical_order (**forward 入参序**, 与 inputs 一一对应); 缺省时退回
        io_spec.inputs 的位置序 — 但 io_spec 按**图 Data 序**排列 (docs §5.3), 两序不同时
        (如 qwen2.5-0.5b) 必须传 logical_order, 否则 .bin 会贴错 logical 标签。
        记录**具体 shape** (张量真实 shape, 不是 io_spec 的 -1) + file 相对路径 (相对 bundle
        目录)。provenance 原样写入 (调用方负责塞 seed/model/soc/dtype/git_commit/版本/时间戳等)。

        golden=None (verify.enabled=false) 时只落 inputs, bundle.golden 写 null —
        运行时仍可跑 (部署态), compare 阶段自动跳过 (pipeline 用 bundle_has_golden 判断)。
        """
        inputs_dir = os.path.join(dir, "inputs")
        os.makedirs(inputs_dir, exist_ok=True)

        in_entries = []
        for i, t in enumerate(inputs):
            logical = (logical_order[i] if logical_order and i < len(logical_order)
                       else _logical(io_spec.inputs, i, f"input_{i}"))
            rel = os.path.join("inputs", f"{logical}.bin")
            _dump(t, os.path.join(dir, rel))
            in_entries.append({"logical": logical, "shape": _shape(t), "file": rel})

        g_entry = None
        if golden is not None:
            g_rel = "golden_logits.bin"
            _dump(golden, os.path.join(dir, g_rel))
            g_entry = {"logical": _logical(io_spec.outputs, 0, "logits"),
                       "shape": _shape(golden), "file": g_rel}

        bundle = {
            "inputs": in_entries,
            "golden": g_entry,
            "provenance": provenance or {},
        }
        bundle_path = os.path.join(dir, "bundle.json")
        with open(bundle_path, "w") as f:
            json.dump(bundle, f, indent=2, ensure_ascii=False)
        return bundle_path

    # ---- compare ----

    def compare_bundle(self, bundle_path, outputs_dir, dtype="float16",
                       rtol=None, atol=None, verbose=True):
        """比对 C++ 运行时输出与 bundle golden — 验证流第三段 (docs §10)。

        输入:
            bundle_path  verification/bundle.json (golden 的**具体 shape** + file)
            outputs_dir  C++ 运行时输出目录 (output_<i>.bin + outputs.json)
            dtype        golden 的 dtype — 取自 io_spec (bundle 不记 dtype, docs §5.4)
        输出项按 logical 与 golden 匹配 (匹配不到取第 0 项), 其 dtype/shape 取自
        outputs.json (运行时实测)。返回 compare.py 的指标 dict (pass_overall 为判定)。
        """
        from tools.compare import PrecisionComparator, load_file

        rtol = self.rtol if rtol is None else rtol
        atol = self.atol if atol is None else atol

        with open(bundle_path) as f:
            golden_entry = (json.load(f).get("golden") or {})
        if not golden_entry.get("file"):
            raise ValueError(
                f"bundle 无 golden 项 (verify.enabled=false 时只落 inputs): {bundle_path} — "
                f"无比对基准, 请开启 verify 重新导出, 或只做 run 不做 compare")

        index_path = os.path.join(outputs_dir, "outputs.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"运行时输出索引不存在: {index_path} (先跑 run 阶段)")
        with open(index_path) as f:
            outs = json.load(f).get("outputs") or []
        if not outs:
            raise ValueError(f"运行时无输出: {index_path}")

        entry = next((o for o in outs if o.get("logical") == golden_entry.get("logical")), outs[0])
        bundle_dir = os.path.dirname(os.path.abspath(bundle_path))

        g = load_file(os.path.join(bundle_dir, golden_entry["file"]), dtype,
                      tuple(golden_entry.get("shape") or ()) or None)
        t = load_file(os.path.join(outputs_dir, entry["file"]), entry.get("dtype") or dtype,
                      tuple(entry.get("shape") or ()) or None)
        # 回归门不做 flatten/截断兜底: 形状不一致本身就是失败 (错序/错 shape 也可能
        # 被截断后 cosine PASS → 静默放行)。人工排查用 tools/compare.py CLI。
        if g.shape != t.shape:
            raise ValueError(
                f"形状不一致, 拒绝比对: golden {g.shape} vs 运行时输出 {t.shape} "
                f"(golden={golden_entry.get('file')}, output={entry.get('file')})")

        return PrecisionComparator.compare_and_report(
            g, t, target_name=entry.get("logical") or "outputs",
            rtol=rtol, atol=atol, verbose=verbose)


# ---- 内部工具 ----

def _logical(nodes, i, default):
    """取 io_spec nodes 第 i 项的 logical 名; 越界/空则退回 default。"""
    if nodes and i < len(nodes):
        name = getattr(nodes[i], "logical", "") or ""
        return name or default
    return default


def _shape(t):
    """张量具体 shape → list[int] (bundle 记真实 shape, 非 io_spec 的 -1)。"""
    return [int(s) for s in tuple(t.shape)]


def _dump(t, path):
    """张量 → .cpu().numpy().tofile(path) (原始字节, 保留 dtype)。"""
    arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    arr.tofile(path)


def bundle_has_golden(bundle_path) -> bool:
    """bundle 里是否有 golden 项 (verify.enabled=false 时只落 inputs, golden 为 null)。"""
    try:
        with open(bundle_path) as f:
            return bool((json.load(f).get("golden") or {}).get("file"))
    except (OSError, ValueError):
        return False


def collect_provenance(seed=None, model=None, soc=None, dtype=None, **extra):
    """便利 helper: 收集 git_commit / torch_npu·transformers 版本 / 时间戳, 并合并入给定字段。

    save_bundle **不依赖**本函数 — 调用方可完全自行构造 provenance dict (docs §10:
    provenance 内容由调用方决定)。本 helper 只为方便, 收集不到的项静默跳过 (不编造)。
    """
    import datetime
    import subprocess

    prov = {}
    if seed is not None:
        prov["seed"] = seed
    if model is not None:
        prov["model"] = model
    if soc is not None:
        prov["soc"] = soc
    if dtype is not None:
        prov["dtype"] = dtype
    try:
        prov["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        pass
    try:
        import torch_npu
        prov["torch_npu"] = getattr(torch_npu, "__version__", "unknown")
    except Exception:
        pass
    try:
        import transformers
        prov["transformers"] = transformers.__version__
    except Exception:
        pass
    prov["timestamp"] = datetime.datetime.now().isoformat()
    prov.update(extra)
    return prov
