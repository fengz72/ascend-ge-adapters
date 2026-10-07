"""验证模块 — golden 生成 + bundle 落盘 + 精度比对 (docs §10/§11)。

验证流三段 (docs §10):
    Python: golden = NPU-eager 前向 (patched 模型 is_compiling()=False, 走 torch_npu 算子)
            save_bundle 把 inputs/golden .bin + 具体 shape + provenance 落盘 → bundle.json
    C++:    backend 跑 OM/GeSession on inputs → outputs (.bin)
    Python: compare_bundle(bundle, outputs_dir) → report (复用 tools/compare.py, 不重写比对数学)

golden 只隔离出"**编译**"这一个变量 — 它来自打过 patch 的同一个模型, 适配写错 (错 mask /
错位置编码 / 错末 token 索引) 时两边一起错, 比对照样 PASS。所以还有第四段:
    Python: reference = **原版 (未 patch) HF** 逐请求前向 → compare_reference(golden, reference)
它隔离出"**适配**"这一个变量, 必须在 adapt 之前算 (patch 是进程级类属性)。

两层 shape (docs §6): io_spec 记动态维声明 (-1), bundle 记**具体 shape** (驱动 .bin 加载)。
bundle.json schema 严格按 docs §5.4: {inputs:[{logical,shape,file}], golden:{...}, provenance:{...}}。
golden 在 trace 之前算 (docs §10); seed 等 provenance 由调用方塞入, 本模块只负责原样写盘。
"""

import json
import os

import numpy as np

# 参考比对 (跨实现: NPU 融合算子 vs 原版 HF eager) 的门限 — 比同源比对
# (compare_bundle 的 0.9999/0.01) 松, 但只松到"数值噪声"的量级。目的是抓**语义错误**:
# 错 mask / 错位置 / 错末 token 索引 / 剪裁错列会让 cosine 掉到 0.9 以下、rel_l2 上到 0.5+,
# 而 fp16 融合算子的实测噪声是 cosine 0.999999 / rel_l2 1.5e-3 (qwen2.5-0.5b prefix+prune 形态)
# —— 门限与噪声之间留 ~2 个数量级, 与真实 bug 之间留 ~1 个数量级。
REF_COSINE_MIN = 0.999
REF_REL_L2_MAX = 0.02

# 编译比对 (同源: C++ 运行时输出 vs patched-eager golden) 的门限 — 两边同出一个模型,
# 只差 AIR→OM/GeSession 的编译与图序喂入, 故比跨实现的 reference 门紧一档。
CMP_COSINE_MIN = 0.9999
CMP_REL_L2_MAX = 0.01

# 四个门限都是**规范默认值**: model.yaml 的 verify 段可逐项覆盖 (VerifyCfg 里默认 None
# = 用这里的值), 换 dtype(bf16)/更大词表/更长序列时噪声量级会变, 改 yaml 即可, 不动框架。


class Verifier:
    """golden 生成 / 原版参考 / bundle 落盘 / 精度比对 (docs §11 verify.py 接口)。

    用法:
        v = Verifier()
        ref = v.reference(raw_model, adapter, inputs)                # 原版 HF, adapt **之前**
        golden = v.golden(model, inputs)                             # NPU-eager logits (在 NPU)
        v.compare_reference(golden, ref, path)                       # 适配是否正确
        bundle = v.save_bundle(dir, inputs, golden, io_spec, prov)   # → bundle.json 路径
        report = v.compare_bundle(bundle, outputs_dir, dtype)        # 编译是否正确
    """

    def __init__(self, rtol=1e-3, atol=1e-5,
                 ref_cosine_min=REF_COSINE_MIN, ref_rel_l2_max=REF_REL_L2_MAX,
                 cmp_cosine_min=CMP_COSINE_MIN, cmp_rel_l2_max=CMP_REL_L2_MAX):
        # 比对默认容忍度 (与 tools/compare.py 一致); compare 可逐次覆盖
        self.rtol = rtol
        self.atol = atol
        # 门① reference (跨实现) 与门② compare (同源) 的判定门限 — 均可由 model.yaml
        # 的 verify 段覆盖 (pipeline 构造 Verifier 时透传), 换 dtype/词表规模时改 yaml 即可
        self.ref_cosine_min = ref_cosine_min
        self.ref_rel_l2_max = ref_rel_l2_max
        self.cmp_cosine_min = cmp_cosine_min
        self.cmp_rel_l2_max = cmp_rel_l2_max

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

    # ---- 原版参考 (隔离"适配"这一个变量) ----

    def reference(self, model, adapter, inputs):
        """**原版 (未 patch)** 模型逐请求前向 → [N, vocab'] 参考 logits。

        调用时机是硬约束: 必须在 `adapter.adapt()` **之前** — patch 是类级 monkey-patch
        (进程全局, 见 core/adapter.py), adapt 之后同进程里任何同架构实例都会走 patched
        forward, 参考就退化成"自己比自己"。

        逐请求还原 (unpack_requests) 与输出列剪裁 (reference_columns) 都是模型专属知识,
        归 adapter; 本函数只负责通用的"一条请求一次前向 + 取末 token"。
        adapter 未实现 unpack_requests → 返回 None, 调用方 WARN 并跳过。
        """
        requests = adapter.unpack_requests(inputs)
        if not requests:
            print("[verify][WARN] adapter 未实现 unpack_requests → 跳过原版参考比对 "
                  "(golden 与图输出同源, 只能证明'编译'正确, 证明不了'适配'正确)")
            return None

        import torch

        outs = []
        with torch.no_grad():
            for ids, pos in requests:
                out = model(input_ids=ids.view(1, -1), position_ids=pos.view(1, -1),
                            use_cache=False)
                logits = getattr(out, "logits", None)
                if logits is None:
                    logits = out[0]
                outs.append(logits[0, -1, :])          # 末 token = 适配后图的输出口径
        ref = torch.stack(outs)

        columns = adapter.reference_columns()
        if columns:
            idx = torch.as_tensor(list(columns), dtype=torch.long, device=ref.device)
            ref = ref.index_select(-1, idx)            # lm_head 剪裁 → 只比保留的那些列
        print(f"[verify] 原版参考: {len(requests)} 条请求 → {tuple(ref.shape)} {ref.dtype}"
              + (f" (剪裁到 {len(columns)} 列)" if columns else ""))
        return ref

    def compare_reference(self, golden, reference, path=None, verbose=False):
        """比对 patched-eager golden 与原版参考 → report (门限比 compare_bundle 松, 见模块头)。

        形状不一致直接抛错 (剪裁列数/请求数对不上就是适配 bug, 不做 flatten/截断兜底)。
        path 非空时把 report 写成 json (io/reference.json)。
        """
        from tools.compare import PrecisionComparator

        g = _to_numpy(golden)
        r = _to_numpy(reference)
        if g.shape != r.shape:
            raise ValueError(
                f"参考比对形状不一致: golden {g.shape} vs 原版参考 {r.shape} "
                "(请求数或输出列数对不上 — 检查 unpack_requests / reference_columns)")

        report = PrecisionComparator.compare_and_report(
            r, g, target_name="适配后 eager (golden)",
            cosine_min=self.ref_cosine_min, rel_l2_max=self.ref_rel_l2_max, verbose=verbose)
        report = dict(report)
        report["gate"] = {"cosine_min": self.ref_cosine_min, "rel_l2_max": self.ref_rel_l2_max}
        report["shape"] = list(g.shape)

        verdict = "PASS" if report["pass_overall"] else "FAIL"
        print(f"[verify] 原版参考比对 {verdict}: cosine={report['cosine_similarity']:.8f} "
              f"rel_l2={report['relative_l2_error']:.3e} "
              f"(门限 cosine>{self.ref_cosine_min}, rel_l2<{self.ref_rel_l2_max})")
        if path:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w") as f:
                json.dump({"schema": "ge-reference/1", **report}, f, indent=2,
                          ensure_ascii=False, default=_json_default)
        return report

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
            bundle_path  io/bundle.json (golden 的**具体 shape** + file)
            outputs_dir  C++ 运行时输出目录 (output_<i>.bin + outputs.json)
            dtype        golden 的 dtype — 取自 io_spec (bundle 不记 dtype, docs §5.4)
        输出项按 logical 与 golden 匹配 (匹配不到取第 0 项), 其 dtype/shape 取自
        outputs.json (运行时实测)。返回 compare.py 的指标 dict (pass_overall 为判定)。
        判定门限用门② 的 self.cmp_cosine_min/cmp_rel_l2_max (同源, 默认 0.9999/0.01;
        可由 model.yaml 的 verify 段覆盖)。
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
            rtol=rtol, atol=atol,
            cosine_min=self.cmp_cosine_min, rel_l2_max=self.cmp_rel_l2_max,
            verbose=verbose)


# ---- 内部工具 ----

def _to_numpy(t):
    """张量 → numpy (比对统一在 CPU float 上做; 已是 ndarray 则原样)。"""
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


def _json_default(obj):
    """json.dump 的 default: metrics 里有 numpy 标量与 max_diff_position 元组。"""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return [_json_default(x) for x in obj]
    return str(obj)


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
    _to_numpy(t).tofile(path)


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
