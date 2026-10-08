"""GE 模型适配基类 — 三层结构中 L0→L1 的机制与 L1→L2 的契约。

三层 (docs §10):
    L0 原模型     source.load_source()          golden = verify.reference (逐请求原版 HF)
    L1 patched    adapter.adapt(raw)            golden = verify.golden (NPU eager)
    L2 图         exporter.trace + Graph.from_air  golden = C++ runtime 输出
两道精度门就是两条层边界: compare_reference (L1 vs L0) / compare_bundle (L2 vs L1)。

本类声明三组契约:
    ① L0→L1 适配机制    patch_specs / setup / adapt      有副作用, 进程级, **破坏性不可逆**
    ② L0↔L1 边界翻译器  unpack_requests / reference_columns
                        可选 — L0 的接口 ≠ L1 (L0 吃逐请求 HF 形态, L1 吃打包 varlen;
                        L1 还可能收窄了输出), 只有要门① 时才需实现
    ③ 图接口契约          mark_dynamic / io_input_nodes
                        两者与 patch_specs 的 forward 签名**同源**: 改 forward 签名要同步这两处
                        (io_input_nodes 的 logical 名还须与 forward 入参名逐字一致 —
                         graph.from_air 靠图 Data 节点的 _source_name 配对)

**激励不在本类**: 输入由 model.yaml 的 `inputs.script` 声明的用户脚本生成并落盘
(<base>/io/{inputs.json, inputs/*.bin}), pipeline 读回后喂给三层 —— 落盘而非进程内造,
激励才是三层真正共用的**独立产物** (L0/L1 的 golden 可脱离 export 重跑, 生成阶段不占卡)。
脚本落盘的 logical 序须与 io_input_nodes 声明的一致, pipeline 会对账 (不一致硬失败)。

注意: 替换实现必须是模型文件里的模块级普通函数, 第一个参数是被替换的 transformers 实例
(RMSNorm/attention 层), 不能写成 GeModelAdapter 的实例方法, 否则 self 错绑。

L0 与 L1 **不能在同进程共存**: patch 作用于 modeling 模块的类/函数, 是进程级全局的;
setup 的实例手术 (如 lm_head 剪裁) 也不可逆。所以 L0 的 golden 必须在 adapt **之前**算完
(pipeline 据此排序; 硬约束见 verify.reference 的 docstring)。

torch/transformers 不在本模块导入 — 基类不依赖 NPU 栈, 故 `import core.adapter` 可在
没装 torch_npu 的机器上做离线单测 (docs §14)。
"""


class GeModelAdapter:
    """GE 适配基类。

    用法 (= pipeline 的实际路径):
        adapter = Qwen25Adapter(prefix=True)
        raw     = load_source(cfg, model_dir, dtype, device)         # L0
        inputs  = <inputs.script 落盘 → tools.varlen.load_inputs>    # 激励 (三层共用)
        ref     = verify.reference(raw, adapter, inputs)             # L0 golden, 必须在 adapt 前
        model   = adapter.adapt(raw, max_seq_len=...)                # L1
    """

    MODELING = None   # 子类指定: transformers.models.<arch>.modeling_<arch>

    def __init__(self, **params):
        """params = model.yaml 的 adapt.params, 由 load_adapter 直接作为构造 kwargs 传入。

        基类收 **params 是为了让 load_adapter 用同一套调用约定实例化任意 adapter —
        最小 adapter 不写 __init__ 也能被加载 (docs §13.8 "onboarding 零框架改动")。
        子类应显式声明自己接受的键 (如 Qwen25Adapter(prefix=..., prune_tokens=...));
        基类不解释、不保存。
        """

    # ---- ① L0→L1 适配机制 (有副作用: 类级 patch 进程全局, 实例手术不可逆) ----

    def adapt(self, model, **setup_kwargs):
        """L0 → L1: 应用 patch + setup, 返回适配后的模型 (**就是传入的那个对象**, 原地改)。

        与 load_source 的分工: load_source 负责加载 (来路① from_pretrained + device),
        adapt 接收任意来源已加载的模型 (来路② importlib 实例化等), 只做适配。
        **破坏性**: 调用后 model 回不到 L0 (见模块 docstring)。
        """
        self.apply_patches()
        model.eval()
        # patched forward 不支持 KV cache (真正支持需结构不同的 forward, 非翻标志即得);
        # 误开只会得到静默空 cache, 故在此统一关掉
        model.config.use_cache = False
        self.setup(model, **setup_kwargs)
        return model

    def setup(self, model, **kwargs):
        """实例级适配钩子 (默认 no-op), 由 adapt 在 patch 之后自动调用。

        做"加载后、可用/可 trace 前"需要的全部实例改造: 结构手术 (如 lm_head
        剪裁、量化)、常量注入 (mask/预计算表)、模式标志等。运行期变量
        (如 actual_seq_lengths) 不在此处 — 由 patched forward 以入参路由。
        """

    def patch_specs(self):
        """返回 [(target, 属性名, 新实现)], 子类必须实现。"""
        raise NotImplementedError

    def apply_patches(self):
        """应用 patch_specs 全部替换 (幂等: 重复 setattr 同一个函数无副作用)。"""
        for target, name, fn in self.patch_specs():
            setattr(target, name, fn)

    # ---- ② L0↔L1 边界翻译器 (可选: 只有要门① reference 时才实现) ----

    def unpack_requests(self, inputs):
        """L1 输入 (打包) → L0 输入 (逐请求): list[(input_ids 1D, position_ids 1D)],
        顺序 = 输出的行序。

        适配后的图接口通常是打包/变形过的 (varlen 拼接、prefix 内嵌、mask 省略…),
        原版模型吃不进去; 只有 adapter 知道怎么还原成"一条请求一次前向"的形态。
        返回 None (默认) = 该 adapter 不支持门①, verify 会 WARN 并跳过。
        """
        return None

    def reference_columns(self):
        """L0 输出 → L1 输出: L1 收窄了输出时 (如 lm_head 词表剪裁) 返回参考输出要取的
        列下标 list[int]; None (默认) = 全词表逐列比对。"""
        return None

    # ---- ③ 图接口契约 (与 patch_specs 的 forward 签名同源) ----
    #
    # **激励不在此**: 输入由 model.yaml 的 inputs.script 声明的用户脚本生成并落盘
    # (tools/varlen.write_inputs → <base>/io/{inputs.json, inputs/*.bin}), pipeline 读回后
    # 喂给三层。本类只剩"图接口怎么声明"。但两者仍与 patched forward 签名同源:
    #   mark_dynamic    哪一维动态由形态决定
    #   io_input_nodes  logical 名须 == forward 入参名 (graph.from_air 靠图 Data 节点的
    #                   _source_name 配对), 且**顺序**须 == 脚本落盘的 logical 序
    #                   (pipeline 对账, 不一致硬失败 — 否则 .bin 会贴错 logical 标签)

    def mark_dynamic(self, inputs, **input_kwargs):
        """标记动态维度 (默认原样返回), 子类按需实现。"""
        return inputs

    def io_input_nodes(self, inputs, **input_kwargs):
        """返回 io_spec 输入节点 list[IoNode] (logical/dtype/shape(-1 动态)/dynamic_dims),
        顺序须与输入脚本落盘的 logical 序 (= patched forward 入参序) 一致。子类实现。

        logical 名须与 forward 入参名一致 — 导出图的 Data 节点带 `_source_name`
        (= forward 入参名), graph.from_air 据此自动完成 node↔logical 配对与图序排列。
        """
        raise NotImplementedError
