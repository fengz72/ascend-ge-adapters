"""
GE 模型适配基类 — 统一加载流程 + patch/restore 机制

子类约定 (替换实现写在各模型的 model.py 里):
    1. MODELING    — transformers 的 modeling 模块
    2. patch_specs — [(target, 属性名, 新实现)] 映射表

注意: 替换实现必须是模型文件里的模块级普通函数, 第一个参数是
被替换的 transformers 实例 (RMSNorm/attention 层), 不能写成
GeModelAdapter 的实例方法, 否则 self 错绑。

monkey-patch 作用于 modeling 模块的类/函数, 是进程级全局的;
不同 modeling 模块的模型可共存, 同一 modeling 模块的实例共享 patch。

torch/transformers 只在 load() 里延迟导入 — 基类本身 (patch/setup/adapt) 不依赖它们,
故 `import core.adapter` 不需要 NPU 栈 (离线单测/CI 可直接实例化 adapter)。
"""


class GeModelAdapter:
    """GE 适配基类。

    用法:
        class Qwen25Adapter(GeModelAdapter):
            MODELING = modeling_qwen2
            def patch_specs(self): ...

        model = Qwen25Adapter().load(path, dtype=torch.float16)
    """

    MODELING = None   # 子类指定: transformers.models.<arch>.modeling_<arch>
    USE_CACHE = False  # 能力声明: load 写入 model.config.use_cache, patched forward 透传各层
                       # (真正支持 KV cache 需结构不同的 forward, 非 翻标志 即得)

    def __init__(self, **params):
        """params = model.yaml 的 adapt.params (子类按需显式取用; 基类只存不解释)。

        基类收 **params 是为了让 load_adapter 用同一套调用约定实例化任意 adapter —
        最小 adapter 不写 __init__ 也能被加载 (docs §13.8 "onboarding 零框架改动")。
        """
        self.params = params
        self.model = None
        self._originals = {}

    def load(self, model_path, dtype=None, **setup_kwargs):
        """便捷入口: from_pretrained 加载 + adapt (form ① 模型名/权重目录)。

        设备由调用方管理 (先 torch_npu.npu.set_device(n)), 模型加载到当前 NPU。
        dtype 缺省 torch.float16; use_cache 是能力声明 (USE_CACHE 类属性), 不做参数 —
        默认 patched attention 不支持 KV cache, 误开只会得到静默空 cache。
        **setup_kwargs 透传给 setup (如 max_seq_len)。
        """
        import torch
        from transformers import AutoModelForCausalLM

        dtype = torch.float16 if dtype is None else dtype
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).npu()
        return self.adapt(model, **setup_kwargs)

    def adapt(self, model, **setup_kwargs):
        """对已加载模型应用 patch + setup (source 加载后调用; 模型需已在 NPU)。

        与 load 的区别: load 自带 from_pretrained (form ①); adapt 接收任意来源
        已加载的模型 (form ② importlib 实例化等), 只做适配。
        """
        self.apply_patches()
        self.model = model
        self.model.eval()
        self.model.config.use_cache = self.USE_CACHE
        self.setup(self.model, **setup_kwargs)
        return self.model

    def setup(self, model, **kwargs):
        """实例级适配钩子 (默认 no-op), 由 load/adapt 在权重加载后自动调用。

        做"加载后、可用/可 trace 前"需要的全部实例改造: 结构手术 (如 lm_head
        剪裁、量化)、常量注入 (mask/预计算表)、模式标志等。运行期变量
        (如 actual_seq_lengths) 不在此处 — 由 patched forward 以入参路由。
        """

    def patch_specs(self):
        """返回 [(target, 属性名, 新实现)], 子类必须实现。"""
        raise NotImplementedError

    # ---- 输入接口 (模型专属: 该模型适配后的图输入怎么构造) ----

    def build_inputs(self, model, **input_kwargs):
        """生成一组输入张量 (顺序 = patched forward 入参序), 子类实现。"""
        raise NotImplementedError

    def mark_dynamic(self, inputs, **input_kwargs):
        """标记动态维度 (默认原样返回), 子类按需实现。"""
        return inputs

    def io_input_nodes(self, inputs, **input_kwargs):
        """返回 io_spec 输入节点 list[IoNode] (logical/dtype/shape(-1 动态)/dynamic_dims),
        顺序须与 build_inputs/forward 入参一致。子类实现。

        logical 名须与 forward 入参名一致 — 导出图的 Data 节点带 `_source_name`
        (= forward 入参名), graph.from_air 据此自动完成 node↔logical 配对与图序排列。
        """
        raise NotImplementedError

    # ---- 原版参考比对 (可选能力: 证明"适配"本身正确, 不只是"编译"正确) ----

    def unpack_requests(self, inputs):
        """把图输入拆回**逐请求**的 (input_ids, position_ids) — 供原版 HF 参考前向。

        适配后的图接口通常是打包/变形过的 (varlen 拼接、prefix 内嵌、mask 省略…),
        原版模型吃不进去; 只有 adapter 知道怎么还原成"一条请求一次前向"的形态。
        返回 None (默认) = 该 adapter 不支持参考比对, verify 会 WARN 并跳过。

        返回: list[(input_ids 1D, position_ids 1D)], 顺序 = 输出的行序。
        """
        return None

    def reference_columns(self):
        """适配收窄了输出时 (如 lm_head 词表剪裁), 返回参考输出要取的列下标 list[int];
        None (默认) = 全词表逐列比对。"""
        return None

    def apply_patches(self):
        """应用 patch_specs 全部替换 (幂等, 重复调用不覆盖备份)。"""
        for target, name, fn in self.patch_specs():
            self._originals.setdefault((target, name), getattr(target, name))
            setattr(target, name, fn)

    def restore(self):
        """回滚全部 patch 为 transformers 原始实现。"""
        for (target, name), original in self._originals.items():
            setattr(target, name, original)
        self._originals.clear()
