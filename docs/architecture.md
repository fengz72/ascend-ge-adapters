# Ascend GE Adapters — 架构设计

> 把客户模型（3 种形态）搬到 NPU GE 上**高效运行**（2 种后端）的 onboarding 管线。
> 本文是方案定稿，作为后续实施的依据。

## 1. 项目目标

客户给出的原始模型有三种形态，需要统一适配到 Ascend GE 上高效推理：

| 形态 | 来源 | 目标图 |
|---|---|---|
| ① 模型名 | transformers hub（如 `Qwen/Qwen2.5-0.5B`） | torch → **AIR** |
| ② PyTorch 源码 | 任意 `nn.Module`（含权重） | torch → **AIR** |
| ③ ONNX 图 | 用户给出 `.onnx` 文件 | **ONNX 原样**（不转换、不优化） |

执行有两种后端：

| 后端 | 路径 | 特点 | 能吃哪些图 |
|---|---|---|---|
| **OM/ACL**（离线） | ATC 编译图 → OM → ACL 运行时 | 编译一次部署多次，运行时开销低，OM 产物可移植 | AIR（`--framework=1`）+ ONNX（`--framework=5`） |
| **GeSession**（在线） | GeSession 直接加载图在线执行 | 在线 JIT，灵活，无离线产物，省掉 ATC（~8min）；C++ 接口 | **仅 GE 图**（AIR/pbtxt）——`ge::Graph::LoadFromFile` 不解析 ONNX |

两后端都能加载 GE pass。**形态③ ONNX 只能走 OM/ACL**：`source.type=onnx` + `backend.type=ge_session` 在 pipeline **配置期**即报错（`GeSessionBackend.compile` 亦兜底），不会拖到 C++ 运行期。

> 注：形态①②本质同一条 `from_pretrained`/`nn.Module` 加载路径；形态③已由 `tests/tiny_onnx_e2e.py` 打通（onnx → io_spec → ATC(fw=5) → OM → 部署态 run，与 onnxruntime CPU 参考一致）。Source 抽象按 YAGNI 推迟（见 §11）。

## 2. 问题矩阵

```
            ┌──────────────┐   ┌─────────────┐   ┌──────────────────────┐
  ① name ───┤              │   │             │   │ 后端1: ATC→OM→ACL     │
  ② torch ──┤ Source/Adapt ├──→│ Graph(AIR)  ├──→│ 后端2: GeSession      │──→ outputs
  ③ onnx ───┤  (onnx 原样) │   │ Graph(ONNX) │   │  (吃 AIR/ONNX + pass) │
            └──────────────┘   └─────────────┘   └──────────────────────┘
                  ↑ 模型级适配          ↑ io_spec        ↑ 图级优化(pass) + 编译选项
```

收敛点是 **Graph**：上游（源 + 适配）只负责产出图，下游（后端）只负责消费图。

## 3. 总体架构

```
Source ──[Adapt]──> Graph(AIR|ONNX) ──[Passes]──> Backend ──> outputs
  3形态              + io_spec          config选       OM/ACL | GeSession
                                            ↑
                              Verify: golden(eager) vs backend outputs
```

### Python / C++ 职责切分

| 侧 | 职责 |
|---|---|
| **Python** (`core/`) | 摄取(torch/onnx)、适配(patch)、导出(→AIR)、golden(eager torch)、pass 构建编排、配置解析、精度比对 |
| **C++** (`runtime/`) | 通用执行运行时：ACL 跑 OM / GeSession 跑 AIR·ONNX，**配置驱动**，含 benchmark |
| **桥** | 产物(AIR/OM/ONNX) + 配置(manifest/io_spec/bundle) + 验证数据(inputs/golden .bin) |

Python 不碰执行，C++ 不碰适配——靠**图 + 配置 + 验证数据**解耦。

## 4. 核心契约

三个契约文件，provenance 与生命周期各不相同，**绝不混在一个文件**：

| 契约 | 文件 | 归属 | 内容 | 消费者 |
|---|---|---|---|---|
| **io_spec** | `air/<name>.io_spec.json` | 图产物（与图同生命周期） | 图接口：node/logical/dtype/format + **动态维标记(-1)**；**不含 file、不含具体 shape** | ATC 编译、C++ 映射输入序 |
| **bundle** | `verification/bundle.json` | 验证产物（一组具体输入） | 该输入集的**具体 shape + file 路径** + golden 引用 + provenance(seed 等) | C++ 分配/喂 .bin、Python 比对 |
| **manifest** | `deploy/manifest.json` | 部署/运行时契约 | backend、graph/om 路径、io_spec 引用、device、pass vendor | C++ 运行时入口 |

- **Graph**：AIR 或 ONNX。后端只认 io_spec，不关心图怎么来的。
- **两层 shape**（关键，见 §6）：io_spec 记**动态维声明**，bundle 记**具体形状**——前者驱动编译，后者驱动 .bin 加载。

## 5. 配置与产物

### 5.1 文件布局（按 provenance 分层）

```
models/qwen2.5-0.5b/
  model.py                       # 人工: Adapter (模型适配, 唯一模型专属代码)
  config/model.yaml              # 人工: 声明 (唯一手写配置)
  air/qwen2.5-0.5b.air           # 生成: 图
  air/qwen2.5-0.5b.io_spec.json  # 生成: 图接口契约 (动态维, 无 file/具体 shape)
  om/qwen2.5-0.5b.om             # 生成: 编译产物 (om_acl 后端)
  deploy/manifest.json           # 生成: C++ 运行时契约 (部署期入口, 不寄居 verification)
  verification/                  # 生成: 验证数据
    bundle.json                  #   具体 shape + file + golden 引用 + provenance
    inputs/*.bin
    golden_logits.bin
```

**人工只维护 `model.py` + `config/model.yaml`**；其余全是生成物。manifest 在 `deploy/`（部署也要用），bundle 在 `verification/`（仅验证用）——部署可只带 `deploy/ + air/ + om/`，不依赖 verification。

### 5.2 model.yaml（人工声明）

```yaml
model:
  name: qwen2.5-0.5b
  soc: Ascend910_9382

source:                          # 3 形态
  type: torch                    # name | torch | onnx
  ref: Qwen/Qwen2.5-0.5B         # name=hub id; torch=权重目录; onnx=.onnx 路径
  # torch 源码形态:
  #   type: torch
  #   module: my.py
  #   class: MyModel
  #   weights: ...

adapt:                           # torch 源适用; onnx 原样跳过
  adapter_class: Qwen25Adapter   # 约定: 同目录 model.py 里的类
  params:                        # 仅适配行为开关
    prefix: false
    prune: false

inputs:                          # 输入生成 (export trace 与 verify golden 共用)
  batch_size: 10
  seq_len: 208
  prefix_len: 0
  seed: 0                        # 随机 token 种子, 进 provenance

graph:
  format: air                    # air | onnx
  dynamic:
    max_seq_len: 2048            # 图常量长度 (RoPE 表 / 因果 mask), 经 adapt(setup_kwargs) 透传
    # 注: 不参与 ATC 分档 — 动态图不传 --input_shape, 见 §6②

custom_ops:                      # 自定义算子安装**脚本** (加载 adapter 前执行), 见 §7
  - models/qwen2.5-0.5b/scripts/install_prefix_attn.sh

passes:                          # fusion pass 安装**脚本** (ATC 编译前执行), 见 §7
  - models/qwen2.5-0.5b/scripts/install_nz_pass.sh

backend:
  type: om_acl                   # om_acl | ge_session
  aicore_num: null

# 注: 不含 device — 用哪张卡是**运行期事实** (每次运行/每台机器都可能不同),
#     由 CLI `--device` **必填**传入; 不设默认值 (默认 0 号卡通常正是被占满的那张)

verify:
  enabled: true
```

> `adapt.params` 只放**适配行为**（prefix/prune）；`inputs` 放**输入形状/分布**（batch/seq/seed）——换输入分布不动 adapt。

### 5.3 io_spec.json（生成，图接口，独立文件）

```json
{
  "inputs": [
    {"node": "arg1_1", "logical": "actual_seq_lengths", "dtype": "int64", "format": "ND", "shape": [-1], "dynamic_dims": [0]},
    {"node": "arg4_1", "logical": "input_ids",          "dtype": "int64", "format": "ND", "shape": [-1], "dynamic_dims": [0]},
    {"node": "arg7_1", "logical": "position_ids",       "dtype": "int64", "format": "ND", "shape": [-1], "dynamic_dims": [0]}
  ],
  "outputs": [
    {"node": "logits", "dtype": "float16", "shape": [-1, 151936], "dynamic_dims": [0]}
  ]
}
```

- `node`：图里真名（dynamo 导出的 Data 节点，如 `arg1_1`）。
- `logical`：语义名 = **forward 入参名**（adapter 的 `io_input_nodes` 声明）。node↔logical 由 Data 节点的 `_source_name` 属性（`local:<入参名>`，torchair PR#3675；本机 torch_npu 未带，由 `core/_torchair_source_name.py` 回移）**按名字自动配对**，不靠位置。
- **inputs 按图 Data 序（index 序）排列**（C++/ATC 按位置喂入），**不一定等于 forward 入参序**——qwen2.5-0.5b 实测图序是 `actual_seq_lengths, input_ids, position_ids`，forward 序是 `input_ids, position_ids, actual_seq_lengths`（见 §15）。bundle 按 forward 序记，两侧靠 `logical` 名配对（C++ `BuildInputPlans` 按 logical 匹配，不按位置）。
- 配对不全（pbtxt 无 `_source_name`，或 `io_input_nodes` 的 logical 名与 forward 入参名不一致）时 **默认硬失败**——静默按位置映射等于喂错张量（表现为运行期 tiling 崩或精度全错，定位成本极高）。仅在 `GE_ALLOW_POSITIONAL_IO_SPEC=1` 时退回 forward 序位置映射并 WARN。
- `shape` 中 `-1` = 动态维，`dynamic_dims` 标出哪些轴动态。**不含 file、不含具体 shape**——那是 bundle 的事。

### 5.4 bundle.json（生成，验证数据契约）

```json
{
  "inputs": [
    {"logical": "actual_seq_lengths", "shape": [11],   "file": "inputs/actual_seq_lengths.bin"},
    {"logical": "input_ids",          "shape": [1900], "file": "inputs/input_ids.bin"},
    {"logical": "position_ids",       "shape": [1900], "file": "inputs/position_ids.bin"}
  ],
  "golden": {"logical": "logits", "shape": [10, 151936], "file": "golden_logits.bin"},
  "provenance": {
    "seed": 0, "batch_size": 10, "seq_len": 208, "prefix_len": 0,
    "model": "qwen2.5-0.5b", "soc": "Ascend910_9382", "dtype": "float16",
    "git_commit": "<repo HEAD>", "torch_npu": "<ver>", "transformers": "<ver>", "timestamp": "<iso>"
  }
}
```

C++ 用 bundle 的**具体 shape** 分配内存、按 `file` 读 .bin；按 **logical 名**（不是位置）与 io_spec 的输入配对，再按 io_spec 的图序喂给 OM/图。bundle 的 inputs 顺序 = forward 入参序（与 io_spec 的图序可不同）。

> **单输出假设**：`golden` 是单个对象（不是列表），文件名固定 `golden_logits.bin` —— 当前契约只支持单输出模型（CausalLM logits）。多输出需扩展为 `golden_<logical>.bin` + 逐输出比对（§10 已知限制）。
> `golden` 可为 `null`（`verify.enabled: false`）：此时 bundle 只含 inputs，运行时照跑，compare 阶段自动跳过。

### 5.5 manifest.json（生成，C++ 运行时契约，部署期入口）

```json
{
  "backend": "om_acl",
  "graph_path": "air/qwen2.5-0.5b.air",
  "om_path": "om/qwen2.5-0.5b.om",
  "io_spec": "air/qwen2.5-0.5b.io_spec.json",
  "device": 0,
  "bundle": "verification/bundle.json"
}
```

- `device` 来自**运行期的 `--device`**（必填），不来自 model.yaml——manifest 是每次生成的产物，把当次用哪张卡记进去正合适；C++ 侧 `--device` 仍可覆盖（换卡重跑不必重新生成 manifest）。
- `bundle` 仅验证态用；**部署态**（`bundle: null`）跑真实输入时由 CLI `--input logical:d0,d1,...:file.bin` 给具体 shape + .bin，dtype/format/node 仍取自 io_spec（不重复声明）。形态③ ONNX 无 adapter/build_inputs → 天然只有部署态。
- C++ 读取链：`manifest.json` → backend/路径/io_spec/device → 输入来自 `bundle.json`（验证态）或 `--input`（部署态）→ 喂入、执行、取输出。

### 5.6 派生关系

```
exporter.to_graph()   → air/<name>.air
graph.from_air/onnx   → 解析图 I/O 节点 + exporter 声明的 logical 序 → air/<name>.io_spec.json
verify.save_bundle()  → 具体 shape/file + golden + provenance → verification/bundle.json
pipeline 编排完        → 汇总 backend/路径/io_spec 引用/device/vendor → deploy/manifest.json
```

## 6. 动态 shape 端到端策略（OM 路径最大技术风险）

动态维贯穿四个环节，必须一起定，缺一环 OM 就跑不起来：

| 环节 | 谁负责 | 产物/动作 |
|---|---|---|
| **① 声明** | export：`build_inputs` 产出具体 trace shape，`mark_dynamic` 标动态轴，`dynamo_export(dynamic=True)` 烙进 AIR | io_spec 记 `shape:[-1]` + `dynamic_dims` |
| **② 编译** | ATC：动态图**不传** `--input_shape`（`OmAclBackend.compile` 检测 io_spec 有动态维即省略），由 GE 运行期特化；静态图才传具体 shape | OM（带动态维支持） |
| **③ 实例** | verify：bundle 记录这组输入的**具体 shape**（T=1900, N=11） | bundle.json |
| **④ 运行** | C++：按 bundle 具体 shape 分配 .bin 缓冲；动态 OM 每次 run 前 `aclmdlSetDatasetTensorDesc` 设实际 shape（CANN 9.0.0 无 `aclmdlSetDynamicInputTensorDesc`）；输出缓冲按 **bundle.golden 的具体 shape 解析 io_spec 动态维精确推导**（无 golden 才用 `--output_reserve` 预留，且执行后校验实际 size ≤ 分配，超出硬失败）；GeSession 由首次执行做 shape 特化 | outputs |

关键约束：

- **动态 OM 跨 shape 可直接复用（实测）**：用 T=32/N=2 导出的 OM 直接喂 T=2080/N=10 的输入执行成功（输出 `[10,151936]`）。所以**换 shape 不需要重编译**；pipeline 重导出是因为 **bundle/golden 与 shape 绑定**（要新的验证数据），不是 OM 的限制。ATC 分档（`--dynamic_batch_size`/`--dynamic_dims`）目前是**未实现的可选优化**，不是必需环节。
- `graph.dynamic.max_seq_len` 的真实作用是**图常量长度**（RoPE 表 / 因果 mask，经 `adapt(setup_kwargs)` 透传给 adapter.setup），不是 ATC 分档参数；它决定 position_ids 的可用上限，超了会 Gather 越界。
- io_spec 的 `[-1]` 只是**声明**，不能用来分配内存；C++ 分配一律用 bundle 的**具体 shape**。
- 旧 `model.conf` 本就有 `INPUT_N_SHAPE`（具体）+ `INPUT_N_SHAPE_ATC`（-1）的区分，本设计把它拆成 io_spec（动态声明）+ bundle（具体实例）两层，语义更清晰。
- GeSession 在线后端的动态维由首次执行的 shape 特化处理，无需 ATC 参与——这是它相对 OM 路径的灵活性优势（省掉 ~8min 编译）。

## 7. 环境准备：fusion pass 与自定义算子（用户脚本接口）

框架**不内置**任何 pass / 算子的构建安装逻辑——每个三方源的方式都不一样（fusion pass 是 cmake 出 `.so` 拷进 vendor；AscendC 自定义算子是 `build.sh` 产 `.run` 再 `--install-path`，还要 pip 装 torch 绑定 wheel）。内置一种就会对不上号，还得跟着上游改版。所以只提供一个稳定接口：**yaml 填脚本路径，框架按序执行**（`core/setup_scripts.py`）。

```yaml
custom_ops:                        # 在**加载 adapter 之前**执行 (model.py 可能 import 算子绑定)
  - models/qwen2.5-0.5b/scripts/install_prefix_attn.sh
passes:                            # 在 **ATC 编译之前**执行
  - models/qwen2.5-0.5b/scripts/install_nz_pass.sh
```

脚本约定（`core/setup_scripts.py`）：路径按 绝对 / 相对仓库根 / 相对 model_dir / 相对 CWD 依次解析；`.py` 用当前解释器、其余用 `bash`（不要求 +x 与 shebang）；继承当前 env；输出直接透传（构建动辄几分钟，要能看进度）；**非 0 退出即抛**（静默继续 = 算子没装上，下游报一堆看不懂的错）；幂等由脚本自己负责（"已装则 exit 0"）。脚本若要回传环境变量，把 `KEY=VALUE` 行写进 `$GE_ENV_FILE`——框架读进 `os.environ`（从而传给后续 ATC / `ge_runtime` 子进程），`PYTHONPATH` 还会同步进本进程 `sys.path`，并 `importlib.invalidate_caches()`（脚本刚 pip 装的绑定包当前进程才 import 得到）。

### 7.1 安装位置：都装到 `opp/vendors/<各自的 vendor 名>/`

**不用 per-model vendor 目录**（`opp/vendors/<model>/`）——因为两类东西的发现机制根本不同：

| | 发现机制 | 能否 per-model 隔离 |
|---|---|---|
| **fusion pass** | CANN **自动扫描** `opp/vendors/*/custom_fusion_passes/` 全部加载，不看 env、无优先级 | ❌ **不能**。per-model 目录只是组织归类，同名 pass 出现在两个 vendor 就是重复注册 → **ATC/TBE 崩**（实测：全局 `custom_nz_pass` 与 per-model 那份都注册 `MatMulWeightNZPass`） |
| **自定义算子** | `ASCEND_CUSTOM_OPP_PATH`（冒号分隔，优先级 2>4>1>3） | ✅ 能，但靠 **env 指向**，与目录名无关；且 vendor 名由三方工程写死（PIA 的 `CMakePresets.json: vendor_name=custom_prefix_attn`，其 `build.sh` 不转发 `-D`，改名要么抄它的两阶段构建、要么装完 `mv` 目录破坏它的 `upgrade.sh`） |

所以约定：**每个 pass / 算子装进它自己的 vendor 目录，全局一份**（`custom_nz_pass`、`custom_prefix_attn`），安装脚本负责"已装即跳过"与"别处已有同名 pass 就复用那份"。

### 7.2 三方源

| 源 | 内容 | 引入方式 |
|---|---|---|
| `third_party/custom_development_code/` | fusion pass 库（19 个：`WeightNzAndMatMulV3Pass`、`rmsnorm_pass`、`AttentionFusionPass`、`fa_pass` …）+ `monkey_patch/`、`model_opti_list/` | git submodule（gitcode） |
| `third_party/ascend-ops/` | AscendC 自定义算子库；当前含 `prefix-attention/`（PIA） | vendored 源码（github git 协议在本环境不可达，见 `third_party/README.md`） |

两者都**只读**：本仓库不改其中任何文件，构建产物落在 `.pass_build/`（仓库内、gitignored）或各源自己的 `build_out/`（被其自带 .gitignore 忽略）。

- **互补**：pass 是**图级**优化（NZ 权重转换、融合等），自定义算子是**算子级**能力（PIA 的 prefix-in-Q 语义），与 Python 侧**模型级**适配（patch）三者互补。
- qwen2.5-0.5b 用 `WeightNzAndMatMulV3Pass`（常量权重 MatMul ND→FRACTAL_NZ + MatMulV3，实测 ATC 日志 `matmul_match num=169`）+ PIA（`adapt.params.prefix: true` 时）。
- 脚本示例：`models/qwen2.5-0.5b/scripts/install_{nz_pass,prefix_attn}.sh`（含幂等、同名冲突检测、`$GE_ENV_FILE` 回传 env 的写法）。
- **踩过的坑**：`set -o pipefail` 下 `strings x.so | grep -q NAME` 会因 grep 提前退出触发 SIGPIPE，管道返回 141 → 冲突检测恒为假（真的把重复 pass 装进去了）。用 `grep -c` + `|| true` 替代。

## 8. 模型适配层（Adapter）

适配是逐模型的人类专家活（无法自动化），框架提供基类 + 模板 + 样例。

### 8.1 三阶段适配协议（`GeModelAdapter`）

```
__init__(**params)          ← model.yaml 的 adapt.params 原样传入 (基类只存 self.params)
load()/adapt():  apply_patches()   ← patch_specs:  类级行为替换 (怎么算)
                 from_pretrained() (load) / 接收已加载模型 (adapt)
                 setup(model, **setup_kwargs)  ← 实例级适配 (结构手术 + 常量注入 + 模式标志)
```

- **`__init__(**params)`**：基类收 `**params`，故**最小 adapter 不写构造函数也能被 `load_adapter` 实例化**（docs §13.8 零框架改动）。特殊键 `prune_token_file` 仅在 yaml 声明时才被配置层载入成 `prune_tokens` 列表传入。
- **setup_kwargs 由 pipeline 提供**：`adapt(raw, max_seq_len=cfg.graph.dynamic.max_seq_len)` —— yaml 声明的 `graph.dynamic.max_seq_len` 是图常量长度（RoPE 表 / 因果 mask）的唯一事实源，不透传就会退回 adapter 默认值而与 yaml 脱节（长序列 Gather 越界）。基类 `setup(self, model, **kwargs)` 吞掉不认识的键。
- **patch_specs**：声明 `[(target, 属性名, 新实现)]`，类级 monkey-patch，对所有实例生效。
- **setup**：实例级——结构手术（如 lm_head 剪裁）、常量注入（mask/rope 表，长度同源于 max_seq_len）、模式标志。
- **restore**：回滚全部 patch（备份表全局，patch 是进程级）。

### 8.2 设计原则（在 qwen2.5-0.5b 上验证过）

1. **patch 决定行为，代码只描述结构**：被 patch 的函数之间经模块属性调用（如 `modeling_qwen2.apply_rotary_pos_emb`），实现由 patch 状态决定，不硬编码。
2. **模式是创建时决策**：prefix 与否在 `Adapter(prefix=...)` 构造时定，`setup` 烙进实例（`_prefix_mode`），运行期不可变（翻标志会导致 eager 与已导出图行为分裂）。
3. **状态决定行为**：`use_cache` 等能力走类属性声明（`USE_CACHE`），`load` 写入 `model.config`，patched forward 透传——单一事实源。
4. **变量由 forward 路由，常量由 setup 注入**：`actual_seq_lengths` 是运行期变量，由 patched forward 每次以入参覆盖每层属性（成为图 Data 节点）；mask/rope 表是常量，由 setup 注入为 buffer（frozen_parameter 成图常量）。
5. **模型手术归 model.py**：lm_head 剪裁等结构手术放模型文件（YAGNI：等出现第二个需要相同手术的具体模型，再上提 common）。

### 8.3 样例（qwen2.5-0.5b）

`models/qwen2.5-0.5b/model.py` 是完整样例：5 处 patch（CausalLM.forward / RMSNorm / apply_rotary_pos_emb / RotaryEmbedding.forward / Attention.forward）+ prefix 变体 + lm_head 剪裁 + setup 注入 + `Qwen25Adapter`。新模型照此复制修改。

## 9. C++ 运行时（重建，通用）

旧 `atb/` 定制 C++ 已**退役删除**（能力全部移植进来；未移植项见 §10 已知限制），重建为配置驱动的通用运行时（`bash runtime/build.sh` → `runtime/build/ge_runtime`）：

```
runtime/
  main.cpp                 # 单入口: 读 manifest → io_spec/bundle(或 --input) → 按 backend 分发 → 落盘输出
  backends/
    acl_backend.{h,cpp}        # OM → ACL 加载+执行 (动态维经 aclmdlSetDatasetTensorDesc)
    gesession_backend.{h,cpp}  # AIR → GeSession 在线执行 (ONNX 不支持)
  io_spec.{h,cpp}          # 三份契约解析 (manifest/io_spec/bundle) + TensorPlan + .bin IO + outputs.json
  bench.{h,cpp}            # 延迟 (warmup+分位数) 与吞吐 (多线程闭环 sweep), 两后端共用
  acl_json.{h,cpp}         # dump/profiling 的 acl.json 生成 (OM/ACL 路径)
  CMakeLists.txt build.sh
```

```
ge_runtime <manifest.json> [--output_dir DIR] [--device N]
                           [--input logical:d0,d1,...:file.bin]   # 部署态(manifest 无 bundle)必填, 可重复
                           # 延迟: [--warmup N] [--bench N]
                           # 吞吐: [--threads N] [--requests M] [--sweep 1,2,4,8]
                           # 在线后端: [--graph_run_mode M] [--precision_mode P] [--aicore_num SPEC]
                           # 输出缓冲: [--output_reserve MB]
                           # 观测: [--dump --dump_path/--dump_mode/--dump_level/--dump_data/--dump_layer]
                                   [--profiling --profiling_output/--profiling_aic_metrics]
```

- 读取链（§5.5）：`manifest.json` → backend/路径/device → `io_spec.json`（node/dtype/format + 动态维声明）→ 输入二选一：**验证态** `bundle.json`（具体 shape + .bin，按 logical 名配对）/ **部署态** `--input`（CLI 给具体 shape + .bin，dtype/format/node 仍取自 io_spec）→ 合成 `TensorPlan`（逐输入 stat 校验字节数 == shape×dtype，不重复读盘）→ 执行 → `output_<i>.bin` + `outputs.json`（logical/dtype/shape/file，供 Python compare）。
- ATC 编译（AIR/ONNX → OM）留在 **Python**（`tools/atc_utils`，argv 列表直传 subprocess，不走 shell；`--framework` 按 `graph.kind` 取 1/5）；C++ 只做运行时。
- OM/ACL 后端：io_spec 声明动态维时逐输入 `aclCreateTensorDesc` + `aclmdlSetDatasetTensorDesc`（CANN 9.0.0 无 `aclmdlSetDynamicInputTensorDesc`）。输出缓冲：验证态按 `bundle.golden` 的具体 shape **精确推导**，部署态用 `--output_reserve`（默认 256MB）；执行后校验实际 size ≤ 分配，超出即硬失败（否则是静默 HBM 越界）。输出 desc 动态图取 `aclmdlGetDatasetTensorDesc`、静态图取 `aclmdlGetOutputDims/DataType`（静态 OM 的 dataset 上不挂 desc）。
- GeSession 后端：C++ 直接加载 AIR 在线执行（API 序列见 §15）；ONNX 不支持（§1）。
- **并发的资源模型（实测约束）**：
  - ACL：**每线程独立 `aclmdlLoadFromFile`**（共享 modelId 并发 `aclmdlExecuteAsync` 实测返回 500002）+ 独立 stream/dataset/缓冲 → HBM ≈ N × OM 大小；模型元数据由首个 context 的 desc 顺带打印（不做 probe 加载，省一次 ~2s 重复加载）。
  - GeSession：**单 Session 多图**——`LoadGraph` 对同一 graphId **不可重复调用**（ge_api.h 约束），故 N 路并发要 N 份 `AddGraph`+`CompileGraph`（串行，qwen2.5-0.5b 约 10s/份）+ 每线程独立 stream 与 `LoadGraph(gid, {}, stream)`；`CompileGraph` 必须在 `aclrtSetDevice` **之前**。
  - 工作线程共享主线程的**默认 context**（`aclrtGetCurrentContext` → 各线程 `aclrtSetCurrentContext`）；显式 `aclrtCreateContext` 会让 GE executor 报 "stream is not in current ctx"。CANN 无 reset 接口，线程退出即释放。
  - `--sweep` 各档复用同一批图绑定/模型实例（GE 的 `setup` 幂等、`release` 为 no-op，统一在退出时释放）。
- **观测**：dump/profiling 在 OM/ACL 路径经 `acl_json.cpp` 生成 `acl.json` 交 `aclInit(configPath)`；GeSession 的 profiling 走 `GEInitialize` 的 `OPTION_EXEC_PROFILING_MODE/OPTIONS`（dump 是 ACL 专属，给了会 WARN 忽略）。产物 `PROF_*` 用 `tools/parse_profiling.py` 解析，dump 数据用 `tools/parse_dump.py`。
- **抽象时机（YAGNI）**：两后端各暴露一个自由函数（`RunAclBackend` / `RunGeSessionBackend`），`main.cpp` 按 `manifest.backend` switch 分发，**不预设 Backend 基类**——公共部分（契约解析、.bin IO、bench）已下沉到 `io_spec`/`bench`，剩下的差异（ACL dataset vs gert::Tensor）不值得抽象。两后端各自的 dtype 枚举映射表**故意不合并**（ACL 与 GE 是两套枚举，合并要引中间层，比重复更贵）。
- 构建：C++17 + `-D_GLIBCXX_USE_CXX11_ABI=0`（GE 头/库为旧 ABI）；JSON 用 vendored `third_party/nlohmann/json.hpp`（header-only，离线可构建）。
- GE 在线路径的运行环境额外要求：`source <model>/env.sh`（CANN + vendor 算子 + 把本地 site-packages 注入 `PYTHONPATH`，否则 GEInitialize 因 tbe pywrapper 缺 numpy 返回 -1）。

## 10. 验证流（跨 Python/C++）

```
Python: load 适配模型 → build_inputs(seed) → eager golden → 存 bundle{inputs, golden, provenance}
C++:    backend 跑 OM/GeSession on inputs → verification/outputs/{output_<i>.bin, outputs.json}
Python: compare(outputs, golden) → report   (verify.compare_bundle → tools/compare.py)
```

- `outputs.json`（`{outputs:[{logical,dtype,shape,bytes,file}]}`）是 C++ → Python 的回传契约：shape/dtype 取**运行时实测值**（动态维特化后的真实形状），compare 按 logical 与 bundle.golden 配对。
- 判定用 `tools/compare.py` 的二重口径（cosine > 0.9999 且 relative_l2 < 0.01）；`pipeline` 在 FAIL 时以非 0 退出（回归门）。
- **门禁不做 flatten/截断兜底**：`compare_bundle` 遇到 golden 与输出 shape 不一致直接抛错——截断后比 cosine 会让"错序/错 shape"也 PASS（静默放行）。人工排查才用 `tools/compare.py` CLI（它保留截断行为）。

- **golden = NPU-eager**：patched 模型的 eager 路径（`is_compiling()=False`，走 torch_npu 算子），与 OM/GeSession 的 graph 路径同源——对比**隔离出"编译"这一个变量**。（"适配是否正确 vs 原版 HF"是另一个更早的检查。）
- **输入要有代表性**：带 `seed` 的随机 token + 真实变长分布（非全 0），确定性可复现，又能压到数值路径。trace 与 golden 共用同一组输入；`seed` 进 bundle.provenance。
- **golden 在 trace 之前算**：eager 先跑干净，再 dynamo_export。
- **bundle 自包含 + provenance**：inputs/golden + metadata（seed/model/soc/batch/seq/prefix/prune/dtype/时间戳/git commit/torch_npu·transformers 版本）。

### 性能验收

- 管线含 **bench 阶段**（C++ `bench.{h,cpp}`），两种口径都已落地：
  - **延迟**：`--warmup N --bench M`，单线程 execute+sync，报 avg/min/p50/p99/max。
  - **吞吐**：`--threads N --requests M`（闭环并发，每线程独立资源）或 `--sweep 1,2,4,8`（串行扫档，每档一份报告），报 wall/QPS/e2e 分位数/errors。
- 口径统一为 **execute + sync**（两后端可比）；首次执行含懒初始化/shape 特化（OM ~330ms、GE ~250ms），故延迟测量必须配 `--warmup`。
- **基线数据归 `models/<model>/docs/`**（逐模型），架构层只规定"有 bench 阶段 + 基线归档位置 + 报告格式"，**不写死全局阈值**——具体验收线（如 OM 相对 eager 的加速比下限）逐模型定。
- 观测：`--profiling`（两后端）/`--dump`（仅 OM/ACL）产出 `PROF_*` 与逐算子数据，交 `tools/parse_profiling.py`、`tools/parse_dump.py` 解析。

### 已知限制（当前契约的边界）

- **单输出假设**：bundle 只记一个 golden（`golden_logits.bin`）、`pipeline._output_node` 固定 `logical="logits"`、compare 只比一个输出、C++ 仅在 `outputs.size()==1` 时精确推导输出缓冲 —— 即当前只支持"单 logits 的 CausalLM"。多输出模型需扩展 bundle schema（`golden_<logical>.bin`）+ 逐输出比对，等出现第二个实例再做（YAGNI）。
- **随机 varlen 负载生成未移植**：旧 `atb/bench_latency.cpp` 的 RequestGenerator（对数正态序列长度 + 闭环随机请求）随 `atb/` 退役删除，需要时从 git 历史取（`8b7ce86:atb/bench_latency.cpp`）；通用替代是用 `tools/varlen.py` 生成多组 bundle 逐组跑。
- **形态② PyTorch 源码**（`source.py:_from_source_code`）的加载约定（无参构造 + 单文件 `torch.load`）是**未经实例验证的猜测**，遇到真客户需按实际约定改写（docs §15）。

## 11. core/ 模块接口

```python
# config.py
@dataclass ModelConfig: model; source; adapt; inputs; graph; passes; backend; verify   # 无 device
def load_config(path) -> ModelConfig
def load_adapter(cfg) -> GeModelAdapter        # importlib 从 <model_dir>/model.py 取 adapter_class
def write_manifest(cfg, graph, om, io_spec, bundle, base_dir, device) -> path
                                                 # 写 deploy/manifest.json; device 必填

# source.py  (YAGNI: 先 torch 分支, onnx 留桩; 第二形态落地再抽 ABC)
def load_source(cfg, model_dir, dtype, device) -> torch.nn.Module | Graph   # device 必填(torch 形态)
    # type=name/torch: from_pretrained / importlib 加载 nn.Module
    # type=onnx:       返回 OnnxGraph (桩, 待实例细化)

# graph.py
@dataclass IoNode: node; logical; dtype; format; shape; dynamic_dims   # 无 file
@dataclass IoSpec: inputs: list[IoNode]; outputs: list[IoNode]
class Graph: kind; path; io_spec
    @staticmethod from_air(air_path, inputs: list[IoNode], outputs: list[IoNode]) -> Graph
        # 解析 dynamo.pbtxt 的 Data 节点 (index/name/_source_name) → 按 _source_name 与
        # inputs 的 logical 名配对 → io_spec.inputs 按 index 序 (图喂入序) 落盘
    @staticmethod from_onnx(onnx_path) -> Graph    # 解析 onnx I/O, 写 io_spec.json

# adapter.py  (现有 GeModelAdapter)
class GeModelAdapter:
    def load(self, model_path, dtype, **setup_kwargs) -> model
    def patch_specs(self); def setup(self, model, **kw)
    def apply_patches(self); def restore(self)
    def build_inputs(self, model, **kw); def mark_dynamic(self, inputs, **kw)
    def io_input_nodes(self, inputs, **kw) -> list[IoNode]   # forward 序; logical 名须 == forward 入参名

# _torchair_source_name.py  (回移 torchair PR#3675)
def native_support() -> bool      # 已装 torchair 是否原生支持 ge.Data(source_name=...)
def enable() -> bool            # 未支持时打补丁 (幂等); 导出前由 GeExporter.trace 调用

# exporter.py  (现有 GeExporter)
class GeExporter:
    def export(self, model_path, dtype, **build_kwargs) -> air_path
    def build_inputs(self, model, **kw); def mark_dynamic(self, inputs, **kw)
    def logical_inputs(self) -> list[str]          # forward 签名的逻辑输入序, 供 graph.from_air

# setup_scripts.py  (pass / 自定义算子的构建安装 = 用户脚本, 框架只按序执行)
def resolve_script(entry, model_dir=None) -> path | None   # 绝对/仓库根/model_dir/CWD
def run_scripts(entries, stage, model_dir=None) -> list    # 按序执行; 非 0 退出即抛;
                                                           # 回收脚本写进 $GE_ENV_FILE 的 env

# backend.py  (时序: compile_graph → write_manifest → run_runtime; 无 Backend 基类, 与 C++ 侧同标准)
def compile_graph(cfg, graph, base_dir=None) -> om_path | None   # om_acl: run_atc; ge_session: None
                                                                 # (并校验 图形态×后端: onnx 不走 ge_session)
def default_output_dir(manifest_path) -> str             # <manifest 根>/verification/outputs
def runtime_argv(manifest, output_dir, device, warmup, bench, extra, inputs) -> list[str]
def run_runtime(manifest, ...) -> output_dir             # 子进程跑 ge_runtime (继承 CANN env)

# verify.py
class Verifier:
    def golden(self, model, inputs) -> Tensor            # eager forward (NPU)
    def save_bundle(self, dir, inputs, golden, io_spec, provenance,
                    logical_order=None) -> bundle_path   # 写 bundle.json + .bin (forward 序标签)
                                                         # golden=None → 只落 inputs (部署/无验证态)
    def compare_bundle(self, bundle_path, outputs_dir, dtype) -> report   # 闭环第三段 (shape 不符即抛)
def bundle_has_golden(bundle_path) -> bool               # pipeline 据此决定是否 compare

# pipeline.py  (YAGNI: 全量 + --skip, 不做 6 阶段枚举)
def run(config_path, skip=(), dtype, device, batch_size, seq_len, work_dir,
        warmup, bench, runtime_extra, runtime_inputs)     # device 必填; skip ⊂ {export,passes,compile,run,compare}
                                                          # runtime_inputs: 部署态 --input 规格 (形态③ 用)
```

## 12. 目录结构（现状）

```
ascend-ge-adapters/
├── README.md                      # 入口: 目录导览 + 快速开始 + 契约速查
├── requirements.txt pytest.ini    # Python 依赖 (实测版本) / pytest 配置 (含 npu marker)
├── core/                          # 通用框架 (Python)
│   ├── source.py adapter.py exporter.py graph.py
│   ├── setup_scripts.py backend.py verify.py config.py pipeline.py
│   └── _torchair_source_name.py   # 回移 torchair PR#3675: Data 节点带 forward 入参名
├── runtime/                       # 通用执行运行时 (C++)
│   ├── main.cpp CMakeLists.txt build.sh
│   ├── backends/{acl_backend, gesession_backend}.{h,cpp}   # 无 Backend 基类, main 按 manifest 分发
│   ├── io_spec.{h,cpp} bench.{h,cpp} acl_json.{h,cpp}
├── third_party/                   # 三方源一律**只读** (见 third_party/README.md)
│   ├── nlohmann/json.hpp          # vendored header-only JSON (C++ 读三份契约)
│   ├── custom_development_code/   # gitcode submodule: fusion_pass/ (19 个 pass)
│   └── ascend-ops/                # vendored: prefix-attention/ (PIA 自定义算子工程)
├── models/
│   └── qwen2.5-0.5b/
│       ├── model.py               # Adapter (模型专属, 唯一手写代码之一)
│       ├── config/model.yaml      # 声明 (基线) + model.prefix.yaml (PIA 变体)
│       ├── scripts/               # 用户脚本: install_nz_pass.sh / install_prefix_attn.sh
│       ├── run.sh env.sh          # 薄封装 core/pipeline + 运行环境
│       └── docs/                  # DEPLOYMENT_GUIDE.md + reports/ aicore/ prefix-attention/
├── tools/                         # varlen / atc_utils / compare / parse_dump / parse_profiling
├── tests/                         # test_*.py (CPU, pytest) + tiny_e2e/tiny_onnx_e2e/smoke (NPU 脚本)
└── docs/architecture.md
```

演进映射（已完成）：`models/common` → `core/`（适配基类），`atb/` → `runtime/`（**已删除**，能力全部移植；未移植项见 §10 已知限制），`atb/models/*` → `models/*`，`atb/tools/*` → `tools/*`，`models/*/pass/` → `third_party/.../fusion_pass/`。

## 13. 设计原则汇总

1. **稳定脚手架与易变适配分离**：source/graph/backend/verify 通用稳定；适配逐模型可插拔。
2. **Graph 是中心契约**：上游产图、下游消费图，io_spec 随行。
3. **三类契约分文件、按 provenance 归位**：io_spec（图）/ bundle（验证实例）/ manifest（部署运行时）/ YAML（人工）互不混。
4. **两层 shape**：io_spec 记动态维声明（驱动编译），bundle 记具体形状（驱动 .bin 加载）——动态 shape 端到端见 §6。
5. **patch=类级行为，setup=实例级状态，模式=创建时决策**。
6. **状态决定行为，代码只描述结构**。
7. **验证内建**：golden（NPU-eager）+ 自包含 bundle + provenance（含 seed）+ 冒烟回归。
8. **模型专属 = Adapter + yaml**，pipeline 通用；新模型 onboarding 零框架改动（由冒烟测试兜底）。
9. **YAGNI**：Source ABC、Backend 抽象、stages 枚举——都等第二个具体案例落地再做。

## 14. 迁移路径（本仓库演进，每步保持可用）

1. **配置层**：`core/config.py`（YAML 解析 + manifest 生成）；qwen2.5 散参数收敛进 `config/model.yaml`
2. **冒烟测试骨架**：小模型 e2e（export→compile→run→compare）脚本，作为后续每步的回归门（先于功能扩张建立）
3. **Graph 抽象 + 两层 shape**：`core/graph.py`——从 AIR pbtxt / ONNX 派生 io_spec（动态维），bundle 记具体 shape；落地 §6 端到端
4. **环境准备（pass / 自定义算子）**：`third_party/custom_development_code` 加 submodule、`third_party/ascend-ops` vendored；`core/setup_scripts.py` 提供"yaml 填脚本路径"的接口（框架不假设构建方式）；`models/qwen2.5-0.5b/scripts/install_{nz_pass,prefix_attn}.sh` 为示例脚本，装到 `opp/vendors/<各自 vendor 名>/`（§7.1：pass 无 per-model 隔离，算子靠 env 指向）
5. **Source（torch 分支 + onnx 桩）**：`core/source.py`——name/torch 加载，onnx 留桩
6. **Backend**：`core/backend.py`——`OmAclBackend`（包 atc_utils）；`GeSessionBackend` 留待阶段二
7. **Verify 成体系**：`core/verify.py`——golden + bundle（含 seed/provenance）+ compare + bench 阶段
8. **Pipeline CLI**：`core/pipeline.py`——读 config 编排全流程（全量 + `--skip`）；退役 per-model `export_air.py`/`prepare_air_inputs.py`
9. **qwen2.5-0.5b 收敛**为：`model.py`（Adapter）+ `config/model.yaml`
10. **C++ 运行时重建**（阶段二，**已落地**）：`runtime/`——`acl_backend` + `gesession_backend` 各为具体实现，`main.cpp` 按 manifest 分发；公共部分下沉到 `io_spec`/`bench`/`acl_json`，**未抽 Backend 基类**（两后端差异只在执行 API，抽象无收益）。`atb/` 已**移植后删除**：OM/GE 执行、延迟与多线程吞吐 bench、dump、profiling 全部进 `runtime/`；唯一未移植的是 `bench_latency.cpp` 里的随机 varlen 负载生成器（模型专属，见 §10 已知限制）。

### 阶段建议

- **阶段一**（Python 侧 + 现有 atb 暂顶 OM/ACL）：步骤 1–9。打通 name/torch → AIR → OM → 验证，配置驱动，冒烟测试兜底。
- **阶段二**（C++ 重建 + GeSession）：步骤 10。先验证 GeSession API 再谈抽象——**已完成**：两后端均跑通 `run + compare` 闭环（`tests/tiny_e2e.py`，cosine=1.0）。

> 第 10 步工作量最大且依赖 GeSession/ACL 具体 API，单独成阶段，避免阻塞 Python 侧管线打通；且遵循"先具体后抽象"，不为未知 API 提前定接口。

### 回归门

| 测试 | 需要 | 覆盖 |
|---|---|---|
| `pytest`（`tests/test_core.py` + `tests/test_atc_utils.py` + `tests/test_runtime_contract.py`） | 纯 CPU；contract 那组另需已构建的 `ge_runtime`（缺失则 skip） | 静默失败点：图序配对/硬失败、yaml→dataclass、adapter 构造约定、compare 门禁、manifest 相对路径、ATC argv 构造、C++ 契约解析与部署态 `--input` |
| `tests/tiny_e2e.py` | NPU（~几百 MB 显存）+ ATC | 极小模型全链路：export（含 `_source_name` 断言）→io_spec/bundle/manifest→ATC→**om_acl** run+compare→**ge_session** run+compare |
| `tests/tiny_onnx_e2e.py` | NPU + ATC + onnxruntime | 形态③：onnx→`from_onnx`(io_spec)→ATC(`--framework=5`)→OM→**部署态** run（无 bundle，`--input`）→ 对比 onnxruntime CPU 参考 |
| `tests/smoke.py` | NPU + 真实权重 | qwen2.5-0.5b 真实模型；默认只到 bundle，`--full` 跑全链路（含 ATC 与 runtime）。断言分**结构不变量**与**模型事实**（后者从 batch/seq 与权重 `config.json` 推导，不写字面量） |

约定：`tests/test_*.py` = 纯 CPU 单测（pytest 收集，CI 可跑，不 import torch/torch_npu）；`tests/{tiny_e2e,tiny_onnx_e2e,smoke}.py` = 需 NPU 的脚本（pytest 不收集，手动跑）。

## 15. 待定/依赖项

- ~~**io_spec 的 node↔logical 配对**~~ **已实测定论（阶段二，OM 运行 + golden 比对验证）**：pbtxt 的 Data 节点带 `index` 属性，按 index 排序得到的是**图侧喂入序**，但它 **≠ dynamo_export 入参序**。Qwen2.5-0.5B 实测：`arg1_1(index0)=actual_seq_lengths, arg4_1(index1)=input_ids, arg7_1(index2)=position_ids`，而 forward 入参序是 `(input_ids, position_ids, actual_seq_lengths)`。
  - 证据：按 forward 序喂 OM → `ApplyRotaryPosEmb` tiling 崩（rope Gather 读到 asl，cos `[2,1,64]` vs q `[32,14,64]`，报 "all input dim1 must equal"；旧 `atb/acl_infer` 同样崩，排除运行时嫌疑）；按图序喂 → 执行通过且与 eager golden 比对 PASS（cosine 0.99996，om_acl 与 ge_session 输出逐字节一致）。
  - **解法（已落地）**：torchair 上游 [PR#3675](https://gitcode.com/Ascend/torchair/pull/3675) 给 Data 节点加了 `_source_name` 属性（dynamo 的 `LocalSource/GlobalSource` → `local:<forward 入参名>`）。本机 torch_npu 2.9.0.post2 的内置 torchair 还没带，故 `core/_torchair_source_name.py` 按同一机制回移（patch `_npu_backend` 采集 `arg_pos_to_source` + `parse_input` 按 `data_index` 取名 + `ge.Data` 写属性；上游原生支持时自动 no-op，异常只 WARN）。`graph.from_air` 据此**按名字自动配对**并输出图序 io_spec —— 无需人工声明、无需探测。
  - 对齐关系（实测）：`parse_input` 的 `data_index = self.graph.num_inputs` 与 `_try_get_metadata_from_dynamo` 返回的 `arg_pos_to_source` 下标一一对应（参数/buffer/符号 shape 各占一位；它们后续被冻结成 Const 或在图里重新编号，但 parse_input 时刻是对齐的）。注意必须在 `_npu_backend(gm, ...)` 处采集——`_NpuFxCompiler.__call__` 拿到的 gm 已无 dynamo 元数据（`_try_get_metadata_from_dynamo` 返回 None）。
  - 注：pbtxt 格式是 `op:"Data"`（非 op_type），且因 frozen 权重内嵌可达 GB 级，用 grep 流式提取。旧 `atb` config 的 `arg1_1→act` 是**对的**（此前文档判其为 bug 有误）。
- ~~**动态 shape 的 ATC 机制**~~ **已定论（阶段二实测）**：动态图**不需要** `--dynamic_batch_size`/`--dynamic_dims` 分档——`OmAclBackend.compile` 检测到 io_spec 有动态维就不传 `--input_shape`，GE 运行期自行特化，同一 OM 可跨 shape 复用（T=32 导出的 OM 跑 T=2080 成功，见 §6）。分档只作为**可选优化**（减少运行期特化开销），当前未实现。`graph.dynamic.max_seq_len` 与 ATC 无关，是图常量长度（§6）。
  - ~~仍待办：`run_atc` 硬编码 `--framework=1`、`shell=True`~~ **已落地（P1/A5）**：`tools/atc_utils.py` 拆出纯函数 `build_atc_argv`/`normalize_aicore`/`framework_of`（单测见 `tests/test_atc_utils.py`），`subprocess.run(argv)` **不走 shell**（路径含空格安全、无注入面），`--framework` 由 `graph.kind` 决定（air=1 / onnx=5）。ONNX 形态已实测打通（`tests/tiny_onnx_e2e.py`：onnx → io_spec → ATC(fw=5) → OM → 部署态 run，与 onnxruntime CPU 参考 cosine=0.99999991）。
- ~~GeSession 具体 C++ API~~ **已实测确认**（CANN 9.0.0，`runtime/backends/gesession_backend.cpp`）：`ge::GEInitialize({ge.graphRunMode, ge.exec.deviceId[, aicore_num]})` → `aclInit` → `ge::Session({ge.session_device_id, ge.exec.precision_mode})` → `ge::Graph::LoadFromFile(<name>.air)` → `AddGraph(id, graph)` → `CompileGraph(id)` → `aclrtSetDevice` + `aclrtCreateStream` → `LoadGraph(id, {}, stream)` → 输入构造为 `gert::Tensor`（`StorageShape` origin+storage 同填、`StorageFormat(FORMAT_ND, FORMAT_ND, {})`、`SetDataType`、`SetData(TensorData(devPtr, nullptr, bytes, kOnDeviceHbm))`）→ `ExecuteGraphWithStreamAsync(id, stream, inputs, outputs)` + `aclrtSynchronizeStream`。**动态维无需 ATC 分档**：首次执行做 shape 特化，输出 device 缓冲由 GE 自动分配（`outputs[i].GetSize()/GetAddr()/GetShape()` 取实际值，用完 `aclrtFree`，多次执行需按地址去重释放）。清理序：free 输入/输出 → `aclrtDestroyStream` → `session.reset()` → `aclrtResetDevice` → `GEFinalize` → `aclFinalize`。链接 `ge_compiler ge_runner ge_common ge_common_base graph graph_base ascendcl`，须 `-D_GLIBCXX_USE_CXX11_ABI=0` + C++17。
- ~~pass 构建/隔离~~ **已定论（见 §7.1）**：fusion pass 由 CANN **自动扫描 `opp/vendors/*/custom_fusion_passes/` 全部加载**（无 env、无优先级）→ **没有 per-model 隔离**，同名 pass 跨 vendor 会重复注册 → ATC/TBE 崩（实测 `MatMulWeightNZPass`）。故约定"每个 pass 装进它自己的 vendor 目录、全局一份"，由安装脚本负责幂等与同名检测。`ASCEND_CUSTOM_OPP_PATH` 只服务**自定义算子**（优先级 2>4>1>3），与 fusion pass 无关；算子的 vendor 名由三方工程写死（PIA = `custom_prefix_attn`），隔离靠 env 指向而非目录名。
- ~~C++ 读 manifest/io_spec/bundle 用 nlohmann/json（header-only）~~ **已落地**：vendored `third_party/nlohmann/json.hpp`（v3.12.0 单头文件，git 跟踪，离线可构建）。
- form ② PyTorch 源码的加载约定（module 路径 + class 名 + 权重；构造参数来源、分片/safetensors、dtype 转换）——遇到实例再细化。
