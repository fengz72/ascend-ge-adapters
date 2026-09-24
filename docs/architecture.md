# Ascend GE Adapters — 架构设计

> 把客户 torch 模型搬到 NPU GE 上**高效运行**（2 种后端）的 onboarding 管线。
> 本文是方案定稿，作为后续实施的依据。

## 1. 项目目标

项目只适配 **torch** 模型，统一导出到 Ascend GE 上高效推理：

| 来路 | 来源 | 目标图 | 状态 |
|---|---|---|---|
| ① 模型名 | transformers hub（如 `Qwen/Qwen2.5-0.5B`） | torch → **AIR** | **已实现**（`ref`→from_pretrained） |
| ② PyTorch 源码 | 任意 `nn.Module`（含权重） | torch → **AIR** | 未实现（YAGNI 推迟，见 §10/§15） |

执行有两种后端：

| 后端 | 路径 | 特点 | 能吃哪些图 |
|---|---|---|---|
| **OM/ACL**（离线） | ATC 编译图 → OM → ACL 运行时 | 编译一次部署多次，运行时开销低，OM 产物可移植 | AIR（`--framework=1`） |
| **GeSession**（在线） | GeSession 直接加载图在线执行 | 在线 JIT，灵活，无离线产物，省掉 ATC（~8min）；C++ 接口 | **仅 GE 图**（AIR/pbtxt）——`ge::Graph::LoadFromFile` |

两后端都能加载 GE pass，都吃 AIR，故 `backend.type` 可在 `om_acl | ge_session` 间自由切换（`tests/tiny_e2e.py` 两后端均跑通闭环）。

> 注：来路①② 本质同一条 `nn.Module` 加载路径（差别只在模型结构从 transformers 拿还是从客户源码拿）；代码**当前只实现来路①**（`ref`→from_pretrained），来路②（客户源码 `module`+`class`）按 YAGNI 推迟、未实现（见 §10/§15）。Source 抽象按 YAGNI 推迟（见 §11）。

## 2. 问题矩阵

```
            ┌──────────────┐   ┌─────────────┐   ┌──────────────────────┐
  ① hf权重 ─┤              │   │             │   │ 后端1: ATC→OM→ACL     │
  ② 源码 ───┤ Source/Adapt ├──→│ Graph(AIR)  ├──→│ 后端2: GeSession      │──→ outputs
            │              │   │             │   │  (吃 AIR + pass)      │
            └──────────────┘   └─────────────┘   └──────────────────────┘
                  ↑ 模型级适配          ↑ io_spec        ↑ 图级优化(pass) + 编译选项
```

收敛点是 **Graph**：上游（源 + 适配）只负责产出图，下游（后端）只负责消费图。
`source` 只有 **torch** 一种：来路①（transformers 权重：hub id / 本地目录）
与②（客户 PyTorch 源码：`module`+`class`）本质同一条加载路径（都产出
`nn.Module`，后续 adapt/export 完全一致），差别只在**模型结构从哪来**，不值得占两个配置值。
（代码当前只实现来路①；来路② 的加载按 YAGNI 推迟，见 §10。）

## 3. 总体架构

```
Source ──[Adapt]──> Graph(AIR) ──[Passes]──> Backend ──> outputs
  torch             + io_spec    config选       OM/ACL | GeSession
                                       ↑
                               Verify: golden(eager) vs backend outputs
```

### Python / C++ 职责切分

| 侧 | 职责 |
|---|---|
| **Python** (`core/`) | 摄取(torch)、适配(patch)、导出(→AIR)、golden(eager torch)、pass 构建编排、配置解析、精度比对 |
| **C++** (`runtime/`) | 通用执行运行时：ACL 跑 OM / GeSession 跑 AIR，**配置驱动**，含 benchmark |
| **桥** | 产物(AIR/OM) + 配置(manifest/io_spec/bundle) + 验证数据(inputs/golden .bin) |

Python 不碰执行，C++ 不碰适配——靠**图 + 配置 + 验证数据**解耦。

## 4. 核心契约

三个契约文件，provenance 与生命周期各不相同，**绝不混在一个文件**：

| 契约 | 文件 | 归属 | 内容 | 消费者 |
|---|---|---|---|---|
| **io_spec** | `air/<name>.io_spec.json` | 图产物（与图同生命周期） | 图接口：node/logical/dtype/format + **动态维标记(-1)**；**不含 file、不含具体 shape** | ATC 编译、C++ 映射输入序 |
| **bundle** | `io/bundle.json` | 验证产物（一组具体输入） | 该输入集的**具体 shape + file 路径** + golden 引用 + provenance(seed 等) | C++ 分配/喂 .bin、Python 比对 |
| **manifest** | `io/manifest.json` | 部署/运行时契约 | backend、graph/om 路径、io_spec 引用、device、pass vendor | C++ 运行时入口 |

- **Graph**：AIR。后端只认 io_spec，不关心图怎么来的。
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
  io/manifest.json               # 生成: C++ 运行时契约 (运行期入口)
  io/                            # 生成: 一次运行的输入/输出数据 (跑完即弃, 可重跑再生)
    bundle.json                  #   具体 shape + file + golden 引用 + provenance
    inputs/*.bin
    golden_logits.bin
    reference.json               #   门① 原版 HF vs 适配后 eager 的指标 (docs §10)
    outputs/                     #   门② C++ 运行时输出 (output_<i>.bin + outputs.json)
    pool/                        #   性能测试请求池 (req_NNNN/{bundle.json, inputs/*.bin})
  results/                       # 生成: 报告归档 (json+md 入库; csv/raw gitignored)
```

**人工只维护 `model.py` + `config/model.yaml`**；其余全是生成物。`io/` 里既有运行时入口
（`manifest.json`）也有验证数据（bundle/inputs/golden/outputs）——两者都是**一次运行的派生物**：
输入由 `config/model.yaml` 的 `inputs`(shape+seed) 现生成、golden 由 patched eager 现算，
所以整个目录可随时删掉重跑。要长期保留的只有小体积报告（`reference.json`、`results/`）。
交付部署（只带图 + 入口）目前没有单独出口，需要时再加 `--emit-deploy`。

> `io/pool/` 是**唯一的例外**：它跨 run 复用（生成 200 套请求要十几秒），幂等由生成脚本自己
> 比 `pool_meta.json` 的参数保证——参数变了才重生。删掉 `io/` 的代价里，池重生是大头。
> 池里每套请求本身就是一份 bundle（与 `io/bundle.json` 同构），C++ 的 `LoadPool` 扫
> `<dir>/*/bundle.json`，故放在 `io/` 下与验证数据同属"输入数据 + bundle 契约"这一类。

### 5.2 model.yaml（人工声明）

```yaml
model:
  name: qwen2.5-0.5b
  soc: Ascend910_9382

source:                          # 只有 torch 一种 (来路①②同一条加载路径, 见 §2)
  ref: Qwen/Qwen2.5-0.5B         # hub id 或本地权重目录
  # 来路② (客户 PyTorch 源码, module+class) 未实现 — 出现真实实例时再加 (见 §10/§15)

adapt:                           # torch 源适用
  adapter_class: Qwen25Adapter   # 约定: 同目录 model.py 里的类
  params:                        # 仅适配行为开关
    prefix: false
    prune_token_file: null       # 填路径 → lm_head 词表剪裁 (如 models/<m>/config/target_tokens.json)

inputs:                          # 输入生成 (export trace 与 verify golden 共用)
  batch_size: 10
  seq_len: 208
  prefix_len: 0
  seed: 0                        # 随机 token 种子, 进 provenance

graph:
  dynamic:
    max_seq_len: 2048            # 图常量长度 (RoPE 表 / 因果 mask), 经 adapt(setup_kwargs) 透传
    # 注: 不参与 ATC 分档 — 动态图不传 --input_shape, 见 §6②

custom_ops:                      # 自定义算子 (加载 adapter 前执行), 见 §7
  - path: third_party/ascend-ops/prefix-attention              # 三方源: 溯源 + $GE_SRC_DIR
    script: models/qwen2.5-0.5b/scripts/install_prefix_attn.sh # 安装脚本

passes:                          # fusion pass (ATC 编译前执行), 见 §7
  - path: third_party/custom_development_code/fusion_pass/WeightNzAndMatMulV3Pass
    script: models/qwen2.5-0.5b/scripts/install_nz_pass.sh

backend:
  type: om_acl                   # om_acl | ge_session
  aicore_num: null

# 注: 不含 device — 用哪张卡是**运行期事实** (每次运行/每台机器都可能不同),
#     由 CLI `--device` **必填**传入; 不设默认值 (默认 0 号卡通常正是被占满的那张)

verify:
  enabled: true

bench:                           # 性能测试口径 (core.bench / tools.sweep); 一个模型一个场景
  instances: 4                   # = 并发 worker (1:1 绑实例, 无锁)
  requests: 2000                 # 总请求 (闭环, 均分到实例)
  warmup: 50                     # 覆盖 每实例 × 每 shape 档
  sample_seed: 0                 # 请求池**抽样**种子 (每实例 seed+instance_id); ≠ inputs.seed
  pool:                          # 请求池生成 = 用户脚本 (§7 同一套接口), 落 io/pool/
    script: models/qwen2.5-0.5b/scripts/gen_requests.py   # 路径写法同 passes/custom_ops (仓库根相对)
    args: [--count, "200", --prefix, "20-25"]   # 只写要覆盖的; 分布口径是脚本的 argparse 默认值
```

> `adapt.params` 只放**适配行为**（prefix/prune）；`inputs` 放**输入形状/分布**（batch/seq/seed）——换输入分布不动 adapt。

> **`bench` 段只放"这次压测怎么压"，模型侧事实一律不重复**（否则两处会静默分叉）：`soc`←`model.soc`、manifest←`<model_dir>/io/manifest.json`、报告←`<model_dir>/results`、池←`<model_dir>/io/pool`、`aicore_num`←`backend.aicore_num`。**没有精度开关**——精度只由 `verify.enabled` 驱动的两道门度量（§10），bench 只出性能数字。请求池生成脚本的**形态参数**由 `bench._form_args` 从配置注入而不是写在 args 里：`--batch`←`inputs.batch_size`（prefix 形态下 act 是静态 `[batch+1]`，池里每套的条数被图烙死）、`--prune-tokens`←`adapt.params.prune_token_file`（决定 golden 宽）、`--prefix`←`inputs.prefix_len` 保底（**范围**属负载口径，args 里写 `"20-25"` 优先）。负载分布（μ/σ/长度截断/词表上界）是模型专属脚本的口径，写死在脚本默认值里，配置只在要覆盖时写 `pool.args`。
>
> **刻意不做"多份 scenario 文件"**（原来是 `bench/varlen.yaml`，已合并）：一个模型当前只需要一个负载场景，独立文件只会把 `soc`/`manifest`/限核/形态参数抄成第二份。**重新引入的触发条件**：同一形态要长期并存多份负载报告（随机 varlen + 固定 shape + 长序列压测）——届时把 `bench` 段抽回独立 yaml、`core.bench` 加回 `--scenario` 即可（形状可从 git 历史的 `bench/varlen.yaml` 取）。

> **一份 yaml 只描述"当前形态"，形态演进靠 git**：模型是不断向下演进的（qwen2.5-0.5b 的 prefix/PIA 就是在 FIA 基线上演化的），任一时刻只有一个当前形态值得被配置描述。切形态 = 改 `adapt.params` 那几行（`prefix` / `prune_token_file` + `inputs.prefix_len`），产物名由 `config.export_name` 自动带后缀（`-prefix` / `-prune`），**不会静默覆盖**另一种形态的 AIR/OM/bundle。历史形态要复现就 `git checkout <commit> -- models/<m>/config/model.yaml`。
>
> 刻意**不做**"变体覆盖表 / 多份 yaml 并存"：那是为"多形态长期并存的矩阵"设计的机制，用在串行演进上只是多一层要理解的抽象（读者得先问"这次生效的是哪份配置"）。**重新引入的触发条件**：同一模型有 ≥2 种形态需要长期并存且都进回归（例如客户同时部署共享 prefix 与普通 varlen 两套）——届时覆盖表的实现可从 commit `3db0189` 取回（约 20 分钟）。
>
> 形态必须**可归因**：`adapt.params` 全量写进 `bundle.provenance`，性能报告顺着 manifest→bundle 取回来（`bench.form_from_manifest`）并印在 `perf.md` / `index.json` 上——否则 A/B 两份报告分不清哪份是哪种形态。

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
  "bundle": "io/bundle.json"
}
```

- `device` 来自**运行期的 `--device`**（必填），不来自 model.yaml——manifest 是每次生成的产物，把当次用哪张卡记进去正合适；C++ 侧 `--device` 仍可覆盖（换卡重跑不必重新生成 manifest）。
- `bundle` 是 C++ 运行时**唯一**的输入来源：具体 shape + .bin 路径都在里面，dtype/format/node 取自 io_spec（不重复声明）。
- C++ 读取链：`manifest.json` → backend/路径/io_spec/device → 输入来自 `bundle.json` → 喂入、执行、取输出。

### 5.6 派生关系

```
exporter.to_graph()   → air/<name>.air
graph.from_air        → 解析图 I/O 节点 + exporter 声明的 logical 序 → air/<name>.io_spec.json
verify.save_bundle()  → 具体 shape/file + golden + provenance → io/bundle.json
pipeline 编排完        → 汇总 backend/路径/io_spec 引用/device/vendor → io/manifest.json
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
  - path: third_party/ascend-ops/prefix-attention              # 三方源在哪 (溯源)
    script: models/qwen2.5-0.5b/scripts/install_prefix_attn.sh # 怎么装 (用户脚本)
passes:                            # 在 **ATC 编译之前**执行
  - path: third_party/custom_development_code/fusion_pass/WeightNzAndMatMulV3Pass
    script: models/qwen2.5-0.5b/scripts/install_nz_pass.sh
```

**条目 = `path` + `script`**（`core.config.SetupEntry`；也允许只写脚本路径的纯字符串条目）：
- `path` 是**三方源目录**，作用是①配置里一眼看出这个 pass/算子从哪来（溯源）②解析成绝对路径后经
  **`$GE_SRC_DIR`** 传给脚本，脚本不必硬编码源码位置（示例脚本写成 `${GE_SRC_DIR:-<回退路径>}`，
  单独手跑也能用）。框架**不拿 path 构建**——构建方式归脚本。源目录不存在（submodule 未克隆）
  只 WARN 并照常执行脚本，由脚本决定跳过还是失败。
- `script` 是安装脚本，路径解析规则见下。

脚本约定（`core/setup_scripts.py`）：`script` 与 `path` **只认绝对路径或相对仓库根**（如 `models/<m>/scripts/x.sh`）——相对 model_dir / CWD 一律硬失败，因为多基准会让同一份配置混两种写法（读的人得先问"这条相对谁"），而 CWD 相对等于"换个目录跑就找不到脚本"；`.py` 用当前解释器、其余用 `bash`（不要求 +x 与 shebang）；继承当前 env；输出直接透传（构建动辄几分钟，要能看进度）；**非 0 退出即抛**（静默继续 = 算子没装上，下游报一堆看不懂的错）；幂等由脚本自己负责（"已装则 exit 0"）。脚本拿到的输入是 `$GE_SRC_DIR`（源目录）与 `$GE_ENV_FILE`；若要回传环境变量，把 `KEY=VALUE` 行写进 `$GE_ENV_FILE`——框架读进 `os.environ`（从而传给后续 ATC / `ge_runtime` 子进程），`PYTHONPATH` 还会同步进本进程 `sys.path`，并 `importlib.invalidate_caches()`（脚本刚 pip 装的绑定包当前进程才 import 得到）。

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
| `third_party/ascend-ops/` | AscendC 自定义算子库；当前含 `prefix-attention/`（PIA） | git **submodule**（`github.com/fengz72/ascend-ops` @ `5be240f`；`github.com:443` 在部分网络环境不可达 → `--init` 会失败，离线兜底：经 `codeload.github.com` 下 tarball 解压到该路径，或从本仓库 git 历史取回旧的 vendored 副本） |

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

- **`__init__(**params)`**：基类收 `**params`，故**最小 adapter 不写构造函数也能被 `load_adapter` 实例化**（docs §13.8 零框架改动）。特殊键 `prune_token_file` 仅在 yaml 声明时才被配置层载入成 `prune_tokens` 列表传入；它的路径规则与 `script`/`path` 一致——**绝对路径或相对仓库根**（`setup_scripts.resolve_path`），找不到即硬失败，不回退 model_dir/CWD。
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

### 8.4 新模型 onboarding 顺序（涉及自定义算子时）

原则：**先基线、后新算子**——把"适配是否正确"与"新算子是否正确"两个变量分开（A5 适配按此走）。

1. **基线先跑通**：只用框架/torch_npu 已有能力（如 FIA `npu_fused_infer_attention_score`）适配 → 导出 → OM → run → compare PASS，报告归档 `models/<m>/results/`。
2. **再接自定义算子**：`custom_ops:` 挂安装脚本（§7）→ adapter 加形态开关（如 `prefix`）走新算子路径 → 用**同一套 golden 口径**比对 → 两份报告靠 `bundle.provenance.adapt_params` 区分（§5.2）。
3. **形态开关写进 yaml 注释**，产物名由 `config.export_name` 带后缀（`-prefix`/`-prune`），A/B 互不覆盖。

配套的两个 YAGNI 触发点（等第二个实例出现再做，别提前抽象）：
- `models/qwen2.5-0.5b/scripts/install_prefix_attn.sh` 目前归 qwen 私有；**A5 是第二个消费者** → 那时上提到仓库级 `scripts/`，两个模型的 yaml 都指过去（同 `models/common → core/` 的套路）。
- 若 A5 与 qwen 的 adapter 出现重复结构（相同的 lm_head 剪裁、相同的 varlen forward 骨架）→ 上提到 `core/`（§8.2 第 5 条）。

## 9. C++ 运行时（重建，通用）

旧 `atb/` 定制 C++ 已**退役删除**（能力全部移植进来；未移植项见 §10 已知限制），重建为配置驱动的通用运行时（`bash runtime/build.sh` → `runtime/build/ge_runtime`）：

```
runtime/
  main.cpp                 # 单入口: 读 manifest → io_spec/bundle → 按 backend 分发 → 落盘输出
  backends/
    acl_backend.{h,cpp}        # OM → ACL 加载+执行 (动态维经 aclmdlSetDatasetTensorDesc)
    gesession_backend.{h,cpp}  # AIR → GeSession 在线执行
  io_spec.{h,cpp}          # 三份契约解析 (manifest/io_spec/bundle) + TensorPlan + .bin IO + outputs.json
  bench.{h,cpp}            # 延迟 (warmup+分位数) / 吞吐 (多线程闭环, 单档) / 变长负载 (BenchPool)
  pool.{h,cpp}             # 请求池: 扫 req_*/bundle.json → 具体 shape + host 数据 + 每实例随机抽样
  bench_plan.{h,cpp}       # --bench-plan: 读 plan.json → 多实例跑池 → 写 perf.json / perf_requests.csv
  acl_json.{h,cpp}         # dump/profiling 的 acl.json 生成 (OM/ACL 路径)
  CMakeLists.txt build.sh
```

```
ge_runtime <manifest.json> [--output_dir DIR] [--device N]
                           # 延迟: [--warmup N] [--bench N]
                           # 吞吐: [--threads N] [--requests M]   (单档; 扫描归 tools/sweep.py)
                           # 变长负载: --bench-plan <plan.json>  (多实例 + 请求池回放, 见 §10)
                           # 在线后端: [--graph_run_mode M] [--precision_mode P] [--aicore_num SPEC]
                           # 输出缓冲: [--output_reserve MB]
                           # 观测: [--dump --dump_path/--dump_mode/--dump_level/--dump_data/--dump_layer]
                                   [--profiling --profiling_output/--profiling_aic_metrics]
```

- 读取链（§5.5）：`manifest.json` → backend/路径/device → `io_spec.json`（node/dtype/format + 动态维声明）→ 输入来自 `bundle.json`（具体 shape + .bin，按 logical 名配对）→ 合成 `TensorPlan`（逐输入 stat 校验字节数 == shape×dtype，不重复读盘）→ 执行 → `output_<i>.bin` + `outputs.json`（logical/dtype/shape/file，供 Python compare）。
- ATC 编译（AIR → OM）留在 **Python**（`tools/atc_utils`，argv 列表直传 subprocess，不走 shell；`--framework=1` 固定为 AIR）；C++ 只做运行时。
- OM/ACL 后端：io_spec 声明动态维时逐输入 `aclCreateTensorDesc` + `aclmdlSetDatasetTensorDesc`（CANN 9.0.0 无 `aclmdlSetDynamicInputTensorDesc`）。输出缓冲：有 `bundle.golden` 时按其具体 shape **精确推导**，无 golden 时用 `--output_reserve`（默认 256MB）预留；执行后校验实际 size ≤ 分配，超出即硬失败（否则是静默 HBM 越界）。输出 desc 动态图取 `aclmdlGetDatasetTensorDesc`、静态图取 `aclmdlGetOutputDims/DataType`（静态 OM 的 dataset 上不挂 desc）。
- GeSession 后端：C++ 直接加载 AIR 在线执行（API 序列见 §15）。
- **并发的资源模型（实测约束）**：
  - ACL：**每线程独立 `aclmdlLoadFromFile`**（共享 modelId 并发 `aclmdlExecuteAsync` 实测返回 500002）+ 独立 stream/dataset/缓冲 → HBM ≈ N × OM 大小；模型元数据由首个 context 的 desc 顺带打印（不做 probe 加载，省一次 ~2s 重复加载）。
  - GeSession：**单 Session 多图**——`LoadGraph` 对同一 graphId **不可重复调用**（ge_api.h 约束），故 N 路并发要 N 份 `AddGraph`+`CompileGraph`（串行，qwen2.5-0.5b 约 10s/份）+ 每线程独立 stream 与 `LoadGraph(gid, {}, stream)`；`CompileGraph` 必须在 `aclrtSetDevice` **之前**。
  - 工作线程共享主线程的**默认 context**（`aclrtGetCurrentContext` → 各线程 `aclrtSetCurrentContext`）；显式 `aclrtCreateContext` 会让 GE executor 报 "stream is not in current ctx"。CANN 无 reset 接口，线程退出即释放。
  - **并发档位扫描不进 C++**（`--sweep` 已删）：扫描 = "同一件事跑 N 遍"，每档都要独立建/销资源，进程级隔离最干净（一档崩了不连累其它档，HBM 彻底归还）。C++ 只负责测准**一档**（`BenchThroughput` / `BenchPool`），档位循环归 `tools/sweep.py`（逐档起进程 + 汇总 scaling 表）。
- **观测**：dump/profiling 在 OM/ACL 路径经 `acl_json.cpp` 生成 `acl.json` 交 `aclInit(configPath)`；GeSession 的 profiling 走 `GEInitialize` 的 `OPTION_EXEC_PROFILING_MODE/OPTIONS`（dump 是 ACL 专属，给了会 WARN 忽略）。产物 `PROF_*` 用 `tools/parse_profiling.py` 解析，dump 数据用 `tools/parse_dump.py`。
- **抽象时机（YAGNI）**：两后端各暴露一个自由函数（`RunAclBackend` / `RunGeSessionBackend`），`main.cpp` 按 `manifest.backend` switch 分发，**不预设 Backend 基类**——公共部分（契约解析、.bin IO、bench）已下沉到 `io_spec`/`bench`，剩下的差异（ACL dataset vs gert::Tensor）不值得抽象。两后端各自的 dtype 枚举映射表**故意不合并**（ACL 与 GE 是两套枚举，合并要引中间层，比重复更贵）。
- 构建：C++17 + `-D_GLIBCXX_USE_CXX11_ABI=0`（GE 头/库为旧 ABI）；JSON 用 vendored `third_party/nlohmann/json.hpp`（header-only，离线可构建）。
- GE 在线路径的运行环境额外要求：`source <model>/env.sh`（CANN + vendor 算子 + 把本地 site-packages 注入 `PYTHONPATH`，否则 GEInitialize 因 tbe pywrapper 缺 numpy 返回 -1）。

## 10. 验证流（跨 Python/C++）

```
Python: load **原版**模型 → build_inputs(seed) → reference (原版逐请求前向, adapt 之前)
        → adapt (patch + setup) → eager golden → compare_reference(golden, reference)   ← 门①适配
        → 存 bundle{inputs, golden, provenance} + io/reference.json
C++:    backend 跑 OM/GeSession on inputs → io/outputs/{output_<i>.bin, outputs.json}
Python: compare(outputs, golden) → report   (verify.compare_bundle → tools/compare.py)   ← 门②编译
```

**两道门，各隔离一个变量**——只有 golden 是不够的：golden 来自**打过 patch 的同一个模型**，
适配写错（错 mask / 错位置编码 / 错末 token 索引 / 剪裁错列）时两边一起错，比对照样 PASS。

| 门 | 比对 | 隔离的变量 | 门限 | 产物 |
|---|---|---|---|---|
| ① `reference` | 原版未 patch 的 HF 逐请求末 token logits vs 适配后 eager golden | **适配**（融合算子替换、varlen 打包、prefix 语义、lm_head 剪裁） | cosine > 0.999 且 rel_l2 < 0.02（跨实现；实测噪声 0.999999 / 1.5e-3，`core/verify.py:REF_*`） | `io/reference.json` |
| ② `compare` | C++ 运行时输出 vs golden | **编译**（AIR→OM/GeSession、图序喂入、动态 shape 特化） | cosine > 0.9999 且 rel_l2 < 0.01（同源） | `io/outputs/` |

- **`reference` 的时机是硬约束**：必须在 `adapter.adapt()` **之前**算——patch 是类级
  monkey-patch（进程全局，见 §8），adapt 之后同进程里任何同架构实例都走 patched forward，
  "原版"就名存实亡。`pipeline` 因此把 `load_source → build_inputs → reference → adapt → golden`
  排成一条线（`core/pipeline.py`）。
- **还原逐请求是模型专属知识**，归 adapter 的两个可选钩子（默认返回 `None` → WARN 跳过）：
  `unpack_requests(inputs)` 把打包的图输入拆回 `[(ids, positions)]`（prefix 形态 = prefix ++ own_i，
  position 用 `arange` 重新生成而**不取**图输入里的 `position_ids`——打包 position 写错正好由比对暴露）；
  `reference_columns()` 在输出被剪裁时给出要取的列（lm_head 剪裁 → 只比保留的那些列）。
  通用的部分（一条请求一次前向 + 取末 token + 比对）在 `core/verify.py`。
- `outputs.json`（`{outputs:[{logical,dtype,shape,bytes,file}]}`）是 C++ → Python 的回传契约：shape/dtype 取**运行时实测值**（动态维特化后的真实形状），compare 按 logical 与 bundle.golden 配对。
- 判定统一用 `tools/compare.py` 的二重口径（cosine + relative_l2，门限可传参）；`pipeline` 在任一门 FAIL 时以非 0 退出（回归门）。
- **门禁不做 flatten/截断兜底**：`compare_bundle`/`compare_reference` 遇到 shape 不一致直接抛错——截断后比 cosine 会让"错序/错 shape"也 PASS（静默放行）。人工排查才用 `tools/compare.py` CLI（它保留截断行为）。
- **golden = NPU-eager**：patched 模型的 eager 路径（`is_compiling()=False`，走 torch_npu 算子），与 OM/GeSession 的 graph 路径同源。
- **输入要有代表性**：带 `seed` 的随机 token（词表宽取自 **embedding 权重形状**，不是 `config.vocab_size`——lm_head 剪裁会把后者改成剪裁宽度）+ 真实变长分布，确定性可复现，又能压到数值路径。全 0 token 会让每条请求逐字节相同，比对退化成"同一行比 N 次"。trace / golden / reference 共用同一组输入；`seed` 进 bundle.provenance。
- **golden 在 trace 之前算**：eager 先跑干净，再 dynamo_export。
- **bundle 自包含 + provenance**：inputs/golden + metadata（seed/model/soc/batch/seq/prefix/prune/dtype/时间戳/git commit/torch_npu·transformers 版本）。
- **用例数量**：一组 = **一个** bundle（`inputs.batch_size/seq_len/prefix_len` 决定），不做用例矩阵——
  这一组输入同时喂 reference / golden / 运行时，三道输出互证已足够定位问题；要换形状就改 yaml 重跑。


### 性能验收

三种口径，都在 C++ 侧测量、Python 侧编排与排版（`core/bench.py`）：

| 口径 | 入口 | 说明 |
|---|---|---|
| **延迟** | `ge_runtime <manifest> --warmup N --bench M` | 单实例、固定输入（H2D 一次），纯 execute+sync 分位数 |
| **吞吐** | `… --threads N --requests M`（单档）/ `tools/sweep.py --levels 1,2,4,8`（扫描） | N 实例闭环并发（1 worker ↔ 1 实例，无锁），报 wall/QPS/e2e 分位数；扫描逐档起进程后汇总 scaling 表 |
| **变长负载** | `python3 -m core.bench --config <model.yaml>` → `ge_runtime --bench-plan <json>` | 多实例 + **请求池回放**（每请求 shape/数据不同），分阶段计时 + 报告归档；口径在 `bench` 段（§5.2） |

**多实例的资源模型**（实测约束，见 §9）：ACL 每实例独立 `aclmdlLoadFromFile`（共享 modelId 并发执行会 500002），qwen2.5-0.5b 实测 **~1.2GB HBM/实例**；GeSession 单 Session 多图，每实例一份 `AddGraph`+`CompileGraph`（**~10s/份，串行**）+ `LoadGraph(gid, stream)`，实测 **~0.6GB HBM/实例**（图实例间共享权重）。

**请求池（变长负载）**：框架**不生成模型专属负载**——语义自洽（`asl` 必须 cumsum、`position_ids` 每段 0..L-1、`asl[-1]==T`）只有模型侧能保证，故由**用户脚本**按分布预生成 K 套输入落盘（`io/pool/req_NNN/{bundle.json, inputs/*.bin}`），与 pass/算子同一套 `{path, script, args}` 接口（§7）；形态参数由框架注入（§5.2）。运行时：所有实例**共享一个池**，各自用 `seed + instance_id` 随机抽样（可复现）；device 缓冲按池内**最大 shape** 预分配，每请求 H2D 实际字节 + 重设 desc（ACL）/ 重建 `gert::Tensor`（GE）；输出缓冲取池内最大 golden。

- **池的 distinct shape 数直接决定 warmup 成本**：warmup 必须覆盖 每实例 × 每 shape（否则测量段付特化代价，csv 里 `first_hit=1` 且 stderr WARN）。实测 qwen 138 种 shape × 2 实例 = 276 次 warmup ≈ 3.4s（ACL）；GE 的图特化更贵，**建议 GE 用分档池**（少量 shape）。
- 每请求记 `h2d / desc / execute+sync / e2e` 四段（定位瓶颈：H2D 占比高说明该增大 batch 或用 pinned memory）。

**报告与归档**（`<model_dir>/results/<run_id>/`，run_id = 时间戳-git短sha-`<model>-bench`；一个模型一个负载场景，故归档名不再单独声明）：

```
run.json          # 快照: git/CANN/torch_npu 版本、device、soc、model.yaml 全文、性能摘要
perf.json         # C++ 出的数据: 聚合 + 每实例 (qps/e2e 分位/exec/h2d/desc/load/特化/HBM)
perf.md           # 人读表 + 口径说明
perf_requests.csv # 逐请求明细 (gitignore)
plan.json         # 传给 C++ 的 bench plan (gitignore)
results/index.json# 历次 run 一行摘要 (趋势)
```

json + md **入库**（小、可 diff、可归档），csv/plan 忽略。基线数据归 `models/<model>/results/`（逐模型），架构层只规定"有 bench 阶段 + 归档位置 + 报告格式"，**不写死全局阈值**——验收线逐模型定。

> **性能报告不含精度**：池模式不落盘输出（每请求 D2H 会污染延迟数字，`runtime/bench_plan.cpp`），精度只由 §10 的两道门度量（`run.sh` 的 `reference` + `compare`）。一个变量只由一处度量——原来 bench 里还有一次"单请求精度复核"（`accuracy.json/md`），与门② 用同一份 bundle、同一份 golden，只是多花一次 ge_session 图编译（实测 ~10s/run），已删除。

- profiling/dump（`--profiling` / `--dump`）产出 `PROF_*` 与逐算子数据，交 `tools/parse_profiling.py`、`tools/parse_dump.py` 解析；算子级 top-N 进报告属 P2（未做）。

### 已知限制（当前契约的边界）

- **单输出假设**：bundle 只记一个 golden（`golden_logits.bin`）、`pipeline._output_node` 固定 `logical="logits"`、compare 只比一个输出、C++ 仅在 `outputs.size()==1` 时精确推导输出缓冲 —— 即当前只支持"单 logits 的 CausalLM"。多输出模型需扩展 bundle schema（`golden_<logical>.bin`）+ 逐输出比对，等出现第二个实例再做（YAGNI）。
- **随机 varlen 负载生成未移植**：旧 `atb/bench_latency.cpp` 的 RequestGenerator（对数正态序列长度 + 闭环随机请求）随 `atb/` 退役删除，需要时从 git 历史取（`8b7ce86:atb/bench_latency.cpp`）；通用替代是用 `tools/varlen.py` 生成多组 bundle 逐组跑。
- **来路② PyTorch 源码未实现**：原 `source.py:_from_source_code` 猜测性桩（无参构造 + 单文件 `torch.load`，**未经实例验证**）连同 `module`/`class`/`weights` 配置字段已按 YAGNI **删除**；遇到真客户源码模型时按实际约定从头实现（docs §15）。

## 11. core/ 模块接口

```python
# config.py
@dataclass ModelConfig: model; source; adapt; inputs; graph; passes; custom_ops;
                        backend; verify; bench                    # 无 device (运行期 --device)
@dataclass SetupEntry: script; path       # passes/custom_ops/bench.pool 的条目 (脚本 + 三方源)
@dataclass BenchCfg: instances; requests; warmup; sample_seed; pool
                                                 # 只放压测口径, 模型侧事实从其它段取 (§5.2)
def load_config(path) -> ModelConfig
def load_adapter(cfg) -> GeModelAdapter        # importlib 从 <model_dir>/model.py 取 adapter_class
def export_name(cfg) -> str                    # 产物名 = 模型名 + 形态后缀 (-prefix/-prune)
def write_manifest(cfg, graph, om, io_spec, bundle, base_dir, device) -> path
                                                 # 写 io/manifest.json; device 必填

# source.py  (YAGNI: 只有 torch 一条加载路径; 第二种来路落地再抽 ABC)
def load_source(cfg, model_dir, dtype, device) -> torch.nn.Module   # device 必填
    # from_pretrained (来路①; 来路② 客户源码未实现, 见 §10)

# graph.py
@dataclass IoNode: node; logical; dtype; format; shape; dynamic_dims   # 无 file
@dataclass IoSpec: inputs: list[IoNode]; outputs: list[IoNode]
class Graph: path; io_spec
    @staticmethod from_air(air_path, inputs: list[IoNode], outputs: list[IoNode]) -> Graph
        # 解析 dynamo.pbtxt 的 Data 节点 (index/name/_source_name) → 按 _source_name 与
        # inputs 的 logical 名配对 → io_spec.inputs 按 index 序 (图喂入序) 落盘

# adapter.py  (现有 GeModelAdapter)
class GeModelAdapter:
    def load(self, model_path, dtype, **setup_kwargs) -> model
    def patch_specs(self); def setup(self, model, **kw)
    def apply_patches(self); def restore(self)
    def build_inputs(self, model, **kw); def mark_dynamic(self, inputs, **kw)
    def io_input_nodes(self, inputs, **kw) -> list[IoNode]   # forward 序; logical 名须 == forward 入参名
    # 原版参考比对的两个**可选**钩子 (默认 None → verify WARN 跳过, 见 §10)
    def unpack_requests(self, inputs) -> list[(ids, positions)] | None   # 打包图输入 → 逐请求
    def reference_columns(self) -> list[int] | None      # 输出被剪裁时取哪些列 (如 lm_head prune)

# _torchair_source_name.py  (回移 torchair PR#3675)
def native_support() -> bool      # 已装 torchair 是否原生支持 ge.Data(source_name=...)
def enable() -> bool            # 未支持时打补丁 (幂等); 导出前由 GeExporter.trace 调用

# exporter.py  (现有 GeExporter)
class GeExporter:
    def export(self, model_path, dtype, **build_kwargs) -> air_path
    def build_inputs(self, model, **kw); def mark_dynamic(self, inputs, **kw)
    def logical_inputs(self) -> list[str]          # forward 签名的逻辑输入序, 供 graph.from_air

# setup_scripts.py  (pass / 自定义算子的构建安装 = 用户脚本, 框架只按序执行)
def resolve_script(entry) -> path | None       # **只认**绝对路径 / 相对仓库根; 其余硬失败
def run_scripts(entries, stage) -> list                # entries: SetupEntry|dict|str;
                                                           # 按序执行, 传 $GE_SRC_DIR, 非 0 即抛,
                                                           # 回收脚本写进 $GE_ENV_FILE 的 env

# backend.py  (时序: compile_graph → write_manifest → run_runtime; 无 Backend 基类, 与 C++ 侧同标准)
def compile_graph(cfg, graph, base_dir=None) -> om_path | None   # om_acl: run_atc; ge_session: None
def default_output_dir(manifest_path) -> str             # <manifest 根>/io/outputs
def runtime_argv(manifest, output_dir, device, warmup, bench, extra) -> list[str]
def run_runtime(manifest, ...) -> output_dir             # 子进程跑 ge_runtime (继承 CANN env)

# bench.py  (性能测试编排: model.yaml 的 bench 段 → plan.json → ge_runtime → 报告归档)
@dataclass Scenario: name; manifest; model_dir; instances; requests; warmup; seed; inputs;
                     generate; report_dir; backend_options; device; soc
def load_bench(config_path, device, instances, requests, warmup, manifest) -> Scenario
                                                 # bench 段 + 派生约定 (io/manifest·io/pool·results)
def _form_args(cfg, args) -> list                # 形态事实 (--batch/--prune-tokens/--prefix) 注入
def build_plan(scenario, run_dir, run_id) -> plan_path      # C++ 只吃 json, 不读 yaml
def run(config_path, ...) -> run_dir                        # 生成池 → 跑 → 写报告 → 更新 index
def render_perf_md(perf, scenario, run_id, provenance)      # 只出性能 (精度归 §10 两道门)
def form_from_manifest(manifest_path) -> dict            # 顺着 manifest→bundle 取形态, 报告自证
def update_index(report_dir, entry) -> index_path

# verify.py
class Verifier:                                  # 门限: compare=同源(0.9999/0.01),
    def reference(self, model, adapter, inputs) -> Tensor | None   # reference=跨实现(0.99/0.05)
                                                 # **原版未 patch** 模型逐请求末 token logits;
                                                 # 必须在 adapt 之前调 (patch 是进程级类属性)
    def compare_reference(self, golden, reference, path=None) -> report   # 门① 适配是否正确
    def golden(self, model, inputs) -> Tensor            # eager forward (NPU)
    def save_bundle(self, dir, inputs, golden, io_spec, provenance,
                     logical_order=None) -> bundle_path   # 写 bundle.json + .bin (forward 序标签)
                                                           # golden=None → 只落 inputs (verify.enabled: false)
    def compare_bundle(self, bundle_path, outputs_dir, dtype) -> report   # 门② 编译是否正确
def bundle_has_golden(bundle_path) -> bool               # pipeline 据此决定是否 compare
def collect_provenance(..., adapt_params, **extra)       # 形态 (prefix/prune) 进 provenance

# pipeline.py  (YAGNI: 全量 + --skip, 不做 6 阶段枚举)
def run(config_path, skip=(), dtype, device, batch_size, seq_len, work_dir,
        warmup, bench, runtime_extra)
                                                 # device 必填
                                                 # skip ⊂ {ops,export,passes,compile,run,compare,reference}
                                                 # 顺序: load_source → build_inputs → reference
                                                 #       → adapt → golden → trace → …
```

## 12. 目录结构（现状）

```
ascend-ge-adapters/
├── README.md                      # 入口: 目录导览 + 快速开始 + 契约速查
├── requirements.txt              # Python 依赖 (实测版本)
├── core/                          # 通用框架 (Python)
│   ├── source.py adapter.py exporter.py graph.py
│   ├── setup_scripts.py backend.py verify.py config.py bench.py pipeline.py
│   └── _torchair_source_name.py   # 回移 torchair PR#3675: Data 节点带 forward 入参名
├── runtime/                       # 通用执行运行时 (C++)
│   ├── main.cpp CMakeLists.txt build.sh
│   ├── backends/{acl_backend, gesession_backend}.{h,cpp}   # 无 Backend 基类, main 按 manifest 分发
│   ├── io_spec.{h,cpp} pool.{h,cpp} bench.{h,cpp} bench_plan.{h,cpp} acl_json.{h,cpp}
├── third_party/                   # 三方源一律**只读** (见 §7.2)
│   ├── nlohmann/json.hpp          # vendored header-only JSON (C++ 读三份契约)
│   ├── custom_development_code/   # gitcode submodule: fusion_pass/ (19 个 pass)
│   └── ascend-ops/                # git submodule: prefix-attention/ (PIA 自定义算子工程)
├── models/
│   └── qwen2.5-0.5b/
│       ├── model.py               # Adapter (模型专属, 唯一手写代码之一)
│       ├── config/model.yaml      # 声明 (唯一手写配置: 形态 + 后端 + 验证 + bench 口径)
│       ├── scripts/               # 用户脚本: install_{nz_pass,prefix_attn}.sh / gen_requests.py
│       ├── results/               # 报告归档 (json+md 入库; csv/raw gitignored)
│       ├── run.sh env.sh          # 薄封装 core/pipeline + 运行环境
│       └── docs/                  # DEPLOYMENT_GUIDE.md + reports/ aicore/ prefix-attention/
├── tools/                         # varlen / atc_utils / compare / sweep / parse_dump / parse_profiling
├── tests/                         # tiny_e2e (需 NPU 的脚本, 手动跑)
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
3. **Graph 抽象 + 两层 shape**：`core/graph.py`——从 AIR pbtxt 派生 io_spec（动态维），bundle 记具体 shape；落地 §6 端到端
4. **环境准备（pass / 自定义算子）**：`third_party/custom_development_code` 与 `third_party/ascend-ops` 均为 submodule；`core/setup_scripts.py` 提供"yaml 填脚本路径"的接口（框架不假设构建方式）；`models/qwen2.5-0.5b/scripts/install_{nz_pass,prefix_attn}.sh` 为示例脚本，装到 `opp/vendors/<各自 vendor 名>/`（§7.1：pass 无 per-model 隔离，算子靠 env 指向）
5. **Source（torch）**：`core/source.py`——torch 加载（来路① hf 权重；来路② 客户源码未实现，见 §10）
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
| `tests/tiny_e2e.py` | NPU（~几百 MB 显存）+ ATC | 极小模型全链路：export（含 `_source_name` 断言）→io_spec/bundle/manifest→ATC→**om_acl** run+compare→**ge_session** run+compare。验**运行时链路**（图序配对、契约解析、两后端执行），不验模型语义 |
| `models/<m>/run.sh --device N` | NPU + 真实权重 | 真实模型全链路 + **两道精度门**（§10）：`reference`（原版 HF vs 适配后 eager，验适配）与 `compare`（运行时输出 vs golden，验编译）；任一 FAIL 非 0 退出 |
| `python3 -m core.bench --config …` | NPU + 请求池 | 多实例并发性能（变长负载回放），报告归档 `models/<m>/results/`；**不含精度**（精度归上一行的两道门） |

约定：`tests/*.py` 都是**需 NPU 的脚本**（手动跑，不进 CI）；仓库不维护纯 CPU 单测——
静默失败点（图序配对/硬失败、契约解析、门禁）由 `tiny_e2e` 端到端兜底，模型语义由两道精度门兜底。

## 15. 待定/依赖项

- ~~**io_spec 的 node↔logical 配对**~~ **已实测定论（阶段二，OM 运行 + golden 比对验证）**：pbtxt 的 Data 节点带 `index` 属性，按 index 排序得到的是**图侧喂入序**，但它 **≠ dynamo_export 入参序**。Qwen2.5-0.5B 实测：`arg1_1(index0)=actual_seq_lengths, arg4_1(index1)=input_ids, arg7_1(index2)=position_ids`，而 forward 入参序是 `(input_ids, position_ids, actual_seq_lengths)`。
  - 证据：按 forward 序喂 OM → `ApplyRotaryPosEmb` tiling 崩（rope Gather 读到 asl，cos `[2,1,64]` vs q `[32,14,64]`，报 "all input dim1 must equal"；旧 `atb/acl_infer` 同样崩，排除运行时嫌疑）；按图序喂 → 执行通过且与 eager golden 比对 PASS（cosine 0.99996，om_acl 与 ge_session 输出逐字节一致）。
  - **解法（已落地）**：torchair 上游 [PR#3675](https://gitcode.com/Ascend/torchair/pull/3675) 给 Data 节点加了 `_source_name` 属性（dynamo 的 `LocalSource/GlobalSource` → `local:<forward 入参名>`）。本机 torch_npu 2.9.0.post2 的内置 torchair 还没带，故 `core/_torchair_source_name.py` 按同一机制回移（patch `_npu_backend` 采集 `arg_pos_to_source` + `parse_input` 按 `data_index` 取名 + `ge.Data` 写属性；上游原生支持时自动 no-op，异常只 WARN）。`graph.from_air` 据此**按名字自动配对**并输出图序 io_spec —— 无需人工声明、无需探测。
  - 对齐关系（实测）：`parse_input` 的 `data_index = self.graph.num_inputs` 与 `_try_get_metadata_from_dynamo` 返回的 `arg_pos_to_source` 下标一一对应（参数/buffer/符号 shape 各占一位；它们后续被冻结成 Const 或在图里重新编号，但 parse_input 时刻是对齐的）。注意必须在 `_npu_backend(gm, ...)` 处采集——`_NpuFxCompiler.__call__` 拿到的 gm 已无 dynamo 元数据（`_try_get_metadata_from_dynamo` 返回 None）。
  - 注：pbtxt 格式是 `op:"Data"`（非 op_type），且因 frozen 权重内嵌可达 GB 级，用 grep 流式提取。旧 `atb` config 的 `arg1_1→act` 是**对的**（此前文档判其为 bug 有误）。
- ~~**动态 shape 的 ATC 机制**~~ **已定论（阶段二实测）**：动态图**不需要** `--dynamic_batch_size`/`--dynamic_dims` 分档——`OmAclBackend.compile` 检测到 io_spec 有动态维就不传 `--input_shape`，GE 运行期自行特化，同一 OM 可跨 shape 复用（T=32 导出的 OM 跑 T=2080 成功，见 §6）。分档只作为**可选优化**（减少运行期特化开销），当前未实现。`graph.dynamic.max_seq_len` 与 ATC 无关，是图常量长度（§6）。
  - ~~仍待办：`run_atc` 硬编码 `--framework=1`、`shell=True`~~ **已落地（P1/A5）**：`tools/atc_utils.py` 拆出纯函数 `build_atc_argv`/`normalize_aicore`，`subprocess.run(argv)` **不走 shell**（路径含空格安全、无注入面）；`--framework=1`（AIR）是常量——图只有 AIR 一种。
- ~~GeSession 具体 C++ API~~ **已实测确认**（CANN 9.0.0，`runtime/backends/gesession_backend.cpp`）：`ge::GEInitialize({ge.graphRunMode, ge.exec.deviceId[, aicore_num]})` → `aclInit` → `ge::Session({ge.session_device_id, ge.exec.precision_mode})` → `ge::Graph::LoadFromFile(<name>.air)` → `AddGraph(id, graph)` → `CompileGraph(id)` → `aclrtSetDevice` + `aclrtCreateStream` → `LoadGraph(id, {}, stream)` → 输入构造为 `gert::Tensor`（`StorageShape` origin+storage 同填、`StorageFormat(FORMAT_ND, FORMAT_ND, {})`、`SetDataType`、`SetData(TensorData(devPtr, nullptr, bytes, kOnDeviceHbm))`）→ `ExecuteGraphWithStreamAsync(id, stream, inputs, outputs)` + `aclrtSynchronizeStream`。**动态维无需 ATC 分档**：首次执行做 shape 特化，输出 device 缓冲由 GE 自动分配（`outputs[i].GetSize()/GetAddr()/GetShape()` 取实际值，用完 `aclrtFree`，多次执行需按地址去重释放）。清理序：free 输入/输出 → `aclrtDestroyStream` → `session.reset()` → `aclrtResetDevice` → `GEFinalize` → `aclFinalize`。链接 `ge_compiler ge_runner ge_common ge_common_base graph graph_base ascendcl`，须 `-D_GLIBCXX_USE_CXX11_ABI=0` + C++17。
- ~~pass 构建/隔离~~ **已定论（见 §7.1）**：fusion pass 由 CANN **自动扫描 `opp/vendors/*/custom_fusion_passes/` 全部加载**（无 env、无优先级）→ **没有 per-model 隔离**，同名 pass 跨 vendor 会重复注册 → ATC/TBE 崩（实测 `MatMulWeightNZPass`）。故约定"每个 pass 装进它自己的 vendor 目录、全局一份"，由安装脚本负责幂等与同名检测。`ASCEND_CUSTOM_OPP_PATH` 只服务**自定义算子**（优先级 2>4>1>3），与 fusion pass 无关；算子的 vendor 名由三方工程写死（PIA = `custom_prefix_attn`），隔离靠 env 指向而非目录名。
- ~~C++ 读 manifest/io_spec/bundle 用 nlohmann/json（header-only）~~ **已落地**：vendored `third_party/nlohmann/json.hpp`（v3.12.0 单头文件，git 跟踪，离线可构建）。
- form ② PyTorch 源码的加载约定（module 路径 + class 名 + 权重；构造参数来源、分片/safetensors、dtype 转换）——遇到实例再细化（原猜测性桩 `_from_source_code` 已按 YAGNI 删除，届时从头实现，见 §10）。
- **形态③ ONNX 与来路② 客户源码均已按 YAGNI 移除**，需要时从 git 历史取回（**重新引入的触发条件**：客户真的给出 ONNX 图，或一份非 transformers 结构的 PyTorch 源码模型）。
