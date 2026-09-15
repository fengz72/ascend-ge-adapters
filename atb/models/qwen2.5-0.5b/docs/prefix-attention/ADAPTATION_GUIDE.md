# Qwen2.5-0.5B prefix-attention 适配部署指南

> 适用版本: prefix-attention 分支 (KV 内嵌版算子, prefix-attention 工程 `d6a5a2d`)
> 性能与精度数据见同目录 [benchmark.md](benchmark.md), 本文档只讲怎么装、改了什么、怎么跑

## 1. PIA 算子安装与测试

### 1.1 构建 + 安装

```bash
cd prefix-attention
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

bash build.sh          # ~4min → build_out/custom_opp_openEuler_aarch64.run
bash build_out/custom_opp_openEuler_aarch64.run --quiet --install-path=$ASCEND_HOME_PATH/opp

cd torch_binding        # torch 接入包 (schema + eager + torchair converter)
pip wheel . --no-deps -w dist
pip install --force-reinstall --no-deps dist/*.whl
```

安装后算子位于 `$ASCEND_HOME_PATH/opp/vendors/custom_prefix_attn/`。

### 1.2 测试 (验收口径: 全 PASS)

```bash
source $ASCEND_HOME_PATH/opp/vendors/custom_prefix_attn/bin/set_env.bash
python3 tests/test_torch_binding.py
```

三项覆盖: eager vs CPU golden (<2e-3)、`.tensor` vs `.default` (bit 级)、
GE 整图 max-autotune (含图复用) vs eager (bit 级)。

**运行前置环境** (后续所有 prefix 相关命令通用):

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source $ASCEND_HOME_PATH/opp/vendors/custom_prefix_attn/bin/set_env.bash   # prefix 必须
# CANN tbe pywrapper 内嵌 /usr/bin/python3 无 numpy, GE 编译必需:
export PYTHONPATH=/usr/local/python3.11.15/lib/python3.11/site-packages:$PYTHONPATH
```

## 2. 模型侧改动

核心思想: **packed prefix-in-Q**。batch 内请求共享同一 prefix (system prompt 语义),
整层输入打包为 `[prefix(P), req0, req1, ...]` 穿过全部层 —— RMSNorm/MLP/Embedding
逐 token 独立, 数学与"每请求展开为 [prefix+own]"逐位等价, 但 prefix 的
K/V/输出只存一份、只算一份。

| 文件 | 改动 |
|---|---|
| `model/attention.py` | ① 新增 `prefix_ia_varlen_forward` + `register_prefix_ia()`: 注册为 transformers AttentionInterface (切入点在 RoPE 之后, 收已旋转的 3D TND q/k/v); 调 `npu_prefix_infer_attention_score.tensor`, q/k/v 传全长 (prefix KV 内嵌 key/value 头部, 算子内部处理, 无需切分), `actual_seq_lengths = actual_seq_lengths_kv = act` 同一 tensor; ② 基线 `npu_fia` 的 `sparse_mode` 3→2 (见 benchmark.md 3.1: sm3 在 GE 图编译下落 MIX 慢 tiling 分支, attention kernel 慢 ~2 倍, 输出逐位不变) |
| `model/varlen_utils.py` | 新增 `setup_prefix_attention(model, act, device)`: 向每层注入 act dummy tensor 与 2048² 压缩 causal mask buffer, 禁用 transformers 自带 `_update_causal_mask`, 预计算 RoPE cos/sin 表 (图常量) |
| `model/export_air.py` | 新增 `PrefixExportWrapper` (3 动态输入) 与 `export_prefix_air()`; `load_model()` 增加 `attn_implementation="prefix_ia"` 路径 |
| `model/prepare_air_inputs.py` | `--prefix P` 模式: 生成 packed 输入 3 个 .bin + eager golden logits |
| `atb/tools/varlen.py` | 新增 `generate_prefix_varlen_inputs(batch, seq, P)`: packed token/position/act 生成 |
| `config` / `run.sh` | prefix 模式 3 图输入; `--prefix` 全流程透传 (export/atc/bench, OM 名追加 `-prefix`) |

加载方式 (不修改 transformers 模型代码):

```python
ALL_ATTENTION_FUNCTIONS.register("prefix_ia", prefix_ia_varlen_forward)
model = AutoModelForCausalLM.from_pretrained(
    path, dtype=torch.float16, attn_implementation="prefix_ia")
```

## 3. 输入输出改动

### 3.1 图输入 (基线 3 个 → prefix 3 个, 结构同构、语义不同)

| 输入 | 基线 (FIA) | prefix (PIA) |
|---|---|---|
| 长度数组 | `actual_seq_lengths [N]` = cumsum(L_i), prefix 逐请求重复计入 | `act [N+1]` = **cumsum([P, L0, L1, ...])** — prefix 独立成 batch 0 (P=act[0]), 请求 i 为 batch i+1; act_q ≡ act_kv |
| `input_ids [T]` | 每请求自带一份 prefix token | `[prefix, req0, ...]` packed, **prefix 只一份** (物理 token 省 (N-1)×P) |
| `position_ids [T]` | 每请求 0..L_i-1 | prefix 行 0..P-1, 请求 i 行 P..P+L_i-1 (与展开布局逐位一致) |

- 示例 (P=20, 10×总长208): 基线 T=2080 token, prefix T'=20+10×188=1900
- **P 只存在于 act 值中** (act 形状固定 [batch+1]): 运行时任意 P (如每请求随机
  [20,25]) 换值不重编译, 图 shape 组合仅由 T' 决定

### 3.2 图输出

两者相同: `logits [N, vocab]` — 每请求最后一个 token。
prefix 图内取行索引: **`last_indices = act[1:] - 1`** (丢弃 prefix 段结束行
act[0]=P, 请求 i 末行 = act[i+1]-1)。算子输出 `[P+sum(L), N, D]`
(prefix 段输出在头部, 被丢弃)。

### 3.3 输入顺序纪律 (重要)

acl_infer / GESession 均按 **inputs 下标**映射图输入, **不校验名字**。
顺序必须与导出 pbtxt 的 Data 节点一致: `act(arg1_1) → input_ids(arg4_1) →
position_ids(arg7_1)`。同 dtype 同 shape 的输入喂反会**静默出错**
(基线链路曾因此 bug 修复, main `34e3e25`)。

## 4. GESession 适配

AIR 免 ATC 在线执行 (参照 cann/ge PR#743 RunGraphAsync 样例), 已提供两个工具:

### 4.1 工具

- **`atb/build/ge_infer`** (单发验证/计时):

```bash
./build/ge_infer --model air/qwen2.5-0.5b-prefix.air --device_id 12 \
    --output_dir <dir> --warmup 10 --bench 100 \
    --input "arg1_1:11:int64:ND:input_data_prefix/act.bin" \
    --input "arg4_1:1900:int64:ND:input_data_prefix/input_ids.bin" \
    --input "arg7_1:1900:int64:ND:input_data_prefix/position_ids.bin"
```

- **`atb/build/bench_ge_latency`** (多线程 sweep, 口径镜像 bench_latency):

```bash
./build/bench_ge_latency --model air/qwen2.5-0.5b-prefix.air \
    --sweep 1,2,3,4,5,6 --requests 8000 --warmup 50 \
    --prefix 20-25 --aicore-num 12 --device-id 12
# --prefix <P> 固定或 <P1>-<P2> 每请求均匀随机; --profiling 开算子级采集
```

架构: 单 Session 多图 (每线程独立 graph_id + stream), 请求闭环
DataGen→H2D→ExecuteGraphWithStreamAsync→Sync→D2H; N 图串行预编译 (~10s/图)
sweep 各档复用。限核经 `{ge::ir_option::AICORE_NUM, "12|24"}` 注入。

### 4.2 关键约束 (踩坑记录, 全部实测验证)

1. **CompileGraph 必须在 aclrtSetDevice 之前** — GE 编译期内部线程先建立
   context 绑定; 顺序反了 ExecuteGraphWithStreamAsync 报 `stream not in current ctx`
2. **不可用显式 `aclrtCreateContext`** (同因); 工作线程共享 SetDevice 的默认 context
3. **AICORE_NUM 必须挂 GEInitialize 全局选项** — 挂 AddGraph 图选项时
   CompileGraph 正常但 LoadGraph 报 `GetPlatformInfo failed`
4. **warmup 轮流打满所有图** (触发各自 shape 特化)
5. **tbe pywrapper 的 numpy 问题** — 见 1.2 环境说明, 缺失时 GEInitialize 返回 -1
6. 输入顺序 = Data 节点顺序 (见 3.3); GE 编译 ~10-24s/图, LoadGraph ~1.3-2.2s

GESession 在线路径与 ATC OM 离线路径**精度逐位一致**, 稳态 execute 相当
(benchmark.md), 部署可任选。

## 5. 基本功能复现

```bash
cd atb/models/qwen2.5-0.5b
source <1.2 节完整环境>

# ① 导出 prefix AIR (动态 shape, 冻结权重/mask/cos-sin 表)
./run.sh export --device 12 --prefix 20
# → air/qwen2.5-0.5b-prefix.air

# ② 生成 packed 输入 + eager golden (固定 P=20, seq=208, batch 10)
PYTHONPATH=<repo根> python3 -m model.prepare_air_inputs --device 12 --prefix 20
# → input_data_prefix/{act,input_ids,position_ids}.bin + golden_logits_prefix.bin

# ③ 精度验证 (GE 在线, vs 基线 golden)
../../build/ge_infer --model air/qwen2.5-0.5b-prefix.air --device_id 12 \
    --output_dir output_ge_prefix \
    --input "arg1_1:11:int64:ND:input_data_prefix/act.bin" \
    --input "arg4_1:1900:int64:ND:input_data_prefix/input_ids.bin" \
    --input "arg7_1:1900:int64:ND:input_data_prefix/position_ids.bin"

# ④ 对比 (python)
#    vs input_data/golden_output_om.bin (基线 FIA OM golden): 预期逐位一致 (max_diff=0)
#    vs input_data_prefix/golden_logits_prefix.bin (eager):  预期 cosine≈0.99996

# ⑤ 性能 (可选, 见 benchmark.md 口径)
../../build/bench_ge_latency --model air/qwen2.5-0.5b-prefix.air \
    --sweep 1,2,3,4,5,6 --requests 8000 --warmup 50 --prefix 20-25 \
    --aicore-num 12 --device-id 12
```

**预期结果** (验收口径):

| 项 | 预期 |
|---|---|
| 算子单测 | 3 项全 PASS |
| 固定回归 vs 基线 golden | max_diff=0, cosine=1.0, argmax 10/10 |
| 固定回归 vs eager golden | cosine≈0.99996 (图-vs-eager 正常水平) |
| 随机抽验 (P=25+随机 seq) | cosine≈0.99996, argmax 10/10 |
| GE 12\|24 随机负载 | PIA QPS +7~8%, E2E avg -6.6~7.6% (见 benchmark.md) |
