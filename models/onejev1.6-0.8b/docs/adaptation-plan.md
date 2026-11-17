# OneJev1.6-0.8B 适配流程与迁移手册（adaptation-plan）

> 状态快照：2026-11-17，旧环境（100 容器，Ascend910_9382 ×8，CANN 9.1.0）。
> 本地 main 已同步 origin/main（604fe84）：上游 `fc9be4b refactor(runtime):
> bind executor inputs positionally from ge-records/3` 已把 executor 泛化为
> N 输入位置绑定（取代本目录早前的 ge-records/1 本地改动，见 §4.1）。
> 本文档面向**换环境续作**：包含全部已核实事实、已完成工作与证据、待办
> runbook、契约定义与坑清单。交付形态参考
> `models/startlux-decision-v1.1-0.8b-instant-text/`（下称 -text）。

## 1. 目标与范围（已与用户确认）

- 把 `models/onejev1.6-0.8b/model/model.py`（一次性原型脚本，已验证过导出
  思路）产品化为 -text 风格的**自包含极简交付**：部件间零 import，只通过
  `work/` 文件契约交互。
- **v1 图契约：单图+文本多模态单一 AIR**（用户确认，非纯文本）：
  9 输入 → `scores [1,26]`（A–Z letter logits），B=1 由图钉死，S/P 动态。
- 验证 case：**官方示例构造 fixtures**（用户确认，无业务文件）：
  `fixtures/cases.jsonl` 14 条（choice K=2..8 / noul / score 3–5 级）+ 5 张
  PIL 生成的截图风 PNG。
- v1 不做（README 需写明限制）：纯文本 case、多图、视频、choice>256 分块、
  debias>1（cyclic shift）、MTP、业务 gate 脚本。

## 2. 背景事实（全部已核实）

### 2.1 官方模型

| 项 | 值 |
|---|---|
| 模型 | https://huggingface.co/OmniJev/OneJev1.6-0.8B（commit `bcd88a382f314b26cb7b1b7f369c1ed3502ee7a0`） |
| 架构 | `Qwen3_5ForConditionalGeneration`（model_type qwen3_5，image-text-to-text，Qwen3.5-0.8B 三个全量微调的权重平均） |
| text | 24 层 = 18 `linear_attention`(GDN) + 6 `full_attention`（interval 4），hidden 1024，head_dim 256，partial_rotary 0.25，MRoPE interleaved [11,11,10]，vocab 248320，`tie_word_embeddings=true` |
| vision | 12 层 768d，patch16 / merge2 / temporal2，位置表 2304（48×48，bilinear、align_corners=True），out_hidden 1024，无 deepstack |
| 特殊 token | image 248056、vision_start 248053、vision_end 248054（与原型一致） |
| 权重 | `model.safetensors` 单文件 **1,706,030,528 B**；473 tensors：`model.language_model.*`(320) + `model.visual.*`(153)，**无 mtp.\*、无独立 lm_head** |
| checkpoint 标注 | transformers_version **5.17.0** |
| decision_config.json | `{"format_id": "onejev-slot-v2", "package": "flide"}`（无 letter_token_ids——与 StartLux 不同，slot ids 须从 tokenizer 推导） |

### 2.2 官方决策语义（flide 0.1.16，PyPI/GitHub OmniJev/Flide）

- `flide_onejev` 即 `onejev-slot-v2`：读出 = tied lm_head 的槽位字母行
  logits **前 K 列** → `softmax(logits / T)`；槽位 = A–Z（可扩双字母至 256，
  v1 只用 26）。
- 仓库**无 readout.json、config 无 modeling_overrides** → `attention=causal`
  （与 StartLux 的双向不同！**不要打双向补丁**）、`readout=slot_lm`。
- **无 calibration.json → 温度全 1.0**（`Calibration()` 默认）。
- prompt = SYSTEM_PROMPT + `<state>` 块 + tagged 问题布局 + chat template
  （`enable_thinking=False`）；每问题一个分支（debias=1）。
- 本机 OneJev tokenizer：A–Z slot ids = **连续 [32..57]**（单 token 往返已验证）。

### 2.3 版本与环境约束（关键）

- 原型依赖的 API **仅 transformers 5.17.0 存在**：`transformers.vision_utils`
  的 `get_vision_{interpolation_indices_and_weights,position_ids,cu_seqlens}`、
  vision forward 返回 `pooler_output` 且原生接受 precomputed kwargs
  （`interp_indices`/`interp_weights`/`position_ids`/`cu_seqlens`——注意原生
  kwarg 键是 `position_ids`，原型的 `precomputed_vpos` patch 在产品化时可删）。
- 容器全局 transformers 是 **5.8.1（StartLux 交付验证组合，禁止改动）**。
  → 方案：task-local `work/deps/py`（pip --no-deps --target 装
  transformers==5.17.0 + tokenizers 0.23.x），脚本自行插入 sys.path 并断言版本。
- 5.17.0 要求 torch≥2.5（本机 2.12.0+cpu ✓）、tokenizers ≥0.23.1,<0.24
  （容器 0.22.2 不满足 → 必须随装）、hub≥1.5（容器 1.33 ✓）、safetensors≥0.8（✓）。
- `get_max_seqlen`（transformers/utils/generic.py:308）：eager 且未预计算时
  返回 **None**，不产生数据依赖 Python int —— 原型路径安全（已读源码确认）。
- 旧环境实测组合：CANN 9.1.0（V100R001C11SPC001B243，aarch64）、npu-smi
  25.5.t6.b001、Ascend910_9382、Python 3.12.13、torch 2.12.0+cpu、
  torch_npu 2.12.0、Pillow 13.0.0（无 torchvision，图片处理用
  `Qwen2VLImageProcessorPil`，与 torchvision 后端已交叉验证 bf16 逐位一致）。

## 3. 交付形态与文件契约

### 3.1 目录（√=已完成，⬜=待做）

```
models/onejev1.6-0.8b/
√ model/model.py            # 原型（保留不动，实验记录；CWD=权重目录运行）
√ model/__init__.py
√ model/generate_inputs.py  # cases.jsonl + 权重目录 → work/inputs/{inputs.bin,records.json,readout.json}
√ model/golden.py           # work/inputs + 权重 → work/golden/{outputs.bin,result.json}（CPU FP32 官方整模）
⬜ model/modeling_onejev.py # Decision 包装 + 全部 NPU/导出 patch（§6.5 设计）
⬜ model/export_air.py      # io_spec + -text 导出机制 → work/build/onejev.air
√ vendor/                   # flide 0.1.16 纯 Python 节选（prompt/calibrate/media/slots/schema + LICENSE + SOURCE.md）
√ native_proto/             # ChunkGatedDeltaRule proto（与两个 startlux 适配器同源同内容）
√ scripts/prepare.sh        # task-local py deps + chunk proto + WeightNzAndMatMulV3Pass
√ fixtures/                 # cases.jsonl(14) + images/(5 PNG) + make_images.py + README.md
⬜ run.sh                   # 一键串联（仿 -text）
⬜ README.md                # 交付文档（仿 -text 结构）
√ docs/adaptation-plan.md   # 本文档
```

共享部件：`runtime/build/executor`（已泛化 N 输入，见 §4.1）、
`tools/compare.py`（cosine≥0.999 / relL2≤0.02）、`runtime/build.sh`。

### 3.2 图 IO 契约（9 输入，顺序=records 顺序=图 Data 顺序，不得偏离）

| # | 名称 | dtype | shape | 动态轴 | 语义 |
|--:|---|---|---|---|---|
| 1 | input_ids | int64 | [1,S] | 1 | 展开后 prompt token（image pad ×P/4） |
| 2 | pixel_values | bfloat16 | [P,1536] | 0 | 官方 processor fp32 产物主机侧 cast bf16 |
| 3 | image_grid_thw | int64 | [1,3] | — | t=1,h,w |
| 4 | position_ids | int64 | [3,1,S] | 2 | 官方 `Qwen3_5Model.get_rope_index`（MRoPE 三轴） |
| 5 | interp_indices | int64 | [P,4] | 0 | bilinear 位置表采样 tap（side=48, align_corners=True, merge=2） |
| 6 | interp_weights | float32 | [P,4] | 0 | 对应权重 |
| 7 | vision_position_ids | int64 | [P,2] | 0 | vision 轴向 (h,w) id（merge-block 序） |
| 8 | cu_seqlens | int32 | [2] | — | [0, P] |
| 9 | feature_index | int64 | [1,S] | 1 | image token 位→merged 特征行 0..P/4-1，其余 -1 |
| 出 | scores | float32 | [1,26] | — | 末 token × 26 slot 行（tied lm_head，fp32） |

records schema `ge-records/3`（上游 executor 的唯一 schema，位置绑定）：
`{"schema","file","output_name":"scores","slot_token_ids":[26 ids 供审计，
executor 忽略],"cases":[{"id","samples":1,"tensors":[{name,dtype,shape,offset,bytes}×9]}]}`；
`tensors` 顺序=图 Data 输入顺序，name 仅注记，`samples` 为业务行数（每 case
一个 System One 请求分支 = 1）。executor 不再做 dtype/shape 校验（上游已放开），
**严格校验全部在 generate_inputs.py 侧**（§4.2 审计断言）。
readout schema 沿用 `startlux-text-readout/1`（labels/letters_asked/temperature=1.0）。
golden/输出布局沿用 `ge-compare/1`。

## 4. 已完成工作与证据（旧环境，全部未 commit）

### 4.1 executor N 输入支持（已由上游完成，本地改动作废）

- 上游 `fc9be4b refactor(runtime): bind executor inputs positionally from
  ge-records/3`（origin/main，2026-11-17）已把 `runtime/executor.cpp` 泛化：
  `tensors[]` 按位绑定图输入（任意个数/dtype/shape），每 case 带 `samples`
  业务行数，`output_name` 命名唯一输出；删除了 text-only 硬编码校验、
  `--case-filter`、AicoreSpec 预检等。上游验证：qwen 100-case 精度门禁 PASS
  （改动前后逐位一致）、5 实例 3×3000 请求 bench 0 错误、ctest 1/1。
- 本地早前的 Lane A 改动（ge-records/1 双 schema 版）**已被取代并作废**：
  runtime 改动存于 `git stash`（`laneA-executor-ge-records-1`，仅作参考，
  可 drop）；配套 `tests/executor_records_contracts.cpp` 已删除。
  Lane A 当时的回归证据（startlux-text 18 case compare PASS、新旧 schema
  输出 sha 一致）验证的是旧实现，仅作历史参考。
- onejev 侧对齐动作（已完成）：generate_inputs 写 `ge-records/3`
  （+`samples:1`），golden/审计脚本同步改 schema 串。
- ⚠️ bench 预热注意：新 executor 的 distinct-shape 预热键是 `tensors[0].shape`
  （即 input_ids 的 [1,S]），之后用同 seed 随机请求补足 warmup 次数；onejev
  的 (S,P) 组合覆盖依赖随机补足，正式 bench 时 warmup 给足（≥32/实例）。

### 4.2 vendor/ + fixtures/ + generate_inputs + golden（模型侧部件）

- vendor：`flide_onejev/prompt.py`、`calibrate.py` **与官方 wheel 逐字节一致**
  （cmp 验证）；`media.py` 保留渲染三函数（逐字）；`slots.py` = 官方
  `_verify_slots` 抽独立函数；`api/schema.py` pydantic→dataclass（字段/校验
  等价）；LICENSE + SOURCE.md 齐。
- fixtures：14 case（S=388..1083，P=1024..3520，grid 1×32×32 / 1×32×56 /
  1×48×64 / 1×44×80 / 1×30×40），make_images.py 固定 seed 可重现。
- generate_inputs.py：严格 JSONL 校验 → vendored 官方渲染（chat template
  enable_thinking=False）→ `Qwen2VLImageProcessorPil` → image pad 展开
  ×(grid.prod/4) → feature_index → **meta-device `Qwen3_5Model.get_rope_index`**
  （不加载权重）→ vision_utils 三件预计算 → 9 张量打包（bf16 经
  `view(torch.uint8)` 落盘）+ slot_token_ids 写 records。
- golden.py：CPU FP32 官方**整模** forward，故意只读 input_ids/pixel_values/
  image_grid_thw 三个张量（其余 6 个是图侧预计算，保持参考独立性）；
  `mm_token_type_ids` 从 input_ids 派生（5.17.0 带图 forward 必需）；
  scores = `logits[0,-1,slot_ids]`；ensure_weights 用 safetensors 头校验
  （download.log 存在才检查 DONE，不绑定下载器产物）。
- **证据**：
  - prompt 交叉验证：vs 官方 flide 0.1.16（/tmp 独立安装）14/14 **byte-identical**；
  - get_rope_index 交叉验证：meta-device vs 手工转写 14/14 一致，纯文本退化=三轴 arange；
  - 图片后端交叉验证：PIL vs torchvision（/tmp 装 0.27.0+cpu）5 张图 bf16 后 diff=0；
  - inputs 审计：9 张量顺序/dtype/shape/bytes、pad 数==P/4、feature_index
    连续递增、cu==[0,P]、温度全 1.0、slot ids==[32..57] 全过；
  - 可重复性：重跑 generate_inputs/make_images 全部 sha256 一致；
  - 负例校验：多余字段/缺图/双问题/单 criterion 均 exit 1 带行号报错。

### 4.3 材料、依赖与 golden 实跑（旧环境产物，work/ 下 git-ignored）

- 权重：`work/onejev-materials/model-bcd88a382f314b26cb7b1b7f369c1ed3502ee7a0/`
  （hf-mirror 下载，大小精确匹配 1,706,030,528 B，download.log 有 DONE）。
- task-local deps：`models/onejev1.6-0.8b/work/deps/py`（transformers 5.17.0
  + tokenizers 0.23.3，pip --no-deps --target）。
- `scripts/prepare.sh` 已实跑 ✓：chunk proto .so 构建成功
  （`work/deps/chunk_gated_delta_rule_proto/install/lib/`），
  WeightNzAndMatMulV3Pass 安装成功（源：`third_party/custom_development_code`）。
- **golden 全量实跑完成**：14/14 case，CPU FP32，每 case 2.3–10.0s（总 ~2min），
  产物 `work/golden/{outputs.bin,result.json}`，日志 `work/logs/golden-run.log`。
- **契约交叉验证通过**（`work/verify/contract_crosscheck.py onejev-00`）：
  图路径（9 张量：预计算 vision 输入 + pooler_output + feature_index scatter +
  packed position_ids + language_model）vs 官方整模 forward vs golden：
  **cosine=1.000000000，maxabs=0**（两条路径都与 golden 逐位一致）。
  → 9 输入契约与官方语义等价已在 CPU 上闭环。
- `work/verify/` 下还有 inputs_audit / rope_crosscheck / prompt_crosscheck /
  contract_crosscheck 四个脚本（git-ignored，**迁移时要带**）。

### 4.4 未做（新环境待办）

- ⬜ **NPU 冒烟导出**（原型脚本，门槛，见 §6.4）——旧环境被中断未跑。
- ⬜ Lane C 产品化部件：modeling_onejev.py / export_air.py / run.sh / README.md。
- ⬜ 精度闭环（executor infer + compare 门禁）。
- ⬜ 性能（单实例逐 case + 多实例扩展 + aicore 对照）与文档附录表。
- √ git：本目录（含本文档）已 commit 并推送 origin/main（见 §5.2）；后续
  Lane C / 精度 / 性能各自独立提交。

## 5. 新环境要求与迁移清单

### 5.1 环境要求

- aarch64 CANN 容器：Ascend-cann-toolkit **9.1.0**（V100R001C11SPC001B243）、
  npu 驱动 25.5.t6.b001 级别、Ascend910_9382（64GB HBM；冒烟/精度 1 芯片即可，
  多实例 bench 建议独占）。
- 容器 Python 3.12 + torch 2.12.0+cpu + **torch_npu 2.12.0（含 torchair）**；
  全局 transformers 5.8.1 保持不动；Pillow、numpy 2.x。
- g++（C++17，`_GLIBCXX_USE_CXX11_ABI=0`）、cmake、git。
- 网络：hf-mirror.com（权重）、pypi.org 或 pypi.tuna.tsinghua.edu.cn
  （transformers 5.17.0 / tokenizers 0.23.x）。github 可选（flide 已 vendor）。
- 磁盘 ≥ 25GB（权重 1.7G + AIR ~2G + kernel_meta + deps + golden/inputs/结果）。

### 5.2 迁移清单（旧环境 → 新环境）

代码已入库：本目录（vendor/fixtures/model/native_proto/scripts/docs）随
`feat(onejev)` 提交推送至 origin/main（github fengz72/ascend-ge-adapters），
executor 的 N 输入支持在更早的上游提交 `fc9be4b`（ge-records/3）里。
新环境直接：

```bash
git clone git@github.com:fengz72/ascend-ge-adapters.git   # 或经 gh-proxy 的 https fetch
cd ascend-ge-adapters && git log --oneline -3             # 应含 feat(onejev) 与 fc9be4b
```

代码之外需要迁移/重建的（均 git-ignored）：
1. `third_party/custom_development_code/`（untracked；prepare.sh 的 WeightNZ
   pass 需要；上游 https://gitcode.com/CharmanderFang/custom_development_code
   main 分支，clone 到 third_party/ 下）
2. 权重：`work/onejev-materials/model-bcd88a3…/`（1.7GB；rsync 或按 §6.1 重下）
3. `models/onejev1.6-0.8b/work/verify/*.py`（4 个验证脚本：inputs_audit /
   rope_crosscheck / prompt_crosscheck / contract_crosscheck；旧环境 rsync：
   `rsync -av <旧>/models/onejev1.6-0.8b/work/verify/ <新>/models/onejev1.6-0.8b/work/verify/`）
4. task-local deps 与 inputs/golden：不用带，按 §6.2/§6.3 重建
   （prepare.sh 需 pypi；generate_inputs ~1min + golden ~2min，CPU 即可）。

## 6. Runbook（新环境从零到交付）

以下 `<ROOT>` = 仓库根，`<ADAPTER>` = `<ROOT>/models/onejev1.6-0.8b`，
`<WEIGHTS>` = 权重目录（§6.1 的下载目标）。

### 6.1 材料下载（若未迁移权重）

```bash
mkdir -p <ROOT>/work/onejev-materials/model-bcd88a382f314b26cb7b1b7f369c1ed3502ee7a0
cd <ROOT>/work/onejev-materials/model-bcd88a382f314b26cb7b1b7f369c1ed3502ee7a0
for f in chat_template.jinja config.json decision_config.json generation_config.json \
         processor_config.json tokenizer.json tokenizer_config.json README.md model.safetensors; do
  curl -sL --retry 5 --retry-all-errors -C - \
    "https://hf-mirror.com/OmniJev/OneJev1.6-0.8B/resolve/main/$f" -o "$f" && echo "OK $f" >> download.log
done
echo "DONE $(stat -c %s model.safetensors)" >> download.log   # 必须 = 1706030528
```

（huggingface.co 直连不通时用 hf-mirror；`-C -` 支持断点续传；下载完核对
model.safetensors 字节数。）

### 6.2 依赖构建

```bash
cd <ADAPTER> && bash scripts/prepare.sh
# 产出: work/deps/py（transformers 5.17.0）、work/deps/chunk_gated_delta_rule_proto/
#       install/lib/libchunk_gated_delta_rule_compat_proto.so、WeightNZ pass 装入 CANN
bash <ROOT>/runtime/build.sh        # executor（上游 ge-records/3 版）
ctest --test-dir <ROOT>/runtime/build --output-on-failure   # 1/1 应 Pass
```

### 6.3 输入与 golden（CPU，无需 NPU）

```bash
cd <ADAPTER>
export ONEJEV_MODEL_DIR=<WEIGHTS>
python3 model/generate_inputs.py            # → work/inputs（14 case，~1min）
python3 model/golden.py                     # → work/golden（~2min，每 case 2–10s）
python3 work/verify/contract_crosscheck.py onejev-00   # 期望 cosine=1.0
```

### 6.4 NPU 冒烟导出（门槛：5.17.0 × torch_npu/torchair 兼容 + 动态导出可行）

原型 `model/model.py` 从 `"."` 加载 → **CWD 必须是权重目录**；导出参数是
argv[1]。挑空闲芯片（`npu-smi info` 看 AICore 0% / HBM 低占用）：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=<空闲chip>
export PYTHONPATH=<ADAPTER>/work/deps/py
export LD_PRELOAD=<ADAPTER>/work/deps/chunk_gated_delta_rule_proto/install/lib/libchunk_gated_delta_rule_compat_proto.so
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TMPDIR=<ROOT>/work/tmp
mkdir -p <ADAPTER>/work/smoke && cd <WEIGHTS>
python3 <ADAPTER>/model/model.py <ADAPTER>/work/smoke   # 期望产出 onejev.air
```

通过判据：`<ADAPTER>/work/smoke/onejev.air` 生成（~1.7GB 量级）；无 dynamo
graph break 致导出失败；kernel_meta/fusion_result.json 落在 CWD（权重目录）
属预期污染，可清理。**失败则停下排查**（常见：LD_PRELOAD 缺失 →
ChunkGatedDeltaRule 原型未注册报错；transformers 版本不对 → 原型 import
`transformers.vision_utils` 直接 ImportError）。
进一步验证（推荐）：用 executor 对冒烟 AIR 跑一条合成 case 确认图可执行
（合成 records 可按 §3.2 契约手工构造，或等 6.5 后直接用正式输入池）。

### 6.5 Lane C 产品化部件（冒烟通过后写）

**modeling_onejev.py**（把原型 patch 收敛成模块，逐条注释理由）：
- `Decision(nn.Module)`：`backbone = Qwen3_5ForConditionalGeneration.model`；
  buffer `rows` = tied lm_head 的 26 slot 行（fp32；slot ids 用 vendor
  `verify_slot_token_ids` 从 tokenizer 推导，与 records 的 slot_token_ids 核对）。
- `forward(input_ids, pixel_values, image_grid_thw, position_ids,
  interp_indices, interp_weights, vision_position_ids, cu_seqlens,
  feature_index)`（顺序=契约）：
  `feats = backbone.visual(pixel_values, grid_thw=image_grid_thw,
  interp_indices=…, interp_weights=…, position_ids=vision_position_ids,
  cu_seqlens=cu_seqlens).pooler_output`（5.17.0 原生 kwargs，**删掉原型的
  `m.get_vision_position_ids` patch**）→ `text = embed(input_ids)` →
  `selected = feats.index_select(0, feature_index.clamp_min(0).flatten())
  .reshape_as(text)` → `embeds = where(feature_index>=0, selected, text)` →
  `hidden = backbone.language_model(inputs_embeds=embeds,
  position_ids=position_ids, use_cache=False).last_hidden_state` →
  `scores = hidden[:, -1].float() @ rows.T`。
- 保留原型 patch（对象是 5.17.0 的 `modeling_qwen3_5`）：
  1. `Qwen3_5VisionAttention.forward` → 单图全量 eager attention（忽略
     cu_seqlens；**v1 单图契约**）；
  2. `torch.library.custom_op("onejev::chunk")` + `register_fx_node_ge_converter`
     → `ge.custom_op("ChunkGatedDeltaRule", …, attrs={"scale_value":
     ge.attr.Float(128**-0.5)}, outputs=["out","final_state"])[0]`；
  3. `m.torch_chunk_gated_delta_rule = delta`（q/k L2-norm eps1e-6 →
     flatten(B,S) → zero initial_state (B,H,D,D) → actual_seq_lengths=full(B,S)
     → g fp32；返回 reshape (B,S,H,D)）；
  4. `m.F.softplus = export_softplus`（t>20 直通，否则 log1p(exp(clamp))）；
  5. `m.causal_conv1d_fn = ge_causal_conv4`（bias-free 4-tap SiLU depthwise
     → conv2d(padding=(0,3)) 截断）；
  6. `torch._dynamo.config.specialize_float = True`（导出期，用
     `torch._dynamo.config.patch` 上下文，参照 -text export_air.py 第 166 行注释）。
- 加载：`from_pretrained(ONEJEV_MODEL_DIR, dtype=torch.bfloat16,
  attn_implementation="eager")`。

**export_air.py**（机制逐段抄 -text 的 `model/export_air.py`）：
`weights_dir()`(ONEJEV_MODEL_DIR)、`_reorder_data_nodes_first`、
`parse_bindings`、`_register_bind_input`（ge_adapters::bind_input 打
`_ge_adapter_name`）、`export_torchair`（mark_static/mark_dynamic 按 io_spec、
`CompilerConfig().experimental_config.frozen_parameter=True`、
`with torch._dynamo.config.patch(specialize_float=True)`、`dynamic=True`、
导出后 parse_bindings 双射校验、删 dynamo.pbtxt）。
- `io_spec()`：§3.2 表（dynamic_dims：input_ids[1]、pixel_values[0]、
  position_ids[2]、interp_indices[0]、interp_weights[0]、
  vision_position_ids[0]、feature_index[1]；image_grid_thw/cu_seqlens 全静态；
  输出 scores float32 [1,26]）。
- `example_inputs(device)`：小 grid 合法样例——如 grid=[1,4,4]→P=16、
  count=P/4=4 个 image token、S≥6（**动态轴 example 尺寸必须 ≥2**）；
  ids 用真实特殊 token（248053 + 248056×4 + 248054 + 文本 token）；
  position_ids 用 meta-device get_rope_index 实算；interp/vpos/cu 用
  vision_utils 实算；feature_index 与 ids 一致；pixel_values bf16 zeros。
  （原型第 165–214 行就是现成模板，改成小 grid + 从 ONEJEV_MODEL_DIR 加载。）
- 产物：`work/build/onejev.air`；CLI `--device/--export-dir/--export-name`。

**run.sh**（仿 -text，固定参数段在顶部）：环境检查（ONEJEV_MODEL_DIR、
`source set_env.sh`、`ASCEND_RT_VISIBLE_DEVICES=$DEVICE_CHIP`、
`export PYTHONPATH=<ADAPTER>/work/deps/py`）→ generate_inputs → golden →
prepare.sh → runtime/build.sh → LD_PRELOAD 导出 → LD_PRELOAD executor infer
（`--air work/build/onejev.air --records work/inputs/records.json
--output-dir work/output`）→ `tools/compare.py --golden work/golden
--output work/output --cosine-min 0.999 --relative-l2-max 0.02` → 可选
bench 段（--iterations/--warmup/--seed/--instances/--aicore-num/--readout，
readout 用 work/inputs/readout.json）。日志 tee 到 work/logs/。

**README.md**：仿 -text 结构（背景/官方地址/业务约束/五部件契约表/环境表/
运行/IO 表/性能表）；性能与调优结果记附录表（AGENTS.md 约定：英文文件名
文档附录表格，不新增 CSV 汇总）。

### 6.6 精度闭环

```bash
cd <ADAPTER> && bash run.sh --device <chip>
```

- 首跑 TBE 冷编译慢属预期（-text 参考：CompileGraph ~17.5min、LoadGraph
  ~3.7min；onejev 含 vision 新 kernel 可能更久）。
- 失败定位三级：golden(CPU 官方) → NPU eager 直跑（modeling_onejev 不导出
  直接 forward 同一 case）→ AIR（executor）。compare 不过时优先看
  vision 段（bf16 pixel cast、interp 预计算）与 GDN 段（chunk proto scale）。
- 通过后（可选强校验）：executor 输出 vs golden 全 14 case compare PASS 即
  精度门禁达成。

### 6.7 性能与收尾

- 单实例：`run.sh --device <chip> --iterations 2000 --warmup 16`（或分档），
  逐 case 模型段时延表进 README。
- 多实例扩展：`--instances 1..N`（参考 -text：N=4–5 达峰、每实例 HBM
  ~3.3GB 线性；onejev 图更大，HBM 会更高，先测 N=1 单实例占用再定档位）、
  `--aicore-num 12|24` 对照、`--readout work/inputs/readout.json` 计时 decider。
  每档 3 seed 轮，跨轮中位数进 README 附录表；共享机器注明负载背景。
- bench 预热：executor 按 `tensors[0].shape`（S）去重预热 + 随机补足，
  warmup 给足（≥32/实例）以覆盖 (S,P) 组合（§4.1 注意项）。
- 提交：英文 Conventional Commits（`feat(onejev): …`、`feat(runtime): …`），
  是否 push origin 由用户决定。

## 7. 坑清单（新旧环境通用）

1. **LD_PRELOAD 单一契约**：任何触碰 GE 的进程（export/infer/bench）都要
   preload chunk proto .so；`ASCEND_CUSTOM_OPP_PATH` 自动加载会 SIGSEGV/SIGILL
   （native_proto/README.md）。
2. **全局 transformers 5.8.1 不许动**（StartLux 验证组合）；OneJev 脚本自带
   sys.path 插入 + `__version__=="5.17.0"` 断言，跑 OneJev 的任何 python
   都要能看见 `work/deps/py`（run.sh 统一 export PYTHONPATH）。
3. dynamic=True 会把**未 mark 的轴也符号化**、把闭包 Python float 提升为
   隐式图输入 → 必须按 io_spec mark_static/mark_dynamic + specialize_float
   patch（-text export_air.py 注释写得很全）。
4. 动态轴 example 尺寸 <2 会被 dynamo 特化成静态图。
5. torchair 2.12 权重烘焙按**位置**对 Data 节点 → 必须 bind_input +
   `_reorder_data_nodes_first`（原型里 `aligned` patch 是粗糙版，产品化用
   -text 机制并做 parse_bindings 双射校验）。
6. pixel_values 契约是 **bf16**（主机侧 cast；图内首层随官方 dtype 行为）；
   golden 是 fp32 官方整模（独立路径），两者等价性已由 contract_crosscheck
   在 CPU 上证明（cosine=1.0）。
7. golden **故意不读** 6 个预计算张量（参考独立性）；不要"顺手优化"成读池。
8. OneJev 是 **causal** full-attention（无 modeling_overrides）——不要套用
   StartLux 的 bidirectional 补丁。
9. executor 只认 `ge-records/3`（上游 fc9be4b 起唯一 schema）：tensors 按位
   绑定、无 dtype/shape 校验、每 case 必带 samples——**输入合法性全靠
   generate_inputs 侧的审计断言**（§4.2），改契约要两边同步。
10. 原型 model.py 只能 CWD=权重目录跑（from_pretrained(".")）；产品化部件
    一律走 ONEJEV_MODEL_DIR。
11. 首跑 kernel_meta/TBE 缓存落在 CWD；run.sh 里 cd 到 adapter 目录保持
    work/ 卫生。
12. 共享 NPU：跑前 npu-smi 挑空闲 chip；bench 数据注明整机负载。

## 8. 附录：fixtures case 表（S/P 供 bench 参考）

| case | 类型 | S | P | grid |
|---|---|---:|---:|---|
| onejev-00 | choice(4) | 417 | 1024 | 1×32×32 |
| onejev-01 | choice(3) | 399 | 1024 | 1×32×32 |
| onejev-02 | score(4) | 417 | 1024 | 1×32×32 |
| onejev-03 | noul | 388 | 1024 | 1×32×32 |
| onejev-04 | choice(5) | 617 | 1792 | 1×32×56 |
| onejev-05 | noul | 588 | 1792 | 1×32×56 |
| onejev-06 | score(4) | 603 | 1792 | 1×32×56 |
| onejev-07 | choice(6) | 953 | 3072 | 1×48×64 |
| onejev-08 | choice(3) | 903 | 3072 | 1×48×64 |
| onejev-09 | score(5) | 1049 | 3520 | 1×44×80 |
| onejev-10 | choice(8) | 1083 | 3520 | 1×44×80 |
| onejev-11 | choice(2) | 430 | 1200 | 1×30×40 |
| onejev-12 | noul | 1011 | 3520 | 1×44×80 |
| onejev-13 | choice(4) | 1039 | 3520 | 1×44×80 |

golden 实测（旧环境 CPU FP32，8 线程）：每 case 2.3–10.0s，全量 ~2min。
