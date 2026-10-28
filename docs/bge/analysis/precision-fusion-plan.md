# BGE：低精度与剩余融合候选

日期：2026-10-26，北京时间。SE/候选开发只执行安装包读取、GE 文本图分析、配置准备和 CPU 图结构测试；本子任务没有初始化或使用 NPU。设备与正式实验由主控独立调度。

## 1. 原始模型同次最终图的事实

读取主控保存的 `work/bge-opt-20261026/graph-dump/pid_1997536_deviceid_2/ge_proto_00000000_graph_1_Build.txt`（38,245,179 bytes），该图是原始模型 GE mode=0 的本次编译产物，图中没有权重数据。

| 目标结构 | 最终图结果 | 对后续实验的影响 |
|---|---:|---|
| LayerNorm | 49 个 `LayerNorm` | 原图 49 组基础算子已被内置 Pass 融合，不再重复实现 |
| GELU | 24 个 `Gelu` | 原图 Erf 结构已融合；这个事实不表示所有 GELU 变体在数值上完全等价 |
| encoder MatMul+bias | 144 个 `MatMulV2`，全部已有三个非空输入 | encoder 的 bias 已融合，不再把源 ONNX Add 数量当待优化 kernel 数 |
| residual Add+LayerNorm | 没有 `AddLayerNorm` | 可以独立尝试剩余的残差融合 |
| 其他矩阵计算 | 55 个 `BatchMatMulV2` | 包括 attention 等，不能一律视为静态 bias MatMul |

包含根图与子图，共解析到 4,133 个 GE op 记录；其中许多是 Data、Const、形状节点，不等于实际执行 kernel 数。原图浮点 descriptor 为 `DT_FLOAT`，没有找到显式 `DT_FLOAT16`。这不证明隐藏的内部累加类型，也不替代 kernel/profiling 证据。

原始图分析结果位于远端候选目录的 `ge_proto_00000000_graph_1_Build.txt.precision.json`。解析工具 `inspect_ge_precision.py` 同时接受 GE proto 文本和 ONNX 风格 pbtxt，逐算子保存 input/output descriptor 与精度/kernel 属性；未知格式不推断为 FP32。

## 2. 低精度候选：先改变 GE 编译策略

CANN 9.0.0 的本机 `atc --help`、`ge_api_types.h` 和 `libge_compiler.so` 已确认 `ge.exec.precision_mode`、`cube_fp16in_fp32out`、`allow_mix_precision_fp16` 等接口。当前 runner 已将 `build.options.precision_mode` 传到 GE Session，因此不必先转换或复制 1.35 GB ONNX。

外部 `build.dtype: float32` 保持实际浮点输入接口，和内部计算策略是不同层。整数 ID/mask 输入保留 INT64，`stat_ratio` 和 `scores` 保留 FP32。不能因为输入输出仍为 FP32 就声称内部计算仍为 FP32。

| 隔离配置 | 策略 | 验证重点 |
|---|---|---|
| `candidates/cube-fp16in-fp32out.yaml` | `cube_fp16in_fp32out` | Cube 算子优先 FP16 输入、FP32 输出；向量算子优先 FP32，实际选择依实现支持；检查最终图 MatMul 输入/输出及敏感片段 |
| `candidates/auto-mixed-fp16.yaml` | `allow_mix_precision_fp16` | 按目标安装包黑白灰名单混精；可能使灰名单敏感算子跟随上游，不能预先声称 LN/Softmax 全部保留 FP32 |

两份配置均引用同一个原始 ONNX，不硬编码 device，并移除历史 `4|8` 限核以使用用户约定的全部核；正式测量参数由主控统一覆盖。准备脚本为 `candidates/prepare_precision_candidates.py`，不会生成 expected 或执行设备任务。

实际 `ascend910_93/*ops-info*.json`：MatMul/MatMulV2/BatchMatMul/BatchMatMulV2 的 `precision_reduce.flag=true`，ReduceMeanD/ReduceSum 为 false；LayerNorm/LayerNormV3/GeluV2/SoftmaxV2/Sqrt/Div 等没有这一项，属于灰名单；AddLayerNorm 为 true。因此自动混精不是“仅 MatMul 半精度”的保证。

`cube_fp16in_fp32out` 的规则以目标实现支持为条件；[官方精度选项说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/API/ascendgraphapi/atlasgeapi_07_0142.html) 提供语义参考，安装 9.0 的帮助和最终执行图负责确认本环境生效情况。若未来需要定制黑白名单，SDK 有 `ge.exec.modify_mixlist`，但当前 runner 未暴露；不能在 YAML 中写一个被忽略的选项。原始 `1e-12` epsilon 和 `-FLT_MAX` mask 是本模型的重点边界。

## 3. 融合候选：残差 Add + LayerNorm

使用 CANN 9.0 已安装公共接口 `register/register_custom_pass.h`：注册 `BgeResidualAddLayerNormFP32`，在 `kAfterBuiltinFusionPass` 阶段运行，以便复用已经生成的 `LayerNorm`。插件由单次进程的 `LD_PRELOAD` 加载，不更改 CANN 安装目录、runner 或原始 ONNX，也不依赖未知的 ONNX custom-domain parser。

产物：`candidates/add_layernorm_pass.cpp`、`candidates/libbge_add_layernorm_pass.so`、`candidates/test_add_layernorm_pass.cpp`、`candidates/test_add_layernorm_pass`。按 CANN 旧 C++ ABI (`_GLIBCXX_USE_CXX11_ABI=0`) 编译，链接目标安装的 GE/graph/register 库。

匹配限制：仅 BGE encoder 层中的 `Add → LayerNorm`；输入和输出 FP32、rank3、最后轴 1024，gamma/beta `[1024]` FP32；LN 的 norm/params axis 都必须等价于最后轴，epsilon 必须为原始 FP32 `1e-12`；Add 输出仅供 LN，LN 统计输出无消费者；不允许控制边。满足后创建已有 `AddLayerNorm(x1,x2,gamma,beta)`，保持加法输入顺序和 epsilon，将 y 消费者重新连接，再删除原两个节点。

源 LayerNorm 的第三个输出为 variance，目标第三个输出为 rstd；本实现只允许两种统计输出均无人使用，复制的是形状/类型描述，不将 variance 值误接成 rstd。目标可选 bias 不使用，`additional_output` 保持默认 false。

7 个 CPU GE 合成图契约测试通过：合法模式替换并检查输入顺序和 y 重连；错误 epsilon、非最后轴、共享 residual、统计输出被使用、控制边、混合输入 dtype 均拒绝。测试仅加载 op schema，不调用 GEInitialize、aclInit 或任何设备运行接口。共享库也已编译链接成功。

正式候选使用原始 FP32 配置和 `must_keep_origin_dtype`，独立于低精度候选；单次作业设置：

```sh
LD_PRELOAD=/export/home/ext.hejun23/workspace/bge/work/bge-opt-20261026/candidates/libbge_add_layernorm_pass.so
```

必须在日志看见 `[BGE_FUSION] AddLayerNorm replacements=N`，并在新的最终图确认替换算子和 descriptor。`N=0` 表示没有证明生效；编译、精度或性能失败均如实保留 OPT 记录。CPU 图结构测试不证明 GE 编译时该阶段一定能命中，也不证明 AddLayerNorm 在本次动态 shape 下能执行或更快。

## 4. 验收、图采集与回退

低精度和融合各为一个清晰候选；不在首轮叠加。通过五例完整 scores 和原 expected 的 Top10 ID（index+1）、顺序与分数门槛后，才比较包含 H2D+GE+D2H 的正式基线。禁止更新 expected 或放宽阈值，禁止从算子数量估算性能收益。

建议单独正确性任务采集 `DUMP_GE_GRAPH=2`、`DUMP_GRAPH_LEVEL=3`、独立 `DUMP_GRAPH_PATH`，随后正式测量关闭图 dump。实际本轮已在 `pid_${pid}_deviceid_${deviceid}` 子目录得到最终图；寻找文件时需递归。官方 [DUMP_GRAPH_LEVEL](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/maintenref/envvar/envref_07_0012.html) 和 [DUMP_GE_GRAPH](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/maintenref/envvar/envref_07_0011.html) 分别说明阶段筛选和去权重输出，接口名称也已在本机 9.0 库核实。官网 9.0 对应直达页面当前不可访问，本文没有把其他版本文档冒充本机执行证据。

磁盘预检仅剩约 15 GB，不复制多份权重、不启用含权重的全阶段 dump。回退低精度仅恢复原 precision_mode；回退融合仅删除该作业 LD_PRELOAD。模型和原始验收输入输出始终不覆盖。是否接纳由主控依据独立正确性、实际吞吐/p99 和预算内的最终复验决定，本报告不填写尚未运行的结果。

## 5. cube 策略同次最终图的独立复核

主控在逻辑设备10执行候选，SE 只读分析 `fp16-graph-dump/pid_2002863_deviceid_10/ge_proto_00000000_graph_1_Build.txt`，42 MB 的无权重最终图。结果与仅看配置的推断不同：该策略已经覆盖**全部199个矩阵节点**，包括 attention、query/candidate 投影、门控和最终打分，不能描述为“只降低 encoder 精度”。

| 最终 GE 类型 | 数量 | 实际输入→输出 descriptor |
|---|---:|---|
| MatMulV2 | 144 | FP16、FP16、FP32 bias → FP32 |
| BatchMatMulV2 | 55 | 前两输入 FP16 → FP32；其中一个还有 FP32 bias |
| LayerNorm | 49 | x/gamma/beta 和三个输出均 FP32，全部 epsilon 属性仍为 `1e-12` |
| SoftmaxV2 | 25 | FP32 → FP32 |
| LpNormV2 | 5 | FP32 → FP32 |
| RealDiv / Sqrt / Gelu | 31 / 48 / 24 | FP32 → FP32 |
| 浮点 Add / Mul / Maximum | 78 / 55 / 7 | 全部浮点输入和输出为 FP32 |

mask 实际路径为 INT64 → FP32 Cast、FP32 Sub、BOOL 条件、`SelectV2(BOOL, FP32, FP32) → FP32`；24层 attention mask Add 的两输入和输出均为 FP32。外部七个整数输入仍是 INT64，stat_ratio 和最终根图 `Node_Output` 仍为 FP32。

矩阵 kernel 属性已有直接佐证，例如 `MatMulV2_ND_ND_ND_FP16_FP32_FP32_false_false_all`，以及匹配的 `_kernel_bin_id`、`AIcoreEngine`。descriptor 和 kernel 名能证明半精度输入与单精度输出，**尚不能单独证明内部每一步累加的精度**。统计包含根图与子图，物理布局 descriptor 也不能一律当原 ONNX 逻辑 shape 使用。

最终图有196个 FP32→FP16 Cast 节点；原始图没有此类 Cast。包含常量分区，因此不等于每个请求196次 kernel 发射。这是潜在转换/调度成本的结构证据，是否抵消矩阵收益只由关闭 dump/profiling 后的正式性能测量判断，不能使用压力正确性任务的 QPS 作为收益。

独立证据在远端 `candidates/fp16-final-graph-analysis.json` 和 `fp16-focused-evidence.json`；可复现脚本为 `analyze_fp16_final_graph.py`。主控/测试已报告2000次原门槛验收通过：完整分数 max_abs≈2.42807e-4、relative_l2≈4.72072e-4，Top10分数最大误差≈1.33679e-4，ID和顺序一致。上述精度结论来自测试执行，本 SE 子任务没有重复运行设备。

此图的 residual Add 与 LayerNorm 均保持 FP32，因此已冻结 AddLayerNorm 插件的严格匹配条件与 cube 策略在 descriptor 层面兼容。先完成 FP32 融合单项实验；若有效，再为组合单独编号并重新做正确性、性能和最终压力验收，不能把两个单项收益相加。

## 6. 自动混精的下一候选边界

`allow_mix_precision_fp16` 配置已准备并通过框架配置校验，不复制模型、不改 runner、不硬编码 device。它与 cube 策略的差异是：灰名单可以沿半精度上游继续降低 Add、Gelu、LayerNorm、Softmax 或 L2 等片段的张量精度，有机会减少 Cast，但结果不能在运行前保证。

LayerNorm 的 epsilon 是浮点**属性**，不会仅因输入变为 FP16 就自动成为半精度零；应核实所选 kernel 的归约和 epsilon 相加实现。相对地，L2 floor 是浮点张量常量，若实际被 Cast 为 FP16，`1e-12` 确实会下溢为0；mask 的 `-FLT_MAX` 若转换会成为负无穷，须检查全屏蔽语义。Softmax和最终打分的误差也必须沿原门槛验收，最终输出 Cast 回 FP32不能恢复损失。

风险/检查清单为 `candidates/auto-mixed-fp16-risk.json`。本 SE 子任务没有启动自动混精的 GE 编译或执行；后续由主控分配 OPT 和预算，保存新最终图后复核真实类型，不用降低数值或业务门槛换取通过。

## 7. 残差融合已在最终图真实生效

主控 `fusion-correctness` 作业日志给出 `[BGE_FUSION] AddLayerNorm replacements=48`。SE 只读复核 `fusion-graph-dump/pid_2005676_deviceid_10/ge_proto_00000000_graph_1_Build.txt`：最终48个 `AddLayerNorm`，剩余1个 embedding `LayerNorm`；前四输入为 x1/x2/gamma/beta FP32，第五 optional bias 为空，四个输出 descriptor 均为 FP32，48处 epsilon 均保留 `1e-12`。

真实 kernel 名为 `AddLayerNorm_normal_input_all_fp32_high_performance`，具有 `_kernel_bin_id`、`AIcoreEngine` 和 `_cube_vector_core_type=AIV` 属性。这证明本次不是只有 Pass 日志或源图节点替换，最终计算图确实选择了已安装的融合 kernel。数值验收仍由测试负责，性能由关闭 dump 的独立任务负责。

图与日志证据为 `candidates/fusion-final-graph-analysis.json`、`fusion-final-graph-summary.json`。作业2000次请求设备 errors 为0，但编译日志另外出现11次 Python multiprocessing forkserver `TypeError: signal handler...`；两种错误不能混为一个统计口径。当前证据只支持候选能完成编译和运行，不支持把这些 Python 警告忽略为已修复。

一种待验证解释是 `LD_PRELOAD` 被编译 Python 子进程继承，在解释器初始化前加载 GE 库并安装非 Python 信号处理器。若后续处理，可在新的隔离插件版本初始化时仅移除其自身的 LD_PRELOAD 继承项，使主 runner 仍持有已加载的插件、Python 子进程不再重复预加载；必须保留其他已有预加载项并独立复验。本文不把这个解释写成已证实根因，也未修改已冻结共享库。

## 8. V1 生命周期失败证据与隔离 V2

`jobs/fusion-search/request.json` 的 argv 明确在 Python `run_runtime.py` 启动前设置了 V1 插件 LD_PRELOAD。`status.json` 记录 task_returncode=0，但 supervisor exit_code=125、reason=`background_children`，剩余 PID 为2006543和2006544，随后 process_group_stopped=true。保存的状态没有两个进程的 cmdline/start_ticks，因此不能把它们直接认定为 forkserver/resource_tracker。即使 native runtime 给出 QPS/p99，该轮生命周期验收失败，不能作为有效优化收益或接纳依据。完整证据见 `candidates/fusion-search-lifecycle-evidence.json`。

主控为下一次隔离尝试分配新的 OPT；新库 `candidates/libbge_add_layernorm_pass_v2.so` 不覆盖 V1，SHA256=`76ea9fb0f1800369b023e22d71298e508dcfaa6f69f34bcf952af4a1b5fa48ed`。V2 原样引用已经测试的融合实现，只增加进程级预加载作用域：仅 `/proc/self/exe` basename 精确为 `ge_runtime` 时，通过 dladdr 找到自身、realpath 进行路径身份匹配，从 LD_PRELOAD 中删除自身 token。冒号或空白分隔都支持，其余预加载项保留；非 ge_runtime、无法解析的路径和不含自身的原值保持不动。主 native 已加载的库不卸载，未来 compiler Python 子进程不再通过这个环境变量预加载该库。

同时，主控的 Python launcher 必须只在创建真实 C++ binary 的 Popen 环境中设置插件 LD_PRELOAD，自己和其他 Python 管理进程不预加载。这解决的是执行环境边界，不能由此预先保证生命周期问题全部解决。

V2 已通过7个原融合 CPU 图测试、11个环境过滤测试和3个实际 CPU 预加载集成检查：名为 ge_runtime 的纯 CPU probe 仍映射 V2 但继承环境已移除自身；它启动的 Python 子进程不映射 V2；其他名字的进程完整保留原预加载值。保留了实际安装的 tcmalloc 预加载项，`ldd -r` 无缺失依赖或未定义符号。测试未调用 GEInitialize、aclInit 或 NPU 执行。构建脚本 `build_add_layernorm_v2.py`、源代码和逐命令验证结果 `v2-cpu-validation.json` 均保存在 candidates。

本 V2 产物已冻结；设备阶段须重新证明48处融合、相同 epsilon/类型、原门槛精度、受管进程组完全退出和正式性能，不能沿用 V1 的有效性结论。

## 9. 自动混精同次最终图的实际边界

只读分析主控 `mixed-graph-dump/pid_2007161_deviceid_10/ge_proto_00000000_graph_1_Build.txt`，SHA256=`c7cd3f0745d9b735e34307ae2c58ecc228726057a9d2648902e4d19b95f9d246`。与 cube 策略相比，自动混精不只是调整 MatMul 输入：

| 区域 | 自动混精最终图 |
|---|---|
| 144 MatMulV2 | x、weight、bias 和输出全部 FP16 |
| 55 BatchMatMulV2 | 前两输入和输出 FP16；有 bias 的一个节点 bias 也为 FP16 |
| 24 Gelu | FP16 输入和输出 |
| 49 LayerNorm | x/gamma/beta 与三个输出仍全 FP32；epsilon 属性仍为 `1e-12` |
| 24 attention Softmax | 仍为 FP32 |
| 门控 `/Softmax` | FP16 输入和输出 |
| candidate `/ReduceL2 → /Clip → /Div` | 三个节点均进入 FP16 张量边界 |
| query `/ReduceL2_1 → /Clip_1 → /Div_1` | 三个节点均进入 FP16 张量边界 |
| 其余3组 L2 | 保持 FP32 |
| mask | INT64→FP32，BOOL 条件下的 SelectV2 浮点输入/输出仍FP32；attention Add为(FP16,FP32)→FP32 |
| 外部 IO | 七个整数输入仍INT64，stat_ratio和最终输出仍FP32 |

实际 FP32→FP16 Cast 从 cube 的196个减少到122个，同时新增1个 FP16→FP32 Cast，名为 `trans_Cast_520`，用于最终半精度结果恢复外部接口类型。这只减少图节点数量，不能直接换算性能收益；最终分数转回FP32也不能恢复内部半精度输出的舍入损失。

LayerNorm 仍为 FP32 是这次真实编译结果，不能推导为自动混精的一般保证。相比之下，candidate 和 query 的 L2 floor 已进入真实 FP16 descriptor：原图的 floor 张量为 `1e-12`，普通FP16表示会下溢为0。这份去权重图没有最终常量payload，不能直接声称已经读到了0；它明确证明了需要验证的实际边界，零/近零范数是否保持原语义仍应单独检查。LayerNorm epsilon 是浮点属性并保持FP32，和上述 floor 张量需要区别对待。mask 路径未降成FP16，因此这里没有仅由目标dtype造成的 `-FLT_MAX→-Inf` 转换证据。

kernel 属性也显示了实际实现路径：LayerNorm仍为 `LayerNorm_23ae115d70bd435e4cb966a52be97b55_high_performance`，attention Softmax仍为 `SoftmaxV2_291664ae3e88b38b8b46c2efbbea27c9_high_performance`。candidate L2切换为新的 `te_lpnormv2_4506dee...` kernel。所查 encoder MatMul 节点没有 `_kernelname/_kernel_bin_id`，而是 `_ge_attr_lowering_func=AclnnLoweringFunc`、`AIcoreEngine` 和实际FP16 descriptor；不能为这些节点编造静态kernel名，具体运行kernel需profiling确认。内部归约/累加精度也不由图descriptor单独保证。

证据为 `candidates/mixed-final-graph-analysis.json`、`mixed-final-graph-summary.json`、`mixed-sensitive-boundaries.json`，分析没有启动NPU。自动混精和融合的组合需要另外核实：自动混精单项的残差Add有混合dtype，融合后AddLayerNorm又在目标降精度白名单，不能简单沿用 cube+fusion 的FP32兼容结论。数值和性能结论由主控/测试的原门槛验收决定。

## 10. 多实例限核候选的静态依据

实际安装的 `aarch64-linux/data/platform_config/Ascend910_9382.ini` 声明 `ai_core_cnt=24`、`cube_core_cnt=24`、`vector_core_cnt=48`，因此目标静态全核为 `24|48`。安装 header 确认参数键 `ge.aicoreNum`；针对该目标，正整数编译上限为 AIC 1–24、AIV 1–48，同时必须不超过当前设备实际可见核数。[官方参数语义](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910/devaids/atctool/atlasatcparam_16_0038.html)与安装值相符，但文档为9.1；9.0精确目标值以本机文件为依据。该INI不证明device10当前没有虚拟化或屏蔽核，实际可见上限本子任务未调用设备API测量。

建议先试 **`12|24` 半核**，保持对应全核候选的实例数、FP32策略、工作负载与计时范围，只改变该 pair；有预算和实测依据再考虑 `6|12`。这是每算子的编译核数上限，不为每个实例硬划分独占核，也不能将“实例数×pair”当实际隔离资源。INI 的 `vir_type_list` 描述虚拟化类型，不是合法限核值枚举。生效程度取决于 kernel/tiling，收益由原精度门槛、受管退出和独立 QPS/p99 测量共同决定。静态证据和文件校验值已保存 `candidates/core-budget-evidence.json`；本次未初始化NPU或采集profiling。

## 11. 剩余最多四配置的搜索建议

本节是 SE 的有条件搜索建议，不启动设备、不分配 OPT。当前主控给出已启动8个配置、设备时间约1200/7200秒。已观测单实例原始FP32基线median175.32 QPS，cube157.29、自动混精152.63、融合V2为171.78；双实例269.90、四实例370.21，八实例压力已运行而纯性能仍待独立结论。压力任务的QPS不用于这组比较，V1生命周期失败的性能也不纳入。以上短搜索和三轮基线的统计强度不同，不能直接视为正式收益。

| 优先顺序 | 新配置 | 进入条件和单一改变量 |
|---|---|---|
| 1 | 原始FP32、16实例、全核 | 八实例正式搜索仍明显优于四实例、内存/Host负载/设备门禁允许时继续翻倍；仅改变实例数 |
| 2 | 最佳已验证实例数N、`12|24`、原始FP32 | 冻结N，只改变核数，与同N全核比较 |
| 3 | 同N、`6|12`、原始FP32 | 仅当半核已有可复现正向趋势时继续；否则跳过，不为用满预算而测试 |
| 4 | 最佳N及已验证核数、融合V2、FP32 | 固定N和core pair，只增加融合；必须重新证明48处命中、原门槛精度和完全退出 |

优先给融合V2一次多实例重试：其单实例结果仅比基线median低约2%，落在原三轮约11.45%的轮间波动范围内；减少Add/LN边界可能在多实例时改变Host调度/访存成本，尚无profiling证明。相对地，两种降精度单实例均下降约10%–13%，并已有122/196个Cast节点及自动混精half L2的实证边界。当前不优先再分配精度组合；只有出现明确的内存容量或带宽证据、已有数值验收仍有效时，才用未消耗的条件槽改测更保守的cube策略，和同N/core的FP32做独立A/B。不会把源码算子减少数当实际kernel发射减少数。

停止规则：实例翻倍后若吞吐持平/下降，停止继续增加实例；`12|24`若无可复现改善，不继续四分核。搜索差异很小、在已有波动范围内时，先复测已有配置，而非再开新组合。可以用约5%作为进入下一轮的粗筛提示，但它不是新增业务验收门槛，也不是统计显著性的替代品；最终以相邻时间段A/B复测及独立三轮的稳定结果决定，小幅而可复现的收益仍可考虑。错误、输出不符、受管清理失败、外部设备占用或统计不足均不能作为有效候选；设备变忙则按既定门禁暂停。最多四个新配置不是必须耗满，重复同一配置不占新配置数但占设备时间，始终保留工作流要求的最终验收时间及余量。

公平性限制：同一device10/同一芯片、相同冻结binary和原始模型/CPU参考、GE mode=0、B=1和五例等概率回放、相同H2D+描述符+GE完成+完整D2H边界，正式性能关闭capture/dump/profiling。core实验固定实例数，融合/精度实验固定实例数和core；只比较正确性与清理均通过的无采集性能任务，记录Host负载、HBM、预热、轮次和随机序列。编译/加载不计请求时延但全部计入设备预算；多实例本身的内存和尾延迟代价必须同时报告。当前没有p99硬上限，不能擅自以某个延迟数字判SLA通过或失败，也不能隐藏高并发导致的p99上升。

16实例的覆盖量需特别注意：2000请求仅平均每实例/每case25次，随机回放容易低于设计每格至少20次。建议先用4000请求并检查实际分布，不足则延长同配置验证，不修改expected和阈值。最终冻结最佳配置后，以独立序列完成完整输出/Top10、每实例覆盖、三轮性能、稳定性和三次加载销毁检查；搜索峰值不直接作为最终交付QPS。结论只适用于这五例短序列闭环负载，不外推为实际业务分布或硬件理论极限。
