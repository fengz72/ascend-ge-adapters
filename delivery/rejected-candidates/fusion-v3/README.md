# Attention/output-only AddLayerNorm V3（尚未部署）

依据 100 的同 case profiling：24 个 attention/output Add+LN 片段均值 8.875→4.318 μs；24 个 FFN output 则 4.590→4.895 μs。V3 **只**匹配 layer.0–23 的 `/attention/output/LayerNorm/`，保留其余 FFN/output、embedding 和未知命名节点。

V2 全部源文件与已构建 so 保持不变。V3 使用独立 `BgeAttentionOutputAddLayerNormFP32V3` 注册名与独立共享库。唯一数学相关实现变更为候选范围缩小；原 epsilon=FP32 1e-12、末轴、hidden1024、所有输入FP32、统计无消费者、残差仅一个消费者、无控制边、输入顺序、输出接线及 native-only preload 生命周期原样保留。

本目录为本地隔离准备材料，没有上传到 100，没有远端编译，没有新的设备测试。主控决定纳入配置预算并排 CPU 构建窗口后，在 100 独立候选目录执行：

```bash
python3 /absolute/candidates/fusion-v3/build_attention_v3.py \
  --cann /usr/local/Ascend/cann-9.0.0 \
  --directory /absolute/candidates/fusion-v3
```

输出 `libbge_add_layernorm_attention_v3.so`，先执行 **13 个真实 GE CPU 图契约**、11 个 preload filter 和 3 个 CPU 环境集成检查，检查 `ldd -r`；这些步骤尚未在 CANN 环境执行，不能把本地 scope/numpy 检查冒充 GE 图重写通过。编译脚本只做 CPU 构建与检查，不初始化 NPU。

主控安排设备验证时，从新的候选快照通过 replay CLI `--preload /absolute/candidates/fusion-v3/libbge_add_layernorm_attention_v3.so` 选择它，避免同时加载 V2。保留 inherited tcmalloc，只给 native runtime 注入。精度、输入、expected 和吞吐计时不变。新增 so SHA 需进入冻结 identity 和独立配置登记；不能继续使用 V2 so 的 SHA 或既有 PASS。

预计命中 24，设备上仍需日志/最终执行证据确认；然后全五例/每实例关联压力与 Top10/full-score 校验，无采集的固定 case、随机五例正式性能和稳定性。未通过这些门禁前，不接纳 V3，也不声称其必然提高端到端 QPS。

本地验证：

```bash
python3 -m unittest discover -s work/bge-opt-100/fusion-v3 -p test_local_v3.py -v
```

5 项通过：实际 48 对 scope 选择恰好 24、非法范围/FFN 排除、Fuse 数学/安全约束及 preload 源码相同、FP32 参考数学对 FP64 检查、低方差可检测 epsilon 错误。scope 谓词由本机 C++17 编译后实际运行；数值测试是 NumPy 语义检查，**不执行 CANN AddLayerNorm 或完整 BGE 模型**。源 SHA、数值误差和未验证项见 `local-validation.json`，修改 diff 见 `v2-to-v3.patch`。
