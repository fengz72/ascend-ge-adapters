# 100 BGE：仅 attention/output 融合 V3 开发交接

2026-10-26，开发完成 **CPU 构建与测试**。没有 GEInitialize、aclInit、模型执行或 NPU job；设备验证和是否占用剩余候选预算由主控决定。

V3 在 `/export/home/ext.hejun23/workspace/bge-opt-100/candidates-v3/` 隔离部署，V2、原始图、expected 和 runner 二进制保持不变，部署前后 SHA 逐项一致。

|交付项|值|
|---|---|
|共享库|`/export/home/ext.hejun23/workspace/bge-opt-100/candidates-v3/libbge_add_layernorm_attention_v3.so`|
|共享库 SHA256|`5472882c2c401e7b99f5dde12470742519397a204650a4e76803df802249e299`|
|大小|91088 bytes|
|pass 源 SHA256|`0f546bc19d58f58af5bf603efa337cd819f6cf12a865706f9295fb2f93700a64`|
|native preload wrapper 源 SHA256|`1508f75b5576c6718ceadd58b1c435c7798b7c29cfc396d9f45cfd21544b118d`|
|scope header SHA256|`91649f5e8377b51b8681e0d280ae4e56d05682b7050bd560520ef8c0c2a2684b`|
|注册名|`BgeAttentionOutputAddLayerNormFP32V3`|
|构建入口|`candidates-v3/build_attention_v3.py --cann /usr/local/Ascend/cann-9.0.0 --directory /export/home/ext.hejun23/workspace/bge-opt-100/candidates-v3`|

候选只匹配 layer.0–23 的 `/attention/output/LayerNorm/`，预期 24 处；FFN output、embedding、其它模型、非规范或越界 layer 名均排除。数学与安全约束、epsilon=FP32 1e-12、原 Add 输入次序、统计无消费者、单消费者/无控制边、原始 FP32 dtype/shape，以及 V2 native-only preload 生命周期不变。实际设备命中 24 仍需后续日志/执行证据。

CPU 验证通过：

- 本地 C++17 scope 对实际 profile 的 48 对节点选择恰好 24 个 attention，并排除 24 个 FFN；5 项本地回归通过。
- 100 CANN9 真实 GE CPU 图契约 **13 项**：合法首/末层正例、epsilon 错误、非末轴、残差共享、统计被使用、控制边、混合 dtype、FFN、其它模型、layer24、embedding、layer 前导零。
- preload filter **11 项**；native 映射、native 子 Python 不映射、其它程序保留 preload **3 项集成**；`ldd -r` 无未解析符号/缺库。
- 100 NumPy1.26.4 的 FP32 残差加法/LN 语义参考，在 S6/S8/S13、普通/低方差/较大输入三组对 FP64 参考最大误差分别 `5.297e-7 / 2.191e-7 / 5.363e-7`；错误 epsilon 可被区分。该检查不执行 CANN AddLayerNorm 或完整 BGE，不能替代设备精度。

100 原始构建/执行日志与每条 CPU 命令退出码：`candidates-v3/v3-cpu-validation.json`；部署/源 SHA/保护文件前后 SHA：`candidates-v3/deployment-cpu-result.json`；数值结果：`candidates-v3/numeric-semantics-result.json`。本地完整拷贝：[remote-build](../../../delivery/supporting-evidence/work/bge-opt-100/fusion-v3/remote-build/)。源码与 diff：[fusion-v3](../../../delivery/supporting-evidence/work/bge-opt-100/fusion-v3/README.md)。

后续必须先更新冻结 identity 和独立配置登记，仅给 native runtime 加载该 so（保留 tcmalloc，不同时加载 V2），再验证命中/全五例 full-score、Top10、请求输出关联压力，最后用无 profiling/capture 的同场景对照判断收益。当前没有 V3 NPU 精度或性能结论。

CPU 窗口已在部署与证据下载结束后释放；后续报告整理全部在本地完成。
