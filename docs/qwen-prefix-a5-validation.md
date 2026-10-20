> 本文保留早期算子定位记录；当前七步命令、采集接口与验证状态以项目 README 和 docs/validation.md 为准。

# A5 Prefix 算子说明

Qwen 的 Prefix 路径由 model.yaml 的 platforms.a5 选择，适配改动集中在 qwen_npu_adapter.py。模型、算子源码和构建脚本直接交付，不经过补丁包、交付清单或构建锁。

桥接修复位于 third_party/ascend-ops/prefix-attention/cannbot-dsl/ge/prefix_attn_torch/prefix_attn_bridge.cpp，已提交为 b4b30cd。getCurrentNPUStream().stream(false) 改为 stream()，先提交 TorchNPU 主机队列中的 Q/K/V 工作，再提交 ACL 自定义算子，没有增加逐算子设备同步。

当前 prepare 在项目 work/qwen2.5-0.5b/dependencies/prefix_attention_a5 下构建并安装依赖。bench 使用同一模型 YAML，读取已有的 100 组 NPY case（当前默认 batch16），按 seed 抽样，无 lock/cases/build 清单，也不要求精度证据。

算子支持变长 T、prefix/bounds 和实际 batch；不再保留固定 shape/bounds 或重复 Prefix batch 配置。当前候选为 FP16、14/2 Q/KV heads、head dim=64。历史 batch10 以原 FP16 适配和 force_fp16 配置，按模型 YAML 的 cosine >= 0.998、relative L2 <= 0.05 阈值通过全部 100 组、1000 个业务样本。当前默认 batch16 的单组回归另记于 validation.md，不能据此推断 batch16 全量精度通过。历史 CPU FP32 golden 已单独保留；同一输出在旧阈值下的失败记录保留在 validation.md。没有增加 FP32 残差/RMSNorm 或修改算子源码。

新入口的 A5 Qwen 性能、Torch/ONNX 精度与 GE profiling 验证均记在 validation.md；ATC 本轮未执行，A3 仍缺少环境。历史 ATC 双实例 profiling 超时仍未解决。
