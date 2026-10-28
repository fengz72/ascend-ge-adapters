# BGE 100 验收工具

从旧 acceptance-v2 最小复制适配；不修改原工具、不访问 NPU。默认项目根 `/export/home/ext.hejun23/workspace/bge-opt-100`。

材料布局：

```text
materials/
  cpu-golden.npz
  cpu-golden-manifest.json
  expected_outputs.json
  original-inputs/inputs.bin
tools/identity.json
work/<run-name>/{result.json,events.jsonl,...}
work/jobs/<run-name>/{request.json,status.json,...}
```

所有模式可用 `--reference-dir` 指定 CPU manifest 所在目录；`--original-inputs` 接受 inputs.bin 或其目录。数值验收还可用 `--golden` / `--expected`。对应环境变量为 `BGE_REFERENCE_DIR`、`BGE_ORIGINAL_INPUTS`、`BGE_GOLDEN`、`BGE_EXPECTED_OUTPUTS`。路径变化不放宽 SHA：manifest 自身、original inputs、golden、expected 仍必须对应冻结身份。

```bash
python3 tools/checks/verify_outputs.py selftest --report work/checks-selftest.json
python3 tools/checks/verify_plan_inputs.py --plan work/base-fixed-r1/plan.json --report work/base-fixed-r1-input-identity.json
python3 tools/checks/verify_outputs.py npz --outputs work/five-case-outputs.npz --report work/five-case-accuracy.json
python3 tools/checks/verify_outputs.py pressure \
  --manifest work/base-pressure/manifest.jsonl --binary work/base-pressure/outputs.bin \
  --events work/base-pressure/events.jsonl --instances 1 --report work/base-pressure-accuracy.json
python3 tools/checks/validate_performance.py \
  --performance work/base-fixed-r1/result.json --events work/base-fixed-r1/events.jsonl \
  --plan work/base-fixed-r1/plan.json --instances 1 --report work/base-fixed-r1-performance.json
python3 tools/checks/validate_performance.py --search --min-seconds 30 --min-requests 2000 \
  --performance work/candidate-search/result.json --events work/candidate-search/events.jsonl \
  --plan work/candidate-search/plan.json --instances 8 --report work/candidate-search-performance.json
python3 tools/checks/validate_run_identity.py --runs base-fixed-r1 \
  --identity-manifest tools/identity.json --report work/base-fixed-r1-run-identity.json
```

数值：五例完整 FP32 scores；cosine>=0.9999、relative L2<=0.001、max abs<=0.001；Top10 index+1、ID 顺序完全一致、score atol0.001/rtol0。压力 CLI 不允许低于 2000 请求或任一 instance×case 20 次。固定 case 性能仍不能代替全五例压力。

性能：从事件独立重算 QPS、nearest-rank p50/p99、闭环请求与窗口完成数；校验真实固定单例/随机五例输入字节、完整 H2D/D2H 和预热。默认单轮正式门槛 60 秒/10000 完成/每 case 10 预热，CLI 不可降低；搜索显式 `--search` 才允许 30 秒/2000 完成，报告 `acceptance_level=search`。拒绝 capture、profiling、dump 计时作为性能结果。单轮 `formal_round` 不等于三轮及最终稳定性完成。

identity.json 由主控在 100 实际构建冻结后生成，示例格式如下（SHA 必须来自实际文件）：

```json
{
  "schema": "bge100-runtime-identity/1",
  "project_root": "/export/home/ext.hejun23/workspace/bge-opt-100",
  "logical_device": 14,
  "physical_devices": ["100/0000:81:00.0", "100/0000:83:00.0"],
  "binary": {"path": "/export/home/ext.hejun23/workspace/bge-opt-100/runner/runtime/build/ge_runtime", "sha256": "ACTUAL_SHA256"},
  "graph": {"path": "/export/home/ext.hejun23/workspace/bge-opt-100/materials/exported/model.onnx", "sha256": "ACTUAL_SHA256"},
  "weights": {"path": "/export/home/ext.hejun23/workspace/bge-opt-100/materials/exported/weights.data", "sha256": "ACTUAL_SHA256"},
  "plugins": {"/export/home/ext.hejun23/workspace/bge-opt-100/candidates/libbge_add_layernorm_pass_v2.so": "ACTUAL_SHA256"},
  "required_preloads": ["/usr/lib64/libtcmalloc.so.4.5.16"]
}
```

`weights` 可选，但建议提供。`plugins` 为允许的候选集合，不要求基线加载。使用插件时须同时包含 required_preloads，且插件只能进入 native runtime 的 plan.runtime_env；将其放在 Python launcher 外层会失败。identity 工具同时绑定真实 logical device、物理锁集合、结果目录、原始图/二进制/可选权重、输入池、runtime sampling、请求 ID 和插件 SHA，不沿用 99 的二进制/PCI 身份。`--run-root` / `--job-root` 可改变布局，不改变物理锁身份要求。

CPU 回归：运行时通过环境变量指定本地归档材料，然后 `python3 -m unittest discover -s tools/checks -p 'test_*.py' -v`。测试中的合成事件和二进制仅验证检查器，不作为设备或性能证据。
