# 核心契约

pipeline.main → pipeline.execute → pipeline.perform → 数据/依赖/单阶段 worker；run 只按 data、baseline、prepare、build、infer、compare、bench 顺序调用 perform。单步不补跑前置阶段。没有独立 check 或强制 eager 精度门禁。原始模型与应用 HF patch 的候选模型运行在不同进程。

Python 脚本和 Skills 直接调用 execute(stage, config, *, device=None, target=None, targets=None, case_ids=None, instances=None, profiling=None)；main(argv=None) 将命令行参数交给同一 execute。config 是 YAML 路径，execute 不接收预先解析的配置字典。report 命令仅读取已有结果。没有独立 cli 模块或兼容转发入口。

配置只读当前 YAML，不生成额外的 resolved、lock 或配置快照文件；实际运行结果记录本次配置以解释测量。execute 接收单个 YAML 路径，在内存中覆盖实例数、精度 case、profiling。source 路径环境变量只在读取原始模型的阶段展开；纯 GE 运行与比较不依赖原始权重路径。自动硬件识别用于选择平台配置，不形成一个检查步骤。

数据脚本实现 generate(config) 和 load(config)，分别负责持久化输入与读取已有 case。case 包含 id、payload、samples，payload 由模型定义。Qwen 用共享 prefix 的三个 int64 NPY 输入保存一组 case，读取时恢复相同业务请求。没有独立脚本的示例提供 generate_cases(params)、input_arrays(case, config)、case_from_inputs(id, arrays, config)，框架保存并读取各输入 NPY。数据池按 case ID 的自然顺序读取，不保存清单。张量名是非空且不含 NUL 的逻辑字符串，允许冒号、斜杠等 ONNX 原始名称。通用 NPY 存储对文件名做可逆 URL 转义，读取时恢复原名；case ID 仍使用安全目录名。

模型适配模块提供 prepare_reference(source, config)、prepare_candidate(source, config)、prepare_runtime(config, io)。reference 负责 encode/run/decode；candidate 负责模型适配和 export_spec；runtime recipe 只负责 encode/decode/output_capacities，不加载权重、不 patch HF、不创建 NPU 模型。模型文件不导入 core，运行配方和适配修改同文件交付。动态输出的容量规则属于模型。

导出器返回具名 IO，AIR 将其写入图属性 _ge_adapters_io，ONNX 将其写入 metadata。图与 IO 一起交付，没有 IO sidecar。infer/bench 从已有图读取 IO，调用模型的无权重运行配方编码已有 case，直接启动原生运行时；不会导出图或安装依赖。内部执行参数通过 stdin 传递，单个打包张量文件在结束后删除。

baseline 保存所选 case 的 golden.npz。infer 单实例保存所选 case 的 outputs.npz。两者都包含按 case 分组的业务输出与对齐信息，覆盖前次数据。compare 仅读取这两个文件和 YAML 比较策略，要求非空策略覆盖全部实际业务输出，拒绝输出缺失、多余或未配置；完成所有选定 case 的数值比较后汇总失败，不依赖 NPU、输入或模型代码。没有精度证据锁；bench 不读取任何精度文件。

bench 从完整输入池按 seed 有放回抽样，使用 YAML 或 execute 参数中的实例数、预热和总调用次数。infer 与 bench 都支持 profiling；带采集开销的性能记录明确标识。GE Session 加载已有 AIR 时仍会在线编译，ATC 运行已有 OM 不调用 ATC。

每个阶段记录自身结果和日志，错误将原始错误文本显示到终端，详细堆栈保留在日志。依赖安装和图文件缺失直接失败，不自动修复。配置改动不静默触发前置步骤：数据或模型变更由用户显式 data/build，依赖源码变更显式 prepare。

原生运行时统一使用 RequestPool，支持静态和动态 shape，不再保留 Bundle/Manifest 或固定输入执行分支。模型适配器只要求三个实际使用的 prepare_* 入口，无 describe 前置契约。性能执行失败时原生程序以非零状态返回，并保留已产生的逐调用记录与部分统计；pipeline 将原始错误和部分结果写入本次 result.json。初始化失败没有实际调用时不写性能字段。
