# 七步执行方式

命令入口统一为 python3 -m core.pipeline。core/cli.py 已删除；现有命令、脚本和 Skills 中的旧模块入口需要直接替换，不提供兼容入口。Python 调用改为 from core.pipeline import execute，并传入模型 YAML 路径，例如 execute("bench", "models/qwen2.5-0.5b/config/model.yaml", device=6, instances=4)；不再先 load_authoring 后传字典。平台选择与参数覆盖全部由 execute 完成。

唯一模型 YAML 保留 source、model.params、data、build、verification、benchmark、dependencies、observability、platforms/chips。只支持容器 Python，不生成清单或兼容旧命令。

| 原操作 | 现在的行为 |
|---|---|
| data | 生成/保存实际 NPY 输入，其他步骤只 load |
| baseline | 读取已有输入，仅对所选 case 生成 golden |
| prepare | 只构建安装所选依赖 |
| build | 只适配并导出，图内携带 IO；不生成 golden 或运行 eager 精度 |
| infer | 新的独立 GE 推理操作，读取已有图和输入，保存 outputs.npz |
| compare | 新的独立精度比较，仅读取 golden/output 和 YAML 阈值 |
| bench | 读取已有图和完整数据池，支持 --instances，绝不补跑前置步骤 |
| run | 按上述顺序调用同一套七步函数 |

check、resolve、verify、profile 命令删除。profiling 改为 infer/bench/run 的执行选项。原 verify 的原始模型、GE 运行、精度比较拆为 baseline/infer/compare。缺少文件或依赖直接报错，不根据安装目录或图的存在与否重建。

旧图没有内嵌 IO，需要显式执行一次 build；不从旧 result.json 或已删除的 build/lock 文件恢复元信息。AIR 内嵌 IO 图属性，ONNX 使用标准 metadata，ATC 路径保留导出图作为 IO 来源。精度默认单实例，baseline/infer/compare 共用 case 选择；性能始终完整池。

日常精度使用默认固定一组，最终验收各步传 --case-ids all 或 run --case-ids all。原始模型、输入或业务输出定义改变后重新生成 golden；模型、适配代码、编译选项改变后按实际后端需要显式 build；依赖修改显式 prepare。

不迁移运行清单或身份锁。实际输入、golden、output、图、依赖安装目录和结果分别保留在既定位置，用户可以明确复用或重建。

精度策略现在必须覆盖 NPZ 中全部真实业务输出；空策略或少配一个输出会直接报错。ONNX 原始张量名不再受文件路径命名规则限制，通用输入 NPY 使用可逆 URL 文件名转义。旧 Bundle/Manifest 与固定输入原生接口已删除；静态输入也通过 RequestPool 执行。

Qwen 默认输入目录固定为 models/qwen2.5-0.5b/io/inputs。batch16 配置通过 data 显式生成；切换前保留既有历史输入和对应 golden/output，防止将 batch10 精度记录误认为 batch16 的验证。源码与配置不引用某一次实验目录。
