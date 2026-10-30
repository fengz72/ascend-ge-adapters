# 离线调测工具

parse_profiling.py、parse_dump.py 和 compare.py 只分析实际采集数据，不负责模型运行。运行、数据选择、导出和采集统一使用 core.pipeline 与模型 YAML，见项目 README。工具参数可通过各脚本 --help 查看。

BGE 可选 GE pass 的源码、host 测试与构建入口见 bge_fias_pass/、bge_fias_all_pass/、bge_candidate_precompute/ 和 build_bge_experimental_passes.sh；状态和性能见 models/bge/README.md 的实验附录。
