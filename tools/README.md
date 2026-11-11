# 离线调测工具

parse_profiling.py、parse_dump.py 和 compare.py 只分析实际采集数据，不负责模型运行。运行、数据选择、导出和采集统一使用 core.pipeline 与模型 YAML，见项目 README。工具参数可通过各脚本 --help 查看。

restore_matmulv3.py 修复部分 msprof 版本导出时丢失 MatMulV3 等 CubeCore 算子的问题（task_time 被标为 UNKNOWN(-1) 导致 op_summary 缺行、msprof db 断链）：对 `msprof --export=on` 之后的 PROF_XXX 目录重建 op_summary/op_statistic（`*_restored.csv`），原地修复 msprof_*.db 与设备侧 sqlite（自动留 `.bak` 备份），校验报告写入 `restore_report.txt`。用法：`python3 tools/restore_matmulv3.py --profiling_dir PROF_XXX`。
