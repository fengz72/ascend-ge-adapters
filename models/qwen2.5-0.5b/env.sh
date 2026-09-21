#!/bin/bash
# 运行环境 (CANN + PIA 算子 + numpy) — 由 run.sh 自动 source, 也可手动 source。

# 1. CANN 基础环境
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 2. PIA 算子 (npu_prefix_infer_attention_score) — prefix 链路必须, 见 §1.1 安装
source $ASCEND_HOME_PATH/opp/vendors/custom_prefix_attn/bin/set_env.bash

# 3. GE 编译期 tbe pywrapper 内嵌 /usr/bin/python3 无 numpy,
#    注入当前 python3 的 site-packages (GEInitialize 返回 -1 时先检查此项)
export PYTHONPATH=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"):$PYTHONPATH
