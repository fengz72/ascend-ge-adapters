#!/bin/bash
# 运行环境 (CANN + PIA 算子 + numpy) — 由 run.sh 自动 source, 也可手动 source。

# 1. CANN 基础环境
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 2. PIA 算子 (npu_prefix_infer_attention_score) — prefix 链路必须
#    未安装时不报错: 管线的 custom_ops 阶段会执行 scripts/install_prefix_attn.sh 构建安装,
#    并通过 $GE_ENV_FILE 回传下面的 env (见 core/setup_scripts.py)
PIA_ENV="$ASCEND_HOME_PATH/opp/vendors/custom_prefix_attn/bin/set_env.bash"
if [ -f "$PIA_ENV" ]; then
    source "$PIA_ENV"
else
    echo "[env.sh][WARN] PIA 算子未安装 ($PIA_ENV 不存在) — prefix 链路会由 custom_ops 脚本安装"
fi

# 3. GE 编译期 tbe pywrapper 内嵌 /usr/bin/python3 无 numpy,
#    注入当前 python3 的 site-packages (GEInitialize 返回 -1 时先检查此项)
export PYTHONPATH=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"):$PYTHONPATH
