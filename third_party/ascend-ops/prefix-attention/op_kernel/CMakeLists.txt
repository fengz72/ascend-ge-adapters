# set custom compile options
if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    add_ops_compile_options(ALL OPTIONS -g -O0)
endif()

# 与源仓ops-transformer的PrefixInferAttentionScore编译选项一致(语义性flag, 缺失将导致
# auto-sync插入/membar消除等行为差异, 产生跨核同步问题与数值错误)
add_ops_compile_options(ALL OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -mllvm -cce-vf-remove-membar=false
        -mllvm -cce-aicore-hoist-movemask=false
        --op_relocatable_kernel_binary=true
)

add_kernels_compile()
