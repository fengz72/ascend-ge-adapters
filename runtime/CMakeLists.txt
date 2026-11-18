cmake_minimum_required(VERSION 3.14)
project(ge_execute LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ASCEND_HOME 解析顺序: 缓存变量 -> $ASCEND_HOME_PATH -> $ASCEND_HOME -> 默认安装路径。
if(NOT DEFINED ASCEND_HOME)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_HOME $ENV{ASCEND_HOME_PATH})
    elseif(DEFINED ENV{ASCEND_HOME})
        set(ASCEND_HOME $ENV{ASCEND_HOME})
    else()
        set(ASCEND_HOME /usr/local/Ascend/ascend-toolkit/latest)
    endif()
endif()
set(ASCEND_HOME "${ASCEND_HOME}" CACHE PATH "Ascend CANN install root")

message(STATUS "ASCEND_HOME: ${ASCEND_HOME}")

set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party")

# 单图执行器: CLI + GE Session + npy/cases/stats 的 CPU 侧实现。
add_executable(ge_execute
    main.cpp
    session.cpp
    npy.cpp
    cases.cpp
    stats.cpp
)

target_include_directories(ge_execute PRIVATE
    ${ASCEND_HOME}/include
    ${THIRD_PARTY_DIR}
)

target_link_directories(ge_execute PRIVATE ${ASCEND_HOME}/lib64)

# GE 头/库以旧 ABI 构建 (同 atb/CMakeLists.txt ge_infer): 去掉这条会因
# std::string/std::map 的符号名不一致在链接期报 undefined reference, 不能删。
target_compile_options(ge_execute PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)

target_link_libraries(ge_execute PRIVATE
    ge_compiler
    ge_runner
    ge_common
    ge_common_base
    graph
    graph_base
    fmk_onnx_parser
    parser_common
    ascendcl
    stdc++
    pthread
)
# 注: 不链 acl_op_compiler —— 本运行时不调任何算子编译/aclnn API (图编译走
# ge::Session::CompileGraph)。曾经链着它但无一处引用, 白增一条对 CANN 组件的依赖。

# CPU 契约测试: 只覆盖 npy/cases/stats, 不碰 GE/ACL, 因此不链任何 CANN 库,
# 也不需要旧 ABI 开关 (它和 GE 头没有共同类型)。
include(CTest)
if(BUILD_TESTING)
    add_executable(runtime_contracts
        ../tests/runtime_contracts.cpp
        npy.cpp
        cases.cpp
        stats.cpp
    )
    target_include_directories(runtime_contracts PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${THIRD_PARTY_DIR}
    )
    target_link_libraries(runtime_contracts PRIVATE pthread)
    add_test(NAME runtime_contracts COMMAND runtime_contracts ${CMAKE_CURRENT_BINARY_DIR}/test-data)
endif()
