cmake_minimum_required(VERSION 3.14)
project(ge_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ASCEND_HOME)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_HOME $ENV{ASCEND_HOME_PATH})
    elseif(DEFINED ENV{ASCEND_HOME})
        set(ASCEND_HOME $ENV{ASCEND_HOME})
    else()
        set(ASCEND_HOME /usr/local/Ascend/ascend-toolkit/latest)
    endif()
endif()

message(STATUS "ASCEND_HOME: ${ASCEND_HOME}")

set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party")

include(CTest)
if(BUILD_TESTING)
    add_executable(runtime_contracts ../tests/runtime_contracts.cpp bench.cpp replay.cpp io_spec.cpp pool.cpp)
    target_include_directories(runtime_contracts PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${THIRD_PARTY_DIR})
    target_link_libraries(runtime_contracts PRIVATE pthread)
    add_test(NAME runtime_contracts COMMAND runtime_contracts ${CMAKE_CURRENT_BINARY_DIR}/test-data)
endif()

add_executable(ge_runtime
    main.cpp
    io_spec.cpp
    pool.cpp
    bench.cpp
    replay.cpp
    acl_json.cpp
    backends/acl_backend.cpp
    backends/gesession_backend.cpp
    backends/graph_loader.cpp
)

target_include_directories(ge_runtime PRIVATE
    ${ASCEND_HOME}/include
    ${THIRD_PARTY_DIR}
)

target_link_directories(ge_runtime PRIVATE ${ASCEND_HOME}/lib64)

# GE 头/库以旧 ABI 构建 (同 atb/CMakeLists.txt ge_infer)
target_compile_options(ge_runtime PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)

target_link_libraries(ge_runtime
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

# 最小两阶段 pipeline runtime: vision AIR -> (device 常驻) features -> text AIR。
# 不嵌 io_spec、不带请求池/回放框架; 输入按 io spec 的固定顺序位置绑定。
add_executable(pipeline_runtime pipeline_runtime.cpp)
target_include_directories(pipeline_runtime PRIVATE
    ${ASCEND_HOME}/include
    ${THIRD_PARTY_DIR}
)
target_link_directories(pipeline_runtime PRIVATE ${ASCEND_HOME}/lib64)
target_compile_options(pipeline_runtime PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
target_link_libraries(pipeline_runtime
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
# 注: 不链 acl_op_compiler —— 本运行时不调任何算子编译/aclnn API (图编译走 ATC 子进程或
# ge::Session::CompileGraph)。曾经链着它但无一处引用, 白增一条对 CANN 组件的依赖。

# 纯文本极简 executor: 单 text AIR, 2 输入位置绑定 (input_ids, position_ids);
# CLI 参数控制 infer/bench (--iterations 出现即 bench); 输出 ge-compare/1 +
# (bench) requests.csv。
add_executable(executor executor.cpp)
target_include_directories(executor PRIVATE
    ${ASCEND_HOME}/include
    ${THIRD_PARTY_DIR}
)
target_link_directories(executor PRIVATE ${ASCEND_HOME}/lib64)
target_compile_options(executor PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
target_link_libraries(executor
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
