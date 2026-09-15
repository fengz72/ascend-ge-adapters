cmake_minimum_required(VERSION 3.14)
project(acl_dynamic_shape_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ASCEND_HOME)
    if(DEFINED ENV{ASCEND_HOME})
        set(ASCEND_HOME $ENV{ASCEND_HOME})
    else()
        set(ASCEND_HOME /usr/local/Ascend/ascend-toolkit/latest)
    endif()
endif()

message(STATUS "ASCEND_HOME: ${ASCEND_HOME}")

set(ACL_INC_DIR "${ASCEND_HOME}/include")
set(ACL_LIB_DIR "${ASCEND_HOME}/lib64")

include_directories(${ACL_INC_DIR})
link_directories(${ACL_LIB_DIR})

add_executable(acl_infer main.cpp)

target_link_libraries(acl_infer
    ascendcl
    acl_op_compiler
    stdc++
)

add_executable(bench_latency bench_latency.cpp)

target_link_libraries(bench_latency
    ascendcl
    stdc++
    pthread
)

# GESession 在线推理 AIR (参照 cann/ge PR#743 RunGraphAsync 样例)
add_executable(ge_infer main_ge.cc)

set_target_properties(ge_infer PROPERTIES CXX_STANDARD 17)

target_compile_options(ge_infer PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)

target_link_libraries(ge_infer
    ge_compiler
    ge_runner
    ge_common
    ge_common_base
    graph
    graph_base
    ascendcl
    stdc++
    pthread
)

# GESession 多线程延迟/吞吐 benchmark (镜像 bench_latency, GE 在线路径)
add_executable(bench_ge_latency bench_ge_latency.cpp)

set_target_properties(bench_ge_latency PROPERTIES CXX_STANDARD 17)

target_compile_options(bench_ge_latency PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)

target_link_libraries(bench_ge_latency
    ge_compiler
    ge_runner
    ge_common
    ge_common_base
    graph
    graph_base
    ascendcl
    stdc++
    pthread
)
