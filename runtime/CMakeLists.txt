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

add_executable(ge_runtime
    main.cpp
    io_spec.cpp
    bench.cpp
    backends/acl_backend.cpp
    backends/gesession_backend.cpp
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
    ascendcl
    acl_op_compiler
    stdc++
    pthread
)
