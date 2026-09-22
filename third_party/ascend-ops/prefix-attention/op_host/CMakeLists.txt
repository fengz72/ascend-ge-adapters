
# 显式源集合: def/infershape/tiling/checker + vendored tiling框架与common host实现
set(ops_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/prefix_infer_attention_score_def.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/prefix_infer_attention_score_infershape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/prefix_infer_attention_score_tiling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/prefix_infer_attention_score_tiling_check.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fia_tiling_impl/fused_infer_attention_score_tiling_info_parser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common_impl/fia_tiling_shape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common_impl/fia_tiling_info.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common_impl/split_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common_impl/arch32/fia_tiling_nonquant.cpp
)

# proto so必须直接编译def+infershape源码: 图模式下GE InferShapePass从opsproto so取infer函数,
# op_build autogen只提取InputsDataDependency等可序列化项(原模板变量OPS_DEF_SRC/OPS_INFER_SRC为空导致infer丢失)
set(OPS_DEF_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/prefix_infer_attention_score_def.cpp
)
set(OPS_INFER_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/prefix_infer_attention_score_infershape.cpp
)
set(OPS_TILING_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/prefix_infer_attention_score_tiling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/prefix_infer_attention_score_tiling_check.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fia_tiling_impl/fused_infer_attention_score_tiling_info_parser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common_impl/fia_tiling_shape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common_impl/fia_tiling_info.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common_impl/split_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common_impl/arch32/fia_tiling_nonquant.cpp
)

# host编译统一include集合: 与源仓验证过的standalone构建一致(CANN头路径) + 本工程vendored目录
set(PIA_CANN_INC
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/include/exe_graph
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include/highlevel_api
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include/highlevel_api/lib
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include/highlevel_api/impl
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include/highlevel_api/tiling
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/pkg_inc/op_common
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/pkg_inc/base
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/pkg_inc/profiling
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/pkg_inc/aicpu_common/context
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/pkg_inc/aicpu
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/include/aclnn
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/include/experiment
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/include/aicpu_common/context
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/include/op_common/op_host
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef/common/util
    ${ASCEND_CANN_PACKAGE_PATH}/include/external
    ${ASCEND_CANN_PACKAGE_PATH}/include/platform
    ${ASCEND_CANN_PACKAGE_PATH}/include/op_common
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling
)
set(PIA_HOST_INC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/fia_tiling_impl
    ${CMAKE_CURRENT_SOURCE_DIR}/common_impl
    ${CMAKE_CURRENT_SOURCE_DIR}/common_impl/arch32
    ${CMAKE_CURRENT_SOURCE_DIR}/common_impl/include
    ${PIA_CANN_INC}
)
set(PIA_INC_FLAGS "")
foreach(dir ${PIA_HOST_INC})
    list(APPEND PIA_INC_FLAGS "-I${dir}")
endforeach()

opbuild(OPS_SRC ${ops_srcs}
        OUT_DIR ${ASCEND_AUTOGEN_PATH}
        INCLUDES ${PIA_INC_FLAGS}
)

file(GLOB group_proto_src ${ASCEND_AUTOGEN_PATH}/group_proto/*.cc)
 
add_library(cust_op_proto SHARED
    ${group_proto_src}
    ${OPS_DEF_SRC}
    ${OPS_INFER_SRC}
    ${ASCEND_AUTOGEN_PATH}/op_proto.cc
)
target_include_directories(cust_op_proto PRIVATE ${PIA_HOST_INC})
target_compile_definitions(cust_op_proto PRIVATE OP_PROTO_LIB)
target_compile_options(cust_op_proto PRIVATE
        -fvisibility=hidden
)
if(ENABLE_CROSS_COMPILE)
    target_link_directories(cust_op_proto PRIVATE
                            ${CMAKE_COMPILE_COMPILER_LIBRARY}
                            ${CMAKE_COMPILE_RUNTIME_LIBRARY}
    )
endif()
target_link_libraries(cust_op_proto PRIVATE
        intf_pub
        exe_graph
        register
        tiling_api
        -Wl,--whole-archive
        rt2_registry
        -Wl,--no-whole-archive
)
set_target_properties(cust_op_proto PROPERTIES OUTPUT_NAME
                      cust_opsproto_rt2.0
)
file(GLOB fallback_src ${ASCEND_AUTOGEN_PATH}/fallback_*.cpp)
add_library(cust_optiling SHARED ${OPS_TILING_SRC})
list(LENGTH fallback_src fallback_src_len)
if (fallback_src_len GREATER 0)
    target_sources(cust_optiling PRIVATE ${fallback_src})
endif()
target_include_directories(cust_optiling PRIVATE ${PIA_HOST_INC})
target_compile_definitions(cust_optiling PRIVATE OP_TILING_LIB)
target_compile_options(cust_optiling PRIVATE
        -fvisibility=hidden
)
if(ENABLE_CROSS_COMPILE)
    target_link_directories(cust_optiling PRIVATE
                            ${CMAKE_COMPILE_COMPILER_LIBRARY}
                            ${CMAKE_COMPILE_RUNTIME_LIBRARY}
    )
endif()
target_link_libraries(cust_optiling PRIVATE
        nnopbase
        intf_pub
        exe_graph
        register
        tiling_api
        -Wl,--whole-archive
        rt2_registry
        -Wl,--no-whole-archive
)
set_target_properties(cust_optiling PROPERTIES OUTPUT_NAME
                      cust_opmaster_rt2.0
)

file(GLOB aclnn_src ${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp)
file(GLOB aclnn_inc ${ASCEND_AUTOGEN_PATH}/aclnn_*.h)
# 排除自动生成的本算子aclnn glue(与手写op_api裁剪包装器符号冲突), 采用手写包装器
list(FILTER aclnn_src EXCLUDE REGEX "aclnn_prefix_infer_attention_score\.cpp$")
list(FILTER aclnn_inc EXCLUDE REGEX "aclnn_prefix_infer_attention_score\.h$")
list(APPEND aclnn_src ${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_prefix_infer_attention_score.cpp
                  ${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_prefix_infer_attention_score_full.cpp
                  ${CMAKE_CURRENT_SOURCE_DIR}/op_api/prefix_infer_attention_score_custom_op.cpp)
list(APPEND aclnn_inc ${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_prefix_infer_attention_score.h)
if(NOT ASCEND_PACK_SHARED_LIBRARY)
    add_library(cust_opapi SHARED ${aclnn_src})
target_include_directories(cust_opapi PRIVATE ${PIA_HOST_INC})
else()
    set(GENERATED_SOURCE ${ASCEND_AUTOGEN_PATH}/custom_op_registry_V2.cpp)
    set_source_files_properties(${GENERATED_SOURCE}
        PROPERTIES
        GENERATED TRUE
    )
    add_library(cust_opapi_obj OBJECT ${aclnn_src} ${GENERATED_SOURCE})
target_include_directories(cust_opapi_obj PRIVATE ${PIA_HOST_INC})
    target_compile_options(cust_opapi_obj PRIVATE -DLOG_CPP)
    target_compile_definitions(cust_opapi_obj PRIVATE ACLNN_WITH_BINARY)
    if(ENABLE_CROSS_COMPILE)
        target_link_directories(cust_opapi_obj PRIVATE
                                ${CMAKE_COMPILE_COMPILER_LIBRARY}
                                ${CMAKE_COMPILE_RUNTIME_LIBRARY}
        )
    endif()
    target_link_libraries(cust_opapi_obj PRIVATE
                        intf_pub
                        exe_graph
                        register
                        tiling_api
                        -Wl,--whole-archive
                        rt2_registry
                        -Wl,--no-whole-archive
                        ascend_opregistry
                        ascend_kernels
    )
    target_compile_options(cust_opapi_obj PRIVATE
                        -fvisibility=hidden
    )
    add_dependencies(cust_opapi_obj ascend_opregistry)
    add_library(cust_op_proto_obj OBJECT
                ${group_proto_src}
                ${OPS_DEF_SRC}
                ${OPS_INFER_SRC}
                ${ASCEND_AUTOGEN_PATH}/op_proto.cc
    )
    target_compile_definitions(cust_op_proto_obj PRIVATE OP_PROTO_LIB)
    target_compile_options(cust_op_proto_obj PRIVATE
                           -fvisibility=hidden
    )
    if(ENABLE_CROSS_COMPILE)
        target_link_directories(cust_op_proto_obj PRIVATE
                                ${CMAKE_COMPILE_COMPILER_LIBRARY}
                                ${CMAKE_COMPILE_RUNTIME_LIBRARY}
    )
    endif()
    target_link_libraries(cust_op_proto_obj PRIVATE
                          intf_pub
                          exe_graph
                          register
                          tiling_api
                          -Wl,--whole-archive
                          rt2_registry
                          -Wl,--no-whole-archive
    )
    add_library(cust_optiling_obj OBJECT ${OPS_TILING_SRC})
    target_compile_definitions(cust_optiling_obj PRIVATE OP_TILING_LIB)
    target_compile_options(cust_optiling_obj PRIVATE
                           -fvisibility=hidden
    )
    if(ENABLE_CROSS_COMPILE)
        target_link_directories(cust_optiling_obj PRIVATE
                                ${CMAKE_COMPILE_COMPILER_LIBRARY}
                                ${CMAKE_COMPILE_RUNTIME_LIBRARY}
        )
    endif()
    target_link_libraries(cust_optiling_obj PRIVATE
                          intf_pub
                          exe_graph
                          register
                          tiling_api
                          -Wl,--whole-archive
                          rt2_registry
                          -Wl,--no-whole-archive
    )
    set(DYNAMIC_LIB_NAME "cust_opapi")
    add_library(${DYNAMIC_LIB_NAME} SHARED
        $<TARGET_OBJECTS:cust_op_proto_obj>
        $<TARGET_OBJECTS:cust_optiling_obj>
        $<TARGET_OBJECTS:cust_opapi_obj>
    )
endif()
if(ENABLE_CROSS_COMPILE)
    target_link_directories(cust_opapi PRIVATE
                            ${CMAKE_COMPILE_COMPILER_LIBRARY}
                            ${CMAKE_COMPILE_RUNTIME_LIBRARY}
    )
endif()
if(NOT ASCEND_PACK_SHARED_LIBRARY)
    # opapi: aclnn注册器init期会dlsym aclnnInplaceCopy*(定义于libopapi.so), 显式依赖使其在任何加载序下可解析
    # (无此依赖时GE/torchair进程先于libopapi加载本so会FATAL中断vendor注册, eager路径靠torch_npu先载libopapi侥幸通过)
    target_link_libraries(cust_opapi PRIVATE intf_pub ascendcl nnopbase opapi)
else()
    target_link_libraries(${DYNAMIC_LIB_NAME} PRIVATE
                        intf_pub
                        exe_graph
                        register
                        tiling_api
                        -Wl,--whole-archive
                        rt2_registry
                        -Wl,--no-whole-archive
    )
    target_link_libraries(${DYNAMIC_LIB_NAME} PRIVATE intf_pub ascendcl nnopbase ascend_opregistry ascend_kernels)
    add_dependencies(${DYNAMIC_LIB_NAME} ascend_opregistry ascend_kernels)

    set(STATIC_LIB_NAME ${vendor_name})
    add_static_library(${STATIC_LIB_NAME} cust_opapi_obj cust_op_proto_obj cust_optiling_obj)
    add_dependencies(${STATIC_LIB_NAME} ascend_opregistry ascend_kernels)
endif()

add_custom_target(optiling_compat ALL
                  COMMAND ln -sf lib/linux/${CMAKE_SYSTEM_PROCESSOR}/$<TARGET_FILE_NAME:cust_optiling>
                          ${CMAKE_CURRENT_BINARY_DIR}/liboptiling.so
)

if (ASCEND_PACK_SHARED_LIBRARY)
    file(REMOVE_RECURSE ${CMAKE_BINARY_DIR}/cmake/${vendor_name})
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/cmake/${vendor_name})
    add_vendor_cmake(${vendor_name} ${CMAKE_BINARY_DIR}/cmake/${vendor_name} ${STATIC_LIB_NAME} ${DYNAMIC_LIB_NAME})
endif()

if(NOT ASCEND_PACK_SHARED_LIBRARY)
        install(TARGETS cust_op_proto
                LIBRARY DESTINATION packages/vendors/${vendor_name}/op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
        install(FILES ${ASCEND_AUTOGEN_PATH}/op_proto.h
                DESTINATION packages/vendors/${vendor_name}/op_proto/inc)
        file(GLOB GROUP_PROTO_HEADERS ${ASCEND_AUTOGEN_PATH}/group_proto/*.h)
        if (GROUP_PROTO_HEADERS)
                install(FILES ${GROUP_PROTO_HEADERS}
                        DESTINATION packages/vendors/${vendor_name}/op_proto/inc)
        endif()
        install(TARGETS cust_optiling
                LIBRARY DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_tiling/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/liboptiling.so
                DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_tiling)
        install(TARGETS cust_opapi
                LIBRARY DESTINATION packages/vendors/${vendor_name}/op_api/lib)
        install(FILES ${aclnn_inc}
                DESTINATION packages/vendors/${vendor_name}/op_api/include)
else()
        file(GLOB group_inc ${ASCEND_AUTOGEN_PATH}/group_proto/*.h)
        install(TARGETS cust_opapi
                LIBRARY DESTINATION op_api/lib)
        install(FILES ${CMAKE_BINARY_DIR}/op_host/lib${STATIC_LIB_NAME}.a
                DESTINATION op_api/lib)
        install(DIRECTORY ${CMAKE_BINARY_DIR}/cmake
                DESTINATION op_api/lib)
        install(FILES ${ASCEND_AUTOGEN_PATH}/op_proto.h
                DESTINATION op_api/include)
        install(FILES ${group_inc}
                DESTINATION op_api/include)
        install(FILES ${aclnn_inc}
                DESTINATION op_api/include)
endif()
