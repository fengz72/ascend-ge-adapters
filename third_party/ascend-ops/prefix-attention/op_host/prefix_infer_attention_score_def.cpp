/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software and you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may refer to the License for details.
 * You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file prefix_infer_attention_score_def.cpp
 * \brief prefix-in-Q prefill attention: TND布局, prefix KV位于key/value张量头部,
 * batch 0为prefix独立batch(act_q/act_kv均为cumsum([P, L0, L1, ...])), prefix只存一份只算一份
 * 输入序/attr序与FusedInferAttentionScore(9.0.0)保持一致(复用其tiling parser与common模板),
 * key/value_shared_prefix与actual_shared_prefix_len输入槽位保留但永不连接(prefix已内嵌于key/value),
 * dtype场景仅NoQuant fp16/bf16; prefix-in-Q语义由算子自身承载(kernel编译期模板标记)
 */
#include "register/op_def_registry.h"

namespace ops {
// 定义于prefix_infer_attention_score_infershape.cpp, 挂接OpDef供GE图模式InferShapePass调用
ge::graphStatus InferShapePrefixInferAttentionScore(gert::InferShapeContext *context);
ge::graphStatus InferDataTypePrefixInferAttentionScore(gert::InferDataTypeContext *context);

class PrefixInferAttentionScore : public OpDef {
public:
    PrefixInferAttentionScore(const char *name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("value")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("pse_shift")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("atten_mask")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BOOL, ge::DT_BOOL})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_lengths")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_lengths_kv")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("dequant_scale1")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_UINT64, ge::DT_UINT64})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("quant_scale1")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("dequant_scale2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_UINT64, ge::DT_UINT64})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("quant_scale2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("quant_offset2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("antiquant_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("antiquant_offset")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("block_table")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("query_padding_size")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("kv_padding_size")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key_antiquant_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key_antiquant_offset")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("value_antiquant_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("value_antiquant_offset")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key_shared_prefix")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("value_shared_prefix")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_shared_prefix_len")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("query_rope")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key_rope")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key_rope_antiquant_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("dequant_scale_query")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("learnable_sink")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("q_start_idx")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("kv_start_idx")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("attention_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Output("softmax_lse").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        // 图模式: infer函数挂接OpDef(GE InferShapePass按此查找, IMPL_OP_INFERSHAPE宏仅进tiling侧registry)
        this->SetInferShape(ops::InferShapePrefixInferAttentionScore)
            .SetInferDataType(ops::InferDataTypePrefixInferAttentionScore);
        // attr声明顺序与FusedInferAttentionScore一致(tiling parser按attr索引读取), 勿调整
        this->Attr("num_heads").AttrType(REQUIRED).Int(1);
        this->Attr("scale").AttrType(OPTIONAL).Float(1.0);
        this->Attr("pre_tokens").AttrType(OPTIONAL).Int(2147483647); // 2147483647: Maximum value of int32_t.
        this->Attr("next_tokens").AttrType(OPTIONAL).Int(2147483647); // 2147483647: Maximum value of int32_t.
        this->Attr("input_layout").AttrType(OPTIONAL).String("TND");
        this->Attr("num_key_value_heads").AttrType(OPTIONAL).Int(0);
        this->Attr("sparse_mode").AttrType(OPTIONAL).Int(2); // 默认值须与torch schema/torchair converter三层对齐(PIA恒为因果场景)
        this->Attr("inner_precise").AttrType(OPTIONAL).Int(1);
        this->Attr("block_size").AttrType(OPTIONAL).Int(0);
        this->Attr("antiquant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("softmax_lse_flag").AttrType(OPTIONAL).Bool(false);
        this->Attr("key_antiquant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("value_antiquant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("query_quant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("pse_type").AttrType(OPTIONAL).Int(0);
        this->Attr("out_dtype").AttrType(OPTIONAL).Int(0);
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "prefix_infer_attention_score")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore().AddConfig("ascend910b", aicore_config); // use 910B
        this->AICore().AddConfig("ascend910_93", aicore_config);
    }
};
OP_ADD(PrefixInferAttentionScore, optiling::FusedInferAttentionScoreCompileInfo);
} // namespace ops
