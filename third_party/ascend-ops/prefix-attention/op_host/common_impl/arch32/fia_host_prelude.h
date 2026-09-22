/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software and you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may refer to the License for details.
 * You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fia_host_prelude.h
 * \brief vendored common host链的编译期符号补全(独立工程裁剪)
 * 源仓构建中, fia_tiling_nonquant.cpp经由 FIA tiling.h → IFA tiling_base.h 链条获得:
 *   - namespace AscendC (tiling/tiling_api.h)
 *   - 全局常量 BYTE_BLOCK / MAX_SPLIT_SIZE (incre_flash_attention_tiling_base.h)
 * 本工程已剥离IFA/FIA头依赖, 在此以最小集合自包含提供, 取值与源仓一致。
 */
#ifndef FIA_HOST_PRELUDE_H
#define FIA_HOST_PRELUDE_H

#include "tiling/tiling_api.h" // namespace AscendC (host侧)

constexpr uint32_t BYTE_BLOCK = 32;       // 与IFA incre_flash_attention_tiling_base.h一致
constexpr uint32_t MAX_SPLIT_SIZE = 8192; // 与IFA incre_flash_attention_tiling_base.h一致

#endif // FIA_HOST_PRELUDE_H
