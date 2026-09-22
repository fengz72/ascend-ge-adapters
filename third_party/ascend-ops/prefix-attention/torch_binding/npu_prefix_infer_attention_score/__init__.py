# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may refer to the License for details.
# You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PrefixInferAttentionScore torch接入(独立交付, 不依赖ops-transformer)。

import本包即完成: torch schema注册 + Meta/FakeTensor支持 + PrivateUse1(eager) dispatch
+ torchair GE converter注册(图模式)。

用法:
    import npu_prefix_infer_attention_score  # noqa
    out = torch.ops.npu_ops_transformer.npu_prefix_infer_attention_score(...)
    # 图模式: torch.compile(model, backend=torchair.get_npu_backend(...))

前置条件:
    - CANN算子包(.run)已安装, ASCEND_CUSTOM_OPP_PATH指向vendors目录
    - import前source了CANN setenv.bash(或自行设置ASCEND_HOME_PATH)
    - 首次import会JIT编译aclnn桥接cpp(~1min), 产物缓存在torch extension cache

注意: 与ops-transformer的npu_ops_transformer扩展同进程互斥(同名torch命名空间)。
"""

import torch
import torch_npu  # noqa: F401  PrivateUse1后端注册须先加载torch_npu

from ._schema import AS_LIBRARY, OP_NAME, SCHEMAS  # noqa: F401  schema+Meta注册
from ._eager import op_module  # noqa: F401  JIT编译+PrivateUse1 dispatch
from ._converter import register_converter

register_converter()

__version__ = "1.0.0"
