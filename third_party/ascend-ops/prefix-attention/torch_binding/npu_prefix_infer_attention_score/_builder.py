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

import os

import torch
import torch_npu  # noqa: F401


def _resolve_cann_path():
    env = os.environ.get("ASCEND_HOME_PATH")
    if env and os.path.exists(env):
        return env
    for cand in ("/usr/local/Ascend/ascend-toolkit/latest",):
        if os.path.exists(cand):
            return cand
    raise RuntimeError(
        "CANN toolkit path not found, set ASCEND_HOME_PATH (e.g. source setenv.bash) and retry")


_EXT_NAME = "npu_prefix_infer_attention_score"
_loaded = None


def load_extension():
    """JIT编译并加载aclnn桥接扩展(带进程级缓存), 编译产物缓存在torch extension cache。"""
    global _loaded
    if _loaded is not None:
        return _loaded
    from torch.utils.cpp_extension import load

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    torch_npu_path = os.path.dirname(os.path.abspath(torch_npu.__file__))
    cann_path = _resolve_cann_path()
    include_paths = [
        os.path.join(torch_npu_path, "include"),
        os.path.join(torch_npu_path, "include/third_party/hccl/inc"),
        os.path.join(torch_npu_path, "include/third_party/acl/inc"),
        os.path.join(cann_path, "include"),
        os.path.join(pkg_dir, "_csrc"),
    ]
    cxx_args = [
        "-fstack-protector-all", "-Wl,-z,relro,-z,now,-z,noexecstack", "-fPIC", "-pie",
        "-s", "-fvisibility=hidden", "-D_FORTIFY_SOURCE=2", "-O2", "-w",
        "-D_GLIBCXX_USE_CXX11_ABI=" + ("1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0"),
    ]
    ldflags = [
        "-L" + os.path.join(cann_path, "lib64"), "-lascendcl",
        "-L" + os.path.join(torch_npu_path, "lib"), "-ltorch_npu",
    ]
    _loaded = load(name=_EXT_NAME,
                   sources=[os.path.join(pkg_dir, "_csrc", "prefix_infer_attention_score.cpp")],
                   extra_include_paths=include_paths,
                   extra_cflags=cxx_args,
                   extra_ldflags=ldflags,
                   verbose=False)
    return _loaded
