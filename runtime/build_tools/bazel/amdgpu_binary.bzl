# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runtime policy wrapper for AMDGPU device binaries."""

load("//build_tools/bazel:iree_amdgpu_binary.bzl", _iree_amdgpu_binary = "iree_amdgpu_binary")
load("//runtime/requirements:package_policy.bzl", "apply_runtime_target_policy")

def iree_amdgpu_binary(**kwargs):
    _iree_amdgpu_binary(**apply_runtime_target_policy(kwargs))
