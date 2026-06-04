# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runtime policy wrappers for AMDGPU device binaries."""

load(
    "//build_tools/amdgpu:binary.bzl",
    _iree_amdgpu_binary = "iree_amdgpu_binary",
    _iree_amdgpu_binary_variants = "iree_amdgpu_binary_variants",
    _iree_amdgpu_binary_variants_embed_data = "iree_amdgpu_binary_variants_embed_data",
)
load("//runtime/requirements:package_policy.bzl", "apply_runtime_target_policy")

def iree_amdgpu_binary(**kwargs):
    _iree_amdgpu_binary(**apply_runtime_target_policy(kwargs))

def iree_amdgpu_binary_variants(**kwargs):
    _iree_amdgpu_binary_variants(**apply_runtime_target_policy(kwargs))

def iree_amdgpu_binary_variants_embed_data(**kwargs):
    _iree_amdgpu_binary_variants_embed_data(**apply_runtime_target_policy(kwargs))
