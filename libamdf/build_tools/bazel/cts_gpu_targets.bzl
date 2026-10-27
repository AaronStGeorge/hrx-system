# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Physical compiler products and wave-mode selection for GPU CTS fixtures."""

load("//loom/build_tools/amdgpu:target_config.bzl", "LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITY_BY_TARGET", "LOOM_AMDGPU_SUPPORTED_TARGETS")

# Generic code-object processors do not identify physical test endpoints.
AMDF_CTS_GPU_KERNEL_TARGETS = sorted([
    target
    for target in LOOM_AMDGPU_SUPPORTED_TARGETS
    if not target.endswith("-generic")
])

# PM4 fixtures include GFX12.5; these selectors are not a product-family name.
AMDF_CTS_GPU_PM4_TARGETS = [
    target
    for target in AMDF_CTS_GPU_KERNEL_TARGETS
    if not target.startswith("gfx9")
]

# The alternate wave64 cases require a physical target supporting both widths.
# Identical select values include a case only once across enabled capabilities.
AMDF_CTS_GPU_DUAL_WAVE_CONFIGS = [
    "//loom/config/arch/amdgpu:descriptor_set_rdna3_core",
    "//loom/config/arch/amdgpu:descriptor_set_rdna3_5_core",
    "//loom/config/arch/amdgpu:descriptor_set_rdna4_core",
    "//loom/config/arch/amdgpu:descriptor_set_rdna4m_core",
]

AMDF_CTS_GPU_DUAL_WAVE_TARGETS = [
    target
    for target in AMDF_CTS_GPU_PM4_TARGETS
    if "//loom/config/arch/amdgpu:" + LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITY_BY_TARGET[target] in AMDF_CTS_GPU_DUAL_WAVE_CONFIGS
]
