# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom build and run requirements."""

load(
    "//build_tools/bazel:requirements.bzl",
    "build_requirement",
    "run_requirement",
)

EMIT_AMDGPU = build_requirement(
    id = "loom.emit.amdgpu",
    label = "//loom/requirements:emit_amdgpu",
    enabled_by = "//loom/config/emit:amdgpu",
    cmake_condition = "LOOM_EMIT_AMDGPU",
)

EMIT_IREEVM = build_requirement(
    id = "loom.emit.ireevm",
    label = "//loom/requirements:emit_ireevm",
    enabled_by = "//loom/config/emit:ireevm",
    cmake_condition = "LOOM_EMIT_IREEVM",
)

EMIT_LLVMIR = build_requirement(
    id = "loom.emit.llvmir",
    label = "//loom/requirements:emit_llvmir",
    enabled_by = "//loom/config/emit:llvmir",
    cmake_condition = "LOOM_EMIT_LLVMIR",
)

EMIT_SPIRV = build_requirement(
    id = "loom.emit.spirv",
    label = "//loom/requirements:emit_spirv",
    enabled_by = "//loom/config/emit:spirv",
    cmake_condition = "LOOM_EMIT_SPIRV",
)

EMIT_WASM = build_requirement(
    id = "loom.emit.wasm",
    label = "//loom/requirements:emit_wasm",
    enabled_by = "//loom/config/emit:wasm",
    cmake_condition = "LOOM_EMIT_WASM",
)

EXECUTE_AMDGPU = build_requirement(
    id = "loom.execute.amdgpu",
    label = "//loom/requirements:execute_amdgpu",
    enabled_by = "//loom/config/execute:amdgpu",
    cmake_condition = "LOOM_EXECUTE_AMDGPU",
)

EXECUTE_IREEVM = build_requirement(
    id = "loom.execute.ireevm",
    label = "//loom/requirements:execute_ireevm",
    enabled_by = "//loom/config/execute:ireevm",
    cmake_condition = "LOOM_EXECUTE_IREEVM",
)

EXECUTE_SPIRV_VULKAN = build_requirement(
    id = "loom.execute.spirv_vulkan",
    label = "//loom/requirements:execute_spirv_vulkan",
    enabled_by = "//loom/config/execute:spirv_vulkan",
    cmake_condition = "LOOM_EXECUTE_SPIRV_VULKAN",
)

IMPORT_MLIR = build_requirement(
    id = "loom.import.mlir",
    label = "//loom/requirements:import_mlir",
    enabled_by = "//loom/config/import:mlir",
    cmake_condition = "LOOM_IMPORT_MLIR",
)

IMPORT_TILELANG = build_requirement(
    id = "loom.import.tilelang",
    label = "//loom/requirements:import_tilelang",
    enabled_by = "//loom/config/import:tilelang",
    cmake_condition = "LOOM_IMPORT_TILELANG",
)

TARGET_ARCH_AMDGPU = build_requirement(
    id = "loom.target.arch.amdgpu",
    label = "//loom/requirements:target_arch_amdgpu",
    enabled_by = "//loom/config/target/arch:amdgpu",
    cmake_condition = "LOOM_TARGET_ARCH_AMDGPU",
)

TARGET_ARCH_IREEVM = build_requirement(
    id = "loom.target.arch.ireevm",
    label = "//loom/requirements:target_arch_ireevm",
    enabled_by = "//loom/config/target/arch:ireevm",
    cmake_condition = "LOOM_TARGET_ARCH_IREEVM",
)

TARGET_ARCH_SPIRV = build_requirement(
    id = "loom.target.arch.spirv",
    label = "//loom/requirements:target_arch_spirv",
    enabled_by = "//loom/config/target/arch:spirv",
    cmake_condition = "LOOM_TARGET_ARCH_SPIRV",
)

TARGET_ARCH_WASM = build_requirement(
    id = "loom.target.arch.wasm",
    label = "//loom/requirements:target_arch_wasm",
    enabled_by = "//loom/config/target/arch:wasm",
    cmake_condition = "LOOM_TARGET_ARCH_WASM",
)

TARGET_ARCH_X86 = build_requirement(
    id = "loom.target.arch.x86",
    label = "//loom/requirements:target_arch_x86",
    enabled_by = "//loom/config/target/arch:x86",
    cmake_condition = "LOOM_TARGET_ARCH_X86",
)

AMDGPU_RESOURCE = run_requirement(
    id = "loom.resource.amd_gpu",
    label = "//loom/requirements:amd_gpu",
    cmake_label = "loom-resource=amd-gpu",
    skip_contract = "Tests skip when no compatible AMD GPU/HSA agent is available.",
)

VULKAN_DEVICE_RESOURCE = run_requirement(
    id = "loom.resource.vulkan_device",
    label = "//loom/requirements:vulkan_device",
    cmake_label = "loom-resource=vulkan-device",
    skip_contract = "Tests skip when no compatible Vulkan device is available.",
)

REQUIREMENTS = [
    AMDGPU_RESOURCE,
    EMIT_AMDGPU,
    EMIT_IREEVM,
    EMIT_LLVMIR,
    EMIT_SPIRV,
    EMIT_WASM,
    EXECUTE_AMDGPU,
    EXECUTE_IREEVM,
    EXECUTE_SPIRV_VULKAN,
    IMPORT_MLIR,
    IMPORT_TILELANG,
    TARGET_ARCH_AMDGPU,
    TARGET_ARCH_IREEVM,
    TARGET_ARCH_SPIRV,
    TARGET_ARCH_WASM,
    TARGET_ARCH_X86,
    VULKAN_DEVICE_RESOURCE,
]
