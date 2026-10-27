# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Canonical Loom compiler capability values and ordinary defaults."""

LOOM_ARCHITECTURES = [
    "aie2p",
    "amdgpu",
    "spirv",
    "vm",
    "wasm",
    "x86",
]

LOOM_ARTIFACT_FORMATS = [
    "elf",
    "spirv",
    "vm",
    "wasm",
]

LOOM_EXECUTION_SUBSTRATES = [
    "iree_hal",
    "iree_vm",
    "wasm_host",
]

# The ordinary build contains every production target except WebAssembly.
# Friendly target presets expand to these same leaves at configuration entry
# points; build graph consumers only observe the canonical lists.
LOOM_DEFAULT_ARCHITECTURES = [
    "aie2p",
    "amdgpu",
    "spirv",
    "vm",
    "x86",
]

LOOM_DEFAULT_ARTIFACT_FORMATS = [
    "elf",
    "spirv",
    "vm",
]

LOOM_DEFAULT_EXECUTION_SUBSTRATES = [
    "iree_hal",
    "iree_vm",
]
