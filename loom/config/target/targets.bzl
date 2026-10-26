# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Canonical Loom target product and architecture build-setting values."""

LOOM_TARGETS = [
    "amdgpu",
    "spirv",
    "vm",
    "wasm",
    "xdna",
    "x86",
]

# Targets whose architecture metadata and artifact emitter can be selected as
# independent source slices. Integrated targets derive their architecture
# capability directly from the public product selection.
LOOM_SPLIT_ARTIFACT_TARGETS = [
    "amdgpu",
    "spirv",
    "wasm",
    "xdna",
]
