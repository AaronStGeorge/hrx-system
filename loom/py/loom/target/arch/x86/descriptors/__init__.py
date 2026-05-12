# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Descriptor input data for x86 target-low shards."""

from __future__ import annotations

from .avx512 import X86_AVX512_CORE_DESCRIPTOR_SET
from .composite import (
    X86_AVX512_PACKED_DOT_DESCRIPTOR_SET,
    _merge_component_descriptors,
)
from .packed_dot import X86_PACKED_DOT_DESCRIPTOR_SET
from .scalar import X86_SCALAR_DESCRIPTOR_SET

__all__ = (
    "X86_AVX512_CORE_DESCRIPTOR_SET",
    "X86_AVX512_PACKED_DOT_DESCRIPTOR_SET",
    "X86_PACKED_DOT_DESCRIPTOR_SET",
    "X86_SCALAR_DESCRIPTOR_SET",
    "_merge_component_descriptors",
)
