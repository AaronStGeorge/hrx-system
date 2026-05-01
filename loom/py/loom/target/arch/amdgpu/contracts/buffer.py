# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU buffer source-to-low contracts."""

from __future__ import annotations

from loom.dialect.buffer import ALL_BUFFER_OPS
from loom.dialect.buffer import defs as buffer
from loom.target.arch.amdgpu.contracts.memory import BYTE_ADDRESSABLE_SCALAR_ELEMENTS
from loom.target.arch.amdgpu.descriptors import build_amdgpu_contract_descriptor_set
from loom.target.contracts import (
    ContractFragment,
    Guard,
    GuardDiagnostic,
    ValueAliasRule,
    ValueRef,
    View,
)

_DESCRIPTOR_SET = build_amdgpu_contract_descriptor_set(
    key="amdgpu.buffer",
    descriptor_keys=(),
)

_VIEW_TYPE_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="memory",
    subject_name="buffer.view",
    reason=(
        "AMDGPU buffer memory lowering requires typed views over "
        "byte-addressable scalar elements"
    ),
)

_STATIC_BYTE_OFFSET_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="memory",
    subject_name="buffer.view",
    reason="AMDGPU HAL buffer views require exact non-negative static byte offsets",
)

AMDGPU_BUFFER_CONTRACT_DIALECT_OPS = {
    "buffer": ALL_BUFFER_OPS,
}

AMDGPU_BUFFER_CONTRACT_FRAGMENT = ContractFragment(
    name="amdgpu.buffer",
    descriptor_set=_DESCRIPTOR_SET,
    cases=(
        ValueAliasRule(
            source_op=buffer.buffer_view,
            source=ValueRef.operand("buffer"),
            result=ValueRef.result("result"),
            guards=(
                Guard.value_type(
                    "result",
                    View(BYTE_ADDRESSABLE_SCALAR_ELEMENTS),
                    diagnostic=_VIEW_TYPE_DIAGNOSTIC,
                ),
                Guard.value_exact_i64(
                    "byte_offset",
                    diagnostic=_STATIC_BYTE_OFFSET_DIAGNOSTIC,
                ),
                Guard.value_i64_range(
                    "byte_offset",
                    0,
                    0x7FFF_FFFF_FFFF_FFFF,
                    diagnostic=_STATIC_BYTE_OFFSET_DIAGNOSTIC,
                ),
            ),
        ),
    ),
)
