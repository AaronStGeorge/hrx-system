# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the kernel dialect declarations."""

from loom.dialect.kernel import (
    ALL_KERNEL_OPS,
    KernelMemorySpace,
    KernelOrdering,
    KernelScope,
    kernel_barrier,
    kernel_ops,
)
from loom.dsl import ATTR_TYPE_ENUM, UNKNOWN_EFFECTS, Op


def _ops() -> dict[str, Op]:
    return {op.name: op for op in ALL_KERNEL_OPS}


class TestKernelDialect:
    def test_dialect_id(self) -> None:
        assert kernel_ops.dialect_id == 0x10

    def test_inventory(self) -> None:
        assert [op.name for op in ALL_KERNEL_OPS] == ["kernel.barrier"]

    def test_public_exports_match_registry(self) -> None:
        assert kernel_barrier in ALL_KERNEL_OPS

    def test_barrier_has_required_attrs(self) -> None:
        op = _ops()["kernel.barrier"]
        assert [attr.name for attr in op.attrs] == [
            "memory_space",
            "ordering",
            "scope",
        ]
        assert all(attr.attr_type == ATTR_TYPE_ENUM for attr in op.attrs)
        memory_attr = op.attr("memory_space")
        ordering_attr = op.attr("ordering")
        scope_attr = op.attr("scope")
        assert memory_attr is not None
        assert ordering_attr is not None
        assert scope_attr is not None
        assert memory_attr.enum_def is KernelMemorySpace
        assert ordering_attr.enum_def is KernelOrdering
        assert scope_attr.enum_def is KernelScope

    def test_barrier_is_effectful(self) -> None:
        op = _ops()["kernel.barrier"]
        assert UNKNOWN_EFFECTS in op.traits
        assert not op.is_pure

    def test_barrier_has_verifier(self) -> None:
        assert _ops()["kernel.barrier"].verify == "loom_kernel_barrier_verify"

    def test_enum_values_match_memory_fact_conventions(self) -> None:
        assert [(case.keyword, case.value) for case in KernelMemorySpace.cases] == [
            ("unknown", 0),
            ("global", 1),
            ("workgroup", 2),
            ("private", 3),
            ("constant", 4),
            ("host", 5),
            ("descriptor", 6),
        ]

    def test_scope_values_match_atomic_scope_conventions(self) -> None:
        assert [(case.keyword, case.value) for case in KernelScope.cases] == [
            ("thread", 0),
            ("subgroup", 1),
            ("workgroup", 2),
            ("device", 3),
            ("system", 4),
        ]

    def test_ordering_values_match_atomic_ordering_conventions(self) -> None:
        assert [(case.keyword, case.value) for case in KernelOrdering.cases] == [
            ("relaxed", 0),
            ("acquire", 1),
            ("release", 2),
            ("acq_rel", 3),
            ("seq_cst", 4),
        ]
