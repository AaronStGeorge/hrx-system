# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the kernel dialect declarations."""

from loom.dialect.cache import CacheScope, CacheTemporal
from loom.dialect.kernel import (
    ALL_KERNEL_OPS,
    ALL_KERNEL_TYPES,
    KernelAsyncDirection,
    KernelMemorySpace,
    KernelOrdering,
    KernelScope,
    kernel_async_cluster_gather,
    kernel_async_cluster_gather_mask,
    kernel_async_copy,
    kernel_async_copy_mask,
    kernel_async_gather,
    kernel_async_gather_mask,
    kernel_async_group,
    kernel_async_group_type,
    kernel_async_tensor_load_to_lds,
    kernel_async_tensor_store_from_lds,
    kernel_async_token_type,
    kernel_async_wait,
    kernel_barrier,
    kernel_ops,
    kernel_tensor_lds_descriptor,
    kernel_tensor_lds_descriptor_type,
)
from loom.dsl import ANY, ATTR_TYPE_ENUM, ATTR_TYPE_I64, I1, INTEGER, UNKNOWN_EFFECTS, VECTOR, VIEW, Op


def _ops() -> dict[str, Op]:
    return {op.name: op for op in ALL_KERNEL_OPS}


class TestKernelDialect:
    def test_dialect_id(self) -> None:
        assert kernel_ops.dialect_id == 0x10

    def test_inventory(self) -> None:
        assert [type_def.name for type_def in ALL_KERNEL_TYPES] == [
            "kernel.async.group",
            "kernel.async.token",
            "kernel.tensor.lds.descriptor",
        ]
        assert [op.name for op in ALL_KERNEL_OPS] == [
            "kernel.barrier",
            "kernel.async.copy",
            "kernel.async.copy.mask",
            "kernel.async.gather",
            "kernel.async.gather.mask",
            "kernel.async.group",
            "kernel.async.wait",
            "kernel.tensor.lds.descriptor",
            "kernel.async.tensor.load.to.lds",
            "kernel.async.tensor.store.from.lds",
            "kernel.async.cluster.gather",
            "kernel.async.cluster.gather.mask",
        ]

    def test_public_exports_match_registry(self) -> None:
        assert kernel_barrier in ALL_KERNEL_OPS
        assert kernel_async_cluster_gather in ALL_KERNEL_OPS
        assert kernel_async_cluster_gather_mask in ALL_KERNEL_OPS
        assert kernel_async_copy in ALL_KERNEL_OPS
        assert kernel_async_copy_mask in ALL_KERNEL_OPS
        assert kernel_async_gather in ALL_KERNEL_OPS
        assert kernel_async_gather_mask in ALL_KERNEL_OPS
        assert kernel_async_group in ALL_KERNEL_OPS
        assert kernel_async_tensor_load_to_lds in ALL_KERNEL_OPS
        assert kernel_async_tensor_store_from_lds in ALL_KERNEL_OPS
        assert kernel_async_wait in ALL_KERNEL_OPS
        assert kernel_tensor_lds_descriptor in ALL_KERNEL_OPS
        assert kernel_async_group_type in ALL_KERNEL_TYPES
        assert kernel_async_token_type in ALL_KERNEL_TYPES
        assert kernel_tensor_lds_descriptor_type in ALL_KERNEL_TYPES

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

    def test_async_copy_shape(self) -> None:
        op = _ops()["kernel.async.copy"]
        assert [operand.name for operand in op.operands] == ["source", "dest"]
        assert [operand.type_constraint for operand in op.operands] == [VIEW, VIEW]
        assert [result.name for result in op.results] == ["token"]
        assert op.results[0].type_constraint == ANY
        assert [attr.name for attr in op.attrs] == [
            "cache_scope",
            "cache_temporal",
            "direction",
        ]
        assert all(attr.attr_type == ATTR_TYPE_ENUM for attr in op.attrs)
        assert op.attrs[0].enum_def is CacheScope
        assert op.attrs[1].enum_def is CacheTemporal
        assert op.attrs[2].enum_def is KernelAsyncDirection
        assert op.constraints == ()
        assert [(effect.operand, effect.kind.value) for effect in op.effects] == [
            ("source", "read"),
            ("dest", "write"),
        ]
        assert op.verify == "loom_kernel_async_copy_verify"

    def test_async_copy_mask_shape(self) -> None:
        op = _ops()["kernel.async.copy.mask"]
        assert [operand.name for operand in op.operands] == [
            "source",
            "dest",
            "predicate",
        ]
        assert [operand.type_constraint for operand in op.operands] == [VIEW, VIEW, I1]
        assert [result.name for result in op.results] == ["token"]
        assert op.results[0].type_constraint == ANY
        assert [attr.name for attr in op.attrs] == [
            "cache_scope",
            "cache_temporal",
            "direction",
        ]
        assert all(attr.attr_type == ATTR_TYPE_ENUM for attr in op.attrs)
        assert op.constraints == ()
        assert op.verify == "loom_kernel_async_copy_mask_verify"

    def test_async_gather_shape(self) -> None:
        op = _ops()["kernel.async.gather"]
        assert [operand.name for operand in op.operands] == ["source", "dest"]
        assert [operand.type_constraint for operand in op.operands] == [VIEW, VIEW]
        assert [result.name for result in op.results] == ["token"]
        assert op.results[0].type_constraint == ANY
        assert [attr.name for attr in op.attrs] == ["cache_scope", "cache_temporal"]
        assert all(attr.attr_type == ATTR_TYPE_ENUM for attr in op.attrs)
        assert op.attrs[0].enum_def is CacheScope
        assert op.attrs[1].enum_def is CacheTemporal
        assert op.constraints == ()
        assert op.verify == "loom_kernel_async_gather_verify"

    def test_async_gather_mask_shape(self) -> None:
        op = _ops()["kernel.async.gather.mask"]
        assert [operand.name for operand in op.operands] == [
            "source",
            "dest",
            "predicate",
        ]
        assert [operand.type_constraint for operand in op.operands] == [VIEW, VIEW, I1]
        assert [result.name for result in op.results] == ["token"]
        assert op.results[0].type_constraint == ANY
        assert [attr.name for attr in op.attrs] == ["cache_scope", "cache_temporal"]
        assert all(attr.attr_type == ATTR_TYPE_ENUM for attr in op.attrs)
        assert op.attrs[0].enum_def is CacheScope
        assert op.attrs[1].enum_def is CacheTemporal
        assert op.constraints == ()
        assert op.verify == "loom_kernel_async_gather_mask_verify"

    def test_async_cluster_gather_shape(self) -> None:
        op = _ops()["kernel.async.cluster.gather"]
        assert [operand.name for operand in op.operands] == [
            "source",
            "dest",
            "cluster_mask",
        ]
        assert [operand.type_constraint for operand in op.operands] == [
            VIEW,
            VIEW,
            INTEGER,
        ]
        assert [result.name for result in op.results] == ["token"]
        assert op.results[0].type_constraint == ANY
        assert [attr.name for attr in op.attrs] == ["cache_scope", "cache_temporal"]
        assert all(attr.attr_type == ATTR_TYPE_ENUM for attr in op.attrs)
        assert op.attrs[0].enum_def is CacheScope
        assert op.attrs[1].enum_def is CacheTemporal
        assert [(effect.operand, effect.kind.value) for effect in op.effects] == [
            ("source", "read"),
            ("dest", "write"),
        ]
        assert op.verify == "loom_kernel_async_cluster_gather_verify"

    def test_async_cluster_gather_mask_shape(self) -> None:
        op = _ops()["kernel.async.cluster.gather.mask"]
        assert [operand.name for operand in op.operands] == [
            "source",
            "dest",
            "cluster_mask",
            "predicate",
        ]
        assert [operand.type_constraint for operand in op.operands] == [
            VIEW,
            VIEW,
            INTEGER,
            I1,
        ]
        assert [result.name for result in op.results] == ["token"]
        assert op.results[0].type_constraint == ANY
        assert [attr.name for attr in op.attrs] == ["cache_scope", "cache_temporal"]
        assert all(attr.attr_type == ATTR_TYPE_ENUM for attr in op.attrs)
        assert op.attrs[0].enum_def is CacheScope
        assert op.attrs[1].enum_def is CacheTemporal
        assert [(effect.operand, effect.kind.value) for effect in op.effects] == [
            ("source", "read"),
            ("dest", "write"),
        ]
        assert op.verify == "loom_kernel_async_cluster_gather_mask_verify"

    def test_tensor_lds_descriptor_shape(self) -> None:
        op = _ops()["kernel.tensor.lds.descriptor"]
        assert [operand.name for operand in op.operands] == ["dgroups"]
        assert op.operands[0].type_constraint == VECTOR
        assert op.operands[0].variadic
        assert [result.name for result in op.results] == ["descriptor"]
        assert op.results[0].type_constraint == ANY
        assert op.constraints == ()
        assert op.is_pure
        assert op.verify == "loom_kernel_tensor_lds_descriptor_verify"

    def test_async_tensor_load_to_lds_shape(self) -> None:
        op = _ops()["kernel.async.tensor.load.to.lds"]
        assert [operand.name for operand in op.operands] == [
            "source",
            "dest",
            "descriptor",
        ]
        assert [operand.type_constraint for operand in op.operands] == [
            VIEW,
            VIEW,
            ANY,
        ]
        assert [result.name for result in op.results] == ["token"]
        assert op.results[0].type_constraint == ANY
        assert [attr.name for attr in op.attrs] == ["cache_scope", "cache_temporal"]
        assert all(attr.attr_type == ATTR_TYPE_ENUM for attr in op.attrs)
        assert op.attrs[0].enum_def is CacheScope
        assert op.attrs[1].enum_def is CacheTemporal
        assert [(effect.operand, effect.kind.value) for effect in op.effects] == [
            ("source", "read"),
            ("dest", "write"),
        ]
        assert op.verify == "loom_kernel_async_tensor_load_to_lds_verify"

    def test_async_tensor_store_from_lds_shape(self) -> None:
        op = _ops()["kernel.async.tensor.store.from.lds"]
        assert [operand.name for operand in op.operands] == [
            "source",
            "dest",
            "descriptor",
        ]
        assert [operand.type_constraint for operand in op.operands] == [
            VIEW,
            VIEW,
            ANY,
        ]
        assert [result.name for result in op.results] == ["token"]
        assert op.results[0].type_constraint == ANY
        assert [attr.name for attr in op.attrs] == ["cache_scope", "cache_temporal"]
        assert all(attr.attr_type == ATTR_TYPE_ENUM for attr in op.attrs)
        assert op.attrs[0].enum_def is CacheScope
        assert op.attrs[1].enum_def is CacheTemporal
        assert [(effect.operand, effect.kind.value) for effect in op.effects] == [
            ("source", "read"),
            ("dest", "write"),
        ]
        assert op.verify == "loom_kernel_async_tensor_store_from_lds_verify"

    def test_async_group_shape(self) -> None:
        op = _ops()["kernel.async.group"]
        assert [operand.name for operand in op.operands] == ["tokens"]
        assert op.operands[0].type_constraint == ANY
        assert op.operands[0].variadic
        assert [result.name for result in op.results] == ["group"]
        assert op.results[0].type_constraint == ANY
        assert UNKNOWN_EFFECTS in op.traits
        assert op.verify == "loom_kernel_async_group_verify"

    def test_async_wait_shape(self) -> None:
        op = _ops()["kernel.async.wait"]
        assert [operand.name for operand in op.operands] == ["group"]
        assert op.operands[0].type_constraint == ANY
        assert not op.results
        assert [attr.name for attr in op.attrs] == ["newer_groups"]
        assert op.attrs[0].attr_type == ATTR_TYPE_I64
        assert UNKNOWN_EFFECTS in op.traits
        assert op.verify == "loom_kernel_async_wait_verify"

    def test_async_direction_values(self) -> None:
        assert [(case.keyword, case.value) for case in KernelAsyncDirection.cases] == [
            ("global_to_workgroup", 0),
            ("workgroup_to_global", 1),
        ]

    def test_async_cache_temporal_values(self) -> None:
        assert [(case.keyword, case.value) for case in CacheTemporal.cases] == [
            ("regular", 0),
            ("non_temporal", 1),
            ("high_temporal", 2),
            ("last_use", 3),
            ("writeback", 4),
            ("non_temporal_regular", 5),
            ("regular_non_temporal", 6),
            ("non_temporal_high_temporal", 7),
            ("non_temporal_writeback", 8),
            ("bypass", 9),
        ]

    def test_async_cache_scope_values(self) -> None:
        assert [(case.keyword, case.value) for case in CacheScope.cases] == [
            ("cu", 0),
            ("se", 1),
            ("device", 2),
            ("system", 3),
        ]

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
