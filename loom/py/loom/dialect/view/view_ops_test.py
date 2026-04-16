# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the view dialect declarations."""

from loom.dialect.view import (
    ALL_VIEW_OPS,
    AtomicKind,
    AtomicOrdering,
    AtomicScope,
    view_atomic_reduce,
    view_atomic_rmw,
    view_load,
    view_ops,
    view_prefetch,
    view_refine,
    view_store,
    view_subview,
)
from loom.dsl import ATTR_TYPE_I64_ARRAY, INDEX, SCALAR, VIEW, EffectKind, Op


def _ops() -> dict[str, Op]:
    return {op.name: op for op in ALL_VIEW_OPS}


class TestViewDialect:
    def test_inventory(self) -> None:
        assert [op.name for op in ALL_VIEW_OPS] == [
            "view.subview",
            "view.refine",
            "view.load",
            "view.store",
            "view.atomic.reduce",
            "view.atomic.rmw",
            "view.prefetch",
        ]

    def test_public_exports_match_registry(self) -> None:
        assert view_subview in ALL_VIEW_OPS
        assert view_refine in ALL_VIEW_OPS
        assert view_load in ALL_VIEW_OPS
        assert view_store in ALL_VIEW_OPS
        assert view_atomic_reduce in ALL_VIEW_OPS
        assert view_atomic_rmw in ALL_VIEW_OPS
        assert view_prefetch in ALL_VIEW_OPS

    def test_all_in_view_namespace(self) -> None:
        for op in ALL_VIEW_OPS:
            assert op.namespace in ("view", "view.atomic")
            assert op.group is view_ops
            assert op.doc
            assert op.format
            assert op.examples


class TestViewLoadStore:
    def test_load_shape(self) -> None:
        op = _ops()["view.load"]
        assert [operand.type_constraint for operand in op.operands] == [VIEW, INDEX]
        assert op.operands[1].variadic
        assert [result.type_constraint for result in op.results] == [SCALAR]
        static_indices = op.attr("static_indices")
        assert static_indices is not None
        assert static_indices.attr_type == ATTR_TYPE_I64_ARRAY
        assert len(op.effects) == 1
        assert op.effects[0].operand == "view"
        assert op.effects[0].kind is EffectKind.READ

    def test_store_shape(self) -> None:
        op = _ops()["view.store"]
        assert [operand.type_constraint for operand in op.operands] == [
            SCALAR,
            VIEW,
            INDEX,
        ]
        assert op.operands[2].variadic
        assert not op.results
        static_indices = op.attr("static_indices")
        assert static_indices is not None
        assert static_indices.attr_type == ATTR_TYPE_I64_ARRAY
        assert len(op.effects) == 1
        assert op.effects[0].operand == "view"
        assert op.effects[0].kind is EffectKind.WRITE


class TestViewAtomics:
    def test_atomic_attrs_are_shared_explicit_enums(self) -> None:
        assert [case.keyword for case in AtomicOrdering.cases] == [
            "relaxed",
            "acquire",
            "release",
            "acq_rel",
            "seq_cst",
        ]
        assert [case.keyword for case in AtomicScope.cases] == [
            "thread",
            "subgroup",
            "workgroup",
            "device",
            "system",
        ]
        assert [case.keyword for case in AtomicKind.cases] == [
            "xchgi",
            "xchgf",
            "addi",
            "addf",
            "subi",
            "andi",
            "ori",
            "xori",
            "minsi",
            "maxsi",
            "minui",
            "maxui",
            "minimumf",
            "maximumf",
            "minnumf",
            "maxnumf",
        ]

    def test_reduce_shape(self) -> None:
        op = _ops()["view.atomic.reduce"]
        assert [operand.type_constraint for operand in op.operands] == [
            SCALAR,
            VIEW,
            INDEX,
        ]
        assert op.operands[2].variadic
        assert not op.results
        assert op.attr("kind") is not None
        assert op.attr("ordering") is not None
        assert op.attr("scope") is not None
        static_indices = op.attr("static_indices")
        assert static_indices is not None
        assert static_indices.attr_type == ATTR_TYPE_I64_ARRAY
        assert len(op.effects) == 1
        assert op.effects[0].operand == "view"
        assert op.effects[0].kind is EffectKind.READWRITE

    def test_rmw_shape(self) -> None:
        op = _ops()["view.atomic.rmw"]
        assert [operand.type_constraint for operand in op.operands] == [
            SCALAR,
            VIEW,
            INDEX,
        ]
        assert op.operands[2].variadic
        assert [result.type_constraint for result in op.results] == [SCALAR]
        assert op.attr("kind") is not None
        assert op.attr("ordering") is not None
        assert op.attr("scope") is not None
        static_indices = op.attr("static_indices")
        assert static_indices is not None
        assert static_indices.attr_type == ATTR_TYPE_I64_ARRAY
        assert len(op.effects) == 1
        assert op.effects[0].operand == "view"
        assert op.effects[0].kind is EffectKind.READWRITE
