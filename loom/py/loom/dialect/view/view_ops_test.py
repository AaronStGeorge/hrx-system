# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the view dialect declarations."""

from loom.dialect.view import (
    ALL_VIEW_OPS,
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
            "view.prefetch",
        ]

    def test_public_exports_match_registry(self) -> None:
        assert view_subview in ALL_VIEW_OPS
        assert view_refine in ALL_VIEW_OPS
        assert view_load in ALL_VIEW_OPS
        assert view_store in ALL_VIEW_OPS
        assert view_prefetch in ALL_VIEW_OPS

    def test_all_in_view_namespace(self) -> None:
        for op in ALL_VIEW_OPS:
            assert op.namespace == "view"
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
