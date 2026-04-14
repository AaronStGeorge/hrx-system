# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the index dialect declarations."""

from loom.dialect.index import (
    ALL_INDEX_OPS,
    IndexCmpPredicate,
    index_ops,
)
from loom.dsl import ADDRESS, I1, INDEX, SCALAR, Op


def _ops() -> dict[str, Op]:
    return {op.name: op for op in ALL_INDEX_OPS}


class TestIndexDialect:
    def test_dialect_id(self) -> None:
        assert index_ops.dialect_id == 0x0F

    def test_inventory(self) -> None:
        assert [op.name for op in ALL_INDEX_OPS] == [
            "index.constant",
            "index.cast",
            "index.add",
            "index.sub",
            "index.mul",
            "index.madd",
            "index.cmpi",
            "index.select",
        ]

    def test_constant_is_address_only(self) -> None:
        op = _ops()["index.constant"]
        assert op.results[0].type_constraint == ADDRESS

    def test_cast_accepts_scalar_and_verifies_address_boundary(self) -> None:
        op = _ops()["index.cast"]
        assert op.operands[0].type_constraint == SCALAR
        assert op.results[0].type_constraint == SCALAR
        assert op.verify == "loom_index_cast_verify"

    def test_coordinate_arithmetic_is_index_typed(self) -> None:
        for name in ("index.add", "index.sub", "index.mul"):
            op = _ops()[name]
            assert op.operands[0].type_constraint == INDEX
            assert op.operands[1].type_constraint == INDEX
            assert op.results[0].type_constraint == INDEX

    def test_madd_is_index_typed(self) -> None:
        op = _ops()["index.madd"]
        assert [operand.type_constraint for operand in op.operands] == [
            INDEX,
            INDEX,
            INDEX,
        ]
        assert op.results[0].type_constraint == INDEX

    def test_cmpi_uses_index_predicates(self) -> None:
        op = _ops()["index.cmpi"]
        assert op.operands[0].type_constraint == INDEX
        assert op.operands[1].type_constraint == INDEX
        assert op.results[0].type_constraint == I1
        assert [case.keyword for case in IndexCmpPredicate.cases] == [
            "eq",
            "ne",
            "slt",
            "sle",
            "sgt",
            "sge",
            "ult",
            "ule",
            "ugt",
            "uge",
        ]

    def test_select_is_index_typed(self) -> None:
        op = _ops()["index.select"]
        assert op.operands[0].type_constraint == I1
        assert op.operands[1].type_constraint == INDEX
        assert op.operands[2].type_constraint == INDEX
        assert op.results[0].type_constraint == INDEX
