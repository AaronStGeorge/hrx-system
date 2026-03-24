# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the IR builder."""

import pytest

from loom.builder import IndexedValue, IRBuilder, TiedResultSpec, ValueRef, tied
from loom.builtin_types import ALL_BUILTIN_TYPES
from loom.dialect.test import ALL_TEST_OPS
from loom.dialect.test.builders import TestBuilders
from loom.ir import (
    F32,
    I32,
    INDEX,
    ShapedType,
    StaticDim,
    TypeKind,
)

# ============================================================================
# Helpers
# ============================================================================


def _builder() -> IRBuilder:
    b = IRBuilder()
    b.register_ops(ALL_TEST_OPS)
    b.register_types(ALL_BUILTIN_TYPES)
    return b


# ============================================================================
# ValueRef
# ============================================================================


class TestValueRef:
    def test_create(self) -> None:
        b = _builder()
        v = b.value("%x", F32)
        assert isinstance(v, ValueRef)
        assert v.id == 0
        assert v.name == "%x"
        assert v.type == F32

    def test_repr(self) -> None:
        b = _builder()
        v = b.value("%x", F32)
        assert "%x" in repr(v)

    def test_int_conversion(self) -> None:
        b = _builder()
        v = b.value("%x", F32)
        assert int(v) == 0


class TestIndexedValue:
    def test_single_static(self) -> None:
        b = _builder()
        tensor = b.value("%t", ShapedType(TypeKind.TENSOR, F32, (StaticDim(4),)))
        indexed = tensor[0]
        assert isinstance(indexed, IndexedValue)
        base, static, dynamic = indexed.decompose()
        assert base == tensor.id
        assert static == [0]
        assert dynamic == []

    def test_single_dynamic(self) -> None:
        b = _builder()
        tensor = b.value("%t", ShapedType(TypeKind.TENSOR, F32, (StaticDim(4),)))
        off = b.value("%off", INDEX)
        indexed = tensor[off]
        base, static, dynamic = indexed.decompose()
        assert base == tensor.id
        sentinel = -(2**63)
        assert static == [sentinel]
        assert dynamic == [off.id]

    def test_mixed(self) -> None:
        b = _builder()
        tile = b.value(
            "%t", ShapedType(TypeKind.TILE, F32, (StaticDim(64), StaticDim(64)))
        )
        off = b.value("%off", INDEX)
        indexed = tile[0, off]
        base, static, dynamic = indexed.decompose()
        sentinel = -(2**63)
        assert static == [0, sentinel]
        assert dynamic == [off.id]

    def test_invalid_index_type(self) -> None:
        b = _builder()
        t = b.value("%t", F32)
        indexed = t["bad"]
        with pytest.raises(TypeError, match="int.*or ValueRef"):
            indexed.decompose()


class TestTiedResultSpec:
    def test_tied_function(self) -> None:
        b = _builder()
        v = b.value("%x", F32)
        spec = tied(v, I32)
        assert isinstance(spec, TiedResultSpec)
        assert spec.operand_id == v.id
        assert spec.result_type == I32

    def test_as_type_method(self) -> None:
        b = _builder()
        v = b.value("%x", F32)
        spec = v.as_type(I32)
        assert isinstance(spec, TiedResultSpec)
        assert spec.operand_id == v.id
        assert spec.result_type == I32


# ============================================================================
# Generic build()
# ============================================================================


class TestBuild:
    def test_binary_op(self) -> None:
        b = _builder()
        a = b.value("%a", I32)
        c = b.value("%b", I32)
        result = b.build("test.addi", [a, c], results=[I32], result_names=["%r"])
        assert isinstance(result, ValueRef)
        assert result.name == "%r"
        assert result.type == I32

    def test_unary_op(self) -> None:
        b = _builder()
        x = b.value("%x", F32)
        result = b.build("test.neg", [x], results=[F32])
        assert isinstance(result, ValueRef)
        assert result.type == F32

    def test_constant(self) -> None:
        b = _builder()
        result = b.build(
            "test.constant",
            [],
            results=[I32],
            result_names=["%c42"],
            attributes={"value": 42},
        )
        assert isinstance(result, ValueRef)
        assert result.name == "%c42"

    def test_no_results(self) -> None:
        b = _builder()
        a = b.value("%a", F32)
        result = b.build("test.yield", [a])
        assert result is None

    def test_multi_result(self) -> None:
        tile_t = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
        b = _builder()
        w = b.value("%weights", tile_t)
        x = b.value("%input", INDEX)
        results = b.build(
            "test.invoke",
            [w, x],
            results=[tile_t, INDEX],
            result_names=["%out", "%count"],
            attributes={"callee": "compute"},
        )
        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0].name == "%out"
        assert results[1].name == "%count"

    def test_tied_result_with_spec(self) -> None:
        tile_t = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
        tensor_t = ShapedType(TypeKind.TENSOR, F32, (StaticDim(4),))
        b = _builder()
        tile = b.value("%tile", tile_t)
        tensor = b.value("%tensor", tensor_t)
        off = b.value("%off", INDEX)
        sentinel = -(2**63)
        result = b.build(
            "test.update",
            [tile, tensor, off],
            results=[tied(tensor, tensor_t)],
            attributes={"static_offsets": [sentinel]},
        )
        assert isinstance(result, ValueRef)

    def test_tied_with_as_type(self) -> None:
        """Use .as_type() method for tied result."""
        tile_t = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
        tensor_t = ShapedType(TypeKind.TENSOR, F32, (StaticDim(4),))
        b = _builder()
        tile = b.value("%tile", tile_t)
        tensor = b.value("%tensor", tensor_t)
        off = b.value("%off", INDEX)
        sentinel = -(2**63)
        result = b.build(
            "test.update",
            [tile, tensor, off],
            results=[tensor.as_type(tensor_t)],
            attributes={"static_offsets": [sentinel]},
        )
        assert isinstance(result, ValueRef)

    def test_call_with_mixed_results(self) -> None:
        """func.call-like: some results tied, some fresh."""
        tile_t = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
        b = _builder()
        weights = b.value("%weights", tile_t)
        input_val = b.value("%input", INDEX)
        results = b.build(
            "test.invoke",
            [weights, input_val],
            results=[tied(weights, tile_t), INDEX],
            result_names=["%out", "%count"],
            attributes={"callee": "compute"},
        )
        assert isinstance(results, list)
        assert len(results) == 2
        # First result tied to weights, second is fresh.
        assert results[0].type == tile_t
        assert results[1].type == INDEX

    def test_unknown_op_fails(self) -> None:
        b = _builder()
        with pytest.raises(ValueError, match="Unknown op"):
            b.build("nonexistent.op", [])

    def test_auto_named_results(self) -> None:
        b = _builder()
        a = b.value("%a", I32)
        c = b.value("%b", I32)
        result = b.build("test.addi", [a, c], results=[I32])
        assert isinstance(result, ValueRef)
        assert result.name == ""  # Unnamed, printer assigns at print time.


# ============================================================================
# Generated builder stubs
# ============================================================================


class TestGeneratedBuilders:
    """Test the generated TestBuilders class."""

    def _builders(self) -> tuple[IRBuilder, TestBuilders]:
        b = _builder()
        return b, TestBuilders(b)

    def test_addi(self) -> None:
        b, t = self._builders()
        a = b.value("%a", I32)
        c = b.value("%b", I32)
        result = t.addi(lhs=a, rhs=c, result_types=[I32])
        assert isinstance(result, ValueRef)
        assert result.type == I32

    def test_neg(self) -> None:
        b, t = self._builders()
        x = b.value("%x", F32)
        result = t.neg(input=x, result_types=[F32])
        assert result.type == F32

    def test_constant(self) -> None:
        b, t = self._builders()
        result = t.constant(value=42, result_types=[I32])
        assert result.type == I32

    def test_cmp(self) -> None:
        b, t = self._builders()
        a = b.value("%a", I32)
        c = b.value("%b", I32)
        result = t.cmp(predicate="lt", lhs=a, rhs=c, result_types=[I32])
        assert result.type == I32

    def test_yield(self) -> None:
        b, t = self._builders()
        a = b.value("%a", F32)
        c = b.value("%b", I32)
        t.yield_(values=[a, c])  # Returns None (void terminator).

    def test_invoke_with_tied(self) -> None:
        tile_t = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
        b, t = self._builders()
        w = b.value("%weights", tile_t)
        x = b.value("%input", INDEX)
        results = t.invoke(
            callee="compute", operands=[w, x], results=[tied(w, tile_t), INDEX]
        )
        assert isinstance(results, list)
        assert len(results) == 2

    def test_update_with_offsets(self) -> None:
        tile_t = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
        tensor_t = ShapedType(TypeKind.TENSOR, F32, (StaticDim(4),))
        b, t = self._builders()
        tile = b.value("%tile", tile_t)
        tensor = b.value("%tensor", tensor_t)
        off = b.value("%off", INDEX)
        result = t.update(
            source=tile,
            target=tensor,
            offsets=[0, off],
            results=[tied(tensor, tensor_t)],
        )
        assert isinstance(result, ValueRef)

    def test_func_optional_attrs(self) -> None:
        b, t = self._builders()
        # visibility, cc, and predicates are optional.
        t.func(callee="main", results=[F32])
        # Should not raise — optional attrs have defaults.

    def test_assume_with_predicates(self) -> None:
        from loom.ir import Predicate, PredicateArg

        b, t = self._builders()
        m = t.constant(value=42, result_types=[INDEX])
        predicates = [
            Predicate("mul", (PredicateArg("value", "%M"), PredicateArg("const", 16))),
        ]
        result = t.assume(values=[m], predicates=predicates, result_types=[INDEX])
        assert isinstance(result, ValueRef | list)

    def test_func_with_predicates(self) -> None:
        from loom.ir import Predicate, PredicateArg

        b, t = self._builders()
        predicates = [
            Predicate("mul", (PredicateArg("value", "%M"), PredicateArg("const", 16))),
        ]
        t.func(callee="constrained", results=[F32], predicates=predicates)
        # Should not raise — predicates are passed through.
