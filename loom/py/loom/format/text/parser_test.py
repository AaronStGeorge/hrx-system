# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the format-driven parser — name scope, type parsing, round-trip."""

from typing import Any

import pytest

from loom.assembly import TypeOf
from loom.builtin_types import ALL_BUILTIN_TYPES
from loom.dialect.func import ALL_FUNC_OPS
from loom.dialect.test import ALL_TEST_OPS
from loom.dsl import ANY, TypeDef, TypeParam
from loom.format.bytecode.reader import read_module
from loom.format.bytecode.writer import write_module
from loom.format.text.parser import (
    NameScope,
    Parser,
    parse_type_string,
)
from loom.format.text.printer import Printer, print_type
from loom.format.text.tokenizer import ParseError
from loom.ir import (
    BF16,
    ENCODING_TYPE,
    F32,
    I8,
    I32,
    I64,
    INDEX,
    CanonicalAttrDict,
    DialectType,
    DynamicDim,
    DynamicEncoding,
    EncodingInstance,
    EncodingType,
    FunctionType,
    GroupScope,
    GroupType,
    Module,
    Operation,
    ScalarType,
    ScalarTypeKind,
    ShapedType,
    StaticDim,
    Type,
    TypeKind,
    Value,
)

# ============================================================================
# Helpers
# ============================================================================


def _parse(text: str, **kwargs: Any) -> tuple[Type, dict[int, int]]:
    """Parse a type string, returning (type, dim_bindings)."""
    result: tuple[Type, dict[int, int]] = parse_type_string(text, **kwargs)
    return result


def _parse_type(text: str, **kwargs: Any) -> Type:
    """Parse a type string, returning just the type."""
    return _parse(text, **kwargs)[0]


# ============================================================================
# Name scope
# ============================================================================


class TestNameScope:
    def test_define_and_lookup(self) -> None:
        scope = NameScope()
        scope.define("x", 0)
        assert scope.lookup("x") == 0

    def test_multiple_definitions(self) -> None:
        scope = NameScope()
        scope.define("a", 0)
        scope.define("b", 1)
        scope.define("c", 2)
        assert scope.lookup("a") == 0
        assert scope.lookup("b") == 1
        assert scope.lookup("c") == 2

    def test_redefinition_fails(self) -> None:
        scope = NameScope()
        scope.define("x", 0)
        with pytest.raises(ValueError, match="already defined"):
            scope.define("x", 1)

    def test_undefined_fails(self) -> None:
        scope = NameScope()
        with pytest.raises(KeyError, match="undefined"):
            scope.lookup("missing")

    def test_child_sees_parent(self) -> None:
        parent = NameScope()
        parent.define("x", 0)
        child = parent.push()
        assert child.lookup("x") == 0

    def test_parent_does_not_see_child(self) -> None:
        parent = NameScope()
        child = parent.push()
        child.define("inner", 42)
        with pytest.raises(KeyError, match="undefined"):
            parent.lookup("inner")

    def test_child_shadows_parent(self) -> None:
        parent = NameScope()
        parent.define("x", 0)
        child = parent.push()
        child.define("x", 99)
        assert child.lookup("x") == 99
        assert parent.lookup("x") == 0

    def test_grandchild_sees_grandparent(self) -> None:
        root = NameScope()
        root.define("global", 0)
        child = root.push()
        grandchild = child.push()
        assert grandchild.lookup("global") == 0

    def test_sibling_scopes_isolated(self) -> None:
        parent = NameScope()
        child_a = parent.push()
        child_b = parent.push()
        child_a.define("only_in_a", 1)
        child_b.define("only_in_b", 2)
        with pytest.raises(KeyError):
            child_b.lookup("only_in_a")
        with pytest.raises(KeyError):
            child_a.lookup("only_in_b")


# ============================================================================
# Scalar types
# ============================================================================


class TestParseScalarTypes:
    def test_f32(self) -> None:
        assert _parse_type("f32") == F32

    def test_i32(self) -> None:
        assert _parse_type("i32") == I32

    def test_index(self) -> None:
        assert _parse_type("index") == INDEX

    def test_bf16(self) -> None:
        assert _parse_type("bf16") == BF16

    def test_i8(self) -> None:
        assert _parse_type("i8") == I8

    def test_i1(self) -> None:
        assert _parse_type("i1") == ScalarType(ScalarTypeKind.I1)

    def test_i64(self) -> None:
        assert _parse_type("i64") == I64

    def test_f16(self) -> None:
        assert _parse_type("f16") == ScalarType(ScalarTypeKind.F16)

    def test_f64(self) -> None:
        assert _parse_type("f64") == ScalarType(ScalarTypeKind.F64)

    def test_f8E4M3(self) -> None:
        assert _parse_type("f8E4M3") == ScalarType(ScalarTypeKind.F8E4M3)

    def test_f8E5M2(self) -> None:
        assert _parse_type("f8E5M2") == ScalarType(ScalarTypeKind.F8E5M2)

    def test_unknown_scalar_fails(self) -> None:
        with pytest.raises(ParseError, match="expected type"):
            _parse_type("f128")


# ============================================================================
# Shaped types — static dims
# ============================================================================


class TestParseShapedStatic:
    def test_tile_1d(self) -> None:
        result = _parse_type("tile<4xf32>")
        assert isinstance(result, ShapedType)
        assert result.type_kind == TypeKind.TILE
        assert result.element_type == F32
        assert result.rank == 1
        assert result.dims == (StaticDim(4),)

    def test_tile_2d(self) -> None:
        result = _parse_type("tile<4x4xf32>")
        assert isinstance(result, ShapedType)
        assert result.rank == 2
        assert result.dims == (StaticDim(4), StaticDim(4))

    def test_tensor_1d(self) -> None:
        result = _parse_type("tensor<256xi8>")
        assert isinstance(result, ShapedType)
        assert result.type_kind == TypeKind.TENSOR
        assert result.element_type == I8
        assert result.dims == (StaticDim(256),)

    def test_tile_0d(self) -> None:
        result = _parse_type("tile<f32>")
        assert isinstance(result, ShapedType)
        assert result.rank == 0
        assert result.element_type == F32

    def test_large_dims(self) -> None:
        result = _parse_type("tile<1024x2048xf16>")
        assert isinstance(result, ShapedType)
        assert result.dims == (StaticDim(1024), StaticDim(2048))

    def test_3d(self) -> None:
        result = _parse_type("tensor<2x3x4xi32>")
        assert isinstance(result, ShapedType)
        assert result.rank == 3
        assert result.dims == (StaticDim(2), StaticDim(3), StaticDim(4))


# ============================================================================
# Shaped types — dynamic dims
# ============================================================================


class TestParseDynamicDims:
    def test_single_dynamic(self) -> None:
        scope = NameScope()
        module = Module()
        dim_id = module.add_value(Value(name="M", type=INDEX))
        scope.define("M", dim_id)

        result, bindings = _parse("tile<[%M]x4xf32>", scope=scope, module=module)
        assert isinstance(result, ShapedType)
        assert result.dims == (DynamicDim(), StaticDim(4))
        assert bindings[0] == dim_id

    def test_all_dynamic(self) -> None:
        scope = NameScope()
        module = Module()
        m_id = module.add_value(Value(name="M", type=INDEX))
        k_id = module.add_value(Value(name="K", type=INDEX))
        scope.define("M", m_id)
        scope.define("K", k_id)

        result, bindings = _parse("tensor<[%M]x[%K]xf32>", scope=scope, module=module)
        assert isinstance(result, ShapedType)
        assert result.dims == (DynamicDim(), DynamicDim())
        assert bindings[0] == m_id
        assert bindings[1] == k_id

    def test_undefined_dim_fails(self) -> None:
        with pytest.raises(ParseError, match="undefined SSA value"):
            _parse("tile<[%M]xf32>")


# ============================================================================
# Shaped types — with encoding
# ============================================================================


class TestParseEncoding:
    def test_simple_encoding(self) -> None:
        result = _parse_type("tile<256xi8, #q8_0>")
        assert isinstance(result, ShapedType)
        assert result.has_encoding
        assert isinstance(result.encoding, EncodingInstance)
        assert result.encoding.name == "q8_0"

    def test_encoding_with_params(self) -> None:
        module = Module()
        result, _ = _parse("tile<256xi8, #q8_0<block=32>>", module=module)
        assert isinstance(result, ShapedType)
        assert result.has_encoding
        assert len(module.encodings) == 1
        assert module.encodings[0].name == "q8_0"
        assert module.encodings[0].params == (("block", "32"),)

    def test_encoding_dedup(self) -> None:
        module = Module()
        _parse("tile<128xi8, #q8_0<block=32>>", module=module)
        _parse("tile<256xi8, #q8_0<block=32>>", module=module)
        assert len(module.encodings) == 1  # Same encoding, deduplicated.

    def test_different_encodings(self) -> None:
        module = Module()
        _parse("tile<128xi8, #q8_0>", module=module)
        _parse("tile<256xi8, #q6_k>", module=module)
        assert len(module.encodings) == 2

    def test_malformed_encoding_params_gives_parse_error(self) -> None:
        """Malformed params raise ParseError, not bare ValueError."""
        from loom.format.text.tokenizer import ParseError as TokenError

        with pytest.raises((ParseError, TokenError), match="expected EQUALS"):
            _parse("tile<256xi8, #q8_0<block32>>")

    def test_encoding_multiple_params(self) -> None:
        module = Module()
        result, _ = _parse("tile<256xi8, #q8_0<block=32, group=128>>", module=module)
        assert isinstance(result, ShapedType)
        assert result.has_encoding
        enc = module.encodings[0]
        assert enc.params == (("block", "32"), ("group", "128"))


# ============================================================================
# Group types
# ============================================================================


class TestParseGroupType:
    def test_basic(self) -> None:
        result = _parse_type("group<workgroup>")
        assert isinstance(result, GroupType)
        assert result.scope == GroupScope.WORKGROUP

    def test_subgroup(self) -> None:
        result = _parse_type("group<subgroup>")
        assert isinstance(result, GroupType)
        assert result.scope == GroupScope.SUBGROUP


# ============================================================================
# Function types
# ============================================================================


class TestParseFunctionType:
    def test_basic(self) -> None:
        result = _parse_type("(f32, i32) -> (f32)")
        assert isinstance(result, FunctionType)
        assert result.arg_types == (F32, I32)
        assert result.result_types == (F32,)

    def test_empty_args(self) -> None:
        result = _parse_type("() -> (f32)")
        assert isinstance(result, FunctionType)
        assert result.arg_types == ()

    def test_empty_results(self) -> None:
        result = _parse_type("(f32) -> ()")
        assert isinstance(result, FunctionType)
        assert result.result_types == ()

    def test_multiple_results(self) -> None:
        result = _parse_type("(f32) -> (i32, f32)")
        assert isinstance(result, FunctionType)
        assert result.result_types == (I32, F32)


# ============================================================================
# Custom dialect types (via type registry)
# ============================================================================


class TestParseDialectTypes:
    def _registry(self) -> dict[str, TypeDef]:
        registry = {td.name: td for td in ALL_BUILTIN_TYPES}
        # Add custom types.
        registry["hal.buffer"] = TypeDef(name="hal.buffer")
        registry["hal.fence"] = TypeDef(name="hal.fence")
        registry["vm.ref"] = TypeDef(
            name="vm.ref",
            params=[TypeParam("object", ANY)],
            format=[TypeOf("object")],
        )
        return registry

    def test_opaque_type(self) -> None:
        result = _parse_type("hal.buffer", type_registry=self._registry())
        assert isinstance(result, DialectType)
        assert result.name == "hal.buffer"
        assert result.params == ()

    def test_parameterized_type(self) -> None:
        result = _parse_type("vm.ref<hal.buffer>", type_registry=self._registry())
        assert isinstance(result, DialectType)
        assert result.name == "vm.ref"
        assert len(result.params) == 1
        assert isinstance(result.params[0], DialectType)
        assert result.params[0].name == "hal.buffer"

    def test_nested_parameterized(self) -> None:
        from loom.assembly import TypeOf
        from loom.dsl import ANY, TypeDef, TypeParam

        registry = self._registry()
        registry["vm.list"] = TypeDef(
            name="vm.list",
            params=[TypeParam("element", ANY)],
            format=[TypeOf("element")],
        )
        result = _parse_type("vm.ref<vm.list<i32>>", type_registry=registry)
        assert isinstance(result, DialectType)
        assert result.name == "vm.ref"
        inner = result.params[0]
        assert isinstance(inner, DialectType)
        assert inner.name == "vm.list"
        assert inner.params[0] == I32

    def test_unknown_type_fails(self) -> None:
        with pytest.raises(ParseError, match="expected type"):
            _parse_type("hal.unknown", type_registry=self._registry())


# ============================================================================
# Type round-trip: print(parse(text)) == text
# ============================================================================


class TestTypeRoundTrip:
    def _roundtrip(self, text: str, **kwargs: Any) -> None:
        """Parse then print, assert identical."""
        parsed_type, _ = _parse(text, **kwargs)
        printed = print_type(parsed_type)
        assert printed == text, f"Round-trip failed: {text!r} -> {printed!r}"

    def test_scalar_roundtrip(self) -> None:
        for name in ["f32", "i32", "index", "bf16", "i8", "i64", "f16", "f64"]:
            self._roundtrip(name)

    def test_shaped_roundtrip(self) -> None:
        self._roundtrip("tile<4xf32>")
        self._roundtrip("tile<4x4xf32>")
        self._roundtrip("tensor<256xi8>")
        self._roundtrip("tile<f32>")

    def test_group_roundtrip(self) -> None:
        self._roundtrip("group<workgroup>")

    def test_dialect_opaque_roundtrip(self) -> None:
        registry = {td.name: td for td in ALL_BUILTIN_TYPES}
        registry["hal.buffer"] = TypeDef(name="hal.buffer")
        self._roundtrip("hal.buffer", type_registry=registry)

    def test_dialect_parameterized_roundtrip(self) -> None:
        registry = {td.name: td for td in ALL_BUILTIN_TYPES}
        registry["hal.buffer"] = TypeDef(name="hal.buffer")
        registry["vm.ref"] = TypeDef(
            name="vm.ref",
            params=[TypeParam("object", ANY)],
            format=[TypeOf("object")],
        )
        self._roundtrip("vm.ref<hal.buffer>", type_registry=registry)

    def test_dialect_nested_roundtrip(self) -> None:
        registry = {td.name: td for td in ALL_BUILTIN_TYPES}
        registry["vm.list"] = TypeDef(
            name="vm.list",
            params=[TypeParam("element", ANY)],
            format=[TypeOf("element")],
        )
        registry["vm.ref"] = TypeDef(
            name="vm.ref",
            params=[TypeParam("object", ANY)],
            format=[TypeOf("object")],
        )
        self._roundtrip("vm.ref<vm.list<i32>>", type_registry=registry)


# ============================================================================
# Op parsing — simple patterns
# ============================================================================


def _op_parser() -> Parser:
    """Create a parser with test + func ops and built-in types."""
    parser = Parser()
    parser.register_ops(ALL_TEST_OPS)
    parser.register_ops(ALL_FUNC_OPS)
    parser.register_types(ALL_BUILTIN_TYPES)
    return parser


def _op_printer(**kwargs: Any) -> Printer:
    """Create a printer with test + func ops and built-in types."""
    printer = Printer(**kwargs)
    printer.register_ops(ALL_TEST_OPS)
    printer.register_ops(ALL_FUNC_OPS)
    printer.register_types(ALL_BUILTIN_TYPES)
    return printer


def _parse_op(
    text: str, module: Module | None = None, scope: NameScope | None = None
) -> Operation:
    """Parse a single op from text."""

    parser = _op_parser()
    return parser.parse_operation_from_text(text, module=module, scope=scope)


def _setup_scope(*names_and_types: tuple[str, Type]) -> tuple[Module, NameScope]:
    """Create module + scope with pre-defined values."""
    module = Module()
    scope = NameScope()
    for name, value_type in names_and_types:
        vid = module.add_value(Value(name=name, type=value_type))
        scope.define(name, vid)
    return module, scope


class TestParseBinaryOp:
    def test_addi(self) -> None:
        module, scope = _setup_scope(("a", I32), ("b", I32))
        op = _parse_op("%r = test.addi %a, %b : i32", module=module, scope=scope)
        assert op.name == "test.addi"
        assert len(op.operands) == 2
        assert len(op.results) == 1
        assert module.values[op.results[0]].name == "r"
        assert module.values[op.results[0]].type == I32


class TestParseUnaryOp:
    def test_neg(self) -> None:
        module, scope = _setup_scope(("x", F32))
        op = _parse_op("%r = test.neg %x : f32", module=module, scope=scope)
        assert op.name == "test.neg"
        assert len(op.operands) == 1
        assert module.values[op.results[0]].type == F32


class TestParseCastOp:
    def test_cast(self) -> None:
        module, scope = _setup_scope(("x", I32))
        op = _parse_op("%r = test.cast %x : i32 to f32", module=module, scope=scope)
        assert op.name == "test.cast"
        assert module.values[op.results[0]].type == F32


class TestParseConvertOp:
    def test_convert_scalar(self) -> None:
        module, scope = _setup_scope(("x", I32))
        op = _parse_op("%r = test.convert %x : i32 -> f32", module=module, scope=scope)
        assert op.name == "test.convert"
        assert module.values[op.results[0]].type == F32

    def test_convert_tile(self) -> None:
        tile_i8 = ShapedType(
            TypeKind.TILE, ScalarType(ScalarTypeKind.I8), (StaticDim(4),)
        )
        module, scope = _setup_scope(("x", tile_i8))
        op = _parse_op(
            "%r = test.convert %x : tile<4xi8> -> tile<4xf32>",
            module=module,
            scope=scope,
        )
        assert op.name == "test.convert"
        result_type = module.values[op.results[0]].type
        assert isinstance(result_type, ShapedType)
        assert result_type.element_type == F32


class TestParseConstantOp:
    def test_integer(self) -> None:
        op = _parse_op("%c42 = test.constant 42 : i32")
        assert op.name == "test.constant"
        assert op.attributes["value"] == 42

    def test_float(self) -> None:
        op = _parse_op("%pi = test.constant 3.14 : f32")
        assert abs(op.attributes["value"] - 3.14) < 1e-10


class TestParseComparisonOp:
    def test_cmp(self) -> None:
        module, scope = _setup_scope(("a", I32), ("b", I32))
        op = _parse_op("%r = test.cmp lt, %a, %b : i32", module=module, scope=scope)
        assert op.attributes["predicate"] == "lt"


class TestParseYieldOp:
    def test_single(self) -> None:
        module, scope = _setup_scope(("a", F32))
        op = _parse_op("test.yield %a : f32", module=module, scope=scope)
        assert op.name == "test.yield"
        assert len(op.operands) == 1
        assert len(op.results) == 0

    def test_multiple(self) -> None:
        module, scope = _setup_scope(("a", F32), ("b", I32))
        op = _parse_op("test.yield %a, %b : f32, i32", module=module, scope=scope)
        assert len(op.operands) == 2


class TestParseAttrDictOp:
    def test_with_attrs(self) -> None:
        module, scope = _setup_scope(("x", F32))
        op = _parse_op(
            '%r = test.attrs %x {axis = 0, label = "foo"} : f32',
            module=module,
            scope=scope,
        )
        assert "dict" in op.attributes
        d = op.attributes["dict"]
        assert isinstance(d, CanonicalAttrDict)
        assert list(d.items()) == [("axis", 0), ("label", "foo")]
        assert d["axis"] == 0
        assert d["label"] == "foo"

    def test_empty_dict(self) -> None:
        module, scope = _setup_scope(("x", F32))
        op = _parse_op(
            "%r = test.attrs %x : f32",
            module=module,
            scope=scope,
        )
        assert "dict" not in op.attributes or op.attributes.get("dict") in (
            None,
            {},
        )

    def test_single_entry(self) -> None:
        module, scope = _setup_scope(("x", F32))
        op = _parse_op(
            "%r = test.attrs %x {count = 42} : f32",
            module=module,
            scope=scope,
        )
        d = op.attributes["dict"]
        assert d["count"] == 42

    def test_bool_values(self) -> None:
        module, scope = _setup_scope(("x", F32))
        op = _parse_op(
            "%r = test.attrs %x {enabled = true, debug = false} : f32",
            module=module,
            scope=scope,
        )
        d = op.attributes["dict"]
        assert d["enabled"] is True
        assert d["debug"] is False

    def test_round_trip(self) -> None:
        """Parse → print → parse produces the same dict entries."""
        import loom.dialect.test.defs as td
        from loom.dsl import Op as DslOp

        ops = [v for v in vars(td).values() if isinstance(v, DslOp)]
        module, scope = _setup_scope(("x", F32))
        op = _parse_op(
            '%r = test.attrs %x {axis = 0, label = "hello"} : f32',
            module=module,
            scope=scope,
        )
        p = Printer()
        p.register_ops(ops)
        text = p.print_operation(op, module)
        assert "axis = 0" in text
        assert 'label = "hello"' in text

    def test_unsorted_input_is_canonicalized(self) -> None:
        module, scope = _setup_scope(("x", F32))
        op = _parse_op(
            '%r = test.attrs %x {label = "foo", axis = 0} : f32',
            module=module,
            scope=scope,
        )
        d = op.attributes["dict"]
        assert isinstance(d, CanonicalAttrDict)
        assert list(d.items()) == [("axis", 0), ("label", "foo")]

    def test_duplicate_key_is_rejected(self) -> None:
        module, scope = _setup_scope(("x", F32))
        with pytest.raises(ParseError, match="duplicate attribute dict key 'axis'"):
            _parse_op(
                "%r = test.attrs %x {axis = 0, axis = 1} : f32",
                module=module,
                scope=scope,
            )


class TestParseSliceOp:
    def test_static_offsets(self) -> None:
        tile_type = ShapedType(
            TypeKind.TILE,
            ScalarType(ScalarTypeKind.F16),
            (StaticDim(64), StaticDim(64)),
        )
        module, scope = _setup_scope(("src", tile_type))
        op = _parse_op(
            "%r = test.slice %src[0, 0] : tile<64x64xf16> -> (tile<16x16xf16>)",
            module=module,
            scope=scope,
        )
        assert op.attributes["static_offsets"] == [0, 0]

    def test_dynamic_offsets(self) -> None:
        tile_type = ShapedType(
            TypeKind.TILE,
            ScalarType(ScalarTypeKind.F16),
            (StaticDim(64), StaticDim(64)),
        )
        module, scope = _setup_scope(("src", tile_type), ("off", INDEX))
        sentinel = -(2**63)
        op = _parse_op(
            "%r = test.slice %src[0, %off] : tile<64x64xf16> -> (tile<16x16xf16>)",
            module=module,
            scope=scope,
        )
        assert op.attributes["static_offsets"] == [0, sentinel]


class TestParseTiedResult:
    def test_update(self) -> None:
        tile_t = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
        tensor_t = ShapedType(TypeKind.TENSOR, F32, (StaticDim(4),))
        module, scope = _setup_scope(
            ("tile", tile_t), ("tensor", tensor_t), ("off", INDEX)
        )
        op = _parse_op(
            "%r = test.update %tile, %tensor[%off] : tile<4xf32> -> (%tensor as tensor<4xf32>)",
            module=module,
            scope=scope,
        )
        assert len(op.tied_results) == 1
        assert op.tied_results[0].operand_index == 1  # %tensor


class TestParseInvokeOp:
    def test_with_tie(self) -> None:
        tile_t = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
        module, scope = _setup_scope(("weights", tile_t), ("input", INDEX))
        op = _parse_op(
            "%output, %count = test.invoke @compute(%weights, %input)"
            " : (tile<4xf32>, index) -> (%weights as tile<4xf32>, index)",
            module=module,
            scope=scope,
        )
        assert op.name == "test.invoke"
        assert op.attributes["callee"] == "compute"
        assert len(op.results) == 2
        assert len(op.tied_results) == 1
        assert op.tied_results[0].operand_index == 0


# ============================================================================
# Module-level parsing
# ============================================================================


class TestParseModule:
    def _parse_module(self, text: str) -> Module:
        return _op_parser().parse(text)

    def test_empty_module(self) -> None:
        module = self._parse_module("")
        assert len(module.symbols) == 0

    def test_simple_function(self) -> None:
        module = self._parse_module(
            "func.def @negate(%input: f32) -> (f32) {\n"
            "  %r = test.neg %input : f32\n"
            "  test.yield %r : f32\n"
            "}\n"
        )
        assert len(module.symbols) == 1
        op = module.symbols[0].op
        assert op is not None
        assert op.attributes.get("callee") == "negate"
        assert len(op.regions[0].blocks[0].arg_ids) == 1
        assert len(op.results) == 1
        assert op.regions
        assert len(op.regions[0].blocks) == 1
        assert len(op.regions[0].blocks[0].ops) == 2

    def test_public_declaration(self) -> None:
        module = self._parse_module("func.decl public @exported(%a: i32) -> (i32)\n")
        assert len(module.symbols) == 1
        op = module.symbols[0].op
        assert op is not None
        assert op.attributes.get("visibility") == "public"
        assert not op.regions  # Declarations have no body region.

    def test_multiple_functions(self) -> None:
        module = self._parse_module(
            "func.def @f1(%a: f32) -> (f32) {\n"
            "  test.yield %a : f32\n"
            "}\n"
            "\n"
            "func.def @f2(%b: i32) -> (i32) {\n"
            "  test.yield %b : i32\n"
            "}\n"
        )
        assert len(module.symbols) == 2
        op0 = module.symbols[0].op
        op1 = module.symbols[1].op
        assert op0 is not None
        assert op1 is not None
        assert op0.attributes.get("callee") == "f1"
        assert op1.attributes.get("callee") == "f2"


class TestParseEncodingAlias:
    def _parse_module(self, text: str) -> Module:
        return _op_parser().parse(text)

    def test_alias_definition(self) -> None:
        module = self._parse_module(
            "#enc = #q8_0<block=32>\n"
            "func.def @f(%a: tile<256xi8, #enc>) -> (tile<256xi8, #enc>) {\n"
            "  test.yield %a : tile<256xi8, #enc>\n"
            "}\n"
        )
        assert len(module.symbols) == 1
        assert len(module.encodings) >= 1
        enc = module.encodings[0]
        assert enc.name == "q8_0"
        assert enc.params == (("block", "32"),)

    def test_alias_without_params(self) -> None:
        module = self._parse_module(
            "#w = #q6_k\n"
            "func.def @f(%a: tile<256xi8, #w>) -> (tile<256xi8, #w>) {\n"
            "  test.yield %a : tile<256xi8, #w>\n"
            "}\n"
        )
        enc = module.encodings[0]
        assert enc.name == "q6_k"
        assert enc.params == ()


class TestEncodingValidation:
    def test_known_encoding_accepted(self) -> None:
        parser = _op_parser()
        parser.register_encodings(["q8_0", "q6_k"])
        # Should not raise.
        parser.parse(
            "func.def @f(%a: tile<256xi8, #q8_0>) -> (tile<256xi8, #q8_0>) {\n"
            "  test.yield %a : tile<256xi8, #q8_0>\n"
            "}\n"
        )

    def test_unknown_encoding_rejected(self) -> None:
        parser = _op_parser()
        parser.register_encodings(["q8_0"])
        with pytest.raises(ParseError, match="unknown encoding"):
            parser.parse(
                "func.def @f(%a: tile<256xi8, #q999>) -> (tile<256xi8, #q999>) {\n"
                "  test.yield %a : tile<256xi8, #q999>\n"
                "}\n"
            )


# ============================================================================
# Round-trip tests: print → parse → print
# ============================================================================


class TestRoundTrip:
    """The gold standard: print(parse(print(construct()))) == print(construct())."""

    def _roundtrip_text(self, text: str) -> None:
        """Parse text, print, assert identical."""
        module = _op_parser().parse(text)
        printed = _op_printer().print_module(module)
        assert printed == text, (
            f"Round-trip failed.\nInput:\n{text}\nOutput:\n{printed}"
        )

    def test_simple_function(self) -> None:
        self._roundtrip_text(
            "func.def @negate(%input: f32) -> (f32) {\n"
            "  %neg0 = test.neg %input : f32\n"
            "  test.yield %neg0 : f32\n"
            "}\n"
        )

    def test_convert_bare_result_type(self) -> None:
        """test.convert uses ResultType (bare, no parens) and round-trips."""
        self._roundtrip_text(
            "func.def @convert(%x: i32) -> (f32) {\n"
            "  %r = test.convert %x : i32 -> f32\n"
            "  test.yield %r : f32\n"
            "}\n"
        )

    def test_convert_tile_result_type(self) -> None:
        """test.convert with tile types round-trips with bare result."""
        self._roundtrip_text(
            "func.def @convert_tile(%x: tile<4xi8>) -> (tile<4xf32>) {\n"
            "  %r = test.convert %x : tile<4xi8> -> tile<4xf32>\n"
            "  test.yield %r : tile<4xf32>\n"
            "}\n"
        )


# ============================================================================
# Op parsing — region-containing ops
# ============================================================================


class TestParseMapOp:
    def test_elementwise(self) -> None:
        tile_t = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
        module, scope = _setup_scope(("x", tile_t))
        op = _parse_op(
            "%r = test.map(%e = %x : tile<4xf32>) {\n"
            "  %v = test.neg %e : f32\n"
            "  test.yield %v : f32\n"
            "} -> (tile<4xf32>)",
            module=module,
            scope=scope,
        )
        assert op.name == "test.map"
        assert len(op.operands) == 1  # %x
        assert len(op.regions) == 1
        body = op.regions[0]
        assert len(body.blocks) == 1
        assert len(body.blocks[0].ops) == 2  # neg + yield

    def test_binding_name_can_shadow_outer_scope_name(self) -> None:
        tile_t = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
        module = _op_parser().parse(
            "func.def @shadow(%x: f32, %tile: tile<4xf32>) -> (f32) {\n"
            "  %mapped = test.map(%x = %tile : tile<4xf32>) {\n"
            "    test.yield %x : f32\n"
            "  } -> (f32)\n"
            "  %negated = test.neg %x : f32\n"
            "  test.yield %negated : f32\n"
            "}\n"
        )
        func_op = module.symbols[0].op
        assert func_op is not None
        func_entry = func_op.regions[0].blocks[0]
        map_op = func_entry.ops[0]
        map_entry = map_op.regions[0].blocks[0]
        assert map_entry.arg_ids[0] != func_entry.arg_ids[0]
        assert map_entry.ops[0].operands[0] == map_entry.arg_ids[0]
        assert func_entry.ops[1].operands[0] == func_entry.arg_ids[0]
        assert module.values[func_entry.arg_ids[1]].type == tile_t

    def test_binding_name_is_region_local(self) -> None:
        with pytest.raises(ParseError, match="undefined SSA value '%element'"):
            _op_parser().parse(
                "func.def @leak(%tile: tile<4xf32>) -> (f32) {\n"
                "  %mapped = test.map(%element = %tile : tile<4xf32>) {\n"
                "    test.yield %element : f32\n"
                "  } -> (f32)\n"
                "  %bad = test.neg %element : f32\n"
                "  test.yield %bad : f32\n"
                "}\n"
            )


class TestParseLoopOp:
    def test_with_iter_args(self) -> None:
        module, scope = _setup_scope(
            ("c0", INDEX), ("n", INDEX), ("c1", INDEX), ("init", F32)
        )
        op = _parse_op(
            "%r = test.loop %i = %c0 to %n step %c1"
            " iter_args(%acc = %init : f32) -> (%init as f32) {\n"
            "  test.yield %acc : f32\n"
            "}",
            module=module,
            scope=scope,
        )
        assert op.name == "test.loop"
        assert len(op.operands) == 4  # lb, ub, step + %init iter_arg
        assert len(op.regions) == 1
        assert len(op.results) == 1
        assert len(op.tied_results) == 1
        assert op.tied_results[0].result_index == 0
        assert op.tied_results[0].operand_index == 3
        assert len(op.regions[0].blocks[0].arg_ids) == 2
        assert (
            op.regions[0].blocks[0].ops[0].operands[0]
            == (op.regions[0].blocks[0].arg_ids[1])
        )

    def test_iter_arg_name_is_not_tied_result_target(self) -> None:
        module, scope = _setup_scope(
            ("c0", INDEX), ("n", INDEX), ("c1", INDEX), ("init", F32)
        )
        with pytest.raises(
            ParseError, match="tied result 'acc' not found in args or operands"
        ):
            _parse_op(
                "%r = test.loop %i = %c0 to %n step %c1"
                " iter_args(%acc = %init : f32) -> (%acc as f32) {\n"
                "  test.yield %acc : f32\n"
                "}",
                module=module,
                scope=scope,
            )

    def test_iv_name_is_not_tied_result_target(self) -> None:
        module, scope = _setup_scope(
            ("c0", INDEX), ("n", INDEX), ("c1", INDEX), ("init", F32)
        )
        with pytest.raises(
            ParseError, match="tied result 'i' not found in args or operands"
        ):
            _parse_op(
                "%r = test.loop %i = %c0 to %n step %c1"
                " iter_args(%acc = %init : f32) -> (%i as f32) {\n"
                "  test.yield %acc : f32\n"
                "}",
                module=module,
                scope=scope,
            )

    def test_iv_name_is_region_local(self) -> None:
        with pytest.raises(ParseError, match="undefined SSA value '%i'"):
            _op_parser().parse(
                "func.def @loop(%lo: index, %hi: index, %step: index) {\n"
                "  test.loop %i = %lo to %hi step %step {\n"
                "  }\n"
                "  test.loop %again = %i to %hi step %step {\n"
                "  }\n"
                "}\n"
            )


class TestParseBranchOp:
    def test_if_else(self) -> None:
        module, scope = _setup_scope(("cond", I32), ("a", F32), ("b", F32))
        op = _parse_op(
            "%r = test.branch %cond -> (f32) {\n"
            "  test.yield %a : f32\n"
            "} else {\n"
            "  test.yield %b : f32\n"
            "}",
            module=module,
            scope=scope,
        )
        assert op.name == "test.branch"
        assert len(op.regions) == 2  # then + else
        assert len(op.regions[0].blocks[0].ops) == 1  # yield in then
        assert len(op.regions[1].blocks[0].ops) == 1  # yield in else


# ============================================================================
# Encoding type and dynamic encoding parsing
# ============================================================================


class TestParseEncodingType:
    def test_encoding_keyword(self) -> None:
        """'encoding' parses to EncodingType."""
        result_type = _parse_type("encoding")
        assert isinstance(result_type, EncodingType)
        assert result_type == ENCODING_TYPE


class TestParseDynamicEncoding:
    def test_tile_with_ssa_encoding(self) -> None:
        """tile<4xf32, %enc> parses to ShapedType with DynamicEncoding."""
        module = Module()
        scope = NameScope()
        enc_id = module.add_value(Value(name="enc", type=ENCODING_TYPE))
        scope.define("enc", enc_id)
        shaped_type, bindings = _parse("tile<4xf32, %enc>", scope=scope, module=module)
        assert isinstance(shaped_type, ShapedType)
        assert shaped_type.type_kind == TypeKind.TILE
        assert shaped_type.element_type == F32
        assert shaped_type.dims == (StaticDim(4),)
        assert isinstance(shaped_type.encoding, DynamicEncoding)
        # Encoding binding uses sentinel key -1.
        assert bindings[-1] == enc_id

    def test_static_encoding_still_works(self) -> None:
        """tile<4xf32, #q8_0> still parses as static encoding."""
        shaped_type = _parse_type("tile<4xf32, #q8_0>")
        assert isinstance(shaped_type, ShapedType)
        assert isinstance(shaped_type.encoding, EncodingInstance)
        assert shaped_type.encoding.name == "q8_0"

    def test_tile_without_encoding_still_works(self) -> None:
        """tile<4xf32> still parses with no encoding."""
        shaped_type = _parse_type("tile<4xf32>")
        assert isinstance(shaped_type, ShapedType)
        assert shaped_type.encoding is None

    def test_undefined_encoding_value_errors(self) -> None:
        """Referencing undefined %enc in encoding position errors."""
        from loom.format.text.tokenizer import ParseError as TokParseError

        with pytest.raises((ParseError, TokParseError, KeyError)):
            _parse("tile<4xf32, %undefined>")

    def test_non_encoding_typed_value_errors(self) -> None:
        """Referencing a non-encoding-typed value in encoding position errors."""
        module = Module()
        scope = NameScope()
        # %x is i32, not encoding.
        x_id = module.add_value(Value(name="x", type=I32))
        scope.define("x", x_id)
        with pytest.raises((ParseError, ValueError)):
            _parse("tile<4xf32, %x>", scope=scope, module=module)


# ============================================================================
# Round-trip: encoding type in function signatures
# ============================================================================


# ============================================================================
# Pool type parsing
# ============================================================================


class TestParsePoolType:
    def test_static_pool(self) -> None:
        """pool<65536> parses to PoolType with static block size."""
        from loom.ir import PoolType

        pool_type = _parse_type("pool<65536>")
        assert isinstance(pool_type, PoolType)
        assert pool_type.block_size == StaticDim(65536)

    def test_dynamic_pool(self) -> None:
        """pool<[%BS]> parses to PoolType with dynamic block size."""
        from loom.ir import PoolType

        module = Module()
        scope = NameScope()
        bs_id = module.add_value(Value(name="BS", type=INDEX))
        scope.define("BS", bs_id)
        pool_type, bindings = _parse("pool<[%BS]>", scope=scope, module=module)
        assert isinstance(pool_type, PoolType)
        assert pool_type.has_dynamic_block_size
        assert bindings[0] == bs_id

    def test_pool_roundtrip(self) -> None:
        """pool<[%BS]> round-trips through parse → print in a function."""
        text = (
            "func.def @use_pool(%BS: index, %pool: pool<[%BS]>) -> (pool<[%BS]>) {\n"
            "  test.yield %pool : pool<[%BS]>\n"
            "}\n"
        )
        module = _op_parser().parse(text)
        printed = _op_printer().print_module(module)
        assert printed == text, (
            f"Round-trip failed.\nInput:\n{text}\nOutput:\n{printed}"
        )


# ============================================================================
# Predicate parsing and round-trip
# ============================================================================


class TestParsePredicates:
    """Tests for parsing where-clause predicates on functions."""

    def _parse_module(self, text: str) -> Module:
        return _op_parser().parse(text)

    def test_single_mul_predicate(self) -> None:
        """Parse a function with where [mul(%M, 16)]."""

        module = self._parse_module(
            "func.decl @f(%M: index, %a: tensor<[%M]xf32>) -> (tensor<[%M]xf32>)"
            " where [mul(%M, 16)]\n"
        )
        op = module.symbols[0].op
        assert op is not None
        predicates = op.attributes.get("predicates", [])
        assert len(predicates) == 1
        pred = predicates[0]
        assert pred.kind == "mul"
        assert len(pred.args) == 2
        assert pred.args[0].tag == "value"
        assert pred.args[0].value == "M"
        assert pred.args[1].tag == "const"
        assert pred.args[1].value == 16

    def test_multiple_predicates(self) -> None:
        """Parse a function with multiple predicates."""

        module = self._parse_module(
            "func.decl @f(%M: index, %K: index, %a: tensor<[%M]x[%K]xf32>) -> (tensor<[%M]xf32>)"
            " where [mul(%M, 16), lt(%K, 1024), range(%M, 32, 512)]\n"
        )
        op = module.symbols[0].op
        assert op is not None
        predicates = op.attributes.get("predicates", [])
        assert len(predicates) == 3
        assert predicates[0].kind == "mul"
        assert predicates[1].kind == "lt"
        assert predicates[2].kind == "range"
        # range has 3 args.
        range_pred = predicates[2]
        assert len(range_pred.args) == 3
        assert range_pred.args[0].value == "M"
        assert range_pred.args[1].value == 32
        assert range_pred.args[2].value == 512

    def test_pow2_single_arg(self) -> None:
        """Parse pow2(%N) — single-argument predicate."""

        module = self._parse_module(
            "func.decl @f(%N: index, %a: tensor<[%N]xf32>) -> (tensor<[%N]xf32>)"
            " where [pow2(%N)]\n"
        )
        op = module.symbols[0].op
        assert op is not None
        predicates = op.attributes.get("predicates", [])
        assert len(predicates) == 1
        assert predicates[0].kind == "pow2"
        assert len(predicates[0].args) == 1

    def test_named_result_predicate(self) -> None:
        """Parse eq(%idx, %M) — result name argument."""

        module = self._parse_module(
            "func.decl @f(%M: index, %a: tensor<[%M]xf32>) -> (tensor<[%M]xf32>, %idx: index)"
            " where [eq(%idx, %M)]\n"
        )
        op = module.symbols[0].op
        assert op is not None
        predicates = op.attributes.get("predicates", [])
        assert len(predicates) == 1
        pred = predicates[0]
        assert pred.kind == "eq"
        assert pred.args[0].tag == "value"
        assert pred.args[0].value == "idx"
        assert pred.args[1].tag == "value"
        assert pred.args[1].value == "M"

    def test_no_predicates(self) -> None:
        """Functions without where clause have empty predicates list."""
        module = self._parse_module(
            "func.def @f(%a: f32) -> (f32) {\n  test.yield %a : f32\n}\n"
        )
        op = module.symbols[0].op
        assert op is not None
        assert op.attributes.get("predicates", []) == []

    def test_predicates_with_body(self) -> None:
        """Where clause followed by function body."""
        module = self._parse_module(
            "func.def @f(%M: index, %a: tensor<[%M]xf32>) -> (tensor<[%M]xf32>)"
            " where [mul(%M, 16)] {\n"
            "  test.yield %a : tensor<[%M]xf32>\n"
            "}\n"
        )
        op = module.symbols[0].op
        assert op is not None
        predicates = op.attributes.get("predicates", [])
        assert len(predicates) == 1
        assert op.regions


class TestPredicateRoundTrip:
    """Round-trip tests: print(parse(text)) == text."""

    def _roundtrip_text(self, text: str) -> None:
        """Parse text, print, assert identical."""
        module = _op_parser().parse(text)
        printed = _op_printer().print_module(module)

        assert printed == text, (
            f"Round-trip failed.\nInput:\n{text}\nOutput:\n{printed}"
        )

    def test_single_predicate_declaration(self) -> None:
        """Declarations must round-trip dim names and predicates."""
        self._roundtrip_text(
            "func.decl @f(%M: index, %a: tensor<[%M]xf32>) -> (tensor<[%M]xf32>)"
            " where [mul(%M, 16)]\n"
        )

    def test_multiple_predicates_declaration(self) -> None:
        self._roundtrip_text(
            "func.decl @f(%M: index, %K: index, %a: tensor<[%M]x[%K]xf32>) -> (tensor<[%M]xf32>)"
            " where [mul(%M, 16), lt(%K, 1024), range(%M, 32, 512)]\n"
        )

    def test_pow2_predicate(self) -> None:
        self._roundtrip_text(
            "func.decl @f(%N: index, %a: tensor<[%N]xf32>) -> (tensor<[%N]xf32>)"
            " where [pow2(%N)]\n"
        )

    def test_result_name_predicate(self) -> None:
        self._roundtrip_text(
            "func.decl @f(%M: index, %a: tensor<[%M]xf32>) -> (tensor<[%M]xf32>, %idx: index)"
            " where [eq(%idx, %M)]\n"
        )

    def test_predicates_with_body(self) -> None:
        self._roundtrip_text(
            "func.def @f(%M: index, %a: tensor<[%M]xf32>) -> (tensor<[%M]xf32>)"
            " where [mul(%M, 16)] {\n"
            "  test.yield %a : tensor<[%M]xf32>\n"
            "}\n"
        )

    def test_public_device_with_predicates(self) -> None:
        self._roundtrip_text(
            "func.decl public device @vnni(%M: index, %K: index, %w: tensor<[%M]x[%K]xi8>) -> (tensor<[%M]xf32>)"
            " where [mul(%M, 16), mul(%K, 32)]\n"
        )


class TestAssumeOpRoundTrip:
    """Round-trip tests for test.assume (predicate-constrained identity)."""

    def _roundtrip_text(self, text: str) -> None:
        module = _op_parser().parse(text)
        printed = _op_printer().print_module(module)
        assert printed == text, (
            f"Round-trip failed.\nInput:\n{text}\nOutput:\n{printed}"
        )

    def test_single_predicate(self) -> None:
        self._roundtrip_text(
            "func.def @f(%M: index) -> (index) {\n"
            "  %M2 = test.assume %M [mul(%M, 16)] : index\n"
            "  test.yield %M2 : index\n"
            "}\n"
        )

    def test_multiple_predicates(self) -> None:
        self._roundtrip_text(
            "func.def @f(%M: index, %K: index) -> (index, index) {\n"
            "  %M2, %K2 = test.assume %M, %K [mul(%M, 16), lt(%K, 1024)] : index, index\n"
            "  test.yield %M2, %K2 : index, index\n"
            "}\n"
        )

    def test_pow2_predicate(self) -> None:
        self._roundtrip_text(
            "func.def @f(%N: index) -> (index) {\n"
            "  %N2 = test.assume %N [pow2(%N)] : index\n"
            "  test.yield %N2 : index\n"
            "}\n"
        )

    def test_range_predicate(self) -> None:
        self._roundtrip_text(
            "func.def @f(%M: index) -> (index) {\n"
            "  %M2 = test.assume %M [range(%M, 32, 512)] : index\n"
            "  test.yield %M2 : index\n"
            "}\n"
        )


# ============================================================================
# Import declarations — parsing and round-trip
# ============================================================================


class TestParseImportDeclaration:
    def _parse_module(self, text: str) -> Module:
        return _op_parser().parse(text)

    def test_basic_import(self) -> None:
        module = self._parse_module(
            'func.decl import("linalg_lib") @matmul(%a: f32, %b: f32) -> (f32)\n'
        )
        assert len(module.symbols) == 1
        sym = module.symbols[0]
        assert sym.is_import
        assert sym.source_module == "linalg_lib"
        assert sym.source_symbol == ""  # defaults to symbol name
        assert sym.name == "matmul"
        assert sym.op is not None
        assert not sym.op.regions  # Declarations have no body region.

    def test_import_with_alias(self) -> None:
        module = self._parse_module(
            'func.decl import("math_lib", "matmul_v2") @my_matmul(%a: f32) -> (f32)\n'
        )
        sym = module.symbols[0]
        assert sym.is_import
        assert sym.source_module == "math_lib"
        assert sym.source_symbol == "matmul_v2"
        assert sym.name == "my_matmul"

    def test_public_import(self) -> None:
        """public and import modifiers together (re-export)."""
        module = self._parse_module(
            'func.decl public import("upstream") @relu(%x: f32) -> (f32)\n'
        )
        sym = module.symbols[0]
        assert sym.is_import
        assert sym.is_public
        assert sym.source_module == "upstream"

    def test_import_with_types(self) -> None:
        """Import with shaped types preserves full signature."""
        module = self._parse_module(
            'func.decl import("kernels") @conv(%N: index, %w: tensor<3x3xf32>, %x: tensor<[%N]xf32>)'
            " -> (tensor<[%N]xf32>)\n"
        )
        sym = module.symbols[0]
        assert sym.is_import
        op = sym.op
        assert op is not None
        # func.decl: all args (including dim params) become operands.
        assert len(op.operands) == 3
        assert len(op.results) == 1

    def test_public_import_canonical_order(self) -> None:
        """Canonical order: visibility before import modifier."""
        module = self._parse_module(
            'func.decl public import("lib") @f(%a: f32) -> (f32)\n'
        )
        sym = module.symbols[0]
        assert sym.is_import
        assert sym.is_public

    def test_non_import_has_no_source(self) -> None:
        """Regular declarations are not imports."""
        module = self._parse_module("func.decl @extern(%a: f32) -> (f32)\n")
        sym = module.symbols[0]
        assert not sym.is_import
        assert sym.source_module == ""

    def test_import_in_multi_function_module(self) -> None:
        """Import mixed with definitions in a module."""
        module = self._parse_module(
            'func.decl import("lib") @imported(%a: f32) -> (f32)\n'
            "\n"
            "func.def @local(%x: f32) -> (f32) {\n"
            "  test.yield %x : f32\n"
            "}\n"
        )
        assert len(module.symbols) == 2
        syms = {s.name: s for s in module.symbols}
        assert syms["imported"].is_import
        assert syms["imported"].source_module == "lib"
        assert not syms["local"].is_import


class TestImportRoundTrip:
    """Text → parse → IR → print → text for import declarations."""

    def _roundtrip_text(self, text: str) -> None:
        module = _op_parser().parse(text)
        printed = _op_printer().print_module(module)
        assert printed == text, (
            f"Round-trip mismatch:\n  expected: {text!r}\n  got:      {printed!r}"
        )

    def test_basic_import_roundtrip(self) -> None:
        self._roundtrip_text(
            'func.decl import("linalg_lib") @matmul(%a: f32, %b: f32) -> (f32)\n'
        )

    def test_import_alias_roundtrip(self) -> None:
        self._roundtrip_text(
            'func.decl import("math_lib", "matmul") @my_matmul(%a: f32) -> (f32)\n'
        )

    def test_public_import_roundtrip(self) -> None:
        self._roundtrip_text(
            'func.decl public import("upstream") @relu(%x: f32) -> (f32)\n'
        )

    def test_mixed_module_roundtrip(self) -> None:
        self._roundtrip_text(
            'func.decl import("lib") @imported(%a: f32) -> (f32)\n'
            "\n"
            "func.def @local(%x: f32) -> (f32) {\n"
            "  test.yield %x : f32\n"
            "}\n"
        )


# ============================================================================
# Dynamic dim and encoding scoping in function signatures
# ============================================================================


class TestDynamicDimScoping:
    """Dynamic dim names in types reference earlier args, not define new ones.

    This is critical for C parser parity: %M in tile<[%M]xf32> must resolve
    to the %M: index arg, not create a duplicate definition.
    """

    def _roundtrip_text(self, text: str) -> None:
        module = _op_parser().parse(text)
        printed = _op_printer().print_module(module)
        assert printed == text, (
            f"Round-trip mismatch:\n  expected: {text!r}\n  got:      {printed!r}"
        )

    def test_dim_references_earlier_arg(self) -> None:
        """Dynamic dim %M in second arg references %M: index first arg."""
        self._roundtrip_text(
            "func.def @f(%M: index, %tile: tile<[%M]xf32>) -> (tile<[%M]xf32>) {\n"
            "  test.yield %tile : tile<[%M]xf32>\n"
            "}\n"
        )

    def test_multiple_dynamic_dims(self) -> None:
        """Two dynamic dims from separate earlier args."""
        self._roundtrip_text(
            "func.decl @g(%M: index, %N: index, %t: tensor<[%M]x[%N]xf32>)"
            " -> (tensor<[%M]x[%N]xf32>)\n"
        )

    def test_mixed_static_dynamic_dims(self) -> None:
        self._roundtrip_text(
            "func.decl @h(%K: index, %t: tile<[%K]x4xf32>) -> (tile<[%K]x4xf32>)\n"
        )

    def test_dim_shared_across_args(self) -> None:
        """Same dim name %M used by two different arg types."""
        self._roundtrip_text(
            "func.decl @shared(%M: index, %a: tile<[%M]xf32>, %b: tile<[%M]xi8>)"
            " -> (tile<[%M]xf32>)\n"
        )

    def test_dim_explicitly_defined(self) -> None:
        """Explicitly defined %M in a type."""
        self._roundtrip_text(
            "func.decl @explicit(%M: index, %a: tile<[%M]xf32>, %b: tile<[%M]xi8>)"
            " -> (tile<[%M]xf32>)\n"
        )

    def test_dynamic_pool_references_earlier_arg(self) -> None:
        """Dynamic pool block size references earlier index arg."""
        self._roundtrip_text(
            "func.decl @p(%BS: index, %pool: pool<[%BS]>) -> (pool<[%BS]>)\n"
        )

    def test_dynamic_encoding_references_earlier_arg(self) -> None:
        """Dynamic encoding %enc in type references earlier encoding arg."""
        self._roundtrip_text(
            "func.decl @enc(%enc: encoding, %t: tile<4xf32, %enc>)"
            " -> (tile<4xf32, %enc>)\n"
        )

    def test_all_dynamic_in_one_signature(self) -> None:
        """Dynamic dims, encoding, and pool all in one function."""
        self._roundtrip_text(
            "func.decl @kitchen_sink(%M: index, %N: index, %enc: encoding,"
            " %t: tile<[%M]x[%N]xf32, %enc>, %p: pool<[%M]>)"
            " -> (tile<[%M]x[%N]xf32, %enc>)\n"
        )

    def test_def_with_dynamic_dims_in_body(self) -> None:
        """Dynamic dims flow into body ops."""
        self._roundtrip_text(
            "func.def @f(%M: index, %t: tile<[%M]xf32>) -> (tile<[%M]xf32>) {\n"
            "  %neg = test.neg %t : tile<[%M]xf32>\n"
            "  test.yield %neg : tile<[%M]xf32>\n"
            "}\n"
        )


class TestImportCrossFormatRoundTrip:
    """Text → IR → bytecode → IR → text for import declarations.

    The bytecode SYMBOLS section stores arg types but not arg names
    (names only live in the IR section as block args, which declarations
    don't have). So cross-format round-trips through bytecode lose
    arg names on declarations. This is a pre-existing limitation of the
    bytecode format, not specific to imports.
    """

    def _cross_roundtrip(self, text: str, expected: str | None = None) -> None:
        module = _op_parser().parse(text)
        loaded = read_module(write_module(module))
        printed = _op_printer().print_module(loaded)
        target = expected if expected is not None else text
        assert printed == target, (
            f"Cross-format round-trip mismatch:\n"
            f"  expected: {target!r}\n"
            f"  got:      {printed!r}"
        )

    def test_import_survives_bytecode(self) -> None:
        # Arg names lost in bytecode (no IR section for declarations).
        self._cross_roundtrip(
            'func.decl import("linalg_lib") @matmul(%a: f32, %b: f32) -> (f32)\n',
            'func.decl import("linalg_lib") @matmul(f32, f32) -> (f32)\n',
        )

    def test_import_alias_survives_bytecode(self) -> None:
        self._cross_roundtrip(
            'func.decl import("math_lib", "matmul") @my_matmul(%a: f32) -> (f32)\n',
            'func.decl import("math_lib", "matmul") @my_matmul(f32) -> (f32)\n',
        )

    def test_import_metadata_preserved(self) -> None:
        """Import source module and symbol survive bytecode round-trip."""
        module = _op_parser().parse(
            'func.decl import("math_lib", "original") @alias(%a: f32) -> (f32)\n'
        )
        loaded = read_module(write_module(module))
        sym = loaded.symbols[0]
        assert sym.is_import
        assert sym.source_module == "math_lib"
        assert sym.source_symbol == "original"
        assert sym.name == "alias"

    def test_mixed_module_survives_bytecode(self) -> None:
        # Declarations lose arg names (no IR section), but definitions
        # recover them from the entry block's values.
        self._cross_roundtrip(
            'func.decl import("lib") @imported(%a: f32) -> (f32)\n'
            "\n"
            "func.def @local(%x: f32) -> (f32) {\n"
            "  test.yield %x : f32\n"
            "}\n",
            'func.decl import("lib") @imported(f32) -> (f32)\n'
            "\n"
            "func.def @local(%x: f32) -> (f32) {\n"
            "  test.yield %x : f32\n"
            "}\n",
        )


# ============================================================================
# Location parsing
# ============================================================================


class TestLocationParsing:
    """Tests for parsing loc(...) annotations on ops."""

    def test_file_location(self) -> None:
        """Parse a FILE location annotation."""
        module, scope = _setup_scope(("x", F32))
        op = _parse_op(
            '%r = test.neg %x : f32 loc("model.loom":42:3 to 42:58)',
            module=module,
            scope=scope,
        )
        from loom.ir import FileLocation

        loc = module.locations.get(op.location_id)
        assert isinstance(loc, FileLocation)
        assert loc.start_line == 42
        assert loc.start_col == 3
        assert loc.end_line == 42
        assert loc.end_col == 58
        # Source name should be registered in the module.
        assert module.sources[loc.source_id] == "model.loom"

    def test_fused_location(self) -> None:
        """Parse a FUSED location annotation."""
        from loom.ir import FileLocation, FusedLocation

        module, scope = _setup_scope(("x", F32))
        op = _parse_op(
            '%r = test.neg %x : f32 loc(fused<"a.loom":1:1, "b.loom":2:2>)',
            module=module,
            scope=scope,
        )
        loc = module.locations.get(op.location_id)
        assert isinstance(loc, FusedLocation)
        assert len(loc.children) == 2
        # Each child should be a FileLocation.
        child_a = module.locations.get(loc.children[0])
        child_b = module.locations.get(loc.children[1])
        assert isinstance(child_a, FileLocation)
        assert isinstance(child_b, FileLocation)
        assert child_a.start_line == 1
        assert child_a.start_col == 1
        assert child_b.start_line == 2
        assert child_b.start_col == 2
        assert module.sources[child_a.source_id] == "a.loom"
        assert module.sources[child_b.source_id] == "b.loom"

    def test_opaque_location(self) -> None:
        """Parse an OPAQUE location annotation."""
        from loom.ir import OpaqueLocation

        module, scope = _setup_scope(("x", F32))
        op = _parse_op(
            '%r = test.neg %x : f32 loc(opaque<"torch", "node_id=42">)',
            module=module,
            scope=scope,
        )
        loc = module.locations.get(op.location_id)
        assert isinstance(loc, OpaqueLocation)
        assert module.sources[loc.source_id] == "torch"
        assert loc.data == b"node_id=42"

    def test_no_location(self) -> None:
        """Ops without explicit location use implicit source position."""
        from loom.ir import FileLocation

        module, scope = _setup_scope(("x", F32))
        op = _parse_op(
            "%r = test.neg %x : f32",
            module=module,
            scope=scope,
        )
        loc = module.locations.get(op.location_id)
        # Implicit locations are FileLocations derived from the source text.
        assert isinstance(loc, FileLocation)
        # source_id is 0 (the default for implicit locations).
        assert loc.source_id == 0


# ============================================================================
# Location round-trip: parse with loc() → print with print_locations → match
# ============================================================================


class TestLocationRoundTrip:
    """Parse text with locations, print with locations, verify identical output."""

    def _roundtrip_with_locations(self, text: str) -> None:
        """Parse text, print with print_locations=True, assert identical."""
        module = _op_parser().parse(text)
        printed = _op_printer(print_locations=True).print_module(module)

        assert printed == text, (
            f"Location round-trip failed.\nInput:\n{text}\nOutput:\n{printed}"
        )

    def test_file_location_roundtrip(self) -> None:
        self._roundtrip_with_locations(
            "func.def @negate(%input: f32) -> (f32) {\n"
            '  %neg0 = test.neg %input : f32 loc("model.loom":42:3 to 42:58)\n'
            '  test.yield %neg0 : f32 loc("model.loom":43:3 to 43:28)\n'
            "}\n"
        )

    def test_fused_location_roundtrip(self) -> None:
        """All ops have explicit locations (fused + file)."""
        self._roundtrip_with_locations(
            "func.def @negate(%input: f32) -> (f32) {\n"
            '  %neg0 = test.neg %input : f32 loc(fused<"a.loom":1:1, "b.loom":2:2>)\n'
            '  test.yield %neg0 : f32 loc("a.loom":3:1 to 3:28)\n'
            "}\n"
        )

    def test_opaque_location_roundtrip(self) -> None:
        """All ops have explicit locations (opaque + file)."""
        self._roundtrip_with_locations(
            "func.def @negate(%input: f32) -> (f32) {\n"
            '  %neg0 = test.neg %input : f32 loc(opaque<"torch", "node_id=42">)\n'
            '  test.yield %neg0 : f32 loc("model.loom":2:3 to 2:28)\n'
            "}\n"
        )

    def test_stable_after_two_rounds(self) -> None:
        """Parse → print → parse → print is stable (mixed implicit/explicit)."""
        text = (
            "func.def @negate(%input: f32) -> (f32) {\n"
            '  %neg0 = test.neg %input : f32 loc("model.loom":10:3 to 10:40)\n'
            "  test.yield %neg0 : f32\n"
            "}\n"
        )

        def parse_and_print(source: str) -> str:
            module = _op_parser().parse(source)
            return _op_printer(print_locations=True).print_module(module)

        round1 = parse_and_print(text)
        round2 = parse_and_print(round1)
        assert round1 == round2, (
            f"Not stable after two rounds.\nRound 1:\n{round1}\nRound 2:\n{round2}"
        )


# ============================================================================
# func.template and func.ukernel round-trip
# ============================================================================


class TestFuncTemplateUkernelRoundTrip:
    """Round-trip tests for func.template and func.ukernel."""

    def _roundtrip_text(self, text: str) -> None:
        """Parse text, print, assert identical."""
        module = _op_parser().parse(text)
        printed = _op_printer().print_module(module)

        assert printed == text, (
            f"Round-trip failed.\nInput:\n{text}\nOutput:\n{printed}"
        )

    def test_template_basic(self) -> None:
        """func.template<tile.contract> with body round-trips."""
        self._roundtrip_text(
            "func.template<tile.contract> @impl(%a: tile<4xf32>)"
            " -> (tile<4xf32>) {\n"
            "  func.return %a : tile<4xf32>\n"
            "}\n"
        )

    def test_template_with_priority(self) -> None:
        """func.template with priority(N) round-trips."""
        self._roundtrip_text(
            "func.template<tile.contract> priority(10) @high_priority(%a: tile<4xf32>)"
            " -> (tile<4xf32>) {\n"
            "  func.return %a : tile<4xf32>\n"
            "}\n"
        )

    def test_template_device_cc(self) -> None:
        """func.template with device calling convention round-trips."""
        self._roundtrip_text(
            "func.template<tile.contract> device @device_impl(%a: tile<4xf32>)"
            " -> (tile<4xf32>) {\n"
            "  func.return %a : tile<4xf32>\n"
            "}\n"
        )

    def test_ukernel_basic(self) -> None:
        """func.ukernel<tile.contract> (no body) round-trips."""
        self._roundtrip_text(
            "func.ukernel<tile.contract> @asm_impl(%a: tile<4xf32>) -> (tile<4xf32>)\n"
        )

    def test_ukernel_device(self) -> None:
        """func.ukernel with device calling convention round-trips."""
        self._roundtrip_text(
            "func.ukernel<tile.contract> device @asm_device(%a: tile<4xf32>)"
            " -> (tile<4xf32>)\n"
        )

    def test_ukernel_with_priority(self) -> None:
        """func.ukernel with priority round-trips."""
        self._roundtrip_text(
            "func.ukernel<tile.contract> priority(5) @prioritized(%a: f32) -> (f32)\n"
        )

    def test_template_implements_stored(self) -> None:
        """Verify the parsed func.template op has the implements attribute set."""
        module = _op_parser().parse(
            "func.template<tile.contract> @impl(%a: f32) -> (f32) {\n"
            "  func.return %a : f32\n"
            "}\n"
        )
        op = module.symbols[0].op
        assert op is not None
        assert op.attributes.get("implements") == "tile.contract"
        assert op.attributes.get("priority") is None

    def test_template_priority_stored(self) -> None:
        """Verify the parsed func.template op has the priority attribute set."""
        module = _op_parser().parse(
            "func.template<tile.contract> priority(42) @impl(%a: f32) -> (f32) {\n"
            "  func.return %a : f32\n"
            "}\n"
        )
        op = module.symbols[0].op
        assert op is not None
        assert op.attributes.get("implements") == "tile.contract"
        assert op.attributes.get("priority") == 42


# ============================================================================
# Cross-format round-trip: text → bytecode → text with locations
# ============================================================================


class TestLocationCrossFormatRoundTrip:
    """Test that locations survive text → bytecode → text."""

    def _cross_roundtrip_with_locations(
        self, text: str, expected: str | None = None
    ) -> None:
        module = _op_parser().parse(text)
        loaded = read_module(write_module(module))
        printed = _op_printer(print_locations=True).print_module(loaded)
        target = expected if expected is not None else text
        assert printed == target, (
            f"Cross-format location round-trip mismatch:\n"
            f"  expected: {target!r}\n"
            f"  got:      {printed!r}"
        )

    def test_file_location_survives_bytecode(self) -> None:
        self._cross_roundtrip_with_locations(
            "func.def @negate(%input: f32) -> (f32) {\n"
            '  %neg0 = test.neg %input : f32 loc("model.loom":42:3 to 42:58)\n'
            '  test.yield %neg0 : f32 loc("model.loom":43:3 to 43:28)\n'
            "}\n"
        )

    def test_opaque_location_survives_bytecode(self) -> None:
        self._cross_roundtrip_with_locations(
            "func.def @negate(%input: f32) -> (f32) {\n"
            '  %neg0 = test.neg %input : f32 loc(opaque<"torch", "node_id=42">)\n'
            '  test.yield %neg0 : f32 loc("model.loom":3:3 to 3:28)\n'
            "}\n"
        )

    def test_fused_location_survives_bytecode(self) -> None:
        self._cross_roundtrip_with_locations(
            "func.def @negate(%input: f32) -> (f32) {\n"
            '  %neg0 = test.neg %input : f32 loc(fused<"a.loom":1:1, "b.loom":2:2>)\n'
            '  test.yield %neg0 : f32 loc("model.loom":3:3 to 3:28)\n'
            "}\n"
        )


# ============================================================================
# Nested dict attribute values
# ============================================================================


class TestNestedDictAttr:
    """Test parsing of nested dict values in attribute dicts."""

    def test_nested_dict(self) -> None:
        """Parse {outer = {inner = 42}} as nested dict."""
        module, scope = _setup_scope(("x", F32))
        op = _parse_op(
            '%r = test.attrs %x {meta = {depth = 3, name = "layer0"}} : f32',
            module=module,
            scope=scope,
        )
        d = op.attributes["dict"]
        assert isinstance(d["meta"], CanonicalAttrDict)
        assert list(d["meta"].items()) == [("depth", 3), ("name", "layer0")]
        assert d["meta"]["depth"] == 3
        assert d["meta"]["name"] == "layer0"

    def test_deeply_nested_dict(self) -> None:
        """Parse three levels of nesting."""
        module, scope = _setup_scope(("x", F32))
        op = _parse_op(
            "%r = test.attrs %x {a = {b = {c = 99}}} : f32",
            module=module,
            scope=scope,
        )
        d = op.attributes["dict"]
        assert isinstance(d["a"], CanonicalAttrDict)
        assert isinstance(d["a"]["b"], CanonicalAttrDict)
        assert d["a"]["b"]["c"] == 99

    def test_nested_duplicate_key_is_rejected(self) -> None:
        module, scope = _setup_scope(("x", F32))
        with pytest.raises(ParseError, match="duplicate attribute dict key 'depth'"):
            _parse_op(
                "%r = test.attrs %x {meta = {depth = 3, depth = 4}} : f32",
                module=module,
                scope=scope,
            )

    def test_unsorted_nested_dict_cross_format_round_trip(self) -> None:
        text = (
            "func.def @f(%x: f32) -> (f32) {\n"
            '  %r = test.attrs %x {meta = {phase = "link", opt = 3}, axis = 0} : f32\n'
            "  test.yield %r : f32\n"
            "}\n"
        )
        module = _op_parser().parse(text)
        loaded = read_module(write_module(module))
        printed = _op_printer().print_module(loaded)
        assert printed == (
            "func.def @f(%x: f32) -> (f32) {\n"
            '  %r = test.attrs %x {axis = 0, meta = {opt = 3, phase = "link"}} : f32\n'
            "  test.yield %r : f32\n"
            "}\n"
        )
