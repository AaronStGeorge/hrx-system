# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the format-driven text printer."""

from loom.builtin_types import ALL_BUILTIN_TYPES
from loom.dialect.func import ALL_FUNC_OPS
from loom.dialect.test import ALL_TEST_OPS
from loom.format.text.parser import Parser
from loom.format.text.printer import Printer, print_type
from loom.ir import (
    BF16,
    ENCODING_TYPE,
    F32,
    I1,
    I8,
    I32,
    INDEX,
    NONE_TYPE,
    Block,
    DynamicDim,
    DynamicEncoding,
    EncodingInstance,
    FunctionType,
    GroupScope,
    GroupType,
    Module,
    Operation,
    PoolType,
    Region,
    ScalarType,
    ScalarTypeKind,
    ShapedType,
    StaticDim,
    Type,
    TypeKind,
    Value,
)
from loom.ir import (
    TiedResult as IRTiedResult,
)

# ============================================================================
# Helpers
# ============================================================================


def _printer() -> Printer:
    """Create a printer with all test ops registered."""
    printer = Printer()
    printer.register_ops(ALL_TEST_OPS)
    return printer


def _module_parser() -> Parser:
    """Create a parser with test + func ops and builtin types."""
    parser = Parser()
    parser.register_ops(ALL_TEST_OPS)
    parser.register_ops(ALL_FUNC_OPS)
    parser.register_types(ALL_BUILTIN_TYPES)
    return parser


def _module_printer(
    *,
    print_locations: bool = False,
    use_aliases: bool = True,
    indent: bool = True,
    print_regions: bool = True,
) -> Printer:
    """Create a printer with test + func ops and builtin types."""
    printer = Printer(
        print_locations=print_locations,
        use_aliases=use_aliases,
        indent=indent,
        print_regions=print_regions,
    )
    printer.register_ops(ALL_TEST_OPS)
    printer.register_ops(ALL_FUNC_OPS)
    printer.register_types(ALL_BUILTIN_TYPES)
    return printer


def _module_with(*names_and_types: tuple[str, Type]) -> tuple[Module, list[int]]:
    """Create a module with named, typed values."""
    module = Module(name="test")
    value_ids = []
    for name, value_type in names_and_types:
        vid = module.add_value(Value(name=name, type=value_type))
        value_ids.append(vid)
    return module, value_ids


_tile_4xf32 = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
_tile_4x4xf32 = ShapedType(TypeKind.TILE, F32, (StaticDim(4), StaticDim(4)))
_tensor_4xf32 = ShapedType(TypeKind.TENSOR, F32, (StaticDim(4),))
_tile_64x64xf16 = ShapedType(
    TypeKind.TILE, ScalarType(ScalarTypeKind.F16), (StaticDim(64), StaticDim(64))
)
_tile_16x16xf16 = ShapedType(
    TypeKind.TILE, ScalarType(ScalarTypeKind.F16), (StaticDim(16), StaticDim(16))
)


# ============================================================================
# Type printing
# ============================================================================


class TestPrintType:
    def test_scalar_types(self) -> None:
        assert print_type(F32) == "f32"
        assert print_type(I32) == "i32"
        assert print_type(INDEX) == "index"
        assert print_type(BF16) == "bf16"
        assert print_type(I8) == "i8"

    def test_tile_1d(self) -> None:
        assert print_type(_tile_4xf32) == "tile<4xf32>"

    def test_tile_2d(self) -> None:
        assert print_type(_tile_4x4xf32) == "tile<4x4xf32>"

    def test_tensor(self) -> None:
        assert print_type(_tensor_4xf32) == "tensor<4xf32>"

    def test_tile_0d(self) -> None:
        assert print_type(ShapedType(TypeKind.TILE, F32, ())) == "tile<f32>"

    def test_dynamic_dims_no_context(self) -> None:
        """Without context, dynamic dims print as '?'."""
        t = ShapedType(TypeKind.TILE, F32, (DynamicDim(), StaticDim(4)))
        assert print_type(t) == "tile<?x4xf32>"

    def test_dynamic_dims_with_context(self) -> None:
        """With context, dynamic dims print as [%name]."""
        from loom.format.text.printer import TypePrintContext

        t = ShapedType(TypeKind.TILE, F32, (DynamicDim(), StaticDim(4)))
        # Set up: value ID 0 is named %M.
        module = Module(name="test")
        module.add_value(Value(name="%M", type=INDEX))
        context = TypePrintContext(
            dim_bindings={0: 0},  # dim position 0 -> value ID 0
            module=module,
        )
        assert print_type(t, context) == "tile<[%M]x4xf32>"

    def test_dynamic_dims_all_dynamic(self) -> None:
        """Multiple dynamic dims with named bindings."""
        from loom.format.text.printer import TypePrintContext

        t = ShapedType(TypeKind.TENSOR, F32, (DynamicDim(), DynamicDim()))
        module = Module(name="test")
        module.add_value(Value(name="%M", type=INDEX))
        module.add_value(Value(name="%K", type=INDEX))
        context = TypePrintContext(
            dim_bindings={0: 0, 1: 1},
            module=module,
        )
        assert print_type(t, context) == "tensor<[%M]x[%K]xf32>"

    def test_ordinal_dim(self) -> None:
        """Ordinal dims print as [#N]."""
        from loom.format.text.printer import TypePrintContext

        t = ShapedType(TypeKind.TENSOR, F32, (DynamicDim(),))
        module = Module(name="test")
        # Ordinal dim: -1 = #0.
        context = TypePrintContext(dim_bindings={0: -1}, module=module)
        assert print_type(t, context) == "tensor<[#0]xf32>"

    def test_ordinal_dim_nonzero(self) -> None:
        """Ordinal dim #1 prints correctly."""
        from loom.format.text.printer import TypePrintContext

        t = ShapedType(TypeKind.TENSOR, F32, (DynamicDim(),))
        module = Module(name="test")
        # Ordinal dim: -2 = #1.
        context = TypePrintContext(dim_bindings={0: -2}, module=module)
        assert print_type(t, context) == "tensor<[#1]xf32>"

    def test_encoding_with_alias(self) -> None:
        """Type with encoding prints the alias when available."""

        enc = EncodingInstance(name="q8_0", alias="#enc")
        t = ShapedType(TypeKind.TILE, I8, (StaticDim(256),), encoding=enc)
        assert print_type(t) == "tile<256xi8, #enc>"

    def test_encoding_without_alias(self) -> None:
        """Without alias, prints #name<params>."""

        enc = EncodingInstance(name="q8_0", params=(("block", "32"),))
        t = ShapedType(TypeKind.TILE, I8, (StaticDim(256),), encoding=enc)
        assert print_type(t) == "tile<256xi8, #q8_0<block=32>>"

    def test_encoding_without_alias_no_params(self) -> None:
        """Without alias or params, prints just #name."""

        enc = EncodingInstance(name="dense")
        t = ShapedType(TypeKind.TILE, I8, (StaticDim(256),), encoding=enc)
        assert print_type(t) == "tile<256xi8, #dense>"

    def test_no_encoding(self) -> None:
        """Type without encoding has no suffix."""
        t = ShapedType(TypeKind.TILE, I8, (StaticDim(256),))
        assert print_type(t) == "tile<256xi8>"

    def test_group_type(self) -> None:
        assert print_type(GroupType(GroupScope.WORKGROUP)) == "group<workgroup>"

    def test_function_type(self) -> None:
        ft = FunctionType((F32, I32), (F32,))
        assert print_type(ft) == "(f32, i32) -> (f32)"

    def test_none_type(self) -> None:
        assert print_type(NONE_TYPE) == "none"

    def test_dialect_type_opaque(self) -> None:
        from loom.ir import DialectType

        assert print_type(DialectType("hal.buffer")) == "hal.buffer"

    def test_dialect_type_parameterized(self) -> None:
        from loom.ir import DialectType

        t = DialectType("vm.ref", (DialectType("hal.buffer"),))
        assert print_type(t) == "vm.ref<hal.buffer>"

    def test_dialect_type_nested(self) -> None:
        from loom.ir import DialectType

        inner = DialectType("vm.list", (I32,))
        outer = DialectType("vm.ref", (inner,))
        assert print_type(outer) == "vm.ref<vm.list<i32>>"

    def test_dialect_type_with_scalar_param(self) -> None:
        from loom.ir import DialectType

        t = DialectType("vm.list", (F32,))
        assert print_type(t) == "vm.list<f32>"

    def test_dialect_type_with_format_spec(self) -> None:
        """Printer walks TypeDef format spec when type_registry is provided."""
        from loom.assembly import TypeOf as AsmTypeOf
        from loom.assembly import kw
        from loom.dsl import ANY, TypeDef, TypeParam
        from loom.ir import DialectType

        # Type with custom format: name<T to T> instead of name<T, T>
        type_def = TypeDef(
            name="test.pair",
            params=[TypeParam("first", ANY), TypeParam("second", ANY)],
            format=[AsmTypeOf("first"), kw("to"), AsmTypeOf("second")],
        )
        registry = {"test.pair": type_def}
        t = DialectType("test.pair", (F32, I32))
        result = print_type(t, type_registry=registry)
        assert result == "test.pair<f32 to i32>"

    def test_dialect_type_format_fallback_without_registry(self) -> None:
        """Without registry, falls back to comma-separated params."""
        from loom.ir import DialectType

        t = DialectType("test.pair", (F32, I32))
        result = print_type(t)
        assert result == "test.pair<f32, i32>"


# ============================================================================
# Simple op printing — one format element type per test
# ============================================================================


class TestPrintBinaryOp:
    """Exercises: Ref, Keyword(','), Keyword(':'), TypeOf, SameType."""

    def test_basic(self) -> None:
        module, [a, b, r] = _module_with(("%a", I32), ("%b", I32), ("%r", I32))
        op = Operation(name="test.addi", operands=[a, b], results=[r])
        assert _printer().print_operation(op, module) == "%r = test.addi %a, %b : i32"


class TestPrintUnaryOp:
    """Exercises: single Ref, TypeOf on result."""

    def test_basic(self) -> None:
        module, [x, r] = _module_with(("%x", F32), ("%r", F32))
        op = Operation(name="test.neg", operands=[x], results=[r])
        assert _printer().print_operation(op, module) == "%r = test.neg %x : f32"


class TestPrintCastOp:
    """Exercises: TypeOf on both input and result, Keyword('to')."""

    def test_basic(self) -> None:
        module, [x, r] = _module_with(("%x", I32), ("%r", F32))
        op = Operation(name="test.cast", operands=[x], results=[r])
        assert (
            _printer().print_operation(op, module) == "%r = test.cast %x : i32 to f32"
        )


class TestPrintConvertOp:
    """Exercises: ResultType (bare result type, no parens)."""

    def test_scalar(self) -> None:
        module, [x, r] = _module_with(("%x", I32), ("%r", F32))
        op = Operation(name="test.convert", operands=[x], results=[r])
        assert (
            _printer().print_operation(op, module)
            == "%r = test.convert %x : i32 -> f32"
        )

    def test_tile(self) -> None:
        tile_i8 = ShapedType(
            TypeKind.TILE, ScalarType(ScalarTypeKind.I8), (StaticDim(4),)
        )
        module, [x, r] = _module_with(("%x", tile_i8), ("%r", _tile_4xf32))
        op = Operation(name="test.convert", operands=[x], results=[r])
        assert (
            _printer().print_operation(op, module)
            == "%r = test.convert %x : tile<4xi8> -> tile<4xf32>"
        )


class TestPrintConstantOp:
    """Exercises: Attr (value before colon), CONSTANT_LIKE."""

    def test_integer(self) -> None:
        module, [r] = _module_with(("%c42", I32))
        op = Operation(name="test.constant", results=[r], attributes={"value": 42})
        assert _printer().print_operation(op, module) == "%c42 = test.constant 42 : i32"

    def test_float(self) -> None:
        module, [r] = _module_with(("%pi", F32))
        op = Operation(name="test.constant", results=[r], attributes={"value": 3.14})
        text = _printer().print_operation(op, module)
        assert text.startswith("%pi = test.constant 3.14")
        assert text.endswith(" : f32")


class TestPrintComparisonOp:
    """Exercises: Attr (enum predicate printed as bare keyword)."""

    def test_basic(self) -> None:
        module, [a, b, r] = _module_with(("%a", I32), ("%b", I32), ("%r", I32))
        op = Operation(
            name="test.cmp",
            operands=[a, b],
            results=[r],
            attributes={"predicate": "lt"},
        )
        assert (
            _printer().print_operation(op, module) == "%r = test.cmp lt, %a, %b : i32"
        )


class TestPrintYield:
    """Exercises: Refs (variadic), TypesOf (variadic), TERMINATOR."""

    def test_single(self) -> None:
        module, [a] = _module_with(("%a", F32))
        op = Operation(name="test.yield", operands=[a])
        assert _printer().print_operation(op, module) == "test.yield %a : f32"

    def test_multiple(self) -> None:
        module, [a, b] = _module_with(("%a", F32), ("%b", I32))
        op = Operation(name="test.yield", operands=[a, b])
        assert _printer().print_operation(op, module) == "test.yield %a, %b : f32, i32"

    def test_no_results_prefix(self) -> None:
        """Yield has no results, so no '= ' prefix."""
        module, [a] = _module_with(("%a", F32))
        op = Operation(name="test.yield", operands=[a])
        text = _printer().print_operation(op, module)
        assert not text.startswith("%")
        assert "=" not in text.split("test.yield")[0]


# ============================================================================
# Complex op printing
# ============================================================================


class TestPrintSlice:
    """Exercises: IndexList (mixed static/dynamic), gluing to Ref."""

    def test_all_static(self) -> None:
        module, [src, r] = _module_with(
            ("%src", _tile_64x64xf16), ("%r", _tile_16x16xf16)
        )
        op = Operation(
            name="test.slice",
            operands=[src],
            results=[r],
            attributes={"static_offsets": [0, 0]},
        )
        assert (
            _printer().print_operation(op, module)
            == "%r = test.slice %src[0, 0] : tile<64x64xf16> -> (tile<16x16xf16>)"
        )

    def test_mixed_static_dynamic(self) -> None:
        sentinel = -(2**63)
        module, [src, off, r] = _module_with(
            ("%src", _tile_64x64xf16), ("%off", INDEX), ("%r", _tile_16x16xf16)
        )
        op = Operation(
            name="test.slice",
            operands=[src, off],
            results=[r],
            attributes={"static_offsets": [0, sentinel]},
        )
        assert (
            _printer().print_operation(op, module)
            == "%r = test.slice %src[0, %off] : tile<64x64xf16> -> (tile<16x16xf16>)"
        )


class TestPrintTiedResult:
    """Exercises: ResultTypeList with tied results."""

    def test_single_tied(self) -> None:
        module, [tile, tensor, off, result] = _module_with(
            ("%tile", _tile_4xf32),
            ("%tensor", _tensor_4xf32),
            ("%off", INDEX),
            ("%result", _tensor_4xf32),
        )
        sentinel = -(2**63)
        op = Operation(
            name="test.update",
            operands=[tile, tensor, off],
            results=[result],
            attributes={"static_offsets": [sentinel]},
            tied_results=[IRTiedResult(result_index=0, operand_index=1)],
        )
        assert (
            _printer().print_operation(op, module)
            == "%result = test.update %tile, %tensor[%off] : tile<4xf32> -> (%tensor as tensor<4xf32>)"
        )


class TestPrintInvoke:
    """Exercises: SymbolRef, Refs, TypesOf, ResultTypeList with variadic + ties."""

    def test_with_tie(self) -> None:
        module, [w, x, out1, out2] = _module_with(
            ("%weights", _tile_4xf32),
            ("%input", INDEX),
            ("%output", _tile_4xf32),
            ("%count", INDEX),
        )
        op = Operation(
            name="test.invoke",
            operands=[w, x],
            results=[out1, out2],
            attributes={"callee": "compute"},
            tied_results=[IRTiedResult(result_index=0, operand_index=0)],
        )
        expected = (
            "%output, %count = test.invoke @compute(%weights, %input)"
            " : (tile<4xf32>, index) -> (%weights as tile<4xf32>, index)"
        )
        assert _printer().print_operation(op, module) == expected


class TestPrintAttrDict:
    """Exercises: AttrDict for extra key-value attributes."""

    def test_dict_attrs(self) -> None:
        module, [x, r] = _module_with(("%x", F32), ("%r", F32))
        op = Operation(
            name="test.attrs",
            operands=[x],
            results=[r],
            attributes={"dict": {"axis": 0, "label": "foo"}},
        )
        text = _printer().print_operation(op, module)
        assert text.startswith("%r = test.attrs %x")
        assert "axis = 0" in text
        assert 'label = "foo"' in text
        assert text.endswith(" : f32")

    def test_empty_dict(self) -> None:
        module, [x, r] = _module_with(("%x", F32), ("%r", F32))
        op = Operation(
            name="test.attrs",
            operands=[x],
            results=[r],
            attributes={"dict": {}},
        )
        text = _printer().print_operation(op, module)
        assert "{" not in text
        assert text == "%r = test.attrs %x : f32"

    def test_no_dict_attr(self) -> None:
        module, [x, r] = _module_with(("%x", F32), ("%r", F32))
        op = Operation(name="test.attrs", operands=[x], results=[r], attributes={})
        text = _printer().print_operation(op, module)
        assert "{" not in text
        assert text == "%r = test.attrs %x : f32"


# ============================================================================
# String attribute escaping
# ============================================================================


class TestStringEscaping:
    """String attributes must escape special characters for round-trip safety."""

    def test_quotes_escaped(self) -> None:
        module, [x, r] = _module_with(("%x", F32), ("%r", F32))
        op = Operation(
            name="test.attrs",
            operands=[x],
            results=[r],
            attributes={"dict": {"msg": 'has "quotes"'}},
        )
        text = _printer().print_operation(op, module)
        assert r"has \"quotes\"" in text

    def test_backslash_escaped(self) -> None:
        module, [x, r] = _module_with(("%x", F32), ("%r", F32))
        op = Operation(
            name="test.attrs",
            operands=[x],
            results=[r],
            attributes={"dict": {"path": r"C:\Users\test"}},
        )
        text = _printer().print_operation(op, module)
        assert r"C:\\Users\\test" in text

    def test_newline_escaped(self) -> None:
        module, [x, r] = _module_with(("%x", F32), ("%r", F32))
        op = Operation(
            name="test.attrs",
            operands=[x],
            results=[r],
            attributes={"dict": {"msg": "line1\nline2"}},
        )
        text = _printer().print_operation(op, module)
        assert r"line1\nline2" in text

    def test_tab_escaped(self) -> None:
        module, [x, r] = _module_with(("%x", F32), ("%r", F32))
        op = Operation(
            name="test.attrs",
            operands=[x],
            results=[r],
            attributes={"dict": {"msg": "col1\tcol2"}},
        )
        text = _printer().print_operation(op, module)
        assert r"col1\tcol2" in text


# ============================================================================
# Region printing
# ============================================================================


class TestRegionPrinting:
    """Regions must print inline with correct indentation and separators."""

    def test_single_region_with_body(self) -> None:
        module, [cond] = _module_with(("%cond", I1))
        tv = module.add_value(Value(name="%tv", type=F32))
        fv = module.add_value(Value(name="%fv", type=F32))
        r = module.add_value(Value(name="%r", type=F32))
        then_block = Block(
            ops=[Operation(name="test.yield", operands=[tv], results=[])]
        )
        else_block = Block(
            ops=[Operation(name="test.yield", operands=[fv], results=[])]
        )
        then_region = Region(blocks=[then_block])
        else_region = Region(blocks=[else_block])
        op = Operation(
            name="test.branch",
            operands=[cond],
            results=[r],
            regions=[then_region, else_region],
        )
        text = _printer().print_operation(op, module)
        assert "{\n" in text
        assert "  test.yield %tv" in text
        assert "else" in text

    def test_two_regions_with_else_separator(self) -> None:
        module, [cond] = _module_with(("%cond", I1))
        tv = module.add_value(Value(name="%tv", type=F32))
        fv = module.add_value(Value(name="%fv", type=F32))
        r = module.add_value(Value(name="%r", type=F32))
        then_block = Block(
            ops=[Operation(name="test.yield", operands=[tv], results=[])]
        )
        else_block = Block(
            ops=[Operation(name="test.yield", operands=[fv], results=[])]
        )
        then_region = Region(blocks=[then_block])
        else_region = Region(blocks=[else_block])
        op = Operation(
            name="test.branch",
            operands=[cond],
            results=[r],
            regions=[then_region, else_region],
        )
        text = _printer().print_operation(op, module)
        # Must have } else { between regions, matching C printer output.
        assert "} else {" in text
        assert "  test.yield %tv" in text
        assert "  test.yield %fv" in text

    def test_region_after_result_types(self) -> None:
        """test.map: region appears before -> (type) in the format."""
        module = Module(name="test")
        tile_type = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
        input_id = module.add_value(Value(name="%input", type=tile_type))
        elem_id = module.add_value(Value(name="%elem", type=F32))
        r = module.add_value(Value(name="%r", type=tile_type))
        body_block = Block(
            arg_ids=[elem_id],
            ops=[Operation(name="test.yield", operands=[elem_id], results=[])],
        )
        body = Region(blocks=[body_block])
        op = Operation(
            name="test.map",
            operands=[input_id],
            results=[r],
            regions=[body],
        )
        text = _printer().print_operation(op, module)
        # The region body must appear BEFORE -> (type), not after.
        brace_pos = text.find("{")
        arrow_pos = text.find("->")
        assert brace_pos >= 0 and arrow_pos >= 0
        assert brace_pos < arrow_pos, (
            f"Region body must appear before -> but got:\n{text}"
        )


# ============================================================================
# Auto-naming
# ============================================================================


class TestAutoNaming:
    """Unnamed values get auto-names based on their defining op."""

    def test_unnamed_results(self) -> None:
        module, [a, b] = _module_with(("", I32), ("", I32))
        r_id = module.add_value(Value(name="", type=I32))
        op = Operation(name="test.addi", operands=[a, b], results=[r_id])
        text = _printer().print_operation(op, module)
        # Unnamed operands get auto-names, result gets %addi0.
        assert "= test.addi" in text
        assert ": i32" in text

    def test_mixed_named_unnamed(self) -> None:
        module, [a] = _module_with(("%x", I32))
        b_id = module.add_value(Value(name="", type=I32))
        r_id = module.add_value(Value(name="", type=I32))
        op = Operation(name="test.addi", operands=[a, b_id], results=[r_id])
        text = _printer().print_operation(op, module)
        # %x is preserved, others get auto-names.
        assert "%x" in text
        assert "= test.addi" in text

    def test_user_and_auto_names_coexist(self) -> None:
        # User name (identifier) and auto name (digit-only) are in
        # separate syntactic namespaces — no collision possible.
        module, [user_named] = _module_with(("%x", I32))
        auto_id = module.add_value(Value(name="", type=I32))
        r_id = module.add_value(Value(name="", type=I32))
        op = Operation(name="test.addi", operands=[user_named, auto_id], results=[r_id])
        text = _printer().print_operation(op, module)
        assert "%x" in text  # User name preserved.
        assert "%1" in text  # Auto name is %{value_id}.
        assert "%2" in text  # Result auto name.


# ============================================================================
# Token spacing edge cases
# ============================================================================


class TestSpacing:
    def test_no_space_before_comma(self) -> None:
        module, [a, b, r] = _module_with(("%a", I32), ("%b", I32), ("%r", I32))
        op = Operation(name="test.addi", operands=[a, b], results=[r])
        text = _printer().print_operation(op, module)
        # Should be "%a, %b" not "%a , %b".
        assert "%a, %b" in text

    def test_space_before_colon(self) -> None:
        module, [x, r] = _module_with(("%x", F32), ("%r", F32))
        op = Operation(name="test.neg", operands=[x], results=[r])
        text = _printer().print_operation(op, module)
        assert "%x : f32" in text

    def test_index_list_glues(self) -> None:
        module, [src, r] = _module_with(
            ("%src", _tile_64x64xf16), ("%r", _tile_16x16xf16)
        )
        op = Operation(
            name="test.slice",
            operands=[src],
            results=[r],
            attributes={"static_offsets": [0, 0]},
        )
        text = _printer().print_operation(op, module)
        # '[' should glue to %src, no space.
        assert "%src[0, 0]" in text
        assert "%src [" not in text

    def test_parens_glue(self) -> None:
        module, [w, x, out] = _module_with(
            ("%w", _tile_4xf32), ("%x", INDEX), ("%out", _tile_4xf32)
        )
        op = Operation(
            name="test.invoke",
            operands=[w, x],
            results=[out],
            attributes={"callee": "f"},
        )
        text = _printer().print_operation(op, module)
        # '(' should glue to @f.
        assert "@f(%w" in text
        # ')' should glue to last operand.
        assert "%x)" in text

    def test_deflate_ordinal_dim(self) -> None:
        """test.deflate prints result ordinal dim as [#1]."""
        tensor_dyn = ShapedType(TypeKind.TENSOR, F32, (DynamicDim(),))
        module = Module(name="test")
        # Input: tensor<[%M]xf32> — has a named dynamic dim.
        m_id = module.add_value(Value(name="%M", type=INDEX))
        input_id = module.add_value(
            Value(name="%input", type=tensor_dyn, dim_bindings={0: m_id})
        )
        # Result 0: tensor<[#1]xf32> — ordinal dim referencing result 1.
        output_id = module.add_value(
            Value(name="%output", type=tensor_dyn, dim_bindings={0: -2})
        )
        # Result 1: index.
        length_id = module.add_value(Value(name="%length", type=INDEX))
        op = Operation(
            name="test.deflate",
            operands=[input_id],
            results=[output_id, length_id],
        )
        text = _printer().print_operation(op, module)
        assert text == (
            "%output, %length = test.deflate %input"
            " : tensor<[%M]xf32> -> (tensor<[#1]xf32>, index)"
        )

    def test_result_type_list_always_parens(self) -> None:
        module, [src, r] = _module_with(
            ("%src", _tile_64x64xf16), ("%r", _tile_16x16xf16)
        )
        op = Operation(
            name="test.slice",
            operands=[src],
            results=[r],
            attributes={"static_offsets": [0, 0]},
        )
        text = _printer().print_operation(op, module)
        # Single result still gets parens.
        assert "-> (tile<16x16xf16>)" in text


# ============================================================================
# Encoding type printing
# ============================================================================


class TestEncodingTypePrinting:
    def test_encoding_type(self) -> None:
        assert print_type(ENCODING_TYPE) == "encoding"

    def test_dynamic_encoding_with_context(self) -> None:
        """ShapedType with DynamicEncoding + encoding_binding prints %name."""
        module = Module(name="test")
        enc_id = module.add_value(Value(name="%enc", type=ENCODING_TYPE))
        tile_type = ShapedType(
            TypeKind.TILE, F32, (StaticDim(4),), encoding=DynamicEncoding()
        )
        module.add_value(Value(name="%t", type=tile_type, encoding_binding=enc_id))
        from loom.format.text.printer import TypePrintContext

        context = TypePrintContext(
            dim_bindings={}, module=module, encoding_binding=enc_id
        )
        text = print_type(tile_type, context)
        assert text == "tile<4xf32, %enc>"

    def test_dynamic_encoding_without_context(self) -> None:
        """ShapedType with DynamicEncoding without context prints ?."""
        tile_type = ShapedType(
            TypeKind.TILE, F32, (StaticDim(4),), encoding=DynamicEncoding()
        )
        text = print_type(tile_type)
        assert text == "tile<4xf32, ?>"

    def test_dynamic_encoding_in_operation(self) -> None:
        """Full op printing with dynamic encoding on a value type."""
        module = Module(name="test")
        enc_id = module.add_value(Value(name="%enc", type=ENCODING_TYPE))
        tile_type = ShapedType(
            TypeKind.TILE, F32, (StaticDim(4),), encoding=DynamicEncoding()
        )
        lhs_id = module.add_value(
            Value(name="%lhs", type=tile_type, encoding_binding=enc_id)
        )
        rhs_id = module.add_value(
            Value(name="%rhs", type=tile_type, encoding_binding=enc_id)
        )
        result_id = module.add_value(
            Value(name="%result", type=tile_type, encoding_binding=enc_id)
        )
        op = Operation(
            name="test.addi",
            operands=[lhs_id, rhs_id],
            results=[result_id],
        )
        text = _printer().print_operation(op, module)
        assert text == "%result = test.addi %lhs, %rhs : tile<4xf32, %enc>"


# ============================================================================
# Pool type printing
# ============================================================================


class TestPoolTypePrinting:
    def test_static_pool(self) -> None:
        assert print_type(PoolType(StaticDim(65536))) == "pool<65536>"

    def test_static_pool_small(self) -> None:
        assert print_type(PoolType(StaticDim(4096))) == "pool<4096>"

    def test_dynamic_pool_no_context(self) -> None:
        """Without context, dynamic block_size prints as ?."""
        assert print_type(PoolType(DynamicDim())) == "pool<?>"

    def test_dynamic_pool_with_context(self) -> None:
        """With context, dynamic block_size prints as [%name]."""
        from loom.format.text.printer import TypePrintContext

        module = Module(name="test")
        module.add_value(Value(name="%BS", type=INDEX))
        context = TypePrintContext(
            dim_bindings={0: 0},
            module=module,
        )
        assert print_type(PoolType(DynamicDim()), context) == "pool<[%BS]>"

    def test_pool_in_operation(self) -> None:
        """Pool type prints correctly as an op operand type."""
        module = Module(name="test")
        bs_id = module.add_value(Value(name="%BS", type=INDEX))
        pool_type = PoolType(DynamicDim())
        pool_id = module.add_value(
            Value(name="%pool", type=pool_type, dim_bindings={0: bs_id})
        )
        result_id = module.add_value(
            Value(name="%result", type=pool_type, dim_bindings={0: bs_id})
        )
        op = Operation(
            name="test.attrs",
            operands=[pool_id],
            results=[result_id],
        )
        text = _printer().print_operation(op, module)
        assert text == "%result = test.attrs %pool : pool<[%BS]>"


# ============================================================================
# Printer flags
# ============================================================================


class TestPrinterFlags:
    """Tests for printer control flags: use_aliases, indent, print_regions."""

    def test_use_aliases_true_default(self) -> None:
        """Default: encoding alias is used when available."""
        enc = EncodingInstance(name="q8_0", alias="#enc", params=(("block", "32"),))
        shaped = ShapedType(TypeKind.TILE, I8, (StaticDim(256),), encoding=enc)
        assert print_type(shaped) == "tile<256xi8, #enc>"

    def test_use_aliases_false(self) -> None:
        """With use_aliases=False, full #name<params> form is printed."""
        from loom.format.text.printer import TypePrintContext

        enc = EncodingInstance(name="q8_0", alias="#enc", params=(("block", "32"),))
        shaped = ShapedType(TypeKind.TILE, I8, (StaticDim(256),), encoding=enc)
        context = TypePrintContext({}, Module(), use_aliases=False)
        assert print_type(shaped, context) == "tile<256xi8, #q8_0<block=32>>"

    def test_use_aliases_false_no_params(self) -> None:
        """With use_aliases=False and no params, prints bare #name."""
        from loom.format.text.printer import TypePrintContext

        enc = EncodingInstance(name="dense", alias="#d")
        shaped = ShapedType(TypeKind.TILE, F32, (StaticDim(4),), encoding=enc)
        context = TypePrintContext({}, Module(), use_aliases=False)
        assert print_type(shaped, context) == "tile<4xf32, #dense>"

    def test_use_aliases_false_in_printer(self) -> None:
        """Printer with use_aliases=False expands aliases in op output."""
        module, [x_id] = _module_with(("%x", I8))
        enc = EncodingInstance(name="q8_0", alias="#enc", params=(("block", "32"),))
        enc_tile = ShapedType(TypeKind.TILE, I8, (StaticDim(256),), encoding=enc)
        r_id = module.add_value(Value(name="%r", type=enc_tile))
        op = Operation(
            name="test.neg",
            operands=[x_id],
            results=[r_id],
        )
        printer = Printer(use_aliases=False)
        printer.register_ops(ALL_TEST_OPS)
        text = printer.print_operation(op, module)
        assert "#q8_0<block=32>" in text
        assert "#enc" not in text

    def test_indent_true_default(self) -> None:
        """Default: ops in regions are indented 2 spaces."""
        module = _module_parser().parse(
            "func.def @f(%x: f32) -> (f32) {\n  test.yield %x : f32\n}\n"
        )
        text = _module_printer().print_module(module)
        lines = text.strip().split("\n")
        # Body ops should be indented.
        assert any(line.startswith("  ") for line in lines)

    def test_indent_false(self) -> None:
        """With indent=False, no leading spaces on any line."""
        module = _module_parser().parse(
            "func.def @f(%x: f32) -> (f32) {\n  test.yield %x : f32\n}\n"
        )
        text = _module_printer(indent=False).print_module(module)
        for line in text.strip().split("\n"):
            assert not line.startswith("  "), f"Line has indentation: {line!r}"

    def test_print_regions_false_module(self) -> None:
        """print_module with print_regions=False prints { ... } placeholders."""
        module = _module_parser().parse(
            "func.def @negate(%input: f32) -> (f32) {\n"
            "  %neg0 = test.neg %input : f32\n"
            "  test.yield %neg0 : f32\n"
            "}\n"
        )

        text = _module_printer(print_regions=False).print_module(module)
        assert "{ ... }" in text
        assert "test.neg" not in text

    def test_defaults_match_current_behavior(self) -> None:
        """Default flags produce identical output to a bare Printer()."""
        module = _module_parser().parse(
            "func.def @negate(%input: f32) -> (f32) {\n"
            "  %neg0 = test.neg %input : f32\n"
            "  test.yield %neg0 : f32\n"
            "}\n"
        )
        default_output = _module_printer().print_module(module)
        explicit_output = _module_printer(
            print_locations=False, use_aliases=True, indent=True, print_regions=True
        ).print_module(module)
        assert default_output == explicit_output
