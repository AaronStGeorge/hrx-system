# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the encoding dialect ops."""

from collections.abc import Sequence

import pytest

from loom.builtin_types import ALL_BUILTIN_TYPES
from loom.dialect.encoding import (
    ALL_ENCODING_OPS,
    encoding_assume_spec,
    encoding_define,
    encoding_isa,
    encoding_layout_assume_dense,
    encoding_layout_assume_strided,
    encoding_layout_dense,
    encoding_layout_strided,
    encoding_ops,
)
from loom.dialect.test import ALL_TEST_OPS
from loom.dsl import (
    ANY_ENCODING,
    ATTR_TYPE_DICT,
    ATTR_TYPE_ENCODING,
    ATTR_TYPE_I64,
    ATTR_TYPE_I64_ARRAY,
    ENCODING_LAYOUT,
    I1,
    Op,
    TypeConstraint,
)
from loom.format.text.parser import Parser
from loom.format.text.printer import Printer
from loom.format.text.tokenizer import ParseError
from loom.ir import CanonicalAttrDict, EncodingInstance, Module


def _all_roundtrip_ops() -> Sequence[Op]:
    return list(ALL_TEST_OPS) + list(ALL_ENCODING_OPS)


def _parse_module(text: str) -> Module:
    parser = Parser()
    parser.register_ops(_all_roundtrip_ops())
    parser.register_types(ALL_BUILTIN_TYPES)
    return parser.parse(text)


def _print_module(module: Module) -> str:
    printer = Printer()
    printer.register_ops(_all_roundtrip_ops())
    printer.register_types(ALL_BUILTIN_TYPES)
    return printer.print_module(module)


def _module_text(*lines: str) -> str:
    return "\n".join(lines) + "\n"


class TestAllOpsRegistered:
    def test_count(self) -> None:
        assert len(ALL_ENCODING_OPS) == 7

    def test_expected_ops_registered_in_order(self) -> None:
        assert [op.name for op in ALL_ENCODING_OPS] == [
            "encoding.layout.dense",
            "encoding.layout.strided",
            "encoding.define",
            "encoding.isa",
            "encoding.layout.assume.dense",
            "encoding.layout.assume.strided",
            "encoding.assume.spec",
        ]

    def test_public_exports_match_registry(self) -> None:
        assert encoding_layout_dense in ALL_ENCODING_OPS
        assert encoding_layout_strided in ALL_ENCODING_OPS
        assert encoding_define in ALL_ENCODING_OPS
        assert encoding_isa in ALL_ENCODING_OPS
        assert encoding_layout_assume_dense in ALL_ENCODING_OPS
        assert encoding_layout_assume_strided in ALL_ENCODING_OPS
        assert encoding_assume_spec in ALL_ENCODING_OPS

    def test_unique_names(self) -> None:
        names = [op.name for op in ALL_ENCODING_OPS]
        assert len(set(names)) == len(names)

    def test_all_in_encoding_dialect(self) -> None:
        for op in ALL_ENCODING_OPS:
            assert op.name.startswith("encoding."), f"{op.name} not in encoding dialect"

    def test_all_have_group(self) -> None:
        for op in ALL_ENCODING_OPS:
            assert op.group is encoding_ops

    def test_all_have_docs(self) -> None:
        for op in ALL_ENCODING_OPS:
            assert op.doc

    def test_all_have_format(self) -> None:
        for op in ALL_ENCODING_OPS:
            assert op.format

    def test_all_have_examples(self) -> None:
        for op in ALL_ENCODING_OPS:
            assert op.examples


class TestEncodingLayoutDense:
    def test_name(self) -> None:
        assert encoding_layout_dense.name == "encoding.layout.dense"

    def test_result_is_encoding(self) -> None:
        assert len(encoding_layout_dense.results) == 1
        assert encoding_layout_dense.results[0].type_constraint == ENCODING_LAYOUT

    def test_is_pure(self) -> None:
        assert encoding_layout_dense.is_pure


class TestEncodingLayoutStrided:
    def test_name(self) -> None:
        assert encoding_layout_strided.name == "encoding.layout.strided"

    def test_dynamic_strides_variadic(self) -> None:
        operand = encoding_layout_strided.operand("strides")
        assert operand is not None
        assert operand.variadic
        assert operand.type_constraint == TypeConstraint.INDEX

    def test_static_strides_attr(self) -> None:
        attr = encoding_layout_strided.attr("static_strides")
        assert attr is not None
        assert attr.attr_type == ATTR_TYPE_I64_ARRAY

    def test_result_is_encoding(self) -> None:
        assert len(encoding_layout_strided.results) == 1
        assert encoding_layout_strided.results[0].type_constraint == ENCODING_LAYOUT

    def test_is_pure(self) -> None:
        assert encoding_layout_strided.is_pure


class TestEncodingLayoutAssumeDense:
    def test_name(self) -> None:
        assert encoding_layout_assume_dense.name == "encoding.layout.assume.dense"

    def test_operand_and_result_are_layout_encodings(self) -> None:
        assert len(encoding_layout_assume_dense.operands) == 1
        assert len(encoding_layout_assume_dense.results) == 1
        assert encoding_layout_assume_dense.operands[0].type_constraint == ENCODING_LAYOUT
        assert encoding_layout_assume_dense.results[0].type_constraint == ENCODING_LAYOUT

    def test_is_pure(self) -> None:
        assert encoding_layout_assume_dense.is_pure


class TestEncodingLayoutAssumeStrided:
    def test_name(self) -> None:
        assert encoding_layout_assume_strided.name == "encoding.layout.assume.strided"

    def test_operand_and_result_are_layout_encodings(self) -> None:
        assert len(encoding_layout_assume_strided.operands) == 1
        assert len(encoding_layout_assume_strided.results) == 1
        assert encoding_layout_assume_strided.operands[0].type_constraint == ENCODING_LAYOUT
        assert encoding_layout_assume_strided.results[0].type_constraint == ENCODING_LAYOUT

    def test_has_rank_attr(self) -> None:
        attr = encoding_layout_assume_strided.attr("rank")
        assert attr is not None
        assert attr.attr_type == ATTR_TYPE_I64

    def test_is_pure(self) -> None:
        assert encoding_layout_assume_strided.is_pure


class TestEncodingDefine:
    def test_name(self) -> None:
        assert encoding_define.name == "encoding.define"

    def test_result_is_encoding(self) -> None:
        assert len(encoding_define.results) == 1
        assert encoding_define.results[0].type_constraint == ANY_ENCODING

    def test_has_spec_attr(self) -> None:
        attr = encoding_define.attr("spec")
        assert attr is not None
        assert attr.attr_type == ATTR_TYPE_ENCODING

    def test_dynamic_params_variadic(self) -> None:
        operand = encoding_define.operand("params")
        assert operand is not None
        assert operand.variadic

    def test_dynamic_param_names_attr(self) -> None:
        attr = encoding_define.attr("param_names")
        assert attr is not None
        assert attr.attr_type == ATTR_TYPE_DICT
        assert attr.optional

    def test_is_pure(self) -> None:
        assert encoding_define.is_pure


class TestEncodingDefineRoundTrip:
    def test_inline_spec_attr(self) -> None:
        module = _parse_module(
            _module_text(
                "test.func @f() {",
                "  %enc = encoding.define #q8_0<block=32> : encoding<schema>",
                "  test.yield",
                "}",
            )
        )
        func_op = module.symbols[0].op
        assert func_op is not None
        define_op = func_op.regions[0].blocks[0].ops[0]
        assert define_op.name == "encoding.define"
        assert define_op.attributes["spec"] == EncodingInstance(
            name="q8_0",
            params=(("block", 32),),
        )
        assert module.encodings == [EncodingInstance(name="q8_0", params=(("block", 32),))]

    def test_dynamic_params_print_in_canonical_order(self) -> None:
        module = _parse_module(
            "test.func @f(%group_size: index, %scales: tensor<[%group_size]xf32>) {\n"
            "  %enc = encoding.define #q8_0<block=32> "
            "{scales = %scales : tensor<[%group_size]xf32>, "
            "group_size = %group_size : index} : encoding<schema>\n"
            "  test.yield\n"
            "}\n"
        )
        printed = _print_module(module)
        assert printed == (
            "test.func @f(%group_size: index, %scales: tensor<[%group_size]xf32>) {\n"
            "  %enc = encoding.define #q8_0<block=32> "
            "{group_size = %group_size : index, "
            "scales = %scales : tensor<[%group_size]xf32>} : encoding<schema>\n"
            "  test.yield\n"
            "}\n"
        )
        func_op = module.symbols[0].op
        assert func_op is not None
        define_op = func_op.regions[0].blocks[0].ops[0]
        assert define_op.name == "encoding.define"
        assert define_op.attributes["param_names"] == CanonicalAttrDict((("group_size", 0), ("scales", 1)))

    def test_rejects_static_dynamic_duplicate(self) -> None:
        with pytest.raises(ParseError, match="both static and dynamic"):
            _parse_module(
                _module_text(
                    "test.func @f(%block: index) {",
                    "  %enc = encoding.define #q8_0<block=32> {block = %block : index} : encoding<schema>",
                    "  test.yield",
                    "}",
                )
            )

    def test_static_encoding_rejects_ssa_parameter(self) -> None:
        with pytest.raises(ParseError, match="cannot use an SSA value"):
            _parse_module(
                _module_text(
                    "test.func @f(%group_size: index) {",
                    "  %enc = encoding.define #q8_0<group_size=%group_size> : encoding<schema>",
                    "  test.yield",
                    "}",
                )
            )

    def test_alias_spec_attr(self) -> None:
        module = _parse_module(
            _module_text(
                "#enc = #q8_0<block=32>",
                "test.func @f() {",
                "  %enc = encoding.define #enc : encoding<schema>",
                "  test.yield",
                "}",
            )
        )
        func_op = module.symbols[0].op
        assert func_op is not None
        define_op = func_op.regions[0].blocks[0].ops[0]
        assert define_op.attributes["spec"] == EncodingInstance(
            name="q8_0",
            alias="enc",
            params=(("block", 32),),
        )
        assert module.encodings == [
            EncodingInstance(
                name="q8_0",
                alias="enc",
                params=(("block", 32),),
            )
        ]


class TestEncodingIsa:
    def test_name(self) -> None:
        assert encoding_isa.name == "encoding.isa"

    def test_operand_is_encoding(self) -> None:
        assert len(encoding_isa.operands) == 1
        assert encoding_isa.operands[0].type_constraint == ANY_ENCODING

    def test_result_is_i1(self) -> None:
        assert len(encoding_isa.results) == 1
        assert encoding_isa.results[0].type_constraint == I1

    def test_has_category_attr(self) -> None:
        attr = encoding_isa.attr("category")
        assert attr is not None
        assert attr.attr_type == "string"

    def test_is_pure(self) -> None:
        assert encoding_isa.is_pure


class TestEncodingAssumeSpec:
    def test_name(self) -> None:
        assert encoding_assume_spec.name == "encoding.assume.spec"

    def test_operand_and_result_are_encodings(self) -> None:
        assert len(encoding_assume_spec.operands) == 1
        assert len(encoding_assume_spec.results) == 1
        assert encoding_assume_spec.operands[0].type_constraint == ANY_ENCODING
        assert encoding_assume_spec.results[0].type_constraint == ANY_ENCODING

    def test_has_spec_attr(self) -> None:
        attr = encoding_assume_spec.attr("spec")
        assert attr is not None
        assert attr.attr_type == ATTR_TYPE_ENCODING

    def test_is_pure(self) -> None:
        assert encoding_assume_spec.is_pure
