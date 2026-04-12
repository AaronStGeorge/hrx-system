# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the encoding dialect ops."""

from loom.dialect.encoding import (
    ALL_ENCODING_OPS,
    encoding_define,
    encoding_isa,
    encoding_ops,
)
from loom.dsl import (
    ATTR_TYPE_DICT,
    ATTR_TYPE_ENCODING,
    ENCODING,
    TypeConstraint,
)


class TestAllOpsRegistered:
    def test_count(self) -> None:
        assert len(ALL_ENCODING_OPS) == 2

    def test_unique_names(self) -> None:
        names = [op.name for op in ALL_ENCODING_OPS]
        assert len(set(names)) == len(names)

    def test_all_in_encoding_namespace(self) -> None:
        for op in ALL_ENCODING_OPS:
            assert op.namespace == "encoding", f"{op.name} not in encoding namespace"

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


class TestEncodingDefine:
    def test_name(self) -> None:
        assert encoding_define.name == "encoding.define"

    def test_result_is_encoding(self) -> None:
        assert len(encoding_define.results) == 1
        assert encoding_define.results[0].type_constraint == ENCODING

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


class TestEncodingIsa:
    def test_name(self) -> None:
        assert encoding_isa.name == "encoding.isa"

    def test_operand_is_encoding(self) -> None:
        assert len(encoding_isa.operands) == 1
        assert encoding_isa.operands[0].type_constraint == ENCODING

    def test_result_is_integer(self) -> None:
        assert len(encoding_isa.results) == 1
        assert encoding_isa.results[0].type_constraint == TypeConstraint.INTEGER

    def test_has_category_attr(self) -> None:
        attr = encoding_isa.attr("category")
        assert attr is not None
        assert attr.attr_type == "string"

    def test_is_pure(self) -> None:
        assert encoding_isa.is_pure
