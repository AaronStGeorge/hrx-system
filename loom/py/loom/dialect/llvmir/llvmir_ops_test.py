# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the LLVM IR target dialect declarations."""

from loom.assembly import Flags, ResultTypeList
from loom.dialect.llvmir import (
    ALL_LLVMIR_OPS,
    AsmFlags,
    llvmir_inline_asm,
    llvmir_ops,
)
from loom.dsl import ANY, ATTR_TYPE_FLAGS, ATTR_TYPE_STRING, UNKNOWN_EFFECTS


class TestLlvmIrDialect:
    def test_dialect_id(self) -> None:
        assert llvmir_ops.dialect_id == 0x11

    def test_inventory(self) -> None:
        assert [op.name for op in ALL_LLVMIR_OPS] == ["llvmir.inline_asm"]

    def test_public_exports_match_registry(self) -> None:
        assert llvmir_inline_asm in ALL_LLVMIR_OPS

    def test_inline_asm_shape(self) -> None:
        op = llvmir_inline_asm
        assert [operand.name for operand in op.operands] == ["operands"]
        assert op.operands[0].type_constraint == ANY
        assert op.operands[0].variadic
        assert [result.name for result in op.results] == ["results"]
        assert op.results[0].type_constraint == ANY
        assert op.results[0].variadic
        assert [attr.name for attr in op.attrs] == [
            "flags",
            "asm_template",
            "constraints",
        ]
        assert op.attrs[0].attr_type == ATTR_TYPE_FLAGS
        assert op.attrs[0].optional
        assert op.attrs[0].enum_def is AsmFlags
        assert op.attrs[1].attr_type == ATTR_TYPE_STRING
        assert op.attrs[2].attr_type == ATTR_TYPE_STRING
        assert UNKNOWN_EFFECTS in op.traits
        assert any(isinstance(element, Flags) for element in op.format)
        assert any(isinstance(element, ResultTypeList) for element in op.format)

    def test_inline_asm_flags_are_bitmasks(self) -> None:
        assert [(case.keyword, case.value) for case in AsmFlags.cases] == [
            ("sideeffect", 1),
            ("alignstack", 2),
            ("inteldialect", 4),
        ]
