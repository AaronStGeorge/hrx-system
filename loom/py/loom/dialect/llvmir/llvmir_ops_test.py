# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the LLVM IR target dialect declarations."""

from loom.assembly import Flags, FormatElement, OptionalGroup, ResultTypeList, Scope, TemplateParam
from loom.dialect.llvmir import (
    ALL_LLVMIR_OPS,
    AsmFlags,
    IntrinsicKind,
    llvmir_inline_asm,
    llvmir_intrinsic,
    llvmir_ops,
)
from loom.dsl import ANY, ATTR_TYPE_ENUM, ATTR_TYPE_FLAGS, ATTR_TYPE_STRING, UNKNOWN_EFFECTS


def _format_contains(elements: tuple[FormatElement, ...], element_type: type[object]) -> bool:
    for element in elements:
        if isinstance(element, element_type):
            return True
        if isinstance(element, OptionalGroup | Scope) and _format_contains(element.elements, element_type):
            return True
    return False


class TestLlvmIrDialect:
    def test_dialect_id(self) -> None:
        assert llvmir_ops.dialect_id == 0x11

    def test_inventory(self) -> None:
        assert [op.name for op in ALL_LLVMIR_OPS] == [
            "llvmir.inline_asm",
            "llvmir.intrinsic",
        ]

    def test_public_exports_match_registry(self) -> None:
        assert llvmir_inline_asm in ALL_LLVMIR_OPS
        assert llvmir_intrinsic in ALL_LLVMIR_OPS

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
        assert _format_contains(op.format, ResultTypeList)

    def test_inline_asm_flags_are_bitmasks(self) -> None:
        assert [(case.keyword, case.value) for case in AsmFlags.cases] == [
            ("sideeffect", 1),
            ("alignstack", 2),
            ("inteldialect", 4),
        ]

    def test_intrinsic_shape(self) -> None:
        op = llvmir_intrinsic
        assert [operand.name for operand in op.operands] == ["operands"]
        assert op.operands[0].type_constraint == ANY
        assert op.operands[0].variadic
        assert [result.name for result in op.results] == ["results"]
        assert op.results[0].type_constraint == ANY
        assert op.results[0].variadic
        assert [attr.name for attr in op.attrs] == ["kind"]
        assert op.attrs[0].attr_type == ATTR_TYPE_ENUM
        assert op.attrs[0].enum_def is IntrinsicKind
        assert UNKNOWN_EFFECTS in op.traits
        assert any(isinstance(element, TemplateParam) for element in op.format)
        assert _format_contains(op.format, ResultTypeList)

    def test_intrinsic_kinds_use_llvm_spellings(self) -> None:
        assert [(case.keyword, case.value) for case in IntrinsicKind.cases] == [
            ("llvm.x86.rdtsc", 0),
            ("llvm.x86.sse2.pause", 1),
            ("llvm.amdgcn.workitem.id.x", 2),
            ("llvm.amdgcn.workitem.id.y", 3),
            ("llvm.amdgcn.workitem.id.z", 4),
            ("llvm.memcpy", 5),
            ("llvm.memset", 6),
            ("llvm.lifetime.start", 7),
            ("llvm.lifetime.end", 8),
        ]
