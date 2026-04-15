# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLVM IR target dialect op definitions."""

from __future__ import annotations

from loom.assembly import (
    ARROW,
    COLON,
    COMMA,
    LPAREN,
    RPAREN,
    Attr,
    Flags,
    OptionalGroup,
    Refs,
    ResultTypeList,
    TypesOf,
)
from loom.dsl import (
    ANY,
    ATTR_TYPE_FLAGS,
    ATTR_TYPE_STRING,
    UNKNOWN_EFFECTS,
    AttrDef,
    Dialect,
    EnumCase,
    EnumDef,
    Op,
    Operand,
    Result,
)

llvmir_ops = Dialect(
    "llvmir",
    dialect_id=0x11,
    doc=("LLVM IR target punch-through operations. These ops preserve target specific intent in Loom IR while still lowering through structured LLVMIR builders instead of raw textual emission."),
)

AsmFlags = EnumDef(
    "AsmFlags",
    [
        EnumCase("sideeffect", 1, doc="Inline asm has side effects."),
        EnumCase("alignstack", 2, doc="Inline asm may require stack realignment."),
        EnumCase("inteldialect", 4, doc="Inline asm uses Intel assembly syntax."),
    ],
    doc="LLVM inline asm call flags.",
)

llvmir_inline_asm = Op(
    "llvmir.inline_asm",
    group=llvmir_ops,
    doc=("Structured LLVM inline assembly call. The asm template and constraint strings use LLVM inline asm syntax; operands/results remain ordinary typed Loom SSA values."),
    operands=[Operand("operands", ANY, variadic=True)],
    results=[Result("results", ANY, variadic=True)],
    attrs=[
        AttrDef("flags", ATTR_TYPE_FLAGS, optional=True, enum_def=AsmFlags),
        AttrDef(
            "asm_template",
            ATTR_TYPE_STRING,
            doc="LLVM inline asm template string.",
        ),
        AttrDef(
            "constraints",
            ATTR_TYPE_STRING,
            doc="LLVM inline asm constraint string.",
        ),
    ],
    traits=[UNKNOWN_EFFECTS],
    format=[
        Flags("flags"),
        Attr("asm_template"),
        COMMA,
        Attr("constraints"),
        LPAREN,
        Refs("operands"),
        RPAREN,
        COLON,
        LPAREN,
        TypesOf("operands"),
        RPAREN,
        OptionalGroup([ARROW, ResultTypeList("results", parens=False)], anchor="results"),
    ],
    examples=[
        '%sum = llvmir.inline_asm<sideeffect> "addl $2, $0", "=r,r,r"(%lhs, %rhs) : (i32, i32) -> i32',
    ],
)

ALL_LLVMIR_OPS = (llvmir_inline_asm,)
