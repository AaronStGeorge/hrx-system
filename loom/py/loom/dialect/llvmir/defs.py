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
    TemplateParam,
    TypesOf,
)
from loom.dsl import (
    ANY,
    ATTR_TYPE_ENUM,
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

IntrinsicKind = EnumDef(
    "IntrinsicKind",
    [
        EnumCase("llvm.x86.rdtsc", 0, doc="Returns the x86 timestamp counter."),
        EnumCase("llvm.x86.sse2.pause", 1, doc="Emits an x86 pause hint."),
        EnumCase(
            "llvm.amdgcn.workitem.id.x",
            2,
            doc="Returns the AMDGPU workitem id in the x dimension.",
        ),
        EnumCase(
            "llvm.amdgcn.workitem.id.y",
            3,
            doc="Returns the AMDGPU workitem id in the y dimension.",
        ),
        EnumCase(
            "llvm.amdgcn.workitem.id.z",
            4,
            doc="Returns the AMDGPU workitem id in the z dimension.",
        ),
        EnumCase(
            "llvm.memcpy",
            5,
            doc="Copies bytes between pointer operands using LLVM's overloaded memcpy intrinsic.",
        ),
        EnumCase(
            "llvm.memset",
            6,
            doc="Fills bytes through a pointer operand using LLVM's overloaded memset intrinsic.",
        ),
        EnumCase(
            "llvm.lifetime.start",
            7,
            doc="Marks the start of a pointer object's lifetime.",
        ),
        EnumCase(
            "llvm.lifetime.end",
            8,
            doc="Marks the end of a pointer object's lifetime.",
        ),
    ],
    doc="LLVM intrinsic selected by llvmir.intrinsic.",
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

llvmir_intrinsic = Op(
    "llvmir.intrinsic",
    group=llvmir_ops,
    doc=("Structured call to a supported LLVM intrinsic. The intrinsic kind is an enum so target-specific spellings stay explicit while lowering still goes through the LLVMIR intrinsic catalog."),
    operands=[Operand("operands", ANY, variadic=True)],
    results=[Result("results", ANY, variadic=True)],
    attrs=[
        AttrDef("kind", ATTR_TYPE_ENUM, enum_def=IntrinsicKind),
    ],
    traits=[UNKNOWN_EFFECTS],
    format=[
        TemplateParam("kind"),
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
        "%ticks = llvmir.intrinsic<llvm.x86.rdtsc> () : () -> i64",
        "llvmir.intrinsic<llvm.x86.sse2.pause> () : ()",
        "llvmir.intrinsic<llvm.memcpy> (%target, %source, %length, %is_volatile) : (!buffer, !buffer, offset, i1)",
    ],
)

ALL_LLVMIR_OPS = (
    llvmir_inline_asm,
    llvmir_intrinsic,
)
