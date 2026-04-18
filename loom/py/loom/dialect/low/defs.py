# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target-low structural dialect op definitions.

The low dialect is a target-bound virtual-register IR that stays inside the
normal Loom module, symbol, bytecode, diagnostic, and pass infrastructure. Its
instruction semantics are descriptor-backed: the IR carries a small set of
structural ops, while target packages define opcode tables, operand classes,
scheduling facts, and emission recipes.
"""

from typing import Any

from loom.assembly import (
    ARROW,
    COLON,
    GLUE,
    LPAREN,
    RPAREN,
    Attr,
    AttrDict,
    FormatElement,
    FuncArgs,
    OpRef,
    OptionalGroup,
    PredicateList,
    Ref,
    Refs,
    Region,
    ResultType,
    ResultTypeList,
    Scope,
    SymbolRef,
    TypeOf,
    TypesOf,
    kw,
)
from loom.dialect.func.defs import CallingConv, Purity, Visibility
from loom.dsl import (
    ANY,
    ATTR_TYPE_ENUM,
    ATTR_TYPE_I64,
    ATTR_TYPE_TYPE,
    ISOLATED_FROM_ABOVE,
    REGISTER,
    SYMBOL_DEFINE,
    TERMINATOR,
    UNKNOWN_EFFECTS,
    AttrDef,
    BlockArgsSatisfy,
    Dialect,
    EnumCase,
    EnumDef,
    FuncLikeInterface,
    Op,
    Operand,
    RegionDef,
    Result,
    SymbolDefinition,
    SymbolReference,
    YieldCountMatchesResults,
    YieldTypesMatchResults,
)

# ============================================================================
# Dialect
# ============================================================================

low_ops = Dialect(
    "low",
    dialect_id=0x14,
    doc="Target-low virtual-register operations.",
)

# ============================================================================
# Shared enums
# ============================================================================

LowAbiConversion = EnumDef(
    "LowAbiConversion",
    [
        EnumCase(
            "direct",
            0,
            doc="Invoke operands/results are already the callee register ABI types.",
        ),
        EnumCase(
            "mapped",
            1,
            doc="Adapter slot records describe semantic and register ABI type crossings.",
        ),
    ],
    doc="Semantic-to-low ABI conversion rule used by a low ABI adapter record.",
)

LowAbiValueConversion = EnumDef(
    "LowAbiValueConversion",
    [
        EnumCase(
            "direct",
            0,
            doc="Semantic and ABI slot types are identical.",
        ),
        EnumCase(
            "scalar_to_register",
            1,
            doc="A semantic scalar is materialized into a callee register ABI slot.",
        ),
        EnumCase(
            "register_to_scalar",
            2,
            doc="A callee register ABI slot is unpacked into a semantic scalar.",
        ),
    ],
    doc="Per-slot conversion rule used by low ABI adapter mapping records.",
)

# ============================================================================
# Shared fragments
# ============================================================================

_FUNC_MODIFIER_ATTRS = [
    AttrDef("callee", "symbol"),
    AttrDef(
        "target",
        "symbol",
        symbol_ref=SymbolReference("record", ["record"]),
    ),
    AttrDef("visibility", "enum", enum_def=Visibility, optional=True),
    AttrDef("cc", "enum", enum_def=CallingConv, optional=True),
    AttrDef("purity", "enum", enum_def=Purity, optional=True),
    AttrDef("predicates", "predicate_list", optional=True),
]

_FUNC_MODIFIER_FORMAT: list[FormatElement] = [
    OptionalGroup([Attr("visibility")], anchor="visibility"),
    OptionalGroup([Attr("cc")], anchor="cc"),
    OptionalGroup([Attr("purity")], anchor="purity"),
    kw("target"),
    GLUE,
    LPAREN,
    SymbolRef("target"),
    GLUE,
    RPAREN,
]

_FUNC_SIGNATURE_FORMAT: list[FormatElement] = [
    SymbolRef("callee"),
    Scope(
        [
            FuncArgs("args"),
            OptionalGroup(
                [ARROW, ResultTypeList("results")],
                anchor="results",
            ),
            OptionalGroup(
                [kw("where"), PredicateList("predicates")],
                anchor="predicates",
            ),
        ]
    ),
]

_FUNC_LIKE_COMMON: dict[str, Any] = dict(
    callee="callee",
    visibility="visibility",
    cc="cc",
    purity="purity",
    predicates="predicates",
)

# ============================================================================
# low.func.def — target-bound low function definition
# ============================================================================

low_func_def = Op(
    "low.func.def",
    group=low_ops,
    doc="Target-bound low function definition with register-typed signature values.",
    traits=[SYMBOL_DEFINE, ISOLATED_FROM_ABOVE],
    attrs=list(_FUNC_MODIFIER_ATTRS),
    symbol_def=SymbolDefinition(
        field="callee",
        name="function",
        interfaces=["func_like"],
        bytecode_kind="LOOM_SYMBOL_FUNC_DEF",
    ),
    results=[Result("results", REGISTER, variadic=True)],
    regions=[RegionDef("body", doc="Low function body.", terminator="low.return")],
    interfaces=[FuncLikeInterface(**_FUNC_LIKE_COMMON, body="body")],
    constraints=[
        BlockArgsSatisfy("body", REGISTER),
        YieldCountMatchesResults("body", "results"),
        YieldTypesMatchResults("body", "results"),
    ],
    format=[
        *_FUNC_MODIFIER_FORMAT,
        *_FUNC_SIGNATURE_FORMAT,
        Region("body"),
    ],
    examples=[
        "low.func.def target(@gfx1100) @add(%lhs: reg<amdgpu.vgpr x1>, %rhs: reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>) {\n  %sum = low.op<amdgpu.v_add_u32>(%lhs, %rhs) : (reg<amdgpu.vgpr x1>, reg<amdgpu.vgpr x1>) -> reg<amdgpu.vgpr x1>\n  low.return %sum : reg<amdgpu.vgpr x1>\n}",
    ],
)

# ============================================================================
# low.func.decl — target-bound low function declaration
# ============================================================================

low_func_decl = Op(
    "low.func.decl",
    group=low_ops,
    doc="Target-bound low function declaration with register-typed signature values.",
    traits=[SYMBOL_DEFINE],
    operands=[Operand("args", REGISTER, variadic=True)],
    attrs=list(_FUNC_MODIFIER_ATTRS),
    symbol_def=SymbolDefinition(
        field="callee",
        name="function",
        interfaces=["func_like"],
        bytecode_kind="LOOM_SYMBOL_FUNC_DECL",
    ),
    results=[Result("results", REGISTER, variadic=True)],
    interfaces=[FuncLikeInterface(**_FUNC_LIKE_COMMON, args_as_operands=True)],
    format=[
        *_FUNC_MODIFIER_FORMAT,
        *_FUNC_SIGNATURE_FORMAT,
    ],
    examples=[
        "low.func.decl target(@gfx1100) @extern_add(%lhs: reg<amdgpu.vgpr x1>, %rhs: reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>)",
    ],
)

# ============================================================================
# low.return — low function return terminator
# ============================================================================

low_return = Op(
    "low.return",
    group=low_ops,
    doc="Return register values from a low function.",
    operands=[Operand("values", REGISTER, variadic=True)],
    traits=[TERMINATOR],
    format=[
        OptionalGroup(
            [Refs("values"), COLON, TypesOf("values")],
            anchor="values",
        ),
    ],
    examples=[
        "low.return",
        "low.return %value : reg<amdgpu.vgpr x1>",
    ],
)

# ============================================================================
# low.op — descriptor-backed target instruction
# ============================================================================

low_op = Op(
    "low.op",
    group=low_ops,
    doc="Descriptor-backed target instruction over virtual registers.",
    operands=[Operand("operands", REGISTER, variadic=True)],
    attrs=[
        AttrDef("opcode", "string"),
        AttrDef("attrs", "dict", optional=True),
    ],
    results=[Result("results", REGISTER, variadic=True)],
    traits=[UNKNOWN_EFFECTS],
    verify="loom_low_op_verify",
    format=[
        OpRef("opcode"),
        GLUE,
        LPAREN,
        Refs("operands"),
        RPAREN,
        AttrDict("attrs"),
        COLON,
        LPAREN,
        TypesOf("operands"),
        RPAREN,
        OptionalGroup(
            [ARROW, ResultTypeList("results", parens=False)],
            anchor="results",
        ),
    ],
    examples=[
        "%sum = low.op<amdgpu.v_add_u32>(%lhs, %rhs) : (reg<amdgpu.vgpr x1>, reg<amdgpu.vgpr x1>) -> reg<amdgpu.vgpr x1>",
        "low.op<amdgpu.s_waitcnt>() {vmcnt = 0} : ()",
    ],
)

# ============================================================================
# low.const — descriptor-backed register constant/materialization
# ============================================================================

low_const = Op(
    "low.const",
    group=low_ops,
    doc="Descriptor-backed constant or immediate materialization into a register.",
    attrs=[
        AttrDef("opcode", "string"),
        AttrDef("attrs", "dict", optional=True),
    ],
    results=[Result("result", REGISTER)],
    traits=[UNKNOWN_EFFECTS],
    verify="loom_low_const_verify",
    format=[
        OpRef("opcode"),
        AttrDict("attrs"),
        COLON,
        ResultType("result"),
    ],
    examples=[
        "%c0 = low.const<amdgpu.s_mov_b32> {imm = 0} : reg<amdgpu.sgpr x1>",
    ],
)

# ============================================================================
# low.copy — explicit virtual-register copy/coalesce boundary
# ============================================================================

low_copy = Op(
    "low.copy",
    group=low_ops,
    doc="Explicit virtual-register copy used by lowering and allocation.",
    operands=[Operand("source", REGISTER)],
    results=[Result("result", REGISTER)],
    traits=[UNKNOWN_EFFECTS],
    format=[
        Ref("source"),
        COLON,
        TypeOf("source"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%copy = low.copy %value : reg<amdgpu.vgpr x1> -> reg<amdgpu.vgpr x1>",
    ],
)

# ============================================================================
# low.abi.adapter — explicit semantic-to-low ABI adapter record
# ============================================================================

low_abi_adapter = Op(
    "low.abi.adapter",
    group=low_ops,
    doc=("Explicit adapter record for calls crossing from semantic Loom values into a low function register ABI."),
    traits=[SYMBOL_DEFINE],
    symbol_def=SymbolDefinition(
        field="symbol",
        name="low ABI adapter",
        interfaces=["record"],
        bytecode_kind="LOOM_SYMBOL_RECORD",
    ),
    attrs=[
        AttrDef("symbol", "symbol"),
        AttrDef(
            "callee",
            "symbol",
            symbol_ref=SymbolReference("function", ["func_like"]),
        ),
        AttrDef("conversion", ATTR_TYPE_ENUM, enum_def=LowAbiConversion),
        AttrDef("operand_count", ATTR_TYPE_I64),
        AttrDef("result_count", ATTR_TYPE_I64),
    ],
    verify="loom_low_abi_adapter_verify",
    format=[
        SymbolRef("symbol"),
        AttrDict(),
    ],
    examples=[
        "low.abi.adapter @extern_add_direct {callee = @extern_add, conversion = direct, operand_count = 2, result_count = 1}",
        "low.abi.adapter @extern_add_i32 {callee = @extern_add, conversion = mapped, operand_count = 2, result_count = 1}",
    ],
)

# ============================================================================
# low.abi.operand — semantic-to-register ABI operand mapping
# ============================================================================

low_abi_operand = Op(
    "low.abi.operand",
    group=low_ops,
    doc=("Mapped low ABI adapter operand slot from semantic value type to callee register ABI type."),
    traits=[SYMBOL_DEFINE],
    symbol_def=SymbolDefinition(
        field="symbol",
        name="low ABI operand mapping",
        interfaces=["record"],
        bytecode_kind="LOOM_SYMBOL_RECORD",
    ),
    attrs=[
        AttrDef("symbol", "symbol"),
        AttrDef(
            "adapter",
            "symbol",
            symbol_ref=SymbolReference("low ABI adapter", ["record"]),
        ),
        AttrDef("index", ATTR_TYPE_I64),
        AttrDef("conversion", ATTR_TYPE_ENUM, enum_def=LowAbiValueConversion),
        AttrDef("semantic_type", ATTR_TYPE_TYPE),
        AttrDef("abi_type", ATTR_TYPE_TYPE),
    ],
    verify="loom_low_abi_operand_verify",
    format=[
        SymbolRef("symbol"),
        AttrDict(),
    ],
    examples=[
        "low.abi.operand @extern_add_i32_lhs {adapter = @extern_add_i32, index = 0, conversion = scalar_to_register, semantic_type = i32, abi_type = reg<vm.i32>}",
    ],
)

# ============================================================================
# low.abi.result — register-to-semantic ABI result mapping
# ============================================================================

low_abi_result = Op(
    "low.abi.result",
    group=low_ops,
    doc=("Mapped low ABI adapter result slot from callee register ABI type to semantic value type."),
    traits=[SYMBOL_DEFINE],
    symbol_def=SymbolDefinition(
        field="symbol",
        name="low ABI result mapping",
        interfaces=["record"],
        bytecode_kind="LOOM_SYMBOL_RECORD",
    ),
    attrs=[
        AttrDef("symbol", "symbol"),
        AttrDef(
            "adapter",
            "symbol",
            symbol_ref=SymbolReference("low ABI adapter", ["record"]),
        ),
        AttrDef("index", ATTR_TYPE_I64),
        AttrDef("conversion", ATTR_TYPE_ENUM, enum_def=LowAbiValueConversion),
        AttrDef("semantic_type", ATTR_TYPE_TYPE),
        AttrDef("abi_type", ATTR_TYPE_TYPE),
    ],
    verify="loom_low_abi_result_verify",
    format=[
        SymbolRef("symbol"),
        AttrDict(),
    ],
    examples=[
        "low.abi.result @extern_add_i32_result {adapter = @extern_add_i32, index = 0, conversion = register_to_scalar, semantic_type = i32, abi_type = reg<vm.i32>}",
    ],
)

# ============================================================================
# low.invoke — call or interop edge to a low function symbol
# ============================================================================

low_invoke = Op(
    "low.invoke",
    group=low_ops,
    doc="Call or interop edge to a low function symbol.",
    operands=[Operand("operands", ANY, variadic=True)],
    attrs=[
        AttrDef(
            "callee",
            "symbol",
            symbol_ref=SymbolReference("function", ["func_like"]),
        ),
        AttrDef(
            "adapter",
            "symbol",
            optional=True,
            symbol_ref=SymbolReference("low ABI adapter", ["record"]),
        ),
    ],
    results=[Result("results", ANY, variadic=True)],
    traits=[UNKNOWN_EFFECTS],
    verify="loom_low_invoke_verify",
    format=[
        SymbolRef("callee"),
        GLUE,
        LPAREN,
        Refs("operands"),
        RPAREN,
        AttrDict(),
        COLON,
        LPAREN,
        TypesOf("operands"),
        RPAREN,
        OptionalGroup(
            [ARROW, ResultTypeList("results")],
            anchor="results",
        ),
    ],
    examples=[
        "%result = low.invoke @extern_add(%lhs, %rhs) : (reg<amdgpu.vgpr x1>, reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>)",
        "%result = low.invoke @extern_add(%lhs, %rhs) {adapter = @extern_add_direct} : (i32, i32) -> (i32)",
    ],
)

ALL_LOW_OPS: tuple[Op, ...] = (
    low_func_def,
    low_func_decl,
    low_return,
    low_op,
    low_const,
    low_copy,
    low_invoke,
    low_abi_adapter,
    low_abi_operand,
    low_abi_result,
)
