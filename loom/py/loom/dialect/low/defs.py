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
    COMMA,
    GLUE,
    LBRACKET,
    LPAREN,
    RBRACKET,
    RPAREN,
    Attr,
    AttrDict,
    BlockRef,
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
    PURE,
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
    RegisterUnitsSumTo,
    Result,
    SameRegisterClass,
    Successor,
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
        EnumCase(
            "value_to_register",
            3,
            doc="A non-register semantic value is materialized into a callee register ABI slot.",
        ),
        EnumCase(
            "register_to_value",
            4,
            doc="A callee register ABI slot is unpacked into a non-register semantic value.",
        ),
    ],
    doc="Per-slot conversion rule used by low ABI adapter mapping records.",
)

LowAbiEffectKind = EnumDef(
    "LowAbiEffectKind",
    [
        EnumCase("read", 1, doc="Reads from the named ABI resource."),
        EnumCase("write", 2, doc="Writes to the named ABI resource."),
        EnumCase("readwrite", 3, doc="Reads and writes the named ABI resource."),
        EnumCase("call", 4, doc="Performs an externally observable call."),
        EnumCase("unknown", 5, doc="Has effects the current vocabulary cannot refine."),
    ],
    doc="Effect summary row attached to a low ABI adapter.",
)

LowAbiResourceKind = EnumDef(
    "LowAbiResourceKind",
    [
        EnumCase(
            "native_pointer",
            1,
            doc="Native object-function pointer argument materialized as a register value.",
        ),
        EnumCase(
            "vm_state",
            2,
            doc="IREE VM module state or context handle materialized as a register value.",
        ),
        EnumCase(
            "vm_import",
            3,
            doc="IREE VM imported-function or imported-resource handle materialized as a register value.",
        ),
        EnumCase(
            "hal_buffer_resource",
            4,
            doc="IREE HAL dispatch binding buffer resource descriptor materialized as a register value.",
        ),
    ],
    doc="Target ABI resource imported into a low function body.",
)

LowAllocationMode = EnumDef(
    "LowAllocationMode",
    [
        EnumCase(
            "virtual",
            1,
            doc="SSA values name virtual registers; allocation is still open.",
        ),
        EnumCase(
            "assigned",
            2,
            doc="Allocation sidecars assign physical registers, but rewriting may still repair copies/spills.",
        ),
        EnumCase(
            "fixed",
            3,
            doc="Physical register assignment is part of the low-function contract.",
        ),
    ],
    doc="Register allocation exactness mode for a low function. Absent means virtual.",
)

LowScheduleMode = EnumDef(
    "LowScheduleMode",
    [
        EnumCase("free", 1, doc="Instruction order may be scheduled."),
        EnumCase(
            "constrained",
            2,
            doc="Instruction order carries target constraints, but legal scheduling may still move packets.",
        ),
        EnumCase(
            "locked",
            3,
            doc="Instruction order is part of the low-function contract.",
        ),
    ],
    doc="Instruction scheduling exactness mode for a low function. Absent means free.",
)

LowCodeImportKind = EnumDef(
    "LowCodeImportKind",
    [
        EnumCase("vm", 1, doc="Imported implementation is an IREE VM symbol."),
        EnumCase("native", 2, doc="Imported implementation is a native callable symbol."),
        EnumCase("rocasm", 3, doc="Imported implementation is an AMDGPU rocasm symbol."),
        EnumCase("object", 4, doc="Imported implementation is a linked object-file symbol."),
    ],
    doc="External code source kind for an imported low function declaration.",
)

LowSlotSpace = EnumDef(
    "LowSlotSpace",
    [
        EnumCase("stack", 1, doc="CPU stack-frame storage."),
        EnumCase("scratch", 2, doc="GPU per-lane scratch storage."),
        EnumCase("private", 3, doc="Target-private per-invocation storage."),
        EnumCase("lds", 4, doc="GPU local data share or workgroup storage."),
    ],
    doc="Storage space represented by a low slot record.",
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
    AttrDef("allocation", "enum", enum_def=LowAllocationMode, optional=True),
    AttrDef("schedule", "enum", enum_def=LowScheduleMode, optional=True),
    AttrDef("import_kind", "enum", enum_def=LowCodeImportKind, optional=True),
    AttrDef("code_symbol", "string", optional=True),
    AttrDef("predicates", "predicate_list", optional=True),
]

_FUNC_MODIFIER_FORMAT: list[FormatElement] = [
    OptionalGroup([Attr("visibility")], anchor="visibility"),
    OptionalGroup([Attr("cc")], anchor="cc"),
    OptionalGroup([Attr("purity")], anchor="purity"),
    OptionalGroup(
        [kw("allocation"), GLUE, LPAREN, Attr("allocation"), GLUE, RPAREN],
        anchor="allocation",
    ),
    OptionalGroup(
        [kw("schedule"), GLUE, LPAREN, Attr("schedule"), GLUE, RPAREN],
        anchor="schedule",
    ),
]

_FUNC_TARGET_FORMAT: list[FormatElement] = [
    kw("target"),
    GLUE,
    LPAREN,
    SymbolRef("target"),
    GLUE,
    RPAREN,
]

_FUNC_IMPORT_FORMAT: list[FormatElement] = [
    OptionalGroup(
        [
            kw("import"),
            GLUE,
            LPAREN,
            Attr("import_kind"),
            COMMA,
            Attr("code_symbol"),
            GLUE,
            RPAREN,
        ],
        anchor="import_kind",
    ),
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
    verify="loom_low_func_def_verify",
    constraints=[
        BlockArgsSatisfy("body", REGISTER),
        YieldCountMatchesResults("body", "results"),
        YieldTypesMatchResults("body", "results"),
    ],
    format=[
        *_FUNC_MODIFIER_FORMAT,
        *_FUNC_TARGET_FORMAT,
        *_FUNC_SIGNATURE_FORMAT,
        Region("body", syntax="low.asm.optional"),
    ],
    examples=[
        "low.func.def target(@gfx1100) @add(%lhs: reg<amdgpu.vgpr x1>, %rhs: reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>) {\n  %sum = low.op<amdgpu.v_add_u32>(%lhs, %rhs) : (reg<amdgpu.vgpr x1>, reg<amdgpu.vgpr x1>) -> reg<amdgpu.vgpr x1>\n  low.return %sum : reg<amdgpu.vgpr x1>\n}",
        "low.func.def allocation(fixed) schedule(locked) target(@gfx1100) @agent_authored(%lhs: reg<amdgpu.vgpr x1>) {\n  low.return\n}",
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
    verify="loom_low_func_decl_verify",
    format=[
        *_FUNC_MODIFIER_FORMAT,
        *_FUNC_IMPORT_FORMAT,
        *_FUNC_TARGET_FORMAT,
        *_FUNC_SIGNATURE_FORMAT,
    ],
    examples=[
        "low.func.decl target(@gfx1100) @extern_add(%lhs: reg<amdgpu.vgpr x1>, %rhs: reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>)",
        'low.func.decl allocation(fixed) schedule(locked) import(rocasm, "mfma_16x16_seq") target(@gfx1100) @mfma_rocasm(%acc: reg<amdgpu.vgpr x4>) -> (reg<amdgpu.vgpr x4>)',
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
# low.br — low unconditional branch
# ============================================================================

low_br = Op(
    "low.br",
    group=low_ops,
    doc="Unconditional branch to a low successor block, forwarding register values.",
    operands=[
        Operand(
            "args",
            REGISTER,
            variadic=True,
            doc="Register values forwarded to the destination block arguments.",
        )
    ],
    successors=[Successor("dest", doc="Destination low block.")],
    traits=[TERMINATOR],
    verify="loom_low_br_verify",
    format=[
        BlockRef("dest"),
        OptionalGroup(
            [GLUE, LPAREN, Refs("args"), COLON, TypesOf("args"), RPAREN],
            anchor="args",
        ),
    ],
    examples=[
        "low.br ^done",
        "low.br ^join(%value : reg<vm.i32>)",
    ],
)

# ============================================================================
# low.cond_br — low conditional branch
# ============================================================================

low_cond_br = Op(
    "low.cond_br",
    group=low_ops,
    doc="Conditional branch to one of two low successor blocks based on a register predicate.",
    operands=[Operand("condition", REGISTER, doc="Register predicate controlling the branch.")],
    successors=[
        Successor("true_dest", doc="Destination block when the predicate is true."),
        Successor("false_dest", doc="Destination block when the predicate is false."),
    ],
    traits=[TERMINATOR],
    verify="loom_low_cond_br_verify",
    format=[
        Ref("condition"),
        COMMA,
        BlockRef("true_dest"),
        COMMA,
        BlockRef("false_dest"),
        COLON,
        TypeOf("condition"),
    ],
    examples=[
        "low.cond_br %condition, ^then, ^else : reg<vm.i32>",
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
    traits=[PURE],
    constraints=[
        SameRegisterClass("source", "result"),
    ],
    verify="loom_low_copy_verify",
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
# low.slice — project a contiguous subrange from a register range
# ============================================================================

low_slice = Op(
    "low.slice",
    group=low_ops,
    doc="Project a contiguous subrange from a register-range value.",
    attrs=[
        AttrDef("offset", "i64"),
    ],
    operands=[Operand("source", REGISTER)],
    results=[Result("result", REGISTER)],
    traits=[PURE],
    constraints=[
        SameRegisterClass("source", "result"),
    ],
    verify="loom_low_slice_verify",
    format=[
        Ref("source"),
        GLUE,
        LBRACKET,
        Attr("offset"),
        RBRACKET,
        COLON,
        TypeOf("source"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%lane = low.slice %quad[2] : reg<amdgpu.vgpr x4> -> reg<amdgpu.vgpr>",
        "%pair = low.slice %quad[1] : reg<amdgpu.vgpr x4> -> reg<amdgpu.vgpr x2>",
    ],
)

# ============================================================================
# low.concat — compose a register range from ordered subranges
# ============================================================================

low_concat = Op(
    "low.concat",
    group=low_ops,
    doc="Compose one register-range value from ordered register subranges.",
    operands=[Operand("sources", REGISTER, variadic=True)],
    results=[Result("result", REGISTER)],
    traits=[PURE],
    constraints=[
        SameRegisterClass("sources", "result"),
        RegisterUnitsSumTo("sources", "result"),
    ],
    format=[
        GLUE,
        LPAREN,
        Refs("sources"),
        RPAREN,
        COLON,
        LPAREN,
        TypesOf("sources"),
        RPAREN,
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%pair = low.concat(%lo, %hi) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x2>",
        "%resource = low.concat(%ptr, %limit, %flags) : (reg<amdgpu.sgpr x2>, reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x4>",
    ],
)

# ============================================================================
# low.live_in — target ABI live-in register value
# ============================================================================

low_live_in = Op(
    "low.live_in",
    group=low_ops,
    doc="Import a target-provided ABI live-in register value at low-function entry.",
    attrs=[
        AttrDef("source", "string"),
        AttrDef("attrs", "dict", optional=True),
    ],
    results=[Result("result", REGISTER)],
    verify="loom_low_live_in_verify",
    format=[
        OpRef("source"),
        AttrDict("attrs"),
        COLON,
        ResultType("result"),
    ],
    examples=[
        "%kernarg = low.live_in<amdgpu.kernarg_segment_ptr> : reg<amdgpu.sgpr x2>",
    ],
)

# ============================================================================
# low.slot — explicit frame/storage slot record
# ============================================================================

low_slot = Op(
    "low.slot",
    group=low_ops,
    doc="Explicit function-owned stack, scratch, private, or LDS storage slot.",
    traits=[SYMBOL_DEFINE],
    symbol_def=SymbolDefinition(
        field="symbol",
        name="low slot",
        interfaces=["record"],
        bytecode_kind="LOOM_SYMBOL_RECORD",
    ),
    attrs=[
        AttrDef("symbol", "symbol"),
        AttrDef(
            "function",
            "symbol",
            symbol_ref=SymbolReference("function", ["func_like"]),
        ),
        AttrDef("space", ATTR_TYPE_ENUM, enum_def=LowSlotSpace),
        AttrDef("size", ATTR_TYPE_I64),
        AttrDef("align", ATTR_TYPE_I64),
    ],
    verify="loom_low_slot_verify",
    format=[
        SymbolRef("symbol"),
        AttrDict(),
    ],
    examples=[
        "low.slot @spill0 {function = @kernel, space = scratch, size = 16, align = 4}",
    ],
)

# ============================================================================
# low.spill — explicit store from a register into a low slot
# ============================================================================

low_spill = Op(
    "low.spill",
    group=low_ops,
    doc="Explicit spill store from a register value into a low slot.",
    operands=[Operand("value", REGISTER)],
    attrs=[
        AttrDef(
            "slot",
            "symbol",
            symbol_ref=SymbolReference("low slot", ["record"]),
        ),
        AttrDef("offset", ATTR_TYPE_I64),
    ],
    traits=[UNKNOWN_EFFECTS],
    verify="loom_low_spill_verify",
    format=[
        Ref("value"),
        COMMA,
        SymbolRef("slot"),
        AttrDict(),
        COLON,
        TypeOf("value"),
    ],
    examples=[
        "low.spill %value, @spill0 {offset = 0} : reg<amdgpu.vgpr x4>",
    ],
)

# ============================================================================
# low.reload — explicit load from a low slot into a register
# ============================================================================

low_reload = Op(
    "low.reload",
    group=low_ops,
    doc="Explicit reload from a low slot into a register value.",
    attrs=[
        AttrDef(
            "slot",
            "symbol",
            symbol_ref=SymbolReference("low slot", ["record"]),
        ),
        AttrDef("offset", ATTR_TYPE_I64),
    ],
    results=[Result("result", REGISTER)],
    traits=[UNKNOWN_EFFECTS],
    verify="loom_low_reload_verify",
    format=[
        SymbolRef("slot"),
        AttrDict(),
        COLON,
        ResultType("result"),
    ],
    examples=[
        "%reload = low.reload @spill0 {offset = 0} : reg<amdgpu.vgpr x4>",
    ],
)

# ============================================================================
# low.frame_index — symbolic address of a low slot before final layout
# ============================================================================

low_frame_index = Op(
    "low.frame_index",
    group=low_ops,
    doc="Symbolic address calculation for a low slot before target frame layout.",
    attrs=[
        AttrDef(
            "slot",
            "symbol",
            symbol_ref=SymbolReference("low slot", ["record"]),
        ),
        AttrDef("offset", ATTR_TYPE_I64),
    ],
    results=[Result("result", REGISTER)],
    traits=[PURE],
    verify="loom_low_frame_index_verify",
    format=[
        SymbolRef("slot"),
        AttrDict(),
        COLON,
        ResultType("result"),
    ],
    examples=[
        "%addr = low.frame_index @spill0 {offset = 0} : reg<x86.gpr>",
    ],
)

# ============================================================================
# low.abi.resource — ABI resource binding record
# ============================================================================

low_abi_resource = Op(
    "low.abi.resource",
    group=low_ops,
    doc="Function-owned target ABI resource imported by low.resource.",
    traits=[SYMBOL_DEFINE],
    symbol_def=SymbolDefinition(
        field="symbol",
        name="low ABI resource",
        interfaces=["record"],
        bytecode_kind="LOOM_SYMBOL_RECORD",
    ),
    attrs=[
        AttrDef("symbol", "symbol"),
        AttrDef(
            "function",
            "symbol",
            symbol_ref=SymbolReference("function", ["func_like"]),
        ),
        AttrDef("kind", ATTR_TYPE_ENUM, enum_def=LowAbiResourceKind),
        AttrDef("index", ATTR_TYPE_I64),
        AttrDef("semantic_type", ATTR_TYPE_TYPE),
        AttrDef("abi_type", ATTR_TYPE_TYPE),
    ],
    verify="loom_low_abi_resource_verify",
    format=[
        SymbolRef("symbol"),
        AttrDict(),
    ],
    examples=[
        "low.abi.resource @vm_state {function = @vm_func, kind = vm_state, index = 0, semantic_type = i64, abi_type = reg<vm.i64>}",
        "low.abi.resource @binding0 {function = @kernel, kind = hal_buffer_resource, index = 0, semantic_type = hal.buffer, abi_type = reg<amdgpu.sgpr x4>}",
    ],
)

# ============================================================================
# low.resource — import ABI resource into a register value
# ============================================================================

low_resource = Op(
    "low.resource",
    group=low_ops,
    doc="Import a function-owned target ABI resource into a low register value.",
    attrs=[
        AttrDef(
            "resource",
            "symbol",
            symbol_ref=SymbolReference("low ABI resource", ["record"]),
        ),
    ],
    results=[Result("result", REGISTER)],
    traits=[PURE],
    verify="loom_low_resource_verify",
    format=[
        SymbolRef("resource"),
        COLON,
        ResultType("result"),
    ],
    examples=[
        "%state = low.resource @vm_state : reg<vm.i64>",
        "%binding = low.resource @binding0 : reg<amdgpu.sgpr x4>",
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
        "low.abi.operand @extern_add_v4_lhs {adapter = @extern_add_v4, index = 0, conversion = value_to_register, semantic_type = vector<4xi32>, abi_type = reg<avx512.zmm>}",
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
        "low.abi.result @extern_add_v4_result {adapter = @extern_add_v4, index = 0, conversion = register_to_value, semantic_type = vector<4xi32>, abi_type = reg<avx512.zmm>}",
    ],
)

# ============================================================================
# low.abi.effect — adapter-visible resource effect
# ============================================================================

low_abi_effect = Op(
    "low.abi.effect",
    group=low_ops,
    doc=("Resource effect exposed by a low ABI adapter to semantic callers."),
    traits=[SYMBOL_DEFINE],
    symbol_def=SymbolDefinition(
        field="symbol",
        name="low ABI effect",
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
        AttrDef("kind", ATTR_TYPE_ENUM, enum_def=LowAbiEffectKind),
        AttrDef("resource", "string", optional=True),
    ],
    verify="loom_low_abi_effect_verify",
    format=[
        SymbolRef("symbol"),
        AttrDict(),
    ],
    examples=[
        'low.abi.effect @extern_add_call {adapter = @extern_add_i32, kind = call, resource = "vm.import"}',
    ],
)

# ============================================================================
# low.abi.clobber — adapter-visible clobbered resource
# ============================================================================

low_abi_clobber = Op(
    "low.abi.clobber",
    group=low_ops,
    doc=("Resource or register class clobbered by a low ABI adapter."),
    traits=[SYMBOL_DEFINE],
    symbol_def=SymbolDefinition(
        field="symbol",
        name="low ABI clobber",
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
        AttrDef("resource", "string"),
    ],
    verify="loom_low_abi_clobber_verify",
    format=[
        SymbolRef("symbol"),
        AttrDict(),
    ],
    examples=[
        'low.abi.clobber @extern_add_vm_state {adapter = @extern_add_i32, resource = "vm.state"}',
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
        AttrDef("purity", "enum", enum_def=Purity, optional=True),
    ],
    results=[Result("results", ANY, variadic=True)],
    traits=[UNKNOWN_EFFECTS],
    effective_traits="loom_low_invoke_effective_traits",
    verify="loom_low_invoke_verify",
    format=[
        OptionalGroup([Attr("purity")], anchor="purity"),
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
        "%result = low.invoke pure @extern_add(%lhs, %rhs) {adapter = @extern_add_i32} : (i32, i32) -> (i32)",
    ],
)

ALL_LOW_OPS: tuple[Op, ...] = (
    low_func_def,
    low_func_decl,
    low_return,
    low_op,
    low_const,
    low_copy,
    low_slice,
    low_concat,
    low_invoke,
    low_abi_adapter,
    low_abi_operand,
    low_abi_result,
    low_abi_effect,
    low_abi_clobber,
    low_slot,
    low_spill,
    low_reload,
    low_frame_index,
    low_br,
    low_cond_br,
    low_resource,
    low_abi_resource,
    low_live_in,
)
