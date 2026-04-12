# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Vector dialect op definitions."""

from __future__ import annotations

from collections.abc import Callable

from loom.assembly import (
    ARROW,
    COLON,
    COMMA,
    GLUE,
    LBRACKET,
    RBRACKET,
    Attr,
    AttrDict,
    Flags,
    FormatElement,
    IndexList,
    Ref,
    Refs,
    ResultType,
    ResultTypeList,
    TemplateParam,
    TypeOf,
    TypesOf,
    kw,
)
from loom.dialect.scalar import IntOverflowFlags
from loom.dialect.scalar.comparison import CmpFPredicate, CmpIPredicate
from loom.dsl import (
    ANY,
    ATTR_TYPE_ANY,
    ATTR_TYPE_ENUM,
    ATTR_TYPE_FLAGS,
    ATTR_TYPE_I64,
    ATTR_TYPE_I64_ARRAY,
    COMMUTATIVE,
    CONSTANT_LIKE,
    ELEMENTWISE,
    FLOAT_ELEMENT,
    I1_ELEMENT,
    INDEX,
    INTEGER_ELEMENT,
    PURE,
    SCALAR,
    VECTOR,
    VIEW,
    AttrDef,
    Constraint,
    Dialect,
    EnumCase,
    EnumDef,
    HasFloatElement,
    HasI1Element,
    HasIntegerElement,
    Op,
    Operand,
    Reads,
    ReadWrites,
    Result,
    SameElementType,
    SameKind,
    SameShape,
    SameType,
    Trait,
    TypeConstraint,
    Writes,
)

# ============================================================================
# Group and shared attrs
# ============================================================================

vector_ops = Dialect(
    "vector",
    dialect_id=0x0E,
    doc="Vector register-lane construction, lanewise computation, and reduction ops.",
)

CombiningKind = EnumDef(
    "CombiningKind",
    [
        EnumCase("addi", 0, doc="Integer addition."),
        EnumCase("addf", 1, doc="Floating-point addition."),
        EnumCase("muli", 2, doc="Integer multiplication."),
        EnumCase("mulf", 3, doc="Floating-point multiplication."),
        EnumCase("minsi", 4, doc="Signed integer minimum."),
        EnumCase("maxsi", 5, doc="Signed integer maximum."),
        EnumCase("minui", 6, doc="Unsigned integer minimum."),
        EnumCase("maxui", 7, doc="Unsigned integer maximum."),
        EnumCase("andi", 8, doc="Bitwise AND."),
        EnumCase("ori", 9, doc="Bitwise OR."),
        EnumCase("xori", 10, doc="Bitwise XOR."),
        EnumCase("minimumf", 11, doc="IEEE 754 floating-point minimum."),
        EnumCase("maximumf", 12, doc="IEEE 754 floating-point maximum."),
        EnumCase("minnumf", 13, doc="C99 fmin-style floating-point minimum."),
        EnumCase("maxnumf", 14, doc="C99 fmax-style floating-point maximum."),
    ],
    doc="Combining operations for vector reductions.",
)

AtomicKind = EnumDef(
    "AtomicKind",
    [
        EnumCase("xchgi", 0, doc="Integer exchange."),
        EnumCase("xchgf", 1, doc="Floating-point exchange."),
        EnumCase("addi", 2, doc="Integer addition."),
        EnumCase("addf", 3, doc="Floating-point addition."),
        EnumCase("subi", 4, doc="Integer subtraction."),
        EnumCase("andi", 5, doc="Bitwise AND."),
        EnumCase("ori", 6, doc="Bitwise OR."),
        EnumCase("xori", 7, doc="Bitwise XOR."),
        EnumCase("minsi", 8, doc="Signed integer minimum."),
        EnumCase("maxsi", 9, doc="Signed integer maximum."),
        EnumCase("minui", 10, doc="Unsigned integer minimum."),
        EnumCase("maxui", 11, doc="Unsigned integer maximum."),
        EnumCase("minimumf", 12, doc="IEEE 754 floating-point minimum."),
        EnumCase("maximumf", 13, doc="IEEE 754 floating-point maximum."),
        EnumCase("minnumf", 14, doc="C99 fmin-style floating-point minimum."),
        EnumCase("maxnumf", 15, doc="C99 fmax-style floating-point maximum."),
    ],
    doc="Read-modify-write operations supported by vector atomics.",
)

AtomicOrdering = EnumDef(
    "AtomicOrdering",
    [
        EnumCase("relaxed", 0, doc="Atomicity without inter-address synchronization."),
        EnumCase("acquire", 1, doc="Acquire ordering."),
        EnumCase("release", 2, doc="Release ordering."),
        EnumCase("acq_rel", 3, doc="Acquire and release ordering."),
        EnumCase("seq_cst", 4, doc="Sequentially consistent ordering."),
    ],
    doc="Atomic memory ordering. The relaxed case lowers to LLVM monotonic RMW ordering.",
)

AtomicScope = EnumDef(
    "AtomicScope",
    [
        EnumCase("thread", 0, doc="Current invocation or thread."),
        EnumCase("subgroup", 1, doc="Current SIMD subgroup or wave."),
        EnumCase("workgroup", 2, doc="Current workgroup or block."),
        EnumCase("device", 3, doc="Current device."),
        EnumCase("system", 4, doc="Whole system."),
    ],
    doc="Synchronization scope for atomic memory effects.",
)

FloatAssumptionFlags = EnumDef(
    "FloatAssumptionFlags",
    [
        EnumCase("nnan", 1, doc="Assume no NaNs."),
        EnumCase("ninf", 2, doc="Assume no infinities."),
        EnumCase("nsz", 4, doc="Assume no signed zeros."),
    ],
    doc="Floating-point value-domain assumptions for vector operations.",
)


def _lanewise_binary(
    name: str,
    *,
    result_constraint: TypeConstraint,
    doc: str,
    commutative: bool = False,
    flags: tuple[str, EnumDef] | None = None,
) -> Op:
    result_element_constraint = _element_constraint_for(result_constraint)
    attrs: list[AttrDef] = []
    fmt: list[FormatElement] = []
    if flags:
        attr_name, enum_def = flags
        attrs.append(AttrDef(attr_name, ATTR_TYPE_FLAGS, optional=True, enum_def=enum_def))
        fmt.append(Flags(attr_name))
    fmt.extend([Ref("lhs"), COMMA, Ref("rhs"), COLON, TypeOf("result")])

    traits: list[Trait] = [PURE, ELEMENTWISE]
    if commutative:
        traits.append(COMMUTATIVE)
    return Op(
        name,
        group=vector_ops,
        doc=doc,
        operands=[
            Operand("lhs", VECTOR),
            Operand("rhs", VECTOR),
        ],
        results=[Result("result", VECTOR)],
        attrs=attrs,
        constraints=[
            result_element_constraint("result"),
            SameType("lhs", "rhs", "result"),
        ],
        traits=traits,
        format=fmt,
    )


def _lanewise_unary(
    name: str,
    *,
    result_constraint: TypeConstraint,
    doc: str,
    traits: list[Trait] | None = None,
    flags: tuple[str, EnumDef] | None = None,
) -> Op:
    result_element_constraint = _element_constraint_for(result_constraint)
    attrs: list[AttrDef] = []
    fmt: list[FormatElement] = []
    if flags:
        attr_name, enum_def = flags
        attrs.append(AttrDef(attr_name, ATTR_TYPE_FLAGS, optional=True, enum_def=enum_def))
        fmt.append(Flags(attr_name))
    fmt.extend([Ref("input"), COLON, TypeOf("result")])

    op_traits: list[Trait] = [PURE, ELEMENTWISE]
    if traits:
        op_traits.extend(traits)
    return Op(
        name,
        group=vector_ops,
        doc=doc,
        operands=[Operand("input", VECTOR)],
        results=[Result("result", VECTOR)],
        attrs=attrs,
        constraints=[
            result_element_constraint("result"),
            SameType("input", "result"),
        ],
        traits=op_traits,
        format=fmt,
    )


def _vector_cast(
    name: str,
    *,
    result_constraint: TypeConstraint,
    source_constraint: Callable[[str], Constraint],
    doc: str,
    verify: str = "",
) -> Op:
    result_element_constraint = _element_constraint_for(result_constraint)
    return Op(
        name,
        group=vector_ops,
        doc=doc,
        operands=[Operand("input", VECTOR)],
        results=[Result("result", VECTOR)],
        constraints=[
            source_constraint("input"),
            result_element_constraint("result"),
            SameKind("input", "result"),
            SameShape("input", "result"),
        ],
        traits=[PURE, ELEMENTWISE],
        verify=verify,
        format=[
            Ref("input"),
            COLON,
            TypeOf("input"),
            kw("to"),
            TypeOf("result"),
        ],
    )


def _element_constraint_for(
    type_constraint: TypeConstraint,
) -> Callable[[str], Constraint]:
    if type_constraint == FLOAT_ELEMENT:
        return HasFloatElement
    if type_constraint == INTEGER_ELEMENT:
        return HasIntegerElement
    if type_constraint == I1_ELEMENT:
        return HasI1Element
    raise ValueError(f"unsupported vector element constraint: {type_constraint}")


# ============================================================================
# Construction
# ============================================================================

vector_constant = Op(
    "vector.constant",
    group=vector_ops,
    doc="Materialize a compile-time splat constant vector value.",
    results=[Result("result", VECTOR)],
    attrs=[AttrDef("value", ATTR_TYPE_ANY, doc="The constant payload.")],
    verify="loom_vector_constant_verify",
    traits=[PURE, CONSTANT_LIKE],
    format=[Attr("value"), COLON, ResultType("result")],
    examples=["%v = vector.constant 0.0 : vector<4xf32>"],
)

vector_splat = Op(
    "vector.splat",
    group=vector_ops,
    doc="Replicate one scalar value to every lane of a vector result.",
    operands=[Operand("scalar", SCALAR)],
    results=[Result("result", VECTOR)],
    constraints=[SameElementType("scalar", "result")],
    traits=[PURE],
    format=[Ref("scalar"), COLON, TypeOf("scalar"), ARROW, ResultType("result")],
    examples=[
        "%vec = vector.splat %scalar : f32 -> vector<16xf32>",
        "%vec = vector.splat %scalar : i32 -> vector<[%n]xi32>",
    ],
)

vector_broadcast = Op(
    "vector.broadcast",
    group=vector_ops,
    doc="Broadcast a vector value to a vector result with a compatible shape.",
    operands=[Operand("source", VECTOR)],
    results=[Result("result", VECTOR)],
    constraints=[SameElementType("source", "result")],
    verify="loom_vector_broadcast_verify",
    traits=[PURE],
    format=[Ref("source"), COLON, TypeOf("source"), ARROW, ResultType("result")],
    examples=["%wide = vector.broadcast %v : vector<4xf32> -> vector<16x4xf32>"],
)

vector_from_elements = Op(
    "vector.from_elements",
    group=vector_ops,
    doc="Build a vector from scalar element operands in lane order.",
    operands=[Operand("elements", SCALAR, variadic=True)],
    results=[Result("result", VECTOR)],
    constraints=[SameElementType("elements", "result")],
    verify="loom_vector_from_elements_verify",
    traits=[PURE],
    format=[
        Refs("elements"),
        COLON,
        TypesOf("elements"),
        ARROW,
        ResultType("result"),
    ],
    examples=["%v = vector.from_elements %a, %b, %c, %d : f32, f32, f32, f32 -> vector<4xf32>"],
)


# ============================================================================
# Access
# ============================================================================

vector_extract = Op(
    "vector.extract",
    group=vector_ops,
    doc="Extract a scalar or subvector from a vector at explicit indices.",
    operands=[
        Operand("source", VECTOR),
        Operand("indices", INDEX, variadic=True),
    ],
    results=[Result("result", ANY)],
    attrs=[
        AttrDef(
            "static_indices",
            ATTR_TYPE_I64_ARRAY,
            doc="Static indices with INT64_MIN sentinels for dynamics.",
        ),
    ],
    constraints=[SameElementType("source", "result")],
    verify="loom_vector_extract_verify",
    traits=[PURE],
    format=[
        Ref("source"),
        IndexList("indices", "static_indices"),
        COLON,
        TypeOf("source"),
        ARROW,
        ResultType("result"),
    ],
    examples=["%x = vector.extract %v[%i] : vector<[%n]xf32> -> f32"],
)

vector_insert = Op(
    "vector.insert",
    group=vector_ops,
    doc="Insert a scalar or subvector into a vector at explicit indices.",
    operands=[
        Operand("value", ANY),
        Operand("dest", VECTOR),
        Operand("indices", INDEX, variadic=True),
    ],
    results=[Result("result", VECTOR)],
    attrs=[
        AttrDef(
            "static_indices",
            ATTR_TYPE_I64_ARRAY,
            doc="Static indices with INT64_MIN sentinels for dynamics.",
        ),
    ],
    constraints=[
        SameElementType("value", "dest", "result"),
        SameType("dest", "result"),
    ],
    verify="loom_vector_insert_verify",
    traits=[PURE],
    format=[
        Ref("value"),
        kw("into"),
        Ref("dest"),
        IndexList("indices", "static_indices"),
        COLON,
        TypeOf("value"),
        COMMA,
        TypeOf("dest"),
        ARROW,
        ResultType("result"),
    ],
    examples=["%r = vector.insert %x into %v[%i] : f32, vector<[%n]xf32> -> vector<[%n]xf32>"],
)


# ============================================================================
# Register layout
# ============================================================================

vector_slice = Op(
    "vector.slice",
    group=vector_ops,
    doc="Extract a rank-preserving contiguous register subvector at explicit offsets.",
    operands=[
        Operand("source", VECTOR),
        Operand("offsets", INDEX, variadic=True),
    ],
    results=[Result("result", VECTOR)],
    attrs=[
        AttrDef(
            "static_offsets",
            ATTR_TYPE_I64_ARRAY,
            doc="Static offsets with INT64_MIN sentinels for dynamics.",
        ),
    ],
    constraints=[SameElementType("source", "result")],
    verify="loom_vector_slice_verify",
    traits=[PURE],
    format=[
        Ref("source"),
        IndexList("offsets", "static_offsets"),
        COLON,
        TypeOf("source"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%tail = vector.slice %v[%i] : vector<[%n]xf32> -> vector<4xf32>",
        "%tile = vector.slice %v[0, 4] : vector<8x16xf32> -> vector<4x8xf32>",
    ],
)

vector_concat = Op(
    "vector.concat",
    group=vector_ops,
    doc="Concatenate same-rank vectors along one explicit result axis.",
    operands=[Operand("inputs", VECTOR, variadic=True)],
    results=[Result("result", VECTOR)],
    attrs=[
        AttrDef(
            "axis",
            ATTR_TYPE_I64,
            doc="Axis along which input extents concatenate.",
        ),
    ],
    constraints=[SameElementType("inputs", "result")],
    verify="loom_vector_concat_verify",
    traits=[PURE],
    format=[
        TemplateParam("axis"),
        Refs("inputs"),
        COLON,
        TypesOf("inputs"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%wide = vector.concat<0> %a, %b : vector<4xf32>, vector<4xf32> -> vector<8xf32>",
        "%cols = vector.concat<1> %a, %b : vector<4x8xf32>, vector<4x8xf32> -> vector<4x16xf32>",
    ],
)

vector_transpose = Op(
    "vector.transpose",
    group=vector_ops,
    doc="Reorder vector register axes using an explicit result-axis to source-axis permutation.",
    operands=[Operand("source", VECTOR)],
    results=[Result("result", VECTOR)],
    attrs=[
        AttrDef(
            "permutation",
            ATTR_TYPE_I64_ARRAY,
            doc="Permutation mapping each result axis to the corresponding source axis.",
        ),
    ],
    constraints=[SameElementType("source", "result")],
    verify="loom_vector_transpose_verify",
    traits=[PURE],
    format=[
        TemplateParam("permutation"),
        Ref("source"),
        COLON,
        TypeOf("source"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%t = vector.transpose<[1, 0]> %v : vector<4x8xf32> -> vector<8x4xf32>",
    ],
)

vector_shuffle = Op(
    "vector.shuffle",
    group=vector_ops,
    doc="Build a same-typed rank-1 vector by selecting source register lanes with a static lane map; duplicate lanes are allowed.",
    operands=[Operand("source", VECTOR)],
    results=[Result("result", VECTOR)],
    attrs=[
        AttrDef(
            "source_lanes",
            ATTR_TYPE_I64_ARRAY,
            doc="Source lane index for each result lane.",
        ),
    ],
    constraints=[SameType("source", "result")],
    verify="loom_vector_shuffle_verify",
    traits=[PURE],
    format=[
        TemplateParam("source_lanes"),
        Ref("source"),
        COLON,
        TypeOf("result"),
    ],
    examples=[
        "%rev = vector.shuffle<[3, 2, 1, 0]> %v : vector<4xf32>",
        "%dup = vector.shuffle<[0, 0, 2, 2]> %v : vector<4xf32>",
    ],
)


vector_interleave = Op(
    "vector.interleave",
    group=vector_ops,
    doc=("Interleave two same-typed vectors along one axis; even lanes come from the first operand and odd lanes come from the second operand."),
    operands=[
        Operand("even", VECTOR, doc="Vector providing result lanes at even positions along the interleaved axis."),
        Operand("odd", VECTOR, doc="Vector providing result lanes at odd positions along the interleaved axis."),
    ],
    results=[Result("result", VECTOR)],
    attrs=[
        AttrDef(
            "axis",
            ATTR_TYPE_I64,
            doc="Axis whose extent is doubled in the result.",
        ),
    ],
    constraints=[
        SameType("even", "odd"),
        SameElementType("even", "result"),
    ],
    verify="loom_vector_interleave_verify",
    traits=[PURE],
    format=[
        TemplateParam("axis"),
        Ref("even"),
        COMMA,
        Ref("odd"),
        COLON,
        TypeOf("even"),
        COMMA,
        TypeOf("odd"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%r = vector.interleave<0> %lo, %hi : vector<16xi8>, vector<16xi8> -> vector<32xi8>",
        "%r = vector.interleave<1> %a, %b : vector<4x8xi8>, vector<4x8xi8> -> vector<4x16xi8>",
    ],
)

vector_deinterleave = Op(
    "vector.deinterleave",
    group=vector_ops,
    doc=("Split one vector along an axis into even-position and odd-position vectors of the same type."),
    operands=[Operand("source", VECTOR)],
    results=[Result("results", VECTOR, variadic=True, doc="Even-position result followed by odd-position result.")],
    attrs=[
        AttrDef(
            "axis",
            ATTR_TYPE_I64,
            doc="Axis whose extent is halved in each result.",
        ),
    ],
    constraints=[
        SameType("results"),
        SameElementType("source", "results"),
    ],
    verify="loom_vector_deinterleave_verify",
    traits=[PURE],
    format=[
        TemplateParam("axis"),
        Ref("source"),
        COLON,
        TypeOf("source"),
        ARROW,
        ResultTypeList("results", parens=False),
    ],
    examples=[
        "%lo, %hi = vector.deinterleave<0> %r : vector<32xi8> -> vector<16xi8>, vector<16xi8>",
        "%a, %b = vector.deinterleave<1> %r : vector<4x16xi8> -> vector<4x8xi8>, vector<4x8xi8>",
    ],
)


vector_table_lookup = Op(
    "vector.table.lookup",
    group=vector_ops,
    doc="Select table vector lanes using explicit integer index lanes; every index lane must be within the table extent.",
    operands=[
        Operand("table", VECTOR, doc="Rank-1 register table containing selectable lane values."),
        Operand("indices", VECTOR, doc="Index vector selecting one table lane for each result lane."),
    ],
    results=[Result("result", VECTOR)],
    constraints=[
        SameElementType("table", "result"),
        SameShape("indices", "result"),
    ],
    verify="loom_vector_table_lookup_verify",
    traits=[PURE],
    format=[
        Ref("table"),
        GLUE,
        LBRACKET,
        Ref("indices"),
        RBRACKET,
        COLON,
        TypeOf("table"),
        COMMA,
        TypeOf("indices"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%values = vector.table.lookup %grid[%codes] : vector<16xf16>, vector<32xi8> -> vector<32xf16>",
        "%values = vector.table.lookup %grid[%codes] : vector<16xf32>, vector<4x8xi8> -> vector<4x8xf32>",
    ],
)


# ============================================================================
# Memory
# ============================================================================

vector_load = Op(
    "vector.load",
    group=vector_ops,
    doc="Load a vector footprint from a typed view at a full-rank logical origin.",
    operands=[
        Operand("view", VIEW, doc="Typed source view."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    results=[Result("result", VECTOR, doc="Loaded vector value.")],
    attrs=[
        AttrDef(
            "static_indices",
            ATTR_TYPE_I64_ARRAY,
            doc="Static logical origin indices with INT64_MIN sentinels for dynamics.",
        ),
    ],
    constraints=[SameElementType("view", "result")],
    effects=[Reads("view")],
    verify="loom_vector_load_verify",
    format=[
        Ref("view"),
        IndexList("indices", "static_indices"),
        COLON,
        TypeOf("view"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%v = vector.load %view[%row, %col] : view<[%m]x[%n]xf32, %layout> -> vector<4x8xf32>",
    ],
)

vector_store = Op(
    "vector.store",
    group=vector_ops,
    doc="Store a vector footprint into a typed view at a full-rank logical origin.",
    operands=[
        Operand("value", VECTOR, doc="Vector value to store."),
        Operand("view", VIEW, doc="Typed destination view."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    attrs=[
        AttrDef(
            "static_indices",
            ATTR_TYPE_I64_ARRAY,
            doc="Static logical origin indices with INT64_MIN sentinels for dynamics.",
        ),
    ],
    constraints=[SameElementType("value", "view")],
    effects=[Writes("view")],
    verify="loom_vector_store_verify",
    format=[
        Ref("value"),
        COMMA,
        Ref("view"),
        IndexList("indices", "static_indices"),
        COLON,
        TypeOf("value"),
        COMMA,
        TypeOf("view"),
    ],
    examples=[
        "vector.store %v, %view[%row, %col] : vector<4x8xf32>, view<[%m]x[%n]xf32, %layout>",
    ],
)

vector_load_mask = Op(
    "vector.load.mask",
    group=vector_ops,
    doc="Masked vector load from a typed view; masked-off lanes take the passthrough value.",
    operands=[
        Operand("view", VIEW, doc="Typed source view."),
        Operand("mask", VECTOR, doc="i1 vector mask selecting loaded lanes."),
        Operand("passthrough", VECTOR, doc="Value used for masked-off lanes."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    results=[Result("result", VECTOR, doc="Loaded vector value.")],
    attrs=[
        AttrDef(
            "static_indices",
            ATTR_TYPE_I64_ARRAY,
            doc="Static logical origin indices with INT64_MIN sentinels for dynamics.",
        ),
    ],
    constraints=[
        HasI1Element("mask"),
        SameElementType("view", "passthrough", "result"),
        SameShape("mask", "passthrough", "result"),
        SameType("passthrough", "result"),
    ],
    effects=[Reads("view")],
    verify="loom_vector_load_mask_verify",
    format=[
        Ref("view"),
        IndexList("indices", "static_indices"),
        COMMA,
        Ref("mask"),
        COMMA,
        Ref("passthrough"),
        COLON,
        TypeOf("view"),
        COMMA,
        TypeOf("mask"),
        COMMA,
        TypeOf("passthrough"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%v = vector.load.mask %view[%row, %col], %mask, %old : view<[%m]x[%n]xf32, %layout>, vector<4x8xi1>, vector<4x8xf32> -> vector<4x8xf32>",
    ],
)

vector_store_mask = Op(
    "vector.store.mask",
    group=vector_ops,
    doc="Masked vector store into a typed view; masked-off lanes leave memory unchanged.",
    operands=[
        Operand("value", VECTOR, doc="Vector value to store."),
        Operand("view", VIEW, doc="Typed destination view."),
        Operand("mask", VECTOR, doc="i1 vector mask selecting stored lanes."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    attrs=[
        AttrDef(
            "static_indices",
            ATTR_TYPE_I64_ARRAY,
            doc="Static logical origin indices with INT64_MIN sentinels for dynamics.",
        ),
    ],
    constraints=[
        HasI1Element("mask"),
        SameElementType("value", "view"),
        SameShape("mask", "value"),
    ],
    effects=[Writes("view")],
    verify="loom_vector_store_mask_verify",
    format=[
        Ref("value"),
        COMMA,
        Ref("view"),
        IndexList("indices", "static_indices"),
        COMMA,
        Ref("mask"),
        COLON,
        TypeOf("value"),
        COMMA,
        TypeOf("view"),
        COMMA,
        TypeOf("mask"),
    ],
    examples=[
        "vector.store.mask %v, %view[%row, %col], %mask : vector<4x8xf32>, view<[%m]x[%n]xf32, %layout>, vector<4x8xi1>",
    ],
)

vector_load_expand = Op(
    "vector.load.expand",
    group=vector_ops,
    doc=("Rank-1 masked expand load from consecutive view elements; active lanes consume memory in increasing lane order and inactive lanes take the passthrough value."),
    operands=[
        Operand("view", VIEW, doc="Typed source view."),
        Operand("mask", VECTOR, doc="i1 rank-1 vector mask selecting loaded lanes."),
        Operand("passthrough", VECTOR, doc="Value used for inactive lanes."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    results=[Result("result", VECTOR, doc="Expanded loaded vector value.")],
    attrs=[
        AttrDef(
            "static_indices",
            ATTR_TYPE_I64_ARRAY,
            doc="Static logical origin indices with INT64_MIN sentinels for dynamics.",
        ),
    ],
    constraints=[
        HasI1Element("mask"),
        SameElementType("view", "passthrough", "result"),
        SameShape("mask", "passthrough", "result"),
        SameType("passthrough", "result"),
    ],
    effects=[Reads("view")],
    verify="loom_vector_load_expand_verify",
    format=[
        Ref("view"),
        IndexList("indices", "static_indices"),
        COMMA,
        Ref("mask"),
        COMMA,
        Ref("passthrough"),
        COLON,
        TypeOf("view"),
        COMMA,
        TypeOf("mask"),
        COMMA,
        TypeOf("passthrough"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%v = vector.load.expand %view[%row, %col], %mask, %old : view<[%m]x[%n]xf32, %layout>, vector<4xi1>, vector<4xf32> -> vector<4xf32>",
    ],
)

vector_store_compress = Op(
    "vector.store.compress",
    group=vector_ops,
    doc=("Rank-1 masked compress store to consecutive view elements; active lanes produce memory in increasing lane order and inactive lanes do not write."),
    operands=[
        Operand("value", VECTOR, doc="Rank-1 vector value to store."),
        Operand("view", VIEW, doc="Typed destination view."),
        Operand("mask", VECTOR, doc="i1 rank-1 vector mask selecting stored lanes."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    attrs=[
        AttrDef(
            "static_indices",
            ATTR_TYPE_I64_ARRAY,
            doc="Static logical origin indices with INT64_MIN sentinels for dynamics.",
        ),
    ],
    constraints=[
        HasI1Element("mask"),
        SameElementType("value", "view"),
        SameShape("mask", "value"),
    ],
    effects=[Writes("view")],
    verify="loom_vector_store_compress_verify",
    format=[
        Ref("value"),
        COMMA,
        Ref("view"),
        IndexList("indices", "static_indices"),
        COMMA,
        Ref("mask"),
        COLON,
        TypeOf("value"),
        COMMA,
        TypeOf("view"),
        COMMA,
        TypeOf("mask"),
    ],
    examples=[
        "vector.store.compress %v, %view[%row, %col], %mask : vector<4xf32>, view<[%m]x[%n]xf32, %layout>, vector<4xi1>",
    ],
)

vector_gather = Op(
    "vector.gather",
    group=vector_ops,
    doc="Gather a vector from per-lane signed element offsets relative to a full-rank view origin.",
    operands=[
        Operand("view", VIEW, doc="Typed source view."),
        Operand("offsets", VECTOR, doc="Per-lane signed element offsets from the logical origin."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    results=[Result("result", VECTOR, doc="Gathered vector value.")],
    attrs=[
        AttrDef(
            "static_indices",
            ATTR_TYPE_I64_ARRAY,
            doc="Static logical origin indices with INT64_MIN sentinels for dynamics.",
        ),
    ],
    constraints=[
        SameElementType("view", "result"),
        SameShape("offsets", "result"),
    ],
    effects=[Reads("view")],
    verify="loom_vector_gather_verify",
    format=[
        Ref("view"),
        IndexList("indices", "static_indices"),
        GLUE,
        LBRACKET,
        Ref("offsets"),
        RBRACKET,
        COLON,
        TypeOf("view"),
        COMMA,
        TypeOf("offsets"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%v = vector.gather %view[%row, %col][%offsets] : view<[%m]x[%n]xf32, %layout>, vector<4xindex> -> vector<4xf32>",
    ],
)

vector_scatter = Op(
    "vector.scatter",
    group=vector_ops,
    doc="Non-atomic scatter of a vector to per-lane signed element offsets relative to a full-rank view origin; active lane addresses must be distinct.",
    operands=[
        Operand("value", VECTOR, doc="Vector value to store."),
        Operand("view", VIEW, doc="Typed destination view."),
        Operand("offsets", VECTOR, doc="Per-lane signed element offsets from the logical origin."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    attrs=[
        AttrDef(
            "static_indices",
            ATTR_TYPE_I64_ARRAY,
            doc="Static logical origin indices with INT64_MIN sentinels for dynamics.",
        ),
    ],
    constraints=[
        SameElementType("value", "view"),
        SameShape("offsets", "value"),
    ],
    effects=[Writes("view")],
    verify="loom_vector_scatter_verify",
    format=[
        Ref("value"),
        COMMA,
        Ref("view"),
        IndexList("indices", "static_indices"),
        GLUE,
        LBRACKET,
        Ref("offsets"),
        RBRACKET,
        COLON,
        TypeOf("value"),
        COMMA,
        TypeOf("view"),
        COMMA,
        TypeOf("offsets"),
    ],
    examples=[
        "vector.scatter %v, %view[%row, %col][%offsets] : vector<4xf32>, view<[%m]x[%n]xf32, %layout>, vector<4xindex>",
    ],
)

vector_gather_mask = Op(
    "vector.gather.mask",
    group=vector_ops,
    doc="Masked vector gather from per-lane signed element offsets; masked-off lanes take the passthrough value.",
    operands=[
        Operand("view", VIEW, doc="Typed source view."),
        Operand("offsets", VECTOR, doc="Per-lane signed element offsets from the logical origin."),
        Operand("mask", VECTOR, doc="i1 vector mask selecting gathered lanes."),
        Operand("passthrough", VECTOR, doc="Value used for masked-off lanes."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    results=[Result("result", VECTOR, doc="Gathered vector value.")],
    attrs=[
        AttrDef(
            "static_indices",
            ATTR_TYPE_I64_ARRAY,
            doc="Static logical origin indices with INT64_MIN sentinels for dynamics.",
        ),
    ],
    constraints=[
        HasI1Element("mask"),
        SameElementType("view", "passthrough", "result"),
        SameShape("offsets", "mask", "passthrough", "result"),
        SameType("passthrough", "result"),
    ],
    effects=[Reads("view")],
    verify="loom_vector_gather_mask_verify",
    format=[
        Ref("view"),
        IndexList("indices", "static_indices"),
        GLUE,
        LBRACKET,
        Ref("offsets"),
        RBRACKET,
        COMMA,
        Ref("mask"),
        COMMA,
        Ref("passthrough"),
        COLON,
        TypeOf("view"),
        COMMA,
        TypeOf("offsets"),
        COMMA,
        TypeOf("mask"),
        COMMA,
        TypeOf("passthrough"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%v = vector.gather.mask %view[%row, %col][%offsets], %mask, %old : view<[%m]x[%n]xf32, %layout>, vector<4xindex>, vector<4xi1>, vector<4xf32> -> vector<4xf32>",
    ],
)

vector_scatter_mask = Op(
    "vector.scatter.mask",
    group=vector_ops,
    doc="Masked non-atomic scatter; masked-off lanes leave memory unchanged and active lane addresses must be distinct.",
    operands=[
        Operand("value", VECTOR, doc="Vector value to store."),
        Operand("view", VIEW, doc="Typed destination view."),
        Operand("offsets", VECTOR, doc="Per-lane signed element offsets from the logical origin."),
        Operand("mask", VECTOR, doc="i1 vector mask selecting stored lanes."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    attrs=[
        AttrDef(
            "static_indices",
            ATTR_TYPE_I64_ARRAY,
            doc="Static logical origin indices with INT64_MIN sentinels for dynamics.",
        ),
    ],
    constraints=[
        HasI1Element("mask"),
        SameElementType("value", "view"),
        SameShape("offsets", "mask", "value"),
    ],
    effects=[Writes("view")],
    verify="loom_vector_scatter_mask_verify",
    format=[
        Ref("value"),
        COMMA,
        Ref("view"),
        IndexList("indices", "static_indices"),
        GLUE,
        LBRACKET,
        Ref("offsets"),
        RBRACKET,
        COMMA,
        Ref("mask"),
        COLON,
        TypeOf("value"),
        COMMA,
        TypeOf("view"),
        COMMA,
        TypeOf("offsets"),
        COMMA,
        TypeOf("mask"),
    ],
    examples=[
        "vector.scatter.mask %v, %view[%row, %col][%offsets], %mask : vector<4xf32>, view<[%m]x[%n]xf32, %layout>, vector<4xindex>, vector<4xi1>",
    ],
)


def _atomic_memory_attrs() -> list[AttrDef]:
    return [
        AttrDef("kind", ATTR_TYPE_ENUM, enum_def=AtomicKind),
        AttrDef(
            "ordering",
            ATTR_TYPE_ENUM,
            enum_def=AtomicOrdering,
            doc="Required atomic memory ordering.",
        ),
        AttrDef(
            "scope",
            ATTR_TYPE_ENUM,
            enum_def=AtomicScope,
            doc="Required atomic synchronization scope.",
        ),
        AttrDef(
            "static_indices",
            ATTR_TYPE_I64_ARRAY,
            doc="Static logical origin indices with INT64_MIN sentinels for dynamics.",
        ),
    ]


vector_atomic_reduce = Op(
    "vector.atomic.reduce",
    group=vector_ops,
    doc="Atomic no-result scatter reduction/update into per-lane signed element offsets. Duplicate active lane addresses are allowed and are serialized by the atomic memory contract.",
    operands=[
        Operand("value", VECTOR, doc="Vector contribution for each lane."),
        Operand("view", VIEW, doc="Typed destination view."),
        Operand("offsets", VECTOR, doc="Per-lane signed element offsets from the logical origin."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    attrs=_atomic_memory_attrs(),
    constraints=[
        SameElementType("value", "view"),
        SameShape("offsets", "value"),
    ],
    effects=[ReadWrites("view")],
    verify="loom_vector_atomic_reduce_verify",
    format=[
        TemplateParam("kind"),
        Ref("value"),
        COMMA,
        Ref("view"),
        IndexList("indices", "static_indices"),
        GLUE,
        LBRACKET,
        Ref("offsets"),
        RBRACKET,
        AttrDict(),
        COLON,
        TypeOf("value"),
        COMMA,
        TypeOf("view"),
        COMMA,
        TypeOf("offsets"),
    ],
    examples=[
        "vector.atomic.reduce<addi> %v, %view[%row, %col][%offsets] {ordering = relaxed, scope = workgroup} : vector<4xi32>, view<[%m]x[%n]xi32, %layout>, vector<4xindex>",
    ],
)

vector_atomic_reduce_mask = Op(
    "vector.atomic.reduce.mask",
    group=vector_ops,
    doc="Masked atomic no-result scatter reduction/update; masked-off lanes do not access memory.",
    operands=[
        Operand("value", VECTOR, doc="Vector contribution for each lane."),
        Operand("view", VIEW, doc="Typed destination view."),
        Operand("offsets", VECTOR, doc="Per-lane signed element offsets from the logical origin."),
        Operand("mask", VECTOR, doc="i1 vector mask selecting active atomic lanes."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    attrs=_atomic_memory_attrs(),
    constraints=[
        HasI1Element("mask"),
        SameElementType("value", "view"),
        SameShape("offsets", "mask", "value"),
    ],
    effects=[ReadWrites("view")],
    verify="loom_vector_atomic_reduce_mask_verify",
    format=[
        TemplateParam("kind"),
        Ref("value"),
        COMMA,
        Ref("view"),
        IndexList("indices", "static_indices"),
        GLUE,
        LBRACKET,
        Ref("offsets"),
        RBRACKET,
        COMMA,
        Ref("mask"),
        AttrDict(),
        COLON,
        TypeOf("value"),
        COMMA,
        TypeOf("view"),
        COMMA,
        TypeOf("offsets"),
        COMMA,
        TypeOf("mask"),
    ],
    examples=[
        "vector.atomic.reduce.mask<addf> %v, %view[%row, %col][%offsets], %mask {ordering = relaxed, scope = device} : vector<4xf32>, view<[%m]x[%n]xf32, %layout>, vector<4xindex>, vector<4xi1>",
    ],
)

vector_atomic_rmw = Op(
    "vector.atomic.rmw",
    group=vector_ops,
    doc="Atomic read-modify-write at per-lane signed element offsets, returning the old memory value for each lane.",
    operands=[
        Operand("value", VECTOR, doc="Vector update value for each lane."),
        Operand("view", VIEW, doc="Typed destination view."),
        Operand("offsets", VECTOR, doc="Per-lane signed element offsets from the logical origin."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    results=[Result("result", VECTOR, doc="Old memory values read by the atomic operations.")],
    attrs=_atomic_memory_attrs(),
    constraints=[
        SameElementType("value", "view", "result"),
        SameShape("offsets", "value", "result"),
        SameType("value", "result"),
    ],
    effects=[ReadWrites("view")],
    verify="loom_vector_atomic_rmw_verify",
    format=[
        TemplateParam("kind"),
        Ref("value"),
        COMMA,
        Ref("view"),
        IndexList("indices", "static_indices"),
        GLUE,
        LBRACKET,
        Ref("offsets"),
        RBRACKET,
        AttrDict(),
        COLON,
        TypeOf("value"),
        COMMA,
        TypeOf("view"),
        COMMA,
        TypeOf("offsets"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%old = vector.atomic.rmw<addi> %v, %view[%row, %col][%offsets] {ordering = relaxed, scope = workgroup} : vector<4xi32>, view<[%m]x[%n]xi32, %layout>, vector<4xindex> -> vector<4xi32>",
    ],
)

vector_atomic_rmw_mask = Op(
    "vector.atomic.rmw.mask",
    group=vector_ops,
    doc="Masked atomic read-modify-write. Masked-off result lanes take the explicit passthrough value.",
    operands=[
        Operand("value", VECTOR, doc="Vector update value for each lane."),
        Operand("view", VIEW, doc="Typed destination view."),
        Operand("offsets", VECTOR, doc="Per-lane signed element offsets from the logical origin."),
        Operand("mask", VECTOR, doc="i1 vector mask selecting active atomic lanes."),
        Operand("passthrough", VECTOR, doc="Value used for masked-off result lanes."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    results=[Result("result", VECTOR, doc="Old memory values for active lanes and passthrough for inactive lanes.")],
    attrs=_atomic_memory_attrs(),
    constraints=[
        HasI1Element("mask"),
        SameElementType("value", "view", "passthrough", "result"),
        SameShape("offsets", "mask", "value"),
        SameType("value", "passthrough", "result"),
    ],
    effects=[ReadWrites("view")],
    verify="loom_vector_atomic_rmw_mask_verify",
    format=[
        TemplateParam("kind"),
        Ref("value"),
        COMMA,
        Ref("view"),
        IndexList("indices", "static_indices"),
        GLUE,
        LBRACKET,
        Ref("offsets"),
        RBRACKET,
        COMMA,
        Ref("mask"),
        COMMA,
        Ref("passthrough"),
        AttrDict(),
        COLON,
        TypeOf("value"),
        COMMA,
        TypeOf("view"),
        COMMA,
        TypeOf("offsets"),
        COMMA,
        TypeOf("mask"),
        COMMA,
        TypeOf("passthrough"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%old = vector.atomic.rmw.mask<addf> %v, %view[%row, %col][%offsets], %mask, %passthrough {ordering = relaxed, scope = device} : vector<4xf32>, view<[%m]x[%n]xf32, %layout>, vector<4xindex>, vector<4xi1>, vector<4xf32> -> vector<4xf32>",
    ],
)


# ============================================================================
# Selection and comparison
# ============================================================================

vector_select = Op(
    "vector.select",
    group=vector_ops,
    doc="Lanewise select from two same-typed vector values using an i1 mask vector.",
    operands=[
        Operand("condition", VECTOR),
        Operand("true_value", VECTOR),
        Operand("false_value", VECTOR),
    ],
    results=[Result("result", VECTOR)],
    constraints=[
        HasI1Element("condition"),
        SameKind("condition", "result"),
        SameShape("condition", "true_value", "false_value", "result"),
        SameType("true_value", "false_value", "result"),
    ],
    traits=[PURE, ELEMENTWISE],
    format=[
        Ref("condition"),
        COMMA,
        Ref("true_value"),
        COMMA,
        Ref("false_value"),
        COLON,
        TypeOf("result"),
    ],
    examples=["%r = vector.select %mask, %a, %b : vector<16xf32>"],
)

vector_cmpi = Op(
    "vector.cmpi",
    group=vector_ops,
    doc="Lanewise integer comparison producing an i1 mask vector.",
    operands=[
        Operand("lhs", VECTOR),
        Operand("rhs", VECTOR),
    ],
    results=[Result("result", VECTOR)],
    attrs=[AttrDef("predicate", ATTR_TYPE_ENUM, enum_def=CmpIPredicate)],
    constraints=[
        HasIntegerElement("lhs"),
        HasI1Element("result"),
        SameKind("lhs", "result"),
        SameShape("lhs", "rhs", "result"),
        SameElementType("lhs", "rhs"),
    ],
    traits=[PURE, ELEMENTWISE],
    format=[
        Attr("predicate"),
        COMMA,
        Ref("lhs"),
        COMMA,
        Ref("rhs"),
        COLON,
        TypeOf("lhs"),
        ARROW,
        ResultType("result"),
    ],
    examples=["%m = vector.cmpi slt, %lhs, %rhs : vector<16xi32> -> vector<16xi1>"],
)

vector_cmpf = Op(
    "vector.cmpf",
    group=vector_ops,
    doc="Lanewise floating-point comparison producing an i1 mask vector.",
    operands=[
        Operand("lhs", VECTOR),
        Operand("rhs", VECTOR),
    ],
    results=[Result("result", VECTOR)],
    attrs=[AttrDef("predicate", ATTR_TYPE_ENUM, enum_def=CmpFPredicate)],
    constraints=[
        HasFloatElement("lhs"),
        HasI1Element("result"),
        SameKind("lhs", "result"),
        SameShape("lhs", "rhs", "result"),
        SameElementType("lhs", "rhs"),
    ],
    traits=[PURE, ELEMENTWISE],
    format=[
        Attr("predicate"),
        COMMA,
        Ref("lhs"),
        COMMA,
        Ref("rhs"),
        COLON,
        TypeOf("lhs"),
        ARROW,
        ResultType("result"),
    ],
    examples=["%m = vector.cmpf olt, %lhs, %rhs : vector<16xf32> -> vector<16xi1>"],
)


# ============================================================================
# Lanewise arithmetic and math
# ============================================================================

vector_addf = _lanewise_binary(
    "vector.addf",
    result_constraint=FLOAT_ELEMENT,
    doc="Lanewise floating-point addition.",
    commutative=True,
    flags=("assumptions", FloatAssumptionFlags),
)

vector_mulf = _lanewise_binary(
    "vector.mulf",
    result_constraint=FLOAT_ELEMENT,
    doc="Lanewise floating-point multiplication.",
    commutative=True,
    flags=("assumptions", FloatAssumptionFlags),
)

vector_fmaf = Op(
    "vector.fmaf",
    group=vector_ops,
    doc="Lanewise fused multiply-add: a*b + c with single rounding.",
    operands=[
        Operand("a", VECTOR),
        Operand("b", VECTOR),
        Operand("c", VECTOR),
    ],
    results=[Result("result", VECTOR)],
    attrs=[
        AttrDef(
            "assumptions",
            ATTR_TYPE_FLAGS,
            optional=True,
            enum_def=FloatAssumptionFlags,
        )
    ],
    constraints=[
        HasFloatElement("result"),
        SameType("a", "b", "c", "result"),
    ],
    traits=[PURE, ELEMENTWISE],
    format=[
        Flags("assumptions"),
        Ref("a"),
        COMMA,
        Ref("b"),
        COMMA,
        Ref("c"),
        COLON,
        TypeOf("result"),
    ],
    examples=["%r = vector.fmaf %a, %b, %c : vector<16xf32>"],
)

vector_addi = _lanewise_binary(
    "vector.addi",
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise integer addition.",
    commutative=True,
    flags=("overflow", IntOverflowFlags),
)

vector_muli = _lanewise_binary(
    "vector.muli",
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise integer multiplication.",
    commutative=True,
    flags=("overflow", IntOverflowFlags),
)

vector_sqrtf = _lanewise_unary(
    "vector.sqrtf",
    result_constraint=FLOAT_ELEMENT,
    doc="Lanewise square root.",
    flags=("assumptions", FloatAssumptionFlags),
)


# ============================================================================
# Conversions
# ============================================================================

vector_extf = _vector_cast(
    "vector.extf",
    source_constraint=HasFloatElement,
    result_constraint=FLOAT_ELEMENT,
    doc="Lanewise floating-point precision extension.",
    verify="loom_vector_extf_verify",
)

vector_fptrunc = _vector_cast(
    "vector.fptrunc",
    source_constraint=HasFloatElement,
    result_constraint=FLOAT_ELEMENT,
    doc="Lanewise floating-point precision truncation.",
    verify="loom_vector_fptrunc_verify",
)

vector_extsi = _vector_cast(
    "vector.extsi",
    source_constraint=HasIntegerElement,
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise signed integer extension.",
    verify="loom_vector_extsi_verify",
)

vector_extui = _vector_cast(
    "vector.extui",
    source_constraint=HasIntegerElement,
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise unsigned integer extension.",
    verify="loom_vector_extui_verify",
)

vector_trunci = _vector_cast(
    "vector.trunci",
    source_constraint=HasIntegerElement,
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise integer truncation.",
    verify="loom_vector_trunci_verify",
)

vector_sitofp = _vector_cast(
    "vector.sitofp",
    source_constraint=HasIntegerElement,
    result_constraint=FLOAT_ELEMENT,
    doc="Lanewise signed integer to floating-point conversion.",
)

vector_uitofp = _vector_cast(
    "vector.uitofp",
    source_constraint=HasIntegerElement,
    result_constraint=FLOAT_ELEMENT,
    doc="Lanewise unsigned integer to floating-point conversion.",
)

vector_fptosi = _vector_cast(
    "vector.fptosi",
    source_constraint=HasFloatElement,
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise floating-point to signed integer conversion.",
)

vector_fptoui = _vector_cast(
    "vector.fptoui",
    source_constraint=HasFloatElement,
    result_constraint=INTEGER_ELEMENT,
    doc="Lanewise floating-point to unsigned integer conversion.",
)

vector_bitcast = Op(
    "vector.bitcast",
    group=vector_ops,
    doc="Bitwise reinterpretation between vector register types with the same total bit count.",
    operands=[Operand("input", VECTOR)],
    results=[Result("result", VECTOR)],
    verify="loom_vector_bitcast_verify",
    traits=[PURE],
    format=[
        Ref("input"),
        COLON,
        TypeOf("input"),
        kw("to"),
        TypeOf("result"),
    ],
    examples=[
        "%r = vector.bitcast %input : vector<16xf32> to vector<16xi32>",
        "%s = vector.bitcast %bytes : vector<2xi8> to vector<1xf16>",
    ],
)


# ============================================================================
# Bitfield packing
# ============================================================================


def _bitfield_attrs() -> list[AttrDef]:
    return [
        AttrDef(
            "offset",
            ATTR_TYPE_I64,
            doc="Least-significant bit position of the field within each source lane.",
        ),
        AttrDef(
            "width",
            ATTR_TYPE_I64,
            doc="Number of bits in the field extracted or inserted in each lane.",
        ),
    ]


vector_bitfield_extractu = Op(
    "vector.bitfield.extractu",
    group=vector_ops,
    doc="Extract and zero-extend one fixed bitfield from each integer source lane.",
    operands=[Operand("source", VECTOR)],
    results=[Result("result", VECTOR)],
    attrs=_bitfield_attrs(),
    constraints=[
        HasIntegerElement("source"),
        HasIntegerElement("result"),
        SameKind("source", "result"),
        SameShape("source", "result"),
    ],
    verify="loom_vector_bitfield_extractu_verify",
    traits=[PURE, ELEMENTWISE],
    format=[
        Ref("source"),
        AttrDict(),
        COLON,
        TypeOf("source"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%lo = vector.bitfield.extractu %bytes {offset = 0, width = 4} : vector<16xi8> -> vector<16xi32>",
    ],
)

vector_bitfield_extracts = Op(
    "vector.bitfield.extracts",
    group=vector_ops,
    doc="Extract and sign-extend one fixed bitfield from each integer source lane.",
    operands=[Operand("source", VECTOR)],
    results=[Result("result", VECTOR)],
    attrs=_bitfield_attrs(),
    constraints=[
        HasIntegerElement("source"),
        HasIntegerElement("result"),
        SameKind("source", "result"),
        SameShape("source", "result"),
    ],
    verify="loom_vector_bitfield_extracts_verify",
    traits=[PURE, ELEMENTWISE],
    format=[
        Ref("source"),
        AttrDict(),
        COLON,
        TypeOf("source"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%signed = vector.bitfield.extracts %bytes {offset = 4, width = 4} : vector<16xi8> -> vector<16xi32>",
    ],
)

vector_bitfield_insert = Op(
    "vector.bitfield.insert",
    group=vector_ops,
    doc="Insert the low bits of each integer field lane into a fixed bitfield of each integer base lane.",
    operands=[
        Operand("field", VECTOR, doc="Integer field values. Only the low `width` bits are inserted."),
        Operand("base", VECTOR, doc="Integer base lanes whose target bitfield is replaced."),
    ],
    results=[Result("result", VECTOR)],
    attrs=_bitfield_attrs(),
    constraints=[
        HasIntegerElement("field"),
        HasIntegerElement("base"),
        HasIntegerElement("result"),
        SameShape("field", "base", "result"),
        SameType("base", "result"),
    ],
    verify="loom_vector_bitfield_insert_verify",
    traits=[PURE, ELEMENTWISE],
    format=[
        Ref("field"),
        kw("into"),
        Ref("base"),
        AttrDict(),
        COLON,
        TypeOf("field"),
        COMMA,
        TypeOf("base"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%packed = vector.bitfield.insert %lo into %zero {offset = 0, width = 4} : vector<16xi32>, vector<16xi8> -> vector<16xi8>",
    ],
)


# ============================================================================
# Reductions
# ============================================================================

vector_reduce = Op(
    "vector.reduce",
    group=vector_ops,
    doc="Reduce all lanes of a vector into a scalar accumulator/result.",
    operands=[
        Operand("input", VECTOR),
        Operand("init", SCALAR),
    ],
    results=[Result("result", SCALAR)],
    attrs=[AttrDef("kind", ATTR_TYPE_ENUM, enum_def=CombiningKind)],
    constraints=[
        SameType("init", "result"),
        SameElementType("input", "init", "result"),
    ],
    verify="loom_vector_reduce_verify",
    traits=[PURE],
    format=[
        TemplateParam("kind"),
        Ref("input"),
        COMMA,
        Ref("init"),
        COLON,
        TypeOf("input"),
        ARROW,
        ResultType("result"),
    ],
    examples=["%sum = vector.reduce<addf> %v, %zero : vector<16xf32> -> f32"],
)


# ============================================================================
# Registry
# ============================================================================

ALL_VECTOR_OPS: tuple[Op, ...] = (
    vector_constant,
    vector_splat,
    vector_broadcast,
    vector_from_elements,
    vector_extract,
    vector_insert,
    vector_slice,
    vector_concat,
    vector_transpose,
    vector_shuffle,
    vector_interleave,
    vector_deinterleave,
    vector_table_lookup,
    vector_load,
    vector_store,
    vector_load_mask,
    vector_store_mask,
    vector_load_expand,
    vector_store_compress,
    vector_gather,
    vector_scatter,
    vector_gather_mask,
    vector_scatter_mask,
    vector_atomic_reduce,
    vector_atomic_reduce_mask,
    vector_atomic_rmw,
    vector_atomic_rmw_mask,
    vector_select,
    vector_cmpi,
    vector_cmpf,
    vector_addf,
    vector_mulf,
    vector_fmaf,
    vector_addi,
    vector_muli,
    vector_sqrtf,
    vector_extf,
    vector_fptrunc,
    vector_extsi,
    vector_extui,
    vector_trunci,
    vector_sitofp,
    vector_uitofp,
    vector_fptosi,
    vector_fptoui,
    vector_bitcast,
    vector_bitfield_extractu,
    vector_bitfield_extracts,
    vector_bitfield_insert,
    vector_reduce,
)
