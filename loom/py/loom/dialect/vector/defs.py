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
    Attr,
    Flags,
    FormatElement,
    IndexList,
    Ref,
    Refs,
    ResultType,
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
    doc="Lanewise bit reinterpretation between same-shaped vector types.",
    operands=[Operand("input", VECTOR)],
    results=[Result("result", VECTOR)],
    constraints=[SameShape("input", "result")],
    verify="loom_vector_bitcast_verify",
    traits=[PURE, ELEMENTWISE],
    format=[
        Ref("input"),
        COLON,
        TypeOf("input"),
        kw("to"),
        TypeOf("result"),
    ],
    examples=["%r = vector.bitcast %input : vector<16xf32> to vector<16xi32>"],
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
    vector_load,
    vector_store,
    vector_load_mask,
    vector_store_mask,
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
    vector_reduce,
)
