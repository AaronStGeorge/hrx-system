# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Index dialect definitions.

The index dialect owns scalar address-domain values: logical coordinates
(`index`), physical byte offsets (`offset`), and explicit conversion to or from
fixed-width integer payloads. It intentionally keeps a small op surface so that
agents can author address math without falling back to scalar integer syntax.
"""

from loom.assembly import (
    COLON,
    COMMA,
    Attr,
    PredicateList,
    Ref,
    Refs,
    ResultType,
    TypeOf,
    TypesOf,
)
from loom.dsl import (
    ADDRESS,
    CONSTANT_LIKE,
    I1,
    INDEX,
    PURE,
    SCALAR,
    AttrDef,
    Dialect,
    EnumCase,
    EnumDef,
    Op,
    Operand,
    Result,
    SameType,
    binary_op,
    cast_op,
    comparison_op,
)

__all__ = [
    "index_ops",
    "IndexPredicate",
    "ALL_INDEX_OPS",
]

index_ops = Dialect(
    "index",
    dialect_id=0x0F,
    doc=("Scalar address-domain operations over index, offset, and explicit integer boundary casts."),
)

IndexPredicate = EnumDef(
    "IndexPredicate",
    [
        EnumCase("eq", 0, doc="Equal."),
        EnumCase("ne", 1, doc="Not equal."),
        EnumCase("slt", 2, doc="Signed less than."),
        EnumCase("sle", 3, doc="Signed less or equal."),
        EnumCase("sgt", 4, doc="Signed greater than."),
        EnumCase("sge", 5, doc="Signed greater or equal."),
        EnumCase("ult", 6, doc="Unsigned less than."),
        EnumCase("ule", 7, doc="Unsigned less or equal."),
        EnumCase("ugt", 8, doc="Unsigned greater than."),
        EnumCase("uge", 9, doc="Unsigned greater or equal."),
    ],
    doc="Address-domain comparison predicates.",
)

# ============================================================================
# Construction and conversion
# ============================================================================

index_constant = Op(
    "index.constant",
    group=index_ops,
    doc=(
        "Materialize a compile-time address-domain constant. The result type "
        "must be index for logical coordinates or offset for physical byte "
        "counts; fixed-width integer payload constants remain scalar.constant."
    ),
    results=[Result("result", ADDRESS)],
    attrs=[AttrDef("value", "any", doc="The constant integer value.")],
    traits=[PURE, CONSTANT_LIKE],
    facts="loom_index_constant_facts",
    verify="loom_index_constant_verify",
    format=[Attr("value"), COLON, TypeOf("result")],
    examples=[
        "%c0 = index.constant 0 : index",
        "%base = index.constant 0 : offset",
    ],
)

index_cast = cast_op(
    "index.cast",
    group=index_ops,
    from_constraint=SCALAR,
    to_constraint=SCALAR,
    doc=("Explicit conversion at an address boundary. At least one side must be index or offset; pure integer width changes use scalar.extsi, scalar.extui, or scalar.trunci."),
    facts="loom_index_cast_facts",
    verify="loom_index_cast_verify",
    examples=[
        "%i = index.cast %n : i64 to index",
        "%bytes = index.cast %raw : i64 to offset",
        "%n = index.cast %i : index to i64",
    ],
)

# ============================================================================
# Assumptions
# ============================================================================

index_assume = Op(
    "index.assume",
    group=index_ops,
    doc="Identity with predicate constraints on index or offset results.",
    operands=[Operand("values", ADDRESS, variadic=True)],
    results=[Result("results", ADDRESS, variadic=True)],
    attrs=[AttrDef("predicates", "predicate_list")],
    traits=[PURE],
    facts="loom_index_assume_facts",
    verify="loom_index_assume_verify",
    format=[
        Refs("values"),
        PredicateList("predicates"),
        COLON,
        TypesOf("results"),
    ],
    examples=[
        "%n2 = index.assume %n [mul(%n, 16)] : index",
        "%end2 = index.assume %end [range(%end, 0, 4096)] : offset",
        "%n2, %off2 = index.assume %n, %off [mul(%n, 16), mul(%off, 64)] : index, offset",
    ],
)

# ============================================================================
# Address-domain arithmetic
# ============================================================================

index_add = binary_op(
    "index.add",
    group=index_ops,
    type_constraint=ADDRESS,
    doc="Address-domain addition. Operands and result must all be index or all offset.",
    commutative=True,
    facts="loom_index_add_facts",
    examples=[
        "%r = index.add %lhs, %rhs : index",
        "%bytes = index.add %base, %delta : offset",
    ],
)

index_sub = binary_op(
    "index.sub",
    group=index_ops,
    type_constraint=ADDRESS,
    doc="Address-domain subtraction. Operands and result must all be index or all offset.",
    facts="loom_index_sub_facts",
    examples=[
        "%r = index.sub %lhs, %rhs : index",
        "%delta = index.sub %end, %base : offset",
    ],
)

index_mul = binary_op(
    "index.mul",
    group=index_ops,
    type_constraint=INDEX,
    doc="Logical coordinate multiplication. Offsets are physical byte counts and cannot be multiplied with this op.",
    commutative=True,
    facts="loom_index_mul_facts",
    examples=["%r = index.mul %lhs, %rhs : index"],
)

index_madd = Op(
    "index.madd",
    group=index_ops,
    doc="Logical coordinate multiply-add: a*b + c. Offsets are physical byte counts and cannot be multiplied with this op.",
    operands=[
        Operand("a", INDEX),
        Operand("b", INDEX),
        Operand("c", INDEX),
    ],
    results=[Result("result", INDEX)],
    constraints=[SameType("a", "b", "c", "result")],
    traits=[PURE],
    format=[
        Ref("a"),
        COMMA,
        Ref("b"),
        COMMA,
        Ref("c"),
        COLON,
        TypeOf("result"),
    ],
    facts="loom_index_madd_facts",
    examples=["%r = index.madd %a, %b, %c : index"],
)

# ============================================================================
# Predicates and selection
# ============================================================================

index_cmp = comparison_op(
    "index.cmp",
    group=index_ops,
    type_constraint=ADDRESS,
    predicates=IndexPredicate,
    doc="Address-domain comparison. Operands must both be index or both be offset.",
    facts="loom_index_cmp_facts",
    examples=[
        "%p = index.cmp slt, %i, %n : index",
        "%inside = index.cmp ule, %byte_offset, %limit : offset",
    ],
)

index_select = Op(
    "index.select",
    group=index_ops,
    doc="Select between two same-typed address-domain values using an i1 condition.",
    operands=[
        Operand("condition", I1),
        Operand("true_value", ADDRESS),
        Operand("false_value", ADDRESS),
    ],
    results=[Result("result", ADDRESS)],
    constraints=[SameType("true_value", "false_value", "result")],
    traits=[PURE],
    format=[
        Ref("condition"),
        COMMA,
        Ref("true_value"),
        COMMA,
        Ref("false_value"),
        COLON,
        ResultType("result"),
    ],
    facts="loom_index_select_facts",
    examples=[
        "%r = index.select %cond, %t, %f : index",
        "%bytes = index.select %cond, %base, %limit : offset",
    ],
)

# ============================================================================
# Registry
# ============================================================================

ALL_INDEX_OPS: tuple[Op, ...] = (
    index_constant,
    index_cast,
    index_assume,
    index_add,
    index_sub,
    index_mul,
    index_madd,
    index_cmp,
    index_select,
)
