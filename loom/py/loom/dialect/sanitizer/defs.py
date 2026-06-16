# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Sanitizer dialect op definitions."""

from loom.assembly import (
    ARROW,
    COLON,
    IndexList,
    PredicateList,
    Ref,
    Refs,
    ResultType,
    TemplateParam,
    TypeOf,
    TypesOf,
)
from loom.dsl import (
    ANY,
    ATTR_TYPE_ENUM,
    ATTR_TYPE_I64_ARRAY,
    FACT_IDENTITY,
    INDEX,
    REFINABLE_RESULT_TYPE_REFS,
    UNKNOWN_EFFECTS,
    VIEW,
    AttrDef,
    Dialect,
    EnumCase,
    EnumDef,
    Op,
    Operand,
    RanksMatch,
    Result,
    SameElementType,
    VariadicValuesMatch,
)

__all__ = [
    "ALL_SANITIZER_OPS",
    "SanitizerAccessKind",
    "sanitizer_assert_access",
    "sanitizer_assert_layout",
    "sanitizer_assert_op",
    "sanitizer_assert_value",
    "sanitizer_ops",
]


sanitizer_ops = Dialect(
    "sanitizer",
    dialect_id=0x1D,
    doc=(
        "Executable diagnostic assertions. Assertions refine SSA facts on the "
        "continuing path while preserving the dynamic trap/report boundary "
        "until canonicalization proves that boundary unreachable."
    ),
)


SanitizerAccessKind = EnumDef(
    "SanitizerAccessKind",
    [
        EnumCase("read", 0, doc="Assertion covers a logical read access."),
        EnumCase("write", 1, doc="Assertion covers a logical write access."),
        EnumCase(
            "read_write",
            2,
            doc="Assertion covers a logical read-modify-write access.",
        ),
    ],
    doc="Logical memory access kind covered by a sanitizer access assertion.",
)


# ============================================================================
# sanitizer.assert.access
# ============================================================================

sanitizer_assert_access = Op(
    "sanitizer.assert.access",
    group=sanitizer_ops,
    doc=(
        "Assert that a logical indexed view access is valid. The assertion "
        "has the same index-list shape as ordinary view memory operations so "
        "source-level memory contracts remain typed until target lowering "
        "materializes address checks."
    ),
    operands=[
        Operand("view", VIEW, doc="Typed view being accessed."),
        Operand(
            "indices",
            INDEX,
            variadic=True,
            doc="Dynamic logical element indices.",
        ),
    ],
    attrs=[
        AttrDef(
            "kind",
            ATTR_TYPE_ENUM,
            enum_def=SanitizerAccessKind,
            doc="Logical access kind being asserted.",
        ),
        AttrDef(
            "static_indices",
            ATTR_TYPE_I64_ARRAY,
            doc="Static logical element indices with INT64_MIN sentinels for dynamics.",
        ),
    ],
    traits=[UNKNOWN_EFFECTS],
    verify="loom_sanitizer_assert_access_verify",
    format=[
        TemplateParam("kind"),
        Ref("view"),
        IndexList("indices", "static_indices"),
        COLON,
        TypeOf("view"),
    ],
    examples=[
        "sanitizer.assert.access<read> %view[%row, %col] : view<[%M]x[%N]xf32, %layout>",
    ],
)


# ============================================================================
# sanitizer.assert.value
# ============================================================================

sanitizer_assert_value = Op(
    "sanitizer.assert.value",
    group=sanitizer_ops,
    doc=(
        "Assert predicate constraints over SSA values and return checked "
        "identity aliases. Unlike assume ops, this op is executable: failing "
        "the assertion reports the site and aborts execution. Passing the "
        "assertion refines facts for the returned aliases."
    ),
    operands=[
        Operand(
            "values",
            ANY,
            variadic=True,
            doc="Values observed by the assertion predicates.",
        ),
    ],
    results=[
        Result(
            "results",
            ANY,
            variadic=True,
            doc="Same values on the path where every predicate held.",
        ),
    ],
    attrs=[
        AttrDef(
            "predicates",
            "predicate_list",
            doc="Predicate constraints that must hold at runtime.",
        ),
    ],
    constraints=[VariadicValuesMatch("values", "results")],
    traits=[UNKNOWN_EFFECTS, FACT_IDENTITY],
    canonicalize="loom_sanitizer_assert_value_canonicalize",
    facts="loom_sanitizer_assert_value_facts",
    verify="loom_sanitizer_assert_value_verify",
    format=[
        Refs("values"),
        PredicateList("predicates"),
        COLON,
        TypesOf("results"),
    ],
    examples=[
        "%n_checked = sanitizer.assert.value %n [range(%n, 0, 4096), mul(%n, 16)] : index",
        "%x_checked, %y_checked = sanitizer.assert.value %x, %y [lt(%x, %y)] : i32, i32",
    ],
)


# ============================================================================
# sanitizer.assert.op
# ============================================================================

sanitizer_assert_op = Op(
    "sanitizer.assert.op",
    group=sanitizer_ops,
    doc=(
        "Assert operation-level predicate constraints without producing "
        "checked aliases. This is the executable form for contracts such as "
        "valid divide operands, shift counts, overflow preconditions, and "
        "other facts where the checked operation itself remains the semantic "
        "anchor."
    ),
    operands=[
        Operand(
            "values",
            ANY,
            variadic=True,
            doc="Values observed by the assertion predicates.",
        ),
    ],
    attrs=[
        AttrDef(
            "predicates",
            "predicate_list",
            doc="Predicate constraints that must hold at runtime.",
        ),
    ],
    traits=[UNKNOWN_EFFECTS],
    canonicalize="loom_sanitizer_assert_op_canonicalize",
    verify="loom_sanitizer_assert_op_verify",
    format=[
        Refs("values"),
        PredicateList("predicates"),
        COLON,
        TypesOf("values"),
    ],
    examples=[
        "sanitizer.assert.op %lhs, %rhs [ne(%rhs, 0)] : i32, i32",
    ],
)


# ============================================================================
# sanitizer.assert.layout
# ============================================================================

sanitizer_assert_layout = Op(
    "sanitizer.assert.layout",
    group=sanitizer_ops,
    doc=(
        "Assert that a view satisfies a refined layout, shape, or encoding "
        "contract and return the same view on the continuing path. This is "
        "the executable counterpart to view.refine for diagnostics that must "
        "abort instead of trusting unchecked layout facts."
    ),
    operands=[Operand("view", VIEW, doc="Source view to assert and refine.")],
    results=[
        Result(
            "result",
            VIEW,
            doc="Same view on the path where the layout assertion held.",
        )
    ],
    constraints=[
        SameElementType("view", "result"),
        RanksMatch("view", "result"),
    ],
    traits=[UNKNOWN_EFFECTS, REFINABLE_RESULT_TYPE_REFS, FACT_IDENTITY],
    facts="loom_sanitizer_assert_layout_facts",
    type_transfer="loom_sanitizer_assert_layout_type_transfer",
    verify="loom_sanitizer_assert_layout_verify",
    format=[
        Ref("view"),
        COLON,
        TypeOf("view"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%checked = sanitizer.assert.layout %view : view<[%M]x[%N]xf32, %layout> -> view<16x[%N]xf32, %layout>",
    ],
)


ALL_SANITIZER_OPS: tuple[Op, ...] = (
    sanitizer_assert_access,
    sanitizer_assert_value,
    sanitizer_assert_op,
    sanitizer_assert_layout,
)
