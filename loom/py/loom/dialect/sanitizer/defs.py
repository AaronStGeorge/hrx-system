# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Sanitizer dialect op definitions."""

from loom.assembly import COLON, PredicateList, Refs, TypesOf
from loom.dsl import (
    ANY,
    FACT_IDENTITY,
    UNKNOWN_EFFECTS,
    AttrDef,
    Dialect,
    Op,
    Operand,
    Result,
    VariadicValuesMatch,
)

__all__ = [
    "ALL_SANITIZER_OPS",
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


ALL_SANITIZER_OPS: tuple[Op, ...] = (sanitizer_assert_value,)
