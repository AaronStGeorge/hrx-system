# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source-of-truth rows for ordinary SPIR-V scalar ALU predicates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IntegerComparePredicate:
    source_predicate: str
    descriptor_suffix: str
    mnemonic: str
    opcode: str


INTEGER_COMPARE_PREDICATES = (
    IntegerComparePredicate("eq", "i_equal", "OpIEqual", "LOOM_SPIRV_OP_I_EQUAL"),
    IntegerComparePredicate(
        "ne",
        "i_not_equal",
        "OpINotEqual",
        "LOOM_SPIRV_OP_I_NOT_EQUAL",
    ),
    IntegerComparePredicate(
        "slt",
        "s_less_than",
        "OpSLessThan",
        "LOOM_SPIRV_OP_S_LESS_THAN",
    ),
    IntegerComparePredicate(
        "sle",
        "s_less_than_equal",
        "OpSLessThanEqual",
        "LOOM_SPIRV_OP_S_LESS_THAN_EQUAL",
    ),
    IntegerComparePredicate(
        "sgt",
        "s_greater_than",
        "OpSGreaterThan",
        "LOOM_SPIRV_OP_S_GREATER_THAN",
    ),
    IntegerComparePredicate(
        "sge",
        "s_greater_than_equal",
        "OpSGreaterThanEqual",
        "LOOM_SPIRV_OP_S_GREATER_THAN_EQUAL",
    ),
    IntegerComparePredicate(
        "ult",
        "u_less_than",
        "OpULessThan",
        "LOOM_SPIRV_OP_U_LESS_THAN",
    ),
    IntegerComparePredicate(
        "ule",
        "u_less_than_equal",
        "OpULessThanEqual",
        "LOOM_SPIRV_OP_U_LESS_THAN_EQUAL",
    ),
    IntegerComparePredicate(
        "ugt",
        "u_greater_than",
        "OpUGreaterThan",
        "LOOM_SPIRV_OP_U_GREATER_THAN",
    ),
    IntegerComparePredicate(
        "uge",
        "u_greater_than_equal",
        "OpUGreaterThanEqual",
        "LOOM_SPIRV_OP_U_GREATER_THAN_EQUAL",
    ),
)

UNSIGNED_INTEGER_COMPARE_PREDICATES = tuple(
    row
    for row in INTEGER_COMPARE_PREDICATES
    if row.source_predicate in ("eq", "ne", "ult", "ule", "ugt", "uge")
)
