# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source-of-truth rows for ordinary SPIR-V scalar ALU predicates."""

from __future__ import annotations

from dataclasses import dataclass

from loom.target.arch.spirv.features import feature_bits_value


@dataclass(frozen=True, slots=True)
class ScalarAluType:
    source_type: str
    suffix: str
    scalar_enum: str
    feature_atoms: tuple[str, ...] = ()

    @property
    def feature_bits(self) -> int:
        return feature_bits_value(self.feature_atoms)


@dataclass(frozen=True, slots=True)
class ScalarBinaryOperation:
    source_op_key: str
    descriptor_suffix: str
    mnemonic: str
    opcode: str


@dataclass(frozen=True, slots=True)
class IntegerComparePredicate:
    source_predicate: str
    descriptor_suffix: str
    mnemonic: str
    opcode: str


INTEGER_SCALAR_ALU_TYPES = (
    ScalarAluType(
        source_type="i8",
        suffix="i8",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_S8",
        feature_atoms=("int8",),
    ),
    ScalarAluType(
        source_type="i16",
        suffix="i16",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_S16",
        feature_atoms=("int16",),
    ),
    ScalarAluType(
        source_type="i32",
        suffix="i32",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_S32",
    ),
    ScalarAluType(
        source_type="i64",
        suffix="i64",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_S64",
        feature_atoms=("int64",),
    ),
)

FLOAT_SCALAR_ALU_TYPES = (
    ScalarAluType(
        source_type="f16",
        suffix="f16",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_F16",
        feature_atoms=("float16",),
    ),
    ScalarAluType(
        source_type="f32",
        suffix="f32",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_F32",
    ),
    ScalarAluType(
        source_type="f64",
        suffix="f64",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_F64",
        feature_atoms=("float64",),
    ),
)

SCALAR_ALU_TYPES = INTEGER_SCALAR_ALU_TYPES + FLOAT_SCALAR_ALU_TYPES

OFFSET64_ALU_TYPE = ScalarAluType(
    source_type="offset",
    suffix="offset64",
    scalar_enum="LOOM_SPIRV_SCALAR_TYPE_U64",
)

INTEGER_BINARY_OPERATIONS = (
    ScalarBinaryOperation("addi", "iadd", "OpIAdd", "LOOM_SPIRV_OP_I_ADD"),
    ScalarBinaryOperation("subi", "isub", "OpISub", "LOOM_SPIRV_OP_I_SUB"),
    ScalarBinaryOperation("muli", "imul", "OpIMul", "LOOM_SPIRV_OP_I_MUL"),
    ScalarBinaryOperation("divsi", "sdiv", "OpSDiv", "LOOM_SPIRV_OP_S_DIV"),
    ScalarBinaryOperation("remsi", "srem", "OpSRem", "LOOM_SPIRV_OP_S_REM"),
)

FLOAT_BINARY_OPERATIONS = (
    ScalarBinaryOperation("addf", "fadd", "OpFAdd", "LOOM_SPIRV_OP_F_ADD"),
    ScalarBinaryOperation("subf", "fsub", "OpFSub", "LOOM_SPIRV_OP_F_SUB"),
    ScalarBinaryOperation("mulf", "fmul", "OpFMul", "LOOM_SPIRV_OP_F_MUL"),
    ScalarBinaryOperation("divf", "fdiv", "OpFDiv", "LOOM_SPIRV_OP_F_DIV"),
    ScalarBinaryOperation("remf", "frem", "OpFRem", "LOOM_SPIRV_OP_F_REM"),
)

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
