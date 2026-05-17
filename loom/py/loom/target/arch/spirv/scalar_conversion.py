# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source-of-truth rows for ordinary SPIR-V scalar conversions."""

from __future__ import annotations

from dataclasses import dataclass

from loom.target.arch.spirv.scalar_alu import (
    FLOAT_SCALAR_ALU_TYPES,
    INTEGER_SCALAR_ALU_TYPES,
    ScalarAluType,
)

_BIT_WIDTHS = {
    "i8": 8,
    "i16": 16,
    "i32": 32,
    "i64": 64,
    "f16": 16,
    "f32": 32,
    "f64": 64,
}


@dataclass(frozen=True, slots=True)
class ScalarConversion:
    source_op_key: str
    descriptor_suffix: str
    mnemonic: str
    opcode: str
    source_type: ScalarAluType
    result_type: ScalarAluType

    @property
    def key(self) -> str:
        return (
            f"spirv.op_{self.descriptor_suffix}."
            f"{self.source_type.suffix}.{self.result_type.suffix}"
        )

    @property
    def display_mnemonic(self) -> str:
        return f"{self.mnemonic}.{self.source_type.suffix}.{self.result_type.suffix}"

    @property
    def feature_bits(self) -> int:
        return self.source_type.feature_bits | self.result_type.feature_bits


def _bit_width(scalar: ScalarAluType) -> int:
    return _BIT_WIDTHS[scalar.source_type]


def _integer_width_conversions() -> tuple[ScalarConversion, ...]:
    rows: list[ScalarConversion] = []
    for source_type in INTEGER_SCALAR_ALU_TYPES:
        for result_type in INTEGER_SCALAR_ALU_TYPES:
            if source_type == result_type:
                continue
            source_width = _bit_width(source_type)
            result_width = _bit_width(result_type)
            rows.append(
                ScalarConversion(
                    source_op_key="extsi" if source_width < result_width else "trunci",
                    descriptor_suffix="s_convert",
                    mnemonic="OpSConvert",
                    opcode="LOOM_SPIRV_OP_S_CONVERT",
                    source_type=source_type,
                    result_type=result_type,
                )
            )
    return tuple(rows)


def _float_width_conversions() -> tuple[ScalarConversion, ...]:
    rows: list[ScalarConversion] = []
    for source_type in FLOAT_SCALAR_ALU_TYPES:
        for result_type in FLOAT_SCALAR_ALU_TYPES:
            if source_type == result_type:
                continue
            source_width = _bit_width(source_type)
            result_width = _bit_width(result_type)
            rows.append(
                ScalarConversion(
                    source_op_key="extf" if source_width < result_width else "fptrunc",
                    descriptor_suffix="f_convert",
                    mnemonic="OpFConvert",
                    opcode="LOOM_SPIRV_OP_F_CONVERT",
                    source_type=source_type,
                    result_type=result_type,
                )
            )
    return tuple(rows)


def _signed_integer_to_float_conversions() -> tuple[ScalarConversion, ...]:
    return tuple(
        ScalarConversion(
            source_op_key="sitofp",
            descriptor_suffix="convert_s_to_f",
            mnemonic="OpConvertSToF",
            opcode="LOOM_SPIRV_OP_CONVERT_S_TO_F",
            source_type=source_type,
            result_type=result_type,
        )
        for source_type in INTEGER_SCALAR_ALU_TYPES
        for result_type in FLOAT_SCALAR_ALU_TYPES
    )


def _float_to_signed_integer_conversions() -> tuple[ScalarConversion, ...]:
    return tuple(
        ScalarConversion(
            source_op_key="fptosi",
            descriptor_suffix="convert_f_to_s",
            mnemonic="OpConvertFToS",
            opcode="LOOM_SPIRV_OP_CONVERT_F_TO_S",
            source_type=source_type,
            result_type=result_type,
        )
        for source_type in FLOAT_SCALAR_ALU_TYPES
        for result_type in INTEGER_SCALAR_ALU_TYPES
    )


def _bitcast_conversions() -> tuple[ScalarConversion, ...]:
    pairs = (
        ("i16", "f16"),
        ("i32", "f32"),
        ("i64", "f64"),
    )
    scalar_types = {
        scalar.source_type: scalar
        for scalar in INTEGER_SCALAR_ALU_TYPES + FLOAT_SCALAR_ALU_TYPES
    }
    rows: list[ScalarConversion] = []
    for lhs_type, rhs_type in pairs:
        for source_type, result_type in (
            (scalar_types[lhs_type], scalar_types[rhs_type]),
            (scalar_types[rhs_type], scalar_types[lhs_type]),
        ):
            rows.append(
                ScalarConversion(
                    source_op_key="bitcast",
                    descriptor_suffix="bitcast",
                    mnemonic="OpBitcast",
                    opcode="LOOM_SPIRV_OP_BITCAST",
                    source_type=source_type,
                    result_type=result_type,
                )
            )
    return tuple(rows)


SCALAR_CONVERSIONS = (
    *_integer_width_conversions(),
    *_float_width_conversions(),
    *_signed_integer_to_float_conversions(),
    *_float_to_signed_integer_conversions(),
    *_bitcast_conversions(),
)
