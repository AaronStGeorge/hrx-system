# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Immutable x86 scalar-family rows shared by descriptor and contract generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScalarReadMemoryFamily:
    """An integer operation with a destructive GPR/read-memory encoding."""

    source_operation: str
    descriptor_operation: str
    mnemonic: str
    semantic: str
    opcode: int
    commutative: bool = True


@dataclass(frozen=True, slots=True)
class ScalarVariableShiftFamily:
    """A source shift and its non-destructive BMI2 machine form."""

    source_operation: str
    descriptor_operation: str
    semantic: str
    mandatory_prefix: int


@dataclass(frozen=True, slots=True)
class ScalarRotationFamily:
    """A source rotation and its legacy destructive machine form."""

    source_operation: str
    mnemonic: str
    semantic: str
    opcode_extension: int


@dataclass(frozen=True, slots=True)
class ScalarMemoryExtensionFamily:
    """A narrow scalar load consumed by one integer extension."""

    source_bit_count: int
    result_bit_counts: tuple[int, ...]
    signed: bool

    @property
    def carrier_bit_count(self) -> int:
        return 64 if self.result_bit_counts == (64,) else 32

    @property
    def mnemonic(self) -> str:
        if self.source_bit_count == 32:
            return "movsxd" if self.signed else "mov"
        return "movsx" if self.signed else "movzx"

    @property
    def opcode(self) -> int:
        if self.source_bit_count == 8:
            return 0x0FBE if self.signed else 0x0FB6
        if self.source_bit_count == 16:
            return 0x0FBF if self.signed else 0x0FB7
        return 0x63 if self.signed else 0x8B


X86_SCALAR_READ_MEMORY_FAMILIES = (
    ScalarReadMemoryFamily("addi", "add", "add", "add", 0x03),
    ScalarReadMemoryFamily("subi", "sub", "sub", "sub", 0x2B, False),
    ScalarReadMemoryFamily("muli", "imul", "imul", "mul", 0x0FAF),
    ScalarReadMemoryFamily("andi", "and", "and", "and", 0x23),
    ScalarReadMemoryFamily("ori", "or", "or", "or", 0x0B),
    ScalarReadMemoryFamily("xori", "xor", "xor", "xor", 0x33),
)

X86_SCALAR_MEMORY_EXTENSION_FAMILIES = tuple(
    ScalarMemoryExtensionFamily(source_bit_count, result_bit_counts, signed)
    for signed in (False, True)
    for source_bit_count, result_bit_counts in (
        (8, (16, 32)),
        (8, (64,)),
        (16, (32,)),
        (16, (64,)),
        (32, (64,)),
    )
)

X86_SCALAR_VARIABLE_SHIFT_FAMILIES = (
    ScalarVariableShiftFamily("shli", "shlx", "shl", 1),
    ScalarVariableShiftFamily("shrsi", "sarx", "shrs", 2),
    ScalarVariableShiftFamily("shrui", "shrx", "shru", 3),
)

X86_SCALAR_ROTATION_FAMILIES = (
    ScalarRotationFamily("rotli", "rol", "rotl", 0),
    ScalarRotationFamily("rotri", "ror", "rotr", 1),
)


def validate_scalar_families() -> None:
    if {
        (family.source_operation, family.semantic, family.opcode_extension)
        for family in X86_SCALAR_ROTATION_FAMILIES
    } != {("rotli", "rotl", 0), ("rotri", "rotr", 1)}:
        raise ValueError("x86 scalar rotation rows must cover left and right")


validate_scalar_families()
