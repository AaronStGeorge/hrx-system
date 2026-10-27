# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Packed integer absolute-value descriptor families."""

from loom.target.arch.x86.native_integer_unary import integer_unary_instruction
from loom.target.arch.x86.vector_families import (
    AVX2_INTEGER_UNARY_FAMILIES,
    AVX512_INTEGER_UNARY_FAMILIES,
)
from loom.target.low_descriptors import Descriptor, DescriptorFlag

from .common import (
    _asm,
    _avx512_descriptor,
    _vector_i32_schedule_class,
    _vector_operand,
    _vector_result,
    _vex_descriptor,
)

_REGISTER_SUFFIXES = {128: "xmm", 256: "ymm", 512: "zmm"}


def integer_unary_descriptors(*, evex: bool) -> tuple[Descriptor, ...]:
    profile = "avx512" if evex else "avx2"
    families = AVX512_INTEGER_UNARY_FAMILIES if evex else AVX2_INTEGER_UNARY_FAMILIES
    widths = (128, 256, 512) if evex else (128, 256)
    descriptors = []
    for family in families:
        instruction = integer_unary_instruction(
            family.mnemonic,
            element_bit_width=family.element.bit_width,
        )
        for vector_bit_width in widths:
            suffix = _REGISTER_SUFFIXES[vector_bit_width]
            descriptor = Descriptor(
                key=f"x86.{profile}.{family.mnemonic}.{suffix}",
                mnemonic=family.mnemonic,
                semantic_tag=f"{family.semantic}.{family.element.name}x{family.element.lane_count(vector_bit_width)}",
                operands=(
                    _vector_result(vector_bit_width),
                    _vector_operand(vector_bit_width, "input"),
                ),
                asm_forms=_asm(
                    mnemonic=f"{'avx512.' if evex else ''}{family.mnemonic}.{suffix}",
                    results=("dst",),
                    operands=("input",),
                ),
                schedule_class=_vector_i32_schedule_class(vector_bit_width),
                flags=(DescriptorFlag.DEAD_REMOVABLE,),
            )
            descriptors.append(
                _avx512_descriptor(descriptor, instruction, scalar=False)
                if evex
                else _vex_descriptor(descriptor, instruction)
            )
    return tuple(descriptors)
