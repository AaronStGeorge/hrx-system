# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native packed integer absolute-value instruction encodings."""

from loom.target.arch.x86.vector_encoding import (
    VectorEncoding,
    VectorEncodingPrefix,
    VectorMachineInstruction,
    VectorOpcodeMap,
    VectorOperandShape,
    VectorRegisterSelector,
    vector_encoding_recipe,
)
from loom.target.low_descriptors import OperandRole


def integer_unary_instruction(
    mnemonic: str, *, element_bit_width: int
) -> VectorMachineInstruction:
    ordinal = {8: 0, 16: 1, 32: 2, 64: 3}[element_bit_width]
    encodings = []
    if element_bit_width < 64:
        encodings.append(
            VectorEncoding(
                prefix=VectorEncodingPrefix.VEX,
                opcode_map=VectorOpcodeMap.MAP_0F38,
                mandatory_prefix=1,
                w=0,
                opcode=0x1C + ordinal,
                vector_bit_widths=(128, 256),
            )
        )
    encodings.append(
        VectorEncoding(
            prefix=VectorEncodingPrefix.EVEX,
            opcode_map=VectorOpcodeMap.MAP_0F38,
            mandatory_prefix=1,
            w=int(element_bit_width == 64),
            opcode=0x1C + ordinal,
            vector_bit_widths=(128, 256, 512),
        )
    )
    return VectorMachineInstruction(
        descriptor_mnemonic=mnemonic,
        encoding_mnemonic=mnemonic,
        encoding_format_id=vector_encoding_recipe(
            VectorRegisterSelector.RESULT,
            VectorRegisterSelector.NONE,
            VectorRegisterSelector.INPUT_0,
        ),
        operands=(
            VectorOperandShape(OperandRole.RESULT, ("x86.simd",)),
            VectorOperandShape(OperandRole.OPERAND, ("x86.simd",)),
        ),
        immediates=(),
        encodings=tuple(encodings),
    )
