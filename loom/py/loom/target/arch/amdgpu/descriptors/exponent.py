# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native floating decomposition and power-of-two scaling."""

from loom.target.arch.amdgpu.descriptor_overlay import (
    AmdgpuDescriptorOverlay,
    AmdgpuOperandOverlay,
)
from loom.target.arch.amdgpu.descriptors.common import (
    _REG_PART_VGPR_LOW16,
    _SCHEDULE_VALU,
    _sgpr_vgpr_operand,
    _vgpr_operand,
    _vgpr_result,
)
from loom.target.low_descriptors import DescriptorFlag, Operand


def _input(name: str, width: int) -> Operand:
    if width == 16:
        return _vgpr_operand(name, register_part=_REG_PART_VGPR_LOW16)
    return _sgpr_vgpr_operand(name, units=width // 32)


def _result(width: int) -> Operand:
    if width == 16:
        return _vgpr_result(register_part=_REG_PART_VGPR_LOW16)
    return _vgpr_result(units=width // 32)


def floating_exponent_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    overlays = []
    for width in (16, 32, 64):
        exponent_width = min(width, 32)
        for operation, result_width in (
            ("ldexp", width),
            ("frexp_mant", width),
            (f"frexp_exp_i{exponent_width}", exponent_width),
        ):
            mnemonic = f"v_{operation}_f{width}"
            operands = (
                AmdgpuOperandOverlay("VDST", _result(result_width)),
                AmdgpuOperandOverlay("SRC0", _input("input", width)),
            )
            if operation == "ldexp":
                operands += (
                    AmdgpuOperandOverlay("SRC1", _input("exponent", exponent_width)),
                )
            overlays.append(
                AmdgpuDescriptorOverlay(
                    descriptor_key=f"amdgpu.{mnemonic}",
                    instruction_name=mnemonic.upper(),
                    mnemonic=mnemonic,
                    encoding_name="ENC_VOP3" if operation == "ldexp" else "ENC_VOP1",
                    semantic_tag=f"float.{operation}.f{width}",
                    schedule_class=_SCHEDULE_VALU,
                    operands=operands,
                    flags=(DescriptorFlag.DEAD_REMOVABLE,),
                )
            )
    return tuple(overlays)
