# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""Scalar, vector, conversion, compare, and select descriptor overlays."""

from __future__ import annotations

from .common import *

_DPP_CTRL_IMMEDIATE = Immediate(
    "dpp_ctrl",
    ImmediateKind.UNSIGNED,
    bit_width=9,
    unsigned_max=0x1FF,
)


def _s_add_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_add_u32",
        instruction_name="S_ADD_U32",
        mnemonic="s_add_u32",
        encoding_name="ENC_SOP2",
        semantic_tag="integer.add.u32",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs")),
        ),
        implicit_operands=(_SCC_CLOBBER_OUTPUT,),
        operand_forms=(
            _literal_operand_form(
                replacement_descriptor="amdgpu.s_add_u32.rhs_inline",
                source_operand="rhs",
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_add_u32_rhs_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_rhs_inline_overlay(
        descriptor_key="amdgpu.s_add_u32.rhs_inline",
        instruction_name="S_ADD_U32",
        mnemonic="s_add_u32",
        semantic_tag="integer.add.u32",
    )


def _s_addc_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_addc_u32",
        instruction_name="S_ADDC_U32",
        mnemonic="s_addc_u32",
        encoding_name="ENC_SOP2",
        semantic_tag="integer.add.carry_in_out.u32",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result("sum")),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs")),
        ),
        implicit_operands=(
            _scc_output(_scc_clobber("carry")),
            _scc_input(_scc_state_read("carry_in")),
        ),
        asm_forms=_asm(results=("sum",), operands=("lhs", "rhs")),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_sub_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_sub_u32",
        instruction_name="S_SUB_U32",
        mnemonic="s_sub_u32",
        encoding_name="ENC_SOP2",
        semantic_tag="integer.sub.u32",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs")),
        ),
        implicit_operands=(_SCC_CLOBBER_OUTPUT,),
        operand_forms=(
            _literal_operand_form(
                replacement_descriptor="amdgpu.s_sub_u32.rhs_inline",
                source_operand="rhs",
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_sub_u32_rhs_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_rhs_inline_overlay(
        descriptor_key="amdgpu.s_sub_u32.rhs_inline",
        instruction_name="S_SUB_U32",
        mnemonic="s_sub_u32",
        semantic_tag="integer.sub.u32",
    )


def _s_mul_i32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_mul_i32",
        instruction_name="S_MUL_I32",
        mnemonic="s_mul_i32",
        encoding_name="ENC_SOP2",
        semantic_tag="integer.mul.lo.i32",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_mul_hi_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_mul_hi_u32",
        instruction_name="S_MUL_HI_U32",
        mnemonic="s_mul_hi_u32",
        encoding_name="ENC_SOP2",
        semantic_tag="integer.mul.hi.u32",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_binary_u32_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    rhs_inline_descriptor_key: str | None = None,
) -> AmdgpuDescriptorOverlay:
    operand_forms: tuple[OperandForm, ...] = ()
    if rhs_inline_descriptor_key is not None:
        operand_forms = (
            _literal_operand_form(
                replacement_descriptor=rhs_inline_descriptor_key,
                source_operand="rhs",
            ),
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SOP2",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs")),
        ),
        implicit_operands=(_SCC_CLOBBER_OUTPUT,),
        operand_forms=operand_forms,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_binary_u32_rhs_inline_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SOP2",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
        ),
        implicit_operands=(_SCC_CLOBBER_OUTPUT,),
        asm_forms=_asm(
            mnemonic=f"{mnemonic}_rhs_inline",
            results=("dst",),
            operands=("lhs",),
            immediates=("imm32",),
        ),
        immediate_fields=("SSRC1",),
        immediates=(_SOURCE_INLINE_U32_IMMEDIATE,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_binary_u64_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SOP2",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result(units=2)),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs", units=2)),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs", units=2)),
        ),
        implicit_operands=(_SCC_CLOBBER_OUTPUT,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_min_i32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_min_i32",
        instruction_name="S_MIN_I32",
        mnemonic="s_min_i32",
        semantic_tag="integer.min.i32",
    )


def _s_max_i32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_max_i32",
        instruction_name="S_MAX_I32",
        mnemonic="s_max_i32",
        semantic_tag="integer.max.i32",
    )


def _s_min_u32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_min_u32",
        instruction_name="S_MIN_U32",
        mnemonic="s_min_u32",
        semantic_tag="integer.min.u32",
    )


def _s_max_u32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_max_u32",
        instruction_name="S_MAX_U32",
        mnemonic="s_max_u32",
        semantic_tag="integer.max.u32",
    )


def _s_cselect_b32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_cselect_b32",
        instruction_name="S_CSELECT_B32",
        mnemonic="s_cselect_b32",
        encoding_name="ENC_SOP2",
        semantic_tag="control.select.b32",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("true_value")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("false_value")),
        ),
        implicit_operands=(_scc_input(_scc_predicate("condition")),),
        asm_forms=_asm(
            results=("dst",),
            operands=("true_value", "false_value", "condition"),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_shift_u64_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SOP2",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result(units=2)),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("value", units=2)),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("shift")),
        ),
        implicit_operands=(_SCC_CLOBBER_OUTPUT,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_cmp_i32_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SOPC",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs")),
        ),
        implicit_operands=(_scc_output(_scc_result()),),
        asm_forms=_asm(results=("scc",), operands=("lhs", "rhs")),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_cmp_u64_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SOPC",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs", units=2)),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs", units=2)),
        ),
        implicit_operands=(_scc_output(_scc_result()),),
        asm_forms=_asm(results=("scc",), operands=("lhs", "rhs")),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_add_u32_overlay(instruction_name: str) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_add_u32",
        instruction_name=instruction_name,
        mnemonic="v_add_u32",
        encoding_name="ENC_VOP2",
        semantic_tag="integer.add.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("lhs")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("rhs")),
        ),
        operand_forms=(
            _literal_operand_form(
                replacement_descriptor="amdgpu.v_add_u32.src0_inline",
                source_operand="lhs",
            ),
            _literal_operand_form(
                replacement_descriptor="amdgpu.v_add_u32.lit",
                source_operand="lhs",
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_add_u32_src0_inline_overlay(instruction_name: str) -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_overlay(
        descriptor_key="amdgpu.v_add_u32.src0_inline",
        instruction_name=instruction_name,
        mnemonic="v_add_u32",
        semantic_tag="integer.add.u32",
    )


def _v_binary_literal_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    rhs_name: str = "rhs",
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="VOP2_INST_LITERAL",
        encoding_condition="has_lit",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand(rhs_name)),
        ),
        asm_forms=_asm(
            mnemonic=f"{mnemonic}_lit",
            results=("dst",),
            operands=(rhs_name,),
            immediates=("imm32",),
        ),
        immediate_fields=("LITERAL",),
        immediates=(_U32_IMMEDIATE,),
        fixed_encoding_fields=(("SRC0", _predefined("SRC_LITERAL", "OPR_SRC")),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_binary_src0_inline_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    rhs_name: str = "rhs",
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_VOP2",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand(rhs_name)),
        ),
        asm_forms=_asm(
            mnemonic=f"{mnemonic}_src0_inline",
            results=("dst",),
            operands=(rhs_name,),
            immediates=("imm32",),
        ),
        immediate_fields=("SRC0",),
        immediates=(_SOURCE_INLINE_U32_IMMEDIATE,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_binary_src0_inline_f32_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    rhs_name: str = "rhs",
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_VOP2",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand(rhs_name)),
        ),
        asm_forms=_asm(
            mnemonic=f"{mnemonic}_src0_inline",
            results=("dst",),
            operands=(rhs_name,),
            immediates=("imm32",),
        ),
        immediate_fields=("SRC0",),
        immediates=(_SOURCE_INLINE_F32_IMMEDIATE,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_binary_f32_operand_forms(
    descriptor: AmdgpuDescriptorOverlay,
) -> tuple[OperandForm, ...]:
    descriptor_key = descriptor.descriptor_key
    return (
        _literal_operand_form(
            replacement_descriptor=f"{descriptor_key}.src0_inline",
            source_operand="lhs",
        ),
        _literal_operand_form(
            replacement_descriptor=f"{descriptor_key}.lit",
            source_operand="lhs",
        ),
    )


def _v_add_u32_literal_overlay(instruction_name: str) -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_add_u32.lit",
        instruction_name=instruction_name,
        mnemonic="v_add_u32",
        semantic_tag="integer.add.u32",
    )


def _v_add_co_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_add_co_u32",
        instruction_name="V_ADD_CO_U32",
        mnemonic="v_add_co_u32",
        encoding_name="VOP3_SDST_ENC",
        semantic_tag="integer.add.carry_out.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result("sum")),
            AmdgpuOperandOverlay("SDST", _sgpr_result("carry", units=2)),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_add_co_ci_u32_overlay(
    *, instruction_name: str = "V_ADD_CO_CI_U32", mnemonic: str = "v_add_co_ci_u32"
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_add_co_ci_u32",
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="VOP3_SDST_ENC",
        semantic_tag="integer.add.carry_in_out.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result("sum")),
            AmdgpuOperandOverlay("SDST", _sgpr_result("carry", units=2)),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _sgpr_operand("carry_in", units=2)),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_sub_u32_overlay(instruction_name: str, mnemonic: str) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_sub_u32",
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_VOP2",
        semantic_tag="integer.sub.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("lhs")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_binary_u32_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    lhs_name: str = "lhs",
    rhs_name: str = "rhs",
    src0_inline_descriptor_key: str | None = None,
    literal_descriptor_key: str | None = None,
) -> AmdgpuDescriptorOverlay:
    operand_forms = []
    if src0_inline_descriptor_key is not None:
        operand_forms.append(
            _literal_operand_form(
                replacement_descriptor=src0_inline_descriptor_key,
                source_operand=lhs_name,
            )
        )
    if literal_descriptor_key is not None:
        operand_forms.append(
            _literal_operand_form(
                replacement_descriptor=literal_descriptor_key,
                source_operand=lhs_name,
            )
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_VOP2",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand(lhs_name)),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand(rhs_name)),
        ),
        operand_forms=tuple(operand_forms),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_mul_lo_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_mul_lo_u32",
        instruction_name="V_MUL_LO_U32",
        mnemonic="v_mul_lo_u32",
        encoding_name="ENC_VOP3",
        semantic_tag="integer.mul.lo.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_mul_hi_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_mul_hi_u32",
        instruction_name="V_MUL_HI_U32",
        mnemonic="v_mul_hi_u32",
        encoding_name="ENC_VOP3",
        semantic_tag="integer.mul.hi.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_mul_u32_u24_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key="amdgpu.v_mul_u32_u24",
        instruction_name="V_MUL_U32_U24",
        mnemonic="v_mul_u32_u24",
        semantic_tag="integer.mul.lo.u24.u32",
        src0_inline_descriptor_key="amdgpu.v_mul_u32_u24.src0_inline",
        literal_descriptor_key="amdgpu.v_mul_u32_u24.lit",
    )


def _v_mul_u32_u24_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_overlay(
        descriptor_key="amdgpu.v_mul_u32_u24.src0_inline",
        instruction_name="V_MUL_U32_U24",
        mnemonic="v_mul_u32_u24",
        semantic_tag="integer.mul.lo.u24.u32",
    )


def _v_mul_u32_u24_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_mul_u32_u24.lit",
        instruction_name="V_MUL_U32_U24",
        mnemonic="v_mul_u32_u24",
        semantic_tag="integer.mul.lo.u24.u32",
    )


def _v_mad_u32_u24_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_mad_u32_u24",
        instruction_name="V_MAD_U32_U24",
        mnemonic="v_mad_u32_u24",
        encoding_name="ENC_VOP3",
        semantic_tag="integer.mad.lo.u24.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay(
                "SRC0",
                _sgpr_vgpr_operand("a"),
                size_exception_reason=_U24_SOURCE_SIZE_REASON,
            ),
            AmdgpuOperandOverlay(
                "SRC1",
                _sgpr_vgpr_operand("b"),
                size_exception_reason=_U24_SOURCE_SIZE_REASON,
            ),
            AmdgpuOperandOverlay("SRC2", _sgpr_vgpr_operand("addend")),
        ),
        operand_forms=(
            _literal_operand_form(
                replacement_descriptor="amdgpu.v_mad_u32_u24.src0_lit",
                source_operand="a",
            ),
            _literal_operand_form(
                replacement_descriptor="amdgpu.v_mad_u32_u24.src1_lit",
                source_operand="b",
            ),
            _literal_operand_form(
                replacement_descriptor="amdgpu.v_mad_u32_u24.src2_lit",
                source_operand="addend",
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_mad_u32_u24_literal_overlay(literal_source: str) -> AmdgpuDescriptorOverlay:
    source_fields = {
        "src0": ("SRC0", "a", _sgpr_vgpr_operand("a"), _U24_SOURCE_SIZE_REASON),
        "src1": ("SRC1", "b", _sgpr_vgpr_operand("b"), _U24_SOURCE_SIZE_REASON),
        "src2": ("SRC2", "addend", _sgpr_vgpr_operand("addend"), None),
    }
    literal_field = source_fields[literal_source][0]
    operands = [AmdgpuOperandOverlay("VDST", _vgpr_result())]
    asm_operands = []
    for source_name, (
        xml_field,
        field_name,
        operand,
        size_reason,
    ) in source_fields.items():
        if source_name == literal_source:
            continue
        asm_operands.append(field_name)
        operands.append(
            AmdgpuOperandOverlay(
                xml_field,
                operand,
                size_exception_reason=size_reason,
            )
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_mad_u32_u24.{literal_source}_lit",
        instruction_name="V_MAD_U32_U24",
        mnemonic=f"v_mad_u32_u24_{literal_source}_lit",
        encoding_name="ENC_VOP3",
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP3_LITERAL,
        semantic_tag="integer.mad.lo.u24.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=tuple(operands),
        asm_forms=_asm(
            results=("dst",),
            operands=tuple(asm_operands),
            immediates=("imm32",),
        ),
        immediates=(_LITERAL_U32_IMMEDIATE,),
        fixed_encoding_fields=((literal_field, _predefined("SRC_LITERAL", "OPR_SRC")),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_minmax_i32_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
) -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
    )


def _v_min_i32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_minmax_i32_overlay(
        descriptor_key="amdgpu.v_min_i32",
        instruction_name="V_MIN_I32",
        mnemonic="v_min_i32",
        semantic_tag="integer.min.i32",
    )


def _v_max_i32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_minmax_i32_overlay(
        descriptor_key="amdgpu.v_max_i32",
        instruction_name="V_MAX_I32",
        mnemonic="v_max_i32",
        semantic_tag="integer.max.i32",
    )


def _v_min_u32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_minmax_i32_overlay(
        descriptor_key="amdgpu.v_min_u32",
        instruction_name="V_MIN_U32",
        mnemonic="v_min_u32",
        semantic_tag="integer.min.u32",
    )


def _v_max_u32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_minmax_i32_overlay(
        descriptor_key="amdgpu.v_max_u32",
        instruction_name="V_MAX_U32",
        mnemonic="v_max_u32",
        semantic_tag="integer.max.u32",
    )


def _v_readfirstlane_b32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_readfirstlane_b32",
        instruction_name="V_READFIRSTLANE_B32",
        mnemonic="v_readfirstlane_b32",
        encoding_name="ENC_VOP1",
        semantic_tag="lane.readfirst.b32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _sgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("value")),
        ),
        effects=(_CONVERGENT_EFFECT,),
    )


def _s_and_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_and_b32",
        instruction_name="S_AND_B32",
        mnemonic="s_and_b32",
        semantic_tag="integer.and.u32",
    )


def _s_or_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_or_b32",
        instruction_name="S_OR_B32",
        mnemonic="s_or_b32",
        semantic_tag="integer.or.u32",
    )


def _s_xor_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_xor_b32",
        instruction_name="S_XOR_B32",
        mnemonic="s_xor_b32",
        semantic_tag="integer.xor.u32",
    )


def _s_and_b64_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u64_overlay(
        descriptor_key="amdgpu.s_and_b64",
        instruction_name="S_AND_B64",
        mnemonic="s_and_b64",
        semantic_tag="integer.and.u64",
    )


def _s_or_b64_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u64_overlay(
        descriptor_key="amdgpu.s_or_b64",
        instruction_name="S_OR_B64",
        mnemonic="s_or_b64",
        semantic_tag="integer.or.u64",
    )


def _s_xor_b64_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u64_overlay(
        descriptor_key="amdgpu.s_xor_b64",
        instruction_name="S_XOR_B64",
        mnemonic="s_xor_b64",
        semantic_tag="integer.xor.u64",
    )


def _s_cmp_i32_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_cmp_i32_overlay(
            descriptor_key="amdgpu.s_cmp_eq_i32",
            instruction_name="S_CMP_EQ_I32",
            mnemonic="s_cmp_eq_i32",
            semantic_tag="integer.compare.eq.i32",
        ),
        _s_cmp_i32_overlay(
            descriptor_key="amdgpu.s_cmp_lg_i32",
            instruction_name="S_CMP_LG_I32",
            mnemonic="s_cmp_lg_i32",
            semantic_tag="integer.compare.ne.i32",
        ),
        _s_cmp_i32_overlay(
            descriptor_key="amdgpu.s_cmp_lt_i32",
            instruction_name="S_CMP_LT_I32",
            mnemonic="s_cmp_lt_i32",
            semantic_tag="integer.compare.slt.i32",
        ),
        _s_cmp_i32_overlay(
            descriptor_key="amdgpu.s_cmp_le_i32",
            instruction_name="S_CMP_LE_I32",
            mnemonic="s_cmp_le_i32",
            semantic_tag="integer.compare.sle.i32",
        ),
        _s_cmp_i32_overlay(
            descriptor_key="amdgpu.s_cmp_gt_i32",
            instruction_name="S_CMP_GT_I32",
            mnemonic="s_cmp_gt_i32",
            semantic_tag="integer.compare.sgt.i32",
        ),
        _s_cmp_i32_overlay(
            descriptor_key="amdgpu.s_cmp_ge_i32",
            instruction_name="S_CMP_GE_I32",
            mnemonic="s_cmp_ge_i32",
            semantic_tag="integer.compare.sge.i32",
        ),
        _s_cmp_i32_overlay(
            descriptor_key="amdgpu.s_cmp_lt_u32",
            instruction_name="S_CMP_LT_U32",
            mnemonic="s_cmp_lt_u32",
            semantic_tag="integer.compare.ult.i32",
        ),
        _s_cmp_i32_overlay(
            descriptor_key="amdgpu.s_cmp_le_u32",
            instruction_name="S_CMP_LE_U32",
            mnemonic="s_cmp_le_u32",
            semantic_tag="integer.compare.ule.i32",
        ),
        _s_cmp_i32_overlay(
            descriptor_key="amdgpu.s_cmp_gt_u32",
            instruction_name="S_CMP_GT_U32",
            mnemonic="s_cmp_gt_u32",
            semantic_tag="integer.compare.ugt.i32",
        ),
        _s_cmp_i32_overlay(
            descriptor_key="amdgpu.s_cmp_ge_u32",
            instruction_name="S_CMP_GE_U32",
            mnemonic="s_cmp_ge_u32",
            semantic_tag="integer.compare.uge.i32",
        ),
    )


def _s_cmp_u64_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_cmp_u64_overlay(
            descriptor_key="amdgpu.s_cmp_eq_u64",
            instruction_name="S_CMP_EQ_U64",
            mnemonic="s_cmp_eq_u64",
            semantic_tag="integer.compare.eq.u64",
        ),
        _s_cmp_u64_overlay(
            descriptor_key="amdgpu.s_cmp_lg_u64",
            instruction_name="S_CMP_LG_U64",
            mnemonic="s_cmp_lg_u64",
            semantic_tag="integer.compare.ne.u64",
        ),
    )


def _s_and_saveexec_b64_overlay(
    encoding_condition: str,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_and_saveexec_b64",
        instruction_name="S_AND_SAVEEXEC_B64",
        mnemonic="s_and_saveexec_b64",
        encoding_name="ENC_SOP1",
        encoding_condition=encoding_condition,
        semantic_tag="control.exec.and_save",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result("saved_exec", units=2)),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("mask", units=2)),
        ),
        implicit_operands=(
            AmdgpuImplicitOperandOverlay(
                "OPR_SDST_EXEC",
                descriptor_operand=_exec_clobber("exec_out"),
                data_format_name="FMT_NUM_M64",
                size_bits=64,
                is_input=False,
                is_output=True,
            ),
            AmdgpuImplicitOperandOverlay(
                "OPR_SDST_EXEC",
                descriptor_operand=_exec_state_read(),
                data_format_name="FMT_NUM_M64",
                size_bits=64,
                is_input=True,
                is_output=False,
            ),
            _scc_output(_scc_result("active")),
        ),
        asm_forms=_asm(results=("saved_exec", "active"), operands=("mask",)),
        effects=(_CONVERGENT_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_lshl_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_lshl_b32",
        instruction_name="S_LSHL_B32",
        mnemonic="s_lshl_b32",
        semantic_tag="integer.shl.u32",
        rhs_inline_descriptor_key="amdgpu.s_lshl_b32.rhs_inline",
    )


def _s_lshl_b32_rhs_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_rhs_inline_overlay(
        descriptor_key="amdgpu.s_lshl_b32.rhs_inline",
        instruction_name="S_LSHL_B32",
        mnemonic="s_lshl_b32",
        semantic_tag="integer.shl.u32",
    )


def _s_lshl_b64_overlay() -> AmdgpuDescriptorOverlay:
    return _s_shift_u64_overlay(
        descriptor_key="amdgpu.s_lshl_b64",
        instruction_name="S_LSHL_B64",
        mnemonic="s_lshl_b64",
        semantic_tag="integer.shl.u64",
    )


def _s_lshr_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_lshr_b32",
        instruction_name="S_LSHR_B32",
        mnemonic="s_lshr_b32",
        semantic_tag="integer.shr.u32",
        rhs_inline_descriptor_key="amdgpu.s_lshr_b32.rhs_inline",
    )


def _s_lshr_b32_rhs_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_rhs_inline_overlay(
        descriptor_key="amdgpu.s_lshr_b32.rhs_inline",
        instruction_name="S_LSHR_B32",
        mnemonic="s_lshr_b32",
        semantic_tag="integer.shr.u32",
    )


def _s_lshr_b64_overlay() -> AmdgpuDescriptorOverlay:
    return _s_shift_u64_overlay(
        descriptor_key="amdgpu.s_lshr_b64",
        instruction_name="S_LSHR_B64",
        mnemonic="s_lshr_b64",
        semantic_tag="integer.shr.u64",
    )


def _s_ashr_i32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_ashr_i32",
        instruction_name="S_ASHR_I32",
        mnemonic="s_ashr_i32",
        semantic_tag="integer.shr.i32",
        rhs_inline_descriptor_key="amdgpu.s_ashr_i32.rhs_inline",
    )


def _s_ashr_i32_rhs_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_rhs_inline_overlay(
        descriptor_key="amdgpu.s_ashr_i32.rhs_inline",
        instruction_name="S_ASHR_I32",
        mnemonic="s_ashr_i32",
        semantic_tag="integer.shr.i32",
    )


def _v_and_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key="amdgpu.v_and_b32",
        instruction_name="V_AND_B32",
        mnemonic="v_and_b32",
        semantic_tag="integer.and.u32",
        src0_inline_descriptor_key="amdgpu.v_and_b32.src0_inline",
        literal_descriptor_key="amdgpu.v_and_b32.lit",
    )


def _v_and_b32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_overlay(
        descriptor_key="amdgpu.v_and_b32.src0_inline",
        instruction_name="V_AND_B32",
        mnemonic="v_and_b32",
        semantic_tag="integer.and.u32",
        rhs_name="rhs",
    )


def _v_and_b32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_and_b32.lit",
        instruction_name="V_AND_B32",
        mnemonic="v_and_b32",
        semantic_tag="integer.and.u32",
        rhs_name="rhs",
    )


def _v_or_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key="amdgpu.v_or_b32",
        instruction_name="V_OR_B32",
        mnemonic="v_or_b32",
        semantic_tag="integer.or.u32",
        src0_inline_descriptor_key="amdgpu.v_or_b32.src0_inline",
        literal_descriptor_key="amdgpu.v_or_b32.lit",
    )


def _v_or_b32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_overlay(
        descriptor_key="amdgpu.v_or_b32.src0_inline",
        instruction_name="V_OR_B32",
        mnemonic="v_or_b32",
        semantic_tag="integer.or.u32",
        rhs_name="rhs",
    )


def _v_or_b32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_or_b32.lit",
        instruction_name="V_OR_B32",
        mnemonic="v_or_b32",
        semantic_tag="integer.or.u32",
        rhs_name="rhs",
    )


def _v_xor_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key="amdgpu.v_xor_b32",
        instruction_name="V_XOR_B32",
        mnemonic="v_xor_b32",
        semantic_tag="integer.xor.u32",
        src0_inline_descriptor_key="amdgpu.v_xor_b32.src0_inline",
        literal_descriptor_key="amdgpu.v_xor_b32.lit",
    )


def _v_xor_b32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_overlay(
        descriptor_key="amdgpu.v_xor_b32.src0_inline",
        instruction_name="V_XOR_B32",
        mnemonic="v_xor_b32",
        semantic_tag="integer.xor.u32",
        rhs_name="rhs",
    )


def _v_xor_b32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_xor_b32.lit",
        instruction_name="V_XOR_B32",
        mnemonic="v_xor_b32",
        semantic_tag="integer.xor.u32",
        rhs_name="rhs",
    )


def _v_lshlrev_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key="amdgpu.v_lshlrev_b32",
        instruction_name="V_LSHLREV_B32",
        mnemonic="v_lshlrev_b32",
        semantic_tag="integer.shl.u32",
        lhs_name="shift",
        rhs_name="value",
        src0_inline_descriptor_key="amdgpu.v_lshlrev_b32.src0_inline",
        literal_descriptor_key="amdgpu.v_lshlrev_b32.lit",
    )


def _v_lshlrev_b32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_overlay(
        descriptor_key="amdgpu.v_lshlrev_b32.src0_inline",
        instruction_name="V_LSHLREV_B32",
        mnemonic="v_lshlrev_b32",
        semantic_tag="integer.shl.u32",
        rhs_name="value",
    )


def _v_lshlrev_b32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_lshlrev_b32.lit",
        instruction_name="V_LSHLREV_B32",
        mnemonic="v_lshlrev_b32",
        semantic_tag="integer.shl.u32",
        rhs_name="value",
    )


def _v_lshlrev_b32_vop3_immediate_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_lshlrev_b32.vop3_imm",
        instruction_name="V_LSHLREV_B32",
        mnemonic="v_lshlrev_b32",
        encoding_name="ENC_VOP3",
        semantic_tag="integer.shl.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("value")),
        ),
        asm_forms=_asm(
            mnemonic="v_lshlrev_b32_vop3_imm",
            results=("dst",),
            operands=("value",),
            immediates=("imm32",),
        ),
        immediate_fields=("SRC0",),
        immediates=(_SOURCE_INLINE_U32_IMMEDIATE,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_lshl_add_u32_shift_immediate_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_lshl_add_u32.shift_imm",
        instruction_name="V_LSHL_ADD_U32",
        mnemonic="v_lshl_add_u32",
        encoding_name="ENC_VOP3",
        semantic_tag="integer.shl.add.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("value")),
            AmdgpuOperandOverlay("SRC2", _sgpr_vgpr_operand("addend")),
        ),
        asm_forms=_asm(
            mnemonic="v_lshl_add_u32_shift_imm",
            results=("dst",),
            operands=("value", "addend"),
            immediates=("shift",),
        ),
        immediate_fields=("SRC1",),
        immediates=(_source_inline_u32_immediate("shift"),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_lshrrev_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key="amdgpu.v_lshrrev_b32",
        instruction_name="V_LSHRREV_B32",
        mnemonic="v_lshrrev_b32",
        semantic_tag="integer.shr.u32",
        lhs_name="shift",
        rhs_name="value",
        src0_inline_descriptor_key="amdgpu.v_lshrrev_b32.src0_inline",
        literal_descriptor_key="amdgpu.v_lshrrev_b32.lit",
    )


def _v_lshrrev_b32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_overlay(
        descriptor_key="amdgpu.v_lshrrev_b32.src0_inline",
        instruction_name="V_LSHRREV_B32",
        mnemonic="v_lshrrev_b32",
        semantic_tag="integer.shr.u32",
        rhs_name="value",
    )


def _v_lshrrev_b32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_lshrrev_b32.lit",
        instruction_name="V_LSHRREV_B32",
        mnemonic="v_lshrrev_b32",
        semantic_tag="integer.shr.u32",
        rhs_name="value",
    )


def _v_ashrrev_i32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key="amdgpu.v_ashrrev_i32",
        instruction_name="V_ASHRREV_I32",
        mnemonic="v_ashrrev_i32",
        semantic_tag="integer.shr.i32",
        lhs_name="shift",
        rhs_name="value",
        src0_inline_descriptor_key="amdgpu.v_ashrrev_i32.src0_inline",
        literal_descriptor_key="amdgpu.v_ashrrev_i32.lit",
    )


def _v_ashrrev_i32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_overlay(
        descriptor_key="amdgpu.v_ashrrev_i32.src0_inline",
        instruction_name="V_ASHRREV_I32",
        mnemonic="v_ashrrev_i32",
        semantic_tag="integer.shr.i32",
        rhs_name="value",
    )


def _v_ashrrev_i32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_ashrrev_i32.lit",
        instruction_name="V_ASHRREV_I32",
        mnemonic="v_ashrrev_i32",
        semantic_tag="integer.shr.i32",
        rhs_name="value",
    )


def _integer_bitwise_shift_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_and_b32_overlay(),
        _s_or_b32_overlay(),
        _s_xor_b32_overlay(),
        _s_and_b64_overlay(),
        _s_or_b64_overlay(),
        _s_xor_b64_overlay(),
        _s_lshl_b32_overlay(),
        _s_lshl_b32_rhs_inline_overlay(),
        _s_lshl_b64_overlay(),
        _s_lshr_b32_overlay(),
        _s_lshr_b32_rhs_inline_overlay(),
        _s_lshr_b64_overlay(),
        _s_ashr_i32_overlay(),
        _s_ashr_i32_rhs_inline_overlay(),
        _v_and_b32_overlay(),
        _v_and_b32_src0_inline_overlay(),
        _v_and_b32_literal_overlay(),
        _v_or_b32_overlay(),
        _v_or_b32_src0_inline_overlay(),
        _v_or_b32_literal_overlay(),
        _v_xor_b32_overlay(),
        _v_xor_b32_src0_inline_overlay(),
        _v_xor_b32_literal_overlay(),
        _v_lshlrev_b32_overlay(),
        _v_lshlrev_b32_src0_inline_overlay(),
        _v_lshlrev_b32_literal_overlay(),
        _v_lshlrev_b32_vop3_immediate_overlay(),
        _v_lshl_add_u32_shift_immediate_overlay(),
        _v_lshrrev_b32_overlay(),
        _v_lshrrev_b32_src0_inline_overlay(),
        _v_lshrrev_b32_literal_overlay(),
        _v_ashrrev_i32_overlay(),
        _v_ashrrev_i32_src0_inline_overlay(),
        _v_ashrrev_i32_literal_overlay(),
    )


def _v_add_f32_overlay() -> AmdgpuDescriptorOverlay:
    descriptor = AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_add_f32",
        instruction_name="V_ADD_F32",
        mnemonic="v_add_f32",
        encoding_name="ENC_VOP2",
        semantic_tag="float.add.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("lhs")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )
    return replace(descriptor, operand_forms=_v_binary_f32_operand_forms(descriptor))


def _v_add_f32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_add_f32.lit",
        instruction_name="V_ADD_F32",
        mnemonic="v_add_f32",
        semantic_tag="float.add.f32",
    )


def _v_add_f32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_f32_overlay(
        descriptor_key="amdgpu.v_add_f32.src0_inline",
        instruction_name="V_ADD_F32",
        mnemonic="v_add_f32",
        semantic_tag="float.add.f32",
    )


def _v_sub_f32_overlay() -> AmdgpuDescriptorOverlay:
    descriptor = AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_sub_f32",
        instruction_name="V_SUB_F32",
        mnemonic="v_sub_f32",
        encoding_name="ENC_VOP2",
        semantic_tag="float.sub.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("lhs")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )
    return replace(descriptor, operand_forms=_v_binary_f32_operand_forms(descriptor))


def _v_sub_f32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_sub_f32.lit",
        instruction_name="V_SUB_F32",
        mnemonic="v_sub_f32",
        semantic_tag="float.sub.f32",
    )


def _v_sub_f32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_f32_overlay(
        descriptor_key="amdgpu.v_sub_f32.src0_inline",
        instruction_name="V_SUB_F32",
        mnemonic="v_sub_f32",
        semantic_tag="float.sub.f32",
    )


def _v_mul_f32_overlay() -> AmdgpuDescriptorOverlay:
    descriptor = AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_mul_f32",
        instruction_name="V_MUL_F32",
        mnemonic="v_mul_f32",
        encoding_name="ENC_VOP2",
        semantic_tag="float.mul.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("lhs")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )
    return replace(descriptor, operand_forms=_v_binary_f32_operand_forms(descriptor))


def _v_mul_f32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_mul_f32.lit",
        instruction_name="V_MUL_F32",
        mnemonic="v_mul_f32",
        semantic_tag="float.mul.f32",
    )


def _v_mul_f32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_f32_overlay(
        descriptor_key="amdgpu.v_mul_f32.src0_inline",
        instruction_name="V_MUL_F32",
        mnemonic="v_mul_f32",
        semantic_tag="float.mul.f32",
    )


def _v_min_f32_overlay() -> AmdgpuDescriptorOverlay:
    descriptor = AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_min_f32",
        instruction_name="V_MIN_F32",
        mnemonic="v_min_f32",
        encoding_name="ENC_VOP2",
        semantic_tag="float.minnum.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("lhs")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )
    return replace(descriptor, operand_forms=_v_binary_f32_operand_forms(descriptor))


def _v_min_f32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_min_f32.lit",
        instruction_name="V_MIN_F32",
        mnemonic="v_min_f32",
        semantic_tag="float.minnum.f32",
    )


def _v_min_f32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_f32_overlay(
        descriptor_key="amdgpu.v_min_f32.src0_inline",
        instruction_name="V_MIN_F32",
        mnemonic="v_min_f32",
        semantic_tag="float.minnum.f32",
    )


def _v_max_f32_overlay() -> AmdgpuDescriptorOverlay:
    descriptor = AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_max_f32",
        instruction_name="V_MAX_F32",
        mnemonic="v_max_f32",
        encoding_name="ENC_VOP2",
        semantic_tag="float.maxnum.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("lhs")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )
    return replace(descriptor, operand_forms=_v_binary_f32_operand_forms(descriptor))


def _v_max_f32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_literal_overlay(
        descriptor_key="amdgpu.v_max_f32.lit",
        instruction_name="V_MAX_F32",
        mnemonic="v_max_f32",
        semantic_tag="float.maxnum.f32",
    )


def _v_max_f32_src0_inline_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_src0_inline_f32_overlay(
        descriptor_key="amdgpu.v_max_f32.src0_inline",
        instruction_name="V_MAX_F32",
        mnemonic="v_max_f32",
        semantic_tag="float.maxnum.f32",
    )


def _v_fma_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_fma_f32",
        instruction_name="V_FMA_F32",
        mnemonic="v_fma_f32",
        encoding_name="ENC_VOP3",
        semantic_tag="float.fma.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("a")),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("b")),
            AmdgpuOperandOverlay("SRC2", _sgpr_vgpr_operand("c")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_fmac_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_fmac_f32",
        instruction_name="V_FMAC_F32",
        mnemonic="v_fmac_f32",
        encoding_name="ENC_VOP2",
        semantic_tag="float.fmac.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay(
                "VDST",
                Operand(
                    "acc",
                    OperandRole.OPERAND,
                    _VGPR_ALT,
                    flags=(OperandFlag.IMPLICIT,),
                ),
                role_exception_reason=(
                    "the encoded destination register is also the tied "
                    "accumulator input"
                ),
            ),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("a")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("b")),
        ),
        constraints=(
            Constraint(ConstraintKind.TIED, 0, 1),
            Constraint(ConstraintKind.DESTRUCTIVE, 0, 1),
        ),
        asm_forms=_asm(results=("dst",), operands=("acc", "a", "b")),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_unary_f32_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_VOP1",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("input")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_exp_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_unary_f32_overlay(
        descriptor_key="amdgpu.v_exp_f32",
        instruction_name="V_EXP_F32",
        mnemonic="v_exp_f32",
        semantic_tag="float.exp2.f32",
    )


def _v_log_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_unary_f32_overlay(
        descriptor_key="amdgpu.v_log_f32",
        instruction_name="V_LOG_F32",
        mnemonic="v_log_f32",
        semantic_tag="float.log2.f32",
    )


def _v_sin_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_unary_f32_overlay(
        descriptor_key="amdgpu.v_sin_f32",
        instruction_name="V_SIN_F32",
        mnemonic="v_sin_f32",
        semantic_tag="float.sin_turns.f32",
    )


def _v_cos_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_unary_f32_overlay(
        descriptor_key="amdgpu.v_cos_f32",
        instruction_name="V_COS_F32",
        mnemonic="v_cos_f32",
        semantic_tag="float.cos_turns.f32",
    )


def _v_native_f32_math_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _v_exp_f32_overlay(),
        _v_log_f32_overlay(),
        _v_sin_f32_overlay(),
        _v_cos_f32_overlay(),
    )


def _v_sqrt_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_unary_f32_overlay(
        descriptor_key="amdgpu.v_sqrt_f32",
        instruction_name="V_SQRT_F32",
        mnemonic="v_sqrt_f32",
        semantic_tag="float.sqrt.f32",
    )


def _v_rsq_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_unary_f32_overlay(
        descriptor_key="amdgpu.v_rsq_f32",
        instruction_name="V_RSQ_F32",
        mnemonic="v_rsq_f32",
        semantic_tag="float.rsqrt.f32",
    )


def _v_rcp_f32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_unary_f32_overlay(
        descriptor_key="amdgpu.v_rcp_f32",
        instruction_name="V_RCP_F32",
        mnemonic="v_rcp_f32",
        semantic_tag="float.reciprocal.f32",
    )


def _v_cvt_f32_i32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_cvt_f32_i32",
        instruction_name="V_CVT_F32_I32",
        mnemonic="v_cvt_f32_i32",
        encoding_name="ENC_VOP1",
        semantic_tag="convert.signed.i32.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("input")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cvt_f32_f16_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_cvt_f32_f16",
        instruction_name="V_CVT_F32_F16",
        mnemonic="v_cvt_f32_f16",
        encoding_name="ENC_VOP1",
        semantic_tag="convert.float.f16.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay(
                "SRC0", _vgpr_operand("input", register_part=_REG_PART_VGPR_LOW16)
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cvt_f16_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_cvt_f16_f32",
        instruction_name="V_CVT_F16_F32",
        mnemonic="v_cvt_f16_f32",
        encoding_name="ENC_VOP1",
        semantic_tag="convert.float.f32.f16",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay(
                "VDST", _vgpr_result(register_part=_REG_PART_VGPR_LOW16)
            ),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("input")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cvt_pk_u16_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_cvt_pk_u16_u32",
        instruction_name="V_CVT_PK_U16_U32",
        mnemonic="v_cvt_pk_u16_u32",
        encoding_name="ENC_VOP3",
        semantic_tag="convert.pack.u32.u16x2",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("low")),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("high")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cvt_pk_bf16_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_cvt_pk_bf16_f32",
        instruction_name="V_CVT_PK_BF16_F32",
        mnemonic="v_cvt_pk_bf16_f32",
        encoding_name="ENC_VOP3",
        semantic_tag="convert.float.f32.bf16x2",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("low")),
            AmdgpuOperandOverlay("SRC1", _sgpr_vgpr_operand("high")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cvt_f32_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_cvt_f32_u32",
        instruction_name="V_CVT_F32_U32",
        mnemonic="v_cvt_f32_u32",
        encoding_name="ENC_VOP1",
        semantic_tag="convert.unsigned.u32.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("input")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cmp_i32_overlay(
    *, predicate: str, instruction_suffix: str, semantic_suffix: str
) -> AmdgpuDescriptorOverlay:
    instruction_predicate = instruction_suffix.lower()
    descriptor_key = f"amdgpu.v_cmp_{predicate}_i32"
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=f"V_CMP_{instruction_suffix}_I32",
        mnemonic=f"v_cmp_{instruction_predicate}_i32",
        encoding_name="ENC_VOP3",
        semantic_tag=f"cmp.i32.{semantic_suffix}",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _sgpr_result("mask", units=2)),
            AmdgpuOperandOverlay("SRC0", _vgpr_const_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_const_operand("rhs")),
        ),
        operand_forms=_v_cmp_inline_operand_forms(descriptor_key),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cmp_u32_overlay(
    *, predicate: str, instruction_suffix: str, semantic_suffix: str
) -> AmdgpuDescriptorOverlay:
    instruction_predicate = instruction_suffix.lower()
    descriptor_key = f"amdgpu.v_cmp_{predicate}_u32"
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=f"V_CMP_{instruction_suffix}_U32",
        mnemonic=f"v_cmp_{instruction_predicate}_u32",
        encoding_name="ENC_VOP3",
        semantic_tag=f"cmp.u32.{semantic_suffix}",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _sgpr_result("mask", units=2)),
            AmdgpuOperandOverlay("SRC0", _vgpr_const_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_const_operand("rhs")),
        ),
        operand_forms=_v_cmp_inline_operand_forms(descriptor_key),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cmp_inline_operand_forms(descriptor_key: str) -> tuple[OperandForm, ...]:
    return (
        _literal_operand_form(
            replacement_descriptor=f"{descriptor_key}.src0_inline",
            source_operand="lhs",
            immediate_field="lhs",
        ),
        _literal_operand_form(
            replacement_descriptor=f"{descriptor_key}.src1_inline",
            source_operand="rhs",
            immediate_field="rhs",
        ),
    )


def _v_cmp_32_source_overlay(
    *,
    predicate: str,
    instruction_suffix: str,
    semantic_suffix: str,
    type_suffix: str,
    literal_source: str,
) -> AmdgpuDescriptorOverlay:
    source_fields = {
        "src0": ("SRC0", "lhs", _vgpr_const_operand("lhs")),
        "src1": ("SRC1", "rhs", _vgpr_const_operand("rhs")),
    }
    literal_field, literal_operand, _ = source_fields[literal_source]
    remaining_operands = [
        (xml_field, field_name, operand)
        for source_name, (xml_field, field_name, operand) in source_fields.items()
        if source_name != literal_source
    ]
    descriptor_key = f"amdgpu.v_cmp_{predicate}_{type_suffix}.{literal_source}_inline"
    instruction_predicate = instruction_suffix.lower()
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=f"V_CMP_{instruction_suffix}_{type_suffix.upper()}",
        mnemonic=f"v_cmp_{instruction_predicate}_{type_suffix}",
        encoding_name="ENC_VOP3",
        semantic_tag=f"cmp.{type_suffix}.{semantic_suffix}",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _sgpr_result("mask", units=2)),
            *(
                AmdgpuOperandOverlay(xml_field, operand)
                for xml_field, _, operand in remaining_operands
            ),
        ),
        asm_forms=_asm(
            mnemonic=f"v_cmp_{instruction_predicate}_{type_suffix}_{literal_source}_inline",
            results=("mask",),
            operands=tuple(field_name for _, field_name, _ in remaining_operands),
            immediates=(literal_operand,),
            named_immediates=True,
        ),
        immediate_fields=(literal_field,),
        immediates=(_source_inline_u32_immediate(literal_operand),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cmp_i32_source_overlays(
    *, predicate: str, instruction_suffix: str, semantic_suffix: str
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _v_cmp_32_source_overlay(
            predicate=predicate,
            instruction_suffix=instruction_suffix,
            semantic_suffix=semantic_suffix,
            type_suffix="i32",
            literal_source=literal_source,
        )
        for literal_source in ("src0", "src1")
    )


def _v_cmp_u32_source_overlays(
    *, predicate: str, instruction_suffix: str, semantic_suffix: str
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _v_cmp_32_source_overlay(
            predicate=predicate,
            instruction_suffix=instruction_suffix,
            semantic_suffix=semantic_suffix,
            type_suffix="u32",
            literal_source=literal_source,
        )
        for literal_source in ("src0", "src1")
    )


def _v_cmp_f32_overlay(
    *, predicate: str, instruction_suffix: str, semantic_suffix: str
) -> AmdgpuDescriptorOverlay:
    instruction_predicate = instruction_suffix.lower()
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_cmp_{predicate}_f32",
        instruction_name=f"V_CMP_{instruction_suffix}_F32",
        mnemonic=f"v_cmp_{instruction_predicate}_f32",
        encoding_name="ENC_VOP3",
        semantic_tag=f"cmp.f32.{semantic_suffix}",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _sgpr_result("mask", units=2)),
            AmdgpuOperandOverlay("SRC0", _vgpr_const_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_const_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cmp_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _v_cmp_i32_overlay(
            predicate="eq", instruction_suffix="EQ", semantic_suffix="eq"
        ),
        *_v_cmp_i32_source_overlays(
            predicate="eq", instruction_suffix="EQ", semantic_suffix="eq"
        ),
        _v_cmp_i32_overlay(
            predicate="ne", instruction_suffix="NE", semantic_suffix="ne"
        ),
        *_v_cmp_i32_source_overlays(
            predicate="ne", instruction_suffix="NE", semantic_suffix="ne"
        ),
        _v_cmp_i32_overlay(
            predicate="slt", instruction_suffix="LT", semantic_suffix="slt"
        ),
        *_v_cmp_i32_source_overlays(
            predicate="slt", instruction_suffix="LT", semantic_suffix="slt"
        ),
        _v_cmp_i32_overlay(
            predicate="sle", instruction_suffix="LE", semantic_suffix="sle"
        ),
        *_v_cmp_i32_source_overlays(
            predicate="sle", instruction_suffix="LE", semantic_suffix="sle"
        ),
        _v_cmp_i32_overlay(
            predicate="sgt", instruction_suffix="GT", semantic_suffix="sgt"
        ),
        *_v_cmp_i32_source_overlays(
            predicate="sgt", instruction_suffix="GT", semantic_suffix="sgt"
        ),
        _v_cmp_i32_overlay(
            predicate="sge", instruction_suffix="GE", semantic_suffix="sge"
        ),
        *_v_cmp_i32_source_overlays(
            predicate="sge", instruction_suffix="GE", semantic_suffix="sge"
        ),
        _v_cmp_u32_overlay(
            predicate="ult", instruction_suffix="LT", semantic_suffix="ult"
        ),
        *_v_cmp_u32_source_overlays(
            predicate="ult", instruction_suffix="LT", semantic_suffix="ult"
        ),
        _v_cmp_u32_overlay(
            predicate="ule", instruction_suffix="LE", semantic_suffix="ule"
        ),
        *_v_cmp_u32_source_overlays(
            predicate="ule", instruction_suffix="LE", semantic_suffix="ule"
        ),
        _v_cmp_u32_overlay(
            predicate="ugt", instruction_suffix="GT", semantic_suffix="ugt"
        ),
        *_v_cmp_u32_source_overlays(
            predicate="ugt", instruction_suffix="GT", semantic_suffix="ugt"
        ),
        _v_cmp_u32_overlay(
            predicate="uge", instruction_suffix="GE", semantic_suffix="uge"
        ),
        *_v_cmp_u32_source_overlays(
            predicate="uge", instruction_suffix="GE", semantic_suffix="uge"
        ),
        _v_cmp_f32_overlay(
            predicate="oeq", instruction_suffix="EQ", semantic_suffix="oeq"
        ),
        _v_cmp_f32_overlay(
            predicate="ogt", instruction_suffix="GT", semantic_suffix="ogt"
        ),
        _v_cmp_f32_overlay(
            predicate="oge", instruction_suffix="GE", semantic_suffix="oge"
        ),
        _v_cmp_f32_overlay(
            predicate="olt", instruction_suffix="LT", semantic_suffix="olt"
        ),
        _v_cmp_f32_overlay(
            predicate="ole", instruction_suffix="LE", semantic_suffix="ole"
        ),
        _v_cmp_f32_overlay(
            predicate="one", instruction_suffix="LG", semantic_suffix="one"
        ),
        _v_cmp_f32_overlay(
            predicate="ord", instruction_suffix="O", semantic_suffix="ord"
        ),
        _v_cmp_f32_overlay(
            predicate="ueq", instruction_suffix="NLG", semantic_suffix="ueq"
        ),
        _v_cmp_f32_overlay(
            predicate="ugt", instruction_suffix="NLE", semantic_suffix="ugt"
        ),
        _v_cmp_f32_overlay(
            predicate="uge", instruction_suffix="NLT", semantic_suffix="uge"
        ),
        _v_cmp_f32_overlay(
            predicate="ult", instruction_suffix="NGE", semantic_suffix="ult"
        ),
        _v_cmp_f32_overlay(
            predicate="ule", instruction_suffix="NGT", semantic_suffix="ule"
        ),
        _v_cmp_f32_overlay(
            predicate="une", instruction_suffix="NEQ", semantic_suffix="une"
        ),
        _v_cmp_f32_overlay(
            predicate="uno", instruction_suffix="U", semantic_suffix="uno"
        ),
    )


def _v_cndmask_b32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_cndmask_b32",
        instruction_name="V_CNDMASK_B32",
        mnemonic="v_cndmask_b32",
        encoding_name="ENC_VOP3",
        semantic_tag="select.mask.b32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_const_operand("false_value")),
            AmdgpuOperandOverlay("SRC1", _vgpr_const_operand("true_value")),
            AmdgpuOperandOverlay("SRC2", _sgpr_predicate("mask", units=2)),
        ),
        operand_forms=(
            _literal_operand_form(
                replacement_descriptor="amdgpu.v_cndmask_b32.src0_inline",
                source_operand="false_value",
                immediate_field="false_value",
            ),
            _literal_operand_form(
                replacement_descriptor="amdgpu.v_cndmask_b32.src1_inline",
                source_operand="true_value",
                immediate_field="true_value",
            ),
            _literal_operand_form(
                replacement_descriptor="amdgpu.v_cndmask_b32.src0_lit",
                source_operand="false_value",
            ),
            _literal_operand_form(
                replacement_descriptor="amdgpu.v_cndmask_b32.src1_lit",
                source_operand="true_value",
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cndmask_b32_source_inline_overlay(
    literal_source: str,
) -> AmdgpuDescriptorOverlay:
    source_fields = {
        "src0": ("SRC0", "false_value", _vgpr_const_operand("false_value")),
        "src1": ("SRC1", "true_value", _vgpr_const_operand("true_value")),
    }
    literal_field, literal_operand, _ = source_fields[literal_source]
    remaining_operands = [
        (xml_field, field_name, operand)
        for source_name, (xml_field, field_name, operand) in source_fields.items()
        if source_name != literal_source
    ]
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_cndmask_b32.{literal_source}_inline",
        instruction_name="V_CNDMASK_B32",
        mnemonic="v_cndmask_b32",
        encoding_name="ENC_VOP3",
        semantic_tag="select.mask.b32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            *(
                AmdgpuOperandOverlay(xml_field, operand)
                for xml_field, _, operand in remaining_operands
            ),
            AmdgpuOperandOverlay("SRC2", _sgpr_predicate("mask", units=2)),
        ),
        asm_forms=_asm(
            mnemonic=f"v_cndmask_b32_{literal_source}_inline",
            results=("dst",),
            operands=(
                *(field_name for _, field_name, _ in remaining_operands),
                "mask",
            ),
            immediates=(literal_operand,),
            named_immediates=True,
        ),
        immediate_fields=(literal_field,),
        immediates=(_source_inline_u32_immediate(literal_operand),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cndmask_b32_source_literal_overlay(
    literal_source: str,
) -> AmdgpuDescriptorOverlay:
    source_fields = {
        "src0": ("SRC0", "false_value", _vgpr_const_operand("false_value")),
        "src1": ("SRC1", "true_value", _vgpr_const_operand("true_value")),
    }
    literal_field, _, _ = source_fields[literal_source]
    remaining_operands = [
        (xml_field, field_name, operand)
        for source_name, (xml_field, field_name, operand) in source_fields.items()
        if source_name != literal_source
    ]
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_cndmask_b32.{literal_source}_lit",
        instruction_name="V_CNDMASK_B32",
        mnemonic=f"v_cndmask_b32_{literal_source}_lit",
        encoding_name="ENC_VOP3",
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP3_LITERAL,
        semantic_tag="select.mask.b32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            *(
                AmdgpuOperandOverlay(xml_field, operand)
                for xml_field, _, operand in remaining_operands
            ),
            AmdgpuOperandOverlay("SRC2", _sgpr_predicate("mask", units=2)),
        ),
        asm_forms=_asm(
            results=("dst",),
            operands=(
                *(field_name for _, field_name, _ in remaining_operands),
                "mask",
            ),
            immediates=("imm32",),
        ),
        immediates=(_LITERAL_U32_IMMEDIATE,),
        fixed_encoding_fields=((literal_field, _predefined("SRC_LITERAL", "OPR_SRC")),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cndmask_b32_literal_inline_overlay(
    literal_source: str,
) -> AmdgpuDescriptorOverlay:
    source_fields = {
        "src0": ("SRC0", "false_value"),
        "src1": ("SRC1", "true_value"),
    }
    inline_source = "src1" if literal_source == "src0" else "src0"
    literal_field, _ = source_fields[literal_source]
    inline_field, inline_operand = source_fields[inline_source]
    return AmdgpuDescriptorOverlay(
        descriptor_key=(
            f"amdgpu.v_cndmask_b32.{literal_source}_lit_{inline_source}_inline"
        ),
        instruction_name="V_CNDMASK_B32",
        mnemonic=f"v_cndmask_b32_{literal_source}_lit_{inline_source}_inline",
        encoding_name="ENC_VOP3",
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP3_LITERAL,
        semantic_tag="select.mask.b32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC2", _sgpr_predicate("mask", units=2)),
        ),
        asm_forms=_asm(
            results=("dst",),
            operands=("mask",),
            immediates=("imm32", inline_operand),
            named_immediates=True,
        ),
        immediate_fields=(inline_field,),
        immediates=(
            _LITERAL_U32_IMMEDIATE,
            replace(
                _source_inline_u32_immediate(inline_operand),
                encoding_field_id=amdgpu_encoding_field_id(inline_field),
            ),
        ),
        fixed_encoding_fields=((literal_field, _predefined("SRC_LITERAL", "OPR_SRC")),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cndmask_b32_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _v_cndmask_b32_overlay(),
        _v_cndmask_b32_source_inline_overlay("src0"),
        _v_cndmask_b32_source_inline_overlay("src1"),
        _v_cndmask_b32_source_literal_overlay("src0"),
        _v_cndmask_b32_source_literal_overlay("src1"),
        _v_cndmask_b32_literal_inline_overlay("src0"),
        _v_cndmask_b32_literal_inline_overlay("src1"),
    )


def _v_mov_b32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_mov_b32",
        instruction_name="V_MOV_B32",
        mnemonic="v_mov_b32",
        encoding_name="VOP1_INST_LITERAL",
        encoding_condition="has_lit",
        semantic_tag="integer.const.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(AmdgpuOperandOverlay("VDST", _vgpr_result()),),
        asm_forms=_asm(results=("dst",), immediates=("imm32",)),
        immediate_fields=("LITERAL",),
        immediates=(_U32_IMMEDIATE,),
        fixed_encoding_fields=(("SRC0", _predefined("SRC_LITERAL", "OPR_SRC")),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_mov_b32_copy_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_mov_b32_copy",
        instruction_name="V_MOV_B32",
        mnemonic="v_mov_b32",
        encoding_name="ENC_VOP1",
        semantic_tag="register.copy.b32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("src")),
        ),
        asm_forms=_asm(mnemonic="v_mov_b32_copy", results=("dst",), operands=("src",)),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_mov_b32_dpp_overlay(
    *,
    descriptor_key: str,
    encoding_name: str,
    encoding_condition: str,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name="V_MOV_B32",
        mnemonic="v_mov_b32",
        encoding_name=encoding_name,
        encoding_condition=encoding_condition,
        semantic_tag="lane.dpp.b32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("VSRC0", _vgpr_operand("src")),
        ),
        asm_forms=_asm(
            mnemonic=descriptor_key.removeprefix("amdgpu."),
            results=("dst",),
            operands=("src",),
            immediates=("dpp_ctrl",),
            named_immediates=True,
        ),
        immediate_fields=("DPP_CTRL",),
        immediates=(_DPP_CTRL_IMMEDIATE,),
        fixed_encoding_fields=(
            ("SRC0", 250),
            ("ROW_MASK", 0xF),
            ("BANK_MASK", 0xF),
            ("BOUND_CTRL", 1),
        ),
        effects=(_CONVERGENT_EFFECT,),
    )


def _v_mov_b32_dpp_legacy_overlay() -> AmdgpuDescriptorOverlay:
    return _v_mov_b32_dpp_overlay(
        descriptor_key="amdgpu.v_mov_b32_dpp",
        encoding_name="VOP1_VOP_DPP",
        encoding_condition="has_dpp",
    )


def _v_mov_b32_dpp16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_mov_b32_dpp_overlay(
        descriptor_key="amdgpu.v_mov_b32_dpp16",
        encoding_name="VOP1_VOP_DPP16",
        encoding_condition="has_dpp16",
    )


__all__ = (
    "_integer_bitwise_shift_overlays",
    "_s_add_u32_overlay",
    "_s_add_u32_rhs_inline_overlay",
    "_s_addc_u32_overlay",
    "_s_and_b32_overlay",
    "_s_and_b64_overlay",
    "_s_and_saveexec_b64_overlay",
    "_s_ashr_i32_overlay",
    "_s_ashr_i32_rhs_inline_overlay",
    "_s_binary_u32_overlay",
    "_s_binary_u32_rhs_inline_overlay",
    "_s_binary_u64_overlay",
    "_s_cmp_i32_overlay",
    "_s_cmp_i32_overlays",
    "_s_cmp_u64_overlay",
    "_s_cmp_u64_overlays",
    "_s_cselect_b32_overlay",
    "_s_lshl_b32_overlay",
    "_s_lshl_b32_rhs_inline_overlay",
    "_s_lshl_b64_overlay",
    "_s_lshr_b32_overlay",
    "_s_lshr_b32_rhs_inline_overlay",
    "_s_lshr_b64_overlay",
    "_s_max_i32_overlay",
    "_s_max_u32_overlay",
    "_s_min_i32_overlay",
    "_s_min_u32_overlay",
    "_s_mul_hi_u32_overlay",
    "_s_mul_i32_overlay",
    "_s_or_b32_overlay",
    "_s_or_b64_overlay",
    "_s_shift_u64_overlay",
    "_s_sub_u32_overlay",
    "_s_sub_u32_rhs_inline_overlay",
    "_s_xor_b32_overlay",
    "_s_xor_b64_overlay",
    "_v_add_co_ci_u32_overlay",
    "_v_add_co_u32_overlay",
    "_v_add_f32_literal_overlay",
    "_v_add_f32_overlay",
    "_v_add_f32_src0_inline_overlay",
    "_v_add_u32_literal_overlay",
    "_v_add_u32_overlay",
    "_v_add_u32_src0_inline_overlay",
    "_v_and_b32_literal_overlay",
    "_v_and_b32_overlay",
    "_v_and_b32_src0_inline_overlay",
    "_v_ashrrev_i32_literal_overlay",
    "_v_ashrrev_i32_overlay",
    "_v_ashrrev_i32_src0_inline_overlay",
    "_v_binary_f32_operand_forms",
    "_v_binary_literal_overlay",
    "_v_binary_src0_inline_f32_overlay",
    "_v_binary_src0_inline_overlay",
    "_v_binary_u32_overlay",
    "_v_cmp_32_source_overlay",
    "_v_cmp_f32_overlay",
    "_v_cmp_i32_overlay",
    "_v_cmp_i32_source_overlays",
    "_v_cmp_inline_operand_forms",
    "_v_cmp_overlays",
    "_v_cmp_u32_overlay",
    "_v_cmp_u32_source_overlays",
    "_v_cndmask_b32_literal_inline_overlay",
    "_v_cndmask_b32_overlay",
    "_v_cndmask_b32_overlays",
    "_v_cndmask_b32_source_inline_overlay",
    "_v_cndmask_b32_source_literal_overlay",
    "_v_cvt_f16_f32_overlay",
    "_v_cvt_f32_f16_overlay",
    "_v_cvt_f32_i32_overlay",
    "_v_cvt_f32_u32_overlay",
    "_v_cvt_pk_bf16_f32_overlay",
    "_v_cvt_pk_u16_u32_overlay",
    "_v_cos_f32_overlay",
    "_v_exp_f32_overlay",
    "_v_fma_f32_overlay",
    "_v_fmac_f32_overlay",
    "_v_lshl_add_u32_shift_immediate_overlay",
    "_v_lshlrev_b32_literal_overlay",
    "_v_lshlrev_b32_overlay",
    "_v_lshlrev_b32_src0_inline_overlay",
    "_v_lshlrev_b32_vop3_immediate_overlay",
    "_v_lshrrev_b32_literal_overlay",
    "_v_lshrrev_b32_overlay",
    "_v_lshrrev_b32_src0_inline_overlay",
    "_v_mad_u32_u24_literal_overlay",
    "_v_mad_u32_u24_overlay",
    "_v_log_f32_overlay",
    "_v_max_f32_literal_overlay",
    "_v_max_f32_overlay",
    "_v_max_f32_src0_inline_overlay",
    "_v_max_i32_overlay",
    "_v_max_u32_overlay",
    "_v_min_f32_literal_overlay",
    "_v_min_f32_overlay",
    "_v_min_f32_src0_inline_overlay",
    "_v_min_i32_overlay",
    "_v_min_u32_overlay",
    "_v_minmax_i32_overlay",
    "_v_mov_b32_copy_overlay",
    "_v_mov_b32_dpp16_overlay",
    "_v_mov_b32_dpp_legacy_overlay",
    "_v_mov_b32_dpp_overlay",
    "_v_mov_b32_literal_overlay",
    "_v_native_f32_math_overlays",
    "_v_mul_f32_literal_overlay",
    "_v_mul_f32_overlay",
    "_v_mul_f32_src0_inline_overlay",
    "_v_mul_hi_u32_overlay",
    "_v_mul_lo_u32_overlay",
    "_v_mul_u32_u24_literal_overlay",
    "_v_mul_u32_u24_overlay",
    "_v_mul_u32_u24_src0_inline_overlay",
    "_v_or_b32_literal_overlay",
    "_v_or_b32_overlay",
    "_v_or_b32_src0_inline_overlay",
    "_v_rcp_f32_overlay",
    "_v_readfirstlane_b32_overlay",
    "_v_rsq_f32_overlay",
    "_v_sin_f32_overlay",
    "_v_sqrt_f32_overlay",
    "_v_sub_f32_literal_overlay",
    "_v_sub_f32_overlay",
    "_v_sub_f32_src0_inline_overlay",
    "_v_sub_u32_overlay",
    "_v_unary_f32_overlay",
    "_v_xor_b32_literal_overlay",
    "_v_xor_b32_overlay",
    "_v_xor_b32_src0_inline_overlay",
)
