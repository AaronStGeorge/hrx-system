# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""Matrix, dot, WMMA, MFMA, and SWMMAC descriptor overlays."""

from __future__ import annotations

from .common import *


def _v_wmma_16x16x16_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    input_units: int,
    accumulator_units: int,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_VOP3P",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_WMMA,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result(units=accumulator_units)),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("a", units=input_units)),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("b", units=input_units)),
            AmdgpuOperandOverlay(
                "SRC2", _vgpr_const_operand("acc", units=accumulator_units)
            ),
        ),
        constraints=_destructive_accumulator_constraints(3),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _with_zero_accumulator_form(
    overlay: AmdgpuDescriptorOverlay,
) -> tuple[AmdgpuDescriptorOverlay, AmdgpuDescriptorOverlay]:
    if overlay.mnemonic is None:
        raise ValueError(
            f"WMMA descriptor '{overlay.descriptor_key}' needs a mnemonic for "
            "its zero-accumulator asm form"
        )
    zero_descriptor_key = f"{overlay.descriptor_key}.acc_zero"
    source_overlay = replace(
        overlay,
        operand_forms=(
            *overlay.operand_forms,
            OperandForm(
                replacement_descriptor=zero_descriptor_key,
                matches=(
                    OperandFormMatch(
                        source_operand="acc",
                        match_kind=OperandFormMatchKind.ALL_EQUAL_I64,
                        match_i64=0,
                    ),
                ),
            ),
        ),
    )

    zero_operands = tuple(
        operand_overlay
        for operand_overlay in overlay.operands
        if operand_overlay.descriptor_operand.field_name != "acc"
    )
    zero_result_names = tuple(
        operand_overlay.descriptor_operand.field_name
        for operand_overlay in zero_operands
        if operand_overlay.descriptor_operand.role is OperandRole.RESULT
    )
    zero_operand_names = tuple(
        operand_overlay.descriptor_operand.field_name
        for operand_overlay in zero_operands
        if operand_overlay.descriptor_operand.role
        in (OperandRole.OPERAND, OperandRole.PREDICATE, OperandRole.RESOURCE)
        and OperandFlag.IMPLICIT not in operand_overlay.descriptor_operand.flags
    )
    zero_overlay = replace(
        overlay,
        descriptor_key=zero_descriptor_key,
        operands=zero_operands,
        fixed_encoding_fields=(
            *overlay.fixed_encoding_fields,
            ("SRC2", _predefined("0")),
        ),
        constraints=(),
        operand_forms=(),
        asm_forms=_asm(
            mnemonic=f"{overlay.mnemonic}_acc_zero",
            results=zero_result_names,
            operands=zero_operand_names,
        ),
    )
    return source_overlay, zero_overlay


def _v_wmma_f32_16x16x16_f16_overlay(
    *, input_units: int = 4
) -> AmdgpuDescriptorOverlay:
    return _v_wmma_16x16x16_overlay(
        descriptor_key="amdgpu.v_wmma_f32_16x16x16_f16",
        instruction_name="V_WMMA_F32_16X16X16_F16",
        mnemonic="v_wmma_f32_16x16x16_f16",
        semantic_tag="matrix.wmma.f32.16x16x16.f16",
        input_units=input_units,
        accumulator_units=8,
    )


def _v_wmma_f32_16x16x16_bf16_overlay(
    *, input_units: int = 4
) -> AmdgpuDescriptorOverlay:
    return _v_wmma_16x16x16_overlay(
        descriptor_key="amdgpu.v_wmma_f32_16x16x16_bf16",
        instruction_name="V_WMMA_F32_16X16X16_BF16",
        mnemonic="v_wmma_f32_16x16x16_bf16",
        semantic_tag="matrix.wmma.f32.16x16x16.bf16",
        input_units=input_units,
        accumulator_units=8,
    )


def _v_wmma_f16_16x16x16_f16_overlay(
    *, input_units: int = 4, accumulator_units: int = 4
) -> AmdgpuDescriptorOverlay:
    return _v_wmma_16x16x16_overlay(
        descriptor_key="amdgpu.v_wmma_f16_16x16x16_f16",
        instruction_name="V_WMMA_F16_16X16X16_F16",
        mnemonic="v_wmma_f16_16x16x16_f16",
        semantic_tag="matrix.wmma.f16.16x16x16.f16",
        input_units=input_units,
        accumulator_units=accumulator_units,
    )


def _v_wmma_bf16_16x16x16_bf16_overlay(
    *, input_units: int = 4, accumulator_units: int = 4
) -> AmdgpuDescriptorOverlay:
    return _v_wmma_16x16x16_overlay(
        descriptor_key="amdgpu.v_wmma_bf16_16x16x16_bf16",
        instruction_name="V_WMMA_BF16_16X16X16_BF16",
        mnemonic="v_wmma_bf16_16x16x16_bf16",
        semantic_tag="matrix.wmma.bf16.16x16x16.bf16",
        input_units=input_units,
        accumulator_units=accumulator_units,
    )


def _v_wmma_f32_16x16x16_packed8_overlay(
    *, lhs_type: str, rhs_type: str, input_units: int = 2
) -> AmdgpuDescriptorOverlay:
    lhs_type_upper = lhs_type.upper()
    rhs_type_upper = rhs_type.upper()
    return _v_wmma_16x16x16_overlay(
        descriptor_key=f"amdgpu.v_wmma_f32_16x16x16_{lhs_type}_{rhs_type}",
        instruction_name=f"V_WMMA_F32_16X16X16_{lhs_type_upper}_{rhs_type_upper}",
        mnemonic=f"v_wmma_f32_16x16x16_{lhs_type}_{rhs_type}",
        semantic_tag=f"matrix.wmma.f32.16x16x16.{lhs_type}.{rhs_type}",
        input_units=input_units,
        accumulator_units=8,
    )


def _v_wmma_i32_16x16x16_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    operand_units: int,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_VOP3P",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_WMMA,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result(units=8)),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("a", units=operand_units)),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("b", units=operand_units)),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc", units=8)),
        ),
        constraints=_destructive_accumulator_constraints(3),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_wmma_i32_16x16x16_iu8_overlay(
    *, operand_units: int = 2
) -> AmdgpuDescriptorOverlay:
    return _v_wmma_i32_16x16x16_overlay(
        descriptor_key="amdgpu.v_wmma_i32_16x16x16_iu8",
        instruction_name="V_WMMA_I32_16X16X16_IU8",
        mnemonic="v_wmma_i32_16x16x16_iu8",
        semantic_tag="matrix.wmma.i32.16x16x16.iu8",
        operand_units=operand_units,
    )


def _v_wmma_i32_16x16x16_iu4_overlay(
    *, operand_units: int = 1
) -> AmdgpuDescriptorOverlay:
    return _v_wmma_i32_16x16x16_overlay(
        descriptor_key="amdgpu.v_wmma_i32_16x16x16_iu4",
        instruction_name="V_WMMA_I32_16X16X16_IU4",
        mnemonic="v_wmma_i32_16x16x16_iu4",
        semantic_tag="matrix.wmma.i32.16x16x16.iu4",
        operand_units=operand_units,
    )


def _v_wmma_i32_16x16x32_iu4_overlay() -> AmdgpuDescriptorOverlay:
    return _v_wmma_i32_16x16x16_overlay(
        descriptor_key="amdgpu.v_wmma_i32_16x16x32_iu4",
        instruction_name="V_WMMA_I32_16X16X32_IU4",
        mnemonic="v_wmma_i32_16x16x32_iu4",
        semantic_tag="matrix.wmma.i32.16x16x32.iu4",
        operand_units=2,
    )


def _v_mfma_f32_16x16x16_f16_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_mfma_f32_16x16x16_f16",
        instruction_name="V_MFMA_F32_16X16X16_F16",
        mnemonic="v_mfma_f32_16x16x16_f16",
        encoding_name="VOP3P_MFMA",
        semantic_tag="matrix.mfma.f32.16x16x16.f16",
        schedule_class=_SCHEDULE_MFMA,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_agpr_result(units=4)),
            AmdgpuOperandOverlay("SRC0", _vgpr_agpr_operand("a", units=2)),
            AmdgpuOperandOverlay("SRC1", _vgpr_agpr_operand("b", units=2)),
            AmdgpuOperandOverlay("SRC2", _vgpr_agpr_const_operand("acc", units=4)),
        ),
        constraints=_destructive_accumulator_constraints(3),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_mfma_f32_16x16x16_bf16_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_mfma_f32_16x16x16_bf16",
        instruction_name="V_MFMA_F32_16X16X16_BF16",
        mnemonic="v_mfma_f32_16x16x16_bf16",
        encoding_name="VOP3P_MFMA",
        semantic_tag="matrix.mfma.f32.16x16x16.bf16.1k",
        schedule_class=_SCHEDULE_MFMA,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_agpr_result(units=4)),
            AmdgpuOperandOverlay("SRC0", _vgpr_agpr_operand("a", units=2)),
            AmdgpuOperandOverlay("SRC1", _vgpr_agpr_operand("b", units=2)),
            AmdgpuOperandOverlay("SRC2", _vgpr_agpr_const_operand("acc", units=4)),
        ),
        constraints=_destructive_accumulator_constraints(3),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_mfma_f32_16x16x4_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_mfma_f32_16x16x4_f32",
        instruction_name="V_MFMA_F32_16X16X4_F32",
        mnemonic="v_mfma_f32_16x16x4_f32",
        encoding_name="VOP3P_MFMA",
        semantic_tag="matrix.mfma.f32.16x16x4.f32",
        schedule_class=_SCHEDULE_MFMA,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_agpr_result(units=4)),
            AmdgpuOperandOverlay("SRC0", _vgpr_agpr_operand("a", units=1)),
            AmdgpuOperandOverlay("SRC1", _vgpr_agpr_operand("b", units=1)),
            AmdgpuOperandOverlay("SRC2", _vgpr_agpr_const_operand("acc", units=4)),
        ),
        constraints=_destructive_accumulator_constraints(3),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_smfmac_f32_16x16x32_bf16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_smfmac_f32_overlay(
        descriptor_key="amdgpu.v_smfmac_f32_16x16x32_bf16",
        instruction_name="V_SMFMAC_F32_16X16X32_BF16",
        mnemonic="v_smfmac_f32_16x16x32_bf16",
        semantic_tag="matrix.smfmac.f32.16x16x32.bf16",
        accumulator_units=4,
        lhs_units=2,
        rhs_units=4,
    )


def _v_smfmac_f32_16x16x32_f16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_smfmac_f32_overlay(
        descriptor_key="amdgpu.v_smfmac_f32_16x16x32_f16",
        instruction_name="V_SMFMAC_F32_16X16X32_F16",
        mnemonic="v_smfmac_f32_16x16x32_f16",
        semantic_tag="matrix.smfmac.f32.16x16x32.f16",
        accumulator_units=4,
        lhs_units=2,
        rhs_units=4,
    )


def _v_smfmac_f32_32x32x16_bf16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_smfmac_f32_overlay(
        descriptor_key="amdgpu.v_smfmac_f32_32x32x16_bf16",
        instruction_name="V_SMFMAC_F32_32X32X16_BF16",
        mnemonic="v_smfmac_f32_32x32x16_bf16",
        semantic_tag="matrix.smfmac.f32.32x32x16.bf16",
        accumulator_units=16,
        lhs_units=2,
        rhs_units=4,
    )


def _v_smfmac_f32_32x32x16_f16_overlay() -> AmdgpuDescriptorOverlay:
    return _v_smfmac_f32_overlay(
        descriptor_key="amdgpu.v_smfmac_f32_32x32x16_f16",
        instruction_name="V_SMFMAC_F32_32X32X16_F16",
        mnemonic="v_smfmac_f32_32x32x16_f16",
        semantic_tag="matrix.smfmac.f32.32x32x16.f16",
        accumulator_units=16,
        lhs_units=2,
        rhs_units=4,
    )


def _v_smfmac_f32_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    accumulator_units: int,
    lhs_units: int,
    rhs_units: int,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="VOP3P_MFMA",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_MFMA,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_agpr_result(units=accumulator_units)),
            AmdgpuOperandOverlay(
                "VDST",
                _vgpr_agpr_operand("acc", units=accumulator_units),
                role_exception_reason=_SMFMAC_VDST_ACCUMULATOR_REASON,
            ),
            AmdgpuOperandOverlay("SRC0", _vgpr_agpr_operand("a", units=lhs_units)),
            AmdgpuOperandOverlay("SRC1", _vgpr_agpr_operand("b", units=rhs_units)),
            AmdgpuOperandOverlay("SRC2", _vgpr_operand("index")),
        ),
        constraints=_DESTRUCTIVE_ACCUMULATOR_CONSTRAINTS,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _vop3p_packed_dot_fixed_fields(
    *,
    op_sel_hi_field: str = "OP_SEL_HI",
    lhs_signed: bool = False,
    rhs_signed: bool = False,
) -> tuple[tuple[str, int], ...]:
    # Packed VOP3P dot instructions use high-half source selection as the
    # canonical no-op modifier spelling. IU dot instructions additionally use
    # NEG low bits as integer signedness selectors for src0/src1.
    neg = (1 if lhs_signed else 0) | (2 if rhs_signed else 0)
    fields = [(op_sel_hi_field, 0x7)]
    if neg != 0:
        fields.append(("NEG", neg))
    return tuple(fields)


def _v_dot2_f32_packed_float_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    op_sel_hi_field: str = "OP_SEL_HI",
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_VOP3P",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc")),
        ),
        fixed_encoding_fields=_vop3p_packed_dot_fixed_fields(
            op_sel_hi_field=op_sel_hi_field
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_dot2_f32_f16_overlay(
    *, op_sel_hi_field: str = "OP_SEL_HI"
) -> AmdgpuDescriptorOverlay:
    return _v_dot2_f32_packed_float_overlay(
        descriptor_key="amdgpu.v_dot2_f32_f16",
        instruction_name="V_DOT2_F32_F16",
        mnemonic="v_dot2_f32_f16",
        semantic_tag="dot.f16f16.f32x1",
        op_sel_hi_field=op_sel_hi_field,
    )


def _v_dot2_f32_bf16_overlay(
    *, op_sel_hi_field: str = "OP_SEL_HI"
) -> AmdgpuDescriptorOverlay:
    return _v_dot2_f32_packed_float_overlay(
        descriptor_key="amdgpu.v_dot2_f32_bf16",
        instruction_name="V_DOT2_F32_BF16",
        mnemonic="v_dot2_f32_bf16",
        semantic_tag="dot.bf16bf16.f32x1",
        op_sel_hi_field=op_sel_hi_field,
    )


def _v_dot4_i32_i8_overlay(
    *, op_sel_hi_field: str = "OP_SEL_HI", signedness_modifiers: bool
) -> AmdgpuDescriptorOverlay:
    instruction_name = "V_DOT4_I32_IU8" if signedness_modifiers else "V_DOT4_I32_I8"
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_dot4_i32_i8",
        instruction_name=instruction_name,
        mnemonic="v_dot4_i32_i8",
        encoding_name="ENC_VOP3P",
        semantic_tag="dot.s8s8.i32x1",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc")),
        ),
        fixed_encoding_fields=_vop3p_packed_dot_fixed_fields(
            op_sel_hi_field=op_sel_hi_field,
            lhs_signed=signedness_modifiers,
            rhs_signed=signedness_modifiers,
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_dot4_i32_iu8_overlay(
    *, op_sel_hi_field: str = "OP_SEL_HI", lhs_signed: bool, rhs_signed: bool
) -> AmdgpuDescriptorOverlay:
    lhs_tag = "s8" if lhs_signed else "u8"
    rhs_tag = "s8" if rhs_signed else "u8"
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_dot4_i32_iu8.{lhs_tag}{rhs_tag}",
        instruction_name="V_DOT4_I32_IU8",
        mnemonic="v_dot4_i32_iu8",
        encoding_name="ENC_VOP3P",
        semantic_tag=f"dot.{lhs_tag}{rhs_tag}.i32x1",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc")),
        ),
        fixed_encoding_fields=_vop3p_packed_dot_fixed_fields(
            op_sel_hi_field=op_sel_hi_field,
            lhs_signed=lhs_signed,
            rhs_signed=rhs_signed,
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
        # Native asm forms cannot yet spell the fixed NEG source selectors, so
        # the target-low descriptor key remains the only unambiguous text form.
        asm_forms=(),
    )


def _v_dot4_u32_u8_overlay(
    *,
    op_sel_hi_field: str = "OP_SEL_HI",
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_dot4_u32_u8",
        instruction_name="V_DOT4_U32_U8",
        mnemonic="v_dot4_u32_u8",
        encoding_name="ENC_VOP3P",
        semantic_tag="dot.u8u8.i32x1",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc")),
        ),
        fixed_encoding_fields=_vop3p_packed_dot_fixed_fields(
            op_sel_hi_field=op_sel_hi_field
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_dot4_f32_packed8_overlay(
    *, lhs_type: str, rhs_type: str
) -> AmdgpuDescriptorOverlay:
    lhs_type_upper = lhs_type.upper()
    rhs_type_upper = rhs_type.upper()
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_dot4_f32_{lhs_type}_{rhs_type}",
        instruction_name=f"V_DOT4_F32_{lhs_type_upper}_{rhs_type_upper}",
        mnemonic=f"v_dot4_f32_{lhs_type}_{rhs_type}",
        encoding_name="ENC_VOP3P",
        semantic_tag=f"dot.{lhs_type}{rhs_type}.f32x1",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_dot8_i32_i4_overlay(
    *, op_sel_hi_field: str = "OP_SEL_HI", signedness_modifiers: bool
) -> AmdgpuDescriptorOverlay:
    instruction_name = "V_DOT8_I32_IU4" if signedness_modifiers else "V_DOT8_I32_I4"
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_dot8_i32_i4",
        instruction_name=instruction_name,
        mnemonic="v_dot8_i32_i4",
        encoding_name="ENC_VOP3P",
        semantic_tag="dot.s4s4.i32x1",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc")),
        ),
        fixed_encoding_fields=_vop3p_packed_dot_fixed_fields(
            op_sel_hi_field=op_sel_hi_field,
            lhs_signed=signedness_modifiers,
            rhs_signed=signedness_modifiers,
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_dot8_i32_iu4_overlay(
    *, op_sel_hi_field: str = "OP_SEL_HI", lhs_signed: bool, rhs_signed: bool
) -> AmdgpuDescriptorOverlay:
    lhs_tag = "s4" if lhs_signed else "u4"
    rhs_tag = "s4" if rhs_signed else "u4"
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_dot8_i32_iu4.{lhs_tag}{rhs_tag}",
        instruction_name="V_DOT8_I32_IU4",
        mnemonic="v_dot8_i32_iu4",
        encoding_name="ENC_VOP3P",
        semantic_tag=f"dot.{lhs_tag}{rhs_tag}.i32x1",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc")),
        ),
        fixed_encoding_fields=_vop3p_packed_dot_fixed_fields(
            op_sel_hi_field=op_sel_hi_field,
            lhs_signed=lhs_signed,
            rhs_signed=rhs_signed,
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
        # Native asm forms cannot yet spell the fixed NEG source selectors, so
        # the target-low descriptor key remains the only unambiguous text form.
        asm_forms=(),
    )


def _v_dot8_u32_u4_overlay(
    *,
    op_sel_hi_field: str = "OP_SEL_HI",
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_dot8_u32_u4",
        instruction_name="V_DOT8_U32_U4",
        mnemonic="v_dot8_u32_u4",
        encoding_name="ENC_VOP3P",
        semantic_tag="dot.u4u4.i32x1",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc")),
        ),
        fixed_encoding_fields=_vop3p_packed_dot_fixed_fields(
            op_sel_hi_field=op_sel_hi_field
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


__all__ = (
    "_v_dot2_f32_bf16_overlay",
    "_v_dot2_f32_f16_overlay",
    "_v_dot2_f32_packed_float_overlay",
    "_v_dot4_f32_packed8_overlay",
    "_v_dot4_i32_i8_overlay",
    "_v_dot4_i32_iu8_overlay",
    "_v_dot4_u32_u8_overlay",
    "_v_dot8_i32_i4_overlay",
    "_v_dot8_i32_iu4_overlay",
    "_v_dot8_u32_u4_overlay",
    "_v_mfma_f32_16x16x16_bf16_overlay",
    "_v_mfma_f32_16x16x16_f16_overlay",
    "_v_mfma_f32_16x16x4_f32_overlay",
    "_v_smfmac_f32_16x16x32_bf16_overlay",
    "_v_smfmac_f32_16x16x32_f16_overlay",
    "_v_smfmac_f32_32x32x16_bf16_overlay",
    "_v_smfmac_f32_32x32x16_f16_overlay",
    "_v_smfmac_f32_overlay",
    "_v_wmma_16x16x16_overlay",
    "_v_wmma_bf16_16x16x16_bf16_overlay",
    "_v_wmma_f16_16x16x16_f16_overlay",
    "_v_wmma_f32_16x16x16_bf16_overlay",
    "_v_wmma_f32_16x16x16_f16_overlay",
    "_v_wmma_f32_16x16x16_packed8_overlay",
    "_v_wmma_i32_16x16x16_iu4_overlay",
    "_v_wmma_i32_16x16x16_iu8_overlay",
    "_v_wmma_i32_16x16x16_overlay",
    "_v_wmma_i32_16x16x32_iu4_overlay",
    "_vop3p_packed_dot_fixed_fields",
    "_with_zero_accumulator_form",
)
