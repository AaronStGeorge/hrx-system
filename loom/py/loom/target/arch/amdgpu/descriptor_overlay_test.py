# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from loom.target.arch.amdgpu.descriptor_overlay import (
    AmdgpuDescriptorOverlay,
    AmdgpuDescriptorOverlayError,
    AmdgpuOperandOverlay,
    materialize_amdgpu_descriptor_overlay,
    materialize_amdgpu_descriptor_overlays,
)
from loom.target.arch.amdgpu.isa_xml import parse_amdgpu_isa_xml_text
from loom.target.arch.amdgpu.isa_xml_test import SAMPLE_XML
from loom.target.low_descriptors import (
    DescriptorFlag,
    Immediate,
    ImmediateKind,
    Operand,
    OperandRole,
    RegClassAlt,
)

_REG_SGPR = "amdgpu.sgpr"
_REG_VGPR = "amdgpu.vgpr"

_SGPR_ALT = (RegClassAlt(_REG_SGPR),)
_VGPR_ALT = (RegClassAlt(_REG_VGPR),)

_U32_IMMEDIATE = Immediate(
    "literal",
    ImmediateKind.UNSIGNED,
    bit_width=32,
    unsigned_max=(2**32) - 1,
)


def _result(field_name: str, reg_alts: tuple[RegClassAlt, ...]) -> Operand:
    return Operand(field_name, OperandRole.RESULT, reg_alts)


def _operand(field_name: str, reg_alts: tuple[RegClassAlt, ...]) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, reg_alts)


def _resource(field_name: str, reg_alts: tuple[RegClassAlt, ...]) -> Operand:
    return Operand(field_name, OperandRole.RESOURCE, reg_alts)


def test_materialize_amdgpu_descriptor_overlays_from_xml_facts() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")

    descriptors = materialize_amdgpu_descriptor_overlays(
        spec,
        (
            AmdgpuDescriptorOverlay(
                descriptor_key="amdgpu.s_add_u32",
                instruction_name="S_ADD_U32",
                mnemonic="s_add_u32",
                encoding_name="ENC_SOP2",
                semantic_tag="integer.add.u32",
                schedule_class="amdgpu.salu",
                operands=(
                    AmdgpuOperandOverlay("SDST", _result("dst", _SGPR_ALT)),
                    AmdgpuOperandOverlay("SSRC0", _operand("lhs", _SGPR_ALT)),
                    AmdgpuOperandOverlay("SSRC1", _operand("rhs", _SGPR_ALT)),
                ),
                flags=(DescriptorFlag.DEAD_REMOVABLE,),
            ),
            AmdgpuDescriptorOverlay(
                descriptor_key="amdgpu.v_add_u32.lit",
                instruction_name="V_ADD_NC_U32",
                mnemonic="v_add_u32",
                encoding_name="VOP2_INST_LITERAL",
                encoding_condition="has_lit",
                semantic_tag="integer.add.u32",
                schedule_class="amdgpu.valu",
                operands=(
                    AmdgpuOperandOverlay("VDST", _result("dst", _VGPR_ALT)),
                    AmdgpuOperandOverlay("VSRC1", _operand("rhs", _VGPR_ALT)),
                ),
                immediate_fields=("LITERAL",),
                immediates=(_U32_IMMEDIATE,),
                flags=(DescriptorFlag.DEAD_REMOVABLE,),
            ),
            AmdgpuDescriptorOverlay(
                descriptor_key="amdgpu.buffer_load_dword",
                instruction_name="BUFFER_LOAD_DWORD",
                mnemonic="buffer_load_dword",
                encoding_name="ENC_VBUFFER",
                semantic_tag="memory.load.u32",
                schedule_class="amdgpu.vmem.load",
                operands=(
                    AmdgpuOperandOverlay("VDATA", _result("dst", _VGPR_ALT)),
                    AmdgpuOperandOverlay("VADDR", _operand("vaddr", _VGPR_ALT)),
                    AmdgpuOperandOverlay("RSRC", _resource("resource", _SGPR_ALT)),
                    AmdgpuOperandOverlay("SOFFSET", _operand("soffset", _SGPR_ALT)),
                ),
            ),
            AmdgpuDescriptorOverlay(
                descriptor_key="amdgpu.s_wait_idle",
                instruction_name="S_WAIT_IDLE",
                encoding_name="ENC_SOPP",
                semantic_tag="control.waitcnt.idle",
                schedule_class="amdgpu.wait",
                operands=(),
                flags=(DescriptorFlag.SIDE_EFFECTING,),
            ),
        ),
    )

    assert [descriptor.key for descriptor in descriptors] == [
        "amdgpu.s_add_u32",
        "amdgpu.v_add_u32.lit",
        "amdgpu.buffer_load_dword",
        "amdgpu.s_wait_idle",
    ]
    assert descriptors[0].mnemonic == "s_add_u32"
    assert [operand.field_name for operand in descriptors[0].operands] == [
        "dst",
        "lhs",
        "rhs",
    ]
    assert descriptors[1].immediates == (_U32_IMMEDIATE,)
    assert descriptors[1].encoding_id == 37
    assert descriptors[1].operands[1].field_name == "rhs"
    assert descriptors[2].encoding_id == 22
    assert descriptors[2].operands[2].role is OperandRole.RESOURCE
    assert descriptors[3].mnemonic == "s_wait_idle"
    assert descriptors[3].encoding_id == 10


def test_materialize_rejects_missing_instruction() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    overlay = AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.missing",
        instruction_name="MISSING",
        encoding_name="ENC_SOP2",
        semantic_tag="missing",
        schedule_class="amdgpu.salu",
        operands=(),
    )

    with pytest.raises(
        AmdgpuDescriptorOverlayError,
        match="references unknown AMDGPU instruction 'MISSING'",
    ):
        materialize_amdgpu_descriptor_overlay(spec, overlay)


def test_materialize_rejects_missing_encoding() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    overlay = AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.bad_encoding",
        instruction_name="S_ADD_U32",
        encoding_name="ENC_VOP2",
        semantic_tag="integer.add.u32",
        schedule_class="amdgpu.salu",
        operands=(),
    )

    with pytest.raises(
        AmdgpuDescriptorOverlayError,
        match="could not find encoding 'ENC_VOP2' condition 'default'",
    ):
        materialize_amdgpu_descriptor_overlay(spec, overlay)


def test_materialize_rejects_operand_role_mismatch() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    overlay = AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.bad_role",
        instruction_name="S_ADD_U32",
        encoding_name="ENC_SOP2",
        semantic_tag="integer.add.u32",
        schedule_class="amdgpu.salu",
        operands=(
            AmdgpuOperandOverlay("SDST", _operand("dst", _SGPR_ALT)),
            AmdgpuOperandOverlay("SSRC0", _operand("lhs", _SGPR_ALT)),
            AmdgpuOperandOverlay("SSRC1", _operand("rhs", _SGPR_ALT)),
        ),
    )

    with pytest.raises(
        AmdgpuDescriptorOverlayError,
        match="maps XML field 'SDST' to low operand 'dst' as input",
    ):
        materialize_amdgpu_descriptor_overlay(spec, overlay)


def test_materialize_rejects_repeated_overlay_xml_field() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    overlay = AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.duplicate",
        instruction_name="V_ADD_NC_U32",
        encoding_name="VOP2_INST_LITERAL",
        encoding_condition="has_lit",
        semantic_tag="integer.add.u32",
        schedule_class="amdgpu.valu",
        operands=(
            AmdgpuOperandOverlay("VDST", _result("dst", _VGPR_ALT)),
            AmdgpuOperandOverlay("LITERAL", _operand("lhs", _VGPR_ALT)),
            AmdgpuOperandOverlay("VSRC1", _operand("rhs", _VGPR_ALT)),
        ),
        immediate_fields=("LITERAL",),
    )

    with pytest.raises(
        AmdgpuDescriptorOverlayError,
        match="repeats XML field 'LITERAL' across operands and immediates",
    ):
        materialize_amdgpu_descriptor_overlay(spec, overlay)


def test_materialize_rejects_uncovered_explicit_xml_operand() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    overlay = AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.incomplete",
        instruction_name="S_ADD_U32",
        encoding_name="ENC_SOP2",
        semantic_tag="integer.add.u32",
        schedule_class="amdgpu.salu",
        operands=(
            AmdgpuOperandOverlay("SDST", _result("dst", _SGPR_ALT)),
            AmdgpuOperandOverlay("SSRC0", _operand("lhs", _SGPR_ALT)),
        ),
    )

    with pytest.raises(
        AmdgpuDescriptorOverlayError,
        match="does not cover XML operand field\\(s\\): SSRC1",
    ):
        materialize_amdgpu_descriptor_overlay(spec, overlay)
