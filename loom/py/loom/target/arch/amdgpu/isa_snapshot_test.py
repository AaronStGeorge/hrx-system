# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json

import pytest

from loom.target.arch.amdgpu.descriptor_overlay import (
    AmdgpuDescriptorOverlay,
    AmdgpuImplicitOperandOverlay,
    AmdgpuOperandOverlay,
    materialize_amdgpu_descriptor_overlay,
)
from loom.target.arch.amdgpu.isa_snapshot import (
    AmdgpuIsaSnapshotAllowlist,
    AmdgpuIsaSnapshotError,
    build_amdgpu_isa_snapshot,
    format_amdgpu_isa_snapshot_json,
    format_amdgpu_isa_snapshot_report_json,
    parse_amdgpu_isa_snapshot_json,
)
from loom.target.arch.amdgpu.isa_xml import (
    AmdgpuIsaBitRange,
    AmdgpuIsaEncoding,
    AmdgpuIsaEncodingField,
    AmdgpuIsaFunctionalGroup,
    AmdgpuIsaInstruction,
    AmdgpuIsaInstructionEncoding,
    AmdgpuIsaInstructionFlags,
    AmdgpuIsaOperand,
    AmdgpuIsaSpec,
    parse_amdgpu_isa_xml_text,
)
from loom.target.arch.amdgpu.isa_xml_test import SAMPLE_XML
from loom.target.low_descriptors import (
    DescriptorFlag,
    Operand,
    OperandRole,
    RegClassAlt,
)

_REG_SGPR = "amdgpu.sgpr"
_REG_VGPR = "amdgpu.vgpr"

_SGPR_ALT = (RegClassAlt(_REG_SGPR),)
_VGPR_ALT = (RegClassAlt(_REG_VGPR),)

_IGNORE_SCC_OUTPUT = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_SSRC_SPECIAL_SCC",
    data_format_name="FMT_NUM_B1",
    size_bits=1,
    is_input=False,
    is_output=True,
    ignore_reason="value-pseudo-drops-scc",
)


def _result(field_name: str, reg_alts: tuple[RegClassAlt, ...]) -> Operand:
    return Operand(field_name, OperandRole.RESULT, reg_alts)


def _operand(field_name: str, reg_alts: tuple[RegClassAlt, ...]) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, reg_alts)


def test_snapshot_round_trips_overlay_and_encoding_bit_layout_facts() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="/tmp/vendor.xml")

    result = build_amdgpu_isa_snapshot(
        spec,
        target_family_key="amdgpu.gfx11",
        snapshot_name="amdgpu.gfx11.core",
        allowlist=AmdgpuIsaSnapshotAllowlist(
            instruction_names=("S_ADD_U32", "V_ADD_NC_U32"),
            encoding_names=("ENC_SOP2", "ENC_VOP2"),
        ),
    )
    snapshot_json = format_amdgpu_isa_snapshot_json(result.snapshot)

    assert "/tmp/vendor.xml" not in snapshot_json
    assert "Description" not in snapshot_json
    assert "BUFFER_LOAD_B32" not in snapshot_json
    assert "S_WAIT_IDLE" not in snapshot_json
    assert "VOP2_INST_LITERAL" not in snapshot_json

    loaded_snapshot = parse_amdgpu_isa_snapshot_json(
        snapshot_json, source_name="generated.json"
    )

    assert loaded_snapshot.snapshot_name == "amdgpu.gfx11.core"
    assert loaded_snapshot.target_family_key == "amdgpu.gfx11"
    assert loaded_snapshot.architecture_name == "AMD RDNA 4"
    assert [encoding.name for encoding in loaded_snapshot.encodings] == [
        "ENC_SOP2",
        "ENC_VOP2",
    ]
    assert [instruction.name for instruction in loaded_snapshot.instructions] == [
        "S_ADD_CO_U32",
        "V_ADD_NC_U32",
    ]

    field_summaries = loaded_snapshot.encoding_field_summaries(["ENC_VOP2"])
    assert len(field_summaries) == 1
    assert field_summaries[0].field_name == "OPSEL_HI"
    assert field_summaries[0].ranges[0].bit_offset == 59
    assert field_summaries[0].ranges[1].bit_offset == 14

    descriptor = materialize_amdgpu_descriptor_overlay(
        loaded_snapshot,
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
            implicit_operands=(_IGNORE_SCC_OUTPUT,),
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
    )

    assert descriptor.key == "amdgpu.s_add_u32"
    assert descriptor.encoding_id == 0
    assert [operand.field_name for operand in descriptor.operands] == [
        "dst",
        "lhs",
        "rhs",
    ]


def test_snapshot_report_tracks_dropped_source_facts_outside_artifact() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")

    result = build_amdgpu_isa_snapshot(
        spec,
        target_family_key="amdgpu.gfx11",
        allowlist=AmdgpuIsaSnapshotAllowlist(
            instruction_names=("V_ADD_NC_U32",),
            encoding_names=("ENC_VOP2", "ENC_SOPP"),
        ),
    )
    snapshot_json = format_amdgpu_isa_snapshot_json(result.snapshot)
    report = json.loads(format_amdgpu_isa_snapshot_report_json(result))

    assert "S_WAIT_IDLE" not in snapshot_json
    assert "ENC_SOPP" in snapshot_json
    assert result.report.dropped_instruction_names == (
        "BUFFER_LOAD_B32",
        "S_ADD_CO_U32",
        "S_WAIT_IDLE",
    )
    assert result.report.dropped_encoding_names == ("ENC_SOP2", "ENC_VBUFFER")
    assert result.report.dropped_instruction_encoding_names == (
        "V_ADD_NC_U32/VOP2_INST_LITERAL/has_lit/opcode=37",
    )
    assert result.report.unreferenced_encoding_names == ("ENC_SOPP",)
    assert report["dropped"]["instructions"] == [
        "BUFFER_LOAD_B32",
        "S_ADD_CO_U32",
        "S_WAIT_IDLE",
    ]
    assert report["unreferenced_selected_encodings"] == ["ENC_SOPP"]


def test_snapshot_can_select_precise_instruction_encodings() -> None:
    spec = AmdgpuIsaSpec(
        source_name="synthetic.xml",
        architecture_name="Synthetic AMDGPU",
        architecture_id=99,
        encodings=(
            AmdgpuIsaEncoding(
                name="ENC_SHARED",
                order=1,
                bit_count=32,
                identifier_mask=0xF0000000,
                identifier_values=(0xA0000000,),
                fields=(
                    AmdgpuIsaEncodingField(
                        name="OP",
                        is_conditional=False,
                        ranges=(AmdgpuIsaBitRange(0, 8, 0),),
                    ),
                ),
            ),
        ),
        instructions=(
            AmdgpuIsaInstruction(
                name="INST",
                aliases=(),
                flags=AmdgpuIsaInstructionFlags(
                    is_branch=False,
                    is_conditional_branch=False,
                    is_indirect_branch=False,
                    is_program_terminator=False,
                    is_immediately_executed=False,
                ),
                encodings=(
                    AmdgpuIsaInstructionEncoding(
                        encoding_name="ENC_SHARED",
                        condition_name="default",
                        opcode=1,
                        operands=(
                            AmdgpuIsaOperand(
                                order=1,
                                field_name="VDST",
                                data_format_name="FMT_NUM_U32",
                                operand_type="OPR_VGPR",
                                size_bits=32,
                                is_input=False,
                                is_output=True,
                                is_implicit=False,
                                is_binary_microcode_required=True,
                            ),
                        ),
                    ),
                    AmdgpuIsaInstructionEncoding(
                        encoding_name="ENC_SHARED",
                        condition_name="literal",
                        opcode=2,
                        operands=(
                            AmdgpuIsaOperand(
                                order=1,
                                field_name="VDST",
                                data_format_name="FMT_NUM_U32",
                                operand_type="OPR_VGPR",
                                size_bits=32,
                                is_input=False,
                                is_output=True,
                                is_implicit=False,
                                is_binary_microcode_required=True,
                            ),
                        ),
                    ),
                ),
                functional_groups=(AmdgpuIsaFunctionalGroup("VALU", ("INTEGER",)),),
            ),
        ),
    )

    result = build_amdgpu_isa_snapshot(
        spec,
        target_family_key="amdgpu.gfx11",
        allowlist=AmdgpuIsaSnapshotAllowlist(
            instruction_names=("INST",),
            encoding_names=("ENC_SHARED",),
            instruction_encoding_names=("INST/ENC_SHARED/default",),
        ),
    )
    snapshot = result.snapshot

    assert [encoding.name for encoding in snapshot.encodings] == ["ENC_SHARED"]
    assert len(snapshot.instructions) == 1
    assert snapshot.instructions[0].name == "INST"
    assert [
        (encoding.encoding_name, encoding.condition_name)
        for encoding in snapshot.instructions[0].encodings
    ] == [("ENC_SHARED", "default")]
    assert result.report.dropped_instruction_encoding_names == (
        "INST/ENC_SHARED/literal/opcode=2",
    )
    assert result.report.unreferenced_encoding_names == ()


def test_snapshot_rejects_unknown_allowlist_instruction() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")

    with pytest.raises(
        AmdgpuIsaSnapshotError,
        match="unknown AMDGPU ISA instruction\\(s\\): MISSING",
    ):
        build_amdgpu_isa_snapshot(
            spec,
            target_family_key="amdgpu.gfx11",
            allowlist=AmdgpuIsaSnapshotAllowlist(
                instruction_names=("MISSING",),
                encoding_names=("ENC_SOP2",),
            ),
        )


def test_snapshot_rejects_unknown_allowlist_encoding() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")

    with pytest.raises(
        AmdgpuIsaSnapshotError,
        match="unknown AMDGPU ISA encoding\\(s\\): MISSING",
    ):
        build_amdgpu_isa_snapshot(
            spec,
            target_family_key="amdgpu.gfx11",
            allowlist=AmdgpuIsaSnapshotAllowlist(
                instruction_names=("S_ADD_U32",),
                encoding_names=("MISSING",),
            ),
        )


def test_snapshot_rejects_duplicate_instruction_encoding_after_alias_resolution() -> (
    None
):
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")

    with pytest.raises(
        AmdgpuIsaSnapshotError,
        match="instruction encoding allowlist repeats 'S_ADD_CO_U32/ENC_SOP2/default'",
    ):
        build_amdgpu_isa_snapshot(
            spec,
            target_family_key="amdgpu.gfx11",
            allowlist=AmdgpuIsaSnapshotAllowlist(
                instruction_names=("S_ADD_U32",),
                encoding_names=("ENC_SOP2",),
                instruction_encoding_names=(
                    "S_ADD_U32/ENC_SOP2/default",
                    "S_ADD_CO_U32/ENC_SOP2/default",
                ),
            ),
        )


def test_snapshot_rejects_instruction_without_selected_encoding() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")

    with pytest.raises(
        AmdgpuIsaSnapshotError,
        match="'S_ADD_CO_U32' has no encodings selected by allowlist",
    ):
        build_amdgpu_isa_snapshot(
            spec,
            target_family_key="amdgpu.gfx11",
            allowlist=AmdgpuIsaSnapshotAllowlist(
                instruction_names=("S_ADD_U32",),
                encoding_names=("ENC_VOP2",),
            ),
        )


def test_snapshot_rejects_malformed_json() -> None:
    snapshot_json = json.dumps(
        {
            "schema_version": 999,
            "snapshot_name": "bad",
            "target_family_key": "amdgpu.bad",
        }
    )

    with pytest.raises(
        AmdgpuIsaSnapshotError,
        match="unsupported AMDGPU ISA snapshot schema_version 999",
    ):
        parse_amdgpu_isa_snapshot_json(snapshot_json, source_name="bad.json")


def test_snapshot_rejects_duplicate_checked_in_facts() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    result = build_amdgpu_isa_snapshot(
        spec,
        target_family_key="amdgpu.gfx11",
        allowlist=AmdgpuIsaSnapshotAllowlist(
            instruction_names=("S_ADD_U32",),
            encoding_names=("ENC_SOP2",),
        ),
    )
    snapshot = json.loads(format_amdgpu_isa_snapshot_json(result.snapshot))
    snapshot["instructions"].append(snapshot["instructions"][0])

    with pytest.raises(
        AmdgpuIsaSnapshotError,
        match="AMDGPU ISA snapshot instruction repeats 'S_ADD_CO_U32'",
    ):
        parse_amdgpu_isa_snapshot_json(json.dumps(snapshot), source_name="bad.json")
