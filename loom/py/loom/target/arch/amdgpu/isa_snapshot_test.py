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
from loom.target.arch.amdgpu.isa_xml import parse_amdgpu_isa_xml_text
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
