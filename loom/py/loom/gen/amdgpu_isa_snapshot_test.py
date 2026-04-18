# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
from pathlib import Path

from loom.gen import amdgpu_isa_snapshot
from loom.target.arch.amdgpu.isa_snapshot import parse_amdgpu_isa_snapshot_json
from loom.target.arch.amdgpu.isa_xml_test import SAMPLE_XML


def test_amdgpu_isa_snapshot_generator_writes_snapshot_and_manifest(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "vendor.xml"
    snapshot_path = tmp_path / "snapshot.json"
    manifest_path = tmp_path / "manifest.json"
    xml_path.write_text(SAMPLE_XML, encoding="utf-8")

    result = amdgpu_isa_snapshot.main(
        [
            "--xml",
            str(xml_path),
            "--out",
            str(snapshot_path),
            "--target-family-key",
            "amdgpu.gfx11",
            "--snapshot-name",
            "amdgpu.gfx11.core",
            "--instruction",
            "S_ADD_U32",
            "--instruction",
            "S_WAIT_IDLE",
            "--encoding",
            "ENC_SOP2",
            "--encoding",
            "ENC_SOPP",
            "--manifest-out",
            str(manifest_path),
        ]
    )

    assert result == 0
    snapshot_json = snapshot_path.read_text(encoding="utf-8")
    snapshot = parse_amdgpu_isa_snapshot_json(snapshot_json)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert snapshot.snapshot_name == "amdgpu.gfx11.core"
    assert [instruction.name for instruction in snapshot.instructions] == [
        "S_ADD_CO_U32",
        "S_WAIT_IDLE",
    ]
    assert "vendor.xml" not in snapshot_json
    assert manifest["selected"]["instructions"] == [
        "S_ADD_CO_U32",
        "S_WAIT_IDLE",
    ]
    assert manifest["dropped"]["instructions"] == [
        "BUFFER_LOAD_B32",
        "V_ADD_NC_U32",
    ]


def test_amdgpu_isa_snapshot_generator_fails_loudly_on_bad_allowlist(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "vendor.xml"
    snapshot_path = tmp_path / "snapshot.json"
    xml_path.write_text(SAMPLE_XML, encoding="utf-8")

    result = amdgpu_isa_snapshot.main(
        [
            "--xml",
            str(xml_path),
            "--out",
            str(snapshot_path),
            "--target-family-key",
            "amdgpu.gfx11",
            "--instruction",
            "MISSING",
            "--encoding",
            "ENC_SOP2",
        ]
    )

    assert result == 1
    assert not snapshot_path.exists()
