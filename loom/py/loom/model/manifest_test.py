# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
from pathlib import Path

import pytest

from loom.model import ManifestError, ModelCard


def _document():
    return json.loads((Path(__file__).parent / "testdata/moonshine.json").read_text())


def test_complete_moonshine_selection():
    document = _document()
    selected = ModelCard(document).select("streaming")
    assert selected.model_id == "moonshine-ai/moonshine-streaming-tiny"
    assert selected.entrypoint["kind"] == "recognition"
    assert selected.settings["execution"] == {"pending_requests": 4}
    assert len(selected.files) == 19
    assert "program/model.loombc" not in selected.files
    assert selected.files["asset/model.safetensors"]["size"] == 176237308


def test_program_name_has_no_format_semantics():
    document = _document()
    programs = document["entrypoints"]["recognition"]["programs"]
    programs["source"] = programs.pop("release")
    document["variants"]["original"]["entrypoints"]["recognition"]["programs"] = [
        "development",
        "source",
    ]
    selected = ModelCard(document).select("streaming", programs="source")
    assert "program/model.loombc" in selected.files
    assert "program/convolution.loom" in selected.files
    assert "program/prepare.loom" not in selected.files


def test_no_implicit_cartesian_product():
    document = _document()
    document["variants"]["original"]["entrypoints"]["recognition"]["programs"] = [
        "development"
    ]
    card = ModelCard(document)
    with pytest.raises(ManifestError, match="not supported"):
        card.select("streaming", programs="release")


def test_owned_snapshots():
    document = _document()
    card = ModelCard(document)
    selected = card.select("streaming")
    document.clear()
    selected.settings["execution"]["pending_requests"] = 77
    assert card.select("streaming").settings["execution"]["pending_requests"] == 4


def test_rejects_duplicate_json_keys():
    with pytest.raises(ManifestError, match="duplicate object key"):
        ModelCard.parse('{"schema_version":1,"schema_version":1}')


def test_rejects_missing_file():
    document = _document()
    del document["files"]["program/control.loom"]
    with pytest.raises(ManifestError, match="unknown reference"):
        ModelCard(document)


def test_rejects_duplicate_provider_role():
    document = _document()
    program = document["entrypoints"]["recognition"]["programs"]["development"]
    program["libraries"].append("program/prepare.loom")
    with pytest.raises(ManifestError, match="duplicate provider"):
        ModelCard(document)


def test_rejects_directory_member_escape():
    document = _document()
    binding = document["variants"]["original"]["entrypoints"]["recognition"][
        "bindings"
    ]["assets"]
    binding["files"].append("program/control.loom")
    with pytest.raises(ManifestError, match="outside the directory"):
        ModelCard(document)


def test_rejects_changed_inline_asset():
    document = _document()
    document["files"]["asset/rotary.safetensors"]["size"] += 1
    with pytest.raises(ManifestError, match="does not match"):
        ModelCard(document)


def test_unknown_settings_are_preserved_for_model_validation():
    document = _document()
    document["entrypoints"]["recognition"]["options"] = ["voice"]
    document["configurations"]["streaming"]["options"] = {
        "voice": {"name": "example", "tokens": [1, 2, 3]}
    }
    selection = ModelCard(document).select("streaming")
    assert selection.settings["options"]["voice"]["tokens"] == [1, 2, 3]


def test_fixed_model_facts_cannot_be_overridden():
    document = _document()
    document["configurations"]["streaming"]["options"] = {"layers": 17}
    with pytest.raises(ManifestError, match="undeclared model options"):
        ModelCard(document)
