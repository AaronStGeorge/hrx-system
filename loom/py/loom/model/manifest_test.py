# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
import json
from pathlib import Path

import pytest

from loom.model import ManifestError, ModelCard


def _document():
    # This is also the native server's executable fixture, not a second Python
    # interpretation of the model's parameter/configuration contract.
    root = Path(__file__).parents[4]
    return json.loads(
        (root / "loom/src/loom/serve/fixture/text/native.card.json").read_text()
    )


def _service(document):
    return document["variants"]["fixture"]["services"]["chat"]


def test_native_selected_closure():
    selected = ModelCard(_document()).select("check")
    assert selected.service == {"kind": "text", "version": 1}
    assert selected.config == {"fixture.card": True, "fixture.bias_delta": 1}
    assert selected.settings["execution"]["rows"] == 4
    assert set(selected.files) == {
        "prepare.loom",
        "control.loom",
        "service.loom",
        "advance.loom",
        "weights.loom",
        "metadata.safetensors",
        "bias.data",
        "tokenizer.json",
    }
    assert selected.files["bias.data"].format == "safetensors"
    assert selected.files["weights.loom"].roles == {"auxiliary"}
    assert selected.parameters["model"]["files"][0] == "metadata.safetensors"
    assert ModelCard(selected.document).select().files == selected.files


def test_source_and_bytecode_names_are_opaque():
    document = _document()
    document["programs"]["textual"] = document["programs"].pop("archive")
    for service in document["variants"]["fixture"]["services"].values():
        service["programs"] = ["source", "textual", "mixed"]
    selected = ModelCard(document).select("check", programs="textual")
    assert "service_module.loombc" in selected.files
    assert "prepare.loom" not in selected.files
    mixed = ModelCard(document).select("check", programs="mixed")
    assert "prepare.loom" in mixed.files and "control_module.loombc" in mixed.files


def test_moonshine_metadata_is_not_a_qualification_claim():
    document = json.loads(
        (Path(__file__).parent / "testdata/moonshine.json").read_text()
    )
    selected = ModelCard(document).select("streaming")
    assert selected.model_id == "moonshine-ai/moonshine-streaming-tiny"
    assert selected.service["kind"] == "recognition"
    assert len(selected.files) == 19
    assert (
        selected.files["asset/model.safetensors"].origin
        == "@checkpoint/model.safetensors"
    )
    assert selected.files["asset/text-vocabulary.data"].format == "huggingface-json"
    assert "asset/config.json" not in selected.files
    assert document["qualification"] == {}


def test_origins_separate_acquisition_from_placement():
    document = _document()
    document["origins"] = {"checkpoint": {"kind": "hf", "repository": "owner/model"}}
    parameters = _service(document)["parameters"]["model"]
    parameters["origin"] = "@checkpoint/"
    parameters["files"][1]["origin"] = "@checkpoint/some/stupid/name.tensor"
    selected = ModelCard(document).select()
    assert (
        selected.files["metadata.safetensors"].origin
        == "@checkpoint/metadata.safetensors"
    )
    assert selected.files["bias.data"].origin == "@checkpoint/some/stupid/name.tensor"
    assert selected.files["bias.data"].path == "bias.data"


def test_owned_snapshots():
    document = _document()
    card = ModelCard(document)
    selected = card.select()
    document.clear()
    selected.settings["execution"]["rows"] = 77
    selected.document["programs"].clear()
    assert card.select().settings["execution"]["rows"] == 4
    assert card.select().document["programs"]


def test_independent_service_versions():
    document = _document()
    document["services"]["future"] = {"kind": "image", "version": 54}
    _service(document)["programs"] = ["source"]
    assert ModelCard(document).select().service["version"] == 1
    document["services"]["chat"].pop("version")
    assert "version" not in ModelCard(document).select().service


def test_explicit_selection_without_canonical_configuration():
    document = _document()
    document.pop("configurations")
    card = ModelCard(document)
    with pytest.raises(ManifestError, match="select explicitly"):
        card.select()
    assert card.select(service="chat", programs="mixed").programs == "mixed"


def test_programs_must_be_supported_by_variant():
    document = _document()
    _service(document)["programs"] = ["source"]
    with pytest.raises(ManifestError, match="unknown reference"):
        ModelCard(document).select("check", programs="archive")


def test_same_destination_combines_roles_but_not_conflicting_sources():
    document = _document()
    parameter = _service(document)["parameters"]["model"]
    parameter["files"].append("weights.loom")
    card = ModelCard(document)
    assert card.select().files["weights.loom"].roles == {"auxiliary", "parameter"}
    parameter["files"][-1] = {"path": "weights.loom", "origin": "other.loom"}
    with pytest.raises(ManifestError, match="conflicting selected"):
        ModelCard(document)


def test_options_preserve_types_but_require_declared_names():
    document = _document()
    document["services"]["chat"]["options"] = ["voice"]
    document["configurations"]["check"]["options"] = {"voice": {"tokens": [1, 2]}}
    assert ModelCard(document).select().settings["options"]["voice"]["tokens"] == [1, 2]
    document["configurations"]["check"]["options"]["layers"] = 17
    with pytest.raises(ManifestError, match="undeclared model options"):
        ModelCard(document)


@pytest.mark.parametrize(
    "source",
    [
        '{"schema_version":1,"schema_version":1}',
        '{"schema_version":1,"schema_versi\\u006fn":1}',
        '{"schema_version":NaN}',
        '{"schema_version":1,}',
        '{/* comment */"schema_version":1}',
        '{"value":1e400}',
    ],
)
def test_rejects_nonportable_json(source):
    with pytest.raises(ManifestError):
        ModelCard.parse(source)


@pytest.mark.parametrize(
    "path", ["/root/a", "../a", "a/../b", "./a", "a//b", "a/", "C:/a", "a\\b", "a\0b"]
)
def test_rejects_unsafe_destinations(path):
    document = _document()
    document["programs"]["source"]["inputs"][0]["path"] = path
    with pytest.raises(ManifestError):
        ModelCard(document)


def test_rejects_file_directory_collision():
    document = _document()
    _service(document)["parameters"]["model"]["files"].append("bias.data/nested")
    with pytest.raises(ManifestError, match="collides"):
        ModelCard(document)


def test_rejects_duplicate_provider_role():
    document = _document()
    document["programs"]["source"]["libraries"].append("prepare.loom")
    with pytest.raises(ManifestError, match="duplicate provider"):
        ModelCard(document)


@pytest.mark.parametrize("value", [None, [], {"nested": 1}])
def test_jit_configuration_is_scalar(value):
    document = _document()
    _service(document)["config"]["invalid"] = value
    with pytest.raises(ManifestError, match="scalar"):
        ModelCard(document)


def test_new_schema_dispatch_precedes_structure():
    document = {"schema_version": 7, "unknown_future_shape": True}
    with pytest.raises(ManifestError, match="unsupported model-card version"):
        ModelCard(document)


def test_inline_and_root_validation():
    document = _document()
    target = document["programs"]["source"]["inputs"][0]
    for origin in (
        {"kind": "inline", "base64": "!"},
        "@unknown/source",
        "https://user:password@example.org/file",
        "https://@example.org/file",
        "https://example.org:invalid/file",
        "https://example.org/line\nbreak",
    ):
        target["origin"] = origin
        with pytest.raises(ManifestError):
            ModelCard(document)
    document = _document()
    document["programs"]["source"]["origin"] = "https://example.org/?path=/"
    with pytest.raises(ManifestError, match="query"):
        ModelCard(document)


def test_projection_excludes_other_selections():
    document = _document()
    document["variants"]["unused"] = copy.deepcopy(document["variants"]["fixture"])
    selected = ModelCard(document).select("check")
    assert set(selected.document["programs"]) == {"source"}
    assert set(selected.document["services"]) == {"chat"}
    assert set(selected.document["variants"]) == {"fixture"}
