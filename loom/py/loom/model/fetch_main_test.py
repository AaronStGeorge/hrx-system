# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CLI destination and selection contracts through actual local acquisition."""

import copy
import json
import unittest
from contextlib import chdir

import pytest

from loom.model.fetch_main import _default_output, main
from loom.model.manifest import ManifestError


def _publish(root, identity="publisher/tiny"):
    directory = root / "source"
    directory.mkdir()
    card = {
        "schema_version": 1,
        "model": {"id": identity, "name": "Tiny CLI fixture"},
        "programs": {"source": {"inputs": ["code.loom"], "libraries": []}},
        "services": {"chat": {"kind": "text", "version": 1}},
        "variants": {
            "original": {
                "services": {
                    "chat": {
                        "programs": ["source"],
                        "parameters": {"model": {"files": ["weights.data"]}},
                    }
                }
            }
        },
        "presets": {
            "chat": {"service": "chat", "variant": "original", "programs": "source"}
        },
    }
    path = directory / "model.json"
    path.write_text(json.dumps(card))
    (directory / "code.loom").write_bytes(b"program")
    (directory / "weights.data").write_bytes(b"weights")
    return path


def _run(root, source, *arguments):
    with chdir(root):
        return main(
            [
                str(source),
                "--cache=" + str(root / "cache"),
                "--offline",
                "--no-lock",
                "--no-input",
                "--json",
                *arguments,
            ]
        )


@pytest.mark.parametrize("identity", ["qwen3.5-9b", "publisher/tiny", "équipe/modèle"])
def test_default_output_preserves_the_model_id(tmp_path, capsys, identity):
    source = _publish(tmp_path, identity)
    assert _run(tmp_path, source) == 0
    result = capsys.readouterr()
    output = tmp_path / identity
    assert json.loads(result.out)["output"] == str(output)
    assert json.loads(result.out)["card"] == str(output / "model.json")
    assert (output / "weights.data").read_bytes() == b"weights"
    assert (output / "code.loom").read_bytes() == b"program"
    assert result.err.index("loom-fetch  Output") < result.err.index(
        "loom-fetch  Fetch"
    )
    assert str(output) in result.err


@pytest.mark.parametrize(
    "identity",
    [
        "",
        "/absolute",
        "../parent",
        "a/../b",
        "./model",
        "a//b",
        "a/",
        ".",
        "..",
        "C:/models",
        r"a\b",
        r"\\host\share",
        "trailing.",
        "trailing ",
        "a\nname",
        "a:stream",
        "a?b",
        "a*b",
        "a|b",
        'a"b',
        "<model>",
        "nul",
        "CON.txt",
        "aux/name",
        "COM1",
        "lpt9.bin",
        "COM¹.txt",
        "LPT²",
        "CONIN$",
        "CONOUT$",
    ],
)
def test_nonportable_identifiers_require_an_explicit_output(identity):
    with pytest.raises(ManifestError, match="specify --output"):
        _default_output(identity)


def test_explicit_destination_accepts_opaque_model_ids(tmp_path, capsys):
    source = _publish(tmp_path, "model:opaque")
    assert _run(tmp_path, source) == 1
    failed = capsys.readouterr()
    assert "specify --output" in json.loads(failed.out)["message"]
    assert "loom-fetch  Fetch" not in failed.err
    assert _run(tmp_path, source, "--output=chosen") == 0
    result = capsys.readouterr()
    assert json.loads(result.out)["output"] == str(tmp_path / "chosen")


def test_implicit_output_does_not_follow_a_namespace_symlink(tmp_path):
    linked = tmp_path / "linked"
    linked.mkdir()
    try:
        (tmp_path / "publisher").symlink_to(linked, target_is_directory=True)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            raise unittest.SkipTest(
                "Windows symlink creation is not enabled"
            ) from error
        raise
    with (
        chdir(tmp_path),
        pytest.raises(ManifestError, match="symbolic link or junction"),
    ):
        _default_output("publisher/tiny")


def test_dry_run_and_listing_do_not_create_default_output(tmp_path, capsys):
    source = _publish(tmp_path)
    assert _run(tmp_path, source, "--dry-run") == 0
    assert json.loads(capsys.readouterr().out)["output"] == str(
        tmp_path / "publisher/tiny"
    )
    assert not (tmp_path / "publisher").exists()
    assert _run(tmp_path, source, "--list") == 0
    listed = capsys.readouterr()
    assert json.loads(listed.out)["model"]["id"] == "publisher/tiny"
    assert "loom-fetch  Output" not in listed.err
    assert not (tmp_path / "publisher").exists()


def test_repeat_keeps_edits_and_conflicting_preset_needs_another_output(
    tmp_path, capsys
):
    source = _publish(tmp_path)
    assert _run(tmp_path, source) == 0
    capsys.readouterr()
    program = tmp_path / "publisher/tiny/code.loom"
    program.write_bytes(b"edited program")
    assert _run(tmp_path, source) == 0
    capsys.readouterr()
    assert program.read_bytes() == b"edited program"
    card = json.loads(source.read_text())
    card["variants"]["alternate"] = copy.deepcopy(card["variants"]["original"])
    card["presets"]["alternate"] = {
        "variant": "alternate",
        "service": "chat",
        "programs": "source",
    }
    source.write_text(json.dumps(card))
    assert _run(tmp_path, source, "--preset=alternate") == 1
    assert (
        "conflicting installation document"
        in json.loads(capsys.readouterr().out)["message"]
    )
    assert program.read_bytes() == b"edited program"
    assert _run(tmp_path, source, "--preset=alternate", "--output=alternative") == 0
    assert json.loads(capsys.readouterr().out)["output"] == str(
        tmp_path / "alternative"
    )


def test_empty_catalog_reports_no_models(tmp_path, capsys):
    catalog = tmp_path / "catalog.json"
    catalog.write_text('{"schema_version":1,"models":{}}')
    assert _run(tmp_path, "--catalog=" + str(catalog)) == 1
    assert (
        "catalog contains no models" in json.loads(capsys.readouterr().out)["message"]
    )
