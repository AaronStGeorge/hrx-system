# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""HF catalog discovery through the real provider, cache, and public CLI."""

import hashlib
import json
import subprocess
import sys

import pytest

from loom.model import ManifestError, ModelCard
from loom.model.catalog import generate
from loom.model.fetch import install, plan_fetch
from loom.model.fetch_catalog import load_catalog
from loom.model.fetch_hf import HuggingFace
from loom.model.fetch_source import FetchError
from loom.model.fetch_storage import ContentCache
from loom.model.hf_test.hub import running_hub
from loom.model.lock import FileIdentity, ModelLock

_DIRECTORY = "publication/model/tiny/"
_CATALOG = "publication/catalog.json"
_URL = "https://huggingface.co/fixture/model/blob/main/" + _CATALOG


def _publish(hub):
    card = ModelCard(
        {
            "schema_version": 1,
            "model": {"id": "example/tiny", "name": "Tiny publication"},
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
        }
    )
    weights = b"tiny checkpoint"
    hub.files = {
        _CATALOG: json.dumps(
            generate({"model/tiny/model.json": card}).document
        ).encode(),
        _DIRECTORY + "model.json": json.dumps(card.document).encode(),
        _DIRECTORY + "model.json.lock": json.dumps(
            ModelLock.snapshot(
                {},
                [
                    FileIdentity(
                        "weights.data",
                        len(weights),
                        hashlib.sha256(weights).hexdigest(),
                    )
                ],
                scope="assets",
            )
        ).encode(),
        _DIRECTORY + "code.loom": b"code",
        _DIRECTORY + "weights.data": weights,
        "unused.data": b"unselected payload",
    }
    hub.credentials.write_text("fixture-first")
    hub.approved = True


def _cache(root, hub):
    return ContentCache(
        root / "cache",
        huggingface=HuggingFace(endpoint=hub.endpoint, cache_dir=hub.cache_dir),
    )


def test_snapshot_survives_branch_advance_and_offline_install(tmp_path):
    with running_hub(tmp_path) as hub:
        _publish(hub)
        cache = _cache(tmp_path, hub)
        first = load_catalog(_URL, cache, offline=False)
        assert first.revision == "a" * 40
        assert first.document.location.endswith("/resolve/" + "a" * 40 + "/" + _CATALOG)
        assert not any("/model/tiny/" in path for _, path in hub.requests)

        hub.snapshots[hub.revision] = dict(hub.files)
        hub.revision = "b" * 40
        hub.files[_DIRECTORY + "code.loom"] = b"changed code"
        # Discovery selected one publication, not the branch's later contents.
        document, card = first.read_model("example/tiny", cache, offline=False)
        installed = install(
            plan_fetch(document, card.select(), cache), cache, tmp_path / "first"
        )
        assert (installed.parent / "code.loom").read_bytes() == b"code"
        assert not any(path.endswith("unused.data") for _, path in hub.requests)

        cache = _cache(tmp_path, hub)
        second = load_catalog(_URL, cache, offline=False)
        assert second.revision == "b" * 40
        document, card = second.read_model("example/tiny", cache, offline=False)
        installed = install(
            plan_fetch(document, card.select(), cache), cache, tmp_path / "second"
        )
        assert (installed.parent / "code.loom").read_bytes() == b"changed code"

        requests = list(hub.requests)
        hub.approved = False
        cache = _cache(tmp_path, hub)
        offline = load_catalog(_URL, cache, offline=True)
        assert offline.revision == second.revision
        document, card = offline.read_model("example/tiny", cache, offline=True)
        installed = install(
            plan_fetch(document, card.select(), cache, offline=True),
            cache,
            tmp_path / "offline",
        )
        assert (installed.parent / "code.loom").read_bytes() == b"changed code"
        assert hub.requests == requests


def test_invalid_refresh_preserves_last_valid_offline_catalog(tmp_path):
    with running_hub(tmp_path) as hub:
        _publish(hub)
        with pytest.raises(FetchError, match="no cached snapshot"):
            load_catalog(_URL, _cache(tmp_path, hub), offline=True)
        assert hub.requests == []
        load_catalog(_URL, _cache(tmp_path, hub), offline=False)
        hub.snapshots[hub.revision] = dict(hub.files)
        hub.revision = "b" * 40
        hub.files[_CATALOG] = b'{"schema_version":99,"models":{}}'
        with pytest.raises(ManifestError, match="unsupported model-catalog"):
            load_catalog(_URL, _cache(tmp_path, hub), offline=False)
        requests = list(hub.requests)
        snapshot = load_catalog(_URL, _cache(tmp_path, hub), offline=True)
        assert snapshot.revision == "a" * 40
        assert hub.requests == requests


def test_cli_lists_acquires_and_reports_access_without_prompts(tmp_path):
    with running_hub(tmp_path) as hub:
        _publish(hub)
        source = hub.endpoint + "/fixture/model/resolve/main/" + _CATALOG
        command = [
            sys.executable,
            "-m",
            "loom.model.fetch_main",
            "--catalog=" + source,
            "--cache=" + str(tmp_path / "cache"),
            "--json",
            "--no-input",
        ]
        listed = subprocess.run(
            [*command, "--list"], env=hub.environment(), capture_output=True, text=True
        )
        assert listed.returncode == 0, listed.stderr
        assert set(json.loads(listed.stdout)["models"]) == {"example/tiny"}
        assert not any("/model/tiny/" in path for _, path in hub.requests)
        result = subprocess.run(
            [
                *command,
                "--model=example/tiny",
                "--output=" + str(tmp_path / "installed"),
            ],
            env=hub.environment(),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        result = json.loads(result.stdout)
        assert result["catalog"]["revision"] == hub.revision
        receipt = json.loads((tmp_path / "installed/.loom-fetch.json").read_text())
        assert receipt["catalog"]["revision"] == hub.revision
        assert (tmp_path / "installed/weights.data").read_bytes() == b"tiny checkpoint"

        hub.approved = False
        denied = subprocess.run(
            [*command, "--list"], env=hub.environment(), capture_output=True, text=True
        )
        assert denied.returncode == 1, denied.stderr
        assert json.loads(denied.stdout)["kind"] == "action_required"
        assert "fixture-first" not in denied.stdout + denied.stderr
