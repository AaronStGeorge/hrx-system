# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import hashlib
import http.server
import json
import os
import threading
from contextlib import contextmanager

import pytest

from loom.model import ManifestError, ModelCard
from loom.model.fetch import check_output, install, plan_fetch
from loom.model.fetch_source import Document, FetchError
from loom.model.fetch_storage import ContentCache
from loom.model.lock import FileIdentity, ModelLock


def _card():
    return {
        "schema_version": 1,
        "model": {"id": "fixture/fetch", "name": "Fetch fixture"},
        "programs": {"p": {"inputs": ["code.loom"], "libraries": []}},
        "services": {"text": {"kind": "text", "version": 1}},
        "variants": {
            "v": {
                "services": {
                    "text": {
                        "programs": ["p"],
                        "parameters": {"scope": {"files": ["weights.data"]}},
                    }
                }
            },
            "unused": {
                "services": {
                    "text": {
                        "programs": ["p"],
                        "parameters": {"scope": {"files": ["large.data"]}},
                    }
                }
            },
        },
        "configurations": {
            "check": {"variant": "v", "service": "text", "programs": "p"}
        },
    }


def _lock(files):
    return ModelLock.snapshot(
        {},
        [
            FileIdentity(name, len(content), hashlib.sha256(content).hexdigest())
            for name, content in files.items()
        ],
    )


@contextmanager
def _server(card, files, *, parallel=False):
    requests = []
    resources = {
        "/model.json": json.dumps(card).encode(),
        "/model.json.lock": json.dumps(_lock(files)).encode(),
        **{"/" + name: content for name, content in files.items()},
    }
    rendezvous = threading.Barrier(2) if parallel else None

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            content = resources.get(self.path)
            if content is None:
                self.send_error(404)
                return
            if rendezvous and self.path in ("/code.loom", "/weights.data"):
                rendezvous.wait()
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *_):
            pass

    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", requests, resources
        finally:
            server.shutdown()
            thread.join()


def _remote_plan(base, cache, **kwargs):
    document = cache.read_document(
        base + "/model.json", offline=kwargs.get("offline", False)
    )
    return plan_fetch(
        document, ModelCard.parse(document.text).select(), cache, **kwargs
    )


def test_cancelled_operation_does_not_publish_an_installation(tmp_path):
    files = {"code.loom": b"code", "weights.data": b"weights"}
    with _server(_card(), files) as (base, requests, _):
        cache = ContentCache(tmp_path / "cache")
        plan = _remote_plan(base, cache)
        before = list(requests)
        cache.cancel()
        with pytest.raises(FetchError, match="cancelled"):
            install(plan, cache, tmp_path / "output")
        assert requests == before
        assert not (tmp_path / "output").exists()


def test_parallel_verified_install_and_offline_reuse(tmp_path):
    cache = ContentCache(tmp_path / "cache")
    output = tmp_path / "first"
    files = {"code.loom": b"code", "weights.data": b"weights", "large.data": b"unused"}
    with _server(_card(), files, parallel=True) as (base, requests, _):
        plan = _remote_plan(base, cache)
        result = install(plan, cache, output)
        assert result == output / "model.json"
        assert sorted(requests) == [
            "/code.loom",
            "/model.json",
            "/model.json.lock",
            "/weights.data",
        ]
    offline = _remote_plan(base, cache, offline=True)
    install(offline, cache, tmp_path / "second")
    assert os.path.samefile(output / "weights.data", tmp_path / "second/weights.data")
    # The installed document retains the original HTTP coordinates, and its
    # selected lock is a usable projection of the publication's superset lock.
    installed = ModelCard.parse(result.read_text()).select()
    locked = ModelLock.parse((output / "model.json.lock").read_text())
    assert len(locked.cover(installed)) == 2
    assert installed.files["code.loom"].origin == "@document/code.loom"
    assert installed.document["origins"]["document"]["url"] == base + "/"
    assert len(locked.files) == 2
    install(offline, cache, output)


def test_tampered_code_does_not_publish_partial_installation(tmp_path):
    cache = ContentCache(tmp_path / "cache")
    with _server(_card(), {"code.loom": b"code", "weights.data": b"weights"}) as (
        base,
        _,
        resources,
    ):
        resources["/code.loom"] = b"evil"
        plan = _remote_plan(base, cache)
        output = tmp_path / "output"
        with pytest.raises(ManifestError, match="SHA-256"):
            install(plan, cache, output)
        assert not (output / "model.json").exists()
        assert not (output / "code.loom").exists()
        assert not (output / "weights.data").exists()


def test_conflicting_output_fails_before_payload_request(tmp_path):
    cache = ContentCache(tmp_path / "cache")
    with _server(_card(), {"code.loom": b"code", "weights.data": b"weights"}) as (
        base,
        requests,
        _,
    ):
        plan = _remote_plan(base, cache)
        output = tmp_path / "output"
        output.mkdir()
        (output / "code.loom").write_bytes(b"mine")
        with pytest.raises(ManifestError, match="integrity mismatch"):
            install(plan, cache, output)
        assert requests == ["/model.json", "/model.json.lock"]
        assert (output / "code.loom").read_bytes() == b"mine"


def test_dry_run_does_not_create_output_or_read_payloads(tmp_path):
    cache = ContentCache(tmp_path / "cache")
    with _server(_card(), {"code.loom": b"code", "weights.data": b"weights"}) as (
        base,
        requests,
        _,
    ):
        plan = _remote_plan(base, cache)
        check_output(plan, tmp_path / "output")
        assert not (tmp_path / "output").exists()
        assert requests == ["/model.json", "/model.json.lock"]
        assert {item["size"] for item in plan.summary()["files"]} == {4, 7}


def test_conflicting_installation_lock_fails_in_plan_and_execution(tmp_path):
    cache = ContentCache(tmp_path / "cache")
    with _server(_card(), {"code.loom": b"code", "weights.data": b"weights"}) as (
        base,
        requests,
        _,
    ):
        plan = _remote_plan(base, cache)
        output = tmp_path / "output"
        output.mkdir()
        (output / "model.json.lock").write_text(json.dumps(_lock({"other": b"other"})))
        for operation in (
            check_output,
            lambda plan, output: install(plan, cache, output),
        ):
            with pytest.raises(FetchError, match="conflicting installation document"):
                operation(plan, output)
        assert requests == ["/model.json", "/model.json.lock"]


def test_no_lock_neither_reads_nor_writes_any_lock(tmp_path):
    card_path = tmp_path / "card.json"
    card_path.write_text(json.dumps(_card()))
    source_lock = tmp_path / "card.json.lock"
    source_lock.write_text("deliberately invalid lock")
    (tmp_path / "code.loom").write_bytes(b"code")
    (tmp_path / "weights.data").write_bytes(b"weights")
    document = Document(str(card_path), card_path.read_text())
    cache = ContentCache(tmp_path / "cache")
    plan = plan_fetch(
        document,
        ModelCard.parse(document.text).select(),
        cache,
        no_lock=True,
        offline=True,
    )
    install(plan, cache, tmp_path / "output")
    assert source_lock.read_text() == "deliberately invalid lock"
    assert not (tmp_path / "output/model.json.lock").exists()
    assert os.path.samefile(tmp_path / "weights.data", tmp_path / "output/weights.data")


def test_update_replaces_exact_set_without_accumulating_records(tmp_path):
    card_path = tmp_path / "card.json"
    card_path.write_text(json.dumps(_card()))
    lock_path = tmp_path / "card.json.lock"
    lock_path.write_text(json.dumps(_lock({"old": b"old", "large.data": b"large"})))
    (tmp_path / "code.loom").write_bytes(b"code")
    (tmp_path / "weights.data").write_bytes(b"weights")
    cache = ContentCache(tmp_path / "cache")
    document = Document(str(card_path), card_path.read_text())
    plan = plan_fetch(
        document,
        ModelCard.parse(document.text).select(),
        cache,
        update_lock=True,
        offline=True,
    )
    install(plan, cache, tmp_path / "output")
    updated = json.loads(lock_path.read_text())
    assert {record["origin"] for record in updated["files"]} == {
        "code.loom",
        "weights.data",
    }
    assert updated["origins"] == {}


def test_override_retains_logical_hf_identity_in_installed_card(tmp_path):
    card = _card()
    card["origins"] = {
        "weights": {"kind": "hf", "repository": "org/gated", "revision": "a" * 40}
    }
    card["variants"]["v"]["services"]["text"]["parameters"]["scope"]["origin"] = (
        "@weights/"
    )
    (tmp_path / "code.loom").write_bytes(b"code")
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "weights.data").write_bytes(b"weights")
    document = Document(str(tmp_path / "card.json"), json.dumps(card))
    cache = ContentCache(tmp_path / "cache")
    plan = plan_fetch(
        document,
        ModelCard(card).select(),
        cache,
        overrides={"weights": str(mirror)},
        offline=True,
        update_lock=True,
    )
    result = install(plan, cache, tmp_path / "output")
    installed = ModelCard.parse(result.read_text()).select()
    assert installed.document["origins"]["weights"] == card["origins"]["weights"]
    assert installed.files["weights.data"].origin == "@weights/weights.data"
    lock = ModelLock.parse((result.parent / "model.json.lock").read_text())
    assert lock.origins["weights"]["resolved"] == {"revision": "a" * 40}
    lock.cover(installed)


def test_parent_symlink_cannot_redirect_installation(tmp_path):
    card = _card()
    card["programs"]["p"]["inputs"] = ["program/code.loom"]
    cache = ContentCache(tmp_path / "cache")
    with _server(card, {"program/code.loom": b"code", "weights.data": b"weights"}) as (
        base,
        requests,
        _,
    ):
        plan = _remote_plan(base, cache)
        output = tmp_path / "output"
        output.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (output / "program").symlink_to(outside, target_is_directory=True)
        with pytest.raises(FetchError, match="ordinary directory"):
            install(plan, cache, output)
        assert requests == ["/model.json", "/model.json.lock"]
        assert list(outside.iterdir()) == []


def test_copy_is_an_explicit_independent_materialization(tmp_path):
    card = _card()
    document = Document(str(tmp_path / "card.json"), json.dumps(card))
    (tmp_path / "code.loom").write_bytes(b"code")
    (tmp_path / "weights.data").write_bytes(b"weights")
    cache = ContentCache(tmp_path / "cache")
    plan = plan_fetch(document, ModelCard(card).select(), cache, no_lock=True)
    install(plan, cache, tmp_path / "output", materialize="copy")
    assert not os.path.samefile(
        tmp_path / "weights.data", tmp_path / "output/weights.data"
    )
    (tmp_path / "weights.data").write_bytes(b"changed")
    assert (tmp_path / "output/weights.data").read_bytes() == b"weights"


def test_shared_logical_source_is_downloaded_only_once(tmp_path):
    card = _card()
    card["programs"]["p"]["libraries"] = [{"path": "copy.loom", "origin": "code.loom"}]
    cache = ContentCache(tmp_path / "cache")
    with _server(card, {"code.loom": b"code", "weights.data": b"weights"}) as (
        base,
        requests,
        _,
    ):
        install(_remote_plan(base, cache), cache, tmp_path / "output")
        assert requests.count("/code.loom") == 1
        assert os.path.samefile(
            tmp_path / "output/code.loom", tmp_path / "output/copy.loom"
        )
