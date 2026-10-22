# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualifies gated SDK acquisition through native source/archive/mixed execution."""

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from loom.model import ModelCard
from loom.model.fetch import install, plan_fetch
from loom.model.fetch_hf import HuggingFace
from loom.model.fetch_storage import ContentCache
from loom.model.hf_test.hub import running_hub
from loom.model.lock import FileIdentity, ModelLock
from model_card_test import check_model


def main():
    serve, fixture = (Path(value).absolute() for value in sys.argv[1:])
    card = ModelCard.parse(fixture.read_text(encoding="utf-8"))
    files = {}
    for programs in ("source", "archive", "mixed"):
        for name in card.select(configuration="check", programs=programs).files:
            files[name] = (fixture.parent / name).read_bytes()
    identities = [
        FileIdentity(name, len(content), hashlib.sha256(content).hexdigest())
        for name, content in files.items()
    ]
    publication = card.document
    publication.pop("$schema", None)
    files["model.json"] = json.dumps(publication).encode()
    files["model.json.lock"] = json.dumps(ModelLock.snapshot({}, identities)).encode()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        outputs = []
        with running_hub(root, files) as hub:
            hub.credentials.write_text("fixture-first")
            actions = []

            def wait(action, delay, cancelled):
                assert action.reason == "gate"
                assert action.account == "first"
                actions.append(action)
                hub.approved = True

            provider = HuggingFace(
                endpoint=hub.endpoint,
                cache_dir=hub.cache_dir,
                wait_for_access=wait,
            )
            cache = ContentCache(root / "cache", huggingface=provider)
            url = hub.endpoint + "/fixture/model/blob/main/model.json"
            document = cache.read_document(url, offline=False)
            for programs in ("source", "archive", "mixed"):
                selection = ModelCard.parse(document.text).select(
                    configuration="check", programs=programs
                )
                plan = plan_fetch(document, selection, cache)
                output = root / programs
                install(plan, cache, output)
                outputs.append(output)
                assert os.path.samefile(
                    output / "bias.data", provider.cached_file(hub.source("bias.data"))
                )
            assert len(actions) == 1
            assert not (cache.root / "blob").exists()
            requests = list(hub.requests)
            hub.approved = False
            offline = cache.read_document(url, offline=True)
            plan = plan_fetch(
                offline,
                ModelCard.parse(offline.text).select(configuration="check"),
                cache,
                offline=True,
            )
            output = root / "offline"
            install(plan, cache, output)
            outputs.append(output)
            assert hub.requests == requests

        # The provider is gone: serving owns only selected ordinary local files.
        for output in outputs:
            deployment = output / "serve.json"
            deployment.write_text(
                json.dumps(
                    {
                        "name": "fixture-text",
                        "model": {"path": "model.json", "configuration": "check"},
                        "definition": {"kind": "text"},
                    }
                ),
                encoding="utf-8",
            )
            check_model(serve, deployment, "abab")


if __name__ == "__main__":
    main()
