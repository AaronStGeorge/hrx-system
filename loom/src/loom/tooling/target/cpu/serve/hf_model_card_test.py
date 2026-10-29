# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Checks SDK acquisition through native execution."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from loom.model import ModelCard
from loom.model.fetch_hf import HuggingFace
from loom.model.hf_test.hub import running_hub
from loom.model.lock import FileIdentity, ModelLock
from model_card_test import check_model


def main():
    serve, fixture = (Path(value).absolute() for value in sys.argv[1:])
    card = ModelCard.parse(fixture.read_text(encoding="utf-8"))
    # Program selection and representation fanout have dedicated fetch and
    # native execution tests. Mixed is the hardest transport-to-runtime seam.
    files = {
        name: (fixture.parent / name).read_bytes()
        for name in card.select(preset="check", programs="mixed").files
    }
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
        with running_hub(root, files) as hub:
            hub.credentials.write_text("fixture-first")
            hub.approved = True
            provider = HuggingFace(
                endpoint=hub.endpoint,
                cache_dir=hub.cache_dir,
            )
            url = hub.endpoint + "/fixture/model/blob/main/model.json"
            online = root / "online"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "loom.model.fetch_main",
                    url,
                    "--cache=" + str(root / "cache"),
                    "--preset=check",
                    "--programs=mixed",
                    "--output=" + str(online),
                    "--no-input",
                    "--json",
                ],
                env=hub.environment(),
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stderr
            assert json.loads(result.stdout)["card"] == str(online / "model.json")
            cached_bias = provider.cached_file(hub.source("bias.data"))
            assert os.path.samefile(online / "bias.data", cached_bias)
            assert not (root / "cache/blob").exists()

        # The provider is gone: serving owns only the acquired ordinary files.
        deployment = online / "serve.json"
        deployment.write_text(
            json.dumps(
                {
                    "name": "fixture-text",
                    "model": {"path": "model.json", "preset": "check"},
                    "definition": {"kind": "text"},
                }
            ),
            encoding="utf-8",
        )
        check_model(serve, deployment, "abab")


if __name__ == "__main__":
    main()
