# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Installs the Python SDK and serves a model acquired by its console command."""

import functools
import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from model_card_test import check_model


def main(wheel, server, deployment):
    wheel = Path(wheel).absolute()
    server = Path(server).absolute()
    deployment = Path(deployment).absolute()
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("PYTHON")
    }
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)

        def run(*arguments):
            result = subprocess.run(
                [str(argument) for argument in arguments],
                cwd=directory,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            assert result.returncode == 0, (result.stdout, result.stderr)
            return result.stdout

        run(sys.executable, "-I", "-m", "venv", directory / "venv")
        scripts = directory / "venv" / ("Scripts" if os.name == "nt" else "bin")
        python = scripts / ("python.exe" if os.name == "nt" else "python")
        fetch = scripts / ("loom-fetch.exe" if os.name == "nt" else "loom-fetch")
        run(
            python,
            "-I",
            "-m",
            "pip",
            "--disable-pip-version-check",
            "install",
            "--no-index",
            "--no-deps",
            "--no-compile",
            wheel,
        )
        run(
            python,
            "-I",
            "-c",
            """
import importlib.metadata
import importlib.util
import json
import sys
from importlib.resources import files
from pathlib import Path
from loom import LoomBuilder, default_ops, default_types
from loom.format.bytecode.reader import read_module
from loom.format.bytecode.writer import write_module
from loom.format.text.parser import Parser
from loom.model import ModelCard

assert importlib.util.find_spec('huggingface_hub') is None
distribution = importlib.metadata.distribution('hrx-loom')
assert distribution.metadata.get_all('Provides-Extra') == ['hf']
assert distribution.requires == ["huggingface_hub<3,>=2.2; extra == 'hf'"], distribution.requires
default_ops()
default_types()
assert all(Path(module.__file__).is_relative_to(sys.prefix)
           for name, module in list(sys.modules.items())
           if name == 'loom' or name.startswith('loom.'))
for name in ('model-card', 'model-lock'):
    schema = json.loads(files('loom.model').joinpath(name + '.v1.schema.json').read_text())
    assert schema['$schema'] == 'https://json-schema.org/draft/2020-12/schema'
""",
        )
        assert "--offline" in run(fetch, "--help")
        card = deployment.parent / "model.json"
        listing = json.loads(run(fetch, card, "--list", "--json"))
        assert listing["model"]["id"] == "loom-fixture/text"
        installed = directory / "installed"
        handler = functools.partial(
            http.server.SimpleHTTPRequestHandler, directory=str(card.parent)
        )
        with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as origin:
            thread = threading.Thread(target=origin.serve_forever)
            thread.start()
            try:
                result = json.loads(
                    run(
                        fetch,
                        f"http://127.0.0.1:{origin.server_port}/model.json",
                        "--configuration=check",
                        "--update-lock",
                        "--no-input",
                        "--json",
                        "--cache",
                        directory / "cache",
                        "--output",
                        installed,
                    )
                )
            finally:
                origin.shutdown()
                thread.join()
        assert (installed / "model.json.lock").is_file()
        run(
            fetch,
            installed / "model.json",
            "--configuration=check",
            "--offline",
            "--no-input",
            "--cache",
            directory / "cache",
            "--output",
            directory / "offline",
        )
        assert Path(result["card"]) == installed / "model.json"
        serving = json.loads(deployment.read_text())
        serving["model"]["path"] = str(installed / "model.json")
        selected = directory / "serve.json"
        selected.write_text(json.dumps(serving))
        check_model(str(server), str(selected), "abab")


if __name__ == "__main__":
    main(*sys.argv[1:])
