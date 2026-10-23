# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Checks a built model card through the native server and weighted execution."""

import json
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from loom_serve_testing.http_harness import control, running_server


def check_model(server, model, expected, *provider_encodings, environment=None):
    server = str(Path(server).absolute())
    # Preserve the runfiles namespace: resolving its symlinks would discard the
    # declared relative relationship between the deployment and its assets.
    model = str(Path(model).absolute())
    document = json.loads(Path(model).read_text())
    model_name = (
        document["model"]["id"] if "schema_version" in document else document["name"]
    )
    for binding in provider_encodings:
        relative, encoding = binding.split("=", 1)
        assert encoding in ("text", "bc"), binding
        with (Path(model).parent / relative).open("rb") as stream:
            assert (stream.read(4) == b"LOOM") == (encoding == "bc"), binding
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        log = directory / "server.log"
        with running_server(
            [
                server,
                "--device=task",
                "--workers=4",
                "--pool_backing=fixed",
                "--listen=127.0.0.1:0",
                "--model=" + model,
            ],
            log,
            cwd=directory,
            environment=environment,
        ) as (process, bound_address):
            assert bound_address is not None, log.read_text()
            address = urlsplit("http://" + bound_address)
            try:
                response = control(
                    address,
                    "POST",
                    "/v1/chat/completions",
                    200,
                    headers={"Content-Type": "application/json"},
                    body=json.dumps(
                        {
                            "model": model_name,
                            "messages": [{"role": "user", "content": "a"}],
                            "max_tokens": 4,
                        }
                    ),
                )
            except RuntimeError as error:
                status = control(address, "GET", "/v1/status", 200)
                raise RuntimeError(
                    f"{error}\nmodel status: {json.dumps(status)}"
                ) from error
            # Independent dense oracle: checkpoint bias 2, plus the selected JIT delta.
            assert response["choices"][0]["message"]["content"] == expected, response
            assert response["usage"]["prompt_tokens"] == 1, response
            assert response["usage"]["completion_tokens"] == 4, response
        assert process.returncode == 0, log.read_text()


if __name__ == "__main__":
    check_model(*sys.argv[1:])
