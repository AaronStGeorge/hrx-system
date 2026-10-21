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


def main():
    server = str(Path(sys.argv[1]).absolute())
    # Preserve the runfiles namespace: resolving its symlinks would discard the
    # declared relative relationship between the deployment and its assets.
    model = str(Path(sys.argv[2]).absolute() / "serve.json")
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
        ) as (process, bound_address):
            assert bound_address is not None, log.read_text()
            address = urlsplit("http://" + bound_address)
            response = control(
                address,
                "POST",
                "/v1/chat/completions",
                200,
                headers={"Content-Type": "application/json"},
                body=json.dumps(
                    {
                        "model": "fixture-text",
                        "messages": [{"role": "user", "content": "a"}],
                        "max_tokens": 4,
                    }
                ),
            )
            # The text fixture's independent dense oracle with checkpoint bias 2.
            assert response["choices"][0]["message"]["content"] == "bbaa", response
            assert response["usage"]["prompt_tokens"] == 1, response
            assert response["usage"]["completion_tokens"] == 4, response
        assert process.returncode == 0, log.read_text()


if __name__ == "__main__":
    main()
