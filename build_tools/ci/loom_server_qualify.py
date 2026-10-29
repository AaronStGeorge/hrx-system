#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercise a standalone loom-serve executable on its destination OS."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from loom_serve_testing.http_harness import control, running_server

_LOADER_ENVIRONMENT = {
    "DYLD_FALLBACK_LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "PYTHONPATH",
}


def _run(binary: Path, argument: str, environment: dict[str, str]) -> str:
    result = subprocess.run(
        [os.fspath(binary), argument],
        cwd=binary.parent,
        env=environment,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.stderr:
        raise ValueError(f"{argument} wrote unexpected diagnostics: {result.stderr!r}")
    return result.stdout


def qualify(binary: Path, version: str, commit: str) -> None:
    binary = binary.resolve(strict=True)
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in _LOADER_ENVIRONMENT
    }
    identity = f"loom-serve {version} (commit {commit})"
    actual = _run(binary, "--version", environment)
    if actual != identity + "\n":
        raise ValueError(f"Unexpected --version output: {actual!r}")
    help_text = _run(binary, "--help", environment)
    if help_text.count(identity + "\n\n") != 1:
        raise ValueError("--help does not contain exactly one stamped build identity")
    for flag in ("--device", "--listen", "--model"):
        if flag not in help_text:
            raise ValueError(f"--help does not document {flag}")

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        log = directory / "server.log"
        command = [
            os.fspath(binary),
            "--device=task",
            "--pool_backing=fixed",
            "--weight_slab_bytes=64mib",
            "--state_slab_bytes=16mib",
            "--memory_bytes=0b",
            "--workers=1",
            "--listen=127.0.0.1:0",
        ]
        with running_server(command, log, cwd=directory, environment=environment) as (
            process,
            bound_address,
        ):
            if bound_address is None:
                raise RuntimeError(f"server did not become ready:\n{log.read_text()}")
            address = urlsplit("http://" + bound_address)
            if control(address, "GET", "/readyz", 200) != {"status": "ready"}:
                raise ValueError(
                    "standalone server returned an invalid readiness result"
                )
            status = control(address, "GET", "/v1/status", 200)
            if status.get("data") != [] or "server" not in status:
                raise ValueError("standalone server returned an invalid empty status")
        if process.returncode != 0:
            raise RuntimeError(
                f"server exited with {process.returncode}:\n{log.read_text()}"
            )


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 3:
        print("usage: loom_server_qualify.py BINARY VERSION COMMIT", file=sys.stderr)
        return 2
    try:
        qualify(Path(arguments[0]), arguments[1], arguments[2])
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"standalone loom-serve qualification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
