# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runs importer checks in child processes.

MLIR, Triton, and TileLang all bring optional native Python extensions into the
process. A failing import should be an ordinary check result, not a lost parent
process, so this runner keeps each source file behind a subprocess boundary.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One importer invocation result."""

    path: Path
    returncode: int
    stdout: str
    stderr: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0

    @property
    def status(self) -> str:
        if self.returncode == 0:
            return "passed"
        if self.returncode < 0:
            return "crashed"
        return "failed"

    def as_json_object(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "returncode": self.returncode,
            "status": self.status,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass(frozen=True, slots=True)
class MlirCheckOptions:
    """Options for `python -m loom.importers.mlir` checks."""

    kernel: str | None = None
    prefer_abi3_extensions: bool = False


def run_mlir_check(
    path: Path,
    *,
    options: MlirCheckOptions,
    env: Mapping[str, str] | None = None,
) -> CheckResult:
    command = build_mlir_command(path, options=options)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        env=dict(env if env is not None else os.environ),
        text=True,
    )
    return CheckResult(
        path=path,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def build_mlir_command(path: Path, *, options: MlirCheckOptions) -> list[str]:
    command = [sys.executable, "-m", "loom.importers.mlir", str(path)]
    if options.kernel is not None:
        command.extend(["--kernel", options.kernel])
    if options.prefer_abi3_extensions:
        command.append("--prefer-abi3-extensions")
    return command


def results_to_json(results: Sequence[CheckResult]) -> str:
    return json.dumps(
        {
            "results": [result.as_json_object() for result in results],
            "summary": summarize_results(results),
        },
        indent=2,
        sort_keys=True,
    )


def summarize_results(results: Sequence[CheckResult]) -> dict[str, int]:
    summary = {
        "passed": 0,
        "failed": 0,
        "crashed": 0,
    }
    for result in results:
        summary[result.status] += 1
    return summary
