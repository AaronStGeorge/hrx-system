# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Importer check result reporting."""

from __future__ import annotations

import difflib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One importer check case result."""

    path: Path
    case_index: int
    returncode: int
    stdout: str
    stderr: str
    mismatch: str | None = None
    diff: str | None = None
    updated: bool = False
    skipped: bool = False

    @property
    def passed(self) -> bool:
        return self.skipped or (self.returncode == 0 and self.mismatch is None)

    @property
    def status(self) -> str:
        if self.skipped:
            return "skipped"
        if self.updated:
            return "updated"
        if self.returncode == 0:
            return "passed" if self.mismatch is None else "failed"
        if self.returncode < 0:
            return "crashed"
        return "failed"

    def as_json_object(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "case_index": self.case_index,
            "returncode": self.returncode,
            "status": self.status,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "mismatch": self.mismatch,
            "diff": self.diff,
            "updated": self.updated,
            "skipped": self.skipped,
        }


def unified_diff(
    expected: str,
    actual: str,
    *,
    fromfile: str,
    tofile: str,
) -> str:
    return "".join(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
        )
    )


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
        "updated": 0,
        "skipped": 0,
        "failed": 0,
        "crashed": 0,
    }
    for result in results:
        summary[result.status] += 1
    return summary
