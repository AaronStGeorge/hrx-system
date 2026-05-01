# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

from loom.importers.check.results import CheckResult, summarize_results


def test_summarize_results_classifies_failures_crashes_and_skips() -> None:
    results = [
        CheckResult(Path("ok.mlir"), 0, 0, "", ""),
        CheckResult(Path("updated.mlir"), 1, 0, "", "", updated=True),
        CheckResult(Path("skipped.mlir"), -1, 0, "", "", skipped=True),
        CheckResult(Path("bad.mlir"), 2, 1, "", ""),
        CheckResult(Path("mismatch.mlir"), 3, 0, "", "", mismatch="different"),
        CheckResult(Path("segv.mlir"), 4, -11, "", ""),
    ]

    assert summarize_results(results) == {
        "passed": 1,
        "updated": 1,
        "skipped": 1,
        "failed": 2,
        "crashed": 1,
    }
