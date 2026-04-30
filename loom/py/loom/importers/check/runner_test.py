# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

from loom.importers.check.runner import (
    CheckResult,
    MlirCheckOptions,
    build_mlir_command,
    summarize_results,
)


def test_build_mlir_command_includes_optional_flags() -> None:
    command = build_mlir_command(
        Path("kernel.mlir"),
        options=MlirCheckOptions(
            kernel="kernel",
            prefer_abi3_extensions=True,
        ),
    )

    assert command[-3:] == [
        "--kernel",
        "kernel",
        "--prefer-abi3-extensions",
    ]


def test_summarize_results_classifies_failures_and_crashes() -> None:
    results = [
        CheckResult(Path("ok.mlir"), 0, "", ""),
        CheckResult(Path("bad.mlir"), 1, "", ""),
        CheckResult(Path("segv.mlir"), -11, "", ""),
    ]

    assert summarize_results(results) == {
        "passed": 1,
        "failed": 1,
        "crashed": 1,
    }
