# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

from loom.importers.check.cases import parse_inline_cases
from loom.importers.check.results import CheckResult
from loom.importers.check.updates import format_updated_case, format_updated_source


def test_format_updated_case_replaces_expected_section() -> None:
    assert (
        format_updated_case("input\n// ----\nold\n", "new\n") == "input\n// ----\nnew\n"
    )


def test_format_updated_source_puts_blank_line_before_case_separator() -> None:
    cases = parse_inline_cases(
        Path("kernels.mlir"),
        "input\n// ----\nold\n// ====\ninput2\n// ----\nold2\n",
    )
    results = [
        CheckResult(Path("kernels.mlir"), 0, 0, "new\n", ""),
        CheckResult(Path("kernels.mlir"), 1, 0, "new2\n", ""),
    ]

    assert (
        format_updated_source("", cases, results)
        == "input\n// ----\nnew\n\n// ====\ninput2\n// ----\nnew2\n"
    )


def test_format_updated_source_keeps_unselected_cases() -> None:
    source = "input\n// ----\nold\n\n// ====\ninput2\n// ----\nold2\n"
    cases = parse_inline_cases(
        Path("kernels.mlir"),
        source,
    )
    results = [CheckResult(Path("kernels.mlir"), 1, 0, "new2\n", "")]

    assert (
        format_updated_source(source, cases, results)
        == "input\n// ----\nold\n\n// ====\ninput2\n// ----\nnew2\n"
    )
