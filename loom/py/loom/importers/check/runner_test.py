# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

from loom.importers.check.runner import (
    CheckResult,
    format_updated_case,
    format_updated_source,
    parse_check_cases,
    parse_mlir_run,
    split_expected,
    split_raw_cases,
    summarize_results,
)


def test_split_raw_cases_uses_loom_check_case_separator() -> None:
    source = "case 0\n// ====\ncase 1\n"

    assert split_raw_cases(source) == ("case 0\n", "case 1\n")


def test_split_raw_cases_ignores_separator_spacer() -> None:
    source = "case 0\n\n// ====\ncase 1\n"

    assert split_raw_cases(source) == ("case 0\n", "case 1\n")


def test_split_expected_uses_loom_check_expected_separator() -> None:
    source, expected, has_expected = split_expected("input\n// ----\nexpected\n")

    assert source == "input\n"
    assert expected == "expected\n"
    assert has_expected


def test_parse_check_cases_inherits_first_run_directive() -> None:
    cases = parse_check_cases(
        Path("kernels.mlir"),
        """
// RUN: mlir --kernel first
module @first {}
// ----
first expected
// ====
module @second {}
// ----
second expected
""",
    )

    assert len(cases) == 2
    assert cases[0].run == "mlir --kernel first"
    assert cases[1].run == "mlir --kernel first"
    assert "// RUN:" not in cases[0].input
    assert cases[0].expected == "first expected\n"


def test_parse_check_cases_defaults_to_mlir_without_run_directives() -> None:
    cases = parse_check_cases(
        Path("kernels.mlir"),
        """
hal.executable @first {}
// ----
first expected
// ====
hal.executable @second {}
// ----
second expected
""",
    )

    assert cases[0].run == "mlir"
    assert cases[1].run == "mlir"


def test_parse_check_cases_keeps_case_local_run_directive() -> None:
    cases = parse_check_cases(
        Path("kernels.mlir"),
        """
// RUN: mlir --kernel first
module @first {}
// ----
first expected
// ====
// RUN: mlir --kernel second
module @second {}
// ----
second expected
""",
    )

    assert cases[1].run == "mlir --kernel second"


def test_parse_mlir_run_accepts_kernel_option() -> None:
    assert parse_mlir_run("mlir --kernel kernel").kernel == "kernel"


def test_format_updated_case_replaces_expected_section() -> None:
    assert (
        format_updated_case("input\n// ----\nold\n", "new\n") == "input\n// ----\nnew\n"
    )


def test_format_updated_source_puts_blank_line_before_case_separator() -> None:
    cases = parse_check_cases(
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


def test_summarize_results_classifies_failures_and_crashes() -> None:
    results = [
        CheckResult(Path("ok.mlir"), 0, 0, "", ""),
        CheckResult(Path("updated.mlir"), 1, 0, "", "", updated=True),
        CheckResult(Path("bad.mlir"), 2, 1, "", ""),
        CheckResult(Path("mismatch.mlir"), 3, 0, "", "", mismatch="different"),
        CheckResult(Path("segv.mlir"), 4, -11, "", ""),
    ]

    assert summarize_results(results) == {
        "passed": 1,
        "updated": 1,
        "failed": 2,
        "crashed": 1,
    }
