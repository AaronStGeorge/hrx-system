# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

from loom.importers.check.cases import (
    CheckCase,
    InlineCheckSyntax,
    parse_inline_cases,
    split_expected,
    split_raw_cases,
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


def test_parse_inline_cases_uses_configured_comment_syntax() -> None:
    syntax = InlineCheckSyntax(
        case_separator_prefix="# ====",
        expected_separator="# ----",
        comment_prefix="#",
    )
    cases = parse_inline_cases(
        Path("kernels.py"),
        "case_0()\n# ----\nexpected 0\n\n# ====\ncase_1()\n# ----\nexpected 1\n",
        syntax=syntax,
    )

    assert len(cases) == 2
    assert cases[0] == CheckCase(
        path=Path("kernels.py"),
        index=0,
        source="case_0()\n",
        input="case_0()\n",
        expected="expected 0\n",
    )
    assert cases[1].input == "case_1()\n"
