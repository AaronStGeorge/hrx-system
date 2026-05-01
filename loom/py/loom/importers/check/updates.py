# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Inline importer check update formatting."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from loom.importers.check.cases import (
    DEFAULT_CHECK_SYNTAX,
    CheckCase,
    InlineCheckSyntax,
    split_expected,
    split_raw_cases,
)
from loom.importers.check.results import CheckResult


def format_updated_source(
    original_source: str,
    cases: Sequence[CheckCase],
    results: Sequence[CheckResult],
    *,
    syntax: InlineCheckSyntax = DEFAULT_CHECK_SYNTAX,
    expected_encoder: Callable[[str], str] | None = None,
) -> str:
    encode_expected = expected_encoder or _identity_expected
    updated_cases: list[str] = []
    result_by_index = {result.case_index: result for result in results}
    for case in cases:
        result = result_by_index.get(case.index)
        if result is None:
            updated_cases.append(_raw_case_source(original_source, case.index, syntax))
            continue
        if result.returncode == 0:
            updated_cases.append(
                format_updated_case(
                    case.source,
                    result.stdout,
                    syntax=syntax,
                    expected_encoder=encode_expected,
                )
            )
        else:
            updated_cases.append(_raw_case_source(original_source, case.index, syntax))
    return f"\n{syntax.case_separator_prefix}\n".join(updated_cases)


def format_updated_case(
    source: str,
    actual: str,
    *,
    syntax: InlineCheckSyntax = DEFAULT_CHECK_SYNTAX,
    expected_encoder: Callable[[str], str] | None = None,
) -> str:
    encode_expected = expected_encoder or _identity_expected
    case_source, _expected, _has_expected = split_expected(source, syntax=syntax)
    return (
        ensure_trailing_newline(case_source)
        + f"{syntax.expected_separator}\n"
        + ensure_trailing_newline(encode_expected(actual))
    )


def ensure_trailing_newline(text: str) -> str:
    return text if not text or text.endswith("\n") else f"{text}\n"


def _identity_expected(text: str) -> str:
    return text


def _raw_case_source(
    source: str,
    case_index: int,
    syntax: InlineCheckSyntax,
) -> str:
    return split_raw_cases(source, syntax=syntax)[case_index]
