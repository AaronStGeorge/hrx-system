# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Inline importer check case parsing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class InlineCheckSyntax:
    """Comment syntax and separators for one inline check file format."""

    case_separator_prefix: str
    expected_separator: str
    run_prefix: str | None = None
    comment_prefix: str | None = None


DEFAULT_CHECK_SYNTAX = InlineCheckSyntax(
    case_separator_prefix="// ====",
    expected_separator="// ----",
    run_prefix="// RUN: ",
    comment_prefix="//",
)


@dataclass(frozen=True, slots=True)
class CheckCase:
    """One parsed inline importer check case."""

    path: Path
    index: int
    source: str
    input: str
    expected: str
    run: str | None = None
    line_start: int = field(default=1, compare=False)
    line_end: int = field(default=1, compare=False)
    raw_source: str = field(default="", compare=False)


def parse_inline_cases(
    path: Path,
    source: str,
    *,
    syntax: InlineCheckSyntax = DEFAULT_CHECK_SYNTAX,
    default_run: str | None = None,
    allow_preamble: bool = False,
) -> tuple[CheckCase, ...]:
    """Parses loom-check style inline cases from one source file."""

    raw_cases = tuple(_split_raw_case_spans(source, syntax=syntax))
    if allow_preamble:
        raw_cases = _strip_preamble_case(raw_cases, syntax=syntax)
    if not raw_cases:
        raise ValueError(f"{path} has no check cases")

    runs = [_run_directive(raw.source, syntax=syntax) for raw in raw_cases]
    file_run = runs[0] or default_run
    cases: list[CheckCase] = []
    for index, raw_case in enumerate(raw_cases):
        case_source, expected, has_expected = split_expected(
            raw_case.source,
            syntax=syntax,
        )
        input_text = strip_case_directives(case_source, syntax=syntax)
        cases.append(
            CheckCase(
                path=path,
                index=index,
                source=case_source,
                input=input_text,
                expected=expected if has_expected else input_text,
                run=runs[index] or file_run,
                line_start=raw_case.line_start,
                line_end=raw_case.line_end,
                raw_source=raw_case.source,
            )
        )
    return tuple(cases)


def split_raw_cases(
    source: str,
    *,
    syntax: InlineCheckSyntax = DEFAULT_CHECK_SYNTAX,
) -> tuple[str, ...]:
    """Splits one inline check source into raw case sources."""

    return tuple(raw.source for raw in _split_raw_case_spans(source, syntax=syntax))


@dataclass(frozen=True, slots=True)
class _RawCheckCase:
    source: str
    line_start: int
    line_end: int
    is_preamble: bool = False


def _strip_preamble_case(
    raw_cases: tuple[_RawCheckCase, ...],
    *,
    syntax: InlineCheckSyntax,
) -> tuple[_RawCheckCase, ...]:
    """Drops a non-case prefix before the first explicit case separator."""

    if len(raw_cases) < 2 or not raw_cases[0].is_preamble:
        return raw_cases
    _case_source, _expected, has_expected = split_expected(
        raw_cases[0].source,
        syntax=syntax,
    )
    if has_expected:
        return raw_cases
    return raw_cases[1:]


def _split_raw_case_spans(
    source: str,
    *,
    syntax: InlineCheckSyntax,
) -> tuple[_RawCheckCase, ...]:
    cases: list[_RawCheckCase] = []
    lines: list[str] = []
    line_start = 1
    line_number = 1
    before_first_separator = True
    for line in source.splitlines(keepends=True):
        if line.startswith(syntax.case_separator_prefix):
            trim_separator_spacer(lines)
            case_source = "".join(lines)
            if case_source.strip():
                cases.append(
                    _RawCheckCase(
                        source=case_source,
                        line_start=line_start,
                        line_end=max(line_start, line_number - 1),
                        is_preamble=before_first_separator,
                    )
                )
            lines = []
            line_start = line_number + 1
            line_number += 1
            before_first_separator = False
            continue
        lines.append(line)
        line_number += 1
    trim_separator_spacer(lines)
    case_source = "".join(lines)
    if case_source.strip():
        cases.append(
            _RawCheckCase(
                source=case_source,
                line_start=line_start,
                line_end=max(line_start, line_number - 1),
                is_preamble=before_first_separator,
            )
        )
    return tuple(cases)


def trim_separator_spacer(lines: list[str]) -> None:
    """Drops formatting-only blank lines before a case separator."""

    while lines and not lines[-1].strip():
        lines.pop()


def split_expected(
    source: str,
    *,
    syntax: InlineCheckSyntax = DEFAULT_CHECK_SYNTAX,
) -> tuple[str, str, bool]:
    """Splits a raw case into input and expected-output sections."""

    input_lines: list[str] = []
    expected_lines: list[str] = []
    in_expected = False
    for line in source.splitlines(keepends=True):
        if line.strip() == syntax.expected_separator:
            in_expected = True
            continue
        if in_expected:
            expected_lines.append(line)
        else:
            input_lines.append(line)
    return "".join(input_lines), "".join(expected_lines), in_expected


def strip_case_directives(
    source: str,
    *,
    syntax: InlineCheckSyntax = DEFAULT_CHECK_SYNTAX,
) -> str:
    """Removes check-runner directives from source input passed to an importer."""

    if syntax.run_prefix is None:
        return source
    lines: list[str] = []
    for line in source.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith(syntax.run_prefix):
            continue
        lines.append(line)
    return "".join(lines)


def case_matches_filter(
    check_case: CheckCase,
    case_filter: str | None,
    *,
    labels: Sequence[str] = (),
) -> bool:
    """Returns whether a parsed case should run under a substring filter."""

    if not case_filter:
        return True
    candidates = [
        str(check_case.path),
        check_case.path.name,
        f"{check_case.path}:case{check_case.index}",
        f"case{check_case.index}",
    ]
    if check_case.run:
        candidates.append(check_case.run)
    candidates.extend(label for label in labels if label)
    return any(case_filter in candidate for candidate in candidates)


def _run_directive(
    source: str,
    *,
    syntax: InlineCheckSyntax,
) -> str | None:
    if syntax.run_prefix is None:
        return None
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(syntax.run_prefix):
            return stripped[len(syntax.run_prefix) :].strip()
        if syntax.comment_prefix is not None and stripped.startswith(
            syntax.comment_prefix
        ):
            continue
        return None
    return None
