# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runs importer checks using the same inline case format as loom-check."""

from __future__ import annotations

import difflib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

CASE_SEPARATOR_PREFIX = "// ===="
EXPECTED_SEPARATOR = "// ----"
RUN_PREFIX = "// RUN: "


@dataclass(frozen=True, slots=True)
class CheckCase:
    """One parsed inline importer check case."""

    path: Path
    index: int
    source: str
    input: str
    expected: str
    run: str


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

    @property
    def passed(self) -> bool:
        return self.returncode == 0 and self.mismatch is None

    @property
    def status(self) -> str:
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
        }


@dataclass(frozen=True, slots=True)
class MlirCheckOptions:
    """Options for inline MLIR importer checks."""

    kernel: str | None = None
    prefer_abi3_extensions: bool = False
    update: bool = False


def run_mlir_check(
    path: Path,
    *,
    options: MlirCheckOptions,
) -> tuple[CheckResult, ...]:
    source = path.read_text()
    cases = parse_check_cases(path, source, default_kernel=options.kernel)
    results = tuple(import_mlir_case(case, options=options) for case in cases)
    if options.update:
        updated_source = format_updated_source(source, cases, results)
        if updated_source != source:
            path.write_text(updated_source)
    return results


def parse_check_cases(
    path: Path,
    source: str,
    *,
    default_kernel: str | None = None,
) -> tuple[CheckCase, ...]:
    raw_cases = tuple(split_raw_cases(source))
    if not raw_cases:
        raise ValueError(f"{path} has no check cases")

    runs = [_run_directive(case_source) for case_source in raw_cases]
    file_run = runs[0] or "mlir"
    cases: list[CheckCase] = []
    for index, raw_case in enumerate(raw_cases):
        case_source, expected, has_expected = split_expected(raw_case)
        run = runs[index] or file_run
        run_options = parse_mlir_run(run)
        input_text = strip_case_directives(case_source)
        expected_text = expected if has_expected else input_text
        kernel = run_options.kernel or default_kernel
        effective_run = "mlir" if kernel is None else f"mlir --kernel {kernel}"
        cases.append(
            CheckCase(
                path=path,
                index=index,
                source=case_source,
                input=input_text,
                expected=expected_text,
                run=effective_run,
            )
        )
    return tuple(cases)


def split_raw_cases(source: str) -> tuple[str, ...]:
    cases: list[list[str]] = [[]]
    for line in source.splitlines(keepends=True):
        if line.startswith(CASE_SEPARATOR_PREFIX):
            trim_separator_spacer(cases[-1])
            if "".join(cases[-1]).strip():
                cases.append([])
            continue
        cases[-1].append(line)
    return tuple(case for lines in cases if (case := "".join(lines)).strip())


def trim_separator_spacer(lines: list[str]) -> None:
    if lines and not lines[-1].strip():
        lines.pop()


def split_expected(source: str) -> tuple[str, str, bool]:
    input_lines: list[str] = []
    expected_lines: list[str] = []
    in_expected = False
    for line in source.splitlines(keepends=True):
        if line.strip() == EXPECTED_SEPARATOR:
            in_expected = True
            continue
        if in_expected:
            expected_lines.append(line)
        else:
            input_lines.append(line)
    return "".join(input_lines), "".join(expected_lines), in_expected


@dataclass(frozen=True, slots=True)
class MlirRunOptions:
    """Per-case options parsed from `// RUN: mlir ...`."""

    kernel: str | None = None


def parse_mlir_run(run: str) -> MlirRunOptions:
    pieces = run.split()
    if not pieces:
        return MlirRunOptions()
    if pieces[0] != "mlir":
        raise ValueError(f"unsupported importer check RUN mode `{pieces[0]}`")
    kernel: str | None = None
    index = 1
    while index < len(pieces):
        piece = pieces[index]
        if piece == "--kernel":
            index += 1
            if index == len(pieces):
                raise ValueError("RUN: mlir --kernel requires a value")
            kernel = pieces[index]
        else:
            raise ValueError(f"unsupported RUN: mlir option `{piece}`")
        index += 1
    return MlirRunOptions(kernel=kernel)


def _run_directive(source: str) -> str | None:
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(RUN_PREFIX):
            return stripped[len(RUN_PREFIX) :].strip()
        if stripped.startswith("//"):
            continue
        return None
    return None


def strip_case_directives(source: str) -> str:
    lines: list[str] = []
    for line in source.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith(RUN_PREFIX):
            continue
        lines.append(line)
    return "".join(lines)


def import_mlir_case(
    case: CheckCase,
    *,
    options: MlirCheckOptions,
) -> CheckResult:
    from loom.importers.core import LoomImportError, print_loom_module
    from loom.importers.mlir.importer import (
        MlirImportOptions,
        import_mlir_module,
    )

    run_options = parse_mlir_run(case.run)
    try:
        result = import_mlir_module(
            case.input,
            options=MlirImportOptions(
                kernel=run_options.kernel,
                prefer_abi3_extensions=options.prefer_abi3_extensions,
            ),
        )
        stdout = print_loom_module(result.module)
    except LoomImportError as exc:
        return CheckResult(
            path=case.path,
            case_index=case.index,
            returncode=1,
            stdout="",
            stderr=f"{exc}\n",
        )
    except Exception as exc:
        return CheckResult(
            path=case.path,
            case_index=case.index,
            returncode=1,
            stdout="",
            stderr=f"{type(exc).__name__}: {exc}\n",
        )

    if stdout == case.expected:
        return CheckResult(
            path=case.path,
            case_index=case.index,
            returncode=0,
            stdout=stdout,
            stderr="",
        )
    if options.update:
        return CheckResult(
            path=case.path,
            case_index=case.index,
            returncode=0,
            stdout=stdout,
            stderr="",
            updated=True,
        )
    return CheckResult(
        path=case.path,
        case_index=case.index,
        returncode=0,
        stdout=stdout,
        stderr="",
        mismatch="output differs from expected output",
        diff=unified_diff(
            case.expected,
            stdout,
            fromfile=f"{case.path}:case{case.index}:expected",
            tofile=f"{case.path}:case{case.index}:actual",
        ),
    )


def format_updated_source(
    original_source: str,
    cases: Sequence[CheckCase],
    results: Sequence[CheckResult],
) -> str:
    updated_cases: list[str] = []
    result_by_index = {result.case_index: result for result in results}
    for case in cases:
        result = result_by_index[case.index]
        if result.returncode == 0:
            updated_cases.append(format_updated_case(case.source, result.stdout))
        else:
            updated_cases.append(_raw_case_source(original_source, case.index))
    return f"\n{CASE_SEPARATOR_PREFIX}\n".join(updated_cases)


def format_updated_case(source: str, actual: str) -> str:
    case_source, _expected, _has_expected = split_expected(source)
    return (
        ensure_trailing_newline(case_source)
        + f"{EXPECTED_SEPARATOR}\n"
        + ensure_trailing_newline(actual)
    )


def _raw_case_source(source: str, case_index: int) -> str:
    return split_raw_cases(source)[case_index]


def ensure_trailing_newline(text: str) -> str:
    return text if not text or text.endswith("\n") else f"{text}\n"


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
        "failed": 0,
        "crashed": 0,
    }
    for result in results:
        summary[result.status] += 1
    return summary
