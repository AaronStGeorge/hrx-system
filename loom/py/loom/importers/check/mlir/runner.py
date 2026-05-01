# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runs MLIR importer checks using the inline loom-check case format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loom.importers.check.cases import (
    DEFAULT_CHECK_SYNTAX,
    CheckCase,
    parse_inline_cases,
)
from loom.importers.check.results import CheckResult, unified_diff
from loom.importers.check.updates import format_updated_source


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
    raw_cases = parse_inline_cases(
        path,
        source,
        syntax=DEFAULT_CHECK_SYNTAX,
        default_run="mlir",
    )
    cases: list[CheckCase] = []
    for raw_case in raw_cases:
        run = raw_case.run or "mlir"
        run_options = parse_mlir_run(run)
        kernel = run_options.kernel or default_kernel
        effective_run = "mlir" if kernel is None else f"mlir --kernel {kernel}"
        cases.append(
            CheckCase(
                path=path,
                index=raw_case.index,
                source=raw_case.source,
                input=raw_case.input,
                expected=raw_case.expected,
                run=effective_run,
            )
        )
    return tuple(cases)


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

    run_options = parse_mlir_run(case.run or "mlir")
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
