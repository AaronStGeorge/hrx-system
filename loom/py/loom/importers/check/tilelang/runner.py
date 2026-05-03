# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runs TileLang importer checks from Python fixtures."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loom.importers.check.loom_verifier import LoomOutputVerifier
from loom.importers.check.python import (
    PythonCheckCase,
    PythonCheckOptions,
    run_python_check,
)
from loom.importers.check.results import CheckResult
from loom.importers.check.tilelang.cases import (
    TileLangImportInput,
    get_tilelang_case_metadata,
    is_tilelang_case,
)
from loom.importers.check.tilelang.harness import TileLangHarness
from loom.importers.core import kernel_module_ops, print_loom_module
from loom.importers.tilelang.importer import TileLangImportOptions, import_tilelang


@dataclass(frozen=True, slots=True)
class TileLangCheckOptions:
    """Options for Python TileLang importer checks."""

    update: bool = False
    target_preset: str | None = None
    case_filter: str | None = None
    verify_structure: bool = True
    output_verifier: LoomOutputVerifier | None = None


def run_tilelang_check(
    path: Path,
    *,
    options: TileLangCheckOptions,
) -> tuple[CheckResult, ...]:
    return run_python_check(
        path,
        options=PythonCheckOptions(
            update=options.update,
            case_filter=options.case_filter,
            output_verifier=options.output_verifier,
        ),
        is_case=is_tilelang_case,
        invoke=lambda case: _invoke_tilelang_case(case, options),
        case_labels=_tilelang_case_labels,
    )


def _invoke_tilelang_case(
    case: PythonCheckCase,
    options: TileLangCheckOptions,
) -> str:
    value = case.function(**_case_kwargs(case.function))
    metadata = get_tilelang_case_metadata(case.function)
    if not isinstance(value, TileLangImportInput):
        value = TileLangImportInput(
            source=value,
            target=options.target_preset,
            name=metadata.name if metadata is not None else None,
        )
    target_preset = options.target_preset or value.target or "tilelang.generic"
    result = import_tilelang(
        value,
        options=TileLangImportOptions(
            target_preset=options.target_preset,
            verify_structure=options.verify_structure,
        ),
    )
    return print_loom_module(result.module, ops=kernel_module_ops(target_preset))


def _case_kwargs(function: Callable[..., Any]) -> dict[str, Any]:
    signature = inspect.signature(function)
    if not signature.parameters:
        return {}
    harness: TileLangHarness | None = None
    kwargs: dict[str, Any] = {}
    for name in signature.parameters:
        if harness is None:
            harness = TileLangHarness()
        if name == "h":
            kwargs[name] = harness
        elif name == "T":
            kwargs[name] = harness.T
        elif name == "tilelang":
            kwargs[name] = harness.tilelang
        elif name == "tvm":
            kwargs[name] = harness.tvm
        elif name == "tir":
            kwargs[name] = harness.tir
        else:
            raise TypeError(f"unsupported TileLang fixture parameter `{name}`")
    return kwargs


def _tilelang_case_labels(case: PythonCheckCase) -> tuple[str, ...]:
    metadata = get_tilelang_case_metadata(case.function)
    if metadata is None:
        return ()
    labels = [metadata.category, *metadata.tags]
    if metadata.name:
        labels.append(metadata.name)
    return tuple(labels)
