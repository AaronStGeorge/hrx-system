# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runs simple module-level pytest-style test functions.

This runner is intentionally tiny: it supports no fixtures and no plugin
surface. Tests that need real pytest should use a future pytest-backed rule.
Most Loom Python unit tests are ordinary zero-argument functions, so this gives
them a Bazel path without making pytest a base dependency.
"""

from __future__ import annotations

import importlib
import inspect
import sys
import traceback
import unittest
from collections.abc import Callable, Iterable, Sequence
from types import ModuleType
from typing import Any


class TestRunnerError(ValueError):
    """Raised when a test module uses unsupported runner features."""


def main(argv: Sequence[str] | None = None) -> int:
    try:
        module_names = _parse_runner_args(tuple(sys.argv[1:] if argv is None else argv))
    except TestRunnerError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    if not module_names:
        sys.stderr.write("error: at least one test module is required\n")
        return 2

    failures = 0
    skips = 0
    test_count = 0
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            for test_name, function in _iter_test_functions(module):
                test_count += 1
                try:
                    _run_test_function(module_name, test_name, function)
                except unittest.SkipTest as exc:
                    skips += 1
                    sys.stdout.write(f"SKIP {module_name}.{test_name}: {exc}\n")
                except Exception:
                    failures += 1
                    sys.stderr.write(f"\nFAILED {module_name}.{test_name}\n")
                    traceback.print_exc(file=sys.stderr)
        except Exception:
            failures += 1
            sys.stderr.write(f"\nFAILED loading {module_name}\n")
            traceback.print_exc(file=sys.stderr)

    sys.stdout.write(
        "loom pytest-style runner: "
        f"{test_count} tests, {failures} failures, {skips} skipped\n"
    )
    if test_count == 0:
        sys.stderr.write("error: no tests were discovered\n")
        return 2
    return 1 if failures else 0


def _iter_test_functions(
    module: ModuleType,
) -> Iterable[tuple[str, Callable[[], object]]]:
    for name, value in inspect.getmembers(module, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        if value.__module__ != module.__name__:
            continue
        signature = inspect.signature(value)
        if signature.parameters:
            raise TestRunnerError(
                f"{module.__name__}.{name} has parameters; "
                "pytest fixtures are not supported by this runner"
            )
        yield name, value


def _run_test_function(
    module_name: str,
    test_name: str,
    function: Callable[[], Any],
) -> None:
    sys.stdout.write(f"RUN {module_name}.{test_name}\n")
    function()


def _parse_runner_args(args: Sequence[str]) -> tuple[str, ...]:
    module_names: list[str] = []
    for arg in args:
        if arg == "--update":
            continue
        if arg.startswith("--"):
            raise TestRunnerError(f"unsupported runner flag {arg!r}")
        module_names.append(arg)
    return tuple(module_names)


if __name__ == "__main__":
    raise SystemExit(main())
