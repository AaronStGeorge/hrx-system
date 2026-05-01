# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from tempfile import TemporaryDirectory

from loom.importers.check.python import (
    PythonCheckCase,
    PythonCheckOptions,
    decode_python_expected,
    encode_python_expected,
    run_python_check,
)


def test_python_expected_codec_keeps_output_in_comments() -> None:
    encoded = encode_python_expected("func @main\n\n  return\n")

    assert encoded == "# func @main\n#\n#   return\n"
    assert decode_python_expected(encoded) == "func @main\n\n  return\n"


def test_run_python_check_executes_decorated_cases() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "cases.py"
        path.write_text(
            """
def case(func):
    func.__case__ = True
    return func

@case
def first():
    return "first\\n"
# ----
# first
# ====
@case
def second():
    return "second\\n"
# ----
# second
"""
        )

        results = run_python_check(
            path,
            options=PythonCheckOptions(),
            is_case=lambda value: bool(getattr(value, "__case__", False)),
            invoke=lambda case: case.function(),
        )

    assert [result.status for result in results] == ["passed", "passed"]


def test_run_python_check_shares_preamble_across_cases() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "cases.py"
        path.write_text(
            """
def case(func):
    func.__case__ = True
    return func


class Shared:
    value = "shared"


# ====
@case
def first():
    return f"{Shared.value}:first\\n"
# ----
# shared:first

# ====
@case
def second():
    return f"{Shared.value}:second\\n"
# ----
# shared:second
"""
        )

        results = run_python_check(
            path,
            options=PythonCheckOptions(),
            is_case=lambda value: bool(getattr(value, "__case__", False)),
            invoke=lambda case: case.function(),
        )

    assert [result.status for result in results] == ["passed", "passed"]


def test_run_python_check_updates_commented_expected_blocks() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "cases.py"
        path.write_text(
            """
def case(func):
    func.__case__ = True
    return func

@case
def first():
    return "new\\n"
# ----
# old
"""
        )

        results = run_python_check(
            path,
            options=PythonCheckOptions(update=True),
            is_case=lambda value: bool(getattr(value, "__case__", False)),
            invoke=lambda case: case.function(),
        )
        updated_source = path.read_text()

    assert results[0].updated
    assert "# new\n" in updated_source
    assert "# old\n" not in updated_source


def test_run_python_check_filters_by_function_name() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "cases.py"
        path.write_text(
            """
def case(func):
    func.__case__ = True
    return func

@case
def first():
    return "first\\n"
# ----
# first

# ====
@case
def second():
    return "second\\n"
# ----
# second
"""
        )

        results = run_python_check(
            path,
            options=PythonCheckOptions(case_filter="second"),
            is_case=lambda value: bool(getattr(value, "__case__", False)),
            invoke=lambda case: case.function(),
        )

    assert len(results) == 1
    assert results[0].case_index == 1
    assert results[0].passed


def test_invoke_receives_python_check_case() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "cases.py"
        path.write_text(
            """
def case(func):
    func.__case__ = True
    return func

@case
def named():
    return "ignored\\n"
# ----
# cases.py:0
"""
        )

        def invoke(case: PythonCheckCase) -> str:
            return f"{case.check_case.path.name}:{case.check_case.index}\n"

        results = run_python_check(
            path,
            options=PythonCheckOptions(),
            is_case=lambda value: bool(getattr(value, "__case__", False)),
            invoke=invoke,
        )

    assert results[0].passed
