#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom project presubmit entry point."""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from build_tools.devtools import project_presubmit

PROJECT_NAME = "loom"
PROJECT_ROOT = "loom/"
CMAKE_TEST_REGEX = "^loom/"
GLOBAL_TEST_TRIGGERS = (
    "BUILD.bazel",
    "MODULE.bazel",
    ".bazelrc",
    ".bazel_to_cmake.cfg.py",
    "requirements",
)
RESOURCE_TEST_TAG_FILTERS = (
    "-iree-run-requirement=runtime.resource.amd_gpu",
    "-iree-run-requirement=runtime.resource.nvidia_gpu",
    "-iree-run-requirement=runtime.resource.vulkan_device",
    "-iree-run-requirement=runtime.resource.webgpu_device",
)
CTEST_RESOURCE_LABEL_EXCLUDE_REGEX = "runtime-resource="


@dataclass(frozen=True)
class PythonPackageSurface:
    path: str
    required_exports: tuple[str, ...]


PYTHON_PACKAGE_SURFACES = (
    PythonPackageSurface(
        "loom/py/loom/__init__.py",
        (
            "LoomBuilder",
            "module_builder",
        ),
    ),
    PythonPackageSurface(
        "loom/py/loom/dialect/__init__.py",
        (
            "scf",
            "target",
            "vector",
        ),
    ),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Loom project presubmit.")
    mutation = parser.add_mutually_exclusive_group()
    mutation.add_argument("--fix", action="store_true", help="Accepted for symmetry.")
    mutation.add_argument("--check", action="store_true", help="Accepted for symmetry.")
    parser.add_argument(
        "--lane",
        choices=("bazel", "cmake"),
        default="bazel",
        help="Build-system lane used for tests. Defaults to bazel.",
    )
    parser.add_argument("--tests", action="store_true", help="Run Loom tests.")
    parser.add_argument(
        "--files-from",
        help="Path to a newline-separated repo-relative changed-file list.",
    )
    return parser.parse_args()


def run_command(command: list[str], description: str) -> bool:
    return project_presubmit.run_command(
        PROJECT_NAME, command, description, cwd=REPO_ROOT
    )


def python_all_exports(path: Path) -> tuple[str, ...] | None:
    try:
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        print(f"{path.relative_to(REPO_ROOT)}:{exc.lineno}: invalid Python syntax")
        return None
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in statement.targets
        ):
            continue
        try:
            exports = ast.literal_eval(statement.value)
        except (TypeError, ValueError):
            return None
        if not isinstance(exports, (list, tuple)):
            return None
        if not all(isinstance(export, str) for export in exports):
            return None
        return tuple(exports)
    return None


def verify_python_package_surfaces() -> bool:
    ok = True
    for surface in PYTHON_PACKAGE_SURFACES:
        path = REPO_ROOT / surface.path
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"{surface.path}: could not read package surface: {exc}")
            ok = False
            continue
        if not text.strip():
            print(f"{surface.path}: protected package surface is empty")
            ok = False
            continue
        exports = python_all_exports(path)
        if exports is None:
            print(f"{surface.path}: protected package surface must define __all__")
            ok = False
            continue
        missing_exports = [
            export for export in surface.required_exports if export not in exports
        ]
        if missing_exports:
            print(
                f"{surface.path}: protected package surface is missing "
                f"required __all__ export(s): {', '.join(missing_exports)}"
            )
            ok = False
    return ok


def run_python_package_surface_check() -> bool:
    print("loom presubmit: Python package surfaces")
    ok = verify_python_package_surfaces()
    if ok:
        print("loom presubmit: Python package surfaces ok")
    return ok


def is_global_trigger(path: str) -> bool:
    if "build_tools" in Path(path).parts:
        return True
    if path.startswith("requirements") and path.endswith(".txt"):
        return True
    return any(
        path == trigger or path.startswith(trigger) for trigger in GLOBAL_TEST_TRIGGERS
    )


def selected_files(files_from: str | None) -> list[str]:
    if not files_from:
        return []
    with open(files_from, encoding="utf-8") as file_list:
        return [line.strip() for line in file_list if line.strip()]


def should_run_tests(files_from: str | None) -> bool:
    paths = selected_files(files_from)
    if not paths:
        return files_from is None
    return any(
        path.startswith(PROJECT_ROOT) or is_global_trigger(path) for path in paths
    )


def bazel_test_command() -> list[str]:
    return [
        "bazel",
        "test",
        "--config=presubmit",
        "--test_tag_filters=" + ",".join(RESOURCE_TEST_TAG_FILTERS),
        "//loom/...",
    ]


def run_bazel_tests() -> bool:
    return run_command(
        bazel_test_command(),
        "Bazel tests",
    )


def run_cmake_tests() -> bool:
    build_dir = project_presubmit.cmake_build_dir(REPO_ROOT)
    if not project_presubmit.validate_cmake_build_tree(PROJECT_NAME, build_dir):
        return False
    if not run_command(
        ["cmake", "--build", str(build_dir), "--parallel"],
        "CMake build",
    ):
        return False
    return run_command(
        [
            "ctest",
            "--test-dir",
            str(build_dir),
            "--output-on-failure",
            "-R",
            CMAKE_TEST_REGEX,
            "-LE",
            CTEST_RESOURCE_LABEL_EXCLUDE_REGEX,
        ],
        "CTest tests",
    )


def main() -> int:
    args = parse_arguments()
    if not args.tests:
        return 0
    if not should_run_tests(args.files_from):
        print("loom presubmit: no Loom-affecting files")
        return 0
    ok = run_python_package_surface_check()
    if not ok:
        return 1
    if args.lane == "bazel":
        ok = run_bazel_tests()
    elif args.lane == "cmake":
        ok = run_cmake_tests()
    else:
        raise ValueError(f"unknown lane: {args.lane}")
    ok = run_python_package_surface_check() and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
