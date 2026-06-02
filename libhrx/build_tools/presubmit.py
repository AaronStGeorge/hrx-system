#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""libhrx project presubmit entry point."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = "libhrx/"
GLOBAL_TEST_TRIGGERS = (
    "BUILD.bazel",
    "MODULE.bazel",
    ".bazelrc",
    ".bazel_to_cmake.cfg.py",
    "requirements",
    "build_tools/bazel/",
    "build_tools/bazel_to_cmake/",
    "build_tools/testing/",
    "build_tools/third_party/",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run libhrx project presubmit.")
    mutation = parser.add_mutually_exclusive_group()
    mutation.add_argument("--fix", action="store_true", help="Accepted for symmetry.")
    mutation.add_argument("--check", action="store_true", help="Accepted for symmetry.")
    parser.add_argument("--tests", action="store_true", help="Run libhrx tests.")
    parser.add_argument(
        "--files-from",
        help="Path to a newline-separated repo-relative changed-file list.",
    )
    return parser.parse_args()


def run_command(command: list[str], description: str) -> bool:
    print(f"libhrx presubmit: {description}")
    print("  " + " ".join(command))
    sys.stdout.flush()
    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode == 0:
        return True
    print(f"libhrx presubmit: {description} failed with exit code {result.returncode}")
    return False


def is_global_trigger(path: str) -> bool:
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


def main() -> int:
    args = parse_arguments()
    if not args.tests:
        return 0
    if not should_run_tests(args.files_from):
        print("libhrx presubmit: no libhrx-affecting files")
        return 0
    ok = run_command(
        ["bazel", "test", "--config=presubmit", "//libhrx/..."],
        "Bazel tests",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
