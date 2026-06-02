#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runtime project presubmit entry point."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run runtime project presubmit.")
    mutation = parser.add_mutually_exclusive_group()
    mutation.add_argument("--fix", action="store_true", help="Accepted for symmetry.")
    mutation.add_argument("--check", action="store_true", help="Accepted for symmetry.")
    parser.add_argument("--tests", action="store_true", help="Run runtime tests.")
    parser.add_argument(
        "--files-from",
        help="Path to a newline-separated repo-relative changed-file list.",
    )
    return parser.parse_args()


def run_command(command: list[str], description: str) -> bool:
    print(f"runtime presubmit: {description}")
    print("  " + " ".join(command))
    sys.stdout.flush()
    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode == 0:
        return True
    print(
        f"runtime presubmit: {description} failed with exit code {result.returncode}"
    )
    return False


def main() -> int:
    args = parse_arguments()
    if not args.tests:
        return 0
    ok = run_command(
        ["bazel", "test", "--config=presubmit", "//runtime/..."],
        "Bazel tests",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
