#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Smoke tests for the repository developer command surface."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dev.py checkout smoke tests.")
    parser.add_argument(
        "--from-working-tree",
        action="store_true",
        help="Copy tracked and untracked working-tree files instead of cloning HEAD.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the temporary checkout for inspection.",
    )
    parser.add_argument(
        "--scenario",
        choices=("dry-run",),
        default="dry-run",
        help="Smoke scenario to run.",
    )
    return parser.parse_args()


def git_paths(*args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [path.decode("utf-8") for path in result.stdout.split(b"\0") if path]


def copy_working_tree(destination: Path) -> None:
    paths = git_paths("ls-files", "-z")
    paths += git_paths("ls-files", "-z", "--others", "--exclude-standard")
    for relative_path in paths:
        source_path = REPO_ROOT / relative_path
        if not source_path.is_file():
            continue
        destination_path = destination / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
    subprocess.run(["git", "init"], cwd=destination, check=True, stdout=subprocess.PIPE)


def clone_head(destination: Path) -> None:
    subprocess.run(
        ["git", "clone", "--local", "--no-hardlinks", str(REPO_ROOT), str(destination)],
        cwd=REPO_ROOT,
        check=True,
    )


def run_command(checkout: Path, args: list[str]) -> None:
    command = [sys.executable, "dev.py", *args]
    print("smoke:", " ".join(command), flush=True)
    subprocess.run(command, cwd=checkout, check=True)


def assert_absent(path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"dry-run smoke unexpectedly created {path}")


def run_dry_run_scenario(checkout: Path) -> None:
    tool_root = checkout.parent / "external-tools"
    run_command(checkout, ["--dry-run", "bazel", "setup"])
    run_command(checkout, ["--dry-run", "bazel", "setup", "--system"])
    run_command(
        checkout, ["--dry-run", "bazel", "setup", "--tool-root", str(tool_root)]
    )
    run_command(
        checkout,
        ["--dry-run", "bazel", "configure", "-DIREE_HAL_DRIVER_AMDGPU=OFF"],
    )
    run_command(
        checkout,
        [
            "--dry-run",
            "cmake",
            "configure",
            "-DIREE_HAL_DRIVER_AMDGPU=OFF",
            "-DLIBHRX_BUILD=OFF",
        ],
    )
    run_command(checkout, ["--dry-run", "bazel", "hook", "--profile", "ci"])
    run_command(checkout, ["--dry-run", "cmake", "hook", "--profile", "paranoid"])
    run_command(checkout, ["--dry-run", "bazel", "precommit", "--profile", "default"])
    run_command(checkout, ["--dry-run", "cmake", "precommit", "--profile", "ci"])
    run_command(checkout, ["--dry-run", "bazel", "presubmit", "--profile", "paranoid"])
    run_command(checkout, ["--dry-run", "cmake", "presubmit", "--profile", "default"])
    assert_absent(checkout / ".bazelrc.configured")
    assert_absent(checkout / ".venv")
    assert_absent(checkout / "lefthook-local.yml")
    assert_absent(tool_root)


def main() -> int:
    args = parse_arguments()
    if args.keep:
        temporary_root = Path(tempfile.mkdtemp(prefix="iree-x-dev-smoke-"))
        checkout = temporary_root / "checkout"
        if args.from_working_tree:
            checkout.mkdir()
            copy_working_tree(checkout)
        else:
            clone_head(checkout)
        if args.scenario == "dry-run":
            run_dry_run_scenario(checkout)
        print(f"smoke: passed in {checkout}")
        print(f"smoke: kept {temporary_root}")
        return 0

    with tempfile.TemporaryDirectory(prefix="iree-x-dev-smoke-") as temporary_name:
        temporary_root = Path(temporary_name)
        checkout = temporary_root / "checkout"
        if args.from_working_tree:
            checkout.mkdir()
            copy_working_tree(checkout)
        else:
            clone_head(checkout)
        if args.scenario == "dry-run":
            run_dry_run_scenario(checkout)
        print(f"smoke: passed in {checkout}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
