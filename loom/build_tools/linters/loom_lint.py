#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom Python linting orchestrator.

Runs code generators, ruff (lint + format), and mypy on the loom Python
package. Used as a pre-commit hook and runnable standalone:

    python loom/build_tools/linters/loom_lint.py

Exit code is 0 only if all steps pass and no files were modified.
"""

import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
RUNTIME_PY = os.path.join(REPO_ROOT, "runtime", "py")


def _run(description: str, cmd: list[str], **kwargs: object) -> bool:
    """Run a command, print status, return True on success."""
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode == 0:
        print(f"  PASS  {description}")
        if result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                print(f"        {line}")
        return True
    print(f"  FAIL  {description}")
    for line in (result.stdout + result.stderr).strip().splitlines():
        print(f"        {line}")
    return False


def main() -> int:
    ok = True

    cwd = RUNTIME_PY

    print("loom-lint: generators")
    ok &= _run(
        "python builders",
        [sys.executable, "loom/py/loom/gen/run.py", "builders"],
        cwd=REPO_ROOT,
    )
    ok &= _run(
        "c tables",
        [sys.executable, "loom/py/loom/gen/run.py", "c_tables"],
        cwd=REPO_ROOT,
    )
    ok &= _run(
        "textmate",
        [sys.executable, "loom/py/loom/gen/run.py", "textmate"],
        cwd=REPO_ROOT,
    )

    print("loom-lint: ruff")
    ok &= _run("format", ["ruff", "format", "loom/"], cwd=cwd)
    ok &= _run("lint", ["ruff", "check", "--fix", "loom/"], cwd=cwd)

    print("loom-lint: mypy")
    ok &= _run("type-check", ["mypy", "loom/"], cwd=cwd)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
