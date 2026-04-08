# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Root-invocable runner for Loom generators.

Usage:
    python3 loom/py/loom/gen/run.py builders
    python3 loom/py/loom/gen/run.py c_tables
    python3 loom/py/loom/gen/run.py textmate
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import bootstrap

GENERATORS = {
    "builders": "loom.gen.builders",
    "c_errors": "loom.gen.c_errors",
    "c_tables": "loom.gen.c_tables",
    "textmate": "loom.gen.textmate",
}


def _usage() -> str:
    names = "|".join(GENERATORS)
    return f"usage: python3 loom/py/loom/gen/run.py <{names}>"


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in GENERATORS:
        print(_usage(), file=sys.stderr)
        return 2

    repo_root = bootstrap.ensure_runtime_py_on_path(Path(__file__))
    if Path.cwd().resolve() != repo_root:
        print(f"error: run from repository root: {repo_root}", file=sys.stderr)
        return 2

    runpy.run_module(GENERATORS[argv[1]], run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
