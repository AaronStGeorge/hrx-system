# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Root-invocable runner for Loom generators.

Usage:
    python3 loom/py/loom/gen/run.py amdgpu_isa_snapshot --xml ... --out ...
    python3 loom/py/loom/gen/run.py builders
    python3 loom/py/loom/gen/run.py c_tables
    python3 loom/py/loom/gen/run.py low_descriptors
    python3 loom/py/loom/gen/run.py textmate
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import bootstrap

GENERATORS = {
    "amdgpu_isa_snapshot": "loom.gen.amdgpu_isa_snapshot",
    "builders": "loom.gen.builders",
    "c_errors": "loom.gen.c_errors",
    "c_tables": "loom.gen.c_tables",
    "low_descriptors": "loom.gen.low_descriptors",
    "textmate": "loom.gen.textmate",
}

ARGUMENT_GENERATORS = {"amdgpu_isa_snapshot"}


def _usage() -> str:
    names = "|".join(GENERATORS)
    return f"usage: python3 loom/py/loom/gen/run.py <{names}> [generator args...]"


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] not in GENERATORS:
        print(_usage(), file=sys.stderr)
        return 2
    if argv[1] not in ARGUMENT_GENERATORS and len(argv) != 2:
        print(_usage(), file=sys.stderr)
        return 2

    repo_root = bootstrap.ensure_runtime_py_on_path(Path(__file__))
    if Path.cwd().resolve() != repo_root:
        print(f"error: run from repository root: {repo_root}", file=sys.stderr)
        return 2

    module = importlib.import_module(GENERATORS[argv[1]])
    if argv[1] in ARGUMENT_GENERATORS:
        return int(module.main(argv[2:]))
    module.main()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
