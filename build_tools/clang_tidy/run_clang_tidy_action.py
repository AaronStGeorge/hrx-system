#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def parse_arguments(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    if "--" not in argv:
        raise ValueError("missing '--' before compile arguments")
    delimiter = argv.index("--")
    parser = argparse.ArgumentParser()
    parser.add_argument("--clang-tidy", required=True, type=Path)
    parser.add_argument("--plugin", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checks", required=True)
    parser.add_argument("--warnings-as-errors", required=True)
    args = parser.parse_args(argv[:delimiter])
    return args, argv[delimiter + 1 :]


def main(argv: list[str]) -> int:
    try:
        args, compile_args = parse_arguments(argv)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    command = [
        str(args.clang_tidy),
        f"--load={args.plugin}",
        f"--checks={args.checks}",
        f"--warnings-as-errors={args.warnings_as_errors}",
        str(args.source),
        "--",
        *compile_args,
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = completed.stdout
    args.output.write_text(output, encoding="utf-8")
    if completed.returncode != 0:
        print("clang-tidy command failed:", file=sys.stderr)
        print(shlex.join(command), file=sys.stderr)
        if output:
            print(output, file=sys.stderr, end="" if output.endswith("\n") else "\n")
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
