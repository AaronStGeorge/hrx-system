#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clang-tidy", required=True, type=Path)
    parser.add_argument("--plugin", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    completed = subprocess.run(
        [
            str(args.clang_tidy),
            f"--load={args.plugin}",
            "--checks=-*,iree-smoke",
            str(args.source),
            "--",
            "-std=c11",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        print(output, file=sys.stderr)
        return completed.returncode
    expected = "IREE clang-tidy plugin smoke diagnostic [iree-smoke]"
    if expected not in output:
        print(output, file=sys.stderr)
        print(f"expected clang-tidy output to contain: {expected}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
