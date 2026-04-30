# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Command line entry point for importer checks."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from loom.importers.check.runner import (
    MlirCheckOptions,
    results_to_json,
    run_mlir_check,
    summarize_results,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="loom-import-check")
    subparsers = parser.add_subparsers(dest="importer", required=True)
    mlir_parser = subparsers.add_parser("mlir", help="check MLIR kernel imports")
    mlir_parser.add_argument("paths", nargs="+", type=Path)
    mlir_parser.add_argument("--kernel")
    mlir_parser.add_argument(
        "--update",
        action="store_true",
        help="update inline expected output sections",
    )
    mlir_parser.add_argument(
        "--prefer-abi3-extensions",
        action="store_true",
        help="prefer local .abi3.so IREE compiler bindings",
    )
    mlir_parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable check results",
    )

    args = parser.parse_args(argv)
    if args.importer != "mlir":
        parser.error(f"unsupported importer: {args.importer}")

    options = MlirCheckOptions(
        kernel=args.kernel,
        prefer_abi3_extensions=args.prefer_abi3_extensions,
        update=args.update,
    )
    results = [
        result
        for path in args.paths
        for result in run_mlir_check(path, options=options)
    ]
    if args.json:
        sys.stdout.write(f"{results_to_json(results)}\n")
    else:
        for result in results:
            sys.stdout.write(
                f"{result.status}: {result.path}:case{result.case_index}\n"
            )
            if result.mismatch:
                sys.stdout.write(f"  {result.mismatch}\n")
                if result.diff:
                    sys.stdout.write(result.diff)
            if not result.passed:
                first_line = _diagnostic_line(result.stderr or result.stdout)
                if first_line:
                    sys.stdout.write(f"  {first_line}\n")
        summary = summarize_results(results)
        sys.stdout.write(
            "summary: "
            f"{summary['passed']} passed, "
            f"{summary['updated']} updated, "
            f"{summary['failed']} failed, "
            f"{summary['crashed']} crashed"
            "\n"
        )
    return 0 if all(result.passed for result in results) else 1


def _diagnostic_line(text: str) -> str | None:
    fallback: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if fallback is None and "RuntimeWarning:" not in stripped:
            fallback = stripped
        if stripped.startswith(("error:", "fatal:", "Traceback ")):
            return stripped
    return fallback


if __name__ == "__main__":
    raise SystemExit(main())
