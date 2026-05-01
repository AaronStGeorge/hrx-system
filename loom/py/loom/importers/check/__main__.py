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

from loom.importers.check.registry import (
    make_default_registry,
    print_importers,
    skip_unavailable_backend,
)
from loom.importers.check.results import (
    CheckResult,
    results_to_json,
    summarize_results,
)


def main(argv: Sequence[str] | None = None) -> int:
    registry = make_default_registry()
    parser = argparse.ArgumentParser(prog="loom-import-check")
    parser.add_argument(
        "--list-importers",
        action="store_true",
        help="list enabled and disabled importer check backends",
    )
    parser.add_argument(
        "--skip-unavailable-importer",
        action="store_true",
        help="skip checks for unavailable optional importer backends",
    )
    subparsers = parser.add_subparsers(dest="importer")
    for candidate in registry.all():
        help_text = candidate.backend.help
        if not candidate.availability.available:
            help_text = f"{help_text} (disabled: {candidate.availability.message()})"
        subparser = subparsers.add_parser(candidate.backend.name, help=help_text)
        candidate.backend.add_arguments(subparser)
    args = parser.parse_args(argv)
    if args.list_importers:
        sys.stdout.write(f"{print_importers(registry)}\n")
        return 0
    if args.importer is None:
        parser.error("importer is required unless --list-importers is used")
    selected = registry.by_name(args.importer)
    if selected is None:
        parser.error(f"unknown importer: {args.importer}")
        raise AssertionError("argparse.error returned")

    availability = selected.availability
    if availability.available:
        availability = selected.backend.prepare(args)
    if not availability.available:
        if args.skip_unavailable_importer:
            results = skip_unavailable_backend(selected, args, availability)
        else:
            parser.error(
                f"importer {args.importer!r} is not available: {availability.message()}"
            )
            raise AssertionError("argparse.error returned")
    else:
        results = selected.backend.run(args)

    if args.json:
        sys.stdout.write(f"{results_to_json(results)}\n")
    else:
        for result in results:
            sys.stdout.write(f"{result.status}: {_result_location(result)}\n")
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
            f"{summary['skipped']} skipped, "
            f"{summary['failed']} failed, "
            f"{summary['crashed']} crashed"
            "\n"
        )
    return 0 if all(result.passed for result in results) else 1


def _result_location(result: CheckResult) -> str:
    if result.case_index < 0:
        return str(result.path)
    return f"{result.path}:case{result.case_index}"


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
