#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validates Git commit message text."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

COMMENT_PREFIX = "#"
SCISSORS_MARKER = ">8"

LITERAL_ESCAPE_PATTERN = re.compile(r"\\[nr]")
BEAD_ID_PATTERN = re.compile(
    r"\b(?P<id>(?:bd-[0-9a-z]+|loom-[0-9a-z]*[0-9][0-9a-z]*)(?:\.\d+)?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CommitMessageDiagnostic:
    line_number: int
    message: str
    text: str


def commit_message_text_lines(message_text: str) -> list[tuple[int, str]]:
    """Returns user-authored commit message lines with Git comments removed."""

    lines: list[tuple[int, str]] = []
    for line_number, line in enumerate(message_text.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith(COMMENT_PREFIX):
            if SCISSORS_MARKER in stripped:
                break
            continue
        lines.append((line_number, line))
    return lines


def validate_commit_message_text(
    message_text: str,
) -> list[CommitMessageDiagnostic]:
    """Returns diagnostics for commit-message policy violations."""

    diagnostics: list[CommitMessageDiagnostic] = []
    for line_number, line in commit_message_text_lines(message_text):
        literal_escape = LITERAL_ESCAPE_PATTERN.search(line)
        if literal_escape:
            diagnostics.append(
                CommitMessageDiagnostic(
                    line_number=line_number,
                    message=(
                        "literal newline/carriage-return escape sequences are "
                        "not allowed; use real paragraph breaks"
                    ),
                    text=literal_escape.group(0),
                )
            )
        for match in BEAD_ID_PATTERN.finditer(line):
            diagnostics.append(
                CommitMessageDiagnostic(
                    line_number=line_number,
                    message="bead identifiers must not appear in commit messages",
                    text=match.group("id"),
                )
            )
    return diagnostics


def validate_commit_message_file(path: Path) -> list[CommitMessageDiagnostic]:
    """Reads and validates a Git commit message file."""

    return validate_commit_message_text(path.read_text(encoding="utf-8"))


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "message_file",
        type=Path,
        help="Git commit message file passed to the commit-msg hook.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_arguments(argv)
    diagnostics = validate_commit_message_file(args.message_file)
    if not diagnostics:
        return 0

    print("commit message policy failed:", file=sys.stderr)
    for diagnostic in diagnostics:
        print(
            f"  {args.message_file}:{diagnostic.line_number}: "
            f"{diagnostic.message}: {diagnostic.text!r}",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
