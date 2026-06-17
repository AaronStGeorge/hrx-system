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
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

COMMENT_PREFIX = "#"
SCISSORS_MARKER = ">8"
LINE_LENGTH_LIMIT = 72

LITERAL_ESCAPE_PATTERN = re.compile(r"\\[nr]")
BEAD_ID_PATTERN = re.compile(
    r"\b(?P<id>(?:bd-[0-9a-z]+|loom-[0-9a-z]*[0-9][0-9a-z]*)(?:\.\d+)?)\b",
    re.IGNORECASE,
)
SUBJECT_TAG_PATTERN = re.compile(
    r"^\[(?P<tag>[A-Za-z][A-Za-z0-9]*(?:/[A-Za-z][A-Za-z0-9]*)?)\](?:\s+|$)"
)
CODE_FENCE_PATTERN = re.compile(r"^\s*(```|~~~)")
TAG_EXAMPLES = ("[Loom]", "[HRX]", "[HAL]", "[Runtime]", "[Infra]", "[CI]")


@dataclass(frozen=True)
class TagSuggestionRule:
    tag: str
    score: int
    path_prefixes: tuple[str, ...]
    exact_paths: tuple[str, ...] = ()


TAG_SUGGESTION_RULES = (
    TagSuggestionRule(
        tag="[HAL/AMDGPU]",
        score=40,
        path_prefixes=(
            "runtime/src/iree/hal/drivers/amdgpu/",
            "runtime/src/iree/hal/drivers/hip/",
            "build_tools/amdgpu/",
        ),
    ),
    TagSuggestionRule(
        tag="[Loom/AMDGPU]",
        score=38,
        path_prefixes=(
            "loom/src/loom/target/arch/amdgpu/",
            "loom/py/loom/target/arch/amdgpu/",
        ),
    ),
    TagSuggestionRule(
        tag="[Runtime/VM]",
        score=36,
        path_prefixes=("runtime/src/iree/vm/",),
    ),
    TagSuggestionRule(
        tag="[Loom]",
        score=30,
        path_prefixes=("loom/",),
    ),
    TagSuggestionRule(
        tag="[HRX]",
        score=30,
        path_prefixes=("libhrx/",),
    ),
    TagSuggestionRule(
        tag="[HAL]",
        score=28,
        path_prefixes=("runtime/src/iree/hal/",),
    ),
    TagSuggestionRule(
        tag="[Runtime]",
        score=24,
        path_prefixes=("runtime/src/iree/",),
    ),
    TagSuggestionRule(
        tag="[CI]",
        score=22,
        path_prefixes=(".github/workflows/", "build_tools/devtools/ci"),
    ),
    TagSuggestionRule(
        tag="[Infra]",
        score=20,
        path_prefixes=(
            ".github/",
            "build_tools/",
            "requirements",
            "third_party/",
        ),
        exact_paths=(
            ".bazelrc",
            ".bazelversion",
            "AGENTS.override.md",
            "BUILD.bazel",
            "CONTRIBUTING.md",
            "MODULE.bazel",
            "MODULE.bazel.lock",
            "README.md",
            "WORKSPACE",
            "WORKSPACE.bazel",
            "WORKSPACE.bzlmod",
            "dev.py",
            "lefthook.yml",
        ),
    ),
)


@dataclass(frozen=True)
class CommitMessageDiagnostic:
    line_number: int
    message: str
    text: str
    hint: str = ""


def staged_commit_paths() -> list[str]:
    """Returns paths staged for the commit being validated."""

    output = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMRT"],
        cwd=REPO_ROOT,
        text=True,
    )
    return [line for line in output.splitlines() if line]


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


def tag_suggestions_for_paths(paths: Sequence[str]) -> list[str]:
    """Returns ranked subsystem tag suggestions for changed paths."""

    scores: dict[str, int] = {}
    for path in paths:
        for rule in TAG_SUGGESTION_RULES:
            if path in rule.exact_paths or any(
                path.startswith(prefix) for prefix in rule.path_prefixes
            ):
                scores[rule.tag] = scores.get(rule.tag, 0) + rule.score
    if not scores:
        return list(TAG_EXAMPLES)

    order = {rule.tag: index for index, rule in enumerate(TAG_SUGGESTION_RULES)}
    return sorted(scores, key=lambda tag: (-scores[tag], order.get(tag, 999)))[:5]


def tag_hint(paths: Sequence[str]) -> str:
    suggestions = tag_suggestions_for_paths(paths)
    hint = (
        "You need to start the subject with a [Tag], such as "
        f"{', '.join(TAG_EXAMPLES)}. Suggested tag"
        f"{'' if len(suggestions) == 1 else 's'} for the staged paths: "
        f"{', '.join(suggestions)}."
    )
    if paths:
        shown_paths = list(paths[:5])
        path_list = ", ".join(shown_paths)
        if len(paths) > len(shown_paths):
            path_list += f", and {len(paths) - len(shown_paths)} more"
        hint += f" Staged paths considered: {path_list}."
    return hint


def validate_subject_line(
    line_number: int,
    subject: str,
    changed_paths: Sequence[str],
) -> list[CommitMessageDiagnostic]:
    diagnostics: list[CommitMessageDiagnostic] = []
    if not subject:
        diagnostics.append(
            CommitMessageDiagnostic(
                line_number=line_number,
                message="commit message must start with a non-empty subject line",
                text=subject,
                hint=(
                    "Use a short subject such as "
                    f"{tag_suggestions_for_paths(changed_paths)[0]} Describe the change."
                ),
            )
        )
        return diagnostics

    if len(subject) > LINE_LENGTH_LIMIT:
        diagnostics.append(
            CommitMessageDiagnostic(
                line_number=line_number,
                message=(
                    f"subject line is {len(subject)} characters; keep it at or "
                    f"below {LINE_LENGTH_LIMIT}"
                ),
                text=subject,
                hint="Move detail into the body and keep the subject crisp.",
            )
        )

    tag_match = SUBJECT_TAG_PATTERN.match(subject)
    if tag_match is None:
        diagnostics.append(
            CommitMessageDiagnostic(
                line_number=line_number,
                message="subject line must start with a bracketed subsystem tag",
                text=subject,
                hint=tag_hint(changed_paths),
            )
        )
    elif subject[tag_match.end() :].strip() == "":
        diagnostics.append(
            CommitMessageDiagnostic(
                line_number=line_number,
                message="subject line must include a description after the tag",
                text=subject,
                hint=(
                    f"Write the subject as {tag_match.group(0).strip()} "
                    "Describe the change."
                ),
            )
        )
    return diagnostics


def validate_line_lengths(
    text_lines: Sequence[tuple[int, str]],
    subject_line_number: int,
) -> list[CommitMessageDiagnostic]:
    diagnostics: list[CommitMessageDiagnostic] = []
    in_code_block = False
    for line_number, line in text_lines:
        if CODE_FENCE_PATTERN.match(line):
            in_code_block = not in_code_block
            continue
        if in_code_block or line_number == subject_line_number or not line:
            continue
        if len(line) > LINE_LENGTH_LIMIT:
            diagnostics.append(
                CommitMessageDiagnostic(
                    line_number=line_number,
                    message=(
                        f"line is {len(line)} characters; keep commit message "
                        f"body lines at or below {LINE_LENGTH_LIMIT}"
                    ),
                    text=line,
                    hint="Wrap prose or put intentionally long examples in a fenced code block.",
                )
            )
    return diagnostics


def validate_commit_message_text(
    message_text: str,
    changed_paths: Sequence[str] = (),
) -> list[CommitMessageDiagnostic]:
    """Returns diagnostics for commit-message policy violations."""

    diagnostics: list[CommitMessageDiagnostic] = []
    text_lines = commit_message_text_lines(message_text)
    if not text_lines:
        diagnostics.append(
            CommitMessageDiagnostic(
                line_number=1,
                message="commit message must start with a non-empty subject line",
                text="",
                hint=(
                    "Use a short subject such as "
                    f"{tag_suggestions_for_paths(changed_paths)[0]} Describe the change."
                ),
            )
        )
        return diagnostics

    subject_line_number, subject = text_lines[0]
    diagnostics.extend(
        validate_subject_line(subject_line_number, subject, changed_paths)
    )
    diagnostics.extend(validate_line_lengths(text_lines, subject_line_number))

    for line_number, line in text_lines:
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


def validate_commit_message_file(
    path: Path,
    changed_paths: Sequence[str] = (),
) -> list[CommitMessageDiagnostic]:
    """Reads and validates a Git commit message file."""

    return validate_commit_message_text(
        path.read_text(encoding="utf-8"),
        changed_paths=changed_paths,
    )


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
    diagnostics = validate_commit_message_file(
        args.message_file,
        changed_paths=staged_commit_paths(),
    )
    if not diagnostics:
        return 0

    print("commit message policy failed:", file=sys.stderr)
    for diagnostic in diagnostics:
        print(
            f"  {args.message_file}:{diagnostic.line_number}: "
            f"{diagnostic.message}: {diagnostic.text!r}",
            file=sys.stderr,
        )
        if diagnostic.hint:
            print(f"    hint: {diagnostic.hint}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
