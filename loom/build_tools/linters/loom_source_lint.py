#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Checks Loom C/C++ source invariants that are easy to regress.

This is intentionally a small guardrail, not a general C parser. It catches the
module-value-cardinality storage pattern that is especially toxic for Loom's
function-local compiler phases: allocating or resetting scratch sized by
`module->values.count`/`capacity` instead of using function-local domains,
maintained facts, or reviewed module-owned structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
LOOM_SOURCE_ROOT = REPO_ROOT / "loom" / "src" / "loom"
SOURCE_SUFFIXES = {".c", ".cc", ".h"}

# Matches common module pointers and nested owner paths ending in module, such
# as `module->values.count`, `source_module->values.capacity`, and
# `state->module->values.count`.
MODULE_VALUE_CARDINALITY_PATTERN = re.compile(
    r"\b(?:[A-Za-z_][A-Za-z0-9_]*->)*"
    r"(?:source_|target_|input_|output_)?module->values\.(?:count|capacity)\b"
)

STORAGE_OR_RESET_PATTERNS = [
    re.compile(r"\biree_arena_allocate(?:_array)?\s*\("),
    re.compile(r"\biree_arena_grow_array\s*\("),
    re.compile(r"\biree_allocator_(?:malloc|realloc|calloc|malloc_aligned)\s*\("),
    re.compile(r"\bmemset\s*\("),
    re.compile(r"\biree_bitfield_[A-Za-z0-9_]*\s*\("),
]

# Local aliases are almost as dangerous as direct allocations: they make the
# storage pattern invisible to line-oriented review and usually precede dense
# tables sized by the whole module.
CARDINALITY_ALIAS_PATTERN = re.compile(
    r"\b(?:value_count|value_capacity|capacity|mask_count|"
    r"defined_bits_length|consuming_op_count|source_value_snapshot_count)\b"
    r"\s*(?:=|\+=)"
)


@dataclass(frozen=True)
class ApprovedStatement:
    """A reviewed statement that intentionally mentions module value cardinality."""

    path: str
    pattern: re.Pattern[str]
    reason: str


APPROVED_STATEMENTS = [
    ApprovedStatement(
        "loom/src/loom/ir/module.c",
        re.compile(r"\bmodule->values\.capacity\b"),
        "core module value-table and module-owned scratch storage",
    ),
    ApprovedStatement(
        "loom/src/loom/pass/interpreter.c",
        re.compile(r"\.value_count\s*=\s*module->values\.count"),
        "pass interpreter epoch snapshot records IR mutation boundaries",
    ),
    ApprovedStatement(
        "loom/src/loom/pass/value_facts.c",
        re.compile(r"\bmodule->values\.capacity\b"),
        "maintained scoped value-fact owner storage",
    ),
    ApprovedStatement(
        "loom/src/loom/passes/refine_boundaries.c",
        re.compile(r"\bboundary_fact_value_capacity\s*=\s*module->values\.capacity"),
        "persistent boundary fact storage owned by the refine-boundaries pass",
    ),
    ApprovedStatement(
        "loom/src/loom/rewrite/rewriter.c",
        re.compile(r"\bvalue_count\s*=\s*rewriter->module->values\.count"),
        "rewriter checkpoint captures mutation boundaries, not scratch storage",
    ),
    ApprovedStatement(
        "loom/src/loom/rewrite/remap.c",
        re.compile(r"\.source_value_snapshot_count\s*=\s*source_module->values\.count"),
        "IR remap snapshot records source-module mutation boundaries",
    ),
    ApprovedStatement(
        "loom/src/loom/target/emit/llvmir/lower/registry.c",
        re.compile(r"\bsource_module->values\.count\b"),
        "LLVMIR lowering maintains explicit whole-module emission state",
    ),
    ApprovedStatement(
        "loom/src/loom/verify/verify.c",
        re.compile(r"\bmodule->values\.count\b"),
        "verifier masks over arbitrary user-provided IR",
    ),
    ApprovedStatement(
        "loom/src/loom/codegen/low/verify.c",
        re.compile(r"\bmodule->values\.count\b"),
        "low verifier masks over arbitrary user-provided IR",
    ),
]


@dataclass(frozen=True)
class Finding:
    """A lint finding with enough context for source review."""

    path: Path
    line: int
    statement: str


def _relative_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _strip_line_comments(statement: str) -> str:
    return "\n".join(line.split("//", 1)[0] for line in statement.splitlines())


def _iter_statements(path: Path) -> list[tuple[int, str]]:
    statements: list[tuple[int, str]] = []
    start_line = 0
    pending: list[str] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not pending:
            start_line = line_number
        pending.append(line.rstrip())
        stripped = line.strip()
        if ";" in stripped or stripped.endswith("{") or stripped.endswith("}"):
            statements.append((start_line, "\n".join(pending)))
            pending = []
    if pending:
        statements.append((start_line, "\n".join(pending)))
    return statements


def _is_approved(path: Path, statement: str) -> bool:
    relative_path = _relative_path(path)
    for approved in APPROVED_STATEMENTS:
        if relative_path == approved.path and approved.pattern.search(statement):
            return True
    return False


def _is_suspicious(statement: str) -> bool:
    uncommented = _strip_line_comments(statement)
    if not MODULE_VALUE_CARDINALITY_PATTERN.search(uncommented):
        return False
    if any(pattern.search(uncommented) for pattern in STORAGE_OR_RESET_PATTERNS):
        return True
    return CARDINALITY_ALIAS_PATTERN.search(uncommented) is not None


def _scan_file(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    for line, statement in _iter_statements(path):
        if _is_suspicious(statement) and not _is_approved(path, statement):
            findings.append(Finding(path=path, line=line, statement=statement))
    return findings


def _iter_source_files() -> list[Path]:
    return sorted(
        path
        for path in LOOM_SOURCE_ROOT.rglob("*")
        if path.is_file() and path.suffix in SOURCE_SUFFIXES
    )


def _format_statement(statement: str) -> str:
    stripped_lines = [line.strip() for line in statement.splitlines()]
    return " ".join(line for line in stripped_lines if line)


def main() -> int:
    findings: list[Finding] = []
    for path in _iter_source_files():
        findings.extend(_scan_file(path))

    if not findings:
        print("loom-source-lint: PASS")
        return 0

    print(
        "loom-source-lint: FAIL: module-value-cardinality storage/reset "
        "requires reviewed infrastructure."
    )
    print(
        "Use function-local value domains, maintained facts, or an explicitly "
        "reviewed module-owned structure instead of local scratch sized by "
        "module->values.count/capacity."
    )
    for finding in findings:
        print(
            f"{_relative_path(finding.path)}:{finding.line}: "
            f"{_format_statement(finding.statement)}"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
