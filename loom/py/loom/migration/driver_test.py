# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

from loom.diagnostics import DiagnosticSeverity
from loom.format.bytecode.writer import FORMAT_VERSION, MAGIC
from loom.migration.driver import (
    MigrationRule,
    MigrationRuleApplication,
    check_bytecode_version,
    discover_migration_files,
    migrate_source_text,
)
from loom.migration.source import SourceEdit


def _edit_rule(
    rule_id: str,
    edit_fn,
) -> MigrationRule:
    return MigrationRule(
        rule_id=rule_id,
        rewrite=lambda document: MigrationRuleApplication(edits=(edit_fn(document),)),
    )


def test_source_rule_applies_precision_edit_and_validates_current_output() -> None:
    text = "func.def @f() {\n  func.return\n}\n"

    result = migrate_source_text(
        text,
        filename=Path("input.loom"),
        rules=(
            _edit_rule(
                "rename-function",
                lambda document: SourceEdit(
                    byte_start=document.text.index("@f") + 1,
                    byte_end=document.text.index("@f") + 2,
                    replacement_text="g",
                ),
            ),
        ),
    )

    assert result.ok
    assert result.changed
    assert result.text == "func.def @g() {\n  func.return\n}\n"


def test_source_rules_reject_overlapping_edits() -> None:
    result = migrate_source_text(
        "abcdef",
        rules=(
            _edit_rule("first", lambda _: SourceEdit(1, 4, "BCD")),
            _edit_rule("second", lambda _: SourceEdit(3, 5, "DE")),
        ),
        validate=False,
    )

    assert not result.ok
    assert result.diagnostics[0].severity == DiagnosticSeverity.ERROR
    assert result.diagnostics[0].rule_id == "second"
    assert "ordered and non-overlapping" in result.diagnostics[0].message


def test_source_migration_validates_rewritten_text_with_current_parser() -> None:
    text = "func.def @f() {\n  func.return\n}\n"

    result = migrate_source_text(
        text,
        filename=Path("broken.loom"),
        rules=(
            _edit_rule(
                "break-current-parse",
                lambda document: SourceEdit(
                    byte_start=document.text.index("func.return"),
                    byte_end=document.text.index("func.return") + len("func.return"),
                    replacement_text="not-an-op",
                ),
            ),
        ),
    )

    assert not result.ok
    assert result.diagnostics[0].rule_id == "loom.migrate.current_parse"
    assert result.diagnostics[0].source_range is not None
    assert result.diagnostics[0].source_range.filename == Path("broken.loom")


def test_current_bytecode_version_is_accepted() -> None:
    assert check_bytecode_version(MAGIC + bytes([FORMAT_VERSION])) == ()


def test_old_bytecode_version_reports_actual_and_current_versions() -> None:
    diagnostics = check_bytecode_version(MAGIC + bytes([FORMAT_VERSION - 1]))

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.rule_id == "loom.migrate.bytecode_version"
    assert diagnostic.actual_version == FORMAT_VERSION - 1
    assert diagnostic.current_version == FORMAT_VERSION
    assert "Regenerate" in diagnostic.fixup_hint


def test_discover_migration_files_sorts_supported_kinds(tmp_path: Path) -> None:
    (tmp_path / "z.loom").write_text("", encoding="utf-8")
    (tmp_path / "a.loombc").write_bytes(MAGIC + bytes([FORMAT_VERSION]))
    (tmp_path / "ignored.txt").write_text("", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "m.loom").write_text("", encoding="utf-8")

    assert discover_migration_files(tmp_path) == (
        tmp_path / "a.loombc",
        nested / "m.loom",
        tmp_path / "z.loom",
    )
