# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

from loom.migration.driver import migrate_loom_test_text, migrate_source_text
from loom.migration.rules import BUFFER_ASSUME_MEMORY_SPACE_ATTR_DICT_RULE_ID


def test_buffer_assume_memory_space_attr_dict_rewrites_source() -> None:
    text = (
        "func.def @f(%buffer: buffer) -> (buffer) {\n"
        "  %global = buffer.assume.memory_space %buffer "
        "{memory_space = global} : buffer\n"
        "  func.return %global : buffer\n"
        "}\n"
    )

    result = migrate_source_text(text, filename=Path("input.loom"))

    assert result.ok
    assert result.changed
    assert result.text == (
        "func.def @f(%buffer: buffer) -> (buffer) {\n"
        "  %global = buffer.assume.memory_space<global> %buffer : buffer\n"
        "  func.return %global : buffer\n"
        "}\n"
    )


def test_buffer_assume_memory_space_current_syntax_is_noop() -> None:
    text = (
        "func.def @f(%buffer: buffer) -> (buffer) {\n"
        "  %global = buffer.assume.memory_space<global> %buffer : buffer\n"
        "  func.return %global : buffer\n"
        "}\n"
    )

    result = migrate_source_text(text, filename=Path("input.loom"))

    assert result.ok
    assert not result.changed
    assert result.text == text


def test_buffer_assume_memory_space_rule_does_not_tokenize_unrelated_source() -> None:
    text = "config.decl @tuner.workgroup_size : %value: index\n"

    result = migrate_source_text(
        text,
        filename=Path("unrelated.loom"),
        validate=False,
    )

    assert result.ok
    assert not result.changed


def test_buffer_assume_memory_space_rewrite_is_idempotent() -> None:
    text = (
        "func.def @f(%buffer: buffer) -> (buffer) {\n"
        "  %global = buffer.assume.memory_space %buffer "
        "{memory_space = global} : buffer\n"
        "  func.return %global : buffer\n"
        "}\n"
    )

    first_result = migrate_source_text(text, filename=Path("input.loom"))
    second_result = migrate_source_text(first_result.text, filename=Path("input.loom"))

    assert first_result.ok
    assert first_result.changed
    assert second_result.ok
    assert not second_result.changed


def test_buffer_assume_memory_space_malformed_legacy_syntax_reports_rule() -> None:
    text = (
        "func.def @f(%buffer: buffer) -> (buffer) {\n"
        "  %global = buffer.assume.memory_space %buffer {space = global} : buffer\n"
        "  func.return %global : buffer\n"
        "}\n"
    )

    result = migrate_source_text(text, filename=Path("broken.loom"))

    assert not result.ok
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.rule_id == BUFFER_ASSUME_MEMORY_SPACE_ATTR_DICT_RULE_ID
    assert diagnostic.source_range is not None
    assert diagnostic.source_range.filename == Path("broken.loom")
    assert "expected legacy" in diagnostic.message


def test_buffer_assume_memory_space_rewrites_loom_test_input_only() -> None:
    text = (
        "// RUN: roundtrip\n"
        "\n"
        "func.def @f(%buffer: buffer) -> (buffer) {\n"
        "  %global = buffer.assume.memory_space %buffer "
        "{memory_space = global} : buffer\n"
        "  func.return %global : buffer\n"
        "}\n"
        "// ----\n"
        "func.def @f(%buffer: buffer) -> (buffer) {\n"
        "  %global = buffer.assume.memory_space %buffer "
        "{memory_space = global} : buffer\n"
        "  func.return %global : buffer\n"
        "}\n"
    )

    result = migrate_loom_test_text(text, filename=Path("case.loom-test"))

    assert result.ok
    assert result.changed
    assert result.text == (
        "// RUN: roundtrip\n"
        "\n"
        "func.def @f(%buffer: buffer) -> (buffer) {\n"
        "  %global = buffer.assume.memory_space<global> %buffer : buffer\n"
        "  func.return %global : buffer\n"
        "}\n"
        "// ----\n"
        "func.def @f(%buffer: buffer) -> (buffer) {\n"
        "  %global = buffer.assume.memory_space %buffer "
        "{memory_space = global} : buffer\n"
        "  func.return %global : buffer\n"
        "}\n"
    )
