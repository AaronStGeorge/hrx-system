# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for AMDGPU callback-registry ownership guardrails."""

from __future__ import annotations

from pathlib import Path

import pytest

from loom.target.arch.amdgpu.contracts.callback_ownership import (
    CallbackDispatchRole,
    CallbackDispatchRow,
    amdgpu_generated_lower_rule_op_kinds,
    parse_callback_dispatch_rows,
    validate_callback_dispatch_rows,
)

_REGISTRY_TABLE_PATH = Path(
    "loom/src/loom/target/arch/amdgpu/lower/registry_tables.inl"
)


def test_parse_callback_dispatch_rows_requires_role_prefix() -> None:
    source = """
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_BITPACK)] =
            LOOM_AMDGPU_DATA_ROW(LOOM_OP_VECTOR_BITPACK, plan_t, select, emit, NULL),
    """

    with pytest.raises(ValueError, match="has no role prefix"):
        parse_callback_dispatch_rows(source)


def test_validate_callback_dispatch_rows_rejects_generated_recipe() -> None:
    rows = (
        CallbackDispatchRow(
            op_kind="LOOM_OP_VECTOR_BITFIELD_EXTRACTU",
            role=CallbackDispatchRole.RECIPE,
            macro_name="RECIPE_DATA_ROW",
        ),
    )

    with pytest.raises(ValueError, match="generated-preselect or legality-only"):
        validate_callback_dispatch_rows(
            rows,
            generated_lower_rule_op_kinds={"LOOM_OP_VECTOR_BITFIELD_EXTRACTU"},
        )


def test_validate_callback_dispatch_rows_accepts_generated_preselect() -> None:
    rows = (
        CallbackDispatchRow(
            op_kind="LOOM_OP_VECTOR_FMAF",
            role=CallbackDispatchRole.GENERATED_PRESELECT,
            macro_name="GENERATED_PRESELECT_DATA_POLICY_ROW",
        ),
    )

    validate_callback_dispatch_rows(
        rows,
        generated_lower_rule_op_kinds={"LOOM_OP_VECTOR_FMAF"},
    )


def test_validate_callback_dispatch_rows_accepts_bounded_recipe() -> None:
    rows = (
        CallbackDispatchRow(
            op_kind="LOOM_OP_VECTOR_BITPACK",
            role=CallbackDispatchRole.RECIPE,
            macro_name="RECIPE_DATA_ROW",
        ),
    )

    validate_callback_dispatch_rows(
        rows,
        generated_lower_rule_op_kinds={"LOOM_OP_VECTOR_FMAF"},
    )


def test_validate_callback_dispatch_rows_rejects_unowned_preselect() -> None:
    rows = (
        CallbackDispatchRow(
            op_kind="LOOM_OP_VECTOR_BITPACK",
            role=CallbackDispatchRole.GENERATED_PRESELECT,
            macro_name="GENERATED_PRESELECT_DATA_POLICY_ROW",
        ),
    )

    with pytest.raises(ValueError, match="fallback owner"):
        validate_callback_dispatch_rows(
            rows,
            generated_lower_rule_op_kinds={"LOOM_OP_VECTOR_FMAF"},
        )


def test_amdgpu_registry_callback_rows_have_generated_ownership() -> None:
    source = _read_repo_file(_REGISTRY_TABLE_PATH)
    rows = parse_callback_dispatch_rows(source)

    validate_callback_dispatch_rows(
        rows,
        generated_lower_rule_op_kinds=amdgpu_generated_lower_rule_op_kinds(),
    )


def _read_repo_file(path: Path) -> str:
    roots = (
        Path.cwd(),
        *Path.cwd().parents,
        Path(__file__).resolve(),
        *Path(__file__).resolve().parents,
    )
    for root in roots:
        candidate = root / path
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError(path)
