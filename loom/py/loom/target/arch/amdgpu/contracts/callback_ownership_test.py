# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for AMDGPU callback-registry ownership guardrails."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from loom.target.arch.amdgpu.contracts.callback_ownership import (
    CallbackDispatchRole,
    CallbackDispatchRow,
    amdgpu_generated_lower_rule_op_kinds,
    callback_dispatch_policy_names_by_kind,
    callback_dispatch_row_macro_names,
    parse_callback_dispatch_rows,
    validate_callback_dispatch_rows,
)

_REGISTRY_SOURCE_PATH = Path("loom/src/loom/target/arch/amdgpu/lower/registry.c")
_REGISTRY_TABLE_PATH = Path(
    "loom/src/loom/target/arch/amdgpu/lower/registry_tables.inl"
)

_POLICY_ENUM_RE = re.compile(
    r"enum loom_amdgpu_(storage|preselect|report)_policy_e\s*\{(?P<body>.*?)\};",
    re.DOTALL,
)
_POLICY_NAME_RE = re.compile(
    r"\b(LOOM_AMDGPU_(?:STORAGE|PRESELECT|REPORT)_[A-Z0-9_]+)\b"
)
_PUBLIC_ROW_MACRO_RE = re.compile(r"#define LOOM_AMDGPU_([A-Z0-9_]+ROW)\b")
_POLICY_SENTINELS_BY_KIND = {
    "storage": frozenset(
        {"LOOM_AMDGPU_STORAGE_SOURCE_OPERANDS", "LOOM_AMDGPU_STORAGE_MAX"}
    ),
    "preselect": frozenset({"LOOM_AMDGPU_PRESELECT_NONE", "LOOM_AMDGPU_PRESELECT_MAX"}),
    "report": frozenset({"LOOM_AMDGPU_REPORT_NONE", "LOOM_AMDGPU_REPORT_MAX"}),
}


def test_parse_callback_dispatch_rows_requires_role_prefix() -> None:
    source = """
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_BITPACK)] =
            LOOM_AMDGPU_DATA_ROW(LOOM_OP_VECTOR_BITPACK, plan_t, select, emit, NULL),
    """

    with pytest.raises(ValueError, match="has no role prefix"):
        parse_callback_dispatch_rows(source)


def test_parse_callback_dispatch_rows_captures_arguments() -> None:
    source = """
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_FMAF)] =
            LOOM_AMDGPU_GENERATED_PRESELECT_DATA_POLICY_ROW(
                LOOM_OP_VECTOR_FMAF, loom_amdgpu_packed_ternary_plan_t,
                loom_amdgpu_select_vector_packed_fmaf_dispatch,
                loom_amdgpu_emit_vector_packed_ternary_dispatch, NULL,
                LOOM_AMDGPU_STORAGE_PLAN_SOURCE_ARRAY_3,
                LOOM_AMDGPU_PRESELECT_PLAN_ID_FMA_DIAGNOSTIC),
    """

    rows = parse_callback_dispatch_rows(source)

    assert rows == (
        CallbackDispatchRow(
            op_kind="LOOM_OP_VECTOR_FMAF",
            role=CallbackDispatchRole.GENERATED_PRESELECT,
            macro_name="GENERATED_PRESELECT_DATA_POLICY_ROW",
            arguments=(
                "LOOM_OP_VECTOR_FMAF",
                "loom_amdgpu_packed_ternary_plan_t",
                "loom_amdgpu_select_vector_packed_fmaf_dispatch",
                "loom_amdgpu_emit_vector_packed_ternary_dispatch",
                "NULL",
                "LOOM_AMDGPU_STORAGE_PLAN_SOURCE_ARRAY_3",
                "LOOM_AMDGPU_PRESELECT_PLAN_ID_FMA_DIAGNOSTIC",
            ),
        ),
    )


def test_validate_callback_dispatch_rows_rejects_unknown_schema() -> None:
    rows = (
        CallbackDispatchRow(
            op_kind="LOOM_OP_VECTOR_BITPACK",
            role=CallbackDispatchRole.VALUE,
            macro_name="VALUE_UNKNOWN_ROW",
            arguments=("LOOM_OP_VECTOR_BITPACK",),
        ),
    )

    with pytest.raises(ValueError, match="no registered dispatch-row schema"):
        validate_callback_dispatch_rows(rows, generated_lower_rule_op_kinds=())


def test_validate_callback_dispatch_rows_rejects_op_kind_mismatch() -> None:
    rows = (
        CallbackDispatchRow(
            op_kind="LOOM_OP_VECTOR_BITPACK",
            role=CallbackDispatchRole.RECIPE,
            macro_name="RECIPE_DATA_ROW",
            arguments=(
                "LOOM_OP_VECTOR_BITUNPACKU",
                "loom_amdgpu_bitpack_plan_t",
                "select",
                "emit",
                "verify",
            ),
        ),
    )

    with pytest.raises(ValueError, match="does not match macro op-kind argument"):
        validate_callback_dispatch_rows(rows, generated_lower_rule_op_kinds=())


def test_validate_callback_dispatch_rows_rejects_wrong_policy_namespace() -> None:
    rows = (
        CallbackDispatchRow(
            op_kind="LOOM_OP_VECTOR_FMAF",
            role=CallbackDispatchRole.GENERATED_PRESELECT,
            macro_name="GENERATED_PRESELECT_DATA_POLICY_ROW",
            arguments=(
                "LOOM_OP_VECTOR_FMAF",
                "loom_amdgpu_packed_ternary_plan_t",
                "select",
                "emit",
                "verify",
                "LOOM_AMDGPU_PRESELECT_PLAN_ID",
                "LOOM_AMDGPU_PRESELECT_PLAN_ID_FMA_DIAGNOSTIC",
            ),
        ),
    )

    with pytest.raises(ValueError, match="expects a storage policy"):
        validate_callback_dispatch_rows(
            rows, generated_lower_rule_op_kinds={"LOOM_OP_VECTOR_FMAF"}
        )


def test_validate_callback_dispatch_rows_rejects_wrong_report_policy_namespace() -> (
    None
):
    rows = (
        CallbackDispatchRow(
            op_kind="LOOM_OP_KERNEL_WORKGROUP_REDUCE",
            role=CallbackDispatchRole.RECIPE,
            macro_name="RECIPE_DATA_STORAGE_REPORT_ROW",
            arguments=(
                "LOOM_OP_KERNEL_WORKGROUP_REDUCE",
                "loom_amdgpu_workgroup_reduce_plan_t",
                "select",
                "emit",
                "verify",
                "LOOM_AMDGPU_STORAGE_PLAN_SOURCE_ARRAY_1",
                "LOOM_AMDGPU_STORAGE_PLAN_SOURCE_ARRAY_2",
            ),
        ),
    )

    with pytest.raises(ValueError, match="expects a report policy"):
        validate_callback_dispatch_rows(rows, generated_lower_rule_op_kinds=())


def test_validate_callback_dispatch_rows_accepts_report_policy() -> None:
    rows = (
        CallbackDispatchRow(
            op_kind="LOOM_OP_KERNEL_WORKGROUP_REDUCE",
            role=CallbackDispatchRole.RECIPE,
            macro_name="RECIPE_DATA_STORAGE_REPORT_ROW",
            arguments=(
                "LOOM_OP_KERNEL_WORKGROUP_REDUCE",
                "loom_amdgpu_workgroup_reduce_plan_t",
                "select",
                "emit",
                "verify",
                "LOOM_AMDGPU_STORAGE_PLAN_SOURCE_ARRAY_1",
                "LOOM_AMDGPU_REPORT_WORKGROUP_REDUCE_PUBLICATION",
            ),
        ),
    )

    validate_callback_dispatch_rows(rows, generated_lower_rule_op_kinds=())


def test_callback_policy_names_match_registry_enums() -> None:
    registry_source = _read_repo_file(_REGISTRY_SOURCE_PATH)

    assert callback_dispatch_policy_names_by_kind() == _parse_registry_policy_enums(
        registry_source
    )


def test_callback_row_macro_schemas_match_registry_macros() -> None:
    registry_source = _read_repo_file(_REGISTRY_SOURCE_PATH)

    assert callback_dispatch_row_macro_names() == _parse_public_row_macros(
        registry_source
    )


def test_validate_callback_dispatch_rows_rejects_policy_in_non_policy_slot() -> None:
    rows = (
        CallbackDispatchRow(
            op_kind="LOOM_OP_VECTOR_CONSTANT",
            role=CallbackDispatchRole.VALUE,
            macro_name="VALUE_DIRECT_ROW",
            arguments=(
                "LOOM_OP_VECTOR_CONSTANT",
                "LOOM_AMDGPU_STORAGE_VALUE_PLAN",
                "emit",
                "verify",
            ),
        ),
    )

    with pytest.raises(ValueError, match=r"policy token .* in non-policy argument"):
        validate_callback_dispatch_rows(rows, generated_lower_rule_op_kinds=())


def test_validate_callback_dispatch_rows_rejects_generated_recipe() -> None:
    rows = (
        CallbackDispatchRow(
            op_kind="LOOM_OP_VECTOR_BITFIELD_EXTRACTU",
            role=CallbackDispatchRole.RECIPE,
            macro_name="RECIPE_DATA_ROW",
            arguments=(
                "LOOM_OP_VECTOR_BITFIELD_EXTRACTU",
                "loom_amdgpu_bitfield_extract_plan_t",
                "select",
                "emit",
                "verify",
            ),
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
            arguments=(
                "LOOM_OP_VECTOR_FMAF",
                "loom_amdgpu_packed_ternary_plan_t",
                "select",
                "emit",
                "verify",
                "LOOM_AMDGPU_STORAGE_PLAN_SOURCE_ARRAY_3",
                "LOOM_AMDGPU_PRESELECT_PLAN_ID",
            ),
        ),
    )

    validate_callback_dispatch_rows(
        rows,
        generated_lower_rule_op_kinds={"LOOM_OP_VECTOR_FMAF"},
    )


def test_validate_callback_dispatch_rows_accepts_generated_direct_preselect() -> None:
    rows = (
        CallbackDispatchRow(
            op_kind="LOOM_OP_INDEX_ADD",
            role=CallbackDispatchRole.GENERATED_PRESELECT,
            macro_name="GENERATED_PRESELECT_DIRECT_POLICY_ROW",
            arguments=(
                "LOOM_OP_INDEX_ADD",
                "select",
                "emit",
                "verify",
                "LOOM_AMDGPU_STORAGE_VALUE_PLAN",
                "LOOM_AMDGPU_PRESELECT_PLAN_ID",
            ),
        ),
    )

    validate_callback_dispatch_rows(
        rows,
        generated_lower_rule_op_kinds={"LOOM_OP_INDEX_ADD"},
    )


def test_validate_callback_dispatch_rows_accepts_bounded_recipe() -> None:
    rows = (
        CallbackDispatchRow(
            op_kind="LOOM_OP_VECTOR_BITPACK",
            role=CallbackDispatchRole.RECIPE,
            macro_name="RECIPE_DATA_ROW",
            arguments=(
                "LOOM_OP_VECTOR_BITPACK",
                "loom_amdgpu_bitpack_plan_t",
                "select",
                "emit",
                "verify",
            ),
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
            arguments=(
                "LOOM_OP_VECTOR_BITPACK",
                "loom_amdgpu_bitpack_plan_t",
                "select",
                "emit",
                "verify",
                "LOOM_AMDGPU_STORAGE_PLAN_SOURCE_ARRAY_3",
                "LOOM_AMDGPU_PRESELECT_PLAN_ID",
            ),
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


def _parse_registry_policy_enums(source: str) -> dict[str, frozenset[str]]:
    policy_names_by_kind: dict[str, frozenset[str]] = {}
    for match in _POLICY_ENUM_RE.finditer(source):
        kind = match.group(1)
        policy_names = frozenset(_POLICY_NAME_RE.findall(match.group("body")))
        policy_names_by_kind[kind] = policy_names - _POLICY_SENTINELS_BY_KIND[kind]
    assert policy_names_by_kind.keys() == _POLICY_SENTINELS_BY_KIND.keys()
    return policy_names_by_kind


def _parse_public_row_macros(source: str) -> frozenset[str]:
    dispatch_macro_region = source.split(
        '#include "loom/target/arch/amdgpu/lower/registry_tables.inl"', maxsplit=1
    )[0]
    return frozenset(
        macro_name
        for macro_name in _PUBLIC_ROW_MACRO_RE.findall(dispatch_macro_region)
        if not macro_name.startswith("INTERNAL_")
    )
