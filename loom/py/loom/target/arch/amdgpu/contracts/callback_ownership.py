# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU callback-registry ownership checks."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, unique

from loom.dsl import Op
from loom.target.contract_fragments import (
    ContractFragmentRegistration,
    iter_contract_fragment_registrations,
)
from loom.target.contracts import compile_lower_rule_set


@unique
class CallbackDispatchRole(Enum):
    """Ownership class declared by an AMDGPU callback dispatch row macro."""

    VALUE = "value"
    MEMORY = "memory"
    RECIPE = "recipe"
    GENERATED_PRESELECT = "generated_preselect"
    LEGALITY = "legality"


@dataclass(frozen=True, slots=True)
class CallbackDispatchRow:
    """Parsed callback dispatch row from the AMDGPU registry table."""

    op_kind: str
    role: CallbackDispatchRole
    macro_name: str


_ROW_RE = re.compile(
    r"\[LOOM_AMDGPU_OP_INDEX\((LOOM_OP_[A-Z0-9_]+)\)\]\s*=\s*"
    r"LOOM_AMDGPU_([A-Z0-9_]+ROW)\("
)


def amdgpu_generated_lower_rule_op_kinds(
    *,
    registrations: Iterable[ContractFragmentRegistration] | None = None,
) -> frozenset[str]:
    """Returns C op-kind names owned by generated AMDGPU lower-rule rows."""

    op_kinds: set[str] = set()
    if registrations is None:
        registrations = iter_contract_fragment_registrations()
    for registration in registrations:
        if not registration.key.startswith("amdgpu."):
            continue
        fragment = registration.load()
        compiled = compile_lower_rule_set(
            fragment, dialect_ops=registration.load_dialect_ops()
        )
        for rule in compiled.rules:
            op_kinds.add(_op_kind_c_name(rule.source_op))
    return frozenset(op_kinds)


def parse_callback_dispatch_rows(source: str) -> tuple[CallbackDispatchRow, ...]:
    """Parses classified AMDGPU callback dispatch rows from registry source."""

    rows: list[CallbackDispatchRow] = []
    for match in _ROW_RE.finditer(source):
        macro_name = match.group(2)
        rows.append(
            CallbackDispatchRow(
                op_kind=match.group(1),
                role=_callback_dispatch_role(macro_name),
                macro_name=macro_name,
            )
        )
    return tuple(rows)


def validate_callback_dispatch_rows(
    rows: Iterable[CallbackDispatchRow],
    *,
    generated_lower_rule_op_kinds: Iterable[str],
) -> None:
    """Validates callback rows against Python-authored generated ownership."""

    generated_op_kinds = frozenset(generated_lower_rule_op_kinds)
    bad_generated_callbacks = tuple(
        row
        for row in rows
        if row.op_kind in generated_op_kinds
        and row.role
        not in (
            CallbackDispatchRole.GENERATED_PRESELECT,
            CallbackDispatchRole.LEGALITY,
        )
    )
    if bad_generated_callbacks:
        raise ValueError(
            "AMDGPU callback rows for generated lower-rule ops must be "
            "generated-preselect or legality-only rows: "
            + _format_rows(bad_generated_callbacks)
        )

    bad_preselect_callbacks = tuple(
        row
        for row in rows
        if row.role == CallbackDispatchRole.GENERATED_PRESELECT
        and row.op_kind not in generated_op_kinds
    )
    if bad_preselect_callbacks:
        raise ValueError(
            "AMDGPU generated-preselect callback rows need a generated lower-rule "
            "fallback owner: " + _format_rows(bad_preselect_callbacks)
        )


def _callback_dispatch_role(macro_name: str) -> CallbackDispatchRole:
    if macro_name == "LEGALITY_ROW":
        return CallbackDispatchRole.LEGALITY
    if macro_name.startswith("GENERATED_PRESELECT_"):
        return CallbackDispatchRole.GENERATED_PRESELECT
    if macro_name.startswith("VALUE_"):
        return CallbackDispatchRole.VALUE
    if macro_name.startswith("MEMORY_"):
        return CallbackDispatchRole.MEMORY
    if macro_name.startswith("RECIPE_"):
        return CallbackDispatchRole.RECIPE
    raise ValueError(f"AMDGPU callback row macro '{macro_name}' has no role prefix")


def _format_rows(rows: Iterable[CallbackDispatchRow]) -> str:
    return ", ".join(f"{row.op_kind} via {row.macro_name}" for row in rows)


def _op_kind_c_name(op: Op) -> str:
    return "LOOM_OP_" + op.name.replace(".", "_").upper()
