# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: Python target contract tables -> compact C ABI tables."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from loom.dsl import Op
from loom.gen.generated_file import line_comment_header
from loom.target.contracts import (
    CONTRACT_ROW_NONE,
    CompiledContractTable,
    ContractSystem,
    ContractTable,
    compile_contract_table,
)

_UINT8_MAX = 0xFF
_UINT16_MAX = 0xFFFF

_CONTRACT_SYSTEM_C_NAMES = {
    ContractSystem.DESCRIPTOR_RULE: "LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_RULE",
    ContractSystem.VALUE_ALIAS: "LOOM_TARGET_CONTRACT_SYSTEM_VALUE_ALIAS",
    ContractSystem.SOURCE_MEMORY: "LOOM_TARGET_CONTRACT_SYSTEM_SOURCE_MEMORY",
    ContractSystem.ENVIRONMENT: "LOOM_TARGET_CONTRACT_SYSTEM_ENVIRONMENT",
    ContractSystem.DESCRIPTOR_MATRIX: "LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_MATRIX",
    ContractSystem.CUSTOM_FAMILY: "LOOM_TARGET_CONTRACT_SYSTEM_CUSTOM_FAMILY",
}


@dataclass(frozen=True, slots=True)
class GeneratedContractTable:
    """Generated C/H contents for one target contract table."""

    header: str
    source: str


def generate_contract_table(
    table: ContractTable,
    *,
    dialect_ops: Mapping[str, Sequence[Op]],
) -> GeneratedContractTable:
    """Generates C/H text for a compact target contract table."""

    compiled = compile_contract_table(table, dialect_ops=dialect_ops)
    _validate_c_table_shape(compiled)
    public_header = _require_table_field(table, "public_header")
    symbol_name = _require_table_field(table, "symbol_name")
    c_table_prefix = _require_table_field(table, "c_table_prefix")
    header_guard = table.header_guard or _header_guard_from_public_header(public_header)
    return GeneratedContractTable(
        header=_generate_header(
            header_guard=header_guard,
            symbol_name=symbol_name,
        ),
        source=_generate_source(
            table=compiled,
            public_header=public_header,
            symbol_name=symbol_name,
            c_table_prefix=_c_identifier(c_table_prefix),
        ),
    )


def write_contract_table_to_paths(
    table: ContractTable,
    *,
    dialect_ops: Mapping[str, Sequence[Op]],
    header_path: Path,
    source_path: Path,
) -> None:
    """Writes generated C/H contents for one target contract table."""

    generated = generate_contract_table(table, dialect_ops=dialect_ops)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(generated.header, encoding="utf-8")
    source_path.write_text(generated.source, encoding="utf-8")


def _generate_header(*, header_guard: str, symbol_name: str) -> str:
    lines: list[str] = []
    lines.extend(
        line_comment_header(
            "//",
            generator="loom/py/loom/gen/target_contract_tables.py",
        )
    )
    lines.extend(
        [
            "",
            f"#ifndef {header_guard}",
            f"#define {header_guard}",
            "",
            '#include "loom/target/contract.h"',
            "",
            "#ifdef __cplusplus",
            'extern "C" {',
            "#endif",
            "",
            f"extern const loom_target_contract_table_t {symbol_name};",
            "",
            "#ifdef __cplusplus",
            '}  // extern "C"',
            "#endif",
            "",
            f"#endif  // {header_guard}",
            "",
        ]
    )
    return "\n".join(lines)


def _generate_source(
    *,
    table: CompiledContractTable,
    public_header: str,
    symbol_name: str,
    c_table_prefix: str,
) -> str:
    lines: list[str] = []
    lines.extend(
        line_comment_header(
            "//",
            generator="loom/py/loom/gen/target_contract_tables.py",
        )
    )
    lines.extend(
        [
            "",
            f'#include "{public_header}"',
            "",
            "#include <stddef.h>",
            "",
        ]
    )

    dialect_table_names: list[str | None] = []
    for dialect in table.dialects:
        if not dialect.op_entries:
            dialect_table_names.append(None)
            continue
        array_name = f"k{c_table_prefix}{_pascal_identifier(dialect.dialect_name)}OpEntries"
        dialect_table_names.append(array_name)
        lines.append(f"static const loom_target_contract_op_entry_t {array_name}[] = {{")
        for entry in dialect.op_entries:
            case_start = "LOOM_TARGET_CONTRACT_ROW_NONE" if entry.case_start == CONTRACT_ROW_NONE else str(entry.case_start)
            lines.append(f"    {{{case_start}, {entry.case_count}}},")
        lines.extend(["};", ""])

    dialects_name = f"k{c_table_prefix}Dialects"
    if table.dialects:
        lines.append(f"static const loom_target_contract_dialect_table_t {dialects_name}[] = {{")
        for dialect, op_entries_name in zip(table.dialects, dialect_table_names, strict=True):
            if op_entries_name is None:
                lines.append("    {0, NULL},")
            else:
                lines.append(f"    {{{len(dialect.op_entries)}, {op_entries_name}}},")
        lines.extend(["};", ""])
        dialects_value = dialects_name
    else:
        dialects_value = "NULL"

    cases_name = f"k{c_table_prefix}Cases"
    if table.cases:
        lines.append(f"static const loom_target_contract_case_t {cases_name}[] = {{")
        for contract_case in table.cases:
            system_name = _CONTRACT_SYSTEM_C_NAMES[contract_case.system]
            row_index = "LOOM_TARGET_CONTRACT_ROW_NONE" if contract_case.row_index == CONTRACT_ROW_NONE else str(contract_case.row_index)
            lines.append(f"    {{{system_name}, 0, {row_index}}},")
        lines.extend(["};", ""])
        cases_value = cases_name
    else:
        cases_value = "NULL"

    descriptor_rules_name = f"k{c_table_prefix}DescriptorRules"
    if table.descriptor_rules:
        lines.append(f"static const loom_target_contract_descriptor_rule_t {descriptor_rules_name}[] = {{")
        lines.extend(f"    {{{descriptor_rule.rule_set_index}, {descriptor_rule.rule_index}}}," for descriptor_rule in table.descriptor_rules)
        lines.extend(["};", ""])
        descriptor_rules_value = descriptor_rules_name
    else:
        descriptor_rules_value = "NULL"

    lines.extend(
        [
            f"const loom_target_contract_table_t {symbol_name} = {{",
            f"    {table.table_index},",
            f"    {table.dialect_base_id},",
            f"    {len(table.dialects)},",
            f"    {dialects_value},",
            f"    {len(table.cases)},",
            f"    {cases_value},",
            f"    {len(table.descriptor_rules)},",
            f"    {descriptor_rules_value},",
            "};",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_c_table_shape(table: CompiledContractTable) -> None:
    if table.table_index > _UINT16_MAX:
        raise ValueError(f"contract table '{table.name}' table index exceeds uint16_t")
    if table.dialect_base_id > _UINT8_MAX:
        raise ValueError(f"contract table '{table.name}' dialect base exceeds uint8_t")
    if len(table.dialects) > _UINT8_MAX:
        raise ValueError(f"contract table '{table.name}' dialect count exceeds uint8_t")
    if len(table.cases) > _UINT16_MAX:
        raise ValueError(f"contract table '{table.name}' case count exceeds uint16_t")
    if len(table.descriptor_rules) > _UINT16_MAX:
        raise ValueError(f"contract table '{table.name}' descriptor-rule count exceeds uint16_t")
    for dialect in table.dialects:
        if len(dialect.op_entries) > _UINT16_MAX:
            raise ValueError(f"contract table '{table.name}' dialect '{dialect.dialect_name}' op count exceeds uint16_t")


def _require_table_field(table: ContractTable, field_name: str) -> str:
    value = getattr(table, field_name)
    if not value:
        raise ValueError(f"contract table '{table.name}' requires {field_name}")
    return str(value)


def _header_guard_from_public_header(public_header: str) -> str:
    return _c_identifier(public_header).upper() + "_"


def _pascal_identifier(value: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in _identifier_parts(value))


def _c_identifier(value: str) -> str:
    parts = _identifier_parts(value)
    if not parts:
        return "_"
    identifier = "_".join(parts)
    if identifier[0].isdigit():
        return "_" + identifier
    return identifier


def _identifier_parts(value: str) -> tuple[str, ...]:
    return tuple(part for part in re.split(r"[^0-9A-Za-z]+", value) if part)
