# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: Python target contracts -> target-low lower-rule C tables."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from loom.dsl import Op
from loom.gen.generated_file import line_comment_header
from loom.target.contracts import (
    LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS,
    LOWER_EMIT_FLAG_RESULT_TYPE_PATTERN,
    LOWER_EMIT_FLAG_SWAP_OPERANDS_0_1,
    CompiledLowerRuleSet,
    ContractTable,
    GuardKind,
    LowerAttrCopy,
    LowerAttrCopyKind,
    LowerEmit,
    LowerEmitKind,
    LowerGuard,
    LowerRule,
    LowerRuleSpan,
    LowerTiedResult,
    LowerValueRef,
    SourceValueKind,
    TypePattern,
    compile_lower_rule_set,
)
from loom.target.low_descriptors import Descriptor, descriptor_set_relative_name

_SCALAR_TYPE_C_NAMES = {
    "index": "LOOM_SCALAR_TYPE_INDEX",
    "offset": "LOOM_SCALAR_TYPE_OFFSET",
    "i1": "LOOM_SCALAR_TYPE_I1",
    "i8": "LOOM_SCALAR_TYPE_I8",
    "i16": "LOOM_SCALAR_TYPE_I16",
    "i32": "LOOM_SCALAR_TYPE_I32",
    "i64": "LOOM_SCALAR_TYPE_I64",
    "f8E4M3": "LOOM_SCALAR_TYPE_F8E4M3",
    "f8E5M2": "LOOM_SCALAR_TYPE_F8E5M2",
    "f16": "LOOM_SCALAR_TYPE_F16",
    "bf16": "LOOM_SCALAR_TYPE_BF16",
    "f32": "LOOM_SCALAR_TYPE_F32",
    "f64": "LOOM_SCALAR_TYPE_F64",
}

_VALUE_REF_KIND_C_NAMES = {
    SourceValueKind.OPERAND: "LOOM_LOW_LOWER_VALUE_REF_OPERAND",
    SourceValueKind.RESULT: "LOOM_LOW_LOWER_VALUE_REF_RESULT",
    SourceValueKind.TEMPORARY: "LOOM_LOW_LOWER_VALUE_REF_TEMPORARY",
}

_ATTR_KIND_C_NAMES = {
    "i64": "LOOM_ATTR_I64",
    "f64": "LOOM_ATTR_F64",
    "string": "LOOM_ATTR_STRING",
    "bool": "LOOM_ATTR_BOOL",
    "enum": "LOOM_ATTR_ENUM",
    "type": "LOOM_ATTR_TYPE",
    "i64_array": "LOOM_ATTR_I64_ARRAY",
    "encoding": "LOOM_ATTR_ENCODING",
    "symbol": "LOOM_ATTR_SYMBOL",
    "flags": "LOOM_ATTR_FLAGS",
    "predicate_list": "LOOM_ATTR_PREDICATE_LIST",
    "dict": "LOOM_ATTR_DICT",
}

_GUARD_KIND_C_NAMES = {
    GuardKind.VALUE_TYPE: "LOOM_LOW_LOWER_GUARD_VALUE_TYPE",
    GuardKind.ATTR_KIND: "LOOM_LOW_LOWER_GUARD_ATTR_KIND",
    GuardKind.ENUM_ATTR_EQUALS: "LOOM_LOW_LOWER_GUARD_ATTR_ENUM_EQ",
    GuardKind.I64_RANGE: "LOOM_LOW_LOWER_GUARD_ATTR_I64_RANGE",
    GuardKind.DESCRIPTOR_AVAILABLE: "LOOM_LOW_LOWER_GUARD_DESCRIPTOR_AVAILABLE",
    GuardKind.VALUE_MATERIALIZABLE: "LOOM_LOW_LOWER_GUARD_VALUE_MATERIALIZABLE",
    GuardKind.LOW_VALUE_REGISTER_CLASS: "LOOM_LOW_LOWER_GUARD_LOW_VALUE_REGISTER_CLASS",
    GuardKind.VALUE_STATIC_DIM0_MULTIPLE: "LOOM_LOW_LOWER_GUARD_VALUE_STATIC_DIM0_MULTIPLE",
    GuardKind.LOW_VALUE_REGISTER_UNIT_COUNT_EQ: "LOOM_LOW_LOWER_GUARD_LOW_VALUE_REGISTER_UNIT_COUNT_EQ",
    GuardKind.I64_ARRAY_COUNT: "LOOM_LOW_LOWER_GUARD_ATTR_I64_ARRAY_COUNT_EQ",
    GuardKind.I64_ARRAY_ELEMENT_RANGE: "LOOM_LOW_LOWER_GUARD_ATTR_I64_ARRAY_ELEMENT_RANGE",
    GuardKind.I64_ARRAY_ELEMENTS_RANGE: "LOOM_LOW_LOWER_GUARD_ATTR_I64_ARRAY_ELEMENTS_RANGE",
    GuardKind.VALUE_SIGNED_BIT_COUNT: "LOOM_LOW_LOWER_GUARD_VALUE_SIGNED_BIT_COUNT",
    GuardKind.VALUE_UNSIGNED_BIT_COUNT: "LOOM_LOW_LOWER_GUARD_VALUE_UNSIGNED_BIT_COUNT",
    GuardKind.VALUE_EXACT_I64: "LOOM_LOW_LOWER_GUARD_VALUE_EXACT_I64",
}

_ATTR_COPY_KIND_C_NAMES = {
    LowerAttrCopyKind.DIRECT: "LOOM_LOW_LOWER_ATTR_COPY_DIRECT",
    LowerAttrCopyKind.I64_ARRAY_ELEMENT: "LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_ELEMENT",
    LowerAttrCopyKind.I64_ARRAY_PACK_ELEMENTS: "LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_PACK_ELEMENTS",
    LowerAttrCopyKind.I64_LITERAL: "LOOM_LOW_LOWER_ATTR_COPY_I64_LITERAL",
}

_EMIT_KIND_C_NAMES = {
    LowerEmitKind.DESCRIPTOR_OP: "LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP",
    LowerEmitKind.DESCRIPTOR_CONST: "LOOM_LOW_LOWER_EMIT_DESCRIPTOR_CONST",
    LowerEmitKind.DESCRIPTOR_OP_PER_LANE: "LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE",
    LowerEmitKind.DESCRIPTOR_OP_ACCUMULATE_LANES: "LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_ACCUMULATE_LANES",
}

_EMIT_FLAG_C_NAMES = {
    LOWER_EMIT_FLAG_SWAP_OPERANDS_0_1: "LOOM_LOW_LOWER_EMIT_FLAG_SWAP_OPERANDS_0_1",
    LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS: "LOOM_LOW_LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS",
    LOWER_EMIT_FLAG_RESULT_TYPE_PATTERN: "LOOM_LOW_LOWER_EMIT_FLAG_RESULT_TYPE_PATTERN",
}


@dataclass(frozen=True, slots=True)
class GeneratedLowerRuleSet:
    """Generated C/H contents for one lower-rule set."""

    header: str
    source: str


def generate_lower_rule_set(
    table: ContractTable,
    *,
    dialect_ops: Mapping[str, Sequence[Op]],
) -> GeneratedLowerRuleSet:
    """Generates C/H text for a generated target-low lower-rule set."""

    compiled = compile_lower_rule_set(table, dialect_ops=dialect_ops)
    public_header = _generated_public_header(table)
    symbol_name = _generated_symbol_name(table)
    c_table_prefix = _c_identifier(_generated_table_prefix(table))
    header_guard = _header_guard_from_public_header(public_header)
    return GeneratedLowerRuleSet(
        header=_generate_header(
            header_guard=header_guard,
            symbol_name=symbol_name,
        ),
        source=_generate_source(
            table=compiled,
            source_contract=table,
            public_header=public_header,
            symbol_name=symbol_name,
            c_table_prefix=c_table_prefix,
        ),
    )


def write_lower_rule_set_to_paths(
    table: ContractTable,
    *,
    dialect_ops: Mapping[str, Sequence[Op]],
    header_path: Path,
    source_path: Path,
) -> None:
    """Writes generated C/H contents for one generated lower-rule set."""

    generated = generate_lower_rule_set(table, dialect_ops=dialect_ops)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(generated.header, encoding="utf-8")
    source_path.write_text(generated.source, encoding="utf-8")


def _generate_header(*, header_guard: str, symbol_name: str) -> str:
    lines: list[str] = []
    lines.extend(
        line_comment_header(
            "//",
            generator="loom/py/loom/gen/target_lower_rules.py",
        )
    )
    lines.extend(
        [
            "",
            f"#ifndef {header_guard}",
            f"#define {header_guard}",
            "",
            '#include "loom/codegen/low/lower_rules.h"',
            "",
            "#ifdef __cplusplus",
            'extern "C" {',
            "#endif",
            "",
            f"extern const loom_low_lower_rule_set_t {symbol_name};",
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
    table: CompiledLowerRuleSet,
    source_contract: ContractTable,
    public_header: str,
    symbol_name: str,
    c_table_prefix: str,
) -> str:
    lines: list[str] = []
    lines.extend(
        line_comment_header(
            "//",
            generator="loom/py/loom/gen/target_lower_rules.py",
        )
    )
    lines.extend(
        [
            "",
            f'#include "{public_header}"',
            "",
            "#include <stddef.h>",
            "",
            f'#include "{source_contract.descriptor_set.public_header}"',
        ]
    )
    lines.extend(f'#include "{include}"' for include in _materializer_includes(source_contract))
    lines.extend(f'#include "{include}"' for include in _op_header_includes(table))
    lines.append("")

    type_patterns_name = f"k{c_table_prefix}TypePatterns"
    lines.extend(
        _emit_optional_array(
            type_patterns_name,
            "loom_low_lower_type_pattern_t",
            [_type_pattern_row(row.type_pattern) for row in table.type_patterns],
        )
    )

    value_refs_name = f"k{c_table_prefix}ValueRefs"
    lines.extend(
        _emit_optional_array(
            value_refs_name,
            "loom_low_lower_value_ref_t",
            [_value_ref_row(row) for row in table.value_refs],
        )
    )

    materializers_name = f"k{c_table_prefix}Materializers"
    lines.extend(
        _emit_optional_array(
            materializers_name,
            "loom_low_lower_value_materializer_t",
            [
                [
                    f".can_materialize = {materializer.can_materialize}",
                    f".materialize = {materializer.materialize}",
                ]
                for materializer in source_contract.materializers
            ],
        )
    )

    diagnostics_name = f"k{c_table_prefix}Diagnostics"
    lines.extend(
        _emit_optional_array(
            diagnostics_name,
            "loom_low_lower_diagnostic_t",
            [
                [
                    f'.subject_kind = IREE_SVL("{_c_string_literal(row.subject_kind)}")',
                    f'.subject_name = IREE_SVL("{_c_string_literal(row.subject_name)}")',
                    f'.reason = IREE_SVL("{_c_string_literal(row.reason)}")',
                ]
                for row in table.diagnostics
            ],
        )
    )

    guards_name = f"k{c_table_prefix}Guards"
    lines.extend(
        _emit_optional_array(
            guards_name,
            "loom_low_lower_guard_t",
            [_guard_row(source_contract, row) for row in table.guards],
        )
    )

    attr_copies_name = f"k{c_table_prefix}AttrCopies"
    lines.extend(
        _emit_optional_array(
            attr_copies_name,
            "loom_low_lower_attr_copy_t",
            [_attr_copy_row(row) for row in table.attr_copies],
        )
    )

    tied_results_name = f"k{c_table_prefix}TiedResults"
    lines.extend(
        _emit_optional_array(
            tied_results_name,
            "loom_tied_result_t",
            [_tied_result_row(row) for row in table.tied_results],
        )
    )

    emits_name = f"k{c_table_prefix}Emits"
    lines.extend(
        _emit_optional_array(
            emits_name,
            "loom_low_lower_emit_t",
            [_emit_row(source_contract, row) for row in table.emits],
        )
    )

    rules_name = f"k{c_table_prefix}Rules"
    lines.extend(
        _emit_optional_array(
            rules_name,
            "loom_low_lower_rule_t",
            [_rule_row(row) for row in table.rules],
        )
    )

    spans_name = f"k{c_table_prefix}Spans"
    lines.extend(
        _emit_optional_array(
            spans_name,
            "loom_low_lower_rule_span_t",
            [_span_row(row) for row in table.spans],
        )
    )

    lines.append(f"const loom_low_lower_rule_set_t {symbol_name} = {{")
    lines.extend(
        f"    {field},"
        for field in _rule_set_row(
            table=table,
            source_contract=source_contract,
            spans_name=spans_name,
            rules_name=rules_name,
            type_patterns_name=type_patterns_name,
            value_refs_name=value_refs_name,
            materializers_name=materializers_name,
            guards_name=guards_name,
            attr_copies_name=attr_copies_name,
            tied_results_name=tied_results_name,
            emits_name=emits_name,
            diagnostics_name=diagnostics_name,
        )
    )
    lines.extend(["};", ""])
    return "\n".join(lines)


def _emit_optional_array(
    name: str,
    c_type: str,
    rows: list[list[str]],
) -> list[str]:
    if not rows:
        return []
    lines = [f"static const {c_type} {name}[] = {{"]
    for row in rows:
        lines.append("    {")
        lines.extend(f"        {field}," for field in row)
        lines.append("    },")
    lines.extend(["};", ""])
    return lines


def _append_field(
    fields: list[str],
    name: str,
    value: str | int,
    *,
    always: bool = False,
    default: str = "0",
) -> None:
    value_string = str(value)
    if always or value_string != default:
        fields.append(f".{name} = {value_string}")


def _value_ref_row(row: LowerValueRef) -> list[str]:
    fields: list[str] = []
    _append_field(fields, "kind", _VALUE_REF_KIND_C_NAMES[row.kind], always=True)
    _append_field(fields, "index", row.index, always=True)
    _append_field(fields, "materializer_index", row.materializer_index)
    return fields


def _guard_row(table: ContractTable, row: LowerGuard) -> list[str]:
    fields: list[str] = []
    _append_field(fields, "kind", _GUARD_KIND_C_NAMES[row.kind], always=True)

    if row.kind in (
        GuardKind.VALUE_TYPE,
        GuardKind.VALUE_MATERIALIZABLE,
        GuardKind.LOW_VALUE_REGISTER_CLASS,
        GuardKind.VALUE_STATIC_DIM0_MULTIPLE,
        GuardKind.LOW_VALUE_REGISTER_UNIT_COUNT_EQ,
        GuardKind.VALUE_SIGNED_BIT_COUNT,
        GuardKind.VALUE_UNSIGNED_BIT_COUNT,
        GuardKind.VALUE_EXACT_I64,
    ):
        _append_field(fields, "value_ref_index", row.value_ref_index, always=True)
    if row.kind == GuardKind.LOW_VALUE_REGISTER_UNIT_COUNT_EQ:
        _append_field(
            fields,
            "other_value_ref_index",
            row.other_value_ref_index,
            always=True,
        )

    if row.kind in (
        GuardKind.ATTR_KIND,
        GuardKind.ENUM_ATTR_EQUALS,
        GuardKind.I64_RANGE,
        GuardKind.I64_ARRAY_COUNT,
        GuardKind.I64_ARRAY_ELEMENT_RANGE,
        GuardKind.I64_ARRAY_ELEMENTS_RANGE,
    ):
        _append_field(fields, "attr_index", row.attr_index, always=True)

    if row.kind == GuardKind.VALUE_TYPE:
        _append_field(fields, "type_pattern_index", row.type_pattern_index, always=True)
    if row.diagnostic_index != 0xFFFF:
        _append_field(
            fields,
            "diagnostic_index",
            _diagnostic_index(row.diagnostic_index),
            always=True,
        )
    if row.kind == GuardKind.ATTR_KIND:
        _append_field(fields, "attr_kind", _attr_kind_c_name(row.attr_kind), always=True)
    if row.kind in (
        GuardKind.ENUM_ATTR_EQUALS,
        GuardKind.VALUE_STATIC_DIM0_MULTIPLE,
        GuardKind.I64_ARRAY_COUNT,
        GuardKind.I64_ARRAY_ELEMENT_RANGE,
        GuardKind.VALUE_SIGNED_BIT_COUNT,
        GuardKind.VALUE_UNSIGNED_BIT_COUNT,
    ):
        _append_field(fields, "u64", row.u64, always=True)
    if row.kind == GuardKind.DESCRIPTOR_AVAILABLE:
        _append_field(
            fields,
            "descriptor_id",
            _guard_descriptor_id(table, row.descriptor),
            always=True,
        )
    if row.kind == GuardKind.LOW_VALUE_REGISTER_CLASS:
        _append_field(fields, "register_class_id", row.register_class_id, always=True)
    if row.kind in (
        GuardKind.I64_RANGE,
        GuardKind.I64_ARRAY_ELEMENT_RANGE,
        GuardKind.I64_ARRAY_ELEMENTS_RANGE,
    ):
        _append_field(fields, "minimum_i64", row.minimum_i64, always=True)
        _append_field(fields, "maximum_i64", row.maximum_i64, always=True)
    return fields


def _attr_copy_row(row: LowerAttrCopy) -> list[str]:
    fields: list[str] = []
    _append_field(fields, "kind", _ATTR_COPY_KIND_C_NAMES[row.kind], always=True)
    _append_field(
        fields,
        "target_name",
        f'IREE_SVL("{_c_string_literal(row.target_name)}")',
        always=True,
    )
    if row.kind in (
        LowerAttrCopyKind.DIRECT,
        LowerAttrCopyKind.I64_ARRAY_ELEMENT,
        LowerAttrCopyKind.I64_ARRAY_PACK_ELEMENTS,
    ):
        _append_field(fields, "source_attr_index", row.source_attr_index, always=True)
    if row.kind in (
        LowerAttrCopyKind.I64_ARRAY_ELEMENT,
        LowerAttrCopyKind.I64_ARRAY_PACK_ELEMENTS,
    ):
        _append_field(
            fields,
            "source_element_index",
            row.source_element_index,
            always=True,
        )
    if row.kind == LowerAttrCopyKind.I64_ARRAY_PACK_ELEMENTS:
        _append_field(
            fields,
            "source_element_count",
            row.source_element_count,
            always=True,
        )
        _append_field(
            fields,
            "source_element_bit_width",
            row.source_element_bit_width,
            always=True,
        )
    _append_field(fields, "target_bit_offset", row.target_bit_offset)
    _append_field(fields, "value_ref_index", row.value_ref_index)
    if row.kind == LowerAttrCopyKind.I64_LITERAL:
        _append_field(fields, "literal_i64", row.literal_i64, always=True)
    return fields


def _tied_result_row(row: LowerTiedResult) -> list[str]:
    fields: list[str] = []
    _append_field(fields, "result_index", row.result_index, always=True)
    _append_field(fields, "operand_index", row.operand_index, always=True)
    if row.has_type_change:
        _append_field(fields, "has_type_change", _c_bool(row.has_type_change))
    return fields


def _emit_row(table: ContractTable, row: LowerEmit) -> list[str]:
    fields: list[str] = []
    flags = _emit_flags(row.flags)
    _append_field(fields, "kind", _EMIT_KIND_C_NAMES[row.kind], always=True)
    _append_field(fields, "flags", flags)
    _append_field(
        fields,
        "descriptor_id",
        _descriptor_id_constant_name(table, row.descriptor.key),
        always=True,
    )
    if row.operand_ref_count:
        _append_field(fields, "operand_ref_start", row.operand_ref_start, always=True)
        _append_field(fields, "operand_ref_count", row.operand_ref_count, always=True)
    _append_field(fields, "copy_operand_mask", row.copy_operand_mask)
    if row.kind == LowerEmitKind.DESCRIPTOR_OP_ACCUMULATE_LANES:
        _append_field(
            fields,
            "accumulator_operand_index",
            row.accumulator_operand_index,
            always=True,
        )
    if row.result_ref_count:
        if row.flags & LOWER_EMIT_FLAG_RESULT_TYPE_PATTERN:
            _append_field(
                fields,
                "result_type_pattern_start",
                row.result_type_pattern_start,
                always=True,
            )
        else:
            _append_field(fields, "result_ref_start", row.result_ref_start, always=True)
        _append_field(fields, "result_ref_count", row.result_ref_count, always=True)
    if row.flags & LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS:
        _append_field(
            fields,
            "result_bind_ref_start",
            row.result_bind_ref_start,
            always=True,
        )
    if row.attr_copy_count:
        _append_field(fields, "attr_copy_start", row.attr_copy_start, always=True)
        _append_field(fields, "attr_copy_count", row.attr_copy_count, always=True)
    if row.tied_result_count:
        _append_field(fields, "tied_result_start", row.tied_result_start, always=True)
        _append_field(fields, "tied_result_count", row.tied_result_count, always=True)
    return fields


def _rule_row(row: LowerRule) -> list[str]:
    fields: list[str] = []
    _append_field(fields, "source_op_kind", _op_c_name(row.source_op), always=True)
    _append_field(fields, "temporary_count", row.temporary_count)
    if row.guard_count:
        _append_field(fields, "guard_start", row.guard_start, always=True)
        _append_field(fields, "guard_count", row.guard_count, always=True)
    if row.emit_count:
        _append_field(fields, "emit_start", row.emit_start, always=True)
        _append_field(fields, "emit_count", row.emit_count, always=True)
    if row.elide_ref_count:
        _append_field(fields, "elide_ref_start", row.elide_ref_start, always=True)
        _append_field(fields, "elide_ref_count", row.elide_ref_count, always=True)
    return fields


def _span_row(row: LowerRuleSpan) -> list[str]:
    fields: list[str] = []
    _append_field(fields, "source_op_kind", _op_c_name(row.source_op), always=True)
    _append_field(fields, "rule_start", row.rule_start, always=True)
    _append_field(fields, "rule_count", row.rule_count, always=True)
    return fields


def _rule_set_row(
    *,
    table: CompiledLowerRuleSet,
    source_contract: ContractTable,
    spans_name: str,
    rules_name: str,
    type_patterns_name: str,
    value_refs_name: str,
    materializers_name: str,
    guards_name: str,
    attr_copies_name: str,
    tied_results_name: str,
    emits_name: str,
    diagnostics_name: str,
) -> list[str]:
    fields = [".flags = LOOM_LOW_LOWER_RULE_SET_FLAG_TARGET_CONTRACT_QUERY"]
    _append_table_fields(fields, "spans", table.spans, spans_name)
    _append_table_fields(fields, "rules", table.rules, rules_name)
    _append_table_fields(
        fields,
        "type_patterns",
        table.type_patterns,
        type_patterns_name,
    )
    _append_table_fields(fields, "value_refs", table.value_refs, value_refs_name)
    _append_table_fields(
        fields,
        "materializers",
        source_contract.materializers,
        materializers_name,
    )
    _append_table_fields(fields, "guards", table.guards, guards_name)
    _append_table_fields(fields, "attr_copies", table.attr_copies, attr_copies_name)
    _append_table_fields(fields, "tied_results", table.tied_results, tied_results_name)
    _append_table_fields(fields, "emits", table.emits, emits_name)
    _append_table_fields(fields, "diagnostics", table.diagnostics, diagnostics_name)
    return fields


def _append_table_fields(
    fields: list[str],
    field_name: str,
    rows: tuple[object, ...],
    table_name: str,
) -> None:
    if not rows:
        return
    fields.append(f".{field_name} = {table_name}")
    fields.append(f".{_table_count_field_name(field_name)} = IREE_ARRAYSIZE({table_name})")


def _table_count_field_name(field_name: str) -> str:
    if field_name == "attr_copies":
        return "attr_copy_count"
    return f"{field_name[:-1]}_count"


def _type_pattern_row(type_pattern: TypePattern) -> list[str]:
    flags = [
        "LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND",
        "LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT",
    ]
    row = [
        ".flags = " + " | ".join(flags),
        f".type_kind = {_type_kind_c_name(type_pattern)}",
        f".element_type_mask = LOOM_LOW_LOWER_SCALAR_TYPE_BIT({_scalar_type_c_name(type_pattern.element)})",
    ]
    if type_pattern.kind == "vector":
        row[0] += " | LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_RANK"
        row.extend(
            [
                ".rank = 1",
            ]
        )
        if type_pattern.lanes is not None:
            row[0] += " | LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM0"
            row.append(f".static_dim0 = {_c_expression(type_pattern.lanes)}")
        elif type_pattern.minimum_lanes is not None and type_pattern.maximum_lanes is not None:
            row[0] += " | LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM0_RANGE"
            row.extend(
                [
                    f".static_dim0_min = {_c_expression(type_pattern.minimum_lanes)}",
                    f".static_dim0_max = {_c_expression(type_pattern.maximum_lanes)}",
                ]
            )
        else:
            raise ValueError("generated vector type patterns require static lanes")
    return row


def _type_kind_c_name(type_pattern: TypePattern) -> str:
    if type_pattern.kind == "scalar":
        return "LOOM_TYPE_SCALAR"
    if type_pattern.kind == "vector":
        return "LOOM_TYPE_VECTOR"
    raise ValueError(f"unknown type pattern kind '{type_pattern.kind}'")


def _scalar_type_c_name(element: str | None) -> str:
    if element is None:
        raise ValueError("type pattern element is required")
    c_name = _SCALAR_TYPE_C_NAMES.get(element)
    if c_name is None:
        raise ValueError(f"unknown scalar type '{element}'")
    return c_name


def _diagnostic_index(index: int) -> str:
    if index == 0xFFFF:
        return "LOOM_LOW_LOWER_DIAGNOSTIC_NONE"
    return str(index)


def _attr_kind_c_name(attr_kind: str | None) -> str:
    if attr_kind is None:
        return "0"
    c_name = _ATTR_KIND_C_NAMES.get(attr_kind)
    if c_name is None:
        raise ValueError(f"unknown attr kind '{attr_kind}'")
    return c_name


def _guard_descriptor_id(table: ContractTable, descriptor: Descriptor | None) -> str:
    if descriptor is None:
        return "LOOM_LOW_DESCRIPTOR_ID_NONE"
    return _descriptor_id_constant_name(table, descriptor.key)


def _emit_flags(flags: int) -> str:
    if flags == 0:
        return "0"
    names = [c_name for bit, c_name in sorted(_EMIT_FLAG_C_NAMES.items()) if flags & bit]
    unknown = flags & ~sum(_EMIT_FLAG_C_NAMES)
    if unknown:
        raise ValueError(f"unknown emit flags 0x{unknown:x}")
    return " | ".join(names)


def _c_bool(value: bool) -> str:
    return "true" if value else "false"


def _op_header_includes(table: CompiledLowerRuleSet) -> tuple[str, ...]:
    dialect_names = {rule.source_op.group.name for rule in table.rules if rule.source_op.group is not None}
    return tuple(f"loom/ops/{name}/ops.h" for name in sorted(dialect_names))


def _materializer_includes(table: ContractTable) -> tuple[str, ...]:
    return tuple(sorted({materializer.header for materializer in table.materializers}))


def _op_c_name(op: Op) -> str:
    return "LOOM_OP_" + _c_identifier(op.name).upper()


def _descriptor_id_constant_name(table: ContractTable, descriptor_key: str) -> str:
    descriptor_name = descriptor_set_relative_name(table.descriptor_set, descriptor_key)
    return f"{table.descriptor_set.c_enum_prefix}_DESCRIPTOR_ID_{_c_identifier(descriptor_name).upper()}"


def _generated_public_header(table: ContractTable) -> str:
    name_parts = _identifier_parts(table.name)
    if len(name_parts) == 2:
        target_name, family_name = name_parts
        return f"loom/target/arch/{target_name}/lower/{family_name}_rules.h"
    if not table.public_header:
        raise ValueError(f"contract table '{table.name}' requires public_header")
    return re.sub(r"contract_table\.h$", "lower_rules.h", table.public_header)


def _generated_symbol_name(table: ContractTable) -> str:
    return f"loom_{_c_identifier(table.name).lower()}_lower_rule_set"


def _generated_table_prefix(table: ContractTable) -> str:
    return f"{_pascal_identifier(table.name)}Lower"


def _c_expression(value: int | str) -> str:
    return str(value)


def _header_guard_from_public_header(public_header: str) -> str:
    return _c_identifier(public_header).upper() + "_"


def _c_string_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


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


def _pascal_identifier(value: str) -> str:
    parts = _identifier_parts(value)
    if not parts:
        return "_"
    return "".join(part[:1].upper() + part[1:] for part in parts)
