# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compilation from descriptor-rule contracts to target-low lowering rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, unique

from loom.dsl import ATTR_TYPE_ENUM, Op
from loom.target.contracts.emits import EmitDescriptorOp
from loom.target.contracts.guards import Guard, GuardKind
from loom.target.contracts.kinds import SourceValueKind
from loom.target.contracts.patterns import TypePattern
from loom.target.contracts.rules import DescriptorRule
from loom.target.contracts.source import ValueRef
from loom.target.contracts.tables import ContractTable
from loom.target.low_descriptors import Descriptor


@unique
class LowerEmitKind(Enum):
    """Interpreter emit operation used by a compiled lower-rule row."""

    DESCRIPTOR_OP = "descriptor_op"
    DESCRIPTOR_OP_PER_LANE = "descriptor_op_per_lane"


@dataclass(frozen=True, slots=True)
class LowerTypePattern:
    """Compiled type-pattern row."""

    type_pattern: TypePattern


@dataclass(frozen=True, slots=True)
class LowerValueRef:
    """Compiled source value-reference row."""

    kind: SourceValueKind
    index: int
    materializer_index: int = 0


@dataclass(frozen=True, slots=True)
class LowerDiagnostic:
    """Compiled rejection diagnostic row."""

    subject_kind: str
    subject_name: str
    reason: str


@dataclass(frozen=True, slots=True)
class LowerGuard:
    """Compiled selection guard row."""

    kind: GuardKind
    value_ref_index: int = 0
    other_value_ref_index: int = 0
    attr_index: int = 0
    type_pattern_index: int = 0
    diagnostic_index: int = 0xFFFF
    u64: int = 0
    minimum_i64: int = 0
    maximum_i64: int = 0


@dataclass(frozen=True, slots=True)
class LowerEmit:
    """Compiled emit-program row."""

    kind: LowerEmitKind
    descriptor: Descriptor
    operand_ref_start: int = 0
    operand_ref_count: int = 0
    result_ref_start: int = 0
    result_ref_count: int = 0


@dataclass(frozen=True, slots=True)
class LowerRule:
    """Compiled lowering rule row."""

    source_op: Op
    guard_start: int
    guard_count: int
    emit_start: int
    emit_count: int


@dataclass(frozen=True, slots=True)
class LowerRuleSpan:
    """Compiled op-kind to rule-range row."""

    source_op: Op
    rule_start: int
    rule_count: int


@dataclass(frozen=True, slots=True)
class CompiledLowerRuleSet:
    """Lower-rule set rows generated from a contract table."""

    name: str
    authored_case_indices: tuple[int, ...]
    rules: tuple[LowerRule, ...]
    spans: tuple[LowerRuleSpan, ...]
    type_patterns: tuple[LowerTypePattern, ...]
    value_refs: tuple[LowerValueRef, ...]
    guards: tuple[LowerGuard, ...]
    emits: tuple[LowerEmit, ...]
    diagnostics: tuple[LowerDiagnostic, ...]


def compile_lower_rule_set(
    table: ContractTable,
    *,
    dialect_ops: Mapping[str, Sequence[Op]],
) -> CompiledLowerRuleSet:
    """Compiles table-authored descriptor rules to target-low rule rows."""

    compiler = _LowerRuleSetCompiler(table, dialect_ops)
    return compiler.compile()


class _LowerRuleSetCompiler:
    def __init__(
        self,
        table: ContractTable,
        dialect_ops: Mapping[str, Sequence[Op]],
    ) -> None:
        self._table = table
        self._op_ordinals = _build_op_ordinals(dialect_ops)
        self._rules: list[LowerRule] = []
        self._type_patterns: list[LowerTypePattern] = []
        self._value_refs: list[LowerValueRef] = []
        self._guards: list[LowerGuard] = []
        self._emits: list[LowerEmit] = []
        self._diagnostics: list[LowerDiagnostic] = []
        self._authored_case_indices: list[int] = []
        self._type_pattern_ordinals: dict[TypePattern, int] = {}
        self._diagnostic_ordinals: dict[LowerDiagnostic, int] = {}

    def compile(self) -> CompiledLowerRuleSet:
        for authored_case_index, contract_case in enumerate(self._table.cases):
            if isinstance(contract_case, DescriptorRule):
                self._append_descriptor_rule(authored_case_index, contract_case)

        spans = _build_spans(self._rules, self._op_ordinals)
        return CompiledLowerRuleSet(
            name=self._table.name,
            authored_case_indices=tuple(self._authored_case_indices),
            rules=tuple(self._rules),
            spans=spans,
            type_patterns=tuple(self._type_patterns),
            value_refs=tuple(self._value_refs),
            guards=tuple(self._guards),
            emits=tuple(self._emits),
            diagnostics=tuple(self._diagnostics),
        )

    def _append_descriptor_rule(
        self,
        authored_case_index: int,
        rule: DescriptorRule,
    ) -> None:
        if not rule.emit:
            raise ValueError(
                f"{rule.source_op.name}: descriptor-rule contracts must "
                "author their emit program in Python"
            )
        if len(rule.emit) != 1:
            raise ValueError(
                f"{rule.source_op.name}: generated lower-rule contracts "
                "currently require one descriptor emit"
            )

        guard_start = len(self._guards)
        type_patterns_by_field: dict[str, TypePattern] = {}
        for guard in rule.guards:
            self._append_guard(rule.source_op, guard, type_patterns_by_field)

        emit_start = len(self._emits)
        self._append_emit(rule.source_op, rule.emit[0], type_patterns_by_field)
        self._rules.append(
            LowerRule(
                source_op=rule.source_op,
                guard_start=guard_start,
                guard_count=len(self._guards) - guard_start,
                emit_start=emit_start,
                emit_count=len(self._emits) - emit_start,
            )
        )
        self._authored_case_indices.append(authored_case_index)

    def _append_guard(
        self,
        source_op: Op,
        guard: Guard,
        type_patterns_by_field: dict[str, TypePattern],
    ) -> None:
        if guard.kind == GuardKind.VALUE_TYPE:
            if guard.type_pattern is None:
                raise ValueError(f"{source_op.name}: value_type guard needs a type")
            type_patterns_by_field[guard.field] = guard.type_pattern
            self._guards.append(
                LowerGuard(
                    kind=guard.kind,
                    value_ref_index=self._append_value_ref(
                        source_op,
                        _value_ref_for_source_field(source_op, guard.field),
                    ),
                    type_pattern_index=self._append_type_pattern(guard.type_pattern),
                    diagnostic_index=self._append_diagnostic(
                        LowerDiagnostic(
                            subject_kind="type",
                            subject_name=guard.field,
                            reason=_type_diagnostic_reason(guard.type_pattern),
                        )
                    ),
                )
            )
            return

        if guard.kind == GuardKind.ENUM_ATTR_EQUALS:
            attr_index = _source_attr_index(source_op, guard.field)
            attr = source_op.attrs[attr_index]
            if attr.attr_type != ATTR_TYPE_ENUM or attr.enum_def is None:
                raise ValueError(
                    f"{source_op.name}: enum guard field '{guard.field}' "
                    "must name an enum attr"
                )
            enum_keyword = guard.enum_keyword
            if enum_keyword is None:
                raise ValueError(f"{source_op.name}: enum guard needs a keyword")
            enum_value = next(
                enum_case.value
                for enum_case in attr.enum_def.cases
                if enum_case.keyword == enum_keyword
            )
            self._guards.append(
                LowerGuard(
                    kind=guard.kind,
                    attr_index=attr_index,
                    diagnostic_index=self._append_diagnostic(
                        LowerDiagnostic(
                            subject_kind="attr",
                            subject_name=guard.field,
                            reason=(
                                f"target contract requires enum case '{enum_keyword}'"
                            ),
                        )
                    ),
                    u64=enum_value,
                )
            )
            return

        raise ValueError(
            f"{source_op.name}: guard kind '{guard.kind.value}' is not "
            "representable by generated descriptor rules yet"
        )

    def _append_emit(
        self,
        source_op: Op,
        emit: EmitDescriptorOp,
        type_patterns_by_field: dict[str, TypePattern],
    ) -> None:
        if emit.immediates:
            raise ValueError(
                f"{source_op.name}: generated descriptor-rule emits do not "
                "support immediate bindings yet"
            )

        emit_kind = _lower_emit_kind(source_op, emit, type_patterns_by_field)
        operand_bindings = emit.operands if emit.operands is not None else {}
        result_bindings = emit.results if emit.results is not None else {}
        operand_refs: list[LowerValueRef] = []
        for descriptor_operand in emit.descriptor.operands:
            value_ref = operand_bindings.get(descriptor_operand.field_name)
            if value_ref is not None:
                operand_refs.append(_lower_value_ref(source_op, value_ref))
        operand_ref_start = self._append_value_ref_sequence(tuple(operand_refs))
        operand_ref_count = len(operand_refs)

        result_refs: list[LowerValueRef] = []
        for descriptor_operand in emit.descriptor.operands:
            value_ref = result_bindings.get(descriptor_operand.field_name)
            if value_ref is not None:
                if value_ref.kind != SourceValueKind.RESULT:
                    raise ValueError(
                        f"{source_op.name}: generated descriptor-rule emits "
                        "must bind descriptor results directly to source results"
                    )
                result_refs.append(_lower_value_ref(source_op, value_ref))
        result_ref_start = self._append_value_ref_sequence(tuple(result_refs))
        result_ref_count = len(result_refs)

        self._emits.append(
            LowerEmit(
                kind=emit_kind,
                descriptor=emit.descriptor,
                operand_ref_start=operand_ref_start,
                operand_ref_count=operand_ref_count,
                result_ref_start=result_ref_start,
                result_ref_count=result_ref_count,
            )
        )

    def _append_type_pattern(self, type_pattern: TypePattern) -> int:
        ordinal = self._type_pattern_ordinals.get(type_pattern)
        if ordinal is not None:
            return ordinal
        ordinal = len(self._type_patterns)
        self._type_pattern_ordinals[type_pattern] = ordinal
        self._type_patterns.append(LowerTypePattern(type_pattern))
        return ordinal

    def _append_value_ref(self, source_op: Op, value_ref: ValueRef) -> int:
        return self._append_value_ref_sequence(
            (_lower_value_ref(source_op, value_ref),)
        )

    def _append_value_ref_sequence(self, sequence: tuple[LowerValueRef, ...]) -> int:
        if not sequence:
            return 0
        sequence_count = len(sequence)
        for start in range(len(self._value_refs) - sequence_count + 1):
            if tuple(self._value_refs[start : start + sequence_count]) == sequence:
                return start
        ordinal = len(self._value_refs)
        self._value_refs.extend(sequence)
        return ordinal

    def _append_diagnostic(self, diagnostic: LowerDiagnostic) -> int:
        ordinal = self._diagnostic_ordinals.get(diagnostic)
        if ordinal is not None:
            return ordinal
        ordinal = len(self._diagnostics)
        self._diagnostic_ordinals[diagnostic] = ordinal
        self._diagnostics.append(diagnostic)
        return ordinal


def _build_spans(
    rules: list[LowerRule],
    op_ordinals: dict[int, int],
) -> tuple[LowerRuleSpan, ...]:
    spans: list[LowerRuleSpan] = []
    i = 0
    while i < len(rules):
        rule_start = i
        rule = rules[i]
        rule_count = 1
        while (
            i + rule_count < len(rules)
            and rules[i + rule_count].source_op is rule.source_op
        ):
            rule_count += 1
        spans.append(
            LowerRuleSpan(
                source_op=rule.source_op,
                rule_start=rule_start,
                rule_count=rule_count,
            )
        )
        i += rule_count
    return tuple(
        sorted(spans, key=lambda span: _op_kind_key(span.source_op, op_ordinals))
    )


def _lower_emit_kind(
    source_op: Op,
    emit: EmitDescriptorOp,
    type_patterns_by_field: dict[str, TypePattern],
) -> LowerEmitKind:
    result_bindings = emit.results if emit.results is not None else {}
    vector_result_lanes: int | None = None
    for descriptor_operand in emit.descriptor.operands:
        value_ref = result_bindings.get(descriptor_operand.field_name)
        if value_ref is None:
            continue
        result_type = _require_type_pattern(
            source_op,
            value_ref.field,
            type_patterns_by_field,
        )
        if result_type.kind != "vector":
            return LowerEmitKind.DESCRIPTOR_OP
        vector_result_lanes = result_type.lanes
        if vector_result_lanes is None:
            raise ValueError(
                f"{source_op.name}: per-lane descriptor emits require a "
                "static vector lane count"
            )
        if descriptor_operand.unit_count == 1:
            return LowerEmitKind.DESCRIPTOR_OP_PER_LANE
    return LowerEmitKind.DESCRIPTOR_OP


def _require_type_pattern(
    source_op: Op,
    field: str,
    type_patterns_by_field: dict[str, TypePattern],
) -> TypePattern:
    type_pattern = type_patterns_by_field.get(field)
    if type_pattern is None:
        raise ValueError(
            f"{source_op.name}: descriptor emit field '{field}' needs a "
            "value_type guard"
        )
    return type_pattern


def _value_ref_for_source_field(source_op: Op, field: str) -> ValueRef:
    if source_op.operand(field) is not None:
        return ValueRef.operand(field)
    if source_op.result(field) is not None:
        return ValueRef.result(field)
    raise ValueError(f"{source_op.name}: source field '{field}' is not a value")


def _lower_value_ref(source_op: Op, value_ref: ValueRef) -> LowerValueRef:
    return LowerValueRef(
        kind=value_ref.kind,
        index=_source_value_index(source_op, value_ref),
    )


def _source_value_index(source_op: Op, value_ref: ValueRef) -> int:
    if value_ref.kind == SourceValueKind.OPERAND:
        operand = source_op.operand(value_ref.field)
        if operand is not None:
            return source_op.operands.index(operand)
    if value_ref.kind == SourceValueKind.RESULT:
        result = source_op.result(value_ref.field)
        if result is not None:
            return source_op.results.index(result)
    raise ValueError(f"source value field '{value_ref.field}' is not declared")


def _source_attr_index(source_op: Op, field: str) -> int:
    attr = source_op.attr(field)
    if attr is None:
        raise ValueError(f"{source_op.name}: source field '{field}' is not an attr")
    return source_op.attrs.index(attr)


def _type_diagnostic_reason(type_pattern: TypePattern) -> str:
    if type_pattern.kind == "scalar":
        return f"target contract requires {type_pattern.element} scalar values"
    if type_pattern.lanes is None:
        return f"target contract requires vector<{type_pattern.element}> values"
    return (
        "target contract requires "
        f"vector<{type_pattern.lanes}x{type_pattern.element}> values"
    )


def _build_op_ordinals(dialect_ops: Mapping[str, Sequence[Op]]) -> dict[int, int]:
    op_ordinals: dict[int, int] = {}
    for ops in dialect_ops.values():
        for op_index, op in enumerate(ops):
            op_ordinals[id(op)] = op_index
    return op_ordinals


def _op_kind_key(op: Op, op_ordinals: dict[int, int]) -> tuple[int, int]:
    if op.group is None:
        raise ValueError(f"op '{op.name}' is not assigned to a dialect")
    try:
        op_index = op_ordinals[id(op)]
    except KeyError as exc:
        raise ValueError(f"op '{op.name}' is not present in dialect_ops") from exc
    return (op.group.dialect_id, op_index)
