# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from loom.target.arch.wasm.descriptors import WASM_CORE_SIMD128_DESCRIPTOR_SET
from loom.target.arch.wasm.vector_shifts import (
    uniform_shift_rules,
    varying_shift_rules,
)
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    GuardKind,
    SourceValueKind,
    ValueProject,
    ValueProjectKind,
    ValueRef,
    descriptor_by_key,
)

ScalarOrVector = int | tuple[int, ...]


def _descriptor(key: str):
    return descriptor_by_key(WASM_CORE_SIMD128_DESCRIPTOR_SET, key)


def _type_guard(field, type_pattern):
    return Guard.value_type(field, type_pattern)


def _all_rules() -> tuple[DescriptorRule, ...]:
    return (*uniform_shift_rules(), *varying_shift_rules(_descriptor, _type_guard))


def _rule_element_type(rule) -> str:
    result_guard = next(
        guard
        for guard in rule.guards
        if guard.kind is GuardKind.VALUE_TYPE
        and guard.field == "result"
        and guard.type_pattern is not None
        and guard.type_pattern.kind == "vector"
    )
    assert result_guard.type_pattern is not None
    return result_guard.type_pattern.elements[0]


def _rule_lane_range(rule: DescriptorRule) -> tuple[int, int]:
    result_guard = next(
        guard
        for guard in rule.guards
        if guard.kind is GuardKind.VALUE_TYPE
        and guard.field == "result"
        and guard.type_pattern is not None
        and guard.type_pattern.kind == "vector"
    )
    assert result_guard.type_pattern is not None
    assert isinstance(result_guard.type_pattern.minimum_lanes, int)
    assert isinstance(result_guard.type_pattern.maximum_lanes, int)
    return (
        result_guard.type_pattern.minimum_lanes,
        result_guard.type_pattern.maximum_lanes,
    )


def _mask(bit_count: int) -> int:
    return (1 << bit_count) - 1


def _signed(value: int, bit_count: int) -> int:
    value &= _mask(bit_count)
    sign_bit = 1 << (bit_count - 1)
    return value - (1 << bit_count) if value & sign_bit else value


def _reference(operation: str, value: int, count: int, bit_count: int) -> int:
    if operation == "shl":
        result = value << count
    elif operation == "shr_s":
        result = _signed(value, bit_count) >> count
    else:
        assert operation == "shr_u"
        result = value >> count
    return result & _mask(bit_count)


def _operation(rule: DescriptorRule) -> str:
    return {
        "vector.shli": "shl",
        "vector.shrsi": "shr_s",
        "vector.shrui": "shr_u",
    }[rule.source_op.name]


def _input_values(bit_count: int) -> tuple[int, ...]:
    lane_count = 128 // bit_count
    boundaries = (
        0,
        1,
        (1 << (bit_count - 1)) - 1,
        1 << (bit_count - 1),
        _mask(bit_count),
    )
    return tuple(
        (boundaries[lane % len(boundaries)] ^ (lane * 0x9E3779B9)) & _mask(bit_count)
        for lane in range(lane_count)
    )


def _project(value: int | ValueProject, count: int) -> int:
    if isinstance(value, int):
        return value
    assert value.kind is ValueProjectKind.EXACT_I64
    assert value.source_value == "rhs"
    return count


def _evaluate(
    rule: DescriptorRule,
    lhs: tuple[int, ...],
    rhs: tuple[int, ...],
    count: int,
) -> tuple[int, ...]:
    bit_count = int(_rule_element_type(rule)[1:])
    lane_mask = _mask(bit_count)
    values: dict[ValueRef, ScalarOrVector] = {
        ValueRef.operand("lhs"): lhs,
        ValueRef.operand("rhs"): rhs,
        ValueRef.uniform_element_origin_operand("rhs"): count,
        ValueRef.exact_uniform_element_origin_operand("rhs"): count,
    }
    for emit in rule.emit:
        assert isinstance(emit, EmitDescriptorOp)
        key = emit.descriptor.key.removeprefix("wasm.")
        operands = {name: values[value] for name, value in emit.operands.items()}
        if key == "i32.const":
            result: ScalarOrVector = _project(emit.immediates["i32_value"], count)
        elif key == "v128.const":
            bits = emit.immediates["lo64"] | (emit.immediates["hi64"] << 64)
            result = tuple(
                (bits >> (lane * bit_count)) & lane_mask
                for lane in range(128 // bit_count)
            )
        elif key == "i32.wrap_i64":
            scalar = operands["input"]
            assert isinstance(scalar, int)
            result = scalar & _mask(32)
        elif key == "v128.and":
            left = operands["lhs"]
            right = operands["rhs"]
            assert isinstance(left, tuple) and isinstance(right, tuple)
            result = tuple(a & b for a, b in zip(left, right, strict=True))
        elif key.endswith(".eq"):
            left = operands["lhs"]
            right = operands["rhs"]
            assert isinstance(left, tuple) and isinstance(right, tuple)
            result = tuple(
                lane_mask if a == b else 0 for a, b in zip(left, right, strict=True)
            )
        elif key.endswith((".shl", ".shr_s", ".shr_u")):
            left = operands["value"]
            shift = operands["count"]
            assert isinstance(left, tuple) and isinstance(shift, int)
            operation = key.rsplit(".", maxsplit=1)[1]
            result = tuple(
                _reference(operation, value, shift & (bit_count - 1), bit_count)
                for value in left
            )
        else:
            assert key == "v128.bitselect"
            true_values = operands["true_value"]
            false_values = operands["false_value"]
            conditions = operands["condition"]
            assert isinstance(true_values, tuple)
            assert isinstance(false_values, tuple)
            assert isinstance(conditions, tuple)
            result = tuple(
                ((true_value & condition) | (false_value & ~condition)) & lane_mask
                for true_value, false_value, condition in zip(
                    true_values, false_values, conditions, strict=True
                )
            )
        assert len(emit.results) == 1
        values[next(iter(emit.results.values()))] = result
    result = values[ValueRef.result("result")]
    assert isinstance(result, tuple)
    return result


def test_integer_shift_rules_cover_the_complete_semantic_family():
    rules = _all_rules()

    actual = {
        (rule.source_op.name, _rule_element_type(rule), rule.priority) for rule in rules
    }
    expected = {
        (source_op, element_type, priority)
        for source_op in ("vector.shli", "vector.shrsi", "vector.shrui")
        for element_type in ("i8", "i16", "i32", "i64")
        for priority in (0, 1, 2)
    }
    assert actual == expected
    assert len(rules) == len(expected)
    expected_maximum_lanes = {"i8": 16, "i16": 8, "i32": 4, "i64": 2}
    for rule in rules:
        assert _rule_lane_range(rule) == (
            1,
            expected_maximum_lanes[_rule_element_type(rule)],
        )


def test_uniform_shift_rules_select_native_simd_instructions():
    rules = tuple(uniform_shift_rules())

    for rule in rules:
        element_type = _rule_element_type(rule)
        operation = {
            "vector.shli": "shl",
            "vector.shrsi": "shr_s",
            "vector.shrui": "shr_u",
        }[rule.source_op.name]
        shape = {
            "i8": "i8x16",
            "i16": "i16x8",
            "i32": "i32x4",
            "i64": "i64x2",
        }[element_type]
        assert rule.report_key == "wasm.integer_shift.uniform.native"
        assert rule.emit[-1].descriptor.key == f"wasm.{shape}.{operation}"
        assert rule.emit[-1].form is DescriptorEmitForm.OP
        assert len(rule.emit) == (
            2 if rule.priority == 1 or element_type == "i64" else 1
        )
        if rule.priority == 2:
            origin_guards = [
                guard
                for guard in rule.guards
                if guard.value_ref is not None
                and guard.value_ref.kind
                is SourceValueKind.UNIFORM_ELEMENT_ORIGIN_OPERAND
            ]
            assert len(origin_guards) == 1


def test_varying_shift_rules_use_one_packed_barrel_stage_per_count_bit():
    rules = varying_shift_rules(_descriptor, _type_guard)

    for rule in rules:
        element_type = _rule_element_type(rule)
        element_bit_count = int(element_type[1:])
        shape = {
            "i8": "i8x16",
            "i16": "i16x8",
            "i32": "i32x4",
            "i64": "i64x2",
        }[element_type]
        operation = {
            "vector.shli": "shl",
            "vector.shrsi": "shr_s",
            "vector.shrui": "shr_u",
        }[rule.source_op.name]
        stage_count = element_bit_count.bit_length() - 1
        assert rule.report_key == "wasm.integer_shift.varying.barrel"
        assert len(rule.emit) == stage_count * 6
        setup_emits = rule.emit[: stage_count * 2]
        assert [emit.descriptor.key for emit in setup_emits] == [
            key
            for _ in range(stage_count)
            for key in ("wasm.v128.const", "wasm.i32.const")
        ]
        sequence_emits = rule.emit[stage_count * 2 :]
        for stage in range(stage_count):
            stage_emits = sequence_emits[stage * 4 : (stage + 1) * 4]
            assert [emit.descriptor.key for emit in stage_emits] == [
                "wasm.v128.and",
                f"wasm.{shape}.eq",
                f"wasm.{shape}.{operation}",
                "wasm.v128.bitselect",
            ]
            assert all(emit.form is DescriptorEmitForm.OP for emit in stage_emits)
        assert all(
            "extract_lane" not in emit.descriptor.key
            and "replace_lane" not in emit.descriptor.key
            for emit in rule.emit
        )


def _assert_rule_counts(rule: DescriptorRule, counts: tuple[int, ...]) -> None:
    bit_count = int(_rule_element_type(rule)[1:])
    values = _input_values(bit_count)
    operation = _operation(rule)
    assert len(counts) == len(values)
    actual = _evaluate(rule, values, counts, counts[0])
    expected = tuple(
        _reference(operation, value, count, bit_count)
        for value, count in zip(values, counts, strict=True)
    )
    assert actual == expected


def test_uniform_shift_rules_match_boundary_counts():
    rules = tuple(uniform_shift_rules())

    for rule in rules:
        bit_count = int(_rule_element_type(rule)[1:])
        lane_count = 128 // bit_count
        for count in (0, 1, bit_count // 2, bit_count - 1):
            _assert_rule_counts(rule, (count,) * lane_count)


def test_varying_shift_rules_match_boundary_count_patterns():
    rules = varying_shift_rules(_descriptor, _type_guard)

    for rule in rules:
        bit_count = int(_rule_element_type(rule)[1:])
        boundaries = (0, 1, bit_count // 2, bit_count - 1)
        counts = tuple(
            boundaries[lane % len(boundaries)] for lane in range(128 // bit_count)
        )
        _assert_rule_counts(rule, counts)


@pytest.mark.exhaustive
def test_uniform_shift_rules_match_every_valid_count():
    for rule in uniform_shift_rules():
        bit_count = int(_rule_element_type(rule)[1:])
        lane_count = 128 // bit_count
        for count in range(bit_count):
            _assert_rule_counts(rule, (count,) * lane_count)


@pytest.mark.exhaustive
def test_varying_shift_rules_match_every_valid_count():
    for rule in varying_shift_rules(_descriptor, _type_guard):
        bit_count = int(_rule_element_type(rule)[1:])
        lane_count = 128 // bit_count
        for first_count in range(0, bit_count, lane_count):
            counts = tuple(
                (first_count + lane) % bit_count for lane in range(lane_count)
            )
            _assert_rule_counts(rule, counts)
