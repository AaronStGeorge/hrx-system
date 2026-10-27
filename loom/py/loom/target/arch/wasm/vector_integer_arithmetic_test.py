# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.target.arch.wasm.descriptors import WASM_CORE_SIMD128_DESCRIPTOR_SET
from loom.target.arch.wasm.vector_integer_arithmetic import (
    integer_arithmetic_rules,
)
from loom.target.contracts import (
    DescriptorRule,
    Guard,
    GuardKind,
    descriptor_by_key,
)


def _descriptor(key: str):
    return descriptor_by_key(WASM_CORE_SIMD128_DESCRIPTOR_SET, key)


def _type_guard(field, type_pattern):
    return Guard.value_type(field, type_pattern)


def _all_rules() -> tuple[DescriptorRule, ...]:
    return integer_arithmetic_rules(_descriptor, _type_guard)


def _rule_element_type(rule: DescriptorRule) -> str:
    result_guard = next(
        guard
        for guard in rule.guards
        if guard.kind is GuardKind.VALUE_TYPE and guard.field == "result"
    )
    assert result_guard.type_pattern is not None
    return result_guard.type_pattern.elements[0]


def _rule_lane_range(rule: DescriptorRule) -> tuple[int, int]:
    result_guard = next(
        guard
        for guard in rule.guards
        if guard.kind is GuardKind.VALUE_TYPE and guard.field == "result"
    )
    assert result_guard.type_pattern is not None
    assert isinstance(result_guard.type_pattern.minimum_lanes, int)
    assert isinstance(result_guard.type_pattern.maximum_lanes, int)
    return (
        result_guard.type_pattern.minimum_lanes,
        result_guard.type_pattern.maximum_lanes,
    )


def _find_rule(source_op: str, element_bit_count: int) -> DescriptorRule:
    element_type = f"i{element_bit_count}"
    return next(
        rule
        for rule in _all_rules()
        if rule.source_op.name == source_op and _rule_element_type(rule) == element_type
    )


def test_rules_cover_the_complete_packed_integer_arithmetic_family():
    rules = _all_rules()
    expected = {
        (source_op, element_type)
        for source_op in (
            "vector.addi",
            "vector.subi",
            "vector.muli",
            "vector.negi",
            "vector.absi",
            "vector.minsi",
            "vector.maxsi",
            "vector.minui",
            "vector.maxui",
            "vector.fmai",
        )
        for element_type in ("i8", "i16", "i32", "i64")
    }
    actual = {(rule.source_op.name, _rule_element_type(rule)) for rule in rules}
    assert actual == expected
    assert len(rules) == len(expected)

    maximum_lanes = {"i8": 16, "i16": 8, "i32": 4, "i64": 2}
    for rule in rules:
        element_type = _rule_element_type(rule)
        assert _rule_lane_range(rule) == (1, maximum_lanes[element_type])

    assert (
        sum(rule.report_key == "wasm.integer_arithmetic.native" for rule in rules) == 31
    )
    assert (
        sum(rule.report_key == "wasm.integer_arithmetic.i8_multiply" for rule in rules)
        == 1
    )
    assert (
        sum(rule.report_key == "wasm.integer_arithmetic.i64_extrema" for rule in rules)
        == 4
    )
    assert sum(rule.report_key == "wasm.integer_arithmetic.fma" for rule in rules) == 4


def test_i8_multiply_uses_two_widening_halves_and_one_byte_pack():
    multiply = _find_rule("vector.muli", 8)
    assert [emit.descriptor.key for emit in multiply.emit] == [
        "wasm.i16x8.extmul_low_i8x16_u",
        "wasm.i16x8.extmul_high_i8x16_u",
        "wasm.i8x16.shuffle",
    ]
    assert [multiply.emit[-1].immediates[f"lane{lane}"] for lane in range(16)] == [
        *range(0, 16, 2),
        *range(16, 32, 2),
    ]


def test_i64_extrema_use_signed_compare_with_unsigned_bias_when_needed():
    for source_op, expected_sequence in (
        ("vector.minsi", ["wasm.i64x2.lt_s", "wasm.v128.bitselect"]),
        ("vector.maxsi", ["wasm.i64x2.lt_s", "wasm.v128.bitselect"]),
        (
            "vector.minui",
            [
                "wasm.v128.const",
                "wasm.v128.xor",
                "wasm.v128.xor",
                "wasm.i64x2.lt_s",
                "wasm.v128.bitselect",
            ],
        ),
        (
            "vector.maxui",
            [
                "wasm.v128.const",
                "wasm.v128.xor",
                "wasm.v128.xor",
                "wasm.i64x2.lt_s",
                "wasm.v128.bitselect",
            ],
        ),
    ):
        rule = _find_rule(source_op, 64)
        assert [emit.descriptor.key for emit in rule.emit] == expected_sequence


def test_fma_composes_the_width_specific_multiply_and_native_add():
    for bit_count in (8, 16, 32, 64):
        rule = _find_rule("vector.fmai", bit_count)
        shape = f"i{bit_count}x{128 // bit_count}"
        if bit_count == 8:
            expected_sequence = [
                "wasm.i16x8.extmul_low_i8x16_u",
                "wasm.i16x8.extmul_high_i8x16_u",
                "wasm.i8x16.shuffle",
                "wasm.i8x16.add",
            ]
        else:
            expected_sequence = [f"wasm.{shape}.mul", f"wasm.{shape}.add"]
        assert [emit.descriptor.key for emit in rule.emit] == expected_sequence
