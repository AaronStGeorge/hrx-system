# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import struct

from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.vector import defs as vector
from loom.target.arch.wasm.descriptors import WASM_CORE_SIMD128_DESCRIPTOR_SET
from loom.target.arch.wasm.float_narrowing import float_narrowing_rules
from loom.target.contracts import (
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    GuardKind,
    ValueRef,
    descriptor_by_key,
)


def _descriptor(key):
    return descriptor_by_key(WASM_CORE_SIMD128_DESCRIPTOR_SET, key)


def _type_guard(field, type_pattern):
    return Guard.value_type(field, type_pattern)


def _rules() -> tuple[DescriptorRule, ...]:
    return float_narrowing_rules(_descriptor, _type_guard)


def _evaluate_lane(rule: DescriptorRule, input_bits: int) -> int:
    values = {ValueRef.operand("input"): input_bits}
    vector_rule = rule.source_op is vector.vector_fptrunc
    for emit in rule.emit:
        assert isinstance(emit, EmitDescriptorOp)
        key = emit.descriptor.key.removeprefix("wasm.")
        operands = {name: values[value] for name, value in emit.operands.items()}
        if key in ("i32.const", "f32.const"):
            result = next(iter(emit.immediates.values()))
        elif key == "v128.const":
            result = emit.immediates["lo64"] & 0xFFFFFFFF
        elif key in ("i32.reinterpret_f32", "f32.reinterpret_i32"):
            result = operands["input"]
        elif key in ("i32.and", "v128.and"):
            result = operands["lhs"] & operands["rhs"]
        elif key in ("i32.or", "v128.or"):
            result = operands["lhs"] | operands["rhs"]
        elif key in ("i32.add", "i32x4.add"):
            result = operands["lhs"] + operands["rhs"]
        elif key in ("i32.sub", "i32x4.sub"):
            result = operands["lhs"] - operands["rhs"]
        elif key in ("i32.shl", "i32x4.shl"):
            value = operands["lhs"] if key == "i32.shl" else operands["value"]
            count = operands["rhs"] if key == "i32.shl" else operands["count"]
            result = value << (count & 31)
        elif key in ("i32.shr_u", "i32x4.shr_u"):
            value = operands["lhs"] if key == "i32.shr_u" else operands["value"]
            count = operands["rhs"] if key == "i32.shr_u" else operands["count"]
            result = value >> (count & 31)
        elif key in ("i32.lt_u", "i32x4.lt_u"):
            result = int(operands["lhs"] < operands["rhs"])
        elif key in ("i32.ge_u", "i32x4.ge_u"):
            result = int(operands["lhs"] >= operands["rhs"])
        elif key in ("i32.gt_u", "i32x4.gt_u"):
            result = int(operands["lhs"] > operands["rhs"])
        elif key == "i32.select":
            result = (
                operands["true_value"]
                if operands["condition"]
                else operands["false_value"]
            )
        elif key == "v128.bitselect":
            condition = 0xFFFFFFFF if operands["condition"] else 0
            result = (operands["true_value"] & condition) | (
                operands["false_value"] & ~condition
            )
        elif key in ("f32.add", "f32x4.add"):
            lhs = struct.unpack("<f", struct.pack("<I", operands["lhs"]))[0]
            rhs = struct.unpack("<f", struct.pack("<I", operands["rhs"]))[0]
            result = struct.unpack("<I", struct.pack("<f", lhs + rhs))[0]
        elif key == "i16x8.narrow_i32x4_u":
            result = min(operands["low"], 0xFFFF)
        else:
            assert key == "i8x16.narrow_i16x8_u"
            result = min(operands["low"], 0xFF)
        if key.startswith("i32x4.") and key.endswith(("lt_u", "ge_u", "gt_u")):
            result = 0xFFFFFFFF if result else 0
        values[next(iter(emit.results.values()))] = result & 0xFFFFFFFF
    result = values[ValueRef.result("result")]
    if vector_rule:
        result &= (
            0xFFFF if rule.emit[-1].descriptor.key.startswith("wasm.i16") else 0xFF
        )
    return result


def _type_guards(rule: DescriptorRule):
    return {
        guard.field: guard.type_pattern
        for guard in rule.guards
        if guard.kind is GuardKind.VALUE_TYPE
    }


def test_rules_cover_the_complete_f32_to_narrow_packet_family():
    rules = tuple(rule for rule in _rules() if rule.source_op is vector.vector_fptrunc)

    assert {rule.report_key for rule in rules} == {
        "wasm.float_narrowing.exact_f32_to_f8e4m3",
        "wasm.float_narrowing.exact_f32_to_f8e5m2",
        "wasm.float_narrowing.exact_f32_to_f16",
        "wasm.float_narrowing.exact_f32_to_bf16",
    }
    for rule in rules:
        type_guards = _type_guards(rule)
        assert type_guards["input"].elements == ("f32",)
        assert type_guards["input"].minimum_lanes == 1
        assert type_guards["input"].maximum_lanes == 4
        assert type_guards["result"].minimum_lanes == 1
        assert type_guards["result"].maximum_lanes == 4
        assert any(
            guard.kind is GuardKind.VALUE_STATIC_ELEMENT_COUNT_EQ
            for guard in rule.guards
        )
        assert all(
            "extract_lane" not in emit.descriptor.key
            and "replace_lane" not in emit.descriptor.key
            for emit in rule.emit
        )
        assert all(
            not emit.descriptor.key.startswith(("wasm.i32.", "wasm.f32."))
            or emit.descriptor.key == "wasm.i32.const"
            for emit in rule.emit
        )
        assert rule.emit[-1].descriptor.key == (
            "wasm.i16x8.narrow_i32x4_u"
            if type_guards["result"].elements[0] in ("f16", "bf16")
            else "wasm.i8x16.narrow_i16x8_u"
        )


def test_packed_recipes_match_exact_scalar_narrowing_at_boundaries():
    scalar_report_keys = {
        "f8E4M3": "exact_binary32_to_f8e4m3",
        "f8E5M2": "exact_binary32_to_f8e5m2",
        "f16": "exact_binary32_to_f16",
        "bf16": "exact_binary32_to_bfloat16",
    }
    scalar_rules = {
        result_element: next(
            rule
            for rule in _rules()
            if rule.source_op is scalar_conversion.scalar_fptrunc
            and rule.report_key == report_key
        )
        for result_element, report_key in scalar_report_keys.items()
    }
    vector_rules = {
        _type_guards(rule)["result"].elements[0]: rule
        for rule in _rules()
        if rule.source_op is vector.vector_fptrunc
    }
    window_bases = (
        0x00000000,
        0x007FFFE0,
        0x00800000,
        0x337FFFE0,
        0x387FFFE0,
        0x3A7FFFE0,
        0x3C7FFFE0,
        0x3EFFFFE0,
        0x3F7FFFE0,
        0x477FFFE0,
        0x477FEFE0,
        0x7F7FFFE0,
        0x7F7FFFFF,
        0x7F800000,
        0x7FC00000,
        0x7FFFFFFF,
    )
    inputs = {
        (base + offset) & 0xFFFFFFFF for base in window_bases for offset in range(-2, 3)
    }
    inputs |= {value | 0x80000000 for value in inputs}

    for result_element, vector_rule in vector_rules.items():
        scalar_rule = scalar_rules[result_element]
        for input_bits in inputs:
            assert _evaluate_lane(vector_rule, input_bits) == _evaluate_lane(
                scalar_rule, input_bits
            ), (result_element, hex(input_bits))
