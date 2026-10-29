# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for AMD XDNA AIE2P floating-point dot-product contracts."""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from loom.dialect.vector import defs as vector
from loom.target.arch.amd.xdna.aie2p.contracts.floating_dot import (
    AIE2P_FLOATING_DOT_RULES,
    _convert_f16_to_bf16,
)
from loom.target.arch.amd.xdna.aie2p.contracts.packet_program import (
    PacketProgram,
)
from loom.target.contracts import (
    DescriptorEmitForm,
    EmitDescriptorOp,
    Guard,
    ValueRef,
    Vector,
)


def _floating_dot_rules(element: str):
    return tuple(
        rule
        for rule in AIE2P_FLOATING_DOT_RULES
        if rule.source_op is vector.vector_dot2f
        and any(
            guard.field == "lhs"
            and guard.type_pattern is not None
            and guard.type_pattern.element == element
            for guard in rule.guards
        )
    )


def _conversion_program():
    program = PacketProgram(16, "test_")
    source = ValueRef.operand("input")
    zero = program.binary("zero", "sub.i16x32", source, source)
    result = _convert_f16_to_bf16(
        program,
        {},
        zero,
        source,
        "input",
    )
    return program, result


def _round_nearest_even(value: int, shift: int) -> int:
    quotient, remainder = divmod(value & 0xFFFFFFFF, 1 << shift)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and quotient & 1):
        quotient += 1
    return quotient & 0xFFFF


def _evaluate_conversion(
    program: PacketProgram,
    conversion_result: ValueRef,
    input_bits: tuple[int, ...],
) -> tuple[int, ...]:
    """Evaluates the emitted conversion program over one packet."""

    assert len(input_bits) == 32
    values: dict[ValueRef, int | tuple[int, ...]] = {
        ValueRef.operand("input"): input_bits,
    }
    state: dict[str, int] = {}
    for emit in program.emits:
        assert isinstance(emit, EmitDescriptorOp)
        key = emit.descriptor.key.removeprefix("amd.xdna.aie2p.")
        if key.startswith("state."):
            state[key.removeprefix("state.").removesuffix(".immediate")] = (
                emit.immediates["i"]
            )
            continue

        operands = {name: values[ref] for name, ref in emit.operands.items()}
        if emit.form is DescriptorEmitForm.CONST:
            result: int | tuple[int, ...] = emit.immediates["i"]
        elif key == "splat.i16x32":
            source = operands["src"]
            assert isinstance(source, int)
            result = (source & 0xFFFF,) * 32
        elif key in ("add.i16x32", "sub.i16x32"):
            lhs = operands["s1"]
            rhs = operands["s2"]
            assert isinstance(lhs, tuple) and isinstance(rhs, tuple)
            operation = int.__add__ if key == "add.i16x32" else int.__sub__
            result = tuple(
                operation(a, b) & 0xFFFF for a, b in zip(lhs, rhs, strict=True)
            )
        elif key in ("and.bits512", "or.bits512"):
            lhs = operands["s1"]
            rhs = operands["s2"]
            assert isinstance(lhs, tuple) and isinstance(rhs, tuple)
            operation = int.__and__ if key == "and.bits512" else int.__or__
            result = tuple(operation(a, b) for a, b in zip(lhs, rhs, strict=True))
        elif key.startswith(("cmp.lt.unsigned.i16x32", "cmp.ge.unsigned.i16x32")):
            lhs = operands["s1"]
            rhs = operands["s2"]
            assert isinstance(lhs, tuple) and isinstance(rhs, tuple)
            compare = int.__lt__ if key.startswith("cmp.lt.") else int.__ge__
            result = sum(
                compare(a & 0xFFFF, b & 0xFFFF) << lane
                for lane, (a, b) in enumerate(zip(lhs, rhs, strict=True))
            )
        elif key == "predicate.complete.zero.high32":
            result = operands["storage"]
            assert isinstance(result, int)
        elif key == "select.i16x32.mask64":
            false_values = operands["s1"]
            true_values = operands["s2"]
            predicate = operands["sel"]
            assert isinstance(false_values, tuple)
            assert isinstance(true_values, tuple)
            assert isinstance(predicate, int)
            result = tuple(
                true_value if predicate & (1 << lane) else false_value
                for lane, (false_value, true_value) in enumerate(
                    zip(false_values, true_values, strict=True)
                )
            )
        elif key == "widen.2x.x-to-c.unsigned.configured":
            source = operands["src"]
            shift = operands["su"]
            assert isinstance(source, tuple) and isinstance(shift, int)
            assert state["saturation"] == 0 and state["ups-mode"] == 0
            result = tuple((value & 0xFFFF) << shift for value in source)
        elif key == "narrow.2x.c-to-x.unsigned.configured":
            source = operands["src"]
            shift = operands["su"]
            assert isinstance(source, tuple) and isinstance(shift, int)
            assert state["saturation"] == 0
            assert state["srs-mode"] == 0 and state["rounding"] == 12
            result = tuple(_round_nearest_even(value, shift) for value in source)
        else:
            raise AssertionError(f"unmodeled conversion descriptor {key}")
        assert len(emit.results) == 1
        values[next(iter(emit.results.values()))] = result

    result = values[conversion_result]
    assert isinstance(result, tuple)
    return result


def _reference_conversion(input_bits: int) -> int:
    sign = input_bits & 0x8000
    magnitude = input_bits & 0x7FFF
    if magnitude < 0x0400:
        return sign
    if magnitude >= 0x7C00:
        return sign | (0x7FC0 if magnitude > 0x7C00 else 0x7F80)

    quotient, remainder = divmod(magnitude, 8)
    if remainder > 4 or (remainder == 4 and quotient & 1):
        quotient += 1
    return sign | (quotient + 0x3800)


def _assert_conversion_packets(input_packets: Iterable[tuple[int, ...]]) -> None:
    program, conversion_result = _conversion_program()
    for input_bits in input_packets:
        assert _evaluate_conversion(program, conversion_result, input_bits) == tuple(
            map(_reference_conversion, input_bits)
        )


def test_bf16_pair_dot_rules_cover_packed_and_interleaved_representations() -> None:
    bf16_rules = _floating_dot_rules("bf16")
    assert len(bf16_rules) == 16
    packed_rules = [rule for rule in bf16_rules if not rule.source_nodes]
    interleaved_rules = [rule for rule in bf16_rules if rule.source_nodes]
    assert len(packed_rules) == 8
    assert len(interleaved_rules) == 8
    assert all(rule.source_op is vector.vector_dot2f for rule in packed_rules)
    assert all(
        Guard.value_type(
            "lhs",
            Vector("bf16", minimum_static_elements=2, maximum_static_elements=32),
        )
        in rule.guards
        for rule in packed_rules
    )


def test_f16_pair_dot_uses_one_bounded_native_packet_rule() -> None:
    rules = _floating_dot_rules("f16")
    assert len(rules) == 1
    rule = rules[0]
    assert rule.source_op is vector.vector_dot2f
    assert rule.report_key == "f16_dot2_bf16_round_nearest"
    assert len(rule.emit) == 61
    input_type = Vector("f16", minimum_static_elements=2, maximum_static_elements=32)
    result_type = Vector("f32", minimum_static_elements=1, maximum_static_elements=16)
    assert Guard.value_type("lhs", input_type) in rule.guards
    assert Guard.value_type("rhs", input_type) in rule.guards
    assert Guard.value_type("acc", result_type) in rule.guards
    assert Guard.value_type("result", result_type) in rule.guards
    descriptor_keys = tuple(
        emit.descriptor.key for emit in rule.emit if isinstance(emit, EmitDescriptorOp)
    )
    assert descriptor_keys.count("amd.xdna.aie2p.accumulate.bf16x16.configured") == 1
    assert descriptor_keys.count("amd.xdna.aie2p.accumulate.bf16x32.configured") == 1


def test_f16_to_bf16_handles_subnormal_flush_and_rounding_ties() -> None:
    values = (
        0x0000,
        0x8000,
        0x0001,
        0x03FF,
        0x0400,
        0x0401,
        0x0402,
        0x0403,
        0x0404,
        0x0405,
        0x0406,
        0x0407,
        0x3C00,
        0x3C07,
        0x7BFF,
        0x7C00,
        0xFC00,
        0x7C01,
        0x7DFF,
        0x7FFF,
        0xFC01,
        0xFFFF,
    )
    packet = values + (0,) * (32 - len(values))
    _assert_conversion_packets((packet,))


@pytest.mark.exhaustive
def test_f16_to_bf16_exhaustive() -> None:
    _assert_conversion_packets(
        tuple(range(base, base + 32)) for base in range(0, 1 << 16, 32)
    )
