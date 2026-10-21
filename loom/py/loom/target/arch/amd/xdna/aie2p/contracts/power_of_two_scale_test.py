# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for exact AIE2P floating power-of-two scaling contracts."""

from __future__ import annotations

import struct
from dataclasses import dataclass

from loom.target.arch.amd.xdna.aie2p.contracts.power_of_two_scale import (
    AIE2P_POWER_OF_TWO_SCALE_RULES,
)
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    ValueProject,
    ValueProjectKind,
    ValueRef,
)
from loom.target.contracts.kinds import SourceValueKind


@dataclass(frozen=True)
class _Format:
    name: str
    bits: int
    exponent_bits: int
    mantissa_bits: int
    bias: int

    @property
    def sign_mask(self) -> int:
        return 1 << (self.bits - 1)

    @property
    def fraction_mask(self) -> int:
        return (1 << self.mantissa_bits) - 1

    @property
    def exponent_mask(self) -> int:
        return (1 << self.exponent_bits) - 1

    @property
    def infinity(self) -> int:
        return self.exponent_mask << self.mantissa_bits

    @property
    def quiet_bit(self) -> int:
        return 1 << (self.mantissa_bits - 1)

    @property
    def minimum_exponent(self) -> int:
        return 1 - self.bias - self.mantissa_bits

    @property
    def maximum_exponent(self) -> int:
        return self.exponent_mask - 1 - self.bias


_F16 = _Format("f16", 16, 5, 10, 15)
_F32 = _Format("f32", 32, 8, 23, 127)


def _u32(value: int) -> int:
    return value & 0xFFFFFFFF


def _s32(value: int) -> int:
    value = _u32(value)
    return value - (1 << 32) if value & 0x80000000 else value


def _logical_shift(value: int, amount: int) -> int:
    amount = _s32(amount)
    assert -31 <= amount <= 31
    if amount < 0:
        return _u32(value) >> -amount
    return _u32(value << amount)


def _power_bits(exponent: int, negative: bool, fmt: _Format) -> int:
    if exponent >= 1 - fmt.bias:
        magnitude = (exponent + fmt.bias) << fmt.mantissa_bits
    else:
        magnitude = 1 << (exponent - fmt.minimum_exponent)
    return magnitude | (fmt.sign_mask if negative else 0)


def _project(
    value: int | ValueProject,
    factor_bits: int,
    exponent: int,
) -> int:
    if isinstance(value, int):
        return value
    if value.kind is ValueProjectKind.FLOAT_BITS:
        return factor_bits << value.target_bit_offset
    if value.kind is ValueProjectKind.FLOAT_AS_F32_I32:
        return _s32(factor_bits)
    if value.kind is ValueProjectKind.FLOAT_POWER_OF_TWO_EXPONENT:
        return exponent << value.target_bit_offset
    if value.kind is ValueProjectKind.FLOAT_POWER_OF_TWO_NEGATED_EXPONENT:
        return -exponent << value.target_bit_offset
    raise AssertionError(f"unmodeled projection {value.kind}")


def _ref_value(
    ref: ValueRef,
    inputs: dict[str, int],
    values: dict[ValueRef, int],
) -> int:
    if ref.kind is SourceValueKind.OPERAND:
        return inputs[ref.field]
    return values[ref]


def _evaluate_scalar(
    rule: DescriptorRule,
    inputs: dict[str, int],
    factor_bits: int,
    exponent: int,
) -> int:
    values: dict[ValueRef, int] = {}
    for emit in rule.emit:
        assert isinstance(emit, EmitDescriptorOp)
        descriptor_key = emit.descriptor.key.removeprefix("amd.xdna.aie2p.")
        result_ref = next(iter(emit.results.values()))
        if emit.form is DescriptorEmitForm.CONST:
            value = _project(emit.immediates["i"], factor_bits, exponent)
        else:
            operands = {
                name: _ref_value(ref, inputs, values)
                for name, ref in emit.operands.items()
            }
            if descriptor_key in (
                "add.i32.immediate",
                "add.no-carry.i32.immediate",
            ):
                value = operands["s0"] + emit.immediates["imm"]
            elif descriptor_key in ("add.i32", "add.no-carry.i32"):
                value = operands["s0"] + operands["s1"]
            elif descriptor_key == "sub.i32":
                value = operands["s0"] - operands["s1"]
            elif descriptor_key == "and.i32":
                value = operands["s0"] & operands["s1"]
            elif descriptor_key == "or.i32":
                value = operands["s0"] | operands["s1"]
            elif descriptor_key == "xor.i32":
                value = operands["s0"] ^ operands["s1"]
            elif descriptor_key == "lshl.i32":
                value = _logical_shift(operands["s0"], operands["s1"])
            elif descriptor_key == "clz.i32":
                value = 32 - _u32(operands["s0"]).bit_length()
            elif descriptor_key == "cmp.eq.i32":
                value = int(_u32(operands["s0"]) == _u32(operands["s1"]))
            elif descriptor_key == "cmp.ult.i32":
                value = int(_u32(operands["s0"]) < _u32(operands["s1"]))
            elif descriptor_key == "cmp.uge.i32":
                value = int(_u32(operands["s0"]) >= _u32(operands["s1"]))
            elif descriptor_key == "cmp.slt.i32":
                value = int(_s32(operands["s0"]) < _s32(operands["s1"]))
            elif descriptor_key == "cmp.eqz.i32":
                value = int(_u32(operands["s0"]) == 0)
            elif descriptor_key == "cmp.nez.i32":
                value = int(_u32(operands["s0"]) != 0)
            elif descriptor_key == "select.zero.i32":
                value = operands["s0"] if operands["s2"] == 0 else operands["s1"]
            elif descriptor_key == "select.nonzero.i32":
                value = operands["s0"] if operands["s2"] != 0 else operands["s1"]
            else:
                raise AssertionError(f"unmodeled scalar descriptor {descriptor_key}")
        values[result_ref] = _u32(value)
    return values[ValueRef.result("result")]


def _evaluate_packet_lane(
    rule: DescriptorRule,
    inputs: dict[str, int],
    factor_bits: int,
    exponent: int,
) -> int:
    values: dict[ValueRef, int] = {}
    state: dict[str, int] = {}
    for emit in rule.emit:
        assert isinstance(emit, EmitDescriptorOp)
        descriptor_key = emit.descriptor.key.removeprefix("amd.xdna.aie2p.")
        if descriptor_key.startswith("state."):
            state[descriptor_key.removeprefix("state.").removesuffix(".immediate")] = (
                emit.immediates["i"]
            )
            continue
        result_ref = next(iter(emit.results.values()))
        if emit.form is DescriptorEmitForm.CONST:
            value = _project(emit.immediates["i"], factor_bits, exponent)
        else:
            operands = {
                name: _ref_value(ref, inputs, values)
                for name, ref in emit.operands.items()
            }
            if descriptor_key == "splat.i32x16":
                value = operands["src"]
            elif descriptor_key == "add.i32x16":
                value = operands["s1"] + operands["s2"]
            elif descriptor_key == "sub.i32x16":
                value = operands["s1"] - operands["s2"]
            elif descriptor_key == "and.bits512":
                value = operands["s1"] & operands["s2"]
            elif descriptor_key == "or.bits512":
                value = operands["s1"] | operands["s2"]
            elif descriptor_key == "cmp.eqz.i32x16.el.low32":
                value = int(_u32(operands["s2"]) == 0)
            elif descriptor_key == "cmp.lt.unsigned.i32x16.el.low32":
                value = int(_u32(operands["s1"]) < _u32(operands["s2"]))
            elif descriptor_key == "cmp.ge.unsigned.i32x16.el.low32":
                value = int(_u32(operands["s1"]) >= _u32(operands["s2"]))
            elif descriptor_key == "cmp.lt.signed.i32x16.el.low32":
                value = int(_s32(operands["s1"]) < _s32(operands["s2"]))
            elif descriptor_key == "predicate.complete.zero.high32":
                value = operands["storage"]
            elif descriptor_key == "select.i32x16.mask64":
                value = operands["s2"] if operands["sel"] else operands["s1"]
            elif descriptor_key.startswith("widen.2x.x-to-c."):
                assert state["saturation"] == 0
                assert state["ups-mode"] == 1
                value = _u32(operands["src"])
                if ".signed." in descriptor_key:
                    value = _s32(value)
                value <<= operands["su"]
            elif descriptor_key.startswith("narrow.2x.c-to-x."):
                assert state["saturation"] == 0
                assert state["srs-mode"] == 1
                assert state["rounding"] == 0
                value = operands["src"]
                if ".unsigned." in descriptor_key:
                    value &= (1 << 64) - 1
                value >>= operands["su"]
            else:
                raise AssertionError(f"unmodeled packet descriptor {descriptor_key}")
        values[result_ref] = value if "widen.2x." in descriptor_key else _u32(value)
    return values[ValueRef.result("result")]


def _float_value(bits: int, fmt: _Format) -> float:
    code = "e" if fmt is _F16 else "f"
    return struct.unpack("<" + code, bits.to_bytes(fmt.bits // 8, "little"))[0]


def _float_bits(value: float, fmt: _Format) -> int:
    code = "e" if fmt is _F16 else "f"
    try:
        return int.from_bytes(struct.pack("<" + code, value), "little")
    except OverflowError:
        return (fmt.sign_mask if value < 0.0 else 0) | fmt.infinity


def _reference(value_bits: int, factor_bits: int, fmt: _Format) -> int:
    magnitude = value_bits & (fmt.sign_mask - 1)
    if magnitude > fmt.infinity:
        return value_bits | fmt.quiet_bit
    return _float_bits(
        _float_value(value_bits, fmt) * _float_value(factor_bits, fmt), fmt
    )


def _boundary_values(fmt: _Format, scale_exponent: int) -> tuple[int, ...]:
    fraction_half = 1 << (fmt.mantissa_bits - 1)
    fractions = (0, fraction_half, fmt.fraction_mask)
    if scale_exponent < 0:
        distance = -scale_exponent
        exponent_candidates = {
            1,
            distance - fmt.mantissa_bits - 2,
            distance - fmt.mantissa_bits - 1,
            distance,
            distance + 1,
            fmt.exponent_mask - 1,
        }
    else:
        overflow_exponent = fmt.exponent_mask - scale_exponent
        exponent_candidates = {
            1,
            overflow_exponent - 1,
            overflow_exponent,
            overflow_exponent + 1,
            fmt.exponent_mask - 1,
        }

    magnitudes = {
        0,
        1,
        2,
        fmt.fraction_mask - 1,
        fmt.fraction_mask,
        1 << fmt.mantissa_bits,
        (1 << fmt.mantissa_bits) + 1,
        fmt.infinity - 1,
        fmt.infinity,
        fmt.infinity + 1,
        fmt.infinity | fmt.quiet_bit,
        fmt.infinity | fmt.fraction_mask,
    }
    for encoded_exponent in exponent_candidates:
        if 0 <= encoded_exponent <= fmt.exponent_mask:
            magnitudes.update(
                (encoded_exponent << fmt.mantissa_bits) | fraction
                for fraction in fractions
            )
    if scale_exponent < 0:
        distance = -scale_exponent
        implicit_bit = 1 << fmt.mantissa_bits
        for encoded_exponent in {
            1,
            max(1, distance - fmt.mantissa_bits),
            distance,
        }:
            if not 1 <= encoded_exponent < fmt.exponent_mask:
                continue
            shift = min(
                distance + 1 - min(max(encoded_exponent, 1), distance),
                fmt.mantissa_bits + 2,
            )
            half = 1 << (shift - 1)
            minimum_quotient = implicit_bit >> shift
            maximum_quotient = ((2 * implicit_bit - 1) - half) >> shift
            for parity in (0, 1):
                quotient = minimum_quotient
                if quotient & 1 != parity:
                    quotient += 1
                if quotient > maximum_quotient:
                    continue
                tie = (quotient << shift) + half
                magnitudes.add(
                    (encoded_exponent << fmt.mantissa_bits) | (tie - implicit_bit)
                )

    # Every factor sees both signed zeros and special values. Finite transition
    # cells use one input sign; factor sign coverage independently checks the
    # product-sign XOR without doubling the arithmetic matrix.
    signed_values = set(magnitudes)
    for magnitude in (
        0,
        1,
        1 << fmt.mantissa_bits,
        fmt.infinity - 1,
        fmt.infinity,
        fmt.infinity + 1,
        fmt.infinity | fmt.quiet_bit,
    ):
        signed_values.add(magnitude | fmt.sign_mask)
    return tuple(sorted(signed_values))


def _oracle_values(fmt: _Format, scale_exponent: int) -> tuple[int, ...]:
    values = {
        0,
        fmt.sign_mask,
        1,
        fmt.fraction_mask,
        1 << fmt.mantissa_bits,
        fmt.infinity - 1,
        fmt.infinity,
        fmt.infinity + 1,
        fmt.infinity | fmt.quiet_bit,
    }
    transition_exponents = {
        fmt.minimum_exponent,
        fmt.minimum_exponent + 1,
        -(fmt.mantissa_bits + 2),
        -(fmt.mantissa_bits + 1),
        -fmt.mantissa_bits,
        -2,
        -1,
        0,
        1,
        fmt.mantissa_bits,
        fmt.mantissa_bits + 1,
        fmt.maximum_exponent // 2,
        fmt.maximum_exponent - 1,
        fmt.maximum_exponent,
    }
    if scale_exponent in transition_exponents:
        values.update(_boundary_values(fmt, scale_exponent))
    return tuple(sorted(values))


def _rule(
    fmt: _Format,
    direction: str,
    factor_field: str,
    *,
    packet: bool = False,
) -> DescriptorRule:
    exponent_range = (
        (fmt.minimum_exponent, -1)
        if direction == "negative"
        else (0, fmt.maximum_exponent)
    )
    rules = [
        rule
        for rule in AIE2P_POWER_OF_TWO_SCALE_RULES
        if rule.guards[0].type_pattern is not None
        and rule.guards[0].type_pattern.element == fmt.name
        and (rule.guards[0].type_pattern.kind == "vector") == packet
        and rule.guards[-1].field == factor_field
        and (rule.guards[-1].minimum, rule.guards[-1].maximum) == exponent_range
    ]
    assert len(rules) == 1
    return rules[0]


def _inputs(
    factor_field: str,
    value_bits: int,
    factor_bits: int,
) -> dict[str, int]:
    value_field = "rhs" if factor_field == "lhs" else "lhs"
    return {value_field: value_bits, factor_field: factor_bits}


def test_power_of_two_scale_rules_cover_the_complete_admitted_family() -> None:
    assert len(AIE2P_POWER_OF_TWO_SCALE_RULES) == 12
    assert {rule.priority for rule in AIE2P_POWER_OF_TWO_SCALE_RULES} == {1}
    assert {rule.report_key for rule in AIE2P_POWER_OF_TWO_SCALE_RULES} == {
        "exact_power_of_two_float_scale"
    }
    for rule in AIE2P_POWER_OF_TWO_SCALE_RULES:
        rule.validate(AIE2P_CORE_DESCRIPTOR_SET)
    packet_rules = [
        rule
        for rule in AIE2P_POWER_OF_TWO_SCALE_RULES
        if rule.guards[0].type_pattern is not None
        and rule.guards[0].type_pattern.kind == "vector"
    ]
    assert len(packet_rules) == 4
    assert all(
        rule.guards[0].type_pattern.minimum_static_elements == 1
        and rule.guards[0].type_pattern.maximum_static_elements == 16
        for rule in packet_rules
        if rule.guards[0].type_pattern is not None
    )
    assert all(
        not any(word in emit.descriptor.key for word in ("extract", "insert"))
        for rule in packet_rules
        for emit in rule.emit
        if isinstance(emit, EmitDescriptorOp)
    )


def test_scalar_power_of_two_scale_matches_boundary_oracle() -> None:
    for fmt in (_F16, _F32):
        for exponent in range(fmt.minimum_exponent, fmt.maximum_exponent + 1):
            direction = "negative" if exponent < 0 else "nonnegative"
            rule = _rule(fmt, direction, "rhs")
            for negative_factor in (False, True):
                factor_bits = _power_bits(exponent, negative_factor, fmt)
                for value_bits in _oracle_values(fmt, exponent):
                    actual = _evaluate_scalar(
                        rule,
                        _inputs("rhs", value_bits, factor_bits),
                        factor_bits,
                        exponent,
                    )
                    expected = _reference(value_bits, factor_bits, fmt)
                    assert actual & ((1 << fmt.bits) - 1) == expected, (
                        fmt.name,
                        f"{value_bits:0{fmt.bits // 4}x}",
                        exponent,
                        negative_factor,
                        f"{actual:08x}",
                        f"{expected:0{fmt.bits // 4}x}",
                    )


def test_packet_power_of_two_scale_matches_boundary_oracle() -> None:
    for exponent in range(_F32.minimum_exponent, _F32.maximum_exponent + 1):
        direction = "negative" if exponent < 0 else "nonnegative"
        rule = _rule(_F32, direction, "rhs", packet=True)
        for negative_factor in (False, True):
            factor_bits = _power_bits(exponent, negative_factor, _F32)
            for value_bits in _oracle_values(_F32, exponent):
                actual = _evaluate_packet_lane(
                    rule,
                    _inputs("rhs", value_bits, factor_bits),
                    factor_bits,
                    exponent,
                )
                expected = _reference(value_bits, factor_bits, _F32)
                assert actual == expected, (
                    f"{value_bits:08x}",
                    exponent,
                    negative_factor,
                    f"{actual:08x}",
                    f"{expected:08x}",
                )


def test_power_of_two_scale_uses_the_selected_factor_operand() -> None:
    cases = (
        (_F16, -24, 0x3555),
        (_F16, 15, 0x7BFF),
        (_F32, -149, 0x3F2AAAAB),
        (_F32, 127, 0x00800001),
    )
    for fmt, exponent, value_bits in cases:
        direction = "negative" if exponent < 0 else "nonnegative"
        factor_bits = _power_bits(exponent, True, fmt)
        for factor_field in ("lhs", "rhs"):
            rule = _rule(fmt, direction, factor_field)
            actual = _evaluate_scalar(
                rule,
                _inputs(factor_field, value_bits, factor_bits),
                factor_bits,
                exponent,
            )
            assert actual & ((1 << fmt.bits) - 1) == _reference(
                value_bits, factor_bits, fmt
            )

        packet_rule = (
            _rule(_F32, direction, "lhs", packet=True) if fmt is _F32 else None
        )
        if packet_rule is not None:
            actual = _evaluate_packet_lane(
                packet_rule,
                _inputs("lhs", value_bits, factor_bits),
                factor_bits,
                exponent,
            )
            assert actual == _reference(value_bits, factor_bits, fmt)
