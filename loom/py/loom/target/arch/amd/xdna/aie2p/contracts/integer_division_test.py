# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Full-word semantic checks for the actual AIE2P descriptor programs."""

import random

from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.target.arch.amd.xdna.aie2p.contracts.integer_division import (
    _high_bit_rule,
    _magic_rule,
)
from loom.target.contracts import (
    Scalar,
    UnsignedDivisorMagicKind,
    ValueProject,
    ValueRef,
)
from loom.target.contracts.immediates import ValueProjectKind

_MASK = (1 << 32) - 1


def _evaluate(rule, numerator, divisor, multiplier=0, shift=0):
    values = {ValueRef.operand("lhs"): numerator, ValueRef.operand("rhs"): divisor}
    for emit in rule.emit:
        operands = {name: values[value] for name, value in emit.operands.items()}
        operation = emit.descriptor.key.removeprefix("amd.xdna.aie2p.")
        if operation.startswith("constant.i32"):
            value = emit.immediates["i"]
            if isinstance(value, ValueProject):
                if value.kind == ValueProjectKind.U32_DIVISOR_MAGIC_MULTIPLIER_AS_I32:
                    value = multiplier
                else:
                    assert value.kind == ValueProjectKind.U32_DIVISOR_MAGIC_SHIFT
                    value = shift
        elif operation == "and.i32":
            value = operands["s0"] & operands["s1"]
        elif operation == "xor.i32":
            value = operands["s0"] ^ operands["s1"]
        elif operation in ("add.i32", "add.no-carry.i32"):
            value = operands["s0"] + operands["s1"]
        elif operation == "sub.i32":
            value = operands["s0"] - operands["s1"]
        elif operation == "mul.i32":
            value = operands["s0"] * operands["s1"]
        elif operation == "madd.i32":
            value = operands["s0"] * operands["s1"] + operands["a0"]
        elif operation == "lshl.i32":
            count = operands["s1"]
            count = count - (1 << 32) if count >= 1 << 31 else count
            value = operands["s0"] >> -count if count < 0 else operands["s0"] << count
        elif operation == "cmp.ult.i32":
            value = int(operands["s0"] < operands["s1"])
        elif operation == "select.nonzero.i32":
            value = operands["s0"] if operands["s2"] else operands["s1"]
        else:
            raise AssertionError(operation)
        assert len(emit.results) == 1
        values[next(iter(emit.results.values()))] = value & _MASK
    return values[ValueRef.result("result")]


def _numerators(divisor, maximum=_MASK):
    values = [0, 1, 0xFFFF, 0x10000, 0x7FFFFFFF, 0x80000000, maximum]
    for quotient in (1, 2, 3, 0xFFFF, maximum // divisor):
        values.extend(
            quotient * divisor + delta
            for delta in (-1, 0, 1)
            if 0 <= quotient * divisor + delta <= maximum
        )
    randomizer = random.Random(divisor)
    values.extend(randomizer.randint(0, maximum) for _ in range(1024))
    return [value for value in values if value <= maximum]


def test_constant_unsigned_division_descriptor_programs():
    for divisor, multiplier, shift, kind, maximum in (
        (3, 0x55555556, 0, UnsignedDivisorMagicKind.MULTIPLY, 255),
        (7, 0x24924925, 0, UnsignedDivisorMagicKind.MULTIPLY, 65535),
        (3, 0xAAAAAAAB, 1, UnsignedDivisorMagicKind.MULTIPLY_SHIFT, _MASK),
        (7, 0x24924925, 2, UnsignedDivisorMagicKind.MULTIPLY_ADD_SHIFT, _MASK),
        (20, 0xCCCCCCCD, 4, UnsignedDivisorMagicKind.MULTIPLY_SHIFT, _MASK),
        (31, 0x08421085, 4, UnsignedDivisorMagicKind.MULTIPLY_ADD_SHIFT, _MASK),
        (0x7FFFFFFF, 3, 30, UnsignedDivisorMagicKind.MULTIPLY_ADD_SHIFT, _MASK),
    ):
        for source_op, remainder in (
            (scalar_arithmetic.scalar_divui, False),
            (scalar_arithmetic.scalar_remui, True),
        ):
            rule = _magic_rule(source_op, Scalar("i32"), kind=kind, remainder=remainder)
            for numerator in _numerators(divisor, maximum):
                expected = numerator % divisor if remainder else numerator // divisor
                assert (
                    _evaluate(rule, numerator, divisor, multiplier, shift) == expected
                )


def test_high_bit_unsigned_division_descriptor_programs():
    for divisor in (0x80000000, 0x80000001, 0xFFFFFFFD, 0xFFFFFFFF):
        for source_op, remainder in (
            (scalar_arithmetic.scalar_divui, False),
            (scalar_arithmetic.scalar_remui, True),
        ):
            rule = _high_bit_rule(source_op, remainder=remainder)
            for numerator in _numerators(divisor):
                expected = numerator % divisor if remainder else numerator // divisor
                assert _evaluate(rule, numerator, divisor) == expected
