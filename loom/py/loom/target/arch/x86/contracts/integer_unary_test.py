# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from functools import partial

from loom.dialect.vector import defs as vector
from loom.target.arch.x86.contracts.integer_unary import integer_unary_rules
from loom.target.arch.x86.descriptors import (
    X86_AVX2_DESCRIPTOR_SET,
    X86_AVX512_CORE_DESCRIPTOR_SET,
)
from loom.target.arch.x86.feature_bits import FEATURE_AVX512_VL
from loom.target.contracts import GuardKind, Vector, descriptor_by_key
from loom.target.low_descriptors import OperandAddressMapKind


def test_native_unary_matrix_includes_partial_and_full_carriers() -> None:
    for evex, descriptor_set in (
        (False, X86_AVX2_DESCRIPTOR_SET),
        (True, X86_AVX512_CORE_DESCRIPTOR_SET),
    ):
        rules = integer_unary_rules(
            partial(descriptor_by_key, descriptor_set), evex=evex
        )
        cells = set()
        for rule in rules:
            rule.validate(descriptor_set)
            types = {
                guard.field: guard.type_pattern
                for guard in rule.guards
                if guard.kind is GuardKind.VALUE_TYPE
            }
            assert types["input"] == types["result"]
            cells.add((rule.source_op, types["input"]))
        expected = set()
        for width in (8, 16, 32, 64) if evex else (8, 16, 32):
            expected.add(
                (
                    vector.vector_absi,
                    Vector(f"i{width}", minimum_lanes=1, maximum_lanes=128 // width),
                )
            )
            expected.add((vector.vector_absi, Vector(f"i{width}", lanes=256 // width)))
            if evex:
                expected.add(
                    (vector.vector_absi, Vector(f"i{width}", lanes=512 // width))
                )
        assert cells == expected
        assert len(cells) == len(rules)


def test_native_encodings_and_register_feature_boundaries() -> None:
    # Map 0F38, prefix 66. Absolute value reserves VEX.vvvv/EVEX.vvvv;
    # the source is ModRM.r/m for every element width and vector length.
    for evex, descriptor_set in (
        (False, X86_AVX2_DESCRIPTOR_SET),
        (True, X86_AVX512_CORE_DESCRIPTOR_SET),
    ):
        profile = "avx512" if evex else "avx2"
        forms = [("vpabsb", 0x1C, 0), ("vpabsw", 0x1D, 0), ("vpabsd", 0x1E, 0)]
        if evex:
            forms.append(("vpabsq", 0x1F, 0x1000))
        widths = [("xmm", 0), ("ymm", 0x4000)]
        if evex:
            widths.append(("zmm", 0x8000))
        for mnemonic, opcode, w_bit in forms:
            for suffix, width_bits in widths:
                descriptor = descriptor_by_key(
                    descriptor_set, f"x86.{profile}.{mnemonic}.{suffix}"
                )
                assert descriptor.encoding_format_id == 0x8140
                assert descriptor.encoding_id == (
                    0x0600 | opcode | w_bit | width_bits | (0x2000 if evex else 0)
                )
                assert descriptor.feature_mask_words == (
                    (FEATURE_AVX512_VL,) if evex and suffix != "zmm" else ()
                )
                for operand in descriptor.operands:
                    if evex:
                        assert (
                            operand.address_map_kind
                            is not OperandAddressMapKind.LOW_SUBSET
                        )
                    else:
                        assert (
                            operand.address_map_kind is OperandAddressMapKind.LOW_SUBSET
                        )
                        assert operand.addressable_unit_count == 16
