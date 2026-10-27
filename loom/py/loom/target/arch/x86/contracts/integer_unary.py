# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native x86 signed integer unary contracts over logical SIMD carriers."""

from loom.dialect.vector import defs as vector
from loom.target.arch.x86.contracts.rule_builders import (
    DescriptorLookup,
    emit_descriptor_op,
    value_type_guards,
    vector_carriers,
)
from loom.target.arch.x86.vector_families import (
    AVX2_CARRIER_VECTOR_BIT_WIDTHS,
    AVX2_INTEGER_UNARY_FAMILIES,
    AVX512_CARRIER_VECTOR_BIT_WIDTHS,
    AVX512_INTEGER_UNARY_FAMILIES,
)
from loom.target.contracts import (
    DescriptorRule,
    Guard,
    ValueRef,
)


def integer_unary_rules(
    descriptor_lookup: DescriptorLookup, *, evex: bool
) -> tuple[DescriptorRule, ...]:
    profile = "avx512" if evex else "avx2"
    families = AVX512_INTEGER_UNARY_FAMILIES if evex else AVX2_INTEGER_UNARY_FAMILIES
    widths = (
        AVX512_CARRIER_VECTOR_BIT_WIDTHS if evex else AVX2_CARRIER_VECTOR_BIT_WIDTHS
    )
    rules = []
    for family in families:
        for carrier in vector_carriers(family.element, widths):
            descriptor = descriptor_lookup(
                f"x86.{profile}.{family.mnemonic}.{carrier.descriptor_suffix}"
            )
            rules.append(
                DescriptorRule(
                    source_op=getattr(vector, f"vector_{family.source_operation}"),
                    descriptor=descriptor,
                    guards=(
                        *(
                            (Guard.descriptor_available(descriptor),)
                            if descriptor.feature_mask_words
                            else ()
                        ),
                        *value_type_guards(("input", "result"), carrier.value_type),
                        Guard.low_value_register_class("input", carrier.register_class),
                        Guard.low_value_register_class(
                            "result", carrier.register_class
                        ),
                    ),
                    emit=(
                        emit_descriptor_op(
                            descriptor=descriptor,
                            operands={"input": ValueRef.operand("input")},
                            results={"dst": ValueRef.result("result")},
                        ),
                    ),
                    # EVEX permits the wider register bank when VL is available.
                    priority=int(evex),
                )
            )
    return tuple(rules)
