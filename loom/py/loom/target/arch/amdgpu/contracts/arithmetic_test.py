# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for AMDGPU arithmetic contract source tables."""

from __future__ import annotations

from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.amdgpu.contracts.arithmetic import (
    AMDGPU_ARITHMETIC_CONTRACT_DIALECT_OPS,
    AMDGPU_ARITHMETIC_CONTRACT_FRAGMENT,
)
from loom.target.contracts import (
    CompiledLowerRuleSet,
    LowerRule,
    compile_lower_rule_set,
)


def _compiled_arithmetic_rules() -> CompiledLowerRuleSet:
    return compile_lower_rule_set(
        AMDGPU_ARITHMETIC_CONTRACT_FRAGMENT,
        dialect_ops=AMDGPU_ARITHMETIC_CONTRACT_DIALECT_OPS,
    )


def _rules_for_source_op(
    compiled: CompiledLowerRuleSet,
    source_op: Op,
) -> tuple[LowerRule, ...]:
    for span in compiled.spans:
        if span.source_op is source_op:
            rule_end = span.rule_start + span.rule_count
            return compiled.rules[span.rule_start : rule_end]
    raise AssertionError(f"no lower-rule span for {source_op.name}")


def _rule_descriptor_keys(
    compiled: CompiledLowerRuleSet,
    rule: LowerRule,
) -> tuple[str, ...]:
    return tuple(
        compiled.emits[emit_index].descriptor.key
        for emit_index in range(rule.emit_start, rule.emit_start + rule.emit_count)
    )


def _descriptor_sequence_positions(
    compiled: CompiledLowerRuleSet,
    source_op: Op,
) -> dict[tuple[str, ...], int]:
    positions: dict[tuple[str, ...], int] = {}
    for ordinal, rule in enumerate(_rules_for_source_op(compiled, source_op)):
        descriptor_keys = _rule_descriptor_keys(compiled, rule)
        if descriptor_keys:
            positions.setdefault(descriptor_keys, ordinal)
    return positions


def test_unsigned_bitfield_extract_rules_try_native_bfe_before_shift_mask() -> None:
    positions = _descriptor_sequence_positions(
        _compiled_arithmetic_rules(),
        vector.vector_bitfield_extractu,
    )

    assert (
        positions[("amdgpu.v_bfe_u32.offset_width_inline",)]
        < positions[
            (
                "amdgpu.v_lshrrev_b32.src0_inline",
                "amdgpu.v_and_b32.src0_inline",
            )
        ]
    )
    assert (
        positions[("amdgpu.v_bfe_u32.offset_width_inline",)]
        < positions[
            (
                "amdgpu.v_lshrrev_b32.src0_inline",
                "amdgpu.v_and_b32.lit",
            )
        ]
    )


def test_signed_bitfield_extract_rules_try_native_bfe_before_shift_pair() -> None:
    positions = _descriptor_sequence_positions(
        _compiled_arithmetic_rules(),
        vector.vector_bitfield_extracts,
    )

    assert (
        positions[("amdgpu.v_bfe_i32.offset_width_inline",)]
        < positions[
            (
                "amdgpu.v_lshlrev_b32.src0_inline",
                "amdgpu.v_ashrrev_i32.src0_inline",
            )
        ]
    )


def test_bitfield_insert_rules_try_native_bfi_before_mask_merge_fallback() -> None:
    positions = _descriptor_sequence_positions(
        _compiled_arithmetic_rules(),
        vector.vector_bitfield_insert,
    )

    native_shift_bfi = (
        "amdgpu.v_lshlrev_b32.src0_inline",
        "amdgpu.v_bfi_b32.src0_lit",
    )
    assert (
        positions[native_shift_bfi]
        < positions[
            (
                "amdgpu.v_and_b32.src0_inline",
                "amdgpu.v_lshlrev_b32.src0_inline",
                "amdgpu.v_and_b32.lit",
                "amdgpu.v_or_b32",
            )
        ]
    )
    assert (
        positions[native_shift_bfi]
        < positions[
            (
                "amdgpu.v_and_b32.lit",
                "amdgpu.v_lshlrev_b32.src0_inline",
                "amdgpu.v_and_b32.lit",
                "amdgpu.v_or_b32",
            )
        ]
    )
