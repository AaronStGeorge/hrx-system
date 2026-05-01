# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.contracts import (
    ContractFragment,
    DescriptorRule,
    DirectDescriptorCase,
    EmitDescriptorOp,
    Guard,
    LowerAttrCopyKind,
    LowerEmitKind,
    Scalar,
    TypePattern,
    ValueAliasRule,
    ValueElideRule,
    ValueProject,
    ValueRef,
    Vector,
    binary_descriptor_rules,
    compile_lower_rule_set,
)
from loom.target.test.descriptors import (
    TEST_LOW_ADD_I32_DESCRIPTOR,
    TEST_LOW_CONST_I32_DESCRIPTOR,
    TEST_LOW_CORE_DESCRIPTOR_SET,
)


def _binary_rule(
    *,
    source_op: Op,
    type_pattern: TypePattern,
) -> DescriptorRule:
    return binary_descriptor_rules(
        (
            DirectDescriptorCase(
                source_op=source_op,
                descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
                type_patterns=type_pattern,
            ),
        )
    )[0]


def test_compile_lower_rule_set_compiles_direct_scalar_rule() -> None:
    table = ContractFragment(
        name="test.scalar",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            _binary_rule(
                source_op=scalar_arithmetic.scalar_addi,
                type_pattern=Scalar("i32"),
            )
        ],
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"scalar": ALL_SCALAR_OPS})

    assert compiled.authored_case_indices == (0,)
    assert len(compiled.spans) == 1
    assert compiled.spans[0].source_op is scalar_arithmetic.scalar_addi
    assert compiled.spans[0].rule_start == 0
    assert compiled.spans[0].rule_count == 1

    assert len(compiled.rules) == 1
    assert compiled.rules[0].source_op is scalar_arithmetic.scalar_addi
    assert compiled.rules[0].guard_count == 3
    assert compiled.rules[0].emit_count == 1

    assert len(compiled.emits) == 1
    assert compiled.emits[0].kind == LowerEmitKind.DESCRIPTOR_OP
    assert compiled.emits[0].descriptor is TEST_LOW_ADD_I32_DESCRIPTOR
    assert compiled.emits[0].operand_ref_count == 2
    assert compiled.emits[0].result_ref_count == 1


def test_compile_lower_rule_set_infers_vector_per_lane_emit() -> None:
    table = ContractFragment(
        name="test.vector",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            _binary_rule(
                source_op=vector.vector_addi,
                type_pattern=Vector("i32", lanes=4),
            )
        ],
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"vector": ALL_VECTOR_OPS})

    assert len(compiled.rules) == 1
    assert compiled.rules[0].source_op is vector.vector_addi
    assert len(compiled.emits) == 1
    assert compiled.emits[0].kind == LowerEmitKind.DESCRIPTOR_OP_PER_LANE
    assert compiled.emits[0].descriptor is TEST_LOW_ADD_I32_DESCRIPTOR


def test_compile_lower_rule_set_skips_value_alias_cases() -> None:
    table = ContractFragment(
        name="test.alias",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            ValueAliasRule(
                source_op=vector.vector_fragment,
                source=ValueRef.operand("data"),
                result=ValueRef.result("result"),
            )
        ],
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"vector": ALL_VECTOR_OPS})

    assert compiled.rules == ()
    assert compiled.spans == ()
    assert compiled.authored_case_indices == ()


def test_compile_lower_rule_set_compiles_value_elide_cases() -> None:
    table = ContractFragment(
        name="test.elide",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            ValueElideRule(
                source_op=vector.vector_extract,
                values=(ValueRef.result("result"),),
            )
        ],
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"vector": ALL_VECTOR_OPS})

    assert compiled.authored_case_indices == (0,)
    assert len(compiled.rules) == 1
    assert compiled.rules[0].source_op is vector.vector_extract
    assert compiled.rules[0].emit_count == 0
    assert compiled.rules[0].elide_ref_count == 1
    assert len(compiled.value_refs) == 1
    assert compiled.spans[0].source_op is vector.vector_extract


def test_compile_lower_rule_set_rejects_descriptor_rule_without_emit() -> None:
    table = ContractFragment(
        name="test.no-emit",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=scalar_arithmetic.scalar_addi,
                descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match=r"scalar.addi: descriptor-rule contracts must author their emit",
    ):
        compile_lower_rule_set(table, dialect_ops={"scalar": ALL_SCALAR_OPS})


def test_compile_lower_rule_set_compiles_const_immediate_emit() -> None:
    table = ContractFragment(
        name="test.immediate",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=scalar_conversion.scalar_constant,
                descriptor=TEST_LOW_CONST_I32_DESCRIPTOR,
                guards=(Guard.value_type("result", Scalar("i32")),),
                emit=(
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_CONST_I32_DESCRIPTOR,
                        results={"dst": ValueRef.result("result")},
                        immediates={"i32_value": 0},
                    ),
                ),
            )
        ],
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"scalar": ALL_SCALAR_OPS})

    assert len(compiled.emits) == 1
    assert compiled.emits[0].kind == LowerEmitKind.DESCRIPTOR_CONST
    assert compiled.emits[0].attr_copy_count == 1
    assert len(compiled.attr_copies) == 1
    assert compiled.attr_copies[0].kind == LowerAttrCopyKind.I64_LITERAL
    assert compiled.attr_copies[0].literal_i64 == 0


def test_compile_lower_rule_set_compiles_value_fact_immediate_emit() -> None:
    table = ContractFragment(
        name="test.value-immediate",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=scalar_arithmetic.scalar_addi,
                descriptor=TEST_LOW_CONST_I32_DESCRIPTOR,
                guards=(Guard.value_type("result", Scalar("i32")),),
                emit=(
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_CONST_I32_DESCRIPTOR,
                        results={"dst": ValueRef.result("result")},
                        immediates={
                            "i32_value": ValueProject.i32_as_u32_bits("lhs"),
                        },
                    ),
                ),
            )
        ],
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"scalar": ALL_SCALAR_OPS})

    assert len(compiled.attr_copies) == 1
    assert compiled.attr_copies[0].kind == LowerAttrCopyKind.VALUE_I32_AS_U32_BITS
    value_ref = compiled.value_refs[compiled.attr_copies[0].value_ref_index]
    assert value_ref.index == 0
