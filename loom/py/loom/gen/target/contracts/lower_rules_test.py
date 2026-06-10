# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.gen.target.contracts.lower_rules import generate_lower_rule_set
from loom.target.contracts import (
    ContractFragment,
    DescriptorAccumulatorSeed,
    DescriptorAccumulatorTree,
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    ValueProject,
    ValueRef,
    Vector,
)
from loom.target.test.descriptors import (
    TEST_LOW_ADD_F32_DESCRIPTOR,
    TEST_LOW_CONST_I32_DESCRIPTOR,
    TEST_LOW_CORE_DESCRIPTOR_SET,
)


def test_generate_lower_rule_set_emits_value_ref_for_f64_equals_guard() -> None:
    table = ContractFragment(
        name="test.low.f64_equals",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=scalar_arithmetic.scalar_mulf,
                descriptor=TEST_LOW_ADD_F32_DESCRIPTOR,
                guards=(
                    Guard.value_type("lhs", Scalar("f32")),
                    Guard.value_f64_equals("rhs", 0.0),
                    Guard.value_type("rhs", Scalar("f32")),
                    Guard.value_type("result", Scalar("f32")),
                ),
                emit=(
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_ADD_F32_DESCRIPTOR,
                        operands={
                            "lhs": ValueRef.operand("lhs"),
                            "rhs": ValueRef.operand("rhs"),
                        },
                        results={"dst": ValueRef.result("result")},
                    ),
                ),
            )
        ],
    )

    generated = generate_lower_rule_set(table, dialect_ops={"scalar": ALL_SCALAR_OPS})

    guard_start = generated.source.index("LOOM_LOW_LOWER_GUARD_VALUE_F64_EQUALS")
    guard_end = generated.source.index("},", guard_start)
    guard_text = generated.source[guard_start:guard_end]
    assert ".value_ref_index = 1," in guard_text
    assert ".u64 = UINT64_C(0)," in guard_text


def test_generate_lower_rule_set_emits_storage_element_format_guard() -> None:
    table = ContractFragment(
        name="test.low.storage_schema",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=vector.vector_fragment_load,
                descriptor=TEST_LOW_ADD_F32_DESCRIPTOR,
                guards=(
                    Guard.value_storage_element_format(
                        "view",
                        "LOOM_VALUE_FACT_NUMERIC_FORMAT_U8",
                    ),
                    Guard.value_type("result", Vector("f32", lanes=4)),
                ),
                emit=(
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_ADD_F32_DESCRIPTOR,
                        operands={
                            "lhs": ValueRef.result("result"),
                            "rhs": ValueRef.result("result"),
                        },
                        results={"dst": ValueRef.result("result")},
                    ),
                ),
            )
        ],
    )

    generated = generate_lower_rule_set(table, dialect_ops={"vector": ALL_VECTOR_OPS})

    guard_start = generated.source.index("LOOM_LOW_LOWER_GUARD_VALUE_STORAGE_ELEMENT_FORMAT")
    guard_end = generated.source.index("},", guard_start)
    guard_text = generated.source[guard_start:guard_end]
    assert ".value_ref_index = 0," in guard_text
    assert ".u64 = LOOM_VALUE_FACT_NUMERIC_FORMAT_U8," in guard_text


def test_generate_lower_rule_set_emits_balanced_accumulator_flag() -> None:
    table = ContractFragment(
        name="test.low.accumulate_tree",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=vector.vector_reduce,
                descriptor=TEST_LOW_ADD_F32_DESCRIPTOR,
                guards=(
                    Guard.value_type("input", Vector("f32", lanes=4)),
                    Guard.value_type("init", Scalar("f32")),
                    Guard.value_type("result", Scalar("f32")),
                ),
                emit=(
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_ADD_F32_DESCRIPTOR,
                        operands={
                            "lhs": ValueRef.operand("input"),
                            "rhs": ValueRef.operand("input"),
                        },
                        results={"dst": ValueRef.result("result")},
                        form=DescriptorEmitForm.ACCUMULATE_LANES,
                        accumulator="lhs",
                        accumulator_seed=DescriptorAccumulatorSeed.FIRST_LANE,
                        accumulator_tree=DescriptorAccumulatorTree.BALANCED,
                    ),
                ),
            )
        ],
    )

    generated = generate_lower_rule_set(table, dialect_ops={"vector": ALL_VECTOR_OPS})

    assert "LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_ACCUMULATE_LANES" in generated.source
    assert "LOOM_LOW_LOWER_EMIT_FLAG_ACCUMULATE_SEED_FIRST_LANE" in generated.source
    assert "LOOM_LOW_LOWER_EMIT_FLAG_ACCUMULATE_TREE_BALANCED" in generated.source


def test_generate_lower_rule_set_emits_balanced_operand_seed() -> None:
    table = ContractFragment(
        name="test.low.accumulate_operand_tree",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=vector.vector_reduce,
                descriptor=TEST_LOW_ADD_F32_DESCRIPTOR,
                guards=(
                    Guard.value_type("input", Vector("f32", lanes=4)),
                    Guard.value_type("init", Scalar("f32")),
                    Guard.value_type("result", Scalar("f32")),
                ),
                emit=(
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_ADD_F32_DESCRIPTOR,
                        operands={
                            "lhs": ValueRef.operand("init"),
                            "rhs": ValueRef.operand("input"),
                        },
                        results={"dst": ValueRef.result("result")},
                        form=DescriptorEmitForm.ACCUMULATE_LANES,
                        accumulator="lhs",
                        accumulator_tree=DescriptorAccumulatorTree.BALANCED,
                    ),
                ),
            )
        ],
    )

    generated = generate_lower_rule_set(table, dialect_ops={"vector": ALL_VECTOR_OPS})

    assert "LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_ACCUMULATE_LANES" in generated.source
    assert "LOOM_LOW_LOWER_EMIT_FLAG_ACCUMULATE_TREE_BALANCED" in generated.source
    assert "LOOM_LOW_LOWER_EMIT_FLAG_ACCUMULATE_SEED_FIRST_LANE" not in generated.source


def test_generate_lower_rule_set_emits_divisor_magic_projection() -> None:
    table = ContractFragment(
        name="test.low.divisor_magic",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=scalar_arithmetic.scalar_addi,
                descriptor=TEST_LOW_CONST_I32_DESCRIPTOR,
                guards=(
                    Guard.value_type("lhs", Scalar("i32")),
                    Guard.value_u32_divisor_magic_is_add("rhs", True),
                    Guard.value_type("rhs", Scalar("i32")),
                    Guard.value_type("result", Scalar("i32")),
                ),
                emit=(
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_CONST_I32_DESCRIPTOR,
                        results={"dst": ValueRef.result("result")},
                        immediates={"i32_value": ValueProject.u32_divisor_magic_multiplier("rhs")},
                    ),
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_CONST_I32_DESCRIPTOR,
                        results={"dst": ValueRef.temporary("shift")},
                        result_types={"dst": ValueRef.result("result")},
                        immediates={"i32_value": ValueProject.u32_divisor_magic_shift("rhs")},
                    ),
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_CONST_I32_DESCRIPTOR,
                        results={"dst": ValueRef.temporary("mask")},
                        result_types={"dst": ValueRef.result("result")},
                        immediates={"i32_value": ValueProject.exact_i64_minus_one("rhs")},
                    ),
                ),
            )
        ],
    )

    generated = generate_lower_rule_set(table, dialect_ops={"scalar": ALL_SCALAR_OPS})

    assert "LOOM_LOW_LOWER_GUARD_VALUE_U32_DIVISOR_MAGIC_IS_ADD" in generated.source
    assert "LOOM_LOW_LOWER_ATTR_COPY_VALUE_U32_DIVISOR_MAGIC_MULTIPLIER" in generated.source
    assert "LOOM_LOW_LOWER_ATTR_COPY_VALUE_U32_DIVISOR_MAGIC_SHIFT" in generated.source
    assert "LOOM_LOW_LOWER_ATTR_COPY_VALUE_EXACT_I64_MINUS_ONE" in generated.source
