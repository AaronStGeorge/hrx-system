# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target contract tables for the backend-independent test-low target."""

from __future__ import annotations

from pathlib import Path

from loom.dialect.index import ALL_INDEX_OPS
from loom.dialect.index import defs as index
from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.contracts import (
    ContractTable,
    DescriptorRule,
    DirectDescriptorCase,
    DirectTypePatterns,
    PredicateDescriptorCase,
    Scalar,
    SelectDescriptorCase,
    TypePattern,
    Vector,
    binary_descriptor_rules,
    compare_descriptor_rules,
    select_descriptor_rules,
    ternary_descriptor_rules,
)
from loom.target.low_descriptors import Descriptor
from loom.target.test.descriptors import (
    TEST_LOW_ADD_F32_DESCRIPTOR,
    TEST_LOW_ADD_I32_DESCRIPTOR,
    TEST_LOW_CMP_EQ_I32_DESCRIPTOR,
    TEST_LOW_CONST_I32_DESCRIPTOR,
    TEST_LOW_CORE_DESCRIPTOR_SET,
    TEST_LOW_DOT4I_S8S8_DESCRIPTOR,
    TEST_LOW_MUL_F32_DESCRIPTOR,
    TEST_LOW_MUL_I32_DESCRIPTOR,
    TEST_LOW_SELECT_I32_DESCRIPTOR,
    TEST_LOW_SHUFFLE_V4I32_DESCRIPTOR,
    TEST_LOW_SUB_F32_DESCRIPTOR,
    TEST_LOW_TIED_ANY_DESCRIPTOR,
)


def _rule(source_op: Op, descriptor: Descriptor) -> DescriptorRule:
    return DescriptorRule(source_op=source_op, descriptor=descriptor)


def _binary_rule(
    source_op: Op,
    descriptor: Descriptor,
    type_patterns: DirectTypePatterns,
    *,
    semantic_tag: str | None = None,
) -> DescriptorRule:
    return binary_descriptor_rules(
        (
            DirectDescriptorCase(
                source_op,
                descriptor,
                type_patterns,
                semantic_tag=semantic_tag,
            ),
        )
    )[0]


def _select_rule(
    source_op: Op,
    descriptor: Descriptor,
    *,
    condition_type: TypePattern,
    value_type: TypePattern,
    semantic_tag: str | None = None,
) -> DescriptorRule:
    return select_descriptor_rules(
        (
            SelectDescriptorCase(
                source_op,
                descriptor,
                condition_type=condition_type,
                value_type=value_type,
                semantic_tag=semantic_tag,
            ),
        )
    )[0]


def _compare_rule(
    source_op: Op,
    descriptor: Descriptor,
    *,
    predicate: str,
    operand_type: TypePattern,
    result_type: TypePattern,
    semantic_tag: str | None = None,
) -> DescriptorRule:
    return compare_descriptor_rules(
        source_op,
        (
            PredicateDescriptorCase(
                predicate,
                descriptor,
                semantic_tag=semantic_tag,
            ),
        ),
        operand_type=operand_type,
        result_type=result_type,
    )[0]


def _ternary_rule(
    source_op: Op,
    descriptor: Descriptor,
    type_patterns: DirectTypePatterns,
    *,
    semantic_tag: str | None = None,
    source_a: str = "a",
    source_b: str = "b",
    source_c: str = "c",
    descriptor_a: str = "a",
    descriptor_b: str = "b",
    descriptor_c: str = "c",
) -> DescriptorRule:
    return ternary_descriptor_rules(
        (
            DirectDescriptorCase(
                source_op,
                descriptor,
                type_patterns,
                semantic_tag=semantic_tag,
            ),
        ),
        source_a=source_a,
        source_b=source_b,
        source_c=source_c,
        descriptor_a=descriptor_a,
        descriptor_b=descriptor_b,
        descriptor_c=descriptor_c,
    )[0]


_I32 = Scalar("i32")
_F32 = Scalar("f32")
_V4I1 = Vector("i1", lanes=4)
_V16I8 = Vector("i8", lanes=16)
_V4I32 = Vector("i32", lanes=4)
_V4F32 = Vector("f32", lanes=4)


TEST_LOW_CORE_CONTRACT_DIALECT_OPS = {
    "scalar": ALL_SCALAR_OPS,
    "vector": ALL_VECTOR_OPS,
    "index": ALL_INDEX_OPS,
}

TEST_LOW_CORE_CONTRACT_TABLE = ContractTable(
    name="test.low.core",
    descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
    table_index=0,
    c_header_path=Path("loom/src/loom/target/test/contract_table.h"),
    c_source_path=Path("loom/src/loom/target/test/contract_table.c"),
    header_guard="LOOM_TARGET_TEST_CONTRACT_TABLE_H_",
    public_header="loom/target/test/contract_table.h",
    symbol_name="loom_test_low_core_contract_table",
    c_table_prefix="TestLowCoreContract",
    cases=[
        _binary_rule(
            scalar_arithmetic.scalar_addi,
            TEST_LOW_ADD_I32_DESCRIPTOR,
            _I32,
            semantic_tag="integer.add.i32",
        ),
        _rule(scalar_arithmetic.scalar_subi, TEST_LOW_TIED_ANY_DESCRIPTOR),
        _binary_rule(
            scalar_arithmetic.scalar_muli,
            TEST_LOW_MUL_I32_DESCRIPTOR,
            _I32,
            semantic_tag="integer.mul.i32",
        ),
        _binary_rule(
            scalar_arithmetic.scalar_addf,
            TEST_LOW_ADD_F32_DESCRIPTOR,
            _F32,
            semantic_tag="float.add.f32",
        ),
        _binary_rule(
            scalar_arithmetic.scalar_subf,
            TEST_LOW_SUB_F32_DESCRIPTOR,
            _F32,
            semantic_tag="float.sub.f32",
        ),
        _binary_rule(
            scalar_arithmetic.scalar_mulf,
            TEST_LOW_MUL_F32_DESCRIPTOR,
            _F32,
            semantic_tag="float.mul.f32",
        ),
        _rule(scalar_conversion.scalar_constant, TEST_LOW_CONST_I32_DESCRIPTOR),
        _rule(index.index_constant, TEST_LOW_CONST_I32_DESCRIPTOR),
        _rule(vector.vector_extract, TEST_LOW_CONST_I32_DESCRIPTOR),
        _rule(vector.vector_shuffle, TEST_LOW_SHUFFLE_V4I32_DESCRIPTOR),
        _select_rule(
            vector.vector_select,
            TEST_LOW_SELECT_I32_DESCRIPTOR,
            condition_type=_V4I1,
            value_type=_V4I32,
            semantic_tag="integer.select.i32",
        ),
        _compare_rule(
            vector.vector_cmpi,
            TEST_LOW_CMP_EQ_I32_DESCRIPTOR,
            predicate="eq",
            operand_type=_V4I32,
            result_type=_V4I1,
            semantic_tag="integer.cmp.eq.i32",
        ),
        _binary_rule(
            vector.vector_addf,
            TEST_LOW_ADD_F32_DESCRIPTOR,
            _V4F32,
            semantic_tag="float.add.f32",
        ),
        _binary_rule(
            vector.vector_subf,
            TEST_LOW_SUB_F32_DESCRIPTOR,
            _V4F32,
            semantic_tag="float.sub.f32",
        ),
        _binary_rule(
            vector.vector_mulf,
            TEST_LOW_MUL_F32_DESCRIPTOR,
            _V4F32,
            semantic_tag="float.mul.f32",
        ),
        _binary_rule(
            vector.vector_addi,
            TEST_LOW_ADD_I32_DESCRIPTOR,
            _V4I32,
            semantic_tag="integer.add.i32",
        ),
        _binary_rule(vector.vector_subi, TEST_LOW_ADD_I32_DESCRIPTOR, _V4I32),
        _binary_rule(
            vector.vector_muli,
            TEST_LOW_MUL_I32_DESCRIPTOR,
            _V4I32,
            semantic_tag="integer.mul.i32",
        ),
        _rule(vector.vector_reduce, TEST_LOW_ADD_I32_DESCRIPTOR),
        _rule(vector.vector_reduce, TEST_LOW_ADD_F32_DESCRIPTOR),
        _ternary_rule(
            vector.vector_dot4i,
            TEST_LOW_DOT4I_S8S8_DESCRIPTOR,
            {
                "lhs": _V16I8,
                "rhs": _V16I8,
                "acc": _V4I32,
                "result": _V4I32,
            },
            semantic_tag="integer.dot4.s8s8",
            source_a="lhs",
            source_b="rhs",
            source_c="acc",
            descriptor_a="lhs",
            descriptor_b="rhs",
            descriptor_c="acc",
        ),
        _rule(index.index_madd, TEST_LOW_ADD_I32_DESCRIPTOR),
    ],
)
