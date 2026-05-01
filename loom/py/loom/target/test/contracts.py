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
from loom.target.contracts import ContractTable, DescriptorRule
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
        _rule(scalar_arithmetic.scalar_addi, TEST_LOW_ADD_I32_DESCRIPTOR),
        _rule(scalar_arithmetic.scalar_subi, TEST_LOW_TIED_ANY_DESCRIPTOR),
        _rule(scalar_arithmetic.scalar_muli, TEST_LOW_MUL_I32_DESCRIPTOR),
        _rule(scalar_arithmetic.scalar_addf, TEST_LOW_ADD_F32_DESCRIPTOR),
        _rule(scalar_arithmetic.scalar_subf, TEST_LOW_SUB_F32_DESCRIPTOR),
        _rule(scalar_arithmetic.scalar_mulf, TEST_LOW_MUL_F32_DESCRIPTOR),
        _rule(scalar_conversion.scalar_constant, TEST_LOW_CONST_I32_DESCRIPTOR),
        _rule(index.index_constant, TEST_LOW_CONST_I32_DESCRIPTOR),
        _rule(vector.vector_extract, TEST_LOW_CONST_I32_DESCRIPTOR),
        _rule(vector.vector_shuffle, TEST_LOW_SHUFFLE_V4I32_DESCRIPTOR),
        _rule(vector.vector_select, TEST_LOW_SELECT_I32_DESCRIPTOR),
        _rule(vector.vector_cmpi, TEST_LOW_CMP_EQ_I32_DESCRIPTOR),
        _rule(vector.vector_addf, TEST_LOW_ADD_F32_DESCRIPTOR),
        _rule(vector.vector_subf, TEST_LOW_SUB_F32_DESCRIPTOR),
        _rule(vector.vector_mulf, TEST_LOW_MUL_F32_DESCRIPTOR),
        _rule(vector.vector_addi, TEST_LOW_ADD_I32_DESCRIPTOR),
        _rule(vector.vector_subi, TEST_LOW_ADD_I32_DESCRIPTOR),
        _rule(vector.vector_muli, TEST_LOW_MUL_I32_DESCRIPTOR),
        _rule(vector.vector_reduce, TEST_LOW_ADD_I32_DESCRIPTOR),
        _rule(vector.vector_reduce, TEST_LOW_ADD_F32_DESCRIPTOR),
        _rule(vector.vector_dot4i, TEST_LOW_DOT4I_S8S8_DESCRIPTOR),
        _rule(index.index_madd, TEST_LOW_ADD_I32_DESCRIPTOR),
    ],
)
