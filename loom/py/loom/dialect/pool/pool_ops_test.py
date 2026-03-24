# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the pool dialect ops."""

from loom.dialect.pool import (
    ALL_POOL_OPS,
    pool_buffer,
    pool_load,
    pool_ops,
    pool_pin,
    pool_store,
    pool_unpin,
)
from loom.dsl import (
    ANY,
    INDEX,
    INTEGER,
    POOL,
    TILE,
)


class TestAllOpsRegistered:
    def test_count(self) -> None:
        assert len(ALL_POOL_OPS) == 5

    def test_unique_names(self) -> None:
        names = [op.name for op in ALL_POOL_OPS]
        assert len(set(names)) == len(names)

    def test_all_in_pool_namespace(self) -> None:
        for op in ALL_POOL_OPS:
            assert op.namespace == "pool", f"{op.name} not in pool namespace"

    def test_all_have_group(self) -> None:
        for op in ALL_POOL_OPS:
            assert op.group is pool_ops

    def test_all_have_docs(self) -> None:
        for op in ALL_POOL_OPS:
            assert op.doc

    def test_all_have_format(self) -> None:
        for op in ALL_POOL_OPS:
            assert op.format

    def test_all_have_examples(self) -> None:
        for op in ALL_POOL_OPS:
            assert op.examples


class TestPoolLoad:
    def test_name(self) -> None:
        assert pool_load.name == "pool.load"

    def test_operand_count(self) -> None:
        assert len(pool_load.operands) == 3

    def test_operand_types(self) -> None:
        assert pool_load.operands[0].type_constraint == POOL
        assert pool_load.operands[1].type_constraint == INTEGER
        assert pool_load.operands[2].type_constraint == INDEX

    def test_result_is_tile(self) -> None:
        assert len(pool_load.results) == 1
        assert pool_load.results[0].type_constraint == TILE

    def test_reads_pool(self) -> None:
        assert not pool_load.is_pure
        assert len(pool_load.effects) == 1
        assert pool_load.effects[0].operand == "pool"


class TestPoolStore:
    def test_name(self) -> None:
        assert pool_store.name == "pool.store"

    def test_operand_count(self) -> None:
        assert len(pool_store.operands) == 5

    def test_operand_types(self) -> None:
        assert pool_store.operands[0].type_constraint == POOL
        assert pool_store.operands[1].type_constraint == INTEGER
        assert pool_store.operands[2].type_constraint == INDEX
        assert pool_store.operands[3].type_constraint == INTEGER
        assert pool_store.operands[4].type_constraint == TILE

    def test_no_results(self) -> None:
        assert len(pool_store.results) == 0

    def test_not_pure(self) -> None:
        assert not pool_store.is_pure


class TestPoolPin:
    def test_name(self) -> None:
        assert pool_pin.name == "pool.pin"

    def test_operand_count(self) -> None:
        assert len(pool_pin.operands) == 2

    def test_operand_types(self) -> None:
        assert pool_pin.operands[0].type_constraint == POOL
        assert pool_pin.operands[1].type_constraint == INTEGER

    def test_no_results(self) -> None:
        assert len(pool_pin.results) == 0

    def test_not_pure(self) -> None:
        assert not pool_pin.is_pure


class TestPoolUnpin:
    def test_name(self) -> None:
        assert pool_unpin.name == "pool.unpin"

    def test_operand_count(self) -> None:
        assert len(pool_unpin.operands) == 2

    def test_operand_types(self) -> None:
        assert pool_unpin.operands[0].type_constraint == POOL
        assert pool_unpin.operands[1].type_constraint == INTEGER

    def test_no_results(self) -> None:
        assert len(pool_unpin.results) == 0

    def test_not_pure(self) -> None:
        assert not pool_unpin.is_pure


class TestPoolBuffer:
    def test_name(self) -> None:
        assert pool_buffer.name == "pool.buffer"

    def test_operand_count(self) -> None:
        assert len(pool_buffer.operands) == 1

    def test_operand_type(self) -> None:
        assert pool_buffer.operands[0].type_constraint == POOL

    def test_result_is_any(self) -> None:
        assert len(pool_buffer.results) == 1
        assert pool_buffer.results[0].type_constraint == ANY

    def test_is_pure(self) -> None:
        assert pool_buffer.is_pure
