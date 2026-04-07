# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from loom.assembly import Attr, AttrDict
from loom.dsl import INTEGER, AttrDef, Dialect, Op, Operand, SameType
from loom.gen.c_tables import generate_builders_c, generate_ops_h, generate_tables_c


def test_generate_tables_rejects_constraint_field_index_above_6_bit_max() -> None:
    op = Op(
        "test.wide",
        group=Dialect("test"),
        operands=[Operand(f"input_{i}", INTEGER) for i in range(65)],
        constraints=[SameType("input_0", "input_64")],
    )

    with pytest.raises(
        ValueError,
        match=(
            r"Op 'test\.wide' constraint SameType: field 'input_64' "
            r"index 64 exceeds LOOM_FIELD_REF 6-bit max 63"
        ),
    ):
        generate_tables_c("test", 0, [op])


def test_generate_builders_use_explicit_flags_for_optional_scalar_attrs() -> None:
    op = Op(
        "test.optional",
        group=Dialect("test"),
        attrs=[AttrDef("priority", "i64", optional=True)],
        format=[Attr("priority")],
    )

    ops_h = generate_ops_h("test", 0, [op])
    builders_c = generate_builders_c("test", [op])

    assert "enum loom_test_optional_build_flag_bits_e {" in ops_h
    assert "LOOM_TEST_OPTIONAL_BUILD_FLAG_HAS_PRIORITY = 1u << 0," in ops_h
    assert "typedef uint32_t loom_test_optional_build_flags_t;" in ops_h
    assert "loom_test_optional_build_flags_t build_flags" in ops_h
    assert "loom_test_optional_build_flags_t build_flags" in builders_c
    assert ("iree_any_bit_set(build_flags, LOOM_TEST_OPTIONAL_BUILD_FLAG_HAS_PRIORITY)") in builders_c
    assert "priority != 0" not in builders_c


def test_generate_builders_keep_count_guard_for_optional_aggregate_attrs() -> None:
    op = Op(
        "test.attrs",
        group=Dialect("test"),
        attrs=[AttrDef("dict", "dict", optional=True)],
        format=[AttrDict("dict")],
    )

    ops_h = generate_ops_h("test", 0, [op])
    builders_c = generate_builders_c("test", [op])

    assert "LOOM_TEST_ATTRS_BUILD_FLAG_HAS_DICT" not in ops_h
    assert "loom_test_attrs_build_flags_t build_flags" not in ops_h
    assert "loom_test_attrs_build_flags_t build_flags" not in builders_c
    assert "iree_any_bit_set(build_flags" not in builders_c
    assert "dict.count > 0" in builders_c
