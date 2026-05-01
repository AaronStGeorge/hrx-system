# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import replace

import pytest

from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.vector import defs as vector
from loom.target.contracts import (
    AttrProject,
    ContractTable,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    ValueAliasRule,
    ValueElideRule,
    ValueRef,
    Vector,
    descriptor_by_semantic_tag,
)
from loom.target.test.descriptors import (
    TEST_LOW_ADD_F32_DESCRIPTOR,
    TEST_LOW_ADD_I32_DESCRIPTOR,
    TEST_LOW_ADD_V4I32_DESCRIPTOR,
    TEST_LOW_CMP_EQ_V4I32_DESCRIPTOR,
    TEST_LOW_CORE_DESCRIPTOR_SET,
    TEST_LOW_EXTRACT_LANE_I32_DESCRIPTOR,
    TEST_LOW_SHUFFLE_BYTES_DESCRIPTOR,
)


def test_alias_rule_validates_source_and_result() -> None:
    table = ContractTable(
        name="value.alias",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            ValueAliasRule(
                source_op=vector.vector_fragment,
                source=ValueRef.operand("data"),
                result=ValueRef.result("result"),
            )
        ],
    )

    assert table.cases[0].source_op == vector.vector_fragment


def test_elide_rule_validates_source_results() -> None:
    table = ContractTable(
        name="value.elide",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            ValueElideRule(
                source_op=vector.vector_extract,
                values=(ValueRef.result("result"),),
            )
        ],
    )

    assert table.cases[0].source_op == vector.vector_extract


def test_elide_rule_rejects_operands() -> None:
    with pytest.raises(
        ValueError,
        match=r"vector.extract: elided values must be results",
    ):
        ContractTable(
            name="value.elide",
            descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
            cases=[
                ValueElideRule(
                    source_op=vector.vector_extract,
                    values=(ValueRef.operand("source"),),
                )
            ],
        )


def test_binary_descriptor_rule_validates_regular_operand_shape() -> None:
    descriptor = TEST_LOW_ADD_V4I32_DESCRIPTOR

    table = ContractTable(
        name="test-low.binary",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=vector.vector_addi,
                descriptor=descriptor,
                guards=[
                    Guard.value_type("lhs", Vector("i32", lanes=4)),
                    Guard.value_type("rhs", Vector("i32", lanes=4)),
                    Guard.value_type("result", Vector("i32", lanes=4)),
                ],
                emit=[
                    EmitDescriptorOp(
                        descriptor=descriptor,
                        operands={
                            "lhs": ValueRef.operand("lhs"),
                            "rhs": ValueRef.operand("rhs"),
                        },
                        results={"dst": ValueRef.result("result")},
                    )
                ],
            )
        ],
    )

    assert table.cases[0].source_op == vector.vector_addi


def test_descriptor_rule_accepts_named_test_low_descriptor_object() -> None:
    table = ContractTable(
        name="test-low.binary",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=scalar_arithmetic.scalar_addi,
                descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
                guards=[
                    Guard.value_type("lhs", Scalar("i32")),
                    Guard.value_type("rhs", Scalar("i32")),
                    Guard.value_type("result", Scalar("i32")),
                ],
                emit=[
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
                        operands={
                            "lhs": ValueRef.operand("lhs"),
                            "rhs": ValueRef.operand("rhs"),
                        },
                        results={"dst": ValueRef.result("result")},
                    )
                ],
            )
        ],
    )

    assert table.cases[0].source_op == scalar_arithmetic.scalar_addi


def test_compare_descriptor_rule_validates_enum_guard() -> None:
    descriptor = TEST_LOW_CMP_EQ_V4I32_DESCRIPTOR

    table = ContractTable(
        name="test-low.compare",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=vector.vector_cmpi,
                descriptor=descriptor,
                guards=[
                    Guard.enum_attr_equals("predicate", "eq"),
                    Guard.value_type("lhs", Vector("i32", lanes=4)),
                    Guard.value_type("rhs", Vector("i32", lanes=4)),
                    Guard.value_type("result", Vector("i1", lanes=4)),
                ],
                emit=[
                    EmitDescriptorOp(
                        descriptor=descriptor,
                        operands={
                            "lhs": ValueRef.operand("lhs"),
                            "rhs": ValueRef.operand("rhs"),
                        },
                        results={"dst": ValueRef.result("result")},
                    )
                ],
            )
        ],
    )

    case = table.cases[0]
    assert isinstance(case, DescriptorRule)
    assert case.descriptor == descriptor


def test_descriptor_rule_validates_source_and_descriptor_fields() -> None:
    descriptor = TEST_LOW_EXTRACT_LANE_I32_DESCRIPTOR

    table = ContractTable(
        name="test-low.extract",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=vector.vector_extract,
                descriptor=descriptor,
                guards=[
                    Guard.value_type("source", Vector("i32", lanes=4)),
                    Guard.value_type("result", Scalar("i32")),
                    Guard.operand_segment_count("indices", 0),
                    Guard.i64_array_count("static_indices", 1),
                    Guard.i64_array_element_range("static_indices", 0, 0, 3),
                ],
                emit=[
                    EmitDescriptorOp(
                        descriptor=descriptor,
                        operands={"source": ValueRef.operand("source")},
                        results={"dst": ValueRef.result("result")},
                        immediates={
                            "lane": AttrProject.i64_array_element(
                                "static_indices", element=0
                            )
                        },
                    )
                ],
            )
        ],
    )

    case = table.cases[0]
    assert isinstance(case, DescriptorRule)
    assert case.descriptor == descriptor


def test_descriptor_rule_rejects_unknown_source_value_field() -> None:
    descriptor = TEST_LOW_EXTRACT_LANE_I32_DESCRIPTOR

    with pytest.raises(
        ValueError,
        match=(
            r"vector.extract: guard value_type field 'missing' is not an "
            r"operand or result"
        ),
    ):
        ContractTable(
            name="bad.source.field",
            descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
            cases=[
                DescriptorRule(
                    source_op=vector.vector_extract,
                    descriptor=descriptor,
                    guards=[Guard.value_type("missing", Scalar("i32"))],
                )
            ],
        )


def test_descriptor_rule_rejects_wrong_attr_kind() -> None:
    descriptor = TEST_LOW_ADD_F32_DESCRIPTOR

    with pytest.raises(
        ValueError,
        match=(
            r"vector.reduce: guard i64_array_count field 'kind' "
            r"must be an i64_array attr"
        ),
    ):
        ContractTable(
            name="bad.attr.kind",
            descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
            cases=[
                DescriptorRule(
                    source_op=vector.vector_reduce,
                    descriptor=descriptor,
                    guards=[Guard.i64_array_count("kind", 1)],
                )
            ],
        )


def test_descriptor_rule_rejects_unknown_enum_case() -> None:
    descriptor = TEST_LOW_CMP_EQ_V4I32_DESCRIPTOR

    with pytest.raises(
        ValueError,
        match=(
            r"vector.cmpi: guard enum_attr_equals field 'predicate' "
            r"has no enum case 'ordered'"
        ),
    ):
        ContractTable(
            name="bad.enum.case",
            descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
            cases=[
                DescriptorRule(
                    source_op=vector.vector_cmpi,
                    descriptor=descriptor,
                    guards=[Guard.enum_attr_equals("predicate", "ordered")],
                )
            ],
        )


def test_descriptor_rule_rejects_unknown_descriptor_operand_field() -> None:
    descriptor = TEST_LOW_EXTRACT_LANE_I32_DESCRIPTOR

    with pytest.raises(
        ValueError,
        match=(
            r"descriptor 'test.extract_lane.i32': descriptor operand "
            r"binding field 'missing' is not a descriptor operand"
        ),
    ):
        ContractTable(
            name="bad.descriptor.field",
            descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
            cases=[
                DescriptorRule(
                    source_op=vector.vector_extract,
                    descriptor=descriptor,
                    emit=[
                        EmitDescriptorOp(
                            descriptor=descriptor,
                            operands={"missing": ValueRef.operand("source")},
                            results={"dst": ValueRef.result("result")},
                            immediates={
                                "lane": AttrProject.i64_array_element(
                                    "static_indices", element=0
                                )
                            },
                        )
                    ],
                )
            ],
        )


def test_descriptor_rule_rejects_unbound_required_immediate() -> None:
    descriptor = TEST_LOW_EXTRACT_LANE_I32_DESCRIPTOR

    with pytest.raises(
        ValueError,
        match=(
            r"vector.extract: descriptor 'test.extract_lane.i32' "
            r"immediate 'lane' is not bound"
        ),
    ):
        ContractTable(
            name="bad.immediate",
            descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
            cases=[
                DescriptorRule(
                    source_op=vector.vector_extract,
                    descriptor=descriptor,
                    emit=[
                        EmitDescriptorOp(
                            descriptor=descriptor,
                            operands={"source": ValueRef.operand("source")},
                            results={"dst": ValueRef.result("result")},
                        )
                    ],
                )
            ],
        )


def test_shuffle_byte_expansion_validates_all_target_immediates() -> None:
    descriptor = TEST_LOW_SHUFFLE_BYTES_DESCRIPTOR

    table = ContractTable(
        name="test-low.shuffle",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=vector.vector_shuffle,
                descriptor=descriptor,
                guards=[
                    Guard.value_type("source", Vector("i32", lanes=4)),
                    Guard.value_type("result", Vector("i32", lanes=4)),
                    Guard.i64_array_count("source_lanes", 4),
                    Guard.i64_array_elements_range("source_lanes", 0, 3),
                ],
                emit=[
                    EmitDescriptorOp(
                        descriptor=descriptor,
                        operands={
                            "lhs": ValueRef.operand("source"),
                            "rhs": ValueRef.operand("source"),
                        },
                        results={"dst": ValueRef.result("result")},
                        immediates=[
                            AttrProject.expand_lane_i64_array_to_byte_lanes(
                                source_attr="source_lanes",
                                source_lane_count=4,
                                bytes_per_lane=4,
                                target_names=[f"lane{i}" for i in range(16)],
                            )
                        ],
                    )
                ],
            )
        ],
    )

    assert table.cases[0].source_op == vector.vector_shuffle


def test_reduction_rule_validates_literal_immediates_and_temporaries() -> None:
    extract = TEST_LOW_EXTRACT_LANE_I32_DESCRIPTOR
    add = TEST_LOW_ADD_I32_DESCRIPTOR

    table = ContractTable(
        name="test-low.reduce",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=vector.vector_reduce,
                descriptor=extract,
                guards=[
                    Guard.enum_attr_equals("kind", "addi"),
                    Guard.value_type("input", Vector("i32", lanes=4)),
                    Guard.value_type("init", Scalar("i32")),
                    Guard.value_type("result", Scalar("i32")),
                ],
                emit=[
                    EmitDescriptorOp(
                        descriptor=extract,
                        operands={"source": ValueRef.operand("input")},
                        results={"dst": ValueRef.temporary("lane0")},
                        immediates={"lane": 0},
                    ),
                    EmitDescriptorOp(
                        descriptor=extract,
                        operands={"source": ValueRef.operand("input")},
                        results={"dst": ValueRef.temporary("lane1")},
                        immediates={"lane": 1},
                    ),
                    EmitDescriptorOp(
                        descriptor=extract,
                        operands={"source": ValueRef.operand("input")},
                        results={"dst": ValueRef.temporary("lane2")},
                        immediates={"lane": 2},
                    ),
                    EmitDescriptorOp(
                        descriptor=extract,
                        operands={"source": ValueRef.operand("input")},
                        results={"dst": ValueRef.temporary("lane3")},
                        immediates={"lane": 3},
                    ),
                    EmitDescriptorOp(
                        descriptor=add,
                        operands={
                            "lhs": ValueRef.operand("init"),
                            "rhs": ValueRef.temporary("lane0"),
                        },
                        results={"dst": ValueRef.temporary("acc0")},
                    ),
                    EmitDescriptorOp(
                        descriptor=add,
                        operands={
                            "lhs": ValueRef.temporary("acc0"),
                            "rhs": ValueRef.temporary("lane1"),
                        },
                        results={"dst": ValueRef.temporary("acc1")},
                    ),
                    EmitDescriptorOp(
                        descriptor=add,
                        operands={
                            "lhs": ValueRef.temporary("acc1"),
                            "rhs": ValueRef.temporary("lane2"),
                        },
                        results={"dst": ValueRef.temporary("acc2")},
                    ),
                    EmitDescriptorOp(
                        descriptor=add,
                        operands={
                            "lhs": ValueRef.temporary("acc2"),
                            "rhs": ValueRef.temporary("lane3"),
                        },
                        results={"dst": ValueRef.result("result")},
                    ),
                ],
            )
        ],
    )

    assert table.cases[0].source_op == vector.vector_reduce


def test_descriptor_rule_rejects_undefined_temporary() -> None:
    descriptor = TEST_LOW_ADD_I32_DESCRIPTOR

    with pytest.raises(
        ValueError,
        match=(
            r"vector.reduce: descriptor operand 'lhs' temporary 'missing' "
            r"is not defined"
        ),
    ):
        ContractTable(
            name="bad.temporary",
            descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
            cases=[
                DescriptorRule(
                    source_op=vector.vector_reduce,
                    descriptor=descriptor,
                    emit=[
                        EmitDescriptorOp(
                            descriptor=descriptor,
                            operands={
                                "lhs": ValueRef.temporary("missing"),
                                "rhs": ValueRef.operand("init"),
                            },
                            results={"dst": ValueRef.result("result")},
                        )
                    ],
                )
            ],
        )


def test_descriptor_rule_rejects_immediate_literal_out_of_range() -> None:
    descriptor = TEST_LOW_EXTRACT_LANE_I32_DESCRIPTOR

    with pytest.raises(
        ValueError,
        match=(
            r"vector.extract: descriptor 'test.extract_lane.i32' "
            r"immediate 'lane' literal 4 is out of range"
        ),
    ):
        ContractTable(
            name="bad.literal",
            descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
            cases=[
                DescriptorRule(
                    source_op=vector.vector_extract,
                    descriptor=descriptor,
                    emit=[
                        EmitDescriptorOp(
                            descriptor=descriptor,
                            operands={"source": ValueRef.operand("source")},
                            results={"dst": ValueRef.result("result")},
                            immediates={"lane": 4},
                        )
                    ],
                )
            ],
        )


def test_shuffle_byte_expansion_rejects_wrong_target_count() -> None:
    descriptor = TEST_LOW_SHUFFLE_BYTES_DESCRIPTOR

    with pytest.raises(
        ValueError,
        match=(
            r"vector.shuffle: immediate projection "
            r"expand_lane_i64_array_to_byte_lanes produces 16 immediates, got 15"
        ),
    ):
        ContractTable(
            name="bad.shuffle",
            descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
            cases=[
                DescriptorRule(
                    source_op=vector.vector_shuffle,
                    descriptor=descriptor,
                    emit=[
                        EmitDescriptorOp(
                            descriptor=descriptor,
                            operands={
                                "lhs": ValueRef.operand("source"),
                                "rhs": ValueRef.operand("source"),
                            },
                            results={"dst": ValueRef.result("result")},
                            immediates=[
                                AttrProject.expand_lane_i64_array_to_byte_lanes(
                                    source_attr="source_lanes",
                                    source_lane_count=4,
                                    bytes_per_lane=4,
                                    target_names=[f"lane{i}" for i in range(15)],
                                )
                            ],
                        )
                    ],
                )
            ],
        )


def test_descriptor_semantic_tag_lookup_rejects_ambiguous_tags() -> None:
    descriptor = TEST_LOW_ADD_V4I32_DESCRIPTOR
    duplicate = replace(descriptor, key="test.add.v4i32.duplicate")
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(*TEST_LOW_CORE_DESCRIPTOR_SET.descriptors, duplicate),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"descriptor set 'test.low.core' has 2 descriptors with "
            r"semantic tag 'vector.add.i32x4'"
        ),
    ):
        descriptor_by_semantic_tag(descriptor_set, "vector.add.i32x4")
