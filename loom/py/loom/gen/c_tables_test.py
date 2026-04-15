# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from loom.assembly import (
    COLON,
    Attr,
    AttrDict,
    Flags,
    OperandDict,
    Ref,
    ResultType,
    TemplateParam,
    TypesOf,
)
from loom.dsl import (
    INTEGER,
    AttrDef,
    Dialect,
    EnumCase,
    EnumDef,
    Op,
    Operand,
    Result,
    SameType,
    TypeConstraint,
)
from loom.gen.c_tables import (
    TYPE_CONSTRAINT_MAP,
    generate_builders_c,
    generate_ops_h,
    generate_tables_c,
)


def test_type_constraint_map_covers_every_constraint() -> None:
    assert set(TYPE_CONSTRAINT_MAP) == set(TypeConstraint)


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


def test_generate_builders_preserve_named_operands_for_non_binary_shapes() -> None:
    op = Op(
        "test.lookup",
        group=Dialect("test"),
        operands=[
            Operand("table", INTEGER),
            Operand("indices", INTEGER),
        ],
        results=[Result("result", INTEGER)],
        format=[Ref("table"), Ref("indices"), ResultType("result")],
    )

    ops_h = generate_ops_h("test", 0, [op])
    builders_c = generate_builders_c("test", [op])

    assert "loom_value_id_t table" in ops_h
    assert "loom_value_id_t indices" in ops_h
    assert "LOOM_DEFINE_BINARY_OP_BUILDER" not in builders_c
    assert "loom_op_operands(*out_op)[0] = table;" in builders_c
    assert "loom_op_operands(*out_op)[1] = indices;" in builders_c


def test_types_of_result_field_generates_result_type_list_format() -> None:
    op = Op(
        "test.results",
        group=Dialect("test"),
        results=[Result("results", INTEGER, variadic=True)],
        format=[COLON, TypesOf("results")],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_FORMAT_KIND_RESULT_TYPE_LIST" in tables_c
    assert "LOOM_FORMAT_KIND_OPERAND_TYPES" not in tables_c


def test_inline_attr_dict_uses_declared_attrs() -> None:
    ordering = EnumDef("Ordering", [EnumCase("relaxed", 0)])
    scope = EnumDef("Scope", [EnumCase("workgroup", 0)])
    op = Op(
        "test.atomic",
        group=Dialect("test"),
        attrs=[
            AttrDef("ordering", "enum", enum_def=ordering),
            AttrDef("scope", "enum", enum_def=scope),
        ],
        format=[AttrDict()],
    )

    ops_h = generate_ops_h("test", 0, [op])
    builders_c = generate_builders_c("test", [op])
    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_ATTR_DICT_FORMAT_INLINE_ATTRS" in tables_c
    assert "uint8_t ordering" in ops_h
    assert "uint8_t scope" in ops_h
    assert "loom_op_attrs(*out_op)[0] = loom_attr_enum(ordering);" in builders_c
    assert "loom_op_attrs(*out_op)[1] = loom_attr_enum(scope);" in builders_c


def test_flags_attrs_do_not_shift_regular_attr_indices() -> None:
    flags = EnumDef("Flags", [EnumCase("hot", 1)])
    op = Op(
        "test.flagged",
        group=Dialect("test"),
        attrs=[
            AttrDef("flags", "flags", optional=True, enum_def=flags),
            AttrDef("name", "string"),
            AttrDef("constraints", "string"),
        ],
        format=[Flags("flags"), Attr("name"), Attr("constraints")],
    )

    builders_c = generate_builders_c("test", [op])
    tables_c = generate_tables_c("test", 0, [op])

    assert "loom_op_attrs(*out_op)[0] = loom_attr_string(name);" in builders_c
    assert "loom_op_attrs(*out_op)[1] = loom_attr_string(constraints);" in builders_c
    assert "loom_op_attrs(*out_op)[2]" not in builders_c
    assert "{LOOM_FORMAT_KIND_ATTR_VALUE, 0, 0}" in tables_c
    assert "{LOOM_FORMAT_KIND_ATTR_VALUE, 1, 0}" in tables_c
    assert "{LOOM_FORMAT_KIND_ATTR_VALUE, 2, 0}" not in tables_c


def test_enum_keywords_with_dots_generate_valid_c_constants() -> None:
    intrinsic_kind = EnumDef("Kind", [EnumCase("llvm.x86.rdtsc", 0)])
    op = Op(
        "test.intrinsic",
        group=Dialect("test"),
        attrs=[AttrDef("kind", "enum", enum_def=intrinsic_kind)],
        format=[TemplateParam("kind")],
    )

    ops_h = generate_ops_h("test", 0, [op])
    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_TEST_INTRINSIC_KIND_LLVM_X86_RDTSC = 0," in ops_h
    assert "LOOM_TEST_INTRINSIC_KIND_LLVM.X86.RDTSC" not in ops_h
    assert '"llvm.x86.rdtsc"' in tables_c


def test_operand_dict_generates_format_and_builder_support() -> None:
    op = Op(
        "test.operand_dict",
        group=Dialect("test"),
        operands=[
            Operand("input", INTEGER),
            Operand("params", INTEGER, variadic=True),
        ],
        results=[Result("result", INTEGER)],
        attrs=[AttrDef("param_names", "dict", optional=True)],
        constraints=[SameType("input", "result")],
        format=[
            Ref("input"),
            OperandDict("params", "param_names"),
            ResultType("result"),
        ],
    )

    ops_h = generate_ops_h("test", 0, [op])
    builders_c = generate_builders_c("test", [op])
    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_FORMAT_KIND_OPERAND_DICT" in tables_c
    assert "const loom_named_value_t* params" in ops_h
    assert "iree_host_size_t params_count" in ops_h
    assert "loom_make_named_value_slice(params, params_count)" in builders_c
    assert "&loom_op_attrs(*out_op)[0]" in builders_c
