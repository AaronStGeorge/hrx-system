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
    AttrTable,
    BlockRef,
    Flags,
    OperandDict,
    Ref,
    Region,
    ResultType,
    ResultTypeList,
    TemplateParam,
    TypesOf,
)
from loom.dsl import (
    ANY,
    ATTR_TYPE_I64_ARRAY,
    INTEGER,
    AttrDef,
    AttrMatchesElementType,
    BitRangeWithinElementWidth,
    CallLikeInterface,
    Dialect,
    ElementWidthAtLeastAttr,
    ElementWidthGreaterThan,
    EnumCase,
    EnumDef,
    Op,
    Operand,
    PackedPayloadBitCountMatchesStorage,
    PositiveBitWidthAttr,
    RegionDef,
    Result,
    SameType,
    Successor,
    TotalBitCountEqual,
    TypeConstraint,
    UnpackedPayloadBitCountMatchesStorage,
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


def test_generate_tables_emits_element_width_constraint() -> None:
    op = Op(
        "test.ext",
        group=Dialect("test"),
        operands=[Operand("input", INTEGER)],
        results=[Result("result", INTEGER)],
        constraints=[ElementWidthGreaterThan("result", "input")],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_RELATION_ELEMENT_WIDTH_ORDER" in tables_c
    assert "LOOM_PROPERTY_ELEMENT_WIDTH_GREATER_THAN" in tables_c
    assert "LOOM_FIELD_REF(1, 0), LOOM_FIELD_REF(0, 0)" in tables_c


def test_generate_tables_emits_bit_width_attr_constraints() -> None:
    op = Op(
        "test.bitfield",
        group=Dialect("test"),
        operands=[Operand("input", INTEGER)],
        results=[Result("result", INTEGER)],
        attrs=[AttrDef("offset", "i64"), AttrDef("width", "i64")],
        constraints=[
            PositiveBitWidthAttr("width"),
            ElementWidthAtLeastAttr("result", "width"),
            BitRangeWithinElementWidth("input", "offset", "width"),
        ],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_RELATION_ATTR_I64_PREDICATE" in tables_c
    assert "LOOM_PROPERTY_BIT_WIDTH_POSITIVE" in tables_c
    assert "LOOM_RELATION_ELEMENT_WIDTH_AT_LEAST_ATTR" in tables_c
    assert "LOOM_RELATION_BIT_RANGE_WITHIN_ELEMENT_WIDTH" in tables_c
    assert "LOOM_FIELD_REF(2, 0), LOOM_FIELD_REF(2, 1)" in tables_c


def test_generate_tables_emits_attr_matches_element_type_constraint() -> None:
    op = Op(
        "test.constant",
        group=Dialect("test"),
        results=[Result("result", INTEGER)],
        attrs=[AttrDef("value", "any")],
        constraints=[AttrMatchesElementType("value", "result")],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_RELATION_ATTR_MATCHES_ELEMENT_TYPE" in tables_c
    assert "LOOM_PROPERTY_ELEMENT_TYPE" in tables_c
    assert "LOOM_FIELD_REF(2, 0), LOOM_FIELD_REF(1, 0)" in tables_c
    assert "LOOM_ERROR_REF(LOOM_ERROR_DOMAIN_TYPE, 5)" in tables_c
    assert "loom_err_" not in tables_c


def test_generate_tables_emits_bit_count_constraints() -> None:
    op = Op(
        "test.bitstream",
        group=Dialect("test"),
        operands=[Operand("source", INTEGER)],
        results=[Result("result", INTEGER)],
        attrs=[AttrDef("width", "i64")],
        constraints=[
            TotalBitCountEqual("source", "result"),
            PackedPayloadBitCountMatchesStorage("source", "width", "result", "result"),
            UnpackedPayloadBitCountMatchesStorage("result", "width", "source", "result"),
        ],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_RELATION_TOTAL_BIT_COUNT_EQUAL" in tables_c
    assert "LOOM_PROPERTY_TOTAL_BIT_COUNT" in tables_c
    assert "LOOM_RELATION_PAYLOAD_BIT_COUNT_MATCHES_STORAGE" in tables_c
    assert "LOOM_PROPERTY_PACKED_PAYLOAD_BIT_COUNT_MATCHES_STORAGE" in tables_c
    assert "LOOM_PROPERTY_UNPACKED_PAYLOAD_BIT_COUNT_MATCHES_STORAGE" in tables_c
    assert "LOOM_FIELD_REF(0, 0), LOOM_FIELD_REF(1, 0)" in tables_c
    assert ("LOOM_FIELD_REF(0, 0), LOOM_FIELD_REF(2, 0), LOOM_FIELD_REF(1, 0), LOOM_FIELD_REF(1, 0)") in tables_c
    assert ("LOOM_FIELD_REF(1, 0), LOOM_FIELD_REF(2, 0), LOOM_FIELD_REF(0, 0), LOOM_FIELD_REF(1, 0)") in tables_c


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


def test_generate_builders_emit_successor_fields() -> None:
    op = Op(
        "test.br",
        group=Dialect("test"),
        successors=[Successor("dest")],
        format=[BlockRef("dest")],
    )

    ops_h = generate_ops_h("test", 0, [op])
    builders_c = generate_builders_c("test", [op])
    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_DEFINE_SUCCESSOR(loom_test_br_dest, 0)" in ops_h
    assert "loom_block_t* dest" in ops_h
    assert "loom_builder_allocate_op_with_successors" in builders_c
    assert "loom_op_successors(*out_op)[0] = dest;" in builders_c
    assert ".fixed_successor_count" not in tables_c
    assert "{LOOM_FORMAT_KIND_SUCCESSOR_REF, 0, 0}," in tables_c


def test_generate_tables_preserves_operand_and_result_descriptor_names() -> None:
    op = Op(
        "test.reduce",
        group=Dialect("test"),
        operands=[
            Operand("input", INTEGER),
            Operand("partials", INTEGER, variadic=True),
        ],
        results=[Result("results", INTEGER, variadic=True)],
        format=[Ref("input"), Ref("partials"), ResultTypeList("results")],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert 'static const uint8_t loom_test_reduce_input_operand_bname[] = "\\x05" "input";' in tables_c
    assert 'static const uint8_t loom_test_reduce_partials_operand_bname[] = "\\x08" "partials";' in tables_c
    assert 'static const uint8_t loom_test_reduce_results_result_bname[] = "\\x07" "results";' in tables_c
    assert "{loom_test_reduce_input_operand_bname, LOOM_TYPE_CONSTRAINT_INTEGER, 0}" in tables_c
    assert ("{loom_test_reduce_partials_operand_bname, LOOM_TYPE_CONSTRAINT_INTEGER, LOOM_OPERAND_VARIADIC}") in tables_c
    assert ("{loom_test_reduce_results_result_bname, LOOM_TYPE_CONSTRAINT_INTEGER, LOOM_RESULT_VARIADIC}") in tables_c


def test_generate_tables_emits_call_like_interface() -> None:
    op = Op(
        "test.call",
        group=Dialect("test"),
        attrs=[AttrDef("callee", "symbol")],
        operands=[Operand("operands", ANY, variadic=True)],
        results=[Result("results", ANY, variadic=True)],
        interfaces=[
            CallLikeInterface(
                callee="callee",
                operands="operands",
                results="results",
            ),
        ],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert "static const loom_call_like_vtable_t loom_test_call_call_like" in tables_c
    assert ".callee_attr_index = 0," in tables_c
    assert ".purity_attr_index = 255," in tables_c
    assert ".operand_offset = 0," in tables_c
    assert ".result_offset = 0," in tables_c
    assert ".kind = LOOM_CALL_LIKE_KIND_SEMANTIC," in tables_c


def test_generate_tables_rejects_call_like_non_variadic_operand() -> None:
    op = Op(
        "test.call",
        group=Dialect("test"),
        attrs=[AttrDef("callee", "symbol")],
        operands=[Operand("operand", ANY)],
        results=[Result("results", ANY, variadic=True)],
        interfaces=[
            CallLikeInterface(
                callee="callee",
                operands="operand",
                results="results",
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            r"CallLikeInterface on 'test\.call': operand 'operand' "
            r"must be variadic"
        ),
    ):
        generate_tables_c("test", 0, [op])


def test_generate_tables_rejects_call_like_non_variadic_result() -> None:
    op = Op(
        "test.call",
        group=Dialect("test"),
        attrs=[AttrDef("callee", "symbol")],
        operands=[Operand("operands", ANY, variadic=True)],
        results=[Result("result", ANY)],
        interfaces=[
            CallLikeInterface(
                callee="callee",
                operands="operands",
                results="result",
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            r"CallLikeInterface on 'test\.call': result 'result' "
            r"must be variadic"
        ),
    ):
        generate_tables_c("test", 0, [op])


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


def test_region_syntax_generates_format_selector() -> None:
    op = Op(
        "test.region_syntax",
        group=Dialect("test"),
        regions=[RegionDef("body")],
        format=[Region("body", syntax="test.do")],
    )

    tables_c = generate_tables_c("test", 0, [op])

    assert "{LOOM_FORMAT_KIND_REGION, 0, LOOM_REGION_SYNTAX_TEST_DO}," in tables_c


def test_unknown_region_syntax_is_rejected() -> None:
    op = Op(
        "test.region_syntax",
        group=Dialect("test"),
        regions=[RegionDef("body")],
        format=[Region("body", syntax="missing.syntax")],
    )

    with pytest.raises(
        ValueError,
        match=(
            r"Op 'test\.region_syntax': unknown region syntax "
            r"'missing\.syntax'"
        ),
    ):
        generate_tables_c("test", 0, [op])


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


def test_attr_table_generates_format_and_builder_support() -> None:
    op = Op(
        "test.attr_table",
        group=Dialect("test"),
        operands=[
            Operand("selector", INTEGER),
            Operand("values", ANY, variadic=True),
        ],
        results=[Result("results", ANY, variadic=True)],
        attrs=[AttrDef("case_keys", ATTR_TYPE_I64_ARRAY)],
        format=[
            Ref("selector"),
            AttrTable("case_keys", "values"),
            COLON,
            ResultTypeList("results", parens=False),
        ],
    )

    ops_h = generate_ops_h("test", 0, [op])
    builders_c = generate_builders_c("test", [op])
    tables_c = generate_tables_c("test", 0, [op])

    assert "LOOM_FORMAT_KIND_ATTR_TABLE" in tables_c
    assert "const int64_t* case_keys" in ops_h
    assert "iree_host_size_t case_keys_count" in ops_h
    assert "const loom_value_id_t* values" in ops_h
    assert "iree_host_size_t values_count" in ops_h
    assert "loom_attr_i64_array(" in builders_c
