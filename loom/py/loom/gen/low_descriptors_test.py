# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
from dataclasses import replace
from typing import cast

import pytest

from loom.gen.low_descriptors import (
    DescriptorAllowlist,
    generate_descriptor_set,
    generate_descriptor_set_shared_source,
)
from loom.target.low_descriptors import (
    LOW_DESCRIPTOR_ENCODING_ID_NONE,
    AsmForm,
    AsmImmediate,
    Constraint,
    ConstraintKind,
    DescriptorCategory,
    DescriptorFlag,
    EncodingFieldValue,
    EnumDomain,
    EnumValue,
    Hazard,
    HazardKind,
    Immediate,
    ImmediateEncodingSlice,
    ImmediateFlag,
    ImmediateKind,
    OperandFlag,
    OperandForm,
    OperandFormMatch,
    OperandFormMatchKind,
    OperandRole,
)
from loom.target.test.descriptors import (
    TEST_LOW_ADD_I32_DESCRIPTOR,
    TEST_LOW_COND_BR_I32_DESCRIPTOR,
    TEST_LOW_CONST_I32_DESCRIPTOR,
    TEST_LOW_CORE_DESCRIPTOR_SET,
    TEST_LOW_MUL_I32_DESCRIPTOR,
)


def test_descriptor_category_validates_stable_key_spelling() -> None:
    category = DescriptorCategory("memory.atomic", doc="Atomic memory.")

    assert category.key == "memory.atomic"
    assert category.doc == "Atomic memory."

    with pytest.raises(
        ValueError,
        match=(
            r"descriptor category key 'Memory/Atomic' must contain only "
            r"lowercase letters, digits, '.', '_', or '-'"
        ),
    ):
        DescriptorCategory("Memory/Atomic")


def test_descriptor_set_validates_category_membership() -> None:
    category = DescriptorCategory("control")
    descriptor = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET.descriptors[0],
        category=category,
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        categories=(category,),
        descriptors=(descriptor,),
    )

    assert descriptor_set.categories == (category,)
    assert descriptor_set.descriptors[0].category == category


def test_descriptor_set_rejects_unknown_default_category() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"DescriptorSet 'test.low.core': default_category 'memory' "
            r"is not declared in categories"
        ),
    ):
        replace(
            TEST_LOW_CORE_DESCRIPTOR_SET,
            categories=(DescriptorCategory("control"),),
            default_category=DescriptorCategory("memory"),
        )


def test_descriptor_set_rejects_unknown_descriptor_category() -> None:
    descriptor = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET.descriptors[0],
        category=DescriptorCategory("memory"),
    )

    with pytest.raises(
        ValueError,
        match=(
            rf"DescriptorSet 'test.low.core': descriptor '{descriptor.key}' "
            r"category 'memory' is not declared in categories"
        ),
    ):
        replace(
            TEST_LOW_CORE_DESCRIPTOR_SET,
            categories=(DescriptorCategory("control"),),
            descriptors=(descriptor,),
        )


def test_allowlist_closes_over_referenced_descriptor_tables() -> None:
    generated = generate_descriptor_set(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        DescriptorAllowlist(keys=("test.add.i32",)),
    )
    manifest = json.loads(generated.manifest_json)

    assert [descriptor["key"] for descriptor in manifest["descriptors"]] == ["test.add.i32"]
    assert manifest["table_counts"]["descriptors"] == 1
    assert manifest["table_counts"]["descriptor_refs"] == 1
    assert manifest["table_counts"]["asm_forms"] == 1
    assert manifest["table_counts"]["reg_classes"] == len(TEST_LOW_CORE_DESCRIPTOR_SET.reg_classes)
    assert manifest["table_counts"]["reg_class_alts"] == 1
    assert manifest["table_counts"]["schedule_classes"] == 1
    assert manifest["table_counts"]["resources"] == 1
    assert "test.call.i32" not in generated.source


def test_allowlist_closes_over_operand_form_replacements() -> None:
    base_descriptor = TEST_LOW_CORE_DESCRIPTOR_SET.descriptors[1]
    replacement_descriptor = replace(
        base_descriptor,
        key="test.add.i32.rhs_zero",
        mnemonic="test.add.i32.rhs_zero",
        operands=base_descriptor.operands[:2],
        asm_forms=(AsmForm(results=("dst",), operands=("lhs",)),),
    )
    source_descriptor = replace(
        base_descriptor,
        operand_forms=(
            OperandForm(
                replacement_descriptor=replacement_descriptor.key,
                matches=(
                    OperandFormMatch(
                        source_operand="rhs",
                        match_kind=OperandFormMatchKind.ALL_EQUAL_I64,
                        match_i64=0,
                    ),
                ),
            ),
        ),
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(source_descriptor, replacement_descriptor),
    )

    generated = generate_descriptor_set(
        descriptor_set,
        DescriptorAllowlist(keys=(source_descriptor.key,)),
    )
    manifest = json.loads(generated.manifest_json)

    assert [descriptor["key"] for descriptor in manifest["descriptors"]] == [
        source_descriptor.key,
        replacement_descriptor.key,
    ]
    assert manifest["table_counts"]["operand_forms"] == 1
    assert manifest["table_counts"]["operand_form_matches"] == 1
    assert manifest["table_counts"]["operand_form_operand_indices"] == 1
    assert manifest["descriptors"][0]["operand_forms"] == 1
    assert manifest["descriptors"][1]["operand_forms"] == 0
    assert ".match_kind = LOOM_LOW_OPERAND_FORM_MATCH_ALL_EQUAL_I64" in generated.source


def test_shared_source_emits_one_storage_table_with_multiple_views() -> None:
    base_view = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=TEST_LOW_CORE_DESCRIPTOR_SET.descriptors[:1],
    )
    extension_view = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        key="test.low.extension.core",
        function_name="loom_test_low_extension_core_descriptor_set",
        c_table_prefix="TestLowExtensionCore",
        c_enum_prefix="TEST_LOW_EXTENSION_CORE",
        descriptors=TEST_LOW_CORE_DESCRIPTOR_SET.descriptors[:2],
    )
    storage_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=extension_view.descriptors,
    )

    source = generate_descriptor_set_shared_source(
        storage_set,
        (base_view, extension_view),
        format_output=False,
    )

    assert source.count("static const loom_low_descriptor_t kTestLowCoreDescriptors[]") == 1
    assert "kTestLowExtensionCoreDescriptors" not in source
    assert ".descriptors = kTestLowCoreDescriptors," in source
    assert ".descriptor_refs = kTestLowCoreDescriptorRefs," in source
    assert ".descriptor_refs = kTestLowExtensionCoreDescriptorRefs," in source
    assert ".descriptor_count = 1," in source
    assert ".descriptor_count = 2," in source
    assert ("const loom_low_descriptor_set_t* loom_test_low_extension_core_descriptor_set(void)") in source


def test_shared_source_emits_sibling_view_descriptor_surfaces() -> None:
    first_view = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(TEST_LOW_CORE_DESCRIPTOR_SET.descriptors[0],),
    )
    sibling_view = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        key="test.low.sibling.core",
        function_name="loom_test_low_sibling_core_descriptor_set",
        c_table_prefix="TestLowSiblingCore",
        c_enum_prefix="TEST_LOW_SIBLING_CORE",
        descriptors=(TEST_LOW_CORE_DESCRIPTOR_SET.descriptors[1],),
    )
    storage_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=TEST_LOW_CORE_DESCRIPTOR_SET.descriptors[:2],
    )

    source = generate_descriptor_set_shared_source(
        storage_set,
        (first_view, sibling_view, storage_set),
        format_output=False,
    )

    assert source.count("static const loom_low_operand_t kTestLowCoreOperands[]") == 1
    assert "static const loom_low_descriptor_t kTestLowSiblingCoreDescriptors[]" in source
    assert "static const loom_low_asm_form_t kTestLowSiblingCoreAsmForms[]" in source
    assert ".descriptors = kTestLowSiblingCoreDescriptors," in source
    assert ".asm_forms = kTestLowSiblingCoreAsmForms," in source
    assert ".descriptor_refs = kTestLowSiblingCoreDescriptorRefs," in source
    assert ".descriptor_count = 1," in source
    assert ".descriptor_ordinal = 0," in source
    assert "test.mul.i32" not in source


def test_generate_test_low_core_descriptor_set() -> None:
    generated = generate_descriptor_set(TEST_LOW_CORE_DESCRIPTOR_SET)
    manifest = json.loads(generated.manifest_json)

    assert manifest["key"] == "test.low.core"
    assert manifest["target"] == "test.low"
    assert manifest["table_counts"]["asm_forms"] >= 13
    assert any(descriptor["key"] == "test.spv.op_iadd.i32" for descriptor in manifest["descriptors"])
    assert any(form["mnemonic"] == "OpIAdd" for form in manifest["asm_forms"])


def test_generator_resolves_symbolic_hazard_resources() -> None:
    schedule_classes = tuple(
        replace(
            schedule_class,
            hazards=(
                Hazard(
                    HazardKind.MIN_DISTANCE,
                    resource="test.scalar",
                    producer_stage=0,
                    consumer_stage=1,
                    distance=2,
                ),
            ),
        )
        if schedule_class.name == "test.scalar.alu"
        else schedule_class
        for schedule_class in TEST_LOW_CORE_DESCRIPTOR_SET.schedule_classes
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        schedule_classes=schedule_classes,
    )

    generated = generate_descriptor_set(
        descriptor_set,
        DescriptorAllowlist(keys=("test.add.i32",)),
    )
    manifest = json.loads(generated.manifest_json)

    assert manifest["table_counts"]["hazards"] == 1
    assert ".reference_kind = LOOM_LOW_HAZARD_REFERENCE_KIND_RESOURCE" in generated.source
    assert ".reference_id = 0" in generated.source


def test_generator_rejects_unknown_hazard_resource() -> None:
    schedule_classes = tuple(
        replace(
            schedule_class,
            hazards=(Hazard(HazardKind.MIN_DISTANCE, resource="missing"),),
        )
        if schedule_class.name == "test.scalar.alu"
        else schedule_class
        for schedule_class in TEST_LOW_CORE_DESCRIPTOR_SET.schedule_classes
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        schedule_classes=schedule_classes,
    )

    with pytest.raises(
        ValueError,
        match=("schedule class 'test.scalar.alu' hazard references unknown resource 'missing'"),
    ):
        generate_descriptor_set(
            descriptor_set,
            DescriptorAllowlist(keys=("test.add.i32",)),
        )


def test_generator_rejects_ambiguous_hazard_reference() -> None:
    schedule_classes = tuple(
        replace(
            schedule_class,
            hazards=(Hazard(HazardKind.MIN_DISTANCE, resource="test.scalar", counter_id=0),),
        )
        if schedule_class.name == "test.scalar.alu"
        else schedule_class
        for schedule_class in TEST_LOW_CORE_DESCRIPTOR_SET.schedule_classes
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        schedule_classes=schedule_classes,
    )

    with pytest.raises(
        ValueError,
        match=("schedule class 'test.scalar.alu' hazard must reference exactly one resource, counter, or target id"),
    ):
        generate_descriptor_set(
            descriptor_set,
            DescriptorAllowlist(keys=("test.add.i32",)),
        )


def test_allowlist_accepts_semantic_tags() -> None:
    generated = generate_descriptor_set(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        DescriptorAllowlist(semantic_tags=("control.return.void",)),
    )
    manifest = json.loads(generated.manifest_json)

    assert [descriptor["key"] for descriptor in manifest["descriptors"]] == ["test.return.void"]


def test_allowlist_rejects_unknown_descriptor_key() -> None:
    with pytest.raises(ValueError, match="allowlist references unknown descriptor key 'missing'"):
        generate_descriptor_set(
            TEST_LOW_CORE_DESCRIPTOR_SET,
            DescriptorAllowlist(keys=("missing",)),
        )


def test_generator_rejects_asm_form_unknown_operand_field() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        asm_forms=(AsmForm(results=("dst",), operands=("lhs", "missing")),),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'test.add.i32' asm form 'test.add.i32' operand references unknown operand field 'missing'"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_asm_form_result_with_operand_role() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        asm_forms=(AsmForm(results=("lhs",), operands=("rhs",)),),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'test.add.i32' asm form 'test.add.i32' result field 'lhs' names a non-result operand"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_accepts_asm_form_implicit_packet_operand() -> None:
    base_descriptor = TEST_LOW_ADD_I32_DESCRIPTOR
    descriptor = replace(
        base_descriptor,
        operands=(
            base_descriptor.operands[0],
            replace(base_descriptor.operands[1], flags=(OperandFlag.IMPLICIT,)),
            *base_descriptor.operands[2:],
        ),
        asm_forms=(AsmForm(results=("dst",), operands=("lhs", "rhs")),),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    generated = generate_descriptor_set(descriptor_set)
    manifest = json.loads(generated.manifest_json)

    assert manifest["asm_forms"][0]["operands"] == ["lhs", "rhs"]


def test_generator_rejects_ambiguous_asm_mnemonics() -> None:
    first = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        asm_forms=(AsmForm(mnemonic="dup", results=("dst",), operands=("lhs", "rhs")),),
    )
    second = replace(
        TEST_LOW_MUL_I32_DESCRIPTOR,
        asm_forms=(AsmForm(mnemonic="dup", results=("dst",), operands=("lhs", "rhs")),),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(first, second))

    with pytest.raises(
        ValueError,
        match=("asm mnemonic 'dup' is ambiguous between descriptors 'test.add.i32' and 'test.mul.i32'"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_asm_form_unknown_immediate_field() -> None:
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        asm_forms=(AsmForm(results=("dst",), immediates=(AsmImmediate("missing"),)),),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'test.const.i32' asm form 'test.const.i32' immediate references unknown immediate field 'missing'"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_duplicate_asm_immediate_name() -> None:
    descriptor = replace(
        TEST_LOW_COND_BR_I32_DESCRIPTOR,
        asm_forms=(
            AsmForm(
                operands=("cond",),
                immediates=(
                    AsmImmediate("true_block", name="target"),
                    AsmImmediate("false_block", name="target"),
                ),
            ),
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'test.cond_br.i32' asm form 'test.cond_br.i32' uses immediate name 'target' more than once"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_asm_form_without_mnemonic() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        mnemonic=None,
        asm_forms=(AsmForm(results=("dst",), operands=("lhs", "rhs")),),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'test.add.i32' asm form must specify a mnemonic because the descriptor has no default mnemonic"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_empty_asm_form_mnemonic() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        asm_forms=(AsmForm(mnemonic="", results=("dst",), operands=("lhs", "rhs")),),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'test.add.i32' asm form specifies an empty mnemonic"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_missing_schedule_resource() -> None:
    bad_resource_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, resources=())

    with pytest.raises(
        ValueError,
        match="schedule class 'test.scalar.alu' references unknown resource 'test.scalar'",
    ):
        generate_descriptor_set(
            bad_resource_set,
            DescriptorAllowlist(keys=("test.add.i32",)),
        )


def test_generator_emits_enum_immediate_domains() -> None:
    domain = EnumDomain(
        "test.condition",
        values=(EnumValue("ne", 1), EnumValue("eq", 0)),
    )
    enum_immediate = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR.immediates[0],
        kind=ImmediateKind.ENUM,
        enum_domain="test.condition",
    )
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        immediates=(enum_immediate,),
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        enum_domains=(domain,),
        descriptors=(descriptor,),
    )

    generated = generate_descriptor_set(descriptor_set)
    manifest = json.loads(generated.manifest_json)

    assert "loom_low_enum_domain_t" in generated.source
    assert "test.condition" in generated.source
    assert "eq" in generated.source
    assert "ne" in generated.source
    assert manifest["table_counts"]["enum_domains"] == 1
    assert manifest["table_counts"]["enum_values"] == 2


def test_generator_rejects_missing_enum_immediate_domain() -> None:
    enum_immediate = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR.immediates[0],
        kind=ImmediateKind.ENUM,
        enum_domain=None,
    )
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        immediates=(enum_immediate,),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'test.const.i32' enum immediate 'i32_value' has no enum domain"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_unknown_enum_immediate_domain() -> None:
    enum_immediate = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR.immediates[0],
        kind=ImmediateKind.ENUM,
        enum_domain="missing",
    )
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        immediates=(enum_immediate,),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'test.const.i32' enum immediate 'i32_value' references unknown enum domain 'missing'"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_non_enum_immediate_domain() -> None:
    enum_immediate = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR.immediates[0],
        enum_domain="test.condition",
    )
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        immediates=(enum_immediate,),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'test.const.i32' non-enum immediate 'i32_value' references enum domain 'test.condition'"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_emits_defaulted_immediate() -> None:
    immediate = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR.immediates[0],
        flags=(ImmediateFlag.DEFAULT_VALUE,),
        default_value=7,
    )
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        immediates=(immediate,),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    generated = generate_descriptor_set(descriptor_set)

    assert "LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE" in generated.source
    assert ".default_value = INT64_C(7)" in generated.source


def test_generator_rejects_default_without_default_flag() -> None:
    immediate = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR.immediates[0],
        default_value=7,
    )
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        immediates=(immediate,),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'test.const.i32' immediate 'i32_value' has a default value without the default-value flag"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_default_outside_unsigned_range() -> None:
    immediate = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR.immediates[0],
        kind=ImmediateKind.UNSIGNED,
        flags=(ImmediateFlag.DEFAULT_VALUE,),
        unsigned_max=3,
        default_value=4,
    )
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        immediates=(immediate,),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'test.const.i32' immediate 'i32_value' default value is out of unsigned range"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_default_outside_enum_domain() -> None:
    domain = EnumDomain(
        "test.condition",
        values=(EnumValue("ne", 1), EnumValue("eq", 0)),
    )
    immediate = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR.immediates[0],
        kind=ImmediateKind.ENUM,
        enum_domain="test.condition",
        flags=(ImmediateFlag.DEFAULT_VALUE,),
        default_value=7,
    )
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        immediates=(immediate,),
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        enum_domains=(domain,),
        descriptors=(descriptor,),
    )

    with pytest.raises(
        ValueError,
        match=("descriptor 'test.const.i32' immediate 'i32_value' default value is not in enum domain 'test.condition'"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_missing_schedule_class() -> None:
    bad_descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        schedule_class=cast(str, None),
    )
    bad_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(bad_descriptor,),
    )

    with pytest.raises(
        ValueError,
        match="descriptor 'test.add.i32' has no schedule class",
    ):
        generate_descriptor_set(bad_set)


def test_generator_rejects_result_after_operand() -> None:
    bad_descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        operands=(
            TEST_LOW_ADD_I32_DESCRIPTOR.operands[1],
            TEST_LOW_ADD_I32_DESCRIPTOR.operands[0],
        ),
    )
    bad_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(bad_descriptor,),
    )

    with pytest.raises(
        ValueError,
        match=("descriptor 'test.add.i32' has result operand 'dst' after non-result operands"),
    ):
        generate_descriptor_set(bad_set)


def test_generator_rejects_operand_result_role() -> None:
    destructive_result = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR.operands[0],
        role=OperandRole.OPERAND_RESULT,
    )
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        operands=(
            destructive_result,
            *TEST_LOW_ADD_I32_DESCRIPTOR.operands[1:],
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'test.add.i32' operand 'dst' uses OPERAND_RESULT; use separate result and operand rows plus an explicit constraint"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_implicit_operand_without_implicit_flag() -> None:
    implicit_operand = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR.operands[1],
        role=OperandRole.IMPLICIT,
        flags=(),
    )
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        operands=(
            TEST_LOW_ADD_I32_DESCRIPTOR.operands[0],
            implicit_operand,
            *TEST_LOW_ADD_I32_DESCRIPTOR.operands[2:],
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'test.add.i32' implicit operand 'lhs' must set the implicit flag"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_accepts_tied_duplicate_operand_encoding_field() -> None:
    base_descriptor = TEST_LOW_ADD_I32_DESCRIPTOR
    descriptor = replace(
        base_descriptor,
        operands=(
            replace(base_descriptor.operands[0], encoding_field_id=7),
            replace(base_descriptor.operands[1], encoding_field_id=7),
            *base_descriptor.operands[2:],
        ),
        constraints=(Constraint(ConstraintKind.TIED, 0, 1),),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    generated = generate_descriptor_set(descriptor_set)

    assert "test.add.i32" in generated.source


def test_generator_rejects_untied_duplicate_operand_encoding_field() -> None:
    base_descriptor = TEST_LOW_ADD_I32_DESCRIPTOR
    descriptor = replace(
        base_descriptor,
        operands=(
            replace(base_descriptor.operands[0], encoding_field_id=7),
            replace(base_descriptor.operands[1], encoding_field_id=7),
            *base_descriptor.operands[2:],
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'test.add.i32' operands 'dst' and 'lhs' share encoding field id 7 without a tied constraint"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_operand_fixed_encoding_field_overlap() -> None:
    base_descriptor = TEST_LOW_ADD_I32_DESCRIPTOR
    descriptor = replace(
        base_descriptor,
        operands=(
            replace(base_descriptor.operands[0], encoding_field_id=7),
            *base_descriptor.operands[1:],
        ),
        encoding_field_values=(EncodingFieldValue(7, 0),),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'test.add.i32' operand 'dst' shares fixed encoding field id 7"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_emits_sliced_immediate_encoding_rows() -> None:
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        immediates=(
            Immediate(
                "i32_value",
                ImmediateKind.SIGNED,
                bit_width=32,
                encoding_slices=(
                    ImmediateEncodingSlice(7, 0, 16),
                    ImmediateEncodingSlice(8, 16, 16),
                ),
                signed_min=-(2**31),
                unsigned_max=(2**31) - 1,
            ),
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    generated = generate_descriptor_set(descriptor_set)

    assert "loom_low_immediate_encoding_slice_t" in generated.source
    assert ".encoding_slice_start = 0," in generated.source
    assert ".encoding_slice_count = 2," in generated.source
    assert ".encoding_field_id = 7," in generated.source
    assert ".source_bit_offset = 16," in generated.source


def test_generator_rejects_immediate_with_direct_and_sliced_encoding() -> None:
    base_descriptor = TEST_LOW_CONST_I32_DESCRIPTOR
    descriptor = replace(
        base_descriptor,
        immediates=(
            replace(
                base_descriptor.immediates[0],
                encoding_field_id=7,
                encoding_slices=(ImmediateEncodingSlice(8, 0, 32),),
            ),
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'test.const.i32' immediate 'i32_value' uses both direct and sliced encoding fields"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_incomplete_sliced_immediate_encoding() -> None:
    base_descriptor = TEST_LOW_CONST_I32_DESCRIPTOR
    descriptor = replace(
        base_descriptor,
        immediates=(
            replace(
                base_descriptor.immediates[0],
                encoding_slices=(ImmediateEncodingSlice(7, 0, 16),),
            ),
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'test.const.i32' immediate 'i32_value' encoding slices cover 0xffff instead of 0xffffffff"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_absent_encoding_without_pseudo_flag() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        encoding_id=LOW_DESCRIPTOR_ENCODING_ID_NONE,
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'test.add.i32' uses absent encoding id without the pseudo flag"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_pseudo_flag_with_target_encoding() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        flags=(
            *TEST_LOW_ADD_I32_DESCRIPTOR.flags,
            DescriptorFlag.PSEUDO,
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'test.add.i32' uses the pseudo flag with a target encoding id"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_pseudo_flag_with_target_encoding_format() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        encoding_format_id=1,
        encoding_id=LOW_DESCRIPTOR_ENCODING_ID_NONE,
        flags=(
            *TEST_LOW_ADD_I32_DESCRIPTOR.flags,
            DescriptorFlag.PSEUDO,
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'test.add.i32' uses the pseudo flag with a target encoding format id"),
    ):
        generate_descriptor_set(descriptor_set)
