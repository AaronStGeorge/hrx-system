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

from loom.gen.low_descriptors import DescriptorAllowlist, generate_descriptor_set
from loom.target.arch.wasm.descriptors import WASM_CORE_SIMD128_DESCRIPTOR_SET
from loom.target.emit.ireevm.descriptors import IREEVM_CORE_DESCRIPTOR_SET
from loom.target.low_descriptors import EnumDomain, EnumValue, ImmediateKind, OperandRole


def test_generate_ireevm_core_descriptor_set() -> None:
    generated = generate_descriptor_set(IREEVM_CORE_DESCRIPTOR_SET)

    assert "loom_ireevm_core_descriptor_set" in generated.header
    assert 'LOOM_BSTRING_LITERAL("\\x0c", "iree.vm.core")' in generated.source
    assert "iree.vm.add.i32" in generated.source
    assert "fingerprint" not in generated.source
    assert "fingerprint" not in generated.manifest_json

    manifest = json.loads(generated.manifest_json)
    assert manifest["key"] == "iree.vm.core"
    assert manifest["abi_version"] == 5
    assert manifest["table_counts"]["descriptors"] >= 9
    assert manifest["table_counts"]["descriptor_refs"] == manifest["table_counts"]["descriptors"]
    assert any(descriptor["key"] == "iree.vm.call.import.i32" for descriptor in manifest["descriptors"])


def test_allowlist_closes_over_referenced_descriptor_tables() -> None:
    generated = generate_descriptor_set(
        IREEVM_CORE_DESCRIPTOR_SET,
        DescriptorAllowlist(keys=("iree.vm.add.i32",)),
    )
    manifest = json.loads(generated.manifest_json)

    assert [descriptor["key"] for descriptor in manifest["descriptors"]] == ["iree.vm.add.i32"]
    assert manifest["table_counts"]["descriptors"] == 1
    assert manifest["table_counts"]["descriptor_refs"] == 1
    assert manifest["table_counts"]["reg_classes"] == 6
    assert manifest["table_counts"]["reg_class_alts"] == 1
    assert manifest["table_counts"]["schedule_classes"] == 1
    assert manifest["table_counts"]["resources"] == 1
    assert "iree.vm.call.import.i32" not in generated.source


def test_generate_wasm_core_simd128_descriptor_set() -> None:
    generated = generate_descriptor_set(WASM_CORE_SIMD128_DESCRIPTOR_SET)

    assert "loom_wasm_core_simd128_descriptor_set" in generated.header
    assert 'LOOM_BSTRING_LITERAL("\\x11", "wasm.core.simd128")' in generated.source
    assert "wasm.i32x4.add" in generated.source
    assert "fingerprint" not in generated.source
    assert "fingerprint" not in generated.manifest_json

    manifest = json.loads(generated.manifest_json)
    assert manifest["key"] == "wasm.core.simd128"
    assert manifest["target"] == "wasm"
    assert manifest["feature_namespace"] == "wasm.simd128.v1"
    assert manifest["abi_version"] == 5
    assert manifest["table_counts"]["descriptors"] >= 12
    assert manifest["table_counts"]["descriptor_refs"] == manifest["table_counts"]["descriptors"]
    assert any(descriptor["key"] == "wasm.v128.load" for descriptor in manifest["descriptors"])


def test_allowlist_accepts_semantic_tags() -> None:
    generated = generate_descriptor_set(
        IREEVM_CORE_DESCRIPTOR_SET,
        DescriptorAllowlist(semantic_tags=("control.return.void",)),
    )
    manifest = json.loads(generated.manifest_json)

    assert [descriptor["key"] for descriptor in manifest["descriptors"]] == ["iree.vm.return.void"]


def test_allowlist_rejects_unknown_descriptor_key() -> None:
    with pytest.raises(ValueError, match="allowlist references unknown descriptor key 'missing'"):
        generate_descriptor_set(
            IREEVM_CORE_DESCRIPTOR_SET,
            DescriptorAllowlist(keys=("missing",)),
        )


def test_generator_rejects_missing_schedule_resource() -> None:
    bad_resource_set = replace(IREEVM_CORE_DESCRIPTOR_SET, resources=())

    with pytest.raises(
        ValueError,
        match="schedule class 'vm.alu.i32' references unknown resource 'vm.alu'",
    ):
        generate_descriptor_set(
            bad_resource_set,
            DescriptorAllowlist(keys=("iree.vm.add.i32",)),
        )


def test_generator_emits_enum_immediate_domains() -> None:
    domain = EnumDomain(
        "vm.condition",
        values=(EnumValue("ne", 1), EnumValue("eq", 0)),
    )
    enum_immediate = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[0].immediates[0],
        kind=ImmediateKind.ENUM,
        enum_domain="vm.condition",
    )
    descriptor = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[0],
        immediates=(enum_immediate,),
    )
    descriptor_set = replace(
        IREEVM_CORE_DESCRIPTOR_SET,
        enum_domains=(domain,),
        descriptors=(descriptor,),
    )

    generated = generate_descriptor_set(descriptor_set)
    manifest = json.loads(generated.manifest_json)

    assert "loom_low_enum_domain_t" in generated.source
    assert "vm.condition" in generated.source
    assert "eq" in generated.source
    assert "ne" in generated.source
    assert manifest["table_counts"]["enum_domains"] == 1
    assert manifest["table_counts"]["enum_values"] == 2


def test_generator_rejects_missing_enum_immediate_domain() -> None:
    enum_immediate = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[0].immediates[0],
        kind=ImmediateKind.ENUM,
        enum_domain=None,
    )
    descriptor = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[0],
        immediates=(enum_immediate,),
    )
    descriptor_set = replace(IREEVM_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'iree.vm.const.i32' enum immediate 'i32_value' has no enum domain"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_unknown_enum_immediate_domain() -> None:
    enum_immediate = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[0].immediates[0],
        kind=ImmediateKind.ENUM,
        enum_domain="missing",
    )
    descriptor = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[0],
        immediates=(enum_immediate,),
    )
    descriptor_set = replace(IREEVM_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'iree.vm.const.i32' enum immediate 'i32_value' references unknown enum domain 'missing'"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_non_enum_immediate_domain() -> None:
    enum_immediate = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[0].immediates[0],
        enum_domain="vm.condition",
    )
    descriptor = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[0],
        immediates=(enum_immediate,),
    )
    descriptor_set = replace(IREEVM_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'iree.vm.const.i32' non-enum immediate 'i32_value' references enum domain 'vm.condition'"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_missing_schedule_class() -> None:
    bad_descriptor = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[1],
        schedule_class=cast(str, None),
    )
    bad_set = replace(
        IREEVM_CORE_DESCRIPTOR_SET,
        descriptors=(bad_descriptor,),
    )

    with pytest.raises(
        ValueError,
        match="descriptor 'iree.vm.add.i32' has no schedule class",
    ):
        generate_descriptor_set(bad_set)


def test_generator_rejects_result_after_operand() -> None:
    bad_descriptor = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[1],
        operands=(
            IREEVM_CORE_DESCRIPTOR_SET.descriptors[1].operands[1],
            IREEVM_CORE_DESCRIPTOR_SET.descriptors[1].operands[0],
        ),
    )
    bad_set = replace(
        IREEVM_CORE_DESCRIPTOR_SET,
        descriptors=(bad_descriptor,),
    )

    with pytest.raises(
        ValueError,
        match=("descriptor 'iree.vm.add.i32' has result operand 'dst' after non-result operands"),
    ):
        generate_descriptor_set(bad_set)


def test_generator_rejects_operand_result_role() -> None:
    destructive_result = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[1].operands[0],
        role=OperandRole.OPERAND_RESULT,
    )
    descriptor = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[1],
        operands=(
            destructive_result,
            *IREEVM_CORE_DESCRIPTOR_SET.descriptors[1].operands[1:],
        ),
    )
    descriptor_set = replace(IREEVM_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=("descriptor 'iree.vm.add.i32' operand 'dst' uses OPERAND_RESULT; use separate result and operand rows plus an explicit constraint"),
    ):
        generate_descriptor_set(descriptor_set)
