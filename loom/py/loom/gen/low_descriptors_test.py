# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from loom.gen.low_descriptors import DescriptorAllowlist, generate_descriptor_set
from loom.target.emit.ireevm.descriptors import IREEVM_CORE_DESCRIPTOR_SET
from loom.target.low_descriptors import OperandRole


def test_generate_ireevm_core_descriptor_set() -> None:
    generated = generate_descriptor_set(IREEVM_CORE_DESCRIPTOR_SET)

    assert "loom_ireevm_core_descriptor_set" in generated.header
    assert 'LOOM_BSTRING_LITERAL("\\x0c", "iree.vm.core")' in generated.source
    assert "iree.vm.add.i32" in generated.source
    assert "fingerprint" not in generated.source
    assert "fingerprint" not in generated.manifest_json

    manifest = json.loads(generated.manifest_json)
    assert manifest["key"] == "iree.vm.core"
    assert manifest["abi_version"] == 3
    assert manifest["table_counts"]["descriptors"] >= 9
    assert any(descriptor["key"] == "iree.vm.call.import.i32" for descriptor in manifest["descriptors"])


def test_allowlist_closes_over_referenced_descriptor_tables() -> None:
    generated = generate_descriptor_set(
        IREEVM_CORE_DESCRIPTOR_SET,
        DescriptorAllowlist(keys=("iree.vm.add.i32",)),
    )
    manifest = json.loads(generated.manifest_json)

    assert [descriptor["key"] for descriptor in manifest["descriptors"]] == ["iree.vm.add.i32"]
    assert manifest["table_counts"]["descriptors"] == 1
    assert manifest["table_counts"]["reg_classes"] == 6
    assert manifest["table_counts"]["reg_class_alts"] == 1
    assert manifest["table_counts"]["schedule_classes"] == 1
    assert manifest["table_counts"]["resources"] == 1
    assert "iree.vm.call.import.i32" not in generated.source


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


def test_operand_result_role_counts_as_result_prefix() -> None:
    destructive_result = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[1].operands[0],
        role=OperandRole.OPERAND_RESULT,
    )
    descriptor = replace(
        IREEVM_CORE_DESCRIPTOR_SET.descriptors[1],
        operands=(destructive_result, *IREEVM_CORE_DESCRIPTOR_SET.descriptors[1].operands[1:]),
    )
    descriptor_set = replace(IREEVM_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    generated = generate_descriptor_set(descriptor_set)
    manifest = json.loads(generated.manifest_json)

    assert manifest["descriptors"][0]["results"] == 1
