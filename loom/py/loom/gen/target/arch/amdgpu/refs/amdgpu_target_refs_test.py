# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from loom.gen.target.arch.amdgpu.refs import amdgpu_target_refs
from loom.target.low_descriptors import AsmForm, Descriptor, DescriptorSet


@contextmanager
def _raises_value_error(match: str) -> Iterator[None]:
    try:
        yield
    except ValueError as exc:
        if re.search(match, str(exc)) is None:
            raise AssertionError(f"ValueError message {exc!s} did not match {match}") from exc
    else:
        raise AssertionError("expected ValueError")


def _descriptor(key: str, asm_forms: tuple[AsmForm, ...]) -> Descriptor:
    return Descriptor(
        key=key,
        mnemonic=None,
        semantic_tag=None,
        operands=(),
        schedule_class="none",
        asm_forms=asm_forms,
    )


def _descriptor_set(*descriptors: Descriptor) -> DescriptorSet:
    return DescriptorSet(
        key="amdgpu.test.core",
        target_key="amdgpu",
        feature_key=None,
        c_header_path=Path("test.h"),
        c_source_path=Path("test.c"),
        header_guard="TEST_H_",
        public_header="test.h",
        function_name="test",
        c_table_prefix="test",
        c_enum_prefix="TEST",
        generator_version=0,
        reg_classes=(),
        resources=(),
        schedule_classes=(),
        descriptors=descriptors,
    )


def _valid_contract_descriptors() -> tuple[Descriptor, ...]:
    return (
        _descriptor(
            "amdgpu.global_load_b32_saddr",
            (AsmForm(operands=("vaddr", "saddr")),),
        ),
        _descriptor(
            "amdgpu.global_load_b64_saddr",
            (AsmForm(operands=("vaddr", "saddr", "m0")),),
        ),
        _descriptor(
            "amdgpu.flat_load_u8",
            (AsmForm(operands=("addr",)),),
        ),
    )


def test_target_refs_header_is_constant_fragment() -> None:
    source = amdgpu_target_refs._emit_tables_header()

    assert "typedef " not in source
    assert "extern " not in source
    assert "loom_amdgpu_descriptor_ref_ordinal" not in source
    assert "loom/codegen/low/descriptors.h" not in source
    assert "#define LOOM_AMDGPU_DESCRIPTOR_REF_COUNT" in source
    assert "LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32" in source


def test_lowering_descriptor_contracts_accept_expected_asm_shapes() -> None:
    amdgpu_target_refs._validate_lowering_descriptor_contracts(_descriptor_set(*_valid_contract_descriptors()))


def test_lowering_descriptor_contracts_reject_missing_canonical_form() -> None:
    with _raises_value_error("amdgpu.global_load_b32_saddr.*exactly one canonical asm form"):
        amdgpu_target_refs._validate_lowering_descriptor_contracts(
            _descriptor_set(
                _descriptor("amdgpu.global_load_b32_saddr", ()),
                *_valid_contract_descriptors()[1:],
            )
        )


def test_lowering_descriptor_contracts_reject_missing_helper_descriptor() -> None:
    with _raises_value_error("missing descriptor 'amdgpu.global_load_b64_saddr' required by target lowering"):
        amdgpu_target_refs._validate_lowering_descriptor_contracts(_descriptor_set(*_valid_contract_descriptors()[:1]))


def test_lowering_descriptor_contracts_reject_bad_operand_count() -> None:
    with _raises_value_error("amdgpu.flat_load_u8.*expected one of: 1, 2"):
        amdgpu_target_refs._validate_lowering_descriptor_contracts(
            _descriptor_set(
                *_valid_contract_descriptors()[:2],
                _descriptor(
                    "amdgpu.flat_load_u8",
                    (AsmForm(operands=("addr", "m0", "extra")),),
                ),
            )
        )
