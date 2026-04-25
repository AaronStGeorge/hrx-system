# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

import pytest

from loom.target.arch.amdgpu.descriptors import (
    _ADDRESS_OFFSET_DS16_ENCODING_ID,
    _ADDRESS_OFFSET_DWORD_ENCODING_ID,
    _ADDRESS_OFFSET_DWORD_STRIDE64_ENCODING_ID,
    _GFX12_TH_ATOMIC_RETURN_VALUE,
    _gfx12_core_overlays,
    _gfx1250_core_overlays,
    _validate_address_immediate_units,
)
from loom.target.low_descriptors import (
    Descriptor,
    DescriptorSet,
    Effect,
    EffectKind,
    Immediate,
    ImmediateKind,
    MemorySpace,
)


def _descriptor_set(descriptor: Descriptor) -> DescriptorSet:
    return DescriptorSet(
        key="amdgpu.test.core",
        target_key="amdgpu.test",
        feature_key="amdgpu.test",
        c_header_path=Path("test.h"),
        c_source_path=Path("test.c"),
        header_guard="TEST_H_",
        public_header="test.h",
        function_name="test_descriptor_set",
        c_table_prefix="test",
        c_enum_prefix="TEST",
        generator_version=1,
        reg_classes=(),
        resources=(),
        schedule_classes=(),
        descriptors=(descriptor,),
    )


def _memory_descriptor(*, immediates: tuple[Immediate, ...]) -> Descriptor:
    return Descriptor(
        key="amdgpu.memory",
        mnemonic="memory",
        semantic_tag="memory.load.u32",
        operands=(),
        schedule_class="amdgpu.vmem.load",
        immediates=immediates,
        effects=(Effect(EffectKind.READ, memory_space=MemorySpace.GLOBAL),),
    )


def _fixed_field_value(
    fixed_encoding_fields: tuple[tuple[str, int], ...], field_name: str
) -> int:
    for actual_field_name, value in fixed_encoding_fields:
        if actual_field_name == field_name:
            return value
    raise AssertionError(f"missing fixed encoding field '{field_name}'")


def test_gfx12_global_atomic_return_uses_temporal_hint_return_bit() -> None:
    for overlays in (_gfx12_core_overlays(), _gfx1250_core_overlays()):
        no_return = next(
            overlay
            for overlay in overlays
            if overlay.descriptor_key == "amdgpu.global_atomic_add_u32_saddr"
        )
        with_return = next(
            overlay
            for overlay in overlays
            if overlay.descriptor_key == "amdgpu.global_atomic_add_u32_rtn_saddr"
        )

        assert _fixed_field_value(no_return.fixed_encoding_fields, "TH") == 0
        assert (
            _fixed_field_value(with_return.fixed_encoding_fields, "TH")
            == _GFX12_TH_ATOMIC_RETURN_VALUE
        )


def test_address_immediate_validation_rejects_missing_unit_metadata() -> None:
    descriptor = _memory_descriptor(
        immediates=(
            Immediate(
                "offset",
                ImmediateKind.UNSIGNED,
                bit_width=8,
                unsigned_max=255,
            ),
        )
    )

    with pytest.raises(ValueError, match="no address-unit encoding"):
        _validate_address_immediate_units(_descriptor_set(descriptor))


def test_address_immediate_validation_rejects_inconsistent_split_units() -> None:
    descriptor = _memory_descriptor(
        immediates=(
            Immediate(
                "offset0",
                ImmediateKind.UNSIGNED,
                bit_width=8,
                encoding_id=_ADDRESS_OFFSET_DWORD_ENCODING_ID,
                unsigned_max=255,
            ),
            Immediate(
                "offset1",
                ImmediateKind.UNSIGNED,
                bit_width=8,
                encoding_id=_ADDRESS_OFFSET_DWORD_STRIDE64_ENCODING_ID,
                unsigned_max=255,
            ),
        )
    )

    with pytest.raises(ValueError, match="inconsistent split address offset units"):
        _validate_address_immediate_units(_descriptor_set(descriptor))


def test_address_immediate_validation_accepts_split_ds16_offset() -> None:
    descriptor = _memory_descriptor(
        immediates=(
            Immediate(
                "offset",
                ImmediateKind.UNSIGNED,
                bit_width=16,
                encoding_id=_ADDRESS_OFFSET_DS16_ENCODING_ID,
                unsigned_max=65535,
            ),
        )
    )

    _validate_address_immediate_units(_descriptor_set(descriptor))
