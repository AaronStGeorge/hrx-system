# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from loom.target.arch.amdgpu.descriptors import (
    _ADDRESS_OFFSET_DS16_ENCODING_ID,
    _ADDRESS_OFFSET_DWORD_ENCODING_ID,
    _ADDRESS_OFFSET_DWORD_STRIDE64_ENCODING_ID,
    _GFX12_TH_ATOMIC_RETURN_VALUE,
    _REG_EXEC,
    _SCHEDULE_SALU,
    _SCHEDULE_VALU,
    _SOURCE_INLINE_F32_ENCODING_ID,
    _SOURCE_INLINE_U32_ENCODING_ID,
    AMDGPU_ATOMIC_DESCRIPTOR_CATEGORY,
    AMDGPU_COMPARE_SELECT_DESCRIPTOR_CATEGORY,
    AMDGPU_CONTROL_DESCRIPTOR_CATEGORY,
    AMDGPU_DESCRIPTOR_CATEGORIES,
    AMDGPU_MEMORY_DESCRIPTOR_CATEGORY,
    AMDGPU_VECTOR_DESCRIPTOR_CATEGORY,
    AmdgpuAtomicKind,
    AmdgpuAtomicMemorySpace,
    AmdgpuAtomicOperationKind,
    AmdgpuAtomicValueKind,
    AmdgpuMemoryAddressForm,
    _categorize_amdgpu_descriptors,
    _gfx11_core_overlays,
    _gfx12_core_overlays,
    _gfx1250_core_overlays,
    _validate_address_immediate_units,
    _with_execution_mask_state_read,
    amdgpu_atomic_descriptor_candidates,
    amdgpu_descriptor_category_groups,
)
from loom.target.low_descriptors import (
    Descriptor,
    DescriptorSet,
    Effect,
    EffectKind,
    Immediate,
    ImmediateKind,
    MemorySpace,
    Operand,
    OperandFlag,
    OperandRole,
    RegClassAlt,
    RegClassAltFlag,
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


def _immediate_default(immediates: tuple[Immediate, ...], name: str) -> int:
    for immediate in immediates:
        if immediate.field_name == name:
            return immediate.default_value
    raise AssertionError(f"missing immediate '{name}'")


def _descriptor(key: str, semantic_tag: str) -> Descriptor:
    return Descriptor(
        key=key,
        mnemonic=None,
        semantic_tag=semantic_tag,
        operands=(),
        schedule_class="amdgpu.test",
    )


def _expect_value_error_contains(
    expected_message: str, thunk: Callable[[], object]
) -> None:
    actual_message: str | None = None
    try:
        thunk()
    except ValueError as exc:
        actual_message = str(exc)
    assert actual_message is not None, "expected ValueError"
    assert expected_message in actual_message


def test_execution_masked_descriptors_read_exec_state() -> None:
    descriptor = Descriptor(
        key="amdgpu.v_add_u32",
        mnemonic="v_add_u32",
        semantic_tag="integer.add.u32",
        operands=(
            Operand("dst", OperandRole.RESULT, (RegClassAlt("amdgpu.vgpr"),)),
            Operand("lhs", OperandRole.OPERAND, (RegClassAlt("amdgpu.vgpr"),)),
            Operand("rhs", OperandRole.OPERAND, (RegClassAlt("amdgpu.vgpr"),)),
        ),
        schedule_class=_SCHEDULE_VALU,
    )

    masked_descriptor = _with_execution_mask_state_read(descriptor)
    exec_operand = masked_descriptor.operands[-1]

    assert exec_operand.field_name == "exec_in"
    assert exec_operand.role is OperandRole.IMPLICIT
    assert exec_operand.reg_alts == (
        RegClassAlt(_REG_EXEC, flags=(RegClassAltFlag.PHYSICAL_ONLY,)),
    )
    assert OperandFlag.IMPLICIT in exec_operand.flags
    assert OperandFlag.STATE_READ in exec_operand.flags
    assert (
        _with_execution_mask_state_read(masked_descriptor).operands
        == masked_descriptor.operands
    )


def test_scalar_descriptors_do_not_get_execution_mask_state_read() -> None:
    descriptor = Descriptor(
        key="amdgpu.s_add_u32",
        mnemonic="s_add_u32",
        semantic_tag="integer.add.u32",
        operands=(
            Operand("dst", OperandRole.RESULT, (RegClassAlt("amdgpu.sgpr"),)),
            Operand("lhs", OperandRole.OPERAND, (RegClassAlt("amdgpu.sgpr"),)),
            Operand("rhs", OperandRole.OPERAND, (RegClassAlt("amdgpu.sgpr"),)),
        ),
        schedule_class=_SCHEDULE_SALU,
    )

    assert _with_execution_mask_state_read(descriptor) is descriptor


def test_amdgpu_descriptor_categories_are_stable() -> None:
    assert tuple(category.key for category in AMDGPU_DESCRIPTOR_CATEGORIES) == (
        "scalar",
        "vector",
        "convert",
        "compare_select",
        "memory",
        "atomic",
        "matrix",
        "control",
        "cache",
        "misc",
    )


def test_amdgpu_descriptor_categorization_uses_semantics() -> None:
    descriptors = _categorize_amdgpu_descriptors(
        (
            _descriptor("amdgpu.v_add_u32", "integer.add.u32"),
            _descriptor("amdgpu.v_cmp_eq_i32", "cmp.i32.eq"),
            _descriptor("amdgpu.buffer_load_dword", "memory.load.u32"),
            _descriptor("amdgpu.global_atomic_add_u32", "memory.global.atomic.add.u32"),
            _descriptor("amdgpu.s_waitcnt", "control.waitcnt"),
        )
    )

    assert tuple(descriptor.category for descriptor in descriptors) == (
        AMDGPU_VECTOR_DESCRIPTOR_CATEGORY,
        AMDGPU_COMPARE_SELECT_DESCRIPTOR_CATEGORY,
        AMDGPU_MEMORY_DESCRIPTOR_CATEGORY,
        AMDGPU_ATOMIC_DESCRIPTOR_CATEGORY,
        AMDGPU_CONTROL_DESCRIPTOR_CATEGORY,
    )


def test_amdgpu_descriptor_category_groups_preserve_category_and_descriptor_order() -> (
    None
):
    descriptors = _categorize_amdgpu_descriptors(
        (
            _descriptor("amdgpu.buffer_load_dword", "memory.load.u32"),
            _descriptor("amdgpu.v_add_u32", "integer.add.u32"),
            _descriptor("amdgpu.global_atomic_add_u32", "memory.global.atomic.add.u32"),
            _descriptor("amdgpu.buffer_store_dword", "memory.store.u32"),
        )
    )

    groups = amdgpu_descriptor_category_groups(descriptors)

    assert [category for category, _ in groups] == [
        AMDGPU_VECTOR_DESCRIPTOR_CATEGORY,
        AMDGPU_MEMORY_DESCRIPTOR_CATEGORY,
        AMDGPU_ATOMIC_DESCRIPTOR_CATEGORY,
    ]
    assert [descriptor.key for _, group in groups for descriptor in group] == [
        "amdgpu.v_add_u32",
        "amdgpu.buffer_load_dword",
        "amdgpu.buffer_store_dword",
        "amdgpu.global_atomic_add_u32",
    ]


def test_atomic_descriptor_candidates_are_derived_from_overlay_metadata() -> None:
    candidates = amdgpu_atomic_descriptor_candidates()

    assert len(candidates) == 104
    assert candidates[0].descriptor_key == "amdgpu.ds_add_u32"
    assert candidates[0].memory_space == AmdgpuAtomicMemorySpace.WORKGROUP
    assert candidates[0].address_form == AmdgpuMemoryAddressForm.DEFAULT
    assert candidates[0].operation_kind == AmdgpuAtomicOperationKind.REDUCE
    assert candidates[0].atomic_kind == AmdgpuAtomicKind.ADDI
    assert candidates[0].value_kind == AmdgpuAtomicValueKind.I32

    global_saddr_add = next(
        candidate
        for candidate in candidates
        if candidate.descriptor_key == "amdgpu.global_atomic_add_u32_saddr"
    )
    assert global_saddr_add.memory_space == AmdgpuAtomicMemorySpace.GLOBAL
    assert global_saddr_add.address_form == AmdgpuMemoryAddressForm.GLOBAL_SADDR

    flat_cmpxchg = next(
        candidate
        for candidate in candidates
        if candidate.descriptor_key == "amdgpu.flat_atomic_cmpswap_b32_rtn"
    )
    assert flat_cmpxchg.memory_space == AmdgpuAtomicMemorySpace.GENERIC
    assert flat_cmpxchg.operation_kind == AmdgpuAtomicOperationKind.CMPXCHG
    assert flat_cmpxchg.atomic_kind == AmdgpuAtomicKind.NONE


def test_atomic_descriptor_candidates_exclude_unsupported_packed_half_rows() -> None:
    keys = {
        candidate.descriptor_key for candidate in amdgpu_atomic_descriptor_candidates()
    }

    assert "amdgpu.buffer_atomic_pk_add_f16" not in keys
    assert "amdgpu.flat_atomic_pk_add_bf16_rtn" not in keys
    assert "amdgpu.ds_pk_add_rtn_f16" not in keys


def test_gfx12_global_atomic_return_uses_temporal_hint_return_bit() -> None:
    for overlays in (_gfx12_core_overlays(), _gfx1250_core_overlays()):
        for descriptor_prefix, descriptor_suffix in (
            ("amdgpu.global_atomic", "_saddr"),
            ("amdgpu.flat_atomic", ""),
        ):
            no_return = next(
                overlay
                for overlay in overlays
                if overlay.descriptor_key
                == f"{descriptor_prefix}_add_u32{descriptor_suffix}"
            )
            with_return = next(
                overlay
                for overlay in overlays
                if overlay.descriptor_key
                == f"{descriptor_prefix}_add_u32_rtn{descriptor_suffix}"
            )

            assert _immediate_default(no_return.immediates, "th") == 0
            assert _immediate_default(with_return.immediates, "th") == (
                _GFX12_TH_ATOMIC_RETURN_VALUE
            )


def test_gfx12_global_cache_controls_expose_scope_immediate() -> None:
    for overlays in (_gfx12_core_overlays(), _gfx1250_core_overlays()):
        for descriptor_key in (
            "amdgpu.global_inv",
            "amdgpu.global_wb",
            "amdgpu.global_wbinv",
        ):
            descriptor = next(
                overlay
                for overlay in overlays
                if overlay.descriptor_key == descriptor_key
            )
            assert _immediate_default(descriptor.immediates, "scope") == 0


def test_vop3_shift_immediate_is_constrained_to_inline_source_selector() -> None:
    descriptor = next(
        overlay
        for overlay in _gfx12_core_overlays()
        if overlay.descriptor_key == "amdgpu.v_lshlrev_b32.vop3_imm"
    )
    assert len(descriptor.immediates) == 1
    immediate = descriptor.immediates[0]
    assert immediate.field_name == "imm32"
    assert immediate.encoding_id == _SOURCE_INLINE_U32_ENCODING_ID
    assert immediate.unsigned_max == 64


def test_vop2_f32_inline_immediate_uses_enum_domain() -> None:
    descriptor = next(
        overlay
        for overlay in _gfx12_core_overlays()
        if overlay.descriptor_key == "amdgpu.v_add_f32.src0_inline"
    )
    assert len(descriptor.immediates) == 1
    immediate = descriptor.immediates[0]
    assert immediate.field_name == "imm32"
    assert immediate.kind == ImmediateKind.ENUM
    assert immediate.encoding_id == _SOURCE_INLINE_F32_ENCODING_ID
    assert immediate.enum_domain == "amdgpu.source_inline_f32"


def test_vop2_f32_uses_inline_then_literal_operand_forms() -> None:
    overlays_by_key = {
        overlay.descriptor_key: overlay for overlay in _gfx12_core_overlays()
    }
    for descriptor_key in (
        "amdgpu.v_add_f32",
        "amdgpu.v_sub_f32",
        "amdgpu.v_mul_f32",
        "amdgpu.v_min_f32",
        "amdgpu.v_max_f32",
    ):
        descriptor = overlays_by_key[descriptor_key]
        assert tuple(
            form.replacement_descriptor for form in descriptor.operand_forms
        ) == (
            f"{descriptor_key}.src0_inline",
            f"{descriptor_key}.lit",
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

    _expect_value_error_contains(
        "no address-unit encoding",
        lambda: _validate_address_immediate_units(_descriptor_set(descriptor)),
    )


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

    _expect_value_error_contains(
        "inconsistent split address offset units",
        lambda: _validate_address_immediate_units(_descriptor_set(descriptor)),
    )


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


def test_plain_ds_memory_offsets_use_split_16_bit_byte_immediates() -> None:
    descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx11_core_overlays()
    }

    for descriptor_key in (
        "amdgpu.ds_read_b128",
        "amdgpu.ds_write_b128",
        "amdgpu.ds_add_u32",
        "amdgpu.ds_cmpst_rtn_b32",
        "amdgpu.ds_read_addtid_b32",
        "amdgpu.ds_write_addtid_b32",
    ):
        descriptor = descriptors[descriptor_key]
        assert len(descriptor.immediates) == 1
        immediate = descriptor.immediates[0]
        assert immediate.field_name == "offset"
        assert immediate.bit_width == 16
        assert immediate.encoding_id == _ADDRESS_OFFSET_DS16_ENCODING_ID
        assert immediate.unsigned_max == 65535
        assert tuple(
            (encoding_slice.source_bit_offset, encoding_slice.bit_count)
            for encoding_slice in immediate.encoding_slices
        ) == ((0, 8), (8, 8))


def test_global_vaddr_memory_forms_have_unique_low_asm_mnemonics() -> None:
    descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx11_core_overlays()
    }

    load_forms = descriptors["amdgpu.global_load_b128"].asm_forms
    assert load_forms is not None
    load_form = load_forms[0]
    assert load_form.mnemonic == "global_load_b128_vaddr"
    assert load_form.results == ("dst",)
    assert load_form.operands == ("addr",)
    assert tuple(immediate.name for immediate in load_form.immediates) == (
        "offset",
        "glc",
        "slc",
        "dlc",
    )

    store_forms = descriptors["amdgpu.global_store_b128"].asm_forms
    assert store_forms is not None
    store_form = store_forms[0]
    assert store_form.mnemonic == "global_store_b128_vaddr"
    assert store_form.results == ()
    assert store_form.operands == ("addr", "value")
    assert tuple(immediate.name for immediate in store_form.immediates) == (
        "offset",
        "glc",
        "slc",
        "dlc",
    )

    assert descriptors["amdgpu.global_load_b128_saddr"].asm_forms is None
    assert descriptors["amdgpu.global_store_b128_saddr"].asm_forms is None
