# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from loom.target.arch.amdgpu.descriptor_overlay import AmdgpuDescriptorOverlay
from loom.target.arch.amdgpu.descriptors import (
    _ADDRESS_OFFSET_DS16_ENCODING_ID,
    _ADDRESS_OFFSET_DWORD_ENCODING_ID,
    _ADDRESS_OFFSET_DWORD_STRIDE64_ENCODING_ID,
    _AMDGPU_TRANS_DESCRIPTOR_KEYS,
    _AMDGPU_TRANS_PROXY_LATENCY_CYCLES,
    _COUNTER_VMEM_LOAD,
    _GFX12_TH_ATOMIC_RETURN_VALUE,
    _REG_EXEC,
    _REG_MODE,
    _SCHEDULE_MODE_CONTROL,
    _SCHEDULE_SALU,
    _SCHEDULE_SMEM_STORE,
    _SCHEDULE_VALU,
    _SCHEDULE_VMEM_LOAD_LDS,
    _SOURCE_INLINE_F32_ENCODING_ID,
    _SOURCE_INLINE_U32_ENCODING_ID,
    AMDGPU_ATOMIC_DESCRIPTOR_CATEGORY,
    AMDGPU_COMPARE_SELECT_DESCRIPTOR_CATEGORY,
    AMDGPU_CONTROL_DESCRIPTOR_CATEGORY,
    AMDGPU_DESCRIPTOR_CATEGORIES,
    AMDGPU_ENCODING_FORMAT_VOP1,
    AMDGPU_ENCODING_FORMAT_VOP2,
    AMDGPU_MEMORY_DESCRIPTOR_CATEGORY,
    AMDGPU_VECTOR_DESCRIPTOR_CATEGORY,
    AmdgpuAtomicKind,
    AmdgpuAtomicMemorySpace,
    AmdgpuAtomicOperationKind,
    AmdgpuAtomicValueKind,
    AmdgpuMemoryAddressForm,
    _amdgpu_core_descriptor_set_bases,
    _amdgpu_trans_schedule_class_name,
    _amdgpu_trans_schedule_classes,
    _categorize_amdgpu_descriptors,
    _gfx11_core_overlays,
    _gfx12_core_overlays,
    _gfx125x_reg_classes,
    _gfx940_core_overlays,
    _gfx950_core_overlays,
    _gfx1250_core_overlays,
    _validate_address_immediate_units,
    _with_execution_mask_state_read,
    _with_gfx125x_vgpr_msb_address_state,
    _with_gfx125x_vgpr_msb_address_states,
    _with_mode_state_read,
    amdgpu_atomic_descriptor_candidates,
    amdgpu_descriptor_category_groups,
    amdgpu_encoding_field_id,
)
from loom.target.arch.amdgpu.descriptors.control import _s_set_vgpr_msb_descriptor
from loom.target.arch.amdgpu.descriptors.memory import (
    _s_buffer_load_64_overlay,
    _s_buffer_load_dword_overlay,
    _s_load_dword_overlay,
    _s_load_dwordx2_overlay,
    _s_load_dwordx4_overlay,
)
from loom.target.low_descriptors import (
    ConstraintKind,
    Descriptor,
    DescriptorSet,
    Effect,
    EffectFlag,
    EffectKind,
    Immediate,
    ImmediateKind,
    LatencyKind,
    MemorySpace,
    Operand,
    OperandAddressMapKind,
    OperandFlag,
    OperandRole,
    RegClassAlt,
    RegClassAltFlag,
    RegClassFlag,
)


def _descriptor_set(*descriptors: Descriptor) -> DescriptorSet:
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
        descriptors=descriptors,
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


def test_trans_descriptors_use_descriptor_specific_schedule_classes() -> None:
    overlays = {
        overlay.descriptor_key: overlay
        for overlay in _gfx11_core_overlays()
        if overlay.descriptor_key in _AMDGPU_TRANS_DESCRIPTOR_KEYS
    }

    assert tuple(overlays) == _AMDGPU_TRANS_DESCRIPTOR_KEYS
    for descriptor_key, overlay in overlays.items():
        assert overlay.schedule_class == _amdgpu_trans_schedule_class_name(
            descriptor_key
        )
        assert overlay.schedule_class != _SCHEDULE_VALU

    for descriptor_set in _amdgpu_core_descriptor_set_bases():
        schedule_classes = {
            schedule_class.name: schedule_class
            for schedule_class in descriptor_set.schedule_classes
        }
        for descriptor_key in _AMDGPU_TRANS_DESCRIPTOR_KEYS:
            schedule_class = schedule_classes[
                _amdgpu_trans_schedule_class_name(descriptor_key)
            ]
            assert schedule_class.latency_kind is LatencyKind.ESTIMATE
            assert schedule_class.latency_cycles == _AMDGPU_TRANS_PROXY_LATENCY_CYCLES


def test_trans_schedule_classes_accept_descriptor_latency_overrides() -> None:
    descriptor_key = "amdgpu.v_rcp_f32"
    schedule_classes = {
        schedule_class.name: schedule_class
        for schedule_class in _amdgpu_trans_schedule_classes(
            latency_cycles_by_descriptor_key={descriptor_key: 5}
        )
    }

    override_schedule_class = schedule_classes[
        _amdgpu_trans_schedule_class_name(descriptor_key)
    ]
    assert override_schedule_class.latency_cycles == 5
    assert (
        schedule_classes[
            _amdgpu_trans_schedule_class_name("amdgpu.v_exp_f32")
        ].latency_cycles
        == _AMDGPU_TRANS_PROXY_LATENCY_CYCLES
    )
    _expect_value_error_contains(
        "unknown descriptor",
        lambda: _amdgpu_trans_schedule_classes(
            latency_cycles_by_descriptor_key={"amdgpu.v_bad_f32": 4}
        ),
    )


def test_trans_descriptors_read_exec_state() -> None:
    descriptor = Descriptor(
        key="amdgpu.v_exp_f32",
        mnemonic="v_exp_f32",
        semantic_tag="float.exp2.f32",
        operands=(
            Operand("dst", OperandRole.RESULT, (RegClassAlt("amdgpu.vgpr"),)),
            Operand("input", OperandRole.OPERAND, (RegClassAlt("amdgpu.vgpr"),)),
        ),
        schedule_class=_amdgpu_trans_schedule_class_name("amdgpu.v_exp_f32"),
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


def test_gfx125x_register_classes_expose_mode_and_high_vgprs() -> None:
    reg_classes = {reg_class.name: reg_class for reg_class in _gfx125x_reg_classes()}

    assert reg_classes["amdgpu.vgpr"].allocatable_count == 1024
    mode = reg_classes[_REG_MODE]
    assert mode.alloc_unit_bits == 32
    assert mode.allocatable_count == 1
    assert RegClassFlag.PHYSICAL in mode.flags
    assert RegClassFlag.UNSPILLABLE in mode.flags


def test_s_set_vgpr_msb_writes_mode_state() -> None:
    descriptor = _s_set_vgpr_msb_descriptor()

    assert descriptor.key == "amdgpu.s_set_vgpr_msb"
    assert descriptor.mnemonic == "s_set_vgpr_msb"
    assert descriptor.schedule_class == _SCHEDULE_MODE_CONTROL
    assert tuple(immediate.field_name for immediate in descriptor.immediates) == (
        "mode",
    )
    mode_operand = descriptor.operands[0]
    assert mode_operand.field_name == "mode"
    assert mode_operand.role is OperandRole.IMPLICIT
    assert mode_operand.reg_alts == (
        RegClassAlt(_REG_MODE, flags=(RegClassAltFlag.PHYSICAL_ONLY,)),
    )
    assert OperandFlag.IMPLICIT in mode_operand.flags
    assert OperandFlag.STATE_WRITE in mode_operand.flags


def test_gfx125x_vop_operands_use_mode_address_state() -> None:
    descriptor = Descriptor(
        key="amdgpu.test.v_add_u32",
        mnemonic="v_add_u32",
        semantic_tag="test.v_add_u32",
        operands=(
            Operand(
                "dst",
                OperandRole.RESULT,
                (RegClassAlt("amdgpu.vgpr"),),
                encoding_field_id=amdgpu_encoding_field_id("VDST"),
            ),
            Operand(
                "lhs",
                OperandRole.OPERAND,
                (RegClassAlt("amdgpu.vgpr"),),
                encoding_field_id=amdgpu_encoding_field_id("SRC0"),
            ),
            Operand(
                "rhs",
                OperandRole.OPERAND,
                (RegClassAlt("amdgpu.vgpr"),),
                encoding_field_id=amdgpu_encoding_field_id("VSRC1"),
            ),
        ),
        schedule_class=_SCHEDULE_VALU,
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP2,
    )
    descriptor = _with_gfx125x_vgpr_msb_address_state(descriptor)
    operands = {operand.field_name: operand for operand in descriptor.operands}

    for field_name in ("dst", "lhs", "rhs"):
        operand = operands[field_name]
        assert operand.address_map_kind is OperandAddressMapKind.TARGET_STATE
        assert operand.addressable_unit_count == 256

    mode_operand = operands["mode_in"]
    assert mode_operand.role is OperandRole.IMPLICIT
    assert mode_operand.reg_alts == (
        RegClassAlt(_REG_MODE, flags=(RegClassAltFlag.PHYSICAL_ONLY,)),
    )
    assert OperandFlag.IMPLICIT in mode_operand.flags
    assert OperandFlag.STATE_READ in mode_operand.flags


def test_gfx125x_uncontrolled_vgpr_operands_use_low_subset() -> None:
    descriptor = Descriptor(
        key="amdgpu.test.uncontrolled_vgpr",
        mnemonic="uncontrolled_vgpr",
        semantic_tag="test.uncontrolled_vgpr",
        operands=(
            Operand(
                "scale_src0",
                OperandRole.OPERAND,
                (RegClassAlt("amdgpu.vgpr"),),
                encoding_field_id=amdgpu_encoding_field_id("SDST"),
            ),
        ),
        schedule_class=_SCHEDULE_VALU,
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP1,
    )
    descriptor = _with_gfx125x_vgpr_msb_address_state(descriptor)
    operands = {operand.field_name: operand for operand in descriptor.operands}

    for field_name in ("scale_src0",):
        operand = operands[field_name]
        assert operand.address_map_kind is OperandAddressMapKind.LOW_SUBSET
        assert operand.addressable_unit_count == 256


def test_gfx125x_target_state_validation_requires_mode_control_descriptor() -> None:
    descriptor = Descriptor(
        key="amdgpu.test.v_mov",
        mnemonic="v_mov_b32",
        semantic_tag="test.v_mov",
        operands=(
            Operand(
                "dst",
                OperandRole.RESULT,
                (RegClassAlt("amdgpu.vgpr"),),
                encoding_field_id=amdgpu_encoding_field_id("VDST"),
            ),
        ),
        schedule_class=_SCHEDULE_VALU,
        encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP1,
    )

    _expect_value_error_contains(
        "amdgpu.s_set_vgpr_msb",
        lambda: _with_gfx125x_vgpr_msb_address_states(_descriptor_set(descriptor)),
    )


def test_gfx125x_target_state_validation_requires_encoding_slot() -> None:
    descriptor = _with_mode_state_read(
        Descriptor(
            key="amdgpu.test.bad_vgpr_msb_slot",
            mnemonic="bad_vgpr_msb_slot",
            semantic_tag="test.bad_vgpr_msb_slot",
            operands=(
                Operand(
                    "dst",
                    OperandRole.RESULT,
                    (RegClassAlt("amdgpu.vgpr"),),
                    encoding_field_id=amdgpu_encoding_field_id("SDST"),
                    address_map_kind=OperandAddressMapKind.TARGET_STATE,
                    addressable_unit_count=256,
                ),
            ),
            schedule_class=_SCHEDULE_VALU,
            encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP1,
        )
    )

    _expect_value_error_contains(
        "has no S_SET_VGPR_MSB slot",
        lambda: _with_gfx125x_vgpr_msb_address_states(
            _descriptor_set(_s_set_vgpr_msb_descriptor(), descriptor)
        ),
    )


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


def test_scalar_memory_loads_early_clobber_results() -> None:
    for descriptor in (
        _s_buffer_load_dword_overlay(),
        _s_buffer_load_64_overlay(),
        _s_load_dword_overlay(),
        _s_load_dwordx2_overlay(),
        _s_load_dwordx4_overlay(),
    ):
        assert tuple(constraint.kind for constraint in descriptor.constraints) == (
            ConstraintKind.EARLY_CLOBBER,
        )
        assert tuple(
            constraint.lhs_operand_index for constraint in descriptor.constraints
        ) == (0,)


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

    saddr_load_forms = descriptors["amdgpu.global_load_b128_saddr"].asm_forms
    assert saddr_load_forms is not None
    saddr_load_form = saddr_load_forms[0]
    assert saddr_load_form.mnemonic == "global_load_b128_saddr"
    assert saddr_load_form.results == ("dst",)
    assert saddr_load_form.operands == ("addr", "saddr")
    assert tuple(immediate.name for immediate in saddr_load_form.immediates) == (
        "offset",
        "glc",
        "slc",
        "dlc",
    )

    saddr_store_forms = descriptors["amdgpu.global_store_b128_saddr"].asm_forms
    assert saddr_store_forms is not None
    saddr_store_form = saddr_store_forms[0]
    assert saddr_store_form.mnemonic == "global_store_b128_saddr"
    assert saddr_store_form.results == ()
    assert saddr_store_form.operands == ("addr", "value", "saddr")
    assert tuple(immediate.name for immediate in saddr_store_form.immediates) == (
        "offset",
        "glc",
        "slc",
        "dlc",
    )


def test_gfx950_global_saddr_memory_asm_forms_include_m0() -> None:
    descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx950_core_overlays()
    }

    load_forms = descriptors["amdgpu.global_load_b128_saddr"].asm_forms
    assert load_forms is not None
    load_form = load_forms[0]
    assert load_form.mnemonic == "global_load_dwordx4_saddr"
    assert load_form.results == ("dst",)
    assert load_form.operands == ("addr", "saddr", "m0")
    assert tuple(immediate.name for immediate in load_form.immediates) == (
        "offset",
        "nt",
        "sc0",
        "sc1",
    )

    store_forms = descriptors["amdgpu.global_store_b128_saddr"].asm_forms
    assert store_forms is not None
    store_form = store_forms[0]
    assert store_form.mnemonic == "global_store_dwordx4_saddr"
    assert store_form.results == ()
    assert store_form.operands == ("addr", "value", "saddr", "m0")
    assert tuple(immediate.name for immediate in store_form.immediates) == (
        "offset",
        "nt",
        "sc0",
        "sc1",
    )


def _assert_memory_width_overlay(
    descriptor: AmdgpuDescriptorOverlay,
    *,
    width_bits: int,
    semantic_tag: str,
    mnemonic: str,
    operand_units: int,
    payload_field_name: str,
    effect_kind: EffectKind,
    memory_space: MemorySpace,
    implicit_data_format: str | None = None,
    implicit_ignore_reason: str | None = None,
) -> None:
    assert descriptor.semantic_tag == semantic_tag
    assert descriptor.mnemonic == mnemonic
    payload_operand = next(
        operand.descriptor_operand
        for operand in descriptor.operands
        if operand.descriptor_operand.field_name == payload_field_name
    )
    assert payload_operand.unit_count == operand_units
    assert descriptor.effects == (
        Effect(
            effect_kind,
            memory_space=memory_space,
            flags=(EffectFlag.DEPENDENCY,),
            width_bits=width_bits,
        ),
    )
    assert any(
        operand.operand_type == "OPR_GPUMEM"
        and operand.data_format_name
        == (implicit_data_format or f"FMT_NUM_B{width_bits}")
        and operand.size_bits == width_bits
        and (
            implicit_ignore_reason is None
            or operand.ignore_reason == implicit_ignore_reason
        )
        for operand in descriptor.implicit_operands
    )


def _assert_global_load_lds_overlay(
    descriptor: AmdgpuDescriptorOverlay,
    *,
    global_width_bits: int,
    workgroup_width_bits: int,
    semantic_tag: str,
    mnemonic: str,
) -> None:
    assert descriptor.semantic_tag == semantic_tag
    assert descriptor.mnemonic == mnemonic
    assert descriptor.schedule_class == _SCHEDULE_VMEM_LOAD_LDS
    assert descriptor.effects == (
        Effect(
            EffectKind.READ,
            memory_space=MemorySpace.GLOBAL,
            flags=(EffectFlag.DEPENDENCY,),
            counter_id=_COUNTER_VMEM_LOAD,
            width_bits=global_width_bits,
        ),
        Effect(
            EffectKind.WRITE,
            memory_space=MemorySpace.WORKGROUP,
            flags=(EffectFlag.DEPENDENCY,),
            counter_id=_COUNTER_VMEM_LOAD,
            width_bits=workgroup_width_bits,
        ),
    )
    assert any(
        operand.xml_field_name == "VDST"
        and operand.ignore_reason == "legacy-lds-dma-has-no-vgpr-result"
        for operand in descriptor.ignored_operands
    )
    assert any(
        operand.operand_type == "OPR_SDST_M0"
        and operand.descriptor_operand is not None
        and operand.descriptor_operand.field_name == "m0"
        and OperandFlag.IMPLICIT in operand.descriptor_operand.flags
        and OperandFlag.STATE_READ in operand.descriptor_operand.flags
        for operand in descriptor.implicit_operands
    )


def _assert_buffer_load_lds_overlay(
    descriptor: AmdgpuDescriptorOverlay,
    *,
    global_width_bits: int,
    workgroup_width_bits: int,
    semantic_tag: str,
    mnemonic: str,
    implicit_data_format: str,
    off_zero: bool,
) -> None:
    assert descriptor.semantic_tag == semantic_tag
    assert descriptor.mnemonic == mnemonic
    assert descriptor.schedule_class == _SCHEDULE_VMEM_LOAD_LDS
    assert descriptor.effects == (
        Effect(
            EffectKind.READ,
            memory_space=MemorySpace.GLOBAL,
            flags=(EffectFlag.DEPENDENCY,),
            counter_id=_COUNTER_VMEM_LOAD,
            width_bits=global_width_bits,
        ),
        Effect(
            EffectKind.WRITE,
            memory_space=MemorySpace.WORKGROUP,
            flags=(EffectFlag.DEPENDENCY,),
            counter_id=_COUNTER_VMEM_LOAD,
            width_bits=workgroup_width_bits,
        ),
    )
    assert ("LDS", 1) in descriptor.fixed_encoding_fields
    assert ("IDXEN", 0) in descriptor.fixed_encoding_fields
    assert ("OFFEN", 0 if off_zero else 1) in descriptor.fixed_encoding_fields
    assert any(
        operand.xml_field_name == "VDATA"
        and operand.ignore_reason == "lds-bit-has-no-vgpr-result"
        for operand in descriptor.ignored_operands
    )
    assert any(
        operand.operand_type == "OPR_SDST_M0"
        and operand.descriptor_operand is not None
        and operand.descriptor_operand.field_name == "m0"
        and OperandFlag.IMPLICIT in operand.descriptor_operand.flags
        and OperandFlag.STATE_READ in operand.descriptor_operand.flags
        and not operand.xml_operand_required
        for operand in descriptor.implicit_operands
    )
    assert any(
        operand.operand_type == "OPR_GPUMEM"
        and operand.data_format_name == implicit_data_format
        and operand.size_bits == global_width_bits
        and operand.ignore_reason == "modeled-by-global-read-effect"
        and operand.xml_operand_required
        for operand in descriptor.implicit_operands
    )
    assert len(descriptor.operands) == (1 if off_zero else 3)


def test_dwordx3_memory_descriptors_cover_cdna_and_rdna_families() -> None:
    for (
        descriptors,
        buffer_load_key,
        buffer_store_key,
        buffer_mnemonic,
        global_mnemonic,
    ) in (
        (
            {
                descriptor.descriptor_key: descriptor
                for descriptor in _gfx940_core_overlays()
            },
            "amdgpu.buffer_load_dwordx3",
            "amdgpu.buffer_store_dwordx3",
            "buffer_load_dwordx3",
            "global_load_dwordx3_saddr",
        ),
        (
            {
                descriptor.descriptor_key: descriptor
                for descriptor in _gfx950_core_overlays()
            },
            "amdgpu.buffer_load_dwordx3",
            "amdgpu.buffer_store_dwordx3",
            "buffer_load_dwordx3",
            "global_load_dwordx3_saddr",
        ),
        (
            {
                descriptor.descriptor_key: descriptor
                for descriptor in _gfx11_core_overlays()
            },
            "amdgpu.buffer_load_b96",
            "amdgpu.buffer_store_b96",
            "buffer_load_b96",
            "global_load_b96_saddr",
        ),
        (
            {
                descriptor.descriptor_key: descriptor
                for descriptor in _gfx12_core_overlays()
            },
            "amdgpu.buffer_load_b96",
            "amdgpu.buffer_store_b96",
            "buffer_load_b96",
            "global_load_b96_saddr",
        ),
        (
            {
                descriptor.descriptor_key: descriptor
                for descriptor in _gfx1250_core_overlays()
            },
            "amdgpu.buffer_load_b96",
            "amdgpu.buffer_store_b96",
            "buffer_load_b96",
            "global_load_b96_saddr",
        ),
    ):
        _assert_memory_width_overlay(
            descriptors[buffer_load_key],
            width_bits=96,
            semantic_tag="memory.load.u96",
            mnemonic=buffer_mnemonic,
            operand_units=3,
            payload_field_name="dst",
            effect_kind=EffectKind.READ,
            memory_space=MemorySpace.GLOBAL,
        )
        _assert_memory_width_overlay(
            descriptors[buffer_store_key],
            width_bits=96,
            semantic_tag="memory.store.u96",
            mnemonic=buffer_mnemonic.replace("load", "store"),
            operand_units=3,
            payload_field_name="value",
            effect_kind=EffectKind.WRITE,
            memory_space=MemorySpace.GLOBAL,
        )

        global_load = descriptors["amdgpu.global_load_b96_saddr"]
        _assert_memory_width_overlay(
            global_load,
            width_bits=96,
            semantic_tag="memory.load.u96",
            mnemonic=global_mnemonic.removesuffix("_saddr"),
            operand_units=3,
            payload_field_name="dst",
            effect_kind=EffectKind.READ,
            memory_space=MemorySpace.GLOBAL,
        )
        assert global_load.asm_forms is not None
        assert global_load.asm_forms[0].mnemonic == global_mnemonic

        global_store = descriptors["amdgpu.global_store_b96_saddr"]
        _assert_memory_width_overlay(
            global_store,
            width_bits=96,
            semantic_tag="memory.store.u96",
            mnemonic=global_mnemonic.removesuffix("_saddr").replace("load", "store"),
            operand_units=3,
            payload_field_name="value",
            effect_kind=EffectKind.WRITE,
            memory_space=MemorySpace.GLOBAL,
        )
        assert global_store.asm_forms is not None
        assert global_store.asm_forms[0].mnemonic == global_mnemonic.replace(
            "load", "store"
        )

        scratch_load = descriptors["amdgpu.scratch_load_b96_vaddr"]
        _assert_memory_width_overlay(
            scratch_load,
            width_bits=96,
            semantic_tag="memory.stack.load.u96",
            mnemonic="scratch_load_b96",
            operand_units=3,
            payload_field_name="dst",
            effect_kind=EffectKind.READ,
            memory_space=MemorySpace.STACK,
        )
        assert scratch_load.asm_forms is not None
        assert scratch_load.asm_forms[0].mnemonic == "scratch_load_b96_vaddr"

        scratch_store = descriptors["amdgpu.scratch_store_b96_vaddr"]
        _assert_memory_width_overlay(
            scratch_store,
            width_bits=96,
            semantic_tag="memory.stack.store.u96",
            mnemonic="scratch_store_b96",
            operand_units=3,
            payload_field_name="value",
            effect_kind=EffectKind.WRITE,
            memory_space=MemorySpace.STACK,
        )
        assert scratch_store.asm_forms is not None
        assert scratch_store.asm_forms[0].mnemonic == "scratch_store_b96_vaddr"


def test_cdna_global_load_lds_descriptors_cover_extension_rows() -> None:
    cdna3_descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx940_core_overlays()
    }
    cdna4_descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx950_core_overlays()
    }
    base_rows = (
        (
            "ubyte",
            "memory.global_to_workgroup.u8.zero_extend",
            8,
            32,
        ),
        (
            "sbyte",
            "memory.global_to_workgroup.i8.sign_extend",
            8,
            32,
        ),
        (
            "ushort",
            "memory.global_to_workgroup.u16.zero_extend",
            16,
            32,
        ),
        (
            "sshort",
            "memory.global_to_workgroup.i16.sign_extend",
            16,
            32,
        ),
        ("dword", "memory.global_to_workgroup.u32", 32, 32),
    )
    for descriptors in (cdna3_descriptors, cdna4_descriptors):
        for suffix, semantic_tag, global_width_bits, workgroup_width_bits in base_rows:
            for descriptor_key_suffix in ("", "_saddr"):
                descriptor_key = f"amdgpu.global_load_lds_{suffix}"
                _assert_global_load_lds_overlay(
                    descriptors[f"{descriptor_key}{descriptor_key_suffix}"],
                    global_width_bits=global_width_bits,
                    workgroup_width_bits=workgroup_width_bits,
                    semantic_tag=semantic_tag,
                    mnemonic=f"global_load_lds_{suffix}",
                )

    for suffix, width_bits in (("dwordx3", 96), ("dwordx4", 128)):
        for descriptor_key_suffix in ("", "_saddr"):
            descriptor_key = f"amdgpu.global_load_lds_{suffix}"
            assert f"{descriptor_key}{descriptor_key_suffix}" not in cdna3_descriptors
            _assert_global_load_lds_overlay(
                cdna4_descriptors[f"{descriptor_key}{descriptor_key_suffix}"],
                global_width_bits=width_bits,
                workgroup_width_bits=width_bits,
                semantic_tag=f"memory.global_to_workgroup.u{width_bits}",
                mnemonic=f"global_load_lds_{suffix}",
            )


def test_cdna_buffer_load_lds_descriptors_cover_fixed_lds_rows() -> None:
    cdna3_descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx940_core_overlays()
    }
    cdna4_descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx950_core_overlays()
    }
    base_rows = (
        (
            "ubyte",
            "memory.global_to_workgroup.u8.zero_extend",
            8,
            8,
            "FMT_NUM_U8",
        ),
        (
            "sbyte",
            "memory.global_to_workgroup.i8.sign_extend",
            8,
            8,
            "FMT_NUM_I8",
        ),
        (
            "ushort",
            "memory.global_to_workgroup.u16.zero_extend",
            16,
            16,
            "FMT_NUM_U16",
        ),
        (
            "sshort",
            "memory.global_to_workgroup.i16.sign_extend",
            16,
            16,
            "FMT_NUM_I16",
        ),
        ("dword", "memory.global_to_workgroup.u32", 32, 32, "FMT_NUM_B32"),
    )
    for descriptors in (cdna3_descriptors, cdna4_descriptors):
        for (
            suffix,
            semantic_tag,
            global_width_bits,
            workgroup_width_bits,
            implicit_data_format,
        ) in base_rows:
            descriptor_key = f"amdgpu.buffer_load_lds_{suffix}"
            _assert_buffer_load_lds_overlay(
                descriptors[descriptor_key],
                global_width_bits=global_width_bits,
                workgroup_width_bits=workgroup_width_bits,
                semantic_tag=semantic_tag,
                mnemonic=f"buffer_load_{suffix}",
                implicit_data_format=implicit_data_format,
                off_zero=False,
            )
            _assert_buffer_load_lds_overlay(
                descriptors[f"{descriptor_key}_off_zero"],
                global_width_bits=global_width_bits,
                workgroup_width_bits=workgroup_width_bits,
                semantic_tag=semantic_tag,
                mnemonic=f"buffer_load_{suffix}",
                implicit_data_format=implicit_data_format,
                off_zero=True,
            )

    for descriptors in (cdna3_descriptors, cdna4_descriptors):
        assert "amdgpu.buffer_load_lds_dwordx2" not in descriptors
        assert "amdgpu.buffer_load_lds_dwordx2_off_zero" not in descriptors

    for suffix, width_bits in (("dwordx3", 96), ("dwordx4", 128)):
        descriptor_key = f"amdgpu.buffer_load_lds_{suffix}"
        assert descriptor_key not in cdna3_descriptors
        assert f"{descriptor_key}_off_zero" not in cdna3_descriptors
        _assert_buffer_load_lds_overlay(
            cdna4_descriptors[descriptor_key],
            global_width_bits=width_bits,
            workgroup_width_bits=width_bits,
            semantic_tag=f"memory.global_to_workgroup.u{width_bits}",
            mnemonic=f"buffer_load_{suffix}",
            implicit_data_format=f"FMT_NUM_B{width_bits}",
            off_zero=False,
        )
        _assert_buffer_load_lds_overlay(
            cdna4_descriptors[f"{descriptor_key}_off_zero"],
            global_width_bits=width_bits,
            workgroup_width_bits=width_bits,
            semantic_tag=f"memory.global_to_workgroup.u{width_bits}",
            mnemonic=f"buffer_load_{suffix}",
            implicit_data_format=f"FMT_NUM_B{width_bits}",
            off_zero=True,
        )

    for descriptors in (
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx11_core_overlays()
        },
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx12_core_overlays()
        },
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx1250_core_overlays()
        },
    ):
        assert "amdgpu.buffer_load_lds_dword" not in descriptors


def test_smem_dword_width_descriptors_cover_active_xml_families() -> None:
    for descriptors in (
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx940_core_overlays()
        },
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx950_core_overlays()
        },
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx11_core_overlays()
        },
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx12_core_overlays()
        },
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx1250_core_overlays()
        },
    ):
        for width_bits, units, descriptor_key, mnemonic in (
            (256, 8, "amdgpu.s_load_dwordx8", "s_load_dwordx8"),
            (512, 16, "amdgpu.s_load_dwordx16", "s_load_dwordx16"),
        ):
            _assert_memory_width_overlay(
                descriptors[descriptor_key],
                width_bits=width_bits,
                semantic_tag=f"memory.load.u{width_bits}",
                mnemonic=mnemonic,
                operand_units=units,
                payload_field_name="dst",
                effect_kind=EffectKind.READ,
                memory_space=MemorySpace.GLOBAL,
            )
            assert f"{descriptor_key}_offset_only" in descriptors

    for descriptors in (
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx940_core_overlays()
        },
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx950_core_overlays()
        },
    ):
        for width_bits, units, descriptor_key, mnemonic in (
            (128, 4, "amdgpu.s_buffer_load_dwordx4", "s_buffer_load_dwordx4"),
            (256, 8, "amdgpu.s_buffer_load_dwordx8", "s_buffer_load_dwordx8"),
            (512, 16, "amdgpu.s_buffer_load_dwordx16", "s_buffer_load_dwordx16"),
        ):
            _assert_memory_width_overlay(
                descriptors[descriptor_key],
                width_bits=width_bits,
                semantic_tag=f"memory.load.u{width_bits}",
                mnemonic=mnemonic,
                operand_units=units,
                payload_field_name="dst",
                effect_kind=EffectKind.READ,
                memory_space=MemorySpace.GLOBAL,
            )

    for descriptors in (
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx11_core_overlays()
        },
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx12_core_overlays()
        },
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx1250_core_overlays()
        },
    ):
        for width_bits, units, descriptor_key, mnemonic in (
            (128, 4, "amdgpu.s_buffer_load_b128", "s_buffer_load_b128"),
            (256, 8, "amdgpu.s_buffer_load_b256", "s_buffer_load_b256"),
            (512, 16, "amdgpu.s_buffer_load_b512", "s_buffer_load_b512"),
        ):
            _assert_memory_width_overlay(
                descriptors[descriptor_key],
                width_bits=width_bits,
                semantic_tag=f"memory.load.u{width_bits}",
                mnemonic=mnemonic,
                operand_units=units,
                payload_field_name="dst",
                effect_kind=EffectKind.READ,
                memory_space=MemorySpace.GLOBAL,
            )

    for descriptors in (
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx12_core_overlays()
        },
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx1250_core_overlays()
        },
    ):
        for descriptor_key, mnemonic in (
            ("amdgpu.s_load_b96", "s_load_b96"),
            ("amdgpu.s_buffer_load_b96", "s_buffer_load_b96"),
        ):
            _assert_memory_width_overlay(
                descriptors[descriptor_key],
                width_bits=96,
                semantic_tag="memory.load.u96",
                mnemonic=mnemonic,
                operand_units=3,
                payload_field_name="dst",
                effect_kind=EffectKind.READ,
                memory_space=MemorySpace.GLOBAL,
            )


def test_rdna4_smem_narrow_load_descriptors_have_extension_semantics() -> None:
    gfx11_descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx11_core_overlays()
    }
    rows = (
        ("i8", 8, "memory.load.i8.sign_extend", "FMT_NUM_I8"),
        ("u8", 8, "memory.load.u8.zero_extend", "FMT_NUM_U8"),
        ("i16", 16, "memory.load.i16.sign_extend", "FMT_NUM_I16"),
        ("u16", 16, "memory.load.u16.zero_extend", "FMT_NUM_U16"),
    )
    for suffix, _width_bits, _semantic_tag, _implicit_data_format in rows:
        assert f"amdgpu.s_load_{suffix}" not in gfx11_descriptors
        assert f"amdgpu.s_buffer_load_{suffix}" not in gfx11_descriptors

    for descriptors in (
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx12_core_overlays()
        },
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx1250_core_overlays()
        },
    ):
        for suffix, width_bits, semantic_tag, implicit_data_format in rows:
            scalar_load_key = f"amdgpu.s_load_{suffix}"
            _assert_memory_width_overlay(
                descriptors[scalar_load_key],
                width_bits=width_bits,
                semantic_tag=semantic_tag,
                mnemonic=f"s_load_{suffix}",
                operand_units=1,
                payload_field_name="dst",
                effect_kind=EffectKind.READ,
                memory_space=MemorySpace.GLOBAL,
                implicit_data_format=implicit_data_format,
            )
            _assert_memory_width_overlay(
                descriptors[f"{scalar_load_key}_offset_only"],
                width_bits=width_bits,
                semantic_tag=semantic_tag,
                mnemonic=f"s_load_{suffix}",
                operand_units=1,
                payload_field_name="dst",
                effect_kind=EffectKind.READ,
                memory_space=MemorySpace.GLOBAL,
                implicit_data_format=implicit_data_format,
            )
            _assert_memory_width_overlay(
                descriptors[f"amdgpu.s_buffer_load_{suffix}"],
                width_bits=width_bits,
                semantic_tag=semantic_tag,
                mnemonic=f"s_buffer_load_{suffix}",
                operand_units=1,
                payload_field_name="dst",
                effect_kind=EffectKind.READ,
                memory_space=MemorySpace.GLOBAL,
                implicit_data_format=implicit_data_format,
            )


def test_cdna_smem_dwordx4_store_and_scratch_descriptors_cover_xml() -> None:
    rows = (
        (32, 1, "dword"),
        (64, 2, "dwordx2"),
        (128, 4, "dwordx4"),
    )
    for rdna_overlays in (
        _gfx11_core_overlays(),
        _gfx12_core_overlays(),
        _gfx1250_core_overlays(),
    ):
        rdna_descriptors = {
            descriptor.descriptor_key: descriptor for descriptor in rdna_overlays
        }
        for _width_bits, _units, suffix in rows:
            assert f"amdgpu.s_store_{suffix}" not in rdna_descriptors
            assert f"amdgpu.s_scratch_load_{suffix}" not in rdna_descriptors
            assert f"amdgpu.s_scratch_store_{suffix}" not in rdna_descriptors
            assert f"amdgpu.s_buffer_store_{suffix}" not in rdna_descriptors

    for descriptors in (
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx940_core_overlays()
        },
        {
            descriptor.descriptor_key: descriptor
            for descriptor in _gfx950_core_overlays()
        },
    ):
        for width_bits, units, suffix in rows:
            store_key = f"amdgpu.s_store_{suffix}"
            _assert_memory_width_overlay(
                descriptors[store_key],
                width_bits=width_bits,
                semantic_tag=f"memory.store.u{width_bits}",
                mnemonic=f"s_store_{suffix}",
                operand_units=units,
                payload_field_name="value",
                effect_kind=EffectKind.WRITE,
                memory_space=MemorySpace.GLOBAL,
            )
            assert descriptors[store_key].schedule_class == _SCHEDULE_SMEM_STORE
            _assert_memory_width_overlay(
                descriptors[f"{store_key}_offset_only"],
                width_bits=width_bits,
                semantic_tag=f"memory.store.u{width_bits}",
                mnemonic=f"s_store_{suffix}",
                operand_units=units,
                payload_field_name="value",
                effect_kind=EffectKind.WRITE,
                memory_space=MemorySpace.GLOBAL,
            )

            scratch_load_key = f"amdgpu.s_scratch_load_{suffix}"
            _assert_memory_width_overlay(
                descriptors[scratch_load_key],
                width_bits=width_bits,
                semantic_tag=f"memory.stack.load.u{width_bits}",
                mnemonic=f"s_scratch_load_{suffix}",
                operand_units=units,
                payload_field_name="dst",
                effect_kind=EffectKind.READ,
                memory_space=MemorySpace.STACK,
                implicit_ignore_reason="modeled-by-stack-read-effect",
            )
            _assert_memory_width_overlay(
                descriptors[f"{scratch_load_key}_offset_only"],
                width_bits=width_bits,
                semantic_tag=f"memory.stack.load.u{width_bits}",
                mnemonic=f"s_scratch_load_{suffix}",
                operand_units=units,
                payload_field_name="dst",
                effect_kind=EffectKind.READ,
                memory_space=MemorySpace.STACK,
                implicit_ignore_reason="modeled-by-stack-read-effect",
            )

            scratch_store_key = f"amdgpu.s_scratch_store_{suffix}"
            _assert_memory_width_overlay(
                descriptors[scratch_store_key],
                width_bits=width_bits,
                semantic_tag=f"memory.stack.store.u{width_bits}",
                mnemonic=f"s_scratch_store_{suffix}",
                operand_units=units,
                payload_field_name="value",
                effect_kind=EffectKind.WRITE,
                memory_space=MemorySpace.STACK,
                implicit_ignore_reason="modeled-by-stack-write-effect",
            )
            assert descriptors[scratch_store_key].schedule_class == _SCHEDULE_SMEM_STORE
            _assert_memory_width_overlay(
                descriptors[f"{scratch_store_key}_offset_only"],
                width_bits=width_bits,
                semantic_tag=f"memory.stack.store.u{width_bits}",
                mnemonic=f"s_scratch_store_{suffix}",
                operand_units=units,
                payload_field_name="value",
                effect_kind=EffectKind.WRITE,
                memory_space=MemorySpace.STACK,
                implicit_ignore_reason="modeled-by-stack-write-effect",
            )

            buffer_store_key = f"amdgpu.s_buffer_store_{suffix}"
            _assert_memory_width_overlay(
                descriptors[buffer_store_key],
                width_bits=width_bits,
                semantic_tag=f"memory.store.u{width_bits}",
                mnemonic=f"s_buffer_store_{suffix}",
                operand_units=units,
                payload_field_name="value",
                effect_kind=EffectKind.WRITE,
                memory_space=MemorySpace.GLOBAL,
            )
            assert descriptors[buffer_store_key].schedule_class == _SCHEDULE_SMEM_STORE


def test_gfx940_scratch_memory_forms_cover_spill_packets() -> None:
    descriptors = {
        descriptor.descriptor_key: descriptor for descriptor in _gfx940_core_overlays()
    }

    load_descriptor = descriptors["amdgpu.scratch_load_b32_offset_only"]
    assert any(
        operand.operand_type == "OPR_SDST_M0"
        and operand.descriptor_operand is not None
        and operand.descriptor_operand.field_name == "m0"
        and operand.descriptor_operand.role is OperandRole.IMPLICIT
        for operand in load_descriptor.implicit_operands
    )
    load_forms = load_descriptor.asm_forms
    assert load_forms is not None
    load_form = load_forms[0]
    assert load_form.mnemonic == "scratch_load_b32_offset_only"
    assert load_form.results == ("dst",)
    assert load_form.operands == ()
    assert tuple(immediate.name for immediate in load_form.immediates) == (
        "offset",
        "nt",
        "sc0",
        "sc1",
    )

    store_descriptor = descriptors["amdgpu.scratch_store_b32_offset_only"]
    assert any(
        operand.operand_type == "OPR_SDST_M0"
        and operand.descriptor_operand is not None
        and operand.descriptor_operand.field_name == "m0"
        and operand.descriptor_operand.role is OperandRole.IMPLICIT
        for operand in store_descriptor.implicit_operands
    )
    store_forms = store_descriptor.asm_forms
    assert store_forms is not None
    store_form = store_forms[0]
    assert store_form.mnemonic == "scratch_store_b32_offset_only"
    assert store_form.results == ()
    assert store_form.operands == ("value",)
    assert tuple(immediate.name for immediate in store_form.immediates) == (
        "offset",
        "nt",
        "sc0",
        "sc1",
    )
