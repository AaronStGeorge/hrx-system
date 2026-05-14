# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""Public AMDGPU descriptor construction API."""

from __future__ import annotations

from .categories import *
from .common import *
from .sets import *


def _descriptor_has_memory_effect(descriptor: Descriptor) -> bool:
    return any(
        effect.kind in (EffectKind.READ, EffectKind.WRITE)
        and effect.memory_space in (MemorySpace.GLOBAL, MemorySpace.WORKGROUP)
        for effect in descriptor.effects
    )


def _descriptor_address_offset_immediates(
    descriptor: Descriptor,
) -> tuple[Immediate, ...]:
    return tuple(
        immediate
        for immediate in descriptor.immediates
        if immediate.field_name in _ADDRESS_OFFSET_IMMEDIATE_FIELD_NAMES
    )


def _validate_address_immediate_units(descriptor_set: DescriptorSet) -> None:
    for descriptor in descriptor_set.descriptors:
        if not _descriptor_has_memory_effect(descriptor):
            continue
        offset_immediates = _descriptor_address_offset_immediates(descriptor)
        if not offset_immediates:
            continue
        for immediate in offset_immediates:
            if immediate.encoding_id not in _ADDRESS_OFFSET_IMMEDIATE_ENCODING_IDS:
                raise ValueError(
                    f"AMDGPU memory descriptor '{descriptor.key}' immediate "
                    f"'{immediate.field_name}' has no address-unit encoding"
                )
        split_offset_immediates = tuple(
            immediate
            for immediate in offset_immediates
            if immediate.field_name in ("offset0", "offset1")
        )
        if split_offset_immediates:
            if len(split_offset_immediates) != 2:
                raise ValueError(
                    f"AMDGPU memory descriptor '{descriptor.key}' has an "
                    "incomplete split address offset"
                )
            first_encoding_id = split_offset_immediates[0].encoding_id
            if any(
                immediate.encoding_id != first_encoding_id
                for immediate in split_offset_immediates[1:]
            ):
                raise ValueError(
                    f"AMDGPU memory descriptor '{descriptor.key}' has "
                    "inconsistent split address offset units"
                )


def amdgpu_descriptor_ref_keys() -> tuple[str, ...]:
    """Returns descriptor keys known to the AMDGPU target family."""

    return tuple(sorted(_amdgpu_descriptor_ref_key_set()))


def amdgpu_descriptor_id_keys() -> tuple[str, ...]:
    """Returns descriptor keys that still need stable-ID compatibility refs."""

    return amdgpu_descriptor_ref_keys()


def amdgpu_immediate_encoding_id_items() -> tuple[tuple[str, int], ...]:
    """Returns target-owned immediate encoding IDs used by AMDGPU descriptors."""

    return (
        ("address_offset_byte", _ADDRESS_OFFSET_BYTE_ENCODING_ID),
        ("address_offset_dword", _ADDRESS_OFFSET_DWORD_ENCODING_ID),
        ("address_offset_qword", _ADDRESS_OFFSET_QWORD_ENCODING_ID),
        ("address_offset_dword_stride64", _ADDRESS_OFFSET_DWORD_STRIDE64_ENCODING_ID),
        ("address_offset_qword_stride64", _ADDRESS_OFFSET_QWORD_STRIDE64_ENCODING_ID),
        ("address_offset_ds16", _ADDRESS_OFFSET_DS16_ENCODING_ID),
        ("source_inline_u32", _SOURCE_INLINE_U32_ENCODING_ID),
        ("source_inline_f32", _SOURCE_INLINE_F32_ENCODING_ID),
        ("wait_counter_vmem", _WAIT_COUNTER_VMEM_ENCODING_ID),
        ("wait_counter_lgkm", _WAIT_COUNTER_LGKM_ENCODING_ID),
        ("wait_counter_vmem_load", _WAIT_COUNTER_VMEM_LOAD_ENCODING_ID),
        ("wait_counter_vmem_store", _WAIT_COUNTER_VMEM_STORE_ENCODING_ID),
        ("wait_counter_lds", _WAIT_COUNTER_LDS_ENCODING_ID),
        ("wait_counter_smem", _WAIT_COUNTER_SMEM_ENCODING_ID),
        ("wait_counter_alu", _WAIT_COUNTER_ALU_ENCODING_ID),
    )


def amdgpu_common_reg_class_ids() -> tuple[tuple[str, int], ...]:
    """Returns descriptor-set-local register-class IDs shared by all AMDGPU sets."""

    result: list[tuple[str, int]] = []
    for reg_class_name in (_REG_SGPR, _REG_VGPR, _REG_SCC, _REG_EXEC):
        expected_reg_class_id: int | None = None
        for descriptor_set in _amdgpu_core_descriptor_set_bases():
            reg_class_id = next(
                i
                for i, reg_class in enumerate(descriptor_set.reg_classes)
                if reg_class.name == reg_class_name
            )
            if expected_reg_class_id is None:
                expected_reg_class_id = reg_class_id
            elif expected_reg_class_id != reg_class_id:
                raise ValueError(
                    f"AMDGPU common register class '{reg_class_name}' has "
                    "inconsistent descriptor-set-local IDs"
                )
        if expected_reg_class_id is None:
            raise ValueError(
                f"AMDGPU common register class '{reg_class_name}' is missing"
            )
        result.append((reg_class_name, expected_reg_class_id))
    return tuple(result)


def _with_overlay_descriptors(
    base: DescriptorSet,
    spec: AmdgpuIsaFactSource,
    overlay_descriptors: tuple[Descriptor, ...],
) -> DescriptorSet:
    manual_descriptors = _manual_scalar_descriptors(spec)
    descriptor_set = replace(
        base,
        descriptors=_categorize_amdgpu_descriptors(
            (
                manual_descriptors[0],
                *overlay_descriptors,
                *manual_descriptors[1:],
                *_hal_buffer_descriptor_pseudos(),
                *base.descriptors,
            )
        ),
    )
    _validate_address_immediate_units(descriptor_set)
    return descriptor_set


_GFX125X_VGPR_MSB_ADDRESSABLE_UNIT_COUNT = 256

_GFX125X_VOP_MODE_FIELDS = frozenset(
    ("VDST", "SRC0", "VSRC0", "SRC1", "VSRC1", "SRC2", "VSRC2")
)
_GFX125X_DS_MODE_FIELDS = frozenset(("ADDR", "DATA0", "DATA1", "VDST"))
_GFX125X_FLAT_MODE_FIELDS = frozenset(
    ("ADDR", "VADDR", "DATA", "VDATA", "VSRC", "VDST")
)
_GFX125X_BUFFER_MODE_FIELDS = frozenset(("VADDR", "VDATA", "VSRC", "VDST"))

_GFX125X_VGPR_MSB_FIELDS_BY_FORMAT = {
    AMDGPU_ENCODING_FORMAT_VOP1: _GFX125X_VOP_MODE_FIELDS,
    AMDGPU_ENCODING_FORMAT_VOP1_LITERAL: _GFX125X_VOP_MODE_FIELDS,
    AMDGPU_ENCODING_FORMAT_VOP1_DPP: _GFX125X_VOP_MODE_FIELDS,
    AMDGPU_ENCODING_FORMAT_VOP1_DPP16: _GFX125X_VOP_MODE_FIELDS,
    AMDGPU_ENCODING_FORMAT_VOP1_SDWA: _GFX125X_VOP_MODE_FIELDS,
    AMDGPU_ENCODING_FORMAT_VOP2: _GFX125X_VOP_MODE_FIELDS,
    AMDGPU_ENCODING_FORMAT_VOP2_LITERAL: _GFX125X_VOP_MODE_FIELDS,
    AMDGPU_ENCODING_FORMAT_VOP3: _GFX125X_VOP_MODE_FIELDS,
    AMDGPU_ENCODING_FORMAT_VOP3_LITERAL: _GFX125X_VOP_MODE_FIELDS,
    AMDGPU_ENCODING_FORMAT_VOP3P: _GFX125X_VOP_MODE_FIELDS,
    AMDGPU_ENCODING_FORMAT_VOP3PX2: _GFX125X_VOP_MODE_FIELDS,
    AMDGPU_ENCODING_FORMAT_VOP3_SDST: _GFX125X_VOP_MODE_FIELDS,
    AMDGPU_ENCODING_FORMAT_DS: _GFX125X_DS_MODE_FIELDS,
    AMDGPU_ENCODING_FORMAT_VDS: _GFX125X_DS_MODE_FIELDS,
    AMDGPU_ENCODING_FORMAT_FLAT: _GFX125X_FLAT_MODE_FIELDS,
    AMDGPU_ENCODING_FORMAT_VFLAT: _GFX125X_FLAT_MODE_FIELDS,
    AMDGPU_ENCODING_FORMAT_VGLOBAL: _GFX125X_FLAT_MODE_FIELDS,
    AMDGPU_ENCODING_FORMAT_VSCRATCH: _GFX125X_FLAT_MODE_FIELDS,
    AMDGPU_ENCODING_FORMAT_MUBUF: _GFX125X_BUFFER_MODE_FIELDS,
    AMDGPU_ENCODING_FORMAT_VBUFFER: _GFX125X_BUFFER_MODE_FIELDS,
}


def _operand_has_vgpr_alt(operand: Operand) -> bool:
    return any(reg_alt.reg_class == _REG_VGPR for reg_alt in operand.reg_alts)


def _operand_is_explicit_register(operand: Operand) -> bool:
    return OperandFlag.IMPLICIT not in operand.flags and operand.role in (
        OperandRole.RESULT,
        OperandRole.OPERAND,
        OperandRole.OPERAND_RESULT,
        OperandRole.PREDICATE,
        OperandRole.RESOURCE,
    )


def _gfx125x_operand_encoding_field_name(operand: Operand) -> str | None:
    if operand.encoding_field_id == 0:
        return None
    return amdgpu_encoding_field_name(operand.encoding_field_id)


def _gfx125x_operand_uses_vgpr_msb_state(
    descriptor: Descriptor, operand: Operand
) -> bool:
    field_name = _gfx125x_operand_encoding_field_name(operand)
    if field_name is None:
        return False
    return field_name in _GFX125X_VGPR_MSB_FIELDS_BY_FORMAT.get(
        descriptor.encoding_format_id, frozenset()
    )


def _with_gfx125x_operand_address_state(
    descriptor: Descriptor, operand: Operand
) -> Operand:
    if not _operand_is_explicit_register(operand) or not _operand_has_vgpr_alt(operand):
        return operand
    if operand.address_map_kind is not OperandAddressMapKind.DIRECT:
        return operand
    if _gfx125x_operand_uses_vgpr_msb_state(descriptor, operand):
        return replace(
            operand,
            address_map_kind=OperandAddressMapKind.TARGET_STATE,
            addressable_unit_count=_GFX125X_VGPR_MSB_ADDRESSABLE_UNIT_COUNT,
        )
    return replace(
        operand,
        address_map_kind=OperandAddressMapKind.LOW_SUBSET,
        addressable_unit_count=_GFX125X_VGPR_MSB_ADDRESSABLE_UNIT_COUNT,
    )


def _with_gfx125x_vgpr_msb_address_state(descriptor: Descriptor) -> Descriptor:
    operands = tuple(
        _with_gfx125x_operand_address_state(descriptor, operand)
        for operand in descriptor.operands
    )
    updated_descriptor = replace(descriptor, operands=operands)
    if any(
        operand.address_map_kind is OperandAddressMapKind.TARGET_STATE
        for operand in operands
    ):
        updated_descriptor = _with_mode_state_read(updated_descriptor)
    return updated_descriptor


def _with_gfx125x_vgpr_msb_address_states(
    descriptor_set: DescriptorSet,
) -> DescriptorSet:
    descriptor_set = replace(
        descriptor_set,
        descriptors=tuple(
            _with_gfx125x_vgpr_msb_address_state(descriptor)
            for descriptor in descriptor_set.descriptors
        ),
    )
    _validate_gfx125x_vgpr_msb_address_state(descriptor_set)
    return descriptor_set


def _descriptor_writes_mode_state(descriptor: Descriptor) -> bool:
    return any(
        OperandFlag.STATE_WRITE in operand.flags
        and len(operand.reg_alts) == 1
        and operand.reg_alts[0].reg_class == _REG_MODE
        for operand in descriptor.operands
    )


def _validate_gfx125x_vgpr_msb_address_state(descriptor_set: DescriptorSet) -> None:
    descriptors_by_key = {
        descriptor.key: descriptor for descriptor in descriptor_set.descriptors
    }
    try:
        mode_descriptor = descriptors_by_key["amdgpu.s_set_vgpr_msb"]
    except KeyError as exc:
        raise ValueError(
            "gfx125x VGPR-MSB target-state operands require 'amdgpu.s_set_vgpr_msb'"
        ) from exc
    if not _descriptor_writes_mode_state(mode_descriptor):
        raise ValueError(
            "gfx125x descriptor 'amdgpu.s_set_vgpr_msb' must write MODE state"
        )
    for descriptor in descriptor_set.descriptors:
        has_target_state_operand = False
        for operand in descriptor.operands:
            if operand.address_map_kind is not OperandAddressMapKind.TARGET_STATE:
                continue
            has_target_state_operand = True
            if not _gfx125x_operand_uses_vgpr_msb_state(descriptor, operand):
                raise ValueError(
                    f"gfx125x descriptor '{descriptor.key}' marks operand "
                    f"'{operand.field_name}' as VGPR-MSB target-state, but "
                    "the operand encoding field has no S_SET_VGPR_MSB slot"
                )
        if has_target_state_operand and not any(
            _is_mode_state_read(operand) for operand in descriptor.operands
        ):
            raise ValueError(
                f"gfx125x descriptor '{descriptor.key}' uses VGPR-MSB "
                "target-state operands but does not read MODE state"
            )


@dataclass(frozen=True, slots=True)
class _AmdgpuCoreDescriptorSetBuilder:
    base: DescriptorSet
    overlay_descriptors: Callable[[AmdgpuIsaFactSource], tuple[Descriptor, ...]]


_AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS = {
    "cdna3": _AmdgpuCoreDescriptorSetBuilder(
        base=_AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE,
        overlay_descriptors=_gfx940_core_overlay_descriptors,
    ),
    "cdna4": _AmdgpuCoreDescriptorSetBuilder(
        base=_AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE,
        overlay_descriptors=_gfx950_core_overlay_descriptors,
    ),
    "rdna3": _AmdgpuCoreDescriptorSetBuilder(
        base=_AMDGPU_RDNA3_CORE_DESCRIPTOR_SET_BASE,
        overlay_descriptors=_gfx11_core_overlay_descriptors,
    ),
    "rdna4": _AmdgpuCoreDescriptorSetBuilder(
        base=_AMDGPU_RDNA4_CORE_DESCRIPTOR_SET_BASE,
        overlay_descriptors=_gfx12_core_overlay_descriptors,
    ),
    "rdna4_gfx125x": _AmdgpuCoreDescriptorSetBuilder(
        base=_AMDGPU_RDNA4_GFX125X_CORE_DESCRIPTOR_SET_BASE,
        overlay_descriptors=_gfx1250_core_overlay_descriptors,
    ),
}

AMDGPU_DESCRIPTOR_SET_GENERATOR_TARGETS = tuple(
    sorted(_AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS)
)


def build_amdgpu_core_descriptor_set_from_spec(
    target: str,
    spec: AmdgpuIsaFactSource,
) -> DescriptorSet:
    try:
        builder = _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS[target]
    except KeyError as exc:
        supported = ", ".join(AMDGPU_DESCRIPTOR_SET_GENERATOR_TARGETS)
        raise ValueError(
            f"unsupported AMDGPU descriptor target '{target}'; "
            f"expected one of: {supported}"
        ) from exc
    validate_amdgpu_descriptor_set_isa_xml(
        amdgpu_descriptor_set_info_by_generator_target(target), spec
    )
    descriptor_set = _with_overlay_descriptors(
        builder.base,
        spec,
        builder.overlay_descriptors(spec),
    )
    if target == "rdna4_gfx125x":
        descriptor_set = _with_gfx125x_vgpr_msb_address_states(descriptor_set)
    return descriptor_set


def build_amdgpu_core_descriptor_set(
    target: str,
    xml_path: str | Path,
) -> DescriptorSet:
    return build_amdgpu_core_descriptor_set_from_spec(
        target, parse_amdgpu_isa_xml_path(xml_path)
    )


__all__ = (
    "AMDGPU_DESCRIPTOR_SET_GENERATOR_TARGETS",
    "_AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS",
    "_AmdgpuCoreDescriptorSetBuilder",
    "_descriptor_address_offset_immediates",
    "_descriptor_has_memory_effect",
    "_gfx125x_operand_uses_vgpr_msb_state",
    "_with_gfx125x_vgpr_msb_address_state",
    "_with_gfx125x_vgpr_msb_address_states",
    "_validate_address_immediate_units",
    "_with_overlay_descriptors",
    "amdgpu_common_reg_class_ids",
    "amdgpu_descriptor_id_keys",
    "amdgpu_descriptor_ref_keys",
    "amdgpu_immediate_encoding_id_items",
    "build_amdgpu_core_descriptor_set",
    "build_amdgpu_core_descriptor_set_from_spec",
)
