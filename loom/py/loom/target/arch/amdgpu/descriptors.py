# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Descriptor input data for AMDGPU target-low shards."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from loom.target.arch.amdgpu.descriptor_overlay import (
    AmdgpuDescriptorOverlay,
    AmdgpuIgnoredOperandOverlay,
    AmdgpuImplicitOperandOverlay,
    AmdgpuOperandOverlay,
    materialize_amdgpu_descriptor_overlays,
)
from loom.target.arch.amdgpu.encoding import (
    AMDGPU_ENCODING_FORMAT_SOP1,
    amdgpu_encoding_field_id,
)
from loom.target.arch.amdgpu.isa_xml import (
    AmdgpuIsaFactSource,
    parse_amdgpu_isa_xml_path,
)
from loom.target.low_descriptors import (
    LOW_DESCRIPTOR_ENCODING_ID_NONE,
    AsmForm,
    AsmImmediate,
    Constraint,
    ConstraintKind,
    Descriptor,
    DescriptorFlag,
    DescriptorSet,
    Effect,
    EffectFlag,
    EffectKind,
    EncodingFieldValue,
    Hazard,
    HazardKind,
    Immediate,
    ImmediateFlag,
    ImmediateKind,
    IssueUse,
    LatencyKind,
    MemorySpace,
    ModelQuality,
    Operand,
    OperandFlag,
    OperandRole,
    RegClass,
    RegClassAlt,
    RegClassAltFlag,
    RegClassFlag,
    Resource,
    ResourceKind,
    ScheduleClass,
    ScheduleClassFlag,
    SpillSlotSpace,
)

_REG_SGPR = "amdgpu.sgpr"
_REG_VGPR = "amdgpu.vgpr"
_REG_AGPR = "amdgpu.agpr"
_REG_M0 = "amdgpu.m0"

_RESOURCE_SALU = "amdgpu.salu"
_RESOURCE_VALU = "amdgpu.valu"
_RESOURCE_SMEM = "amdgpu.smem"
_RESOURCE_VMEM_LOAD = "amdgpu.vmem.load"
_RESOURCE_VMEM_STORE = "amdgpu.vmem.store"
_RESOURCE_LDS_LOAD = "amdgpu.lds.load"
_RESOURCE_LDS_STORE = "amdgpu.lds.store"
_RESOURCE_LDS_CROSSLANE = "amdgpu.lds.crosslane"
_RESOURCE_MFMA = "amdgpu.mfma"
_RESOURCE_WMMA = "amdgpu.wmma"
_RESOURCE_SWMMAC = "amdgpu.swmmac"
_RESOURCE_CONTROL = "amdgpu.control"

_SCHEDULE_SALU = "amdgpu.salu"
_SCHEDULE_VALU = "amdgpu.valu"
_SCHEDULE_SMEM_LOAD = "amdgpu.smem.load"
_SCHEDULE_VMEM_LOAD = "amdgpu.vmem.load"
_SCHEDULE_VMEM_LOAD_LDS = "amdgpu.vmem.load.lds"
_SCHEDULE_VMEM_STORE = "amdgpu.vmem.store"
_SCHEDULE_VMEM_ATOMIC_RETURN = "amdgpu.vmem.atomic.return"
_SCHEDULE_VMEM_ATOMIC_NO_RETURN = "amdgpu.vmem.atomic.no_return"
_SCHEDULE_LDS_LOAD = "amdgpu.lds.load"
_SCHEDULE_LDS_STORE = "amdgpu.lds.store"
_SCHEDULE_LDS_ATOMIC = "amdgpu.lds.atomic"
_SCHEDULE_LDS_CROSSLANE = "amdgpu.lds.crosslane"
_SCHEDULE_BARRIER = "amdgpu.barrier"
_SCHEDULE_MFMA = "amdgpu.mfma"
_SCHEDULE_WMMA = "amdgpu.wmma"
_SCHEDULE_WMMA_SCALE = "amdgpu.wmma.scale"
_SCHEDULE_SWMMAC = "amdgpu.swmmac"
_SCHEDULE_CACHE_CONTROL = "amdgpu.cache.control"
_SCHEDULE_WAIT_MEMORY = "amdgpu.wait.memory"
_SCHEDULE_WAIT_VMEM_STORE = "amdgpu.wait.vmem.store"
_SCHEDULE_WAIT_LDS = "amdgpu.wait.lds"
_SCHEDULE_WAIT_SMEM = "amdgpu.wait.smem"
_SCHEDULE_WAIT_LOAD = "amdgpu.wait.load"
_SCHEDULE_WAIT_STORE = "amdgpu.wait.store"
_SCHEDULE_WAIT_ALU = "amdgpu.wait.alu"
_SCHEDULE_WAIT_IDLE = "amdgpu.wait.idle"

_COUNTER_VMEM_LOAD = 1
_COUNTER_VMEM_STORE = 2
_COUNTER_LDS = 3
_COUNTER_SMEM = 4
_COUNTER_ALU = 5
_MUBUF_SOFFSET_INLINE_ZERO = 0x80

_VMEM_LOAD_COUNTER_HAZARD = Hazard(
    HazardKind.WAIT_COUNTER, counter_id=_COUNTER_VMEM_LOAD
)
_VMEM_STORE_COUNTER_HAZARD = Hazard(
    HazardKind.WAIT_COUNTER, counter_id=_COUNTER_VMEM_STORE
)
_LDS_COUNTER_HAZARD = Hazard(HazardKind.WAIT_COUNTER, counter_id=_COUNTER_LDS)
_SMEM_COUNTER_HAZARD = Hazard(HazardKind.WAIT_COUNTER, counter_id=_COUNTER_SMEM)
_ALU_COUNTER_HAZARD = Hazard(HazardKind.WAIT_COUNTER, counter_id=_COUNTER_ALU)
_GFX950_MEMORY_WAIT_HAZARDS = (
    _VMEM_LOAD_COUNTER_HAZARD,
    _VMEM_STORE_COUNTER_HAZARD,
    _LDS_COUNTER_HAZARD,
    _SMEM_COUNTER_HAZARD,
)
_GFX11_MEMORY_WAIT_HAZARDS = (
    _VMEM_LOAD_COUNTER_HAZARD,
    _LDS_COUNTER_HAZARD,
    _SMEM_COUNTER_HAZARD,
)
_VMEM_LOAD_WAIT_HAZARDS = (_VMEM_LOAD_COUNTER_HAZARD,)
_VMEM_STORE_WAIT_HAZARDS = (_VMEM_STORE_COUNTER_HAZARD,)
_LDS_WAIT_HAZARDS = (_LDS_COUNTER_HAZARD,)
_SMEM_WAIT_HAZARDS = (_SMEM_COUNTER_HAZARD,)
_ALU_WAIT_HAZARDS = (_ALU_COUNTER_HAZARD,)
_IDLE_WAIT_HAZARDS = (
    _VMEM_LOAD_COUNTER_HAZARD,
    _VMEM_STORE_COUNTER_HAZARD,
    _LDS_COUNTER_HAZARD,
    _SMEM_COUNTER_HAZARD,
    _ALU_COUNTER_HAZARD,
)

_ADDRESS_OFFSET_BYTE_ENCODING_ID = 1
_ADDRESS_OFFSET_DWORD_ENCODING_ID = 2
_ADDRESS_OFFSET_QWORD_ENCODING_ID = 3
_ADDRESS_OFFSET_DWORD_STRIDE64_ENCODING_ID = 4
_ADDRESS_OFFSET_QWORD_STRIDE64_ENCODING_ID = 5
_ADDRESS_OFFSET_DS16_ENCODING_ID = 6
_WAIT_COUNTER_VMEM_ENCODING_ID = 16
_WAIT_COUNTER_LGKM_ENCODING_ID = 17
_WAIT_COUNTER_VMEM_LOAD_ENCODING_ID = 18
_WAIT_COUNTER_VMEM_STORE_ENCODING_ID = 19
_WAIT_COUNTER_LDS_ENCODING_ID = 20
_WAIT_COUNTER_SMEM_ENCODING_ID = 21
_WAIT_COUNTER_ALU_ENCODING_ID = 22
_GFX9_11_VECTOR_CACHE_FIELDS = (("GLC", 1), ("SLC", 1), ("DLC", 1))
_GFX950_VECTOR_CACHE_FIELDS = (("NT", 1), ("SC0", 1), ("SC1", 1))
_GFX12_VECTOR_CACHE_FIELDS = (("NV", 1), ("SCOPE", 2), ("TH", 3))
_GFX12_ATOMIC_CACHE_IMMEDIATE_FIELDS = ("SCOPE", "TH")
# VGLOBAL atomic instructions use TH bit 0 to request returning the old value.
_GFX12_TH_ATOMIC_RETURN_VALUE = 0x1
_ADDRESS_OFFSET_IMMEDIATE_ENCODING_IDS = frozenset(
    (
        _ADDRESS_OFFSET_BYTE_ENCODING_ID,
        _ADDRESS_OFFSET_DWORD_ENCODING_ID,
        _ADDRESS_OFFSET_QWORD_ENCODING_ID,
        _ADDRESS_OFFSET_DWORD_STRIDE64_ENCODING_ID,
        _ADDRESS_OFFSET_QWORD_STRIDE64_ENCODING_ID,
        _ADDRESS_OFFSET_DS16_ENCODING_ID,
    )
)
_ADDRESS_OFFSET_IMMEDIATE_FIELD_NAMES = frozenset(("offset", "offset0", "offset1"))
_GLOBAL_SADDR_OFF = 0x7C
_GLOBAL_GFX950_SADDR_OFF = 0x7F

_SGPR_ALT = (RegClassAlt(_REG_SGPR),)
_VGPR_ALT = (RegClassAlt(_REG_VGPR),)
_SGPR_VGPR_ALT = (RegClassAlt(_REG_SGPR), RegClassAlt(_REG_VGPR))
_VGPR_CONST_ALT = (
    RegClassAlt(_REG_VGPR),
    RegClassAlt(None, flags=(RegClassAltFlag.IMMEDIATE,)),
)
_VGPR_AGPR_ALT = (RegClassAlt(_REG_VGPR), RegClassAlt(_REG_AGPR))
_VGPR_AGPR_CONST_ALT = (
    RegClassAlt(_REG_VGPR),
    RegClassAlt(_REG_AGPR),
    RegClassAlt(None, flags=(RegClassAltFlag.IMMEDIATE,)),
)
_M0_ALT = (RegClassAlt(_REG_M0, flags=(RegClassAltFlag.PHYSICAL_ONLY,)),)


def _matrix_hazards(resource: str) -> tuple[Hazard, ...]:
    return (
        _ALU_COUNTER_HAZARD,
        Hazard(
            HazardKind.MIN_DISTANCE,
            resource=resource,
            producer_stage=0,
            consumer_stage=0,
            distance=2,
        ),
    )


def _common_scalar_vector_memory_resources() -> tuple[Resource, ...]:
    return (
        Resource(_RESOURCE_SALU, capacity_per_cycle=1, kind=ResourceKind.SCALAR_ALU),
        Resource(_RESOURCE_VALU, capacity_per_cycle=1, kind=ResourceKind.VECTOR_ALU),
        Resource(_RESOURCE_SMEM, capacity_per_cycle=1, kind=ResourceKind.LOAD),
        Resource(_RESOURCE_VMEM_LOAD, capacity_per_cycle=1, kind=ResourceKind.LOAD),
        Resource(_RESOURCE_VMEM_STORE, capacity_per_cycle=1, kind=ResourceKind.STORE),
        Resource(_RESOURCE_LDS_LOAD, capacity_per_cycle=1, kind=ResourceKind.LOAD),
        Resource(_RESOURCE_LDS_STORE, capacity_per_cycle=1, kind=ResourceKind.STORE),
        Resource(_RESOURCE_LDS_CROSSLANE, capacity_per_cycle=1, kind=ResourceKind.LOAD),
    )


def _common_scalar_vector_memory_schedule_classes(
    *,
    smem_load_hazards: tuple[Hazard, ...],
    vmem_load_hazards: tuple[Hazard, ...],
    vmem_store_hazards: tuple[Hazard, ...],
    lds_load_hazards: tuple[Hazard, ...],
    lds_store_hazards: tuple[Hazard, ...],
    lds_atomic_hazards: tuple[Hazard, ...],
    lds_crosslane_hazards: tuple[Hazard, ...],
) -> tuple[ScheduleClass, ...]:
    return (
        ScheduleClass(
            _SCHEDULE_SALU,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_SALU, cycles=1, units=1),),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_VALU,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_VALU, cycles=1, units=1),),
            hazards=_ALU_WAIT_HAZARDS,
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_SMEM_LOAD,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=8,
            issue_uses=(IssueUse(_RESOURCE_SMEM, cycles=1, units=1),),
            hazards=smem_load_hazards,
            flags=(ScheduleClassFlag.MAY_LOAD,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_VMEM_LOAD,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=16,
            issue_uses=(IssueUse(_RESOURCE_VMEM_LOAD, cycles=1, units=1),),
            hazards=vmem_load_hazards,
            flags=(ScheduleClassFlag.MAY_LOAD,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_VMEM_STORE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=16,
            issue_uses=(IssueUse(_RESOURCE_VMEM_STORE, cycles=1, units=1),),
            hazards=vmem_store_hazards,
            flags=(ScheduleClassFlag.MAY_STORE,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_VMEM_ATOMIC_RETURN,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=16,
            issue_uses=(
                IssueUse(_RESOURCE_VMEM_LOAD, cycles=1, units=1),
                IssueUse(_RESOURCE_VMEM_STORE, cycles=1, units=1),
            ),
            hazards=vmem_load_hazards,
            flags=(ScheduleClassFlag.MAY_LOAD, ScheduleClassFlag.MAY_STORE),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_VMEM_ATOMIC_NO_RETURN,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=16,
            issue_uses=(
                IssueUse(_RESOURCE_VMEM_LOAD, cycles=1, units=1),
                IssueUse(_RESOURCE_VMEM_STORE, cycles=1, units=1),
            ),
            hazards=vmem_store_hazards,
            flags=(ScheduleClassFlag.MAY_LOAD, ScheduleClassFlag.MAY_STORE),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_LDS_LOAD,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=8,
            issue_uses=(IssueUse(_RESOURCE_LDS_LOAD, cycles=1, units=1),),
            hazards=lds_load_hazards,
            flags=(ScheduleClassFlag.MAY_LOAD,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_LDS_STORE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=8,
            issue_uses=(IssueUse(_RESOURCE_LDS_STORE, cycles=1, units=1),),
            hazards=lds_store_hazards,
            flags=(ScheduleClassFlag.MAY_STORE,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_LDS_ATOMIC,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=8,
            issue_uses=(
                IssueUse(_RESOURCE_LDS_LOAD, cycles=1, units=1),
                IssueUse(_RESOURCE_LDS_STORE, cycles=1, units=1),
            ),
            hazards=lds_atomic_hazards,
            flags=(ScheduleClassFlag.MAY_LOAD, ScheduleClassFlag.MAY_STORE),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_LDS_CROSSLANE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=8,
            issue_uses=(IssueUse(_RESOURCE_LDS_CROSSLANE, cycles=1, units=1),),
            hazards=lds_crosslane_hazards,
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_BARRIER,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_CACHE_CONTROL,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
    )


def _asm(
    *,
    mnemonic: str | None = None,
    results: tuple[str, ...] = (),
    operands: tuple[str, ...] = (),
    immediates: tuple[str, ...] = (),
    named_immediates: bool = False,
) -> tuple[AsmForm, ...]:
    return (
        AsmForm(
            mnemonic=mnemonic,
            results=results,
            operands=operands,
            immediates=tuple(
                AsmImmediate(field_name, name=field_name if named_immediates else None)
                for field_name in immediates
            ),
        ),
    )


def _sgpr_result(field_name: str = "dst", *, units: int = 1) -> Operand:
    return Operand(field_name, OperandRole.RESULT, _SGPR_ALT, unit_count=units)


def _sgpr_operand(field_name: str, *, units: int = 1) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _SGPR_ALT, unit_count=units)


def _sgpr_predicate(field_name: str, *, units: int = 1) -> Operand:
    return Operand(field_name, OperandRole.PREDICATE, _SGPR_ALT, unit_count=units)


def _sgpr_resource(field_name: str, *, units: int = 1) -> Operand:
    return Operand(field_name, OperandRole.RESOURCE, _SGPR_ALT, unit_count=units)


def _m0_implicit_resource(field_name: str = "m0") -> Operand:
    return Operand(
        field_name,
        OperandRole.RESOURCE,
        _M0_ALT,
        flags=(OperandFlag.IMPLICIT,),
        unit_count=1,
    )


def _m0_result(field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _M0_ALT, unit_count=1)


def _vgpr_result(field_name: str = "dst", *, units: int = 1) -> Operand:
    return Operand(field_name, OperandRole.RESULT, _VGPR_ALT, unit_count=units)


def _vgpr_operand(field_name: str, *, units: int = 1) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _VGPR_ALT, unit_count=units)


def _sgpr_vgpr_operand(field_name: str, *, units: int = 1) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _SGPR_VGPR_ALT, unit_count=units)


def _vgpr_const_operand(field_name: str, *, units: int = 1) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _VGPR_CONST_ALT, unit_count=units)


def _vgpr_agpr_result(field_name: str = "dst", *, units: int = 1) -> Operand:
    return Operand(field_name, OperandRole.RESULT, _VGPR_AGPR_ALT, unit_count=units)


def _vgpr_agpr_operand(field_name: str, *, units: int = 1) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _VGPR_AGPR_ALT, unit_count=units)


def _vgpr_agpr_const_operand(field_name: str, *, units: int = 1) -> Operand:
    return Operand(
        field_name, OperandRole.OPERAND, _VGPR_AGPR_CONST_ALT, unit_count=units
    )


_U32_IMMEDIATE = Immediate(
    "imm32",
    ImmediateKind.UNSIGNED,
    bit_width=32,
    unsigned_max=(2**32) - 1,
)


def _s_mov_b32_descriptors() -> tuple[Descriptor, ...]:
    return (
        Descriptor(
            key="amdgpu.s_mov_b32",
            mnemonic="s_mov_b32",
            semantic_tag="integer.const.u32",
            operands=(_sgpr_result(),),
            immediates=(_U32_IMMEDIATE,),
            asm_forms=_asm(results=("dst",), immediates=("imm32",)),
            schedule_class=_SCHEDULE_SALU,
            encoding_format_id=AMDGPU_ENCODING_FORMAT_SOP1,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="amdgpu.s_mov_b32_m0",
            mnemonic="s_mov_b32",
            semantic_tag="special.m0.move.u32",
            operands=(
                _m0_result(),
                Operand(
                    "src",
                    OperandRole.OPERAND,
                    _SGPR_ALT,
                    encoding_field_id=amdgpu_encoding_field_id("SSRC0"),
                ),
            ),
            encoding_field_values=(
                EncodingFieldValue(amdgpu_encoding_field_id("SDST"), 124),
            ),
            asm_forms=_asm(
                mnemonic="s_mov_b32_m0", results=("dst",), operands=("src",)
            ),
            schedule_class=_SCHEDULE_SALU,
            encoding_format_id=AMDGPU_ENCODING_FORMAT_SOP1,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
    )


_VMCNT_IMMEDIATE = Immediate(
    "vmcnt",
    ImmediateKind.UNSIGNED,
    bit_width=6,
    encoding_id=_WAIT_COUNTER_VMEM_ENCODING_ID,
    unsigned_max=(2**6) - 1,
)

_LGKMCNT_IMMEDIATE = Immediate(
    "lgkmcnt",
    ImmediateKind.UNSIGNED,
    bit_width=6,
    encoding_id=_WAIT_COUNTER_LGKM_ENCODING_ID,
    unsigned_max=(2**6) - 1,
)

_LOADCNT_IMMEDIATE = Immediate(
    "loadcnt",
    ImmediateKind.UNSIGNED,
    bit_width=6,
    encoding_id=_WAIT_COUNTER_VMEM_LOAD_ENCODING_ID,
    unsigned_max=(2**6) - 1,
)

_STORECNT_IMMEDIATE = Immediate(
    "storecnt",
    ImmediateKind.UNSIGNED,
    bit_width=6,
    encoding_id=_WAIT_COUNTER_VMEM_STORE_ENCODING_ID,
    unsigned_max=(2**6) - 1,
)

_VSCNT_IMMEDIATE = Immediate(
    "vscnt",
    ImmediateKind.UNSIGNED,
    bit_width=6,
    encoding_id=_WAIT_COUNTER_VMEM_STORE_ENCODING_ID,
    unsigned_max=(2**6) - 1,
)

_DSCNT_IMMEDIATE = Immediate(
    "dscnt",
    ImmediateKind.UNSIGNED,
    bit_width=6,
    encoding_id=_WAIT_COUNTER_LDS_ENCODING_ID,
    unsigned_max=(2**6) - 1,
)

_KMCNT_IMMEDIATE = Immediate(
    "kmcnt",
    ImmediateKind.UNSIGNED,
    bit_width=5,
    encoding_id=_WAIT_COUNTER_SMEM_ENCODING_ID,
    unsigned_max=(2**5) - 1,
)

_DEPCTR_IMMEDIATE = Immediate(
    "depctr",
    ImmediateKind.UNSIGNED,
    bit_width=16,
    encoding_id=_WAIT_COUNTER_ALU_ENCODING_ID,
    unsigned_max=(2**16) - 1,
)

_PREFETCH_COUNT_IMMEDIATE = Immediate(
    "count",
    ImmediateKind.UNSIGNED,
    bit_width=5,
    unsigned_max=(2**5) - 1,
)

_PREFETCH_DISTANCE_IMMEDIATE = Immediate(
    "distance",
    ImmediateKind.UNSIGNED,
    bit_width=16,
    unsigned_max=(2**16) - 1,
)

_MATRIX_A_FORMAT_IMMEDIATE = Immediate(
    "matrix_a_fmt",
    ImmediateKind.UNSIGNED,
    bit_width=8,
    unsigned_max=(2**8) - 1,
)

_MATRIX_B_FORMAT_IMMEDIATE = Immediate(
    "matrix_b_fmt",
    ImmediateKind.UNSIGNED,
    bit_width=8,
    unsigned_max=(2**8) - 1,
)

_MATRIX_A_SCALE_IMMEDIATE = Immediate(
    "matrix_a_scale",
    ImmediateKind.UNSIGNED,
    bit_width=8,
    unsigned_max=(2**8) - 1,
)

_MATRIX_B_SCALE_IMMEDIATE = Immediate(
    "matrix_b_scale",
    ImmediateKind.UNSIGNED,
    bit_width=8,
    unsigned_max=(2**8) - 1,
)

_MATRIX_A_SCALE_FORMAT_IMMEDIATE = Immediate(
    "matrix_a_scale_fmt",
    ImmediateKind.UNSIGNED,
    bit_width=8,
    unsigned_max=(2**8) - 1,
)

_MATRIX_B_SCALE_FORMAT_IMMEDIATE = Immediate(
    "matrix_b_scale_fmt",
    ImmediateKind.UNSIGNED,
    bit_width=8,
    unsigned_max=(2**8) - 1,
)

_MATRIX_A_REUSE_IMMEDIATE = Immediate(
    "matrix_a_reuse",
    ImmediateKind.UNSIGNED,
    bit_width=1,
    unsigned_max=1,
)

_MATRIX_B_REUSE_IMMEDIATE = Immediate(
    "matrix_b_reuse",
    ImmediateKind.UNSIGNED,
    bit_width=1,
    unsigned_max=1,
)

_INDEX_KEY_16_IMMEDIATE = Immediate(
    "index_key_16bit",
    ImmediateKind.UNSIGNED,
    bit_width=32,
    unsigned_max=(2**32) - 1,
)

_GLOBAL_LOAD_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=32,
)

_GLOBAL_LOAD_B64_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=64,
)

_GLOBAL_LOAD_B128_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=128,
)

_GLOBAL_STORE_EFFECT = Effect(
    EffectKind.WRITE,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=32,
)

_GLOBAL_STORE_B64_EFFECT = Effect(
    EffectKind.WRITE,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=64,
)

_GLOBAL_STORE_B128_EFFECT = Effect(
    EffectKind.WRITE,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=128,
)


def _workgroup_memory_effect(kind: EffectKind, width_bits: int) -> Effect:
    return Effect(
        kind,
        memory_space=MemorySpace.WORKGROUP,
        flags=(EffectFlag.DEPENDENCY,),
        width_bits=width_bits,
    )


_WORKGROUP_BARRIER_EFFECT = Effect(
    EffectKind.BARRIER,
    memory_space=MemorySpace.WORKGROUP,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
)

_CACHE_CONTROL_EFFECT = Effect(
    EffectKind.BARRIER,
    memory_space=MemorySpace.GENERIC,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
)

_GLOBAL_PREFETCH_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
)

_INSTRUCTION_PREFETCH_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.GENERIC,
    flags=(EffectFlag.DEPENDENCY,),
)

_CONVERGENT_EFFECT = Effect(
    EffectKind.CONVERGENT,
    flags=(EffectFlag.ORDERED,),
)

_WAIT_EFFECT = Effect(
    EffectKind.COUNTER,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
)

_VMEM_LOAD_WAIT_EFFECT = Effect(
    EffectKind.COUNTER,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
    counter_id=_COUNTER_VMEM_LOAD,
)

_VMEM_STORE_WAIT_EFFECT = Effect(
    EffectKind.COUNTER,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
    counter_id=_COUNTER_VMEM_STORE,
)

_LDS_WAIT_EFFECT = Effect(
    EffectKind.COUNTER,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
    counter_id=_COUNTER_LDS,
)

_SMEM_WAIT_EFFECT = Effect(
    EffectKind.COUNTER,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
    counter_id=_COUNTER_SMEM,
)

_ALU_WAIT_EFFECT = Effect(
    EffectKind.COUNTER,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
    counter_id=_COUNTER_ALU,
)

_DESTRUCTIVE_ACCUMULATOR_CONSTRAINTS = (
    Constraint(ConstraintKind.TIED, 0, 1),
    Constraint(ConstraintKind.DESTRUCTIVE, 0, 1),
    Constraint(ConstraintKind.EARLY_CLOBBER, 0),
)
_BUFFER_ATOMIC_VDATA_INPUT_REASON = "xml-models-buffer-atomic-vdata-as-output-only"
_DESTRUCTIVE_BUFFER_ATOMIC_CONSTRAINTS = (
    Constraint(ConstraintKind.TIED, 0, 1),
    Constraint(ConstraintKind.DESTRUCTIVE, 0, 1),
)
_PSEUDO_DEAD_REMOVABLE_FLAGS = (DescriptorFlag.DEAD_REMOVABLE, DescriptorFlag.PSEUDO)


def _offset_immediate(
    bit_width: int,
    *,
    encoding_id: int = _ADDRESS_OFFSET_BYTE_ENCODING_ID,
) -> Immediate:
    return _named_offset_immediate("offset", bit_width, encoding_id=encoding_id)


def _named_offset_immediate(
    field_name: str,
    bit_width: int,
    *,
    encoding_id: int = _ADDRESS_OFFSET_BYTE_ENCODING_ID,
) -> Immediate:
    return Immediate(
        field_name,
        ImmediateKind.UNSIGNED,
        bit_width=bit_width,
        encoding_id=encoding_id,
        unsigned_max=(2**bit_width) - 1,
    )


def _signed_offset_immediate(
    bit_width: int,
    *,
    encoding_id: int = _ADDRESS_OFFSET_BYTE_ENCODING_ID,
) -> Immediate:
    return Immediate(
        "offset",
        ImmediateKind.SIGNED,
        bit_width=bit_width,
        encoding_id=encoding_id,
        signed_min=-(2 ** (bit_width - 1)),
        unsigned_max=(2 ** (bit_width - 1)) - 1,
    )


def _cache_immediate(
    field_name: str, bit_width: int, *, default_value: int = 0
) -> Immediate:
    return Immediate(
        field_name.lower(),
        ImmediateKind.UNSIGNED,
        flags=(ImmediateFlag.DEFAULT_VALUE,),
        bit_width=bit_width,
        unsigned_max=(2**bit_width) - 1,
        default_value=default_value,
    )


def _cache_field_names(cache_fields: tuple[tuple[str, int], ...]) -> tuple[str, ...]:
    return tuple(field_name for field_name, _bit_width in cache_fields)


def _cache_immediates(
    cache_fields: tuple[tuple[str, int], ...],
) -> tuple[Immediate, ...]:
    return tuple(
        _cache_immediate(field_name, bit_width)
        for field_name, bit_width in cache_fields
    )


def _cache_immediates_with_defaults(
    cache_fields: tuple[tuple[str, int], ...],
    defaults: dict[str, int],
) -> tuple[Immediate, ...]:
    return tuple(
        _cache_immediate(
            field_name, bit_width, default_value=defaults.get(field_name, 0)
        )
        for field_name, bit_width in cache_fields
    )


_IGNORE_SCC_OUTPUT = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_SSRC_SPECIAL_SCC",
    data_format_name="FMT_NUM_B1",
    size_bits=1,
    is_input=False,
    is_output=True,
    ignore_reason="value-pseudo-drops-scc",
)

_IGNORE_GLOBAL_READ_MEMORY = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B32",
    size_bits=32,
    is_input=True,
    is_output=False,
    ignore_reason="modeled-by-global-read-effect",
)

_IGNORE_GLOBAL_READ_MEMORY_B64 = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B64",
    size_bits=64,
    is_input=True,
    is_output=False,
    ignore_reason="modeled-by-global-read-effect",
)

_IGNORE_GLOBAL_READ_MEMORY_B128 = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B128",
    size_bits=128,
    is_input=True,
    is_output=False,
    ignore_reason="modeled-by-global-read-effect",
)

_IGNORE_GLOBAL_WRITE_MEMORY = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B32",
    size_bits=32,
    is_input=False,
    is_output=True,
    ignore_reason="modeled-by-global-write-effect",
)

_IGNORE_GLOBAL_WRITE_MEMORY_B64 = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B64",
    size_bits=64,
    is_input=False,
    is_output=True,
    ignore_reason="modeled-by-global-write-effect",
)

_IGNORE_GLOBAL_WRITE_MEMORY_B128 = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B128",
    size_bits=128,
    is_input=False,
    is_output=True,
    ignore_reason="modeled-by-global-write-effect",
)


def _ignore_workgroup_memory(
    *, width_bits: int, is_input: bool, data_format_name: str | None = None
) -> AmdgpuImplicitOperandOverlay:
    return AmdgpuImplicitOperandOverlay(
        operand_type="OPR_DSMEM",
        data_format_name=data_format_name or f"FMT_NUM_B{width_bits}",
        size_bits=width_bits,
        is_input=is_input,
        is_output=not is_input,
        ignore_reason=(
            "modeled-by-workgroup-read-effect"
            if is_input
            else "modeled-by-workgroup-write-effect"
        ),
    )


def _ignore_global_read_memory(width_bits: int) -> AmdgpuImplicitOperandOverlay:
    match width_bits:
        case 32:
            return _IGNORE_GLOBAL_READ_MEMORY
        case 64:
            return _IGNORE_GLOBAL_READ_MEMORY_B64
        case 128:
            return _IGNORE_GLOBAL_READ_MEMORY_B128
        case _:
            raise ValueError(f"unsupported global read width {width_bits}")


def _ignore_global_write_memory(width_bits: int) -> AmdgpuImplicitOperandOverlay:
    match width_bits:
        case 32:
            return _IGNORE_GLOBAL_WRITE_MEMORY
        case 64:
            return _IGNORE_GLOBAL_WRITE_MEMORY_B64
        case 128:
            return _IGNORE_GLOBAL_WRITE_MEMORY_B128
        case _:
            raise ValueError(f"unsupported global write width {width_bits}")


def _global_read_effect(width_bits: int) -> Effect:
    match width_bits:
        case 32:
            return _GLOBAL_LOAD_EFFECT
        case 64:
            return _GLOBAL_LOAD_B64_EFFECT
        case 128:
            return _GLOBAL_LOAD_B128_EFFECT
        case _:
            raise ValueError(f"unsupported global read width {width_bits}")


def _global_write_effect(width_bits: int) -> Effect:
    match width_bits:
        case 32:
            return _GLOBAL_STORE_EFFECT
        case 64:
            return _GLOBAL_STORE_B64_EFFECT
        case 128:
            return _GLOBAL_STORE_B128_EFFECT
        case _:
            raise ValueError(f"unsupported global write width {width_bits}")


def _ignore_global_atomic_memory(
    *, data_format_name: str, is_input: bool
) -> AmdgpuImplicitOperandOverlay:
    return AmdgpuImplicitOperandOverlay(
        operand_type="OPR_GPUMEM",
        data_format_name=data_format_name,
        size_bits=32,
        is_input=is_input,
        is_output=not is_input,
        ignore_reason=(
            "modeled-by-global-atomic-read-effect"
            if is_input
            else "modeled-by-global-atomic-write-effect"
        ),
    )


def _global_atomic_effects(
    width_bits: int, *, counter_id: int
) -> tuple[Effect, Effect]:
    return (
        Effect(
            EffectKind.READ,
            memory_space=MemorySpace.GLOBAL,
            flags=(EffectFlag.DEPENDENCY,),
            counter_id=counter_id,
            width_bits=width_bits,
        ),
        Effect(
            EffectKind.WRITE,
            memory_space=MemorySpace.GLOBAL,
            flags=(EffectFlag.DEPENDENCY,),
            counter_id=counter_id,
            width_bits=width_bits,
        ),
    )


def _global_to_lds_effects(width_bits: int) -> tuple[Effect, Effect]:
    return (
        Effect(
            EffectKind.READ,
            memory_space=MemorySpace.GLOBAL,
            flags=(EffectFlag.DEPENDENCY,),
            counter_id=_COUNTER_VMEM_LOAD,
            width_bits=width_bits,
        ),
        Effect(
            EffectKind.WRITE,
            memory_space=MemorySpace.WORKGROUP,
            flags=(EffectFlag.DEPENDENCY,),
            counter_id=_COUNTER_VMEM_LOAD,
            width_bits=width_bits,
        ),
    )


def _implicit_m0_input() -> AmdgpuImplicitOperandOverlay:
    return AmdgpuImplicitOperandOverlay(
        operand_type="OPR_SDST_M0",
        descriptor_operand=_m0_implicit_resource(),
        data_format_name="FMT_NUM_B32",
        size_bits=32,
        is_input=True,
        is_output=False,
    )


def _s_add_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_add_u32",
        instruction_name="S_ADD_U32",
        mnemonic="s_add_u32",
        encoding_name="ENC_SOP2",
        semantic_tag="integer.add.u32",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs")),
        ),
        implicit_operands=(_IGNORE_SCC_OUTPUT,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_sub_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_sub_u32",
        instruction_name="S_SUB_U32",
        mnemonic="s_sub_u32",
        encoding_name="ENC_SOP2",
        semantic_tag="integer.sub.u32",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs")),
        ),
        implicit_operands=(_IGNORE_SCC_OUTPUT,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_binary_u32_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SOP2",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs")),
        ),
        implicit_operands=(_IGNORE_SCC_OUTPUT,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_add_u32_overlay(instruction_name: str) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_add_u32",
        instruction_name=instruction_name,
        mnemonic="v_add_u32",
        encoding_name="ENC_VOP2",
        semantic_tag="integer.add.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_sub_u32_overlay(instruction_name: str, mnemonic: str) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_sub_u32",
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_VOP2",
        semantic_tag="integer.sub.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_binary_u32_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    lhs_name: str = "lhs",
    rhs_name: str = "rhs",
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_VOP2",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand(lhs_name)),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand(rhs_name)),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_mul_lo_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_mul_lo_u32",
        instruction_name="V_MUL_LO_U32",
        mnemonic="v_mul_lo_u32",
        encoding_name="ENC_VOP3",
        semantic_tag="integer.mul.lo.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_and_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_and_b32",
        instruction_name="S_AND_B32",
        mnemonic="s_and_b32",
        semantic_tag="integer.and.u32",
    )


def _s_or_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_or_b32",
        instruction_name="S_OR_B32",
        mnemonic="s_or_b32",
        semantic_tag="integer.or.u32",
    )


def _s_xor_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_xor_b32",
        instruction_name="S_XOR_B32",
        mnemonic="s_xor_b32",
        semantic_tag="integer.xor.u32",
    )


def _s_lshl_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_lshl_b32",
        instruction_name="S_LSHL_B32",
        mnemonic="s_lshl_b32",
        semantic_tag="integer.shl.u32",
    )


def _s_lshr_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_lshr_b32",
        instruction_name="S_LSHR_B32",
        mnemonic="s_lshr_b32",
        semantic_tag="integer.shr.u32",
    )


def _s_ashr_i32_overlay() -> AmdgpuDescriptorOverlay:
    return _s_binary_u32_overlay(
        descriptor_key="amdgpu.s_ashr_i32",
        instruction_name="S_ASHR_I32",
        mnemonic="s_ashr_i32",
        semantic_tag="integer.shr.i32",
    )


def _v_and_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key="amdgpu.v_and_b32",
        instruction_name="V_AND_B32",
        mnemonic="v_and_b32",
        semantic_tag="integer.and.u32",
    )


def _v_or_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key="amdgpu.v_or_b32",
        instruction_name="V_OR_B32",
        mnemonic="v_or_b32",
        semantic_tag="integer.or.u32",
    )


def _v_xor_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key="amdgpu.v_xor_b32",
        instruction_name="V_XOR_B32",
        mnemonic="v_xor_b32",
        semantic_tag="integer.xor.u32",
    )


def _v_lshlrev_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key="amdgpu.v_lshlrev_b32",
        instruction_name="V_LSHLREV_B32",
        mnemonic="v_lshlrev_b32",
        semantic_tag="integer.shl.u32",
        lhs_name="shift",
        rhs_name="value",
    )


def _v_lshrrev_b32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key="amdgpu.v_lshrrev_b32",
        instruction_name="V_LSHRREV_B32",
        mnemonic="v_lshrrev_b32",
        semantic_tag="integer.shr.u32",
        lhs_name="shift",
        rhs_name="value",
    )


def _v_ashrrev_i32_overlay() -> AmdgpuDescriptorOverlay:
    return _v_binary_u32_overlay(
        descriptor_key="amdgpu.v_ashrrev_i32",
        instruction_name="V_ASHRREV_I32",
        mnemonic="v_ashrrev_i32",
        semantic_tag="integer.shr.i32",
        lhs_name="shift",
        rhs_name="value",
    )


def _i32_bitwise_shift_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_and_b32_overlay(),
        _s_or_b32_overlay(),
        _s_xor_b32_overlay(),
        _s_lshl_b32_overlay(),
        _s_lshr_b32_overlay(),
        _s_ashr_i32_overlay(),
        _v_and_b32_overlay(),
        _v_or_b32_overlay(),
        _v_xor_b32_overlay(),
        _v_lshlrev_b32_overlay(),
        _v_lshrrev_b32_overlay(),
        _v_ashrrev_i32_overlay(),
    )


def _v_add_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_add_f32",
        instruction_name="V_ADD_F32",
        mnemonic="v_add_f32",
        encoding_name="ENC_VOP2",
        semantic_tag="float.add.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_sub_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_sub_f32",
        instruction_name="V_SUB_F32",
        mnemonic="v_sub_f32",
        encoding_name="ENC_VOP2",
        semantic_tag="float.sub.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_mul_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_mul_f32",
        instruction_name="V_MUL_F32",
        mnemonic="v_mul_f32",
        encoding_name="ENC_VOP2",
        semantic_tag="float.mul.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_min_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_min_f32",
        instruction_name="V_MIN_F32",
        mnemonic="v_min_f32",
        encoding_name="ENC_VOP2",
        semantic_tag="float.minnum.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_max_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_max_f32",
        instruction_name="V_MAX_F32",
        mnemonic="v_max_f32",
        encoding_name="ENC_VOP2",
        semantic_tag="float.maxnum.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("VSRC1", _vgpr_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_fma_f32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_fma_f32",
        instruction_name="V_FMA_F32",
        mnemonic="v_fma_f32",
        encoding_name="ENC_VOP3",
        semantic_tag="float.fma.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("a")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("b")),
            AmdgpuOperandOverlay("SRC2", _vgpr_operand("c")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cvt_f32_i32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_cvt_f32_i32",
        instruction_name="V_CVT_F32_I32",
        mnemonic="v_cvt_f32_i32",
        encoding_name="ENC_VOP1",
        semantic_tag="convert.signed.i32.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("input")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cvt_f32_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_cvt_f32_u32",
        instruction_name="V_CVT_F32_U32",
        mnemonic="v_cvt_f32_u32",
        encoding_name="ENC_VOP1",
        semantic_tag="convert.unsigned.u32.f32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _sgpr_vgpr_operand("input")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cmp_i32_overlay(
    *, predicate: str, instruction_suffix: str, semantic_suffix: str
) -> AmdgpuDescriptorOverlay:
    instruction_predicate = instruction_suffix.lower()
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_cmp_{predicate}_i32",
        instruction_name=f"V_CMP_{instruction_suffix}_I32",
        mnemonic=f"v_cmp_{instruction_predicate}_i32",
        encoding_name="ENC_VOP3",
        semantic_tag=f"cmp.i32.{semantic_suffix}",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _sgpr_result("mask", units=2)),
            AmdgpuOperandOverlay("SRC0", _vgpr_const_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_const_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cmp_u32_overlay(
    *, predicate: str, instruction_suffix: str, semantic_suffix: str
) -> AmdgpuDescriptorOverlay:
    instruction_predicate = instruction_suffix.lower()
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_cmp_{predicate}_u32",
        instruction_name=f"V_CMP_{instruction_suffix}_U32",
        mnemonic=f"v_cmp_{instruction_predicate}_u32",
        encoding_name="ENC_VOP3",
        semantic_tag=f"cmp.u32.{semantic_suffix}",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _sgpr_result("mask", units=2)),
            AmdgpuOperandOverlay("SRC0", _vgpr_const_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_const_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cmp_f32_overlay(
    *, predicate: str, instruction_suffix: str, semantic_suffix: str
) -> AmdgpuDescriptorOverlay:
    instruction_predicate = instruction_suffix.lower()
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_cmp_{predicate}_f32",
        instruction_name=f"V_CMP_{instruction_suffix}_F32",
        mnemonic=f"v_cmp_{instruction_predicate}_f32",
        encoding_name="ENC_VOP3",
        semantic_tag=f"cmp.f32.{semantic_suffix}",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _sgpr_result("mask", units=2)),
            AmdgpuOperandOverlay("SRC0", _vgpr_const_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_const_operand("rhs")),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_cmp_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _v_cmp_i32_overlay(
            predicate="eq", instruction_suffix="EQ", semantic_suffix="eq"
        ),
        _v_cmp_i32_overlay(
            predicate="ne", instruction_suffix="NE", semantic_suffix="ne"
        ),
        _v_cmp_i32_overlay(
            predicate="slt", instruction_suffix="LT", semantic_suffix="slt"
        ),
        _v_cmp_i32_overlay(
            predicate="sle", instruction_suffix="LE", semantic_suffix="sle"
        ),
        _v_cmp_i32_overlay(
            predicate="sgt", instruction_suffix="GT", semantic_suffix="sgt"
        ),
        _v_cmp_i32_overlay(
            predicate="sge", instruction_suffix="GE", semantic_suffix="sge"
        ),
        _v_cmp_u32_overlay(
            predicate="ult", instruction_suffix="LT", semantic_suffix="ult"
        ),
        _v_cmp_u32_overlay(
            predicate="ule", instruction_suffix="LE", semantic_suffix="ule"
        ),
        _v_cmp_u32_overlay(
            predicate="ugt", instruction_suffix="GT", semantic_suffix="ugt"
        ),
        _v_cmp_u32_overlay(
            predicate="uge", instruction_suffix="GE", semantic_suffix="uge"
        ),
        _v_cmp_f32_overlay(
            predicate="oeq", instruction_suffix="EQ", semantic_suffix="oeq"
        ),
        _v_cmp_f32_overlay(
            predicate="ogt", instruction_suffix="GT", semantic_suffix="ogt"
        ),
        _v_cmp_f32_overlay(
            predicate="oge", instruction_suffix="GE", semantic_suffix="oge"
        ),
        _v_cmp_f32_overlay(
            predicate="olt", instruction_suffix="LT", semantic_suffix="olt"
        ),
        _v_cmp_f32_overlay(
            predicate="ole", instruction_suffix="LE", semantic_suffix="ole"
        ),
        _v_cmp_f32_overlay(
            predicate="one", instruction_suffix="LG", semantic_suffix="one"
        ),
        _v_cmp_f32_overlay(
            predicate="ord", instruction_suffix="O", semantic_suffix="ord"
        ),
        _v_cmp_f32_overlay(
            predicate="ueq", instruction_suffix="NLG", semantic_suffix="ueq"
        ),
        _v_cmp_f32_overlay(
            predicate="ugt", instruction_suffix="NLE", semantic_suffix="ugt"
        ),
        _v_cmp_f32_overlay(
            predicate="uge", instruction_suffix="NLT", semantic_suffix="uge"
        ),
        _v_cmp_f32_overlay(
            predicate="ult", instruction_suffix="NGE", semantic_suffix="ult"
        ),
        _v_cmp_f32_overlay(
            predicate="ule", instruction_suffix="NGT", semantic_suffix="ule"
        ),
        _v_cmp_f32_overlay(
            predicate="une", instruction_suffix="NEQ", semantic_suffix="une"
        ),
        _v_cmp_f32_overlay(
            predicate="uno", instruction_suffix="U", semantic_suffix="uno"
        ),
    )


def _v_cndmask_b32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_cndmask_b32",
        instruction_name="V_CNDMASK_B32",
        mnemonic="v_cndmask_b32",
        encoding_name="ENC_VOP3",
        semantic_tag="select.mask.b32",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_const_operand("false_value")),
            AmdgpuOperandOverlay("SRC1", _vgpr_const_operand("true_value")),
            AmdgpuOperandOverlay("SRC2", _sgpr_predicate("mask", units=2)),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_mov_b32_literal_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_mov_b32",
        instruction_name="V_MOV_B32",
        mnemonic="v_mov_b32",
        encoding_name="VOP1_INST_LITERAL",
        encoding_condition="has_lit",
        semantic_tag="integer.const.u32",
        schedule_class=_SCHEDULE_VALU,
        operands=(AmdgpuOperandOverlay("VDST", _vgpr_result()),),
        asm_forms=_asm(results=("dst",), immediates=("imm32",)),
        immediate_fields=("LITERAL",),
        immediates=(_U32_IMMEDIATE,),
        fixed_encoding_fields=(("SRC0", 255),),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_buffer_load_dword_overlay(
    offset_field_name: str = "OFFSET", offset_bit_width: int = 21
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_buffer_load_dword",
        instruction_name="S_BUFFER_LOAD_DWORD",
        mnemonic="s_buffer_load_dword",
        encoding_name="ENC_SMEM",
        semantic_tag="memory.load.u32",
        schedule_class=_SCHEDULE_SMEM_LOAD,
        operands=(
            AmdgpuOperandOverlay("SDATA", _sgpr_result()),
            AmdgpuOperandOverlay("SBASE", _sgpr_resource("resource", units=4)),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY,),
        immediate_fields=(offset_field_name,),
        immediates=(_offset_immediate(offset_bit_width),),
        effects=(_GLOBAL_LOAD_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_buffer_load_64_overlay(
    *,
    descriptor_key: str = "amdgpu.s_buffer_load_b64",
    instruction_name: str = "S_BUFFER_LOAD_B64",
    mnemonic: str = "s_buffer_load_b64",
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 21,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SMEM",
        semantic_tag="memory.load.u64",
        schedule_class=_SCHEDULE_SMEM_LOAD,
        operands=(
            AmdgpuOperandOverlay("SDATA", _sgpr_result(units=2)),
            AmdgpuOperandOverlay("SBASE", _sgpr_resource("resource", units=4)),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY_B64,),
        immediate_fields=(offset_field_name,),
        immediates=(_offset_immediate(offset_bit_width),),
        effects=(_GLOBAL_LOAD_B64_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_load_dwordx2_overlay(
    offset_field_name: str = "OFFSET", offset_bit_width: int = 21
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_load_dwordx2",
        instruction_name="S_LOAD_DWORDX2",
        mnemonic="s_load_dwordx2",
        encoding_name="ENC_SMEM",
        semantic_tag="memory.load.u64",
        schedule_class=_SCHEDULE_SMEM_LOAD,
        operands=(
            AmdgpuOperandOverlay("SDATA", _sgpr_result(units=2)),
            AmdgpuOperandOverlay("SBASE", _sgpr_operand("base", units=2)),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY_B64,),
        immediate_fields=(offset_field_name,),
        immediates=(_offset_immediate(offset_bit_width),),
        effects=(_GLOBAL_LOAD_B64_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _buffer_load_dword_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.buffer_load_dword",
        instruction_name="BUFFER_LOAD_DWORD",
        mnemonic="buffer_load_dword",
        encoding_name=encoding_name,
        semantic_tag="memory.load.u32",
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_result()),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            AmdgpuOperandOverlay("VADDR", _vgpr_operand("vaddr")),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY,),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("OFFEN", 1),),
        effects=(_GLOBAL_LOAD_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _buffer_load_dword_off_zero_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.buffer_load_dword_off_zero",
        instruction_name="BUFFER_LOAD_DWORD",
        mnemonic="buffer_load_dword",
        encoding_name=encoding_name,
        semantic_tag="memory.load.u32",
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_result()),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
        ),
        implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY,),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(
            ("VADDR", 0),
            ("SOFFSET", _MUBUF_SOFFSET_INLINE_ZERO),
            ("OFFEN", 0),
        ),
        effects=(_GLOBAL_LOAD_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=(),
    )


def _buffer_load_64_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_load_b64",
    instruction_name: str = "BUFFER_LOAD_B64",
    mnemonic: str = "buffer_load_b64",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag="memory.load.u64",
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_result(units=2)),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            AmdgpuOperandOverlay("VADDR", _vgpr_operand("vaddr")),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY_B64,),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("OFFEN", 1),),
        effects=(_GLOBAL_LOAD_B64_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _buffer_load_128_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_load_b128",
    instruction_name: str = "BUFFER_LOAD_B128",
    mnemonic: str = "buffer_load_b128",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag="memory.load.u128",
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_result(units=4)),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            AmdgpuOperandOverlay("VADDR", _vgpr_operand("vaddr")),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY_B128,),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("OFFEN", 1),),
        effects=(_GLOBAL_LOAD_B128_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _buffer_store_dword_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.buffer_store_dword",
        instruction_name="BUFFER_STORE_DWORD",
        mnemonic="buffer_store_dword",
        encoding_name=encoding_name,
        semantic_tag="memory.store.u32",
        schedule_class=_SCHEDULE_VMEM_STORE,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_operand("value")),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            AmdgpuOperandOverlay("VADDR", _vgpr_operand("vaddr")),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_IGNORE_GLOBAL_WRITE_MEMORY,),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("OFFEN", 1),),
        effects=(_GLOBAL_STORE_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _buffer_store_dword_off_zero_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.buffer_store_dword_off_zero",
        instruction_name="BUFFER_STORE_DWORD",
        mnemonic="buffer_store_dword",
        encoding_name=encoding_name,
        semantic_tag="memory.store.u32",
        schedule_class=_SCHEDULE_VMEM_STORE,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_operand("value")),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
        ),
        implicit_operands=(_IGNORE_GLOBAL_WRITE_MEMORY,),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(
            ("VADDR", 0),
            ("SOFFSET", _MUBUF_SOFFSET_INLINE_ZERO),
            ("OFFEN", 0),
        ),
        effects=(_GLOBAL_STORE_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=(),
    )


def _buffer_store_64_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_store_b64",
    instruction_name: str = "BUFFER_STORE_B64",
    mnemonic: str = "buffer_store_b64",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag="memory.store.u64",
        schedule_class=_SCHEDULE_VMEM_STORE,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_operand("value", units=2)),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            AmdgpuOperandOverlay("VADDR", _vgpr_operand("vaddr")),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_IGNORE_GLOBAL_WRITE_MEMORY_B64,),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("OFFEN", 1),),
        effects=(_GLOBAL_STORE_B64_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _buffer_store_128_overlay(
    *,
    descriptor_key: str = "amdgpu.buffer_store_b128",
    instruction_name: str = "BUFFER_STORE_B128",
    mnemonic: str = "buffer_store_b128",
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 12,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag="memory.store.u128",
        schedule_class=_SCHEDULE_VMEM_STORE,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_operand("value", units=4)),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            AmdgpuOperandOverlay("VADDR", _vgpr_operand("vaddr")),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(_IGNORE_GLOBAL_WRITE_MEMORY_B128,),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=(("OFFEN", 1),),
        effects=(_GLOBAL_STORE_B128_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _global_load_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    saddr_off: int | None,
    width_bits: int,
    units: int,
    address_units: int,
    implicit_m0: bool = False,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    implicit_operands: tuple[AmdgpuImplicitOperandOverlay, ...] = (
        _ignore_global_read_memory(width_bits),
    )
    if implicit_m0:
        implicit_operands += (_implicit_m0_input(),)
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        AmdgpuOperandOverlay(data_field_name, _vgpr_result(units=units)),
        AmdgpuOperandOverlay(
            address_field_name, _vgpr_operand("addr", units=address_units)
        ),
    )
    fixed_encoding_fields: tuple[tuple[str, int], ...] = ()
    if saddr_off is None:
        operands += (AmdgpuOperandOverlay("SADDR", _sgpr_operand("saddr", units=2)),)
    else:
        fixed_encoding_fields = (("SADDR", saddr_off),)
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=f"memory.load.u{width_bits}",
        schedule_class=_SCHEDULE_VMEM_LOAD,
        operands=operands,
        implicit_operands=implicit_operands,
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _signed_offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_global_read_effect(width_bits),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=None if saddr_off is None else (),
    )


def _global_store_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    saddr_off: int | None,
    width_bits: int,
    units: int,
    address_units: int,
    implicit_m0: bool = False,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    implicit_operands: tuple[AmdgpuImplicitOperandOverlay, ...] = (
        _ignore_global_write_memory(width_bits),
    )
    if implicit_m0:
        implicit_operands += (_implicit_m0_input(),)
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        AmdgpuOperandOverlay(
            address_field_name, _vgpr_operand("addr", units=address_units)
        ),
        AmdgpuOperandOverlay(data_field_name, _vgpr_operand("value", units=units)),
    )
    fixed_encoding_fields: tuple[tuple[str, int], ...] = ()
    if saddr_off is None:
        operands += (AmdgpuOperandOverlay("SADDR", _sgpr_operand("saddr", units=2)),)
    else:
        fixed_encoding_fields = (("SADDR", saddr_off),)
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=f"memory.store.u{width_bits}",
        schedule_class=_SCHEDULE_VMEM_STORE,
        operands=operands,
        implicit_operands=implicit_operands,
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _signed_offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_global_write_effect(width_bits),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=None if saddr_off is None else (),
    )


def _global_load_lds_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    width_bits: int,
    address_units: int,
    saddr_off: int | None,
    offset_field_name: str = "OFFSET",
    offset_bit_width: int = 13,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr", units=address_units)),
    )
    fixed_encoding_fields: tuple[tuple[str, int], ...] = ()
    if saddr_off is None:
        operands += (AmdgpuOperandOverlay("SADDR", _sgpr_operand("saddr", units=2)),)
    else:
        fixed_encoding_fields = (("SADDR", saddr_off),)
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_FLAT_GLBL",
        semantic_tag=f"memory.global_to_workgroup.u{width_bits}",
        schedule_class=_SCHEDULE_VMEM_LOAD_LDS,
        operands=operands,
        ignored_operands=(
            AmdgpuIgnoredOperandOverlay(
                "VDST",
                ignore_reason="legacy-lds-dma-has-no-vgpr-result",
                fixed_encoding_value=0,
            ),
        ),
        implicit_operands=(_implicit_m0_input(),),
        immediate_fields=(offset_field_name, *_cache_field_names(cache_fields)),
        immediates=(
            _signed_offset_immediate(offset_bit_width),
            *_cache_immediates(cache_fields),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=_global_to_lds_effects(width_bits),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=(),
    )


def _global_load_lds_overlays(
    *,
    descriptor_key_suffix: str = "",
    address_units: int,
    saddr_off: int | None,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    variants = (
        ("DWORD", "dword", 32),
        ("DWORDX3", "dwordx3", 96),
        ("DWORDX4", "dwordx4", 128),
    )
    return tuple(
        _global_load_lds_overlay(
            descriptor_key=(
                f"amdgpu.global_load_lds_{mnemonic_suffix}{descriptor_key_suffix}"
            ),
            instruction_name=f"GLOBAL_LOAD_LDS_{instruction_suffix}",
            mnemonic=f"global_load_lds_{mnemonic_suffix}",
            width_bits=width_bits,
            address_units=address_units,
            saddr_off=saddr_off,
            cache_fields=cache_fields,
        )
        for instruction_suffix, mnemonic_suffix, width_bits in variants
    )


def _global_memory_overlays(
    *,
    instruction_suffixes: tuple[str, str, str],
    mnemonic_suffixes: tuple[str, str, str],
    encoding_name: str,
    address_field_name: str,
    load_data_field_name: str,
    store_data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    saddr_off: int | None,
    address_units: int,
    descriptor_key_suffix: str = "",
    implicit_m0: bool = False,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    widths = ((32, 1), (64, 2), (128, 4))
    return (
        *(
            _global_load_overlay(
                descriptor_key=(
                    f"amdgpu.global_load_b{width_bits}{descriptor_key_suffix}"
                ),
                instruction_name=f"GLOBAL_LOAD_{instruction_suffix}",
                mnemonic=f"global_load_{mnemonic_suffix}",
                encoding_name=encoding_name,
                address_field_name=address_field_name,
                data_field_name=load_data_field_name,
                offset_field_name=offset_field_name,
                offset_bit_width=offset_bit_width,
                saddr_off=saddr_off,
                width_bits=width_bits,
                units=units,
                address_units=address_units,
                implicit_m0=implicit_m0,
                cache_fields=cache_fields,
            )
            for (width_bits, units), instruction_suffix, mnemonic_suffix in zip(
                widths, instruction_suffixes, mnemonic_suffixes, strict=True
            )
        ),
        *(
            _global_store_overlay(
                descriptor_key=(
                    f"amdgpu.global_store_b{width_bits}{descriptor_key_suffix}"
                ),
                instruction_name=f"GLOBAL_STORE_{instruction_suffix}",
                mnemonic=f"global_store_{mnemonic_suffix}",
                encoding_name=encoding_name,
                address_field_name=address_field_name,
                data_field_name=store_data_field_name,
                offset_field_name=offset_field_name,
                offset_bit_width=offset_bit_width,
                saddr_off=saddr_off,
                width_bits=width_bits,
                units=units,
                address_units=address_units,
                implicit_m0=implicit_m0,
                cache_fields=cache_fields,
            )
            for (width_bits, units), instruction_suffix, mnemonic_suffix in zip(
                widths, instruction_suffixes, mnemonic_suffixes, strict=True
            )
        ),
    )


def _global_atomic_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    data_format_name: str,
    returns_old_value: bool,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    return_field_name: str,
    return_field_value: int,
    cache_fields: tuple[tuple[str, int], ...],
    cache_immediate_field_names: tuple[str, ...],
    saddr_off: int | None,
    address_units: int,
) -> AmdgpuDescriptorOverlay:
    schedule_class = (
        _SCHEDULE_VMEM_ATOMIC_RETURN
        if returns_old_value
        else _SCHEDULE_VMEM_ATOMIC_NO_RETURN
    )
    counter_id = _COUNTER_VMEM_LOAD if returns_old_value else _COUNTER_VMEM_STORE
    ignored_operands: tuple[AmdgpuIgnoredOperandOverlay, ...] = ()
    if returns_old_value:
        operands: tuple[AmdgpuOperandOverlay, ...] = (
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay(
                address_field_name, _vgpr_operand("addr", units=address_units)
            ),
            AmdgpuOperandOverlay(data_field_name, _vgpr_operand("value")),
        )
    else:
        operands = (
            AmdgpuOperandOverlay(
                address_field_name, _vgpr_operand("addr", units=address_units)
            ),
            AmdgpuOperandOverlay(data_field_name, _vgpr_operand("value")),
        )
        ignored_operands = (
            AmdgpuIgnoredOperandOverlay(
                "VDST",
                ignore_reason="no-return-global-atomic-has-no-vgpr-result",
                fixed_encoding_value=0,
            ),
        )

    cache_immediate_fields = tuple(
        (field_name, bit_width)
        for field_name, bit_width in cache_fields
        if field_name in cache_immediate_field_names
    )
    cache_immediate_defaults = {
        return_field_name: return_field_value if returns_old_value else 0
    }
    fixed_encoding_fields: tuple[tuple[str, int], ...] = tuple(
        (
            field_name,
            return_field_value
            if returns_old_value and field_name == return_field_name
            else 0,
        )
        for field_name, _bit_width in cache_fields
        if field_name not in cache_immediate_field_names
    )
    if saddr_off is None:
        operands += (AmdgpuOperandOverlay("SADDR", _sgpr_operand("saddr", units=2)),)
    else:
        fixed_encoding_fields += (("SADDR", saddr_off),)
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=schedule_class,
        operands=operands,
        ignored_operands=ignored_operands,
        implicit_operands=(
            _ignore_global_atomic_memory(
                data_format_name=data_format_name, is_input=False
            ),
            _ignore_global_atomic_memory(
                data_format_name=data_format_name, is_input=True
            ),
        ),
        immediate_fields=(
            offset_field_name,
            *_cache_field_names(cache_immediate_fields),
        ),
        immediates=(
            _signed_offset_immediate(offset_bit_width),
            *_cache_immediates_with_defaults(
                cache_immediate_fields, cache_immediate_defaults
            ),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=_global_atomic_effects(32, counter_id=counter_id),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=(),
    )


def _global_atomic_cmpswap_overlay(
    *,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    return_field_name: str,
    return_field_value: int,
    cache_fields: tuple[tuple[str, int], ...],
    cache_immediate_field_names: tuple[str, ...],
    saddr_off: int | None,
    address_units: int,
    descriptor_key_suffix: str,
) -> AmdgpuDescriptorOverlay:
    cache_immediate_fields = tuple(
        (field_name, bit_width)
        for field_name, bit_width in cache_fields
        if field_name in cache_immediate_field_names
    )
    cache_immediate_defaults = {return_field_name: return_field_value}
    fixed_encoding_fields: tuple[tuple[str, int], ...] = tuple(
        (
            field_name,
            return_field_value if field_name == return_field_name else 0,
        )
        for field_name, _bit_width in cache_fields
        if field_name not in cache_immediate_field_names
    )
    operands: tuple[AmdgpuOperandOverlay, ...] = (
        AmdgpuOperandOverlay("VDST", _vgpr_result()),
        AmdgpuOperandOverlay(
            address_field_name, _vgpr_operand("addr", units=address_units)
        ),
        AmdgpuOperandOverlay(data_field_name, _vgpr_operand("value", units=2)),
    )
    if saddr_off is None:
        operands += (AmdgpuOperandOverlay("SADDR", _sgpr_operand("saddr", units=2)),)
    else:
        fixed_encoding_fields += (("SADDR", saddr_off),)
    return AmdgpuDescriptorOverlay(
        descriptor_key=(f"amdgpu.global_atomic_cmpswap_b32_rtn{descriptor_key_suffix}"),
        instruction_name="GLOBAL_ATOMIC_CMPSWAP",
        mnemonic="global_atomic_cmpswap_b32",
        encoding_name=encoding_name,
        semantic_tag="memory.global.atomic.compare_exchange.b32.return",
        schedule_class=_SCHEDULE_VMEM_ATOMIC_RETURN,
        operands=operands,
        implicit_operands=(
            _ignore_global_atomic_memory(
                data_format_name="FMT_NUM_U32", is_input=False
            ),
            _ignore_global_atomic_memory(data_format_name="FMT_NUM_U32", is_input=True),
        ),
        immediate_fields=(
            offset_field_name,
            *_cache_field_names(cache_immediate_fields),
        ),
        immediates=(
            _signed_offset_immediate(offset_bit_width),
            *_cache_immediates_with_defaults(
                cache_immediate_fields, cache_immediate_defaults
            ),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=_global_atomic_effects(32, counter_id=_COUNTER_VMEM_LOAD),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=(),
    )


def _global_atomic_overlays(
    *,
    encoding_name: str,
    address_field_name: str,
    data_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    return_field_name: str,
    return_field_value: int,
    cache_fields: tuple[tuple[str, int], ...],
    cache_immediate_field_names: tuple[str, ...] = (),
    saddr_off: int | None,
    address_units: int,
    descriptor_key_suffix: str = "",
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    rows = (
        ("add_u32", "GLOBAL_ATOMIC_ADD_U32", "add.u32", "FMT_NUM_U32", True),
        ("sub_u32", "GLOBAL_ATOMIC_SUB_U32", "sub.u32", "FMT_NUM_U32", True),
        ("min_i32", "GLOBAL_ATOMIC_MIN_I32", "min.i32", "FMT_NUM_I32", True),
        ("max_i32", "GLOBAL_ATOMIC_MAX_I32", "max.i32", "FMT_NUM_I32", True),
        ("min_u32", "GLOBAL_ATOMIC_MIN_U32", "min.u32", "FMT_NUM_U32", True),
        ("max_u32", "GLOBAL_ATOMIC_MAX_U32", "max.u32", "FMT_NUM_U32", True),
        ("and_b32", "GLOBAL_ATOMIC_AND_B32", "and.b32", "FMT_NUM_B32", True),
        ("or_b32", "GLOBAL_ATOMIC_OR_B32", "or.b32", "FMT_NUM_B32", True),
        ("xor_b32", "GLOBAL_ATOMIC_XOR_B32", "xor.b32", "FMT_NUM_B32", True),
        (
            "swap_b32",
            "GLOBAL_ATOMIC_SWAP_B32",
            "exchange.b32",
            "FMT_NUM_B32",
            False,
        ),
        ("add_f32", "GLOBAL_ATOMIC_ADD_F32", "add.f32", "FMT_NUM_F32", True),
    )
    overlays: list[AmdgpuDescriptorOverlay] = []
    for (
        mnemonic_suffix,
        instruction_name,
        semantic_suffix,
        data_format_name,
        has_no_return_form,
    ) in rows:
        if has_no_return_form:
            overlays.append(
                _global_atomic_overlay(
                    descriptor_key=(
                        f"amdgpu.global_atomic_{mnemonic_suffix}{descriptor_key_suffix}"
                    ),
                    instruction_name=instruction_name,
                    mnemonic=f"global_atomic_{mnemonic_suffix}",
                    semantic_tag=f"memory.global.atomic.{semantic_suffix}",
                    data_format_name=data_format_name,
                    returns_old_value=False,
                    encoding_name=encoding_name,
                    address_field_name=address_field_name,
                    data_field_name=data_field_name,
                    offset_field_name=offset_field_name,
                    offset_bit_width=offset_bit_width,
                    return_field_name=return_field_name,
                    return_field_value=return_field_value,
                    cache_fields=cache_fields,
                    cache_immediate_field_names=cache_immediate_field_names,
                    saddr_off=saddr_off,
                    address_units=address_units,
                )
            )
        overlays.append(
            _global_atomic_overlay(
                descriptor_key=(
                    f"amdgpu.global_atomic_{mnemonic_suffix}_rtn{descriptor_key_suffix}"
                ),
                instruction_name=instruction_name,
                mnemonic=f"global_atomic_{mnemonic_suffix}",
                semantic_tag=f"memory.global.atomic.{semantic_suffix}.return",
                data_format_name=data_format_name,
                returns_old_value=True,
                encoding_name=encoding_name,
                address_field_name=address_field_name,
                data_field_name=data_field_name,
                offset_field_name=offset_field_name,
                offset_bit_width=offset_bit_width,
                return_field_name=return_field_name,
                return_field_value=return_field_value,
                cache_fields=cache_fields,
                cache_immediate_field_names=cache_immediate_field_names,
                saddr_off=saddr_off,
                address_units=address_units,
            )
        )
    overlays.append(
        _global_atomic_cmpswap_overlay(
            encoding_name=encoding_name,
            address_field_name=address_field_name,
            data_field_name=data_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            return_field_name=return_field_name,
            return_field_value=return_field_value,
            cache_fields=cache_fields,
            cache_immediate_field_names=cache_immediate_field_names,
            saddr_off=saddr_off,
            address_units=address_units,
            descriptor_key_suffix=descriptor_key_suffix,
        )
    )
    return tuple(overlays)


def _buffer_atomic_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    data_format_name: str,
    returns_old_value: bool,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    return_field_name: str,
    return_field_value: int,
    cache_field_names: tuple[str, ...],
) -> AmdgpuDescriptorOverlay:
    schedule_class = (
        _SCHEDULE_VMEM_ATOMIC_RETURN
        if returns_old_value
        else _SCHEDULE_VMEM_ATOMIC_NO_RETURN
    )
    counter_id = _COUNTER_VMEM_LOAD if returns_old_value else _COUNTER_VMEM_STORE
    constraints: tuple[Constraint, ...]
    if returns_old_value:
        operands: tuple[AmdgpuOperandOverlay, ...] = (
            AmdgpuOperandOverlay("VDATA", _vgpr_result()),
            AmdgpuOperandOverlay(
                "VDATA",
                _vgpr_operand("value"),
                role_exception_reason=_BUFFER_ATOMIC_VDATA_INPUT_REASON,
            ),
        )
        constraints = _DESTRUCTIVE_BUFFER_ATOMIC_CONSTRAINTS
    else:
        operands = (
            AmdgpuOperandOverlay(
                "VDATA",
                _vgpr_operand("value"),
                role_exception_reason=_BUFFER_ATOMIC_VDATA_INPUT_REASON,
            ),
        )
        constraints = ()

    operands += (
        AmdgpuOperandOverlay(resource_field_name, _sgpr_resource("resource", units=4)),
        AmdgpuOperandOverlay("VADDR", _vgpr_operand("vaddr")),
        AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
    )
    fixed_encoding_fields = tuple(
        (
            field_name,
            return_field_value
            if returns_old_value and field_name == return_field_name
            else 0,
        )
        for field_name in cache_field_names
    )
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=schedule_class,
        operands=operands,
        implicit_operands=(
            _ignore_global_atomic_memory(
                data_format_name=data_format_name, is_input=False
            ),
            _ignore_global_atomic_memory(
                data_format_name=data_format_name, is_input=True
            ),
        ),
        immediate_fields=(offset_field_name,),
        immediates=(_offset_immediate(offset_bit_width),),
        fixed_encoding_fields=(("OFFEN", 1), *fixed_encoding_fields),
        effects=_global_atomic_effects(32, counter_id=counter_id),
        constraints=constraints,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=(),
    )


def _buffer_atomic_cmpswap_overlay(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    return_field_name: str,
    return_field_value: int,
    cache_field_names: tuple[str, ...],
) -> AmdgpuDescriptorOverlay:
    fixed_encoding_fields = tuple(
        (
            field_name,
            return_field_value if field_name == return_field_name else 0,
        )
        for field_name in cache_field_names
    )
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.buffer_atomic_cmpswap_b32_rtn",
        instruction_name="BUFFER_ATOMIC_CMPSWAP",
        mnemonic="buffer_atomic_cmpswap_b32",
        encoding_name=encoding_name,
        semantic_tag="memory.global.atomic.compare_exchange.b32.return",
        schedule_class=_SCHEDULE_VMEM_ATOMIC_RETURN,
        operands=(
            AmdgpuOperandOverlay("VDATA", _vgpr_result(units=2)),
            AmdgpuOperandOverlay(
                "VDATA",
                _vgpr_operand("value", units=2),
                role_exception_reason=_BUFFER_ATOMIC_VDATA_INPUT_REASON,
            ),
            AmdgpuOperandOverlay(
                resource_field_name, _sgpr_resource("resource", units=4)
            ),
            AmdgpuOperandOverlay("VADDR", _vgpr_operand("vaddr")),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        implicit_operands=(
            _ignore_global_atomic_memory(
                data_format_name="FMT_NUM_U32", is_input=False
            ),
            _ignore_global_atomic_memory(data_format_name="FMT_NUM_U32", is_input=True),
        ),
        immediate_fields=(offset_field_name,),
        immediates=(_offset_immediate(offset_bit_width),),
        fixed_encoding_fields=(("OFFEN", 1), *fixed_encoding_fields),
        effects=_global_atomic_effects(32, counter_id=_COUNTER_VMEM_LOAD),
        constraints=_DESTRUCTIVE_BUFFER_ATOMIC_CONSTRAINTS,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=(),
    )


def _buffer_atomic_overlays(
    *,
    encoding_name: str,
    resource_field_name: str,
    offset_field_name: str,
    offset_bit_width: int,
    return_field_name: str,
    return_field_value: int,
    cache_fields: tuple[tuple[str, int], ...],
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    rows = (
        ("add_u32", "BUFFER_ATOMIC_ADD_U32", "add.u32", "FMT_NUM_U32", True),
        ("sub_u32", "BUFFER_ATOMIC_SUB_U32", "sub.u32", "FMT_NUM_U32", True),
        ("min_i32", "BUFFER_ATOMIC_MIN_I32", "min.i32", "FMT_NUM_I32", True),
        ("max_i32", "BUFFER_ATOMIC_MAX_I32", "max.i32", "FMT_NUM_I32", True),
        ("min_u32", "BUFFER_ATOMIC_MIN_U32", "min.u32", "FMT_NUM_U32", True),
        ("max_u32", "BUFFER_ATOMIC_MAX_U32", "max.u32", "FMT_NUM_U32", True),
        ("and_b32", "BUFFER_ATOMIC_AND_B32", "and.b32", "FMT_NUM_B32", True),
        ("or_b32", "BUFFER_ATOMIC_OR_B32", "or.b32", "FMT_NUM_B32", True),
        ("xor_b32", "BUFFER_ATOMIC_XOR_B32", "xor.b32", "FMT_NUM_B32", True),
        (
            "swap_b32",
            "BUFFER_ATOMIC_SWAP_B32",
            "exchange.b32",
            "FMT_NUM_B32",
            False,
        ),
        ("add_f32", "BUFFER_ATOMIC_ADD_F32", "add.f32", "FMT_NUM_F32", True),
    )
    cache_field_names = tuple(field_name for field_name, _ in cache_fields)
    overlays: list[AmdgpuDescriptorOverlay] = []
    for (
        mnemonic_suffix,
        instruction_name,
        semantic_suffix,
        data_format_name,
        has_no_return_form,
    ) in rows:
        if has_no_return_form:
            overlays.append(
                _buffer_atomic_overlay(
                    descriptor_key=f"amdgpu.buffer_atomic_{mnemonic_suffix}",
                    instruction_name=instruction_name,
                    mnemonic=f"buffer_atomic_{mnemonic_suffix}",
                    semantic_tag=f"memory.global.atomic.{semantic_suffix}",
                    data_format_name=data_format_name,
                    returns_old_value=False,
                    encoding_name=encoding_name,
                    resource_field_name=resource_field_name,
                    offset_field_name=offset_field_name,
                    offset_bit_width=offset_bit_width,
                    return_field_name=return_field_name,
                    return_field_value=return_field_value,
                    cache_field_names=cache_field_names,
                )
            )
        overlays.append(
            _buffer_atomic_overlay(
                descriptor_key=f"amdgpu.buffer_atomic_{mnemonic_suffix}_rtn",
                instruction_name=instruction_name,
                mnemonic=f"buffer_atomic_{mnemonic_suffix}",
                semantic_tag=f"memory.global.atomic.{semantic_suffix}.return",
                data_format_name=data_format_name,
                returns_old_value=True,
                encoding_name=encoding_name,
                resource_field_name=resource_field_name,
                offset_field_name=offset_field_name,
                offset_bit_width=offset_bit_width,
                return_field_name=return_field_name,
                return_field_value=return_field_value,
                cache_field_names=cache_field_names,
            )
        )
    overlays.append(
        _buffer_atomic_cmpswap_overlay(
            encoding_name=encoding_name,
            resource_field_name=resource_field_name,
            offset_field_name=offset_field_name,
            offset_bit_width=offset_bit_width,
            return_field_name=return_field_name,
            return_field_value=return_field_value,
            cache_field_names=cache_field_names,
        )
    )
    return tuple(overlays)


def _ds_read_overlay(
    *,
    width_bits: int,
    units: int,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, int], ...] = (("OFFSET1", 0), ("GDS", 0)),
) -> AmdgpuDescriptorOverlay:
    suffix = f"b{width_bits}"
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.ds_read_{suffix}",
        instruction_name=f"DS_READ_{suffix.upper()}",
        mnemonic=f"ds_read_{suffix}",
        encoding_name=encoding_name,
        semantic_tag=f"memory.workgroup.load.u{width_bits}",
        schedule_class=_SCHEDULE_LDS_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result(units=units)),
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
        ),
        implicit_operands=(
            _ignore_workgroup_memory(width_bits=width_bits, is_input=True),
        ),
        immediate_fields=("OFFSET0",),
        immediates=(_offset_immediate(8),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_workgroup_memory_effect(EffectKind.READ, width_bits),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_write_overlay(
    *,
    width_bits: int,
    units: int,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, int], ...] = (("OFFSET1", 0), ("GDS", 0)),
) -> AmdgpuDescriptorOverlay:
    suffix = f"b{width_bits}"
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.ds_write_{suffix}",
        instruction_name=f"DS_WRITE_{suffix.upper()}",
        mnemonic=f"ds_write_{suffix}",
        encoding_name=encoding_name,
        semantic_tag=f"memory.workgroup.store.u{width_bits}",
        schedule_class=_SCHEDULE_LDS_STORE,
        operands=(
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
            AmdgpuOperandOverlay("DATA0", _vgpr_operand("value", units=units)),
        ),
        implicit_operands=(
            _ignore_workgroup_memory(width_bits=width_bits, is_input=False),
        ),
        immediate_fields=("OFFSET0",),
        immediates=(_offset_immediate(8),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_workgroup_memory_effect(EffectKind.WRITE, width_bits),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_read2_overlay(
    *,
    element_width_bits: int,
    value_units: int,
    offset_encoding_id: int | None = None,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, int], ...] = (("GDS", 0),),
) -> AmdgpuDescriptorOverlay:
    suffix = f"b{element_width_bits}"
    if offset_encoding_id is None:
        offset_encoding_id = (
            _ADDRESS_OFFSET_DWORD_ENCODING_ID
            if element_width_bits == 32
            else _ADDRESS_OFFSET_QWORD_ENCODING_ID
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.ds_read2_{suffix}",
        instruction_name=f"DS_READ2_{suffix.upper()}",
        mnemonic=f"ds_read2_{suffix}",
        encoding_name=encoding_name,
        semantic_tag=f"memory.workgroup.load2.u{element_width_bits}",
        schedule_class=_SCHEDULE_LDS_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result(units=value_units * 2)),
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
        ),
        implicit_operands=(
            _ignore_workgroup_memory(width_bits=element_width_bits, is_input=True),
        ),
        immediate_fields=("OFFSET0", "OFFSET1"),
        immediates=(
            _named_offset_immediate("offset0", 8, encoding_id=offset_encoding_id),
            _named_offset_immediate("offset1", 8, encoding_id=offset_encoding_id),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_workgroup_memory_effect(EffectKind.READ, element_width_bits * 2),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_write2_overlay(
    *,
    element_width_bits: int,
    value_units: int,
    offset_encoding_id: int | None = None,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, int], ...] = (("GDS", 0),),
) -> AmdgpuDescriptorOverlay:
    suffix = f"b{element_width_bits}"
    if offset_encoding_id is None:
        offset_encoding_id = (
            _ADDRESS_OFFSET_DWORD_ENCODING_ID
            if element_width_bits == 32
            else _ADDRESS_OFFSET_QWORD_ENCODING_ID
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.ds_write2_{suffix}",
        instruction_name=f"DS_WRITE2_{suffix.upper()}",
        mnemonic=f"ds_write2_{suffix}",
        encoding_name=encoding_name,
        semantic_tag=f"memory.workgroup.store2.u{element_width_bits}",
        schedule_class=_SCHEDULE_LDS_STORE,
        operands=(
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
            AmdgpuOperandOverlay("DATA0", _vgpr_operand("value0", units=value_units)),
            AmdgpuOperandOverlay("DATA1", _vgpr_operand("value1", units=value_units)),
        ),
        implicit_operands=(
            _ignore_workgroup_memory(width_bits=element_width_bits, is_input=False),
        ),
        immediate_fields=("OFFSET0", "OFFSET1"),
        immediates=(
            _named_offset_immediate("offset0", 8, encoding_id=offset_encoding_id),
            _named_offset_immediate("offset1", 8, encoding_id=offset_encoding_id),
        ),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_workgroup_memory_effect(EffectKind.WRITE, element_width_bits * 2),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_atomic_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    data_format_name: str,
    returns_old_value: bool,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, int], ...] = (("OFFSET1", 0), ("GDS", 0)),
) -> AmdgpuDescriptorOverlay:
    operands: tuple[AmdgpuOperandOverlay, ...]
    if returns_old_value:
        operands = (
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
            AmdgpuOperandOverlay("DATA0", _vgpr_operand("value")),
        )
    else:
        operands = (
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
            AmdgpuOperandOverlay("DATA0", _vgpr_operand("value")),
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_LDS_ATOMIC,
        operands=operands,
        implicit_operands=(
            _ignore_workgroup_memory(
                width_bits=32, is_input=False, data_format_name=data_format_name
            ),
            _ignore_workgroup_memory(
                width_bits=32, is_input=True, data_format_name=data_format_name
            ),
        ),
        immediate_fields=("OFFSET0",),
        immediates=(_offset_immediate(8),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(
            _workgroup_memory_effect(EffectKind.READ, 32),
            _workgroup_memory_effect(EffectKind.WRITE, 32),
        ),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_atomic_cmpstore_overlay(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, int], ...] = (("OFFSET1", 0), ("GDS", 0)),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.ds_cmpst_rtn_b32",
        instruction_name="DS_CMPST_RTN_B32",
        mnemonic="ds_cmpst_rtn_b32",
        encoding_name=encoding_name,
        semantic_tag="memory.workgroup.atomic.compare_exchange.b32.return",
        schedule_class=_SCHEDULE_LDS_ATOMIC,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
            AmdgpuOperandOverlay("DATA0", _vgpr_operand("expected")),
            AmdgpuOperandOverlay("DATA1", _vgpr_operand("replacement")),
        ),
        implicit_operands=(
            _ignore_workgroup_memory(
                width_bits=32, is_input=False, data_format_name="FMT_NUM_B32"
            ),
            _ignore_workgroup_memory(
                width_bits=32, is_input=True, data_format_name="FMT_NUM_B32"
            ),
        ),
        immediate_fields=("OFFSET0",),
        immediates=(_offset_immediate(8),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(
            _workgroup_memory_effect(EffectKind.READ, 32),
            _workgroup_memory_effect(EffectKind.WRITE, 32),
        ),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_atomic_overlays(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, int], ...] = (("OFFSET1", 0), ("GDS", 0)),
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    rows = (
        ("ds_add_u32", "DS_ADD_U32", "add.u32", "FMT_NUM_U32", False),
        ("ds_sub_u32", "DS_SUB_U32", "sub.u32", "FMT_NUM_U32", False),
        ("ds_min_i32", "DS_MIN_I32", "min.i32", "FMT_NUM_I32", False),
        ("ds_max_i32", "DS_MAX_I32", "max.i32", "FMT_NUM_I32", False),
        ("ds_min_u32", "DS_MIN_U32", "min.u32", "FMT_NUM_U32", False),
        ("ds_max_u32", "DS_MAX_U32", "max.u32", "FMT_NUM_U32", False),
        ("ds_and_b32", "DS_AND_B32", "and.b32", "FMT_NUM_B32", False),
        ("ds_or_b32", "DS_OR_B32", "or.b32", "FMT_NUM_B32", False),
        ("ds_xor_b32", "DS_XOR_B32", "xor.b32", "FMT_NUM_B32", False),
        ("ds_add_f32", "DS_ADD_F32", "add.f32", "FMT_NUM_F32", False),
        ("ds_add_rtn_u32", "DS_ADD_RTN_U32", "add.return.u32", "FMT_NUM_U32", True),
        ("ds_sub_rtn_u32", "DS_SUB_RTN_U32", "sub.return.u32", "FMT_NUM_U32", True),
        ("ds_min_rtn_i32", "DS_MIN_RTN_I32", "min.return.i32", "FMT_NUM_I32", True),
        ("ds_max_rtn_i32", "DS_MAX_RTN_I32", "max.return.i32", "FMT_NUM_I32", True),
        ("ds_min_rtn_u32", "DS_MIN_RTN_U32", "min.return.u32", "FMT_NUM_U32", True),
        ("ds_max_rtn_u32", "DS_MAX_RTN_U32", "max.return.u32", "FMT_NUM_U32", True),
        ("ds_and_rtn_b32", "DS_AND_RTN_B32", "and.return.b32", "FMT_NUM_B32", True),
        ("ds_or_rtn_b32", "DS_OR_RTN_B32", "or.return.b32", "FMT_NUM_B32", True),
        ("ds_xor_rtn_b32", "DS_XOR_RTN_B32", "xor.return.b32", "FMT_NUM_B32", True),
        (
            "ds_wrxchg_rtn_b32",
            "DS_WRXCHG_RTN_B32",
            "exchange.return.b32",
            "FMT_NUM_B32",
            True,
        ),
        ("ds_add_rtn_f32", "DS_ADD_RTN_F32", "add.return.f32", "FMT_NUM_F32", True),
    )
    overlays = [
        _ds_atomic_overlay(
            descriptor_key=f"amdgpu.{mnemonic}",
            instruction_name=instruction_name,
            mnemonic=mnemonic,
            semantic_tag=f"memory.workgroup.atomic.{semantic_suffix}",
            data_format_name=data_format_name,
            returns_old_value=returns_old_value,
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        )
        for (
            mnemonic,
            instruction_name,
            semantic_suffix,
            data_format_name,
            returns_old_value,
        ) in rows
    ]
    overlays.append(
        _ds_atomic_cmpstore_overlay(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        )
    )
    return tuple(overlays)


def _ds_stride64_read2_overlay(
    *,
    element_width_bits: int,
    value_units: int,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, int], ...] = (("GDS", 0),),
) -> AmdgpuDescriptorOverlay:
    suffix = f"b{element_width_bits}"
    offset_encoding_id = (
        _ADDRESS_OFFSET_DWORD_STRIDE64_ENCODING_ID
        if element_width_bits == 32
        else _ADDRESS_OFFSET_QWORD_STRIDE64_ENCODING_ID
    )
    return replace(
        _ds_read2_overlay(
            element_width_bits=element_width_bits,
            value_units=value_units,
            offset_encoding_id=offset_encoding_id,
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        descriptor_key=f"amdgpu.ds_read2st64_{suffix}",
        instruction_name=f"DS_READ2ST64_{suffix.upper()}",
        mnemonic=f"ds_read2st64_{suffix}",
        semantic_tag=f"memory.workgroup.load2.stride64.u{element_width_bits}",
    )


def _ds_stride64_write2_overlay(
    *,
    element_width_bits: int,
    value_units: int,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, int], ...] = (("GDS", 0),),
) -> AmdgpuDescriptorOverlay:
    suffix = f"b{element_width_bits}"
    offset_encoding_id = (
        _ADDRESS_OFFSET_DWORD_STRIDE64_ENCODING_ID
        if element_width_bits == 32
        else _ADDRESS_OFFSET_QWORD_STRIDE64_ENCODING_ID
    )
    return replace(
        _ds_write2_overlay(
            element_width_bits=element_width_bits,
            value_units=value_units,
            offset_encoding_id=offset_encoding_id,
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        descriptor_key=f"amdgpu.ds_write2st64_{suffix}",
        instruction_name=f"DS_WRITE2ST64_{suffix.upper()}",
        mnemonic=f"ds_write2st64_{suffix}",
        semantic_tag=f"memory.workgroup.store2.stride64.u{element_width_bits}",
    )


def _ds_read_addtid_b32_overlay(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, int], ...] = (("OFFSET1", 0), ("GDS", 0)),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.ds_read_addtid_b32",
        instruction_name="DS_READ_ADDTID_B32",
        mnemonic="ds_read_addtid_b32",
        encoding_name=encoding_name,
        semantic_tag="memory.workgroup.load.addtid.u32",
        schedule_class=_SCHEDULE_LDS_LOAD,
        operands=(AmdgpuOperandOverlay("VDST", _vgpr_result()),),
        implicit_operands=(
            _ignore_workgroup_memory(width_bits=32, is_input=True),
            _implicit_m0_input(),
        ),
        immediate_fields=("OFFSET0",),
        immediates=(_offset_immediate(8),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_workgroup_memory_effect(EffectKind.READ, 32),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_write_addtid_b32_overlay(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, int], ...] = (("OFFSET1", 0), ("GDS", 0)),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.ds_write_addtid_b32",
        instruction_name="DS_WRITE_ADDTID_B32",
        mnemonic="ds_write_addtid_b32",
        encoding_name=encoding_name,
        semantic_tag="memory.workgroup.store.addtid.u32",
        schedule_class=_SCHEDULE_LDS_STORE,
        operands=(AmdgpuOperandOverlay("DATA0", _vgpr_operand("value")),),
        implicit_operands=(
            _ignore_workgroup_memory(width_bits=32, is_input=False),
            _implicit_m0_input(),
        ),
        immediate_fields=("OFFSET0",),
        immediates=(_offset_immediate(8),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_workgroup_memory_effect(EffectKind.WRITE, 32),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_crosslane_offset_immediate() -> Immediate:
    return _offset_immediate(16, encoding_id=_ADDRESS_OFFSET_DS16_ENCODING_ID)


def _ds_swizzle_b32_overlay(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, int], ...] = (("GDS", 0),),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.ds_swizzle_b32",
        instruction_name="DS_SWIZZLE_B32",
        mnemonic="ds_swizzle_b32",
        encoding_name=encoding_name,
        semantic_tag="memory.crosslane.swizzle.u32",
        schedule_class=_SCHEDULE_LDS_CROSSLANE,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
        ),
        immediates=(_ds_crosslane_offset_immediate(),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_CONVERGENT_EFFECT,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _ds_permute_b32_overlay(
    *,
    descriptor_key: str = "amdgpu.ds_permute_b32",
    instruction_name: str = "DS_PERMUTE_B32",
    mnemonic: str = "ds_permute_b32",
    semantic_tag: str = "memory.crosslane.permute.u32",
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, int], ...] = (("GDS", 0),),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_LDS_CROSSLANE,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
            AmdgpuOperandOverlay("DATA0", _vgpr_operand("value")),
        ),
        immediates=(_ds_crosslane_offset_immediate(),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_CONVERGENT_EFFECT,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _ds_bpermute_b32_overlay(
    *,
    descriptor_key: str = "amdgpu.ds_bpermute_b32",
    instruction_name: str = "DS_BPERMUTE_B32",
    mnemonic: str = "ds_bpermute_b32",
    semantic_tag: str = "memory.crosslane.bpermute.u32",
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, int], ...] = (("GDS", 0),),
) -> AmdgpuDescriptorOverlay:
    return _ds_permute_b32_overlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        encoding_name=encoding_name,
        fixed_encoding_fields=fixed_encoding_fields,
    )


def _ds_bpermute_fi_b32_overlay(
    *,
    encoding_name: str = "ENC_VDS",
    fixed_encoding_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return _ds_bpermute_b32_overlay(
        descriptor_key="amdgpu.ds_bpermute_fi_b32",
        instruction_name="DS_BPERMUTE_FI_B32",
        mnemonic="ds_bpermute_fi_b32",
        semantic_tag="memory.crosslane.bpermute.fi.u32",
        encoding_name=encoding_name,
        fixed_encoding_fields=fixed_encoding_fields,
    )


def _ds_crosslane_overlays(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, int], ...] = (("GDS", 0),),
    include_fetch_invalid: bool = False,
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    overlays = (
        _ds_swizzle_b32_overlay(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        _ds_permute_b32_overlay(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        _ds_bpermute_b32_overlay(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
    )
    if not include_fetch_invalid:
        return overlays
    return (
        *overlays,
        _ds_bpermute_fi_b32_overlay(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
    )


def _ds_transpose_read_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    width_bits: int,
    units: int,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, int], ...] = (("GDS", 0),),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_LDS_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result(units=units)),
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
        ),
        implicit_operands=(
            _ignore_workgroup_memory(width_bits=width_bits, is_input=True),
        ),
        immediates=(_ds_crosslane_offset_immediate(),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_workgroup_memory_effect(EffectKind.READ, width_bits),),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _gfx950_ds_transpose_read_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _ds_transpose_read_overlay(
            descriptor_key="amdgpu.ds_read_b64_tr_b4",
            instruction_name="DS_READ_B64_TR_B4",
            mnemonic="ds_read_b64_tr_b4",
            semantic_tag="memory.workgroup.transpose.load.b4.u64",
            width_bits=64,
            units=2,
        ),
        _ds_transpose_read_overlay(
            descriptor_key="amdgpu.ds_read_b96_tr_b6",
            instruction_name="DS_READ_B96_TR_B6",
            mnemonic="ds_read_b96_tr_b6",
            semantic_tag="memory.workgroup.transpose.load.b6.u96",
            width_bits=96,
            units=3,
        ),
        _ds_transpose_read_overlay(
            descriptor_key="amdgpu.ds_read_b64_tr_b8",
            instruction_name="DS_READ_B64_TR_B8",
            mnemonic="ds_read_b64_tr_b8",
            semantic_tag="memory.workgroup.transpose.load.b8.u64",
            width_bits=64,
            units=2,
        ),
        _ds_transpose_read_overlay(
            descriptor_key="amdgpu.ds_read_b64_tr_b16",
            instruction_name="DS_READ_B64_TR_B16",
            mnemonic="ds_read_b64_tr_b16",
            semantic_tag="memory.workgroup.transpose.load.b16.u64",
            width_bits=64,
            units=2,
        ),
    )


def _ds_fixed_fields_without_offset1(
    fixed_encoding_fields: tuple[tuple[str, int], ...],
) -> tuple[tuple[str, int], ...]:
    return tuple(
        fixed_field
        for fixed_field in fixed_encoding_fields
        if fixed_field[0] != "OFFSET1"
    )


def _ds_memory_overlays(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, int], ...] = (("OFFSET1", 0), ("GDS", 0)),
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    widths = ((32, 1), (64, 2), (96, 3), (128, 4))
    two_addr_widths = ((32, 1), (64, 2))
    two_addr_fixed_encoding_fields = _ds_fixed_fields_without_offset1(
        fixed_encoding_fields
    )
    return (
        *(
            _ds_read_overlay(
                width_bits=width_bits,
                units=units,
                encoding_name=encoding_name,
                fixed_encoding_fields=fixed_encoding_fields,
            )
            for width_bits, units in widths
        ),
        *(
            _ds_write_overlay(
                width_bits=width_bits,
                units=units,
                encoding_name=encoding_name,
                fixed_encoding_fields=fixed_encoding_fields,
            )
            for width_bits, units in widths
        ),
        *_ds_atomic_overlays(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        *(
            _ds_read2_overlay(
                element_width_bits=width_bits,
                value_units=units,
                encoding_name=encoding_name,
                fixed_encoding_fields=two_addr_fixed_encoding_fields,
            )
            for width_bits, units in two_addr_widths
        ),
        *(
            _ds_stride64_read2_overlay(
                element_width_bits=width_bits,
                value_units=units,
                encoding_name=encoding_name,
                fixed_encoding_fields=two_addr_fixed_encoding_fields,
            )
            for width_bits, units in two_addr_widths
        ),
        *(
            _ds_write2_overlay(
                element_width_bits=width_bits,
                value_units=units,
                encoding_name=encoding_name,
                fixed_encoding_fields=two_addr_fixed_encoding_fields,
            )
            for width_bits, units in two_addr_widths
        ),
        *(
            _ds_stride64_write2_overlay(
                element_width_bits=width_bits,
                value_units=units,
                encoding_name=encoding_name,
                fixed_encoding_fields=two_addr_fixed_encoding_fields,
            )
            for width_bits, units in two_addr_widths
        ),
        _ds_read_addtid_b32_overlay(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        _ds_write_addtid_b32_overlay(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
    )


def _v_wmma_f32_16x16x16_f16_overlay(
    *, operand_units: int = 4
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_wmma_f32_16x16x16_f16",
        instruction_name="V_WMMA_F32_16X16X16_F16",
        mnemonic="v_wmma_f32_16x16x16_f16",
        encoding_name="ENC_VOP3P",
        semantic_tag="matrix.wmma.f32.16x16x16.f16",
        schedule_class=_SCHEDULE_WMMA,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result(units=8)),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("a", units=operand_units)),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("b", units=operand_units)),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc", units=8)),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_wmma_i32_16x16x16_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    operand_units: int,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_VOP3P",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_WMMA,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result(units=8)),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("a", units=operand_units)),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("b", units=operand_units)),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc", units=8)),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_wmma_i32_16x16x16_iu8_overlay(
    *, operand_units: int = 2
) -> AmdgpuDescriptorOverlay:
    return _v_wmma_i32_16x16x16_overlay(
        descriptor_key="amdgpu.v_wmma_i32_16x16x16_iu8",
        instruction_name="V_WMMA_I32_16X16X16_IU8",
        mnemonic="v_wmma_i32_16x16x16_iu8",
        semantic_tag="matrix.wmma.i32.16x16x16.iu8",
        operand_units=operand_units,
    )


def _v_wmma_i32_16x16x16_iu4_overlay(
    *, operand_units: int = 1
) -> AmdgpuDescriptorOverlay:
    return _v_wmma_i32_16x16x16_overlay(
        descriptor_key="amdgpu.v_wmma_i32_16x16x16_iu4",
        instruction_name="V_WMMA_I32_16X16X16_IU4",
        mnemonic="v_wmma_i32_16x16x16_iu4",
        semantic_tag="matrix.wmma.i32.16x16x16.iu4",
        operand_units=operand_units,
    )


def _v_mfma_f32_16x16x16_f16_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_mfma_f32_16x16x16_f16",
        instruction_name="V_MFMA_F32_16X16X16_F16",
        mnemonic="v_mfma_f32_16x16x16_f16",
        encoding_name="VOP3P_MFMA",
        semantic_tag="matrix.mfma.f32.16x16x16.f16",
        schedule_class=_SCHEDULE_MFMA,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_agpr_result(units=4)),
            AmdgpuOperandOverlay("SRC0", _vgpr_agpr_operand("a", units=2)),
            AmdgpuOperandOverlay("SRC1", _vgpr_agpr_operand("b", units=2)),
            AmdgpuOperandOverlay("SRC2", _vgpr_agpr_const_operand("acc", units=4)),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _vop3p_packed_dot_fixed_fields(
    *,
    op_sel_hi_field: str = "OP_SEL_HI",
    lhs_signed: bool = False,
    rhs_signed: bool = False,
) -> tuple[tuple[str, int], ...]:
    # Packed VOP3P dot instructions use high-half source selection as the
    # canonical no-op modifier spelling. IU dot instructions additionally use
    # NEG low bits as integer signedness selectors for src0/src1.
    neg = (1 if lhs_signed else 0) | (2 if rhs_signed else 0)
    fields = [(op_sel_hi_field, 0x7)]
    if neg != 0:
        fields.append(("NEG", neg))
    return tuple(fields)


def _v_dot4_i32_i8_overlay(
    *, op_sel_hi_field: str = "OP_SEL_HI", signedness_modifiers: bool
) -> AmdgpuDescriptorOverlay:
    instruction_name = "V_DOT4_I32_IU8" if signedness_modifiers else "V_DOT4_I32_I8"
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_dot4_i32_i8",
        instruction_name=instruction_name,
        mnemonic="v_dot4_i32_i8",
        encoding_name="ENC_VOP3P",
        semantic_tag="dot.s8s8.i32x1",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc")),
        ),
        fixed_encoding_fields=_vop3p_packed_dot_fixed_fields(
            op_sel_hi_field=op_sel_hi_field,
            lhs_signed=signedness_modifiers,
            rhs_signed=signedness_modifiers,
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_dot4_i32_iu8_overlay(
    *, op_sel_hi_field: str = "OP_SEL_HI", lhs_signed: bool, rhs_signed: bool
) -> AmdgpuDescriptorOverlay:
    lhs_tag = "s8" if lhs_signed else "u8"
    rhs_tag = "s8" if rhs_signed else "u8"
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_dot4_i32_iu8.{lhs_tag}{rhs_tag}",
        instruction_name="V_DOT4_I32_IU8",
        mnemonic="v_dot4_i32_iu8",
        encoding_name="ENC_VOP3P",
        semantic_tag=f"dot.{lhs_tag}{rhs_tag}.i32x1",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc")),
        ),
        fixed_encoding_fields=_vop3p_packed_dot_fixed_fields(
            op_sel_hi_field=op_sel_hi_field,
            lhs_signed=lhs_signed,
            rhs_signed=rhs_signed,
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
        # Native asm forms cannot yet spell the fixed NEG source selectors, so
        # the target-low descriptor key remains the only unambiguous text form.
        asm_forms=(),
    )


def _v_dot4_u32_u8_overlay(
    *,
    op_sel_hi_field: str = "OP_SEL_HI",
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_dot4_u32_u8",
        instruction_name="V_DOT4_U32_U8",
        mnemonic="v_dot4_u32_u8",
        encoding_name="ENC_VOP3P",
        semantic_tag="dot.u8u8.i32x1",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc")),
        ),
        fixed_encoding_fields=_vop3p_packed_dot_fixed_fields(
            op_sel_hi_field=op_sel_hi_field
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_dot8_i32_i4_overlay(
    *, op_sel_hi_field: str = "OP_SEL_HI", signedness_modifiers: bool
) -> AmdgpuDescriptorOverlay:
    instruction_name = "V_DOT8_I32_IU4" if signedness_modifiers else "V_DOT8_I32_I4"
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_dot8_i32_i4",
        instruction_name=instruction_name,
        mnemonic="v_dot8_i32_i4",
        encoding_name="ENC_VOP3P",
        semantic_tag="dot.s4s4.i32x1",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc")),
        ),
        fixed_encoding_fields=_vop3p_packed_dot_fixed_fields(
            op_sel_hi_field=op_sel_hi_field,
            lhs_signed=signedness_modifiers,
            rhs_signed=signedness_modifiers,
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _v_dot8_i32_iu4_overlay(
    *, op_sel_hi_field: str = "OP_SEL_HI", lhs_signed: bool, rhs_signed: bool
) -> AmdgpuDescriptorOverlay:
    lhs_tag = "s4" if lhs_signed else "u4"
    rhs_tag = "s4" if rhs_signed else "u4"
    return AmdgpuDescriptorOverlay(
        descriptor_key=f"amdgpu.v_dot8_i32_iu4.{lhs_tag}{rhs_tag}",
        instruction_name="V_DOT8_I32_IU4",
        mnemonic="v_dot8_i32_iu4",
        encoding_name="ENC_VOP3P",
        semantic_tag=f"dot.{lhs_tag}{rhs_tag}.i32x1",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc")),
        ),
        fixed_encoding_fields=_vop3p_packed_dot_fixed_fields(
            op_sel_hi_field=op_sel_hi_field,
            lhs_signed=lhs_signed,
            rhs_signed=rhs_signed,
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
        # Native asm forms cannot yet spell the fixed NEG source selectors, so
        # the target-low descriptor key remains the only unambiguous text form.
        asm_forms=(),
    )


def _v_dot8_u32_u4_overlay(
    *,
    op_sel_hi_field: str = "OP_SEL_HI",
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_dot8_u32_u4",
        instruction_name="V_DOT8_U32_U4",
        mnemonic="v_dot8_u32_u4",
        encoding_name="ENC_VOP3P",
        semantic_tag="dot.u4u4.i32x1",
        schedule_class=_SCHEDULE_VALU,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("lhs")),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("rhs")),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc")),
        ),
        fixed_encoding_fields=_vop3p_packed_dot_fixed_fields(
            op_sel_hi_field=op_sel_hi_field
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_waitcnt_overlay(
    *,
    effects: tuple[Effect, ...],
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_waitcnt",
        instruction_name="S_WAITCNT",
        mnemonic="s_waitcnt",
        encoding_name="ENC_SOPP",
        semantic_tag="control.waitcnt",
        schedule_class=_SCHEDULE_WAIT_MEMORY,
        operands=(),
        immediate_fields=("SIMM16",),
        immediates=(_VMCNT_IMMEDIATE, _LGKMCNT_IMMEDIATE),
        effects=effects,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_waitcnt_depctr_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_waitcnt_depctr",
        instruction_name="S_WAITCNT_DEPCTR",
        mnemonic="s_waitcnt_depctr",
        encoding_name="ENC_SOPP",
        semantic_tag="control.waitcnt.alu",
        schedule_class=_SCHEDULE_WAIT_ALU,
        operands=(),
        immediate_fields=("SIMM16",),
        immediates=(_DEPCTR_IMMEDIATE,),
        effects=(_ALU_WAIT_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_waitcnt_vscnt_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_waitcnt_vscnt",
        instruction_name="S_WAITCNT_VSCNT",
        mnemonic="s_waitcnt_vscnt",
        encoding_name="ENC_SOPK",
        semantic_tag="control.waitcnt.vmem_store",
        schedule_class=_SCHEDULE_WAIT_VMEM_STORE,
        operands=(),
        immediate_fields=("SIMM16",),
        immediates=(_VSCNT_IMMEDIATE,),
        fixed_encoding_fields=(("SDST", 124),),
        effects=(_VMEM_STORE_WAIT_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_wait_loadcnt_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_wait_loadcnt",
        instruction_name="S_WAIT_LOADCNT",
        mnemonic="s_wait_loadcnt",
        encoding_name="ENC_SOPP",
        semantic_tag="control.waitcnt.load",
        schedule_class=_SCHEDULE_WAIT_LOAD,
        operands=(),
        immediate_fields=("SIMM16",),
        immediates=(_LOADCNT_IMMEDIATE,),
        effects=(_VMEM_LOAD_WAIT_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_wait_storecnt_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_wait_storecnt",
        instruction_name="S_WAIT_STORECNT",
        mnemonic="s_wait_storecnt",
        encoding_name="ENC_SOPP",
        semantic_tag="control.waitcnt.store",
        schedule_class=_SCHEDULE_WAIT_STORE,
        operands=(),
        immediate_fields=("SIMM16",),
        immediates=(_STORECNT_IMMEDIATE,),
        effects=(_VMEM_STORE_WAIT_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_wait_dscnt_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_wait_dscnt",
        instruction_name="S_WAIT_DSCNT",
        mnemonic="s_wait_dscnt",
        encoding_name="ENC_SOPP",
        semantic_tag="control.waitcnt.lds",
        schedule_class=_SCHEDULE_WAIT_LDS,
        operands=(),
        immediate_fields=("SIMM16",),
        immediates=(_DSCNT_IMMEDIATE,),
        effects=(_LDS_WAIT_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_wait_kmcnt_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_wait_kmcnt",
        instruction_name="S_WAIT_KMCNT",
        mnemonic="s_wait_kmcnt",
        encoding_name="ENC_SOPP",
        semantic_tag="control.waitcnt.smem",
        schedule_class=_SCHEDULE_WAIT_SMEM,
        operands=(),
        immediate_fields=("SIMM16",),
        immediates=(_KMCNT_IMMEDIATE,),
        effects=(_SMEM_WAIT_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_wait_alu_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_wait_alu",
        instruction_name="S_WAIT_ALU",
        mnemonic="s_wait_alu",
        encoding_name="ENC_SOPP",
        semantic_tag="control.waitcnt.alu",
        schedule_class=_SCHEDULE_WAIT_ALU,
        operands=(),
        immediate_fields=("SIMM16",),
        immediates=(_DEPCTR_IMMEDIATE,),
        effects=(_ALU_WAIT_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_wait_idle_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_wait_idle",
        instruction_name="S_WAIT_IDLE",
        mnemonic="s_wait_idle",
        encoding_name="ENC_SOPP",
        semantic_tag="control.waitcnt.idle",
        schedule_class=_SCHEDULE_WAIT_IDLE,
        operands=(),
        effects=(_WAIT_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_barrier_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_barrier",
        instruction_name="S_BARRIER",
        mnemonic="s_barrier",
        encoding_name="ENC_SOPP",
        semantic_tag="control.barrier.workgroup",
        schedule_class=_SCHEDULE_BARRIER,
        operands=(),
        effects=(_WORKGROUP_BARRIER_EFFECT, _CONVERGENT_EFFECT),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _cache_control_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    encoding_name: str,
    semantic_tag: str,
    cache_fields: tuple[tuple[str, int], ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_CACHE_CONTROL,
        operands=(),
        immediate_fields=_cache_field_names(cache_fields),
        immediates=_cache_immediates(cache_fields),
        effects=(_CACHE_CONTROL_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_set_inst_prefetch_distance_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_set_inst_prefetch_distance",
        instruction_name="S_SET_INST_PREFETCH_DISTANCE",
        mnemonic="s_set_inst_prefetch_distance",
        encoding_name="ENC_SOPP",
        semantic_tag="memory.cache.prefetch.instruction.distance",
        schedule_class=_SCHEDULE_CACHE_CONTROL,
        operands=(),
        immediate_fields=("SIMM16",),
        immediates=(_PREFETCH_DISTANCE_IMMEDIATE,),
        effects=(_CACHE_CONTROL_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_dcache_discard_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SMEM",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_CACHE_CONTROL,
        operands=(
            AmdgpuOperandOverlay("SBASE", _sgpr_operand("base", units=2)),
            AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),
        ),
        effects=(_CACHE_CONTROL_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _s_prefetch_overlay(
    *,
    descriptor_key: str,
    instruction_name: str,
    mnemonic: str,
    semantic_tag: str,
    base_operand: Operand | None,
    effect: Effect,
) -> AmdgpuDescriptorOverlay:
    operands = []
    if base_operand is not None:
        operands.append(AmdgpuOperandOverlay("SBASE", base_operand))
    operands.append(AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")))
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=instruction_name,
        mnemonic=mnemonic,
        encoding_name="ENC_SMEM",
        semantic_tag=semantic_tag,
        schedule_class=_SCHEDULE_SMEM_LOAD,
        operands=tuple(operands),
        immediate_fields=("IOFFSET", "SDATA"),
        immediates=(
            _offset_immediate(24),
            _PREFETCH_COUNT_IMMEDIATE,
        ),
        effects=(effect,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _gfx12_prefetch_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_prefetch_overlay(
            descriptor_key="amdgpu.s_prefetch_inst",
            instruction_name="S_PREFETCH_INST",
            mnemonic="s_prefetch_inst",
            semantic_tag="memory.cache.prefetch.instruction",
            base_operand=_sgpr_operand("base", units=2),
            effect=_INSTRUCTION_PREFETCH_EFFECT,
        ),
        _s_prefetch_overlay(
            descriptor_key="amdgpu.s_prefetch_inst_pc_rel",
            instruction_name="S_PREFETCH_INST_PC_REL",
            mnemonic="s_prefetch_inst_pc_rel",
            semantic_tag="memory.cache.prefetch.instruction.pc_relative",
            base_operand=None,
            effect=_INSTRUCTION_PREFETCH_EFFECT,
        ),
        _s_prefetch_overlay(
            descriptor_key="amdgpu.s_prefetch_data",
            instruction_name="S_PREFETCH_DATA",
            mnemonic="s_prefetch_data",
            semantic_tag="memory.cache.prefetch.data",
            base_operand=_sgpr_operand("base", units=2),
            effect=_GLOBAL_PREFETCH_EFFECT,
        ),
        _s_prefetch_overlay(
            descriptor_key="amdgpu.s_buffer_prefetch_data",
            instruction_name="S_BUFFER_PREFETCH_DATA",
            mnemonic="s_buffer_prefetch_data",
            semantic_tag="memory.cache.prefetch.data.buffer",
            base_operand=_sgpr_resource("resource", units=4),
            effect=_GLOBAL_PREFETCH_EFFECT,
        ),
        _s_prefetch_overlay(
            descriptor_key="amdgpu.s_prefetch_data_pc_rel",
            instruction_name="S_PREFETCH_DATA_PC_REL",
            mnemonic="s_prefetch_data_pc_rel",
            semantic_tag="memory.cache.prefetch.data.pc_relative",
            base_operand=None,
            effect=_GLOBAL_PREFETCH_EFFECT,
        ),
    )


def _gfx950_cache_control_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _cache_control_overlay(
            descriptor_key="amdgpu.buffer_inv",
            instruction_name="BUFFER_INV",
            mnemonic="buffer_inv",
            encoding_name="ENC_MUBUF",
            semantic_tag="memory.cache.invalidate.buffer",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.buffer_wbl2",
            instruction_name="BUFFER_WBL2",
            mnemonic="buffer_wbl2",
            encoding_name="ENC_MUBUF",
            semantic_tag="memory.cache.writeback.buffer.l2",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.s_dcache_inv",
            instruction_name="S_DCACHE_INV",
            mnemonic="s_dcache_inv",
            encoding_name="ENC_SMEM",
            semantic_tag="memory.cache.invalidate.data",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.s_dcache_wb",
            instruction_name="S_DCACHE_WB",
            mnemonic="s_dcache_wb",
            encoding_name="ENC_SMEM",
            semantic_tag="memory.cache.writeback.data",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.s_dcache_inv_vol",
            instruction_name="S_DCACHE_INV_VOL",
            mnemonic="s_dcache_inv_vol",
            encoding_name="ENC_SMEM",
            semantic_tag="memory.cache.invalidate.data.volatile",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.s_dcache_wb_vol",
            instruction_name="S_DCACHE_WB_VOL",
            mnemonic="s_dcache_wb_vol",
            encoding_name="ENC_SMEM",
            semantic_tag="memory.cache.writeback.data.volatile",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.s_icache_inv",
            instruction_name="S_ICACHE_INV",
            mnemonic="s_icache_inv",
            encoding_name="ENC_SOPP",
            semantic_tag="memory.cache.invalidate.instruction",
        ),
        _s_dcache_discard_overlay(
            descriptor_key="amdgpu.s_dcache_discard",
            instruction_name="S_DCACHE_DISCARD",
            mnemonic="s_dcache_discard",
            semantic_tag="memory.cache.discard.data",
        ),
        _s_dcache_discard_overlay(
            descriptor_key="amdgpu.s_dcache_discard_x2",
            instruction_name="S_DCACHE_DISCARD_X2",
            mnemonic="s_dcache_discard_x2",
            semantic_tag="memory.cache.discard.data.x2",
        ),
    )


def _gfx11_cache_control_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _cache_control_overlay(
            descriptor_key="amdgpu.buffer_gl0_inv",
            instruction_name="BUFFER_GL0_INV",
            mnemonic="buffer_gl0_inv",
            encoding_name="ENC_MUBUF",
            semantic_tag="memory.cache.invalidate.buffer.gl0",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.buffer_gl1_inv",
            instruction_name="BUFFER_GL1_INV",
            mnemonic="buffer_gl1_inv",
            encoding_name="ENC_MUBUF",
            semantic_tag="memory.cache.invalidate.buffer.gl1",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.s_dcache_inv",
            instruction_name="S_DCACHE_INV",
            mnemonic="s_dcache_inv",
            encoding_name="ENC_SMEM",
            semantic_tag="memory.cache.invalidate.data",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.s_gl1_inv",
            instruction_name="S_GL1_INV",
            mnemonic="s_gl1_inv",
            encoding_name="ENC_SMEM",
            semantic_tag="memory.cache.invalidate.global.l1",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.s_icache_inv",
            instruction_name="S_ICACHE_INV",
            mnemonic="s_icache_inv",
            encoding_name="ENC_SOPP",
            semantic_tag="memory.cache.invalidate.instruction",
        ),
        _s_set_inst_prefetch_distance_overlay(),
    )


def _gfx12_cache_control_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _cache_control_overlay(
            descriptor_key="amdgpu.global_inv",
            instruction_name="GLOBAL_INV",
            mnemonic="global_inv",
            encoding_name="ENC_VGLOBAL",
            semantic_tag="memory.cache.invalidate.global",
            cache_fields=(("SCOPE", 2),),
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.global_wb",
            instruction_name="GLOBAL_WB",
            mnemonic="global_wb",
            encoding_name="ENC_VGLOBAL",
            semantic_tag="memory.cache.writeback.global",
            cache_fields=(("SCOPE", 2),),
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.global_wbinv",
            instruction_name="GLOBAL_WBINV",
            mnemonic="global_wbinv",
            encoding_name="ENC_VGLOBAL",
            semantic_tag="memory.cache.writeback_invalidate.global",
            cache_fields=(("SCOPE", 2),),
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.s_dcache_inv",
            instruction_name="S_DCACHE_INV",
            mnemonic="s_dcache_inv",
            encoding_name="ENC_SMEM",
            semantic_tag="memory.cache.invalidate.data",
        ),
        _cache_control_overlay(
            descriptor_key="amdgpu.s_icache_inv",
            instruction_name="S_ICACHE_INV",
            mnemonic="s_icache_inv",
            encoding_name="ENC_SOPP",
            semantic_tag="memory.cache.invalidate.instruction",
        ),
    )


def _gfx950_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_add_u32_overlay(),
        _s_sub_u32_overlay(),
        _v_add_u32_overlay("V_ADD_U32"),
        _v_sub_u32_overlay("V_SUB_U32", "v_sub_u32"),
        _v_mov_b32_literal_overlay(),
        _v_mul_lo_u32_overlay(),
        *_i32_bitwise_shift_overlays(),
        _v_add_f32_overlay(),
        _v_sub_f32_overlay(),
        _v_mul_f32_overlay(),
        _v_min_f32_overlay(),
        _v_max_f32_overlay(),
        _v_fma_f32_overlay(),
        _v_cvt_f32_i32_overlay(),
        _v_cvt_f32_u32_overlay(),
        *_v_cmp_overlays(),
        _v_cndmask_b32_overlay(),
        _s_load_dwordx2_overlay(),
        _s_buffer_load_dword_overlay(),
        _s_buffer_load_64_overlay(
            descriptor_key="amdgpu.s_buffer_load_dwordx2",
            instruction_name="S_BUFFER_LOAD_DWORDX2",
            mnemonic="s_buffer_load_dwordx2",
        ),
        _buffer_load_dword_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_dword_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_64_overlay(
            descriptor_key="amdgpu.buffer_load_dwordx2",
            instruction_name="BUFFER_LOAD_DWORDX2",
            mnemonic="buffer_load_dwordx2",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_128_overlay(
            descriptor_key="amdgpu.buffer_load_dwordx4",
            instruction_name="BUFFER_LOAD_DWORDX4",
            mnemonic="buffer_load_dwordx4",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_dword_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_dword_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_64_overlay(
            descriptor_key="amdgpu.buffer_store_dwordx2",
            instruction_name="BUFFER_STORE_DWORDX2",
            mnemonic="buffer_store_dwordx2",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_128_overlay(
            descriptor_key="amdgpu.buffer_store_dwordx4",
            instruction_name="BUFFER_STORE_DWORDX4",
            mnemonic="buffer_store_dwordx4",
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_global_memory_overlays(
            instruction_suffixes=("DWORD", "DWORDX2", "DWORDX4"),
            mnemonic_suffixes=("dword", "dwordx2", "dwordx4"),
            encoding_name="ENC_FLAT_GLBL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=_GLOBAL_GFX950_SADDR_OFF,
            address_units=2,
            implicit_m0=True,
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_global_memory_overlays(
            instruction_suffixes=("DWORD", "DWORDX2", "DWORDX4"),
            mnemonic_suffixes=("dword", "dwordx2", "dwordx4"),
            encoding_name="ENC_FLAT_GLBL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            implicit_m0=True,
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_global_load_lds_overlays(
            address_units=2,
            saddr_off=_GLOBAL_GFX950_SADDR_OFF,
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_global_load_lds_overlays(
            address_units=1,
            saddr_off=None,
            descriptor_key_suffix="_saddr",
            cache_fields=_GFX950_VECTOR_CACHE_FIELDS,
        ),
        *_ds_memory_overlays(),
        *_ds_crosslane_overlays(),
        _v_dot4_i32_i8_overlay(signedness_modifiers=False),
        _v_dot4_u32_u8_overlay(),
        _v_dot8_i32_i4_overlay(signedness_modifiers=False),
        _v_dot8_u32_u4_overlay(),
        *_gfx950_ds_transpose_read_overlays(),
        _v_mfma_f32_16x16x16_f16_overlay(),
        _s_barrier_overlay(),
        *_gfx950_cache_control_overlays(),
        _s_waitcnt_overlay(
            effects=(
                _VMEM_LOAD_WAIT_EFFECT,
                _VMEM_STORE_WAIT_EFFECT,
                _LDS_WAIT_EFFECT,
                _SMEM_WAIT_EFFECT,
            )
        ),
    )


def _gfx950_core_overlay_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    return materialize_amdgpu_descriptor_overlays(spec, _gfx950_core_overlays())


def _gfx11_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_add_u32_overlay(),
        _s_sub_u32_overlay(),
        _v_add_u32_overlay("V_ADD_NC_U32"),
        _v_sub_u32_overlay("V_SUB_NC_U32", "v_sub_nc_u32"),
        _v_mov_b32_literal_overlay(),
        _v_mul_lo_u32_overlay(),
        *_i32_bitwise_shift_overlays(),
        _v_add_f32_overlay(),
        _v_sub_f32_overlay(),
        _v_mul_f32_overlay(),
        _v_min_f32_overlay(),
        _v_max_f32_overlay(),
        _v_fma_f32_overlay(),
        _v_cvt_f32_i32_overlay(),
        _v_cvt_f32_u32_overlay(),
        *_v_cmp_overlays(),
        _v_cndmask_b32_overlay(),
        _s_load_dwordx2_overlay(),
        _s_buffer_load_dword_overlay(),
        _s_buffer_load_64_overlay(),
        _buffer_load_dword_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_dword_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_64_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_128_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_dword_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_dword_off_zero_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_64_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_128_overlay(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_buffer_atomic_overlays(
            encoding_name="ENC_MUBUF",
            resource_field_name="SRSRC",
            offset_field_name="OFFSET",
            offset_bit_width=12,
            return_field_name="GLC",
            return_field_value=1,
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_global_memory_overlays(
            instruction_suffixes=("B32", "B64", "B128"),
            mnemonic_suffixes=("b32", "b64", "b128"),
            encoding_name="ENC_FLAT_GLOBAL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=_GLOBAL_SADDR_OFF,
            address_units=2,
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_global_memory_overlays(
            instruction_suffixes=("B32", "B64", "B128"),
            mnemonic_suffixes=("b32", "b64", "b128"),
            encoding_name="ENC_FLAT_GLOBAL",
            address_field_name="ADDR",
            load_data_field_name="VDST",
            store_data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
        ),
        *_global_atomic_overlays(
            encoding_name="ENC_FLAT_GLOBAL",
            address_field_name="ADDR",
            data_field_name="DATA",
            offset_field_name="OFFSET",
            offset_bit_width=13,
            return_field_name="GLC",
            return_field_value=1,
            cache_fields=_GFX9_11_VECTOR_CACHE_FIELDS,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
        ),
        *_ds_memory_overlays(),
        *_ds_crosslane_overlays(),
        _v_dot4_i32_i8_overlay(signedness_modifiers=True),
        _v_dot4_i32_iu8_overlay(lhs_signed=True, rhs_signed=False),
        _v_dot4_i32_iu8_overlay(lhs_signed=False, rhs_signed=True),
        _v_dot4_u32_u8_overlay(),
        _v_dot8_i32_i4_overlay(signedness_modifiers=True),
        _v_dot8_i32_iu4_overlay(lhs_signed=True, rhs_signed=False),
        _v_dot8_i32_iu4_overlay(lhs_signed=False, rhs_signed=True),
        _v_dot8_u32_u4_overlay(),
        _v_wmma_f32_16x16x16_f16_overlay(operand_units=8),
        _v_wmma_i32_16x16x16_iu8_overlay(operand_units=4),
        _v_wmma_i32_16x16x16_iu4_overlay(operand_units=2),
        _s_barrier_overlay(),
        *_gfx11_cache_control_overlays(),
        _s_waitcnt_overlay(
            effects=(
                _VMEM_LOAD_WAIT_EFFECT,
                _LDS_WAIT_EFFECT,
                _SMEM_WAIT_EFFECT,
            )
        ),
        _s_waitcnt_vscnt_overlay(),
        _s_waitcnt_depctr_overlay(),
        _s_wait_idle_overlay(),
    )


def _gfx11_core_overlay_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    return materialize_amdgpu_descriptor_overlays(spec, _gfx11_core_overlays())


def _gfx12_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_add_u32_overlay(),
        _s_sub_u32_overlay(),
        _v_add_u32_overlay("V_ADD_NC_U32"),
        _v_sub_u32_overlay("V_SUB_NC_U32", "v_sub_nc_u32"),
        _v_mov_b32_literal_overlay(),
        _v_mul_lo_u32_overlay(),
        *_i32_bitwise_shift_overlays(),
        _v_add_f32_overlay(),
        _v_sub_f32_overlay(),
        _v_mul_f32_overlay(),
        _v_min_f32_overlay(),
        _v_max_f32_overlay(),
        _v_fma_f32_overlay(),
        _v_cvt_f32_i32_overlay(),
        _v_cvt_f32_u32_overlay(),
        *_v_cmp_overlays(),
        _v_cndmask_b32_overlay(),
        _s_load_dwordx2_overlay("IOFFSET", offset_bit_width=24),
        _s_buffer_load_dword_overlay("IOFFSET", offset_bit_width=24),
        _s_buffer_load_64_overlay(offset_field_name="IOFFSET", offset_bit_width=24),
        _buffer_load_dword_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_64_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_128_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_dword_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_64_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_128_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_memory_overlays(
            instruction_suffixes=("B32", "B64", "B128"),
            mnemonic_suffixes=("b32", "b64", "b128"),
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            saddr_off=_GLOBAL_SADDR_OFF,
            address_units=2,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_memory_overlays(
            instruction_suffixes=("B32", "B64", "B128"),
            mnemonic_suffixes=("b32", "b64", "b128"),
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_atomic_overlays(
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            return_field_name="TH",
            return_field_value=_GFX12_TH_ATOMIC_RETURN_VALUE,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            cache_immediate_field_names=_GFX12_ATOMIC_CACHE_IMMEDIATE_FIELDS,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
        ),
        *_ds_memory_overlays(
            encoding_name="ENC_VDS",
            fixed_encoding_fields=(("OFFSET1", 0),),
        ),
        *_ds_crosslane_overlays(
            encoding_name="ENC_VDS",
            fixed_encoding_fields=(),
            include_fetch_invalid=True,
        ),
        _v_dot4_i32_i8_overlay(op_sel_hi_field="OPSEL_HI", signedness_modifiers=True),
        _v_dot4_i32_iu8_overlay(
            op_sel_hi_field="OPSEL_HI", lhs_signed=True, rhs_signed=False
        ),
        _v_dot4_i32_iu8_overlay(
            op_sel_hi_field="OPSEL_HI", lhs_signed=False, rhs_signed=True
        ),
        _v_dot4_u32_u8_overlay(op_sel_hi_field="OPSEL_HI"),
        _v_dot8_i32_i4_overlay(op_sel_hi_field="OPSEL_HI", signedness_modifiers=True),
        _v_dot8_i32_iu4_overlay(
            op_sel_hi_field="OPSEL_HI", lhs_signed=True, rhs_signed=False
        ),
        _v_dot8_i32_iu4_overlay(
            op_sel_hi_field="OPSEL_HI", lhs_signed=False, rhs_signed=True
        ),
        _v_dot8_u32_u4_overlay(op_sel_hi_field="OPSEL_HI"),
        _v_wmma_f32_16x16x16_f16_overlay(),
        _v_wmma_i32_16x16x16_iu8_overlay(),
        _v_wmma_i32_16x16x16_iu4_overlay(),
        *_gfx12_cache_control_overlays(),
        *_gfx12_prefetch_overlays(),
        _s_wait_loadcnt_overlay(),
        _s_wait_storecnt_overlay(),
        _s_wait_dscnt_overlay(),
        _s_wait_kmcnt_overlay(),
        _s_wait_alu_overlay(),
        _s_wait_idle_overlay(),
    )


def _gfx12_core_overlay_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    return materialize_amdgpu_descriptor_overlays(spec, _gfx12_core_overlays())


def _gfx1250_core_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _s_add_u32_overlay(),
        _s_sub_u32_overlay(),
        _v_add_u32_overlay("V_ADD_NC_U32"),
        _v_sub_u32_overlay("V_SUB_NC_U32", "v_sub_nc_u32"),
        _v_mov_b32_literal_overlay(),
        _v_mul_lo_u32_overlay(),
        *_i32_bitwise_shift_overlays(),
        _v_add_f32_overlay(),
        _v_sub_f32_overlay(),
        _v_mul_f32_overlay(),
        _v_min_f32_overlay(),
        _v_max_f32_overlay(),
        _v_fma_f32_overlay(),
        _v_cvt_f32_i32_overlay(),
        _v_cvt_f32_u32_overlay(),
        *_v_cmp_overlays(),
        _v_cndmask_b32_overlay(),
        _s_load_dwordx2_overlay("IOFFSET", offset_bit_width=24),
        _s_buffer_load_dword_overlay("IOFFSET", offset_bit_width=24),
        _s_buffer_load_64_overlay(offset_field_name="IOFFSET", offset_bit_width=24),
        _buffer_load_dword_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_64_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        _buffer_load_128_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_dword_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_64_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        _buffer_store_128_overlay(
            encoding_name="ENC_VBUFFER",
            resource_field_name="RSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_memory_overlays(
            instruction_suffixes=("B32", "B64", "B128"),
            mnemonic_suffixes=("b32", "b64", "b128"),
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            saddr_off=_GLOBAL_SADDR_OFF,
            address_units=2,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_memory_overlays(
            instruction_suffixes=("B32", "B64", "B128"),
            mnemonic_suffixes=("b32", "b64", "b128"),
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            load_data_field_name="VDST",
            store_data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
        ),
        *_global_atomic_overlays(
            encoding_name="ENC_VGLOBAL",
            address_field_name="VADDR",
            data_field_name="VSRC",
            offset_field_name="IOFFSET",
            offset_bit_width=24,
            return_field_name="TH",
            return_field_value=_GFX12_TH_ATOMIC_RETURN_VALUE,
            cache_fields=_GFX12_VECTOR_CACHE_FIELDS,
            cache_immediate_field_names=_GFX12_ATOMIC_CACHE_IMMEDIATE_FIELDS,
            saddr_off=None,
            address_units=1,
            descriptor_key_suffix="_saddr",
        ),
        *_ds_memory_overlays(
            encoding_name="ENC_VDS",
            fixed_encoding_fields=(("OFFSET1", 0),),
        ),
        *_ds_crosslane_overlays(
            encoding_name="ENC_VDS",
            fixed_encoding_fields=(),
            include_fetch_invalid=True,
        ),
        _v_dot4_i32_i8_overlay(op_sel_hi_field="OPSEL_HI", signedness_modifiers=True),
        _v_dot4_i32_iu8_overlay(
            op_sel_hi_field="OPSEL_HI", lhs_signed=True, rhs_signed=False
        ),
        _v_dot4_i32_iu8_overlay(
            op_sel_hi_field="OPSEL_HI", lhs_signed=False, rhs_signed=True
        ),
        _v_dot4_u32_u8_overlay(op_sel_hi_field="OPSEL_HI"),
        _v_dot8_i32_i4_overlay(op_sel_hi_field="OPSEL_HI", signedness_modifiers=True),
        _v_dot8_i32_iu4_overlay(
            op_sel_hi_field="OPSEL_HI", lhs_signed=True, rhs_signed=False
        ),
        _v_dot8_i32_iu4_overlay(
            op_sel_hi_field="OPSEL_HI", lhs_signed=False, rhs_signed=True
        ),
        _v_dot8_u32_u4_overlay(op_sel_hi_field="OPSEL_HI"),
        _v_wmma_f32_16x16x16_f16_overlay(),
        _v_wmma_i32_16x16x16_iu8_overlay(),
        _v_wmma_i32_16x16x16_iu4_overlay(),
        *_gfx12_cache_control_overlays(),
        *_gfx12_prefetch_overlays(),
        _s_wait_loadcnt_overlay(),
        _s_wait_storecnt_overlay(),
        _s_wait_dscnt_overlay(),
        _s_wait_kmcnt_overlay(),
        _s_wait_alu_overlay(),
        _s_wait_idle_overlay(),
    )


def _gfx1250_core_overlay_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    return materialize_amdgpu_descriptor_overlays(spec, _gfx1250_core_overlays())


_AMDGPU_GFX950_CORE_DESCRIPTOR_SET_BASE = DescriptorSet(
    key="amdgpu.gfx950.core",
    target_key="amdgpu",
    feature_key="amdgpu.gfx950.v1",
    c_header_path=Path("loom/src/loom/target/arch/amdgpu/gfx950_descriptors.h"),
    c_source_path=Path("loom/src/loom/target/arch/amdgpu/gfx950_descriptors.c"),
    header_guard="LOOM_TARGET_ARCH_AMDGPU_GFX950_DESCRIPTORS_H_",
    public_header="loom/target/arch/amdgpu/gfx950_descriptors.h",
    function_name="loom_amdgpu_gfx950_core_descriptor_set",
    c_table_prefix="AmdgpuGfx950Core",
    c_enum_prefix="AMDGPU_GFX950_CORE",
    generator_version=1,
    reg_classes=(
        RegClass(
            _REG_SGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=106,
        ),
        RegClass(
            _REG_VGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=1024,
        ),
        RegClass(
            _REG_AGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=256,
        ),
        RegClass(
            _REG_M0,
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            physical_count=1,
        ),
    ),
    resources=(
        *_common_scalar_vector_memory_resources(),
        Resource(_RESOURCE_MFMA, capacity_per_cycle=1, kind=ResourceKind.MATRIX),
        Resource(_RESOURCE_CONTROL, capacity_per_cycle=1, kind=ResourceKind.CONTROL),
    ),
    schedule_classes=(
        *_common_scalar_vector_memory_schedule_classes(
            smem_load_hazards=_SMEM_WAIT_HAZARDS,
            vmem_load_hazards=_VMEM_LOAD_WAIT_HAZARDS,
            vmem_store_hazards=_VMEM_STORE_WAIT_HAZARDS,
            lds_load_hazards=_LDS_WAIT_HAZARDS,
            lds_store_hazards=_LDS_WAIT_HAZARDS,
            lds_atomic_hazards=_LDS_WAIT_HAZARDS,
            lds_crosslane_hazards=_LDS_WAIT_HAZARDS,
        ),
        ScheduleClass(
            _SCHEDULE_VMEM_LOAD_LDS,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=16,
            issue_uses=(
                IssueUse(_RESOURCE_VMEM_LOAD, cycles=1, units=1),
                IssueUse(_RESOURCE_LDS_STORE, cycles=1, units=1),
            ),
            hazards=_VMEM_LOAD_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.MAY_LOAD, ScheduleClassFlag.MAY_STORE),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_MFMA,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=32,
            issue_uses=(IssueUse(_RESOURCE_MFMA, cycles=1, units=1),),
            hazards=_matrix_hazards(_RESOURCE_MFMA),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_MEMORY,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_GFX950_MEMORY_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
    ),
    descriptors=_s_mov_b32_descriptors(),
)


_AMDGPU_GFX11_CORE_DESCRIPTOR_SET_BASE = DescriptorSet(
    key="amdgpu.gfx11.core",
    target_key="amdgpu",
    feature_key="amdgpu.gfx11.v1",
    c_header_path=Path("loom/src/loom/target/arch/amdgpu/gfx11_descriptors.h"),
    c_source_path=Path("loom/src/loom/target/arch/amdgpu/gfx11_descriptors.c"),
    header_guard="LOOM_TARGET_ARCH_AMDGPU_GFX11_DESCRIPTORS_H_",
    public_header="loom/target/arch/amdgpu/gfx11_descriptors.h",
    function_name="loom_amdgpu_gfx11_core_descriptor_set",
    c_table_prefix="AmdgpuGfx11Core",
    c_enum_prefix="AMDGPU_GFX11_CORE",
    generator_version=1,
    reg_classes=(
        RegClass(
            _REG_SGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=106,
        ),
        RegClass(
            _REG_VGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=1024,
        ),
        RegClass(
            _REG_M0,
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            physical_count=1,
        ),
    ),
    resources=(
        *_common_scalar_vector_memory_resources(),
        Resource(_RESOURCE_WMMA, capacity_per_cycle=1, kind=ResourceKind.MATRIX),
        Resource(_RESOURCE_CONTROL, capacity_per_cycle=1, kind=ResourceKind.CONTROL),
    ),
    schedule_classes=(
        *_common_scalar_vector_memory_schedule_classes(
            smem_load_hazards=_SMEM_WAIT_HAZARDS,
            vmem_load_hazards=_VMEM_LOAD_WAIT_HAZARDS,
            vmem_store_hazards=_VMEM_STORE_WAIT_HAZARDS,
            lds_load_hazards=_LDS_WAIT_HAZARDS,
            lds_store_hazards=_LDS_WAIT_HAZARDS,
            lds_atomic_hazards=_LDS_WAIT_HAZARDS,
            lds_crosslane_hazards=_LDS_WAIT_HAZARDS,
        ),
        ScheduleClass(
            _SCHEDULE_WMMA,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=32,
            issue_uses=(IssueUse(_RESOURCE_WMMA, cycles=1, units=1),),
            hazards=_matrix_hazards(_RESOURCE_WMMA),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_MEMORY,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_GFX11_MEMORY_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_VMEM_STORE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_VMEM_STORE_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_ALU,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_ALU_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_IDLE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_IDLE_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
    ),
    descriptors=_s_mov_b32_descriptors(),
)

_AMDGPU_GFX12_CORE_DESCRIPTOR_SET_BASE = DescriptorSet(
    key="amdgpu.gfx12.core",
    target_key="amdgpu",
    feature_key="amdgpu.gfx12.v1",
    c_header_path=Path("loom/src/loom/target/arch/amdgpu/gfx12_descriptors.h"),
    c_source_path=Path("loom/src/loom/target/arch/amdgpu/gfx12_descriptors.c"),
    header_guard="LOOM_TARGET_ARCH_AMDGPU_GFX12_DESCRIPTORS_H_",
    public_header="loom/target/arch/amdgpu/gfx12_descriptors.h",
    function_name="loom_amdgpu_gfx12_core_descriptor_set",
    c_table_prefix="AmdgpuGfx12Core",
    c_enum_prefix="AMDGPU_GFX12_CORE",
    generator_version=1,
    reg_classes=(
        RegClass(
            _REG_SGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=106,
        ),
        RegClass(
            _REG_VGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=1024,
        ),
        RegClass(
            _REG_M0,
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            physical_count=1,
        ),
    ),
    resources=(
        *_common_scalar_vector_memory_resources(),
        Resource(_RESOURCE_WMMA, capacity_per_cycle=1, kind=ResourceKind.MATRIX),
        Resource(_RESOURCE_CONTROL, capacity_per_cycle=1, kind=ResourceKind.CONTROL),
    ),
    schedule_classes=(
        *_common_scalar_vector_memory_schedule_classes(
            smem_load_hazards=_SMEM_WAIT_HAZARDS,
            vmem_load_hazards=_VMEM_LOAD_WAIT_HAZARDS,
            vmem_store_hazards=_VMEM_STORE_WAIT_HAZARDS,
            lds_load_hazards=_LDS_WAIT_HAZARDS,
            lds_store_hazards=_LDS_WAIT_HAZARDS,
            lds_atomic_hazards=_LDS_WAIT_HAZARDS,
            lds_crosslane_hazards=_LDS_WAIT_HAZARDS,
        ),
        ScheduleClass(
            _SCHEDULE_WMMA,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=32,
            issue_uses=(IssueUse(_RESOURCE_WMMA, cycles=1, units=1),),
            hazards=_matrix_hazards(_RESOURCE_WMMA),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_LOAD,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_VMEM_LOAD_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_STORE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_VMEM_STORE_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_LDS,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_LDS_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_SMEM,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_SMEM_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_ALU,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_ALU_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_IDLE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_IDLE_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
    ),
    descriptors=_s_mov_b32_descriptors(),
)

_AMDGPU_GFX1250_CORE_DESCRIPTOR_SET_BASE = DescriptorSet(
    key="amdgpu.gfx1250.core",
    target_key="amdgpu",
    feature_key="amdgpu.gfx1250.v1",
    c_header_path=Path("loom/src/loom/target/arch/amdgpu/gfx1250_descriptors.h"),
    c_source_path=Path("loom/src/loom/target/arch/amdgpu/gfx1250_descriptors.c"),
    header_guard="LOOM_TARGET_ARCH_AMDGPU_GFX1250_DESCRIPTORS_H_",
    public_header="loom/target/arch/amdgpu/gfx1250_descriptors.h",
    function_name="loom_amdgpu_gfx1250_core_descriptor_set",
    c_table_prefix="AmdgpuGfx1250Core",
    c_enum_prefix="AMDGPU_GFX1250_CORE",
    generator_version=1,
    reg_classes=(
        RegClass(
            _REG_SGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=106,
        ),
        RegClass(
            _REG_VGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=1024,
        ),
        RegClass(
            _REG_M0,
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            physical_count=1,
        ),
    ),
    resources=(
        *_common_scalar_vector_memory_resources(),
        Resource(_RESOURCE_WMMA, capacity_per_cycle=1, kind=ResourceKind.MATRIX),
        Resource(_RESOURCE_SWMMAC, capacity_per_cycle=1, kind=ResourceKind.MATRIX),
        Resource(_RESOURCE_CONTROL, capacity_per_cycle=1, kind=ResourceKind.CONTROL),
    ),
    schedule_classes=(
        *_common_scalar_vector_memory_schedule_classes(
            smem_load_hazards=_SMEM_WAIT_HAZARDS,
            vmem_load_hazards=_VMEM_LOAD_WAIT_HAZARDS,
            vmem_store_hazards=_VMEM_STORE_WAIT_HAZARDS,
            lds_load_hazards=_LDS_WAIT_HAZARDS,
            lds_store_hazards=_LDS_WAIT_HAZARDS,
            lds_atomic_hazards=_LDS_WAIT_HAZARDS,
            lds_crosslane_hazards=_LDS_WAIT_HAZARDS,
        ),
        ScheduleClass(
            _SCHEDULE_WMMA,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=32,
            issue_uses=(IssueUse(_RESOURCE_WMMA, cycles=1, units=1),),
            hazards=_matrix_hazards(_RESOURCE_WMMA),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_WMMA_SCALE,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=32,
            issue_uses=(IssueUse(_RESOURCE_WMMA, cycles=1, units=1),),
            hazards=_matrix_hazards(_RESOURCE_WMMA),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_SWMMAC,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=32,
            issue_uses=(IssueUse(_RESOURCE_SWMMAC, cycles=1, units=1),),
            hazards=_matrix_hazards(_RESOURCE_SWMMAC),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_LOAD,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_VMEM_LOAD_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_STORE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_VMEM_STORE_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_LDS,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_LDS_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_SMEM,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_SMEM_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_ALU,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_ALU_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_IDLE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_IDLE_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
    ),
    descriptors=(
        *_s_mov_b32_descriptors(),
        Descriptor(
            key="amdgpu.v_wmma_f32_16x16x32_f16",
            mnemonic="v_wmma_f32_16x16x32_f16",
            semantic_tag="matrix.wmma.f32.16x16x32.f16",
            operands=(
                _vgpr_result(units=8),
                _vgpr_operand("a", units=8),
                _vgpr_operand("b", units=8),
                _vgpr_const_operand("acc", units=8),
            ),
            asm_forms=_asm(results=("dst",), operands=("a", "b", "acc")),
            schedule_class=_SCHEDULE_WMMA,
            encoding_id=LOW_DESCRIPTOR_ENCODING_ID_NONE,
            flags=_PSEUDO_DEAD_REMOVABLE_FLAGS,
        ),
        Descriptor(
            key="amdgpu.v_wmma_scale_f32_16x16x128_f8f6f4_f8_f8",
            mnemonic="v_wmma_scale_f32_16x16x128_f8f6f4_f8_f8",
            semantic_tag="matrix.wmma.scale.f32.16x16x128.f8f6f4.f8.f8",
            operands=(
                _vgpr_result(units=8),
                _vgpr_operand("a", units=16),
                _vgpr_operand("b", units=16),
                _vgpr_const_operand("acc", units=8),
                _vgpr_operand("scale_src0", units=1),
                _vgpr_operand("scale_src1", units=1),
            ),
            immediates=(
                _MATRIX_A_FORMAT_IMMEDIATE,
                _MATRIX_B_FORMAT_IMMEDIATE,
                _MATRIX_A_SCALE_IMMEDIATE,
                _MATRIX_B_SCALE_IMMEDIATE,
                _MATRIX_A_SCALE_FORMAT_IMMEDIATE,
                _MATRIX_B_SCALE_FORMAT_IMMEDIATE,
                _MATRIX_A_REUSE_IMMEDIATE,
                _MATRIX_B_REUSE_IMMEDIATE,
            ),
            asm_forms=_asm(
                results=("dst",),
                operands=("a", "b", "acc", "scale_src0", "scale_src1"),
                immediates=(
                    "matrix_a_fmt",
                    "matrix_b_fmt",
                    "matrix_a_scale",
                    "matrix_b_scale",
                    "matrix_a_scale_fmt",
                    "matrix_b_scale_fmt",
                    "matrix_a_reuse",
                    "matrix_b_reuse",
                ),
                named_immediates=True,
            ),
            schedule_class=_SCHEDULE_WMMA_SCALE,
            encoding_id=LOW_DESCRIPTOR_ENCODING_ID_NONE,
            flags=_PSEUDO_DEAD_REMOVABLE_FLAGS,
        ),
        Descriptor(
            key="amdgpu.v_wmma_scale16_f32_16x16x128_f8f6f4_f8_f8",
            mnemonic="v_wmma_scale16_f32_16x16x128_f8f6f4_f8_f8",
            semantic_tag="matrix.wmma.scale16.f32.16x16x128.f8f6f4.f8.f8",
            operands=(
                _vgpr_result(units=8),
                _vgpr_operand("a", units=16),
                _vgpr_operand("b", units=16),
                _vgpr_const_operand("acc", units=8),
                _vgpr_operand("scale_src0", units=2),
                _vgpr_operand("scale_src1", units=2),
            ),
            immediates=(
                _MATRIX_A_FORMAT_IMMEDIATE,
                _MATRIX_B_FORMAT_IMMEDIATE,
                _MATRIX_A_SCALE_IMMEDIATE,
                _MATRIX_B_SCALE_IMMEDIATE,
                _MATRIX_A_SCALE_FORMAT_IMMEDIATE,
                _MATRIX_B_SCALE_FORMAT_IMMEDIATE,
                _MATRIX_A_REUSE_IMMEDIATE,
                _MATRIX_B_REUSE_IMMEDIATE,
            ),
            asm_forms=_asm(
                results=("dst",),
                operands=("a", "b", "acc", "scale_src0", "scale_src1"),
                immediates=(
                    "matrix_a_fmt",
                    "matrix_b_fmt",
                    "matrix_a_scale",
                    "matrix_b_scale",
                    "matrix_a_scale_fmt",
                    "matrix_b_scale_fmt",
                    "matrix_a_reuse",
                    "matrix_b_reuse",
                ),
                named_immediates=True,
            ),
            schedule_class=_SCHEDULE_WMMA_SCALE,
            encoding_id=LOW_DESCRIPTOR_ENCODING_ID_NONE,
            flags=_PSEUDO_DEAD_REMOVABLE_FLAGS,
        ),
        Descriptor(
            key="amdgpu.v_swmmac_f32_16x16x64_f16",
            mnemonic="v_swmmac_f32_16x16x64_f16",
            semantic_tag="matrix.swmmac.f32.16x16x64.f16",
            operands=(
                _vgpr_result(units=8),
                _vgpr_operand("acc", units=8),
                _vgpr_operand("a", units=8),
                _vgpr_operand("b", units=16),
                _vgpr_operand("index", units=1),
            ),
            immediates=(_INDEX_KEY_16_IMMEDIATE,),
            constraints=_DESTRUCTIVE_ACCUMULATOR_CONSTRAINTS,
            asm_forms=_asm(
                results=("dst",),
                operands=("acc", "a", "b", "index"),
                immediates=("index_key_16bit",),
                named_immediates=True,
            ),
            schedule_class=_SCHEDULE_SWMMAC,
            encoding_id=LOW_DESCRIPTOR_ENCODING_ID_NONE,
            flags=_PSEUDO_DEAD_REMOVABLE_FLAGS,
        ),
    ),
)


def _amdgpu_core_descriptor_set_bases() -> tuple[DescriptorSet, ...]:
    return (
        _AMDGPU_GFX950_CORE_DESCRIPTOR_SET_BASE,
        _AMDGPU_GFX11_CORE_DESCRIPTOR_SET_BASE,
        _AMDGPU_GFX12_CORE_DESCRIPTOR_SET_BASE,
        _AMDGPU_GFX1250_CORE_DESCRIPTOR_SET_BASE,
    )


def _amdgpu_descriptor_id_key_set() -> set[str]:
    keys: set[str] = set()
    for descriptor_set in _amdgpu_core_descriptor_set_bases():
        keys.update(descriptor.key for descriptor in descriptor_set.descriptors)
    for overlays in (
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx12_core_overlays(),
        _gfx1250_core_overlays(),
    ):
        keys.update(overlay.descriptor_key for overlay in overlays)
    return keys


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


def amdgpu_descriptor_id_keys() -> tuple[str, ...]:
    """Returns descriptor keys known to the AMDGPU target family."""

    return tuple(sorted(_amdgpu_descriptor_id_key_set()))


def amdgpu_immediate_encoding_id_items() -> tuple[tuple[str, int], ...]:
    """Returns target-owned immediate encoding IDs used by AMDGPU descriptors."""

    return (
        ("address_offset_byte", _ADDRESS_OFFSET_BYTE_ENCODING_ID),
        ("address_offset_dword", _ADDRESS_OFFSET_DWORD_ENCODING_ID),
        ("address_offset_qword", _ADDRESS_OFFSET_QWORD_ENCODING_ID),
        ("address_offset_dword_stride64", _ADDRESS_OFFSET_DWORD_STRIDE64_ENCODING_ID),
        ("address_offset_qword_stride64", _ADDRESS_OFFSET_QWORD_STRIDE64_ENCODING_ID),
        ("address_offset_ds16", _ADDRESS_OFFSET_DS16_ENCODING_ID),
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
    for reg_class_name in (_REG_SGPR, _REG_VGPR):
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
    overlay_descriptors: tuple[Descriptor, ...],
) -> DescriptorSet:
    descriptor_set = replace(
        base,
        descriptors=(
            base.descriptors[0],
            *overlay_descriptors,
            *base.descriptors[1:],
        ),
    )
    _validate_address_immediate_units(descriptor_set)
    return descriptor_set


def build_amdgpu_gfx950_core_descriptor_set(
    xml_path: str | Path,
) -> DescriptorSet:
    spec = parse_amdgpu_isa_xml_path(xml_path)
    return _with_overlay_descriptors(
        _AMDGPU_GFX950_CORE_DESCRIPTOR_SET_BASE,
        _gfx950_core_overlay_descriptors(spec),
    )


def build_amdgpu_gfx11_core_descriptor_set(
    xml_path: str | Path,
) -> DescriptorSet:
    spec = parse_amdgpu_isa_xml_path(xml_path)
    return _with_overlay_descriptors(
        _AMDGPU_GFX11_CORE_DESCRIPTOR_SET_BASE,
        _gfx11_core_overlay_descriptors(spec),
    )


def build_amdgpu_gfx12_core_descriptor_set(
    xml_path: str | Path,
) -> DescriptorSet:
    spec = parse_amdgpu_isa_xml_path(xml_path)
    return _with_overlay_descriptors(
        _AMDGPU_GFX12_CORE_DESCRIPTOR_SET_BASE,
        _gfx12_core_overlay_descriptors(spec),
    )


def build_amdgpu_gfx1250_core_descriptor_set(
    xml_path: str | Path,
) -> DescriptorSet:
    spec = parse_amdgpu_isa_xml_path(xml_path)
    return _with_overlay_descriptors(
        _AMDGPU_GFX1250_CORE_DESCRIPTOR_SET_BASE,
        _gfx1250_core_overlay_descriptors(spec),
    )


AMDGPU_DESCRIPTOR_SET_BUILDERS = {
    "gfx950": build_amdgpu_gfx950_core_descriptor_set,
    "gfx11": build_amdgpu_gfx11_core_descriptor_set,
    "gfx12": build_amdgpu_gfx12_core_descriptor_set,
    "gfx1250": build_amdgpu_gfx1250_core_descriptor_set,
}


def build_amdgpu_core_descriptor_set(
    target: str,
    xml_path: str | Path,
) -> DescriptorSet:
    try:
        builder = AMDGPU_DESCRIPTOR_SET_BUILDERS[target]
    except KeyError as exc:
        supported = ", ".join(sorted(AMDGPU_DESCRIPTOR_SET_BUILDERS))
        raise ValueError(
            f"unsupported AMDGPU descriptor target '{target}'; "
            f"expected one of: {supported}"
        ) from exc
    return builder(xml_path)
