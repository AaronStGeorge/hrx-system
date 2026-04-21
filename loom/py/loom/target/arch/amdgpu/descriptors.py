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
    AmdgpuImplicitOperandOverlay,
    AmdgpuOperandOverlay,
    materialize_amdgpu_descriptor_overlays,
)
from loom.target.arch.amdgpu.encoding import AMDGPU_ENCODING_FORMAT_SOP1
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
    Hazard,
    HazardKind,
    Immediate,
    ImmediateKind,
    IssueUse,
    LatencyKind,
    MemorySpace,
    ModelQuality,
    Operand,
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

_RESOURCE_SALU = "amdgpu.salu"
_RESOURCE_VALU = "amdgpu.valu"
_RESOURCE_SMEM = "amdgpu.smem"
_RESOURCE_VMEM_LOAD = "amdgpu.vmem.load"
_RESOURCE_VMEM_STORE = "amdgpu.vmem.store"
_RESOURCE_LDS_LOAD = "amdgpu.lds.load"
_RESOURCE_LDS_STORE = "amdgpu.lds.store"
_RESOURCE_MFMA = "amdgpu.mfma"
_RESOURCE_WMMA = "amdgpu.wmma"
_RESOURCE_SWMMAC = "amdgpu.swmmac"
_RESOURCE_CONTROL = "amdgpu.control"

_SCHEDULE_SALU = "amdgpu.salu"
_SCHEDULE_VALU = "amdgpu.valu"
_SCHEDULE_SMEM_LOAD = "amdgpu.smem.load"
_SCHEDULE_VMEM_LOAD = "amdgpu.vmem.load"
_SCHEDULE_VMEM_STORE = "amdgpu.vmem.store"
_SCHEDULE_LDS_LOAD = "amdgpu.lds.load"
_SCHEDULE_LDS_STORE = "amdgpu.lds.store"
_SCHEDULE_BARRIER = "amdgpu.barrier"
_SCHEDULE_MFMA = "amdgpu.mfma"
_SCHEDULE_WMMA = "amdgpu.wmma"
_SCHEDULE_WMMA_SCALE = "amdgpu.wmma.scale"
_SCHEDULE_SWMMAC = "amdgpu.swmmac"
_SCHEDULE_WAIT_MEMORY = "amdgpu.wait.memory"
_SCHEDULE_WAIT_LOAD = "amdgpu.wait.load"
_SCHEDULE_WAIT_STORE = "amdgpu.wait.store"
_SCHEDULE_WAIT_ALU = "amdgpu.wait.alu"
_SCHEDULE_WAIT_IDLE = "amdgpu.wait.idle"

_COUNTER_LOAD = 1
_COUNTER_STORE = 2
_COUNTER_ALU = 3

_LOAD_COUNTER_HAZARD = Hazard(HazardKind.WAIT_COUNTER, counter_id=_COUNTER_LOAD)
_STORE_COUNTER_HAZARD = Hazard(HazardKind.WAIT_COUNTER, counter_id=_COUNTER_STORE)
_ALU_COUNTER_HAZARD = Hazard(HazardKind.WAIT_COUNTER, counter_id=_COUNTER_ALU)
_MEMORY_WAIT_HAZARDS = (_LOAD_COUNTER_HAZARD, _STORE_COUNTER_HAZARD)
_LOAD_WAIT_HAZARDS = (_LOAD_COUNTER_HAZARD,)
_STORE_WAIT_HAZARDS = (_STORE_COUNTER_HAZARD,)
_ALU_WAIT_HAZARDS = (_ALU_COUNTER_HAZARD,)
_IDLE_WAIT_HAZARDS = (
    _LOAD_COUNTER_HAZARD,
    _STORE_COUNTER_HAZARD,
    _ALU_COUNTER_HAZARD,
)

_SGPR_ALT = (RegClassAlt(_REG_SGPR),)
_VGPR_ALT = (RegClassAlt(_REG_VGPR),)
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
    )


def _common_scalar_vector_memory_schedule_classes() -> tuple[ScheduleClass, ...]:
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
            hazards=_LOAD_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.MAY_LOAD,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_VMEM_LOAD,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=16,
            issue_uses=(IssueUse(_RESOURCE_VMEM_LOAD, cycles=1, units=1),),
            hazards=_LOAD_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.MAY_LOAD,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_VMEM_STORE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=16,
            issue_uses=(IssueUse(_RESOURCE_VMEM_STORE, cycles=1, units=1),),
            hazards=_STORE_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.MAY_STORE,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_LDS_LOAD,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=8,
            issue_uses=(IssueUse(_RESOURCE_LDS_LOAD, cycles=1, units=1),),
            hazards=_LOAD_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.MAY_LOAD,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_LDS_STORE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=8,
            issue_uses=(IssueUse(_RESOURCE_LDS_STORE, cycles=1, units=1),),
            hazards=_STORE_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.MAY_STORE,),
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
    )


def _asm(
    *,
    results: tuple[str, ...] = (),
    operands: tuple[str, ...] = (),
    immediates: tuple[str, ...] = (),
    named_immediates: bool = False,
) -> tuple[AsmForm, ...]:
    return (
        AsmForm(
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


def _sgpr_resource(field_name: str, *, units: int = 1) -> Operand:
    return Operand(field_name, OperandRole.RESOURCE, _SGPR_ALT, unit_count=units)


def _vgpr_result(field_name: str = "dst", *, units: int = 1) -> Operand:
    return Operand(field_name, OperandRole.RESULT, _VGPR_ALT, unit_count=units)


def _vgpr_operand(field_name: str, *, units: int = 1) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _VGPR_ALT, unit_count=units)


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

_VMCNT_IMMEDIATE = Immediate(
    "vmcnt",
    ImmediateKind.UNSIGNED,
    bit_width=6,
    unsigned_max=(2**6) - 1,
)

_LGKMCNT_IMMEDIATE = Immediate(
    "lgkmcnt",
    ImmediateKind.UNSIGNED,
    bit_width=6,
    unsigned_max=(2**6) - 1,
)

_LOADCNT_IMMEDIATE = Immediate(
    "loadcnt",
    ImmediateKind.UNSIGNED,
    bit_width=6,
    unsigned_max=(2**6) - 1,
)

_STORECNT_IMMEDIATE = Immediate(
    "storecnt",
    ImmediateKind.UNSIGNED,
    bit_width=6,
    unsigned_max=(2**6) - 1,
)

_DEPCTR_IMMEDIATE = Immediate(
    "depctr",
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

_WORKGROUP_LOAD_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.WORKGROUP,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=32,
)

_WORKGROUP_LOAD_B128_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.WORKGROUP,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=128,
)

_WORKGROUP_STORE_EFFECT = Effect(
    EffectKind.WRITE,
    memory_space=MemorySpace.WORKGROUP,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=32,
)

_WORKGROUP_STORE_B128_EFFECT = Effect(
    EffectKind.WRITE,
    memory_space=MemorySpace.WORKGROUP,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=128,
)

_WORKGROUP_BARRIER_EFFECT = Effect(
    EffectKind.BARRIER,
    memory_space=MemorySpace.WORKGROUP,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
)

_CONVERGENT_EFFECT = Effect(
    EffectKind.CONVERGENT,
    flags=(EffectFlag.ORDERED,),
)

_WAIT_EFFECT = Effect(
    EffectKind.COUNTER,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
)

_LOAD_WAIT_EFFECT = Effect(
    EffectKind.COUNTER,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
    counter_id=_COUNTER_LOAD,
)

_STORE_WAIT_EFFECT = Effect(
    EffectKind.COUNTER,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
    counter_id=_COUNTER_STORE,
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
_PSEUDO_DEAD_REMOVABLE_FLAGS = (DescriptorFlag.DEAD_REMOVABLE, DescriptorFlag.PSEUDO)


def _offset_immediate(bit_width: int) -> Immediate:
    return Immediate(
        "offset",
        ImmediateKind.UNSIGNED,
        bit_width=bit_width,
        unsigned_max=(2**bit_width) - 1,
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

_IGNORE_WORKGROUP_READ_MEMORY = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_DSMEM",
    data_format_name="FMT_NUM_B32",
    size_bits=32,
    is_input=True,
    is_output=False,
    ignore_reason="modeled-by-workgroup-read-effect",
)

_IGNORE_WORKGROUP_READ_MEMORY_B128 = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_DSMEM",
    data_format_name="FMT_NUM_B128",
    size_bits=128,
    is_input=True,
    is_output=False,
    ignore_reason="modeled-by-workgroup-read-effect",
)

_IGNORE_WORKGROUP_WRITE_MEMORY = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_DSMEM",
    data_format_name="FMT_NUM_B32",
    size_bits=32,
    is_input=False,
    is_output=True,
    ignore_reason="modeled-by-workgroup-write-effect",
)

_IGNORE_WORKGROUP_WRITE_MEMORY_B128 = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_DSMEM",
    data_format_name="FMT_NUM_B128",
    size_bits=128,
    is_input=False,
    is_output=True,
    ignore_reason="modeled-by-workgroup-write-effect",
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
        immediate_fields=(offset_field_name,),
        immediates=(_offset_immediate(offset_bit_width),),
        fixed_encoding_fields=(("OFFEN", 1),),
        effects=(_GLOBAL_LOAD_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
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
        immediate_fields=(offset_field_name,),
        immediates=(_offset_immediate(offset_bit_width),),
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
        immediate_fields=(offset_field_name,),
        immediates=(_offset_immediate(offset_bit_width),),
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
        immediate_fields=(offset_field_name,),
        immediates=(_offset_immediate(offset_bit_width),),
        fixed_encoding_fields=(("OFFEN", 1),),
        effects=(_GLOBAL_STORE_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
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
        immediate_fields=(offset_field_name,),
        immediates=(_offset_immediate(offset_bit_width),),
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
        immediate_fields=(offset_field_name,),
        immediates=(_offset_immediate(offset_bit_width),),
        fixed_encoding_fields=(("OFFEN", 1),),
        effects=(_GLOBAL_STORE_B128_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_read_b32_overlay(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, int], ...] = (("OFFSET1", 0), ("GDS", 0)),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.ds_read_b32",
        instruction_name="DS_READ_B32",
        mnemonic="ds_read_b32",
        encoding_name=encoding_name,
        semantic_tag="memory.workgroup.load.u32",
        schedule_class=_SCHEDULE_LDS_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result()),
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
        ),
        implicit_operands=(_IGNORE_WORKGROUP_READ_MEMORY,),
        immediate_fields=("OFFSET0",),
        immediates=(_offset_immediate(8),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_WORKGROUP_LOAD_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_read_b128_overlay(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, int], ...] = (("OFFSET1", 0), ("GDS", 0)),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.ds_read_b128",
        instruction_name="DS_READ_B128",
        mnemonic="ds_read_b128",
        encoding_name=encoding_name,
        semantic_tag="memory.workgroup.load.u128",
        schedule_class=_SCHEDULE_LDS_LOAD,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result(units=4)),
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
        ),
        implicit_operands=(_IGNORE_WORKGROUP_READ_MEMORY_B128,),
        immediate_fields=("OFFSET0",),
        immediates=(_offset_immediate(8),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_WORKGROUP_LOAD_B128_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_write_b32_overlay(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, int], ...] = (("OFFSET1", 0), ("GDS", 0)),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.ds_write_b32",
        instruction_name="DS_WRITE_B32",
        mnemonic="ds_write_b32",
        encoding_name=encoding_name,
        semantic_tag="memory.workgroup.store.u32",
        schedule_class=_SCHEDULE_LDS_STORE,
        operands=(
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
            AmdgpuOperandOverlay("DATA0", _vgpr_operand("value")),
        ),
        implicit_operands=(_IGNORE_WORKGROUP_WRITE_MEMORY,),
        immediate_fields=("OFFSET0",),
        immediates=(_offset_immediate(8),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_WORKGROUP_STORE_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_write_b128_overlay(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, int], ...] = (("OFFSET1", 0), ("GDS", 0)),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.ds_write_b128",
        instruction_name="DS_WRITE_B128",
        mnemonic="ds_write_b128",
        encoding_name=encoding_name,
        semantic_tag="memory.workgroup.store.u128",
        schedule_class=_SCHEDULE_LDS_STORE,
        operands=(
            AmdgpuOperandOverlay("ADDR", _vgpr_operand("addr")),
            AmdgpuOperandOverlay("DATA0", _vgpr_operand("value", units=4)),
        ),
        implicit_operands=(_IGNORE_WORKGROUP_WRITE_MEMORY_B128,),
        immediate_fields=("OFFSET0",),
        immediates=(_offset_immediate(8),),
        fixed_encoding_fields=fixed_encoding_fields,
        effects=(_WORKGROUP_STORE_B128_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _ds_memory_overlays(
    *,
    encoding_name: str = "ENC_DS",
    fixed_encoding_fields: tuple[tuple[str, int], ...] = (("OFFSET1", 0), ("GDS", 0)),
) -> tuple[AmdgpuDescriptorOverlay, ...]:
    return (
        _ds_read_b32_overlay(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        _ds_read_b128_overlay(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        _ds_write_b32_overlay(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
        _ds_write_b128_overlay(
            encoding_name=encoding_name,
            fixed_encoding_fields=fixed_encoding_fields,
        ),
    )


def _v_wmma_f32_16x16x16_f16_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_wmma_f32_16x16x16_f16",
        instruction_name="V_WMMA_F32_16X16X16_F16",
        mnemonic="v_wmma_f32_16x16x16_f16",
        encoding_name="ENC_VOP3P",
        semantic_tag="matrix.wmma.f32.16x16x16.f16",
        schedule_class=_SCHEDULE_WMMA,
        operands=(
            AmdgpuOperandOverlay("VDST", _vgpr_result(units=8)),
            AmdgpuOperandOverlay("SRC0", _vgpr_operand("a", units=4)),
            AmdgpuOperandOverlay("SRC1", _vgpr_operand("b", units=4)),
            AmdgpuOperandOverlay("SRC2", _vgpr_const_operand("acc", units=8)),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
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


def _s_waitcnt_overlay() -> AmdgpuDescriptorOverlay:
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
        effects=(_WAIT_EFFECT,),
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
        effects=(_LOAD_WAIT_EFFECT,),
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
        effects=(_STORE_WAIT_EFFECT,),
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


def _gfx950_core_overlay_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    return materialize_amdgpu_descriptor_overlays(
        spec,
        (
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
            _v_fma_f32_overlay(),
            _s_load_dwordx2_overlay(),
            _s_buffer_load_dword_overlay(),
            _s_buffer_load_64_overlay(
                descriptor_key="amdgpu.s_buffer_load_dwordx2",
                instruction_name="S_BUFFER_LOAD_DWORDX2",
                mnemonic="s_buffer_load_dwordx2",
            ),
            _buffer_load_dword_overlay(
                encoding_name="ENC_MUBUF", resource_field_name="SRSRC"
            ),
            _buffer_load_64_overlay(
                descriptor_key="amdgpu.buffer_load_dwordx2",
                instruction_name="BUFFER_LOAD_DWORDX2",
                mnemonic="buffer_load_dwordx2",
                encoding_name="ENC_MUBUF",
                resource_field_name="SRSRC",
            ),
            _buffer_load_128_overlay(
                descriptor_key="amdgpu.buffer_load_dwordx4",
                instruction_name="BUFFER_LOAD_DWORDX4",
                mnemonic="buffer_load_dwordx4",
                encoding_name="ENC_MUBUF",
                resource_field_name="SRSRC",
            ),
            _buffer_store_dword_overlay(
                encoding_name="ENC_MUBUF", resource_field_name="SRSRC"
            ),
            _buffer_store_64_overlay(
                descriptor_key="amdgpu.buffer_store_dwordx2",
                instruction_name="BUFFER_STORE_DWORDX2",
                mnemonic="buffer_store_dwordx2",
                encoding_name="ENC_MUBUF",
                resource_field_name="SRSRC",
            ),
            _buffer_store_128_overlay(
                descriptor_key="amdgpu.buffer_store_dwordx4",
                instruction_name="BUFFER_STORE_DWORDX4",
                mnemonic="buffer_store_dwordx4",
                encoding_name="ENC_MUBUF",
                resource_field_name="SRSRC",
            ),
            *_ds_memory_overlays(),
            _v_mfma_f32_16x16x16_f16_overlay(),
            _s_barrier_overlay(),
            _s_waitcnt_overlay(),
        ),
    )


def _gfx11_core_overlay_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    return materialize_amdgpu_descriptor_overlays(
        spec,
        (
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
            _v_fma_f32_overlay(),
            _s_load_dwordx2_overlay(),
            _s_buffer_load_dword_overlay(),
            _s_buffer_load_64_overlay(),
            _buffer_load_dword_overlay(
                encoding_name="ENC_MUBUF", resource_field_name="SRSRC"
            ),
            _buffer_load_64_overlay(
                encoding_name="ENC_MUBUF", resource_field_name="SRSRC"
            ),
            _buffer_load_128_overlay(
                encoding_name="ENC_MUBUF", resource_field_name="SRSRC"
            ),
            _buffer_store_dword_overlay(
                encoding_name="ENC_MUBUF", resource_field_name="SRSRC"
            ),
            _buffer_store_64_overlay(
                encoding_name="ENC_MUBUF", resource_field_name="SRSRC"
            ),
            _buffer_store_128_overlay(
                encoding_name="ENC_MUBUF", resource_field_name="SRSRC"
            ),
            *_ds_memory_overlays(),
            _v_wmma_f32_16x16x16_f16_overlay(),
            _s_barrier_overlay(),
            _s_waitcnt_overlay(),
            _s_waitcnt_depctr_overlay(),
            _s_wait_idle_overlay(),
        ),
    )


def _gfx12_core_overlay_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    return materialize_amdgpu_descriptor_overlays(
        spec,
        (
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
            _v_fma_f32_overlay(),
            _s_load_dwordx2_overlay("IOFFSET", offset_bit_width=24),
            _s_buffer_load_dword_overlay("IOFFSET", offset_bit_width=24),
            _s_buffer_load_64_overlay(offset_field_name="IOFFSET", offset_bit_width=24),
            _buffer_load_dword_overlay(
                encoding_name="ENC_VBUFFER",
                resource_field_name="RSRC",
                offset_field_name="IOFFSET",
                offset_bit_width=24,
            ),
            _buffer_load_64_overlay(
                encoding_name="ENC_VBUFFER",
                resource_field_name="RSRC",
                offset_field_name="IOFFSET",
                offset_bit_width=24,
            ),
            _buffer_load_128_overlay(
                encoding_name="ENC_VBUFFER",
                resource_field_name="RSRC",
                offset_field_name="IOFFSET",
                offset_bit_width=24,
            ),
            _buffer_store_dword_overlay(
                encoding_name="ENC_VBUFFER",
                resource_field_name="RSRC",
                offset_field_name="IOFFSET",
                offset_bit_width=24,
            ),
            _buffer_store_64_overlay(
                encoding_name="ENC_VBUFFER",
                resource_field_name="RSRC",
                offset_field_name="IOFFSET",
                offset_bit_width=24,
            ),
            _buffer_store_128_overlay(
                encoding_name="ENC_VBUFFER",
                resource_field_name="RSRC",
                offset_field_name="IOFFSET",
                offset_bit_width=24,
            ),
            *_ds_memory_overlays(
                encoding_name="ENC_VDS",
                fixed_encoding_fields=(("OFFSET1", 0),),
            ),
            _v_wmma_f32_16x16x16_f16_overlay(),
            _s_wait_loadcnt_overlay(),
            _s_wait_storecnt_overlay(),
            _s_wait_alu_overlay(),
            _s_wait_idle_overlay(),
        ),
    )


def _gfx1250_core_overlay_descriptors(
    spec: AmdgpuIsaFactSource,
) -> tuple[Descriptor, ...]:
    return materialize_amdgpu_descriptor_overlays(
        spec,
        (
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
            _v_fma_f32_overlay(),
            _s_load_dwordx2_overlay("IOFFSET", offset_bit_width=24),
            _s_buffer_load_dword_overlay("IOFFSET", offset_bit_width=24),
            _s_buffer_load_64_overlay(offset_field_name="IOFFSET", offset_bit_width=24),
            _buffer_load_dword_overlay(
                encoding_name="ENC_VBUFFER",
                resource_field_name="RSRC",
                offset_field_name="IOFFSET",
                offset_bit_width=24,
            ),
            _buffer_load_64_overlay(
                encoding_name="ENC_VBUFFER",
                resource_field_name="RSRC",
                offset_field_name="IOFFSET",
                offset_bit_width=24,
            ),
            _buffer_load_128_overlay(
                encoding_name="ENC_VBUFFER",
                resource_field_name="RSRC",
                offset_field_name="IOFFSET",
                offset_bit_width=24,
            ),
            _buffer_store_dword_overlay(
                encoding_name="ENC_VBUFFER",
                resource_field_name="RSRC",
                offset_field_name="IOFFSET",
                offset_bit_width=24,
            ),
            _buffer_store_64_overlay(
                encoding_name="ENC_VBUFFER",
                resource_field_name="RSRC",
                offset_field_name="IOFFSET",
                offset_bit_width=24,
            ),
            _buffer_store_128_overlay(
                encoding_name="ENC_VBUFFER",
                resource_field_name="RSRC",
                offset_field_name="IOFFSET",
                offset_bit_width=24,
            ),
            *_ds_memory_overlays(
                encoding_name="ENC_VDS",
                fixed_encoding_fields=(("OFFSET1", 0),),
            ),
            _s_wait_loadcnt_overlay(),
            _s_wait_storecnt_overlay(),
            _s_wait_alu_overlay(),
            _s_wait_idle_overlay(),
        ),
    )


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
    ),
    resources=(
        *_common_scalar_vector_memory_resources(),
        Resource(_RESOURCE_MFMA, capacity_per_cycle=1, kind=ResourceKind.MATRIX),
        Resource(_RESOURCE_CONTROL, capacity_per_cycle=1, kind=ResourceKind.CONTROL),
    ),
    schedule_classes=(
        *_common_scalar_vector_memory_schedule_classes(),
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
            hazards=_MEMORY_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
    ),
    descriptors=(
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
    ),
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
    ),
    resources=(
        *_common_scalar_vector_memory_resources(),
        Resource(_RESOURCE_WMMA, capacity_per_cycle=1, kind=ResourceKind.MATRIX),
        Resource(_RESOURCE_CONTROL, capacity_per_cycle=1, kind=ResourceKind.CONTROL),
    ),
    schedule_classes=(
        *_common_scalar_vector_memory_schedule_classes(),
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
            hazards=_MEMORY_WAIT_HAZARDS,
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
    ),
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
    ),
    resources=(
        *_common_scalar_vector_memory_resources(),
        Resource(_RESOURCE_WMMA, capacity_per_cycle=1, kind=ResourceKind.MATRIX),
        Resource(_RESOURCE_CONTROL, capacity_per_cycle=1, kind=ResourceKind.CONTROL),
    ),
    schedule_classes=(
        *_common_scalar_vector_memory_schedule_classes(),
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
            hazards=_LOAD_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_STORE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_STORE_WAIT_HAZARDS,
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
    ),
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
    ),
    resources=(
        *_common_scalar_vector_memory_resources(),
        Resource(_RESOURCE_WMMA, capacity_per_cycle=1, kind=ResourceKind.MATRIX),
        Resource(_RESOURCE_SWMMAC, capacity_per_cycle=1, kind=ResourceKind.MATRIX),
        Resource(_RESOURCE_CONTROL, capacity_per_cycle=1, kind=ResourceKind.CONTROL),
    ),
    schedule_classes=(
        *_common_scalar_vector_memory_schedule_classes(),
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
            hazards=_LOAD_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_STORE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_STORE_WAIT_HAZARDS,
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


def _with_overlay_descriptors(
    base: DescriptorSet,
    overlay_descriptors: tuple[Descriptor, ...],
) -> DescriptorSet:
    return replace(
        base,
        descriptors=(
            base.descriptors[0],
            *overlay_descriptors,
            *base.descriptors[1:],
        ),
    )


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
