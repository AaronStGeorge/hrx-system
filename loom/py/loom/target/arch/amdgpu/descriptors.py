# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Descriptor input data for the initial AMDGPU GFX950 target-low shard."""

from __future__ import annotations

from pathlib import Path

from loom.target.low_descriptors import (
    Descriptor,
    DescriptorFlag,
    DescriptorSet,
    Effect,
    EffectFlag,
    EffectKind,
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
)

_REG_SGPR = "amdgpu.sgpr"
_REG_VGPR = "amdgpu.vgpr"
_REG_AGPR = "amdgpu.agpr"

_RESOURCE_SALU = "amdgpu.salu"
_RESOURCE_VALU = "amdgpu.valu"
_RESOURCE_SMEM = "amdgpu.smem"
_RESOURCE_VMEM_LOAD = "amdgpu.vmem.load"
_RESOURCE_VMEM_STORE = "amdgpu.vmem.store"
_RESOURCE_MFMA = "amdgpu.mfma"
_RESOURCE_CONTROL = "amdgpu.control"

_SCHEDULE_SALU = "amdgpu.salu"
_SCHEDULE_VALU = "amdgpu.valu"
_SCHEDULE_SMEM_LOAD = "amdgpu.smem.load"
_SCHEDULE_VMEM_LOAD = "amdgpu.vmem.load"
_SCHEDULE_VMEM_STORE = "amdgpu.vmem.store"
_SCHEDULE_MFMA = "amdgpu.mfma"
_SCHEDULE_WAIT = "amdgpu.wait"

_SGPR_ALT = (RegClassAlt(_REG_SGPR),)
_VGPR_ALT = (RegClassAlt(_REG_VGPR),)
_VGPR_AGPR_ALT = (RegClassAlt(_REG_VGPR), RegClassAlt(_REG_AGPR))
_VGPR_AGPR_CONST_ALT = (
    RegClassAlt(_REG_VGPR),
    RegClassAlt(_REG_AGPR),
    RegClassAlt(None, flags=(RegClassAltFlag.IMMEDIATE,)),
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

_OFFSET_IMMEDIATE = Immediate(
    "offset",
    ImmediateKind.UNSIGNED,
    bit_width=12,
    unsigned_max=(2**12) - 1,
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

_GLOBAL_LOAD_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=32,
)

_GLOBAL_STORE_EFFECT = Effect(
    EffectKind.WRITE,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=32,
)

_WAIT_EFFECT = Effect(
    EffectKind.COUNTER,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
)

AMDGPU_GFX950_CORE_DESCRIPTOR_SET = DescriptorSet(
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
        RegClass(_REG_SGPR, 32, flags=(RegClassFlag.PHYSICAL,), physical_count=106),
        RegClass(_REG_VGPR, 32, flags=(RegClassFlag.PHYSICAL,), physical_count=1024),
        RegClass(_REG_AGPR, 32, flags=(RegClassFlag.PHYSICAL,), physical_count=256),
    ),
    resources=(
        Resource(_RESOURCE_SALU, capacity_per_cycle=1, kind=ResourceKind.SCALAR_ALU),
        Resource(_RESOURCE_VALU, capacity_per_cycle=1, kind=ResourceKind.VECTOR_ALU),
        Resource(_RESOURCE_SMEM, capacity_per_cycle=1, kind=ResourceKind.LOAD),
        Resource(_RESOURCE_VMEM_LOAD, capacity_per_cycle=1, kind=ResourceKind.LOAD),
        Resource(_RESOURCE_VMEM_STORE, capacity_per_cycle=1, kind=ResourceKind.STORE),
        Resource(_RESOURCE_MFMA, capacity_per_cycle=1, kind=ResourceKind.MATRIX),
        Resource(_RESOURCE_CONTROL, capacity_per_cycle=1, kind=ResourceKind.CONTROL),
    ),
    schedule_classes=(
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
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_SMEM_LOAD,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=8,
            issue_uses=(IssueUse(_RESOURCE_SMEM, cycles=1, units=1),),
            flags=(ScheduleClassFlag.MAY_LOAD,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_VMEM_LOAD,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=16,
            issue_uses=(IssueUse(_RESOURCE_VMEM_LOAD, cycles=1, units=1),),
            flags=(ScheduleClassFlag.MAY_LOAD,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_VMEM_STORE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=16,
            issue_uses=(IssueUse(_RESOURCE_VMEM_STORE, cycles=1, units=1),),
            flags=(ScheduleClassFlag.MAY_STORE,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_MFMA,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=32,
            issue_uses=(IssueUse(_RESOURCE_MFMA, cycles=1, units=1),),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
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
            schedule_class=_SCHEDULE_SALU,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="amdgpu.s_add_u32",
            mnemonic="s_add_u32",
            semantic_tag="integer.add.u32",
            operands=(_sgpr_result(), _sgpr_operand("lhs"), _sgpr_operand("rhs")),
            schedule_class=_SCHEDULE_SALU,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="amdgpu.v_add_u32",
            mnemonic="v_add_u32",
            semantic_tag="integer.add.u32",
            operands=(_vgpr_result(), _vgpr_operand("lhs"), _vgpr_operand("rhs")),
            schedule_class=_SCHEDULE_VALU,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="amdgpu.s_buffer_load_dword",
            mnemonic="s_buffer_load_dword",
            semantic_tag="memory.load.u32",
            operands=(
                _sgpr_result(),
                _sgpr_resource("resource", units=4),
                _sgpr_operand("soffset"),
            ),
            immediates=(_OFFSET_IMMEDIATE,),
            effects=(_GLOBAL_LOAD_EFFECT,),
            schedule_class=_SCHEDULE_SMEM_LOAD,
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
        Descriptor(
            key="amdgpu.buffer_load_dword",
            mnemonic="buffer_load_dword",
            semantic_tag="memory.load.u32",
            operands=(
                _vgpr_result(),
                _sgpr_resource("resource", units=4),
                _vgpr_operand("vaddr"),
                _sgpr_operand("soffset"),
            ),
            immediates=(_OFFSET_IMMEDIATE,),
            effects=(_GLOBAL_LOAD_EFFECT,),
            schedule_class=_SCHEDULE_VMEM_LOAD,
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
        Descriptor(
            key="amdgpu.buffer_store_dword",
            mnemonic="buffer_store_dword",
            semantic_tag="memory.store.u32",
            operands=(
                _vgpr_operand("value"),
                _sgpr_resource("resource", units=4),
                _vgpr_operand("vaddr"),
                _sgpr_operand("soffset"),
            ),
            immediates=(_OFFSET_IMMEDIATE,),
            effects=(_GLOBAL_STORE_EFFECT,),
            schedule_class=_SCHEDULE_VMEM_STORE,
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
        Descriptor(
            key="amdgpu.v_mfma_f32_16x16x16_f16",
            mnemonic="v_mfma_f32_16x16x16_f16",
            semantic_tag="matrix.mfma.f32.16x16x16.f16",
            operands=(
                _vgpr_agpr_result(units=4),
                _vgpr_agpr_operand("a", units=2),
                _vgpr_agpr_operand("b", units=2),
                _vgpr_agpr_const_operand("acc", units=4),
            ),
            schedule_class=_SCHEDULE_MFMA,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="amdgpu.s_waitcnt",
            mnemonic="s_waitcnt",
            semantic_tag="control.waitcnt",
            operands=(),
            immediates=(_VMCNT_IMMEDIATE, _LGKMCNT_IMMEDIATE),
            effects=(_WAIT_EFFECT,),
            schedule_class=_SCHEDULE_WAIT,
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
    ),
)
