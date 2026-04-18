# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Descriptor input data for AMDGPU target-low shards."""

from __future__ import annotations

from pathlib import Path

from loom.target.arch.amdgpu.descriptor_overlay import (
    AmdgpuDescriptorOverlay,
    AmdgpuImplicitOperandOverlay,
    AmdgpuOperandOverlay,
    materialize_amdgpu_descriptor_overlays,
)
from loom.target.arch.amdgpu.isa_snapshot import (
    AmdgpuIsaSnapshot,
    parse_amdgpu_isa_snapshot_json,
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
_RESOURCE_MFMA = "amdgpu.mfma"
_RESOURCE_WMMA = "amdgpu.wmma"
_RESOURCE_SWMMAC = "amdgpu.swmmac"
_RESOURCE_CONTROL = "amdgpu.control"

_SCHEDULE_SALU = "amdgpu.salu"
_SCHEDULE_VALU = "amdgpu.valu"
_SCHEDULE_SMEM_LOAD = "amdgpu.smem.load"
_SCHEDULE_VMEM_LOAD = "amdgpu.vmem.load"
_SCHEDULE_VMEM_STORE = "amdgpu.vmem.store"
_SCHEDULE_MFMA = "amdgpu.mfma"
_SCHEDULE_WMMA = "amdgpu.wmma"
_SCHEDULE_WMMA_SCALE = "amdgpu.wmma.scale"
_SCHEDULE_SWMMAC = "amdgpu.swmmac"
_SCHEDULE_WAIT = "amdgpu.wait"

_COUNTER_LOAD = 1
_COUNTER_STORE = 2
_COUNTER_ALU = 3

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

_IGNORE_GLOBAL_WRITE_MEMORY = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B32",
    size_bits=32,
    is_input=False,
    is_output=True,
    ignore_reason="modeled-by-global-write-effect",
)


def _load_amdgpu_isa_snapshot(filename: str) -> AmdgpuIsaSnapshot:
    path = Path(__file__).with_name(filename)
    return parse_amdgpu_isa_snapshot_json(
        path.read_text(encoding="utf-8"), source_name=str(path)
    )


_GFX950_ISA_SNAPSHOT = _load_amdgpu_isa_snapshot("gfx950_isa_snapshot.json")
_GFX11_ISA_SNAPSHOT = _load_amdgpu_isa_snapshot("gfx11_isa_snapshot.json")
_GFX12_ISA_SNAPSHOT = _load_amdgpu_isa_snapshot("gfx12_isa_snapshot.json")
_GFX1250_ISA_SNAPSHOT = _load_amdgpu_isa_snapshot("gfx1250_isa_snapshot.json")


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


def _s_buffer_load_dword_overlay() -> AmdgpuDescriptorOverlay:
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
        immediates=(_OFFSET_IMMEDIATE,),
        effects=(_GLOBAL_LOAD_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _buffer_load_dword_overlay(
    *, encoding_name: str, resource_field_name: str
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
        immediates=(_OFFSET_IMMEDIATE,),
        effects=(_GLOBAL_LOAD_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _buffer_store_dword_overlay(
    *, encoding_name: str, resource_field_name: str
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
        immediates=(_OFFSET_IMMEDIATE,),
        effects=(_GLOBAL_STORE_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
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
        schedule_class=_SCHEDULE_WAIT,
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
        schedule_class=_SCHEDULE_WAIT,
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
        schedule_class=_SCHEDULE_WAIT,
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
        schedule_class=_SCHEDULE_WAIT,
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
        schedule_class=_SCHEDULE_WAIT,
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
        schedule_class=_SCHEDULE_WAIT,
        operands=(),
        effects=(_WAIT_EFFECT,),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


_GFX950_CORE_OVERLAY_DESCRIPTORS = materialize_amdgpu_descriptor_overlays(
    _GFX950_ISA_SNAPSHOT,
    (
        _s_add_u32_overlay(),
        _v_add_u32_overlay("V_ADD_U32"),
        _s_buffer_load_dword_overlay(),
        _buffer_load_dword_overlay(
            encoding_name="ENC_MUBUF", resource_field_name="SRSRC"
        ),
        _buffer_store_dword_overlay(
            encoding_name="ENC_MUBUF", resource_field_name="SRSRC"
        ),
        _v_mfma_f32_16x16x16_f16_overlay(),
        _s_waitcnt_overlay(),
    ),
)

_GFX11_CORE_OVERLAY_DESCRIPTORS = materialize_amdgpu_descriptor_overlays(
    _GFX11_ISA_SNAPSHOT,
    (
        _s_add_u32_overlay(),
        _v_add_u32_overlay("V_ADD_NC_U32"),
        _s_buffer_load_dword_overlay(),
        _buffer_load_dword_overlay(
            encoding_name="ENC_MUBUF", resource_field_name="SRSRC"
        ),
        _buffer_store_dword_overlay(
            encoding_name="ENC_MUBUF", resource_field_name="SRSRC"
        ),
        _v_wmma_f32_16x16x16_f16_overlay(),
        _s_waitcnt_overlay(),
        _s_waitcnt_depctr_overlay(),
        _s_wait_idle_overlay(),
    ),
)

_GFX12_CORE_OVERLAY_DESCRIPTORS = materialize_amdgpu_descriptor_overlays(
    _GFX12_ISA_SNAPSHOT,
    (
        _s_add_u32_overlay(),
        _v_add_u32_overlay("V_ADD_NC_U32"),
        _s_buffer_load_dword_overlay(),
        _buffer_load_dword_overlay(
            encoding_name="ENC_VBUFFER", resource_field_name="RSRC"
        ),
        _buffer_store_dword_overlay(
            encoding_name="ENC_VBUFFER", resource_field_name="RSRC"
        ),
        _v_wmma_f32_16x16x16_f16_overlay(),
        _s_wait_loadcnt_overlay(),
        _s_wait_storecnt_overlay(),
        _s_wait_alu_overlay(),
        _s_wait_idle_overlay(),
    ),
)

_GFX1250_CORE_OVERLAY_DESCRIPTORS = materialize_amdgpu_descriptor_overlays(
    _GFX1250_ISA_SNAPSHOT,
    (
        _s_add_u32_overlay(),
        _v_add_u32_overlay("V_ADD_NC_U32"),
        _s_buffer_load_dword_overlay(),
        _buffer_load_dword_overlay(
            encoding_name="ENC_VBUFFER", resource_field_name="RSRC"
        ),
        _buffer_store_dword_overlay(
            encoding_name="ENC_VBUFFER", resource_field_name="RSRC"
        ),
        _s_wait_loadcnt_overlay(),
        _s_wait_storecnt_overlay(),
        _s_wait_alu_overlay(),
        _s_wait_idle_overlay(),
    ),
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
            asm_forms=_asm(results=("dst",), immediates=("imm32",)),
            schedule_class=_SCHEDULE_SALU,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        *_GFX950_CORE_OVERLAY_DESCRIPTORS,
    ),
)

AMDGPU_GFX11_CORE_DESCRIPTOR_SET = DescriptorSet(
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
        Resource(_RESOURCE_SALU, capacity_per_cycle=1, kind=ResourceKind.SCALAR_ALU),
        Resource(_RESOURCE_VALU, capacity_per_cycle=1, kind=ResourceKind.VECTOR_ALU),
        Resource(_RESOURCE_SMEM, capacity_per_cycle=1, kind=ResourceKind.LOAD),
        Resource(_RESOURCE_VMEM_LOAD, capacity_per_cycle=1, kind=ResourceKind.LOAD),
        Resource(_RESOURCE_VMEM_STORE, capacity_per_cycle=1, kind=ResourceKind.STORE),
        Resource(_RESOURCE_WMMA, capacity_per_cycle=1, kind=ResourceKind.MATRIX),
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
            _SCHEDULE_WMMA,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=32,
            issue_uses=(IssueUse(_RESOURCE_WMMA, cycles=1, units=1),),
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
            asm_forms=_asm(results=("dst",), immediates=("imm32",)),
            schedule_class=_SCHEDULE_SALU,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        *_GFX11_CORE_OVERLAY_DESCRIPTORS,
    ),
)

AMDGPU_GFX12_CORE_DESCRIPTOR_SET = DescriptorSet(
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
        Resource(_RESOURCE_SALU, capacity_per_cycle=1, kind=ResourceKind.SCALAR_ALU),
        Resource(_RESOURCE_VALU, capacity_per_cycle=1, kind=ResourceKind.VECTOR_ALU),
        Resource(_RESOURCE_SMEM, capacity_per_cycle=1, kind=ResourceKind.LOAD),
        Resource(_RESOURCE_VMEM_LOAD, capacity_per_cycle=1, kind=ResourceKind.LOAD),
        Resource(_RESOURCE_VMEM_STORE, capacity_per_cycle=1, kind=ResourceKind.STORE),
        Resource(_RESOURCE_WMMA, capacity_per_cycle=1, kind=ResourceKind.MATRIX),
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
            _SCHEDULE_WMMA,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=32,
            issue_uses=(IssueUse(_RESOURCE_WMMA, cycles=1, units=1),),
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
            asm_forms=_asm(results=("dst",), immediates=("imm32",)),
            schedule_class=_SCHEDULE_SALU,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        *_GFX12_CORE_OVERLAY_DESCRIPTORS,
    ),
)

AMDGPU_GFX1250_CORE_DESCRIPTOR_SET = DescriptorSet(
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
        Resource(_RESOURCE_SALU, capacity_per_cycle=1, kind=ResourceKind.SCALAR_ALU),
        Resource(_RESOURCE_VALU, capacity_per_cycle=1, kind=ResourceKind.VECTOR_ALU),
        Resource(_RESOURCE_SMEM, capacity_per_cycle=1, kind=ResourceKind.LOAD),
        Resource(_RESOURCE_VMEM_LOAD, capacity_per_cycle=1, kind=ResourceKind.LOAD),
        Resource(_RESOURCE_VMEM_STORE, capacity_per_cycle=1, kind=ResourceKind.STORE),
        Resource(_RESOURCE_WMMA, capacity_per_cycle=1, kind=ResourceKind.MATRIX),
        Resource(_RESOURCE_SWMMAC, capacity_per_cycle=1, kind=ResourceKind.MATRIX),
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
            _SCHEDULE_WMMA,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=32,
            issue_uses=(IssueUse(_RESOURCE_WMMA, cycles=1, units=1),),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_WMMA_SCALE,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=32,
            issue_uses=(IssueUse(_RESOURCE_WMMA, cycles=1, units=1),),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_SWMMAC,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=32,
            issue_uses=(IssueUse(_RESOURCE_SWMMAC, cycles=1, units=1),),
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
            asm_forms=_asm(results=("dst",), immediates=("imm32",)),
            schedule_class=_SCHEDULE_SALU,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        *_GFX1250_CORE_OVERLAY_DESCRIPTORS,
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
