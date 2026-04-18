# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Descriptor input data for x86 AVX512 target-low shards."""

from __future__ import annotations

from pathlib import Path

from loom.target.low_descriptors import (
    Constraint,
    ConstraintKind,
    Descriptor,
    DescriptorFlag,
    DescriptorSet,
    Effect,
    EffectFlag,
    EffectKind,
    Immediate,
    ImmediateFlag,
    ImmediateKind,
    IssueUse,
    LatencyKind,
    MemorySpace,
    ModelQuality,
    Operand,
    OperandRole,
    RegClass,
    RegClassAlt,
    RegClassFlag,
    Resource,
    ResourceKind,
    ScheduleClass,
    ScheduleClassFlag,
)

_REG_GPR64 = "x86.gpr64"
_REG_ZMM = "x86.zmm"
_REG_K = "x86.k"

_RESOURCE_SCALAR = "x86.scalar"
_RESOURCE_VECTOR = "x86.vector"
_RESOURCE_DOT = "x86.vector.dot"
_RESOURCE_MASK = "x86.mask"
_RESOURCE_LOAD = "x86.load"
_RESOURCE_STORE = "x86.store"
_RESOURCE_ADDRESS = "x86.address"
_RESOURCE_CONTROL = "x86.control"

_SCHEDULE_SCALAR = "x86.scalar"
_SCHEDULE_VECTOR_I32 = "x86.vector.i32"
_SCHEDULE_VECTOR_DOT = "x86.vector.dot"
_SCHEDULE_MASK = "x86.mask"
_SCHEDULE_MEMORY_LOAD = "x86.memory.load"
_SCHEDULE_MEMORY_STORE = "x86.memory.store"
_SCHEDULE_CONTROL = "x86.control"

_GPR64_ALT = (RegClassAlt(_REG_GPR64),)
_ZMM_ALT = (RegClassAlt(_REG_ZMM),)
_K_ALT = (RegClassAlt(_REG_K),)


def _gpr64_result(field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _GPR64_ALT)


def _gpr64_operand(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _GPR64_ALT)


def _gpr64_resource(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.RESOURCE, _GPR64_ALT)


def _zmm_result(field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _ZMM_ALT)


def _zmm_operand(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _ZMM_ALT)


def _k_result(field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _K_ALT)


def _k_operand(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _K_ALT)


_DISP32_IMMEDIATE = Immediate(
    "disp32",
    ImmediateKind.SIGNED,
    bit_width=32,
    signed_min=-(2**31),
    unsigned_max=(2**31) - 1,
)

_TARGET_BLOCK_IMMEDIATE = Immediate(
    "target_block",
    ImmediateKind.ORDINAL,
    flags=(ImmediateFlag.SYMBOLIC, ImmediateFlag.RELATIVE),
    bit_width=32,
    unsigned_max=(2**32) - 1,
)

_LOAD_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.GENERIC,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=512,
)

_STORE_EFFECT = Effect(
    EffectKind.WRITE,
    memory_space=MemorySpace.GENERIC,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=512,
)

_CONTROL_EFFECT = Effect(
    EffectKind.CONTROL,
    flags=(EffectFlag.ORDERED,),
)

_DOT_ACCUMULATOR_CONSTRAINTS = (
    Constraint(ConstraintKind.TIED, 0, 1),
    Constraint(ConstraintKind.DESTRUCTIVE, 0, 1),
)

X86_AVX512_CORE_DESCRIPTOR_SET = DescriptorSet(
    key="x86.avx512.core",
    target_key="x86",
    feature_key="x86.avx512.v1",
    c_header_path=Path("loom/src/loom/target/arch/x86/avx512_descriptors.h"),
    c_source_path=Path("loom/src/loom/target/arch/x86/avx512_descriptors.c"),
    header_guard="LOOM_TARGET_ARCH_X86_AVX512_DESCRIPTORS_H_",
    public_header="loom/target/arch/x86/avx512_descriptors.h",
    function_name="loom_x86_avx512_core_descriptor_set",
    c_table_prefix="X86Avx512Core",
    c_enum_prefix="X86_AVX512_CORE",
    generator_version=1,
    reg_classes=(
        RegClass(_REG_GPR64, 64, flags=(RegClassFlag.PHYSICAL,), physical_count=16),
        RegClass(_REG_ZMM, 512, flags=(RegClassFlag.PHYSICAL,), physical_count=32),
        RegClass(_REG_K, 64, flags=(RegClassFlag.PHYSICAL,), physical_count=8),
    ),
    resources=(
        Resource(_RESOURCE_SCALAR, capacity_per_cycle=1, kind=ResourceKind.SCALAR_ALU),
        Resource(_RESOURCE_VECTOR, capacity_per_cycle=1, kind=ResourceKind.VECTOR_ALU),
        Resource(_RESOURCE_DOT, capacity_per_cycle=1, kind=ResourceKind.VECTOR_ALU),
        Resource(_RESOURCE_MASK, capacity_per_cycle=1, kind=ResourceKind.SCALAR_ALU),
        Resource(_RESOURCE_LOAD, capacity_per_cycle=1, kind=ResourceKind.LOAD),
        Resource(_RESOURCE_STORE, capacity_per_cycle=1, kind=ResourceKind.STORE),
        Resource(_RESOURCE_ADDRESS, capacity_per_cycle=1, kind=ResourceKind.ADDRESS),
        Resource(_RESOURCE_CONTROL, capacity_per_cycle=1, kind=ResourceKind.CONTROL),
    ),
    schedule_classes=(
        ScheduleClass(
            _SCHEDULE_SCALAR,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_SCALAR, cycles=1, units=1),),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_VECTOR_I32,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_VECTOR, cycles=1, units=1),),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_VECTOR_DOT,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=4,
            issue_uses=(IssueUse(_RESOURCE_DOT, cycles=1, units=1),),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_MASK,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_MASK, cycles=1, units=1),),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_MEMORY_LOAD,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=4,
            issue_uses=(
                IssueUse(_RESOURCE_ADDRESS, cycles=1, units=1),
                IssueUse(_RESOURCE_LOAD, cycles=1, units=1),
            ),
            flags=(ScheduleClassFlag.MAY_LOAD,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_MEMORY_STORE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(
                IssueUse(_RESOURCE_ADDRESS, cycles=1, units=1),
                IssueUse(_RESOURCE_STORE, cycles=1, units=1),
            ),
            flags=(ScheduleClassFlag.MAY_STORE,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_CONTROL,
            latency_kind=LatencyKind.EXACT,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.EXACT,
        ),
    ),
    descriptors=(
        Descriptor(
            key="x86.avx512.vpaddd.zmm",
            mnemonic="vpaddd",
            semantic_tag="integer.add.i32x16",
            operands=(_zmm_result(), _zmm_operand("lhs"), _zmm_operand("rhs")),
            schedule_class=_SCHEDULE_VECTOR_I32,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="x86.avx512.vmovdqu32.load.zmm",
            mnemonic="vmovdqu32",
            semantic_tag="memory.load.i32x16",
            operands=(_zmm_result(), _gpr64_resource("base")),
            immediates=(_DISP32_IMMEDIATE,),
            effects=(_LOAD_EFFECT,),
            schedule_class=_SCHEDULE_MEMORY_LOAD,
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
        Descriptor(
            key="x86.avx512.vmovdqu32.store.zmm",
            mnemonic="vmovdqu32",
            semantic_tag="memory.store.i32x16",
            operands=(_zmm_operand("value"), _gpr64_resource("base")),
            immediates=(_DISP32_IMMEDIATE,),
            effects=(_STORE_EFFECT,),
            schedule_class=_SCHEDULE_MEMORY_STORE,
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
        Descriptor(
            key="x86.avx512.vpdpbusd.zmm",
            mnemonic="vpdpbusd",
            semantic_tag="dot.u8s8.i32x16",
            operands=(
                _zmm_result(),
                _zmm_operand("acc"),
                _zmm_operand("lhs"),
                _zmm_operand("rhs"),
            ),
            constraints=_DOT_ACCUMULATOR_CONSTRAINTS,
            schedule_class=_SCHEDULE_VECTOR_DOT,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="x86.avx512.vdpbf16ps.zmm",
            mnemonic="vdpbf16ps",
            semantic_tag="dot.bf16.f32x16",
            operands=(
                _zmm_result(),
                _zmm_operand("acc"),
                _zmm_operand("lhs"),
                _zmm_operand("rhs"),
            ),
            constraints=_DOT_ACCUMULATOR_CONSTRAINTS,
            schedule_class=_SCHEDULE_VECTOR_DOT,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="x86.avx512.kandq",
            mnemonic="kandq",
            semantic_tag="mask.and.i64",
            operands=(_k_result(), _k_operand("lhs"), _k_operand("rhs")),
            schedule_class=_SCHEDULE_MASK,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="x86.avx512.mov.gpr64",
            mnemonic="mov",
            semantic_tag="integer.move.i64",
            operands=(_gpr64_result(), _gpr64_operand("src")),
            schedule_class=_SCHEDULE_SCALAR,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="x86.avx512.jmp",
            mnemonic="jmp",
            semantic_tag="control.branch",
            operands=(),
            immediates=(_TARGET_BLOCK_IMMEDIATE,),
            effects=(_CONTROL_EFFECT,),
            schedule_class=_SCHEDULE_CONTROL,
            flags=(DescriptorFlag.SIDE_EFFECTING, DescriptorFlag.TERMINATOR),
        ),
    ),
)
