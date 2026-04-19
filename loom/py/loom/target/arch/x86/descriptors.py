# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Descriptor input data for x86 AVX512 target-low shards."""

from __future__ import annotations

from pathlib import Path

from loom.target.arch.x86.packed_dot_data import (
    CONTRACT_FLAG_SATURATING,
    NUMERIC_BF16,
    NUMERIC_F16,
    NUMERIC_F32,
    NUMERIC_I8,
    NUMERIC_I16,
    NUMERIC_I32,
    NUMERIC_U8,
    NUMERIC_U16,
    X86_PACKED_DOT_DESCRIPTORS,
    PackedDotDescriptor,
)
from loom.target.low_descriptors import (
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
    SpillSlotSpace,
)

_REG_GPR64 = "x86.gpr64"
_REG_XMM = "x86.xmm"
_REG_YMM = "x86.ymm"
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
_SCHEDULE_VECTOR_I32_ZMM = "x86.vector.i32.512"
_SCHEDULE_VECTOR_DOT_XMM = "x86.vector.dot.128"
_SCHEDULE_VECTOR_DOT_YMM = "x86.vector.dot.256"
_SCHEDULE_VECTOR_DOT_ZMM = "x86.vector.dot.512"
_SCHEDULE_MASK = "x86.mask"
_SCHEDULE_MEMORY_LOAD = "x86.memory.load"
_SCHEDULE_MEMORY_STORE = "x86.memory.store"
_SCHEDULE_CONTROL = "x86.control"

_GPR64_ALT = (RegClassAlt(_REG_GPR64),)
_XMM_ALT = (RegClassAlt(_REG_XMM),)
_YMM_ALT = (RegClassAlt(_REG_YMM),)
_ZMM_ALT = (RegClassAlt(_REG_ZMM),)
_K_ALT = (RegClassAlt(_REG_K),)


def _vector_lane_units(vector_bit_width: int) -> int:
    if vector_bit_width % 128 != 0:
        raise ValueError(f"unsupported x86 vector width {vector_bit_width}")
    return vector_bit_width // 128


def _vector_dot_schedule_class(vector_bit_width: int) -> str:
    match vector_bit_width:
        case 128:
            return _SCHEDULE_VECTOR_DOT_XMM
        case 256:
            return _SCHEDULE_VECTOR_DOT_YMM
        case 512:
            return _SCHEDULE_VECTOR_DOT_ZMM
        case _:
            raise ValueError(f"unsupported x86 dot schedule width {vector_bit_width}")


def _asm(
    *,
    results: tuple[str, ...] = (),
    operands: tuple[str, ...] = (),
    immediates: tuple[str, ...] = (),
) -> tuple[AsmForm, ...]:
    return (
        AsmForm(
            results=results,
            operands=operands,
            immediates=tuple(AsmImmediate(field_name) for field_name in immediates),
        ),
    )


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


def _vector_reg_alt(vector_bit_width: int) -> tuple[RegClassAlt, ...]:
    match vector_bit_width:
        case 128:
            return _XMM_ALT
        case 256:
            return _YMM_ALT
        case 512:
            return _ZMM_ALT
        case _:
            raise ValueError(
                f"unsupported x86 vector register width {vector_bit_width}"
            )


def _vector_result(vector_bit_width: int, field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _vector_reg_alt(vector_bit_width))


def _vector_operand(vector_bit_width: int, field_name: str) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _vector_reg_alt(vector_bit_width))


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

_PACKED_DOT_NUMERIC_TAGS = {
    NUMERIC_I8: "s8",
    NUMERIC_U8: "u8",
    NUMERIC_I16: "s16",
    NUMERIC_U16: "u16",
    NUMERIC_F16: "f16",
    NUMERIC_BF16: "bf16",
    NUMERIC_I32: "i32",
    NUMERIC_F32: "f32",
}


def _packed_dot_semantic_tag(descriptor: PackedDotDescriptor) -> str:
    lhs_tag = _PACKED_DOT_NUMERIC_TAGS[descriptor.lhs_numeric_type]
    rhs_tag = _PACKED_DOT_NUMERIC_TAGS[descriptor.rhs_numeric_type]
    result_tag = _PACKED_DOT_NUMERIC_TAGS[descriptor.result_numeric_type]
    saturating_suffix = ".sat" if descriptor.flags & CONTRACT_FLAG_SATURATING else ""
    return (
        f"dot.{lhs_tag}{rhs_tag}.{result_tag}x"
        f"{descriptor.result_lane_count}{saturating_suffix}"
    )


def _packed_dot_descriptor(descriptor: PackedDotDescriptor) -> Descriptor:
    return Descriptor(
        key=descriptor.key,
        mnemonic=descriptor.mnemonic,
        semantic_tag=_packed_dot_semantic_tag(descriptor),
        operands=(
            _vector_result(descriptor.vector_bit_width),
            _vector_operand(descriptor.vector_bit_width, "acc"),
            _vector_operand(descriptor.vector_bit_width, "lhs"),
            _vector_operand(descriptor.vector_bit_width, "rhs"),
        ),
        constraints=_DOT_ACCUMULATOR_CONSTRAINTS,
        schedule_class=_vector_dot_schedule_class(descriptor.vector_bit_width),
        feature_mask_words=(descriptor.required_feature_bits,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
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
        RegClass(
            _REG_GPR64,
            64,
            SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=16,
        ),
        RegClass(
            _REG_ZMM,
            512,
            SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=32,
        ),
        RegClass(
            _REG_K,
            64,
            SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=8,
        ),
    ),
    resources=(
        Resource(_RESOURCE_SCALAR, capacity_per_cycle=1, kind=ResourceKind.SCALAR_ALU),
        Resource(
            _RESOURCE_VECTOR,
            capacity_per_cycle=4,
            kind=ResourceKind.VECTOR_ALU,
            contention_group_id=1,
        ),
        Resource(
            _RESOURCE_DOT,
            capacity_per_cycle=4,
            kind=ResourceKind.VECTOR_ALU,
            contention_group_id=1,
        ),
        Resource(_RESOURCE_MASK, capacity_per_cycle=1, kind=ResourceKind.SCALAR_ALU),
        Resource(
            _RESOURCE_LOAD,
            capacity_per_cycle=4,
            kind=ResourceKind.LOAD,
            contention_group_id=2,
        ),
        Resource(
            _RESOURCE_STORE,
            capacity_per_cycle=4,
            kind=ResourceKind.STORE,
            contention_group_id=3,
        ),
        Resource(
            _RESOURCE_ADDRESS,
            capacity_per_cycle=1,
            kind=ResourceKind.ADDRESS,
            contention_group_id=4,
        ),
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
            _SCHEDULE_VECTOR_I32_ZMM,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(
                IssueUse(_RESOURCE_VECTOR, cycles=1, units=_vector_lane_units(512)),
            ),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_VECTOR_DOT_ZMM,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=4,
            issue_uses=(
                IssueUse(_RESOURCE_DOT, cycles=1, units=_vector_lane_units(512)),
            ),
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
                IssueUse(_RESOURCE_LOAD, cycles=1, units=_vector_lane_units(512)),
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
                IssueUse(_RESOURCE_STORE, cycles=1, units=_vector_lane_units(512)),
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
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_VECTOR_I32_ZMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="x86.avx512.vpmulld.zmm",
            mnemonic="vpmulld",
            semantic_tag="integer.mul.i32x16",
            operands=(_zmm_result(), _zmm_operand("lhs"), _zmm_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_VECTOR_I32_ZMM,
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
            asm_forms=_asm(results=("dst",), operands=("acc", "lhs", "rhs")),
            schedule_class=_SCHEDULE_VECTOR_DOT_ZMM,
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
            asm_forms=_asm(results=("dst",), operands=("acc", "lhs", "rhs")),
            schedule_class=_SCHEDULE_VECTOR_DOT_ZMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="x86.avx512.kandq",
            mnemonic="kandq",
            semantic_tag="mask.and.i64",
            operands=(_k_result(), _k_operand("lhs"), _k_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_MASK,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="x86.avx512.mov.gpr64",
            mnemonic="mov",
            semantic_tag="integer.move.i64",
            operands=(_gpr64_result(), _gpr64_operand("src")),
            asm_forms=_asm(results=("dst",), operands=("src",)),
            schedule_class=_SCHEDULE_SCALAR,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="x86.avx512.jmp",
            mnemonic="jmp",
            semantic_tag="control.branch",
            operands=(),
            immediates=(_TARGET_BLOCK_IMMEDIATE,),
            asm_forms=_asm(immediates=("target_block",)),
            effects=(_CONTROL_EFFECT,),
            schedule_class=_SCHEDULE_CONTROL,
            flags=(DescriptorFlag.SIDE_EFFECTING, DescriptorFlag.TERMINATOR),
        ),
    ),
)

X86_PACKED_DOT_DESCRIPTOR_SET = DescriptorSet(
    key="x86.packed_dot.core",
    target_key="x86",
    feature_key="x86.packed_dot.v1",
    c_header_path=Path("loom/src/loom/target/arch/x86/packed_dot_descriptors.h"),
    c_source_path=Path("loom/src/loom/target/arch/x86/packed_dot_descriptors.c"),
    header_guard="LOOM_TARGET_ARCH_X86_PACKED_DOT_DESCRIPTORS_H_",
    public_header="loom/target/arch/x86/packed_dot_descriptors.h",
    function_name="loom_x86_packed_dot_core_descriptor_set",
    c_table_prefix="X86PackedDotCore",
    c_enum_prefix="X86_PACKED_DOT_CORE",
    generator_version=1,
    reg_classes=(
        RegClass(
            _REG_XMM,
            128,
            SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=32,
        ),
        RegClass(
            _REG_YMM,
            256,
            SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=32,
        ),
        RegClass(
            _REG_ZMM,
            512,
            SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=32,
        ),
    ),
    resources=(
        Resource(
            _RESOURCE_DOT,
            capacity_per_cycle=4,
            kind=ResourceKind.VECTOR_ALU,
            contention_group_id=1,
        ),
    ),
    schedule_classes=(
        ScheduleClass(
            _SCHEDULE_VECTOR_DOT_XMM,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=4,
            issue_uses=(
                IssueUse(_RESOURCE_DOT, cycles=1, units=_vector_lane_units(128)),
            ),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_VECTOR_DOT_YMM,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=4,
            issue_uses=(
                IssueUse(_RESOURCE_DOT, cycles=1, units=_vector_lane_units(256)),
            ),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_VECTOR_DOT_ZMM,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=4,
            issue_uses=(
                IssueUse(_RESOURCE_DOT, cycles=1, units=_vector_lane_units(512)),
            ),
            model_quality=ModelQuality.ESTIMATED,
        ),
    ),
    descriptors=tuple(
        _packed_dot_descriptor(descriptor) for descriptor in X86_PACKED_DOT_DESCRIPTORS
    ),
)
