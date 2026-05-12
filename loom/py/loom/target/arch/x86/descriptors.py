# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Descriptor input data for x86 AVX512 target-low shards."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TypeVar

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
    EnumDomain,
    EnumValue,
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

_T = TypeVar("_T", RegClass, Resource, ScheduleClass, EnumDomain)

_REG_GPR32 = "x86.gpr32"
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
_SCHEDULE_VECTOR_I32_XMM = "x86.vector.i32.128"
_SCHEDULE_VECTOR_I32_ZMM = "x86.vector.i32.512"
_SCHEDULE_VECTOR_F32_XMM = "x86.vector.f32.128"
_SCHEDULE_VECTOR_F32_ZMM = "x86.vector.f32.512"
_SCHEDULE_VECTOR_FMA_F32_XMM = "x86.vector.fma.f32.128"
_SCHEDULE_VECTOR_FMA_F32_ZMM = "x86.vector.fma.f32.512"
_SCHEDULE_VECTOR_COMPARE_XMM = "x86.vector.compare.128"
_SCHEDULE_VECTOR_COMPARE_ZMM = "x86.vector.compare.512"
_SCHEDULE_VECTOR_DOT_XMM = "x86.vector.dot.128"
_SCHEDULE_VECTOR_DOT_YMM = "x86.vector.dot.256"
_SCHEDULE_VECTOR_DOT_ZMM = "x86.vector.dot.512"
_SCHEDULE_MASK = "x86.mask"
_SCHEDULE_MEMORY_LOAD_XMM = "x86.memory.load.128"
_SCHEDULE_MEMORY_LOAD_ZMM = "x86.memory.load"
_SCHEDULE_MEMORY_STORE_XMM = "x86.memory.store.128"
_SCHEDULE_MEMORY_STORE_ZMM = "x86.memory.store"
_SCHEDULE_CONTROL = "x86.control"

_GPR32_ALT = (RegClassAlt(_REG_GPR32),)
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


def _vector_i32_schedule_class(vector_bit_width: int) -> str:
    match vector_bit_width:
        case 128:
            return _SCHEDULE_VECTOR_I32_XMM
        case 512:
            return _SCHEDULE_VECTOR_I32_ZMM
        case _:
            raise ValueError(f"unsupported x86 i32 vector width {vector_bit_width}")


def _vector_f32_schedule_class(vector_bit_width: int) -> str:
    match vector_bit_width:
        case 128:
            return _SCHEDULE_VECTOR_F32_XMM
        case 512:
            return _SCHEDULE_VECTOR_F32_ZMM
        case _:
            raise ValueError(f"unsupported x86 f32 vector width {vector_bit_width}")


def _vector_fma_f32_schedule_class(vector_bit_width: int) -> str:
    match vector_bit_width:
        case 128:
            return _SCHEDULE_VECTOR_FMA_F32_XMM
        case 512:
            return _SCHEDULE_VECTOR_FMA_F32_ZMM
        case _:
            raise ValueError(f"unsupported x86 f32 FMA vector width {vector_bit_width}")


def _vector_compare_schedule_class(vector_bit_width: int) -> str:
    match vector_bit_width:
        case 128:
            return _SCHEDULE_VECTOR_COMPARE_XMM
        case 512:
            return _SCHEDULE_VECTOR_COMPARE_ZMM
        case _:
            raise ValueError(f"unsupported x86 compare vector width {vector_bit_width}")


def _memory_load_schedule_class(vector_bit_width: int) -> str:
    match vector_bit_width:
        case 128:
            return _SCHEDULE_MEMORY_LOAD_XMM
        case 512:
            return _SCHEDULE_MEMORY_LOAD_ZMM
        case _:
            raise ValueError(f"unsupported x86 memory vector width {vector_bit_width}")


def _memory_store_schedule_class(vector_bit_width: int) -> str:
    match vector_bit_width:
        case 128:
            return _SCHEDULE_MEMORY_STORE_XMM
        case 512:
            return _SCHEDULE_MEMORY_STORE_ZMM
        case _:
            raise ValueError(f"unsupported x86 memory vector width {vector_bit_width}")


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


def _gpr64_result(field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _GPR64_ALT)


def _gpr64_operand(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _GPR64_ALT)


def _gpr64_resource(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.RESOURCE, _GPR64_ALT)


def _gpr32_result(field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _GPR32_ALT)


def _gpr32_operand(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _GPR32_ALT)


def _xmm_operand(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _XMM_ALT)


def _xmm_result(field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _XMM_ALT)


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


def _vector_asm_mnemonic(mnemonic: str, vector_bit_width: int) -> str:
    match vector_bit_width:
        case 128:
            return f"{mnemonic}.xmm"
        case 256:
            return f"{mnemonic}.ymm"
        case 512:
            return mnemonic
        case _:
            raise ValueError(f"unsupported x86 vector width {vector_bit_width}")


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

_IMM64_IMMEDIATE = Immediate(
    "imm64",
    ImmediateKind.SIGNED,
    bit_width=64,
    signed_min=-(2**63) + 1,
    unsigned_max=(2**63) - 1,
)

_LANE_I32X4_IMMEDIATE = Immediate(
    "lane",
    ImmediateKind.UNSIGNED,
    bit_width=8,
    unsigned_max=3,
)

_SHUFFLE_4X2_CONTROL_IMMEDIATE = Immediate(
    "control",
    ImmediateKind.UNSIGNED,
    bit_width=8,
    unsigned_max=255,
)

_INSERTPS_CONTROL_IMMEDIATE = Immediate(
    "control",
    ImmediateKind.UNSIGNED,
    bit_width=8,
    unsigned_max=255,
)

_ADDRESS_SCALE_ENUM = "x86.address.scale"

_ADDRESS_SCALE_IMMEDIATE = Immediate(
    "scale",
    ImmediateKind.ENUM,
    enum_domain=_ADDRESS_SCALE_ENUM,
    bit_width=8,
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


def _load_effect(width_bits: int) -> Effect:
    return Effect(
        EffectKind.READ,
        memory_space=MemorySpace.GENERIC,
        flags=(EffectFlag.DEPENDENCY,),
        width_bits=width_bits,
    )


def _store_effect(width_bits: int) -> Effect:
    return Effect(
        EffectKind.WRITE,
        memory_space=MemorySpace.GENERIC,
        flags=(EffectFlag.DEPENDENCY,),
        width_bits=width_bits,
    )


_CONTROL_EFFECT = Effect(
    EffectKind.CONTROL,
    flags=(EffectFlag.ORDERED,),
)

_DESTRUCTIVE_ACCUMULATOR_CONSTRAINTS = (
    Constraint(ConstraintKind.TIED, 0, 1),
    Constraint(ConstraintKind.DESTRUCTIVE, 0, 1),
)

_GPR_DESTRUCTIVE_LHS_CONSTRAINTS = (
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
    asm_forms: tuple[AsmForm, ...] = ()
    if descriptor.vector_bit_width == 512:
        asm_forms = _asm(
            mnemonic=_vector_asm_mnemonic(
                descriptor.mnemonic, descriptor.vector_bit_width
            ),
            results=("dst",),
            operands=("acc", "lhs", "rhs"),
        )
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
        constraints=_DESTRUCTIVE_ACCUMULATOR_CONSTRAINTS,
        asm_forms=asm_forms,
        schedule_class=_vector_dot_schedule_class(descriptor.vector_bit_width),
        feature_mask_words=(descriptor.required_feature_bits,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _zmm_i32_binary_descriptor(
    *,
    key: str,
    mnemonic: str,
    semantic_tag: str,
) -> Descriptor:
    return _vector_i32_binary_descriptor(
        vector_bit_width=512,
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
    )


def _vector_i32_binary_descriptor(
    *,
    vector_bit_width: int,
    key: str,
    mnemonic: str,
    semantic_tag: str,
) -> Descriptor:
    return Descriptor(
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        operands=(
            _vector_result(vector_bit_width),
            _vector_operand(vector_bit_width, "lhs"),
            _vector_operand(vector_bit_width, "rhs"),
        ),
        asm_forms=_asm(
            mnemonic=_vector_asm_mnemonic(mnemonic, vector_bit_width),
            results=("dst",),
            operands=("lhs", "rhs"),
        ),
        schedule_class=_vector_i32_schedule_class(vector_bit_width),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _vector_f32_binary_descriptor(
    *,
    vector_bit_width: int,
    key: str,
    mnemonic: str,
    semantic_tag: str,
) -> Descriptor:
    return Descriptor(
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        operands=(
            _vector_result(vector_bit_width),
            _vector_operand(vector_bit_width, "lhs"),
            _vector_operand(vector_bit_width, "rhs"),
        ),
        asm_forms=_asm(
            mnemonic=_vector_asm_mnemonic(mnemonic, vector_bit_width),
            results=("dst",),
            operands=("lhs", "rhs"),
        ),
        schedule_class=_vector_f32_schedule_class(vector_bit_width),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _scalar_f32_binary_descriptor(
    *,
    key: str,
    mnemonic: str,
    semantic_tag: str,
) -> Descriptor:
    return Descriptor(
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        operands=(
            _xmm_result(),
            _xmm_operand("lhs"),
            _xmm_operand("rhs"),
        ),
        asm_forms=_asm(
            mnemonic=f"{mnemonic}.xmm",
            results=("dst",),
            operands=("lhs", "rhs"),
        ),
        schedule_class=_SCHEDULE_VECTOR_F32_XMM,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _zmm_splat_descriptor(
    *,
    key: str,
    mnemonic: str,
    semantic_tag: str,
    operand: Operand,
    schedule_class: str,
) -> Descriptor:
    return _vector_splat_descriptor(
        vector_bit_width=512,
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        operand=operand,
        schedule_class=schedule_class,
    )


def _vector_splat_descriptor(
    *,
    vector_bit_width: int,
    key: str,
    mnemonic: str,
    semantic_tag: str,
    operand: Operand,
    schedule_class: str,
) -> Descriptor:
    return Descriptor(
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        operands=(_vector_result(vector_bit_width), operand),
        asm_forms=_asm(
            mnemonic=_vector_asm_mnemonic(mnemonic, vector_bit_width),
            results=("dst",),
            operands=("value",),
        ),
        schedule_class=schedule_class,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _zmm_i32_compare_descriptor(
    *,
    key: str,
    mnemonic: str,
    semantic_tag: str,
) -> Descriptor:
    return _vector_i32_compare_descriptor(
        vector_bit_width=512,
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
    )


def _vector_i32_compare_descriptor(
    *,
    vector_bit_width: int,
    key: str,
    mnemonic: str,
    semantic_tag: str,
) -> Descriptor:
    return Descriptor(
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        operands=(
            _k_result(),
            _vector_operand(vector_bit_width, "lhs"),
            _vector_operand(vector_bit_width, "rhs"),
        ),
        asm_forms=_asm(
            mnemonic=_vector_asm_mnemonic(mnemonic, vector_bit_width),
            results=("dst",),
            operands=("lhs", "rhs"),
        ),
        schedule_class=_vector_compare_schedule_class(vector_bit_width),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _zmm_f32_compare_descriptor(
    *,
    key: str,
    mnemonic: str,
    semantic_tag: str,
) -> Descriptor:
    return _vector_f32_compare_descriptor(
        vector_bit_width=512,
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
    )


def _vector_f32_compare_descriptor(
    *,
    vector_bit_width: int,
    key: str,
    mnemonic: str,
    semantic_tag: str,
) -> Descriptor:
    return Descriptor(
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        operands=(
            _k_result(),
            _vector_operand(vector_bit_width, "lhs"),
            _vector_operand(vector_bit_width, "rhs"),
        ),
        asm_forms=_asm(
            mnemonic=_vector_asm_mnemonic(mnemonic, vector_bit_width),
            results=("dst",),
            operands=("lhs", "rhs"),
        ),
        schedule_class=_vector_compare_schedule_class(vector_bit_width),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _zmm_mask_select_descriptor(
    *,
    key: str,
    mnemonic: str,
    semantic_tag: str,
    schedule_class: str,
) -> Descriptor:
    return _vector_mask_select_descriptor(
        vector_bit_width=512,
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        schedule_class=schedule_class,
    )


def _vector_mask_select_descriptor(
    *,
    vector_bit_width: int,
    key: str,
    mnemonic: str,
    semantic_tag: str,
    schedule_class: str,
) -> Descriptor:
    return Descriptor(
        key=key,
        mnemonic=mnemonic,
        semantic_tag=semantic_tag,
        operands=(
            _vector_result(vector_bit_width),
            _k_operand("mask"),
            _vector_operand(vector_bit_width, "true_value"),
            _vector_operand(vector_bit_width, "false_value"),
        ),
        asm_forms=_asm(
            mnemonic=_vector_asm_mnemonic(mnemonic, vector_bit_width),
            results=("dst",),
            operands=("mask", "true_value", "false_value"),
        ),
        schedule_class=schedule_class,
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
            _REG_GPR32,
            32,
            SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=16,
            alias_set_id=1,
        ),
        RegClass(
            _REG_GPR64,
            64,
            SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=16,
            alias_set_id=1,
        ),
        RegClass(
            _REG_XMM,
            128,
            SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=32,
            alias_set_id=2,
        ),
        RegClass(
            _REG_ZMM,
            512,
            SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=32,
            alias_set_id=2,
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
            _SCHEDULE_VECTOR_I32_XMM,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(
                IssueUse(_RESOURCE_VECTOR, cycles=1, units=_vector_lane_units(128)),
            ),
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
            _SCHEDULE_VECTOR_F32_XMM,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(
                IssueUse(_RESOURCE_VECTOR, cycles=1, units=_vector_lane_units(128)),
            ),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_VECTOR_F32_ZMM,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(
                IssueUse(_RESOURCE_VECTOR, cycles=1, units=_vector_lane_units(512)),
            ),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_VECTOR_FMA_F32_XMM,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=4,
            issue_uses=(
                IssueUse(_RESOURCE_VECTOR, cycles=1, units=_vector_lane_units(128)),
            ),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_VECTOR_FMA_F32_ZMM,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=4,
            issue_uses=(
                IssueUse(_RESOURCE_VECTOR, cycles=1, units=_vector_lane_units(512)),
            ),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_VECTOR_COMPARE_XMM,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(
                IssueUse(_RESOURCE_VECTOR, cycles=1, units=_vector_lane_units(128)),
            ),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_VECTOR_COMPARE_ZMM,
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
            _SCHEDULE_MEMORY_LOAD_ZMM,
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
            _SCHEDULE_MEMORY_LOAD_XMM,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=4,
            issue_uses=(
                IssueUse(_RESOURCE_ADDRESS, cycles=1, units=1),
                IssueUse(_RESOURCE_LOAD, cycles=1, units=_vector_lane_units(128)),
            ),
            flags=(ScheduleClassFlag.MAY_LOAD,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_MEMORY_STORE_ZMM,
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
            _SCHEDULE_MEMORY_STORE_XMM,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(
                IssueUse(_RESOURCE_ADDRESS, cycles=1, units=1),
                IssueUse(_RESOURCE_STORE, cycles=1, units=_vector_lane_units(128)),
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
    enum_domains=(
        EnumDomain(
            _ADDRESS_SCALE_ENUM,
            (
                EnumValue("1", 1),
                EnumValue("2", 2),
                EnumValue("4", 4),
                EnumValue("8", 8),
            ),
        ),
    ),
    descriptors=(
        Descriptor(
            key="x86.scalar.add.gpr32",
            mnemonic="add",
            semantic_tag="integer.add.i32",
            operands=(
                _gpr32_result(),
                _gpr32_operand("lhs"),
                _gpr32_operand("rhs"),
            ),
            constraints=_GPR_DESTRUCTIVE_LHS_CONSTRAINTS,
            asm_forms=_asm(
                mnemonic="add.gpr32",
                results=("dst",),
                operands=("lhs", "rhs"),
            ),
            schedule_class=_SCHEDULE_SCALAR,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        _vector_splat_descriptor(
            vector_bit_width=128,
            key="x86.avx512.vpbroadcastd.xmm",
            mnemonic="vpbroadcastd",
            semantic_tag="integer.splat.i32x4",
            operand=_gpr32_operand("value"),
            schedule_class=_SCHEDULE_VECTOR_I32_XMM,
        ),
        Descriptor(
            key="x86.avx512.vpextrd.gpr32.xmm",
            mnemonic="vpextrd",
            semantic_tag="integer.extract.i32x4",
            operands=(_gpr32_result(), _xmm_operand("source")),
            immediates=(_LANE_I32X4_IMMEDIATE,),
            asm_forms=_asm(
                mnemonic="vpextrd.xmm",
                results=("dst",),
                operands=("source",),
                immediates=("lane",),
            ),
            schedule_class=_SCHEDULE_VECTOR_I32_XMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="x86.avx512.vpinsrd.xmm",
            mnemonic="vpinsrd",
            semantic_tag="integer.insert.i32x4",
            operands=(
                _vector_result(128),
                _xmm_operand("dest"),
                _gpr32_operand("value"),
            ),
            immediates=(_LANE_I32X4_IMMEDIATE,),
            asm_forms=_asm(
                mnemonic="vpinsrd.xmm",
                results=("dst",),
                operands=("dest", "value"),
                immediates=("lane",),
            ),
            schedule_class=_SCHEDULE_VECTOR_I32_XMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="x86.avx512.vpshufd.xmm",
            mnemonic="vpshufd",
            semantic_tag="integer.shuffle.i32x4",
            operands=(_vector_result(128), _xmm_operand("source")),
            immediates=(_SHUFFLE_4X2_CONTROL_IMMEDIATE,),
            asm_forms=_asm(
                mnemonic="vpshufd.xmm",
                results=("dst",),
                operands=("source",),
                immediates=("control",),
            ),
            schedule_class=_SCHEDULE_VECTOR_I32_XMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        _vector_splat_descriptor(
            vector_bit_width=128,
            key="x86.avx512.vbroadcastss.xmm",
            mnemonic="vbroadcastss",
            semantic_tag="float.splat.f32x4",
            operand=_xmm_operand("value"),
            schedule_class=_SCHEDULE_VECTOR_F32_XMM,
        ),
        Descriptor(
            key="x86.avx512.vpermilps.xmm",
            mnemonic="vpermilps",
            semantic_tag="float.shuffle.f32x4",
            operands=(_vector_result(128), _xmm_operand("source")),
            immediates=(_SHUFFLE_4X2_CONTROL_IMMEDIATE,),
            asm_forms=_asm(
                mnemonic="vpermilps.xmm",
                results=("dst",),
                operands=("source",),
                immediates=("control",),
            ),
            schedule_class=_SCHEDULE_VECTOR_F32_XMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="x86.avx512.vinsertps.xmm",
            mnemonic="vinsertps",
            semantic_tag="float.insert.f32x4",
            operands=(
                _vector_result(128),
                _xmm_operand("dest"),
                _xmm_operand("value"),
            ),
            immediates=(_INSERTPS_CONTROL_IMMEDIATE,),
            asm_forms=_asm(
                mnemonic="vinsertps.xmm",
                results=("dst",),
                operands=("dest", "value"),
                immediates=("control",),
            ),
            schedule_class=_SCHEDULE_VECTOR_F32_XMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="x86.avx512.vextractf32x4.xmm.zmm",
            mnemonic="vextractf32x4",
            semantic_tag="float.extract.f32x16.quarter",
            operands=(_xmm_result(), _zmm_operand("source")),
            immediates=(_LANE_I32X4_IMMEDIATE,),
            asm_forms=_asm(
                mnemonic="vextractf32x4.xmm",
                results=("dst",),
                operands=("source",),
                immediates=("lane",),
            ),
            schedule_class=_SCHEDULE_VECTOR_F32_ZMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        _vector_i32_binary_descriptor(
            vector_bit_width=128,
            key="x86.avx512.vpaddd.xmm",
            mnemonic="vpaddd",
            semantic_tag="integer.add.i32x4",
        ),
        _vector_i32_binary_descriptor(
            vector_bit_width=128,
            key="x86.avx512.vpsubd.xmm",
            mnemonic="vpsubd",
            semantic_tag="integer.sub.i32x4",
        ),
        _vector_i32_binary_descriptor(
            vector_bit_width=128,
            key="x86.avx512.vpmulld.xmm",
            mnemonic="vpmulld",
            semantic_tag="integer.mul.i32x4",
        ),
        _vector_i32_compare_descriptor(
            vector_bit_width=128,
            key="x86.avx512.vpcmpd.slt.xmm",
            mnemonic="vpcmpd.slt",
            semantic_tag="integer.cmp.slt.i32x4",
        ),
        _vector_mask_select_descriptor(
            vector_bit_width=128,
            key="x86.avx512.vpblendmd.xmm",
            mnemonic="vpblendmd",
            semantic_tag="integer.select.i32x4",
            schedule_class=_SCHEDULE_VECTOR_I32_XMM,
        ),
        _vector_f32_binary_descriptor(
            vector_bit_width=128,
            key="x86.avx512.vaddps.xmm",
            mnemonic="vaddps",
            semantic_tag="float.add.f32x4",
        ),
        _scalar_f32_binary_descriptor(
            key="x86.avx512.vaddss.xmm",
            mnemonic="vaddss",
            semantic_tag="float.add.f32",
        ),
        _vector_f32_binary_descriptor(
            vector_bit_width=128,
            key="x86.avx512.vsubps.xmm",
            mnemonic="vsubps",
            semantic_tag="float.sub.f32x4",
        ),
        _scalar_f32_binary_descriptor(
            key="x86.avx512.vsubss.xmm",
            mnemonic="vsubss",
            semantic_tag="float.sub.f32",
        ),
        _vector_f32_binary_descriptor(
            vector_bit_width=128,
            key="x86.avx512.vmulps.xmm",
            mnemonic="vmulps",
            semantic_tag="float.mul.f32x4",
        ),
        _scalar_f32_binary_descriptor(
            key="x86.avx512.vmulss.xmm",
            mnemonic="vmulss",
            semantic_tag="float.mul.f32",
        ),
        Descriptor(
            key="x86.avx512.vfmadd231ss.xmm",
            mnemonic="vfmadd231ss",
            semantic_tag="float.fma.f32",
            operands=(
                _xmm_result(),
                _xmm_operand("acc"),
                _xmm_operand("lhs"),
                _xmm_operand("rhs"),
            ),
            constraints=_DESTRUCTIVE_ACCUMULATOR_CONSTRAINTS,
            asm_forms=_asm(
                mnemonic="vfmadd231ss.xmm",
                results=("dst",),
                operands=("acc", "lhs", "rhs"),
            ),
            schedule_class=_SCHEDULE_VECTOR_FMA_F32_XMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        _vector_f32_compare_descriptor(
            vector_bit_width=128,
            key="x86.avx512.vcmpps.olt.xmm",
            mnemonic="vcmpps.olt",
            semantic_tag="float.cmp.olt.f32x4",
        ),
        _vector_mask_select_descriptor(
            vector_bit_width=128,
            key="x86.avx512.vblendmps.xmm",
            mnemonic="vblendmps",
            semantic_tag="float.select.f32x4",
            schedule_class=_SCHEDULE_VECTOR_F32_XMM,
        ),
        Descriptor(
            key="x86.avx512.vmovdqu32.load.xmm",
            mnemonic="vmovdqu32",
            semantic_tag="memory.load.i32x4",
            operands=(_vector_result(128), _gpr64_resource("base")),
            immediates=(_DISP32_IMMEDIATE,),
            asm_forms=_asm(
                mnemonic="vmovdqu32.load.xmm",
                results=("dst",),
                operands=("base",),
                immediates=("disp32",),
                named_immediates=True,
            ),
            effects=(_load_effect(128),),
            schedule_class=_SCHEDULE_MEMORY_LOAD_XMM,
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
        Descriptor(
            key="x86.avx512.vmovdqu32.load.indexed.xmm",
            mnemonic="vmovdqu32",
            semantic_tag="memory.load.indexed.i32x4",
            operands=(
                _vector_result(128),
                _gpr64_resource("base"),
                _gpr64_resource("index"),
            ),
            immediates=(_DISP32_IMMEDIATE, _ADDRESS_SCALE_IMMEDIATE),
            asm_forms=_asm(
                mnemonic="vmovdqu32.load.indexed.xmm",
                results=("dst",),
                operands=("base", "index"),
                immediates=("disp32", "scale"),
                named_immediates=True,
            ),
            effects=(_load_effect(128),),
            schedule_class=_SCHEDULE_MEMORY_LOAD_XMM,
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
        Descriptor(
            key="x86.avx512.vmovdqu32.store.xmm",
            mnemonic="vmovdqu32",
            semantic_tag="memory.store.i32x4",
            operands=(_vector_operand(128, "value"), _gpr64_resource("base")),
            immediates=(_DISP32_IMMEDIATE,),
            asm_forms=_asm(
                mnemonic="vmovdqu32.store.xmm",
                operands=("value", "base"),
                immediates=("disp32",),
                named_immediates=True,
            ),
            effects=(_store_effect(128),),
            schedule_class=_SCHEDULE_MEMORY_STORE_XMM,
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
        Descriptor(
            key="x86.avx512.vmovdqu32.store.indexed.xmm",
            mnemonic="vmovdqu32",
            semantic_tag="memory.store.indexed.i32x4",
            operands=(
                _vector_operand(128, "value"),
                _gpr64_resource("base"),
                _gpr64_resource("index"),
            ),
            immediates=(_DISP32_IMMEDIATE, _ADDRESS_SCALE_IMMEDIATE),
            asm_forms=_asm(
                mnemonic="vmovdqu32.store.indexed.xmm",
                operands=("value", "base", "index"),
                immediates=("disp32", "scale"),
                named_immediates=True,
            ),
            effects=(_store_effect(128),),
            schedule_class=_SCHEDULE_MEMORY_STORE_XMM,
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
        _zmm_splat_descriptor(
            key="x86.avx512.vpbroadcastd.zmm",
            mnemonic="vpbroadcastd",
            semantic_tag="integer.splat.i32x16",
            operand=_gpr32_operand("value"),
            schedule_class=_SCHEDULE_VECTOR_I32_ZMM,
        ),
        _zmm_splat_descriptor(
            key="x86.avx512.vbroadcastss.zmm",
            mnemonic="vbroadcastss",
            semantic_tag="float.splat.f32x16",
            operand=_xmm_operand("value"),
            schedule_class=_SCHEDULE_VECTOR_F32_ZMM,
        ),
        _zmm_i32_binary_descriptor(
            key="x86.avx512.vpaddd.zmm",
            mnemonic="vpaddd",
            semantic_tag="integer.add.i32x16",
        ),
        _zmm_i32_binary_descriptor(
            key="x86.avx512.vpsubd.zmm",
            mnemonic="vpsubd",
            semantic_tag="integer.sub.i32x16",
        ),
        _zmm_i32_binary_descriptor(
            key="x86.avx512.vpmulld.zmm",
            mnemonic="vpmulld",
            semantic_tag="integer.mul.i32x16",
        ),
        _zmm_i32_binary_descriptor(
            key="x86.avx512.vpminsd.zmm",
            mnemonic="vpminsd",
            semantic_tag="integer.mins.i32x16",
        ),
        _zmm_i32_binary_descriptor(
            key="x86.avx512.vpmaxsd.zmm",
            mnemonic="vpmaxsd",
            semantic_tag="integer.maxs.i32x16",
        ),
        _zmm_i32_binary_descriptor(
            key="x86.avx512.vpminud.zmm",
            mnemonic="vpminud",
            semantic_tag="integer.minu.i32x16",
        ),
        _zmm_i32_binary_descriptor(
            key="x86.avx512.vpmaxud.zmm",
            mnemonic="vpmaxud",
            semantic_tag="integer.maxu.i32x16",
        ),
        _zmm_i32_binary_descriptor(
            key="x86.avx512.vpandd.zmm",
            mnemonic="vpandd",
            semantic_tag="integer.and.i32x16",
        ),
        _zmm_i32_binary_descriptor(
            key="x86.avx512.vpord.zmm",
            mnemonic="vpord",
            semantic_tag="integer.or.i32x16",
        ),
        _zmm_i32_binary_descriptor(
            key="x86.avx512.vpxord.zmm",
            mnemonic="vpxord",
            semantic_tag="integer.xor.i32x16",
        ),
        _zmm_i32_binary_descriptor(
            key="x86.avx512.vpsllvd.zmm",
            mnemonic="vpsllvd",
            semantic_tag="integer.shl.i32x16",
        ),
        _zmm_i32_binary_descriptor(
            key="x86.avx512.vpsravd.zmm",
            mnemonic="vpsravd",
            semantic_tag="integer.shrs.i32x16",
        ),
        _zmm_i32_binary_descriptor(
            key="x86.avx512.vpsrlvd.zmm",
            mnemonic="vpsrlvd",
            semantic_tag="integer.shru.i32x16",
        ),
        _zmm_i32_compare_descriptor(
            key="x86.avx512.vpcmpd.eq.zmm",
            mnemonic="vpcmpd.eq",
            semantic_tag="integer.cmp.eq.i32x16",
        ),
        _zmm_i32_compare_descriptor(
            key="x86.avx512.vpcmpd.ne.zmm",
            mnemonic="vpcmpd.ne",
            semantic_tag="integer.cmp.ne.i32x16",
        ),
        _zmm_i32_compare_descriptor(
            key="x86.avx512.vpcmpd.slt.zmm",
            mnemonic="vpcmpd.slt",
            semantic_tag="integer.cmp.slt.i32x16",
        ),
        _zmm_i32_compare_descriptor(
            key="x86.avx512.vpcmpd.sle.zmm",
            mnemonic="vpcmpd.sle",
            semantic_tag="integer.cmp.sle.i32x16",
        ),
        _zmm_i32_compare_descriptor(
            key="x86.avx512.vpcmpd.sgt.zmm",
            mnemonic="vpcmpd.sgt",
            semantic_tag="integer.cmp.sgt.i32x16",
        ),
        _zmm_i32_compare_descriptor(
            key="x86.avx512.vpcmpd.sge.zmm",
            mnemonic="vpcmpd.sge",
            semantic_tag="integer.cmp.sge.i32x16",
        ),
        _zmm_i32_compare_descriptor(
            key="x86.avx512.vpcmpud.ult.zmm",
            mnemonic="vpcmpud.ult",
            semantic_tag="integer.cmp.ult.i32x16",
        ),
        _zmm_i32_compare_descriptor(
            key="x86.avx512.vpcmpud.ule.zmm",
            mnemonic="vpcmpud.ule",
            semantic_tag="integer.cmp.ule.i32x16",
        ),
        _zmm_i32_compare_descriptor(
            key="x86.avx512.vpcmpud.ugt.zmm",
            mnemonic="vpcmpud.ugt",
            semantic_tag="integer.cmp.ugt.i32x16",
        ),
        _zmm_i32_compare_descriptor(
            key="x86.avx512.vpcmpud.uge.zmm",
            mnemonic="vpcmpud.uge",
            semantic_tag="integer.cmp.uge.i32x16",
        ),
        _zmm_mask_select_descriptor(
            key="x86.avx512.vpblendmd.zmm",
            mnemonic="vpblendmd",
            semantic_tag="integer.select.i32x16",
            schedule_class=_SCHEDULE_VECTOR_I32_ZMM,
        ),
        _vector_f32_binary_descriptor(
            vector_bit_width=512,
            key="x86.avx512.vaddps.zmm",
            mnemonic="vaddps",
            semantic_tag="float.add.f32x16",
        ),
        _vector_f32_binary_descriptor(
            vector_bit_width=512,
            key="x86.avx512.vsubps.zmm",
            mnemonic="vsubps",
            semantic_tag="float.sub.f32x16",
        ),
        _vector_f32_binary_descriptor(
            vector_bit_width=512,
            key="x86.avx512.vmulps.zmm",
            mnemonic="vmulps",
            semantic_tag="float.mul.f32x16",
        ),
        Descriptor(
            key="x86.avx512.vfmadd231ps.zmm",
            mnemonic="vfmadd231ps",
            semantic_tag="float.fma.f32x16",
            operands=(
                _zmm_result(),
                _zmm_operand("acc"),
                _zmm_operand("lhs"),
                _zmm_operand("rhs"),
            ),
            constraints=_DESTRUCTIVE_ACCUMULATOR_CONSTRAINTS,
            asm_forms=_asm(results=("dst",), operands=("acc", "lhs", "rhs")),
            schedule_class=_SCHEDULE_VECTOR_FMA_F32_ZMM,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        _zmm_f32_compare_descriptor(
            key="x86.avx512.vcmpps.oeq.zmm",
            mnemonic="vcmpps.oeq",
            semantic_tag="float.cmp.oeq.f32x16",
        ),
        _zmm_f32_compare_descriptor(
            key="x86.avx512.vcmpps.ogt.zmm",
            mnemonic="vcmpps.ogt",
            semantic_tag="float.cmp.ogt.f32x16",
        ),
        _zmm_f32_compare_descriptor(
            key="x86.avx512.vcmpps.oge.zmm",
            mnemonic="vcmpps.oge",
            semantic_tag="float.cmp.oge.f32x16",
        ),
        _zmm_f32_compare_descriptor(
            key="x86.avx512.vcmpps.olt.zmm",
            mnemonic="vcmpps.olt",
            semantic_tag="float.cmp.olt.f32x16",
        ),
        _zmm_f32_compare_descriptor(
            key="x86.avx512.vcmpps.ole.zmm",
            mnemonic="vcmpps.ole",
            semantic_tag="float.cmp.ole.f32x16",
        ),
        _zmm_f32_compare_descriptor(
            key="x86.avx512.vcmpps.one.zmm",
            mnemonic="vcmpps.one",
            semantic_tag="float.cmp.one.f32x16",
        ),
        _zmm_f32_compare_descriptor(
            key="x86.avx512.vcmpps.ord.zmm",
            mnemonic="vcmpps.ord",
            semantic_tag="float.cmp.ord.f32x16",
        ),
        _zmm_f32_compare_descriptor(
            key="x86.avx512.vcmpps.ueq.zmm",
            mnemonic="vcmpps.ueq",
            semantic_tag="float.cmp.ueq.f32x16",
        ),
        _zmm_f32_compare_descriptor(
            key="x86.avx512.vcmpps.ugt.zmm",
            mnemonic="vcmpps.ugt",
            semantic_tag="float.cmp.ugt.f32x16",
        ),
        _zmm_f32_compare_descriptor(
            key="x86.avx512.vcmpps.uge.zmm",
            mnemonic="vcmpps.uge",
            semantic_tag="float.cmp.uge.f32x16",
        ),
        _zmm_f32_compare_descriptor(
            key="x86.avx512.vcmpps.ult.zmm",
            mnemonic="vcmpps.ult",
            semantic_tag="float.cmp.ult.f32x16",
        ),
        _zmm_f32_compare_descriptor(
            key="x86.avx512.vcmpps.ule.zmm",
            mnemonic="vcmpps.ule",
            semantic_tag="float.cmp.ule.f32x16",
        ),
        _zmm_f32_compare_descriptor(
            key="x86.avx512.vcmpps.une.zmm",
            mnemonic="vcmpps.une",
            semantic_tag="float.cmp.une.f32x16",
        ),
        _zmm_f32_compare_descriptor(
            key="x86.avx512.vcmpps.uno.zmm",
            mnemonic="vcmpps.uno",
            semantic_tag="float.cmp.uno.f32x16",
        ),
        _zmm_mask_select_descriptor(
            key="x86.avx512.vblendmps.zmm",
            mnemonic="vblendmps",
            semantic_tag="float.select.f32x16",
            schedule_class=_SCHEDULE_VECTOR_F32_ZMM,
        ),
        Descriptor(
            key="x86.avx512.vmovdqu32.load.zmm",
            mnemonic="vmovdqu32",
            semantic_tag="memory.load.i32x16",
            operands=(_zmm_result(), _gpr64_resource("base")),
            immediates=(_DISP32_IMMEDIATE,),
            asm_forms=_asm(
                mnemonic="vmovdqu32.load",
                results=("dst",),
                operands=("base",),
                immediates=("disp32",),
                named_immediates=True,
            ),
            effects=(_LOAD_EFFECT,),
            schedule_class=_SCHEDULE_MEMORY_LOAD_ZMM,
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
        Descriptor(
            key="x86.avx512.vmovdqu32.load.indexed.zmm",
            mnemonic="vmovdqu32",
            semantic_tag="memory.load.indexed.i32x16",
            operands=(
                _zmm_result(),
                _gpr64_resource("base"),
                _gpr64_resource("index"),
            ),
            immediates=(_DISP32_IMMEDIATE, _ADDRESS_SCALE_IMMEDIATE),
            asm_forms=_asm(
                mnemonic="vmovdqu32.load.indexed",
                results=("dst",),
                operands=("base", "index"),
                immediates=("disp32", "scale"),
                named_immediates=True,
            ),
            effects=(_LOAD_EFFECT,),
            schedule_class=_SCHEDULE_MEMORY_LOAD_ZMM,
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
        Descriptor(
            key="x86.avx512.vmovdqu32.store.zmm",
            mnemonic="vmovdqu32",
            semantic_tag="memory.store.i32x16",
            operands=(_zmm_operand("value"), _gpr64_resource("base")),
            immediates=(_DISP32_IMMEDIATE,),
            asm_forms=_asm(
                mnemonic="vmovdqu32.store",
                operands=("value", "base"),
                immediates=("disp32",),
                named_immediates=True,
            ),
            effects=(_STORE_EFFECT,),
            schedule_class=_SCHEDULE_MEMORY_STORE_ZMM,
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
        Descriptor(
            key="x86.avx512.vmovdqu32.store.indexed.zmm",
            mnemonic="vmovdqu32",
            semantic_tag="memory.store.indexed.i32x16",
            operands=(
                _zmm_operand("value"),
                _gpr64_resource("base"),
                _gpr64_resource("index"),
            ),
            immediates=(_DISP32_IMMEDIATE, _ADDRESS_SCALE_IMMEDIATE),
            asm_forms=_asm(
                mnemonic="vmovdqu32.store.indexed",
                operands=("value", "base", "index"),
                immediates=("disp32", "scale"),
                named_immediates=True,
            ),
            effects=(_STORE_EFFECT,),
            schedule_class=_SCHEDULE_MEMORY_STORE_ZMM,
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
            constraints=_DESTRUCTIVE_ACCUMULATOR_CONSTRAINTS,
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
            constraints=_DESTRUCTIVE_ACCUMULATOR_CONSTRAINTS,
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
            key="x86.avx512.korq",
            mnemonic="korq",
            semantic_tag="mask.or.i64",
            operands=(_k_result(), _k_operand("lhs"), _k_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_MASK,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="x86.avx512.kxorq",
            mnemonic="kxorq",
            semantic_tag="mask.xor.i64",
            operands=(_k_result(), _k_operand("lhs"), _k_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_MASK,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="x86.scalar.mov.gpr64",
            mnemonic="mov",
            semantic_tag="integer.move.i64",
            operands=(_gpr64_result(), _gpr64_operand("src")),
            asm_forms=_asm(results=("dst",), operands=("src",)),
            schedule_class=_SCHEDULE_SCALAR,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="x86.scalar.movimm.gpr64",
            mnemonic="mov",
            semantic_tag="integer.const.i64",
            operands=(_gpr64_result(),),
            immediates=(_IMM64_IMMEDIATE,),
            asm_forms=_asm(
                mnemonic="mov.imm64",
                results=("dst",),
                immediates=("imm64",),
            ),
            schedule_class=_SCHEDULE_SCALAR,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="x86.scalar.lea.add.gpr64",
            mnemonic="lea",
            semantic_tag="integer.add.i64",
            operands=(
                _gpr64_result(),
                _gpr64_operand("lhs"),
                _gpr64_operand("rhs"),
            ),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SCALAR,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="x86.scalar.imul.gpr64",
            mnemonic="imul",
            semantic_tag="integer.mul.i64",
            operands=(
                _gpr64_result(),
                _gpr64_operand("lhs"),
                _gpr64_operand("rhs"),
            ),
            constraints=_GPR_DESTRUCTIVE_LHS_CONSTRAINTS,
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SCALAR,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="x86.scalar.jmp",
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

_X86_SCALAR_DESCRIPTOR_KEYS = frozenset(
    (
        "x86.scalar.add.gpr32",
        "x86.scalar.mov.gpr64",
        "x86.scalar.movimm.gpr64",
        "x86.scalar.lea.add.gpr64",
        "x86.scalar.imul.gpr64",
        "x86.scalar.jmp",
    )
)

X86_SCALAR_DESCRIPTOR_SET = replace(
    X86_AVX512_CORE_DESCRIPTOR_SET,
    key="x86.scalar.core",
    feature_key="x86.scalar.v1",
    c_header_path=Path("loom/src/loom/target/arch/x86/scalar_descriptors.h"),
    c_source_path=Path("loom/src/loom/target/arch/x86/scalar_descriptors.c"),
    header_guard="LOOM_TARGET_ARCH_X86_SCALAR_DESCRIPTORS_H_",
    public_header="loom/target/arch/x86/scalar_descriptors.h",
    function_name="loom_x86_scalar_core_descriptor_set",
    c_table_prefix="X86ScalarCore",
    c_enum_prefix="X86_SCALAR_CORE",
    reg_classes=tuple(
        reg_class
        for reg_class in X86_AVX512_CORE_DESCRIPTOR_SET.reg_classes
        if reg_class.name in (_REG_GPR32, _REG_GPR64)
    ),
    resources=tuple(
        resource
        for resource in X86_AVX512_CORE_DESCRIPTOR_SET.resources
        if resource.name in (_RESOURCE_SCALAR, _RESOURCE_CONTROL)
    ),
    schedule_classes=tuple(
        schedule_class
        for schedule_class in X86_AVX512_CORE_DESCRIPTOR_SET.schedule_classes
        if schedule_class.name in (_SCHEDULE_SCALAR, _SCHEDULE_CONTROL)
    ),
    enum_domains=(),
    descriptors=tuple(
        descriptor
        for descriptor in X86_AVX512_CORE_DESCRIPTOR_SET.descriptors
        if descriptor.key in _X86_SCALAR_DESCRIPTOR_KEYS
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
            alias_set_id=2,
        ),
        RegClass(
            _REG_YMM,
            256,
            SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=32,
            alias_set_id=2,
        ),
        RegClass(
            _REG_ZMM,
            512,
            SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            physical_count=32,
            alias_set_id=2,
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

_X86_DESCRIPTOR_SET_COMPONENTS = tuple[tuple[DescriptorSet, frozenset[str]], ...]


_X86_AVX512_PACKED_DOT_COMPONENTS: _X86_DESCRIPTOR_SET_COMPONENTS = (
    (
        X86_AVX512_CORE_DESCRIPTOR_SET,
        frozenset(
            (
                "x86.avx512.vpdpbusd.zmm",
                "x86.avx512.vdpbf16ps.zmm",
            )
        ),
    ),
    (X86_PACKED_DOT_DESCRIPTOR_SET, frozenset()),
)


def _merge_named_items(item_groups: tuple[tuple[_T, ...], ...]) -> tuple[_T, ...]:
    merged_items: list[_T] = []
    seen_names: set[str] = set()
    for items in item_groups:
        for item in items:
            if item.name in seen_names:
                continue
            merged_items.append(item)
            seen_names.add(item.name)
    return tuple(merged_items)


def _merge_component_descriptors(
    components: tuple[tuple[DescriptorSet, frozenset[str]], ...],
) -> tuple[Descriptor, ...]:
    descriptors = []
    seen_keys: set[str] = set()
    for descriptor_set, excluded_keys in components:
        for descriptor in descriptor_set.descriptors:
            if descriptor.key in excluded_keys:
                continue
            if descriptor.key in seen_keys:
                raise ValueError(
                    "x86 descriptor set component repeats descriptor "
                    f"'{descriptor.key}'"
                )
            descriptors.append(descriptor)
            seen_keys.add(descriptor.key)
    return tuple(descriptors)


X86_AVX512_PACKED_DOT_DESCRIPTOR_SET = DescriptorSet(
    key="x86.avx512_packed_dot.core",
    target_key="x86",
    feature_key="x86.avx512_packed_dot.v1",
    c_header_path=Path("loom/src/loom/target/arch/x86/avx512_packed_dot_descriptors.h"),
    c_source_path=Path("loom/src/loom/target/arch/x86/avx512_packed_dot_descriptors.c"),
    header_guard="LOOM_TARGET_ARCH_X86_AVX512_PACKED_DOT_DESCRIPTORS_H_",
    public_header="loom/target/arch/x86/avx512_packed_dot_descriptors.h",
    function_name="loom_x86_avx512_packed_dot_core_descriptor_set",
    c_table_prefix="X86Avx512PackedDotCore",
    c_enum_prefix="X86_AVX512_PACKED_DOT_CORE",
    generator_version=1,
    reg_classes=_merge_named_items(
        tuple(
            descriptor_set.reg_classes
            for descriptor_set, _ in _X86_AVX512_PACKED_DOT_COMPONENTS
        )
    ),
    resources=_merge_named_items(
        tuple(
            descriptor_set.resources
            for descriptor_set, _ in _X86_AVX512_PACKED_DOT_COMPONENTS
        )
    ),
    schedule_classes=_merge_named_items(
        tuple(
            descriptor_set.schedule_classes
            for descriptor_set, _ in _X86_AVX512_PACKED_DOT_COMPONENTS
        )
    ),
    enum_domains=_merge_named_items(
        tuple(
            descriptor_set.enum_domains
            for descriptor_set, _ in _X86_AVX512_PACKED_DOT_COMPONENTS
        )
    ),
    descriptors=_merge_component_descriptors(_X86_AVX512_PACKED_DOT_COMPONENTS),
)
