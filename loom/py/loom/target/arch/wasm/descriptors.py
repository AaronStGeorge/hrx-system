# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Descriptor input data for the Wasm core+SIMD128 target."""

from __future__ import annotations

from pathlib import Path

from loom.target.low_descriptors import (
    AsmForm,
    AsmImmediate,
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

_REG_I32 = "wasm.i32"
_REG_I64 = "wasm.i64"
_REG_F32 = "wasm.f32"
_REG_F64 = "wasm.f64"
_REG_V128 = "wasm.v128"

_RESOURCE_SCALAR = "wasm.scalar"
_RESOURCE_SIMD = "wasm.simd"
_RESOURCE_LOAD = "wasm.load"
_RESOURCE_STORE = "wasm.store"
_RESOURCE_CONTROL = "wasm.control"

_SCHEDULE_CONST = "wasm.const"
_SCHEDULE_SCALAR_I32 = "wasm.scalar.i32"
_SCHEDULE_SIMD_I32X4 = "wasm.simd.i32x4"
_SCHEDULE_SIMD_F32X4 = "wasm.simd.f32x4"
_SCHEDULE_MEMORY_LOAD = "wasm.memory.load"
_SCHEDULE_MEMORY_STORE = "wasm.memory.store"
_SCHEDULE_CONTROL = "wasm.control"

_I32_ALT = (RegClassAlt(_REG_I32),)
_V128_ALT = (RegClassAlt(_REG_V128),)


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


def _i32_result(field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _I32_ALT)


def _i32_operand(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _I32_ALT)


def _i32_predicate(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.PREDICATE, _I32_ALT)


def _i32_resource(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.RESOURCE, _I32_ALT)


def _v128_result(field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _V128_ALT)


def _v128_operand(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _V128_ALT)


_I32_VALUE_IMMEDIATE = Immediate(
    "i32_value",
    ImmediateKind.SIGNED,
    bit_width=32,
    signed_min=-(2**31),
    unsigned_max=(2**31) - 1,
)

_V128_LO_IMMEDIATE = Immediate(
    "lo64",
    ImmediateKind.UNSIGNED,
    bit_width=64,
    unsigned_max=(2**64) - 1,
)

_V128_HI_IMMEDIATE = Immediate(
    "hi64",
    ImmediateKind.UNSIGNED,
    bit_width=64,
    unsigned_max=(2**64) - 1,
)

_OP_BR = 0x0C
_OP_BR_IF = 0x0D
_OP_RETURN = 0x0F
_OP_I32_CONST = 0x41
_OP_I32_LT_U = 0x49
_OP_I32_ADD = 0x6A
_OP_I32_SUB = 0x6B
_OP_SIMD_PREFIX = 0xFD


def _simd_encoding_id(subopcode: int) -> int:
    return (_OP_SIMD_PREFIX << 8) | subopcode


_OP_V128_LOAD = _simd_encoding_id(0x00)
_OP_V128_STORE = _simd_encoding_id(0x0B)
_OP_V128_CONST = _simd_encoding_id(0x0C)
_OP_I32X4_SPLAT = _simd_encoding_id(0x11)
_OP_I32X4_ADD = _simd_encoding_id(0xAE)
_OP_I32X4_SUB = _simd_encoding_id(0xB1)
_OP_I32X4_MUL = _simd_encoding_id(0xB5)
_OP_F32X4_ADD = _simd_encoding_id(0xE4)
_OP_F32X4_MUL = _simd_encoding_id(0xE6)

_TARGET_BLOCK_IMMEDIATE = Immediate(
    "target_block",
    ImmediateKind.ORDINAL,
    flags=(ImmediateFlag.SYMBOLIC,),
    bit_width=32,
    unsigned_max=(2**32) - 1,
)

_LOAD_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.WASM_MEMORY,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=128,
)

_STORE_EFFECT = Effect(
    EffectKind.WRITE,
    memory_space=MemorySpace.WASM_MEMORY,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=128,
)

_CONTROL_EFFECT = Effect(
    EffectKind.CONTROL,
    flags=(EffectFlag.ORDERED,),
)

WASM_CORE_SIMD128_DESCRIPTOR_SET = DescriptorSet(
    key="wasm.core.simd128",
    target_key="wasm",
    feature_key="wasm.simd128.v1",
    c_header_path=Path("loom/src/loom/target/arch/wasm/descriptors.h"),
    c_source_path=Path("loom/src/loom/target/arch/wasm/descriptors.c"),
    header_guard="LOOM_TARGET_ARCH_WASM_DESCRIPTORS_H_",
    public_header="loom/target/arch/wasm/descriptors.h",
    function_name="loom_wasm_core_simd128_descriptor_set",
    c_table_prefix="WasmCoreSimd128",
    c_enum_prefix="WASM_CORE_SIMD128",
    generator_version=1,
    reg_classes=(
        RegClass(
            _REG_I32, 32, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
        RegClass(
            _REG_I64, 64, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
        RegClass(
            _REG_F32, 32, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
        RegClass(
            _REG_F64, 64, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
        RegClass(
            _REG_V128, 128, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
    ),
    resources=(
        Resource(_RESOURCE_SCALAR, capacity_per_cycle=1, kind=ResourceKind.SCALAR_ALU),
        Resource(_RESOURCE_SIMD, capacity_per_cycle=1, kind=ResourceKind.VECTOR_ALU),
        Resource(_RESOURCE_LOAD, capacity_per_cycle=1, kind=ResourceKind.LOAD),
        Resource(_RESOURCE_STORE, capacity_per_cycle=1, kind=ResourceKind.STORE),
        Resource(_RESOURCE_CONTROL, capacity_per_cycle=1, kind=ResourceKind.CONTROL),
    ),
    schedule_classes=(
        ScheduleClass(
            _SCHEDULE_CONST,
            latency_kind=LatencyKind.EXACT,
            model_quality=ModelQuality.EXACT,
        ),
        ScheduleClass(
            _SCHEDULE_SCALAR_I32,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_SCALAR, cycles=1, units=1),),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_SIMD_I32X4,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_SIMD, cycles=1, units=1),),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_SIMD_F32X4,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_SIMD, cycles=1, units=1),),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_MEMORY_LOAD,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_LOAD, cycles=1, units=1),),
            flags=(ScheduleClassFlag.MAY_LOAD,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_MEMORY_STORE,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_STORE, cycles=1, units=1),),
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
            key="wasm.i32.const",
            mnemonic="i32.const",
            semantic_tag="integer.const.i32",
            encoding_id=_OP_I32_CONST,
            operands=(_i32_result(),),
            immediates=(_I32_VALUE_IMMEDIATE,),
            asm_forms=_asm(results=("dst",), immediates=("i32_value",)),
            schedule_class=_SCHEDULE_CONST,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.i32.add",
            mnemonic="i32.add",
            semantic_tag="integer.add.i32",
            encoding_id=_OP_I32_ADD,
            operands=(_i32_result(), _i32_operand("lhs"), _i32_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SCALAR_I32,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.i32.sub",
            mnemonic="i32.sub",
            semantic_tag="integer.sub.i32",
            encoding_id=_OP_I32_SUB,
            operands=(_i32_result(), _i32_operand("lhs"), _i32_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SCALAR_I32,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.i32.lt_u",
            mnemonic="i32.lt_u",
            semantic_tag="integer.cmp.lt.u32",
            encoding_id=_OP_I32_LT_U,
            operands=(_i32_result(), _i32_operand("lhs"), _i32_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SCALAR_I32,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.v128.const",
            mnemonic="v128.const",
            semantic_tag="vector.const.v128",
            encoding_id=_OP_V128_CONST,
            operands=(_v128_result(),),
            immediates=(_V128_LO_IMMEDIATE, _V128_HI_IMMEDIATE),
            asm_forms=_asm(results=("dst",), immediates=("lo64", "hi64")),
            schedule_class=_SCHEDULE_CONST,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.i32x4.splat",
            mnemonic="i32x4.splat",
            semantic_tag="vector.splat.i32x4",
            encoding_id=_OP_I32X4_SPLAT,
            operands=(_v128_result(), _i32_operand("value")),
            asm_forms=_asm(results=("dst",), operands=("value",)),
            schedule_class=_SCHEDULE_SIMD_I32X4,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.i32x4.add",
            mnemonic="i32x4.add",
            semantic_tag="vector.add.i32x4",
            encoding_id=_OP_I32X4_ADD,
            operands=(_v128_result(), _v128_operand("lhs"), _v128_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SIMD_I32X4,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.i32x4.sub",
            mnemonic="i32x4.sub",
            semantic_tag="vector.sub.i32x4",
            encoding_id=_OP_I32X4_SUB,
            operands=(_v128_result(), _v128_operand("lhs"), _v128_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SIMD_I32X4,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.i32x4.mul",
            mnemonic="i32x4.mul",
            semantic_tag="vector.mul.i32x4",
            encoding_id=_OP_I32X4_MUL,
            operands=(_v128_result(), _v128_operand("lhs"), _v128_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SIMD_I32X4,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.f32x4.add",
            mnemonic="f32x4.add",
            semantic_tag="vector.add.f32x4",
            encoding_id=_OP_F32X4_ADD,
            operands=(_v128_result(), _v128_operand("lhs"), _v128_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SIMD_F32X4,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.f32x4.mul",
            mnemonic="f32x4.mul",
            semantic_tag="vector.mul.f32x4",
            encoding_id=_OP_F32X4_MUL,
            operands=(_v128_result(), _v128_operand("lhs"), _v128_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_SIMD_F32X4,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="wasm.v128.load",
            mnemonic="v128.load",
            semantic_tag="memory.load.v128",
            encoding_id=_OP_V128_LOAD,
            operands=(_v128_result(), _i32_resource("address")),
            asm_forms=_asm(results=("dst",), operands=("address",)),
            effects=(_LOAD_EFFECT,),
            schedule_class=_SCHEDULE_MEMORY_LOAD,
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
        Descriptor(
            key="wasm.v128.store",
            mnemonic="v128.store",
            semantic_tag="memory.store.v128",
            encoding_id=_OP_V128_STORE,
            operands=(_i32_resource("address"), _v128_operand("value")),
            asm_forms=_asm(operands=("address", "value")),
            effects=(_STORE_EFFECT,),
            schedule_class=_SCHEDULE_MEMORY_STORE,
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
        Descriptor(
            key="wasm.br",
            mnemonic="br",
            semantic_tag="control.branch",
            encoding_id=_OP_BR,
            operands=(),
            immediates=(_TARGET_BLOCK_IMMEDIATE,),
            asm_forms=_asm(immediates=("target_block",)),
            effects=(_CONTROL_EFFECT,),
            schedule_class=_SCHEDULE_CONTROL,
            flags=(DescriptorFlag.SIDE_EFFECTING, DescriptorFlag.TERMINATOR),
        ),
        Descriptor(
            key="wasm.br_if.i32",
            mnemonic="br_if",
            semantic_tag="control.cond_branch.i32",
            encoding_id=_OP_BR_IF,
            operands=(_i32_predicate("cond"),),
            immediates=(_TARGET_BLOCK_IMMEDIATE,),
            asm_forms=_asm(operands=("cond",), immediates=("target_block",)),
            effects=(_CONTROL_EFFECT,),
            schedule_class=_SCHEDULE_CONTROL,
            flags=(DescriptorFlag.SIDE_EFFECTING, DescriptorFlag.TERMINATOR),
        ),
        Descriptor(
            key="wasm.return.v128",
            mnemonic="return",
            semantic_tag="control.return.v128",
            encoding_id=_OP_RETURN,
            operands=(_v128_operand("value"),),
            asm_forms=_asm(operands=("value",)),
            effects=(_CONTROL_EFFECT,),
            schedule_class=_SCHEDULE_CONTROL,
            flags=(DescriptorFlag.SIDE_EFFECTING, DescriptorFlag.TERMINATOR),
        ),
    ),
)
