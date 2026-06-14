# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Descriptor input data for the generic LLVMIR oracle target."""

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

_REG_I1 = "llvmir.i1"
_REG_I8 = "llvmir.i8"
_REG_I16 = "llvmir.i16"
_REG_I32 = "llvmir.i32"
_REG_I64 = "llvmir.i64"
_REG_F16 = "llvmir.f16"
_REG_BF16 = "llvmir.bf16"
_REG_F32 = "llvmir.f32"
_REG_F64 = "llvmir.f64"
_REG_PTR = "llvmir.ptr"

_RESOURCE_ALU = "llvmir.alu"
_RESOURCE_LOAD = "llvmir.load"
_RESOURCE_STORE = "llvmir.store"

_SCHEDULE_CONST = "llvmir.const"
_SCHEDULE_ALU = "llvmir.alu"
_SCHEDULE_LOAD = "llvmir.load"
_SCHEDULE_STORE = "llvmir.store"

_VECTOR_LANE_COUNTS = (2, 4, 8, 16)

_INTEGER_PREDICATES = (
    "eq",
    "ne",
    "slt",
    "sle",
    "sgt",
    "sge",
    "ult",
    "ule",
    "ugt",
    "uge",
)

_FLOAT_PREDICATES = (
    "oeq",
    "ogt",
    "oge",
    "olt",
    "ole",
    "one",
    "ord",
    "ueq",
    "ugt",
    "uge",
    "ult",
    "ule",
    "une",
    "uno",
)

_ALT_BY_TYPE = {
    "i1": (RegClassAlt(_REG_I1),),
    "i8": (RegClassAlt(_REG_I8),),
    "i16": (RegClassAlt(_REG_I16),),
    "i32": (RegClassAlt(_REG_I32),),
    "i64": (RegClassAlt(_REG_I64),),
    "f16": (RegClassAlt(_REG_F16),),
    "bf16": (RegClassAlt(_REG_BF16),),
    "f32": (RegClassAlt(_REG_F32),),
    "f64": (RegClassAlt(_REG_F64),),
    "ptr": (RegClassAlt(_REG_PTR),),
}

_UNIT_BITS_BY_TYPE = {
    "i1": 1,
    "i8": 8,
    "i16": 16,
    "i32": 32,
    "i64": 64,
    "f16": 16,
    "bf16": 16,
    "f32": 32,
    "f64": 64,
    "ptr": 64,
}


def _asm(
    *,
    mnemonic: str,
    results: tuple[str, ...] = (),
    operands: tuple[str, ...] = (),
    immediates: tuple[str, ...] = (),
) -> tuple[AsmForm, ...]:
    return (
        AsmForm(
            mnemonic=mnemonic,
            results=results,
            operands=operands,
            immediates=tuple(AsmImmediate(field_name) for field_name in immediates),
        ),
    )


def _value(
    type_name: str,
    role: OperandRole,
    field_name: str,
    unit_count: int = 1,
) -> Operand:
    return Operand(
        field_name,
        role,
        _ALT_BY_TYPE[type_name],
        unit_count=unit_count,
    )


def _result(type_name: str, suffix: str = "dst", unit_count: int = 1) -> Operand:
    return _value(type_name, OperandRole.RESULT, suffix, unit_count)


def _operand(type_name: str, field_name: str, unit_count: int = 1) -> Operand:
    return _value(type_name, OperandRole.OPERAND, field_name, unit_count)


def _predicate(
    type_name: str,
    field_name: str = "condition",
    unit_count: int = 1,
) -> Operand:
    return _value(type_name, OperandRole.PREDICATE, field_name, unit_count)


def _ptr_operand(field_name: str) -> Operand:
    return _value("ptr", OperandRole.RESOURCE, field_name)


def _descriptor_suffix(type_name: str, unit_count: int, *, vector: bool = False) -> str:
    if vector:
        return f"v{unit_count}{type_name}"
    return type_name if unit_count == 1 else f"v{unit_count}{type_name}"


_I64_VALUE_IMMEDIATE = Immediate(
    "value",
    ImmediateKind.SIGNED,
    bit_width=64,
    signed_min=-(2**63),
    unsigned_max=(2**63) - 1,
)

_F32_BITS_IMMEDIATE = Immediate(
    "bits",
    ImmediateKind.UNSIGNED,
    bit_width=32,
    unsigned_max=(2**32) - 1,
)

_F64_BITS_IMMEDIATE = Immediate(
    "bits",
    ImmediateKind.UNSIGNED,
    bit_width=64,
    unsigned_max=(2**64) - 1,
)

_BYTE_OFFSET_IMMEDIATE = Immediate(
    "byte_offset",
    ImmediateKind.SIGNED,
    bit_width=64,
    signed_min=-(2**63),
    unsigned_max=(2**63) - 1,
)

_BYTE_STRIDE_IMMEDIATE = Immediate(
    "byte_stride",
    ImmediateKind.SIGNED,
    bit_width=64,
    signed_min=-(2**63),
    unsigned_max=(2**63) - 1,
)


def _read_effect(width_bits: int) -> Effect:
    return Effect(
        EffectKind.READ,
        memory_space=MemorySpace.GENERIC,
        flags=(EffectFlag.DEPENDENCY,),
        width_bits=width_bits,
    )


def _write_effect(width_bits: int) -> Effect:
    return Effect(
        EffectKind.WRITE,
        memory_space=MemorySpace.GENERIC,
        flags=(EffectFlag.DEPENDENCY,),
        width_bits=width_bits,
    )


def _const_i_descriptor(type_name: str) -> Descriptor:
    return Descriptor(
        key=f"llvmir.const.{type_name}",
        mnemonic="const",
        semantic_tag=f"llvmir.const.{type_name}",
        operands=(_result(type_name),),
        immediates=(_I64_VALUE_IMMEDIATE,),
        asm_forms=_asm(
            mnemonic=f"const.{type_name}",
            results=("dst",),
            immediates=("value",),
        ),
        schedule_class=_SCHEDULE_CONST,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _const_f_descriptor(type_name: str, immediate: Immediate) -> Descriptor:
    return Descriptor(
        key=f"llvmir.const.{type_name}",
        mnemonic="const",
        semantic_tag=f"llvmir.const.{type_name}",
        operands=(_result(type_name),),
        immediates=(immediate,),
        asm_forms=_asm(
            mnemonic=f"const.{type_name}",
            results=("dst",),
            immediates=("bits",),
        ),
        schedule_class=_SCHEDULE_CONST,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _binary_descriptor(
    *,
    stem: str,
    type_name: str,
    semantic_stem: str,
    unit_count: int = 1,
    vector: bool = False,
) -> Descriptor:
    suffix = _descriptor_suffix(type_name, unit_count, vector=vector)
    return Descriptor(
        key=f"llvmir.{stem}.{suffix}",
        mnemonic=stem,
        semantic_tag=f"llvmir.{semantic_stem}.{suffix}",
        operands=(
            _result(type_name, unit_count=unit_count),
            _operand(type_name, "lhs", unit_count=unit_count),
            _operand(type_name, "rhs", unit_count=unit_count),
        ),
        asm_forms=_asm(
            mnemonic=f"{stem}.{suffix}",
            results=("dst",),
            operands=("lhs", "rhs"),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _compare_descriptor(
    *,
    predicate: str,
    type_name: str,
    unit_count: int = 1,
    vector: bool = False,
) -> Descriptor:
    suffix = _descriptor_suffix(type_name, unit_count, vector=vector)
    result_unit_count = unit_count
    return Descriptor(
        key=f"llvmir.cmp.{predicate}.{suffix}",
        mnemonic="cmp",
        semantic_tag=f"llvmir.cmp.{predicate}.{suffix}",
        operands=(
            _result("i1", unit_count=result_unit_count),
            _operand(type_name, "lhs", unit_count=unit_count),
            _operand(type_name, "rhs", unit_count=unit_count),
        ),
        asm_forms=_asm(
            mnemonic=f"cmp.{predicate}.{suffix}",
            results=("dst",),
            operands=("lhs", "rhs"),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _select_descriptor(
    type_name: str,
    unit_count: int = 1,
    *,
    vector: bool = False,
) -> Descriptor:
    suffix = _descriptor_suffix(type_name, unit_count, vector=vector)
    return Descriptor(
        key=f"llvmir.select.{suffix}",
        mnemonic="select",
        semantic_tag=f"llvmir.select.{suffix}",
        operands=(
            _result(type_name, unit_count=unit_count),
            _predicate("i1", unit_count=unit_count),
            _operand(type_name, "true_value", unit_count=unit_count),
            _operand(type_name, "false_value", unit_count=unit_count),
        ),
        asm_forms=_asm(
            mnemonic=f"select.{suffix}",
            results=("dst",),
            operands=("condition", "true_value", "false_value"),
        ),
        schedule_class=_SCHEDULE_ALU,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _load_descriptor(
    type_name: str,
    *,
    unit_count: int = 1,
    indexed: bool = False,
) -> Descriptor:
    suffix = _descriptor_suffix(type_name, unit_count)
    operands = [_result(type_name, unit_count=unit_count), _ptr_operand("ptr")]
    immediates = [_BYTE_OFFSET_IMMEDIATE]
    asm_operands = ["ptr"]
    asm_immediates = ["byte_offset"]
    key_suffix = f"indexed.{suffix}" if indexed else suffix
    mnemonic_suffix = f"indexed.{suffix}" if indexed else suffix
    if indexed:
        operands.append(_operand("i64", "index"))
        immediates.append(_BYTE_STRIDE_IMMEDIATE)
        asm_operands.append("index")
        asm_immediates.append("byte_stride")
    return Descriptor(
        key=f"llvmir.load.{key_suffix}",
        mnemonic="load",
        semantic_tag=f"llvmir.load.{key_suffix}",
        operands=tuple(operands),
        immediates=tuple(immediates),
        asm_forms=_asm(
            mnemonic=f"load.{mnemonic_suffix}",
            results=("dst",),
            operands=tuple(asm_operands),
            immediates=tuple(asm_immediates),
        ),
        effects=(_read_effect(_UNIT_BITS_BY_TYPE[type_name] * unit_count),),
        schedule_class=_SCHEDULE_LOAD,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _store_descriptor(
    type_name: str,
    *,
    unit_count: int = 1,
    indexed: bool = False,
) -> Descriptor:
    suffix = _descriptor_suffix(type_name, unit_count)
    operands = [
        _operand(type_name, "value", unit_count=unit_count),
        _ptr_operand("ptr"),
    ]
    immediates = [_BYTE_OFFSET_IMMEDIATE]
    asm_operands = ["value", "ptr"]
    asm_immediates = ["byte_offset"]
    key_suffix = f"indexed.{suffix}" if indexed else suffix
    mnemonic_suffix = f"indexed.{suffix}" if indexed else suffix
    if indexed:
        operands.append(_operand("i64", "index"))
        immediates.append(_BYTE_STRIDE_IMMEDIATE)
        asm_operands.append("index")
        asm_immediates.append("byte_stride")
    return Descriptor(
        key=f"llvmir.store.{key_suffix}",
        mnemonic="store",
        semantic_tag=f"llvmir.store.{key_suffix}",
        operands=tuple(operands),
        immediates=tuple(immediates),
        asm_forms=_asm(
            mnemonic=f"store.{mnemonic_suffix}",
            operands=tuple(asm_operands),
            immediates=tuple(asm_immediates),
        ),
        effects=(_write_effect(_UNIT_BITS_BY_TYPE[type_name] * unit_count),),
        schedule_class=_SCHEDULE_STORE,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    )


def _arithmetic_descriptors() -> tuple[Descriptor, ...]:
    descriptors: list[Descriptor] = []
    for type_name in ("i32", "i64"):
        descriptors.extend(
            (_binary_descriptor(stem=stem, type_name=type_name, semantic_stem=stem))
            for stem in ("add", "sub", "mul")
        )
    for type_name in ("f32", "f64"):
        descriptors.extend(
            (_binary_descriptor(stem=stem, type_name=type_name, semantic_stem=stem))
            for stem in ("add", "sub", "mul", "div")
        )
    for type_name in ("i32", "f32"):
        descriptors.extend(
            (
                _binary_descriptor(
                    stem=stem,
                    type_name=type_name,
                    semantic_stem=stem,
                    unit_count=lane_count,
                    vector=True,
                )
            )
            for lane_count in _VECTOR_LANE_COUNTS
            for stem in ("add", "sub", "mul")
        )
    return tuple(descriptors)


def _bitwise_descriptors() -> tuple[Descriptor, ...]:
    descriptors: list[Descriptor] = []
    for type_name in ("i1", "i32", "i64"):
        descriptors.extend(
            (_binary_descriptor(stem=stem, type_name=type_name, semantic_stem=stem))
            for stem in ("and", "or", "xor")
        )
    for type_name in ("i32", "i64"):
        descriptors.extend(
            (_binary_descriptor(stem=stem, type_name=type_name, semantic_stem=stem))
            for stem in ("shl", "lshr", "ashr")
        )
    for type_name, stems in (
        ("i1", ("and", "or", "xor")),
        ("i32", ("and", "or", "xor", "shl", "lshr", "ashr")),
    ):
        descriptors.extend(
            (
                _binary_descriptor(
                    stem=stem,
                    type_name=type_name,
                    semantic_stem=stem,
                    unit_count=lane_count,
                    vector=True,
                )
            )
            for lane_count in _VECTOR_LANE_COUNTS
            for stem in stems
        )
    return tuple(descriptors)


def _compare_descriptors() -> tuple[Descriptor, ...]:
    descriptors: list[Descriptor] = []
    for type_name in ("i32", "i64"):
        descriptors.extend(
            _compare_descriptor(predicate=predicate, type_name=type_name)
            for predicate in _INTEGER_PREDICATES
        )
    for type_name in ("f32", "f64"):
        descriptors.extend(
            _compare_descriptor(predicate=predicate, type_name=type_name)
            for predicate in _FLOAT_PREDICATES
        )
    for type_name in ("i32", "f32"):
        predicates = _INTEGER_PREDICATES if type_name == "i32" else _FLOAT_PREDICATES
        descriptors.extend(
            _compare_descriptor(
                predicate=predicate,
                type_name=type_name,
                unit_count=lane_count,
                vector=True,
            )
            for lane_count in _VECTOR_LANE_COUNTS
            for predicate in predicates
        )
    return tuple(descriptors)


def _select_descriptors() -> tuple[Descriptor, ...]:
    return tuple(
        _select_descriptor(type_name, unit_count=unit_count)
        for type_name, unit_count in (
            ("i32", 1),
            ("i64", 1),
            ("f32", 1),
            ("f64", 1),
        )
    ) + tuple(
        _select_descriptor(type_name, unit_count=lane_count, vector=True)
        for type_name in ("i8", "i16", "i32", "i64", "f16", "bf16", "f32", "f64")
        for lane_count in _VECTOR_LANE_COUNTS
    )


def _memory_descriptors() -> tuple[Descriptor, ...]:
    descriptors: list[Descriptor] = []
    for type_name, unit_count in (
        ("i32", 1),
        ("i64", 1),
        ("f32", 1),
        ("f64", 1),
        ("i32", 4),
        ("f32", 4),
    ):
        for indexed in (False, True):
            descriptors.append(
                _load_descriptor(
                    type_name,
                    unit_count=unit_count,
                    indexed=indexed,
                )
            )
            descriptors.append(
                _store_descriptor(
                    type_name,
                    unit_count=unit_count,
                    indexed=indexed,
                )
            )
    return tuple(descriptors)


LLVMIR_GENERIC_CORE_DESCRIPTOR_SET = DescriptorSet(
    key="llvmir.generic.core",
    target_key="llvmir",
    feature_key="llvmir.generic.v1",
    c_header_path=Path("loom/src/loom/target/arch/llvmir/descriptors.h"),
    c_source_path=Path("loom/src/loom/target/arch/llvmir/descriptors.c"),
    header_guard="LOOM_TARGET_ARCH_LLVMIR_DESCRIPTORS_H_",
    public_header="loom/target/arch/llvmir/descriptors/descriptors.h",
    function_name="loom_llvmir_generic_core_descriptor_set",
    c_table_prefix="LlvmirGenericCore",
    c_enum_prefix="LLVMIR_GENERIC_CORE",
    generator_version=1,
    reg_classes=(
        RegClass(
            _REG_I1, 1, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
        RegClass(
            _REG_I8, 8, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
        RegClass(
            _REG_I16, 16, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
        RegClass(
            _REG_I32, 32, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
        RegClass(
            _REG_I64, 64, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
        RegClass(
            _REG_F16, 16, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
        RegClass(
            _REG_BF16, 16, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
        RegClass(
            _REG_F32, 32, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
        RegClass(
            _REG_F64, 64, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
        RegClass(
            _REG_PTR, 64, SpillSlotSpace.PRIVATE, flags=(RegClassFlag.VIRTUAL_ONLY,)
        ),
    ),
    resources=(
        Resource(_RESOURCE_ALU, capacity_per_cycle=1, kind=ResourceKind.VECTOR_ALU),
        Resource(_RESOURCE_LOAD, capacity_per_cycle=1, kind=ResourceKind.LOAD),
        Resource(_RESOURCE_STORE, capacity_per_cycle=1, kind=ResourceKind.STORE),
    ),
    schedule_classes=(
        ScheduleClass(
            _SCHEDULE_CONST,
            latency_kind=LatencyKind.EXACT,
            model_quality=ModelQuality.EXACT,
        ),
        ScheduleClass(
            _SCHEDULE_ALU,
            latency_kind=LatencyKind.EXACT,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_ALU, cycles=1, units=1),),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_LOAD,
            latency_kind=LatencyKind.VARIABLE,
            issue_uses=(IssueUse(_RESOURCE_LOAD, cycles=1, units=1),),
            model_quality=ModelQuality.FALLBACK,
            flags=(ScheduleClassFlag.MAY_LOAD,),
        ),
        ScheduleClass(
            _SCHEDULE_STORE,
            latency_kind=LatencyKind.VARIABLE,
            issue_uses=(IssueUse(_RESOURCE_STORE, cycles=1, units=1),),
            model_quality=ModelQuality.FALLBACK,
            flags=(ScheduleClassFlag.MAY_STORE,),
        ),
    ),
    descriptors=(
        _const_i_descriptor("i32"),
        _const_i_descriptor("i64"),
        _const_f_descriptor("f32", _F32_BITS_IMMEDIATE),
        _const_f_descriptor("f64", _F64_BITS_IMMEDIATE),
        *_arithmetic_descriptors(),
        *_bitwise_descriptors(),
        *_compare_descriptors(),
        *_select_descriptors(),
        *_memory_descriptors(),
    ),
)
