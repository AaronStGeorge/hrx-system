# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Descriptor input data for the IREE VM low target."""

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

_REG_I32 = "vm.i32"
_REG_I64 = "vm.i64"
_REG_F32 = "vm.f32"
_REG_F64 = "vm.f64"
_REG_REF = "vm.ref"
_REG_LIST = "vm.list"

_SCHEDULE_CONST = "vm.const"
_SCHEDULE_ALU_I32 = "vm.alu.i32"
_SCHEDULE_CONTROL = "vm.control"
_SCHEDULE_CALL = "vm.call"

_RESOURCE_ALU = "vm.alu"
_RESOURCE_CONTROL = "vm.control"
_RESOURCE_CALL = "vm.call"

_I32_ALT = (RegClassAlt(_REG_I32),)


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


_I32_VALUE_IMMEDIATE = Immediate(
    "i32_value",
    ImmediateKind.SIGNED,
    bit_width=32,
    signed_min=-(2**31),
    unsigned_max=(2**31) - 1,
)

_TARGET_BLOCK_IMMEDIATE = Immediate(
    "target_block",
    ImmediateKind.ORDINAL,
    flags=(ImmediateFlag.SYMBOLIC,),
    bit_width=32,
    unsigned_max=(2**32) - 1,
)

_TRUE_BLOCK_IMMEDIATE = Immediate(
    "true_block",
    ImmediateKind.ORDINAL,
    flags=(ImmediateFlag.SYMBOLIC,),
    bit_width=32,
    unsigned_max=(2**32) - 1,
)

_FALSE_BLOCK_IMMEDIATE = Immediate(
    "false_block",
    ImmediateKind.ORDINAL,
    flags=(ImmediateFlag.SYMBOLIC,),
    bit_width=32,
    unsigned_max=(2**32) - 1,
)

_CALLEE_IMMEDIATE = Immediate(
    "callee_ordinal",
    ImmediateKind.ORDINAL,
    flags=(ImmediateFlag.SYMBOLIC,),
    bit_width=32,
    unsigned_max=(2**32) - 1,
)

_CONTROL_EFFECT = Effect(
    EffectKind.CONTROL,
    flags=(EffectFlag.ORDERED,),
)

_CALL_EFFECT = Effect(
    EffectKind.CALL,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
)

IREEVM_CORE_DESCRIPTOR_SET = DescriptorSet(
    key="iree.vm.core",
    target_key="iree.vm",
    feature_key="iree.vm.v1",
    c_header_path=Path("loom/src/loom/target/emit/ireevm/descriptors.h"),
    c_source_path=Path("loom/src/loom/target/emit/ireevm/descriptors.c"),
    header_guard="LOOM_TARGET_EMIT_IREEVM_DESCRIPTORS_H_",
    public_header="loom/target/emit/ireevm/descriptors.h",
    function_name="loom_ireevm_core_descriptor_set",
    c_table_prefix="IreeVmCore",
    c_enum_prefix="IREE_VM_CORE",
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
            _REG_REF,
            64,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.VIRTUAL_ONLY, RegClassFlag.REFERENCE),
        ),
        RegClass(
            _REG_LIST,
            64,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.VIRTUAL_ONLY, RegClassFlag.REFERENCE),
        ),
    ),
    resources=(
        Resource(_RESOURCE_ALU, capacity_per_cycle=1, kind=ResourceKind.SCALAR_ALU),
        Resource(_RESOURCE_CONTROL, capacity_per_cycle=1, kind=ResourceKind.CONTROL),
        Resource(_RESOURCE_CALL, capacity_per_cycle=1, kind=ResourceKind.CONTROL),
    ),
    schedule_classes=(
        ScheduleClass(
            _SCHEDULE_CONST,
            latency_kind=LatencyKind.EXACT,
            model_quality=ModelQuality.EXACT,
        ),
        ScheduleClass(
            _SCHEDULE_ALU_I32,
            latency_kind=LatencyKind.EXACT,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_ALU, cycles=1, units=1),),
            model_quality=ModelQuality.EXACT,
        ),
        ScheduleClass(
            _SCHEDULE_CONTROL,
            latency_kind=LatencyKind.EXACT,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.EXACT,
        ),
        ScheduleClass(
            _SCHEDULE_CALL,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CALL, cycles=1, units=1),),
            flags=(ScheduleClassFlag.MAY_CALL,),
            model_quality=ModelQuality.FALLBACK,
        ),
    ),
    descriptors=(
        Descriptor(
            key="iree.vm.const.i32",
            mnemonic="vm.const.i32",
            semantic_tag="integer.const.i32",
            operands=(_i32_result(),),
            immediates=(_I32_VALUE_IMMEDIATE,),
            asm_forms=_asm(results=("dst",), immediates=("i32_value",)),
            schedule_class=_SCHEDULE_CONST,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="iree.vm.add.i32",
            mnemonic="vm.add.i32",
            semantic_tag="integer.add.i32",
            operands=(_i32_result(), _i32_operand("lhs"), _i32_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_ALU_I32,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="iree.vm.sub.i32",
            mnemonic="vm.sub.i32",
            semantic_tag="integer.sub.i32",
            operands=(_i32_result(), _i32_operand("lhs"), _i32_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_ALU_I32,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="iree.vm.cmp.eq.i32",
            mnemonic="vm.cmp.eq.i32",
            semantic_tag="integer.cmp.eq.i32",
            operands=(_i32_result(), _i32_operand("lhs"), _i32_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_ALU_I32,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="iree.vm.br",
            mnemonic="vm.br",
            semantic_tag="control.branch",
            operands=(),
            immediates=(_TARGET_BLOCK_IMMEDIATE,),
            asm_forms=_asm(immediates=("target_block",)),
            effects=(_CONTROL_EFFECT,),
            schedule_class=_SCHEDULE_CONTROL,
            flags=(DescriptorFlag.SIDE_EFFECTING, DescriptorFlag.TERMINATOR),
        ),
        Descriptor(
            key="iree.vm.cond_br.i32",
            mnemonic="vm.cond_br.i32",
            semantic_tag="control.cond_branch.i32",
            operands=(_i32_predicate("cond"),),
            immediates=(_TRUE_BLOCK_IMMEDIATE, _FALSE_BLOCK_IMMEDIATE),
            asm_forms=_asm(
                operands=("cond",), immediates=("true_block", "false_block")
            ),
            effects=(_CONTROL_EFFECT,),
            schedule_class=_SCHEDULE_CONTROL,
            flags=(DescriptorFlag.SIDE_EFFECTING, DescriptorFlag.TERMINATOR),
        ),
        Descriptor(
            key="iree.vm.call.import.i32",
            mnemonic="vm.call.import.i32",
            semantic_tag="call.import.i32",
            operands=(_i32_result(), _i32_operand("arg0")),
            immediates=(_CALLEE_IMMEDIATE,),
            asm_forms=_asm(
                results=("dst",), operands=("arg0",), immediates=("callee_ordinal",)
            ),
            effects=(_CALL_EFFECT,),
            schedule_class=_SCHEDULE_CALL,
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
        Descriptor(
            key="iree.vm.return.i32",
            mnemonic="vm.return.i32",
            semantic_tag="control.return.i32",
            operands=(_i32_operand("value"),),
            asm_forms=_asm(operands=("value",)),
            effects=(_CONTROL_EFFECT,),
            schedule_class=_SCHEDULE_CONTROL,
            flags=(DescriptorFlag.SIDE_EFFECTING, DescriptorFlag.TERMINATOR),
        ),
        Descriptor(
            key="iree.vm.return.void",
            mnemonic="vm.return.void",
            semantic_tag="control.return.void",
            operands=(),
            asm_forms=_asm(),
            effects=(_CONTROL_EFFECT,),
            schedule_class=_SCHEDULE_CONTROL,
            flags=(DescriptorFlag.SIDE_EFFECTING, DescriptorFlag.TERMINATOR),
        ),
    ),
)
