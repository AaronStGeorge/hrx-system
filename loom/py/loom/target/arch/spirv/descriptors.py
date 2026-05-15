# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Descriptor input data for the SPIR-V logical target."""

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

_REG_ID = "spirv.id"
_REG_OFFSET64 = "spirv.offset64"
_REG_PTR_FUNCTION = "spirv.ptr.function"
_REG_PTR_WORKGROUP = "spirv.ptr.workgroup"
_REG_PTR_STORAGE_BUFFER = "spirv.ptr.storage_buffer"

_RESOURCE_ALU = "spirv.alu"
_RESOURCE_LOAD = "spirv.load"
_RESOURCE_STORE = "spirv.store"
_RESOURCE_VARIABLE = "spirv.variable"

_SCHEDULE_ALU = "spirv.alu"
_SCHEDULE_LOAD = "spirv.load"
_SCHEDULE_STORE = "spirv.store"
_SCHEDULE_VARIABLE = "spirv.variable"

_ID_ALT = (RegClassAlt(_REG_ID),)
_OFFSET64_ALT = (RegClassAlt(_REG_OFFSET64),)
_PTR_FUNCTION_ALT = (RegClassAlt(_REG_PTR_FUNCTION),)
_PTR_WORKGROUP_ALT = (RegClassAlt(_REG_PTR_WORKGROUP),)
_PTR_STORAGE_BUFFER_ALT = (RegClassAlt(_REG_PTR_STORAGE_BUFFER),)

_I32_VALUE_IMMEDIATE = Immediate(
    "i32_value",
    ImmediateKind.SIGNED,
    bit_width=32,
    signed_min=-(2**31),
    unsigned_max=(2**31) - 1,
)

_OFFSET64_VALUE_IMMEDIATE = Immediate(
    "offset64_value",
    ImmediateKind.SIGNED,
    bit_width=64,
    signed_min=-(2**63),
    unsigned_max=(2**63) - 1,
)


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


def _id_result(field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _ID_ALT)


def _id_operand(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _ID_ALT)


def _offset64_result(field_name: str = "dst") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _OFFSET64_ALT)


def _offset64_operand(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, _OFFSET64_ALT)


def _ptr_function_result(field_name: str = "ptr") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _PTR_FUNCTION_ALT)


def _ptr_workgroup_result(field_name: str = "ptr") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _PTR_WORKGROUP_ALT)


def _ptr_storage_buffer_result(field_name: str = "ptr") -> Operand:
    return Operand(field_name, OperandRole.RESULT, _PTR_STORAGE_BUFFER_ALT)


def _ptr_storage_buffer_operand(field_name: str) -> Operand:
    return Operand(field_name, OperandRole.RESOURCE, _PTR_STORAGE_BUFFER_ALT)


_LOAD_STORAGE_BUFFER_EFFECT = Effect(
    EffectKind.READ,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=32,
)

_STORE_STORAGE_BUFFER_EFFECT = Effect(
    EffectKind.WRITE,
    memory_space=MemorySpace.GLOBAL,
    flags=(EffectFlag.DEPENDENCY,),
    width_bits=32,
)

SPIRV_LOGICAL_CORE_DESCRIPTOR_SET = DescriptorSet(
    key="spirv.logical.core",
    target_key="spirv",
    feature_key="spirv.logical.v1",
    c_header_path=Path("loom/src/loom/target/arch/spirv/descriptors.h"),
    c_source_path=Path("loom/src/loom/target/arch/spirv/descriptors.c"),
    header_guard="LOOM_TARGET_ARCH_SPIRV_DESCRIPTORS_H_",
    public_header="loom/target/arch/spirv/descriptors.h",
    function_name="loom_spirv_logical_core_descriptor_set",
    c_table_prefix="SpirvLogicalCore",
    c_enum_prefix="SPIRV_LOGICAL_CORE",
    generator_version=1,
    reg_classes=(
        RegClass(
            _REG_ID,
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.VIRTUAL_ONLY,),
        ),
        RegClass(
            _REG_OFFSET64,
            64,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.VIRTUAL_ONLY,),
        ),
        RegClass(
            _REG_PTR_FUNCTION,
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.VIRTUAL_ONLY,),
        ),
        RegClass(
            _REG_PTR_WORKGROUP,
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.VIRTUAL_ONLY, RegClassFlag.UNSPILLABLE),
        ),
        RegClass(
            _REG_PTR_STORAGE_BUFFER,
            64,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.VIRTUAL_ONLY,),
        ),
    ),
    resources=(
        Resource(_RESOURCE_ALU, capacity_per_cycle=1, kind=ResourceKind.SCALAR_ALU),
        Resource(_RESOURCE_LOAD, capacity_per_cycle=1, kind=ResourceKind.LOAD),
        Resource(_RESOURCE_STORE, capacity_per_cycle=1, kind=ResourceKind.STORE),
        Resource(_RESOURCE_VARIABLE, capacity_per_cycle=1, kind=ResourceKind.ADDRESS),
    ),
    schedule_classes=(
        ScheduleClass(
            _SCHEDULE_ALU,
            latency_kind=LatencyKind.EXACT,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_ALU, cycles=1, units=1),),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_LOAD,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=4,
            issue_uses=(IssueUse(_RESOURCE_LOAD, cycles=1, units=1),),
            flags=(ScheduleClassFlag.MAY_LOAD,),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_STORE,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=4,
            issue_uses=(IssueUse(_RESOURCE_STORE, cycles=1, units=1),),
            flags=(ScheduleClassFlag.MAY_STORE,),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_VARIABLE,
            latency_kind=LatencyKind.EXACT,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_VARIABLE, cycles=1, units=1),),
            model_quality=ModelQuality.ESTIMATED,
        ),
    ),
    descriptors=(
        Descriptor(
            key="spirv.op_constant.i32",
            mnemonic="OpConstant.i32",
            semantic_tag="spirv.op_constant.i32",
            operands=(_id_result(),),
            immediates=(_I32_VALUE_IMMEDIATE,),
            asm_forms=_asm(results=("dst",), immediates=("i32_value",)),
            schedule_class=_SCHEDULE_ALU,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="spirv.op_constant.offset64",
            mnemonic="OpConstant.offset64",
            semantic_tag="spirv.op_constant.offset64",
            operands=(_offset64_result(),),
            immediates=(_OFFSET64_VALUE_IMMEDIATE,),
            asm_forms=_asm(results=("dst",), immediates=("offset64_value",)),
            schedule_class=_SCHEDULE_ALU,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="spirv.op_iadd.i32",
            mnemonic="OpIAdd",
            semantic_tag="spirv.op_iadd.i32",
            operands=(_id_result(), _id_operand("lhs"), _id_operand("rhs")),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_ALU,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="spirv.op_iadd.offset64",
            mnemonic="OpIAdd.offset64",
            semantic_tag="spirv.op_iadd.offset64",
            operands=(
                _offset64_result(),
                _offset64_operand("lhs"),
                _offset64_operand("rhs"),
            ),
            asm_forms=_asm(results=("dst",), operands=("lhs", "rhs")),
            schedule_class=_SCHEDULE_ALU,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="spirv.op_ptr_access_chain.storage_buffer.byte_offset",
            mnemonic="OpPtrAccessChain.storage_buffer.byte_offset",
            semantic_tag="spirv.op_ptr_access_chain.storage_buffer.byte_offset",
            operands=(
                _ptr_storage_buffer_result(),
                _ptr_storage_buffer_operand("base"),
                _offset64_operand("byte_offset"),
            ),
            asm_forms=_asm(results=("ptr",), operands=("base", "byte_offset")),
            schedule_class=_SCHEDULE_VARIABLE,
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
        Descriptor(
            key="spirv.op_load.storage_buffer.i32",
            mnemonic="OpLoad.storage_buffer.i32",
            semantic_tag="spirv.op_load.storage_buffer.i32",
            operands=(_id_result(), _ptr_storage_buffer_operand("ptr")),
            effects=(_LOAD_STORAGE_BUFFER_EFFECT,),
            asm_forms=_asm(results=("dst",), operands=("ptr",)),
            schedule_class=_SCHEDULE_LOAD,
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
        Descriptor(
            key="spirv.op_store.storage_buffer.i32",
            mnemonic="OpStore.storage_buffer.i32",
            semantic_tag="spirv.op_store.storage_buffer.i32",
            operands=(_ptr_storage_buffer_operand("ptr"), _id_operand("value")),
            effects=(_STORE_STORAGE_BUFFER_EFFECT,),
            asm_forms=_asm(operands=("ptr", "value")),
            schedule_class=_SCHEDULE_STORE,
            flags=(DescriptorFlag.SIDE_EFFECTING,),
        ),
        Descriptor(
            key="spirv.op_variable.function.ptr",
            mnemonic="OpVariable.function.ptr",
            semantic_tag="spirv.op_variable.function.ptr",
            operands=(_ptr_function_result(),),
            asm_forms=_asm(results=("ptr",)),
            schedule_class=_SCHEDULE_VARIABLE,
        ),
        Descriptor(
            key="spirv.op_variable.workgroup.ptr",
            mnemonic="OpVariable.workgroup.ptr",
            semantic_tag="spirv.op_variable.workgroup.ptr",
            operands=(_ptr_workgroup_result(),),
            asm_forms=_asm(results=("ptr",)),
            schedule_class=_SCHEDULE_VARIABLE,
        ),
    ),
)
