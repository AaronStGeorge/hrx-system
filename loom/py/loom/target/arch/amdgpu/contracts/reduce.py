# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU vector reduction source-to-low contracts."""

from __future__ import annotations

from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.target.arch.amdgpu.contracts.materializers import I32_VGPR_MATERIALIZER
from loom.target.arch.amdgpu.descriptors import build_amdgpu_contract_descriptor_set
from loom.target.contracts import (
    ContractFragment,
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    TypePattern,
    ValueRef,
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_DESCRIPTOR_KEYS = (
    "amdgpu.v_add_u32",
    "amdgpu.v_mul_lo_u32",
    "amdgpu.v_and_b32",
    "amdgpu.v_or_b32",
    "amdgpu.v_xor_b32",
    "amdgpu.v_add_f32",
    "amdgpu.v_mul_f32",
    "amdgpu.v_min_f32",
    "amdgpu.v_max_f32",
)

_DESCRIPTOR_SET = build_amdgpu_contract_descriptor_set(
    key="amdgpu.reduce",
    descriptor_keys=_DESCRIPTOR_KEYS,
)

_VEC_I32 = Vector(
    "i32",
    minimum_lanes=1,
    maximum_lanes="LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES",
)
_VEC_F32 = Vector(
    "f32",
    minimum_lanes=1,
    maximum_lanes="LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES",
)
_I32 = Scalar("i32")
_F32 = Scalar("f32")


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(_DESCRIPTOR_SET, key)


def _vector_reduce_rule(
    *,
    kind: str,
    input_type: TypePattern,
    accumulator_type: TypePattern,
    descriptor: Descriptor,
    materialized_init: ValueRef | None = None,
) -> DescriptorRule:
    init = materialized_init or ValueRef.operand("init")
    guards = [
        Guard.enum_attr_equals("kind", kind),
        Guard.value_type("input", input_type),
        Guard.value_type("init", accumulator_type),
        Guard.value_type("result", accumulator_type),
        Guard.low_value_register_class("result", "amdgpu.vgpr"),
    ]
    if materialized_init is not None:
        materializer = materialized_init.materializer
        if materializer is None:
            raise ValueError("materialized init requires a materializer name")
        guards.append(Guard.value_materializable("init", materializer))
    guards.append(Guard.descriptor_available(descriptor))
    return DescriptorRule(
        source_op=vector.vector_reduce,
        descriptor=descriptor,
        guards=guards,
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "lhs": init,
                    "rhs": ValueRef.operand("input"),
                },
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.ACCUMULATE_LANES,
                accumulator="lhs",
            ),
        ),
    )


def _i32_rule(kind: str, descriptor_key: str) -> DescriptorRule:
    return _vector_reduce_rule(
        kind=kind,
        input_type=_VEC_I32,
        accumulator_type=_I32,
        descriptor=_descriptor(descriptor_key),
        materialized_init=ValueRef.operand(
            "init",
            materializer=I32_VGPR_MATERIALIZER.name,
        ),
    )


def _f32_rule(kind: str, descriptor_key: str) -> DescriptorRule:
    return _vector_reduce_rule(
        kind=kind,
        input_type=_VEC_F32,
        accumulator_type=_F32,
        descriptor=_descriptor(descriptor_key),
    )


AMDGPU_REDUCE_CONTRACT_DIALECT_OPS = {
    "vector": ALL_VECTOR_OPS,
}

AMDGPU_REDUCE_CONTRACT_FRAGMENT = ContractFragment(
    name="amdgpu.reduce",
    descriptor_set=_DESCRIPTOR_SET,
    c_source_includes=("loom/target/arch/amdgpu/lower/kinds.h",),
    materializers=(I32_VGPR_MATERIALIZER,),
    cases=[
        _i32_rule("addi", "amdgpu.v_add_u32"),
        _i32_rule("muli", "amdgpu.v_mul_lo_u32"),
        _i32_rule("andi", "amdgpu.v_and_b32"),
        _i32_rule("ori", "amdgpu.v_or_b32"),
        _i32_rule("xori", "amdgpu.v_xor_b32"),
        _f32_rule("addf", "amdgpu.v_add_f32"),
        _f32_rule("mulf", "amdgpu.v_mul_f32"),
        _f32_rule("minnumf", "amdgpu.v_min_f32"),
        _f32_rule("maxnumf", "amdgpu.v_max_f32"),
    ],
)
