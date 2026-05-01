# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU scalar integer comparison source-to-low contracts."""

from __future__ import annotations

from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import comparison as scalar
from loom.target.arch.amdgpu.contracts.materializers import I32_VGPR_MATERIALIZER
from loom.target.arch.amdgpu.descriptors import build_amdgpu_contract_descriptor_set
from loom.target.contracts import (
    ContractTable,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    ValueRef,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_CMP_I32_CASES = (
    ("eq", "amdgpu.s_cmp_eq_i32", "amdgpu.v_cmp_eq_i32"),
    ("ne", "amdgpu.s_cmp_lg_i32", "amdgpu.v_cmp_ne_i32"),
    ("slt", "amdgpu.s_cmp_lt_i32", "amdgpu.v_cmp_slt_i32"),
    ("sle", "amdgpu.s_cmp_le_i32", "amdgpu.v_cmp_sle_i32"),
    ("sgt", "amdgpu.s_cmp_gt_i32", "amdgpu.v_cmp_sgt_i32"),
    ("sge", "amdgpu.s_cmp_ge_i32", "amdgpu.v_cmp_sge_i32"),
    ("ult", "amdgpu.s_cmp_lt_u32", "amdgpu.v_cmp_ult_u32"),
    ("ule", "amdgpu.s_cmp_le_u32", "amdgpu.v_cmp_ule_u32"),
    ("ugt", "amdgpu.s_cmp_gt_u32", "amdgpu.v_cmp_ugt_u32"),
    ("uge", "amdgpu.s_cmp_ge_u32", "amdgpu.v_cmp_uge_u32"),
)

_DESCRIPTOR_KEYS = tuple(
    descriptor_key
    for _, scalar_descriptor_key, mask_descriptor_key in _CMP_I32_CASES
    for descriptor_key in (scalar_descriptor_key, mask_descriptor_key)
)

_DESCRIPTOR_SET = build_amdgpu_contract_descriptor_set(
    key="amdgpu.compare",
    descriptor_keys=_DESCRIPTOR_KEYS,
)

_I32 = Scalar("i32")
_I1 = Scalar("i1")
_I32_VGPR_LHS = ValueRef.operand("lhs", materializer=I32_VGPR_MATERIALIZER.name)
_I32_VGPR_RHS = ValueRef.operand("rhs", materializer=I32_VGPR_MATERIALIZER.name)


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(_DESCRIPTOR_SET, key)


def _scc_rule(predicate: str, descriptor: Descriptor) -> DescriptorRule:
    return DescriptorRule(
        source_op=scalar.scalar_cmpi,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate),
            Guard.value_type("lhs", _I32),
            Guard.value_type("rhs", _I32),
            Guard.value_type("result", _I1),
            Guard.low_value_register_class("lhs", "amdgpu.sgpr"),
            Guard.low_value_register_class("rhs", "amdgpu.sgpr"),
            Guard.low_value_register_class("result", "amdgpu.scc"),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"scc": ValueRef.result("result")},
            ),
        ),
    )


def _mask_rule(predicate: str, descriptor: Descriptor) -> DescriptorRule:
    return DescriptorRule(
        source_op=scalar.scalar_cmpi,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate),
            Guard.value_type("lhs", _I32),
            Guard.value_type("rhs", _I32),
            Guard.value_type("result", _I1),
            Guard.low_value_register_class("result", "amdgpu.sgpr"),
            Guard.value_materializable("lhs", I32_VGPR_MATERIALIZER.name),
            Guard.value_materializable("rhs", I32_VGPR_MATERIALIZER.name),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "lhs": _I32_VGPR_LHS,
                    "rhs": _I32_VGPR_RHS,
                },
                results={"mask": ValueRef.result("result")},
            ),
        ),
    )


def _rules() -> tuple[DescriptorRule, ...]:
    scalar_rules = tuple(
        _scc_rule(predicate, _descriptor(descriptor_key))
        for predicate, descriptor_key, _ in _CMP_I32_CASES
    )
    mask_rules = tuple(
        _mask_rule(predicate, _descriptor(descriptor_key))
        for predicate, _, descriptor_key in _CMP_I32_CASES
    )
    return scalar_rules + mask_rules


AMDGPU_COMPARE_CONTRACT_DIALECT_OPS = {
    "scalar": ALL_SCALAR_OPS,
}

AMDGPU_COMPARE_CONTRACT_TABLE = ContractTable(
    name="amdgpu.compare",
    descriptor_set=_DESCRIPTOR_SET,
    materializers=(I32_VGPR_MATERIALIZER,),
    cases=_rules(),
)
