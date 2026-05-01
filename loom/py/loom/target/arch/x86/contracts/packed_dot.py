# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""x86 packed-dot source-to-low contract fragment."""

from __future__ import annotations

from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.target.arch.x86.descriptors import X86_PACKED_DOT_DESCRIPTOR_SET
from loom.target.arch.x86.packed_dot_data import (
    CONTRACT_FLAG_SATURATING,
    NUMERIC_BF16,
    NUMERIC_F16,
    NUMERIC_F32,
    NUMERIC_I8,
    NUMERIC_I32,
    NUMERIC_U8,
    X86_PACKED_DOT_DESCRIPTORS,
    PackedDotDescriptor,
)
from loom.target.contracts import (
    ContractCase,
    ContractFragment,
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    GuardDiagnostic,
    TypePattern,
    ValueRef,
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_DOT_TYPE_DIAGNOSTICS = {
    "lhs": GuardDiagnostic(
        subject_kind="type",
        subject_name="lhs",
        reason="x86 packed-dot lowering requires a descriptor-compatible lhs vector",
    ),
    "rhs": GuardDiagnostic(
        subject_kind="type",
        subject_name="rhs",
        reason="x86 packed-dot lowering requires a descriptor-compatible rhs vector",
    ),
    "acc": GuardDiagnostic(
        subject_kind="type",
        subject_name="acc",
        reason=(
            "x86 packed-dot lowering requires a descriptor-compatible "
            "accumulator vector"
        ),
    ),
    "result": GuardDiagnostic(
        subject_kind="type",
        subject_name="result",
        reason="x86 packed-dot lowering requires a descriptor-compatible result vector",
    ),
}

_DOT_KIND_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="attr",
    subject_name="kind",
    reason="x86 packed-dot lowering requires a supported vector dot kind",
)

_DOT_DESCRIPTOR_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="descriptor",
    subject_name="x86.packed_dot",
    reason=(
        "x86 packed-dot lowering requires a descriptor available in the active "
        "target contract and enabled by target features"
    ),
)

_DOT4I_KIND_BY_NUMERIC = {
    (NUMERIC_I8, NUMERIC_I8): "s8s8",
    (NUMERIC_U8, NUMERIC_I8): "u8s8",
    (NUMERIC_I8, NUMERIC_U8): "s8u8",
    (NUMERIC_U8, NUMERIC_U8): "u8u8",
}

_VECTOR_ELEMENT_BY_NUMERIC = {
    NUMERIC_BF16: "bf16",
    NUMERIC_F16: "f16",
    NUMERIC_F32: "f32",
    NUMERIC_I8: "i8",
    NUMERIC_I32: "i32",
    NUMERIC_U8: "i8",
}


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(X86_PACKED_DOT_DESCRIPTOR_SET, key)


def _vector_type(numeric_type: str, lane_count: int) -> TypePattern:
    return Vector(_VECTOR_ELEMENT_BY_NUMERIC[numeric_type], lanes=lane_count)


def _type_guard(field: str, type_pattern: TypePattern) -> Guard:
    return Guard.value_type(
        field,
        type_pattern,
        diagnostic=_DOT_TYPE_DIAGNOSTICS[field],
    )


def _emit(descriptor: Descriptor) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        operands={
            "acc": ValueRef.operand("acc"),
            "lhs": ValueRef.operand("lhs"),
            "rhs": ValueRef.operand("rhs"),
        },
        results={"dst": ValueRef.result("result")},
        form=DescriptorEmitForm.OP,
    )


def _dot2f_rule(descriptor_data: PackedDotDescriptor) -> DescriptorRule | None:
    if descriptor_data.reduction_group_size != 2:
        return None
    if descriptor_data.accumulator_numeric_type != NUMERIC_F32:
        return None
    if descriptor_data.result_numeric_type != NUMERIC_F32:
        return None
    if descriptor_data.lhs_numeric_type != descriptor_data.rhs_numeric_type:
        return None
    if descriptor_data.lhs_numeric_type not in (NUMERIC_F16, NUMERIC_BF16):
        return None

    descriptor = _descriptor(descriptor_data.key)
    source_type = _vector_type(
        descriptor_data.lhs_numeric_type,
        descriptor_data.input_lane_count,
    )
    result_type = _vector_type(NUMERIC_F32, descriptor_data.result_lane_count)
    return DescriptorRule(
        source_op=vector.vector_dot2f,
        descriptor=descriptor,
        guards=(
            _type_guard("lhs", source_type),
            _type_guard("rhs", source_type),
            _type_guard("acc", result_type),
            _type_guard("result", result_type),
            Guard.descriptor_available(
                descriptor,
                diagnostic=_DOT_DESCRIPTOR_DIAGNOSTIC,
            ),
        ),
        emit=(_emit(descriptor),),
    )


def _dot4i_rule(descriptor_data: PackedDotDescriptor) -> DescriptorRule | None:
    if descriptor_data.reduction_group_size != 4:
        return None
    if descriptor_data.accumulator_numeric_type != NUMERIC_I32:
        return None
    if descriptor_data.result_numeric_type != NUMERIC_I32:
        return None
    kind = _DOT4I_KIND_BY_NUMERIC.get(
        (descriptor_data.lhs_numeric_type, descriptor_data.rhs_numeric_type)
    )
    if kind is None:
        return None

    descriptor = _descriptor(descriptor_data.key)
    source_type = _vector_type(NUMERIC_I8, descriptor_data.input_lane_count)
    result_type = _vector_type(NUMERIC_I32, descriptor_data.result_lane_count)
    return DescriptorRule(
        source_op=vector.vector_dot4i,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("kind", kind, diagnostic=_DOT_KIND_DIAGNOSTIC),
            _type_guard("lhs", source_type),
            _type_guard("rhs", source_type),
            _type_guard("acc", result_type),
            _type_guard("result", result_type),
            Guard.descriptor_available(
                descriptor,
                diagnostic=_DOT_DESCRIPTOR_DIAGNOSTIC,
            ),
        ),
        emit=(_emit(descriptor),),
    )


def _packed_dot_rules() -> tuple[ContractCase, ...]:
    rules: list[ContractCase] = []
    for descriptor_data in X86_PACKED_DOT_DESCRIPTORS:
        if descriptor_data.flags & CONTRACT_FLAG_SATURATING:
            continue
        rule = _dot2f_rule(descriptor_data)
        if rule is None:
            rule = _dot4i_rule(descriptor_data)
        if rule is not None:
            rules.append(rule)
    return tuple(rules)


X86_PACKED_DOT_CONTRACT_DIALECT_OPS = {
    "vector": ALL_VECTOR_OPS,
}

X86_PACKED_DOT_CONTRACT_FRAGMENT = ContractFragment(
    name="x86.packed_dot",
    descriptor_set=X86_PACKED_DOT_DESCRIPTOR_SET,
    cases=_packed_dot_rules(),
)
