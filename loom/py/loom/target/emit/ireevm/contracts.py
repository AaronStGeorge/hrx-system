# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""IREE VM source-to-low contract fragment."""

from __future__ import annotations

from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import comparison as scalar_comparison
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dsl import Op
from loom.target.contracts import (
    AttrProject,
    ContractFragment,
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    GuardDiagnostic,
    Scalar,
    TypePattern,
    ValueRef,
    descriptor_by_key,
)
from loom.target.emit.ireevm.descriptors import IREEVM_CORE_DESCRIPTOR_SET
from loom.target.low_descriptors import Descriptor

_I1 = Scalar("i1")
_I32 = Scalar("i32")

_I1_OR_I32_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="type",
    subject_name="source",
    reason="IREE VM lowering currently supports only i1/i32 scalar values",
)
_I32_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="type",
    subject_name="i32",
    reason="IREE VM lowering requires i32 scalar operands",
)
_I64_ATTR_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="attr",
    subject_name="value",
    reason="IREE VM constant lowering requires an i64 value",
)
_I1_CONSTANT_RANGE_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="attr",
    subject_name="value",
    reason="IREE VM i1 constants must be either zero or one",
)
_I32_CONSTANT_RANGE_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="attr",
    subject_name="value",
    reason="IREE VM i32 constants must fit in signed i32",
)
_CMPI_EQ_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="attr",
    subject_name="predicate",
    reason="IREE VM scalar.cmpi lowering supports eq only",
)


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(IREEVM_CORE_DESCRIPTOR_SET, key)


def _const_i32_rule(
    result_type: TypePattern,
    range_diagnostic: GuardDiagnostic,
) -> DescriptorRule:
    descriptor = _descriptor("iree.vm.const.i32")
    return DescriptorRule(
        source_op=scalar_conversion.scalar_constant,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "i64", diagnostic=_I64_ATTR_DIAGNOSTIC),
            Guard.value_type("result", result_type, diagnostic=_I1_OR_I32_DIAGNOSTIC),
            Guard.i64_range(
                "value",
                0 if result_type == _I1 else -(2**31),
                1 if result_type == _I1 else (2**31) - 1,
                diagnostic=range_diagnostic,
            ),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"i32_value": AttrProject.direct("value")},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _binary_i32_rule(source_op: Op, descriptor_key: str) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.value_type("lhs", _I32, diagnostic=_I32_DIAGNOSTIC),
            Guard.value_type("rhs", _I32, diagnostic=_I32_DIAGNOSTIC),
            Guard.value_type("result", _I32, diagnostic=_I32_DIAGNOSTIC),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _cmpi_eq_rule(result_type: TypePattern) -> DescriptorRule:
    descriptor = _descriptor("iree.vm.cmp.eq.i32")
    return DescriptorRule(
        source_op=scalar_comparison.scalar_cmpi,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals(
                "predicate",
                "eq",
                diagnostic=_CMPI_EQ_DIAGNOSTIC,
            ),
            Guard.value_type("lhs", _I32, diagnostic=_I32_DIAGNOSTIC),
            Guard.value_type("rhs", _I32, diagnostic=_I32_DIAGNOSTIC),
            Guard.value_type(
                "result",
                result_type,
                diagnostic=_I1_OR_I32_DIAGNOSTIC,
            ),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


IREEVM_CORE_CONTRACT_DIALECT_OPS = {
    "scalar": ALL_SCALAR_OPS,
}

IREEVM_CORE_CONTRACT_FRAGMENT = ContractFragment(
    name="iree.vm.core",
    descriptor_set=IREEVM_CORE_DESCRIPTOR_SET,
    public_header="loom/target/emit/ireevm/lower/rules.h",
    cases=(
        _binary_i32_rule(scalar_arithmetic.scalar_addi, "iree.vm.add.i32"),
        _binary_i32_rule(scalar_arithmetic.scalar_subi, "iree.vm.sub.i32"),
        _cmpi_eq_rule(_I1),
        _cmpi_eq_rule(_I32),
        _const_i32_rule(_I32, _I32_CONSTANT_RANGE_DIAGNOSTIC),
        _const_i32_rule(_I1, _I1_CONSTANT_RANGE_DIAGNOSTIC),
    ),
)
