# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""x86 scalar source-to-low contract fragment."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from loom.dialect.buffer import ALL_BUFFER_OPS
from loom.dialect.buffer import defs as buffer
from loom.dialect.index import ALL_INDEX_OPS
from loom.dialect.index import defs as index
from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dsl import Op
from loom.target.arch.x86.descriptors import X86_SCALAR_DESCRIPTOR_SET
from loom.target.contracts import (
    AttrProject,
    ContractCase,
    ContractFragment,
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    GuardDiagnostic,
    Scalar,
    TypePattern,
    ValueAliasRule,
    ValueProject,
    ValueRef,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_I32 = Scalar("i32")
_INDEX = Scalar("index")
_OFFSET = Scalar("offset")

_DISP32_MIN = -(2**31)
_DISP32_MAX = (2**31) - 1

_ADDRESS_FORM_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="address-form",
    subject_name="x86-scalar",
    constraint_key="x86.scalar.address_form",
)

_DescriptorLookup = Callable[[str], Descriptor]


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(X86_SCALAR_DESCRIPTOR_SET, key)


def _op_emit(
    *,
    descriptor: Descriptor,
    operands: dict[str, ValueRef] | None = None,
    results: dict[str, ValueRef] | None = None,
    result_types: dict[str, TypePattern] | None = None,
    immediates: dict[str, AttrProject | ValueProject | int] | None = None,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        operands={} if operands is None else operands,
        results={} if results is None else results,
        result_types=result_types,
        immediates={} if immediates is None else immediates,
        form=DescriptorEmitForm.OP,
    )


def _typed_guards(
    fields: Iterable[str],
    type_pattern: TypePattern,
) -> tuple[Guard, ...]:
    return tuple(Guard.value_type(field, type_pattern) for field in fields)


def _exact_i64_literal_guards(field: str, value: int) -> tuple[Guard, ...]:
    return (
        Guard.value_exact_i64(field, diagnostic=_ADDRESS_FORM_DIAGNOSTIC),
        Guard.value_i64_range(
            field,
            value,
            value,
            diagnostic=_ADDRESS_FORM_DIAGNOSTIC,
        ),
    )


def _disp32_guards(field: str) -> tuple[Guard, ...]:
    return (
        Guard.value_exact_i64(field, diagnostic=_ADDRESS_FORM_DIAGNOSTIC),
        Guard.value_i64_range(
            field,
            _DISP32_MIN,
            _DISP32_MAX,
            diagnostic=_ADDRESS_FORM_DIAGNOSTIC,
        ),
    )


def _negated_disp32_guards(field: str) -> tuple[Guard, ...]:
    return (
        Guard.value_exact_i64(field, diagnostic=_ADDRESS_FORM_DIAGNOSTIC),
        Guard.value_i64_range(
            field,
            -_DISP32_MAX,
            -_DISP32_MIN,
            diagnostic=_ADDRESS_FORM_DIAGNOSTIC,
        ),
    )


def _binary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=_typed_guards(("lhs", "rhs", "result"), type_pattern),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _const_i64_rule(
    result_type: TypePattern,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup("x86.scalar.movimm.gpr64")
    return DescriptorRule(
        source_op=index.index_constant,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "i64"),
            Guard.value_type("result", result_type),
            Guard.i64_range("value", -(2**63) + 1, (2**63) - 1),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"imm64": AttrProject.direct("value")},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _buffer_view_rule() -> ValueAliasRule:
    return ValueAliasRule(
        source_op=buffer.buffer_view,
        source=ValueRef.operand("buffer"),
        result=ValueRef.result("result"),
    )


def _add_disp_rule(
    type_pattern: TypePattern,
    *,
    base_field: str,
    displacement_field: str,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup("x86.scalar.lea.disp.gpr64")
    return DescriptorRule(
        source_op=index.index_add,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), type_pattern),
            *_disp32_guards(displacement_field),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"base": ValueRef.operand(base_field)},
                results={"dst": ValueRef.result("result")},
                immediates={
                    "disp32": ValueProject.exact_i64(displacement_field),
                },
            ),
        ),
    )


def _sub_disp_rule(
    type_pattern: TypePattern,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup("x86.scalar.lea.disp.gpr64")
    return DescriptorRule(
        source_op=index.index_sub,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), type_pattern),
            *_negated_disp32_guards("rhs"),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"base": ValueRef.operand("lhs")},
                results={"dst": ValueRef.result("result")},
                immediates={"disp32": ValueProject.exact_i64_negate("rhs")},
            ),
        ),
    )


def _mul_scale_rule(
    *,
    index_field: str,
    scale_field: str,
    scale: int,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup("x86.scalar.lea.scale.gpr64")
    return DescriptorRule(
        source_op=index.index_mul,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), _INDEX),
            *_exact_i64_literal_guards(scale_field, scale),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"index": ValueRef.operand(index_field)},
                results={"dst": ValueRef.result("result")},
                immediates={"disp32": 0, "scale": scale},
            ),
        ),
    )


def _shli_scale_rule(
    *,
    shift_amount: int,
    scale: int,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup("x86.scalar.lea.scale.gpr64")
    return DescriptorRule(
        source_op=index.index_shli,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), _INDEX),
            *_exact_i64_literal_guards("rhs", shift_amount),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"index": ValueRef.operand("lhs")},
                results={"dst": ValueRef.result("result")},
                immediates={"disp32": 0, "scale": scale},
            ),
        ),
    )


def _madd_scale_disp_rule(
    *,
    index_field: str,
    scale_field: str,
    scale: int,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup("x86.scalar.lea.scale.gpr64")
    return DescriptorRule(
        source_op=index.index_madd,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("a", "b", "c", "result"), _INDEX),
            *_exact_i64_literal_guards(scale_field, scale),
            *_disp32_guards("c"),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"index": ValueRef.operand(index_field)},
                results={"dst": ValueRef.result("result")},
                immediates={
                    "disp32": ValueProject.exact_i64("c"),
                    "scale": scale,
                },
            ),
        ),
    )


def _madd_add_scale_rule(
    *,
    index_field: str,
    scale_field: str,
    scale: int,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup("x86.scalar.lea.add_scale.gpr64")
    return DescriptorRule(
        source_op=index.index_madd,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("a", "b", "c", "result"), _INDEX),
            *_exact_i64_literal_guards(scale_field, scale),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={
                    "base": ValueRef.operand("c"),
                    "index": ValueRef.operand(index_field),
                },
                results={"dst": ValueRef.result("result")},
                immediates={"disp32": 0, "scale": scale},
            ),
        ),
    )


def _madd_product_disp_rule(descriptor_lookup: _DescriptorLookup) -> DescriptorRule:
    multiply = descriptor_lookup("x86.scalar.imul.gpr64")
    add = descriptor_lookup("x86.scalar.lea.disp.gpr64")
    return DescriptorRule(
        source_op=index.index_madd,
        descriptor=add,
        guards=(
            *_typed_guards(("a", "b", "c", "result"), _INDEX),
            *_disp32_guards("c"),
        ),
        emit=(
            _op_emit(
                descriptor=multiply,
                operands={
                    "lhs": ValueRef.operand("a"),
                    "rhs": ValueRef.operand("b"),
                },
                results={"dst": ValueRef.temporary("product")},
                result_types={"dst": _INDEX},
            ),
            _op_emit(
                descriptor=add,
                operands={"base": ValueRef.temporary("product")},
                results={"dst": ValueRef.result("result")},
                immediates={"disp32": ValueProject.exact_i64("c")},
            ),
        ),
    )


def _index_madd_rule(descriptor_lookup: _DescriptorLookup) -> DescriptorRule:
    multiply = descriptor_lookup("x86.scalar.imul.gpr64")
    add = descriptor_lookup("x86.scalar.lea.add.gpr64")
    return DescriptorRule(
        source_op=index.index_madd,
        descriptor=add,
        guards=_typed_guards(("a", "b", "c", "result"), _INDEX),
        emit=(
            _op_emit(
                descriptor=multiply,
                operands={
                    "lhs": ValueRef.operand("a"),
                    "rhs": ValueRef.operand("b"),
                },
                results={"dst": ValueRef.temporary("product")},
                result_types={"dst": _INDEX},
            ),
            _op_emit(
                descriptor=add,
                operands={
                    "lhs": ValueRef.temporary("product"),
                    "rhs": ValueRef.operand("c"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _scale_rules(descriptor_lookup: _DescriptorLookup) -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for scale in (2, 4, 8):
        rules.append(
            _mul_scale_rule(
                index_field="lhs",
                scale_field="rhs",
                scale=scale,
                descriptor_lookup=descriptor_lookup,
            )
        )
        rules.append(
            _mul_scale_rule(
                index_field="rhs",
                scale_field="lhs",
                scale=scale,
                descriptor_lookup=descriptor_lookup,
            )
        )
    for shift_amount, scale in ((1, 2), (2, 4), (3, 8)):
        rules.append(
            _shli_scale_rule(
                shift_amount=shift_amount,
                scale=scale,
                descriptor_lookup=descriptor_lookup,
            )
        )
    return tuple(rules)


def _madd_address_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for scale in (1, 2, 4, 8):
        for index_field, scale_field in (("a", "b"), ("b", "a")):
            rules.append(
                _madd_scale_disp_rule(
                    index_field=index_field,
                    scale_field=scale_field,
                    scale=scale,
                    descriptor_lookup=descriptor_lookup,
                )
            )
        for index_field, scale_field in (("a", "b"), ("b", "a")):
            rules.append(
                _madd_add_scale_rule(
                    index_field=index_field,
                    scale_field=scale_field,
                    scale=scale,
                    descriptor_lookup=descriptor_lookup,
                )
            )
    rules.append(_madd_product_disp_rule(descriptor_lookup))
    rules.append(_index_madd_rule(descriptor_lookup))
    return tuple(rules)


def x86_scalar_core_cases(
    descriptor_lookup: _DescriptorLookup,
) -> Sequence[ContractCase]:
    return (
        _buffer_view_rule(),
        _binary_rule(
            scalar_arithmetic.scalar_addi,
            _I32,
            "x86.scalar.add.gpr32",
            descriptor_lookup,
        ),
        _binary_rule(
            scalar_arithmetic.scalar_subi,
            _I32,
            "x86.scalar.sub.gpr32",
            descriptor_lookup,
        ),
        _binary_rule(
            scalar_arithmetic.scalar_muli,
            _I32,
            "x86.scalar.imul.gpr32",
            descriptor_lookup,
        ),
        _const_i64_rule(_INDEX, descriptor_lookup),
        _const_i64_rule(_OFFSET, descriptor_lookup),
        _add_disp_rule(
            _INDEX,
            base_field="lhs",
            displacement_field="rhs",
            descriptor_lookup=descriptor_lookup,
        ),
        _add_disp_rule(
            _INDEX,
            base_field="rhs",
            displacement_field="lhs",
            descriptor_lookup=descriptor_lookup,
        ),
        _add_disp_rule(
            _OFFSET,
            base_field="lhs",
            displacement_field="rhs",
            descriptor_lookup=descriptor_lookup,
        ),
        _add_disp_rule(
            _OFFSET,
            base_field="rhs",
            displacement_field="lhs",
            descriptor_lookup=descriptor_lookup,
        ),
        _sub_disp_rule(_INDEX, descriptor_lookup),
        _sub_disp_rule(_OFFSET, descriptor_lookup),
        _binary_rule(
            index.index_add, _INDEX, "x86.scalar.lea.add.gpr64", descriptor_lookup
        ),
        _binary_rule(
            index.index_add, _OFFSET, "x86.scalar.lea.add.gpr64", descriptor_lookup
        ),
        *_scale_rules(descriptor_lookup),
        _binary_rule(
            index.index_mul, _INDEX, "x86.scalar.imul.gpr64", descriptor_lookup
        ),
        *_madd_address_rules(descriptor_lookup),
    )


X86_SCALAR_CONTRACT_DIALECT_OPS = {
    "buffer": ALL_BUFFER_OPS,
    "index": ALL_INDEX_OPS,
    "scalar": ALL_SCALAR_OPS,
}

X86_SCALAR_CONTRACT_FRAGMENT = ContractFragment(
    name="x86.scalar",
    descriptor_set=X86_SCALAR_DESCRIPTOR_SET,
    cases=x86_scalar_core_cases(_descriptor),
)
