# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU scalar integer and index arithmetic source-to-low contracts."""

from __future__ import annotations

from loom.dialect.index import ALL_INDEX_OPS
from loom.dialect.index import defs as index
from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import bitwise as scalar_bitwise
from loom.dsl import Op
from loom.target.arch.amdgpu.contracts.materializers import (
    ADDRESS_VGPR_MATERIALIZER,
    I1_NATIVE_MASK_MATERIALIZER,
    I32_VGPR_MATERIALIZER,
)
from loom.target.arch.amdgpu.descriptors import build_amdgpu_contract_descriptor_set
from loom.target.contracts import (
    ContractFragment,
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    GuardDiagnostic,
    Scalar,
    TypePattern,
    ValueMaterializer,
    ValueProject,
    ValueRef,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_DESCRIPTOR_KEYS = (
    "amdgpu.s_add_u32",
    "amdgpu.s_sub_u32",
    "amdgpu.s_mul_i32",
    "amdgpu.s_min_i32",
    "amdgpu.s_max_i32",
    "amdgpu.s_min_u32",
    "amdgpu.s_max_u32",
    "amdgpu.s_and_b32",
    "amdgpu.s_or_b32",
    "amdgpu.s_xor_b32",
    "amdgpu.s_and_b64",
    "amdgpu.s_or_b64",
    "amdgpu.s_xor_b64",
    "amdgpu.s_lshl_b32",
    "amdgpu.s_lshr_b32",
    "amdgpu.s_ashr_i32",
    "amdgpu.v_add_u32",
    "amdgpu.v_sub_u32",
    "amdgpu.v_mul_lo_u32",
    "amdgpu.v_min_i32",
    "amdgpu.v_max_i32",
    "amdgpu.v_min_u32",
    "amdgpu.v_max_u32",
    "amdgpu.v_and_b32",
    "amdgpu.v_or_b32",
    "amdgpu.v_xor_b32",
    "amdgpu.v_lshlrev_b32",
    "amdgpu.v_lshlrev_b32.lit",
    "amdgpu.v_lshlrev_b32.vop3_imm",
    "amdgpu.v_lshrrev_b32",
    "amdgpu.v_lshrrev_b32.lit",
    "amdgpu.v_ashrrev_i32",
    "amdgpu.v_ashrrev_i32.lit",
)

_DESCRIPTOR_SET = build_amdgpu_contract_descriptor_set(
    key="amdgpu.integer",
    descriptor_keys=_DESCRIPTOR_KEYS,
)

_I1 = Scalar("i1")
_I32 = Scalar("i32")
_INDEX = Scalar("index")
_OFFSET = Scalar("offset")
_DIRECT_LHS = ValueRef.operand("lhs")
_DIRECT_RHS = ValueRef.operand("rhs")
_RESULT = ValueRef.result("result")
_ADDRESS_U32_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="address-width",
    subject_name="u32",
    constraint_key="amdgpu.address.u32",
)
_SHIFT_AMOUNT_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="shift-amount",
    subject_name="u5",
    constraint_key="amdgpu.shift_amount.u5",
)


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(_DESCRIPTOR_SET, key)


def _typed_binary_guards(
    type_pattern: TypePattern,
    *,
    unsigned_bit_count: int | None = None,
    unsigned_diagnostic: GuardDiagnostic | None = None,
) -> tuple[Guard, ...]:
    guards = [
        Guard.value_type("lhs", type_pattern),
        Guard.value_type("rhs", type_pattern),
        Guard.value_type("result", type_pattern),
    ]
    if unsigned_bit_count is not None:
        guards.append(
            Guard.value_unsigned_bit_count(
                "result",
                unsigned_bit_count,
                diagnostic=unsigned_diagnostic,
            )
        )
    return tuple(guards)


def _sgpr_binary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor: Descriptor,
    *,
    unsigned_bit_count: int | None = None,
    unsigned_diagnostic: GuardDiagnostic | None = None,
) -> DescriptorRule:
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_binary_guards(
                type_pattern,
                unsigned_bit_count=unsigned_bit_count,
                unsigned_diagnostic=unsigned_diagnostic,
            ),
            Guard.low_value_register_class("result", "amdgpu.sgpr"),
            Guard.low_value_register_class("lhs", "amdgpu.sgpr"),
            Guard.low_value_register_class("rhs", "amdgpu.sgpr"),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"lhs": _DIRECT_LHS, "rhs": _DIRECT_RHS},
                results={"dst": _RESULT},
            ),
        ),
    )


def _materialized_operand(field: str, materializer: ValueMaterializer) -> ValueRef:
    return ValueRef.operand(field, materializer=materializer.name)


def _vgpr_binary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor: Descriptor,
    materializer: ValueMaterializer,
    *,
    descriptor_lhs: str = "lhs",
    descriptor_rhs: str = "rhs",
    source_lhs: str = "lhs",
    source_rhs: str = "rhs",
    unsigned_bit_count: int | None = None,
    unsigned_diagnostic: GuardDiagnostic | None = None,
) -> DescriptorRule:
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_binary_guards(
                type_pattern,
                unsigned_bit_count=unsigned_bit_count,
                unsigned_diagnostic=unsigned_diagnostic,
            ),
            Guard.low_value_register_class("result", "amdgpu.vgpr"),
            Guard.value_materializable("lhs", materializer.name),
            Guard.value_materializable("rhs", materializer.name),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    descriptor_lhs: _materialized_operand(source_lhs, materializer),
                    descriptor_rhs: _materialized_operand(source_rhs, materializer),
                },
                results={"dst": _RESULT},
            ),
        ),
    )


def _vgpr_literal_shift_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor: Descriptor,
    materializer: ValueMaterializer,
    *,
    preserve_source_register: bool = False,
    unsigned_bit_count: int | None = None,
    unsigned_diagnostic: GuardDiagnostic | None = None,
) -> DescriptorRule:
    value_guard = (
        Guard.low_value_register_class("lhs", "amdgpu.sgpr")
        if preserve_source_register
        else Guard.value_materializable("lhs", materializer.name)
    )
    value_operand = (
        ValueRef.operand("lhs")
        if preserve_source_register
        else _materialized_operand("lhs", materializer)
    )
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_binary_guards(
                type_pattern,
                unsigned_bit_count=unsigned_bit_count,
                unsigned_diagnostic=unsigned_diagnostic,
            ),
            Guard.low_value_register_class("result", "amdgpu.vgpr"),
            value_guard,
            Guard.value_exact_i64("rhs", diagnostic=_SHIFT_AMOUNT_DIAGNOSTIC),
            Guard.value_i64_range(
                "rhs",
                0,
                31,
                diagnostic=_SHIFT_AMOUNT_DIAGNOSTIC,
            ),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"value": value_operand},
                results={"dst": _RESULT},
                immediates={"imm32": ValueProject.exact_i64("rhs")},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _i32_sgpr_vgpr_rules(
    source_op: Op,
    sgpr_descriptor_key: str,
    vgpr_descriptor_key: str,
) -> tuple[DescriptorRule, ...]:
    sgpr_descriptor = _descriptor(sgpr_descriptor_key)
    vgpr_descriptor = _descriptor(vgpr_descriptor_key)
    return (
        _sgpr_binary_rule(source_op, _I32, sgpr_descriptor),
        _vgpr_binary_rule(
            source_op,
            _I32,
            vgpr_descriptor,
            I32_VGPR_MATERIALIZER,
        ),
    )


def _i1_sgpr_mask_rule(
    source_op: Op,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_binary_guards(_I1),
            Guard.low_value_register_class("result", "amdgpu.sgpr"),
            Guard.value_materializable("lhs", I1_NATIVE_MASK_MATERIALIZER.name),
            Guard.value_materializable("rhs", I1_NATIVE_MASK_MATERIALIZER.name),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "lhs": _materialized_operand("lhs", I1_NATIVE_MASK_MATERIALIZER),
                    "rhs": _materialized_operand("rhs", I1_NATIVE_MASK_MATERIALIZER),
                },
                results={"dst": _RESULT},
            ),
        ),
    )


def _i32_shift_rules(
    source_op: Op,
    sgpr_descriptor_key: str,
    vgpr_descriptor_key: str,
    literal_descriptor_key: str,
    *,
    preserve_descriptor_key: str | None = None,
) -> tuple[DescriptorRule, ...]:
    sgpr_descriptor = _descriptor(sgpr_descriptor_key)
    vgpr_descriptor = _descriptor(vgpr_descriptor_key)
    literal_descriptor = _descriptor(literal_descriptor_key)
    preserve_rules = (
        (
            _vgpr_literal_shift_rule(
                source_op,
                _I32,
                _descriptor(preserve_descriptor_key),
                I32_VGPR_MATERIALIZER,
                preserve_source_register=True,
            ),
        )
        if preserve_descriptor_key is not None
        else ()
    )
    return (
        _sgpr_binary_rule(source_op, _I32, sgpr_descriptor),
        *preserve_rules,
        _vgpr_literal_shift_rule(
            source_op,
            _I32,
            literal_descriptor,
            I32_VGPR_MATERIALIZER,
        ),
        _vgpr_binary_rule(
            source_op,
            _I32,
            vgpr_descriptor,
            I32_VGPR_MATERIALIZER,
            descriptor_lhs="shift",
            descriptor_rhs="value",
            source_lhs="rhs",
            source_rhs="lhs",
        ),
    )


def _address_rules(
    source_op: Op,
    sgpr_descriptor_key: str,
    vgpr_descriptor_key: str,
) -> tuple[DescriptorRule, ...]:
    sgpr_descriptor = _descriptor(sgpr_descriptor_key)
    vgpr_descriptor = _descriptor(vgpr_descriptor_key)
    return tuple(
        _sgpr_binary_rule(
            source_op,
            type_pattern,
            sgpr_descriptor,
            unsigned_bit_count=32,
            unsigned_diagnostic=_ADDRESS_U32_DIAGNOSTIC,
        )
        for type_pattern in (_INDEX, _OFFSET)
    ) + tuple(
        _vgpr_binary_rule(
            source_op,
            type_pattern,
            vgpr_descriptor,
            ADDRESS_VGPR_MATERIALIZER,
            unsigned_bit_count=32,
            unsigned_diagnostic=_ADDRESS_U32_DIAGNOSTIC,
        )
        for type_pattern in (_INDEX, _OFFSET)
    )


def _address_shift_rules(
    source_op: Op,
    sgpr_descriptor_key: str,
    vgpr_descriptor_key: str,
    literal_descriptor_key: str,
    *,
    preserve_descriptor_key: str | None = None,
) -> tuple[DescriptorRule, ...]:
    sgpr_descriptor = _descriptor(sgpr_descriptor_key)
    vgpr_descriptor = _descriptor(vgpr_descriptor_key)
    literal_descriptor = _descriptor(literal_descriptor_key)
    preserve_descriptor = (
        _descriptor(preserve_descriptor_key)
        if preserve_descriptor_key is not None
        else None
    )
    rules: list[DescriptorRule] = []
    for type_pattern in (_INDEX, _OFFSET):
        rules.append(
            _sgpr_binary_rule(
                source_op,
                type_pattern,
                sgpr_descriptor,
                unsigned_bit_count=32,
                unsigned_diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            )
        )
        if preserve_descriptor is not None:
            rules.append(
                _vgpr_literal_shift_rule(
                    source_op,
                    type_pattern,
                    preserve_descriptor,
                    ADDRESS_VGPR_MATERIALIZER,
                    preserve_source_register=True,
                    unsigned_bit_count=32,
                    unsigned_diagnostic=_ADDRESS_U32_DIAGNOSTIC,
                )
            )
        rules.append(
            _vgpr_literal_shift_rule(
                source_op,
                type_pattern,
                literal_descriptor,
                ADDRESS_VGPR_MATERIALIZER,
                unsigned_bit_count=32,
                unsigned_diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            )
        )
        rules.append(
            _vgpr_binary_rule(
                source_op,
                type_pattern,
                vgpr_descriptor,
                ADDRESS_VGPR_MATERIALIZER,
                descriptor_lhs="shift",
                descriptor_rhs="value",
                source_lhs="rhs",
                source_rhs="lhs",
                unsigned_bit_count=32,
                unsigned_diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            )
        )
    return tuple(rules)


def _index_madd_sgpr_rule() -> DescriptorRule:
    multiply = _descriptor("amdgpu.s_mul_i32")
    add = _descriptor("amdgpu.s_add_u32")
    return DescriptorRule(
        source_op=index.index_madd,
        descriptor=multiply,
        guards=(
            Guard.value_type("a", _INDEX),
            Guard.value_type("b", _INDEX),
            Guard.value_type("c", _INDEX),
            Guard.value_type("result", _INDEX),
            Guard.value_unsigned_bit_count(
                "result",
                32,
                diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            ),
            Guard.low_value_register_class("a", "amdgpu.sgpr"),
            Guard.low_value_register_class("b", "amdgpu.sgpr"),
            Guard.low_value_register_class("c", "amdgpu.sgpr"),
            Guard.low_value_register_class("result", "amdgpu.sgpr"),
            Guard.descriptor_available(multiply),
            Guard.descriptor_available(add),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=multiply,
                operands={"lhs": ValueRef.operand("a"), "rhs": ValueRef.operand("b")},
                results={"dst": ValueRef.temporary("product")},
                result_types={"dst": ValueRef.result("result")},
            ),
            EmitDescriptorOp(
                descriptor=add,
                operands={
                    "lhs": ValueRef.temporary("product"),
                    "rhs": ValueRef.operand("c"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    rules.extend(
        _i32_sgpr_vgpr_rules(
            scalar_arithmetic.scalar_addi,
            "amdgpu.s_add_u32",
            "amdgpu.v_add_u32",
        )
    )
    rules.extend(
        _i32_sgpr_vgpr_rules(
            scalar_arithmetic.scalar_subi,
            "amdgpu.s_sub_u32",
            "amdgpu.v_sub_u32",
        )
    )
    rules.extend(
        _i32_sgpr_vgpr_rules(
            scalar_arithmetic.scalar_muli,
            "amdgpu.s_mul_i32",
            "amdgpu.v_mul_lo_u32",
        )
    )
    rules.append(
        _i1_sgpr_mask_rule(
            scalar_bitwise.scalar_andi,
            "amdgpu.s_and_b64",
        )
    )
    rules.extend(
        _i32_sgpr_vgpr_rules(
            scalar_bitwise.scalar_andi,
            "amdgpu.s_and_b32",
            "amdgpu.v_and_b32",
        )
    )
    rules.append(
        _i1_sgpr_mask_rule(
            scalar_bitwise.scalar_ori,
            "amdgpu.s_or_b64",
        )
    )
    rules.extend(
        _i32_sgpr_vgpr_rules(
            scalar_bitwise.scalar_ori,
            "amdgpu.s_or_b32",
            "amdgpu.v_or_b32",
        )
    )
    rules.append(
        _i1_sgpr_mask_rule(
            scalar_bitwise.scalar_xori,
            "amdgpu.s_xor_b64",
        )
    )
    rules.extend(
        _i32_sgpr_vgpr_rules(
            scalar_bitwise.scalar_xori,
            "amdgpu.s_xor_b32",
            "amdgpu.v_xor_b32",
        )
    )
    rules.extend(
        _i32_shift_rules(
            scalar_bitwise.scalar_shli,
            "amdgpu.s_lshl_b32",
            "amdgpu.v_lshlrev_b32",
            "amdgpu.v_lshlrev_b32.lit",
            preserve_descriptor_key="amdgpu.v_lshlrev_b32.vop3_imm",
        )
    )
    rules.extend(
        _i32_shift_rules(
            scalar_bitwise.scalar_shrsi,
            "amdgpu.s_ashr_i32",
            "amdgpu.v_ashrrev_i32",
            "amdgpu.v_ashrrev_i32.lit",
        )
    )
    rules.extend(
        _i32_shift_rules(
            scalar_bitwise.scalar_shrui,
            "amdgpu.s_lshr_b32",
            "amdgpu.v_lshrrev_b32",
            "amdgpu.v_lshrrev_b32.lit",
        )
    )
    rules.extend(
        _address_rules(index.index_add, "amdgpu.s_add_u32", "amdgpu.v_add_u32")
    )
    rules.extend(
        _address_rules(index.index_sub, "amdgpu.s_sub_u32", "amdgpu.v_sub_u32")
    )
    rules.extend(
        _address_rules(
            index.index_mul,
            "amdgpu.s_mul_i32",
            "amdgpu.v_mul_lo_u32",
        )
    )
    rules.extend(
        _address_rules(index.index_min, "amdgpu.s_min_i32", "amdgpu.v_min_i32")
    )
    rules.extend(
        _address_rules(index.index_max, "amdgpu.s_max_i32", "amdgpu.v_max_i32")
    )
    rules.extend(
        _address_rules(index.index_andi, "amdgpu.s_and_b32", "amdgpu.v_and_b32")
    )
    rules.extend(_address_rules(index.index_ori, "amdgpu.s_or_b32", "amdgpu.v_or_b32"))
    rules.extend(
        _address_rules(index.index_xori, "amdgpu.s_xor_b32", "amdgpu.v_xor_b32")
    )
    rules.extend(
        _address_shift_rules(
            index.index_shli,
            "amdgpu.s_lshl_b32",
            "amdgpu.v_lshlrev_b32",
            "amdgpu.v_lshlrev_b32.lit",
            preserve_descriptor_key="amdgpu.v_lshlrev_b32.vop3_imm",
        )
    )
    rules.extend(
        _address_shift_rules(
            index.index_shrsi,
            "amdgpu.s_ashr_i32",
            "amdgpu.v_ashrrev_i32",
            "amdgpu.v_ashrrev_i32.lit",
        )
    )
    rules.extend(
        _address_shift_rules(
            index.index_shrui,
            "amdgpu.s_lshr_b32",
            "amdgpu.v_lshrrev_b32",
            "amdgpu.v_lshrrev_b32.lit",
        )
    )
    rules.append(_index_madd_sgpr_rule())
    return tuple(rules)


AMDGPU_INTEGER_CONTRACT_DIALECT_OPS = {
    "index": ALL_INDEX_OPS,
    "scalar": ALL_SCALAR_OPS,
}

AMDGPU_INTEGER_CONTRACT_FRAGMENT = ContractFragment(
    name="amdgpu.integer",
    descriptor_set=_DESCRIPTOR_SET,
    materializers=(
        I32_VGPR_MATERIALIZER,
        ADDRESS_VGPR_MATERIALIZER,
        I1_NATIVE_MASK_MATERIALIZER,
    ),
    cases=_rules(),
)
