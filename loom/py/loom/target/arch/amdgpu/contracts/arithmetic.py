# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU vector, float scalar, and index madd source-to-low contracts."""

from __future__ import annotations

from loom.dialect.index import ALL_INDEX_OPS
from loom.dialect.index import defs as index
from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.scalar import math as scalar_math
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.amdgpu.contracts.materializers import (
    ADDRESS_VGPR_MATERIALIZER,
    F32_VGPR_MATERIALIZER,
)
from loom.target.arch.amdgpu.descriptors import build_amdgpu_contract_descriptor_set
from loom.target.contracts import (
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
    ValueMaterializer,
    ValueProject,
    ValueRef,
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_DESCRIPTOR_KEYS = (
    "amdgpu.v_add_f32",
    "amdgpu.v_add_f32.lit",
    "amdgpu.v_sub_f32",
    "amdgpu.v_sub_f32.lit",
    "amdgpu.v_mul_f32",
    "amdgpu.v_mul_f32.lit",
    "amdgpu.v_min_f32",
    "amdgpu.v_min_f32.lit",
    "amdgpu.v_max_f32",
    "amdgpu.v_max_f32.lit",
    "amdgpu.v_fma_f32",
    "amdgpu.v_exp_f32",
    "amdgpu.v_sqrt_f32",
    "amdgpu.v_rsq_f32",
    "amdgpu.v_rcp_f32",
    "amdgpu.v_cvt_f32_f16",
    "amdgpu.v_cvt_f16_f32",
    "amdgpu.v_cvt_f32_i32",
    "amdgpu.v_cvt_f32_u32",
    "amdgpu.v_add_u32",
    "amdgpu.v_add_u32.lit",
    "amdgpu.v_sub_u32",
    "amdgpu.v_mul_lo_u32",
    "amdgpu.v_min_i32",
    "amdgpu.v_max_i32",
    "amdgpu.v_min_u32",
    "amdgpu.v_max_u32",
    "amdgpu.v_and_b32",
    "amdgpu.v_and_b32.lit",
    "amdgpu.v_or_b32",
    "amdgpu.v_or_b32.lit",
    "amdgpu.v_xor_b32",
    "amdgpu.v_xor_b32.lit",
    "amdgpu.v_lshlrev_b32",
    "amdgpu.v_lshlrev_b32.lit",
    "amdgpu.v_lshlrev_b32.vop3_imm",
    "amdgpu.v_ashrrev_i32",
    "amdgpu.v_ashrrev_i32.lit",
    "amdgpu.v_lshrrev_b32",
    "amdgpu.v_lshrrev_b32.lit",
)

_DESCRIPTOR_SET = build_amdgpu_contract_descriptor_set(
    key="amdgpu.arithmetic",
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
_F16 = Scalar("f16")
_F32 = Scalar("f32")
_INDEX = Scalar("index")
_F32_ABS_MASK = 0x7FFFFFFF
_F32_SIGN_MASK = 0x80000000

_VEC_I32_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="type",
    subject_name="vector<i32>",
    constraint_key="amdgpu.arithmetic.vector_i32",
)
_VEC_F32_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="type",
    subject_name="vector<f32>",
    constraint_key="amdgpu.arithmetic.vector_f32",
)
_I32_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="type",
    subject_name="i32",
    constraint_key="amdgpu.arithmetic.i32",
)
_F16_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="type",
    subject_name="f16",
    constraint_key="amdgpu.arithmetic.f16",
)
_F32_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="type",
    subject_name="f32",
    constraint_key="amdgpu.arithmetic.f32",
)
_INDEX_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="type",
    subject_name="index",
    constraint_key="amdgpu.index.scalar",
)
_ADDRESS_U32_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="address-width",
    subject_name="u32",
    constraint_key="amdgpu.address.u32",
)
_ADDRESS_EXACT_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="address-literal",
    subject_name="i64",
    constraint_key="amdgpu.address.exact_i64",
)
_ADDRESS_POWER_OF_TWO_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="address-scale",
    subject_name="i64",
    constraint_key="amdgpu.address.exact_power_of_two_i64",
)
_ADDRESS_VGPR_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="materializer",
    subject_name="address-vgpr",
    constraint_key="amdgpu.address.vgpr_materializer",
)
_ADDRESS_SGPR_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="register-class",
    subject_name="sgpr",
    constraint_key="amdgpu.address.sgpr",
)
_RESULT_VGPR_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="register-class",
    subject_name="vgpr",
    constraint_key="amdgpu.arithmetic.result_vgpr",
)
_LITERAL_EXACT_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="literal",
    subject_name="i64",
    constraint_key="amdgpu.literal.exact_i64",
)
_LITERAL_EXACT_F32_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="literal",
    subject_name="f32",
    constraint_key="amdgpu.literal.exact_f32",
)
_LITERAL_I32_BITS_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="literal-bits",
    subject_name="i32",
    constraint_key="amdgpu.literal.i32_bits",
)


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(_DESCRIPTOR_SET, key)


def _type_diagnostic(type_pattern: TypePattern) -> GuardDiagnostic:
    if type_pattern == _VEC_I32:
        return _VEC_I32_DIAGNOSTIC
    if type_pattern == _VEC_F32:
        return _VEC_F32_DIAGNOSTIC
    if type_pattern == _I32:
        return _I32_DIAGNOSTIC
    if type_pattern == _F16:
        return _F16_DIAGNOSTIC
    if type_pattern == _F32:
        return _F32_DIAGNOSTIC
    if type_pattern == _INDEX:
        return _INDEX_DIAGNOSTIC
    raise ValueError(f"unknown AMDGPU arithmetic type pattern: {type_pattern!r}")


def _value_type(field: str, type_pattern: TypePattern) -> Guard:
    return Guard.value_type(
        field,
        type_pattern,
        diagnostic=_type_diagnostic(type_pattern),
    )


def _typed_guards(
    fields: tuple[str, ...],
    type_pattern: TypePattern,
) -> tuple[Guard, ...]:
    return tuple(_value_type(field, type_pattern) for field in fields)


def _f32_vgpr_operand(field: str) -> ValueRef:
    return _materialized_operand(field, F32_VGPR_MATERIALIZER)


def _bitcast_alias_rule(
    input_type: TypePattern,
    result_type: TypePattern,
) -> ValueAliasRule:
    return ValueAliasRule(
        source_op=scalar_conversion.scalar_bitcast,
        source=ValueRef.operand("input"),
        result=ValueRef.result("result"),
        guards=(
            _value_type("input", input_type),
            _value_type("result", result_type),
        ),
    )


def _emit_form(type_pattern: TypePattern) -> DescriptorEmitForm:
    if type_pattern.kind == "vector":
        return DescriptorEmitForm.PER_LANE
    return DescriptorEmitForm.OP


def _binary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
    *,
    descriptor_lhs: str = "lhs",
    descriptor_rhs: str = "rhs",
    source_lhs: str = "lhs",
    source_rhs: str = "rhs",
    f32_operands: bool = False,
    extra_guards: tuple[Guard, ...] = (),
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    operands = (
        {
            descriptor_lhs: _f32_vgpr_operand(source_lhs),
            descriptor_rhs: _f32_vgpr_operand(source_rhs),
        }
        if f32_operands
        else {
            descriptor_lhs: ValueRef.operand(source_lhs),
            descriptor_rhs: ValueRef.operand(source_rhs),
        }
    )
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_guards((source_lhs, source_rhs, "result"), type_pattern),
            *extra_guards,
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands=operands,
                results={"dst": ValueRef.result("result")},
                form=_emit_form(type_pattern),
            ),
        ),
    )


def _unary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
    *,
    f32_operand: bool = False,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("input", "result"), type_pattern),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "input": _f32_vgpr_operand("input")
                    if f32_operand
                    else ValueRef.operand("input")
                },
                results={"dst": ValueRef.result("result")},
                form=_emit_form(type_pattern),
            ),
        ),
    )


def _f32_neg_rule(
    source_op: Op,
    type_pattern: TypePattern,
    *,
    f32_operand: bool = False,
) -> DescriptorRule:
    descriptor = _descriptor("amdgpu.v_xor_b32.lit")
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("input", "result"), type_pattern),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "rhs": _f32_vgpr_operand("input")
                    if f32_operand
                    else ValueRef.operand("input")
                },
                results={"dst": ValueRef.result("result")},
                immediates={"imm32": _F32_SIGN_MASK},
                form=_emit_form(type_pattern),
            ),
        ),
    )


def _f32_abs_rule(
    source_op: Op,
    type_pattern: TypePattern,
    *,
    f32_operand: bool = False,
) -> DescriptorRule:
    descriptor = _descriptor("amdgpu.v_and_b32.lit")
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("input", "result"), type_pattern),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "rhs": _f32_vgpr_operand("input")
                    if f32_operand
                    else ValueRef.operand("input")
                },
                results={"dst": ValueRef.result("result")},
                immediates={"imm32": _F32_ABS_MASK},
                form=_emit_form(type_pattern),
            ),
        ),
    )


def _ternary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
    *,
    f32_operands: bool = False,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("a", "b", "c", "result"), type_pattern),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "a": _f32_vgpr_operand("a")
                    if f32_operands
                    else ValueRef.operand("a"),
                    "b": _f32_vgpr_operand("b")
                    if f32_operands
                    else ValueRef.operand("b"),
                    "c": _f32_vgpr_operand("c")
                    if f32_operands
                    else ValueRef.operand("c"),
                },
                results={"dst": ValueRef.result("result")},
                form=_emit_form(type_pattern),
            ),
        ),
    )


def _divf_arcp_one_rule(
    source_op: Op,
    type_pattern: TypePattern,
    *,
    f32_operand: bool = False,
) -> DescriptorRule:
    reciprocal = _descriptor("amdgpu.v_rcp_f32")
    return DescriptorRule(
        source_op=source_op,
        descriptor=reciprocal,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), type_pattern),
            Guard.instance_flags_has_all("fastmath", "arcp"),
            Guard.value_f64_equals("lhs", 1.0),
            Guard.descriptor_available(reciprocal),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=reciprocal,
                operands={
                    "input": _f32_vgpr_operand("rhs")
                    if f32_operand
                    else ValueRef.operand("rhs")
                },
                results={"dst": ValueRef.result("result")},
                form=_emit_form(type_pattern),
            ),
        ),
    )


def _divf_arcp_rule(
    source_op: Op,
    type_pattern: TypePattern,
    *,
    f32_operands: bool = False,
) -> DescriptorRule:
    reciprocal = _descriptor("amdgpu.v_rcp_f32")
    multiply = _descriptor("amdgpu.v_mul_f32")
    return DescriptorRule(
        source_op=source_op,
        descriptor=reciprocal,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), type_pattern),
            Guard.instance_flags_has_all("fastmath", "arcp"),
            Guard.descriptor_available(reciprocal),
            Guard.descriptor_available(multiply),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=reciprocal,
                operands={
                    "input": _f32_vgpr_operand("rhs")
                    if f32_operands
                    else ValueRef.operand("rhs")
                },
                results={"dst": ValueRef.temporary("reciprocal")},
                result_types={"dst": ValueRef.result("result")},
                form=_emit_form(type_pattern),
            ),
            EmitDescriptorOp(
                descriptor=multiply,
                operands={
                    "lhs": _f32_vgpr_operand("lhs")
                    if f32_operands
                    else ValueRef.operand("lhs"),
                    "rhs": ValueRef.temporary("reciprocal"),
                },
                results={"dst": ValueRef.result("result")},
                form=_emit_form(type_pattern),
            ),
        ),
    )


def _cast_rule(
    source_op: Op,
    input_type: TypePattern,
    result_type: TypePattern,
    descriptor_key: str,
    *,
    f32_input: bool = False,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            _value_type("input", input_type),
            _value_type("result", result_type),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "input": _f32_vgpr_operand("input")
                    if f32_input
                    else ValueRef.operand("input")
                },
                results={"dst": ValueRef.result("result")},
                form=_emit_form(result_type),
            ),
        ),
    )


def _literal_binary_rule(
    source_op: Op,
    descriptor_key: str,
    *,
    literal_source: str,
    nonliteral_source: str,
    descriptor_operand: str = "rhs",
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), _VEC_I32),
            Guard.value_exact_i64(
                literal_source,
                diagnostic=_LITERAL_EXACT_DIAGNOSTIC,
            ),
            Guard.value_signed_bit_count(
                literal_source,
                32,
                diagnostic=_LITERAL_I32_BITS_DIAGNOSTIC,
            ),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={descriptor_operand: ValueRef.operand(nonliteral_source)},
                results={"dst": ValueRef.result("result")},
                immediates={
                    "imm32": ValueProject.i32_as_u32_bits(literal_source),
                },
                form=DescriptorEmitForm.PER_LANE,
            ),
        ),
    )


def _f32_literal_binary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
    *,
    literal_source: str,
    nonliteral_source: str,
    f32_operand: bool = False,
    extra_guards: tuple[Guard, ...] = (),
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), type_pattern),
            Guard.value_exact_f64(
                literal_source,
                diagnostic=_LITERAL_EXACT_F32_DIAGNOSTIC,
            ),
            *extra_guards,
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "rhs": _f32_vgpr_operand(nonliteral_source)
                    if f32_operand
                    else ValueRef.operand(nonliteral_source)
                },
                results={"dst": ValueRef.result("result")},
                immediates={
                    "imm32": ValueProject.f64_as_f32_bits(literal_source),
                },
                form=_emit_form(type_pattern),
            ),
        ),
    )


def _index_madd_power_of_two_rule(
    *,
    scale_source: str,
    value_source: str,
    literal_addend: bool,
    preserve_value_register: bool,
) -> DescriptorRule:
    shift = _descriptor(
        "amdgpu.v_lshlrev_b32.vop3_imm"
        if preserve_value_register
        else "amdgpu.v_lshlrev_b32.lit"
    )
    add = _descriptor("amdgpu.v_add_u32.lit" if literal_addend else "amdgpu.v_add_u32")
    value_guard = (
        Guard.low_value_register_class(
            value_source,
            "amdgpu.sgpr",
            diagnostic=_ADDRESS_SGPR_DIAGNOSTIC,
        )
        if preserve_value_register
        else Guard.value_materializable(
            value_source,
            ADDRESS_VGPR_MATERIALIZER.name,
            diagnostic=_ADDRESS_VGPR_DIAGNOSTIC,
        )
    )
    value_operand = (
        ValueRef.operand(value_source)
        if preserve_value_register
        else _materialized_operand(value_source, ADDRESS_VGPR_MATERIALIZER)
    )
    addend_guards = (
        (
            Guard.value_exact_i64("c", diagnostic=_ADDRESS_EXACT_DIAGNOSTIC),
            Guard.value_unsigned_bit_count(
                "c",
                32,
                diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            ),
        )
        if literal_addend
        else (
            Guard.value_materializable(
                "c",
                ADDRESS_VGPR_MATERIALIZER.name,
                diagnostic=_ADDRESS_VGPR_DIAGNOSTIC,
            ),
        )
    )
    add_emit = (
        EmitDescriptorOp(
            descriptor=add,
            operands={"rhs": ValueRef.temporary("scaled")},
            results={"dst": ValueRef.result("result")},
            immediates={"imm32": ValueProject.exact_i64("c")},
            form=DescriptorEmitForm.OP,
        )
        if literal_addend
        else EmitDescriptorOp(
            descriptor=add,
            operands={
                "lhs": ValueRef.temporary("scaled"),
                "rhs": _materialized_operand("c", ADDRESS_VGPR_MATERIALIZER),
            },
            results={"dst": ValueRef.result("result")},
            form=DescriptorEmitForm.OP,
        )
    )
    return DescriptorRule(
        source_op=index.index_madd,
        descriptor=shift,
        guards=(
            *_typed_guards(("a", "b", "c", "result"), _INDEX),
            Guard.value_unsigned_bit_count(
                "result",
                32,
                diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            ),
            Guard.low_value_register_class(
                "result",
                "amdgpu.vgpr",
                diagnostic=_RESULT_VGPR_DIAGNOSTIC,
            ),
            Guard.value_exact_power_of_two_i64(
                scale_source,
                diagnostic=_ADDRESS_POWER_OF_TWO_DIAGNOSTIC,
            ),
            Guard.value_unsigned_bit_count(
                scale_source,
                32,
                diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            ),
            value_guard,
            *addend_guards,
            Guard.descriptor_available(shift),
            Guard.descriptor_available(add),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=shift,
                operands={"value": value_operand},
                results={"dst": ValueRef.temporary("scaled")},
                result_types={"dst": ValueRef.result("result")},
                immediates={
                    "imm32": ValueProject.exact_i64_log2(scale_source),
                },
                form=DescriptorEmitForm.OP,
            ),
            add_emit,
        ),
    )


def _index_madd_literal_rule() -> DescriptorRule:
    multiply = _descriptor("amdgpu.v_mul_lo_u32")
    add = _descriptor("amdgpu.v_add_u32.lit")
    return DescriptorRule(
        source_op=index.index_madd,
        descriptor=multiply,
        guards=(
            *_typed_guards(("a", "b", "c", "result"), _INDEX),
            Guard.value_unsigned_bit_count(
                "result",
                32,
                diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            ),
            Guard.low_value_register_class(
                "result",
                "amdgpu.vgpr",
                diagnostic=_RESULT_VGPR_DIAGNOSTIC,
            ),
            Guard.value_materializable(
                "a",
                ADDRESS_VGPR_MATERIALIZER.name,
                diagnostic=_ADDRESS_VGPR_DIAGNOSTIC,
            ),
            Guard.value_materializable(
                "b",
                ADDRESS_VGPR_MATERIALIZER.name,
                diagnostic=_ADDRESS_VGPR_DIAGNOSTIC,
            ),
            Guard.value_exact_i64("c", diagnostic=_ADDRESS_EXACT_DIAGNOSTIC),
            Guard.value_unsigned_bit_count(
                "c",
                32,
                diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            ),
            Guard.descriptor_available(multiply),
            Guard.descriptor_available(add),
        ),
        emit=(
            _index_madd_product_emit(multiply),
            EmitDescriptorOp(
                descriptor=add,
                operands={"rhs": ValueRef.temporary("product")},
                results={"dst": ValueRef.result("result")},
                immediates={"imm32": ValueProject.exact_i64("c")},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _index_madd_rule() -> DescriptorRule:
    multiply = _descriptor("amdgpu.v_mul_lo_u32")
    add = _descriptor("amdgpu.v_add_u32")
    return DescriptorRule(
        source_op=index.index_madd,
        descriptor=multiply,
        guards=(
            *_typed_guards(("a", "b", "c", "result"), _INDEX),
            Guard.value_unsigned_bit_count(
                "result",
                32,
                diagnostic=_ADDRESS_U32_DIAGNOSTIC,
            ),
            Guard.low_value_register_class(
                "result",
                "amdgpu.vgpr",
                diagnostic=_RESULT_VGPR_DIAGNOSTIC,
            ),
            Guard.value_materializable(
                "a",
                ADDRESS_VGPR_MATERIALIZER.name,
                diagnostic=_ADDRESS_VGPR_DIAGNOSTIC,
            ),
            Guard.value_materializable(
                "b",
                ADDRESS_VGPR_MATERIALIZER.name,
                diagnostic=_ADDRESS_VGPR_DIAGNOSTIC,
            ),
            Guard.value_materializable(
                "c",
                ADDRESS_VGPR_MATERIALIZER.name,
                diagnostic=_ADDRESS_VGPR_DIAGNOSTIC,
            ),
            Guard.descriptor_available(multiply),
            Guard.descriptor_available(add),
        ),
        emit=(
            _index_madd_product_emit(multiply),
            EmitDescriptorOp(
                descriptor=add,
                operands={
                    "lhs": ValueRef.temporary("product"),
                    "rhs": _materialized_operand("c", ADDRESS_VGPR_MATERIALIZER),
                },
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _index_madd_product_emit(descriptor: Descriptor) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        operands={
            "lhs": _materialized_operand("a", ADDRESS_VGPR_MATERIALIZER),
            "rhs": _materialized_operand("b", ADDRESS_VGPR_MATERIALIZER),
        },
        results={"dst": ValueRef.temporary("product")},
        result_types={"dst": ValueRef.result("result")},
        form=DescriptorEmitForm.OP,
    )


def _materialized_operand(field: str, materializer: ValueMaterializer) -> ValueRef:
    return ValueRef.operand(field, materializer=materializer.name)


def _register_class(field: str, register_class: str) -> Guard:
    return Guard.low_value_register_class(field, register_class)


def _commutative_f32_vector_binary_rules(
    source_op: Op,
    descriptor_key: str,
) -> tuple[DescriptorRule, DescriptorRule]:
    return (
        _binary_rule(
            source_op,
            _VEC_F32,
            descriptor_key,
            descriptor_lhs="lhs",
            descriptor_rhs="rhs",
            source_lhs="rhs",
            source_rhs="lhs",
            extra_guards=(
                _register_class("rhs", "amdgpu.sgpr"),
                _register_class("lhs", "amdgpu.vgpr"),
            ),
        ),
        _binary_rule(
            source_op,
            _VEC_F32,
            descriptor_key,
            extra_guards=(_register_class("rhs", "amdgpu.vgpr"),),
        ),
    )


def _rules() -> tuple[ContractCase, ...]:
    rules: list[ContractCase] = []
    for source_op, descriptor_key in (
        (vector.vector_addf, "amdgpu.v_add_f32.lit"),
        (vector.vector_mulf, "amdgpu.v_mul_f32.lit"),
        (vector.vector_minnumf, "amdgpu.v_min_f32.lit"),
        (vector.vector_maxnumf, "amdgpu.v_max_f32.lit"),
    ):
        rules.extend(
            (
                _f32_literal_binary_rule(
                    source_op,
                    _VEC_F32,
                    descriptor_key,
                    literal_source="lhs",
                    nonliteral_source="rhs",
                    extra_guards=(_register_class("rhs", "amdgpu.vgpr"),),
                ),
                _f32_literal_binary_rule(
                    source_op,
                    _VEC_F32,
                    descriptor_key,
                    literal_source="rhs",
                    nonliteral_source="lhs",
                    extra_guards=(_register_class("lhs", "amdgpu.vgpr"),),
                ),
            )
        )
    rules.append(
        _f32_literal_binary_rule(
            vector.vector_subf,
            _VEC_F32,
            "amdgpu.v_sub_f32.lit",
            literal_source="lhs",
            nonliteral_source="rhs",
            extra_guards=(_register_class("rhs", "amdgpu.vgpr"),),
        )
    )
    rules.extend(
        (
            *_commutative_f32_vector_binary_rules(
                vector.vector_addf,
                "amdgpu.v_add_f32",
            ),
            _binary_rule(
                vector.vector_subf,
                _VEC_F32,
                "amdgpu.v_sub_f32",
                extra_guards=(_register_class("rhs", "amdgpu.vgpr"),),
            ),
            *_commutative_f32_vector_binary_rules(
                vector.vector_mulf,
                "amdgpu.v_mul_f32",
            ),
            _f32_neg_rule(vector.vector_negf, _VEC_F32),
            _f32_abs_rule(vector.vector_absf, _VEC_F32),
            _divf_arcp_one_rule(vector.vector_divf, _VEC_F32),
            _divf_arcp_rule(vector.vector_divf, _VEC_F32),
            *_commutative_f32_vector_binary_rules(
                vector.vector_minnumf,
                "amdgpu.v_min_f32",
            ),
            *_commutative_f32_vector_binary_rules(
                vector.vector_maxnumf,
                "amdgpu.v_max_f32",
            ),
            _ternary_rule(vector.vector_fmaf, _VEC_F32, "amdgpu.v_fma_f32"),
            _unary_rule(vector.vector_exp2f, _VEC_F32, "amdgpu.v_exp_f32"),
            _unary_rule(vector.vector_sqrtf, _VEC_F32, "amdgpu.v_sqrt_f32"),
            _unary_rule(vector.vector_rsqrtf, _VEC_F32, "amdgpu.v_rsq_f32"),
        )
    )
    rules.extend(
        (
            _literal_binary_rule(
                vector.vector_addi,
                "amdgpu.v_add_u32.lit",
                literal_source="lhs",
                nonliteral_source="rhs",
            ),
            _literal_binary_rule(
                vector.vector_addi,
                "amdgpu.v_add_u32.lit",
                literal_source="rhs",
                nonliteral_source="lhs",
            ),
            _binary_rule(vector.vector_addi, _VEC_I32, "amdgpu.v_add_u32"),
        )
    )
    rules.extend(
        (
            _binary_rule(vector.vector_subi, _VEC_I32, "amdgpu.v_sub_u32"),
            _binary_rule(vector.vector_muli, _VEC_I32, "amdgpu.v_mul_lo_u32"),
            _binary_rule(vector.vector_minsi, _VEC_I32, "amdgpu.v_min_i32"),
            _binary_rule(vector.vector_maxsi, _VEC_I32, "amdgpu.v_max_i32"),
            _binary_rule(vector.vector_minui, _VEC_I32, "amdgpu.v_min_u32"),
            _binary_rule(vector.vector_maxui, _VEC_I32, "amdgpu.v_max_u32"),
        )
    )
    for source_op, descriptor_key in (
        (vector.vector_andi, "amdgpu.v_and_b32"),
        (vector.vector_ori, "amdgpu.v_or_b32"),
        (vector.vector_xori, "amdgpu.v_xor_b32"),
    ):
        rules.extend(
            (
                _literal_binary_rule(
                    source_op,
                    f"{descriptor_key}.lit",
                    literal_source="lhs",
                    nonliteral_source="rhs",
                ),
                _literal_binary_rule(
                    source_op,
                    f"{descriptor_key}.lit",
                    literal_source="rhs",
                    nonliteral_source="lhs",
                ),
                _binary_rule(source_op, _VEC_I32, descriptor_key),
            )
        )
    for source_op, descriptor_key in (
        (vector.vector_shli, "amdgpu.v_lshlrev_b32"),
        (vector.vector_shrsi, "amdgpu.v_ashrrev_i32"),
        (vector.vector_shrui, "amdgpu.v_lshrrev_b32"),
    ):
        rules.extend(
            (
                _literal_binary_rule(
                    source_op,
                    f"{descriptor_key}.lit",
                    literal_source="rhs",
                    nonliteral_source="lhs",
                    descriptor_operand="value",
                ),
                _binary_rule(
                    source_op,
                    _VEC_I32,
                    descriptor_key,
                    descriptor_lhs="shift",
                    descriptor_rhs="value",
                    source_lhs="rhs",
                    source_rhs="lhs",
                ),
            )
        )
    for source_op, descriptor_key in (
        (scalar_arithmetic.scalar_addf, "amdgpu.v_add_f32.lit"),
        (scalar_arithmetic.scalar_mulf, "amdgpu.v_mul_f32.lit"),
        (scalar_arithmetic.scalar_minnumf, "amdgpu.v_min_f32.lit"),
        (scalar_arithmetic.scalar_maxnumf, "amdgpu.v_max_f32.lit"),
    ):
        rules.extend(
            (
                _f32_literal_binary_rule(
                    source_op,
                    _F32,
                    descriptor_key,
                    literal_source="lhs",
                    nonliteral_source="rhs",
                    f32_operand=True,
                ),
                _f32_literal_binary_rule(
                    source_op,
                    _F32,
                    descriptor_key,
                    literal_source="rhs",
                    nonliteral_source="lhs",
                    f32_operand=True,
                ),
            )
        )
    rules.append(
        _f32_literal_binary_rule(
            scalar_arithmetic.scalar_subf,
            _F32,
            "amdgpu.v_sub_f32.lit",
            literal_source="lhs",
            nonliteral_source="rhs",
            f32_operand=True,
        )
    )
    rules.extend(
        (
            _cast_rule(
                vector.vector_sitofp,
                _VEC_I32,
                _VEC_F32,
                "amdgpu.v_cvt_f32_i32",
            ),
            _cast_rule(
                vector.vector_uitofp,
                _VEC_I32,
                _VEC_F32,
                "amdgpu.v_cvt_f32_u32",
            ),
            _binary_rule(
                scalar_arithmetic.scalar_addf,
                _F32,
                "amdgpu.v_add_f32",
                f32_operands=True,
            ),
            _binary_rule(
                scalar_arithmetic.scalar_subf,
                _F32,
                "amdgpu.v_sub_f32",
                f32_operands=True,
            ),
            _binary_rule(
                scalar_arithmetic.scalar_mulf,
                _F32,
                "amdgpu.v_mul_f32",
                f32_operands=True,
            ),
            _f32_neg_rule(scalar_arithmetic.scalar_negf, _F32, f32_operand=True),
            _f32_abs_rule(scalar_arithmetic.scalar_absf, _F32, f32_operand=True),
            _divf_arcp_one_rule(
                scalar_arithmetic.scalar_divf,
                _F32,
                f32_operand=True,
            ),
            _divf_arcp_rule(
                scalar_arithmetic.scalar_divf,
                _F32,
                f32_operands=True,
            ),
            _binary_rule(
                scalar_arithmetic.scalar_minnumf,
                _F32,
                "amdgpu.v_min_f32",
                f32_operands=True,
            ),
            _binary_rule(
                scalar_arithmetic.scalar_maxnumf,
                _F32,
                "amdgpu.v_max_f32",
                f32_operands=True,
            ),
            _binary_rule(
                scalar_arithmetic.scalar_minsi,
                _I32,
                "amdgpu.v_min_i32",
            ),
            _binary_rule(
                scalar_arithmetic.scalar_maxsi,
                _I32,
                "amdgpu.v_max_i32",
            ),
            _binary_rule(
                scalar_arithmetic.scalar_minui,
                _I32,
                "amdgpu.v_min_u32",
            ),
            _binary_rule(
                scalar_arithmetic.scalar_maxui,
                _I32,
                "amdgpu.v_max_u32",
            ),
            _ternary_rule(
                scalar_math.scalar_fmaf,
                _F32,
                "amdgpu.v_fma_f32",
                f32_operands=True,
            ),
            _unary_rule(
                scalar_math.scalar_exp2f,
                _F32,
                "amdgpu.v_exp_f32",
                f32_operand=True,
            ),
            _unary_rule(
                scalar_math.scalar_sqrtf,
                _F32,
                "amdgpu.v_sqrt_f32",
                f32_operand=True,
            ),
            _unary_rule(
                scalar_math.scalar_rsqrtf,
                _F32,
                "amdgpu.v_rsq_f32",
                f32_operand=True,
            ),
            _cast_rule(
                scalar_conversion.scalar_extf,
                _F16,
                _F32,
                "amdgpu.v_cvt_f32_f16",
            ),
            _cast_rule(
                scalar_conversion.scalar_fptrunc,
                _F32,
                _F16,
                "amdgpu.v_cvt_f16_f32",
                f32_input=True,
            ),
            _cast_rule(
                scalar_conversion.scalar_sitofp,
                _I32,
                _F32,
                "amdgpu.v_cvt_f32_i32",
            ),
            _cast_rule(
                scalar_conversion.scalar_uitofp,
                _I32,
                _F32,
                "amdgpu.v_cvt_f32_u32",
            ),
            _bitcast_alias_rule(_I32, _F32),
            _bitcast_alias_rule(_F32, _I32),
            _index_madd_power_of_two_rule(
                scale_source="a",
                value_source="b",
                literal_addend=True,
                preserve_value_register=True,
            ),
            _index_madd_power_of_two_rule(
                scale_source="b",
                value_source="a",
                literal_addend=True,
                preserve_value_register=True,
            ),
            _index_madd_power_of_two_rule(
                scale_source="a",
                value_source="b",
                literal_addend=False,
                preserve_value_register=True,
            ),
            _index_madd_power_of_two_rule(
                scale_source="b",
                value_source="a",
                literal_addend=False,
                preserve_value_register=True,
            ),
            _index_madd_power_of_two_rule(
                scale_source="a",
                value_source="b",
                literal_addend=True,
                preserve_value_register=False,
            ),
            _index_madd_power_of_two_rule(
                scale_source="b",
                value_source="a",
                literal_addend=True,
                preserve_value_register=False,
            ),
            _index_madd_power_of_two_rule(
                scale_source="a",
                value_source="b",
                literal_addend=False,
                preserve_value_register=False,
            ),
            _index_madd_power_of_two_rule(
                scale_source="b",
                value_source="a",
                literal_addend=False,
                preserve_value_register=False,
            ),
            _index_madd_literal_rule(),
            _index_madd_rule(),
        )
    )
    return tuple(rules)


AMDGPU_ARITHMETIC_CONTRACT_DIALECT_OPS = {
    "index": ALL_INDEX_OPS,
    "scalar": ALL_SCALAR_OPS,
    "vector": ALL_VECTOR_OPS,
}

AMDGPU_ARITHMETIC_CONTRACT_FRAGMENT = ContractFragment(
    name="amdgpu.arithmetic",
    descriptor_set=_DESCRIPTOR_SET,
    c_source_includes=("loom/target/arch/amdgpu/lower/kinds.h",),
    materializers=(ADDRESS_VGPR_MATERIALIZER, F32_VGPR_MATERIALIZER),
    cases=_rules(),
)
