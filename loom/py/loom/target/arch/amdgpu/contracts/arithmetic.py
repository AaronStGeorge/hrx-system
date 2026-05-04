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
from loom.target.arch.amdgpu.contracts.materializers import ADDRESS_VGPR_MATERIALIZER
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
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_DESCRIPTOR_KEYS = (
    "amdgpu.v_add_f32",
    "amdgpu.v_sub_f32",
    "amdgpu.v_mul_f32",
    "amdgpu.v_min_f32",
    "amdgpu.v_max_f32",
    "amdgpu.v_fma_f32",
    "amdgpu.v_cvt_f32_i32",
    "amdgpu.v_cvt_f32_u32",
    "amdgpu.v_add_u32",
    "amdgpu.v_add_u32.lit",
    "amdgpu.v_sub_u32",
    "amdgpu.v_mul_lo_u32",
    "amdgpu.v_and_b32",
    "amdgpu.v_and_b32.lit",
    "amdgpu.v_or_b32",
    "amdgpu.v_or_b32.lit",
    "amdgpu.v_xor_b32",
    "amdgpu.v_xor_b32.lit",
    "amdgpu.v_lshlrev_b32",
    "amdgpu.v_lshlrev_b32.lit",
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
_F32 = Scalar("f32")
_INDEX = Scalar("index")

_VEC_I32_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="type",
    subject_name="vector<i32>",
    expected_text=(
        "AMDGPU arithmetic lowering requires a rank-1 static i32 vector with "
        "1 to 8 lanes"
    ),
)
_VEC_F32_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="type",
    subject_name="vector<f32>",
    expected_text=(
        "AMDGPU arithmetic lowering requires a rank-1 static f32 vector with "
        "1 to 8 lanes"
    ),
)
_I32_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="type",
    subject_name="i32",
    expected_text="AMDGPU arithmetic lowering requires an i32 scalar",
)
_F32_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="type",
    subject_name="f32",
    expected_text="AMDGPU arithmetic lowering requires an f32 scalar",
)
_INDEX_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="type",
    subject_name="index",
    expected_text="AMDGPU index lowering requires index scalar values",
)
_ADDRESS_U32_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="address-width",
    subject_name="u32",
    expected_text=(
        "AMDGPU index lowering requires address results proven non-negative and 32-bit"
    ),
)
_ADDRESS_EXACT_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="address-literal",
    subject_name="i64",
    expected_text="AMDGPU literal address operands require exact integer value facts",
)
_ADDRESS_VGPR_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="materializer",
    subject_name="address-vgpr",
    expected_text=(
        "AMDGPU index lowering requires address values that can materialize as "
        "VGPR operands"
    ),
)
_RESULT_VGPR_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="register-class",
    subject_name="vgpr",
    expected_text=(
        "AMDGPU arithmetic lowering requires a VGPR result for vector "
        "descriptor emission"
    ),
)
_LITERAL_EXACT_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="literal",
    subject_name="i64",
    expected_text=(
        "AMDGPU literal arithmetic operands require exact integer value facts"
    ),
)
_LITERAL_I32_BITS_DIAGNOSTIC = GuardDiagnostic(
    subject_kind="literal-bits",
    subject_name="i32",
    expected_text=(
        "AMDGPU literal arithmetic operands must fit in a signed i32 bit pattern"
    ),
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
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_guards((source_lhs, source_rhs, "result"), type_pattern),
            Guard.descriptor_available(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    descriptor_lhs: ValueRef.operand(source_lhs),
                    descriptor_rhs: ValueRef.operand(source_rhs),
                },
                results={"dst": ValueRef.result("result")},
                form=_emit_form(type_pattern),
            ),
        ),
    )


def _ternary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
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
                    "a": ValueRef.operand("a"),
                    "b": ValueRef.operand("b"),
                    "c": ValueRef.operand("c"),
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
                operands={"input": ValueRef.operand("input")},
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


def _rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    rules.extend(
        (
            _binary_rule(vector.vector_addf, _VEC_F32, "amdgpu.v_add_f32"),
            _binary_rule(vector.vector_subf, _VEC_F32, "amdgpu.v_sub_f32"),
            _binary_rule(vector.vector_mulf, _VEC_F32, "amdgpu.v_mul_f32"),
            _binary_rule(vector.vector_minnumf, _VEC_F32, "amdgpu.v_min_f32"),
            _binary_rule(vector.vector_maxnumf, _VEC_F32, "amdgpu.v_max_f32"),
            _ternary_rule(vector.vector_fmaf, _VEC_F32, "amdgpu.v_fma_f32"),
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
            ),
            _binary_rule(
                scalar_arithmetic.scalar_subf,
                _F32,
                "amdgpu.v_sub_f32",
            ),
            _binary_rule(
                scalar_arithmetic.scalar_mulf,
                _F32,
                "amdgpu.v_mul_f32",
            ),
            _binary_rule(
                scalar_arithmetic.scalar_minnumf,
                _F32,
                "amdgpu.v_min_f32",
            ),
            _binary_rule(
                scalar_arithmetic.scalar_maxnumf,
                _F32,
                "amdgpu.v_max_f32",
            ),
            _ternary_rule(scalar_math.scalar_fmaf, _F32, "amdgpu.v_fma_f32"),
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
    materializers=(ADDRESS_VGPR_MATERIALIZER,),
    cases=_rules(),
)
