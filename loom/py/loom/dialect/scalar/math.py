# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar math: transcendental functions, rounding, special functions."""

from loom.assembly import (
    COLON,
    COMMA,
    Flags,
    Ref,
    TypeOf,
)
from loom.dialect.scalar import FastMathFlags, scalar_ops
from loom.dsl import (
    FLOAT,
    IDEMPOTENT,
    PURE,
    AttrDef,
    Op,
    Operand,
    Result,
    SameType,
    binary_op,
    unary_op,
)

__all__ = ["ALL_MATH_OPS"]

# All math ops are FLOAT and PURE with optional FastMathFlags unless noted.
_FM = ("fastmath", FastMathFlags)

# ============================================================================
# Exponential / logarithm
# ============================================================================

scalar_expf = unary_op(
    "scalar.expf",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Exponential: e^x.",
    flags=_FM,
    fold="loom_scalar_expf_fold",
    examples=["%result = scalar.expf %input : f32"],
)
scalar_exp2f = unary_op(
    "scalar.exp2f",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Base-2 exponential: 2^x.",
    flags=_FM,
    fold="loom_scalar_exp2f_fold",
    examples=["%result = scalar.exp2f %input : f32"],
)
scalar_expm1f = unary_op(
    "scalar.expm1f",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Exponential minus one: e^x - 1 (numerically stable near 0).",
    flags=_FM,
    fold="loom_scalar_expm1f_fold",
    examples=["%result = scalar.expm1f %input : f32"],
)
scalar_logf = unary_op(
    "scalar.logf",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Natural logarithm: ln(x).",
    flags=_FM,
    fold="loom_scalar_logf_fold",
    examples=["%result = scalar.logf %input : f32"],
)
scalar_log2f = unary_op(
    "scalar.log2f",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Base-2 logarithm.",
    flags=_FM,
    fold="loom_scalar_log2f_fold",
    examples=["%result = scalar.log2f %input : f32"],
)
scalar_log10f = unary_op(
    "scalar.log10f",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Base-10 logarithm.",
    flags=_FM,
    fold="loom_scalar_log10f_fold",
    examples=["%result = scalar.log10f %input : f32"],
)
scalar_log1pf = unary_op(
    "scalar.log1pf",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Natural logarithm of 1+x: ln(1+x) (numerically stable near 0).",
    flags=_FM,
    fold="loom_scalar_log1pf_fold",
    examples=["%result = scalar.log1pf %input : f32"],
)

# ============================================================================
# Power / root
# ============================================================================

scalar_powf = binary_op(
    "scalar.powf",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Power: x^y.",
    flags=_FM,
    fold="loom_scalar_powf_fold",
    examples=["%result = scalar.powf %lhs, %rhs : f32"],
)
scalar_sqrtf = unary_op(
    "scalar.sqrtf",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Square root.",
    flags=_FM,
    fold="loom_scalar_sqrtf_fold",
    examples=["%result = scalar.sqrtf %input : f32"],
)
scalar_rsqrtf = unary_op(
    "scalar.rsqrtf",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Reciprocal square root: 1/sqrt(x).",
    flags=_FM,
    fold="loom_scalar_rsqrtf_fold",
    examples=["%result = scalar.rsqrtf %input : f32"],
)
scalar_cbrtf = unary_op(
    "scalar.cbrtf",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Cube root.",
    flags=_FM,
    fold="loom_scalar_cbrtf_fold",
    examples=["%result = scalar.cbrtf %input : f32"],
)

# ============================================================================
# Trigonometric
# ============================================================================

scalar_sinf = unary_op(
    "scalar.sinf",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Sine.",
    flags=_FM,
    fold="loom_scalar_sinf_fold",
    examples=["%result = scalar.sinf %input : f32"],
)
scalar_cosf = unary_op(
    "scalar.cosf",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Cosine.",
    flags=_FM,
    fold="loom_scalar_cosf_fold",
    examples=["%result = scalar.cosf %input : f32"],
)
scalar_tanf = unary_op(
    "scalar.tanf",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Tangent.",
    flags=_FM,
    fold="loom_scalar_tanf_fold",
    examples=["%result = scalar.tanf %input : f32"],
)
scalar_asinf = unary_op(
    "scalar.asinf",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Arcsine.",
    flags=_FM,
    fold="loom_scalar_asinf_fold",
    examples=["%result = scalar.asinf %input : f32"],
)
scalar_acosf = unary_op(
    "scalar.acosf",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Arccosine.",
    flags=_FM,
    fold="loom_scalar_acosf_fold",
    examples=["%result = scalar.acosf %input : f32"],
)
scalar_atanf = unary_op(
    "scalar.atanf",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Arctangent.",
    flags=_FM,
    fold="loom_scalar_atanf_fold",
    examples=["%result = scalar.atanf %input : f32"],
)
scalar_atan2f = binary_op(
    "scalar.atan2f",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Two-argument arctangent: atan2(y, x).",
    flags=_FM,
    fold="loom_scalar_atan2f_fold",
    examples=["%result = scalar.atan2f %lhs, %rhs : f32"],
)

# ============================================================================
# Hyperbolic
# ============================================================================

scalar_sinhf = unary_op(
    "scalar.sinhf",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Hyperbolic sine.",
    flags=_FM,
    fold="loom_scalar_sinhf_fold",
    examples=["%result = scalar.sinhf %input : f32"],
)
scalar_coshf = unary_op(
    "scalar.coshf",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Hyperbolic cosine.",
    flags=_FM,
    fold="loom_scalar_coshf_fold",
    examples=["%result = scalar.coshf %input : f32"],
)
scalar_tanhf = unary_op(
    "scalar.tanhf",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Hyperbolic tangent.",
    flags=_FM,
    fold="loom_scalar_tanhf_fold",
    examples=["%result = scalar.tanhf %input : f32"],
)
scalar_asinhf = unary_op(
    "scalar.asinhf",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Inverse hyperbolic sine.",
    flags=_FM,
    fold="loom_scalar_asinhf_fold",
    examples=["%result = scalar.asinhf %input : f32"],
)
scalar_acoshf = unary_op(
    "scalar.acoshf",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Inverse hyperbolic cosine.",
    flags=_FM,
    fold="loom_scalar_acoshf_fold",
    examples=["%result = scalar.acoshf %input : f32"],
)
scalar_atanhf = unary_op(
    "scalar.atanhf",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Inverse hyperbolic tangent.",
    flags=_FM,
    fold="loom_scalar_atanhf_fold",
    examples=["%result = scalar.atanhf %input : f32"],
)

# ============================================================================
# Special functions
# ============================================================================

scalar_erff = unary_op(
    "scalar.erff",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Error function (used in GeLU activation).",
    flags=_FM,
    fold="loom_scalar_erff_fold",
    examples=["%result = scalar.erff %input : f32"],
)
scalar_erfcf = unary_op(
    "scalar.erfcf",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Complementary error function: 1 - erf(x).",
    flags=_FM,
    fold="loom_scalar_erfcf_fold",
    examples=["%result = scalar.erfcf %input : f32"],
)

scalar_fmaf = Op(
    "scalar.fmaf",
    group=scalar_ops,
    doc="Fused multiply-add: a*b + c with single rounding.",
    operands=[
        Operand("a", FLOAT),
        Operand("b", FLOAT),
        Operand("c", FLOAT),
    ],
    results=[Result("result", FLOAT)],
    attrs=[AttrDef("fastmath", "flags", optional=True, enum_def=FastMathFlags)],
    constraints=[SameType("a", "b", "c", "result")],
    traits=[PURE],
    format=[
        Flags("fastmath"),
        Ref("a"),
        COMMA,
        Ref("b"),
        COMMA,
        Ref("c"),
        COLON,
        TypeOf("result"),
    ],
    fold="loom_scalar_fmaf_fold",
    examples=["%result = scalar.fmaf %a, %b, %c : f32"],
)

# ============================================================================
# Rounding
# ============================================================================

scalar_ceilf = unary_op(
    "scalar.ceilf",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Round toward positive infinity.",
    traits=[IDEMPOTENT],
    flags=_FM,
    fold="loom_scalar_ceilf_fold",
    examples=["%result = scalar.ceilf %input : f32"],
)
scalar_floorf = unary_op(
    "scalar.floorf",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Round toward negative infinity.",
    traits=[IDEMPOTENT],
    flags=_FM,
    fold="loom_scalar_floorf_fold",
    examples=["%result = scalar.floorf %input : f32"],
)
scalar_roundf = unary_op(
    "scalar.roundf",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Round to nearest, ties away from zero.",
    traits=[IDEMPOTENT],
    flags=_FM,
    fold="loom_scalar_roundf_fold",
    examples=["%result = scalar.roundf %input : f32"],
)
scalar_roundevenf = unary_op(
    "scalar.roundevenf",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Round to nearest, ties to even (IEEE 754 default rounding).",
    traits=[IDEMPOTENT],
    flags=_FM,
    fold="loom_scalar_roundevenf_fold",
    examples=["%result = scalar.roundevenf %input : f32"],
)
scalar_truncf = unary_op(
    "scalar.truncf",
    group=scalar_ops,
    type_constraint=FLOAT,
    doc="Round toward zero (C trunc).",
    traits=[IDEMPOTENT],
    flags=_FM,
    fold="loom_scalar_truncf_fold",
    examples=["%result = scalar.truncf %input : f32"],
)

# ============================================================================
# Registry
# ============================================================================

ALL_MATH_OPS: tuple[Op, ...] = (
    scalar_expf,
    scalar_exp2f,
    scalar_expm1f,
    scalar_logf,
    scalar_log2f,
    scalar_log10f,
    scalar_log1pf,
    scalar_powf,
    scalar_sqrtf,
    scalar_rsqrtf,
    scalar_cbrtf,
    scalar_sinf,
    scalar_cosf,
    scalar_tanf,
    scalar_asinf,
    scalar_acosf,
    scalar_atanf,
    scalar_atan2f,
    scalar_sinhf,
    scalar_coshf,
    scalar_tanhf,
    scalar_asinhf,
    scalar_acoshf,
    scalar_atanhf,
    scalar_erff,
    scalar_erfcf,
    scalar_fmaf,
    scalar_ceilf,
    scalar_floorf,
    scalar_roundf,
    scalar_roundevenf,
    scalar_truncf,
)
