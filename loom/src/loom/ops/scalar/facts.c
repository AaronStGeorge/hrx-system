// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fact implementations for the scalar dialect.
//
// Each fact inference function computes output facts from operand facts. For
// exact inputs, this IS constant folding. For range inputs, it
// propagates range and divisibility information.

#include "loom/ir/facts.h"

#include <math.h>

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/compare.h"
#include "loom/ops/scalar/ops.h"
#include "loom/util/math.h"

//===----------------------------------------------------------------------===//
// Macros for mechanical fact inference functions
//===----------------------------------------------------------------------===//

#define BINARY_FACTS(name, transfer_fn)                                  \
  iree_status_t name(loom_fact_context_t* context,                       \
                     const loom_module_t* module, const loom_op_t* op,   \
                     const loom_value_facts_t* operand_facts,            \
                     loom_value_facts_t* result_facts) {                 \
    transfer_fn(&operand_facts[0], &operand_facts[1], &result_facts[0]); \
    return iree_ok_status();                                             \
  }

#define UNARY_FACTS(name, transfer_fn)                                 \
  iree_status_t name(loom_fact_context_t* context,                     \
                     const loom_module_t* module, const loom_op_t* op, \
                     const loom_value_facts_t* operand_facts,          \
                     loom_value_facts_t* result_facts) {               \
    transfer_fn(&operand_facts[0], &result_facts[0]);                  \
    return iree_ok_status();                                           \
  }

// Float facts: exact-only constant folding via C library functions.
static void loom_float_facts_unary(const loom_value_facts_t* input,
                                   double (*fn)(double),
                                   loom_value_facts_t* out) {
  if (!loom_value_facts_is_exact(*input) ||
      !loom_value_facts_is_float(*input)) {
    *out = loom_value_facts_unknown();
    return;
  }
  *out = loom_value_facts_exact_f64(fn(loom_value_facts_as_f64(*input)));
}

static void loom_float_facts_binary(const loom_value_facts_t* lhs,
                                    const loom_value_facts_t* rhs,
                                    double (*fn)(double, double),
                                    loom_value_facts_t* out) {
  if (!loom_value_facts_is_exact(*lhs) || !loom_value_facts_is_float(*lhs) ||
      !loom_value_facts_is_exact(*rhs) || !loom_value_facts_is_float(*rhs)) {
    *out = loom_value_facts_unknown();
    return;
  }
  *out = loom_value_facts_exact_f64(
      fn(loom_value_facts_as_f64(*lhs), loom_value_facts_as_f64(*rhs)));
}

#define FLOAT_BINARY_FACTS(name, fn)                                   \
  iree_status_t name(loom_fact_context_t* context,                     \
                     const loom_module_t* module, const loom_op_t* op, \
                     const loom_value_facts_t* operand_facts,          \
                     loom_value_facts_t* result_facts) {               \
    loom_float_facts_binary(&operand_facts[0], &operand_facts[1], fn,  \
                            &result_facts[0]);                         \
    return iree_ok_status();                                           \
  }

#define FLOAT_UNARY_FACTS(name, fn)                                    \
  iree_status_t name(loom_fact_context_t* context,                     \
                     const loom_module_t* module, const loom_op_t* op, \
                     const loom_value_facts_t* operand_facts,          \
                     loom_value_facts_t* result_facts) {               \
    loom_float_facts_unary(&operand_facts[0], fn, &result_facts[0]);   \
    return iree_ok_status();                                           \
  }

// Helper wrappers for C math functions that need adapting.
static double add_f64(double a, double b) { return a + b; }
static double sub_f64(double a, double b) { return a - b; }
static double mul_f64(double a, double b) { return a * b; }
static double div_f64(double a, double b) { return a / b; }
static double negate_f64(double a) { return -a; }
static double rsqrt_f64(double x) { return 1.0 / sqrt(x); }
static double roundeven_f64(double x) { return nearbyint(x); }
static double logistic_f64(double x) { return 1.0 / (1.0 + exp(-x)); }
static double silu_f64(double x) { return x * logistic_f64(x); }
static double softplus_f64(double x) {
  return log1p(exp(-fabs(x))) + fmax(x, 0.0);
}
static double gelu_erf_f64(double x) {
  const double inverse_sqrt2 = 0.70710678118654752440;
  return 0.5 * x * (1.0 + erf(x * inverse_sqrt2));
}
static double gelu_tanh_f64(double x) {
  const double sqrt_2_over_pi = 0.79788456080286535588;
  return 0.5 * x * (1.0 + tanh(sqrt_2_over_pi * (x + 0.044715 * x * x * x)));
}
static double gelu_logistic_f64(double x, double scale) {
  return x * logistic_f64(scale * x);
}
static double minimum_f64(double a, double b) {
  return (isnan(a) || isnan(b)) ? NAN : fmin(a, b);
}
static double maximum_f64(double a, double b) {
  return (isnan(a) || isnan(b)) ? NAN : fmax(a, b);
}

//===----------------------------------------------------------------------===//
// scalar.constant
//===----------------------------------------------------------------------===//

iree_status_t loom_scalar_constant_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_attribute_t attr = loom_op_attrs(op)[0];
  loom_value_id_t result_id = loom_scalar_constant_result(op);
  loom_type_t result_type = loom_module_value_type(module, result_id);
  if (loom_scalar_type_is_float(loom_type_element_type(result_type))) {
    result_facts[0] = loom_value_facts_exact_f64(loom_attr_as_f64(attr));
  } else {
    result_facts[0] = loom_value_facts_exact_i64(loom_attr_as_i64(attr));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Integer arithmetic
//===----------------------------------------------------------------------===//

BINARY_FACTS(loom_scalar_addi_facts, loom_value_facts_addi)
BINARY_FACTS(loom_scalar_subi_facts, loom_value_facts_subi)
BINARY_FACTS(loom_scalar_muli_facts, loom_value_facts_muli)
BINARY_FACTS(loom_scalar_divsi_facts, loom_value_facts_divsi)
BINARY_FACTS(loom_scalar_divui_facts, loom_value_facts_divui)
BINARY_FACTS(loom_scalar_remsi_facts, loom_value_facts_remsi)
BINARY_FACTS(loom_scalar_remui_facts, loom_value_facts_remui)
UNARY_FACTS(loom_scalar_negi_facts, loom_value_facts_negi)
UNARY_FACTS(loom_scalar_absi_facts, loom_value_facts_absi)
BINARY_FACTS(loom_scalar_minsi_facts, loom_value_facts_minsi)
BINARY_FACTS(loom_scalar_maxsi_facts, loom_value_facts_maxsi)
BINARY_FACTS(loom_scalar_minui_facts, loom_value_facts_minui)
BINARY_FACTS(loom_scalar_maxui_facts, loom_value_facts_maxui)

// fmai has 3 operands: a*b + c.
iree_status_t loom_scalar_fmai_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  loom_value_facts_fmai(&operand_facts[0], &operand_facts[1], &operand_facts[2],
                        &result_facts[0]);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Float arithmetic
//===----------------------------------------------------------------------===//

FLOAT_BINARY_FACTS(loom_scalar_addf_facts, add_f64)
FLOAT_BINARY_FACTS(loom_scalar_subf_facts, sub_f64)
FLOAT_BINARY_FACTS(loom_scalar_mulf_facts, mul_f64)
FLOAT_BINARY_FACTS(loom_scalar_divf_facts, div_f64)
FLOAT_BINARY_FACTS(loom_scalar_remf_facts, fmod)
FLOAT_UNARY_FACTS(loom_scalar_negf_facts, negate_f64)
FLOAT_UNARY_FACTS(loom_scalar_absf_facts, fabs)
FLOAT_BINARY_FACTS(loom_scalar_minimumf_facts, minimum_f64)
FLOAT_BINARY_FACTS(loom_scalar_maximumf_facts, maximum_f64)
FLOAT_BINARY_FACTS(loom_scalar_minnumf_facts, fmin)
FLOAT_BINARY_FACTS(loom_scalar_maxnumf_facts, fmax)
FLOAT_BINARY_FACTS(loom_scalar_copysignf_facts, copysign)

//===----------------------------------------------------------------------===//
// Math functions
//===----------------------------------------------------------------------===//

FLOAT_UNARY_FACTS(loom_scalar_expf_facts, exp)
FLOAT_UNARY_FACTS(loom_scalar_exp2f_facts, exp2)
FLOAT_UNARY_FACTS(loom_scalar_expm1f_facts, expm1)
FLOAT_UNARY_FACTS(loom_scalar_logf_facts, log)
FLOAT_UNARY_FACTS(loom_scalar_log2f_facts, log2)
FLOAT_UNARY_FACTS(loom_scalar_log10f_facts, log10)
FLOAT_UNARY_FACTS(loom_scalar_log1pf_facts, log1p)
FLOAT_BINARY_FACTS(loom_scalar_powf_facts, pow)
FLOAT_UNARY_FACTS(loom_scalar_sqrtf_facts, sqrt)
FLOAT_UNARY_FACTS(loom_scalar_rsqrtf_facts, rsqrt_f64)
FLOAT_UNARY_FACTS(loom_scalar_cbrtf_facts, cbrt)
FLOAT_UNARY_FACTS(loom_scalar_sinf_facts, sin)
FLOAT_UNARY_FACTS(loom_scalar_cosf_facts, cos)
FLOAT_UNARY_FACTS(loom_scalar_tanf_facts, tan)
FLOAT_UNARY_FACTS(loom_scalar_asinf_facts, asin)
FLOAT_UNARY_FACTS(loom_scalar_acosf_facts, acos)
FLOAT_UNARY_FACTS(loom_scalar_atanf_facts, atan)
FLOAT_BINARY_FACTS(loom_scalar_atan2f_facts, atan2)
FLOAT_UNARY_FACTS(loom_scalar_sinhf_facts, sinh)
FLOAT_UNARY_FACTS(loom_scalar_coshf_facts, cosh)
FLOAT_UNARY_FACTS(loom_scalar_tanhf_facts, tanh)
FLOAT_UNARY_FACTS(loom_scalar_asinhf_facts, asinh)
FLOAT_UNARY_FACTS(loom_scalar_acoshf_facts, acosh)
FLOAT_UNARY_FACTS(loom_scalar_atanhf_facts, atanh)
FLOAT_UNARY_FACTS(loom_scalar_erff_facts, erf)
FLOAT_UNARY_FACTS(loom_scalar_erfcf_facts, erfc)
FLOAT_UNARY_FACTS(loom_scalar_logisticf_facts, logistic_f64)
FLOAT_UNARY_FACTS(loom_scalar_siluf_facts, silu_f64)
FLOAT_UNARY_FACTS(loom_scalar_softplusf_facts, softplus_f64)

iree_status_t loom_scalar_geluf_facts(loom_fact_context_t* context,
                                      const loom_module_t* module,
                                      const loom_op_t* op,
                                      const loom_value_facts_t* operand_facts,
                                      loom_value_facts_t* result_facts) {
  if (!loom_value_facts_is_exact(operand_facts[0]) ||
      !loom_value_facts_is_float(operand_facts[0])) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }

  double input = loom_value_facts_as_f64(operand_facts[0]);
  switch (loom_scalar_geluf_variant(op)) {
    case LOOM_SCALAR_GELUF_VARIANT_ERF:
      result_facts[0] = loom_value_facts_exact_f64(gelu_erf_f64(input));
      return iree_ok_status();
    case LOOM_SCALAR_GELUF_VARIANT_TANH:
      result_facts[0] = loom_value_facts_exact_f64(gelu_tanh_f64(input));
      return iree_ok_status();
    case LOOM_SCALAR_GELUF_VARIANT_LOGISTIC: {
      loom_attribute_t scale_attr = loom_op_attrs(op)[1];
      if (loom_attr_is_absent(scale_attr)) {
        result_facts[0] = loom_value_facts_unknown();
        return iree_ok_status();
      }
      result_facts[0] = loom_value_facts_exact_f64(
          gelu_logistic_f64(input, loom_attr_as_f64(scale_attr)));
      return iree_ok_status();
    }
    case LOOM_SCALAR_GELUF_VARIANT_COUNT_:
      break;
  }
  result_facts[0] = loom_value_facts_unknown();
  return iree_ok_status();
}

// fmaf has 3 operands.
iree_status_t loom_scalar_fmaf_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  for (int i = 0; i < 3; ++i) {
    if (!loom_value_facts_is_exact(operand_facts[i]) ||
        !loom_value_facts_is_float(operand_facts[i])) {
      result_facts[0] = loom_value_facts_unknown();
      return iree_ok_status();
    }
  }
  result_facts[0] = loom_value_facts_exact_f64(
      fma(loom_value_facts_as_f64(operand_facts[0]),
          loom_value_facts_as_f64(operand_facts[1]),
          loom_value_facts_as_f64(operand_facts[2])));
  return iree_ok_status();
}

FLOAT_UNARY_FACTS(loom_scalar_ceilf_facts, ceil)
FLOAT_UNARY_FACTS(loom_scalar_floorf_facts, floor)
FLOAT_UNARY_FACTS(loom_scalar_roundf_facts, round)
FLOAT_UNARY_FACTS(loom_scalar_roundevenf_facts, roundeven_f64)
FLOAT_UNARY_FACTS(loom_scalar_truncf_facts, trunc)

//===----------------------------------------------------------------------===//
// Bitwise
//===----------------------------------------------------------------------===//

BINARY_FACTS(loom_scalar_andi_facts, loom_value_facts_andi)
BINARY_FACTS(loom_scalar_ori_facts, loom_value_facts_ori)
BINARY_FACTS(loom_scalar_xori_facts, loom_value_facts_xori)
BINARY_FACTS(loom_scalar_shli_facts, loom_value_facts_shli)
BINARY_FACTS(loom_scalar_shrsi_facts, loom_value_facts_shrsi)
BINARY_FACTS(loom_scalar_shrui_facts, loom_value_facts_shrui)

// Rotates: no transfer function yet — exact-only fact inference.
iree_status_t loom_scalar_rotli_facts(loom_fact_context_t* context,
                                      const loom_module_t* module,
                                      const loom_op_t* op,
                                      const loom_value_facts_t* operand_facts,
                                      loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_unknown();
  return iree_ok_status();
}
iree_status_t loom_scalar_rotri_facts(loom_fact_context_t* context,
                                      const loom_module_t* module,
                                      const loom_op_t* op,
                                      const loom_value_facts_t* operand_facts,
                                      loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_unknown();
  return iree_ok_status();
}

// Bit counting: exact-only over the declared integer width.
#define BIT_COUNT_FACTS(name, result_accessor, fn)                      \
  iree_status_t name(loom_fact_context_t* context,                      \
                     const loom_module_t* module, const loom_op_t* op,  \
                     const loom_value_facts_t* operand_facts,           \
                     loom_value_facts_t* result_facts) {                \
    if (!loom_value_facts_is_exact(operand_facts[0])) {                 \
      result_facts[0] = loom_value_facts_unknown();                     \
      return iree_ok_status();                                          \
    }                                                                   \
    loom_type_t result_type =                                           \
        loom_module_value_type(module, result_accessor(op));            \
    int32_t bitwidth =                                                  \
        loom_scalar_type_bitwidth(loom_type_element_type(result_type)); \
    if (bitwidth <= 0) {                                                \
      result_facts[0] = loom_value_facts_unknown();                     \
      return iree_ok_status();                                          \
    }                                                                   \
    result_facts[0] = loom_value_facts_exact_i64(                       \
        fn((uint64_t)operand_facts[0].range_lo, bitwidth));             \
    return iree_ok_status();                                            \
  }

BIT_COUNT_FACTS(loom_scalar_ctlzi_facts, loom_scalar_ctlzi_result,
                loom_count_leading_zeros_u64_width)
BIT_COUNT_FACTS(loom_scalar_cttzi_facts, loom_scalar_cttzi_result,
                loom_count_trailing_zeros_u64_width)
BIT_COUNT_FACTS(loom_scalar_ctpopi_facts, loom_scalar_ctpopi_result,
                loom_count_ones_u64_width)

#undef BIT_COUNT_FACTS

//===----------------------------------------------------------------------===//
// Comparison
//===----------------------------------------------------------------------===//

iree_status_t loom_scalar_cmpi_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  bool result = false;
  if ((loom_scalar_cmpi_lhs(op) == loom_scalar_cmpi_rhs(op) &&
       loom_scalar_cmpi_same_value_result(loom_scalar_cmpi_predicate(op),
                                          &result)) ||
      loom_scalar_cmpi_result_from_facts(loom_scalar_cmpi_predicate(op),
                                         &operand_facts[0], &operand_facts[1],
                                         &result)) {
    result_facts[0] = loom_value_facts_exact_i64(result ? 1 : 0);
    return iree_ok_status();
  }
  result_facts[0] = loom_value_facts_make(0, 1, 1);
  return iree_ok_status();
}

iree_status_t loom_scalar_cmpf_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  bool result = false;
  if (loom_scalar_cmpf_lhs(op) == loom_scalar_cmpf_rhs(op) &&
      (loom_scalar_cmpf_fastmath(op) & LOOM_SCALAR_FASTMATHFLAGS_NNAN) != 0 &&
      loom_scalar_cmpf_same_value_result(loom_scalar_cmpf_predicate(op),
                                         &result)) {
    result_facts[0] = loom_value_facts_exact_i64(result ? 1 : 0);
    return iree_ok_status();
  }
  if (loom_value_facts_is_exact(operand_facts[0]) &&
      loom_value_facts_is_float(operand_facts[0]) &&
      loom_value_facts_is_exact(operand_facts[1]) &&
      loom_value_facts_is_float(operand_facts[1]) &&
      loom_scalar_cmpf_exact_result(loom_scalar_cmpf_predicate(op),
                                    loom_value_facts_as_f64(operand_facts[0]),
                                    loom_value_facts_as_f64(operand_facts[1]),
                                    &result)) {
    result_facts[0] = loom_value_facts_exact_i64(result ? 1 : 0);
    return iree_ok_status();
  }
  result_facts[0] = loom_value_facts_make(0, 1, 1);
  return iree_ok_status();
}

// Float predicates: exact-only.
static bool isnan_f64(double v) { return v != v; }
static bool isinf_f64(double v) { return isinf(v) != 0; }
static bool isfinite_f64(double v) { return isfinite(v) != 0; }

#define FLOAT_PREDICATE_FACTS(name, fn)                                \
  iree_status_t name(loom_fact_context_t* context,                     \
                     const loom_module_t* module, const loom_op_t* op, \
                     const loom_value_facts_t* operand_facts,          \
                     loom_value_facts_t* result_facts) {               \
    if (!loom_value_facts_is_exact(operand_facts[0]) ||                \
        !loom_value_facts_is_float(operand_facts[0])) {                \
      result_facts[0] = loom_value_facts_make(0, 1, 1);                \
      return iree_ok_status();                                         \
    }                                                                  \
    double v = loom_value_facts_as_f64(operand_facts[0]);              \
    result_facts[0] = loom_value_facts_exact_i64(fn(v) ? 1 : 0);       \
    return iree_ok_status();                                           \
  }

FLOAT_PREDICATE_FACTS(loom_scalar_isnanf_facts, isnan_f64)
FLOAT_PREDICATE_FACTS(loom_scalar_isinff_facts, isinf_f64)
FLOAT_PREDICATE_FACTS(loom_scalar_isfinitef_facts, isfinite_f64)

iree_status_t loom_scalar_signf_facts(loom_fact_context_t* context,
                                      const loom_module_t* module,
                                      const loom_op_t* op,
                                      const loom_value_facts_t* operand_facts,
                                      loom_value_facts_t* result_facts) {
  if (!loom_value_facts_is_exact(operand_facts[0]) ||
      !loom_value_facts_is_float(operand_facts[0])) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  double v = loom_value_facts_as_f64(operand_facts[0]);
  result_facts[0] = loom_value_facts_exact_f64((v > 0.0)   ? 1.0
                                               : (v < 0.0) ? -1.0
                                                           : 0.0);
  return iree_ok_status();
}

iree_status_t loom_scalar_signi_facts(loom_fact_context_t* context,
                                      const loom_module_t* module,
                                      const loom_op_t* op,
                                      const loom_value_facts_t* operand_facts,
                                      loom_value_facts_t* result_facts) {
  if (!loom_value_facts_is_exact(operand_facts[0])) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  int64_t v = operand_facts[0].range_lo;
  result_facts[0] = loom_value_facts_exact_i64((v > 0) ? 1 : (v < 0) ? -1 : 0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Conversion
//===----------------------------------------------------------------------===//

iree_status_t loom_scalar_sitofp_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  if (!loom_value_facts_is_exact(operand_facts[0])) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  result_facts[0] =
      loom_value_facts_exact_f64((double)operand_facts[0].range_lo);
  return iree_ok_status();
}

iree_status_t loom_scalar_uitofp_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  if (!loom_value_facts_is_exact(operand_facts[0])) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  result_facts[0] =
      loom_value_facts_exact_f64((double)(uint64_t)operand_facts[0].range_lo);
  return iree_ok_status();
}

iree_status_t loom_scalar_fptosi_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  if (!loom_value_facts_is_exact(operand_facts[0]) ||
      !loom_value_facts_is_float(operand_facts[0])) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  result_facts[0] = loom_value_facts_exact_i64(
      (int64_t)loom_value_facts_as_f64(operand_facts[0]));
  return iree_ok_status();
}

iree_status_t loom_scalar_fptoui_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  if (!loom_value_facts_is_exact(operand_facts[0]) ||
      !loom_value_facts_is_float(operand_facts[0])) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  result_facts[0] = loom_value_facts_exact_i64(
      (int64_t)(uint64_t)loom_value_facts_as_f64(operand_facts[0]));
  return iree_ok_status();
}

iree_status_t loom_scalar_extf_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  result_facts[0] = operand_facts[0];
  return iree_ok_status();
}

iree_status_t loom_scalar_fptrunc_facts(loom_fact_context_t* context,
                                        const loom_module_t* module,
                                        const loom_op_t* op,
                                        const loom_value_facts_t* operand_facts,
                                        loom_value_facts_t* result_facts) {
  if (!loom_value_facts_is_exact(operand_facts[0]) ||
      !loom_value_facts_is_float(operand_facts[0])) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  double v = loom_value_facts_as_f64(operand_facts[0]);
  result_facts[0] = loom_value_facts_exact_f64((double)(float)v);
  return iree_ok_status();
}

iree_status_t loom_scalar_extsi_facts(loom_fact_context_t* context,
                                      const loom_module_t* module,
                                      const loom_op_t* op,
                                      const loom_value_facts_t* operand_facts,
                                      loom_value_facts_t* result_facts) {
  result_facts[0] = operand_facts[0];
  return iree_ok_status();
}

iree_status_t loom_scalar_extui_facts(loom_fact_context_t* context,
                                      const loom_module_t* module,
                                      const loom_op_t* op,
                                      const loom_value_facts_t* operand_facts,
                                      loom_value_facts_t* result_facts) {
  if (loom_value_facts_is_non_negative(operand_facts[0])) {
    result_facts[0] = operand_facts[0];
  } else {
    result_facts[0] = loom_value_facts_unknown();
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_trunci_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  if (loom_value_facts_is_exact(operand_facts[0])) {
    result_facts[0] = operand_facts[0];
  } else {
    result_facts[0] = loom_value_facts_unknown();
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_bitcast_facts(loom_fact_context_t* context,
                                        const loom_module_t* module,
                                        const loom_op_t* op,
                                        const loom_value_facts_t* operand_facts,
                                        loom_value_facts_t* result_facts) {
  if (loom_value_facts_is_exact(operand_facts[0])) {
    result_facts[0] = operand_facts[0];
  } else {
    result_facts[0] = loom_value_facts_unknown();
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// scalar.assume
//===----------------------------------------------------------------------===//

iree_status_t loom_scalar_assume_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  uint16_t fact_count = op->operand_count < op->result_count ? op->operand_count
                                                             : op->result_count;
  for (uint16_t i = 0; i < fact_count; ++i) {
    result_facts[i] = operand_facts[i];
  }
  for (uint16_t i = fact_count; i < op->result_count; ++i) {
    result_facts[i] = loom_value_facts_unknown();
  }
  loom_attribute_t pred_attr = loom_op_attrs(op)[0];
  const loom_predicate_t* predicates = pred_attr.predicate_list;
  uint16_t predicate_count = pred_attr.count;
  for (uint16_t p = 0; p < predicate_count; ++p) {
    const loom_predicate_t* pred = &predicates[p];
    if (pred->arg_tags[0] != LOOM_PRED_ARG_VALUE) continue;
    loom_value_slice_t values = loom_scalar_assume_values(op);
    loom_value_id_t target_id = (loom_value_id_t)pred->args[0];
    uint16_t target = 0;
    bool found = false;
    for (uint16_t i = 0; i < values.count; ++i) {
      if (values.values[i] == target_id) {
        target = i;
        found = true;
        break;
      }
    }
    if (!found) continue;
    if (target < fact_count) {
      loom_value_facts_apply_predicate(&result_facts[target], pred);
    }
  }
  return iree_ok_status();
}
