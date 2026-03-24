// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fold implementations for the scalar dialect.
//
// Each fold function computes output facts from operand facts. For
// exact inputs, this IS constant folding. For range inputs, it
// propagates range and divisibility information.

#include <math.h>

#include "iree/base/internal/math.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"

//===----------------------------------------------------------------------===//
// Macros for mechanical fold functions
//===----------------------------------------------------------------------===//

#define BINARY_FOLD(name, transfer_fn)                                   \
  void name(const loom_module_t* module, const loom_op_t* op,            \
            const loom_value_facts_t* operand_facts,                     \
            loom_value_facts_t* result_facts) {                          \
    transfer_fn(&operand_facts[0], &operand_facts[1], &result_facts[0]); \
  }

#define UNARY_FOLD(name, transfer_fn)                         \
  void name(const loom_module_t* module, const loom_op_t* op, \
            const loom_value_facts_t* operand_facts,          \
            loom_value_facts_t* result_facts) {               \
    transfer_fn(&operand_facts[0], &result_facts[0]);         \
  }

// Float folds: exact-only constant folding via C library functions.
static void loom_float_fold_unary(const loom_value_facts_t* input,
                                  double (*fn)(double),
                                  loom_value_facts_t* out) {
  if (!loom_value_facts_is_exact(*input) ||
      !loom_value_facts_is_float(*input)) {
    *out = loom_value_facts_unknown();
    return;
  }
  *out = loom_value_facts_exact_f64(fn(loom_value_facts_as_f64(*input)));
}

static void loom_float_fold_binary(const loom_value_facts_t* lhs,
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

#define FLOAT_BINARY_FOLD(name, fn)                                  \
  void name(const loom_module_t* module, const loom_op_t* op,        \
            const loom_value_facts_t* operand_facts,                 \
            loom_value_facts_t* result_facts) {                      \
    loom_float_fold_binary(&operand_facts[0], &operand_facts[1], fn, \
                           &result_facts[0]);                        \
  }

#define FLOAT_UNARY_FOLD(name, fn)                                  \
  void name(const loom_module_t* module, const loom_op_t* op,       \
            const loom_value_facts_t* operand_facts,                \
            loom_value_facts_t* result_facts) {                     \
    loom_float_fold_unary(&operand_facts[0], fn, &result_facts[0]); \
  }

// Helper wrappers for C math functions that need adapting.
static double add_f64(double a, double b) { return a + b; }
static double sub_f64(double a, double b) { return a - b; }
static double mul_f64(double a, double b) { return a * b; }
static double div_f64(double a, double b) { return a / b; }
static double negate_f64(double a) { return -a; }
static double rsqrt_f64(double x) { return 1.0 / sqrt(x); }
static double roundeven_f64(double x) { return nearbyint(x); }

//===----------------------------------------------------------------------===//
// scalar.constant
//===----------------------------------------------------------------------===//

void loom_scalar_constant_fold(const loom_module_t* module, const loom_op_t* op,
                               const loom_value_facts_t* operand_facts,
                               loom_value_facts_t* result_facts) {
  loom_attribute_t attr = loom_op_attrs(op)[0];
  loom_value_id_t result_id = loom_scalar_constant_result(op);
  loom_type_t result_type = loom_module_value_type(module, result_id);
  if (loom_scalar_type_is_float(loom_type_element_type(result_type))) {
    result_facts[0] = loom_value_facts_exact_f64(loom_attr_as_f64(attr));
  } else {
    result_facts[0] = loom_value_facts_exact_i64(loom_attr_as_i64(attr));
  }
}

//===----------------------------------------------------------------------===//
// Integer arithmetic
//===----------------------------------------------------------------------===//

BINARY_FOLD(loom_scalar_addi_fold, loom_value_facts_addi)
BINARY_FOLD(loom_scalar_subi_fold, loom_value_facts_subi)
BINARY_FOLD(loom_scalar_muli_fold, loom_value_facts_muli)
BINARY_FOLD(loom_scalar_divsi_fold, loom_value_facts_divsi)
BINARY_FOLD(loom_scalar_divui_fold, loom_value_facts_divui)
BINARY_FOLD(loom_scalar_remsi_fold, loom_value_facts_remsi)
BINARY_FOLD(loom_scalar_remui_fold, loom_value_facts_remui)
UNARY_FOLD(loom_scalar_negi_fold, loom_value_facts_negi)
UNARY_FOLD(loom_scalar_absi_fold, loom_value_facts_absi)
BINARY_FOLD(loom_scalar_minsi_fold, loom_value_facts_minsi)
BINARY_FOLD(loom_scalar_maxsi_fold, loom_value_facts_maxsi)
BINARY_FOLD(loom_scalar_minui_fold, loom_value_facts_minui)
BINARY_FOLD(loom_scalar_maxui_fold, loom_value_facts_maxui)

// fmai has 3 operands: a*b + c.
void loom_scalar_fmai_fold(const loom_module_t* module, const loom_op_t* op,
                           const loom_value_facts_t* operand_facts,
                           loom_value_facts_t* result_facts) {
  loom_value_facts_fmai(&operand_facts[0], &operand_facts[1], &operand_facts[2],
                        &result_facts[0]);
}

//===----------------------------------------------------------------------===//
// Float arithmetic
//===----------------------------------------------------------------------===//

FLOAT_BINARY_FOLD(loom_scalar_addf_fold, add_f64)
FLOAT_BINARY_FOLD(loom_scalar_subf_fold, sub_f64)
FLOAT_BINARY_FOLD(loom_scalar_mulf_fold, mul_f64)
FLOAT_BINARY_FOLD(loom_scalar_divf_fold, div_f64)
FLOAT_BINARY_FOLD(loom_scalar_remf_fold, fmod)
FLOAT_UNARY_FOLD(loom_scalar_negf_fold, negate_f64)
FLOAT_UNARY_FOLD(loom_scalar_absf_fold, fabs)
FLOAT_BINARY_FOLD(loom_scalar_minimumf_fold, fmin)
FLOAT_BINARY_FOLD(loom_scalar_maximumf_fold, fmax)
FLOAT_BINARY_FOLD(loom_scalar_minnumf_fold, fmin)
FLOAT_BINARY_FOLD(loom_scalar_maxnumf_fold, fmax)
FLOAT_BINARY_FOLD(loom_scalar_copysignf_fold, copysign)

//===----------------------------------------------------------------------===//
// Math functions
//===----------------------------------------------------------------------===//

FLOAT_UNARY_FOLD(loom_scalar_expf_fold, exp)
FLOAT_UNARY_FOLD(loom_scalar_exp2f_fold, exp2)
FLOAT_UNARY_FOLD(loom_scalar_expm1f_fold, expm1)
FLOAT_UNARY_FOLD(loom_scalar_logf_fold, log)
FLOAT_UNARY_FOLD(loom_scalar_log2f_fold, log2)
FLOAT_UNARY_FOLD(loom_scalar_log10f_fold, log10)
FLOAT_UNARY_FOLD(loom_scalar_log1pf_fold, log1p)
FLOAT_BINARY_FOLD(loom_scalar_powf_fold, pow)
FLOAT_UNARY_FOLD(loom_scalar_sqrtf_fold, sqrt)
FLOAT_UNARY_FOLD(loom_scalar_rsqrtf_fold, rsqrt_f64)
FLOAT_UNARY_FOLD(loom_scalar_cbrtf_fold, cbrt)
FLOAT_UNARY_FOLD(loom_scalar_sinf_fold, sin)
FLOAT_UNARY_FOLD(loom_scalar_cosf_fold, cos)
FLOAT_UNARY_FOLD(loom_scalar_tanf_fold, tan)
FLOAT_UNARY_FOLD(loom_scalar_asinf_fold, asin)
FLOAT_UNARY_FOLD(loom_scalar_acosf_fold, acos)
FLOAT_UNARY_FOLD(loom_scalar_atanf_fold, atan)
FLOAT_BINARY_FOLD(loom_scalar_atan2f_fold, atan2)
FLOAT_UNARY_FOLD(loom_scalar_sinhf_fold, sinh)
FLOAT_UNARY_FOLD(loom_scalar_coshf_fold, cosh)
FLOAT_UNARY_FOLD(loom_scalar_tanhf_fold, tanh)
FLOAT_UNARY_FOLD(loom_scalar_asinhf_fold, asinh)
FLOAT_UNARY_FOLD(loom_scalar_acoshf_fold, acosh)
FLOAT_UNARY_FOLD(loom_scalar_atanhf_fold, atanh)
FLOAT_UNARY_FOLD(loom_scalar_erff_fold, erf)
FLOAT_UNARY_FOLD(loom_scalar_erfcf_fold, erfc)

// fmaf has 3 operands.
void loom_scalar_fmaf_fold(const loom_module_t* module, const loom_op_t* op,
                           const loom_value_facts_t* operand_facts,
                           loom_value_facts_t* result_facts) {
  for (int i = 0; i < 3; ++i) {
    if (!loom_value_facts_is_exact(operand_facts[i]) ||
        !loom_value_facts_is_float(operand_facts[i])) {
      result_facts[0] = loom_value_facts_unknown();
      return;
    }
  }
  result_facts[0] = loom_value_facts_exact_f64(
      fma(loom_value_facts_as_f64(operand_facts[0]),
          loom_value_facts_as_f64(operand_facts[1]),
          loom_value_facts_as_f64(operand_facts[2])));
}

FLOAT_UNARY_FOLD(loom_scalar_ceilf_fold, ceil)
FLOAT_UNARY_FOLD(loom_scalar_floorf_fold, floor)
FLOAT_UNARY_FOLD(loom_scalar_roundf_fold, round)
FLOAT_UNARY_FOLD(loom_scalar_roundevenf_fold, roundeven_f64)
FLOAT_UNARY_FOLD(loom_scalar_truncf_fold, trunc)

//===----------------------------------------------------------------------===//
// Bitwise
//===----------------------------------------------------------------------===//

BINARY_FOLD(loom_scalar_andi_fold, loom_value_facts_andi)
BINARY_FOLD(loom_scalar_ori_fold, loom_value_facts_ori)
BINARY_FOLD(loom_scalar_xori_fold, loom_value_facts_xori)
BINARY_FOLD(loom_scalar_shli_fold, loom_value_facts_shli)
BINARY_FOLD(loom_scalar_shrsi_fold, loom_value_facts_shrsi)
BINARY_FOLD(loom_scalar_shrui_fold, loom_value_facts_shrui)

// Rotates: no transfer function yet — exact-only fold.
void loom_scalar_rotli_fold(const loom_module_t* module, const loom_op_t* op,
                            const loom_value_facts_t* operand_facts,
                            loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_unknown();
}
void loom_scalar_rotri_fold(const loom_module_t* module, const loom_op_t* op,
                            const loom_value_facts_t* operand_facts,
                            loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_unknown();
}

// Bit counting: exact-only.
static int64_t clz64(uint64_t x) {
  return x == 0 ? 64 : (int64_t)iree_math_count_leading_zeros_u64(x);
}
static int64_t ctz64(uint64_t x) {
  return x == 0 ? 64 : (int64_t)iree_math_count_trailing_zeros_u64(x);
}
static int64_t popcount64(uint64_t x) {
  x = x - ((x >> 1) & 0x5555555555555555ULL);
  x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
  return (int64_t)(((x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL) *
                       0x0101010101010101ULL >>
                   56);
}

#define EXACT_UNARY_I64_FOLD(name, fn)                                       \
  void name(const loom_module_t* module, const loom_op_t* op,                \
            const loom_value_facts_t* operand_facts,                         \
            loom_value_facts_t* result_facts) {                              \
    if (!loom_value_facts_is_exact(operand_facts[0])) {                      \
      result_facts[0] = loom_value_facts_unknown();                          \
      return;                                                                \
    }                                                                        \
    result_facts[0] =                                                        \
        loom_value_facts_exact_i64(fn((uint64_t)operand_facts[0].range_lo)); \
  }

EXACT_UNARY_I64_FOLD(loom_scalar_ctlzi_fold, clz64)
EXACT_UNARY_I64_FOLD(loom_scalar_cttzi_fold, ctz64)
EXACT_UNARY_I64_FOLD(loom_scalar_ctpopi_fold, popcount64)

//===----------------------------------------------------------------------===//
// Comparison
//===----------------------------------------------------------------------===//

// cmpi/cmpf: boolean result, no predicate-aware folding yet.
void loom_scalar_cmpi_fold(const loom_module_t* module, const loom_op_t* op,
                           const loom_value_facts_t* operand_facts,
                           loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_make(0, 1, 1);
}

void loom_scalar_cmpf_fold(const loom_module_t* module, const loom_op_t* op,
                           const loom_value_facts_t* operand_facts,
                           loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_make(0, 1, 1);
}

void loom_scalar_select_fold(const loom_module_t* module, const loom_op_t* op,
                             const loom_value_facts_t* operand_facts,
                             loom_value_facts_t* result_facts) {
  if (loom_value_facts_is_exact(operand_facts[0])) {
    result_facts[0] =
        operand_facts[0].range_lo ? operand_facts[1] : operand_facts[2];
    return;
  }
  loom_value_facts_meet(&operand_facts[1], &operand_facts[2], &result_facts[0]);
}

// Float predicates: exact-only.
static bool isnan_f64(double v) { return v != v; }
static bool isinf_f64(double v) { return isinf(v) != 0; }
static bool isfinite_f64(double v) { return isfinite(v) != 0; }

#define FLOAT_PREDICATE_FOLD(name, fn)                           \
  void name(const loom_module_t* module, const loom_op_t* op,    \
            const loom_value_facts_t* operand_facts,             \
            loom_value_facts_t* result_facts) {                  \
    if (!loom_value_facts_is_exact(operand_facts[0]) ||          \
        !loom_value_facts_is_float(operand_facts[0])) {          \
      result_facts[0] = loom_value_facts_make(0, 1, 1);          \
      return;                                                    \
    }                                                            \
    double v = loom_value_facts_as_f64(operand_facts[0]);        \
    result_facts[0] = loom_value_facts_exact_i64(fn(v) ? 1 : 0); \
  }

FLOAT_PREDICATE_FOLD(loom_scalar_isnanf_fold, isnan_f64)
FLOAT_PREDICATE_FOLD(loom_scalar_isinff_fold, isinf_f64)
FLOAT_PREDICATE_FOLD(loom_scalar_isfinitef_fold, isfinite_f64)

void loom_scalar_signf_fold(const loom_module_t* module, const loom_op_t* op,
                            const loom_value_facts_t* operand_facts,
                            loom_value_facts_t* result_facts) {
  if (!loom_value_facts_is_exact(operand_facts[0]) ||
      !loom_value_facts_is_float(operand_facts[0])) {
    result_facts[0] = loom_value_facts_unknown();
    return;
  }
  double v = loom_value_facts_as_f64(operand_facts[0]);
  result_facts[0] = loom_value_facts_exact_f64((v > 0.0)   ? 1.0
                                               : (v < 0.0) ? -1.0
                                                           : 0.0);
}

void loom_scalar_signi_fold(const loom_module_t* module, const loom_op_t* op,
                            const loom_value_facts_t* operand_facts,
                            loom_value_facts_t* result_facts) {
  if (!loom_value_facts_is_exact(operand_facts[0])) {
    result_facts[0] = loom_value_facts_unknown();
    return;
  }
  int64_t v = operand_facts[0].range_lo;
  result_facts[0] = loom_value_facts_exact_i64((v > 0) ? 1 : (v < 0) ? -1 : 0);
}

//===----------------------------------------------------------------------===//
// Conversion
//===----------------------------------------------------------------------===//

void loom_scalar_sitofp_fold(const loom_module_t* module, const loom_op_t* op,
                             const loom_value_facts_t* operand_facts,
                             loom_value_facts_t* result_facts) {
  if (!loom_value_facts_is_exact(operand_facts[0])) {
    result_facts[0] = loom_value_facts_unknown();
    return;
  }
  result_facts[0] =
      loom_value_facts_exact_f64((double)operand_facts[0].range_lo);
}

void loom_scalar_uitofp_fold(const loom_module_t* module, const loom_op_t* op,
                             const loom_value_facts_t* operand_facts,
                             loom_value_facts_t* result_facts) {
  if (!loom_value_facts_is_exact(operand_facts[0])) {
    result_facts[0] = loom_value_facts_unknown();
    return;
  }
  result_facts[0] =
      loom_value_facts_exact_f64((double)(uint64_t)operand_facts[0].range_lo);
}

void loom_scalar_fptosi_fold(const loom_module_t* module, const loom_op_t* op,
                             const loom_value_facts_t* operand_facts,
                             loom_value_facts_t* result_facts) {
  if (!loom_value_facts_is_exact(operand_facts[0]) ||
      !loom_value_facts_is_float(operand_facts[0])) {
    result_facts[0] = loom_value_facts_unknown();
    return;
  }
  result_facts[0] = loom_value_facts_exact_i64(
      (int64_t)loom_value_facts_as_f64(operand_facts[0]));
}

void loom_scalar_fptoui_fold(const loom_module_t* module, const loom_op_t* op,
                             const loom_value_facts_t* operand_facts,
                             loom_value_facts_t* result_facts) {
  if (!loom_value_facts_is_exact(operand_facts[0]) ||
      !loom_value_facts_is_float(operand_facts[0])) {
    result_facts[0] = loom_value_facts_unknown();
    return;
  }
  result_facts[0] = loom_value_facts_exact_i64(
      (int64_t)(uint64_t)loom_value_facts_as_f64(operand_facts[0]));
}

void loom_scalar_extf_fold(const loom_module_t* module, const loom_op_t* op,
                           const loom_value_facts_t* operand_facts,
                           loom_value_facts_t* result_facts) {
  result_facts[0] = operand_facts[0];
}

void loom_scalar_fptrunc_fold(const loom_module_t* module, const loom_op_t* op,
                              const loom_value_facts_t* operand_facts,
                              loom_value_facts_t* result_facts) {
  if (!loom_value_facts_is_exact(operand_facts[0]) ||
      !loom_value_facts_is_float(operand_facts[0])) {
    result_facts[0] = loom_value_facts_unknown();
    return;
  }
  double v = loom_value_facts_as_f64(operand_facts[0]);
  result_facts[0] = loom_value_facts_exact_f64((double)(float)v);
}

void loom_scalar_extsi_fold(const loom_module_t* module, const loom_op_t* op,
                            const loom_value_facts_t* operand_facts,
                            loom_value_facts_t* result_facts) {
  result_facts[0] = operand_facts[0];
}

void loom_scalar_extui_fold(const loom_module_t* module, const loom_op_t* op,
                            const loom_value_facts_t* operand_facts,
                            loom_value_facts_t* result_facts) {
  if (loom_value_facts_is_non_negative(operand_facts[0])) {
    result_facts[0] = operand_facts[0];
  } else {
    result_facts[0] = loom_value_facts_unknown();
  }
}

void loom_scalar_trunci_fold(const loom_module_t* module, const loom_op_t* op,
                             const loom_value_facts_t* operand_facts,
                             loom_value_facts_t* result_facts) {
  if (loom_value_facts_is_exact(operand_facts[0])) {
    result_facts[0] = operand_facts[0];
  } else {
    result_facts[0] = loom_value_facts_unknown();
  }
}

void loom_scalar_index_cast_fold(const loom_module_t* module,
                                 const loom_op_t* op,
                                 const loom_value_facts_t* operand_facts,
                                 loom_value_facts_t* result_facts) {
  result_facts[0] = operand_facts[0];
}

void loom_scalar_bitcast_fold(const loom_module_t* module, const loom_op_t* op,
                              const loom_value_facts_t* operand_facts,
                              loom_value_facts_t* result_facts) {
  if (loom_value_facts_is_exact(operand_facts[0])) {
    result_facts[0] = operand_facts[0];
  } else {
    result_facts[0] = loom_value_facts_unknown();
  }
}

//===----------------------------------------------------------------------===//
// scalar.assume
//===----------------------------------------------------------------------===//

void loom_scalar_assume_fold(const loom_module_t* module, const loom_op_t* op,
                             const loom_value_facts_t* operand_facts,
                             loom_value_facts_t* result_facts) {
  for (uint16_t i = 0; i < op->result_count; ++i) {
    result_facts[i] = operand_facts[i];
  }
  loom_attribute_t pred_attr = loom_op_attrs(op)[0];
  const loom_predicate_t* predicates = pred_attr.predicate_list;
  uint16_t predicate_count = pred_attr.count;
  for (uint16_t p = 0; p < predicate_count; ++p) {
    const loom_predicate_t* pred = &predicates[p];
    uint16_t target = 0;
    if (pred->arg_tags[0] == LOOM_PRED_ARG_ORDINAL) {
      target = (uint16_t)pred->args[0];
    } else if (pred->arg_tags[0] == LOOM_PRED_ARG_VALUE) {
      loom_value_slice_t values = loom_scalar_assume_values(op);
      loom_value_id_t target_id = (loom_value_id_t)pred->args[0];
      for (uint16_t i = 0; i < values.count; ++i) {
        if (values.values[i] == target_id) {
          target = i;
          break;
        }
      }
    }
    if (target < op->result_count) {
      loom_value_facts_apply_predicate(&result_facts[target], pred);
    }
  }
}
