// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fact implementations for the vector dialect.
//
// Vector facts deliberately summarize register values instead of interpreting
// every lane. Uniform-element facts let construction and lanewise ops preserve
// "all lanes have the same scalar facts"; reductions and dot products can then
// fold to scalar facts without teaching the canonicalizer to materialize vector
// constants. Iota and prefix-mask facts keep structural vector producers
// visible to later lowering/fact consumers without making every pass walk
// vector lanes.

#include <math.h>

#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ops/vector/ops.h"
#include "loom/util/fact_table.h"

#define LOOM_VECTOR_FACT_STATIC_LOOP_LIMIT 1024

typedef void (*loom_vector_integer_binary_transfer_fn_t)(
    const loom_value_facts_t* lhs, const loom_value_facts_t* rhs,
    loom_value_facts_t* out);

//===----------------------------------------------------------------------===//
// Scalar element helpers
//===----------------------------------------------------------------------===//

static bool loom_vector_facts_query_uniform_element(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_facts_t* out_element) {
  loom_value_fact_uniform_element_t uniform = {0};
  if (!loom_value_facts_query_uniform_element(context, facts, &uniform)) {
    return false;
  }
  *out_element = uniform.element;
  return true;
}

static loom_value_facts_t loom_vector_attr_element_facts(
    loom_attribute_t attr, loom_scalar_type_t element_type) {
  if (loom_scalar_type_is_float(element_type)) {
    return loom_value_facts_exact_f64(loom_attr_as_f64(attr));
  }
  if (element_type == LOOM_SCALAR_TYPE_I1 && attr.kind == LOOM_ATTR_BOOL) {
    return loom_value_facts_exact_i64(loom_attr_as_bool(attr) ? 1 : 0);
  }
  return loom_value_facts_exact_i64(loom_attr_as_i64(attr));
}

static bool loom_vector_facts_exact_i64_is(loom_value_facts_t facts,
                                           int64_t expected) {
  return loom_value_facts_is_exact(facts) &&
         !loom_value_facts_is_float(facts) && facts.range_lo == expected;
}

static bool loom_vector_facts_query_exact_f64(loom_value_facts_t facts,
                                              double* out_value) {
  if (!loom_value_facts_is_exact(facts) || !loom_value_facts_is_float(facts)) {
    return false;
  }
  *out_value = loom_value_facts_as_f64(facts);
  return true;
}

static double loom_vector_add_f64(double lhs, double rhs) { return lhs + rhs; }

static double loom_vector_mul_f64(double lhs, double rhs) { return lhs * rhs; }

static double loom_vector_minimum_f64(double lhs, double rhs) {
  return fmin(lhs, rhs);
}

static double loom_vector_maximum_f64(double lhs, double rhs) {
  return fmax(lhs, rhs);
}

//===----------------------------------------------------------------------===//
// Construction
//===----------------------------------------------------------------------===//

iree_status_t loom_vector_constant_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_id_t result_id = loom_vector_constant_result(op);
  loom_type_t result_type = loom_module_value_type(module, result_id);
  loom_value_facts_t element = loom_vector_attr_element_facts(
      loom_vector_constant_value(op), loom_type_element_type(result_type));
  return loom_value_facts_make_uniform_element(context, element,
                                               &result_facts[0]);
}

iree_status_t loom_vector_splat_facts(loom_fact_context_t* context,
                                      const loom_module_t* module,
                                      const loom_op_t* op,
                                      const loom_value_facts_t* operand_facts,
                                      loom_value_facts_t* result_facts) {
  return loom_value_facts_make_uniform_element(context, operand_facts[0],
                                               &result_facts[0]);
}

iree_status_t loom_vector_iota_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  loom_value_fact_vector_iota_t iota = {
      .base = operand_facts[0],
      .step = operand_facts[1],
  };
  return loom_value_facts_make_vector_iota(context, iota, &result_facts[0]);
}

iree_status_t loom_vector_mask_range_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  if (loom_value_facts_is_exact(operand_facts[0]) &&
      loom_value_facts_is_exact(operand_facts[1]) &&
      operand_facts[0].range_lo >= operand_facts[1].range_lo) {
    return loom_value_facts_make_uniform_element(
        context, loom_value_facts_exact_i64(0), &result_facts[0]);
  }
  loom_value_fact_vector_prefix_mask_t mask = {
      .lower_bound = operand_facts[0],
      .upper_bound = operand_facts[1],
      .step = operand_facts[2],
  };
  return loom_value_facts_make_vector_prefix_mask(context, mask,
                                                  &result_facts[0]);
}

//===----------------------------------------------------------------------===//
// Lanewise uniform propagation
//===----------------------------------------------------------------------===//

static iree_status_t loom_vector_integer_binary_uniform_facts(
    loom_fact_context_t* context, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts,
    loom_vector_integer_binary_transfer_fn_t transfer_fn) {
  loom_value_facts_t lhs = {0};
  loom_value_facts_t rhs = {0};
  if (!loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                               &lhs) ||
      !loom_vector_facts_query_uniform_element(context, operand_facts[1],
                                               &rhs)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  loom_value_facts_t element = loom_value_facts_unknown();
  transfer_fn(&lhs, &rhs, &element);
  return loom_value_facts_make_uniform_element(context, element,
                                               &result_facts[0]);
}

static iree_status_t loom_vector_float_binary_uniform_facts(
    loom_fact_context_t* context, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts, double (*fn)(double, double)) {
  loom_value_facts_t lhs = {0};
  loom_value_facts_t rhs = {0};
  if (!loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                               &lhs) ||
      !loom_vector_facts_query_uniform_element(context, operand_facts[1],
                                               &rhs)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  double lhs_value = 0.0;
  double rhs_value = 0.0;
  loom_value_facts_t element = loom_value_facts_unknown();
  if (loom_vector_facts_query_exact_f64(lhs, &lhs_value) &&
      loom_vector_facts_query_exact_f64(rhs, &rhs_value)) {
    element = loom_value_facts_exact_f64(fn(lhs_value, rhs_value));
  }
  return loom_value_facts_make_uniform_element(context, element,
                                               &result_facts[0]);
}

#define LOOM_VECTOR_INTEGER_BINARY_FACTS(name, transfer_fn)            \
  iree_status_t name(loom_fact_context_t* context,                     \
                     const loom_module_t* module, const loom_op_t* op, \
                     const loom_value_facts_t* operand_facts,          \
                     loom_value_facts_t* result_facts) {               \
    return loom_vector_integer_binary_uniform_facts(                   \
        context, operand_facts, result_facts, transfer_fn);            \
  }

#define LOOM_VECTOR_FLOAT_BINARY_FACTS(name, fn)                          \
  iree_status_t name(loom_fact_context_t* context,                        \
                     const loom_module_t* module, const loom_op_t* op,    \
                     const loom_value_facts_t* operand_facts,             \
                     loom_value_facts_t* result_facts) {                  \
    return loom_vector_float_binary_uniform_facts(context, operand_facts, \
                                                  result_facts, fn);      \
  }

LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_addf_facts, loom_vector_add_f64)
LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_mulf_facts, loom_vector_mul_f64)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_addi_facts, loom_value_facts_addi)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_muli_facts, loom_value_facts_muli)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_andi_facts, loom_value_facts_andi)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_ori_facts, loom_value_facts_ori)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_xori_facts, loom_value_facts_xori)

iree_status_t loom_vector_fmaf_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  loom_value_facts_t a = {0};
  loom_value_facts_t b = {0};
  loom_value_facts_t c = {0};
  if (!loom_vector_facts_query_uniform_element(context, operand_facts[0], &a) ||
      !loom_vector_facts_query_uniform_element(context, operand_facts[1], &b) ||
      !loom_vector_facts_query_uniform_element(context, operand_facts[2], &c)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  double a_value = 0.0;
  double b_value = 0.0;
  double c_value = 0.0;
  loom_value_facts_t element = loom_value_facts_unknown();
  if (loom_vector_facts_query_exact_f64(a, &a_value) &&
      loom_vector_facts_query_exact_f64(b, &b_value) &&
      loom_vector_facts_query_exact_f64(c, &c_value)) {
    element = loom_value_facts_exact_f64(fma(a_value, b_value, c_value));
  }
  return loom_value_facts_make_uniform_element(context, element,
                                               &result_facts[0]);
}

iree_status_t loom_vector_select_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  if (loom_value_facts_equal(operand_facts[1], operand_facts[2])) {
    result_facts[0] = operand_facts[1];
    return iree_ok_status();
  }
  loom_value_facts_t condition = {0};
  if (!loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                               &condition) ||
      !loom_value_facts_is_exact(condition)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  result_facts[0] = condition.range_lo ? operand_facts[1] : operand_facts[2];
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Scalar-producing reductions
//===----------------------------------------------------------------------===//

static bool loom_vector_reduce_apply_integer(
    uint8_t kind, const loom_value_facts_t* accumulator,
    const loom_value_facts_t* element, loom_value_facts_t* out) {
  switch (kind) {
    case LOOM_VECTOR_REDUCE_KIND_ADDI:
      loom_value_facts_addi(accumulator, element, out);
      return true;
    case LOOM_VECTOR_REDUCE_KIND_MULI:
      loom_value_facts_muli(accumulator, element, out);
      return true;
    case LOOM_VECTOR_REDUCE_KIND_MINSI:
      loom_value_facts_minsi(accumulator, element, out);
      return true;
    case LOOM_VECTOR_REDUCE_KIND_MAXSI:
      loom_value_facts_maxsi(accumulator, element, out);
      return true;
    case LOOM_VECTOR_REDUCE_KIND_MINUI:
      loom_value_facts_minui(accumulator, element, out);
      return true;
    case LOOM_VECTOR_REDUCE_KIND_MAXUI:
      loom_value_facts_maxui(accumulator, element, out);
      return true;
    case LOOM_VECTOR_REDUCE_KIND_ANDI:
      loom_value_facts_andi(accumulator, element, out);
      return true;
    case LOOM_VECTOR_REDUCE_KIND_ORI:
      loom_value_facts_ori(accumulator, element, out);
      return true;
    case LOOM_VECTOR_REDUCE_KIND_XORI:
      loom_value_facts_xori(accumulator, element, out);
      return true;
    default:
      return false;
  }
}

static bool loom_vector_reduce_apply_float(uint8_t kind, double accumulator,
                                           double element, double* out) {
  switch (kind) {
    case LOOM_VECTOR_REDUCE_KIND_ADDF:
      *out = accumulator + element;
      return true;
    case LOOM_VECTOR_REDUCE_KIND_MULF:
      *out = accumulator * element;
      return true;
    case LOOM_VECTOR_REDUCE_KIND_MINIMUMF:
    case LOOM_VECTOR_REDUCE_KIND_MINNUMF:
      *out = loom_vector_minimum_f64(accumulator, element);
      return true;
    case LOOM_VECTOR_REDUCE_KIND_MAXIMUMF:
    case LOOM_VECTOR_REDUCE_KIND_MAXNUMF:
      *out = loom_vector_maximum_f64(accumulator, element);
      return true;
    default:
      return false;
  }
}

static bool loom_vector_reduce_dynamic_identity(uint8_t kind,
                                                loom_value_facts_t element,
                                                loom_value_facts_t init,
                                                loom_value_facts_t* out) {
  switch (kind) {
    case LOOM_VECTOR_REDUCE_KIND_ADDI:
    case LOOM_VECTOR_REDUCE_KIND_ORI:
    case LOOM_VECTOR_REDUCE_KIND_XORI:
      if (loom_vector_facts_exact_i64_is(element, 0)) {
        *out = init;
        return true;
      }
      return false;
    case LOOM_VECTOR_REDUCE_KIND_MULI:
      if (loom_vector_facts_exact_i64_is(element, 1)) {
        *out = init;
        return true;
      }
      return false;
    default:
      return false;
  }
}

static bool loom_vector_reduce_static_uniform(uint8_t kind,
                                              uint64_t element_count,
                                              loom_value_facts_t element,
                                              loom_value_facts_t init,
                                              loom_value_facts_t* out) {
  if (element_count == 0) {
    *out = init;
    return true;
  }
  if (element_count > LOOM_VECTOR_FACT_STATIC_LOOP_LIMIT) return false;

  double float_accumulator = 0.0;
  double float_element = 0.0;
  bool float_reduce =
      loom_vector_facts_query_exact_f64(init, &float_accumulator) &&
      loom_vector_facts_query_exact_f64(element, &float_element);
  if (float_reduce) {
    for (uint64_t i = 0; i < element_count; ++i) {
      if (!loom_vector_reduce_apply_float(kind, float_accumulator,
                                          float_element, &float_accumulator)) {
        return false;
      }
    }
    *out = loom_value_facts_exact_f64(float_accumulator);
    return true;
  }

  loom_value_facts_t accumulator = init;
  for (uint64_t i = 0; i < element_count; ++i) {
    loom_value_facts_t next = loom_value_facts_unknown();
    if (!loom_vector_reduce_apply_integer(kind, &accumulator, &element,
                                          &next)) {
      return false;
    }
    accumulator = next;
  }
  *out = accumulator;
  return true;
}

iree_status_t loom_vector_reduce_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  uint64_t element_count = 0;
  loom_type_t input_type =
      loom_module_value_type(module, loom_vector_reduce_input(op));
  if (loom_type_static_element_count(input_type, &element_count) &&
      element_count == 0) {
    result_facts[0] = operand_facts[1];
    return iree_ok_status();
  }

  loom_value_facts_t element = {0};
  if (!loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                               &element)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }

  uint8_t kind = loom_vector_reduce_kind(op);
  if (loom_type_static_element_count(input_type, &element_count)) {
    if (loom_vector_reduce_static_uniform(kind, element_count, element,
                                          operand_facts[1], &result_facts[0])) {
      return iree_ok_status();
    }
  } else if (loom_vector_reduce_dynamic_identity(
                 kind, element, operand_facts[1], &result_facts[0])) {
    return iree_ok_status();
  }

  result_facts[0] = loom_value_facts_unknown();
  return iree_ok_status();
}

iree_status_t loom_vector_dotf_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  uint64_t element_count = 0;
  loom_type_t lhs_type =
      loom_module_value_type(module, loom_vector_dotf_lhs(op));
  if (loom_type_static_element_count(lhs_type, &element_count) &&
      element_count == 0) {
    result_facts[0] = operand_facts[2];
    return iree_ok_status();
  }
  if (!loom_type_static_element_count(lhs_type, &element_count) ||
      element_count > LOOM_VECTOR_FACT_STATIC_LOOP_LIMIT) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }

  loom_value_facts_t lhs = {0};
  loom_value_facts_t rhs = {0};
  if (!loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                               &lhs) ||
      !loom_vector_facts_query_uniform_element(context, operand_facts[1],
                                               &rhs)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }

  double lhs_value = 0.0;
  double rhs_value = 0.0;
  double accumulator = 0.0;
  if (!loom_vector_facts_query_exact_f64(lhs, &lhs_value) ||
      !loom_vector_facts_query_exact_f64(rhs, &rhs_value) ||
      !loom_vector_facts_query_exact_f64(operand_facts[2], &accumulator)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  for (uint64_t i = 0; i < element_count; ++i) {
    accumulator = fma(lhs_value, rhs_value, accumulator);
  }
  result_facts[0] = loom_value_facts_exact_f64(accumulator);
  return iree_ok_status();
}

#undef LOOM_VECTOR_FLOAT_BINARY_FACTS
#undef LOOM_VECTOR_INTEGER_BINARY_FACTS
