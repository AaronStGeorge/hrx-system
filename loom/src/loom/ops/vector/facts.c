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

static bool loom_vector_facts_query_small_lanes(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_small_static_lanes_t* out_lanes) {
  return loom_value_facts_query_small_static_lanes(context, facts, out_lanes);
}

static bool loom_vector_facts_query_lane(const loom_fact_context_t* context,
                                         loom_value_facts_t facts,
                                         iree_host_size_t lane,
                                         loom_value_facts_t* out_element) {
  if (loom_vector_facts_query_uniform_element(context, facts, out_element)) {
    return true;
  }
  loom_value_fact_small_static_lanes_t lanes = {0};
  if (!loom_vector_facts_query_small_lanes(context, facts, &lanes) ||
      lane >= lanes.count) {
    return false;
  }
  *out_element = lanes.lanes[lane];
  return true;
}

static bool loom_vector_facts_query_binary_lane_count(
    const loom_fact_context_t* context, loom_value_facts_t lhs,
    loom_value_facts_t rhs, iree_host_size_t* out_lane_count) {
  loom_value_fact_small_static_lanes_t lhs_lanes = {0};
  loom_value_fact_small_static_lanes_t rhs_lanes = {0};
  bool lhs_is_small =
      loom_vector_facts_query_small_lanes(context, lhs, &lhs_lanes);
  bool rhs_is_small =
      loom_vector_facts_query_small_lanes(context, rhs, &rhs_lanes);
  if (lhs_is_small && rhs_is_small) {
    if (lhs_lanes.count != rhs_lanes.count) return false;
    *out_lane_count = lhs_lanes.count;
    return true;
  }
  if (lhs_is_small) {
    *out_lane_count = lhs_lanes.count;
    return true;
  }
  if (rhs_is_small) {
    *out_lane_count = rhs_lanes.count;
    return true;
  }
  return false;
}

static bool loom_vector_facts_query_ternary_lane_count(
    const loom_fact_context_t* context, loom_value_facts_t a,
    loom_value_facts_t b, loom_value_facts_t c,
    iree_host_size_t* out_lane_count) {
  loom_value_fact_small_static_lanes_t lane_sets[3] = {{0}};
  loom_value_facts_t facts[3] = {a, b, c};
  bool found_count = false;
  iree_host_size_t lane_count = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(facts); ++i) {
    if (!loom_vector_facts_query_small_lanes(context, facts[i],
                                             &lane_sets[i])) {
      continue;
    }
    if (found_count && lane_sets[i].count != lane_count) return false;
    lane_count = lane_sets[i].count;
    found_count = true;
  }
  if (!found_count) return false;
  *out_lane_count = lane_count;
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

iree_status_t loom_vector_from_elements_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_small_static_lanes_t lanes = {
      .lanes = operand_facts,
      .count = op->operand_count,
  };
  return loom_value_facts_make_small_static_lanes(context, lanes,
                                                  &result_facts[0]);
}

iree_status_t loom_vector_extract_facts(loom_fact_context_t* context,
                                        const loom_module_t* module,
                                        const loom_op_t* op,
                                        const loom_value_facts_t* operand_facts,
                                        loom_value_facts_t* result_facts) {
  loom_value_id_t result_id = loom_vector_extract_result(op);
  loom_type_t result_type = loom_module_value_type(module, result_id);
  loom_attribute_t static_indices = loom_vector_extract_static_indices(op);
  if (!loom_type_is_scalar(result_type) || static_indices.count != 1 ||
      static_indices.i64_array[0] == INT64_MIN ||
      static_indices.i64_array[0] < 0) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  iree_host_size_t lane = (iree_host_size_t)static_indices.i64_array[0];
  if (!loom_vector_facts_query_lane(context, operand_facts[0], lane,
                                    &result_facts[0])) {
    result_facts[0] = loom_value_facts_unknown();
  }
  return iree_ok_status();
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

iree_status_t loom_vector_shuffle_facts(loom_fact_context_t* context,
                                        const loom_module_t* module,
                                        const loom_op_t* op,
                                        const loom_value_facts_t* operand_facts,
                                        loom_value_facts_t* result_facts) {
  loom_value_facts_t uniform = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &uniform)) {
    return loom_value_facts_make_uniform_element(context, uniform,
                                                 &result_facts[0]);
  }

  loom_attribute_t source_lanes = loom_vector_shuffle_source_lanes(op);
  if (source_lanes.count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (uint16_t i = 0; i < source_lanes.count; ++i) {
    int64_t source_lane = source_lanes.i64_array[i];
    if (source_lane < 0 || !loom_vector_facts_query_lane(
                               context, operand_facts[0],
                               (iree_host_size_t)source_lane, &lanes[i])) {
      result_facts[0] = loom_value_facts_unknown();
      return iree_ok_status();
    }
  }
  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = source_lanes.count,
  };
  return loom_value_facts_make_small_static_lanes(context, lane_slice,
                                                  &result_facts[0]);
}

//===----------------------------------------------------------------------===//
// Lanewise summary propagation
//===----------------------------------------------------------------------===//

static iree_status_t loom_vector_integer_binary_summary_facts(
    loom_fact_context_t* context, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts,
    loom_vector_integer_binary_transfer_fn_t transfer_fn) {
  loom_value_facts_t lhs = {0};
  loom_value_facts_t rhs = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &lhs) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[1],
                                              &rhs)) {
    loom_value_facts_t element = loom_value_facts_unknown();
    transfer_fn(&lhs, &rhs, &element);
    return loom_value_facts_make_uniform_element(context, element,
                                                 &result_facts[0]);
  }

  iree_host_size_t lane_count = 0;
  if (!loom_vector_facts_query_binary_lane_count(
          context, operand_facts[0], operand_facts[1], &lane_count)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t i = 0; i < lane_count; ++i) {
    if (!loom_vector_facts_query_lane(context, operand_facts[0], i, &lhs) ||
        !loom_vector_facts_query_lane(context, operand_facts[1], i, &rhs)) {
      result_facts[0] = loom_value_facts_unknown();
      return iree_ok_status();
    }
    transfer_fn(&lhs, &rhs, &lanes[i]);
  }
  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = lane_count,
  };
  return loom_value_facts_make_small_static_lanes(context, lane_slice,
                                                  &result_facts[0]);
}

static iree_status_t loom_vector_float_binary_summary_facts(
    loom_fact_context_t* context, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts, double (*fn)(double, double)) {
  loom_value_facts_t lhs = {0};
  loom_value_facts_t rhs = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &lhs) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[1],
                                              &rhs)) {
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

  iree_host_size_t lane_count = 0;
  if (!loom_vector_facts_query_binary_lane_count(
          context, operand_facts[0], operand_facts[1], &lane_count)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t i = 0; i < lane_count; ++i) {
    if (!loom_vector_facts_query_lane(context, operand_facts[0], i, &lhs) ||
        !loom_vector_facts_query_lane(context, operand_facts[1], i, &rhs)) {
      result_facts[0] = loom_value_facts_unknown();
      return iree_ok_status();
    }
    double lhs_value = 0.0;
    double rhs_value = 0.0;
    lanes[i] = loom_value_facts_unknown();
    if (loom_vector_facts_query_exact_f64(lhs, &lhs_value) &&
        loom_vector_facts_query_exact_f64(rhs, &rhs_value)) {
      lanes[i] = loom_value_facts_exact_f64(fn(lhs_value, rhs_value));
    }
  }
  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = lane_count,
  };
  return loom_value_facts_make_small_static_lanes(context, lane_slice,
                                                  &result_facts[0]);
}

#define LOOM_VECTOR_INTEGER_BINARY_FACTS(name, transfer_fn)            \
  iree_status_t name(loom_fact_context_t* context,                     \
                     const loom_module_t* module, const loom_op_t* op, \
                     const loom_value_facts_t* operand_facts,          \
                     loom_value_facts_t* result_facts) {               \
    return loom_vector_integer_binary_summary_facts(                   \
        context, operand_facts, result_facts, transfer_fn);            \
  }

#define LOOM_VECTOR_FLOAT_BINARY_FACTS(name, fn)                          \
  iree_status_t name(loom_fact_context_t* context,                        \
                     const loom_module_t* module, const loom_op_t* op,    \
                     const loom_value_facts_t* operand_facts,             \
                     loom_value_facts_t* result_facts) {                  \
    return loom_vector_float_binary_summary_facts(context, operand_facts, \
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
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0], &a) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[1], &b) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[2], &c)) {
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

  iree_host_size_t lane_count = 0;
  if (!loom_vector_facts_query_ternary_lane_count(
          context, operand_facts[0], operand_facts[1], operand_facts[2],
          &lane_count)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }

  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t i = 0; i < lane_count; ++i) {
    if (!loom_vector_facts_query_lane(context, operand_facts[0], i, &a) ||
        !loom_vector_facts_query_lane(context, operand_facts[1], i, &b) ||
        !loom_vector_facts_query_lane(context, operand_facts[2], i, &c)) {
      result_facts[0] = loom_value_facts_unknown();
      return iree_ok_status();
    }
    double a_value = 0.0;
    double b_value = 0.0;
    double c_value = 0.0;
    lanes[i] = loom_value_facts_unknown();
    if (loom_vector_facts_query_exact_f64(a, &a_value) &&
        loom_vector_facts_query_exact_f64(b, &b_value) &&
        loom_vector_facts_query_exact_f64(c, &c_value)) {
      lanes[i] = loom_value_facts_exact_f64(fma(a_value, b_value, c_value));
    }
  }
  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = lane_count,
  };
  return loom_value_facts_make_small_static_lanes(context, lane_slice,
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
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &condition)) {
    if (!loom_value_facts_is_exact(condition)) {
      result_facts[0] = loom_value_facts_unknown();
      return iree_ok_status();
    }
    result_facts[0] = condition.range_lo ? operand_facts[1] : operand_facts[2];
    return iree_ok_status();
  }

  loom_value_fact_small_static_lanes_t condition_lanes = {0};
  if (!loom_vector_facts_query_small_lanes(context, operand_facts[0],
                                           &condition_lanes)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t i = 0; i < condition_lanes.count; ++i) {
    condition = condition_lanes.lanes[i];
    if (!loom_value_facts_is_exact(condition)) {
      loom_value_facts_t true_lane = {0};
      loom_value_facts_t false_lane = {0};
      if (loom_vector_facts_query_lane(context, operand_facts[1], i,
                                       &true_lane) &&
          loom_vector_facts_query_lane(context, operand_facts[2], i,
                                       &false_lane) &&
          loom_value_facts_equal(true_lane, false_lane)) {
        lanes[i] = true_lane;
      } else {
        lanes[i] = loom_value_facts_unknown();
      }
      continue;
    }
    if (!loom_vector_facts_query_lane(
            context, condition.range_lo ? operand_facts[1] : operand_facts[2],
            i, &lanes[i])) {
      result_facts[0] = loom_value_facts_unknown();
      return iree_ok_status();
    }
  }
  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = condition_lanes.count,
  };
  return loom_value_facts_make_small_static_lanes(context, lane_slice,
                                                  &result_facts[0]);
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

static bool loom_vector_reduce_small_static_lanes(
    uint8_t kind, loom_value_fact_small_static_lanes_t lanes,
    loom_value_facts_t init, loom_value_facts_t* out) {
  if (lanes.count == 0) {
    *out = init;
    return true;
  }

  double float_accumulator = 0.0;
  if (loom_vector_facts_query_exact_f64(init, &float_accumulator)) {
    for (iree_host_size_t i = 0; i < lanes.count; ++i) {
      double element = 0.0;
      if (!loom_vector_facts_query_exact_f64(lanes.lanes[i], &element) ||
          !loom_vector_reduce_apply_float(kind, float_accumulator, element,
                                          &float_accumulator)) {
        return false;
      }
    }
    *out = loom_value_facts_exact_f64(float_accumulator);
    return true;
  }

  loom_value_facts_t accumulator = init;
  for (iree_host_size_t i = 0; i < lanes.count; ++i) {
    loom_value_facts_t next = loom_value_facts_unknown();
    if (!loom_vector_reduce_apply_integer(kind, &accumulator, &lanes.lanes[i],
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

  uint8_t kind = loom_vector_reduce_kind(op);
  loom_value_fact_small_static_lanes_t lanes = {0};
  if (loom_vector_facts_query_small_lanes(context, operand_facts[0], &lanes)) {
    if (loom_vector_reduce_small_static_lanes(kind, lanes, operand_facts[1],
                                              &result_facts[0])) {
      return iree_ok_status();
    }
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }

  loom_value_facts_t element = {0};
  if (!loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                               &element)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }

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

  iree_host_size_t lane_count = 0;
  if (loom_vector_facts_query_binary_lane_count(
          context, operand_facts[0], operand_facts[1], &lane_count)) {
    double accumulator = 0.0;
    if (!loom_vector_facts_query_exact_f64(operand_facts[2], &accumulator)) {
      result_facts[0] = loom_value_facts_unknown();
      return iree_ok_status();
    }
    for (iree_host_size_t i = 0; i < lane_count; ++i) {
      loom_value_facts_t lhs = {0};
      loom_value_facts_t rhs = {0};
      double lhs_value = 0.0;
      double rhs_value = 0.0;
      if (!loom_vector_facts_query_lane(context, operand_facts[0], i, &lhs) ||
          !loom_vector_facts_query_lane(context, operand_facts[1], i, &rhs) ||
          !loom_vector_facts_query_exact_f64(lhs, &lhs_value) ||
          !loom_vector_facts_query_exact_f64(rhs, &rhs_value)) {
        result_facts[0] = loom_value_facts_unknown();
        return iree_ok_status();
      }
      accumulator = fma(lhs_value, rhs_value, accumulator);
    }
    result_facts[0] = loom_value_facts_exact_f64(accumulator);
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
