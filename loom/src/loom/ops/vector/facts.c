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
#include "loom/util/math.h"

#define LOOM_VECTOR_FACT_STATIC_LOOP_LIMIT 1024

typedef void (*loom_vector_unary_transfer_fn_t)(const loom_value_facts_t* input,
                                                loom_value_facts_t* out);
typedef void (*loom_vector_integer_binary_transfer_fn_t)(
    const loom_value_facts_t* lhs, const loom_value_facts_t* rhs,
    loom_value_facts_t* out);
typedef void (*loom_vector_ternary_transfer_fn_t)(const loom_value_facts_t* a,
                                                  const loom_value_facts_t* b,
                                                  const loom_value_facts_t* c,
                                                  loom_value_facts_t* out);
typedef int64_t (*loom_vector_bit_count_fn_t)(uint64_t value, int32_t bitwidth);

typedef double (*loom_vector_float_unary_transfer_fn_t)(double input);
typedef double (*loom_vector_float_binary_transfer_fn_t)(double lhs,
                                                         double rhs);

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

static bool loom_vector_mask_range_exact_lane(int64_t lower_bound,
                                              int64_t upper_bound, int64_t step,
                                              uint64_t lane_ordinal,
                                              bool* out_value) {
  if (lane_ordinal > (uint64_t)INT64_MAX) return false;
  int64_t lane_delta = 0;
  if (!loom_checked_mul_i64((int64_t)lane_ordinal, step, &lane_delta)) {
    return false;
  }
  int64_t lane_value = 0;
  if (!loom_checked_add_i64(lower_bound, lane_delta, &lane_value)) {
    return false;
  }
  *out_value = lane_value < upper_bound;
  return true;
}

static iree_status_t loom_vector_mask_range_exact_static_facts(
    loom_fact_context_t* context, uint64_t lane_count, int64_t lower_bound,
    int64_t upper_bound, int64_t step, loom_value_facts_t* out_facts,
    bool* out_handled) {
  *out_handled = true;
  if (lane_count == 0) {
    return loom_value_facts_make_uniform_element(
        context, loom_value_facts_exact_i64(0), out_facts);
  }

  bool first_value = false;
  if (!loom_vector_mask_range_exact_lane(lower_bound, upper_bound, step, 0,
                                         &first_value)) {
    *out_facts = loom_value_facts_unknown();
    return iree_ok_status();
  }

  bool last_value = first_value;
  if (lane_count > 1) {
    if (!loom_vector_mask_range_exact_lane(lower_bound, upper_bound, step,
                                           lane_count - 1, &last_value)) {
      *out_facts = loom_value_facts_unknown();
      return iree_ok_status();
    }
  }

  if (first_value == last_value) {
    return loom_value_facts_make_uniform_element(
        context, loom_value_facts_exact_i64(first_value ? 1 : 0), out_facts);
  }

  if (lane_count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT) {
    *out_handled = false;
    return iree_ok_status();
  }

  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (uint64_t i = 0; i < lane_count; ++i) {
    bool lane_value = false;
    if (!loom_vector_mask_range_exact_lane(lower_bound, upper_bound, step, i,
                                           &lane_value)) {
      *out_facts = loom_value_facts_unknown();
      return iree_ok_status();
    }
    lanes[i] = loom_value_facts_exact_i64(lane_value ? 1 : 0);
  }

  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = (iree_host_size_t)lane_count,
  };
  return loom_value_facts_make_small_static_lanes(context, lane_slice,
                                                  out_facts);
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

static bool loom_vector_facts_query_exact_i64(loom_value_facts_t facts,
                                              int64_t* out_value) {
  if (!loom_value_facts_is_exact(facts) || loom_value_facts_is_float(facts)) {
    return false;
  }
  *out_value = facts.range_lo;
  return true;
}

static bool loom_vector_facts_query_exact_i32(loom_value_facts_t facts,
                                              int32_t* out_value) {
  int64_t value = 0;
  if (!loom_vector_facts_query_exact_i64(facts, &value) || value < INT32_MIN ||
      value > INT32_MAX) {
    return false;
  }
  *out_value = (int32_t)value;
  return true;
}

static int32_t loom_vector_extend_integer_field_i32(int64_t value,
                                                    uint8_t bit_count,
                                                    bool is_signed) {
  if (bit_count == 0) return 0;
  if (bit_count > 32) bit_count = 32;
  uint32_t mask =
      bit_count == 32 ? UINT32_MAX : (((uint32_t)1) << bit_count) - 1;
  uint32_t masked = ((uint32_t)value) & mask;
  if (!is_signed) return (int32_t)masked;
  uint32_t sign_bit = ((uint32_t)1) << (bit_count - 1);
  return (int32_t)((masked ^ sign_bit) - sign_bit);
}

typedef struct loom_vector_grouped_dot_shape_t {
  // Number of logical result lanes.
  iree_host_size_t result_lane_count;
  // Static last-axis extent of each source vector.
  iree_host_size_t source_last_extent;
  // Static last-axis extent of the result vector.
  iree_host_size_t result_last_extent;
} loom_vector_grouped_dot_shape_t;

static bool loom_vector_query_grouped_dot_shape(
    loom_type_t source_type, loom_type_t result_type, uint8_t group_size,
    loom_vector_grouped_dot_shape_t* out_shape) {
  uint8_t rank = loom_type_rank(result_type);
  if (rank == 0 || loom_type_rank(source_type) != rank || group_size == 0) {
    return false;
  }

  uint64_t result_lane_count = 0;
  if (!loom_type_static_element_count(result_type, &result_lane_count) ||
      result_lane_count > (uint64_t)IREE_HOST_SIZE_MAX) {
    return false;
  }

  if (loom_type_dim_is_dynamic_at(source_type, rank - 1) ||
      loom_type_dim_is_dynamic_at(result_type, rank - 1)) {
    return false;
  }
  int64_t source_last_extent =
      loom_type_dim_static_size_at(source_type, rank - 1);
  int64_t result_last_extent =
      loom_type_dim_static_size_at(result_type, rank - 1);
  if (source_last_extent < 0 || result_last_extent < 0 ||
      (uint64_t)source_last_extent > (uint64_t)IREE_HOST_SIZE_MAX ||
      (uint64_t)result_last_extent > (uint64_t)IREE_HOST_SIZE_MAX) {
    return false;
  }

  int64_t expected_source_last_extent = 0;
  if (!iree_checked_mul_i64(result_last_extent, (int64_t)group_size,
                            &expected_source_last_extent) ||
      source_last_extent != expected_source_last_extent) {
    return false;
  }

  for (uint8_t axis = 0; axis + 1 < rank; ++axis) {
    if (loom_type_dim_is_dynamic_at(source_type, axis) ||
        loom_type_dim_is_dynamic_at(result_type, axis) ||
        loom_type_dim_static_size_at(source_type, axis) !=
            loom_type_dim_static_size_at(result_type, axis)) {
      return false;
    }
  }

  *out_shape = (loom_vector_grouped_dot_shape_t){
      .result_lane_count = (iree_host_size_t)result_lane_count,
      .source_last_extent = (iree_host_size_t)source_last_extent,
      .result_last_extent = (iree_host_size_t)result_last_extent,
  };
  return true;
}

static bool loom_vector_grouped_dot_source_lane(
    loom_vector_grouped_dot_shape_t shape, iree_host_size_t result_lane,
    uint8_t group_size, uint8_t group_lane, iree_host_size_t* out_source_lane) {
  if (shape.result_last_extent == 0 || group_lane >= group_size) return false;
  iree_host_size_t leading_lane = result_lane / shape.result_last_extent;
  iree_host_size_t result_last_lane = result_lane % shape.result_last_extent;
  iree_host_size_t source_lane = 0;
  if (!iree_host_size_checked_mul(leading_lane, shape.source_last_extent,
                                  &source_lane)) {
    return false;
  }
  iree_host_size_t source_last_lane = 0;
  if (!iree_host_size_checked_mul(result_last_lane, group_size,
                                  &source_last_lane) ||
      !iree_host_size_checked_add(source_last_lane, group_lane,
                                  &source_last_lane) ||
      !iree_host_size_checked_add(source_lane, source_last_lane,
                                  &source_lane)) {
    return false;
  }
  *out_source_lane = source_lane;
  return true;
}

static bool loom_vector_dot4i_lhs_is_signed(uint8_t kind) {
  return kind == LOOM_VECTOR_DOT4I_KIND_S8S8 ||
         kind == LOOM_VECTOR_DOT4I_KIND_S8U8;
}

static bool loom_vector_dot4i_rhs_is_signed(uint8_t kind) {
  return kind == LOOM_VECTOR_DOT4I_KIND_S8S8 ||
         kind == LOOM_VECTOR_DOT4I_KIND_U8S8;
}

static bool loom_vector_dot8i4_lhs_is_signed(uint8_t kind) {
  return kind == LOOM_VECTOR_DOT8I4_KIND_S4S4 ||
         kind == LOOM_VECTOR_DOT8I4_KIND_S4U4;
}

static bool loom_vector_dot8i4_rhs_is_signed(uint8_t kind) {
  return kind == LOOM_VECTOR_DOT8I4_KIND_S4S4 ||
         kind == LOOM_VECTOR_DOT8I4_KIND_U4S4;
}

static bool loom_vector_dot4i_apply(uint8_t kind, int64_t lhs_raw,
                                    int64_t rhs_raw, int32_t* accumulator) {
  if (kind >= LOOM_VECTOR_DOT4I_KIND_COUNT_) return false;
  int32_t lhs = loom_vector_extend_integer_field_i32(
      lhs_raw, 8, loom_vector_dot4i_lhs_is_signed(kind));
  int32_t rhs = loom_vector_extend_integer_field_i32(
      rhs_raw, 8, loom_vector_dot4i_rhs_is_signed(kind));
  int32_t next = 0;
  if (!iree_checked_mul_add_i32(*accumulator, lhs, rhs, &next)) return false;
  *accumulator = next;
  return true;
}

static bool loom_vector_dot8i4_apply(uint8_t kind, int64_t lhs_raw,
                                     int64_t rhs_raw, int32_t* accumulator) {
  if (kind >= LOOM_VECTOR_DOT8I4_KIND_COUNT_) return false;
  bool lhs_is_signed = loom_vector_dot8i4_lhs_is_signed(kind);
  bool rhs_is_signed = loom_vector_dot8i4_rhs_is_signed(kind);
  for (uint8_t field_ordinal = 0; field_ordinal < 8; ++field_ordinal) {
    uint8_t shift = (uint8_t)(4 * field_ordinal);
    int32_t lhs = loom_vector_extend_integer_field_i32(
        ((uint32_t)lhs_raw) >> shift, 4, lhs_is_signed);
    int32_t rhs = loom_vector_extend_integer_field_i32(
        ((uint32_t)rhs_raw) >> shift, 4, rhs_is_signed);
    int32_t next = 0;
    if (!iree_checked_mul_add_i32(*accumulator, lhs, rhs, &next)) {
      return false;
    }
    *accumulator = next;
  }
  return true;
}

static double loom_vector_add_f64(double lhs, double rhs) { return lhs + rhs; }

static double loom_vector_sub_f64(double lhs, double rhs) { return lhs - rhs; }

static double loom_vector_mul_f64(double lhs, double rhs) { return lhs * rhs; }

static double loom_vector_div_f64(double lhs, double rhs) { return lhs / rhs; }

static double loom_vector_neg_f64(double input) { return -input; }

static double loom_vector_rsqrt_f64(double input) { return 1.0 / sqrt(input); }

static double loom_vector_roundeven_f64(double input) {
  return nearbyint(input);
}

static double loom_vector_minimum_f64(double lhs, double rhs) {
  return (isnan(lhs) || isnan(rhs)) ? NAN : fmin(lhs, rhs);
}

static double loom_vector_maximum_f64(double lhs, double rhs) {
  return (isnan(lhs) || isnan(rhs)) ? NAN : fmax(lhs, rhs);
}

static double loom_vector_minnum_f64(double lhs, double rhs) {
  return fmin(lhs, rhs);
}

static double loom_vector_maxnum_f64(double lhs, double rhs) {
  return fmax(lhs, rhs);
}

static void loom_vector_isnanf_transfer(const loom_value_facts_t* input,
                                        loom_value_facts_t* out) {
  double value = 0.0;
  if (!loom_vector_facts_query_exact_f64(*input, &value)) {
    *out = loom_value_facts_make(0, 1, 1);
    return;
  }
  *out = loom_value_facts_exact_i64(isnan(value) ? 1 : 0);
}

static void loom_vector_isinff_transfer(const loom_value_facts_t* input,
                                        loom_value_facts_t* out) {
  double value = 0.0;
  if (!loom_vector_facts_query_exact_f64(*input, &value)) {
    *out = loom_value_facts_make(0, 1, 1);
    return;
  }
  *out = loom_value_facts_exact_i64(isinf(value) ? 1 : 0);
}

static void loom_vector_isfinitef_transfer(const loom_value_facts_t* input,
                                           loom_value_facts_t* out) {
  double value = 0.0;
  if (!loom_vector_facts_query_exact_f64(*input, &value)) {
    *out = loom_value_facts_make(0, 1, 1);
    return;
  }
  *out = loom_value_facts_exact_i64(isfinite(value) ? 1 : 0);
}

static void loom_vector_signf_transfer(const loom_value_facts_t* input,
                                       loom_value_facts_t* out) {
  double value = 0.0;
  if (!loom_vector_facts_query_exact_f64(*input, &value)) {
    *out = loom_value_facts_unknown();
    return;
  }
  *out = loom_value_facts_exact_f64((value > 0.0)   ? 1.0
                                    : (value < 0.0) ? -1.0
                                                    : 0.0);
}

static void loom_vector_signi_transfer(const loom_value_facts_t* input,
                                       loom_value_facts_t* out) {
  if (!loom_value_facts_is_exact(*input)) {
    *out = loom_value_facts_unknown();
    return;
  }
  int64_t value = input->range_lo;
  *out = loom_value_facts_exact_i64((value > 0) ? 1 : (value < 0) ? -1 : 0);
}

static void loom_vector_fmai_transfer(const loom_value_facts_t* a,
                                      const loom_value_facts_t* b,
                                      const loom_value_facts_t* c,
                                      loom_value_facts_t* out) {
  loom_value_facts_fmai(a, b, c, out);
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
      loom_value_facts_is_exact(operand_facts[2]) &&
      !loom_value_facts_is_float(operand_facts[0]) &&
      !loom_value_facts_is_float(operand_facts[1]) &&
      !loom_value_facts_is_float(operand_facts[2])) {
    loom_type_t result_type =
        loom_module_value_type(module, loom_vector_mask_range_result(op));
    uint64_t lane_count = 0;
    if (loom_type_static_element_count(result_type, &lane_count)) {
      bool handled = false;
      IREE_RETURN_IF_ERROR(loom_vector_mask_range_exact_static_facts(
          context, lane_count, operand_facts[0].range_lo,
          operand_facts[1].range_lo, operand_facts[2].range_lo,
          &result_facts[0], &handled));
      if (handled) return iree_ok_status();
    }

    int64_t lower_bound = operand_facts[0].range_lo;
    int64_t upper_bound = operand_facts[1].range_lo;
    int64_t step = operand_facts[2].range_lo;
    if ((step >= 0 && lower_bound >= upper_bound) ||
        (step <= 0 && lower_bound < upper_bound)) {
      return loom_value_facts_make_uniform_element(
          context,
          loom_value_facts_exact_i64(lower_bound < upper_bound ? 1 : 0),
          &result_facts[0]);
    }
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

static iree_status_t loom_vector_unary_summary_facts(
    loom_fact_context_t* context, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts, loom_vector_unary_transfer_fn_t fn) {
  loom_value_facts_t input = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &input)) {
    loom_value_facts_t element = loom_value_facts_unknown();
    fn(&input, &element);
    return loom_value_facts_make_uniform_element(context, element,
                                                 &result_facts[0]);
  }

  loom_value_fact_small_static_lanes_t input_lanes = {0};
  if (!loom_vector_facts_query_small_lanes(context, operand_facts[0],
                                           &input_lanes)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t i = 0; i < input_lanes.count; ++i) {
    fn(&input_lanes.lanes[i], &lanes[i]);
  }
  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = input_lanes.count,
  };
  return loom_value_facts_make_small_static_lanes(context, lane_slice,
                                                  &result_facts[0]);
}

static void loom_vector_bit_count_element_facts(const loom_value_facts_t* input,
                                                int32_t bitwidth,
                                                loom_vector_bit_count_fn_t fn,
                                                loom_value_facts_t* out) {
  if (!loom_value_facts_is_exact(*input)) {
    *out = loom_value_facts_unknown();
    return;
  }
  *out = loom_value_facts_exact_i64(fn((uint64_t)input->range_lo, bitwidth));
}

static iree_status_t loom_vector_bit_count_summary_facts(
    loom_fact_context_t* context, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts, int32_t bitwidth,
    loom_vector_bit_count_fn_t fn) {
  if (bitwidth <= 0) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }

  loom_value_facts_t input = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &input)) {
    loom_value_facts_t element = loom_value_facts_unknown();
    loom_vector_bit_count_element_facts(&input, bitwidth, fn, &element);
    return loom_value_facts_make_uniform_element(context, element,
                                                 &result_facts[0]);
  }

  loom_value_fact_small_static_lanes_t input_lanes = {0};
  if (!loom_vector_facts_query_small_lanes(context, operand_facts[0],
                                           &input_lanes)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t i = 0; i < input_lanes.count; ++i) {
    loom_vector_bit_count_element_facts(&input_lanes.lanes[i], bitwidth, fn,
                                        &lanes[i]);
  }
  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = input_lanes.count,
  };
  return loom_value_facts_make_small_static_lanes(context, lane_slice,
                                                  &result_facts[0]);
}

static iree_status_t loom_vector_float_unary_summary_facts(
    loom_fact_context_t* context, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts,
    loom_vector_float_unary_transfer_fn_t fn) {
  loom_value_facts_t input = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &input)) {
    double input_value = 0.0;
    loom_value_facts_t element = loom_value_facts_unknown();
    if (loom_vector_facts_query_exact_f64(input, &input_value)) {
      element = loom_value_facts_exact_f64(fn(input_value));
    }
    return loom_value_facts_make_uniform_element(context, element,
                                                 &result_facts[0]);
  }

  loom_value_fact_small_static_lanes_t input_lanes = {0};
  if (!loom_vector_facts_query_small_lanes(context, operand_facts[0],
                                           &input_lanes)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t i = 0; i < input_lanes.count; ++i) {
    double input_value = 0.0;
    lanes[i] = loom_value_facts_unknown();
    if (loom_vector_facts_query_exact_f64(input_lanes.lanes[i], &input_value)) {
      lanes[i] = loom_value_facts_exact_f64(fn(input_value));
    }
  }
  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = input_lanes.count,
  };
  return loom_value_facts_make_small_static_lanes(context, lane_slice,
                                                  &result_facts[0]);
}

static iree_status_t loom_vector_float_binary_summary_facts(
    loom_fact_context_t* context, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts,
    loom_vector_float_binary_transfer_fn_t fn) {
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

static iree_status_t loom_vector_ternary_summary_facts(
    loom_fact_context_t* context, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts, loom_vector_ternary_transfer_fn_t fn) {
  loom_value_facts_t a = {0};
  loom_value_facts_t b = {0};
  loom_value_facts_t c = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0], &a) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[1], &b) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[2], &c)) {
    loom_value_facts_t element = loom_value_facts_unknown();
    fn(&a, &b, &c, &element);
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
    fn(&a, &b, &c, &lanes[i]);
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

#define LOOM_VECTOR_FLOAT_UNARY_FACTS(name, fn)                          \
  iree_status_t name(loom_fact_context_t* context,                       \
                     const loom_module_t* module, const loom_op_t* op,   \
                     const loom_value_facts_t* operand_facts,            \
                     loom_value_facts_t* result_facts) {                 \
    return loom_vector_float_unary_summary_facts(context, operand_facts, \
                                                 result_facts, fn);      \
  }

#define LOOM_VECTOR_UNARY_FACTS(name, fn)                              \
  iree_status_t name(loom_fact_context_t* context,                     \
                     const loom_module_t* module, const loom_op_t* op, \
                     const loom_value_facts_t* operand_facts,          \
                     loom_value_facts_t* result_facts) {               \
    return loom_vector_unary_summary_facts(context, operand_facts,     \
                                           result_facts, fn);          \
  }

#define LOOM_VECTOR_BIT_COUNT_FACTS(name, result_accessor, fn)              \
  iree_status_t name(loom_fact_context_t* context,                          \
                     const loom_module_t* module, const loom_op_t* op,      \
                     const loom_value_facts_t* operand_facts,               \
                     loom_value_facts_t* result_facts) {                    \
    loom_type_t result_type =                                               \
        loom_module_value_type(module, result_accessor(op));                \
    int32_t bitwidth =                                                      \
        loom_scalar_type_bitwidth(loom_type_element_type(result_type));     \
    return loom_vector_bit_count_summary_facts(context, operand_facts,      \
                                               result_facts, bitwidth, fn); \
  }

LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_addf_facts, loom_vector_add_f64)
LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_subf_facts, loom_vector_sub_f64)
LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_mulf_facts, loom_vector_mul_f64)
LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_divf_facts, loom_vector_div_f64)
LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_remf_facts, fmod)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_negf_facts, loom_vector_neg_f64)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_absf_facts, fabs)
LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_minimumf_facts,
                               loom_vector_minimum_f64)
LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_maximumf_facts,
                               loom_vector_maximum_f64)
LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_minnumf_facts,
                               loom_vector_minnum_f64)
LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_maxnumf_facts,
                               loom_vector_maxnum_f64)
LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_copysignf_facts, copysign)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_addi_facts, loom_value_facts_addi)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_subi_facts, loom_value_facts_subi)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_muli_facts, loom_value_facts_muli)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_divsi_facts,
                                 loom_value_facts_divsi)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_divui_facts,
                                 loom_value_facts_divui)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_remsi_facts,
                                 loom_value_facts_remsi)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_remui_facts,
                                 loom_value_facts_remui)
LOOM_VECTOR_UNARY_FACTS(loom_vector_negi_facts, loom_value_facts_negi)
LOOM_VECTOR_UNARY_FACTS(loom_vector_absi_facts, loom_value_facts_absi)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_minsi_facts,
                                 loom_value_facts_minsi)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_maxsi_facts,
                                 loom_value_facts_maxsi)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_minui_facts,
                                 loom_value_facts_minui)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_maxui_facts,
                                 loom_value_facts_maxui)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_andi_facts, loom_value_facts_andi)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_ori_facts, loom_value_facts_ori)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_xori_facts, loom_value_facts_xori)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_shli_facts, loom_value_facts_shli)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_shrsi_facts,
                                 loom_value_facts_shrsi)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_shrui_facts,
                                 loom_value_facts_shrui)
LOOM_VECTOR_BIT_COUNT_FACTS(loom_vector_ctlzi_facts, loom_vector_ctlzi_result,
                            loom_count_leading_zeros_u64_width)
LOOM_VECTOR_BIT_COUNT_FACTS(loom_vector_cttzi_facts, loom_vector_cttzi_result,
                            loom_count_trailing_zeros_u64_width)
LOOM_VECTOR_BIT_COUNT_FACTS(loom_vector_ctpopi_facts, loom_vector_ctpopi_result,
                            loom_count_ones_u64_width)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_expf_facts, exp)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_exp2f_facts, exp2)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_expm1f_facts, expm1)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_logf_facts, log)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_log2f_facts, log2)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_log10f_facts, log10)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_log1pf_facts, log1p)
LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_powf_facts, pow)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_sqrtf_facts, sqrt)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_rsqrtf_facts, loom_vector_rsqrt_f64)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_cbrtf_facts, cbrt)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_sinf_facts, sin)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_cosf_facts, cos)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_tanf_facts, tan)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_asinf_facts, asin)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_acosf_facts, acos)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_atanf_facts, atan)
LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_atan2f_facts, atan2)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_sinhf_facts, sinh)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_coshf_facts, cosh)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_tanhf_facts, tanh)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_asinhf_facts, asinh)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_acoshf_facts, acosh)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_atanhf_facts, atanh)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_erff_facts, erf)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_erfcf_facts, erfc)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_ceilf_facts, ceil)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_floorf_facts, floor)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_roundf_facts, round)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_roundevenf_facts,
                              loom_vector_roundeven_f64)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_truncf_facts, trunc)
LOOM_VECTOR_UNARY_FACTS(loom_vector_isnanf_facts, loom_vector_isnanf_transfer)
LOOM_VECTOR_UNARY_FACTS(loom_vector_isinff_facts, loom_vector_isinff_transfer)
LOOM_VECTOR_UNARY_FACTS(loom_vector_isfinitef_facts,
                        loom_vector_isfinitef_transfer)
LOOM_VECTOR_UNARY_FACTS(loom_vector_signf_facts, loom_vector_signf_transfer)
LOOM_VECTOR_UNARY_FACTS(loom_vector_signi_facts, loom_vector_signi_transfer)

iree_status_t loom_vector_fmai_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  return loom_vector_ternary_summary_facts(context, operand_facts, result_facts,
                                           loom_vector_fmai_transfer);
}

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

static iree_status_t loom_vector_make_unknown_facts(
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_unknown();
  return iree_ok_status();
}

iree_status_t loom_vector_dot2f_facts(loom_fact_context_t* context,
                                      const loom_module_t* module,
                                      const loom_op_t* op,
                                      const loom_value_facts_t* operand_facts,
                                      loom_value_facts_t* result_facts) {
  loom_value_facts_t lhs_element = {0};
  loom_value_facts_t rhs_element = {0};
  loom_value_facts_t acc_element = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &lhs_element) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[1],
                                              &rhs_element) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[2],
                                              &acc_element)) {
    double lhs_value = 0.0;
    double rhs_value = 0.0;
    double accumulator = 0.0;
    if (!loom_vector_facts_query_exact_f64(lhs_element, &lhs_value) ||
        !loom_vector_facts_query_exact_f64(rhs_element, &rhs_value) ||
        !loom_vector_facts_query_exact_f64(acc_element, &accumulator)) {
      return loom_vector_make_unknown_facts(result_facts);
    }
    for (uint8_t group_lane = 0; group_lane < 2; ++group_lane) {
      accumulator = fma(lhs_value, rhs_value, accumulator);
    }
    return loom_value_facts_make_uniform_element(
        context, loom_value_facts_exact_f64(accumulator), &result_facts[0]);
  }

  loom_type_t lhs_type =
      loom_module_value_type(module, loom_vector_dot2f_lhs(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_dot2f_result(op));
  loom_vector_grouped_dot_shape_t shape = {0};
  if (!loom_vector_query_grouped_dot_shape(lhs_type, result_type, 2, &shape) ||
      shape.result_lane_count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT) {
    return loom_vector_make_unknown_facts(result_facts);
  }

  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t result_lane = 0; result_lane < shape.result_lane_count;
       ++result_lane) {
    loom_value_facts_t acc = {0};
    double accumulator = 0.0;
    if (!loom_vector_facts_query_lane(context, operand_facts[2], result_lane,
                                      &acc) ||
        !loom_vector_facts_query_exact_f64(acc, &accumulator)) {
      return loom_vector_make_unknown_facts(result_facts);
    }
    for (uint8_t group_lane = 0; group_lane < 2; ++group_lane) {
      iree_host_size_t source_lane = 0;
      loom_value_facts_t lhs = {0};
      loom_value_facts_t rhs = {0};
      double lhs_value = 0.0;
      double rhs_value = 0.0;
      if (!loom_vector_grouped_dot_source_lane(shape, result_lane, 2,
                                               group_lane, &source_lane) ||
          !loom_vector_facts_query_lane(context, operand_facts[0], source_lane,
                                        &lhs) ||
          !loom_vector_facts_query_lane(context, operand_facts[1], source_lane,
                                        &rhs) ||
          !loom_vector_facts_query_exact_f64(lhs, &lhs_value) ||
          !loom_vector_facts_query_exact_f64(rhs, &rhs_value)) {
        return loom_vector_make_unknown_facts(result_facts);
      }
      accumulator = fma(lhs_value, rhs_value, accumulator);
    }
    lanes[result_lane] = loom_value_facts_exact_f64(accumulator);
  }

  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = shape.result_lane_count,
  };
  return loom_value_facts_make_small_static_lanes(context, lane_slice,
                                                  &result_facts[0]);
}

iree_status_t loom_vector_dot4i_facts(loom_fact_context_t* context,
                                      const loom_module_t* module,
                                      const loom_op_t* op,
                                      const loom_value_facts_t* operand_facts,
                                      loom_value_facts_t* result_facts) {
  uint8_t kind = loom_vector_dot4i_kind(op);
  loom_value_facts_t lhs_element = {0};
  loom_value_facts_t rhs_element = {0};
  loom_value_facts_t acc_element = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &lhs_element) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[1],
                                              &rhs_element) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[2],
                                              &acc_element)) {
    int64_t lhs_value = 0;
    int64_t rhs_value = 0;
    int32_t accumulator = 0;
    if (!loom_vector_facts_query_exact_i64(lhs_element, &lhs_value) ||
        !loom_vector_facts_query_exact_i64(rhs_element, &rhs_value) ||
        !loom_vector_facts_query_exact_i32(acc_element, &accumulator)) {
      return loom_vector_make_unknown_facts(result_facts);
    }
    for (uint8_t group_lane = 0; group_lane < 4; ++group_lane) {
      if (!loom_vector_dot4i_apply(kind, lhs_value, rhs_value, &accumulator)) {
        return loom_vector_make_unknown_facts(result_facts);
      }
    }
    return loom_value_facts_make_uniform_element(
        context, loom_value_facts_exact_i64(accumulator), &result_facts[0]);
  }

  loom_type_t lhs_type =
      loom_module_value_type(module, loom_vector_dot4i_lhs(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_dot4i_result(op));
  loom_vector_grouped_dot_shape_t shape = {0};
  if (!loom_vector_query_grouped_dot_shape(lhs_type, result_type, 4, &shape) ||
      shape.result_lane_count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT) {
    return loom_vector_make_unknown_facts(result_facts);
  }

  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t result_lane = 0; result_lane < shape.result_lane_count;
       ++result_lane) {
    loom_value_facts_t acc = {0};
    int32_t accumulator = 0;
    if (!loom_vector_facts_query_lane(context, operand_facts[2], result_lane,
                                      &acc) ||
        !loom_vector_facts_query_exact_i32(acc, &accumulator)) {
      return loom_vector_make_unknown_facts(result_facts);
    }
    for (uint8_t group_lane = 0; group_lane < 4; ++group_lane) {
      iree_host_size_t source_lane = 0;
      loom_value_facts_t lhs = {0};
      loom_value_facts_t rhs = {0};
      int64_t lhs_value = 0;
      int64_t rhs_value = 0;
      if (!loom_vector_grouped_dot_source_lane(shape, result_lane, 4,
                                               group_lane, &source_lane) ||
          !loom_vector_facts_query_lane(context, operand_facts[0], source_lane,
                                        &lhs) ||
          !loom_vector_facts_query_lane(context, operand_facts[1], source_lane,
                                        &rhs) ||
          !loom_vector_facts_query_exact_i64(lhs, &lhs_value) ||
          !loom_vector_facts_query_exact_i64(rhs, &rhs_value) ||
          !loom_vector_dot4i_apply(kind, lhs_value, rhs_value, &accumulator)) {
        return loom_vector_make_unknown_facts(result_facts);
      }
    }
    lanes[result_lane] = loom_value_facts_exact_i64(accumulator);
  }

  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = shape.result_lane_count,
  };
  return loom_value_facts_make_small_static_lanes(context, lane_slice,
                                                  &result_facts[0]);
}

iree_status_t loom_vector_dot8i4_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  uint8_t kind = loom_vector_dot8i4_kind(op);
  loom_value_facts_t lhs_element = {0};
  loom_value_facts_t rhs_element = {0};
  loom_value_facts_t acc_element = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &lhs_element) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[1],
                                              &rhs_element) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[2],
                                              &acc_element)) {
    int64_t lhs_value = 0;
    int64_t rhs_value = 0;
    int32_t accumulator = 0;
    if (!loom_vector_facts_query_exact_i64(lhs_element, &lhs_value) ||
        !loom_vector_facts_query_exact_i64(rhs_element, &rhs_value) ||
        !loom_vector_facts_query_exact_i32(acc_element, &accumulator) ||
        !loom_vector_dot8i4_apply(kind, lhs_value, rhs_value, &accumulator)) {
      return loom_vector_make_unknown_facts(result_facts);
    }
    return loom_value_facts_make_uniform_element(
        context, loom_value_facts_exact_i64(accumulator), &result_facts[0]);
  }

  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_dot8i4_result(op));
  uint64_t result_lane_count = 0;
  if (!loom_type_static_element_count(result_type, &result_lane_count) ||
      result_lane_count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT) {
    return loom_vector_make_unknown_facts(result_facts);
  }

  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t result_lane = 0;
       result_lane < (iree_host_size_t)result_lane_count; ++result_lane) {
    loom_value_facts_t lhs = {0};
    loom_value_facts_t rhs = {0};
    loom_value_facts_t acc = {0};
    int64_t lhs_value = 0;
    int64_t rhs_value = 0;
    int32_t accumulator = 0;
    if (!loom_vector_facts_query_lane(context, operand_facts[0], result_lane,
                                      &lhs) ||
        !loom_vector_facts_query_lane(context, operand_facts[1], result_lane,
                                      &rhs) ||
        !loom_vector_facts_query_lane(context, operand_facts[2], result_lane,
                                      &acc) ||
        !loom_vector_facts_query_exact_i64(lhs, &lhs_value) ||
        !loom_vector_facts_query_exact_i64(rhs, &rhs_value) ||
        !loom_vector_facts_query_exact_i32(acc, &accumulator) ||
        !loom_vector_dot8i4_apply(kind, lhs_value, rhs_value, &accumulator)) {
      return loom_vector_make_unknown_facts(result_facts);
    }
    lanes[result_lane] = loom_value_facts_exact_i64(accumulator);
  }

  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = (iree_host_size_t)result_lane_count,
  };
  return loom_value_facts_make_small_static_lanes(context, lane_slice,
                                                  &result_facts[0]);
}

#undef LOOM_VECTOR_FLOAT_BINARY_FACTS
#undef LOOM_VECTOR_FLOAT_UNARY_FACTS
#undef LOOM_VECTOR_BIT_COUNT_FACTS
#undef LOOM_VECTOR_INTEGER_BINARY_FACTS
#undef LOOM_VECTOR_UNARY_FACTS
