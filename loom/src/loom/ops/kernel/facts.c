// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fact implementations for the kernel dialect.

#include <stdint.h>

#include "loom/ops/kernel/ops.h"
#include "loom/target/types.h"
#include "loom/util/fact_table.h"

#define LOOM_KERNEL_DEFAULT_MAX_SUBGROUP_SIZE 128u

static loom_value_facts_t loom_kernel_hal_coordinate_facts(void) {
  return loom_value_facts_make(0, (int64_t)UINT32_MAX, 1);
}

static loom_value_facts_t loom_kernel_hal_positive_u32_facts(void) {
  return loom_value_facts_make(1, (int64_t)UINT32_MAX, 1);
}

static uint32_t loom_kernel_workgroup_size_dim(
    const loom_target_workgroup_size_t* size,
    loom_kernel_dimension_t dimension) {
  switch (dimension) {
    case LOOM_KERNEL_DIMENSION_X:
      return size->x;
    case LOOM_KERNEL_DIMENSION_Y:
      return size->y;
    case LOOM_KERNEL_DIMENSION_Z:
      return size->z;
    default:
      return 0;
  }
}

static uint32_t loom_kernel_grid_size_dim(const loom_target_grid_size_t* size,
                                          loom_kernel_dimension_t dimension) {
  switch (dimension) {
    case LOOM_KERNEL_DIMENSION_X:
      return size->x;
    case LOOM_KERNEL_DIMENSION_Y:
      return size->y;
    case LOOM_KERNEL_DIMENSION_Z:
      return size->z;
    default:
      return 0;
  }
}

static uint32_t loom_kernel_workgroup_count_dim(
    const loom_target_workgroup_count_limit_t* size,
    loom_kernel_dimension_t dimension) {
  switch (dimension) {
    case LOOM_KERNEL_DIMENSION_X:
      return size->x;
    case LOOM_KERNEL_DIMENSION_Y:
      return size->y;
    case LOOM_KERNEL_DIMENSION_Z:
      return size->z;
    default:
      return 0;
  }
}

static bool loom_kernel_workgroup_size_flat_product(
    const loom_target_workgroup_size_t* size, uint32_t* out_flat_size) {
  if (size->x == 0 || size->y == 0 || size->z == 0) {
    return false;
  }
  const uint64_t flat_size = (uint64_t)size->x * size->y * size->z;
  if (flat_size > UINT32_MAX) {
    return false;
  }
  *out_flat_size = (uint32_t)flat_size;
  return true;
}

static uint32_t loom_kernel_ceil_div_u32(uint32_t numerator,
                                         uint32_t denominator) {
  IREE_ASSERT_NE(denominator, 0u);
  return numerator == 0 ? 0 : 1u + (numerator - 1u) / denominator;
}

static uint32_t loom_kernel_max_workgroup_count(
    const loom_target_snapshot_t* snapshot,
    const loom_target_workgroup_size_t* required_workgroup_size,
    loom_kernel_dimension_t dimension) {
  uint32_t max_count = loom_kernel_workgroup_count_dim(
      &snapshot->max_workgroup_count, dimension);
  const uint32_t max_grid_size =
      loom_kernel_grid_size_dim(&snapshot->max_grid_size, dimension);
  if (max_grid_size != 0) {
    uint32_t workgroup_size =
        loom_kernel_workgroup_size_dim(required_workgroup_size, dimension);
    if (workgroup_size == 0) {
      workgroup_size = 1;
    }
    const uint32_t grid_limited_count = max_grid_size / workgroup_size;
    max_count = max_count == 0 ? grid_limited_count
                               : iree_min(max_count, grid_limited_count);
  }
  return max_count;
}

static const loom_target_bundle_t* loom_kernel_target_bundle(
    const loom_fact_context_t* context) {
  IREE_ASSERT_ARGUMENT(context);
  const loom_target_bundle_t* bundle = context->target_bundle;
  if (!bundle) {
    return NULL;
  }
  IREE_ASSERT(bundle->snapshot != NULL);
  IREE_ASSERT(bundle->export_plan != NULL);
  return bundle;
}

static uint32_t loom_kernel_context_fixed_subgroup_size(
    const loom_fact_context_t* context) {
  const loom_target_bundle_t* bundle = loom_kernel_target_bundle(context);
  if (!bundle || bundle->export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return 0;
  }
  return bundle->snapshot->subgroup_size;
}

static bool loom_kernel_context_required_workgroup_size(
    const loom_fact_context_t* context,
    loom_target_workgroup_size_t* out_size) {
  *out_size = (loom_target_workgroup_size_t){0};

  uint32_t x = 0;
  uint32_t y = 0;
  uint32_t z = 0;
  if (loom_func_like_isa(context->function) &&
      loom_kernel_def_isa(context->function.op) &&
      loom_func_like_workgroup_size(context->function, &x, &y, &z)) {
    *out_size = (loom_target_workgroup_size_t){.x = x, .y = y, .z = z};
    return true;
  }

  const loom_target_bundle_t* bundle = loom_kernel_target_bundle(context);
  if (!bundle || bundle->export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return false;
  }
  *out_size = bundle->export_plan->hal_kernel.required_workgroup_size;
  return out_size->x != 0 || out_size->y != 0 || out_size->z != 0;
}

static bool loom_kernel_context_fixed_flat_workgroup_size(
    const loom_fact_context_t* context, uint32_t* out_flat_size) {
  IREE_ASSERT_ARGUMENT(out_flat_size);
  *out_flat_size = 0;
  loom_target_workgroup_size_t required_workgroup_size = {0};
  if (!loom_kernel_context_required_workgroup_size(context,
                                                   &required_workgroup_size)) {
    return false;
  }
  return loom_kernel_workgroup_size_flat_product(&required_workgroup_size,
                                                 out_flat_size);
}

static uint32_t loom_kernel_context_max_flat_workgroup_size(
    const loom_fact_context_t* context) {
  uint32_t flat_size = 0;
  if (loom_kernel_context_fixed_flat_workgroup_size(context, &flat_size)) {
    return flat_size;
  }

  const loom_target_bundle_t* bundle = loom_kernel_target_bundle(context);
  if (!bundle || bundle->export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return 0;
  }
  if (bundle->snapshot->max_flat_workgroup_size != 0) {
    return bundle->snapshot->max_flat_workgroup_size;
  }
  if (loom_kernel_workgroup_size_flat_product(
          &bundle->snapshot->max_workgroup_size, &flat_size)) {
    return flat_size;
  }
  return 0;
}

static loom_value_facts_t loom_kernel_workitem_id_target_facts(
    const loom_fact_context_t* context, loom_kernel_dimension_t dimension) {
  loom_target_workgroup_size_t required_workgroup_size = {0};
  if (loom_kernel_context_required_workgroup_size(context,
                                                  &required_workgroup_size)) {
    const uint32_t fixed_workgroup_size =
        loom_kernel_workgroup_size_dim(&required_workgroup_size, dimension);
    if (fixed_workgroup_size != 0) {
      return loom_value_facts_make(0, (int64_t)fixed_workgroup_size - 1, 1);
    }
  }

  const loom_target_bundle_t* bundle = loom_kernel_target_bundle(context);
  if (!bundle || bundle->export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return loom_kernel_hal_coordinate_facts();
  }
  const uint32_t max_workgroup_size = loom_kernel_workgroup_size_dim(
      &bundle->snapshot->max_workgroup_size, dimension);
  if (max_workgroup_size != 0) {
    return loom_value_facts_make(0, (int64_t)max_workgroup_size - 1, 1);
  }
  return loom_kernel_hal_coordinate_facts();
}

static loom_value_facts_t loom_kernel_workgroup_id_target_facts(
    const loom_fact_context_t* context, loom_kernel_dimension_t dimension) {
  const loom_target_bundle_t* bundle = loom_kernel_target_bundle(context);
  if (!bundle || bundle->export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return loom_kernel_hal_coordinate_facts();
  }
  loom_target_workgroup_size_t required_workgroup_size = {0};
  if (!loom_kernel_context_required_workgroup_size(context,
                                                   &required_workgroup_size)) {
    required_workgroup_size =
        bundle->export_plan->hal_kernel.required_workgroup_size;
  }
  const uint32_t max_workgroup_count = loom_kernel_max_workgroup_count(
      bundle->snapshot, &required_workgroup_size, dimension);
  if (max_workgroup_count == 0) {
    return loom_kernel_hal_coordinate_facts();
  }
  return loom_value_facts_make(0, (int64_t)max_workgroup_count - 1, 1);
}

static loom_value_facts_t loom_kernel_workgroup_size_target_facts(
    const loom_fact_context_t* context, loom_kernel_dimension_t dimension) {
  loom_target_workgroup_size_t required_workgroup_size = {0};
  if (loom_kernel_context_required_workgroup_size(context,
                                                  &required_workgroup_size)) {
    const uint32_t fixed_workgroup_size =
        loom_kernel_workgroup_size_dim(&required_workgroup_size, dimension);
    if (fixed_workgroup_size != 0) {
      return loom_value_facts_exact_i64((int64_t)fixed_workgroup_size);
    }
  }

  const loom_target_bundle_t* bundle = loom_kernel_target_bundle(context);
  if (!bundle || bundle->export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return loom_kernel_hal_positive_u32_facts();
  }
  const uint32_t max_workgroup_size = loom_kernel_workgroup_size_dim(
      &bundle->snapshot->max_workgroup_size, dimension);
  if (max_workgroup_size != 0) {
    return loom_value_facts_make(1, (int64_t)max_workgroup_size, 1);
  }
  return loom_kernel_hal_positive_u32_facts();
}

static loom_value_facts_t loom_kernel_workgroup_count_target_facts(
    const loom_fact_context_t* context, loom_kernel_dimension_t dimension) {
  const loom_target_bundle_t* bundle = loom_kernel_target_bundle(context);
  if (!bundle || bundle->export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return loom_kernel_hal_positive_u32_facts();
  }
  loom_target_workgroup_size_t required_workgroup_size = {0};
  if (!loom_kernel_context_required_workgroup_size(context,
                                                   &required_workgroup_size)) {
    required_workgroup_size =
        bundle->export_plan->hal_kernel.required_workgroup_size;
  }
  const uint32_t max_workgroup_count = loom_kernel_max_workgroup_count(
      bundle->snapshot, &required_workgroup_size, dimension);
  if (max_workgroup_count == 0) {
    return loom_kernel_hal_positive_u32_facts();
  }
  return loom_value_facts_make(1, (int64_t)max_workgroup_count, 1);
}

static loom_value_facts_t loom_kernel_workitem_dispatch_id_target_facts(
    const loom_fact_context_t* context, loom_kernel_dimension_t dimension) {
  const loom_target_bundle_t* bundle = loom_kernel_target_bundle(context);
  if (!bundle || bundle->export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return loom_kernel_hal_coordinate_facts();
  }
  const uint32_t max_grid_size =
      loom_kernel_grid_size_dim(&bundle->snapshot->max_grid_size, dimension);
  if (max_grid_size != 0) {
    return loom_value_facts_make(0, (int64_t)max_grid_size - 1, 1);
  }

  loom_target_workgroup_size_t workgroup_size = {0};
  if (!loom_kernel_context_required_workgroup_size(context, &workgroup_size)) {
    workgroup_size = bundle->snapshot->max_workgroup_size;
  }
  const uint32_t size =
      loom_kernel_workgroup_size_dim(&workgroup_size, dimension);
  const uint32_t count = loom_kernel_workgroup_count_dim(
      &bundle->snapshot->max_workgroup_count, dimension);
  if (size == 0 || count == 0) {
    return loom_kernel_hal_coordinate_facts();
  }
  const uint64_t max_dispatch_size = (uint64_t)size * count;
  if (max_dispatch_size == 0 || max_dispatch_size > UINT32_MAX) {
    return loom_kernel_hal_coordinate_facts();
  }
  return loom_value_facts_make(0, (int64_t)max_dispatch_size - 1, 1);
}

static uint32_t loom_kernel_max_subgroup_size(
    const loom_fact_context_t* context) {
  const uint32_t fixed_subgroup_size =
      loom_kernel_context_fixed_subgroup_size(context);
  if (fixed_subgroup_size != 0) {
    return fixed_subgroup_size;
  }
  uint32_t max_subgroup_size = LOOM_KERNEL_DEFAULT_MAX_SUBGROUP_SIZE;
  const uint32_t max_flat_workgroup_size =
      loom_kernel_context_max_flat_workgroup_size(context);
  if (max_flat_workgroup_size != 0) {
    max_subgroup_size = iree_min(max_subgroup_size, max_flat_workgroup_size);
  }
  return max_subgroup_size;
}

static uint32_t loom_kernel_max_subgroup_lane_count(
    const loom_fact_context_t* context) {
  uint32_t max_lane_count = loom_kernel_max_subgroup_size(context);
  const uint32_t max_flat_workgroup_size =
      loom_kernel_context_max_flat_workgroup_size(context);
  if (max_flat_workgroup_size != 0) {
    max_lane_count = iree_min(max_lane_count, max_flat_workgroup_size);
  }
  return max_lane_count;
}

static uint32_t loom_kernel_max_subgroup_count(
    const loom_fact_context_t* context) {
  const uint32_t max_flat_workgroup_size =
      loom_kernel_context_max_flat_workgroup_size(context);
  if (max_flat_workgroup_size == 0) {
    return 0;
  }
  const uint32_t fixed_subgroup_size =
      loom_kernel_context_fixed_subgroup_size(context);
  if (fixed_subgroup_size != 0) {
    return loom_kernel_ceil_div_u32(max_flat_workgroup_size,
                                    fixed_subgroup_size);
  }
  return max_flat_workgroup_size;
}

static bool loom_kernel_exact_subgroup_count(const loom_fact_context_t* context,
                                             uint32_t* out_count) {
  IREE_ASSERT_ARGUMENT(out_count);
  *out_count = 0;
  const uint32_t fixed_subgroup_size =
      loom_kernel_context_fixed_subgroup_size(context);
  if (fixed_subgroup_size == 0) {
    return false;
  }
  uint32_t flat_workgroup_size = 0;
  if (!loom_kernel_context_fixed_flat_workgroup_size(context,
                                                     &flat_workgroup_size)) {
    return false;
  }
  *out_count =
      loom_kernel_ceil_div_u32(flat_workgroup_size, fixed_subgroup_size);
  return *out_count != 0;
}

static uint32_t loom_kernel_min_subgroup_count(
    const loom_fact_context_t* context) {
  uint32_t flat_workgroup_size = 0;
  if (!loom_kernel_context_fixed_flat_workgroup_size(context,
                                                     &flat_workgroup_size)) {
    return 1;
  }
  const uint32_t max_subgroup_size = loom_kernel_max_subgroup_size(context);
  if (max_subgroup_size == 0) {
    return 1;
  }
  const uint32_t min_count =
      loom_kernel_ceil_div_u32(flat_workgroup_size, max_subgroup_size);
  return min_count == 0 ? 1 : min_count;
}

iree_status_t loom_kernel_workitem_id_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)module;
  (void)operand_facts;
  result_facts[0] = loom_kernel_workitem_id_target_facts(
      context, loom_kernel_workitem_id_dimension(op));
  return iree_ok_status();
}

iree_status_t loom_kernel_workgroup_id_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)module;
  (void)operand_facts;
  result_facts[0] = loom_kernel_workgroup_id_target_facts(
      context, loom_kernel_workgroup_id_dimension(op));
  return iree_ok_status();
}

iree_status_t loom_kernel_workgroup_size_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)module;
  (void)operand_facts;
  result_facts[0] = loom_kernel_workgroup_size_target_facts(
      context, loom_kernel_workgroup_size_dimension(op));
  return iree_ok_status();
}

iree_status_t loom_kernel_workgroup_count_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)module;
  (void)operand_facts;
  result_facts[0] = loom_kernel_workgroup_count_target_facts(
      context, loom_kernel_workgroup_count_dimension(op));
  return iree_ok_status();
}

iree_status_t loom_kernel_workitem_dispatch_id_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)module;
  (void)operand_facts;
  result_facts[0] = loom_kernel_workitem_dispatch_id_target_facts(
      context, loom_kernel_workitem_dispatch_id_dimension(op));
  return iree_ok_status();
}

iree_status_t loom_kernel_subgroup_lane_id_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)module;
  (void)op;
  (void)operand_facts;
  const uint32_t max_lane_count = loom_kernel_max_subgroup_lane_count(context);
  result_facts[0] = loom_value_facts_make(0, (int64_t)max_lane_count - 1, 1);
  return iree_ok_status();
}

iree_status_t loom_kernel_subgroup_size_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)module;
  (void)op;
  (void)operand_facts;
  const uint32_t fixed_subgroup_size =
      loom_kernel_context_fixed_subgroup_size(context);
  if (fixed_subgroup_size != 0) {
    result_facts[0] = loom_value_facts_exact_i64((int64_t)fixed_subgroup_size);
    return iree_ok_status();
  }
  const uint32_t max_subgroup_size = loom_kernel_max_subgroup_size(context);
  result_facts[0] = loom_value_facts_make(1, (int64_t)max_subgroup_size, 1);
  return iree_ok_status();
}

iree_status_t loom_kernel_subgroup_id_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)module;
  (void)op;
  (void)operand_facts;
  uint32_t exact_count = 0;
  if (loom_kernel_exact_subgroup_count(context, &exact_count)) {
    result_facts[0] = loom_value_facts_make(0, (int64_t)exact_count - 1, 1);
    return iree_ok_status();
  }
  const uint32_t max_subgroup_count = loom_kernel_max_subgroup_count(context);
  if (max_subgroup_count == 0) {
    result_facts[0] = loom_kernel_hal_coordinate_facts();
  } else {
    result_facts[0] =
        loom_value_facts_make(0, (int64_t)max_subgroup_count - 1, 1);
  }
  return iree_ok_status();
}

iree_status_t loom_kernel_subgroup_count_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)module;
  (void)op;
  (void)operand_facts;
  uint32_t exact_count = 0;
  if (loom_kernel_exact_subgroup_count(context, &exact_count)) {
    result_facts[0] = loom_value_facts_exact_i64((int64_t)exact_count);
    return iree_ok_status();
  }
  const uint32_t max_subgroup_count = loom_kernel_max_subgroup_count(context);
  if (max_subgroup_count == 0) {
    result_facts[0] = loom_kernel_hal_positive_u32_facts();
  } else {
    uint32_t min_subgroup_count = loom_kernel_min_subgroup_count(context);
    if (min_subgroup_count > max_subgroup_count) {
      min_subgroup_count = max_subgroup_count;
    }
    result_facts[0] = loom_value_facts_make((int64_t)min_subgroup_count,
                                            (int64_t)max_subgroup_count, 1);
  }
  return iree_ok_status();
}
