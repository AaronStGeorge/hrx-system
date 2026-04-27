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

static loom_value_facts_t loom_kernel_hal_coordinate_facts(void) {
  return loom_value_facts_make(0, (int64_t)UINT32_MAX, 1);
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
  IREE_ASSERT_ARGUMENT(bundle->snapshot);
  IREE_ASSERT_ARGUMENT(bundle->export_plan);
  return bundle;
}

static bool loom_kernel_context_required_workgroup_size(
    const loom_fact_context_t* context,
    loom_target_workgroup_size_t* out_size) {
  *out_size = (loom_target_workgroup_size_t){0};

  uint32_t x = 0;
  uint32_t y = 0;
  uint32_t z = 0;
  if (context && loom_kernel_def_isa(context->function.op) &&
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
