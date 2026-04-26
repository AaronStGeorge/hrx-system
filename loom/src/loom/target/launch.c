// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/launch.h"

#include <inttypes.h>

bool loom_target_workgroup_size_is_empty(
    const loom_target_workgroup_size_t* size) {
  IREE_ASSERT_ARGUMENT(size);
  return size->x == 0 && size->y == 0 && size->z == 0;
}

bool loom_target_workgroup_size_is_concrete(
    const loom_target_workgroup_size_t* size) {
  IREE_ASSERT_ARGUMENT(size);
  return size->x != 0 && size->y != 0 && size->z != 0;
}

bool loom_target_workgroup_size_is_partial(
    const loom_target_workgroup_size_t* size) {
  IREE_ASSERT_ARGUMENT(size);
  return !loom_target_workgroup_size_is_empty(size) &&
         !loom_target_workgroup_size_is_concrete(size);
}

uint64_t loom_target_workgroup_size_flat_product(
    const loom_target_workgroup_size_t* size) {
  IREE_ASSERT_ARGUMENT(size);
  return (uint64_t)size->x * size->y * size->z;
}

static iree_status_t loom_target_validate_required_workgroup_size(
    const loom_target_workgroup_size_t* limit,
    const loom_target_workgroup_size_t* required) {
  if (limit->x != 0 && required->x > limit->x) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL kernel required workgroup x size %" PRIu32
                            " exceeds target limit %" PRIu32,
                            required->x, limit->x);
  }
  if (limit->y != 0 && required->y > limit->y) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL kernel required workgroup y size %" PRIu32
                            " exceeds target limit %" PRIu32,
                            required->y, limit->y);
  }
  if (limit->z != 0 && required->z > limit->z) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL kernel required workgroup z size %" PRIu32
                            " exceeds target limit %" PRIu32,
                            required->z, limit->z);
  }
  return iree_ok_status();
}

iree_status_t loom_target_validate_hal_kernel_launch(
    const loom_target_snapshot_t* snapshot,
    const loom_target_hal_kernel_abi_t* hal_kernel) {
  IREE_ASSERT_ARGUMENT(snapshot);
  IREE_ASSERT_ARGUMENT(hal_kernel);
  const loom_target_workgroup_size_t* required =
      &hal_kernel->required_workgroup_size;
  if (loom_target_workgroup_size_is_partial(required)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL kernel required workgroup size must be all zero or all non-zero");
  }
  const uint32_t max_flat_workgroup_size = snapshot->max_flat_workgroup_size;
  const uint32_t flat_min = hal_kernel->flat_workgroup_size_min;
  const uint32_t flat_max = hal_kernel->flat_workgroup_size_max;
  if ((flat_min == 0) != (flat_max == 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL kernel flat workgroup size range must be both zero or both "
        "non-zero");
  }
  if (flat_min != 0) {
    if (flat_min > flat_max) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "HAL kernel flat workgroup size range must be ordered");
    }
    if (max_flat_workgroup_size != 0 && flat_max > max_flat_workgroup_size) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HAL kernel flat workgroup size max %" PRIu32
                              " exceeds target limit %" PRIu32,
                              flat_max, max_flat_workgroup_size);
    }
  }
  if (loom_target_workgroup_size_is_empty(required)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_target_validate_required_workgroup_size(
      &snapshot->max_workgroup_size, required));

  const uint64_t flat_size = loom_target_workgroup_size_flat_product(required);
  if (max_flat_workgroup_size != 0 && flat_size > max_flat_workgroup_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL kernel required flat workgroup size %" PRIu64
                            " exceeds target limit %" PRIu32,
                            flat_size, max_flat_workgroup_size);
  }
  if (flat_min != 0) {
    if (flat_size < flat_min || flat_size > flat_max) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HAL kernel required flat workgroup size %" PRIu64
                              " must be inside selected range %" PRIu32
                              "..%" PRIu32,
                              flat_size, flat_min, flat_max);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_target_require_concrete_hal_kernel_launch(
    const loom_target_hal_kernel_abi_t* hal_kernel,
    iree_string_view_t consumer_name) {
  IREE_ASSERT_ARGUMENT(hal_kernel);
  if (loom_target_workgroup_size_is_partial(
          &hal_kernel->required_workgroup_size)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL kernel required workgroup size must be all zero or all non-zero");
  }
  if (loom_target_workgroup_size_is_concrete(
          &hal_kernel->required_workgroup_size)) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "%.*s requires a selected HAL kernel workgroup size",
                          (int)consumer_name.size, consumer_name.data);
}
