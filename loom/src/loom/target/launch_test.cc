// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/launch.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

loom_target_snapshot_t TestSnapshot() {
  return loom_target_snapshot_t{
      .name = IREE_SV("test-target"),
      .max_workgroup_size = {.x = 1024, .y = 1024, .z = 1024},
      .max_flat_workgroup_size = 1024,
  };
}

loom_target_hal_kernel_abi_t TestHalKernelAbi(
    loom_target_workgroup_size_t required_workgroup_size) {
  return loom_target_hal_kernel_abi_t{
      .binding_alignment = 16,
      .required_workgroup_size = required_workgroup_size,
  };
}

TEST(TargetLaunchTest, AllowsDynamicRequiredWorkgroupSize) {
  loom_target_snapshot_t snapshot = TestSnapshot();
  loom_target_hal_kernel_abi_t hal_kernel =
      TestHalKernelAbi({.x = 0, .y = 0, .z = 0});
  IREE_EXPECT_OK(
      loom_target_validate_hal_kernel_launch(&snapshot, &hal_kernel));
}

TEST(TargetLaunchTest, RejectsPartialRequiredWorkgroupSize) {
  loom_target_snapshot_t snapshot = TestSnapshot();
  loom_target_hal_kernel_abi_t hal_kernel =
      TestHalKernelAbi({.x = 64, .y = 1, .z = 0});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_validate_hal_kernel_launch(&snapshot, &hal_kernel));
}

TEST(TargetLaunchTest, RejectsRequiredWorkgroupSizeDimensionAboveLimit) {
  loom_target_snapshot_t snapshot = TestSnapshot();
  loom_target_hal_kernel_abi_t hal_kernel =
      TestHalKernelAbi({.x = 1025, .y = 1, .z = 1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_validate_hal_kernel_launch(&snapshot, &hal_kernel));
}

TEST(TargetLaunchTest, RejectsRequiredFlatWorkgroupSizeAboveLimit) {
  loom_target_snapshot_t snapshot = TestSnapshot();
  loom_target_hal_kernel_abi_t hal_kernel =
      TestHalKernelAbi({.x = 33, .y = 33, .z = 1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_validate_hal_kernel_launch(&snapshot, &hal_kernel));
}

TEST(TargetLaunchTest, ValidatesSelectedFlatWorkgroupRange) {
  loom_target_snapshot_t snapshot = TestSnapshot();
  loom_target_hal_kernel_abi_t hal_kernel =
      TestHalKernelAbi({.x = 64, .y = 2, .z = 1});
  hal_kernel.flat_workgroup_size_min = 64;
  hal_kernel.flat_workgroup_size_max = 256;
  IREE_EXPECT_OK(
      loom_target_validate_hal_kernel_launch(&snapshot, &hal_kernel));

  hal_kernel.flat_workgroup_size_min = 256;
  hal_kernel.flat_workgroup_size_max = 512;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_validate_hal_kernel_launch(&snapshot, &hal_kernel));
}

TEST(TargetLaunchTest, RequiresConcreteLaunchForConcreteConsumers) {
  loom_target_hal_kernel_abi_t hal_kernel =
      TestHalKernelAbi({.x = 0, .y = 0, .z = 0});
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_target_require_concrete_hal_kernel_launch(
                            &hal_kernel, IREE_SV("test consumer")));

  hal_kernel.required_workgroup_size =
      loom_target_workgroup_size_t{.x = 64, .y = 1, .z = 0};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_target_require_concrete_hal_kernel_launch(
                            &hal_kernel, IREE_SV("test consumer")));

  hal_kernel.required_workgroup_size =
      loom_target_workgroup_size_t{.x = 64, .y = 1, .z = 1};
  IREE_EXPECT_OK(loom_target_require_concrete_hal_kernel_launch(
      &hal_kernel, IREE_SV("test consumer")));
}

}  // namespace
}  // namespace loom
