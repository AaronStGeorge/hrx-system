// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device_spec_builder.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

TEST(DeviceSpecTest, EncodesAndDecodesPayload) {
  iree_hal_amdgpu_device_spec_t source = {
      /*.gfx_major=*/11,
      /*.gfx_minor=*/0,
      /*.gfx_stepping=*/0,
      /*.wavefront_size=*/64,
      /*.simd_per_compute_unit=*/4,
      /*.compute_unit_count=*/60,
      /*.maximum_waves_per_compute_unit=*/32,
      /*.local_memory_per_compute_unit=*/64 * 1024,
      /*.flags=*/IREE_HAL_AMDGPU_DEVICE_SPEC_FLAG_NONE,
  };
  std::vector<uint8_t> payload_storage(
      iree_hal_amdgpu_device_spec_payload_size());
  IREE_ASSERT_OK(iree_hal_amdgpu_device_spec_encode(
      &source,
      iree_make_byte_span(payload_storage.data(), payload_storage.size())));

  iree_hal_amdgpu_device_spec_t decoded = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_device_spec_decode(
      iree_make_const_byte_span(payload_storage.data(), payload_storage.size()),
      &decoded));
  EXPECT_EQ(decoded.gfx_major, 11);
  EXPECT_EQ(decoded.wavefront_size, 64);
  EXPECT_EQ(decoded.compute_unit_count, 60);
}

TEST(DeviceSpecTest, AddsAndFindsCoreFacet) {
  iree_hal_amdgpu_device_spec_t source = {
      /*.gfx_major=*/12,
      /*.gfx_minor=*/0,
      /*.gfx_stepping=*/1,
      /*.wavefront_size=*/64,
      /*.simd_per_compute_unit=*/4,
      /*.compute_unit_count=*/120,
      /*.maximum_waves_per_compute_unit=*/32,
      /*.local_memory_per_compute_unit=*/64 * 1024,
      /*.flags=*/IREE_HAL_AMDGPU_DEVICE_SPEC_FLAG_NONE,
  };

  iree_hal_device_spec_builder_t builder;
  iree_hal_device_spec_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(
      iree_hal_amdgpu_device_spec_builder_add_facet(&builder, &source));

  iree_hal_device_spec_t* device_spec = NULL;
  IREE_ASSERT_OK(iree_hal_device_spec_builder_finalize(&builder, &device_spec));
  const iree_hal_device_spec_facet_t* facet =
      iree_hal_amdgpu_device_spec_find_facet(device_spec);
  ASSERT_NE(facet, nullptr);

  iree_hal_amdgpu_device_spec_t decoded = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_device_spec_decode_facet(facet, &decoded));
  EXPECT_EQ(decoded.gfx_major, 12);
  EXPECT_EQ(decoded.gfx_stepping, 1);
  EXPECT_EQ(decoded.compute_unit_count, 120);

  iree_hal_device_spec_release(device_spec);
  iree_hal_device_spec_builder_deinitialize(&builder);
}

}  // namespace
}  // namespace iree::hal::amdgpu
