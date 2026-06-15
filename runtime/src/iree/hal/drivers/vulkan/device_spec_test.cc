// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/device_spec.h"

#include <vector>

#include "iree/hal/drivers/vulkan/device_spec_builder.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "vulkan/vulkan_core.h"

namespace iree::hal::vulkan {
namespace {

static iree_hal_vulkan_device_spec_t MakeTestSpec() {
  return {
      /*.api_version=*/VK_MAKE_API_VERSION(0, 1, 3, 0),
      /*.driver_version=*/1234,
      /*.physical_device_type=*/VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
      /*.enabled_features=*/IREE_HAL_VULKAN_FEATURE_REQUIRED_BASELINE |
          IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT16 |
          IREE_HAL_VULKAN_FEATURE_ENABLE_SUBGROUP_SIZE_CONTROL,
      /*.flags=*/IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_NONE,
  };
}

TEST(DeviceSpecTest, EncodesAndDecodesPayload) {
  iree_hal_vulkan_device_spec_t source = MakeTestSpec();
  std::vector<uint8_t> payload_storage(
      iree_hal_vulkan_device_spec_payload_size());
  IREE_ASSERT_OK(iree_hal_vulkan_device_spec_encode(
      &source,
      iree_make_byte_span(payload_storage.data(), payload_storage.size())));

  iree_hal_vulkan_device_spec_t decoded = {0};
  IREE_ASSERT_OK(iree_hal_vulkan_device_spec_decode(
      iree_make_const_byte_span(payload_storage.data(), payload_storage.size()),
      &decoded));
  EXPECT_EQ(decoded.api_version, VK_MAKE_API_VERSION(0, 1, 3, 0));
  EXPECT_EQ(decoded.driver_version, 1234u);
  EXPECT_EQ(decoded.physical_device_type, VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
  EXPECT_TRUE(
      iree_all_bits_set(decoded.enabled_features,
                        IREE_HAL_VULKAN_FEATURE_ENABLE_SUBGROUP_SIZE_CONTROL));
}

TEST(DeviceSpecTest, AddsAndFindsCoreFacet) {
  iree_hal_vulkan_device_spec_t source = MakeTestSpec();

  iree_hal_device_spec_builder_t builder;
  iree_hal_device_spec_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(
      iree_hal_vulkan_device_spec_builder_add_facet(&builder, &source));

  iree_hal_device_spec_t* device_spec = NULL;
  IREE_ASSERT_OK(iree_hal_device_spec_builder_finalize(&builder, &device_spec));
  const iree_hal_device_spec_facet_t* facet =
      iree_hal_vulkan_device_spec_find_facet(device_spec);
  ASSERT_NE(facet, nullptr);

  iree_hal_vulkan_device_spec_t decoded = {0};
  IREE_ASSERT_OK(iree_hal_vulkan_device_spec_decode_facet(facet, &decoded));
  EXPECT_EQ(decoded.enabled_features, source.enabled_features);
  EXPECT_EQ(decoded.physical_device_type, source.physical_device_type);

  iree_hal_device_spec_release(device_spec);
  iree_hal_device_spec_builder_deinitialize(&builder);
}

}  // namespace
}  // namespace iree::hal::vulkan
