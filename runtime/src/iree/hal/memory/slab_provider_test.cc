// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/slab_provider.h"

#include "iree/hal/memory/cpu_slab_provider.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static iree_hal_asan_pool_options_t ShadowOptions() {
  iree_hal_asan_pool_options_t options = {};
  options.mode = IREE_HAL_ASAN_POOL_MODE_SHADOW;
  options.shadow_granule_size = 8;
  options.redzone_size = 16;
  return options;
}

TEST(SlabProviderASANTest, DisabledOptionsAreAlwaysValid) {
  iree_hal_slab_provider_t* provider = nullptr;
  IREE_ASSERT_OK(
      iree_hal_cpu_slab_provider_create(iree_allocator_system(), &provider));

  iree_hal_asan_pool_options_t options = {};
  IREE_EXPECT_OK(
      iree_hal_slab_provider_validate_asan_options(provider, &options));

  iree_hal_slab_provider_release(provider);
}

TEST(SlabProviderASANTest, EnabledOptionsRequireProviderSupport) {
  iree_hal_slab_provider_t* provider = nullptr;
  IREE_ASSERT_OK(
      iree_hal_cpu_slab_provider_create(iree_allocator_system(), &provider));

  iree_hal_asan_pool_options_t options = ShadowOptions();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_slab_provider_validate_asan_options(provider, &options));

  iree_hal_slab_provider_release(provider);
}

TEST(SlabProviderASANTest, AdviseRejectsInvalidFlags) {
  iree_hal_slab_provider_t* provider = nullptr;
  IREE_ASSERT_OK(
      iree_hal_cpu_slab_provider_create(iree_allocator_system(), &provider));

  iree_hal_slab_t slab;
  IREE_ASSERT_OK(iree_hal_slab_provider_acquire_slab(provider, 128, &slab));

  iree_hal_asan_pool_options_t options = ShadowOptions();
  iree_hal_asan_allocation_layout_t layout;
  IREE_ASSERT_OK(
      iree_hal_asan_calculate_allocation_layout(&options, 16, 8, &layout));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_slab_provider_advise_asan_range(
          provider, &slab, 0, IREE_HAL_ASAN_RANGE_ADVICE_FLAG_NONE, &layout));

  iree_hal_slab_provider_release_slab(provider, &slab);
  iree_hal_slab_provider_release(provider);
}

TEST(SlabProviderASANTest, AdviseRejectsOutOfRangeLayout) {
  iree_hal_slab_provider_t* provider = nullptr;
  IREE_ASSERT_OK(
      iree_hal_cpu_slab_provider_create(iree_allocator_system(), &provider));

  iree_hal_slab_t slab;
  IREE_ASSERT_OK(iree_hal_slab_provider_acquire_slab(provider, 64, &slab));

  iree_hal_asan_pool_options_t options = ShadowOptions();
  iree_hal_asan_allocation_layout_t layout;
  IREE_ASSERT_OK(
      iree_hal_asan_calculate_allocation_layout(&options, 16, 8, &layout));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_slab_provider_advise_asan_range(
          provider, &slab, 32, IREE_HAL_ASAN_RANGE_ADVICE_FLAG_ALLOCATED,
          &layout));

  iree_hal_slab_provider_release_slab(provider, &slab);
  iree_hal_slab_provider_release(provider);
}

TEST(SlabProviderASANTest, AdviseRejectsMisalignedBackingOffset) {
  iree_hal_slab_provider_t* provider = nullptr;
  IREE_ASSERT_OK(
      iree_hal_cpu_slab_provider_create(iree_allocator_system(), &provider));

  iree_hal_slab_t slab;
  IREE_ASSERT_OK(iree_hal_slab_provider_acquire_slab(provider, 128, &slab));

  iree_hal_asan_pool_options_t options = ShadowOptions();
  iree_hal_asan_allocation_layout_t layout;
  IREE_ASSERT_OK(
      iree_hal_asan_calculate_allocation_layout(&options, 16, 8, &layout));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_slab_provider_advise_asan_range(
          provider, &slab, 1, IREE_HAL_ASAN_RANGE_ADVICE_FLAG_ALLOCATED,
          &layout));

  iree_hal_slab_provider_release_slab(provider, &slab);
  iree_hal_slab_provider_release(provider);
}

TEST(SlabProviderASANTest, AdviseDispatchesValidRange) {
  iree_hal_slab_provider_t* provider = nullptr;
  IREE_ASSERT_OK(
      iree_hal_cpu_slab_provider_create(iree_allocator_system(), &provider));

  iree_hal_slab_t slab;
  IREE_ASSERT_OK(iree_hal_slab_provider_acquire_slab(provider, 128, &slab));

  iree_hal_asan_pool_options_t options = ShadowOptions();
  iree_hal_asan_allocation_layout_t layout;
  IREE_ASSERT_OK(
      iree_hal_asan_calculate_allocation_layout(&options, 16, 8, &layout));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_slab_provider_advise_asan_range(
          provider, &slab, 0, IREE_HAL_ASAN_RANGE_ADVICE_FLAG_ALLOCATED,
          &layout));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_slab_provider_advise_asan_range(
                            provider, &slab, 0,
                            IREE_HAL_ASAN_RANGE_ADVICE_FLAG_RELEASED, &layout));

  iree_hal_slab_provider_release_slab(provider, &slab);
  iree_hal_slab_provider_release(provider);
}

}  // namespace
