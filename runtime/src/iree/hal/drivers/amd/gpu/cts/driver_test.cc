// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <string>

#include "iree/async/util/proactor_pool.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amd/gpu/cts/test_device.h"
#include "iree/hal/drivers/amd/registration/driver_module.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::hal::amd::gpu::cts::DeviceAllocator;

class DriverTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(
        iree_hal_driver_registry_allocate(iree_allocator_system(), &registry_));
    IREE_ASSERT_OK(iree_hal_amd_driver_module_register(registry_));
    IREE_ASSERT_OK(iree_hal_driver_registry_acquire_driver(
        registry_, IREE_SV("rdna"), &driver_));
    IREE_ASSERT_OK(iree_hal_driver_query_available_devices(
        driver_, IREE_SV("rdna"), iree_allocator_system(), &count_, &infos_));
    if (!count_) {
      GTEST_SKIP() << "No RDNA endpoint";
    }
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        1, nullptr, iree_async_proactor_pool_options_default(),
        iree_allocator_system(), &proactors_));
  }

  void TearDown() override {
    for (size_t i = 0; i < devices_.size(); ++i) {
      iree_hal_semaphore_release(semaphores_[i]);
      if (devices_[i]) {
        iree_hal_device_release(devices_[i]);
        allocators_[i].reclaimed.acquire();
      }
      EXPECT_EQ(allocators_[i].live_count.load(), 0u);
    }
    iree_async_proactor_pool_release(proactors_);
    iree_allocator_free(iree_allocator_system(), infos_);
    iree_hal_driver_release(driver_);
    iree_hal_driver_registry_free(registry_);
  }

  void Create(size_t index, iree_string_view_t path,
              iree_string_view_t scheme = IREE_SV("rdna")) {
    auto params = iree_hal_device_create_params_default();
    params.proactor_pool = proactors_;
    const iree_string_pair_t options[] = {
        {.key = IREE_SV("queue_count"), .value = IREE_SV("2")}};
    iree::Status status = iree_hal_driver_create_device_by_path(
        driver_, scheme, path, IREE_ARRAYSIZE(options), options, &params,
        allocators_[index].allocator(), &devices_[index]);
    IREE_ASSERT_OK(status);
    allocators_[index].armed.store(true);
    EXPECT_TRUE(iree_string_view_equal(iree_hal_device_id(devices_[index]),
                                       IREE_SV("rdna")));
    ASSERT_NE(iree_hal_device_queue(devices_[index], 0, 0), nullptr);
    ASSERT_NE(iree_hal_device_queue(devices_[index], 0, 1), nullptr);
  }

  void Run(size_t index) {
    iree_hal_device_t* device = devices_[index];
    iree_hal_semaphore_t** semaphore = &semaphores_[index];
    IREE_ASSERT_OK(
        iree_hal_semaphore_create(device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                  IREE_HAL_SEMAPHORE_FLAG_NONE, semaphore));
    uint64_t first = 1, second = 2;
    IREE_ASSERT_OK(iree_hal_queue_barrier(
        iree_hal_device_queue(device, 0, 1), {1, semaphore, &first},
        {1, semaphore, &second}, nullptr, IREE_HAL_QUEUE_BARRIER_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_queue_barrier(
        iree_hal_device_queue(device, 0, 0), iree_hal_semaphore_list_empty(),
        {1, semaphore, &first}, nullptr, IREE_HAL_QUEUE_BARRIER_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_wait(*semaphore, second,
                                           iree_infinite_timeout(),
                                           IREE_ASYNC_WAIT_FLAG_NONE));
  }

  // Public registry retaining the canonical AMD driver until explicitly freed.
  iree_hal_driver_registry_t* registry_ = nullptr;
  // Independent reference acquired through the RDNA scheme.
  iree_hal_driver_t* driver_ = nullptr;
  // Public discovery rows, independent of native execution resources.
  iree_hal_device_info_t* infos_ = nullptr;
  // Number of discovery rows with usable native compute families.
  iree_host_size_t count_ = 0;
  // Process progress service supplied to public device construction.
  iree_async_proactor_pool_t* proactors_ = nullptr;
  // Device ownership includes asynchronous final worker cleanup.
  std::array<DeviceAllocator, 2> allocators_;
  // Independently provisioned devices sharing one native fabric.
  std::array<iree_hal_device_t*, 2> devices_ = {};
  // Cross-queue dependencies and final completions observed by this test.
  std::array<iree_hal_semaphore_t*, 2> semaphores_ = {};
};

TEST_F(DriverTest, DefaultAndNativeIdentityRetainOneOwner) {
  iree_hal_driver_t* xdna_driver = nullptr;
  IREE_ASSERT_OK(iree_hal_driver_registry_acquire_driver(
      registry_, IREE_SV("xdna"), &xdna_driver));
  EXPECT_EQ(driver_, xdna_driver);
  iree_hal_driver_release(xdna_driver);
  iree_hal_driver_t* gpu_driver = nullptr;
  IREE_ASSERT_OK(iree_hal_driver_registry_acquire_driver(
      registry_, IREE_SV("amdgpu2"), &gpu_driver));
  EXPECT_EQ(driver_, gpu_driver);
  iree_hal_driver_release(gpu_driver);

  iree_host_size_t gpu_count = 0;
  iree_hal_device_info_t* gpu_infos = nullptr;
  IREE_ASSERT_OK(iree_hal_driver_query_available_devices(
      driver_, IREE_SV("amdgpu2"), iree_allocator_system(), &gpu_count,
      &gpu_infos));
  if (!gpu_count) {
    iree_allocator_free(iree_allocator_system(), gpu_infos);
    GTEST_SKIP() << "No visible unified GPU endpoint";
  }
  const std::string gpu_path(gpu_infos[0].path.data, gpu_infos[0].path.size);
  iree_allocator_free(iree_allocator_system(), gpu_infos);

  auto params = iree_hal_device_create_params_default();
  params.proactor_pool = proactors_;
  const iree_string_pair_t invalid[] = {
      {.key = IREE_SV("columns"), .value = IREE_SV("1")}};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_driver_create_device_by_id(
          driver_, infos_[0].device_id, IREE_ARRAYSIZE(invalid), invalid,
          &params, allocators_[0].allocator(), &devices_[0]));
  ASSERT_EQ(devices_[0], nullptr);

  // The registry selects PROCESS native lifetime for host registration. Both
  // devices must share that owner: KFD cannot reacquire a destroyed primary VM.
  Create(0, IREE_SV(""));
  if (HasFatalFailure()) {
    return;
  }
  Create(1, iree_make_cstring_view(gpu_path.c_str()), IREE_SV("amdgpu2"));
  if (HasFatalFailure()) {
    return;
  }
  iree_hal_driver_release(driver_);
  driver_ = nullptr;
  iree_hal_driver_registry_free(registry_);
  registry_ = nullptr;
  Run(0);
  Run(1);
}

}  // namespace
