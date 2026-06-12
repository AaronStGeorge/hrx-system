// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU ASAN manual hook CTS coverage.

#include <cstdint>

#include "iree/hal/drivers/amdgpu/cts/asan_executable_test_util.h"

namespace iree::hal::cts {

class ManualAsanExecutableTest : public ::testing::TestWithParam<BackendInfo> {
};

TEST_P(ManualAsanExecutableTest, ReportsStore4HookThroughFeedback) {
  AsanIsolatedBackendDevice isolated_device;
  iree_status_t status = isolated_device.Initialize(GetParam());
  if (iree_status_is_unavailable(status)) {
    iree_status_free(status);
    GTEST_SKIP() << "Backend '" << GetParam().name
                 << "' unavailable on this system";
  }
  IREE_ASSERT_OK(status);

  Ref<iree_hal_executable_cache_t> executable_cache;
  IREE_ASSERT_OK(iree_hal_executable_cache_create(
      isolated_device.device(), iree_make_cstring_view("default"),
      executable_cache.out()));

  iree_const_byte_span_t executable_data = GetParam().executable_data(
      iree_make_cstring_view("manual_asan_test.bin"));
  ASSERT_FALSE(iree_const_byte_span_is_empty(executable_data));

  Ref<iree_hal_executable_t> executable;
  iree_hal_executable_params_t executable_params;
  iree_hal_executable_params_initialize(&executable_params);
  executable_params.caching_mode =
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
  executable_params.executable_format =
      iree_make_cstring_view(GetParam().executable_format);
  executable_params.executable_data = executable_data;
  status = iree_hal_executable_cache_prepare_executable(
      executable_cache, &executable_params, executable.out());
  if (iree_status_is_incompatible(status)) {
    iree_status_free(status);
    GTEST_SKIP() << "Executable format '" << GetParam().executable_format
                 << "' is incompatible with CTS backend/device '"
                 << GetParam().name << "'";
  }
  IREE_ASSERT_OK(status);

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(AsanCreateDeviceBuffer(isolated_device.allocator(),
                                        sizeof(uint64_t), output_buffer.out()));

  iree_hal_buffer_ref_t binding_refs[1];
  binding_refs[0] = iree_hal_make_buffer_ref(
      output_buffer, /*offset=*/0, iree_hal_buffer_byte_length(output_buffer));
  iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  SemaphoreList empty_wait;
  SemaphoreList dispatch_signal(isolated_device.device(), {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      isolated_device.device(), IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait,
      dispatch_signal, executable, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        AsanWaitForQueueFailure(isolated_device.device()));
}

CTS_REGISTER_EXECUTABLE_TEST_SUITE(ManualAsanExecutableTest);

}  // namespace iree::hal::cts
