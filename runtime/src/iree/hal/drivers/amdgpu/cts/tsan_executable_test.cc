// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU TSAN executable CTS coverage.

#include <cstdint>
#include <string_view>
#include <vector>

#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amdgpu/abi/feedback.h"
#include "iree/hal/drivers/amdgpu/abi/tsan.h"
#include "iree/hal/drivers/amdgpu/api.h"

namespace iree::hal::cts {

class TsanExecutableTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) return;

    IREE_ASSERT_OK(iree_hal_executable_cache_create(
        device_, iree_make_cstring_view("default"), &executable_cache_));

    PrepareExecutableOrSkipUnsupported(executable_cache_, "executable_test.bin",
                                       &executable_);
  }

  void TearDown() override {
    iree_hal_executable_release(executable_);
    executable_ = nullptr;
    iree_hal_executable_cache_release(executable_cache_);
    executable_cache_ = nullptr;
    CtsTestBase::TearDown();
  }

  iree_hal_executable_cache_t* executable_cache_ = nullptr;
  iree_hal_executable_t* executable_ = nullptr;
};

TEST_P(TsanExecutableTest, PublishesTsanConfigGlobal) {
  bool found = false;
  iree_hal_executable_global_t global = iree_hal_executable_global_invalid();
  IREE_ASSERT_OK(iree_hal_executable_try_lookup_global_by_name(
      executable_, IREE_SV(IREE_HAL_AMDGPU_TSAN_CONFIG_GLOBAL_NAME), &found,
      &global));
  ASSERT_TRUE(found);

  iree_hal_executable_global_info_t info;
  IREE_ASSERT_OK(iree_hal_executable_global_info(executable_, global, &info));
  EXPECT_EQ(std::string_view(info.name.data, info.name.size),
            IREE_HAL_AMDGPU_TSAN_CONFIG_GLOBAL_NAME);
  ASSERT_EQ(info.byte_length, sizeof(iree_hal_amdgpu_tsan_config_t));

  iree_hal_buffer_t* global_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_executable_global_buffer(
      executable_, global, IREE_HAL_QUEUE_AFFINITY_ANY, &global_buffer));
  ASSERT_NE(global_buffer, nullptr);

  std::vector<iree_hal_amdgpu_tsan_config_t> configs =
      ReadBufferData<iree_hal_amdgpu_tsan_config_t>(global_buffer);
  ASSERT_EQ(configs.size(), 1u);
  const iree_hal_amdgpu_tsan_config_t& config = configs[0];
  EXPECT_EQ(config.record_length, sizeof(config));
  EXPECT_EQ(config.abi_version, IREE_HAL_AMDGPU_TSAN_CONFIG_ABI_VERSION_0);
  EXPECT_NE(config.flags & IREE_HAL_AMDGPU_TSAN_CONFIG_FLAG_ENABLED, 0u);
  EXPECT_NE(config.shadow_base, 0u);
  EXPECT_NE(config.shadow_size, 0u);
  EXPECT_EQ(config.memory_granule_shift,
            IREE_HAL_AMDGPU_TSAN_DEFAULT_MEMORY_GRANULE_SHIFT);
  EXPECT_EQ(config.workgroup_capacity,
            IREE_HAL_AMDGPU_TSAN_DEFAULT_WORKGROUP_CAPACITY);
  EXPECT_EQ(config.shadow_entry_size, IREE_HAL_AMDGPU_TSAN_SHADOW_ENTRY_SIZE);
  EXPECT_NE(config.workgroup_shadow_stride, 0u);

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(8 * sizeof(uint64_t), output_buffer.out()));
  Ref<iree_hal_buffer_t> fallback_buffer;
  const uint64_t fallback_value = 0xAAAAAAAA55555555ull;
  IREE_ASSERT_OK(CreateDeviceBufferWithData(
      &fallback_value, sizeof(fallback_value), fallback_buffer.out()));

  iree_hal_buffer_ref_t binding_refs[2];
  binding_refs[0] = iree_hal_make_buffer_ref(
      output_buffer, /*offset=*/0, iree_hal_buffer_byte_length(output_buffer));
  binding_refs[1] =
      iree_hal_make_buffer_ref(fallback_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(fallback_buffer));
  iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  const uint32_t constant_data[] = {0x5453414Eu, 0x43464721u};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  SemaphoreList empty_wait;
  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, dispatch_signal,
      executable_, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint64_t> output_data = ReadBufferData<uint64_t>(output_buffer);
  ASSERT_EQ(output_data.size(), 8u);
  EXPECT_EQ(output_data[0], config.record_length);
  EXPECT_EQ(output_data[1], config.flags);
  EXPECT_EQ(output_data[2], config.shadow_base);
  EXPECT_EQ(output_data[3], config.shadow_size);
  EXPECT_EQ(output_data[4], config.workgroup_shadow_stride);
  EXPECT_EQ(output_data[5], config.workgroup_capacity);
  EXPECT_EQ(output_data[6], config.shadow_entry_size);
  EXPECT_EQ(output_data[7], config.memory_granule_shift);
}

TEST_P(TsanExecutableTest, EnablesFeedbackConfigGlobal) {
  bool found = false;
  iree_hal_executable_global_t global = iree_hal_executable_global_invalid();
  IREE_ASSERT_OK(iree_hal_executable_try_lookup_global_by_name(
      executable_, IREE_SV(IREE_HAL_AMDGPU_FEEDBACK_CONFIG_GLOBAL_NAME), &found,
      &global));
  ASSERT_TRUE(found);

  iree_hal_executable_global_info_t info;
  IREE_ASSERT_OK(iree_hal_executable_global_info(executable_, global, &info));
  EXPECT_EQ(std::string_view(info.name.data, info.name.size),
            IREE_HAL_AMDGPU_FEEDBACK_CONFIG_GLOBAL_NAME);
  ASSERT_EQ(info.byte_length, sizeof(iree_hal_amdgpu_feedback_config_t));

  iree_hal_buffer_t* global_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_executable_global_buffer(
      executable_, global, IREE_HAL_QUEUE_AFFINITY_ANY, &global_buffer));
  ASSERT_NE(global_buffer, nullptr);

  std::vector<iree_hal_amdgpu_feedback_config_t> configs =
      ReadBufferData<iree_hal_amdgpu_feedback_config_t>(global_buffer);
  ASSERT_EQ(configs.size(), 1u);
  const iree_hal_amdgpu_feedback_config_t& config = configs[0];
  EXPECT_EQ(config.record_length, sizeof(config));
  EXPECT_EQ(config.abi_version, IREE_HAL_AMDGPU_FEEDBACK_CONFIG_ABI_VERSION_0);
  EXPECT_NE(config.flags & IREE_HAL_AMDGPU_FEEDBACK_CONFIG_FLAG_ENABLED, 0u);
  EXPECT_NE(config.channel_base, 0u);
  EXPECT_NE(config.notify_signal.handle, 0u);
  EXPECT_NE(config.source_context, 0u);
}

CTS_REGISTER_EXECUTABLE_TEST_SUITE(TsanExecutableTest);

}  // namespace iree::hal::cts
