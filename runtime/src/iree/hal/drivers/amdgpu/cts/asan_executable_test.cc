// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU ASAN executable CTS coverage.

#include <cstdint>
#include <string_view>
#include <vector>

#include "iree/base/threading/thread.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amdgpu/abi/asan.h"
#include "iree/hal/drivers/amdgpu/abi/feedback.h"
#include "iree/hal/drivers/amdgpu/api.h"

namespace iree::hal::cts {

namespace {

class IsolatedBackendDevice {
 public:
  IsolatedBackendDevice() = default;
  ~IsolatedBackendDevice() { Reset(); }

  IsolatedBackendDevice(const IsolatedBackendDevice&) = delete;
  IsolatedBackendDevice& operator=(const IsolatedBackendDevice&) = delete;

  iree_status_t Initialize(const BackendInfo& backend) {
    IREE_RETURN_IF_ERROR(create_context_.Initialize(iree_allocator_system()));
    iree_status_t status =
        backend.factory(create_context_.params(), &driver_, &device_);
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_group_create_from_device(
          device_, create_context_.frontier_tracker(), iree_allocator_system(),
          &device_group_);
    }
    if (iree_status_is_ok(status)) {
      allocator_ = iree_hal_device_allocator(device_);
      iree_hal_allocator_retain(allocator_);
    } else {
      Reset();
    }
    return status;
  }

  iree_hal_device_t* device() const { return device_; }
  iree_hal_allocator_t* allocator() const { return allocator_; }

 private:
  void Reset() {
    iree_hal_allocator_release(allocator_);
    allocator_ = nullptr;
    iree_hal_device_release(device_);
    device_ = nullptr;
    iree_hal_device_group_release(device_group_);
    device_group_ = nullptr;
    iree_hal_driver_release(driver_);
    driver_ = nullptr;
    create_context_.Deinitialize();
  }

  DeviceCreateContext create_context_;
  iree_hal_driver_t* driver_ = nullptr;
  iree_hal_device_group_t* device_group_ = nullptr;
  iree_hal_device_t* device_ = nullptr;
  iree_hal_allocator_t* allocator_ = nullptr;
};

static iree_status_t CreateDeviceBuffer(iree_hal_allocator_t* allocator,
                                        iree_device_size_t buffer_size,
                                        iree_hal_buffer_t** out_buffer) {
  *out_buffer = nullptr;
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage =
      IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
  return iree_hal_allocator_allocate_buffer(allocator, params, buffer_size,
                                            out_buffer);
}

static iree_status_t WaitForQueueFailure(iree_hal_device_t* device) {
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_flush(device, IREE_HAL_QUEUE_AFFINITY_ANY);
    if (iree_status_is_ok(status)) {
      iree_thread_yield();
    }
  }
  return status;
}

}  // namespace

class AsanExecutableTest : public CtsTestBase<> {
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

TEST_P(AsanExecutableTest, PublishesConfigGlobal) {
  bool found = false;
  iree_hal_executable_global_t global = iree_hal_executable_global_invalid();
  IREE_ASSERT_OK(iree_hal_executable_try_lookup_global_by_name(
      executable_, IREE_SV(IREE_HAL_AMDGPU_ASAN_CONFIG_GLOBAL_NAME), &found,
      &global));
  ASSERT_TRUE(found);

  iree_hal_executable_global_info_t info;
  IREE_ASSERT_OK(iree_hal_executable_global_info(executable_, global, &info));
  EXPECT_EQ(std::string_view(info.name.data, info.name.size),
            IREE_HAL_AMDGPU_ASAN_CONFIG_GLOBAL_NAME);
  ASSERT_EQ(info.byte_length, sizeof(iree_hal_amdgpu_asan_config_t));

  iree_hal_buffer_t* global_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_executable_global_buffer(
      executable_, global, IREE_HAL_QUEUE_AFFINITY_ANY, &global_buffer));
  ASSERT_NE(global_buffer, nullptr);

  std::vector<iree_hal_amdgpu_asan_config_t> configs =
      ReadBufferData<iree_hal_amdgpu_asan_config_t>(global_buffer);
  ASSERT_EQ(configs.size(), 1u);
  const iree_hal_amdgpu_asan_config_t& config = configs[0];
  EXPECT_EQ(config.record_length, sizeof(config));
  EXPECT_EQ(config.abi_version, IREE_HAL_AMDGPU_ASAN_CONFIG_ABI_VERSION_0);
  EXPECT_NE(config.flags & IREE_HAL_AMDGPU_ASAN_CONFIG_FLAG_ENABLED, 0u);
  EXPECT_NE(config.shadow_base, 0u);
  EXPECT_EQ(config.shadow_size, IREE_HAL_AMDGPU_ASAN_DEFAULT_SHADOW_SIZE);
  EXPECT_GE(config.shadow_slab_size,
            IREE_HAL_AMDGPU_ASAN_DEFAULT_SHADOW_SLAB_SIZE);

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(5 * sizeof(uint64_t), output_buffer.out()));
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

  const uint32_t constant_data[] = {0x4153414Eu, 0x43464721u};
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
  ASSERT_EQ(output_data.size(), 5u);
  EXPECT_EQ(output_data[0], config.record_length);
  EXPECT_EQ(output_data[1], config.flags);
  EXPECT_EQ(output_data[2], config.shadow_base);
  EXPECT_EQ(output_data[3], config.shadow_size);
  EXPECT_EQ(output_data[4], config.shadow_slab_size);
}

TEST_P(AsanExecutableTest, PublishesFeedbackConfigGlobal) {
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

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(4 * sizeof(uint64_t), output_buffer.out()));
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

  const uint32_t constant_data[] = {0x4644424Bu, 0x43464721u};
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
  ASSERT_EQ(output_data.size(), 4u);
  EXPECT_EQ(output_data[0], config.record_length);
  EXPECT_EQ(output_data[1], config.flags);
  EXPECT_EQ(output_data[2], config.channel_base);
  EXPECT_EQ(output_data[3], config.notify_signal.handle);
}

TEST_P(AsanExecutableTest, ReportsAsanPacketThroughFeedback) {
  IsolatedBackendDevice isolated_device;
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

  Ref<iree_hal_executable_t> executable;
  iree_hal_executable_params_t executable_params;
  iree_hal_executable_params_initialize(&executable_params);
  executable_params.caching_mode =
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
  executable_params.executable_format =
      iree_make_cstring_view(executable_format());
  executable_params.executable_data =
      executable_data(iree_make_cstring_view("executable_test.bin"));
  status = iree_hal_executable_cache_prepare_executable(
      executable_cache, &executable_params, executable.out());
  if (iree_status_is_incompatible(status)) {
    iree_status_free(status);
    GTEST_SKIP() << "Executable format '" << executable_format()
                 << "' is incompatible with CTS backend/device '"
                 << GetParam().name << "'";
  }
  IREE_ASSERT_OK(status);

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateDeviceBuffer(isolated_device.allocator(),
                                    sizeof(uint64_t), output_buffer.out()));
  Ref<iree_hal_buffer_t> fallback_buffer;
  IREE_ASSERT_OK(CreateDeviceBuffer(isolated_device.allocator(),
                                    sizeof(uint64_t), fallback_buffer.out()));

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

  const uint32_t constant_data[] = {0x4153414Eu, 0x52505421u};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  SemaphoreList empty_wait;
  SemaphoreList dispatch_signal(isolated_device.device(), {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      isolated_device.device(), IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait,
      dispatch_signal, executable, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        WaitForQueueFailure(isolated_device.device()));
}

CTS_REGISTER_EXECUTABLE_TEST_SUITE(AsanExecutableTest);

}  // namespace iree::hal::cts
