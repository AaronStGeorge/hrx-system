// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU TSAN executable CTS coverage.

#include <cstdint>
#include <map>
#include <string_view>
#include <vector>

#include "iree/hal/cts/sanitizer/asan_test_util.h"
#include "iree/hal/drivers/amdgpu/abi/feedback.h"
#include "iree/hal/drivers/amdgpu/abi/tsan.h"
#include "iree/hal/drivers/amdgpu/api.h"
#include "iree/hal/drivers/amdgpu/registration/driver_module.h"

namespace iree::hal::cts {

namespace {

static iree_status_t CreateAmdgpuTsanDevice(
    const iree_hal_device_create_params_t* create_params,
    iree_hal_driver_t** out_driver, iree_hal_device_t** out_device) {
  *out_driver = nullptr;
  *out_device = nullptr;

  iree_status_t status = iree_hal_amdgpu_driver_module_register(
      iree_hal_driver_registry_default());
  if (iree_status_is_already_exists(status)) {
    iree_status_free(status);
    status = iree_ok_status();
  }

  iree_hal_driver_t* driver = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(), IREE_SV("amdgpu"),
        iree_allocator_system(), &driver);
  }

  iree_hal_device_t* device = nullptr;
  if (iree_status_is_ok(status)) {
    iree_string_pair_t params[1] = {};
    params[0].key = IREE_SV("hal.sanitizer");
    params[0].value = IREE_SV("tsan");
    status = iree_hal_driver_create_device_by_id(
        driver, IREE_HAL_DEVICE_ID_DEFAULT, IREE_ARRAYSIZE(params), params,
        create_params, iree_allocator_system(), &device);
  }

  if (iree_status_is_ok(status)) {
    *out_driver = driver;
    *out_device = device;
  } else {
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
  }
  return status;
}

struct TsanCachedBackendResources {
  ~TsanCachedBackendResources() { Reset(); }

  void Reset() {
    iree_hal_allocator_release(allocator);
    allocator = nullptr;
    iree_hal_device_release(device);
    device = nullptr;
    iree_hal_device_group_release(device_group);
    device_group = nullptr;
    iree_hal_driver_release(driver);
    driver = nullptr;
    create_context.Deinitialize();
    unavailable = false;
  }

  DeviceCreateContext create_context;
  iree_hal_driver_t* driver = nullptr;
  iree_hal_device_group_t* device_group = nullptr;
  iree_hal_device_t* device = nullptr;
  iree_hal_allocator_t* allocator = nullptr;
  bool unavailable = false;
};

static std::map<std::string, TsanCachedBackendResources>&
GetTsanBackendCache() {
  static std::map<std::string, TsanCachedBackendResources> cache;
  return cache;
}

}  // namespace

class TsanExecutableTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    const BackendInfo& backend = GetParam();
    auto [cached_it, inserted] =
        GetTsanBackendCache().try_emplace(GetBackendDeviceCacheKey(backend));
    (void)inserted;
    TsanCachedBackendResources& cached = cached_it->second;
    if (!cached.device && !cached.unavailable) {
      iree_hal_driver_t* driver = nullptr;
      iree_hal_device_t* device = nullptr;
      iree_status_t status =
          cached.create_context.Initialize(iree_allocator_system());
      if (iree_status_is_ok(status)) {
        status = CreateAmdgpuTsanDevice(cached.create_context.params(), &driver,
                                        &device);
      }
      if (AsanStatusCodeIsBackendUnavailable(iree_status_code(status))) {
        iree_status_free(status);
        cached.unavailable = true;
        cached.create_context.Deinitialize();
      } else if (!iree_status_is_ok(status)) {
        cached.create_context.Deinitialize();
        IREE_ASSERT_OK(status);
      } else {
        cached.driver = driver;
        cached.device = device;
        IREE_ASSERT_OK(iree_hal_device_group_create_from_device(
            device, cached.create_context.frontier_tracker(),
            iree_allocator_system(), &cached.device_group));
        cached.allocator = iree_hal_device_allocator(device);
        iree_hal_allocator_retain(cached.allocator);
      }
    }
    if (cached.unavailable) {
      GTEST_SKIP() << "Backend '" << backend.name
                   << "' unavailable on this system";
      return;
    }

    driver_ = cached.driver;
    iree_hal_driver_retain(driver_);
    device_group_ = cached.device_group;
    iree_hal_device_group_retain(device_group_);
    device_ = cached.device;
    iree_hal_device_retain(device_);
    device_allocator_ = cached.allocator;
    iree_hal_allocator_retain(device_allocator_);

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
    iree_hal_allocator_release(device_allocator_);
    device_allocator_ = nullptr;
    iree_hal_device_release(device_);
    device_ = nullptr;
    iree_hal_device_group_release(device_group_);
    device_group_ = nullptr;
    iree_hal_driver_release(driver_);
    driver_ = nullptr;
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
  EXPECT_NE(config.flags & IREE_HAL_AMDGPU_TSAN_CONFIG_FLAG_QUEUE_STATE, 0u);
  EXPECT_NE(config.shadow_base, 0u);
  EXPECT_NE(config.shadow_size, 0u);
  EXPECT_EQ(config.memory_granule_shift,
            IREE_HAL_AMDGPU_TSAN_DEFAULT_MEMORY_GRANULE_SHIFT);
  EXPECT_EQ(config.workgroup_capacity,
            IREE_HAL_AMDGPU_TSAN_DEFAULT_WORKGROUP_CAPACITY);
  EXPECT_EQ(config.shadow_entry_size, IREE_HAL_AMDGPU_TSAN_SHADOW_ENTRY_SIZE);
  EXPECT_EQ(config.shadow_slot_count,
            IREE_HAL_AMDGPU_TSAN_DEFAULT_SHADOW_SLOT_COUNT);
  EXPECT_NE(config.workgroup_shadow_stride, 0u);
  EXPECT_NE(config.queue_aql_base, 0u);
  EXPECT_NE(config.queue_aql_slot_mask, 0u);
  EXPECT_NE(config.queue_state_base, 0u);

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(12 * sizeof(uint64_t), output_buffer.out()));
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
  ASSERT_EQ(output_data.size(), 12u);
  EXPECT_EQ(output_data[0], config.record_length);
  EXPECT_EQ(output_data[1], config.flags);
  EXPECT_EQ(output_data[2], config.shadow_base);
  EXPECT_EQ(output_data[3], config.shadow_size);
  EXPECT_EQ(output_data[4], config.workgroup_shadow_stride);
  EXPECT_EQ(output_data[5], config.workgroup_capacity);
  EXPECT_EQ(output_data[6], config.shadow_entry_size);
  EXPECT_EQ(output_data[7], config.memory_granule_shift);
  EXPECT_EQ(output_data[8], config.queue_aql_base);
  EXPECT_EQ(output_data[9], config.queue_aql_slot_mask);
  EXPECT_EQ(output_data[10], config.queue_state_base);
  EXPECT_EQ(output_data[11], config.shadow_slot_count);
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
