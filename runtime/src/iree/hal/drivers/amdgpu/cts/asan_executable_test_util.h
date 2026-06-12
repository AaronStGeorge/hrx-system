// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared helpers for AMDGPU ASAN executable CTS coverage.

#ifndef IREE_HAL_DRIVERS_AMDGPU_CTS_ASAN_EXECUTABLE_TEST_UTIL_H_
#define IREE_HAL_DRIVERS_AMDGPU_CTS_ASAN_EXECUTABLE_TEST_UTIL_H_

#include "iree/base/threading/thread.h"
#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

// Owns an isolated backend device for fatal ASAN tests that intentionally put
// the logical device into a sticky-failure state.
class AsanIsolatedBackendDevice {
 public:
  AsanIsolatedBackendDevice() = default;
  ~AsanIsolatedBackendDevice() { Reset(); }

  AsanIsolatedBackendDevice(const AsanIsolatedBackendDevice&) = delete;
  AsanIsolatedBackendDevice& operator=(const AsanIsolatedBackendDevice&) =
      delete;

  // Creates a fresh backend device using the supplied CTS backend factory.
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

  // Returns the isolated HAL device.
  iree_hal_device_t* device() const { return device_; }

  // Returns the isolated HAL device allocator.
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

  // Device creation storage borrowed by the backend factory.
  DeviceCreateContext create_context_;
  // Owned HAL driver for the isolated device.
  iree_hal_driver_t* driver_ = nullptr;
  // Owned group keeping the isolated device topology alive.
  iree_hal_device_group_t* device_group_ = nullptr;
  // Owned isolated HAL device.
  iree_hal_device_t* device_ = nullptr;
  // Retained allocator from the isolated HAL device.
  iree_hal_allocator_t* allocator_ = nullptr;
};

// Allocates a dispatch/transfer-capable device-local buffer.
inline iree_status_t AsanCreateDeviceBuffer(iree_hal_allocator_t* allocator,
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

// Flushes until the current fatal ASAN path surfaces the device sticky failure.
inline iree_status_t AsanWaitForQueueFailure(iree_hal_device_t* device) {
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_flush(device, IREE_HAL_QUEUE_AFFINITY_ANY);
    if (iree_status_is_ok(status)) {
      iree_thread_yield();
    }
  }
  return status;
}

}  // namespace iree::hal::cts

#endif  // IREE_HAL_DRIVERS_AMDGPU_CTS_ASAN_EXECUTABLE_TEST_UTIL_H_
