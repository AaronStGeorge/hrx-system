// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared helpers for HAL ASAN executable CTS coverage.

#ifndef IREE_HAL_CTS_SANITIZER_ASAN_TEST_UTIL_H_
#define IREE_HAL_CTS_SANITIZER_ASAN_TEST_UTIL_H_

#include <condition_variable>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

// Captures ASAN device events produced by the ASAN CTS backend device.
class AsanDeviceEventRecorder {
 public:
  AsanDeviceEventRecorder() = default;

  AsanDeviceEventRecorder(const AsanDeviceEventRecorder&) = delete;
  AsanDeviceEventRecorder& operator=(const AsanDeviceEventRecorder&) = delete;

  iree_hal_device_event_sink_t sink() {
    iree_hal_device_event_sink_t sink;
    sink.fn = AsanDeviceEventRecorder::Capture;
    sink.user_data = this;
    return sink;
  }

  void Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    asan_report_count_ = 0;
    last_source_ = iree_hal_device_event_source_default();
    std::memset(&last_report_, 0, sizeof(last_report_));
  }

  void WaitForAsanReportCount(iree_host_size_t expected_count) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return asan_report_count_ >= expected_count; });
  }

  iree_host_size_t asan_report_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return asan_report_count_;
  }

  iree_hal_device_event_source_t last_source() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_source_;
  }

  iree_hal_device_asan_report_t last_report() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_report_;
  }

 private:
  static void Capture(void* user_data, const iree_hal_device_event_t* event) {
    auto* recorder = static_cast<AsanDeviceEventRecorder*>(user_data);
    if (event->type != IREE_HAL_DEVICE_EVENT_TYPE_ASAN_REPORT ||
        !event->payload.data ||
        event->payload.data_length < sizeof(iree_hal_device_asan_report_t)) {
      return;
    }
    std::lock_guard<std::mutex> lock(recorder->mutex_);
    recorder->last_source_ = event->source;
    std::memcpy(&recorder->last_report_, event->payload.data,
                sizeof(recorder->last_report_));
    ++recorder->asan_report_count_;
    recorder->condition_.notify_all();
  }

  // Protects captured report state.
  mutable std::mutex mutex_;
  // Notifies tests waiting for captured ASAN reports.
  std::condition_variable condition_;
  // Count of ASAN reports captured since the last reset.
  iree_host_size_t asan_report_count_ = 0;
  // Source attribution from the last captured ASAN report.
  iree_hal_device_event_source_t last_source_ =
      iree_hal_device_event_source_default();
  // Payload from the last captured ASAN report.
  iree_hal_device_asan_report_t last_report_ = {0};
};

// Cached backend resources for ASAN report tests.
struct AsanCachedBackendResources {
  ~AsanCachedBackendResources() { Reset(); }

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

  // Event recorder bound as the device creation sink.
  AsanDeviceEventRecorder recorder;
  // Device creation storage borrowed by the backend factory.
  DeviceCreateContext create_context;
  // Owned HAL driver for the cached device.
  iree_hal_driver_t* driver = nullptr;
  // Owned group keeping the cached device topology alive.
  iree_hal_device_group_t* device_group = nullptr;
  // Owned cached HAL device.
  iree_hal_device_t* device = nullptr;
  // Retained allocator from the cached HAL device.
  iree_hal_allocator_t* allocator = nullptr;
  // True when the backend factory reported unavailability.
  bool unavailable = false;
};

inline std::map<std::string, AsanCachedBackendResources>&
GetAsanBackendCache() {
  static std::map<std::string, AsanCachedBackendResources> cache;
  return cache;
}

// Returns true if |status_code| means the CTS backend cannot run on this host.
inline bool AsanStatusCodeIsBackendUnavailable(iree_status_code_t status_code) {
  return status_code == IREE_STATUS_UNAVAILABLE ||
         status_code == IREE_STATUS_NOT_FOUND;
}

// Borrows the cached ASAN CTS backend device for one test.
class AsanCachedBackendDevice {
 public:
  AsanCachedBackendDevice() = default;
  ~AsanCachedBackendDevice() { Reset(); }

  AsanCachedBackendDevice(const AsanCachedBackendDevice&) = delete;
  AsanCachedBackendDevice& operator=(const AsanCachedBackendDevice&) = delete;

  // Gets or creates a cached ASAN backend device using |backend|.
  iree_status_t Initialize(const BackendInfo& backend) {
    auto [cached_it, inserted] =
        GetAsanBackendCache().try_emplace(backend.name);
    (void)inserted;
    AsanCachedBackendResources& cached = cached_it->second;
    if (!cached.device && !cached.unavailable) {
      iree_hal_driver_t* driver = nullptr;
      iree_hal_device_t* device = nullptr;
      iree_status_t status = cached.create_context.Initialize(
          iree_allocator_system(), cached.recorder.sink());
      if (iree_status_is_ok(status)) {
        status =
            backend.factory(cached.create_context.params(), &driver, &device);
      }
      if (AsanStatusCodeIsBackendUnavailable(iree_status_code(status))) {
        iree_status_free(status);
        cached.unavailable = true;
        cached.create_context.Deinitialize();
      } else if (!iree_status_is_ok(status)) {
        cached.create_context.Deinitialize();
        return status;
      } else {
        cached.driver = driver;
        cached.device = device;
        status = iree_hal_device_group_create_from_device(
            device, cached.create_context.frontier_tracker(),
            iree_allocator_system(), &cached.device_group);
        if (iree_status_is_ok(status)) {
          cached.allocator = iree_hal_device_allocator(device);
          iree_hal_allocator_retain(cached.allocator);
        } else {
          cached.Reset();
          return status;
        }
      }
    }
    if (cached.unavailable) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "ASAN CTS backend '%s' is unavailable",
                              backend.name.c_str());
    }

    cached_ = &cached;
    driver_ = cached.driver;
    iree_hal_driver_retain(driver_);
    device_group_ = cached.device_group;
    iree_hal_device_group_retain(device_group_);
    device_ = cached.device;
    iree_hal_device_retain(device_);
    allocator_ = cached.allocator;
    iree_hal_allocator_retain(allocator_);
    cached.recorder.Reset();
    return iree_ok_status();
  }

  // Returns the cached HAL device.
  iree_hal_device_t* device() const { return device_; }

  // Returns the cached HAL device allocator.
  iree_hal_allocator_t* allocator() const { return allocator_; }

  // Returns the ASAN event recorder bound to the cached HAL device.
  AsanDeviceEventRecorder* recorder() const { return &cached_->recorder; }

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
    cached_ = nullptr;
  }

  // Borrowed cached resources for this test.
  AsanCachedBackendResources* cached_ = nullptr;
  // Retained HAL driver from the cached device.
  iree_hal_driver_t* driver_ = nullptr;
  // Retained group keeping the cached device topology alive.
  iree_hal_device_group_t* device_group_ = nullptr;
  // Retained cached HAL device.
  iree_hal_device_t* device_ = nullptr;
  // Retained allocator from the cached HAL device.
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

// Reads device-local data through a queue copy into host-visible memory.
inline iree_status_t AsanReadBufferBytes(iree_hal_device_t* device,
                                         iree_hal_allocator_t* allocator,
                                         iree_hal_buffer_t* source_buffer,
                                         iree_device_size_t source_offset,
                                         iree_device_size_t length,
                                         std::vector<uint8_t>* out_data) {
  IREE_ASSERT_ARGUMENT(out_data);
  out_data->clear();
  if (length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HAL ASAN CTS readback length %" PRIu64
                            " exceeds host vector capacity",
                            (uint64_t)length);
  }

  std::vector<uint8_t> data((iree_host_size_t)length);
  iree_hal_buffer_params_t staging_params = {0};
  staging_params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  staging_params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;

  Ref<iree_hal_buffer_t> staging_buffer;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      allocator, staging_params, length, staging_buffer.out()));

  SemaphoreList empty_wait;
  SemaphoreList copy_signal(device, {0}, {1});
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_copy(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, copy_signal,
      source_buffer, source_offset, staging_buffer, /*target_offset=*/0, length,
      IREE_HAL_COPY_FLAG_NONE));
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(
      copy_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_read(
      staging_buffer, /*offset=*/0, data.data(), (iree_host_size_t)length));

  *out_data = std::move(data);
  return iree_ok_status();
}

// Reads buffer contents back to host as a vector of T.
template <typename T>
inline iree_status_t AsanReadBufferData(iree_hal_device_t* device,
                                        iree_hal_allocator_t* allocator,
                                        iree_hal_buffer_t* source_buffer,
                                        iree_device_size_t source_offset,
                                        std::vector<T>* out_data) {
  IREE_ASSERT_ARGUMENT(out_data);
  out_data->clear();
  if (source_offset > iree_hal_buffer_byte_length(source_buffer)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "HAL ASAN CTS readback offset %" PRIu64
        " exceeds buffer length %" PRIu64,
        (uint64_t)source_offset,
        (uint64_t)iree_hal_buffer_byte_length(source_buffer));
  }
  const iree_device_size_t byte_length =
      iree_hal_buffer_byte_length(source_buffer) - source_offset;
  if (byte_length % sizeof(T) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL ASAN CTS readback length %" PRIu64
                            " is not divisible by element size %" PRIhsz,
                            (uint64_t)byte_length, sizeof(T));
  }

  std::vector<uint8_t> bytes;
  IREE_RETURN_IF_ERROR(AsanReadBufferBytes(device, allocator, source_buffer,
                                           source_offset, byte_length, &bytes));
  std::vector<T> data((iree_host_size_t)(byte_length / sizeof(T)));
  std::memcpy(data.data(), bytes.data(), (iree_host_size_t)byte_length);
  *out_data = std::move(data);
  return iree_ok_status();
}

template <typename T>
inline iree_status_t AsanReadBufferData(iree_hal_device_t* device,
                                        iree_hal_allocator_t* allocator,
                                        iree_hal_buffer_t* source_buffer,
                                        std::vector<T>* out_data) {
  return AsanReadBufferData(device, allocator, source_buffer,
                            /*source_offset=*/0, out_data);
}

}  // namespace iree::hal::cts

#endif  // IREE_HAL_CTS_SANITIZER_ASAN_TEST_UTIL_H_
