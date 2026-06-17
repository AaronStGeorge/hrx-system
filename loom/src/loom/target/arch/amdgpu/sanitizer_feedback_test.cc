// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/internal/arena.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/abi/asan.h"
#include "iree/hal/drivers/amdgpu/abi/feedback.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/registration/driver_module.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/sanitizer/ops.h"
#include "loom/sanitizer/options.h"
#include "loom/sanitizer/site_table.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/hal/kernel_abi.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/feedback.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_access.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_race_report.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_report.h"
#include "loom/target/arch/amdgpu/lower/system_memory.h"
#include "loom/target/arch/amdgpu/ops/ops.h"
#include "loom/target/arch/amdgpu/ops/registry.h"
#include "loom/target/arch/amdgpu/planning/wait_packets.h"
#include "loom/target/arch/amdgpu/provider.h"
#include "loom/target/arch/amdgpu/records/target_records.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/registers.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/testing/module_ptr.h"
#include "loom/tooling/execution/hal/invocation.h"
#include "loom/tooling/target/amdgpu/artifact_provider.h"

namespace loom {
namespace {

using ::loom::testing::CapturedDiagnostic;
using ::loom::testing::DiagnosticCapture;
using ::loom::testing::ModulePtr;

constexpr uint64_t kReportFaultAddress = UINT64_C(0x0123456789ABCDEF);
constexpr uint64_t kReportAccessLength = 16;
constexpr loom_sanitizer_site_id_t kReportSiteId = 0;
constexpr uint64_t kReportShadowAddress = UINT64_C(0x0000056789ABCDEF);
constexpr uint64_t kReportShadowValue = UINT64_C(0xF0);

constexpr loom_sanitizer_site_id_t kTsanCurrentSiteId = 0;
constexpr loom_sanitizer_site_id_t kTsanPriorSiteId = 1;
constexpr uint64_t kTsanMemoryAddress = 0;
constexpr uint64_t kTsanShadowAddress = 0;
constexpr uint64_t kTsanShadowValue = 0;
constexpr uint32_t kTsanAccessLength = sizeof(uint32_t);

uint32_t LoadLeU32(const uint8_t* data, iree_host_size_t offset) {
  return ((uint32_t)data[offset]) | ((uint32_t)data[offset + 1] << 8) |
         ((uint32_t)data[offset + 2] << 16) |
         ((uint32_t)data[offset + 3] << 24);
}

uint16_t LoadLeU16(const uint8_t* data, iree_host_size_t offset) {
  return ((uint16_t)data[offset]) | ((uint16_t)data[offset + 1] << 8);
}

uint64_t LoadLeU64(const uint8_t* data, iree_host_size_t offset) {
  return ((uint64_t)LoadLeU32(data, offset)) |
         ((uint64_t)LoadLeU32(data, offset + 4) << 32);
}

std::string StatusToStringAndFree(iree_status_t status) {
  iree_host_size_t length = 0;
  iree_status_format(status, /*buffer_capacity=*/0, /*buffer=*/nullptr,
                     &length);
  std::vector<char> buffer(length + 1);
  if (length != 0) {
    iree_status_format(status, buffer.size(), buffer.data(), &length);
  }
  std::string message(buffer.data(), length);
  iree_status_free(status);
  return message;
}

std::string DiagnosticSummary(const DiagnosticCapture& capture) {
  std::string result;
  for (const CapturedDiagnostic& diagnostic : capture.diagnostics) {
    if (!result.empty()) {
      result += "\n";
    }
    result += diagnostic.error ? diagnostic.error->error_id : "<unknown>";
    result += ": ";
    result += diagnostic.error ? diagnostic.error->summary : "";
    for (const loom_diagnostic_param_t& param : diagnostic.params) {
      result += " [";
      switch (param.kind) {
        case LOOM_PARAM_STRING:
          result.append(param.string.data, param.string.size);
          break;
        case LOOM_PARAM_I64:
          result += std::to_string(param.i64);
          break;
        case LOOM_PARAM_U32:
          result += std::to_string(param.u32);
          break;
        case LOOM_PARAM_U64:
          result += std::to_string(param.u64);
          break;
        default:
          result += "?";
          break;
      }
      result += "]";
    }
  }
  return result;
}

iree_status_t InitializeAmdgpuContext(
    const loom_target_environment_t* target_environment,
    loom_context_t* context) {
  loom_context_initialize(iree_allocator_system(), context);
  iree_status_t status = loom_op_registry_register_all_dialects(context);
  if (iree_status_is_ok(status)) {
    status =
        loom_target_environment_register_context(target_environment, context);
  }
  if (iree_status_is_ok(status)) {
    status = loom_context_finalize(context);
  }
  if (!iree_status_is_ok(status)) {
    loom_context_deinitialize(context);
  }
  return status;
}

iree_status_t ReadDeviceBufferBytes(iree_hal_device_t* device,
                                    iree_hal_allocator_t* allocator,
                                    iree_hal_buffer_t* source_buffer,
                                    iree_device_size_t source_offset,
                                    iree_device_size_t length,
                                    std::vector<uint8_t>* out_data) {
  out_data->clear();
  if (source_offset > iree_hal_buffer_byte_length(source_buffer)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "readback offset %" PRIu64 " exceeds buffer length %" PRIu64,
        (uint64_t)source_offset,
        (uint64_t)iree_hal_buffer_byte_length(source_buffer));
  }
  if (length > iree_hal_buffer_byte_length(source_buffer) - source_offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "readback length %" PRIu64 " at offset %" PRIu64
        " exceeds buffer length %" PRIu64,
        (uint64_t)length, (uint64_t)source_offset,
        (uint64_t)iree_hal_buffer_byte_length(source_buffer));
  }
  if (length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "readback length %" PRIu64
                            " exceeds host vector capacity",
                            (uint64_t)length);
  }

  std::vector<uint8_t> data((iree_host_size_t)length);
  iree_hal_buffer_t* staging_buffer = nullptr;
  iree_hal_semaphore_t* semaphore = nullptr;

  iree_hal_buffer_params_t staging_params = {};
  staging_params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  staging_params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;

  iree_status_t status = iree_hal_allocator_allocate_buffer(
      allocator, staging_params, length, &staging_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore);
  }
  uint64_t signal_value = 1;
  iree_hal_semaphore_list_t signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&semaphore,
      /*.payload_values=*/&signal_value,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_copy(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
        signal_list, source_buffer, source_offset, staging_buffer,
        /*target_offset=*/0, length, IREE_HAL_COPY_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_wait(signal_list, iree_infinite_timeout(),
                                          IREE_ASYNC_WAIT_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_map_read(staging_buffer, /*offset=*/0, data.data(),
                                      data.size());
  }
  if (iree_status_is_ok(status)) {
    *out_data = std::move(data);
  }

  iree_hal_semaphore_release(semaphore);
  iree_hal_buffer_release(staging_buffer);
  return status;
}

template <typename T>
iree_status_t ReadDeviceBufferData(iree_hal_device_t* device,
                                   iree_hal_allocator_t* allocator,
                                   iree_hal_buffer_t* source_buffer,
                                   std::vector<T>* out_data) {
  out_data->clear();
  const iree_device_size_t byte_length =
      iree_hal_buffer_byte_length(source_buffer);
  if (byte_length % sizeof(T) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "readback length %" PRIu64
                            " is not divisible by element size %" PRIhsz,
                            (uint64_t)byte_length, sizeof(T));
  }

  std::vector<uint8_t> bytes;
  IREE_RETURN_IF_ERROR(ReadDeviceBufferBytes(device, allocator, source_buffer,
                                             /*source_offset=*/0, byte_length,
                                             &bytes));
  std::vector<T> data((iree_host_size_t)(byte_length / sizeof(T)));
  std::memcpy(data.data(), bytes.data(), (iree_host_size_t)byte_length);
  *out_data = std::move(data);
  return iree_ok_status();
}

iree_status_t ReadAmdgpuAsanConfig(iree_hal_device_t* device,
                                   iree_hal_allocator_t* allocator,
                                   iree_hal_executable_t* executable,
                                   iree_hal_amdgpu_asan_config_t* out_config) {
  *out_config = {};
  bool found = false;
  iree_hal_executable_global_t global = iree_hal_executable_global_invalid();
  IREE_RETURN_IF_ERROR(iree_hal_executable_try_lookup_global_by_name(
      executable, IREE_SV(IREE_HAL_AMDGPU_ASAN_CONFIG_GLOBAL_NAME), &found,
      &global));
  if (!found) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "executable has no AMDGPU ASAN config global");
  }

  iree_hal_executable_global_info_t info = {};
  IREE_RETURN_IF_ERROR(
      iree_hal_executable_global_info(executable, global, &info));
  if (!iree_string_view_equal(
          info.name, IREE_SV(IREE_HAL_AMDGPU_ASAN_CONFIG_GLOBAL_NAME))) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "AMDGPU ASAN config global name mismatch");
  }
  if (info.byte_length != sizeof(iree_hal_amdgpu_asan_config_t)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU ASAN config global has length %" PRIu64
                            " but expected %" PRIhsz,
                            (uint64_t)info.byte_length,
                            sizeof(iree_hal_amdgpu_asan_config_t));
  }

  iree_hal_buffer_t* global_buffer = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_executable_global_buffer(
      executable, global, IREE_HAL_QUEUE_AFFINITY_ANY, &global_buffer));
  std::vector<iree_hal_amdgpu_asan_config_t> configs;
  IREE_RETURN_IF_ERROR(
      ReadDeviceBufferData(device, allocator, global_buffer, &configs));
  if (configs.size() != 1) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "AMDGPU ASAN config read returned %" PRIhsz
                            " records instead of 1",
                            configs.size());
  }
  *out_config = configs[0];
  return iree_ok_status();
}

void ExpectAmdgpuAsanConfigEnabled(
    const iree_hal_amdgpu_asan_config_t& config) {
  EXPECT_EQ(config.record_length, sizeof(config));
  EXPECT_EQ(config.abi_version, IREE_HAL_AMDGPU_ASAN_CONFIG_ABI_VERSION_0);
  EXPECT_NE(config.flags & IREE_HAL_AMDGPU_ASAN_CONFIG_FLAG_ENABLED, 0u);
  EXPECT_LT(config.shadow_scale_shift, 63u);
  EXPECT_NE(config.shadow_base, 0u);
  EXPECT_NE(config.application_window_size, 0u);
  EXPECT_NE(config.shadow_size, 0u);
  EXPECT_NE(config.shadow_slab_size, 0u);
}

uint64_t AmdgpuAsanShadowAddress(const iree_hal_amdgpu_asan_config_t& config,
                                 uint64_t application_address) {
  return config.shadow_base +
         (application_address >> config.shadow_scale_shift);
}

class AmdgpuSanitizerDeviceEventRecorder {
 public:
  AmdgpuSanitizerDeviceEventRecorder() = default;

  AmdgpuSanitizerDeviceEventRecorder(
      const AmdgpuSanitizerDeviceEventRecorder&) = delete;
  AmdgpuSanitizerDeviceEventRecorder& operator=(
      const AmdgpuSanitizerDeviceEventRecorder&) = delete;

  iree_hal_device_event_sink_t sink() {
    iree_hal_device_event_sink_t sink;
    sink.fn = AmdgpuSanitizerDeviceEventRecorder::Capture;
    sink.user_data = this;
    return sink;
  }

  void Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    asan_report_count_ = 0;
    tsan_report_count_ = 0;
    last_source_ = iree_hal_device_event_source_default();
    last_site_available_ = false;
    last_site_ = iree_hal_device_event_site_default();
    last_site_source_file_.clear();
    last_site_function_name_.clear();
    last_site_operation_name_.clear();
    last_site_producer_payload_.clear();
    last_asan_report_ = {};
    last_tsan_report_ = {};
  }

  void WaitForAsanReportCount(iree_host_size_t expected_count) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return asan_report_count_ >= expected_count; });
  }

  void WaitForTsanReportCount(iree_host_size_t expected_count) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return tsan_report_count_ >= expected_count; });
  }

  iree_host_size_t asan_report_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return asan_report_count_;
  }

  iree_host_size_t tsan_report_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tsan_report_count_;
  }

  iree_hal_device_event_source_t last_source() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_source_;
  }

  iree_hal_device_asan_report_t last_report() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_asan_report_;
  }

  iree_hal_device_tsan_report_t last_tsan_report() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_tsan_report_;
  }

  bool last_site_available() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_site_available_;
  }

  iree_hal_device_event_site_t last_site() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_site_;
  }

 private:
  static iree_string_view_t StringViewFromStorage(const std::string& storage) {
    return iree_make_string_view(storage.data(), storage.size());
  }

  static void CaptureStringView(iree_string_view_t view,
                                std::string* out_storage,
                                iree_string_view_t* out_view) {
    if (!iree_string_view_is_empty(view)) {
      out_storage->assign(view.data, view.size);
      *out_view = StringViewFromStorage(*out_storage);
    } else {
      out_storage->clear();
      *out_view = iree_string_view_empty();
    }
  }

  static void CaptureByteSpan(iree_const_byte_span_t span,
                              std::vector<uint8_t>* out_storage,
                              iree_const_byte_span_t* out_span) {
    if (!iree_const_byte_span_is_empty(span)) {
      out_storage->assign(span.data, span.data + span.data_length);
      *out_span =
          iree_make_const_byte_span(out_storage->data(), out_storage->size());
    } else {
      out_storage->clear();
      *out_span = iree_const_byte_span_empty();
    }
  }

  static void CaptureSite(AmdgpuSanitizerDeviceEventRecorder* recorder,
                          const iree_hal_device_event_site_t* site) {
    recorder->last_site_available_ = site != nullptr;
    recorder->last_site_ = iree_hal_device_event_site_default();
    if (!site) return;
    recorder->last_site_ = *site;
    CaptureStringView(site->source_file, &recorder->last_site_source_file_,
                      &recorder->last_site_.source_file);
    CaptureStringView(site->function_name, &recorder->last_site_function_name_,
                      &recorder->last_site_.function_name);
    CaptureStringView(site->operation_name,
                      &recorder->last_site_operation_name_,
                      &recorder->last_site_.operation_name);
    CaptureByteSpan(site->producer_payload,
                    &recorder->last_site_producer_payload_,
                    &recorder->last_site_.producer_payload);
  }

  static void Capture(void* user_data, const iree_hal_device_event_t* event) {
    auto* recorder =
        static_cast<AmdgpuSanitizerDeviceEventRecorder*>(user_data);
    if (!event->payload.data) {
      return;
    }
    std::lock_guard<std::mutex> lock(recorder->mutex_);
    if (event->type == IREE_HAL_DEVICE_EVENT_TYPE_ASAN_REPORT &&
        event->payload.data_length >= sizeof(iree_hal_device_asan_report_t)) {
      recorder->last_source_ = event->source;
      CaptureSite(recorder, event->site);
      std::memcpy(&recorder->last_asan_report_, event->payload.data,
                  sizeof(recorder->last_asan_report_));
      ++recorder->asan_report_count_;
    } else if (event->type == IREE_HAL_DEVICE_EVENT_TYPE_TSAN_REPORT &&
               event->payload.data_length >=
                   sizeof(iree_hal_device_tsan_report_t)) {
      recorder->last_source_ = event->source;
      CaptureSite(recorder, event->site);
      std::memcpy(&recorder->last_tsan_report_, event->payload.data,
                  sizeof(recorder->last_tsan_report_));
      ++recorder->tsan_report_count_;
    } else {
      return;
    }
    recorder->condition_.notify_all();
  }

  // Protects captured report state.
  mutable std::mutex mutex_;
  // Notifies tests waiting for captured ASAN reports.
  std::condition_variable condition_;
  // Count of ASAN reports captured since the last reset.
  iree_host_size_t asan_report_count_ = 0;
  // Count of TSAN reports captured since the last reset.
  iree_host_size_t tsan_report_count_ = 0;
  // Source attribution from the last captured sanitizer report.
  iree_hal_device_event_source_t last_source_ =
      iree_hal_device_event_source_default();
  // True when the last captured sanitizer report carried resolved site
  // metadata.
  bool last_site_available_ = false;
  // Resolved site metadata from the last captured sanitizer report.
  iree_hal_device_event_site_t last_site_ =
      iree_hal_device_event_site_default();
  // Owned source file storage referenced by |last_site_|.
  std::string last_site_source_file_;
  // Owned function name storage referenced by |last_site_|.
  std::string last_site_function_name_;
  // Owned operation name storage referenced by |last_site_|.
  std::string last_site_operation_name_;
  // Owned producer payload storage referenced by |last_site_|.
  std::vector<uint8_t> last_site_producer_payload_;
  // Payload from the last captured ASAN report.
  iree_hal_device_asan_report_t last_asan_report_ = {};
  // Payload from the last captured TSAN report.
  iree_hal_device_tsan_report_t last_tsan_report_ = {};
};

iree_status_t AllocateAsanDeviceBuffer(iree_hal_allocator_t* allocator,
                                       iree_device_size_t byte_length,
                                       iree_hal_buffer_t** out_buffer) {
  *out_buffer = nullptr;
  iree_hal_buffer_params_t params = {};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage =
      IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
  return iree_hal_allocator_allocate_buffer(allocator, params, byte_length,
                                            out_buffer);
}

uint64_t AmdgpuDevicePointer(iree_hal_buffer_t* buffer) {
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer);
  void* base_pointer = iree_hal_amdgpu_buffer_device_pointer(allocated_buffer);
  return (uint64_t)((uintptr_t)base_pointer +
                    (uintptr_t)iree_hal_buffer_byte_offset(buffer));
}

iree_status_t QueueDispatchTwoBuffers(iree_hal_device_t* device,
                                      iree_hal_executable_t* executable,
                                      iree_string_view_t function_name,
                                      iree_hal_buffer_t* binding0,
                                      iree_hal_buffer_t* binding1) {
  iree_hal_executable_function_t function =
      iree_hal_executable_function_invalid();
  IREE_RETURN_IF_ERROR(iree_hal_executable_lookup_function_by_name(
      executable, function_name, &function));

  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_buffer_ref(binding0, 0,
                               iree_hal_buffer_byte_length(binding0)),
      iree_hal_make_buffer_ref(binding1, 0,
                               iree_hal_buffer_byte_length(binding1)),
  };
  iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  iree_hal_semaphore_t* semaphore = nullptr;
  iree_status_t status = iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore);
  uint64_t signal_value = 1;
  iree_hal_semaphore_list_t signal_semaphores = {
      /*.count=*/1,
      /*.semaphores=*/&semaphore,
      /*.payload_values=*/&signal_value,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_dispatch(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
        signal_semaphores, executable, function,
        iree_hal_make_static_dispatch_config(1, 1, 1),
        iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(semaphore, signal_value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }
  iree_hal_semaphore_release(semaphore);
  return status;
}

class AmdgpuAsanHalRuntime {
 public:
  AmdgpuAsanHalRuntime() = default;
  ~AmdgpuAsanHalRuntime() { Deinitialize(); }

  AmdgpuAsanHalRuntime(const AmdgpuAsanHalRuntime&) = delete;
  AmdgpuAsanHalRuntime& operator=(const AmdgpuAsanHalRuntime&) = delete;

  iree_status_t Initialize() {
    return Initialize(IREE_SV("asan"), iree_hal_device_event_sink_discard());
  }

  iree_status_t Initialize(iree_hal_device_event_sink_t event_sink) {
    return Initialize(IREE_SV("asan"), event_sink);
  }

  iree_status_t Initialize(iree_string_view_t sanitizer,
                           iree_hal_device_event_sink_t event_sink) {
    iree_async_proactor_pool_t* proactor_pool = nullptr;
    iree_async_frontier_tracker_t* frontier_tracker = nullptr;

    iree_status_t status = iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/nullptr,
        iree_async_proactor_pool_options_default(), iree_allocator_system(),
        &proactor_pool);
    if (iree_status_is_ok(status)) {
      status = iree_async_frontier_tracker_create(
          iree_async_frontier_tracker_options_default(),
          iree_allocator_system(), &frontier_tracker);
    }

    iree_hal_driver_registry_t* driver_registry = nullptr;
    if (iree_status_is_ok(status)) {
      status = iree_hal_driver_registry_allocate(iree_allocator_system(),
                                                 &driver_registry);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_driver_module_register(driver_registry);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_driver_registry_try_create(
          driver_registry, IREE_SV("amdgpu"), iree_allocator_system(),
          &driver_);
    }
    iree_hal_driver_registry_free(driver_registry);

    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.event_sink = event_sink;
    create_params.proactor_pool = proactor_pool;
    iree_string_pair_t device_parameters[1] = {
        iree_make_string_pair(IREE_SV("hal.sanitizer"), sanitizer),
    };
    if (iree_status_is_ok(status)) {
      status = iree_hal_driver_create_device_by_ordinal(
          driver_, /*device_ordinal=*/0, IREE_ARRAYSIZE(device_parameters),
          device_parameters, &create_params, iree_allocator_system(),
          &runtime_.device);
    }
    iree_async_proactor_pool_release(proactor_pool);

    if (iree_status_is_ok(status)) {
      status = iree_hal_device_group_create_from_device(
          runtime_.device, frontier_tracker, iree_allocator_system(),
          &runtime_.device_group);
    }
    iree_async_frontier_tracker_release(frontier_tracker);

    if (iree_status_is_ok(status)) {
      status = iree_hal_executable_cache_create(
          runtime_.device, IREE_SV("loom"), &runtime_.executable_cache);
    }
    if (!iree_status_is_ok(status)) {
      Deinitialize();
    }
    return status;
  }

  const loom_run_hal_runtime_t* runtime() const { return &runtime_; }

  iree_hal_allocator_t* allocator() const {
    return iree_hal_device_allocator(runtime_.device);
  }

 private:
  void Deinitialize() {
    loom_run_hal_runtime_deinitialize(&runtime_);
    iree_hal_driver_release(driver_);
    driver_ = nullptr;
  }

  loom_run_hal_runtime_t runtime_ = {};
  iree_hal_driver_t* driver_ = nullptr;
};

iree_status_t GetOrCreateModuleSymbol(loom_module_t* module,
                                      iree_string_view_t name,
                                      loom_symbol_ref_t* out_symbol_ref) {
  *out_symbol_ref = loom_symbol_ref_null();
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, name, &name_id));
  uint16_t symbol_id = loom_module_find_symbol(module, name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_module_add_symbol(module, name_id, &symbol_id));
  }
  *out_symbol_ref = (loom_symbol_ref_t){
      /*.module_id=*/0,
      /*.symbol_id=*/symbol_id,
  };
  return iree_ok_status();
}

iree_status_t BuildU32Attr(loom_builder_t* builder, iree_string_view_t name,
                           uint32_t value, loom_named_attr_t* out_attr) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(builder, name, &name_id));
  *out_attr = {};
  out_attr->name_id = name_id;
  out_attr->value = loom_attr_i64(value);
  return iree_ok_status();
}

iree_status_t AppendSingleSanitizerSiteTable(loom_module_t* module,
                                             loom_sanitizer_site_id_t site_id,
                                             uint32_t op_kind) {
  loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_register_source(
      module, IREE_SV("amdgpu_hal_sanitizer_feedback_test.loom"), &source_id));
  loom_location_entry_t source_entry =
      loom_location_file_range(source_id, /*start_line=*/13,
                               /*start_col=*/3, /*end_line=*/13,
                               /*end_col=*/11);
  loom_location_id_t source_location = LOOM_LOCATION_UNKNOWN;
  IREE_RETURN_IF_ERROR(
      loom_module_add_location(module, source_entry, &source_location));

  loom_sanitizer_site_row_t row = {};
  row.site_id = site_id;
  row.op_kind = op_kind;
  row.location = source_location;
  row.payload_location = LOOM_LOCATION_UNKNOWN;
  row.source_location = source_location;

  loom_sanitizer_site_collection_t collection = {};
  collection.rows = &row;
  collection.row_count = 1;

  iree_const_byte_span_t site_table = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(loom_sanitizer_site_table_encode(
      module, &collection, &module->arena, &site_table));

  loom_symbol_ref_t symbol_ref = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(GetOrCreateModuleSymbol(
      module, IREE_SV(LOOM_SANITIZER_SITE_TABLE_SYMBOL_NAME), &symbol_ref));
  loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->defining_op != nullptr) {
    return iree_make_status(
        IREE_STATUS_ALREADY_EXISTS,
        "test sanitizer site table symbol is already defined");
  }

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  loom_op_t* rodata_op = nullptr;
  return loom_global_rodata_build(
      &builder, LOOM_GLOBAL_RODATA_BUILD_FLAG_HAS_ALIGNMENT, symbol_ref,
      site_table, 8, LOOM_LOCATION_UNKNOWN, &rodata_op);
}

void ExpectSingleSanitizerSiteTable(loom_module_t* module,
                                    loom_sanitizer_site_id_t site_id,
                                    uint32_t op_kind) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(
      module, IREE_SV(LOOM_SANITIZER_SITE_TABLE_SYMBOL_NAME), &name_id));
  const uint16_t symbol_id = loom_module_find_symbol(module, name_id);
  ASSERT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
  ASSERT_NE(symbol->defining_op, nullptr);
  ASSERT_TRUE(loom_global_rodata_isa(symbol->defining_op));

  const iree_const_byte_span_t contents =
      loom_global_rodata_contents(symbol->defining_op);
  ASSERT_GE(contents.data_length, LOOM_SANITIZER_SITE_TABLE_HEADER_LENGTH +
                                      LOOM_SANITIZER_SITE_TABLE_RECORD_LENGTH);
  EXPECT_EQ(
      LoadLeU32(contents.data, LOOM_SANITIZER_SITE_TABLE_HEADER_MAGIC_OFFSET),
      LOOM_SANITIZER_SITE_TABLE_MAGIC);
  EXPECT_EQ(contents.data[LOOM_SANITIZER_SITE_TABLE_HEADER_VERSION_OFFSET],
            LOOM_SANITIZER_SITE_TABLE_VERSION);
  EXPECT_EQ(LoadLeU32(contents.data,
                      LOOM_SANITIZER_SITE_TABLE_HEADER_ROW_COUNT_OFFSET),
            1u);

  const uint8_t* record =
      contents.data + LOOM_SANITIZER_SITE_TABLE_HEADER_LENGTH;
  EXPECT_EQ(LoadLeU32(record, LOOM_SANITIZER_SITE_TABLE_RECORD_SITE_ID_OFFSET),
            site_id);
  EXPECT_EQ(LoadLeU32(record, LOOM_SANITIZER_SITE_TABLE_RECORD_OP_KIND_OFFSET),
            op_kind);
  EXPECT_EQ(LoadLeU32(record, LOOM_SANITIZER_SITE_TABLE_RECORD_FLAGS_OFFSET),
            LOOM_SANITIZER_SITE_TABLE_RECORD_HAS_SOURCE_LOCATION);
  EXPECT_NE(
      LoadLeU32(record,
                LOOM_SANITIZER_SITE_TABLE_RECORD_SOURCE_NAME_LENGTH_OFFSET),
      0u);
  EXPECT_EQ(
      LoadLeU16(record, LOOM_SANITIZER_SITE_TABLE_RECORD_SOURCE_KIND_OFFSET),
      LOOM_SANITIZER_SITE_TABLE_SOURCE_KIND_FILE);
}

iree_status_t BuildConstU32(loom_builder_t* builder,
                            const loom_low_descriptor_set_t* descriptor_set,
                            loom_amdgpu_descriptor_ref_t descriptor_ref,
                            uint32_t value, loom_type_t result_type,
                            loom_location_id_t location,
                            loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_low_descriptor_t* descriptor = nullptr;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_descriptor_ref(
      builder, descriptor_set, descriptor_ref, &descriptor, &opcode_id));

  loom_named_attr_t imm32_attr = {};
  IREE_RETURN_IF_ERROR(
      BuildU32Attr(builder, IREE_SV("imm32"), value, &imm32_attr));
  loom_op_t* const_op = nullptr;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_const(
      builder, descriptor_set, descriptor, opcode_id,
      loom_make_named_attr_slice(&imm32_attr, 1), result_type, location,
      &const_op));
  *out_value = loom_low_const_result(const_op);
  return iree_ok_status();
}

iree_status_t BuildRegisterU32Constant(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint16_t register_class, uint32_t value, loom_location_id_t location,
    loom_value_id_t* out_value) {
  loom_type_t register_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_build_register_type(descriptor_set, register_class,
                                   /*unit_count=*/1, &register_type));
  const loom_amdgpu_descriptor_ref_t descriptor_ref =
      register_class == LOOM_AMDGPU_REG_CLASS_ID_SGPR
          ? LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32
          : LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32;
  return BuildConstU32(builder, descriptor_set, descriptor_ref, value,
                       register_type, location, out_value);
}

iree_status_t BuildRegisterU64Constant(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint16_t register_class, uint64_t value, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(BuildRegisterU32Constant(builder, descriptor_set,
                                                register_class, (uint32_t)value,
                                                location, &low_value));
  loom_value_id_t high_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      BuildRegisterU32Constant(builder, descriptor_set, register_class,
                               (uint32_t)(value >> 32), location, &high_value));

  loom_type_t register_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_build_register_type(descriptor_set, register_class,
                                   /*unit_count=*/2, &register_x2_type));
  const loom_value_id_t parts[] = {low_value, high_value};
  loom_op_t* concat_op = nullptr;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(builder, parts, IREE_ARRAYSIZE(parts),
                            register_x2_type, location, &concat_op));
  *out_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

iree_status_t BuildSgprLoadDwordX2(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t base_address, uint32_t byte_offset,
    loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
      /*unit_count=*/2, &sgpr_x2_type));

  const loom_low_descriptor_t* descriptor = nullptr;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_descriptor_ref(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORDX2_OFFSET_ONLY, &descriptor,
      &opcode_id));
  loom_named_attr_t offset_attr = {};
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_offset_attr(
      builder, byte_offset, &offset_attr));
  loom_op_t* load_op = nullptr;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, opcode_id, &base_address,
      /*operand_count=*/1, loom_make_named_attr_slice(&offset_attr, 1),
      &sgpr_x2_type, /*result_count=*/1, /*tied_results=*/nullptr,
      /*tied_result_count=*/0, location, &load_op));
  *out_value = loom_value_slice_get(loom_low_op_results(load_op), 0);
  return iree_ok_status();
}

iree_status_t BuildWaitCounterMask(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t counter_mask, uint16_t target_count, loom_location_id_t location) {
  loom_amdgpu_wait_packet_selection_t selection = {};
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_select_counter_mask(
      descriptor_set, counter_mask, target_count, &selection));
  loom_named_attr_t
      attrs[LOOM_AMDGPU_WAIT_PACKET_SELECTION_IMMEDIATE_CAPACITY] = {};
  for (iree_host_size_t i = 0; i < selection.immediate_count; ++i) {
    IREE_RETURN_IF_ERROR(BuildU32Attr(builder, selection.immediates[i].name,
                                      selection.immediates[i].value,
                                      &attrs[i]));
  }
  iree_string_view_t key = loom_low_descriptor_set_string(
      descriptor_set, selection.descriptor->key_string_offset);
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(builder, key, &opcode_id));
  loom_op_t* wait_op = nullptr;
  return loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, selection.descriptor, opcode_id,
      /*operands=*/nullptr, /*operand_count=*/0,
      loom_make_named_attr_slice(attrs, selection.immediate_count),
      /*result_types=*/nullptr, /*result_count=*/0, /*tied_results=*/nullptr,
      /*tied_result_count=*/0, location, &wait_op);
}

iree_status_t BuildLiveIn(loom_builder_t* builder, iree_string_view_t source,
                          loom_type_t result_type, loom_location_id_t location,
                          loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_string_id_t source_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(builder->module, source, &source_id));
  loom_op_t* live_in_op = nullptr;
  IREE_RETURN_IF_ERROR(loom_low_live_in_build(
      builder, source_id, loom_make_named_attr_slice(nullptr, 0), result_type,
      location, &live_in_op));
  *out_value = loom_low_live_in_result(live_in_op);
  return iree_ok_status();
}

iree_status_t FindKernelOpByName(loom_module_t* module, iree_string_view_t name,
                                 loom_op_t** out_op) {
  *out_op = nullptr;
  loom_block_t* module_block = loom_module_block(module);
  loom_op_t* op = nullptr;
  loom_block_for_each_op(module_block, op) {
    if (!loom_low_kernel_def_isa(op)) continue;
    const loom_symbol_ref_t callee_ref = loom_low_kernel_def_callee(op);
    if (callee_ref.module_id != 0 ||
        callee_ref.symbol_id >= module->symbols.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low.kernel.def has an invalid callee symbol");
    }
    const loom_symbol_t* symbol =
        &module->symbols.entries[callee_ref.symbol_id];
    if (symbol->name_id == LOOM_STRING_ID_INVALID ||
        symbol->name_id >= module->strings.count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low.kernel.def callee symbol has an invalid name");
    }
    if (!iree_string_view_equal(module->strings.entries[symbol->name_id],
                                name)) {
      continue;
    }
    if (*out_op != nullptr) {
      return iree_make_status(
          IREE_STATUS_ALREADY_EXISTS,
          "expected one low.kernel.def named '%.*s' but found more than one",
          (int)name.size, name.data);
    }
    *out_op = op;
  }
  if (*out_op == nullptr) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "low.kernel.def named '%.*s' was not found",
                            (int)name.size, name.data);
  }
  return iree_ok_status();
}

iree_status_t InsertBlockAfter(loom_module_t* module, loom_block_t* block,
                               loom_block_t** out_block) {
  *out_block = nullptr;
  if (block->parent_region == nullptr) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "expected block to belong to a region");
  }
  return loom_region_insert_block(module, block->parent_region,
                                  (uint16_t)(block->region_index + 1),
                                  out_block);
}

iree_status_t BuildAsanReportPublishAndReturn(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t feedback_config_symbol,
    const loom_amdgpu_sanitizer_report_source_t* source,
    const loom_amdgpu_sanitizer_access_report_t* report,
    loom_location_id_t location) {
  loom_block_t* entry_block = builder->ip.block;
  loom_block_t* feedback_block = nullptr;
  IREE_RETURN_IF_ERROR(
      InsertBlockAfter(builder->module, entry_block, &feedback_block));
  loom_block_t* return_block = nullptr;
  IREE_RETURN_IF_ERROR(
      InsertBlockAfter(builder->module, feedback_block, &return_block));

  loom_amdgpu_feedback_config_values_t config_values = {};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_config_values(
      builder, descriptor_set, feedback_config_symbol, location,
      &config_values));
  loom_value_id_t feedback_enabled_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_config_enabled_scc(
      builder, descriptor_set, config_values.flags, location,
      &feedback_enabled_scc));
  loom_op_t* enabled_branch = nullptr;
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(builder, feedback_enabled_scc,
                                              feedback_block, return_block,
                                              location, &enabled_branch));

  loom_builder_set_block(builder, feedback_block);
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_channel_header_values(
      builder, descriptor_set, config_values.channel_base, location,
      &channel_values));
  const uint32_t packet_length = (uint32_t)loom_amdgpu_feedback_packet_length(
      LOOM_AMDGPU_SANITIZER_ACCESS_REPORT_BYTE_LENGTH);
  loom_amdgpu_feedback_reservation_t reservation = {};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_reservation(
      builder, descriptor_set, channel_values.address, channel_values.ring_base,
      channel_values.ring_capacity, packet_length, location, &reservation));

  loom_block_t* reservation_block = builder->ip.block;
  loom_block_t* report_block = nullptr;
  IREE_RETURN_IF_ERROR(
      InsertBlockAfter(builder->module, reservation_block, &report_block));
  loom_value_id_t reservation_succeeded_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_reservation_succeeded_scc(
      builder, descriptor_set, reservation.reserved_mask, location,
      &reservation_succeeded_scc));
  loom_op_t* reserved_branch = nullptr;
  IREE_RETURN_IF_ERROR(
      loom_low_cond_br_build(builder, reservation_succeeded_scc, report_block,
                             return_block, location, &reserved_branch));

  loom_builder_set_block(builder, report_block);
  const loom_amdgpu_feedback_packet_header_t header = {
      /*.record_length=*/packet_length,
      /*.kind=*/LOOM_AMDGPU_FEEDBACK_PACKET_KIND_ASAN,
      /*.flags=*/LOOM_AMDGPU_FEEDBACK_PACKET_FLAG_NONE,
      /*.sequence=*/reservation.sequence,
      /*.source_dispatch_ptr=*/source->dispatch_ptr,
      /*.source_workgroup_id_x=*/source->workgroup_id_x,
      /*.source_workitem_id_x=*/source->workitem_id_x,
      /*.source_context=*/config_values.source_context,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_header(
      builder, descriptor_set, &reservation.packet_address, &header, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_sanitizer_access_report_payload(
      builder, descriptor_set, &reservation.packet_address, report, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_publish_packet(
      builder, descriptor_set, &reservation.packet_address,
      config_values.notify_signal, location));
  loom_op_t* publish_branch = nullptr;
  IREE_RETURN_IF_ERROR(loom_low_br_build(builder, return_block,
                                         /*args=*/nullptr, /*args_count=*/0,
                                         location, &publish_branch));

  loom_builder_set_block(builder, return_block);
  loom_op_t* return_op = nullptr;
  return loom_low_return_build(builder, /*values=*/nullptr, /*values_count=*/0,
                               location, &return_op);
}

iree_status_t BuildTsanReportPublishAndReturn(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t feedback_config_symbol,
    const loom_amdgpu_sanitizer_report_source_t* source,
    const loom_amdgpu_sanitizer_race_report_t* report,
    loom_location_id_t location) {
  loom_block_t* entry_block = builder->ip.block;
  loom_block_t* feedback_block = nullptr;
  IREE_RETURN_IF_ERROR(
      InsertBlockAfter(builder->module, entry_block, &feedback_block));
  loom_block_t* return_block = nullptr;
  IREE_RETURN_IF_ERROR(
      InsertBlockAfter(builder->module, feedback_block, &return_block));

  loom_amdgpu_feedback_config_values_t config_values = {};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_config_values(
      builder, descriptor_set, feedback_config_symbol, location,
      &config_values));
  loom_value_id_t feedback_enabled_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_config_enabled_scc(
      builder, descriptor_set, config_values.flags, location,
      &feedback_enabled_scc));
  loom_op_t* enabled_branch = nullptr;
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(builder, feedback_enabled_scc,
                                              feedback_block, return_block,
                                              location, &enabled_branch));

  loom_builder_set_block(builder, feedback_block);
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_channel_header_values(
      builder, descriptor_set, config_values.channel_base, location,
      &channel_values));
  const uint32_t packet_length = (uint32_t)loom_amdgpu_feedback_packet_length(
      LOOM_AMDGPU_TSAN_REPORT_BYTE_LENGTH);
  loom_amdgpu_feedback_reservation_t reservation = {};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_reservation(
      builder, descriptor_set, channel_values.address, channel_values.ring_base,
      channel_values.ring_capacity, packet_length, location, &reservation));

  loom_block_t* reservation_block = builder->ip.block;
  loom_block_t* report_block = nullptr;
  IREE_RETURN_IF_ERROR(
      InsertBlockAfter(builder->module, reservation_block, &report_block));
  loom_value_id_t reservation_succeeded_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_reservation_succeeded_scc(
      builder, descriptor_set, reservation.reserved_mask, location,
      &reservation_succeeded_scc));
  loom_op_t* reserved_branch = nullptr;
  IREE_RETURN_IF_ERROR(
      loom_low_cond_br_build(builder, reservation_succeeded_scc, report_block,
                             return_block, location, &reserved_branch));

  loom_builder_set_block(builder, report_block);
  const loom_amdgpu_feedback_packet_header_t header = {
      /*.record_length=*/packet_length,
      /*.kind=*/LOOM_AMDGPU_FEEDBACK_PACKET_KIND_TSAN,
      /*.flags=*/LOOM_AMDGPU_FEEDBACK_PACKET_FLAG_ASYNC,
      /*.sequence=*/reservation.sequence,
      /*.source_dispatch_ptr=*/source->dispatch_ptr,
      /*.source_workgroup_id_x=*/source->workgroup_id_x,
      /*.source_workitem_id_x=*/source->workitem_id_x,
      /*.source_context=*/config_values.source_context,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_header(
      builder, descriptor_set, &reservation.packet_address, &header, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_sanitizer_race_report_payload(
      builder, descriptor_set, &reservation.packet_address, report, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_publish_packet(
      builder, descriptor_set, &reservation.packet_address,
      config_values.notify_signal, location));
  loom_op_t* publish_branch = nullptr;
  IREE_RETURN_IF_ERROR(loom_low_br_build(builder, return_block,
                                         /*args=*/nullptr, /*args_count=*/0,
                                         location, &publish_branch));

  loom_builder_set_block(builder, return_block);
  loom_op_t* return_op = nullptr;
  return loom_low_return_build(builder, /*values=*/nullptr, /*values_count=*/0,
                               location, &return_op);
}

iree_status_t AppendAsanReportProducer(
    loom_module_t* module, const loom_low_descriptor_set_t* descriptor_set,
    iree_string_view_t kernel_name) {
  loom_op_t* kernel_op = nullptr;
  IREE_RETURN_IF_ERROR(FindKernelOpByName(module, kernel_name, &kernel_op));
  loom_region_t* body = loom_low_kernel_def_body(kernel_op);
  loom_block_t* entry_block = loom_region_entry_block(body);
  if (entry_block->op_count != 1 ||
      !loom_low_return_isa(entry_block->last_op)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "expected report kernel skeleton to contain only "
                            "one low.return terminator");
  }
  IREE_RETURN_IF_ERROR(loom_op_erase(module, entry_block->last_op));

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, entry_block, &builder);
  builder.ip.parent_op = kernel_op;
  constexpr loom_location_id_t location = LOOM_LOCATION_UNKNOWN;

  loom_symbol_ref_t feedback_config_symbol = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(GetOrCreateModuleSymbol(
      module, IREE_SV(IREE_HAL_AMDGPU_FEEDBACK_CONFIG_GLOBAL_NAME),
      &feedback_config_symbol));

  loom_amdgpu_sanitizer_report_source_t source = {};
  IREE_RETURN_IF_ERROR(BuildRegisterU64Constant(
      &builder, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
      /*value=*/0, location, &source.dispatch_ptr));
  IREE_RETURN_IF_ERROR(BuildRegisterU32Constant(
      &builder, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
      /*value=*/0, location, &source.workgroup_id_x));
  IREE_RETURN_IF_ERROR(BuildRegisterU32Constant(
      &builder, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
      /*value=*/0, location, &source.workitem_id_x));

  loom_amdgpu_sanitizer_access_report_t report = {
      /*.access_kind=*/LOOM_AMDGPU_SANITIZER_ACCESS_KIND_WRITE,
      /*.flags=*/LOOM_AMDGPU_SANITIZER_REPORT_FLAG_NONE,
  };
  IREE_RETURN_IF_ERROR(BuildRegisterU64Constant(
      &builder, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
      kReportFaultAddress, location, &report.fault_address));
  IREE_RETURN_IF_ERROR(BuildRegisterU64Constant(
      &builder, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
      kReportAccessLength, location, &report.access_size));
  IREE_RETURN_IF_ERROR(BuildRegisterU64Constant(
      &builder, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
      (uint64_t)kReportSiteId, location, &report.site_id));
  IREE_RETURN_IF_ERROR(BuildRegisterU64Constant(
      &builder, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
      kReportShadowAddress, location, &report.shadow_address));
  IREE_RETURN_IF_ERROR(BuildRegisterU64Constant(
      &builder, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
      kReportShadowValue, location, &report.shadow_value));

  return BuildAsanReportPublishAndReturn(&builder, descriptor_set,
                                         feedback_config_symbol, &source,
                                         &report, location);
}

iree_status_t AppendTsanReportProducer(
    loom_module_t* module, const loom_low_descriptor_set_t* descriptor_set,
    iree_string_view_t kernel_name) {
  loom_op_t* kernel_op = nullptr;
  IREE_RETURN_IF_ERROR(FindKernelOpByName(module, kernel_name, &kernel_op));
  loom_region_t* body = loom_low_kernel_def_body(kernel_op);
  loom_block_t* entry_block = loom_region_entry_block(body);
  if (entry_block->op_count != 1 ||
      !loom_low_return_isa(entry_block->last_op)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "expected report kernel skeleton to contain only "
                            "one low.return terminator");
  }
  IREE_RETURN_IF_ERROR(loom_op_erase(module, entry_block->last_op));

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, entry_block, &builder);
  builder.ip.parent_op = kernel_op;
  constexpr loom_location_id_t location = LOOM_LOCATION_UNKNOWN;

  loom_symbol_ref_t feedback_config_symbol = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(GetOrCreateModuleSymbol(
      module, IREE_SV(IREE_HAL_AMDGPU_FEEDBACK_CONFIG_GLOBAL_NAME),
      &feedback_config_symbol));

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
      /*unit_count=*/1, &sgpr_type));
  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
      /*unit_count=*/2, &sgpr_x2_type));
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
      /*unit_count=*/1, &vgpr_type));

  loom_amdgpu_sanitizer_report_source_t source = {};
  IREE_RETURN_IF_ERROR(BuildLiveIn(
      &builder, IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_DISPATCH_PTR_SOURCE),
      sgpr_x2_type, location, &source.dispatch_ptr));
  IREE_RETURN_IF_ERROR(BuildLiveIn(
      &builder, IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_WORKGROUP_ID_X_SOURCE),
      sgpr_type, location, &source.workgroup_id_x));
  IREE_RETURN_IF_ERROR(BuildLiveIn(
      &builder, IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_X_SOURCE),
      vgpr_type, location, &source.workitem_id_x));

  loom_value_id_t vgpr_zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(BuildRegisterU32Constant(
      &builder, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
      /*value=*/0, location, &vgpr_zero));
  loom_value_id_t vgpr_one = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(BuildRegisterU32Constant(
      &builder, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
      /*value=*/1, location, &vgpr_one));

  loom_amdgpu_sanitizer_race_report_t report = {};
  IREE_RETURN_IF_ERROR(BuildRegisterU32Constant(
      &builder, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
      LOOM_AMDGPU_TSAN_CHECK_KIND_DATA_RACE, location, &report.check_kind));
  IREE_RETURN_IF_ERROR(BuildRegisterU32Constant(
      &builder, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
      /*value=*/0, location, &report.flags));
  IREE_RETURN_IF_ERROR(BuildRegisterU32Constant(
      &builder, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
      LOOM_AMDGPU_TSAN_MEMORY_SPACE_WORKGROUP, location, &report.memory_space));
  IREE_RETURN_IF_ERROR(BuildRegisterU32Constant(
      &builder, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
      LOOM_AMDGPU_TSAN_ACCESS_KIND_WRITE, location,
      &report.current_access_kind));
  IREE_RETURN_IF_ERROR(BuildRegisterU32Constant(
      &builder, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
      LOOM_AMDGPU_TSAN_ACCESS_KIND_READ, location, &report.prior_access_kind));
  IREE_RETURN_IF_ERROR(BuildRegisterU32Constant(
      &builder, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
      kTsanAccessLength, location, &report.access_size));
  IREE_RETURN_IF_ERROR(BuildRegisterU64Constant(
      &builder, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
      (uint64_t)kTsanCurrentSiteId, location, &report.current_site_id));
  IREE_RETURN_IF_ERROR(BuildRegisterU64Constant(
      &builder, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
      (uint64_t)kTsanPriorSiteId, location, &report.prior_site_id));
  IREE_RETURN_IF_ERROR(BuildRegisterU64Constant(
      &builder, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
      kTsanMemoryAddress, location, &report.memory_address));
  IREE_RETURN_IF_ERROR(BuildRegisterU64Constant(
      &builder, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
      kTsanShadowAddress, location, &report.shadow_address));
  IREE_RETURN_IF_ERROR(BuildRegisterU64Constant(
      &builder, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, kTsanShadowValue,
      location, &report.shadow_value));
  report.current_workgroup_id_x = source.workgroup_id_x;
  report.current_workgroup_id_y = vgpr_zero;
  report.current_workgroup_id_z = vgpr_zero;
  report.current_workitem_id_x = source.workitem_id_x;
  report.current_workitem_id_y = vgpr_zero;
  report.current_workitem_id_z = vgpr_zero;
  report.prior_workgroup_id_x = source.workgroup_id_x;
  report.prior_workgroup_id_y = vgpr_zero;
  report.prior_workgroup_id_z = vgpr_zero;
  report.prior_workitem_id_x = vgpr_one;
  report.prior_workitem_id_y = vgpr_zero;
  report.prior_workitem_id_z = vgpr_zero;

  return BuildTsanReportPublishAndReturn(&builder, descriptor_set,
                                         feedback_config_symbol, &source,
                                         &report, location);
}

iree_status_t AppendAsanShadowProbeProducer(
    loom_module_t* module, const loom_low_descriptor_set_t* descriptor_set,
    iree_string_view_t kernel_name, uint32_t wavefront_size) {
  loom_op_t* kernel_op = nullptr;
  IREE_RETURN_IF_ERROR(FindKernelOpByName(module, kernel_name, &kernel_op));
  loom_region_t* body = loom_low_kernel_def_body(kernel_op);
  loom_block_t* entry_block = loom_region_entry_block(body);
  if (!loom_low_return_isa(entry_block->last_op)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "expected shadow probe kernel skeleton to end with low.return");
  }

  IREE_RETURN_IF_ERROR(loom_op_erase(module, entry_block->last_op));

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, entry_block, &builder);
  builder.ip.parent_op = kernel_op;
  constexpr loom_location_id_t location = LOOM_LOCATION_UNKNOWN;

  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
      /*unit_count=*/2, &sgpr_x2_type));
  loom_string_id_t kernarg_source_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("amdgpu.kernarg_segment_ptr"), &kernarg_source_id));
  loom_op_t* kernarg_live_in_op = nullptr;
  IREE_RETURN_IF_ERROR(loom_low_live_in_build(
      &builder, kernarg_source_id, loom_make_named_attr_slice(nullptr, 0),
      sgpr_x2_type, location, &kernarg_live_in_op));
  const loom_value_id_t kernarg_segment_ptr =
      loom_low_live_in_result(kernarg_live_in_op);

  loom_value_id_t input_resource = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      BuildSgprLoadDwordX2(&builder, descriptor_set, kernarg_segment_ptr,
                           /*byte_offset=*/0, location, &input_resource));
  loom_value_id_t output_resource = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      BuildSgprLoadDwordX2(&builder, descriptor_set, kernarg_segment_ptr,
                           /*byte_offset=*/8, location, &output_resource));
  IREE_RETURN_IF_ERROR(BuildWaitCounterMask(&builder, descriptor_set,
                                            LOOM_AMDGPU_WAIT_COUNTER_MASK_SMEM,
                                            /*target_count=*/0, location));

  loom_amdgpu_feedback_packet_address_t output_address = {};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_uniform_packet_address(
      &builder, descriptor_set, output_resource, location, &output_address));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_u32_constant(
      &builder, descriptor_set, &output_address, /*byte_offset=*/0,
      /*value=*/0xCAFE0001u, location));

  loom_symbol_ref_t asan_config_symbol = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(GetOrCreateModuleSymbol(
      module, IREE_SV(IREE_HAL_AMDGPU_ASAN_CONFIG_GLOBAL_NAME),
      &asan_config_symbol));
  loom_amdgpu_sanitizer_access_check_t check = {};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_sanitizer_access_check(
      &builder, descriptor_set, asan_config_symbol, input_resource,
      /*access_size=*/sizeof(uint32_t), wavefront_size, location, &check));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b64(
      &builder, descriptor_set, &output_address, /*byte_offset=*/8,
      check.shadow_value, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b64(
      &builder, descriptor_set, &output_address, /*byte_offset=*/16,
      check.failure_mask, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b64(
      &builder, descriptor_set, &output_address, /*byte_offset=*/24,
      check.shadow_address, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_u32_constant(
      &builder, descriptor_set, &output_address, /*byte_offset=*/32,
      /*value=*/0xCAFE0002u, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_release_ordering(
      &builder, descriptor_set, location));

  loom_op_t* return_op = nullptr;
  return loom_low_return_build(&builder, /*values=*/nullptr,
                               /*values_count=*/0, location, &return_op);
}

class AmdgpuHalSanitizerFeedbackTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    recorder_ = new AmdgpuSanitizerDeviceEventRecorder();
    runtime_ = new AmdgpuAsanHalRuntime();
    iree_status_t status = runtime_->Initialize(recorder_->sink());
    runtime_status_code_ = iree_status_code(status);
    if (!iree_status_is_ok(status)) {
      runtime_status_message_ = StatusToStringAndFree(status);
      delete runtime_;
      runtime_ = nullptr;
      delete recorder_;
      recorder_ = nullptr;
    }
  }

  static void TearDownTestSuite() {
    delete runtime_;
    runtime_ = nullptr;
    delete recorder_;
    recorder_ = nullptr;
    runtime_status_code_ = IREE_STATUS_OK;
    runtime_status_message_.clear();
  }

  void SetUp() override {
    if (recorder_ != nullptr) {
      recorder_->Reset();
    }
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(loom_target_environment_initialize(
        &loom_amdgpu_target_provider_set, &target_environment_));
    IREE_ASSERT_OK(InitializeAmdgpuContext(&target_environment_, &context_));
    IREE_ASSERT_OK(loom_target_environment_initialize_low_descriptor_registry(
        &target_environment_, &low_registry_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    loom_target_environment_deinitialize(&target_environment_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_text_parse_options_t ParseOptions() const {
    loom_text_parse_options_t options = {
        /*.diagnostic_sink=*/{},
        /*.max_errors=*/20,
    };
    loom_low_descriptor_text_asm_environment_initialize(
        &low_registry_.registry, &options.low_asm_environment);
    return options;
  }

  iree_status_t ParseModuleSource(iree_string_view_t source,
                                  iree_string_view_t source_name,
                                  ModulePtr* out_module) {
    loom_text_parse_options_t options = ParseOptions();
    loom_module_t* module = nullptr;
    IREE_RETURN_IF_ERROR(loom_text_parse(source, source_name, &context_,
                                         &block_pool_, &options, &module));
    *out_module = ModulePtr(module);
    return iree_ok_status();
  }

  iree_status_t ParseArithmeticModule(iree_string_view_t target_key,
                                      ModulePtr* out_module) {
    std::string source;
    source += "amdgpu.target<";
    source.append(target_key.data, target_key.size);
    source += "> @gfx_target\n";
    source +=
        "low.kernel.def target(@gfx_target) workgroup_size(64, 1, 1) "
        "@loom_kernel() {\n"
        "  %zero = low.const<amdgpu.v_mov_b32> {imm32 = 0} : "
        "reg<amdgpu.vgpr>\n"
        "  %one = low.const<amdgpu.v_mov_b32> {imm32 = 1} : "
        "reg<amdgpu.vgpr>\n"
        "  %sum = low.op<amdgpu.v_add_u32>(%zero, %one) : "
        "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
        "  low.return\n"
        "}\n";

    IREE_RETURN_IF_ERROR(ParseModuleSource(
        iree_make_string_view(source.data(), source.size()),
        IREE_SV("amdgpu_hal_sanitizer_feedback_test.loom"), out_module));
    return iree_ok_status();
  }

  iree_status_t ParseReportModule(const loom_run_hal_device_target_t* target,
                                  ModulePtr* out_module) {
    if (target->target_bundle == nullptr ||
        target->target_bundle->config == nullptr) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "selected AMDGPU target has no bundle config");
    }
    const loom_low_descriptor_set_t* descriptor_set =
        loom_low_descriptor_registry_lookup(
            &low_registry_.registry,
            target->target_bundle->config->contract_set_key);
    if (descriptor_set == nullptr) {
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE, "AMDGPU descriptor set is not linked: %.*s",
          (int)target->target_bundle->config->contract_set_key.size,
          target->target_bundle->config->contract_set_key.data);
    }

    std::string source;
    source += "amdgpu.target<";
    source.append(target->target_key.data, target->target_key.size);
    source += "> @gfx_target\n";
    source +=
        "low.kernel.def target(@gfx_target) workgroup_size(1, 1, 1) "
        "@loom_idle_kernel() {\n"
        "  low.return\n"
        "}\n"
        "low.kernel.def target(@gfx_target) workgroup_size(1, 1, 1) "
        "@loom_report_kernel() {\n"
        "  low.return\n"
        "}\n";

    ModulePtr module;
    IREE_RETURN_IF_ERROR(ParseModuleSource(
        iree_make_string_view(source.data(), source.size()),
        IREE_SV("amdgpu_hal_sanitizer_feedback_test.loom"), &module));
    IREE_RETURN_IF_ERROR(AppendSingleSanitizerSiteTable(
        module.get(), kReportSiteId, LOOM_OP_SANITIZER_ASSERT_ACCESS));
    IREE_RETURN_IF_ERROR(AppendAsanReportProducer(
        module.get(), descriptor_set, IREE_SV("loom_report_kernel")));
    *out_module = std::move(module);
    return iree_ok_status();
  }

  iree_status_t ParseTsanReportModule(
      const loom_run_hal_device_target_t* target, ModulePtr* out_module) {
    if (target->target_bundle == nullptr ||
        target->target_bundle->config == nullptr) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "selected AMDGPU target has no bundle config");
    }
    const loom_low_descriptor_set_t* descriptor_set =
        loom_low_descriptor_registry_lookup(
            &low_registry_.registry,
            target->target_bundle->config->contract_set_key);
    if (descriptor_set == nullptr) {
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE, "AMDGPU descriptor set is not linked: %.*s",
          (int)target->target_bundle->config->contract_set_key.size,
          target->target_bundle->config->contract_set_key.data);
    }

    std::string source;
    source += "amdgpu.target<";
    source.append(target->target_key.data, target->target_key.size);
    source += "> @gfx_target\n";
    source +=
        "low.kernel.def target(@gfx_target) workgroup_size(1, 1, 1) "
        "@loom_tsan_report_kernel() {\n"
        "  low.return\n"
        "}\n";

    ModulePtr module;
    IREE_RETURN_IF_ERROR(ParseModuleSource(
        iree_make_string_view(source.data(), source.size()),
        IREE_SV("amdgpu_hal_sanitizer_feedback_test.loom"), &module));
    IREE_RETURN_IF_ERROR(AppendSingleSanitizerSiteTable(
        module.get(), kTsanCurrentSiteId, LOOM_OP_SANITIZER_RACE_ACCESS));
    IREE_RETURN_IF_ERROR(AppendTsanReportProducer(
        module.get(), descriptor_set, IREE_SV("loom_tsan_report_kernel")));
    *out_module = std::move(module);
    return iree_ok_status();
  }

  iree_status_t ParseShadowProbeModule(
      const loom_run_hal_device_target_t* target, ModulePtr* out_module) {
    if (target->target_bundle == nullptr ||
        target->target_bundle->config == nullptr) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "selected AMDGPU target has no bundle config");
    }
    const loom_low_descriptor_set_t* descriptor_set =
        loom_low_descriptor_registry_lookup(
            &low_registry_.registry,
            target->target_bundle->config->contract_set_key);
    if (descriptor_set == nullptr) {
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE, "AMDGPU descriptor set is not linked: %.*s",
          (int)target->target_bundle->config->contract_set_key.size,
          target->target_bundle->config->contract_set_key.data);
    }

    std::string source;
    source += "amdgpu.target<";
    source.append(target->target_key.data, target->target_key.size);
    source += "> @gfx_target\n";
    source +=
        "low.kernel.def target(@gfx_target) abi_layout({constant_count = 0, "
        "direct_arg_count = 0, direct_arg_names = {}, direct_arg_sizes = [], "
        "resource_count = 2, uses_kernarg_segment_ptr = true}) "
        "workgroup_size(1, 1, 1) "
        "@loom_shadow_probe_kernel() {\n"
        "  low.return\n"
        "}\n";

    ModulePtr module;
    IREE_RETURN_IF_ERROR(ParseModuleSource(
        iree_make_string_view(source.data(), source.size()),
        IREE_SV("amdgpu_hal_sanitizer_feedback_test.loom"), &module));
    const uint32_t wavefront_size =
        target->target_bundle->snapshot
            ? target->target_bundle->snapshot->subgroup_size
            : 0;
    IREE_RETURN_IF_ERROR(AppendAsanShadowProbeProducer(
        module.get(), descriptor_set, IREE_SV("loom_shadow_probe_kernel"),
        wavefront_size));
    *out_module = std::move(module);
    return iree_ok_status();
  }

  static bool ShouldSkipForRuntimeStatus() {
    return runtime_status_code_ == IREE_STATUS_UNAVAILABLE ||
           runtime_status_code_ == IREE_STATUS_NOT_FOUND ||
           runtime_status_code_ == IREE_STATUS_UNIMPLEMENTED;
  }

  const loom_run_hal_runtime_t* runtime() const { return runtime_->runtime(); }

  iree_hal_allocator_t* allocator() const { return runtime_->allocator(); }

  AmdgpuSanitizerDeviceEventRecorder* recorder() const { return recorder_; }

  static AmdgpuSanitizerDeviceEventRecorder* recorder_;
  static AmdgpuAsanHalRuntime* runtime_;
  static iree_status_code_t runtime_status_code_;
  static std::string runtime_status_message_;

  iree_arena_block_pool_t block_pool_;
  loom_target_environment_t target_environment_ = {};
  loom_context_t context_ = {};
  loom_target_low_descriptor_registry_t low_registry_ = {};
};

AmdgpuSanitizerDeviceEventRecorder* AmdgpuHalSanitizerFeedbackTest::recorder_ =
    nullptr;
AmdgpuAsanHalRuntime* AmdgpuHalSanitizerFeedbackTest::runtime_ = nullptr;
iree_status_code_t AmdgpuHalSanitizerFeedbackTest::runtime_status_code_ =
    IREE_STATUS_OK;
std::string AmdgpuHalSanitizerFeedbackTest::runtime_status_message_;

class AmdgpuHalArtifact {
 public:
  AmdgpuHalArtifact() = default;
  ~AmdgpuHalArtifact() { Deinitialize(); }

  AmdgpuHalArtifact(const AmdgpuHalArtifact&) = delete;
  AmdgpuHalArtifact& operator=(const AmdgpuHalArtifact&) = delete;

  loom_run_hal_artifact_t* value() { return &artifact_; }

 private:
  void Deinitialize() {
    loom_amdgpu_hal_artifact_provider.deinitialize_artifact(
        &loom_amdgpu_hal_artifact_provider, &artifact_,
        iree_allocator_system());
    artifact_ = {};
  }

  loom_run_hal_artifact_t artifact_ = {};
};

class AmdgpuHalPreparedCandidate {
 public:
  AmdgpuHalPreparedCandidate() = default;
  ~AmdgpuHalPreparedCandidate() {
    loom_run_hal_prepared_candidate_deinitialize(&candidate_);
  }

  AmdgpuHalPreparedCandidate(const AmdgpuHalPreparedCandidate&) = delete;
  AmdgpuHalPreparedCandidate& operator=(const AmdgpuHalPreparedCandidate&) =
      delete;

  loom_run_hal_prepared_candidate_t* value() { return &candidate_; }

 private:
  loom_run_hal_prepared_candidate_t candidate_ = {};
};

TEST_F(AmdgpuHalSanitizerFeedbackTest,
       PublishesRuntimeGlobalsFromLoomNativeHsaco) {
  if (ShouldSkipForRuntimeStatus()) {
    GTEST_SKIP() << runtime_status_message_;
  }
  ASSERT_EQ(runtime_status_code_, IREE_STATUS_OK) << runtime_status_message_;
  ASSERT_NE(runtime_, nullptr);

  loom_run_hal_device_target_t target = {};
  iree_status_t status = loom_amdgpu_hal_artifact_provider.select_device_target(
      &loom_amdgpu_hal_artifact_provider, runtime(), iree_allocator_system(),
      &target);
  if (iree_status_is_unavailable(status) ||
      iree_status_is_unimplemented(status)) {
    GTEST_SKIP() << StatusToStringAndFree(status);
  }
  IREE_ASSERT_OK(status);

  ModulePtr module;
  IREE_ASSERT_OK(ParseArithmeticModule(target.target_key, &module));

  const loom_target_pipeline_options_t target_pipeline_options = {
      /*.source_to_low_max_errors=*/{},
      /*.source_to_low_legality_diagnostic_flags=*/{},
      /*.control_flow_lowering=*/{},
      /*.sanitizer=*/
      {
          /*.checks=*/LOOM_SANITIZER_CHECK_ACCESS,
      },
  };
  AmdgpuHalArtifact artifact;
  bool emitted = false;
  DiagnosticCapture capture;
  IREE_ASSERT_OK(loom_amdgpu_hal_artifact_provider.emit_artifact(
      &loom_amdgpu_hal_artifact_provider, module.get(), &target,
      /*diagnostic_sink=*/capture.sink(),
      /*source_resolver=*/(loom_source_resolver_t){}, /*max_errors=*/20,
      &target_pipeline_options,
      /*artifact_flags=*/LOOM_RUN_CANDIDATE_ARTIFACT_FLAG_NONE,
      /*artifact_manifest=*/nullptr, /*report=*/nullptr,
      iree_allocator_system(), &emitted, artifact.value()));
  ASSERT_TRUE(emitted) << DiagnosticSummary(capture);
  EXPECT_TRUE(capture.diagnostics.empty()) << DiagnosticSummary(capture);

  AmdgpuHalPreparedCandidate candidate;
  IREE_ASSERT_OK(loom_run_hal_prepared_candidate_prepare(
      runtime(), artifact.value(), candidate.value()));

  iree_hal_amdgpu_asan_config_t asan_config = {};
  IREE_ASSERT_OK(ReadAmdgpuAsanConfig(runtime()->device, allocator(),
                                      candidate.value()->executable,
                                      &asan_config));
  ExpectAmdgpuAsanConfigEnabled(asan_config);

  bool found = false;
  iree_hal_executable_global_t feedback_global =
      iree_hal_executable_global_invalid();
  IREE_ASSERT_OK(iree_hal_executable_try_lookup_global_by_name(
      candidate.value()->executable,
      IREE_SV(IREE_HAL_AMDGPU_FEEDBACK_CONFIG_GLOBAL_NAME), &found,
      &feedback_global));
  ASSERT_TRUE(found);
  iree_hal_executable_global_info_t feedback_info = {};
  IREE_ASSERT_OK(iree_hal_executable_global_info(
      candidate.value()->executable, feedback_global, &feedback_info));
  EXPECT_TRUE(iree_string_view_equal(
      feedback_info.name,
      IREE_SV(IREE_HAL_AMDGPU_FEEDBACK_CONFIG_GLOBAL_NAME)));
  ASSERT_EQ(feedback_info.byte_length,
            sizeof(iree_hal_amdgpu_feedback_config_t));

  iree_hal_buffer_t* feedback_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_executable_global_buffer(
      candidate.value()->executable, feedback_global,
      IREE_HAL_QUEUE_AFFINITY_ANY, &feedback_buffer));
  ASSERT_NE(feedback_buffer, nullptr);
  std::vector<iree_hal_amdgpu_feedback_config_t> feedback_configs;
  IREE_ASSERT_OK(ReadDeviceBufferData(runtime()->device, allocator(),
                                      feedback_buffer, &feedback_configs));
  ASSERT_EQ(feedback_configs.size(), 1u);
  EXPECT_EQ(feedback_configs[0].record_length, sizeof(feedback_configs[0]));
  EXPECT_EQ(feedback_configs[0].abi_version,
            IREE_HAL_AMDGPU_FEEDBACK_CONFIG_ABI_VERSION_0);
  EXPECT_NE(
      feedback_configs[0].flags & IREE_HAL_AMDGPU_FEEDBACK_CONFIG_FLAG_ENABLED,
      0u);
  EXPECT_NE(feedback_configs[0].channel_base, 0u);
  EXPECT_NE(feedback_configs[0].notify_signal.handle, 0u);
  EXPECT_NE(feedback_configs[0].source_context, 0u);
}

TEST_F(AmdgpuHalSanitizerFeedbackTest, ProbesAsanShadowLoadThroughHalShadow) {
  constexpr iree_device_size_t kBufferLength = 64;
  constexpr iree_device_size_t kRedzoneCrossingSubspanOffset =
      kBufferLength - 1;
  constexpr iree_device_size_t kRedzoneCrossingSubspanLength = 1;
  constexpr uint64_t kAssertedAccessLength = sizeof(uint32_t);
  constexpr iree_device_size_t kOutputLength = 40;

  if (ShouldSkipForRuntimeStatus()) {
    GTEST_SKIP() << runtime_status_message_;
  }
  ASSERT_EQ(runtime_status_code_, IREE_STATUS_OK) << runtime_status_message_;
  ASSERT_NE(runtime_, nullptr);

  loom_run_hal_device_target_t target = {};
  iree_status_t status = loom_amdgpu_hal_artifact_provider.select_device_target(
      &loom_amdgpu_hal_artifact_provider, runtime(), iree_allocator_system(),
      &target);
  if (iree_status_is_unavailable(status) ||
      iree_status_is_unimplemented(status)) {
    GTEST_SKIP() << StatusToStringAndFree(status);
  }
  IREE_ASSERT_OK(status);

  ModulePtr module;
  status = ParseShadowProbeModule(&target, &module);
  if (iree_status_is_unavailable(status)) {
    GTEST_SKIP() << StatusToStringAndFree(status);
  }
  IREE_ASSERT_OK(status);

  const loom_target_pipeline_options_t target_pipeline_options = {
      /*.source_to_low_max_errors=*/{},
      /*.source_to_low_legality_diagnostic_flags=*/{},
      /*.control_flow_lowering=*/{},
      /*.sanitizer=*/
      {
          /*.checks=*/LOOM_SANITIZER_CHECK_ACCESS,
      },
  };
  AmdgpuHalArtifact artifact;
  bool emitted = false;
  DiagnosticCapture capture;
  IREE_ASSERT_OK(loom_amdgpu_hal_artifact_provider.emit_artifact(
      &loom_amdgpu_hal_artifact_provider, module.get(), &target,
      /*diagnostic_sink=*/capture.sink(),
      /*source_resolver=*/(loom_source_resolver_t){}, /*max_errors=*/20,
      &target_pipeline_options,
      /*artifact_flags=*/LOOM_RUN_CANDIDATE_ARTIFACT_FLAG_NONE,
      /*artifact_manifest=*/nullptr, /*report=*/nullptr,
      iree_allocator_system(), &emitted, artifact.value()));
  ASSERT_TRUE(emitted) << DiagnosticSummary(capture);
  EXPECT_TRUE(capture.diagnostics.empty()) << DiagnosticSummary(capture);

  AmdgpuHalPreparedCandidate candidate;
  IREE_ASSERT_OK(loom_run_hal_prepared_candidate_prepare(
      runtime(), artifact.value(), candidate.value()));
  ASSERT_EQ(iree_hal_executable_function_count(candidate.value()->executable),
            1u);

  iree_hal_amdgpu_asan_config_t asan_config = {};
  IREE_ASSERT_OK(ReadAmdgpuAsanConfig(runtime()->device, allocator(),
                                      candidate.value()->executable,
                                      &asan_config));
  ExpectAmdgpuAsanConfigEnabled(asan_config);

  iree_hal_buffer_t* input_buffer = nullptr;
  IREE_ASSERT_OK(
      AllocateAsanDeviceBuffer(allocator(), kBufferLength, &input_buffer));
  iree_hal_buffer_t* output_buffer = nullptr;
  IREE_ASSERT_OK(
      AllocateAsanDeviceBuffer(allocator(), kOutputLength, &output_buffer));

  const uint64_t allocation_pointer = AmdgpuDevicePointer(input_buffer);
  const uint64_t first_shadow_address =
      AmdgpuAsanShadowAddress(asan_config, allocation_pointer);
  const uint64_t redzone_fault_pointer =
      allocation_pointer + kRedzoneCrossingSubspanOffset;
  const uint64_t redzone_shadow_address = AmdgpuAsanShadowAddress(
      asan_config, redzone_fault_pointer + kAssertedAccessLength - 1);
  ASSERT_NE(allocation_pointer, 0u);
  ASSERT_NE(first_shadow_address, 0u);
  ASSERT_NE(redzone_shadow_address, 0u);

  IREE_ASSERT_OK(QueueDispatchTwoBuffers(
      runtime()->device, candidate.value()->executable,
      IREE_SV("loom_shadow_probe_kernel"), input_buffer, output_buffer));

  std::vector<uint8_t> output;
  IREE_ASSERT_OK(
      ReadDeviceBufferBytes(runtime()->device, allocator(), output_buffer,
                            /*source_offset=*/0, kOutputLength, &output));
  ASSERT_EQ(output.size(), kOutputLength);
  EXPECT_EQ(LoadLeU32(output.data(), 0), 0xCAFE0001u);
  EXPECT_EQ(LoadLeU64(output.data(), 8), 0u);
  EXPECT_EQ(LoadLeU64(output.data(), 16), 0u);
  EXPECT_EQ(LoadLeU64(output.data(), 24), first_shadow_address);
  EXPECT_EQ(LoadLeU32(output.data(), 32), 0xCAFE0002u);

  // This remains an allocation-shadow ASAN proof: the HAL binding subspan
  // shifts the device pointer to the allocation tail, and the fixed i32 access
  // crosses into the allocation redzone without requiring a binding-length
  // kernarg.
  iree_hal_buffer_t* redzone_crossing_subspan = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(
      input_buffer, kRedzoneCrossingSubspanOffset,
      kRedzoneCrossingSubspanLength, iree_allocator_system(),
      &redzone_crossing_subspan));
  ASSERT_EQ(iree_hal_buffer_byte_offset(redzone_crossing_subspan),
            kRedzoneCrossingSubspanOffset);
  ASSERT_EQ(iree_hal_buffer_byte_length(redzone_crossing_subspan),
            kRedzoneCrossingSubspanLength);
  ASSERT_EQ(AmdgpuDevicePointer(redzone_crossing_subspan),
            redzone_fault_pointer);
  IREE_ASSERT_OK(
      QueueDispatchTwoBuffers(runtime()->device, candidate.value()->executable,
                              IREE_SV("loom_shadow_probe_kernel"),
                              redzone_crossing_subspan, output_buffer));
  output.clear();
  IREE_ASSERT_OK(
      ReadDeviceBufferBytes(runtime()->device, allocator(), output_buffer,
                            /*source_offset=*/0, kOutputLength, &output));
  ASSERT_EQ(output.size(), kOutputLength);
  EXPECT_EQ(LoadLeU32(output.data(), 0), 0xCAFE0001u);
  EXPECT_NE(LoadLeU64(output.data(), 8), 0u);
  EXPECT_NE(LoadLeU64(output.data(), 16), 0u);
  EXPECT_EQ(LoadLeU64(output.data(), 24), redzone_shadow_address);
  EXPECT_EQ(LoadLeU32(output.data(), 32), 0xCAFE0002u);

  iree_hal_buffer_release(redzone_crossing_subspan);
  iree_hal_buffer_release(output_buffer);
  iree_hal_buffer_release(input_buffer);
}

TEST_F(AmdgpuHalSanitizerFeedbackTest,
       PublishesAsanFeedbackPacketFromLoomNativeHsaco) {
  if (ShouldSkipForRuntimeStatus()) {
    GTEST_SKIP() << runtime_status_message_;
  }
  ASSERT_EQ(runtime_status_code_, IREE_STATUS_OK) << runtime_status_message_;
  ASSERT_NE(runtime_, nullptr);

  loom_run_hal_device_target_t target = {};
  iree_status_t status = loom_amdgpu_hal_artifact_provider.select_device_target(
      &loom_amdgpu_hal_artifact_provider, runtime(), iree_allocator_system(),
      &target);
  if (iree_status_is_unavailable(status) ||
      iree_status_is_unimplemented(status)) {
    GTEST_SKIP() << StatusToStringAndFree(status);
  }
  IREE_ASSERT_OK(status);

  ModulePtr module;
  status = ParseReportModule(&target, &module);
  if (iree_status_is_unavailable(status)) {
    GTEST_SKIP() << StatusToStringAndFree(status);
  }
  IREE_ASSERT_OK(status);
  ASSERT_NO_FATAL_FAILURE(ExpectSingleSanitizerSiteTable(
      module.get(), kReportSiteId, LOOM_OP_SANITIZER_ASSERT_ACCESS));

  const loom_target_pipeline_options_t target_pipeline_options = {
      /*.source_to_low_max_errors=*/{},
      /*.source_to_low_legality_diagnostic_flags=*/{},
      /*.control_flow_lowering=*/{},
      /*.sanitizer=*/
      {
          /*.checks=*/LOOM_SANITIZER_CHECK_ACCESS,
      },
  };
  AmdgpuHalArtifact artifact;
  bool emitted = false;
  DiagnosticCapture capture;
  IREE_ASSERT_OK(loom_amdgpu_hal_artifact_provider.emit_artifact(
      &loom_amdgpu_hal_artifact_provider, module.get(), &target,
      /*diagnostic_sink=*/capture.sink(),
      /*source_resolver=*/(loom_source_resolver_t){}, /*max_errors=*/20,
      &target_pipeline_options,
      /*artifact_flags=*/LOOM_RUN_CANDIDATE_ARTIFACT_FLAG_NONE,
      /*artifact_manifest=*/nullptr, /*report=*/nullptr,
      iree_allocator_system(), &emitted, artifact.value()));
  ASSERT_TRUE(emitted) << DiagnosticSummary(capture);
  EXPECT_TRUE(capture.diagnostics.empty()) << DiagnosticSummary(capture);

  AmdgpuHalPreparedCandidate candidate;
  IREE_ASSERT_OK(loom_run_hal_prepared_candidate_prepare(
      runtime(), artifact.value(), candidate.value()));
  ASSERT_EQ(iree_hal_executable_function_count(candidate.value()->executable),
            2u);

  loom_run_hal_binding_list_t bindings = {};
  loom_run_hal_binding_list_initialize(&bindings);
  loom_run_hal_invocation_options_t options;
  loom_run_hal_invocation_options_initialize(&options);
  options.function_name = IREE_SV("loom_report_kernel");
  status = loom_run_hal_dispatch(
      runtime()->device, candidate.value()->executable, &bindings, &options);
  loom_run_hal_binding_list_deinitialize(&bindings);

  const iree_status_code_t dispatch_code = iree_status_code(status);
  if (dispatch_code == IREE_STATUS_ABORTED) {
    iree_status_free(status);
  } else {
    IREE_ASSERT_OK(status);
  }

  IREE_ASSERT_OK(iree_hal_device_queue_flush(runtime()->device,
                                             IREE_HAL_QUEUE_AFFINITY_ANY));
  recorder()->WaitForAsanReportCount(1);
  EXPECT_EQ(recorder()->asan_report_count(), 1u);
  iree_hal_device_asan_report_t report = recorder()->last_report();
  EXPECT_EQ(report.record_length, sizeof(report));
  EXPECT_EQ(report.abi_version, IREE_HAL_DEVICE_ASAN_REPORT_ABI_VERSION_0);
  EXPECT_EQ(report.access_kind, IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE);
  EXPECT_EQ(report.fault_address, kReportFaultAddress);
  EXPECT_EQ(report.access_length, kReportAccessLength);
  EXPECT_EQ(report.site_id, (uint64_t)kReportSiteId);
  EXPECT_EQ(report.shadow_address, kReportShadowAddress);
  EXPECT_EQ(report.shadow_value, kReportShadowValue);
  EXPECT_EQ(report.workgroup_id[0], 0u);
  EXPECT_EQ(report.workitem_id[0], 0u);
  EXPECT_EQ(report.source_dispatch_ptr, 0u);

  iree_hal_device_event_source_t source = recorder()->last_source();
  EXPECT_TRUE(iree_string_view_equal(source.driver_id, IREE_SV("amdgpu")));
  EXPECT_NE(source.executable_id, 0u);
  EXPECT_NE(source.physical_device_ordinal, UINT32_MAX);

  ASSERT_TRUE(recorder()->last_site_available());
  iree_hal_device_event_site_t site = recorder()->last_site();
  EXPECT_EQ(site.site_id, (uint64_t)kReportSiteId);
  EXPECT_TRUE(iree_string_view_equal(
      site.source_file, IREE_SV("amdgpu_hal_sanitizer_feedback_test.loom")));
  EXPECT_EQ(site.start_line, 13u);
  EXPECT_EQ(site.start_column, 3u);
  EXPECT_EQ(site.end_line, 13u);
  EXPECT_EQ(site.end_column, 11u);
  IREE_EXPECT_OK(iree_hal_device_queue_flush(runtime()->device,
                                             IREE_HAL_QUEUE_AFFINITY_ANY));
}

TEST_F(AmdgpuHalSanitizerFeedbackTest,
       PublishesTsanFeedbackPacketFromLoomNativeHsaco) {
  AmdgpuSanitizerDeviceEventRecorder tsan_recorder;
  AmdgpuAsanHalRuntime tsan_runtime;
  iree_status_t status =
      tsan_runtime.Initialize(IREE_SV("tsan"), tsan_recorder.sink());
  const iree_status_code_t tsan_runtime_code = iree_status_code(status);
  if (tsan_runtime_code == IREE_STATUS_UNAVAILABLE ||
      tsan_runtime_code == IREE_STATUS_NOT_FOUND ||
      tsan_runtime_code == IREE_STATUS_UNIMPLEMENTED) {
    GTEST_SKIP() << StatusToStringAndFree(status);
  }
  IREE_ASSERT_OK(status);

  loom_run_hal_device_target_t target = {};
  status = loom_amdgpu_hal_artifact_provider.select_device_target(
      &loom_amdgpu_hal_artifact_provider, tsan_runtime.runtime(),
      iree_allocator_system(), &target);
  if (iree_status_is_unavailable(status) ||
      iree_status_is_unimplemented(status)) {
    GTEST_SKIP() << StatusToStringAndFree(status);
  }
  IREE_ASSERT_OK(status);

  ModulePtr module;
  status = ParseTsanReportModule(&target, &module);
  if (iree_status_is_unavailable(status)) {
    GTEST_SKIP() << StatusToStringAndFree(status);
  }
  IREE_ASSERT_OK(status);
  ASSERT_NO_FATAL_FAILURE(ExpectSingleSanitizerSiteTable(
      module.get(), kTsanCurrentSiteId, LOOM_OP_SANITIZER_RACE_ACCESS));

  const loom_target_pipeline_options_t target_pipeline_options = {
      /*.source_to_low_max_errors=*/{},
      /*.source_to_low_legality_diagnostic_flags=*/{},
      /*.control_flow_lowering=*/{},
      /*.sanitizer=*/
      {
          /*.checks=*/LOOM_SANITIZER_CHECK_RACE,
      },
  };
  AmdgpuHalArtifact artifact;
  bool emitted = false;
  DiagnosticCapture capture;
  IREE_ASSERT_OK(loom_amdgpu_hal_artifact_provider.emit_artifact(
      &loom_amdgpu_hal_artifact_provider, module.get(), &target,
      /*diagnostic_sink=*/capture.sink(),
      /*source_resolver=*/(loom_source_resolver_t){}, /*max_errors=*/20,
      &target_pipeline_options,
      /*artifact_flags=*/LOOM_RUN_CANDIDATE_ARTIFACT_FLAG_NONE,
      /*artifact_manifest=*/nullptr, /*report=*/nullptr,
      iree_allocator_system(), &emitted, artifact.value()));
  ASSERT_TRUE(emitted) << DiagnosticSummary(capture);
  EXPECT_TRUE(capture.diagnostics.empty()) << DiagnosticSummary(capture);

  AmdgpuHalPreparedCandidate candidate;
  IREE_ASSERT_OK(loom_run_hal_prepared_candidate_prepare(
      tsan_runtime.runtime(), artifact.value(), candidate.value()));
  ASSERT_EQ(iree_hal_executable_function_count(candidate.value()->executable),
            1u);

  loom_run_hal_binding_list_t bindings = {};
  loom_run_hal_binding_list_initialize(&bindings);
  loom_run_hal_invocation_options_t options;
  loom_run_hal_invocation_options_initialize(&options);
  options.function_name = IREE_SV("loom_tsan_report_kernel");
  status =
      loom_run_hal_dispatch(tsan_runtime.runtime()->device,
                            candidate.value()->executable, &bindings, &options);
  loom_run_hal_binding_list_deinitialize(&bindings);

  const iree_status_code_t dispatch_code = iree_status_code(status);
  if (dispatch_code == IREE_STATUS_ABORTED) {
    iree_status_free(status);
  } else {
    IREE_ASSERT_OK(status);
  }

  status = iree_hal_device_queue_flush(tsan_runtime.runtime()->device,
                                       IREE_HAL_QUEUE_AFFINITY_ANY);
  const iree_status_code_t flush_code = iree_status_code(status);
  if (flush_code == IREE_STATUS_ABORTED) {
    iree_status_free(status);
  } else {
    IREE_ASSERT_OK(status);
  }
  tsan_recorder.WaitForTsanReportCount(1);
  EXPECT_EQ(tsan_recorder.tsan_report_count(), 1u);
  iree_hal_device_tsan_report_t report = tsan_recorder.last_tsan_report();
  EXPECT_EQ(report.record_length, sizeof(report));
  EXPECT_EQ(report.abi_version, IREE_HAL_DEVICE_TSAN_REPORT_ABI_VERSION_0);
  EXPECT_EQ(report.check_kind, IREE_HAL_DEVICE_TSAN_CHECK_KIND_DATA_RACE);
  EXPECT_EQ(report.memory_space, IREE_HAL_DEVICE_TSAN_MEMORY_SPACE_WORKGROUP);
  EXPECT_EQ(report.current_access_kind, IREE_HAL_DEVICE_TSAN_ACCESS_KIND_WRITE);
  EXPECT_EQ(report.prior_access_kind, IREE_HAL_DEVICE_TSAN_ACCESS_KIND_READ);
  EXPECT_EQ(report.access_length, kTsanAccessLength);
  EXPECT_EQ(report.current_site_id, (uint64_t)kTsanCurrentSiteId);
  EXPECT_EQ(report.prior_site_id, (uint64_t)kTsanPriorSiteId);
  EXPECT_EQ(report.memory_address, kTsanMemoryAddress);
  EXPECT_EQ(report.shadow_address, kTsanShadowAddress);
  EXPECT_EQ(report.shadow_value, kTsanShadowValue);
  EXPECT_EQ(report.current_workgroup_id[0], 0u);
  EXPECT_EQ(report.current_workgroup_id[1], 0u);
  EXPECT_EQ(report.current_workgroup_id[2], 0u);
  EXPECT_EQ(report.current_workitem_id[0], 0u);
  EXPECT_EQ(report.current_workitem_id[1], 0u);
  EXPECT_EQ(report.current_workitem_id[2], 0u);
  EXPECT_EQ(report.prior_workgroup_id[0], 0u);
  EXPECT_EQ(report.prior_workgroup_id[1], 0u);
  EXPECT_EQ(report.prior_workgroup_id[2], 0u);
  EXPECT_EQ(report.prior_workitem_id[0], 1u);
  EXPECT_EQ(report.prior_workitem_id[1], 0u);
  EXPECT_EQ(report.prior_workitem_id[2], 0u);
  EXPECT_NE(report.source_dispatch_ptr, 0u);

  iree_hal_device_event_source_t source = tsan_recorder.last_source();
  EXPECT_TRUE(iree_string_view_equal(source.driver_id, IREE_SV("amdgpu")));
  EXPECT_NE(source.executable_id, 0u);
  EXPECT_NE(source.physical_device_ordinal, UINT32_MAX);

  ASSERT_TRUE(tsan_recorder.last_site_available());
  iree_hal_device_event_site_t site = tsan_recorder.last_site();
  EXPECT_EQ(site.site_id, (uint64_t)kTsanCurrentSiteId);
  EXPECT_TRUE(iree_string_view_equal(
      site.source_file, IREE_SV("amdgpu_hal_sanitizer_feedback_test.loom")));
  EXPECT_EQ(site.start_line, 13u);
  EXPECT_EQ(site.start_column, 3u);
  EXPECT_EQ(site.end_line, 13u);
  EXPECT_EQ(site.end_column, 11u);
  EXPECT_TRUE(iree_string_view_equal(site.operation_name,
                                     IREE_SV("sanitizer.race.access")));
}

}  // namespace
}  // namespace loom
