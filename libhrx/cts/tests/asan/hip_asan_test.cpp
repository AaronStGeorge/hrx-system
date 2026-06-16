// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <cinttypes>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>

#include "hrx_test_fixture.hpp"
#include "libhrx/cts/amdgpu_asan_hip_module_test_kernels.h"

namespace {

typedef struct hipModule_st* hipModule_t;
typedef struct hipFunction_st* hipFunction_t;
typedef struct hipStream_st* hipStream_t;
typedef void* hipDeviceptr_t;

enum hipError_t {
  hipSuccess = 0,
  hipErrorNoDevice = 100,
  hipErrorInvalidImage = 200,
  hipErrorNoBinaryForGpu = 209,
  hipErrorNotFound = 500,
  hipErrorNotReady = 600,
};

enum hipMemcpyKind {
  hipMemcpyHostToDevice = 1,
  hipMemcpyDeviceToHost = 2,
};

#define HIP_LAUNCH_PARAM_BUFFER_POINTER ((void*)0x01)
#define HIP_LAUNCH_PARAM_BUFFER_SIZE ((void*)0x02)
#define HIP_LAUNCH_PARAM_END ((void*)0x03)

using HipGetErrorNameFn = const char* (*)(hipError_t);
using HipGetErrorStringFn = const char* (*)(hipError_t);
using HipInitFn = hipError_t (*)(unsigned int);
using HipSetDeviceFn = hipError_t (*)(int);
using HipGetDeviceCountFn = hipError_t (*)(int*);
using HipMallocFn = hipError_t (*)(void**, size_t);
using HipMemcpyFn = hipError_t (*)(void*, const void*, size_t, hipMemcpyKind);
using HipFreeFn = hipError_t (*)(void*);
using HipDeviceSynchronizeFn = hipError_t (*)(void);
using HipModuleLoadDataFn = hipError_t (*)(hipModule_t*, const void*);
using HipModuleUnloadFn = hipError_t (*)(hipModule_t);
using HipModuleGetGlobalFn = hipError_t (*)(hipDeviceptr_t*, size_t*,
                                            hipModule_t, const char*);
using HipModuleGetFunctionFn = hipError_t (*)(hipFunction_t*, hipModule_t,
                                              const char*);
using HipModuleLaunchKernelFn = hipError_t (*)(hipFunction_t, unsigned int,
                                               unsigned int, unsigned int,
                                               unsigned int, unsigned int,
                                               unsigned int, unsigned int,
                                               hipStream_t, void**, void**);
using HrxRuntimeSetDeviceEventSinkFn =
    hrx_status_t (*)(hrx_device_event_sink_t);
using HrxStatusToStringFn = hrx_status_t (*)(hrx_status_t, char**, size_t*);
using HrxStatusFreeMessageFn = void (*)(char*);
using HrxStatusIgnoreFn = void (*)(hrx_status_t);

const char* default_hip_library_name() {
#if defined(_WIN32)
  return "amdhip64.dll";
#else
  return "libamdhip64.so";
#endif
}

class HipApi {
 public:
  void load(const std::string& path) {
    library_.load(path.empty() ? default_hip_library_name() : path);

#define LOAD_SYMBOL(field, symbol) \
  field = reinterpret_cast<decltype(field)>(library_.loadSymbol(#symbol))

    LOAD_SYMBOL(get_error_name, hipGetErrorName);
    LOAD_SYMBOL(get_error_string, hipGetErrorString);
    LOAD_SYMBOL(init, hipInit);
    LOAD_SYMBOL(set_device, hipSetDevice);
    LOAD_SYMBOL(get_device_count, hipGetDeviceCount);
    LOAD_SYMBOL(malloc, hipMalloc);
    LOAD_SYMBOL(memcpy, hipMemcpy);
    LOAD_SYMBOL(free, hipFree);
    LOAD_SYMBOL(device_synchronize, hipDeviceSynchronize);
    LOAD_SYMBOL(module_load_data, hipModuleLoadData);
    LOAD_SYMBOL(module_unload, hipModuleUnload);
    LOAD_SYMBOL(module_get_global, hipModuleGetGlobal);
    LOAD_SYMBOL(module_get_function, hipModuleGetFunction);
    LOAD_SYMBOL(module_launch_kernel, hipModuleLaunchKernel);
    LOAD_SYMBOL(set_device_event_sink, hrx_runtime_set_device_event_sink);
    LOAD_SYMBOL(status_to_string, hrx_status_to_string);
    LOAD_SYMBOL(status_free_message, hrx_status_free_message);
    LOAD_SYMBOL(status_ignore, hrx_status_ignore);

#undef LOAD_SYMBOL
  }

  HipGetErrorNameFn get_error_name = nullptr;
  HipGetErrorStringFn get_error_string = nullptr;
  HipInitFn init = nullptr;
  HipSetDeviceFn set_device = nullptr;
  HipGetDeviceCountFn get_device_count = nullptr;
  HipMallocFn malloc = nullptr;
  HipMemcpyFn memcpy = nullptr;
  HipFreeFn free = nullptr;
  HipDeviceSynchronizeFn device_synchronize = nullptr;
  HipModuleLoadDataFn module_load_data = nullptr;
  HipModuleUnloadFn module_unload = nullptr;
  HipModuleGetGlobalFn module_get_global = nullptr;
  HipModuleGetFunctionFn module_get_function = nullptr;
  HipModuleLaunchKernelFn module_launch_kernel = nullptr;
  HrxRuntimeSetDeviceEventSinkFn set_device_event_sink = nullptr;
  HrxStatusToStringFn status_to_string = nullptr;
  HrxStatusFreeMessageFn status_free_message = nullptr;
  HrxStatusIgnoreFn status_ignore = nullptr;

 private:
  HrxDynamicLibrary library_;
};

std::string hip_error_message(const HipApi& hip, hipError_t result) {
  std::string message = std::to_string(static_cast<int>(result));
  const char* error_name = hip.get_error_name(result);
  if (error_name) {
    message += " (";
    message += error_name;
    message += ")";
  }
  const char* error_string = hip.get_error_string(result);
  if (error_string) {
    message += ": ";
    message += error_string;
  }
  return message;
}

void require_hip(const HipApi& hip, const char* expression, hipError_t result) {
  if (result == hipSuccess) return;
  FAIL(std::string(expression) + " failed: " + hip_error_message(hip, result));
}

#define REQUIRE_HIP(hip, expr) require_hip((hip), #expr, (expr))

void require_hrx(const HipApi& hip, const char* expression,
                 hrx_status_t status) {
  if (hrx_status_is_ok(status)) return;
  char* message = nullptr;
  size_t message_length = 0;
  hrx_status_t string_status =
      hip.status_to_string(status, &message, &message_length);
  std::string failure_message = std::string(expression) + " failed";
  if (hrx_status_is_ok(string_status)) {
    failure_message += ": ";
    failure_message += std::string(message ? message : "", message_length);
    hip.status_free_message(message);
  } else {
    hip.status_ignore(string_status);
  }
  hip.status_ignore(status);
  FAIL(failure_message);
}

#define REQUIRE_HRX(hip, expr) require_hrx((hip), #expr, (expr))

bool hip_result_is_unavailable(hipError_t result) {
  return result == hipErrorNotFound || result == hipErrorNotReady ||
         result == hipErrorNoDevice;
}

class EventRecorder {
 public:
  hrx_device_event_sink_t sink() {
    hrx_device_event_sink_t sink;
    sink.fn = EventRecorder::capture;
    sink.user_data = this;
    return sink;
  }

  void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    asan_report_count_ = 0;
    last_source_ = {};
    last_report_ = {};
  }

  void wait_for_asan_report_count(size_t expected_count) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return asan_report_count_ >= expected_count; });
  }

  size_t asan_report_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return asan_report_count_;
  }

  hrx_device_event_source_t last_source() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_source_;
  }

  hrx_device_asan_report_t last_report() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_report_;
  }

 private:
  static void capture(void* user_data, const hrx_device_event_t* event) {
    auto* recorder = static_cast<EventRecorder*>(user_data);
    if (event->type != HRX_DEVICE_EVENT_TYPE_ASAN_REPORT ||
        !event->payload.data ||
        event->payload.data_length < sizeof(hrx_device_asan_report_t)) {
      return;
    }
    std::lock_guard<std::mutex> lock(recorder->mutex_);
    recorder->last_source_ = event->source;
    std::memcpy(&recorder->last_report_, event->payload.data,
                sizeof(recorder->last_report_));
    ++recorder->asan_report_count_;
    recorder->condition_.notify_all();
  }

  mutable std::mutex mutex_;
  std::condition_variable condition_;
  size_t asan_report_count_ = 0;
  hrx_device_event_source_t last_source_ = {};
  hrx_device_asan_report_t last_report_ = {};
};

enum class LaunchMode {
  kArgsArray,
  kPrePacked,
};

bool hip_result_is_unsupported_image(hipError_t result) {
  return result == hipErrorInvalidImage || result == hipErrorNoBinaryForGpu;
}

hipModule_t load_supported_module(const HipApi& hip) {
  const iree_file_toc_t* toc = hrx_cts_amdgpu_asan_test_kernels_create();
  for (size_t i = 0; i < hrx_cts_amdgpu_asan_test_kernels_size(); ++i) {
    hipModule_t module = nullptr;
    hipError_t result = hip.module_load_data(&module, toc[i].data);
    if (result == hipSuccess) {
      return module;
    }
    if (!hip_result_is_unsupported_image(result)) {
      require_hip(hip, "hipModuleLoadData", result);
    }
  }
  return nullptr;
}

void launch_kernel(const HipApi& hip, hipFunction_t function, void* app_buffer,
                   void* output_buffer, LaunchMode mode) {
  if (mode == LaunchMode::kPrePacked) {
    uint64_t packed_args[] = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(app_buffer)),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(output_buffer)),
    };
    size_t packed_args_size = sizeof(packed_args);
    void* extra[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        packed_args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &packed_args_size,
        HIP_LAUNCH_PARAM_END,
    };
    REQUIRE_HIP(hip, hip.module_launch_kernel(function, 1, 1, 1, 1, 1, 1, 0,
                                              nullptr, nullptr, extra));
  } else {
    void* kernel_args[] = {&app_buffer, &output_buffer};
    REQUIRE_HIP(hip, hip.module_launch_kernel(function, 1, 1, 1, 1, 1, 1, 0,
                                              nullptr, kernel_args, nullptr));
  }
  REQUIRE_HIP(hip, hip.device_synchronize());
}

void expect_asan_report(EventRecorder& recorder, uint64_t fault_address) {
  recorder.wait_for_asan_report_count(1);
  REQUIRE(recorder.asan_report_count() == 1);

  hrx_device_asan_report_t report = recorder.last_report();
  REQUIRE(report.record_length == sizeof(report));
  REQUIRE(report.abi_version == HRX_DEVICE_ASAN_REPORT_ABI_VERSION_0);
  REQUIRE(report.access_kind == HRX_DEVICE_ASAN_ACCESS_KIND_READ);
  REQUIRE(report.access_length == 1);
  REQUIRE(report.fault_address == fault_address);
  REQUIRE(report.shadow_address != 0);
  REQUIRE(report.shadow_value != 0);

  hrx_device_event_source_t source = recorder.last_source();
  REQUIRE(source.driver_id.size == 6);
  REQUIRE(std::memcmp(source.driver_id.data, "amdgpu", 6) == 0);
}

void run_in_bounds(const HipApi& hip, hipFunction_t function, void* app_buffer,
                   void* output_buffer, LaunchMode mode,
                   EventRecorder& recorder) {
  recorder.reset();
  uint32_t output = 0;
  REQUIRE_HIP(hip, hip.memcpy(output_buffer, &output, sizeof(output),
                              hipMemcpyHostToDevice));
  launch_kernel(hip, function, app_buffer, output_buffer, mode);
  REQUIRE(recorder.asan_report_count() == 0);
  REQUIRE_HIP(hip, hip.memcpy(&output, output_buffer, sizeof(output),
                              hipMemcpyDeviceToHost));
  REQUIRE(output == 31);
}

void run_reporting_kernel(const HipApi& hip, hipFunction_t function,
                          void* app_buffer, void* output_buffer,
                          uint64_t fault_address, LaunchMode mode,
                          EventRecorder& recorder) {
  recorder.reset();
  launch_kernel(hip, function, app_buffer, output_buffer, mode);
  expect_asan_report(recorder, fault_address);
}

}  // namespace

TEST_CASE("HIP module ASAN reports route through HRX") {
  HipApi hip;
  hip.load(g_test_hip_library_path);

  EventRecorder recorder;
  REQUIRE_HRX(hip, hip.set_device_event_sink(recorder.sink()));

  hipError_t init_result = hip.init(0);
  if (init_result != hipSuccess) {
    if (hip_result_is_unavailable(init_result)) {
      SUCCEED("Skipping HRX HIP ASAN CTS: hipInit unavailable");
      return;
    }
    require_hip(hip, "hipInit(0)", init_result);
  }
  REQUIRE_HIP(hip, hip.set_device(0));

  int device_count = 0;
  hipError_t count_result = hip.get_device_count(&device_count);
  if (count_result != hipSuccess) {
    if (hip_result_is_unavailable(count_result)) {
      SUCCEED("Skipping HRX HIP ASAN CTS: hipGetDeviceCount unavailable");
      return;
    }
    require_hip(hip, "hipGetDeviceCount(&device_count)", count_result);
  }
  if (device_count <= 0) {
    SUCCEED("Skipping HRX HIP ASAN CTS: no HIP devices");
    return;
  }

  hipModule_t module = load_supported_module(hip);
  if (!module) {
    SUCCEED("Skipping HRX HIP ASAN CTS: no compatible HSACO asset");
    return;
  }

  uint8_t* app_buffer = nullptr;
  uint32_t* output_buffer = nullptr;
  REQUIRE_HIP(hip, hip.malloc(reinterpret_cast<void**>(&app_buffer), 64));
  REQUIRE_HIP(hip, hip.malloc(reinterpret_cast<void**>(&output_buffer),
                              sizeof(*output_buffer)));

  uint8_t app_data[64] = {};
  app_data[0] = 30;
  REQUIRE_HIP(hip, hip.memcpy(app_buffer, app_data, sizeof(app_data),
                              hipMemcpyHostToDevice));

  hipDeviceptr_t asan_config_ptr = nullptr;
  size_t asan_config_size = 0;
  REQUIRE_HIP(hip, hip.module_get_global(&asan_config_ptr, &asan_config_size,
                                         module, "iree_asan_config"));
  REQUIRE(asan_config_ptr != nullptr);
  REQUIRE(asan_config_size != 0);

  hipFunction_t plain_in_bounds = nullptr;
  hipFunction_t in_bounds = nullptr;
  hipFunction_t left_redzone = nullptr;
  hipFunction_t check_only = nullptr;
  REQUIRE_HIP(hip, hip.module_get_function(&plain_in_bounds, module,
                                           "hrx_asan_plain_in_bounds"));
  REQUIRE_HIP(
      hip, hip.module_get_function(&in_bounds, module, "hrx_asan_in_bounds"));
  REQUIRE_HIP(hip, hip.module_get_function(&left_redzone, module,
                                           "hrx_asan_left_redzone"));
  REQUIRE_HIP(
      hip, hip.module_get_function(&check_only, module, "hrx_asan_check_only"));

  run_in_bounds(hip, plain_in_bounds, app_buffer, output_buffer,
                LaunchMode::kArgsArray, recorder);
  run_in_bounds(hip, in_bounds, app_buffer, output_buffer,
                LaunchMode::kArgsArray, recorder);
  run_reporting_kernel(
      hip, left_redzone, app_buffer, output_buffer,
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(app_buffer)) - 1u,
      LaunchMode::kArgsArray, recorder);
  run_in_bounds(hip, in_bounds, app_buffer, output_buffer,
                LaunchMode::kPrePacked, recorder);
  run_reporting_kernel(
      hip, left_redzone, app_buffer, output_buffer,
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(app_buffer)) - 1u,
      LaunchMode::kPrePacked, recorder);

  uint8_t* stale_buffer = app_buffer;
  REQUIRE_HIP(hip, hip.free(app_buffer));
  app_buffer = nullptr;
  run_reporting_kernel(
      hip, check_only, stale_buffer, output_buffer,
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(stale_buffer)),
      LaunchMode::kArgsArray, recorder);

  REQUIRE_HIP(hip, hip.module_unload(module));
  REQUIRE_HIP(hip, hip.free(output_buffer));
}
