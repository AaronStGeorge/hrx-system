// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HAL backend interface for iree-run-loom.
//
// The runner owns generic HAL device setup, binding parsing, dispatch, and
// result checking. Target providers own device-target selection and compilation
// of a parsed Loom module into a HAL executable container accepted by that
// target's production HAL loader.

#ifndef LOOM_TOOLS_IREE_RUN_LOOM_HAL_BACKEND_H_
#define LOOM_TOOLS_IREE_RUN_LOOM_HAL_BACKEND_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"
#include "loom/error/diagnostic.h"
#include "loom/ir/module.h"
#include "loom/verify/verify.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_amdgpu_hal_backend_t iree_amdgpu_hal_backend_t;

// Shared HAL runtime state created by the runner for the selected backend.
typedef struct iree_run_loom_hal_runtime_t {
  // VM instance retaining HAL ref-type registrations for function I/O helpers.
  iree_vm_instance_t* instance;
  // Selected HAL device used for executable preparation and dispatch.
  iree_hal_device_t* device;
  // Executable cache owned by |device| and used for target probing/loading.
  iree_hal_executable_cache_t* executable_cache;
} iree_run_loom_hal_runtime_t;

// Target selected from the active HAL device for one compilation.
typedef struct iree_run_loom_hal_selected_target_t {
  // Backend-owned target payload. Usually points at static target info.
  const void* data;
  // Preset key that aliases the selected target, if the backend has one.
  iree_string_view_t preset_key;
} iree_run_loom_hal_selected_target_t;

// Generic executable bytes ready for iree_hal_executable_cache_prepare.
typedef struct iree_run_loom_hal_executable_t {
  // HAL executable format string consumed by the selected HAL loader.
  iree_string_view_t executable_format;
  // Backend-owned executable container bytes.
  iree_const_byte_span_t executable_data;
  // Backend-owned storage released by |deinitialize_executable|.
  void* storage;
} iree_run_loom_hal_executable_t;

typedef iree_status_t (*iree_run_loom_hal_select_target_fn_t)(
    const iree_amdgpu_hal_backend_t* backend,
    const iree_run_loom_hal_runtime_t* runtime, iree_allocator_t allocator,
    iree_run_loom_hal_selected_target_t* out_target);

typedef iree_status_t (*iree_run_loom_hal_format_target_fn_t)(
    const iree_amdgpu_hal_backend_t* backend,
    const iree_run_loom_hal_selected_target_t* target,
    iree_string_builder_t* output);

typedef iree_status_t (*iree_run_loom_hal_compile_fn_t)(
    const iree_amdgpu_hal_backend_t* backend, loom_module_t* module,
    const iree_run_loom_hal_selected_target_t* target,
    iree_string_view_t target_symbol, loom_diagnostic_sink_t diagnostic_sink,
    loom_source_resolver_t source_resolver, uint32_t max_errors,
    iree_allocator_t allocator, iree_run_loom_hal_executable_t* out_executable);

typedef void (*iree_run_loom_hal_deinitialize_executable_fn_t)(
    const iree_amdgpu_hal_backend_t* backend,
    iree_run_loom_hal_executable_t* executable, iree_allocator_t allocator);

struct iree_amdgpu_hal_backend_t {
  // User-facing backend name accepted by --loom_backend.
  iree_string_view_t name;
  // IREE HAL driver name used to create the runtime device.
  iree_string_view_t hal_driver_name;
  // Human-readable target-family name used in status messages.
  iree_string_view_t target_family_name;
  // Selects a concrete target supported by the active HAL executable cache.
  iree_run_loom_hal_select_target_fn_t select_target;
  // Formats the selected target into a HAL executable target string.
  iree_run_loom_hal_format_target_fn_t format_target;
  // Compiles a parsed Loom module to a HAL executable container.
  iree_run_loom_hal_compile_fn_t compile;
  // Releases storage owned by an executable returned from |compile|.
  iree_run_loom_hal_deinitialize_executable_fn_t deinitialize_executable;
};

// A registry of HAL backend providers linked into the runner binary.
typedef struct iree_amdgpu_hal_backend_registry_t {
  // Linked backend provider table.
  const iree_amdgpu_hal_backend_t* const* backends;
  // Number of entries in |backends|.
  iree_host_size_t backend_count;
} iree_amdgpu_hal_backend_registry_t;

// Initializes |out_registry| with the HAL backends linked into this binary.
void iree_amdgpu_hal_backend_registry_initialize(
    iree_amdgpu_hal_backend_registry_t* out_registry);

// Looks up a HAL backend by user-facing --loom_backend name.
const iree_amdgpu_hal_backend_t* iree_amdgpu_hal_backend_registry_lookup(
    const iree_amdgpu_hal_backend_registry_t* registry,
    iree_string_view_t name);

// Appends a comma-separated list of registered HAL backend names.
iree_status_t iree_amdgpu_hal_backend_registry_format_names(
    const iree_amdgpu_hal_backend_registry_t* registry,
    iree_string_builder_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_RUN_LOOM_HAL_BACKEND_H_
