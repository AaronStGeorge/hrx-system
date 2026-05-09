// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HAL backend provider interface for Loom execution tools.
//
// The shared execution layer owns generic HAL device setup, executable loading,
// dispatch, and binding/result handling. Target providers own device-target
// selection and emission of prepared target-low Loom modules into the HAL
// executable package accepted by that target's production loader.

#ifndef LOOM_TOOLING_EXECUTION_HAL_BACKEND_H_
#define LOOM_TOOLING_EXECUTION_HAL_BACKEND_H_

#include "iree/base/api.h"
#include "loom/error/diagnostic.h"
#include "loom/ir/module.h"
#include "loom/target/compile_report.h"
#include "loom/target/types.h"
#include "loom/verify/verify.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_run_hal_backend_t loom_run_hal_backend_t;
typedef struct loom_run_hal_runtime_t loom_run_hal_runtime_t;

// Target selected for one HAL candidate.
typedef struct loom_run_hal_selected_target_t {
  // Backend-owned target payload. Usually points at static target info. NULL
  // requests that the backend emit from the module's target records without a
  // runtime processor override.
  const void* data;
  // Target-neutral bundle resolved for the selected backend target.
  const loom_target_bundle_t* target_bundle;
  // Backend-facing target key selected for emission, if any.
  iree_string_view_t target_key;
} loom_run_hal_selected_target_t;

// Generic executable bytes ready for iree_hal_executable_cache_prepare.
typedef struct loom_run_hal_executable_t {
  // HAL executable format string consumed by the selected HAL loader.
  iree_string_view_t executable_format;
  // Target-neutral bundle resolved for the executable, when available.
  const loom_target_bundle_t* target_bundle;
  // Target-native artifact format before any runtime-loader packaging.
  iree_string_view_t target_artifact_format;
  // Target-native artifact bytes before any runtime-loader packaging.
  iree_const_byte_span_t target_artifact_data;
  // Backend-owned executable container bytes.
  iree_const_byte_span_t executable_data;
  // Backend-owned storage released by |deinitialize_executable|.
  void* storage;
} loom_run_hal_executable_t;

typedef iree_status_t (*loom_run_hal_select_target_fn_t)(
    const loom_run_hal_backend_t* backend,
    const loom_run_hal_runtime_t* runtime, iree_allocator_t allocator,
    loom_run_hal_selected_target_t* out_target);

typedef iree_status_t (*loom_run_hal_format_target_fn_t)(
    const loom_run_hal_backend_t* backend,
    const loom_run_hal_selected_target_t* target,
    iree_string_builder_t* output);

typedef iree_status_t (*loom_run_hal_emit_executable_fn_t)(
    const loom_run_hal_backend_t* backend, loom_module_t* module,
    const loom_run_hal_selected_target_t* target,
    iree_string_view_t entry_symbol, loom_diagnostic_sink_t diagnostic_sink,
    loom_source_resolver_t source_resolver, uint32_t max_errors,
    loom_target_compile_report_t* report, iree_allocator_t allocator,
    bool* out_emitted, loom_run_hal_executable_t* out_executable);

typedef void (*loom_run_hal_deinitialize_executable_fn_t)(
    const loom_run_hal_backend_t* backend,
    loom_run_hal_executable_t* executable, iree_allocator_t allocator);

struct loom_run_hal_backend_t {
  // User-facing backend name accepted by execution tools.
  iree_string_view_t name;
  // IREE HAL driver name used to create the runtime device.
  iree_string_view_t hal_driver_name;
  // Human-readable target-family name used in status messages.
  iree_string_view_t target_family_name;
  // Selects a concrete target supported by the active HAL executable cache.
  loom_run_hal_select_target_fn_t select_target;
  // Formats the selected target into a HAL executable target string.
  loom_run_hal_format_target_fn_t format_target;
  // Emits a prepared target-low Loom module to a HAL executable package.
  loom_run_hal_emit_executable_fn_t emit_executable;
  // Releases storage owned by an executable returned from |emit_executable|.
  loom_run_hal_deinitialize_executable_fn_t deinitialize_executable;
};

// A registry of HAL backend providers linked into a runner binary.
typedef struct loom_run_hal_backend_registry_t {
  // Linked backend provider table.
  const loom_run_hal_backend_t* const* backends;
  // Number of entries in |backends|.
  iree_host_size_t backend_count;
} loom_run_hal_backend_registry_t;

// Initializes |out_registry| from a caller-owned backend table.
void loom_run_hal_backend_registry_initialize_from_entries(
    const loom_run_hal_backend_t* const* backends,
    iree_host_size_t backend_count,
    loom_run_hal_backend_registry_t* out_registry);

// Looks up a HAL backend by user-facing backend name.
const loom_run_hal_backend_t* loom_run_hal_backend_registry_lookup(
    const loom_run_hal_backend_registry_t* registry, iree_string_view_t name);

// Appends a comma-separated list of registered HAL backend names.
iree_status_t loom_run_hal_backend_registry_format_names(
    const loom_run_hal_backend_registry_t* registry,
    iree_string_builder_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_BACKEND_H_
