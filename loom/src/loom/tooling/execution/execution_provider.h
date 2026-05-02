// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Execution provider composition for Loom run, benchmark, and tuning tools.

#ifndef LOOM_TOOLING_EXECUTION_EXECUTION_PROVIDER_H_
#define LOOM_TOOLING_EXECUTION_EXECUTION_PROVIDER_H_

#include "iree/base/api.h"
#include "loom/tooling/execution/execution_backend.h"
#include "loom/tooling/execution/hal_backend.h"
#include "loom/tooling/execution/session.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initializes a target-low descriptor registry package. Registry tables are
// linked into the provider library and do not allocate.
typedef void (*loom_run_low_descriptor_registry_initializer_t)(
    loom_target_low_descriptor_registry_t* out_registry);

// Target-owned execution contribution linked into run/benchmark/tune tools.
typedef struct loom_run_execution_provider_t {
  // Stable provider name used in diagnostics and help text.
  iree_string_view_t name;
  // Optional function that initializes a target-low descriptor registry
  // package.
  loom_run_low_descriptor_registry_initializer_t
      initialize_low_descriptor_registry;
  // Optional HAL backend table contributed by this provider.
  const loom_run_hal_backend_t* const* hal_backends;
  // Number of entries in |hal_backends|.
  iree_host_size_t hal_backend_count;
  // Optional execution backend table contributed by this provider.
  const loom_run_execution_backend_t* const* execution_backends;
  // Number of entries in |execution_backends|.
  iree_host_size_t execution_backend_count;
} loom_run_execution_provider_t;

// Static provider table linked into a run/benchmark/tune binary or embedding.
typedef struct loom_run_execution_provider_set_t {
  // Provider contribution table.
  const loom_run_execution_provider_t* const* providers;
  // Number of entries in |providers|.
  iree_host_size_t provider_count;
} loom_run_execution_provider_set_t;

enum {
  LOOM_RUN_EXECUTION_PROVIDER_DESCRIPTOR_SET_PROVIDER_CAPACITY = 256,
  LOOM_RUN_EXECUTION_PROVIDER_HAL_BACKEND_CAPACITY = 64,
  LOOM_RUN_EXECUTION_PROVIDER_EXECUTION_BACKEND_CAPACITY = 64,
};

// Composed execution environment derived from a provider set.
typedef struct loom_run_execution_environment_t {
  // Provider table selected by the linked binary or embedding.
  const loom_run_execution_provider_set_t* provider_set;
  // Descriptor-set provider scratch table assembled on demand.
  loom_low_descriptor_set_provider_t descriptor_set_providers
      [LOOM_RUN_EXECUTION_PROVIDER_DESCRIPTOR_SET_PROVIDER_CAPACITY];
  // Number of entries in |descriptor_set_providers|.
  iree_host_size_t descriptor_set_provider_count;
  // HAL backend table assembled once for the environment.
  const loom_run_hal_backend_t*
      hal_backends[LOOM_RUN_EXECUTION_PROVIDER_HAL_BACKEND_CAPACITY];
  // Number of entries in |hal_backends|.
  iree_host_size_t hal_backend_count;
  // HAL backend registry view over |hal_backends|.
  loom_run_hal_backend_registry_t hal_backend_registry;
  // Execution backend table assembled once for the environment.
  const loom_run_execution_backend_t* execution_backends
      [LOOM_RUN_EXECUTION_PROVIDER_EXECUTION_BACKEND_CAPACITY];
  // Number of entries in |execution_backends|.
  iree_host_size_t execution_backend_count;
  // Execution backend registry view over |execution_backends|.
  loom_run_execution_backend_registry_t execution_backend_registry;
} loom_run_execution_environment_t;

// Initializes |out_environment| from |provider_set|.
iree_status_t loom_run_execution_environment_initialize(
    const loom_run_execution_provider_set_t* provider_set,
    loom_run_execution_environment_t* out_environment);

// Resets |environment| to an empty state. No provider-owned storage is freed.
void loom_run_execution_environment_deinitialize(
    loom_run_execution_environment_t* environment);

// Returns a session descriptor-registry callback backed by |environment|. Each
// callback invocation resets environment-owned scratch tables, so the returned
// registry view remains valid only until the next callback invocation or
// |environment| deinitialization.
loom_run_initialize_low_descriptor_registry_callback_t
loom_run_execution_environment_low_descriptor_registry_callback(
    loom_run_execution_environment_t* environment);

// Returns the HAL backend registry composed from |environment|'s providers.
const loom_run_hal_backend_registry_t*
loom_run_execution_environment_hal_backend_registry(
    const loom_run_execution_environment_t* environment);

// Returns the execution backend registry composed from |environment|'s
// providers.
const loom_run_execution_backend_registry_t*
loom_run_execution_environment_execution_backend_registry(
    const loom_run_execution_environment_t* environment);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_EXECUTION_PROVIDER_H_
