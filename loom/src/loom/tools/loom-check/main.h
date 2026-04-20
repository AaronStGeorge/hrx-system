// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared loom-check command-line entry point.
//
// Tool binaries provide a loom_check_environment_t that selects the dialects,
// target-low descriptor package, and source-to-low lowering policies linked
// into that binary. The shared entry point owns flag parsing, file IO, test
// update handling, JSON output, and result reporting.

#ifndef LOOM_TOOLS_LOOM_CHECK_MAIN_H_
#define LOOM_TOOLS_LOOM_CHECK_MAIN_H_

#include "iree/base/api.h"
#include "loom/tools/loom-check/execute.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initializes a target-low descriptor registry package. Registry tables are
// linked into the runner binary and do not allocate.
typedef void (*loom_check_low_descriptor_registry_initializer_t)(
    loom_target_low_descriptor_registry_t* out_registry);

// Initializes a source-to-target-low lowering policy registry package. Registry
// tables are linked into the runner binary and do not allocate.
typedef void (*loom_check_low_lower_policy_registry_initializer_t)(
    loom_low_lower_policy_registry_t* out_registry);

// Target-owned contribution linked into a loom-check environment.
typedef struct loom_check_provider_t {
  // Stable provider name used in diagnostics and help text.
  iree_string_view_t name;
  // Optional function that initializes a target-low descriptor registry
  // package.
  loom_check_low_descriptor_registry_initializer_t
      initialize_low_descriptor_registry;
  // Optional function that initializes a source-to-low lowering policy package.
  loom_check_low_lower_policy_registry_initializer_t
      initialize_low_lower_policy_registry;
  // Optional emit provider table contributed by this provider.
  const loom_check_emit_provider_t* const* emit_providers;
  // Number of entries in |emit_providers|.
  iree_host_size_t emit_provider_count;
  // Optional in-process RUN: run provider table contributed by this provider.
  const loom_check_run_provider_t* const* run_providers;
  // Number of entries in |run_providers|.
  iree_host_size_t run_provider_count;
  // Optional requirement provider table contributed by this provider.
  const loom_check_requirement_provider_t* const* requirement_providers;
  // Number of entries in |requirement_providers|.
  iree_host_size_t requirement_provider_count;
} loom_check_provider_t;

// Static provider table linked into a loom-check binary or embedding.
typedef struct loom_check_provider_set_t {
  // Provider contribution table.
  const loom_check_provider_t* const* providers;
  // Number of entries in |providers|.
  iree_host_size_t provider_count;
} loom_check_provider_set_t;

// Registers the production Loom dialect surface without the synthetic test
// dialect. This is suitable for backend-owned .loom-test runners.
iree_status_t loom_check_register_production_context(void* user_data,
                                                     loom_context_t* context);

// Runs loom-check using |environment| as the linked tool environment.
int loom_check_main(int argc, char** argv,
                    const loom_check_environment_t* environment);

// Runs loom-check using production dialects plus |provider_set|'s target
// contributions.
int loom_check_provider_main(int argc, char** argv,
                             const loom_check_provider_set_t* provider_set);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_MAIN_H_
