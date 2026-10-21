// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-linked benchmark runner configuration.

#ifndef LOOM_TOOLS_LOOM_BENCHMARK_CONFIGURATION_H_
#define LOOM_TOOLS_LOOM_BENCHMARK_CONFIGURATION_H_

#include "iree/base/api.h"
#include "loom/tooling/input/format.h"
#include "loom/tooling/input/loomc.h"
#include "loom/tooling/testbench/compiled_provider.h"
#include "loom/tooling/testbench/requirements.h"
#include "loomc/target.h"
#include "loomc/target/iree_hal.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_run_hal_testbench_context_t
    loom_run_hal_testbench_context_t;
typedef struct loom_run_hal_target_route_t loom_run_hal_target_route_t;

// Appends target-linked requirement providers to |providers|.
typedef iree_status_t (*loom_benchmark_populate_requirement_providers_fn_t)(
    void* user_data, loom_run_hal_testbench_context_t* hal_context,
    iree_host_size_t provider_capacity,
    loom_testbench_requirement_provider_t* providers,
    iree_host_size_t* inout_provider_count);

typedef struct loom_benchmark_populate_requirement_providers_callback_t {
  // Callback implementation, or NULL when no extra providers are linked.
  loom_benchmark_populate_requirement_providers_fn_t fn;
  // Opaque callback state passed to |fn|.
  void* user_data;
} loom_benchmark_populate_requirement_providers_callback_t;

typedef struct loom_benchmark_configuration_t {
  // Borrowed optional source formats selected by the final application.
  loom_input_format_list_t input_formats;
  // Null-terminated executable name used in help and diagnostics.
  const char* tool_name;
  // Public target environment composed from linked compiler providers.
  loomc_target_environment_t* target_environment;
  // Optional foreign-source importer dispatch.
  loom_tooling_input_import_loomc_fn_t import;
  // Opaque state forwarded to |import|.
  void* import_user_data;
  // HAL driver-to-compiler-target routes linked into the final binary.
  const loom_run_hal_target_route_t* hal_target_routes;
  // Number of entries in |hal_target_routes|.
  iree_host_size_t hal_target_route_count;
  // Binds ordinary function calls once for all cases in the parsed module.
  loom_testbench_function_call_provider_callback_t function_call_provider;
  // Binds the independent oracle for check.compare scenario actions.
  loom_testbench_bind_scenario_profile_callback_t scenario_oracle_profile;
  // Appends target-specific requirement providers linked into this runner.
  loom_benchmark_populate_requirement_providers_callback_t
      populate_requirement_providers;
} loom_benchmark_configuration_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_BENCHMARK_CONFIGURATION_H_
