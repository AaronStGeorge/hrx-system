// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Canonical compiler requests for one concrete check.case sample.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_CASE_REQUEST_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_CASE_REQUEST_H_

#include "iree/base/api.h"
#include "loom/tooling/testbench/testbench.h"
#include "loomc/compile.h"
#include "loomc/config.h"
#include "loomc/module.h"
#include "loomc/product.h"
#include "loomc/result.h"
#include "loomc/target.h"
#include "loomc/workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_benchmark_loom_case_request_options_t {
  // Invocation-local workspace used to clone and plan the source module.
  loomc_workspace_t* workspace;
  // Verified source module containing the authored check.case.
  loomc_module_t* module;
  // Source-order case ordinal established by benchmark planning.
  iree_host_size_t case_index;
  // Concrete selection, borrowing benchmark assignments for this call.
  loom_testbench_case_sample_t case_sample;
  // Resolved invocation configuration consumed by product materialization.
  const loomc_config_options_t* config;
  // Optional exact kernel target profile selected without a HAL device.
  const loomc_target_profile_t* target_profile;
} iree_benchmark_loom_case_request_options_t;

typedef struct iree_benchmark_loom_case_request_capture_t {
  // Host allocator owning arrays and strings retained below.
  iree_allocator_t host_allocator;
  // Command product whose requirement ordinals are referenced by requests.
  loomc_product_t* command_product;
  // Canonical independently compilable requests in publication order.
  loomc_request_t** requests;
  // Number of entries in |requests|.
  iree_host_size_t request_count;
  // Allocated entries in |requests|.
  iree_host_size_t request_capacity;
  // Authored storage-root value IDs in command binding order.
  loom_value_id_t* binding_value_ids;
  // Number of entries in |binding_value_ids|.
  iree_host_size_t binding_count;
  // Source-order case ordinal represented by this capture.
  iree_host_size_t case_index;
  // Generated command root borrowed from |root_symbol_storage|.
  iree_string_view_t root_symbol;
  // Owned NUL-terminated storage backing |root_symbol|.
  char* root_symbol_storage;
} iree_benchmark_loom_case_request_capture_t;

// Publishes the canonical kernel requests for one concrete case sample.
//
// The source module remains unchanged. The operation clones and replans it,
// indexes the clone-local plan with the case ordinal and applies the exact
// parameter selection established by benchmark planning, appends one private
// command root, and delegates provider classification and request
// materialization to the ordinary command product builder. Published requests
// own their bytecode source closure and remain valid after the temporary clone
// is released.
//
// An OK status means product construction ran to |out_result|. The capture is
// populated only when that result succeeded. The caller releases |out_result|
// and deinitializes |out_capture|.
iree_status_t iree_benchmark_loom_case_request_capture(
    const iree_benchmark_loom_case_request_options_t* options,
    iree_allocator_t host_allocator,
    iree_benchmark_loom_case_request_capture_t* out_capture,
    loomc_result_t** out_result);

// Releases all requests, the command product, and retained metadata.
void iree_benchmark_loom_case_request_capture_deinitialize(
    iree_benchmark_loom_case_request_capture_t* capture);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_CASE_REQUEST_H_
