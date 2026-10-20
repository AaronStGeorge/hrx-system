// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Durable canonical compile requests for selected benchmark case samples.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_COMPILE_REQUESTS_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_COMPILE_REQUESTS_H_

#include "iree/base/api.h"
#include "loom/config/text_binding.h"
#include "loom/sanitizer/options.h"
#include "loom/tools/iree-benchmark-loom/event.h"
#include "loom/tools/iree-benchmark-loom/output.h"
#include "loom/tools/iree-benchmark-loom/work_plan.h"
#include "loomc/compile.h"
#include "loomc/config.h"
#include "loomc/context.h"
#include "loomc/module.h"
#include "loomc/pass.h"
#include "loomc/target.h"
#include "loomc/workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_benchmark_loom_compile_request_export_options_t {
  // Run identity attached to structured failure events.
  const iree_benchmark_loom_run_identity_t* run;
  // Public compiler context used to resolve request root names.
  loomc_context_t* context;
  // Prepared compiler used to compile each canonical request.
  loomc_compiler_t* compiler;
  // Invocation-local workspace shared by sequential capture/compile steps.
  loomc_workspace_t* workspace;
  // Prepared pass program used to compile every request.
  const loomc_pass_program_t* pass_program;
  // Verified source module cloned by each case-sample capture.
  loomc_module_t* module;
  // Exact-version source module referenced by |module_plan| and provenance.
  const loom_module_t* native_module;
  // Testbench plan owning every case and benchmark referenced by |work_plan|.
  const loom_testbench_module_plan_t* module_plan;
  // Selected benchmark windows to capture in authored order.
  const iree_benchmark_loom_work_plan_t* work_plan;
  // Resolved compiler configuration consumed by request production.
  const loomc_config_options_t* config;
  // Ordered normalized text bindings recorded in the durable manifest.
  const loom_config_text_binding_set_t* config_text;
  // Exact target profile applied while producing requests.
  const loomc_target_profile_t* target_profile;
  // Explicit target specification used to select |target_profile|.
  iree_string_view_t target;
  // Compiler pipeline spelling selected for request compilation.
  iree_string_view_t pipeline;
  // Sanitizer policy represented by |pass_program|.
  const loom_sanitizer_options_t* sanitizer;
  // Enabled artifact bundle receiving requests and their manifest.
  iree_benchmark_loom_artifact_bundle_t* artifact_bundle;
  // Structured event sink receiving capture and compile rejections.
  const iree_benchmark_loom_event_sink_t* event_sink;
  // Host allocator for transient capture and manifest storage.
  iree_allocator_t host_allocator;
} iree_benchmark_loom_compile_request_export_options_t;

typedef struct iree_benchmark_loom_compile_request_export_result_t {
  // Number of unique case samples captured.
  iree_host_size_t capture_count;
  // Number of request bytecode files written.
  iree_host_size_t request_count;
  // Number of capture or request compiler rejections emitted as failures.
  iree_host_size_t failure_count;
  // Number of selected benchmarks referencing at least one failed capture.
  iree_host_size_t failed_selection_count;
} iree_benchmark_loom_compile_request_export_result_t;

// Captures, compiles, and writes canonical requests for selected case samples.
//
// Infrastructure failures return a non-OK status. Compiler-domain rejections
// are retained in the request manifest, emitted through |event_sink|, and
// counted in |out_result| while the remaining selected samples continue.
iree_status_t iree_benchmark_loom_export_compile_requests(
    const iree_benchmark_loom_compile_request_export_options_t* options,
    iree_benchmark_loom_compile_request_export_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_COMPILE_REQUESTS_H_
