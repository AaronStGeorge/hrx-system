// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Typed benchmark lifecycle events and sink adapters.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_EVENT_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_EVENT_H_

#include "iree/base/api.h"
#include "loom/ir/module.h"
#include "loom/tools/iree-benchmark-loom/model.h"
#include "loom/tools/iree-benchmark-loom/options.h"
#include "loom/tools/iree-benchmark-loom/output.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum iree_benchmark_loom_event_kind_e {
  // Invalid or uninitialized event.
  IREE_BENCHMARK_LOOM_EVENT_NONE = 0,
  // Run metadata emitted once after output setup.
  IREE_BENCHMARK_LOOM_EVENT_RUN = 1,
  // One selected benchmark plan row.
  IREE_BENCHMARK_LOOM_EVENT_PLAN = 2,
  // Final aggregate run summary.
  IREE_BENCHMARK_LOOM_EVENT_SUMMARY = 3,
} iree_benchmark_loom_event_kind_t;

typedef struct iree_benchmark_loom_run_event_t {
  // Run identity shared by all emitted events.
  const iree_benchmark_loom_run_identity_t* run;
  // True when the run stops after planning selected benchmarks.
  bool dry_run;
  // Requested dispatch sample-compilation mode for this run.
  iree_benchmark_loom_sample_compilation_mode_t sample_compilation_mode;
} iree_benchmark_loom_run_event_t;

typedef struct iree_benchmark_loom_plan_event_t {
  // Run identity shared by all emitted events.
  const iree_benchmark_loom_run_identity_t* run;
  // Parsed module that owns case and benchmark plan references.
  const loom_module_t* module;
  // Selected benchmark record being reported.
  const iree_benchmark_loom_selected_benchmark_t* selection;
  // Effective benchmark runner options for plan-policy reporting.
  const iree_benchmark_loom_options_t* options;
  // Requested dispatch sample-compilation mode for this run.
  iree_benchmark_loom_sample_compilation_mode_t sample_compilation_mode;
} iree_benchmark_loom_plan_event_t;

typedef struct iree_benchmark_loom_summary_event_t {
  // Run identity shared by all emitted events.
  const iree_benchmark_loom_run_identity_t* run;
  // Optional artifact bundle observed during execution.
  const iree_benchmark_loom_artifact_bundle_t* artifact_bundle;
  // Number of planned check.case records.
  iree_host_size_t planned_case_count;
  // Number of planned check.benchmark records.
  iree_host_size_t planned_benchmark_count;
  // Number of selected benchmark candidates.
  iree_host_size_t selected_benchmark_count;
  // Number of top-level failure rows emitted.
  iree_host_size_t failure_count;
  // Number of benchmark rows that failed or did not execute successfully.
  iree_host_size_t failed_benchmark_count;
  // Number of correctness samples executed.
  iree_host_size_t correctness_sample_count;
  // Number of correctness samples that failed expectations.
  iree_host_size_t correctness_failed_sample_count;
  // True when the run stopped after planning selected benchmarks.
  bool dry_run;
  // Requested dispatch sample-compilation mode for this run.
  iree_benchmark_loom_sample_compilation_mode_t sample_compilation_mode;
} iree_benchmark_loom_summary_event_t;

typedef struct iree_benchmark_loom_event_t {
  // Event kind and union discriminator.
  iree_benchmark_loom_event_kind_t kind;
  union {
    // Payload for IREE_BENCHMARK_LOOM_EVENT_RUN.
    iree_benchmark_loom_run_event_t run;
    // Payload for IREE_BENCHMARK_LOOM_EVENT_PLAN.
    iree_benchmark_loom_plan_event_t plan;
    // Payload for IREE_BENCHMARK_LOOM_EVENT_SUMMARY.
    iree_benchmark_loom_summary_event_t summary;
  };
} iree_benchmark_loom_event_t;

typedef iree_status_t (*iree_benchmark_loom_event_sink_emit_fn_t)(
    void* user_data, const iree_benchmark_loom_event_t* event);

typedef struct iree_benchmark_loom_event_sink_t {
  // Callback invoked once per event.
  iree_benchmark_loom_event_sink_emit_fn_t emit;
  // Opaque sink state passed to |emit|.
  void* user_data;
} iree_benchmark_loom_event_sink_t;

typedef struct iree_benchmark_loom_jsonl_event_sink_t {
  // Borrowed JSONL row sink receiving rendered events.
  iree_benchmark_loom_jsonl_sink_t* jsonl_sink;
} iree_benchmark_loom_jsonl_event_sink_t;

// Emits |event| to |sink|.
iree_status_t iree_benchmark_loom_event_sink_emit(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_event_t* event);

// Emits the run metadata event.
iree_status_t iree_benchmark_loom_event_sink_emit_run(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_run_identity_t* run, bool dry_run,
    iree_benchmark_loom_sample_compilation_mode_t sample_compilation_mode);

// Emits one selected benchmark plan event.
iree_status_t iree_benchmark_loom_event_sink_emit_plan(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_run_identity_t* run, const loom_module_t* module,
    const iree_benchmark_loom_selected_benchmark_t* selection,
    const iree_benchmark_loom_options_t* options,
    iree_benchmark_loom_sample_compilation_mode_t sample_compilation_mode);

// Emits the terminal aggregate summary event.
iree_status_t iree_benchmark_loom_event_sink_emit_summary(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_artifact_bundle_t* artifact_bundle,
    iree_host_size_t planned_case_count,
    iree_host_size_t planned_benchmark_count,
    iree_host_size_t selected_benchmark_count, iree_host_size_t failure_count,
    iree_host_size_t failed_benchmark_count,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count, bool dry_run,
    iree_benchmark_loom_sample_compilation_mode_t sample_compilation_mode);

// Initializes a JSONL event adapter over |jsonl_sink|.
void iree_benchmark_loom_jsonl_event_sink_initialize(
    iree_benchmark_loom_jsonl_sink_t* jsonl_sink,
    iree_benchmark_loom_jsonl_event_sink_t* out_adapter,
    iree_benchmark_loom_event_sink_t* out_sink);

// Releases borrowed references held by |adapter|.
void iree_benchmark_loom_jsonl_event_sink_deinitialize(
    iree_benchmark_loom_jsonl_event_sink_t* adapter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_EVENT_H_
