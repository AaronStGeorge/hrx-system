// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Typed benchmark lifecycle events and sink adapters.
//
// Event payloads are borrowed from the benchmark run and are only valid for the
// duration of the sink callback receiving them. Streaming sinks should render
// rows before returning, and aggregating sinks must copy any strings or derived
// JSON they need after the run advances.

#ifndef LOOM_TOOLS_LOOM_BENCHMARK_EVENT_H_
#define LOOM_TOOLS_LOOM_BENCHMARK_EVENT_H_

#include "iree/base/api.h"
#include "loom/ir/module.h"
#include "loom/tooling/testbench/scenario/executor.h"
#include "loom/tooling/testbench/testbench.h"
#include "loom/tools/loom-benchmark/model.h"
#include "loom/tools/loom-benchmark/options.h"
#include "loom/tools/loom-benchmark/output.h"
#include "loom/tools/loom-benchmark/work_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_benchmark_event_kind_e {
  // Invalid or uninitialized event.
  LOOM_BENCHMARK_EVENT_NONE = 0,
  // Run metadata emitted once after output setup.
  LOOM_BENCHMARK_EVENT_RUN = 1,
  // One selected benchmark plan row.
  LOOM_BENCHMARK_EVENT_PLAN = 2,
  // Final aggregate run summary.
  LOOM_BENCHMARK_EVENT_SUMMARY = 3,
  // Selected HAL device metadata.
  LOOM_BENCHMARK_EVENT_DEVICE = 4,
  // One canonical compiled entry's metadata.
  LOOM_BENCHMARK_EVENT_COMPILE = 5,
  // One correctness sample result.
  LOOM_BENCHMARK_EVENT_SAMPLE = 6,
  // One benchmark measurement result.
  LOOM_BENCHMARK_EVENT_BENCHMARK_RESULT = 7,
  // Optional profile-replay rows from a distinct post-measurement execution.
  LOOM_BENCHMARK_EVENT_PROFILE_REPLAY = 8,
  // Parse, verify, planning, or infrastructure failure metadata.
  LOOM_BENCHMARK_EVENT_FAILURE = 9,
  // One interleaved comparison benchmark repetition.
  LOOM_BENCHMARK_EVENT_BENCHMARK_REPETITION = 10,
  // Aggregate interleaved comparison result.
  LOOM_BENCHMARK_EVENT_COMPARISON = 11,
  // Deduplicated planned work graph for dry-run/reporting sinks.
  LOOM_BENCHMARK_EVENT_WORK_PLAN = 12,
  // One untimed scenario trial's correctness observations.
  LOOM_BENCHMARK_EVENT_SCENARIO_SAMPLE = 13,
} loom_benchmark_event_kind_t;

typedef struct loom_benchmark_run_event_t {
  // Run identity shared by all emitted events.
  const loom_benchmark_run_identity_t* run;
  // True when the run stops after planning selected benchmarks.
  bool dry_run;
  // Sanitizer checks and reporting mode used for compiler-backed work.
  loom_sanitizer_options_t sanitizer;
} loom_benchmark_run_event_t;

typedef struct loom_benchmark_plan_event_t {
  // Run identity shared by all emitted events.
  const loom_benchmark_run_identity_t* run;
  // Parsed module that owns case and benchmark plan references.
  const loom_module_t* module;
  // Selected benchmark record being reported.
  const loom_benchmark_selected_benchmark_t* selection;
  // Effective benchmark runner options for plan-policy reporting.
  const loom_benchmark_options_t* options;
} loom_benchmark_plan_event_t;

typedef struct loom_benchmark_summary_event_t {
  // Run identity shared by all emitted events.
  const loom_benchmark_run_identity_t* run;
  // Optional artifact bundle observed during execution.
  const loom_benchmark_artifact_bundle_t* artifact_bundle;
  // Number of planned check.case records.
  iree_host_size_t planned_case_count;
  // Number of planned check.benchmark records.
  iree_host_size_t planned_benchmark_count;
  // Number of selected benchmark candidates.
  iree_host_size_t selected_benchmark_count;
  // Number of logical benchmark samples selected for reporting.
  iree_host_size_t logical_sample_count;
  // Number of unique physical work items selected for execution.
  iree_host_size_t work_item_count;
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
} loom_benchmark_summary_event_t;

typedef struct loom_benchmark_device_event_t {
  // Run identity shared by all emitted events.
  const loom_benchmark_run_identity_t* run;
  // Borrowed HAL context whose current device state is reported.
  const loom_benchmark_hal_context_t* context;
} loom_benchmark_device_event_t;

typedef struct loom_benchmark_compile_event_t {
  // Run identity shared by all emitted events.
  const loom_benchmark_run_identity_t* run;
  // Stable candidate identity for the compile row.
  const loom_benchmark_candidate_identity_t* candidate;
  // Planned compilation shared by benchmark aliases and their work items.
  const loom_benchmark_hal_compile_item_t* compile_item;
  // Benchmark plan owning the candidate.
  const loom_testbench_benchmark_plan_t* benchmark_plan;
  // Case plan referenced by |benchmark_plan|.
  const loom_testbench_case_plan_t* case_plan;
  // HAL actual provider containing compile diagnostics and artifact paths.
  const loom_benchmark_hal_actual_provider_t* provider;
} loom_benchmark_compile_event_t;

typedef struct loom_benchmark_sample_event_t {
  // Run identity shared by all emitted events.
  const loom_benchmark_run_identity_t* run;
  // Stable candidate identity for the sample row.
  const loom_benchmark_candidate_identity_t* candidate;
  // Stable physical work item ordinal, or LOOM_BENCHMARK_INDEX_INVALID.
  iree_host_size_t work_item_index;
  // Parsed module used to render sample parameter assignments.
  const loom_module_t* module;
  // Benchmark plan owning the sample.
  const loom_testbench_benchmark_plan_t* benchmark_plan;
  // Case plan referenced by |benchmark_plan|.
  const loom_testbench_case_plan_t* case_plan;
  // Zero-based sample ordinal within the benchmark's selected sample space.
  iree_host_size_t benchmark_sample_ordinal;
  // Borrowed correctness result for this sample.
  const loom_testbench_case_sample_result_t* sample_result;
} loom_benchmark_sample_event_t;

typedef struct loom_benchmark_scenario_sample_event_t {
  // Run identity shared by all emitted events.
  const loom_benchmark_run_identity_t* run;
  // Logical benchmark selection that executed this correctness trial.
  const loom_benchmark_selected_benchmark_t* selection;
  // Physical work item shared by all aliases of this trial.
  iree_host_size_t work_item_index;
  // Zero-based sample ordinal within the selected benchmark.
  iree_host_size_t benchmark_sample_ordinal;
  // Borrowed observations, valid only while the sink handles this event.
  const loom_testbench_scenario_trial_result_t* result;
} loom_benchmark_scenario_sample_event_t;

typedef struct loom_benchmark_benchmark_result_event_t {
  // Run identity shared by all emitted events.
  const loom_benchmark_run_identity_t* run;
  // Stable candidate identity for the benchmark row.
  const loom_benchmark_candidate_identity_t* candidate;
  // Borrowed physical work item, including its representative selection.
  const loom_benchmark_work_item_t* work_item;
  // Parsed module used to render sample parameter assignments.
  const loom_module_t* module;
  // Benchmark plan owning the result.
  const loom_testbench_benchmark_plan_t* benchmark_plan;
  // Case plan referenced by |benchmark_plan|, or NULL for a scenario result.
  const loom_testbench_case_plan_t* case_plan;
  // Scenario plan referenced by |benchmark_plan|, or NULL for a case result.
  const loom_testbench_scenario_plan_t* scenario_plan;
  // Effective benchmark policy used by the measurement.
  const loom_benchmark_benchmark_policy_t* policy;
  // Borrowed benchmark result payload.
  const loom_benchmark_benchmark_result_t* benchmark_result;
  // Number of correctness samples executed before measurement.
  iree_host_size_t correctness_sample_count;
  // Number of correctness samples that failed expectations.
  iree_host_size_t correctness_failed_sample_count;
} loom_benchmark_benchmark_result_event_t;

typedef struct loom_benchmark_profile_replay_event_t {
  // Run identity shared by all emitted events.
  const loom_benchmark_run_identity_t* run;
  // Stable candidate identity for the profile-replay rows.
  const loom_benchmark_candidate_identity_t* candidate;
  // Stable physical work item ordinal, or LOOM_BENCHMARK_INDEX_INVALID.
  iree_host_size_t work_item_index;
  // Parsed module used to render sample parameter assignments.
  const loom_module_t* module;
  // Benchmark plan owning the replay result.
  const loom_testbench_benchmark_plan_t* benchmark_plan;
  // Case plan referenced by |benchmark_plan|, or NULL for a scenario replay.
  const loom_testbench_case_plan_t* case_plan;
  // Scenario plan referenced by |benchmark_plan|, or NULL for a case replay.
  const loom_testbench_scenario_plan_t* scenario_plan;
  // Effective benchmark policy used by the replay.
  const loom_benchmark_benchmark_policy_t* policy;
  // Borrowed benchmark result payload.
  const loom_benchmark_benchmark_result_t* benchmark_result;
} loom_benchmark_profile_replay_event_t;

typedef struct loom_benchmark_failure_event_t {
  // Run identity shared by all emitted events.
  const loom_benchmark_run_identity_t* run;
  // Pipeline stage that rejected the input.
  iree_string_view_t stage;
  // Failure kind within |stage|.
  iree_string_view_t kind;
  // Human-readable failure summary.
  iree_string_view_t message;
  // Optional captured diagnostics for this failure.
  const loom_benchmark_diagnostic_capture_t* diagnostics;
  // Optional testbench module plan that owns |planning_issues|.
  const loom_testbench_module_plan_t* testbench_plan;
  // Optional testbench planning issues explaining this failure.
  const loom_testbench_issue_t* planning_issues;
  // Number of entries in |planning_issues|.
  iree_host_size_t planning_issue_count;
} loom_benchmark_failure_event_t;

typedef struct loom_benchmark_benchmark_repetition_event_t {
  // Run identity shared by all emitted events.
  const loom_benchmark_run_identity_t* run;
  // Interleaved comparison candidate that produced this repetition.
  const loom_benchmark_dispatch_comparison_candidate_t* candidate;
  // Baseline candidate identity for the comparison group.
  const loom_benchmark_candidate_identity_t* baseline;
  // Comparison group label shared by all candidates in this comparison.
  iree_string_view_t comparison_group;
  // Interleaving method name.
  iree_string_view_t method;
  // Zero-based global execution order within the comparison schedule.
  iree_host_size_t order_index;
  // Zero-based repetition ordinal for |candidate|.
  iree_host_size_t repetition_index;
  // Compact schedule label used in ABABA/round-robin output.
  char schedule_token;
  // True when final-batch profiling was suppressed for interleaving stability.
  bool profile_suppressed;
  // Borrowed benchmark result payload for this repetition.
  const loom_benchmark_benchmark_result_t* benchmark_result;
} loom_benchmark_benchmark_repetition_event_t;

typedef struct loom_benchmark_comparison_event_t {
  // Run identity shared by all emitted events.
  const loom_benchmark_run_identity_t* run;
  // Baseline interleaved comparison candidate.
  const loom_benchmark_dispatch_comparison_candidate_t* baseline;
  // Candidate compared against |baseline|.
  const loom_benchmark_dispatch_comparison_candidate_t* candidate;
  // Comparison group label shared by all candidates in this comparison.
  iree_string_view_t comparison_group;
  // Interleaving method name.
  iree_string_view_t method;
} loom_benchmark_comparison_event_t;

typedef struct loom_benchmark_work_plan_event_t {
  // Run identity shared by all emitted events.
  const loom_benchmark_run_identity_t* run;
  // Parsed module that owns case and benchmark plan references.
  const loom_module_t* module;
  // Selected benchmark and logical-to-physical work plan.
  const loom_benchmark_work_plan_t* work_plan;
} loom_benchmark_work_plan_event_t;

typedef struct loom_benchmark_event_t {
  // Event kind and union discriminator.
  loom_benchmark_event_kind_t kind;
  union {
    // Payload for LOOM_BENCHMARK_EVENT_RUN.
    loom_benchmark_run_event_t run;
    // Payload for LOOM_BENCHMARK_EVENT_PLAN.
    loom_benchmark_plan_event_t plan;
    // Payload for LOOM_BENCHMARK_EVENT_SUMMARY.
    loom_benchmark_summary_event_t summary;
    // Payload for LOOM_BENCHMARK_EVENT_DEVICE.
    loom_benchmark_device_event_t device;
    // Payload for LOOM_BENCHMARK_EVENT_COMPILE.
    loom_benchmark_compile_event_t compile;
    // Payload for LOOM_BENCHMARK_EVENT_SAMPLE.
    loom_benchmark_sample_event_t sample;
    // Payload for LOOM_BENCHMARK_EVENT_SCENARIO_SAMPLE.
    loom_benchmark_scenario_sample_event_t scenario_sample;
    // Payload for LOOM_BENCHMARK_EVENT_BENCHMARK_RESULT.
    loom_benchmark_benchmark_result_event_t benchmark_result;
    // Payload for LOOM_BENCHMARK_EVENT_PROFILE_REPLAY.
    loom_benchmark_profile_replay_event_t profile_replay;
    // Payload for LOOM_BENCHMARK_EVENT_FAILURE.
    loom_benchmark_failure_event_t failure;
    // Payload for LOOM_BENCHMARK_EVENT_BENCHMARK_REPETITION.
    loom_benchmark_benchmark_repetition_event_t benchmark_repetition;
    // Payload for LOOM_BENCHMARK_EVENT_COMPARISON.
    loom_benchmark_comparison_event_t comparison;
    // Payload for LOOM_BENCHMARK_EVENT_WORK_PLAN.
    loom_benchmark_work_plan_event_t work_plan;
  };
} loom_benchmark_event_t;

typedef iree_status_t (*loom_benchmark_event_sink_emit_fn_t)(
    void* user_data, const loom_benchmark_event_t* event);

typedef struct loom_benchmark_event_sink_t {
  // Callback invoked once per event with a borrowed event payload.
  loom_benchmark_event_sink_emit_fn_t emit;
  // Opaque sink state passed to |emit|.
  void* user_data;
} loom_benchmark_event_sink_t;

typedef struct loom_benchmark_jsonl_event_sink_t {
  // Borrowed JSONL row sink receiving rendered events during emission.
  loom_benchmark_jsonl_sink_t* jsonl_sink;
  // Renderer-local selected-device row suppression state.
  loom_benchmark_device_row_state_t device_row_state;
} loom_benchmark_jsonl_event_sink_t;

// Emits |event| to |sink|.
iree_status_t loom_benchmark_event_sink_emit(
    const loom_benchmark_event_sink_t* sink,
    const loom_benchmark_event_t* event);

// Emits the run metadata event.
iree_status_t loom_benchmark_event_sink_emit_run(
    const loom_benchmark_event_sink_t* sink,
    const loom_benchmark_run_identity_t* run, bool dry_run,
    const loom_sanitizer_options_t* sanitizer);

// Emits one selected benchmark plan event.
iree_status_t loom_benchmark_event_sink_emit_plan(
    const loom_benchmark_event_sink_t* sink,
    const loom_benchmark_run_identity_t* run, const loom_module_t* module,
    const loom_benchmark_selected_benchmark_t* selection,
    const loom_benchmark_options_t* options);

// Emits the deduplicated planned work graph.
iree_status_t loom_benchmark_event_sink_emit_work_plan(
    const loom_benchmark_event_sink_t* sink,
    const loom_benchmark_run_identity_t* run, const loom_module_t* module,
    const loom_benchmark_work_plan_t* work_plan);

// Emits the terminal aggregate summary event.
iree_status_t loom_benchmark_event_sink_emit_summary(
    const loom_benchmark_event_sink_t* sink,
    const loom_benchmark_run_identity_t* run,
    const loom_benchmark_artifact_bundle_t* artifact_bundle,
    iree_host_size_t planned_case_count,
    iree_host_size_t planned_benchmark_count,
    iree_host_size_t selected_benchmark_count,
    iree_host_size_t logical_sample_count, iree_host_size_t work_item_count,
    iree_host_size_t failure_count, iree_host_size_t failed_benchmark_count,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count, bool dry_run);

// Emits current HAL device metadata without initializing the runtime.
iree_status_t loom_benchmark_event_sink_emit_device(
    const loom_benchmark_event_sink_t* sink,
    const loom_benchmark_run_identity_t* run,
    const loom_benchmark_hal_context_t* context);

// Emits one canonical compiled entry's metadata.
iree_status_t loom_benchmark_event_sink_emit_compile(
    const loom_benchmark_event_sink_t* sink,
    const loom_benchmark_run_identity_t* run,
    const loom_benchmark_candidate_identity_t* candidate,
    const loom_benchmark_hal_compile_item_t* compile_item,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const loom_benchmark_hal_actual_provider_t* provider);

// Emits one correctness sample result.
iree_status_t loom_benchmark_event_sink_emit_sample(
    const loom_benchmark_event_sink_t* sink,
    const loom_benchmark_run_identity_t* run,
    const loom_benchmark_candidate_identity_t* candidate,
    iree_host_size_t work_item_index, const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    iree_host_size_t benchmark_sample_ordinal,
    const loom_testbench_case_sample_result_t* sample_result);

// Emits one logical alias of a physical benchmark result. The work item's
// representative selection owns the shared evidence; other aliases refer to it.
iree_status_t loom_benchmark_event_sink_emit_benchmark_result(
    const loom_benchmark_event_sink_t* sink,
    const loom_benchmark_run_identity_t* run,
    const loom_benchmark_candidate_identity_t* candidate,
    const loom_benchmark_work_item_t* work_item, const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_scenario_plan_t* scenario_plan,
    const loom_benchmark_benchmark_policy_t* policy,
    const loom_benchmark_benchmark_result_t* benchmark_result,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count);

// Emits optional profile-replay rows for one benchmark result.
iree_status_t loom_benchmark_event_sink_emit_profile_replay(
    const loom_benchmark_event_sink_t* sink,
    const loom_benchmark_run_identity_t* run,
    const loom_benchmark_candidate_identity_t* candidate,
    iree_host_size_t work_item_index, const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_scenario_plan_t* scenario_plan,
    const loom_benchmark_benchmark_policy_t* policy,
    const loom_benchmark_benchmark_result_t* benchmark_result);

// Emits parse, verify, planning, or infrastructure failure metadata.
iree_status_t loom_benchmark_event_sink_emit_failure(
    const loom_benchmark_event_sink_t* sink,
    const loom_benchmark_run_identity_t* run, iree_string_view_t stage,
    iree_string_view_t kind, iree_string_view_t message,
    const loom_benchmark_diagnostic_capture_t* diagnostics);

// Emits a failure row with attached testbench planning issues.
iree_status_t loom_benchmark_event_sink_emit_planning_failure(
    const loom_benchmark_event_sink_t* sink,
    const loom_benchmark_run_identity_t* run, iree_string_view_t stage,
    iree_string_view_t kind, iree_string_view_t message,
    const loom_testbench_module_plan_t* testbench_plan,
    const loom_testbench_issue_t* planning_issues,
    iree_host_size_t planning_issue_count);

// Emits one interleaved comparison benchmark repetition.
iree_status_t loom_benchmark_event_sink_emit_benchmark_repetition(
    const loom_benchmark_event_sink_t* sink,
    const loom_benchmark_run_identity_t* run,
    const loom_benchmark_dispatch_comparison_candidate_t* candidate,
    const loom_benchmark_candidate_identity_t* baseline,
    iree_string_view_t comparison_group, iree_string_view_t method,
    iree_host_size_t order_index, iree_host_size_t repetition_index,
    char schedule_token, bool profile_suppressed,
    const loom_benchmark_benchmark_result_t* benchmark_result);

// Emits one aggregate interleaved comparison result.
iree_status_t loom_benchmark_event_sink_emit_comparison(
    const loom_benchmark_event_sink_t* sink,
    const loom_benchmark_run_identity_t* run,
    const loom_benchmark_dispatch_comparison_candidate_t* baseline,
    const loom_benchmark_dispatch_comparison_candidate_t* candidate,
    iree_string_view_t comparison_group, iree_string_view_t method);

// Initializes a JSONL event adapter over |jsonl_sink|.
void loom_benchmark_jsonl_event_sink_initialize(
    loom_benchmark_jsonl_sink_t* jsonl_sink,
    loom_benchmark_jsonl_event_sink_t* out_adapter,
    loom_benchmark_event_sink_t* out_sink);

// Releases borrowed references held by |adapter|.
void loom_benchmark_jsonl_event_sink_deinitialize(
    loom_benchmark_jsonl_event_sink_t* adapter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_BENCHMARK_EVENT_H_
