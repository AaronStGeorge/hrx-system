// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared loom-benchmark run, benchmark, and HAL candidate model.

#ifndef LOOM_TOOLS_LOOM_BENCHMARK_MODEL_H_
#define LOOM_TOOLS_LOOM_BENCHMARK_MODEL_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "loom/tooling/execution/hal/benchmark.h"
#include "loom/tooling/execution/hal/testbench_actual.h"
#include "loom/tooling/testbench/executor.h"
#include "loom/tooling/testbench/requirements.h"
#include "loom/tools/loom-benchmark/configuration.h"
#include "loom/tools/loom-benchmark/output.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"
#include "loomc/artifact.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_benchmark_launch_evidence_t
    loom_benchmark_launch_evidence_t;
typedef struct loom_benchmark_hal_compile_item_t
    loom_benchmark_hal_compile_item_t;

// Invalid stable ordinal for optional benchmark model indexes.
#define LOOM_BENCHMARK_INDEX_INVALID IREE_HOST_SIZE_MAX

typedef enum loom_benchmark_measure_e {
  // Full testbench case execution wall time.
  LOOM_BENCHMARK_MEASURE_CASE_END_TO_END = 0,
  // Prepared HAL dispatch completion timing with correctness outside timing.
  LOOM_BENCHMARK_MEASURE_DISPATCH_COMPLETE = 1,
} loom_benchmark_measure_t;

typedef struct loom_benchmark_benchmark_policy_t {
  // Parsed measurement mode.
  loom_benchmark_measure_t measure_kind;
  // Declared measurement mode.
  iree_string_view_t measure;
  // Number of warmup iterations run before measured timing.
  iree_host_size_t warmup_iterations;
  // Number of measured iterations.
  iree_host_size_t iterations;
  // Counter-set selection storage referenced by |hal_options|.
  iree_hal_profile_counter_set_selection_t profile_counter_set;
  // HAL benchmark options for dispatch_complete measurement.
  loom_run_hal_benchmark_options_t hal_options;
} loom_benchmark_benchmark_policy_t;

typedef struct loom_benchmark_diagnostic_capture_t {
  // Canonical diagnostic objects emitted by this candidate compile.
  loom_json_value_list_t json_values;
  // True after |json_values| has been initialized.
  bool initialized;
  // Number of error diagnostics captured.
  iree_host_size_t error_count;
  // Number of warning diagnostics captured.
  iree_host_size_t warning_count;
  // Number of remark diagnostics captured.
  iree_host_size_t remark_count;
} loom_benchmark_diagnostic_capture_t;

typedef struct loom_benchmark_device_row_state_t {
  // True once a selected-device row has been appended.
  bool appended;
} loom_benchmark_device_row_state_t;

typedef struct loom_benchmark_candidate_identity_t {
  // Deterministic candidate identifier within the source/run selection.
  iree_string_view_t candidate_id;
  // Zero-based selected benchmark ordinal within this run.
  iree_host_size_t candidate_index;
} loom_benchmark_candidate_identity_t;

typedef struct loom_benchmark_selected_benchmark_t {
  // Stable candidate identity for rows produced by this selected benchmark.
  loom_benchmark_candidate_identity_t identity;
  // Inline storage backing |identity.candidate_id|.
  char candidate_id_storage[32];
  // Borrowed benchmark plan selected from the module plan.
  const loom_testbench_benchmark_plan_t* benchmark_plan;
  // Source-order ordinal of |benchmark_plan| in its module plan.
  iree_host_size_t benchmark_index;
  // Borrowed case plan referenced by |benchmark_plan|.
  const loom_testbench_case_plan_t* case_plan;
  // Borrowed scenario plan referenced by |benchmark_plan|.
  const loom_testbench_scenario_plan_t* scenario_plan;
  // First selected benchmark-local sample ordinal.
  iree_host_size_t begin_benchmark_sample;
  // One-past-end selected benchmark-local sample ordinal.
  iree_host_size_t end_benchmark_sample;
  // Run-arena-owned reason shared by all selections of a skipped record.
  // NULL for admitted records and unevaluated dry-run/compile-only plans.
  const loom_testbench_requirement_result_t* skip_reason;
  // Effective benchmark policy after command-line overrides.
  loom_benchmark_benchmark_policy_t policy;
} loom_benchmark_selected_benchmark_t;

typedef struct loom_benchmark_timing_stats_t {
  // Number of measured timing samples.
  iree_host_size_t count;
  // Sum of measured durations in nanoseconds.
  int64_t total_ns;
  // Minimum measured duration in nanoseconds.
  int64_t minimum_ns;
  // Maximum measured duration in nanoseconds.
  int64_t maximum_ns;
  // Arithmetic mean of measured durations in nanoseconds.
  double mean_ns;
  // Nearest-rank median duration in nanoseconds.
  int64_t p50_ns;
  // Nearest-rank p90 duration in nanoseconds.
  int64_t p90_ns;
} loom_benchmark_timing_stats_t;

typedef uint8_t loom_benchmark_buffer_materialization_t;
enum loom_benchmark_buffer_materialization_e {
  // The report does not know how this buffer family was materialized.
  LOOM_BENCHMARK_BUFFER_MATERIALIZATION_UNKNOWN = 0,
  // Buffers were materialized as host-local device-visible mappings.
  LOOM_BENCHMARK_BUFFER_MATERIALIZATION_HOST_VISIBLE = 1,
  // Buffers were materialized using the materializer default device-local
  // placement.
  LOOM_BENCHMARK_BUFFER_MATERIALIZATION_DEVICE_LOCAL = 2,
};

typedef struct loom_benchmark_data_cache_summary_t {
  // True when the dispatch benchmark has populated this summary.
  bool populated;
  // Placement used for correctness samples before benchmark timing.
  loom_benchmark_buffer_materialization_t correctness_materialization;
  // Placement used for measured dispatch-complete binding-ring buffers.
  loom_benchmark_buffer_materialization_t measurement_materialization;
  // Number of HAL binding references in each dispatch binding set.
  iree_host_size_t binding_count;
  // Number of physical binding sets materialized from check ops.
  iree_host_size_t binding_ring_count;
  // Number of pre-recorded command buffers rotated across benchmark batches.
  iree_host_size_t command_buffer_ring_count;
  // Number of physical kernel dispatches in each timed logical batch. This is
  // zero for command programs containing only transfer operations.
  iree_host_size_t dispatches_per_batch;
  // Requested minimum byte size for unique reachable binding ranges.
  uint64_t requested_min_ring_bytes;
  // Sum of binding-reference byte lengths in the first materialized set.
  uint64_t binding_set_bytes;
  // Sum of binding-reference byte lengths across the physical ring.
  uint64_t binding_ring_bytes;
  // Number of disjoint reachable binding ranges in one materialized set.
  iree_host_size_t binding_range_count;
  // Sum of reachable binding ranges in one materialized set.
  uint64_t binding_range_set_bytes;
  // Sum of reachable binding ranges across the physical ring.
  uint64_t binding_range_ring_bytes;
  // Number of unique logical input allocations in one materialized set.
  iree_host_size_t materialized_storage_count;
  // Sum of unique logical input-allocation bytes in one materialized set.
  uint64_t materialized_storage_set_bytes;
  // Sum of unique logical input-allocation bytes across the physical ring.
  uint64_t materialized_storage_ring_bytes;
} loom_benchmark_data_cache_summary_t;

typedef struct loom_benchmark_benchmark_result_t {
  // Stable benchmark state string when it differs from the default.
  iree_string_view_t state;
  // Borrowed requirement evidence when execution was skipped, otherwise NULL.
  const loom_testbench_requirement_result_t* skip_reason;
  // True when failure fields below describe why the benchmark did not run.
  bool has_failure;
  // Kernel entry whose candidate failed, when the failure is entry-specific.
  iree_string_view_t failure_entry;
  // Product stage that failed: compile, prepare, benchmark, etc.
  iree_string_view_t failure_stage;
  // Failure kind within |failure_stage|.
  iree_string_view_t failure_kind;
  // Optional human-facing failure message.
  iree_string_view_t failure_message;
  // Number of error diagnostics associated with the failure.
  iree_host_size_t diagnostic_error_count;
  // Number of warning diagnostics associated with the failure.
  iree_host_size_t diagnostic_warning_count;
  // Number of remark diagnostics associated with the failure.
  iree_host_size_t diagnostic_remark_count;
  // JSON array entries for structured diagnostics associated with the failure.
  iree_string_view_t diagnostic_json;
  // Planned compilation whose per-entry evidence is shared by case results.
  // NULL for products whose compilation is owned by the work item itself.
  const loom_benchmark_hal_compile_item_t* compile_item;
  // Public structured compile report, serialized inline only when the product
  // has no shared |compile_item|.
  const loomc_artifact_t* compile_report;
  // Sidecar compile report artifact path for debug/full bundles, if any.
  iree_string_view_t compile_report_artifact_path;
  // Sidecar artifact manifest path for debug/full bundles, if any.
  iree_string_view_t artifact_manifest_path;
  // Target-native executable artifact path for debug/full bundles, if any.
  iree_string_view_t target_artifact_path;
  // Target-native textual listing path for debug/full bundles, if any.
  iree_string_view_t target_listing_path;
  // HAL executable artifact path for debug/full bundles, if any.
  iree_string_view_t hal_executable_path;
  // True when benchmark setup and timing completed.
  bool executed;
  // True when no measured or warmup sample failed expectations.
  bool passed;
  // True when |benchmark_sample_ordinal| identifies one selected benchmark
  // sample and can be passed back through --sample.
  bool has_benchmark_sample_ordinal;
  // Concrete ordinal within the selected benchmark's sample space.
  iree_host_size_t benchmark_sample_ordinal;
  // True when |scenario_coordinate| identifies the measured scenario trial.
  bool has_scenario_coordinate;
  // Concrete scenario coordinate measured by dispatch_complete.
  loom_testbench_scenario_sample_coordinate_t scenario_coordinate;
  // Number of selected subject samples run per benchmark iteration.
  iree_host_size_t samples_per_iteration;
  // Borrowed concrete workload and resolved launch records for the timed work
  // item. Empty for benchmarks that do not submit HAL kernel launches.
  const loom_benchmark_launch_evidence_t* launch_evidence;
  // Failed sample executions observed during warmup and measured iterations.
  iree_host_size_t failed_sample_count;
  // Timing summary for measured iterations.
  loom_benchmark_timing_stats_t timing;
  // True when |hal_benchmark| contains dispatch_complete evidence.
  bool has_hal_benchmark;
  // HAL batch benchmark evidence.
  loom_run_hal_benchmark_result_t hal_benchmark;
  // Device-buffer reuse shape used by the HAL batch benchmark.
  loom_benchmark_data_cache_summary_t data_cache;
} loom_benchmark_benchmark_result_t;

typedef struct loom_benchmark_hal_context_t {
  // Tool configuration with linked artifact-provider registries.
  const loom_benchmark_configuration_t* configuration;
  // Optional artifact bundle receiving HAL profile artifact references.
  loom_benchmark_artifact_bundle_t* artifact_bundle;
  // Shared HAL runtime and artifact-provider state.
  loom_run_hal_testbench_context_t execution;
} loom_benchmark_hal_context_t;

typedef struct loom_benchmark_hal_actual_provider_t {
  // Shared HAL runtime and artifact-provider state.
  loom_benchmark_hal_context_t* context;
  // Shared HAL actual provider owning compilation and dispatch state.
  loom_run_hal_testbench_actual_provider_t execution;
  // First source-order launch of this root within the compile item. All calls
  // of that root share its report; single-provider items use zero.
  iree_host_size_t compile_entry_index;
  // Optional suffix appended to debug/full artifact leaves.
  iree_string_view_t artifact_path_suffix;
  // Structured diagnostics emitted while compiling this candidate.
  loom_benchmark_diagnostic_capture_t diagnostics;
  // Borrowed view into |compile_report_artifact_path_storage|.
  iree_string_view_t compile_report_artifact_path;
  // Owned debug/full bundle compile-report artifact path.
  char* compile_report_artifact_path_storage;
  // Borrowed view into |artifact_manifest_path_storage|.
  iree_string_view_t artifact_manifest_path;
  // Owned debug/full bundle artifact-manifest path.
  char* artifact_manifest_path_storage;
  // Borrowed view into |target_artifact_path_storage|.
  iree_string_view_t target_artifact_path;
  // Owned debug/full bundle target-native artifact path.
  char* target_artifact_path_storage;
  // Borrowed view into |target_listing_path_storage|.
  iree_string_view_t target_listing_path;
  // Owned debug/full bundle target-native listing path.
  char* target_listing_path_storage;
  // Borrowed view into |hal_executable_path_storage|.
  iree_string_view_t hal_executable_path;
  // Owned debug/full bundle HAL executable artifact path.
  char* hal_executable_path_storage;
} loom_benchmark_hal_actual_provider_t;

typedef struct loom_benchmark_hal_actual_sequence_t {
  // Host allocator used for sequence-owned provider storage.
  iree_allocator_t host_allocator;
  // Sequence-owned providers in check.case source order.
  loom_benchmark_hal_actual_provider_t* providers;
  // Number of entries in |providers|.
  iree_host_size_t provider_count;
  // Prepared ordered execution over the embedded execution providers.
  loom_run_hal_testbench_actual_sequence_execution_t* execution;
} loom_benchmark_hal_actual_sequence_t;

typedef struct loom_benchmark_dispatch_comparison_candidate_t {
  // Selected benchmark/subject/policy identity for this comparison member.
  const loom_benchmark_selected_benchmark_t* selection;
  // Module owning the selected benchmark and subject plans.
  const loom_module_t* module;
  // Deduplicated physical work item index used for setup.
  iree_host_size_t work_item_index;
  // First benchmark-local sample ordinal included in the comparison window.
  iree_host_size_t begin_sample;
  // One-past-end benchmark-local sample ordinal included in the comparison
  // window.
  iree_host_size_t end_sample;
  // Number of correctness samples executed before interleaved timing.
  iree_host_size_t correctness_sample_count;
  // Number of failed correctness samples observed before interleaved timing.
  iree_host_size_t correctness_failed_sample_count;
  // True when compile and correctness succeeded and timing windows can run.
  bool runnable;
  // Per-repetition p50 logical-operation timings collected for aggregate rows.
  iree_duration_t* p50_samples;
  // Per-repetition p90 logical-operation timings collected for aggregate rows.
  iree_duration_t* p90_samples;
  // Number of entries populated in |p50_samples| and |p90_samples|.
  iree_host_size_t sample_count;
  // Capacity of |p50_samples| and |p90_samples|.
  iree_host_size_t sample_capacity;
} loom_benchmark_dispatch_comparison_candidate_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_BENCHMARK_MODEL_H_
