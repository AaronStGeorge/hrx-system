// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// JSON row emission and artifact report writing for iree-tune-loom.

#ifndef LOOM_TOOLS_IREE_TUNE_LOOM_REPORT_H_
#define LOOM_TOOLS_IREE_TUNE_LOOM_REPORT_H_

#include "iree/base/api.h"
#include "loom/tooling/testbench/testbench.h"
#include "loom/tools/iree-tune-loom/model.h"
#include "loom/tools/iree-tune-loom/output.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// Writes a borrowed status as a structured JSON object.
iree_status_t iree_tune_loom_write_status_object_json(
    iree_status_t status, loom_output_stream_t* stream);

// Writes the run identifier field shared by every JSONL row.
iree_status_t iree_tune_loom_write_run_id_field_json(
    const iree_tune_loom_run_identity_t* run, loom_output_stream_t* stream);

// Writes stable benchmark candidate identity fields.
iree_status_t iree_tune_loom_write_candidate_identity_json(
    const iree_tune_loom_candidate_identity_t* candidate,
    loom_output_stream_t* stream);

// Writes a specialization field when |specialization| is non-empty.
iree_status_t iree_tune_loom_write_specialization_field_json(
    iree_string_view_t specialization, loom_output_stream_t* stream);

// Writes shape fields for one concrete case sample when the case is shaped.
iree_status_t iree_tune_loom_write_shape_point_fields_json(
    const loom_module_t* module, const loom_testbench_case_plan_t* case_plan,
    iree_host_size_t sample_ordinal, loom_output_stream_t* stream);

// Writes shape-plan fields for a parameterized case.
iree_status_t iree_tune_loom_write_shape_plan_fields_json(
    const loom_module_t* module, const loom_testbench_case_plan_t* case_plan,
    loom_output_stream_t* stream);

// Writes selected HAL context identity fields into an open JSON object.
iree_status_t iree_tune_loom_write_hal_context_identity_fields_json(
    const iree_tune_loom_hal_context_t* context, loom_output_stream_t* stream);

// Writes a HAL profile summary object.
iree_status_t iree_tune_loom_write_hal_profile_summary_json(
    const loom_run_hal_profile_summary_t* profile,
    loom_output_stream_t* stream);

// Writes one field separator/name pair inside a JSON object.
iree_status_t iree_tune_loom_write_json_object_field_name(
    loom_output_stream_t* stream, bool* first_field, const char* name);

// Writes a required string field inside a JSON object.
iree_status_t iree_tune_loom_write_json_string_field(
    loom_output_stream_t* stream, bool* first_field, const char* name,
    iree_string_view_t value);

// Writes a string field only when |value| is non-empty.
iree_status_t iree_tune_loom_write_json_optional_string_field(
    loom_output_stream_t* stream, bool* first_field, const char* name,
    iree_string_view_t value);

// Writes a required uint32 field inside a JSON object.
iree_status_t iree_tune_loom_write_json_u32_field(loom_output_stream_t* stream,
                                                  bool* first_field,
                                                  const char* name,
                                                  uint32_t value);

// Writes a required uint64 field inside a JSON object.
iree_status_t iree_tune_loom_write_json_u64_field(loom_output_stream_t* stream,
                                                  bool* first_field,
                                                  const char* name,
                                                  uint64_t value);

// Writes a required host-size field inside a JSON object.
iree_status_t iree_tune_loom_write_json_size_field(loom_output_stream_t* stream,
                                                   bool* first_field,
                                                   const char* name,
                                                   iree_host_size_t value);

// Writes a benchmark result object without the surrounding JSONL row wrapper.
iree_status_t iree_tune_loom_write_benchmark_result_json(
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_tune_loom_benchmark_policy_t* policy,
    const iree_tune_loom_benchmark_result_t* benchmark_result,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count,
    loom_output_stream_t* stream);

// Appends the initial run row.
iree_status_t iree_tune_loom_append_run_row(
    const iree_tune_loom_run_identity_t* run, bool dry_run,
    iree_tune_loom_shape_specialization_mode_t shape_specialization_mode,
    iree_string_builder_t* output);

// Appends the selected HAL device row once per run.
iree_status_t iree_tune_loom_append_device_row(
    const iree_tune_loom_run_identity_t* run,
    iree_tune_loom_hal_context_t* context,
    iree_tune_loom_device_row_state_t* state, iree_string_builder_t* output);

// Writes target, listing, and HAL executable artifacts for a candidate.
iree_status_t iree_tune_loom_write_compiled_artifacts(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    iree_tune_loom_hal_actual_provider_t* provider, iree_allocator_t allocator);

// Writes a compile-report sidecar for a candidate when the bundle wants one.
iree_status_t iree_tune_loom_write_compile_report_artifact(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    iree_tune_loom_hal_actual_provider_t* provider, iree_allocator_t allocator);

// Appends a candidate compile row.
iree_status_t iree_tune_loom_append_compile_row(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_tune_loom_hal_actual_provider_t* provider,
    iree_string_builder_t* compile_output);

// Appends a benchmark result row.
iree_status_t iree_tune_loom_append_benchmark_result(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_tune_loom_benchmark_policy_t* policy,
    const iree_tune_loom_benchmark_result_t* benchmark_result,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count,
    iree_string_builder_t* benchmark_output);

// Appends a parse, verify, planning, or infrastructure failure row.
iree_status_t iree_tune_loom_append_failure_row(
    const iree_tune_loom_run_identity_t* run, iree_string_view_t stage,
    iree_string_view_t kind, iree_string_view_t message,
    const iree_tune_loom_diagnostic_capture_t* diagnostics,
    iree_string_builder_t* failure_output);

// Appends the final summary row.
iree_status_t iree_tune_loom_append_summary_row(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_artifact_bundle_t* bundle,
    iree_host_size_t planned_case_count,
    iree_host_size_t planned_benchmark_count,
    iree_host_size_t selected_benchmark_count, iree_host_size_t failure_count,
    iree_host_size_t failed_benchmark_count,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count, bool dry_run,
    iree_tune_loom_shape_specialization_mode_t shape_specialization_mode,
    iree_string_builder_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_TUNE_LOOM_REPORT_H_
