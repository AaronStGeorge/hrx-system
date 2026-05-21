// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/snapshot.h"

#include "iree/base/internal/json.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

static iree_string_view_t SnapshotJson(
    iree_benchmark_loom_snapshot_sink_t* snapshot,
    iree_string_builder_t* storage) {
  iree_string_builder_reset(storage);
  IREE_EXPECT_OK(
      iree_benchmark_loom_snapshot_sink_append_json(snapshot, storage));
  return iree_string_builder_view(storage);
}

static iree_string_view_t ParseJsonDocument(iree_string_view_t json) {
  iree_string_view_t cursor = json;
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_consume_value(&cursor, &value));
  IREE_EXPECT_OK(iree_json_consume_insignificant(&cursor));
  EXPECT_TRUE(iree_string_view_is_empty(cursor));
  return value;
}

TEST(BenchmarkSnapshotSinkTest, AggregatesDeduplicatedWorkItems) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_benchmark_loom_snapshot_sink_t snapshot = {};
  IREE_ASSERT_OK(
      iree_benchmark_loom_snapshot_sink_initialize(allocator, &snapshot));
  iree_benchmark_loom_event_sink_t event_sink = {};
  iree_benchmark_loom_snapshot_event_sink_initialize(&snapshot, &event_sink);

  iree_benchmark_loom_run_identity_t run = {};
  run.run_id = IREE_SV("run");
  run.source = IREE_SV("input.loom");
  run.results_path = IREE_SV("-");
  run.file_output_dir = IREE_SV("/tmp/loom");
  iree_benchmark_loom_candidate_identity_t candidate0 = {};
  candidate0.candidate_id = IREE_SV("c0");
  candidate0.candidate_index = 0;
  iree_benchmark_loom_candidate_identity_t candidate1 = {};
  candidate1.candidate_id = IREE_SV("c1");
  candidate1.candidate_index = 1;
  loom_module_t module = {};
  loom_testbench_benchmark_plan_t benchmark_plan = {};
  benchmark_plan.name = IREE_SV("kernel_latency");
  loom_testbench_case_plan_t case_plan = {};
  case_plan.name = IREE_SV("kernel_case");
  iree_benchmark_loom_benchmark_policy_t policy = {};
  policy.measure_kind = IREE_BENCHMARK_LOOM_MEASURE_CASE_END_TO_END;
  policy.measure = IREE_SV("case_end_to_end");
  iree_benchmark_loom_benchmark_result_t result = {};
  result.executed = true;
  result.passed = true;
  result.has_sample_ordinal = true;
  result.sample_ordinal = 0;
  result.samples_per_iteration = 1;
  result.timing.count = 3;
  result.timing.total_ns = 90;
  result.timing.minimum_ns = 20;
  result.timing.maximum_ns = 40;
  result.timing.mean_ns = 30.0;
  result.timing.p50_ns = 30;
  result.timing.p90_ns = 40;

  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_run(
      &event_sink, &run, /*dry_run=*/false,
      IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_ONCE));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_benchmark_result(
      &event_sink, &run, &candidate0, /*work_item_index=*/7, &module,
      &benchmark_plan, &case_plan, &policy, &result,
      /*correctness_sample_count=*/1,
      /*correctness_failed_sample_count=*/0));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_benchmark_result(
      &event_sink, &run, &candidate1, /*work_item_index=*/7, &module,
      &benchmark_plan, &case_plan, &policy, &result,
      /*correctness_sample_count=*/1,
      /*correctness_failed_sample_count=*/0));
  iree_benchmark_loom_artifact_bundle_t bundle = {};
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_summary(
      &event_sink, &run, &bundle, /*planned_case_count=*/1,
      /*planned_benchmark_count=*/2, /*selected_benchmark_count=*/2,
      /*logical_sample_count=*/2, /*work_item_count=*/1,
      /*failure_count=*/0, /*failed_benchmark_count=*/0,
      /*correctness_sample_count=*/1, /*correctness_failed_sample_count=*/0,
      /*dry_run=*/false, IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_ONCE));

  iree_string_builder_t output;
  iree_string_builder_initialize(allocator, &output);
  iree_string_view_t root = ParseJsonDocument(SnapshotJson(&snapshot, &output));

  iree_string_view_t output_format = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_lookup_object_value(root, IREE_SV("output_format"),
                                               &output_format));
  EXPECT_TRUE(iree_string_view_equal(output_format, IREE_SV("snapshot")));

  iree_string_view_t work_items = iree_string_view_empty();
  IREE_ASSERT_OK(
      iree_json_lookup_object_value(root, IREE_SV("work_items"), &work_items));
  iree_host_size_t work_item_count = 0;
  IREE_ASSERT_OK(iree_json_array_length(work_items, &work_item_count));
  EXPECT_EQ(work_item_count, 1u);

  iree_string_view_t benchmarks = iree_string_view_empty();
  IREE_ASSERT_OK(
      iree_json_lookup_object_value(root, IREE_SV("benchmarks"), &benchmarks));
  iree_host_size_t benchmark_count = 0;
  IREE_ASSERT_OK(iree_json_array_length(benchmarks, &benchmark_count));
  EXPECT_EQ(benchmark_count, 2u);

  iree_string_view_t first_work_item = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_array_get(work_items, 0, &first_work_item));
  iree_string_view_t timing = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_lookup_object_value(first_work_item,
                                               IREE_SV("timing_ns"), &timing));
  iree_string_view_t redundant_timing = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_try_lookup_object_value(
      first_work_item, IREE_SV("batch_timing_ns"), &redundant_timing));
  EXPECT_TRUE(iree_string_view_is_empty(redundant_timing));
  iree_string_view_t compile_report = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_try_lookup_object_value(
      first_work_item, IREE_SV("compile_report"), &compile_report));
  EXPECT_TRUE(iree_string_view_is_empty(compile_report));

  iree_string_builder_deinitialize(&output);
  iree_benchmark_loom_snapshot_sink_deinitialize(&snapshot);
}

TEST(BenchmarkSnapshotSinkTest, DryRunReportsPlannedBenchmarksWithoutWork) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_benchmark_loom_snapshot_sink_t snapshot = {};
  IREE_ASSERT_OK(
      iree_benchmark_loom_snapshot_sink_initialize(allocator, &snapshot));
  iree_benchmark_loom_event_sink_t event_sink = {};
  iree_benchmark_loom_snapshot_event_sink_initialize(&snapshot, &event_sink);

  iree_benchmark_loom_run_identity_t run = {};
  run.run_id = IREE_SV("run");
  run.source = IREE_SV("input.loom");
  run.results_path = IREE_SV("-");
  run.file_output_dir = IREE_SV("/tmp/loom");
  iree_benchmark_loom_selected_benchmark_t selection = {};
  selection.identity.candidate_id = IREE_SV("c0");
  selection.identity.candidate_index = 0;
  loom_testbench_benchmark_plan_t benchmark_plan = {};
  benchmark_plan.name = IREE_SV("kernel_latency");
  benchmark_plan.sample_count = 4;
  loom_testbench_case_plan_t case_plan = {};
  case_plan.name = IREE_SV("kernel_case");
  selection.benchmark_plan = &benchmark_plan;
  selection.case_plan = &case_plan;
  selection.policy.measure = IREE_SV("case_end_to_end");
  iree_benchmark_loom_options_t options = {};
  iree_benchmark_loom_options_initialize(&options);
  loom_module_t module = {};
  iree_benchmark_loom_artifact_bundle_t bundle = {};

  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_run(
      &event_sink, &run, /*dry_run=*/true,
      IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_ONCE));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_plan(
      &event_sink, &run, &module, &selection, &options,
      IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_ONCE));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_summary(
      &event_sink, &run, &bundle, /*planned_case_count=*/1,
      /*planned_benchmark_count=*/1, /*selected_benchmark_count=*/1,
      /*logical_sample_count=*/4, /*work_item_count=*/0,
      /*failure_count=*/0, /*failed_benchmark_count=*/0,
      /*correctness_sample_count=*/0, /*correctness_failed_sample_count=*/0,
      /*dry_run=*/true, IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_ONCE));

  iree_string_builder_t output;
  iree_string_builder_initialize(allocator, &output);
  iree_string_view_t root = ParseJsonDocument(SnapshotJson(&snapshot, &output));

  bool dry_run = false;
  IREE_ASSERT_OK(iree_json_lookup_bool(root, IREE_SV("dry_run"), &dry_run));
  EXPECT_TRUE(dry_run);
  iree_string_view_t benchmarks = iree_string_view_empty();
  IREE_ASSERT_OK(
      iree_json_lookup_object_value(root, IREE_SV("benchmarks"), &benchmarks));
  iree_host_size_t benchmark_count = 0;
  IREE_ASSERT_OK(iree_json_array_length(benchmarks, &benchmark_count));
  EXPECT_EQ(benchmark_count, 1u);
  iree_string_view_t work_items = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_try_lookup_object_value(root, IREE_SV("work_items"),
                                                   &work_items));
  EXPECT_TRUE(iree_string_view_is_empty(work_items));

  iree_string_builder_deinitialize(&output);
  iree_benchmark_loom_snapshot_sink_deinitialize(&snapshot);
}

}  // namespace
}  // namespace loom
