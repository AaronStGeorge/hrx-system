// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/event.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

typedef struct event_collector_t {
  iree_host_size_t event_count;
  iree_benchmark_loom_event_t events[10];
} event_collector_t;

static iree_status_t collect_event(void* user_data,
                                   const iree_benchmark_loom_event_t* event) {
  event_collector_t* collector = (event_collector_t*)user_data;
  if (collector->event_count == IREE_ARRAYSIZE(collector->events)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "benchmark event collector is full");
  }
  collector->events[collector->event_count++] = *event;
  return iree_ok_status();
}

static iree_status_t reject_event(void* user_data,
                                  const iree_benchmark_loom_event_t* event) {
  (void)user_data;
  (void)event;
  return iree_make_status(IREE_STATUS_ABORTED, "sink stopped");
}

TEST(BenchmarkEventSinkTest, EmitsTypedLifecycleEvents) {
  event_collector_t collector = {};
  iree_benchmark_loom_event_sink_t sink = {};
  sink.emit = collect_event;
  sink.user_data = &collector;
  iree_benchmark_loom_run_identity_t run = {};
  run.run_id = IREE_SV("run");
  run.source = IREE_SV("input.loom");
  run.results_path = IREE_SV("-");
  loom_module_t module = {};
  iree_benchmark_loom_work_plan_t work_plan = {};
  iree_benchmark_loom_artifact_bundle_t artifact_bundle = {};

  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_run(
      &sink, &run, /*dry_run=*/true,
      IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_ONCE));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_work_plan(
      &sink, &run, &module, &work_plan));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_summary(
      &sink, &run, &artifact_bundle, /*planned_case_count=*/3,
      /*planned_benchmark_count=*/2, /*selected_benchmark_count=*/1,
      /*logical_sample_count=*/4, /*work_item_count=*/2,
      /*failure_count=*/0, /*failed_benchmark_count=*/0,
      /*correctness_sample_count=*/5, /*correctness_failed_sample_count=*/0,
      /*dry_run=*/true, IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_ONCE));

  ASSERT_EQ(collector.event_count, 3u);
  EXPECT_EQ(collector.events[0].kind, IREE_BENCHMARK_LOOM_EVENT_RUN);
  EXPECT_EQ(collector.events[0].run.run, &run);
  EXPECT_TRUE(collector.events[0].run.dry_run);
  EXPECT_EQ(collector.events[1].kind, IREE_BENCHMARK_LOOM_EVENT_WORK_PLAN);
  EXPECT_EQ(collector.events[1].work_plan.run, &run);
  EXPECT_EQ(collector.events[1].work_plan.module, &module);
  EXPECT_EQ(collector.events[1].work_plan.work_plan, &work_plan);
  EXPECT_EQ(collector.events[2].kind, IREE_BENCHMARK_LOOM_EVENT_SUMMARY);
  EXPECT_EQ(collector.events[2].summary.run, &run);
  EXPECT_EQ(collector.events[2].summary.artifact_bundle, &artifact_bundle);
  EXPECT_EQ(collector.events[2].summary.planned_case_count, 3u);
  EXPECT_EQ(collector.events[2].summary.planned_benchmark_count, 2u);
  EXPECT_EQ(collector.events[2].summary.selected_benchmark_count, 1u);
  EXPECT_EQ(collector.events[2].summary.logical_sample_count, 4u);
  EXPECT_EQ(collector.events[2].summary.work_item_count, 2u);
  EXPECT_EQ(collector.events[2].summary.correctness_sample_count, 5u);
}

TEST(BenchmarkEventSinkTest, EmitsTypedOutputRowEvents) {
  event_collector_t collector = {};
  iree_benchmark_loom_event_sink_t sink = {};
  sink.emit = collect_event;
  sink.user_data = &collector;
  iree_benchmark_loom_run_identity_t run = {};
  run.run_id = IREE_SV("run");
  iree_benchmark_loom_candidate_identity_t candidate = {};
  candidate.candidate_id = IREE_SV("c0");
  loom_module_t module = {};
  loom_testbench_benchmark_plan_t benchmark_plan = {};
  loom_testbench_case_plan_t case_plan = {};
  iree_benchmark_loom_benchmark_policy_t policy = {};
  iree_benchmark_loom_benchmark_result_t benchmark_result = {};
  loom_testbench_case_sample_result_t sample_result = {};
  iree_benchmark_loom_hal_context_t hal_context = {};
  iree_benchmark_loom_hal_actual_provider_t provider = {};
  iree_benchmark_loom_diagnostic_capture_t diagnostics = {};
  iree_benchmark_loom_selected_benchmark_t baseline_selection = {};
  baseline_selection.identity.candidate_id = IREE_SV("c0");
  iree_benchmark_loom_dispatch_comparison_candidate_t baseline = {};
  baseline.selection = &baseline_selection;
  iree_benchmark_loom_selected_benchmark_t comparison_selection = {};
  comparison_selection.identity.candidate_id = IREE_SV("c1");
  iree_benchmark_loom_dispatch_comparison_candidate_t comparison = {};
  comparison.selection = &comparison_selection;

  IREE_ASSERT_OK(
      iree_benchmark_loom_event_sink_emit_device(&sink, &run, &hal_context));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_compile(
      &sink, &run, &candidate, &benchmark_plan, &case_plan, &provider));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_sample(
      &sink, &run, &candidate, /*work_item_index=*/7, &module, &benchmark_plan,
      &case_plan, IREE_SV("once"), /*benchmark_sample_ordinal=*/1,
      /*case_sample_ordinal=*/2, &sample_result));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_benchmark_result(
      &sink, &run, &candidate, /*work_item_index=*/7, &module, &benchmark_plan,
      &case_plan, &policy, &benchmark_result,
      /*correctness_sample_count=*/3,
      /*correctness_failed_sample_count=*/4));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_profile(
      &sink, &run, &candidate, /*work_item_index=*/7, &module, &benchmark_plan,
      &case_plan, &policy, &benchmark_result));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_failure(
      &sink, &run, IREE_SV("verify"), IREE_SV("diagnostics"), IREE_SV("failed"),
      &diagnostics));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_benchmark_repetition(
      &sink, &run, &comparison, &baseline_selection.identity, IREE_SV("group"),
      IREE_SV("round_robin"), /*order_index=*/5,
      /*repetition_index=*/6, 'B', /*profile_suppressed=*/true,
      &benchmark_result));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_comparison(
      &sink, &run, &baseline, &comparison, IREE_SV("group"),
      IREE_SV("round_robin")));

  ASSERT_EQ(collector.event_count, 8u);
  EXPECT_EQ(collector.events[0].kind, IREE_BENCHMARK_LOOM_EVENT_DEVICE);
  EXPECT_EQ(collector.events[1].kind, IREE_BENCHMARK_LOOM_EVENT_COMPILE);
  EXPECT_EQ(collector.events[2].kind, IREE_BENCHMARK_LOOM_EVENT_SAMPLE);
  EXPECT_EQ(collector.events[2].sample.work_item_index, 7u);
  EXPECT_EQ(collector.events[2].sample.benchmark_sample_ordinal, 1u);
  EXPECT_EQ(collector.events[2].sample.case_sample_ordinal, 2u);
  EXPECT_EQ(collector.events[3].kind,
            IREE_BENCHMARK_LOOM_EVENT_BENCHMARK_RESULT);
  EXPECT_EQ(collector.events[3].benchmark_result.work_item_index, 7u);
  EXPECT_EQ(collector.events[3].benchmark_result.correctness_sample_count, 3u);
  EXPECT_EQ(
      collector.events[3].benchmark_result.correctness_failed_sample_count, 4u);
  EXPECT_EQ(collector.events[4].kind, IREE_BENCHMARK_LOOM_EVENT_PROFILE);
  EXPECT_EQ(collector.events[4].profile.work_item_index, 7u);
  EXPECT_EQ(collector.events[5].kind, IREE_BENCHMARK_LOOM_EVENT_FAILURE);
  EXPECT_EQ(collector.events[5].failure.diagnostics, &diagnostics);
  EXPECT_EQ(collector.events[6].kind,
            IREE_BENCHMARK_LOOM_EVENT_BENCHMARK_REPETITION);
  EXPECT_EQ(collector.events[6].benchmark_repetition.order_index, 5u);
  EXPECT_EQ(collector.events[6].benchmark_repetition.schedule_token, 'B');
  EXPECT_TRUE(collector.events[6].benchmark_repetition.profile_suppressed);
  EXPECT_EQ(collector.events[7].kind, IREE_BENCHMARK_LOOM_EVENT_COMPARISON);
  EXPECT_EQ(collector.events[7].comparison.candidate, &comparison);
}

TEST(BenchmarkEventSinkTest, PropagatesSinkStatus) {
  iree_benchmark_loom_event_sink_t sink = {};
  sink.emit = reject_event;
  iree_benchmark_loom_run_identity_t run = {};
  run.run_id = IREE_SV("run");
  run.source = IREE_SV("input.loom");
  run.results_path = IREE_SV("-");

  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        iree_benchmark_loom_event_sink_emit_run(
                            &sink, &run, /*dry_run=*/false,
                            IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_PER_SAMPLE));
}

}  // namespace
}  // namespace loom
