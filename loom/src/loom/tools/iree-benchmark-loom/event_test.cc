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
  iree_benchmark_loom_event_t events[4];
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
  iree_benchmark_loom_artifact_bundle_t artifact_bundle = {};

  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_run(
      &sink, &run, /*dry_run=*/true,
      IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_ONCE));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_summary(
      &sink, &run, &artifact_bundle, /*planned_case_count=*/3,
      /*planned_benchmark_count=*/2, /*selected_benchmark_count=*/1,
      /*failure_count=*/0, /*failed_benchmark_count=*/0,
      /*correctness_sample_count=*/5, /*correctness_failed_sample_count=*/0,
      /*dry_run=*/true, IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_ONCE));

  ASSERT_EQ(collector.event_count, 2u);
  EXPECT_EQ(collector.events[0].kind, IREE_BENCHMARK_LOOM_EVENT_RUN);
  EXPECT_EQ(collector.events[0].run.run, &run);
  EXPECT_TRUE(collector.events[0].run.dry_run);
  EXPECT_EQ(collector.events[1].kind, IREE_BENCHMARK_LOOM_EVENT_SUMMARY);
  EXPECT_EQ(collector.events[1].summary.run, &run);
  EXPECT_EQ(collector.events[1].summary.artifact_bundle, &artifact_bundle);
  EXPECT_EQ(collector.events[1].summary.planned_case_count, 3u);
  EXPECT_EQ(collector.events[1].summary.planned_benchmark_count, 2u);
  EXPECT_EQ(collector.events[1].summary.selected_benchmark_count, 1u);
  EXPECT_EQ(collector.events[1].summary.correctness_sample_count, 5u);
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
