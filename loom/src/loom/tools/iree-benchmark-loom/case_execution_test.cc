// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/case_execution.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/check/ops.h"
#include "loom/tools/iree-benchmark-loom/options.h"

namespace loom {
namespace {

typedef struct event_collector_t {
  // Successfully collected events in emission order.
  iree_host_size_t event_count;
  // Event ordinal at which the sink fails instead of accepting an event.
  iree_host_size_t fail_at = IREE_HOST_SIZE_MAX;
  // Event envelopes; borrowed plan and identity pointers remain live in tests.
  iree_benchmark_loom_event_t events[16];
  // Owned copies of temporary result payloads referenced by collected events.
  iree_benchmark_loom_benchmark_result_t results[16];
} event_collector_t;

static iree_status_t collect_event(void* user_data,
                                   const iree_benchmark_loom_event_t* event) {
  event_collector_t* collector = (event_collector_t*)user_data;
  if (collector->event_count == collector->fail_at) {
    return iree_make_status(IREE_STATUS_DATA_LOSS, "event sink failed");
  }
  if (collector->event_count == IREE_ARRAYSIZE(collector->events)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "benchmark event collector is full");
  }
  const iree_host_size_t index = collector->event_count++;
  collector->events[index] = *event;
  if (event->kind == IREE_BENCHMARK_LOOM_EVENT_BENCHMARK_RESULT) {
    collector->results[index] = *event->benchmark_result.benchmark_result;
    collector->events[index].benchmark_result.benchmark_result =
        &collector->results[index];
  } else if (event->kind == IREE_BENCHMARK_LOOM_EVENT_PROFILE_REPLAY) {
    collector->results[index] = *event->profile_replay.benchmark_result;
    collector->events[index].profile_replay.benchmark_result =
        &collector->results[index];
  }
  return iree_ok_status();
}

TEST(BenchmarkResultAliasesTest, PreservesLocalIdentityAndFailureAccounting) {
  const loom_testbench_parameter_plan_t parameter = {.sample_count = 2};
  const loom_testbench_case_plan_t case_plan = {
      .parameters = &parameter,
      .parameter_count = 1,
      .cartesian_sample_count = 2,
      .sample_count = 2,
  };
  const iree_host_size_t assignments[] = {
      1, LOOM_TESTBENCH_PARAMETER_SAMPLE_ORDINAL_ALL, 0, 1};
  loom_testbench_benchmark_plan_t benchmarks[4] = {};
  for (size_t i = 0; i < IREE_ARRAYSIZE(benchmarks); ++i) {
    benchmarks[i] = {
        .case_index = 0,
        .scenario_index = LOOM_TESTBENCH_SCENARIO_INDEX_INVALID,
        .parameter_sample_ordinals = &assignments[i],
        .parameter_sample_ordinal_count = 1,
        .cartesian_sample_count = i == 1 ? 2u : 1u,
        .sample_count = i == 1 ? 2u : 1u,
    };
  }
  loom_module_t module = {};
  const loom_testbench_module_plan_t module_plan = {
      .module = &module,
      .cases = &case_plan,
      .case_count = 1,
      .benchmarks = benchmarks,
      .benchmark_count = IREE_ARRAYSIZE(benchmarks),
  };
  iree_benchmark_loom_options_t options = {};
  iree_benchmark_loom_options_initialize(&options);
  options.measure = IREE_SV("dispatch_complete");
  iree_benchmark_loom_work_plan_t plan = {};
  IREE_ASSERT_OK(iree_benchmark_loom_work_plan_initialize(
      &module_plan, &options, iree_allocator_system(), &plan));
  ASSERT_EQ(plan.work_item_count, 2u);

  const iree_benchmark_loom_run_identity_t run = {};
  for (int outcome = 0; outcome < 4; ++outcome) {
    SCOPED_TRACE(outcome);
    const iree_benchmark_loom_benchmark_result_t result = {
        .state = outcome == 3 ? IREE_SV("skipped") : iree_string_view_empty(),
        .executed = outcome < 2,
        .passed = outcome == 0,
    };
    event_collector_t collector = {};
    const iree_benchmark_loom_event_sink_t sink = {
        .emit = collect_event,
        .user_data = &collector,
    };
    iree_host_size_t failed_count = 7;
    for (size_t i = 0; i < plan.work_item_count; ++i) {
      IREE_ASSERT_OK(iree_benchmark_loom_emit_work_item_result_aliases(
          &run, &module_plan, &plan, &plan.work_items[i], &result, 3, 2, &sink,
          &failed_count));
    }
    EXPECT_EQ(failed_count, outcome == 0 || outcome == 3 ? 7u : 12u);
    ASSERT_EQ(collector.event_count, 10u);
    // First-selected physical order differs from sample order. Aliases are
    // interleaved in the logical array, and pinned local ordinals differ from
    // the unpinned selection even when the physical sample is the same.
    const size_t candidates[] = {0, 1, 3, 1, 2};
    const size_t ordinals[] = {0, 1, 0, 0, 0};
    for (size_t i = 0; i < IREE_ARRAYSIZE(candidates); ++i) {
      const auto& result_event = collector.events[2 * i];
      const auto& replay_event = collector.events[2 * i + 1];
      ASSERT_EQ(result_event.kind, IREE_BENCHMARK_LOOM_EVENT_BENCHMARK_RESULT);
      ASSERT_EQ(replay_event.kind, IREE_BENCHMARK_LOOM_EVENT_PROFILE_REPLAY);
      const auto& row = result_event.benchmark_result;
      const auto& replay = replay_event.profile_replay;
      const auto& selection = plan.selected_benchmarks[candidates[i]];
      EXPECT_EQ(row.candidate, &selection.identity);
      EXPECT_EQ(row.benchmark_plan, selection.benchmark_plan);
      EXPECT_EQ(row.work_item, &plan.work_items[i < 3 ? 0 : 1]);
      EXPECT_EQ(row.correctness_sample_count, 3u);
      EXPECT_EQ(row.correctness_failed_sample_count, 2u);
      EXPECT_EQ(replay.candidate, row.candidate);
      EXPECT_EQ(replay.benchmark_plan, row.benchmark_plan);
      EXPECT_EQ(replay.work_item_index, row.work_item->work_item_index);
      for (const auto* payload :
           {row.benchmark_result, replay.benchmark_result}) {
        EXPECT_EQ(payload->executed, result.executed);
        EXPECT_EQ(payload->passed, result.passed);
        EXPECT_TRUE(iree_string_view_equal(payload->state, result.state));
        EXPECT_TRUE(payload->has_benchmark_sample_ordinal);
        EXPECT_EQ(payload->benchmark_sample_ordinal, ordinals[i]);
        EXPECT_FALSE(payload->has_scenario_coordinate);
        EXPECT_EQ(payload->samples_per_iteration, 1u);
      }
    }
    EXPECT_FALSE(result.has_benchmark_sample_ordinal);
    EXPECT_EQ(result.samples_per_iteration, 0u);
  }

  // A sink failure stops at either half of a result/replay pair. Failure
  // accounting includes the attempted alias, but never a later alias.
  const iree_benchmark_loom_benchmark_result_t failed_result = {};
  for (size_t fail_at = 0; fail_at < 6; ++fail_at) {
    SCOPED_TRACE(fail_at);
    event_collector_t collector = {.fail_at = fail_at};
    const iree_benchmark_loom_event_sink_t sink = {
        .emit = collect_event,
        .user_data = &collector,
    };
    iree_host_size_t failed_count = 7;
    iree::Status status = iree_benchmark_loom_emit_work_item_result_aliases(
        &run, &module_plan, &plan, &plan.work_items[0], &failed_result, 0, 0,
        &sink, &failed_count);
    EXPECT_THAT(status,
                ::iree::testing::status::StatusIs(iree::StatusCode::kDataLoss));
    EXPECT_EQ(collector.event_count, fail_at);
    EXPECT_EQ(failed_count, 8 + fail_at / 2);
  }
  iree_benchmark_loom_work_plan_deinitialize(&plan);
}

class BenchmarkCaseExecutionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &plan_arena_);
    iree_arena_initialize(&block_pool_, &execution_arena_);

    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_CHECK, loom_check_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    for (loom_module_t* module : owned_modules_) {
      loom_module_free(module);
    }
    owned_modules_.clear();
    iree_arena_deinitialize(&execution_arena_);
    iree_arena_deinitialize(&plan_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn = const loom_op_vtable_t* const* (*)(iree_host_size_t *
                                                              out_count);

  void RegisterDialect(loom_dialect_id_t dialect_id, DialectVtablesFn fn) {
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables = fn(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)vtable_count));
  }

  loom_testbench_module_plan_t PlanModule(const char* source) {
    loom_text_parse_options_t parse_options = {.max_errors = 20};
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_text_parse(
        iree_make_cstring_view(source), IREE_SV("case_execution_test.loom"),
        &context_, &block_pool_, &parse_options, &module));
    EXPECT_NE(module, nullptr);
    owned_modules_.push_back(module);

    loom_testbench_module_plan_t module_plan = {};
    IREE_EXPECT_OK(loom_testbench_plan_module(module, nullptr, &plan_arena_,
                                              &module_plan));
    EXPECT_EQ(module_plan.issue_count, 0u);
    return module_plan;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t plan_arena_;
  iree_arena_allocator_t execution_arena_;
  loom_context_t context_;
  std::vector<loom_module_t*> owned_modules_;
};

TEST_F(BenchmarkCaseExecutionTest, RunsCaseEndToEndWorkItemThroughEventSink) {
  const loom_testbench_module_plan_t module_plan = PlanModule(R"(
check.case @sampled {
  %value = check.param.choice values([5, 7]) name("value") : i32
  check.expect.equal actual(%value) expected(%value) : i32
  check.return
}

check.benchmark<@sampled> @all_a
check.benchmark<@sampled> @all_b
)");

  iree_benchmark_loom_options_t options = {};
  iree_benchmark_loom_options_initialize(&options);
  options.measure = IREE_SV("case_end_to_end");
  options.iterations = 1;
  options.warmup_iterations = 0;

  iree_benchmark_loom_work_plan_t work_plan = {};
  IREE_ASSERT_OK(iree_benchmark_loom_work_plan_initialize(
      &module_plan, &options, iree_allocator_system(), &work_plan));
  ASSERT_EQ(work_plan.work_item_count, 1u);

  loom_testbench_case_execution_options_t execution_options = {};
  loom_testbench_case_execution_options_initialize(&execution_options);
  execution_options.materializer.host_allocator = iree_allocator_system();

  event_collector_t collector = {};
  iree_benchmark_loom_event_sink_t event_sink = {
      .emit = collect_event,
      .user_data = &collector,
  };
  iree_benchmark_loom_run_identity_t run = {
      .run_id = IREE_SV("run"),
      .source = IREE_SV("input.loom"),
      .results_path = IREE_SV("-"),
  };

  iree_host_size_t correctness_sample_count = 0;
  iree_host_size_t correctness_failed_sample_count = 0;
  iree_host_size_t failed_benchmark_count = 0;
  IREE_ASSERT_OK(iree_benchmark_loom_run_case_end_to_end_work_item(
      &run, &module_plan, &work_plan, &work_plan.work_items[0],
      &execution_options, &execution_arena_, iree_allocator_system(),
      &event_sink, &correctness_sample_count, &correctness_failed_sample_count,
      &failed_benchmark_count));

  EXPECT_EQ(correctness_sample_count, 2u);
  EXPECT_EQ(correctness_failed_sample_count, 0u);
  EXPECT_EQ(failed_benchmark_count, 0u);
  ASSERT_EQ(collector.event_count, 8u);
  EXPECT_EQ(collector.events[0].kind, IREE_BENCHMARK_LOOM_EVENT_SAMPLE);
  EXPECT_EQ(collector.events[0].sample.benchmark_sample_ordinal, 0u);
  EXPECT_TRUE(iree_string_view_equal(
      collector.events[0].sample.candidate->candidate_id, IREE_SV("c0")));
  EXPECT_EQ(collector.events[1].kind, IREE_BENCHMARK_LOOM_EVENT_SAMPLE);
  EXPECT_TRUE(iree_string_view_equal(
      collector.events[1].sample.candidate->candidate_id, IREE_SV("c1")));
  EXPECT_EQ(collector.events[2].kind, IREE_BENCHMARK_LOOM_EVENT_SAMPLE);
  EXPECT_EQ(collector.events[2].sample.benchmark_sample_ordinal, 1u);
  EXPECT_EQ(collector.events[2].sample.candidate->candidate_index, 0u);
  EXPECT_EQ(collector.events[3].kind, IREE_BENCHMARK_LOOM_EVENT_SAMPLE);
  EXPECT_EQ(collector.events[3].sample.benchmark_sample_ordinal, 1u);
  EXPECT_EQ(collector.events[3].sample.candidate->candidate_index, 1u);
  EXPECT_EQ(collector.events[4].kind,
            IREE_BENCHMARK_LOOM_EVENT_BENCHMARK_RESULT);
  EXPECT_EQ(collector.events[4].benchmark_result.work_item,
            &work_plan.work_items[0]);
  EXPECT_EQ(collector.events[4].benchmark_result.correctness_sample_count, 2u);
  EXPECT_EQ(collector.events[5].kind, IREE_BENCHMARK_LOOM_EVENT_PROFILE_REPLAY);
  EXPECT_EQ(collector.events[6].kind,
            IREE_BENCHMARK_LOOM_EVENT_BENCHMARK_RESULT);
  EXPECT_TRUE(iree_string_view_equal(
      collector.events[6].benchmark_result.candidate->candidate_id,
      IREE_SV("c1")));
  EXPECT_EQ(collector.events[7].kind, IREE_BENCHMARK_LOOM_EVENT_PROFILE_REPLAY);

  iree_benchmark_loom_work_plan_deinitialize(&work_plan);
}

}  // namespace
}  // namespace loom
