// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/work_plan.h"

#include <algorithm>
#include <array>
#include <map>
#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/check/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/tools/iree-benchmark-loom/comparison_execution.h"

namespace loom {
namespace {

using ::iree::testing::status::StatusIs;
using ::testing::HasSubstr;

TEST(BenchmarkWorkPlanApiTest, EmptyModuleNeedsNoSelectionStorage) {
  const loom_testbench_module_plan_t module_plan = {};
  iree_benchmark_loom_options_t options = {};
  iree_benchmark_loom_options_initialize(&options);
  iree_benchmark_loom_work_plan_t plan = {};
  IREE_ASSERT_OK(iree_benchmark_loom_work_plan_initialize(
      &module_plan, &options, iree_allocator_null(), &plan));
  EXPECT_EQ(plan.selected_benchmark_count, 0u);
  EXPECT_EQ(plan.logical_sample_count, 0u);
  EXPECT_EQ(plan.work_item_count, 0u);
  EXPECT_EQ(plan.hal_compile_item_count, 0u);
  iree_benchmark_loom_work_plan_deinitialize(&plan);
}

TEST(BenchmarkWorkPlanApiTest, ExecutionSlotsCoverOnlySelectedPhysicalWork) {
  const loom_testbench_parameter_plan_t parameter = {.sample_count = 32};
  const loom_testbench_case_plan_t case_plan = {
      .parameters = &parameter,
      .parameter_count = 1,
      .kernel_launch_count = 2,
      .cartesian_sample_count = 32,
      .sample_count = 2,
      .sample_count_truncated = true,
  };
  const iree_host_size_t assignments[] = {29, 7, 29};
  loom_testbench_benchmark_plan_t benchmarks[3] = {};
  for (size_t i = 0; i < IREE_ARRAYSIZE(benchmarks); ++i) {
    benchmarks[i] = {
        .case_index = 0,
        .scenario_index = LOOM_TESTBENCH_SCENARIO_INDEX_INVALID,
        .parameter_sample_ordinals = &assignments[i],
        .parameter_sample_ordinal_count = 1,
        .cartesian_sample_count = 1,
        .sample_count = 1,
    };
  }
  const loom_testbench_module_plan_t module_plan = {
      .cases = &case_plan,
      .case_count = 1,
      .benchmarks = benchmarks,
      .benchmark_count = IREE_ARRAYSIZE(benchmarks),
  };
  iree_benchmark_loom_options_t options = {};
  iree_benchmark_loom_options_initialize(&options);
  for (const char* measure : {"case_end_to_end", "dispatch_complete"}) {
    SCOPED_TRACE(measure);
    options.measure = iree_make_cstring_view(measure);
    iree_benchmark_loom_work_plan_t plan = {};
    IREE_ASSERT_OK(iree_benchmark_loom_work_plan_initialize(
        &module_plan, &options, iree_allocator_system(), &plan));
    ASSERT_EQ(plan.hal_compile_item_count, 1u);
    ASSERT_EQ(plan.work_item_count, 2u);
    EXPECT_EQ(plan.hal_compile_items[0].execution_sample_count, 2u);
    EXPECT_EQ(plan.work_items[0].execution_sample_base, 0u);
    EXPECT_EQ(plan.work_items[1].execution_sample_base, 1u);
    EXPECT_EQ(plan.logical_samples[2].work_item_index, 0u);
    iree_benchmark_loom_work_plan_deinitialize(&plan);
  }
}

TEST(BenchmarkWorkPlanApiTest, DistinguishesSamplesBeyondFlatOrdinalCapacity) {
  constexpr iree_host_size_t radix = iree_host_size_t{1}
                                     << (sizeof(iree_host_size_t) * 4);
  const loom_testbench_parameter_plan_t parameters[] = {
      {.sample_count = radix}, {.sample_count = radix}, {.sample_count = 2}};
  const loom_testbench_case_plan_t case_plan = {
      .parameters = parameters,
      .parameter_count = IREE_ARRAYSIZE(parameters),
      .kernel_launch_count = 2,
      .cartesian_sample_count = IREE_HOST_SIZE_MAX,
      .sample_count = 2,
      .sample_count_truncated = true,
  };
  constexpr auto all = LOOM_TESTBENCH_PARAMETER_SAMPLE_ORDINAL_ALL;
  const iree_host_size_t assignments[5][3] = {{all, all, 0},
                                              {all, all, 1},
                                              {all, all, 1},
                                              {radix - 1, radix - 1, 0},
                                              {radix - 1, radix - 1, 1}};
  loom_testbench_benchmark_plan_t benchmarks[5] = {};
  for (size_t i = 0; i < IREE_ARRAYSIZE(benchmarks); ++i) {
    benchmarks[i] = {
        .case_index = 0,
        .scenario_index = LOOM_TESTBENCH_SCENARIO_INDEX_INVALID,
        .parameter_sample_ordinals = assignments[i],
        .parameter_sample_ordinal_count = 3,
        .cartesian_sample_count = i < 3 ? IREE_HOST_SIZE_MAX : 1,
        .sample_count = i < 3 ? 2u : 1u,
        .sample_count_truncated = i < 3,
    };
  }
  const loom_testbench_module_plan_t module_plan = {
      .cases = &case_plan,
      .case_count = 1,
      .benchmarks = benchmarks,
      .benchmark_count = IREE_ARRAYSIZE(benchmarks),
  };
  iree_benchmark_loom_options_t options = {};
  iree_benchmark_loom_options_initialize(&options);
  for (const char* measure : {"case_end_to_end", "dispatch_complete"}) {
    SCOPED_TRACE(measure);
    options.measure = iree_make_cstring_view(measure);
    const size_t windows =
        iree_string_view_equal(options.measure, IREE_SV("case_end_to_end")) ? 1
                                                                            : 2;
    iree_benchmark_loom_work_plan_t plan = {};
    IREE_ASSERT_OK(iree_benchmark_loom_work_plan_initialize(
        &module_plan, &options, iree_allocator_system(), &plan));
    ASSERT_EQ(plan.hal_compile_item_count, 1u);
    EXPECT_EQ(plan.hal_compile_items[0].execution_sample_count, 6u);
    ASSERT_EQ(plan.work_item_count, 2 * windows + 2);
    ASSERT_EQ(plan.logical_sample_count, 3 * windows + 2);
    for (size_t i = 0; i < windows; ++i) {
      EXPECT_NE(plan.logical_samples[i].work_item_index,
                plan.logical_samples[windows + i].work_item_index);
      EXPECT_EQ(plan.logical_samples[windows + i].work_item_index,
                plan.logical_samples[2 * windows + i].work_item_index);
    }
    // A representable HOST_SIZE_MAX position and a larger position share a
    // saturated sort key but must remain separate physical samples.
    EXPECT_EQ(plan.logical_samples[3 * windows].work_item_index, 2 * windows);
    EXPECT_EQ(plan.logical_samples[3 * windows + 1].work_item_index,
              2 * windows + 1);
    iree_benchmark_loom_work_plan_deinitialize(&plan);
  }
}

TEST(BenchmarkWorkPlanApiTest, CompareLookupPreservesRequestsAndDiagnostics) {
  const loom_testbench_trial_plan_t trial = {
      .trial_count = 1,
      .action = {.kind = LOOM_TESTBENCH_SCENARIO_ACTION_INVOKE},
  };
  const loom_testbench_scenario_plan_t scenario = {
      .name = IREE_SV("subject"),
      .configuration_count = 1,
      .trials = &trial,
      .trial_count = 1,
  };
  std::array<std::string, 128> names;
  std::array<loom_testbench_benchmark_plan_t, 128> benchmarks = {};
  for (size_t i = 0; i < benchmarks.size(); ++i) {
    names[i] = "benchmark_" + std::to_string(benchmarks.size() - 1 - i);
    benchmarks[i] = {
        .name = iree_make_cstring_view(names[i].c_str()),
        .case_index = LOOM_TESTBENCH_CASE_INDEX_INVALID,
        .scenario_index = 0,
        .cartesian_sample_count = 1,
        .sample_count = 1,
    };
  }
  const loom_testbench_module_plan_t module_plan = {
      .scenarios = &scenario,
      .scenario_count = 1,
      .benchmarks = benchmarks.data(),
      .benchmark_count = benchmarks.size(),
  };
  iree_benchmark_loom_options_t options = {};
  iree_benchmark_loom_options_initialize(&options);
  // Exercise direct and indexed lookup, and repeated requests after all
  // names have been selected once. Request order differs from both module
  // and lexical order.
  for (size_t selection_count : {2u, 8u, 9u, 256u}) {
    SCOPED_TRACE(selection_count);
    std::string compare;
    for (size_t i = 0; i < selection_count; ++i) {
      if (i != 0) {
        compare += ',';
      }
      compare += " @" + names[(i * 37) % names.size()] + " ";
    }
    options.compare = iree_make_cstring_view(compare.c_str());
    iree_benchmark_loom_work_plan_t plan = {};
    IREE_ASSERT_OK(iree_benchmark_loom_work_plan_initialize(
        &module_plan, &options, iree_allocator_system(), &plan));
    ASSERT_EQ(plan.selected_benchmark_count, selection_count);
    ASSERT_EQ(plan.logical_sample_count, selection_count);
    ASSERT_EQ(plan.work_item_count, 1u);
    EXPECT_EQ(plan.work_items[0].representative_selection_index, 0u);
    for (size_t i = 0; i < selection_count; ++i) {
      const auto& selection = plan.selected_benchmarks[i];
      const size_t expected_index = (i * 37) % benchmarks.size();
      EXPECT_EQ(selection.benchmark_index, expected_index);
      EXPECT_EQ(selection.benchmark_plan, &benchmarks[expected_index]);
      EXPECT_EQ(selection.identity.candidate_index, i);
      const std::string candidate_id = "c" + std::to_string(i);
      EXPECT_TRUE(
          iree_string_view_equal(selection.identity.candidate_id,
                                 iree_make_cstring_view(candidate_id.c_str())));
      EXPECT_EQ(plan.logical_samples[i].selection_index, i);
      EXPECT_EQ(plan.logical_samples[i].work_item_index, 0u);
    }
    iree_benchmark_loom_work_plan_deinitialize(&plan);

    // Parsing and name failures retain request order even after indexing.
    for (bool missing_first : {false, true}) {
      const std::string invalid =
          compare + (missing_first ? ",@absent,@" : ",@,@absent");
      options.compare = iree_make_cstring_view(invalid.c_str());
      iree::Status status = iree_benchmark_loom_work_plan_initialize(
          &module_plan, &options, iree_allocator_system(), &plan);
      EXPECT_THAT(status,
                  StatusIs(missing_first ? iree::StatusCode::kNotFound
                                         : iree::StatusCode::kInvalidArgument));
      EXPECT_THAT(status.ToString(),
                  HasSubstr(missing_first ? "'@absent' was not found"
                                          : "empty benchmark name"));
      EXPECT_EQ(plan.selected_benchmarks, nullptr);
      EXPECT_EQ(plan.selected_benchmark_count, 0u);
    }
  }
}

TEST(BenchmarkWorkPlanApiTest, PreservesScenarioOrderAcrossAliasesAndDomains) {
  // Unequal domains exercise both trial and configuration transitions.
  // Reverse record order exercises sorted grouping without making sorted order
  // the execution order.
  std::array<loom_testbench_trial_plan_t, 4> trials = {};
  for (size_t i = 0; i < trials.size(); ++i) {
    trials[i].trial_count = std::array<size_t, 4>{2, 1, 2, 1}[i];
    trials[i].action.kind = LOOM_TESTBENCH_SCENARIO_ACTION_INVOKE;
  }
  constexpr size_t kRecordCount = 64;
  constexpr size_t kAliasCount = 3;
  constexpr size_t kSampleCount = 18;
  std::vector<loom_testbench_scenario_plan_t> scenarios(kRecordCount);
  std::vector<loom_testbench_benchmark_plan_t> benchmarks(kRecordCount *
                                                          kAliasCount);
  std::vector<std::string> names(benchmarks.size());
  for (auto& scenario : scenarios) {
    scenario.configuration_count = 3;
    scenario.trials = trials.data();
    scenario.trial_count = trials.size();
  }
  for (size_t i = 0; i < benchmarks.size(); ++i) {
    names[i] = "benchmark_" + std::to_string(i);
    benchmarks[i].name = iree_make_cstring_view(names[i].c_str());
    benchmarks[i].case_index = LOOM_TESTBENCH_CASE_INDEX_INVALID;
    benchmarks[i].scenario_index = kRecordCount - 1 - i % kRecordCount;
    benchmarks[i].sample_count = kSampleCount;
    benchmarks[i].cartesian_sample_count = kSampleCount;
  }
  loom_testbench_module_plan_t module_plan = {
      .scenarios = scenarios.data(),
      .scenario_count = scenarios.size(),
      .benchmarks = benchmarks.data(),
      .benchmark_count = benchmarks.size(),
  };
  iree_benchmark_loom_options_t options = {};
  iree_benchmark_loom_options_initialize(&options);
  options.batch_size = 7;
  options.warmup_iterations = 3;
  options.iterations = 5;

  for (int sample_ordinal : {-1, 5, 6, 17}) {
    SCOPED_TRACE(sample_ordinal);
    options.sample_ordinal = sample_ordinal;
    iree_benchmark_loom_work_plan_t plan = {};
    IREE_ASSERT_OK(iree_benchmark_loom_work_plan_initialize(
        &module_plan, &options, iree_allocator_system(), &plan));
    const size_t selected_samples = sample_ordinal < 0 ? kSampleCount : 1;
    ASSERT_EQ(plan.work_item_count, kRecordCount * selected_samples);
    ASSERT_EQ(plan.logical_sample_count, benchmarks.size() * selected_samples);
    EXPECT_EQ(plan.hal_compile_item_count, 0u);
    for (size_t i = 0; i < plan.logical_sample_count; ++i) {
      const auto& logical = plan.logical_samples[i];
      const size_t selection_index = i / selected_samples;
      const size_t flat_ordinal =
          sample_ordinal < 0 ? i % selected_samples : sample_ordinal;
      const size_t owner_selection = selection_index % kRecordCount;
      const size_t work_index =
          owner_selection * selected_samples + i % selected_samples;
      ASSERT_EQ(logical.work_item_index, work_index);
      EXPECT_EQ(logical.selection_index, selection_index);
      const auto& work = plan.work_items[work_index];
      EXPECT_EQ(work.work_item_index, work_index);
      EXPECT_EQ(work.representative_selection_index, owner_selection);
      EXPECT_EQ(work.hal_compile_item_index,
                IREE_BENCHMARK_LOOM_WORK_PLAN_INDEX_INVALID);
      const auto coordinate =
          loom_testbench_benchmark_sample_scenario_coordinate(
              &scenarios[benchmarks[selection_index].scenario_index],
              &benchmarks[selection_index], flat_ordinal);
      EXPECT_EQ(logical.scenario_coordinate.configuration_ordinal,
                coordinate.configuration_ordinal);
      EXPECT_EQ(logical.scenario_coordinate.trial_index,
                coordinate.trial_index);
      EXPECT_EQ(logical.scenario_coordinate.trial_ordinal,
                coordinate.trial_ordinal);
      EXPECT_EQ(work.scenario_coordinate.configuration_ordinal,
                coordinate.configuration_ordinal);
      EXPECT_EQ(work.scenario_coordinate.trial_index, coordinate.trial_index);
      EXPECT_EQ(work.scenario_coordinate.trial_ordinal,
                coordinate.trial_ordinal);
    }
    iree_benchmark_loom_work_plan_deinitialize(&plan);
  }
}

TEST(BenchmarkWorkPlanApiTest, CasePrefixIdentityMatchesEnumeratedSamples) {
  // Enumerate every pinned/unpinned assignment over a small mixed-radix
  // domain. Caps hide later dimensions; size-one dimensions have no observable
  // variation. The decoder supplies an independent expanded-sequence oracle.
  constexpr std::array<size_t, 4> kRadices = {2, 1, 3, 2};
  constexpr size_t kAssignmentCount = 3 * 2 * 4 * 3;
  std::array<loom_testbench_parameter_plan_t, kRadices.size()> parameters = {};
  for (size_t i = 0; i < parameters.size(); ++i) {
    parameters[i].sample_count = kRadices[i];
  }
  loom_testbench_case_plan_t case_plan = {
      .parameters = parameters.data(),
      .parameter_count = parameters.size(),
      .kernel_launch_count = 1,
      .cartesian_sample_count = 12,
      .sample_count = 12,
  };
  std::array<std::array<iree_host_size_t, kRadices.size()>, kAssignmentCount>
      assignments = {};
  std::array<std::string, kAssignmentCount> names;
  for (size_t i = 0; i < kAssignmentCount; ++i) {
    names[i] = "assignment_" + std::to_string(i);
    size_t code = i;
    for (size_t axis = 0; axis < kRadices.size(); ++axis) {
      const size_t digit = code % (kRadices[axis] + 1);
      code /= kRadices[axis] + 1;
      assignments[i][axis] = digit == kRadices[axis]
                                 ? LOOM_TESTBENCH_PARAMETER_SAMPLE_ORDINAL_ALL
                                 : digit;
    }
  }

  for (size_t cap : {1u, 2u, 3u, 5u, 12u}) {
    for (int window : {-1, 0, 1, 4}) {
      SCOPED_TRACE(::testing::Message()
                   << "cap=" << cap << " window=" << window);
      std::vector<loom_testbench_benchmark_plan_t> benchmarks;
      for (size_t i = 0; i < kAssignmentCount; ++i) {
        size_t sample_count = 1;
        for (size_t axis = 0; axis < kRadices.size(); ++axis) {
          if (assignments[i][axis] ==
              LOOM_TESTBENCH_PARAMETER_SAMPLE_ORDINAL_ALL) {
            sample_count *= kRadices[axis];
          }
        }
        loom_testbench_benchmark_plan_t benchmark = {
            .name = iree_make_cstring_view(names[i].c_str()),
            .case_index = 0,
            .scenario_index = LOOM_TESTBENCH_SCENARIO_INDEX_INVALID,
            .parameter_sample_ordinals = assignments[i].data(),
            .parameter_sample_ordinal_count = kRadices.size(),
            .cartesian_sample_count = sample_count,
            .sample_count = std::min(cap, sample_count),
            .sample_count_truncated = cap < sample_count,
        };
        if (window < 0 ||
            static_cast<size_t>(window) < benchmark.sample_count) {
          benchmarks.push_back(benchmark);
        }
      }
      std::reverse(benchmarks.begin(), benchmarks.end());
      loom_testbench_module_plan_t module_plan = {
          .cases = &case_plan,
          .case_count = 1,
          .benchmarks = benchmarks.data(),
          .benchmark_count = benchmarks.size(),
      };
      iree_benchmark_loom_options_t options = {};
      iree_benchmark_loom_options_initialize(&options);
      options.sample_ordinal = window;
      iree_benchmark_loom_work_plan_t plan = {};
      IREE_ASSERT_OK(iree_benchmark_loom_work_plan_initialize(
          &module_plan, &options, iree_allocator_system(), &plan));
      ASSERT_EQ(plan.logical_sample_count, benchmarks.size());
      EXPECT_EQ(plan.hal_compile_item_count, benchmarks.empty() ? 0u : 1u);

      std::map<std::vector<size_t>, size_t> owners;
      for (size_t i = 0; i < benchmarks.size(); ++i) {
        const auto& benchmark = benchmarks[i];
        const size_t begin = window < 0 ? 0 : window;
        const size_t end = window < 0 ? benchmark.sample_count : begin + 1;
        std::vector<size_t> sequence = {begin, end};
        for (size_t sample = begin; sample < end; ++sample) {
          iree_host_size_t ordinal = 0;
          ASSERT_TRUE(loom_testbench_case_sample_flatten(
              &case_plan,
              loom_testbench_benchmark_case_sample(&benchmark, sample),
              &ordinal));
          sequence.push_back(ordinal);
        }
        auto [owner, inserted] = owners.emplace(sequence, owners.size());
        ASSERT_LT(owner->second, plan.work_item_count);
        EXPECT_EQ(plan.logical_samples[i].work_item_index, owner->second);
        const auto& work = plan.work_items[owner->second];
        if (inserted) {
          EXPECT_EQ(work.representative_selection_index, i);
        }
        EXPECT_EQ(work.begin_benchmark_sample, begin);
        EXPECT_EQ(work.end_benchmark_sample, end);
        EXPECT_EQ(work.hal_compile_item_index, 0u);
      }
      EXPECT_EQ(plan.work_item_count, owners.size());
      size_t execution_sample_count = 0;
      for (size_t i = 0; i < plan.work_item_count; ++i) {
        const auto& work = plan.work_items[i];
        EXPECT_EQ(work.execution_sample_base, execution_sample_count);
        execution_sample_count +=
            work.end_benchmark_sample - work.begin_benchmark_sample;
      }
      if (plan.hal_compile_item_count != 0) {
        EXPECT_EQ(plan.hal_compile_items[0].execution_sample_count,
                  execution_sample_count);
      }
      iree_benchmark_loom_work_plan_deinitialize(&plan);
    }
  }
}

TEST(BenchmarkWorkPlanApiTest, PreservesCaseCompileAndDispatchOwnerOrder) {
  std::array<loom_testbench_parameter_plan_t, 2> parameters = {};
  parameters[0].sample_count = 3;
  parameters[1].sample_count = 2;
  std::array<loom_testbench_case_plan_t, 3> cases = {};
  for (auto& case_plan : cases) {
    case_plan.parameters = parameters.data();
    case_plan.parameter_count = parameters.size();
    case_plan.sample_count = 6;
    case_plan.cartesian_sample_count = 6;
  }
  const iree_host_size_t pinned[] = {
      LOOM_TESTBENCH_PARAMETER_SAMPLE_ORDINAL_ALL, 1};
  std::array<loom_testbench_benchmark_plan_t, 5> benchmarks = {};
  const size_t case_indices[] = {2, 0, 2, 1, 0};
  const char* names[] = {"last_pinned", "first", "last", "middle",
                         "first_pinned"};
  for (size_t i = 0; i < benchmarks.size(); ++i) {
    benchmarks[i].name = iree_make_cstring_view(names[i]);
    benchmarks[i].case_index = case_indices[i];
    benchmarks[i].scenario_index = LOOM_TESTBENCH_SCENARIO_INDEX_INVALID;
    benchmarks[i].sample_count = i == 0 || i == 4 ? 3 : 6;
    benchmarks[i].cartesian_sample_count = benchmarks[i].sample_count;
    if (i == 0 || i == 4) {
      benchmarks[i].parameter_sample_ordinals = pinned;
      benchmarks[i].parameter_sample_ordinal_count = 2;
    }
  }
  loom_testbench_module_plan_t module_plan = {
      .cases = cases.data(),
      .case_count = cases.size(),
      .benchmarks = benchmarks.data(),
      .benchmark_count = benchmarks.size(),
  };
  iree_benchmark_loom_options_t options = {};
  iree_benchmark_loom_options_initialize(&options);
  options.measure = IREE_SV("dispatch_complete");
  iree_benchmark_loom_work_plan_t plan = {};
  IREE_ASSERT_OK(iree_benchmark_loom_work_plan_initialize(
      &module_plan, &options, iree_allocator_system(), &plan));
  ASSERT_EQ(plan.hal_compile_item_count, 3u);
  const size_t compile_owners[] = {0, 1, 3};
  const size_t case_compile_indices[] = {1, 2, 0};
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(plan.hal_compile_items[i].compile_item_index, i);
    EXPECT_EQ(plan.hal_compile_items[i].representative_selection_index,
              compile_owners[i]);
  }
  ASSERT_EQ(plan.logical_sample_count, 24u);
  ASSERT_EQ(plan.work_item_count, 18u);
  std::map<std::pair<size_t, size_t>, size_t> owners;
  for (size_t i = 0; i < plan.logical_sample_count; ++i) {
    const auto& logical = plan.logical_samples[i];
    const size_t selection = logical.selection_index;
    const size_t case_index = case_indices[selection];
    iree_host_size_t ordinal = 0;
    ASSERT_TRUE(loom_testbench_case_sample_flatten(
        &cases[case_index],
        loom_testbench_benchmark_case_sample(&benchmarks[selection],
                                             logical.begin_benchmark_sample),
        &ordinal));
    auto [owner, inserted] =
        owners.emplace(std::make_pair(case_index, ordinal), owners.size());
    ASSERT_EQ(logical.work_item_index, owner->second);
    const auto& work = plan.work_items[owner->second];
    EXPECT_EQ(work.hal_compile_item_index, case_compile_indices[case_index]);
    if (inserted) {
      EXPECT_EQ(work.representative_selection_index, selection);
      EXPECT_EQ(work.begin_benchmark_sample, logical.begin_benchmark_sample);
    }
  }
  iree_benchmark_loom_work_plan_deinitialize(&plan);
}

class BenchmarkWorkPlanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &plan_arena_);

    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_CHECK, loom_check_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    for (loom_module_t* module : owned_modules_) {
      loom_module_free(module);
    }
    owned_modules_.clear();
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

  loom_module_t* ParseModule(const char* source) {
    loom_text_parse_options_t options = {.max_errors = 20};
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_text_parse(iree_make_cstring_view(source),
                                   IREE_SV("work_plan_test.loom"), &context_,
                                   &block_pool_, &options, &module));
    EXPECT_NE(module, nullptr);
    return module;
  }

  loom_testbench_module_plan_t PlanModuleAllowingIssues(const char* source) {
    loom_module_t* module = ParseModule(source);
    owned_modules_.push_back(module);
    loom_testbench_module_plan_t module_plan = {};
    IREE_EXPECT_OK(loom_testbench_plan_module(module, nullptr, &plan_arena_,
                                              &module_plan));
    return module_plan;
  }

  loom_testbench_module_plan_t PlanModule(const char* source) {
    loom_testbench_module_plan_t module_plan = PlanModuleAllowingIssues(source);
    EXPECT_EQ(module_plan.issue_count, 0u);
    return module_plan;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t plan_arena_;
  loom_context_t context_;
  std::vector<loom_module_t*> owned_modules_;
};

TEST_F(BenchmarkWorkPlanTest, DeduplicatesPinnedDispatchSamples) {
  const loom_testbench_module_plan_t module_plan = PlanModule(R"(
check.case @mlp {
  %rows = check.param.choice values([2, 3584]) name("rows") : index
  check.return
}

check.benchmark<@mlp> @mlp_sweep
check.benchmark<@mlp> @decode {rows = 2}
check.benchmark<@mlp> @full {rows = 3584}
)");

  iree_benchmark_loom_options_t options = {};
  iree_benchmark_loom_options_initialize(&options);
  options.measure = IREE_SV("dispatch_complete");

  iree_benchmark_loom_work_plan_t work_plan = {};
  IREE_ASSERT_OK(iree_benchmark_loom_work_plan_initialize(
      &module_plan, &options, iree_allocator_system(), &work_plan));

  EXPECT_EQ(work_plan.selected_benchmark_count, 3u);
  EXPECT_EQ(work_plan.logical_sample_count, 4u);
  EXPECT_EQ(work_plan.hal_compile_item_count, 1u);
  EXPECT_EQ(work_plan.work_item_count, 2u);

  EXPECT_EQ(work_plan.selected_benchmarks[0].benchmark_index, 0u);
  EXPECT_EQ(work_plan.selected_benchmarks[0].begin_benchmark_sample, 0u);
  EXPECT_EQ(work_plan.selected_benchmarks[0].end_benchmark_sample, 2u);
  EXPECT_EQ(work_plan.selected_benchmarks[1].benchmark_index, 1u);
  EXPECT_EQ(work_plan.selected_benchmarks[1].begin_benchmark_sample, 0u);
  EXPECT_EQ(work_plan.selected_benchmarks[1].end_benchmark_sample, 1u);
  EXPECT_EQ(work_plan.selected_benchmarks[2].benchmark_index, 2u);
  EXPECT_EQ(work_plan.selected_benchmarks[2].begin_benchmark_sample, 0u);
  EXPECT_EQ(work_plan.selected_benchmarks[2].end_benchmark_sample, 1u);

  EXPECT_EQ(work_plan.logical_samples[0].selection_index, 0u);
  EXPECT_EQ(work_plan.logical_samples[0].work_item_index, 0u);
  EXPECT_EQ(work_plan.logical_samples[1].selection_index, 0u);
  EXPECT_EQ(work_plan.logical_samples[1].work_item_index, 1u);
  EXPECT_EQ(work_plan.logical_samples[2].selection_index, 1u);
  EXPECT_EQ(work_plan.logical_samples[2].work_item_index, 0u);
  EXPECT_EQ(work_plan.logical_samples[3].selection_index, 2u);
  EXPECT_EQ(work_plan.logical_samples[3].work_item_index, 1u);

  EXPECT_EQ(work_plan.work_items[0].representative_selection_index, 0u);
  EXPECT_EQ(work_plan.work_items[0].hal_compile_item_index, 0u);
  EXPECT_EQ(work_plan.work_items[1].representative_selection_index, 0u);
  EXPECT_EQ(work_plan.work_items[1].hal_compile_item_index, 0u);

  iree_benchmark_loom_work_plan_deinitialize(&work_plan);
}

TEST_F(BenchmarkWorkPlanTest, DeduplicatesEquivalentCaseEndToEndSamples) {
  const loom_testbench_module_plan_t module_plan = PlanModule(R"(
check.case @sampled {
  %value = check.param.choice values([5, 7]) name("value") : i32
  check.expect.equal actual(%value) expected(%value) : i32
  check.return
}

check.benchmark<@sampled> @all_a
check.benchmark<@sampled> @all_b
check.benchmark<@sampled> @value7_a {value = 7}
check.benchmark<@sampled> @value7_b {value = 7}
)");

  iree_benchmark_loom_options_t options = {};
  iree_benchmark_loom_options_initialize(&options);
  options.measure = IREE_SV("case_end_to_end");

  iree_benchmark_loom_work_plan_t work_plan = {};
  IREE_ASSERT_OK(iree_benchmark_loom_work_plan_initialize(
      &module_plan, &options, iree_allocator_system(), &work_plan));

  EXPECT_EQ(work_plan.selected_benchmark_count, 4u);
  EXPECT_EQ(work_plan.logical_sample_count, 4u);
  EXPECT_EQ(work_plan.hal_compile_item_count, 0u);
  EXPECT_EQ(work_plan.work_item_count, 2u);

  EXPECT_EQ(work_plan.logical_samples[0].selection_index, 0u);
  EXPECT_EQ(work_plan.logical_samples[0].work_item_index, 0u);
  EXPECT_EQ(work_plan.logical_samples[1].selection_index, 1u);
  EXPECT_EQ(work_plan.logical_samples[1].work_item_index, 0u);
  EXPECT_EQ(work_plan.logical_samples[2].selection_index, 2u);
  EXPECT_EQ(work_plan.logical_samples[2].work_item_index, 1u);
  EXPECT_EQ(work_plan.logical_samples[3].selection_index, 3u);
  EXPECT_EQ(work_plan.logical_samples[3].work_item_index, 1u);

  EXPECT_EQ(work_plan.work_items[0].representative_selection_index, 0u);
  EXPECT_EQ(work_plan.work_items[0].begin_benchmark_sample, 0u);
  EXPECT_EQ(work_plan.work_items[0].end_benchmark_sample, 2u);
  EXPECT_EQ(work_plan.work_items[1].representative_selection_index, 2u);
  EXPECT_EQ(work_plan.work_items[1].begin_benchmark_sample, 0u);
  EXPECT_EQ(work_plan.work_items[1].end_benchmark_sample, 1u);

  iree_benchmark_loom_work_plan_deinitialize(&work_plan);
}

TEST_F(BenchmarkWorkPlanTest, AppliesSampleWindowBeforeDedupe) {
  const loom_testbench_module_plan_t module_plan = PlanModule(R"(
check.case @sampled {
  %value = check.param.choice values([5, 7, 11]) name("value") : i32
  check.return
}

check.benchmark<@sampled> @all_a
check.benchmark<@sampled> @all_b
)");

  iree_benchmark_loom_options_t options = {};
  iree_benchmark_loom_options_initialize(&options);
  options.measure = IREE_SV("dispatch_complete");
  options.sample_ordinal = 1;

  iree_benchmark_loom_work_plan_t work_plan = {};
  IREE_ASSERT_OK(iree_benchmark_loom_work_plan_initialize(
      &module_plan, &options, iree_allocator_system(), &work_plan));

  EXPECT_EQ(work_plan.selected_benchmark_count, 2u);
  EXPECT_EQ(work_plan.logical_sample_count, 2u);
  EXPECT_EQ(work_plan.hal_compile_item_count, 1u);
  EXPECT_EQ(work_plan.work_item_count, 1u);

  EXPECT_EQ(work_plan.selected_benchmarks[0].benchmark_index, 0u);
  EXPECT_EQ(work_plan.selected_benchmarks[0].begin_benchmark_sample, 1u);
  EXPECT_EQ(work_plan.selected_benchmarks[0].end_benchmark_sample, 2u);
  EXPECT_EQ(work_plan.selected_benchmarks[1].benchmark_index, 1u);
  EXPECT_EQ(work_plan.selected_benchmarks[1].begin_benchmark_sample, 1u);
  EXPECT_EQ(work_plan.selected_benchmarks[1].end_benchmark_sample, 2u);

  EXPECT_EQ(work_plan.logical_samples[0].selection_index, 0u);
  EXPECT_EQ(work_plan.logical_samples[0].begin_benchmark_sample, 1u);
  EXPECT_EQ(work_plan.logical_samples[0].end_benchmark_sample, 2u);
  EXPECT_EQ(work_plan.logical_samples[0].work_item_index, 0u);
  EXPECT_EQ(work_plan.logical_samples[1].selection_index, 1u);
  EXPECT_EQ(work_plan.logical_samples[1].begin_benchmark_sample, 1u);
  EXPECT_EQ(work_plan.logical_samples[1].end_benchmark_sample, 2u);
  EXPECT_EQ(work_plan.logical_samples[1].work_item_index, 0u);

  EXPECT_EQ(work_plan.work_items[0].begin_benchmark_sample, 1u);
  EXPECT_EQ(work_plan.work_items[0].end_benchmark_sample, 2u);

  iree_benchmark_loom_work_plan_deinitialize(&work_plan);
}

TEST_F(BenchmarkWorkPlanTest, CompareSelectionsShareDuplicatePhysicalWork) {
  const loom_testbench_module_plan_t module_plan = PlanModule(R"(
check.case @mlp {
  %rows = check.param.choice values([2, 3584]) name("rows") : index
  check.return
}

check.benchmark<@mlp> @decode_a {rows = 2}
check.benchmark<@mlp> @decode_b {rows = 2}
)");

  iree_benchmark_loom_options_t options = {};
  iree_benchmark_loom_options_initialize(&options);
  options.measure = IREE_SV("dispatch_complete");
  options.compare = IREE_SV("@decode_b,@decode_a");

  iree_benchmark_loom_work_plan_t work_plan = {};
  IREE_ASSERT_OK(iree_benchmark_loom_work_plan_initialize(
      &module_plan, &options, iree_allocator_system(), &work_plan));

  EXPECT_EQ(work_plan.selected_benchmark_count, 2u);
  EXPECT_EQ(work_plan.logical_sample_count, 2u);
  EXPECT_EQ(work_plan.hal_compile_item_count, 1u);
  EXPECT_EQ(work_plan.work_item_count, 1u);
  EXPECT_EQ(work_plan.selected_benchmarks[0].benchmark_index, 1u);
  EXPECT_EQ(work_plan.selected_benchmarks[0].begin_benchmark_sample, 0u);
  EXPECT_EQ(work_plan.selected_benchmarks[0].end_benchmark_sample, 1u);
  EXPECT_EQ(work_plan.selected_benchmarks[1].benchmark_index, 0u);
  EXPECT_EQ(work_plan.selected_benchmarks[1].begin_benchmark_sample, 0u);
  EXPECT_EQ(work_plan.selected_benchmarks[1].end_benchmark_sample, 1u);
  EXPECT_EQ(work_plan.logical_samples[0].selection_index, 0u);
  EXPECT_EQ(work_plan.logical_samples[0].work_item_index, 0u);
  EXPECT_EQ(work_plan.logical_samples[1].selection_index, 1u);
  EXPECT_EQ(work_plan.logical_samples[1].work_item_index, 0u);

  iree_benchmark_loom_work_plan_deinitialize(&work_plan);
}

TEST_F(BenchmarkWorkPlanTest, CompareSampleFlagNarrowsBeforeDedupe) {
  const loom_testbench_module_plan_t module_plan = PlanModule(R"(
check.case @sampled {
  %value = check.param.choice values([5, 7, 11]) name("value") : i32
  check.return
}

check.benchmark<@sampled> @all_a
check.benchmark<@sampled> @all_b
)");

  iree_benchmark_loom_options_t options = {};
  iree_benchmark_loom_options_initialize(&options);
  options.measure = IREE_SV("dispatch_complete");
  options.compare = IREE_SV("@all_a,@all_b");
  options.sample_ordinal = 1;

  iree_benchmark_loom_work_plan_t work_plan = {};
  IREE_ASSERT_OK(iree_benchmark_loom_work_plan_initialize(
      &module_plan, &options, iree_allocator_system(), &work_plan));

  EXPECT_EQ(work_plan.selected_benchmark_count, 2u);
  EXPECT_EQ(work_plan.logical_sample_count, 2u);
  EXPECT_EQ(work_plan.hal_compile_item_count, 1u);
  EXPECT_EQ(work_plan.work_item_count, 1u);
  EXPECT_EQ(work_plan.logical_samples[0].begin_benchmark_sample, 1u);
  EXPECT_EQ(work_plan.logical_samples[1].begin_benchmark_sample, 1u);

  iree_benchmark_loom_work_plan_deinitialize(&work_plan);
}

TEST_F(BenchmarkWorkPlanTest, PlansAndDeduplicatesScenarioTrials) {
  const loom_testbench_module_plan_t module_plan = PlanModule(R"(
test.func @identity(%input: i32) -> (i32) {
  test.yield %input : i32
}

check.scenario @scenario configure[2](%configuration: index, %configuration_entropy: check.entropy) {
  check.trial[3](%trial: index, %entropy: check.entropy) {
    %input = check.literal value(1) : i32
    check.invoke<@identity>(%input) : (i32) -> (i32)
  }
  check.trial[2](%trial: index, %entropy: check.entropy) {
    %input = check.literal value(2) : i32
    check.invoke<@identity>(%input) : (i32) -> (i32)
  }
  check.return
}

check.benchmark<@scenario> @first
check.benchmark<@scenario> @second
)");

  iree_benchmark_loom_options_t options = {};
  iree_benchmark_loom_options_initialize(&options);
  options.selected_case = IREE_SV("scenario");
  options.sample_ordinal = 4;

  iree_benchmark_loom_work_plan_t work_plan = {};
  IREE_ASSERT_OK(iree_benchmark_loom_work_plan_initialize(
      &module_plan, &options, iree_allocator_system(), &work_plan));

  ASSERT_EQ(work_plan.selected_benchmark_count, 2u);
  EXPECT_EQ(work_plan.selected_benchmarks[0].benchmark_index, 0u);
  EXPECT_EQ(work_plan.selected_benchmarks[0].case_plan, nullptr);
  EXPECT_EQ(work_plan.selected_benchmarks[0].scenario_plan,
            &module_plan.scenarios[0]);
  EXPECT_EQ(work_plan.selected_benchmarks[0].begin_benchmark_sample, 4u);
  EXPECT_EQ(work_plan.selected_benchmarks[0].end_benchmark_sample, 5u);
  EXPECT_EQ(work_plan.selected_benchmarks[1].benchmark_index, 1u);
  EXPECT_EQ(work_plan.selected_benchmarks[1].begin_benchmark_sample, 4u);
  EXPECT_EQ(work_plan.selected_benchmarks[1].end_benchmark_sample, 5u);
  EXPECT_EQ(work_plan.selected_benchmarks[0].policy.measure_kind,
            IREE_BENCHMARK_LOOM_MEASURE_DISPATCH_COMPLETE);
  EXPECT_TRUE(
      iree_string_view_equal(work_plan.selected_benchmarks[0].policy.measure,
                             IREE_SV("dispatch_complete")));
  ASSERT_EQ(work_plan.logical_sample_count, 2u);
  EXPECT_EQ(work_plan.hal_compile_item_count, 0u);
  ASSERT_EQ(work_plan.work_item_count, 1u);
  EXPECT_EQ(work_plan.work_items[0].kind,
            IREE_BENCHMARK_LOOM_WORK_ITEM_SCENARIO_TRIAL);
  EXPECT_EQ(work_plan.work_items[0].hal_compile_item_index,
            IREE_BENCHMARK_LOOM_WORK_PLAN_INDEX_INVALID);
  EXPECT_EQ(work_plan.work_items[0].scenario_coordinate.configuration_ordinal,
            0u);
  EXPECT_EQ(work_plan.work_items[0].scenario_coordinate.trial_index, 1u);
  EXPECT_EQ(work_plan.work_items[0].scenario_coordinate.trial_ordinal, 1u);
  EXPECT_EQ(work_plan.logical_samples[0].work_item_index, 0u);
  EXPECT_EQ(work_plan.logical_samples[1].work_item_index, 0u);

  iree_benchmark_loom_work_plan_deinitialize(&work_plan);
}

TEST_F(BenchmarkWorkPlanTest, RejectsCaseEndToEndScenarioMeasurement) {
  const loom_testbench_module_plan_t module_plan = PlanModule(R"(
test.func @identity(%input: i32) -> (i32) {
  test.yield %input : i32
}

check.scenario @scenario {
  check.trial[1](%trial: index, %entropy: check.entropy) {
    %input = check.literal value(1) : i32
    check.invoke<@identity>(%input) : (i32) -> (i32)
  }
  check.return
}

check.benchmark<@scenario> @scenario_latency
)");

  iree_benchmark_loom_options_t options = {};
  iree_benchmark_loom_options_initialize(&options);
  options.measure = IREE_SV("case_end_to_end");

  iree_benchmark_loom_work_plan_t work_plan = {};
  iree::Status status(iree_benchmark_loom_work_plan_initialize(
      &module_plan, &options, iree_allocator_system(), &work_plan));

  EXPECT_THAT(status, StatusIs(iree::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.ToString(),
              HasSubstr("requires measure = \"dispatch_complete\""));
}

TEST_F(BenchmarkWorkPlanTest, RejectsSelectedBenchmarkWithZeroSamples) {
  const loom_testbench_module_plan_t module_plan = PlanModuleAllowingIssues(R"(
check.case @sampled {
  %value = check.param.choice values([5, 7]) name("value") : i32
  check.return
}

check.benchmark<@sampled> @missing_value {value = 9}
)");
  ASSERT_EQ(module_plan.benchmark_count, 1u);
  ASSERT_EQ(module_plan.benchmarks[0].sample_count, 0u);

  iree_benchmark_loom_options_t options = {};
  iree_benchmark_loom_options_initialize(&options);
  options.measure = IREE_SV("dispatch_complete");

  iree_benchmark_loom_work_plan_t work_plan = {};
  iree::Status status(iree_benchmark_loom_work_plan_initialize(
      &module_plan, &options, iree_allocator_system(), &work_plan));

  EXPECT_THAT(status, StatusIs(iree::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.ToString(), HasSubstr("zero executable samples"));
  EXPECT_THAT(status.ToString(), HasSubstr("missing_value"));
  EXPECT_THAT(status.ToString(), HasSubstr("sampled"));
}

TEST_F(BenchmarkWorkPlanTest, AcceptsModuleWithoutBenchmarks) {
  const loom_testbench_module_plan_t module_plan = PlanModule(R"(
check.case @only_case {
  check.return
}
)");

  iree_benchmark_loom_options_t options = {};
  iree_benchmark_loom_options_initialize(&options);

  iree_benchmark_loom_work_plan_t work_plan = {};
  IREE_ASSERT_OK(iree_benchmark_loom_work_plan_initialize(
      &module_plan, &options, iree_allocator_system(), &work_plan));

  EXPECT_EQ(work_plan.selected_benchmark_count, 0u);
  EXPECT_EQ(work_plan.logical_sample_count, 0u);
  EXPECT_EQ(work_plan.hal_compile_item_count, 0u);
  EXPECT_EQ(work_plan.work_item_count, 0u);

  iree_benchmark_loom_work_plan_deinitialize(&work_plan);
}

TEST_F(BenchmarkWorkPlanTest, RejectsMissingBenchmarkSelector) {
  const loom_testbench_module_plan_t module_plan = PlanModule(R"(
check.case @sampled {
  check.return
}

check.benchmark<@sampled> @available
)");

  iree_benchmark_loom_options_t options = {};
  iree_benchmark_loom_options_initialize(&options);
  options.selected_benchmark = IREE_SV("missing");

  iree_benchmark_loom_work_plan_t work_plan = {};
  iree::Status status(iree_benchmark_loom_work_plan_initialize(
      &module_plan, &options, iree_allocator_system(), &work_plan));

  EXPECT_THAT(status, StatusIs(iree::StatusCode::kNotFound));
  EXPECT_THAT(status.ToString(), HasSubstr("no check.benchmark matched"));
  EXPECT_THAT(status.ToString(), HasSubstr("missing"));
}

TEST(BenchmarkComparisonExecutionTest, AbabaSampleCapacityKeepsBaselinePairs) {
  EXPECT_EQ(iree_benchmark_loom_dispatch_comparison_sample_capacity(
                IREE_BENCHMARK_LOOM_INTERLEAVE_ABABA,
                /*candidate_count=*/2, /*candidate_index=*/0,
                /*repetitions=*/3),
            4u);
  EXPECT_EQ(iree_benchmark_loom_dispatch_comparison_sample_capacity(
                IREE_BENCHMARK_LOOM_INTERLEAVE_ABABA,
                /*candidate_count=*/2, /*candidate_index=*/1,
                /*repetitions=*/3),
            3u);
  EXPECT_EQ(iree_benchmark_loom_dispatch_comparison_sample_capacity(
                IREE_BENCHMARK_LOOM_INTERLEAVE_ROUND_ROBIN,
                /*candidate_count=*/4, /*candidate_index=*/0,
                /*repetitions=*/3),
            3u);
  EXPECT_EQ(iree_benchmark_loom_dispatch_comparison_sample_capacity(
                IREE_BENCHMARK_LOOM_INTERLEAVE_ROUND_ROBIN,
                /*candidate_count=*/4, /*candidate_index=*/3,
                /*repetitions=*/3),
            3u);
}

}  // namespace
}  // namespace loom
