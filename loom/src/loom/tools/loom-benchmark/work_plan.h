// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmark selection and logical-to-physical work planning.

#ifndef LOOM_TOOLS_LOOM_BENCHMARK_WORK_PLAN_H_
#define LOOM_TOOLS_LOOM_BENCHMARK_WORK_PLAN_H_

#include "iree/base/api.h"
#include "loom/tools/loom-benchmark/model.h"
#include "loom/tools/loom-benchmark/options.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_benchmark_work_item_kind_e {
  // Invalid or uninitialized work item.
  LOOM_BENCHMARK_WORK_ITEM_NONE = 0,
  // A case_end_to_end benchmark window over one selected benchmark.
  LOOM_BENCHMARK_WORK_ITEM_CASE_END_TO_END = 1,
  // One dispatch_complete measurement for one concrete case sample.
  LOOM_BENCHMARK_WORK_ITEM_DISPATCH_SAMPLE = 2,
  // One target-only measurement for one concrete scenario trial.
  LOOM_BENCHMARK_WORK_ITEM_SCENARIO_TRIAL = 3,
} loom_benchmark_work_item_kind_t;

#define LOOM_BENCHMARK_WORK_PLAN_INDEX_INVALID LOOM_BENCHMARK_INDEX_INVALID

typedef struct loom_benchmark_hal_compile_item_t {
  // Stable compile item ordinal within the plan.
  iree_host_size_t compile_item_index;
  // Representative selected benchmark ordinal used for compilation.
  iree_host_size_t representative_selection_index;
  // Dense execution slots retained across the case's physical work windows.
  iree_host_size_t execution_sample_count;
} loom_benchmark_hal_compile_item_t;

typedef struct loom_benchmark_logical_sample_t {
  // Selected benchmark ordinal that owns this logical report sample.
  iree_host_size_t selection_index;
  // First benchmark-local sample ordinal covered by this logical sample.
  iree_host_size_t begin_benchmark_sample;
  // One-past-end benchmark-local sample ordinal covered by this logical sample.
  iree_host_size_t end_benchmark_sample;
  // Scenario coordinate selected by this logical sample. Meaningful only when
  // the selected benchmark references a scenario.
  loom_testbench_scenario_sample_coordinate_t scenario_coordinate;
  // Deduplicated physical work item index satisfying this logical sample.
  iree_host_size_t work_item_index;
  // Next logical sample sharing the physical work item, in report order, or
  // INDEX_INVALID for the last alias.
  iree_host_size_t next_alias_index;
} loom_benchmark_logical_sample_t;

typedef struct loom_benchmark_work_item_t {
  // Work kind and payload discriminator.
  loom_benchmark_work_item_kind_t kind;
  // Stable work item ordinal within the plan.
  iree_host_size_t work_item_index;
  // Representative selected benchmark ordinal used for execution.
  iree_host_size_t representative_selection_index;
  // First logical sample in the nonempty, report-ordered alias chain.
  iree_host_size_t first_logical_sample_index;
  // HAL compile item index, or INDEX_INVALID when compilation is owned by the
  // work item or the subject has no actual call.
  iree_host_size_t hal_compile_item_index;
  // First dense sequence slot in the HAL compile item's execution storage.
  iree_host_size_t execution_sample_base;
  // First benchmark-local sample ordinal covered by this work item.
  iree_host_size_t begin_benchmark_sample;
  // One-past-end benchmark-local sample ordinal covered by this work item.
  iree_host_size_t end_benchmark_sample;
  // Scenario coordinate measured by this work item. Meaningful only for
  // SCENARIO_TRIAL work.
  loom_testbench_scenario_sample_coordinate_t scenario_coordinate;
} loom_benchmark_work_item_t;

typedef struct loom_benchmark_work_plan_t {
  // Host allocator used for arrays owned by this plan.
  iree_allocator_t host_allocator;
  // Original selected issues borrowed from the module plan, in module issue
  // order. Nonempty diagnostics exclude all executable selections and work.
  const loom_testbench_issue_t** planning_issues;
  // Number of entries in |planning_issues|, each present once across aliases.
  iree_host_size_t planning_issue_count;
  // Selected benchmark records in execution/report order.
  loom_benchmark_selected_benchmark_t* selected_benchmarks;
  // Number of populated selected benchmark records.
  iree_host_size_t selected_benchmark_count;
  // Logical report samples in selected benchmark order.
  loom_benchmark_logical_sample_t* logical_samples;
  // Number of populated logical report samples.
  iree_host_size_t logical_sample_count;
  // Deduplicated HAL candidate compile items needed by physical work items.
  loom_benchmark_hal_compile_item_t* hal_compile_items;
  // Number of populated HAL compile items.
  iree_host_size_t hal_compile_item_count;
  // Deduplicated physical work items needed by |logical_samples|.
  loom_benchmark_work_item_t* work_items;
  // Number of populated physical work items.
  iree_host_size_t work_item_count;
} loom_benchmark_work_plan_t;

// Builds a selected benchmark and deduplicated physical work plan.
// Logical samples retain selection order. Shared work and compile products use
// the first selected representative, preserving its execution order.
// Authored planning errors return OK with |planning_issues| and no executable
// work. CLI selection/policy errors and allocation failures return a status.
// The module plan must outlive this plan.
iree_status_t loom_benchmark_work_plan_initialize(
    const loom_testbench_module_plan_t* module_plan,
    const loom_benchmark_options_t* options, iree_allocator_t allocator,
    loom_benchmark_work_plan_t* out_plan);

// Releases arrays owned by |plan|.
void loom_benchmark_work_plan_deinitialize(loom_benchmark_work_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_BENCHMARK_WORK_PLAN_H_
