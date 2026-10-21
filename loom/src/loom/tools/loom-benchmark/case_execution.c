// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-benchmark/case_execution.h"

#include <inttypes.h>
#include <string.h>

#include "loom/tools/loom-benchmark/timing.h"

static iree_status_t loom_benchmark_prepare_case_executor(
    const loom_testbench_module_plan_t* module_plan,
    iree_host_size_t case_index,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_arena_allocator_t* arena,
    loom_testbench_prepared_case_t* out_prepared_case,
    loom_testbench_case_executor_t* out_executor, bool* out_initialized) {
  *out_initialized = false;
  IREE_RETURN_IF_ERROR(loom_testbench_prepare_case_execution(
      execution_options, module_plan, case_index, arena, out_prepared_case));
  IREE_RETURN_IF_ERROR(loom_testbench_case_executor_initialize(
      out_prepared_case, execution_options, out_executor));
  *out_initialized = true;
  return iree_ok_status();
}

static iree_status_t loom_benchmark_run_case_iteration(
    loom_testbench_case_executor_t* executor,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    iree_host_size_t begin_sample, iree_host_size_t end_sample,
    iree_host_size_t execution_sample_base,
    iree_host_size_t* inout_failed_sample_count) {
  for (iree_host_size_t benchmark_sample_ordinal = begin_sample;
       benchmark_sample_ordinal < end_sample; ++benchmark_sample_ordinal) {
    const loom_testbench_case_sample_t sample =
        loom_testbench_benchmark_case_sample(benchmark_plan,
                                             benchmark_sample_ordinal);
    loom_testbench_case_sample_result_t sample_result = {0};
    IREE_RETURN_IF_ERROR(loom_testbench_run_case_sample(
        executor, sample,
        execution_sample_base + benchmark_sample_ordinal - begin_sample,
        &sample_result));
    if (!sample_result.passed) {
      ++*inout_failed_sample_count;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_benchmark_run_benchmark_iterations(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_execution_options_t* execution_options,
    const loom_benchmark_benchmark_policy_t* policy,
    iree_host_size_t begin_sample, iree_host_size_t end_sample,
    iree_host_size_t execution_sample_base, iree_arena_allocator_t* arena,
    iree_allocator_t host_allocator,
    loom_benchmark_benchmark_result_t* out_result) {
  memset(out_result, 0, sizeof(*out_result));

  out_result->samples_per_iteration = end_sample - begin_sample;
  if (end_sample == begin_sample + 1) {
    out_result->has_benchmark_sample_ordinal = true;
    out_result->benchmark_sample_ordinal = begin_sample;
  }

  iree_duration_t* durations = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, policy->iterations * sizeof(*durations),
      (void**)&durations));

  iree_status_t status = iree_ok_status();
  loom_testbench_prepared_case_t prepared_case = {0};
  loom_testbench_case_executor_t executor = {0};
  bool executor_initialized = false;
  status = loom_benchmark_prepare_case_executor(
      module_plan, benchmark_plan->case_index, execution_options, arena,
      &prepared_case, &executor, &executor_initialized);

  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < policy->warmup_iterations; ++i) {
    status = loom_benchmark_run_case_iteration(
        &executor, benchmark_plan, begin_sample, end_sample,
        execution_sample_base, &out_result->failed_sample_count);
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < policy->iterations; ++i) {
    const iree_time_t start_time_ns = iree_time_now();
    status = loom_benchmark_run_case_iteration(
        &executor, benchmark_plan, begin_sample, end_sample,
        execution_sample_base, &out_result->failed_sample_count);
    const iree_time_t end_time_ns = iree_time_now();
    durations[i] =
        end_time_ns >= start_time_ns ? end_time_ns - start_time_ns : 0;
  }

  if (iree_status_is_ok(status)) {
    out_result->executed = true;
    out_result->passed = out_result->failed_sample_count == 0;
    loom_benchmark_compute_timing_stats(durations, policy->iterations,
                                        &out_result->timing);
  }

  if (executor_initialized) {
    loom_testbench_case_executor_deinitialize(&executor);
  }
  iree_arena_reset(arena);
  iree_allocator_free(host_allocator, durations);
  return status;
}

static iree_status_t loom_benchmark_emit_work_item_sample_aliases(
    const loom_benchmark_run_identity_t* run,
    const loom_testbench_module_plan_t* module_plan,
    const loom_benchmark_work_plan_t* work_plan,
    const loom_benchmark_work_item_t* work_item, iree_host_size_t sample_offset,
    const loom_testbench_case_sample_result_t* sample_result,
    const loom_benchmark_event_sink_t* event_sink) {
  for (iree_host_size_t logical_sample_index =
           work_item->first_logical_sample_index;
       logical_sample_index != LOOM_BENCHMARK_WORK_PLAN_INDEX_INVALID;
       logical_sample_index =
           work_plan->logical_samples[logical_sample_index].next_alias_index) {
    const loom_benchmark_logical_sample_t* logical_sample =
        &work_plan->logical_samples[logical_sample_index];
    const loom_benchmark_selected_benchmark_t* selection =
        &work_plan->selected_benchmarks[logical_sample->selection_index];
    const iree_host_size_t benchmark_sample_ordinal =
        logical_sample->begin_benchmark_sample + sample_offset;
    if (benchmark_sample_ordinal >= logical_sample->end_benchmark_sample) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_benchmark_event_sink_emit_sample(
        event_sink, run, &selection->identity, work_item->work_item_index,
        module_plan->module, selection->benchmark_plan, selection->case_plan,
        benchmark_sample_ordinal, sample_result));
  }
  return iree_ok_status();
}

static bool loom_benchmark_benchmark_result_counts_as_failed(
    const loom_benchmark_benchmark_result_t* benchmark_result) {
  if (iree_string_view_equal(benchmark_result->state, IREE_SV("skipped"))) {
    return false;
  }
  return !benchmark_result->executed || !benchmark_result->passed;
}

iree_status_t loom_benchmark_emit_work_item_result_aliases(
    const loom_benchmark_run_identity_t* run,
    const loom_testbench_module_plan_t* module_plan,
    const loom_benchmark_work_plan_t* work_plan,
    const loom_benchmark_work_item_t* work_item,
    const loom_benchmark_benchmark_result_t* benchmark_result,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count,
    const loom_benchmark_event_sink_t* event_sink,
    iree_host_size_t* inout_failed_benchmark_count) {
  for (iree_host_size_t logical_sample_index =
           work_item->first_logical_sample_index;
       logical_sample_index != LOOM_BENCHMARK_WORK_PLAN_INDEX_INVALID;
       logical_sample_index =
           work_plan->logical_samples[logical_sample_index].next_alias_index) {
    const loom_benchmark_logical_sample_t* logical_sample =
        &work_plan->logical_samples[logical_sample_index];
    const loom_benchmark_selected_benchmark_t* selection =
        &work_plan->selected_benchmarks[logical_sample->selection_index];
    loom_benchmark_benchmark_result_t alias_result = *benchmark_result;
    if (logical_sample->end_benchmark_sample ==
        logical_sample->begin_benchmark_sample + 1) {
      alias_result.has_benchmark_sample_ordinal = true;
      alias_result.benchmark_sample_ordinal =
          logical_sample->begin_benchmark_sample;
    }
    if (selection->scenario_plan != NULL) {
      alias_result.has_scenario_coordinate = true;
      alias_result.scenario_coordinate = logical_sample->scenario_coordinate;
    }
    if (alias_result.samples_per_iteration == 0 &&
        logical_sample->end_benchmark_sample ==
            logical_sample->begin_benchmark_sample + 1) {
      alias_result.samples_per_iteration = 1;
    }
    if (loom_benchmark_benchmark_result_counts_as_failed(&alias_result)) {
      ++*inout_failed_benchmark_count;
    }
    IREE_RETURN_IF_ERROR(loom_benchmark_event_sink_emit_benchmark_result(
        event_sink, run, &selection->identity, work_item, module_plan->module,
        selection->benchmark_plan, selection->case_plan,
        selection->scenario_plan, &selection->policy, &alias_result,
        correctness_sample_count, correctness_failed_sample_count));
    IREE_RETURN_IF_ERROR(loom_benchmark_event_sink_emit_profile_replay(
        event_sink, run, &selection->identity, work_item->work_item_index,
        module_plan->module, selection->benchmark_plan, selection->case_plan,
        selection->scenario_plan, &selection->policy, &alias_result));
  }
  return iree_ok_status();
}

iree_status_t loom_benchmark_run_work_item_correctness_range(
    const loom_benchmark_run_identity_t* run,
    const loom_testbench_module_plan_t* module_plan,
    const loom_benchmark_work_plan_t* work_plan,
    const loom_benchmark_work_item_t* work_item,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_arena_allocator_t* arena,
    loom_benchmark_case_sample_observer_t sample_observer,
    const loom_benchmark_event_sink_t* event_sink,
    iree_host_size_t* out_sample_count,
    iree_host_size_t* out_failed_sample_count) {
  *out_sample_count = 0;
  *out_failed_sample_count = 0;
  const loom_benchmark_selected_benchmark_t* selection =
      &work_plan
           ->selected_benchmarks[work_item->representative_selection_index];
  const loom_testbench_benchmark_plan_t* benchmark_plan =
      selection->benchmark_plan;

  loom_testbench_prepared_case_t prepared_case = {0};
  loom_testbench_case_executor_t executor = {0};
  bool executor_initialized = false;
  iree_status_t status = loom_benchmark_prepare_case_executor(
      module_plan, benchmark_plan->case_index, execution_options, arena,
      &prepared_case, &executor, &executor_initialized);

  for (iree_host_size_t benchmark_sample_ordinal =
           work_item->begin_benchmark_sample;
       iree_status_is_ok(status) &&
       benchmark_sample_ordinal < work_item->end_benchmark_sample;
       ++benchmark_sample_ordinal) {
    const iree_host_size_t sample_offset =
        benchmark_sample_ordinal - work_item->begin_benchmark_sample;
    const loom_testbench_case_sample_t sample =
        loom_testbench_benchmark_case_sample(benchmark_plan,
                                             benchmark_sample_ordinal);
    loom_testbench_case_sample_result_t sample_result = {0};
    status = loom_testbench_run_case_sample(
        &executor, sample, work_item->execution_sample_base + sample_offset,
        &sample_result);
    if (iree_status_is_ok(status) && sample_observer.fn != NULL) {
      sample_observer.fn(sample_observer.user_data, sample);
    }
    if (iree_status_is_ok(status)) {
      status = loom_benchmark_emit_work_item_sample_aliases(
          run, module_plan, work_plan, work_item, sample_offset, &sample_result,
          event_sink);
    }
    if (iree_status_is_ok(status)) {
      ++*out_sample_count;
      if (!sample_result.passed) {
        ++*out_failed_sample_count;
      }
    }
  }

  if (executor_initialized) {
    loom_testbench_case_executor_deinitialize(&executor);
  }
  iree_arena_reset(arena);
  return status;
}

iree_status_t loom_benchmark_measure_case_end_to_end_work_item(
    const loom_testbench_module_plan_t* module_plan,
    const loom_benchmark_work_plan_t* work_plan,
    const loom_benchmark_work_item_t* work_item,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_arena_allocator_t* execution_arena, iree_allocator_t allocator,
    loom_benchmark_benchmark_result_t* out_result) {
  const loom_benchmark_selected_benchmark_t* selection =
      &work_plan
           ->selected_benchmarks[work_item->representative_selection_index];
  return loom_benchmark_run_benchmark_iterations(
      module_plan, selection->benchmark_plan, execution_options,
      &selection->policy, work_item->begin_benchmark_sample,
      work_item->end_benchmark_sample, work_item->execution_sample_base,
      execution_arena, allocator, out_result);
}

iree_status_t loom_benchmark_run_case_end_to_end_work_item(
    const loom_benchmark_run_identity_t* run,
    const loom_testbench_module_plan_t* module_plan,
    const loom_benchmark_work_plan_t* work_plan,
    const loom_benchmark_work_item_t* work_item,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_arena_allocator_t* execution_arena, iree_allocator_t allocator,
    const loom_benchmark_event_sink_t* event_sink,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count) {
  loom_testbench_case_execution_options_t benchmark_execution_options =
      *execution_options;

  iree_host_size_t correctness_sample_count = 0;
  iree_host_size_t correctness_failed_sample_count = 0;
  iree_status_t status = loom_benchmark_run_work_item_correctness_range(
      run, module_plan, work_plan, work_item, &benchmark_execution_options,
      execution_arena, (loom_benchmark_case_sample_observer_t){0}, event_sink,
      &correctness_sample_count, &correctness_failed_sample_count);
  if (iree_status_is_ok(status)) {
    *inout_correctness_sample_count += correctness_sample_count;
    *inout_correctness_failed_sample_count += correctness_failed_sample_count;
  }

  loom_benchmark_benchmark_result_t benchmark_result = {0};
  if (iree_status_is_ok(status) && correctness_failed_sample_count == 0) {
    status = loom_benchmark_measure_case_end_to_end_work_item(
        module_plan, work_plan, work_item, &benchmark_execution_options,
        execution_arena, allocator, &benchmark_result);
  }
  if (iree_status_is_ok(status) && correctness_failed_sample_count != 0) {
    benchmark_result = (loom_benchmark_benchmark_result_t){
        .executed = false,
        .passed = false,
        .samples_per_iteration = correctness_sample_count,
        .failed_sample_count = correctness_failed_sample_count,
    };
  }
  if (iree_status_is_ok(status)) {
    status = loom_benchmark_emit_work_item_result_aliases(
        run, module_plan, work_plan, work_item, &benchmark_result,
        correctness_sample_count, correctness_failed_sample_count, event_sink,
        inout_failed_benchmark_count);
  }
  return status;
}
