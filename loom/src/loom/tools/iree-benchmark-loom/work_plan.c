// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/work_plan.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "loom/tools/iree-benchmark-loom/module_query.h"
#include "loom/util/adaptive_sort.h"

static iree_status_t iree_benchmark_loom_duration_ms_to_ns(
    int64_t duration_ms, iree_duration_t* out_duration_ns) {
  if (duration_ms < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duration must be non-negative; got %" PRIi64,
                            duration_ms);
  }
  if (duration_ms > INT64_MAX / 1000000) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "duration %" PRIi64 "ms overflows nanoseconds",
                            duration_ms);
  }
  *out_duration_ns = (iree_duration_t)(duration_ms * 1000000);
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_policy_from_benchmark(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const iree_benchmark_loom_options_t* options,
    iree_benchmark_loom_benchmark_policy_t* out_policy) {
  memset(out_policy, 0, sizeof(*out_policy));
  loom_run_hal_benchmark_options_initialize(&out_policy->hal_options);

  const bool is_scenario =
      benchmark_plan->scenario_index < module_plan->scenario_count;
  iree_string_view_t measure = options->measure;
  iree_benchmark_loom_measure_t measure_kind =
      IREE_BENCHMARK_LOOM_MEASURE_CASE_END_TO_END;
  if (iree_string_view_equal(measure, IREE_SV("auto"))) {
    measure =
        is_scenario ? IREE_SV("dispatch_complete") : IREE_SV("case_end_to_end");
    measure_kind = is_scenario ? IREE_BENCHMARK_LOOM_MEASURE_DISPATCH_COMPLETE
                               : IREE_BENCHMARK_LOOM_MEASURE_CASE_END_TO_END;
  } else if (iree_string_view_equal(measure, IREE_SV("case_end_to_end")) ||
             iree_string_view_equal(measure, IREE_SV("end_to_end"))) {
    measure = IREE_SV("case_end_to_end");
    measure_kind = IREE_BENCHMARK_LOOM_MEASURE_CASE_END_TO_END;
  } else if (iree_string_view_equal(measure, IREE_SV("dispatch_complete"))) {
    measure_kind = IREE_BENCHMARK_LOOM_MEASURE_DISPATCH_COMPLETE;
  } else {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "benchmark `%.*s` requested unsupported V0 "
                            "measure '%.*s'",
                            (int)benchmark_plan->name.size,
                            benchmark_plan->name.data, (int)measure.size,
                            measure.data);
  }
  if (is_scenario &&
      measure_kind != IREE_BENCHMARK_LOOM_MEASURE_DISPATCH_COMPLETE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "scenario benchmark `%.*s` requires measure = \"dispatch_complete\"",
        (int)benchmark_plan->name.size, benchmark_plan->name.data);
  }
  if (measure_kind != IREE_BENCHMARK_LOOM_MEASURE_DISPATCH_COMPLETE &&
      (!iree_string_view_is_empty(options->profile_artifacts_dir) ||
       options->profile_data_requested ||
       options->profile_counters.count != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "final-batch profile flags require benchmark `%.*s` "
        "to use measure = \"dispatch_complete\"",
        (int)benchmark_plan->name.size, benchmark_plan->name.data);
  }

  *out_policy = (iree_benchmark_loom_benchmark_policy_t){
      .measure_kind = measure_kind,
      .measure = measure,
      .warmup_iterations = options->warmup_iterations,
      .iterations = options->iterations,
  };
  loom_run_hal_benchmark_options_initialize(&out_policy->hal_options);
  if (measure_kind == IREE_BENCHMARK_LOOM_MEASURE_DISPATCH_COMPLETE) {
    bool profile_final_batch = options->profile_final_batch;
    const bool profile_artifacts_requested =
        !iree_string_view_is_empty(options->profile_artifacts_dir);
    if ((profile_artifacts_requested || options->profile_data_requested ||
         options->profile_counters.count != 0) &&
        !profile_final_batch) {
      if (options->profile_final_batch_specified &&
          !options->profile_final_batch) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "--profile-final-batch=false conflicts with "
                                "requested final-batch profile data");
      }
      profile_final_batch = true;
    }
    iree_duration_t min_duration_ns = 0;
    iree_duration_t warmup_min_duration_ns = 0;
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_duration_ms_to_ns(
        options->min_time_ms, &min_duration_ns));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_duration_ms_to_ns(
        options->warmup_time_ms, &warmup_min_duration_ns));
    out_policy->hal_options.timing.batch_size = options->batch_size;
    out_policy->hal_options.timing.warmup_batch_count =
        options->warmup_iterations;
    out_policy->hal_options.timing.warmup_min_duration_ns =
        warmup_min_duration_ns;
    out_policy->hal_options.timing.min_batch_count = options->iterations;
    out_policy->hal_options.timing.min_duration_ns = min_duration_ns;
    out_policy->hal_options.timing.max_batch_count = options->max_batches;
    out_policy->hal_options.timing.stable_p90_to_p50_delta_ppm =
        options->stable_p90_to_p50_ppm;
    out_policy->hal_options.dispatch_batch.dispatch_count = options->batch_size;
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_parse_profile_data_families(
        options->profile_data, &out_policy->hal_options.profile_data_families));
    const iree_hal_device_profiling_data_families_t counter_data =
        IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES |
        IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_RANGES;
    if (options->profile_counters.count != 0) {
      if (!iree_any_bit_set(out_policy->hal_options.profile_data_families,
                            counter_data)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "--profile-counter requires --profile-data to include counters or "
            "counter-ranges");
      }
      out_policy->profile_counter_set =
          (iree_hal_profile_counter_set_selection_t){
              .name = IREE_SV("iree-benchmark-loom"),
              .counter_name_count = options->profile_counters.count,
              .counter_names = options->profile_counters.values,
          };
      out_policy->hal_options.profile_counter_set_count = 1;
      out_policy->hal_options.profile_counter_sets =
          &out_policy->profile_counter_set;
    }
    if (profile_final_batch) {
      out_policy->hal_options.flags |=
          LOOM_RUN_HAL_BENCHMARK_FLAG_PROFILE_FINAL_BATCH;
    }
  }
  return iree_ok_status();
}

static iree_host_size_t iree_benchmark_loom_compare_token_count(
    iree_string_view_t compare_list) {
  iree_host_size_t count = 0;
  iree_string_view_t remaining = iree_string_view_trim(compare_list);
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_split(remaining, ',', &token, &remaining);
    token = iree_string_view_trim(token);
    if (!iree_string_view_is_empty(token)) {
      ++count;
    }
    remaining = iree_string_view_trim(remaining);
  }
  return count;
}

static bool iree_benchmark_loom_benchmark_name_less(
    const loom_testbench_module_plan_t* module_plan,
    const iree_host_size_t* lhs, const iree_host_size_t* rhs) {
  const int order = iree_string_view_compare(
      module_plan->benchmarks[*lhs].name, module_plan->benchmarks[*rhs].name);
  return order != 0 ? order < 0 : *lhs < *rhs;
}

LOOM_DEFINE_ADAPTIVE_SORT_WITH_CONTEXT(iree_benchmark_loom_sort_benchmark_names,
                                       iree_host_size_t,
                                       const loom_testbench_module_plan_t*,
                                       iree_benchmark_loom_benchmark_name_less)

static iree_status_t iree_benchmark_loom_find_benchmark_by_name(
    const loom_testbench_module_plan_t* module_plan, iree_string_view_t name,
    const iree_host_size_t* sorted_indices,
    iree_host_size_t* out_benchmark_index,
    const loom_testbench_benchmark_plan_t** out_benchmark) {
  *out_benchmark_index = IREE_HOST_SIZE_MAX;
  *out_benchmark = NULL;
  if (sorted_indices != NULL) {
    iree_host_size_t begin = 0;
    iree_host_size_t end = module_plan->benchmark_count;
    while (begin < end) {
      const iree_host_size_t middle = begin + (end - begin) / 2;
      if (iree_string_view_compare(
              module_plan->benchmarks[sorted_indices[middle]].name, name) < 0) {
        begin = middle + 1;
      } else {
        end = middle;
      }
    }
    if (begin < module_plan->benchmark_count &&
        iree_string_view_equal(
            module_plan->benchmarks[sorted_indices[begin]].name, name)) {
      *out_benchmark_index = sorted_indices[begin];
    }
  } else {
    for (iree_host_size_t i = 0; i < module_plan->benchmark_count; ++i) {
      if (iree_string_view_equal(module_plan->benchmarks[i].name, name)) {
        *out_benchmark_index = i;
        break;
      }
    }
  }
  if (*out_benchmark_index != IREE_HOST_SIZE_MAX) {
    *out_benchmark = &module_plan->benchmarks[*out_benchmark_index];
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "--compare benchmark '@%.*s' was not found",
                          (int)name.size, name.data);
}

static iree_status_t iree_benchmark_loom_sample_window(
    const iree_benchmark_loom_options_t* options, iree_host_size_t sample_count,
    iree_host_size_t* out_begin_sample, iree_host_size_t* out_end_sample);

static iree_status_t iree_benchmark_loom_selected_benchmark_initialize(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const iree_benchmark_loom_options_t* options,
    iree_host_size_t benchmark_index, iree_host_size_t candidate_index,
    iree_benchmark_loom_selected_benchmark_t* out_selection) {
  const bool is_case = benchmark_plan->case_index < module_plan->case_count;
  const bool is_scenario =
      benchmark_plan->scenario_index < module_plan->scenario_count;
  if (is_case == is_scenario) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "benchmark `%.*s` must reference exactly one planned check.case or "
        "check.scenario",
        (int)benchmark_plan->name.size, benchmark_plan->name.data);
  }
  if (benchmark_plan->sample_count == 0) {
    const iree_string_view_t record_name =
        is_case ? module_plan->cases[benchmark_plan->case_index].name
                : module_plan->scenarios[benchmark_plan->scenario_index].name;
    const char* record_kind = is_case ? "case" : "scenario";
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "selected benchmark `%.*s` for check.%s `%.*s` has zero executable "
        "samples; selected_benchmark='%.*s', selected_case='%.*s', "
        "sample=%d",
        (int)benchmark_plan->name.size, benchmark_plan->name.data, record_kind,
        (int)record_name.size, record_name.data,
        (int)options->selected_benchmark.size, options->selected_benchmark.data,
        (int)options->selected_case.size, options->selected_case.data,
        options->sample_ordinal);
  }
  memset(out_selection, 0, sizeof(*out_selection));
  snprintf(out_selection->candidate_id_storage,
           sizeof(out_selection->candidate_id_storage), "c%" PRIhsz,
           candidate_index);
  out_selection->identity = (iree_benchmark_loom_candidate_identity_t){
      .candidate_id =
          iree_make_cstring_view(out_selection->candidate_id_storage),
      .candidate_index = candidate_index,
  };
  out_selection->benchmark_plan = benchmark_plan;
  out_selection->benchmark_index = benchmark_index;
  out_selection->case_plan =
      is_case ? &module_plan->cases[benchmark_plan->case_index] : NULL;
  out_selection->scenario_plan =
      is_scenario ? &module_plan->scenarios[benchmark_plan->scenario_index]
                  : NULL;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_policy_from_benchmark(
      module_plan, benchmark_plan, options, &out_selection->policy));
  return iree_benchmark_loom_sample_window(
      options, benchmark_plan->sample_count,
      &out_selection->begin_benchmark_sample,
      &out_selection->end_benchmark_sample);
}

static iree_status_t iree_benchmark_loom_sample_window(
    const iree_benchmark_loom_options_t* options, iree_host_size_t sample_count,
    iree_host_size_t* out_begin_sample, iree_host_size_t* out_end_sample) {
  if (options->sample_ordinal < 0) {
    *out_begin_sample = 0;
    *out_end_sample = sample_count;
    return iree_ok_status();
  }
  const iree_host_size_t sample_ordinal =
      (iree_host_size_t)options->sample_ordinal;
  if (sample_ordinal >= sample_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "--sample=%" PRIhsz
                            " exceeds selected benchmark sample count %" PRIhsz,
                            sample_ordinal, sample_count);
  }
  *out_begin_sample = sample_ordinal;
  *out_end_sample = sample_ordinal + 1;
  return iree_ok_status();
}

static iree_host_size_t iree_benchmark_loom_find_or_append_compile_item(
    iree_benchmark_loom_work_plan_t* plan, iree_host_size_t selection_index,
    iree_host_size_t* case_compile_indices) {
  const iree_benchmark_loom_selected_benchmark_t* selection =
      &plan->selected_benchmarks[selection_index];
  // Zero means unseen; populated entries are compile indices plus one.
  iree_host_size_t* entry =
      &case_compile_indices[selection->benchmark_plan->case_index];
  if (*entry != 0) {
    return *entry - 1;
  }
  const iree_host_size_t compile_item_index = plan->hal_compile_item_count++;
  *entry = compile_item_index + 1;
  plan->hal_compile_items[compile_item_index] =
      (iree_benchmark_loom_hal_compile_item_t){
          .compile_item_index = compile_item_index,
          .representative_selection_index = selection_index,
      };
  return compile_item_index;
}

static int iree_benchmark_loom_compare_ordinal(iree_host_size_t lhs,
                                               iree_host_size_t rhs) {
  return (lhs > rhs) - (lhs < rhs);
}

// Multi-sample windows are prefixes: --sample selects exactly one sample.
// A parameter's prefix is identified by its initial value and its increment
// period. Pinned parameters and dimensions that never advance within the
// prefix are constant. Comparing these keys avoids expanding the Cartesian
// product, including when different assignments describe the same capped
// prefix.
static int iree_benchmark_loom_compare_sample_prefixes(
    const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_benchmark_plan_t* lhs,
    const loom_testbench_benchmark_plan_t* rhs, iree_host_size_t sample_count) {
  iree_host_size_t strides[2] = {1, 1};
  const loom_testbench_benchmark_plan_t* benchmarks[2] = {lhs, rhs};
  for (iree_host_size_t i = 0; i < case_plan->parameter_count; ++i) {
    const iree_host_size_t radix = case_plan->parameters[i].sample_count;
    iree_host_size_t initial_values[2] = {0, 0};
    iree_host_size_t periods[2] = {0, 0};
    for (int side = 0; side < 2; ++side) {
      const loom_testbench_benchmark_plan_t* benchmark = benchmarks[side];
      const iree_host_size_t ordinal =
          i < benchmark->parameter_sample_ordinal_count
              ? benchmark->parameter_sample_ordinals[i]
              : LOOM_TESTBENCH_PARAMETER_SAMPLE_ORDINAL_ALL;
      if (ordinal != LOOM_TESTBENCH_PARAMETER_SAMPLE_ORDINAL_ALL) {
        initial_values[side] = ordinal;
      } else {
        if (radix > 1 && strides[side] < sample_count) {
          periods[side] = strides[side];
        }
        // Strides at or beyond the prefix length all describe a constant
        // dimension. Saturation also avoids multiplying an unbounded domain.
        strides[side] = strides[side] > sample_count / radix
                            ? sample_count
                            : strides[side] * radix;
      }
    }
    int order = iree_benchmark_loom_compare_ordinal(initial_values[0],
                                                    initial_values[1]);
    if (order != 0) {
      return order;
    }
    order = iree_benchmark_loom_compare_ordinal(periods[0], periods[1]);
    if (order != 0) {
      return order;
    }
  }
  return 0;
}

// Policy is fixed by the invocation's options and record kind, so it is not
// part of the key. Physical identity is the record and its measured sample.
typedef struct iree_benchmark_loom_work_item_key_t {
  // Work kind discriminating the record and measured sample domain.
  iree_benchmark_loom_work_item_kind_t kind;
  // Case or scenario index within the module, selected by |kind|.
  iree_host_size_t record_index;
  // Scenario ordinal or saturated Cartesian case ordinal. Equal saturated case
  // keys require coordinate comparison. Retained only during deduplication.
  iree_host_size_t sample_key;
} iree_benchmark_loom_work_item_key_t;

typedef struct iree_benchmark_loom_work_item_index_t {
  // Plan retaining selections for exact comparison of windows and large keys.
  const iree_benchmark_loom_work_plan_t* plan;
  // Cached keys in original work-item order, alive only during deduplication.
  const iree_benchmark_loom_work_item_key_t* keys;
} iree_benchmark_loom_work_item_index_t;

static IREE_ATTRIBUTE_NOINLINE int iree_benchmark_loom_compare_sample_details(
    const iree_benchmark_loom_work_item_index_t* index,
    iree_host_size_t lhs_index, iree_host_size_t rhs_index) {
  const iree_benchmark_loom_work_plan_t* plan = index->plan;
  const iree_benchmark_loom_work_item_key_t* lhs_key = &index->keys[lhs_index];
  const iree_benchmark_loom_work_item_key_t* rhs_key = &index->keys[rhs_index];
  int order = 0;
  const iree_benchmark_loom_work_item_t* lhs = &plan->work_items[lhs_index];
  const iree_benchmark_loom_work_item_t* rhs = &plan->work_items[rhs_index];
  const iree_benchmark_loom_selected_benchmark_t* lhs_selection =
      &plan->selected_benchmarks[lhs->representative_selection_index];
  const iree_benchmark_loom_selected_benchmark_t* rhs_selection =
      &plan->selected_benchmarks[rhs->representative_selection_index];
  const loom_testbench_benchmark_plan_t* lhs_benchmark =
      lhs_selection->benchmark_plan;
  const loom_testbench_benchmark_plan_t* rhs_benchmark =
      rhs_selection->benchmark_plan;
  if (lhs->kind == IREE_BENCHMARK_LOOM_WORK_ITEM_CASE_END_TO_END) {
    order = iree_benchmark_loom_compare_ordinal(lhs->begin_benchmark_sample,
                                                rhs->begin_benchmark_sample);
    if (order != 0) {
      return order;
    }
    order = iree_benchmark_loom_compare_ordinal(lhs->end_benchmark_sample,
                                                rhs->end_benchmark_sample);
    if (order != 0) {
      return order;
    }
    if (lhs->end_benchmark_sample - lhs->begin_benchmark_sample != 1) {
      return iree_benchmark_loom_compare_sample_prefixes(
          lhs_selection->case_plan, lhs_benchmark, rhs_benchmark,
          lhs->end_benchmark_sample);
    }
  }
  order = iree_benchmark_loom_compare_ordinal(lhs_key->sample_key,
                                              rhs_key->sample_key);
  if (order != 0 || lhs_key->sample_key != IREE_HOST_SIZE_MAX) {
    return order;
  }
  return loom_testbench_case_sample_compare(
      lhs_selection->case_plan,
      loom_testbench_benchmark_case_sample(lhs_benchmark,
                                           lhs->begin_benchmark_sample),
      loom_testbench_benchmark_case_sample(rhs_benchmark,
                                           rhs->begin_benchmark_sample));
}

static int iree_benchmark_loom_compare_work_items(
    const iree_benchmark_loom_work_item_index_t* index,
    iree_host_size_t lhs_index, iree_host_size_t rhs_index) {
  const iree_benchmark_loom_work_item_key_t* lhs = &index->keys[lhs_index];
  const iree_benchmark_loom_work_item_key_t* rhs = &index->keys[rhs_index];
  int order = iree_benchmark_loom_compare_ordinal(lhs->kind, rhs->kind);
  if (order != 0) {
    return order;
  }
  order =
      iree_benchmark_loom_compare_ordinal(lhs->record_index, rhs->record_index);
  if (order != 0) {
    return order;
  }
  if (lhs->kind != IREE_BENCHMARK_LOOM_WORK_ITEM_CASE_END_TO_END &&
      (lhs->sample_key != IREE_HOST_SIZE_MAX ||
       rhs->sample_key != IREE_HOST_SIZE_MAX)) {
    return iree_benchmark_loom_compare_ordinal(lhs->sample_key,
                                               rhs->sample_key);
  }
  return iree_benchmark_loom_compare_sample_details(index, lhs_index,
                                                    rhs_index);
}

static bool iree_benchmark_loom_work_item_less(
    const iree_benchmark_loom_work_item_index_t* index,
    const iree_host_size_t* lhs, const iree_host_size_t* rhs) {
  const int order = iree_benchmark_loom_compare_work_items(index, *lhs, *rhs);
  return order != 0 ? order < 0 : *lhs < *rhs;
}

LOOM_DEFINE_ADAPTIVE_SORT_WITH_CONTEXT(
    iree_benchmark_loom_sort_work_items, iree_host_size_t,
    const iree_benchmark_loom_work_item_index_t*,
    iree_benchmark_loom_work_item_less)

static void iree_benchmark_loom_deduplicate_work_items(
    iree_benchmark_loom_work_plan_t* plan, iree_host_size_t* order,
    iree_benchmark_loom_work_item_key_t* keys) {
  for (iree_host_size_t i = 0; i < plan->work_item_count; ++i) {
    const iree_benchmark_loom_work_item_t* work_item = &plan->work_items[i];
    const iree_benchmark_loom_selected_benchmark_t* selection =
        &plan->selected_benchmarks[work_item->representative_selection_index];
    order[i] = i;
    keys[i] = (iree_benchmark_loom_work_item_key_t){
        .kind = work_item->kind,
        .record_index = selection->case_plan != NULL
                            ? selection->benchmark_plan->case_index
                            : selection->benchmark_plan->scenario_index,
        .sample_key = work_item->begin_benchmark_sample,
    };
    if (selection->case_plan != NULL &&
        work_item->end_benchmark_sample - work_item->begin_benchmark_sample ==
            1 &&
        !loom_testbench_case_sample_flatten(
            selection->case_plan,
            loom_testbench_benchmark_case_sample(
                selection->benchmark_plan, work_item->begin_benchmark_sample),
            &keys[i].sample_key)) {
      keys[i].sample_key = IREE_HOST_SIZE_MAX;
    }
  }
  const iree_benchmark_loom_work_item_index_t index = {
      .plan = plan,
      .keys = keys,
  };
  iree_benchmark_loom_sort_work_items(&index, order, plan->work_item_count);
  iree_host_size_t representative = 0;
  for (iree_host_size_t i = 0; i < plan->work_item_count; ++i) {
    if (i == 0 || iree_benchmark_loom_compare_work_items(&index, representative,
                                                         order[i]) != 0) {
      representative = order[i];
    }
    plan->logical_samples[order[i]].work_item_index = representative;
  }

  // Candidates and logical samples were appended one-to-one in report order.
  // Equal candidates sort by original ordinal, leaving every alias pointing
  // backward to its first owner. Compact in that original order, translating
  // aliases through the already-finalized owner's logical sample.
  plan->work_item_count = 0;
  for (iree_host_size_t i = 0; i < plan->logical_sample_count; ++i) {
    iree_benchmark_loom_logical_sample_t* logical = &plan->logical_samples[i];
    representative = logical->work_item_index;
    if (representative == i) {
      const iree_host_size_t work_item_index = plan->work_item_count++;
      plan->work_items[work_item_index] = plan->work_items[i];
      plan->work_items[work_item_index].work_item_index = work_item_index;
      logical->work_item_index = work_item_index;
    } else {
      logical->work_item_index =
          plan->logical_samples[representative].work_item_index;
    }
  }
}

static iree_host_size_t iree_benchmark_loom_append_work_item(
    iree_benchmark_loom_work_plan_t* plan,
    iree_benchmark_loom_work_item_kind_t kind, iree_host_size_t selection_index,
    iree_host_size_t begin_sample, iree_host_size_t end_sample,
    loom_testbench_scenario_sample_coordinate_t scenario_coordinate,
    iree_host_size_t hal_compile_item_index) {
  const iree_host_size_t work_item_index = plan->work_item_count++;
  plan->work_items[work_item_index] = (iree_benchmark_loom_work_item_t){
      .kind = kind,
      .work_item_index = work_item_index,
      .representative_selection_index = selection_index,
      .hal_compile_item_index = hal_compile_item_index,
      .begin_benchmark_sample = begin_sample,
      .end_benchmark_sample = end_sample,
      .scenario_coordinate = scenario_coordinate,
  };
  return work_item_index;
}

static void iree_benchmark_loom_append_logical_sample(
    iree_benchmark_loom_work_plan_t* plan, iree_host_size_t selection_index,
    iree_host_size_t begin_sample, iree_host_size_t end_sample,
    loom_testbench_scenario_sample_coordinate_t scenario_coordinate,
    iree_host_size_t work_item_index) {
  plan->logical_samples[plan->logical_sample_count++] =
      (iree_benchmark_loom_logical_sample_t){
          .selection_index = selection_index,
          .begin_benchmark_sample = begin_sample,
          .end_benchmark_sample = end_sample,
          .scenario_coordinate = scenario_coordinate,
          .work_item_index = work_item_index,
      };
}

static iree_status_t iree_benchmark_loom_count_logical_samples(
    const iree_benchmark_loom_selected_benchmark_t* selection,
    iree_host_size_t* inout_logical_sample_count) {
  const iree_host_size_t sample_count =
      selection->policy.measure_kind ==
              IREE_BENCHMARK_LOOM_MEASURE_CASE_END_TO_END
          ? 1
          : selection->end_benchmark_sample - selection->begin_benchmark_sample;
  if (sample_count > IREE_HOST_SIZE_MAX - *inout_logical_sample_count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "selected benchmark sample count exceeds host "
                            "limits");
  }
  *inout_logical_sample_count += sample_count;
  return iree_ok_status();
}

static void iree_benchmark_loom_append_case_end_to_end_work(
    iree_benchmark_loom_work_plan_t* plan, iree_host_size_t selection_index,
    iree_host_size_t compile_item_index) {
  const iree_benchmark_loom_selected_benchmark_t* selection =
      &plan->selected_benchmarks[selection_index];
  const iree_host_size_t begin_sample = selection->begin_benchmark_sample;
  const iree_host_size_t end_sample = selection->end_benchmark_sample;
  const iree_host_size_t work_item_index = iree_benchmark_loom_append_work_item(
      plan, IREE_BENCHMARK_LOOM_WORK_ITEM_CASE_END_TO_END, selection_index,
      begin_sample, end_sample,
      (loom_testbench_scenario_sample_coordinate_t){0}, compile_item_index);
  iree_benchmark_loom_append_logical_sample(
      plan, selection_index, begin_sample, end_sample,
      (loom_testbench_scenario_sample_coordinate_t){0}, work_item_index);
}

static void iree_benchmark_loom_append_dispatch_sample_work(
    iree_benchmark_loom_work_plan_t* plan, iree_host_size_t selection_index,
    iree_host_size_t benchmark_sample_ordinal,
    iree_host_size_t compile_item_index) {
  const iree_host_size_t work_item_index = iree_benchmark_loom_append_work_item(
      plan, IREE_BENCHMARK_LOOM_WORK_ITEM_DISPATCH_SAMPLE, selection_index,
      benchmark_sample_ordinal, benchmark_sample_ordinal + 1,
      (loom_testbench_scenario_sample_coordinate_t){0}, compile_item_index);
  iree_benchmark_loom_append_logical_sample(
      plan, selection_index, benchmark_sample_ordinal,
      benchmark_sample_ordinal + 1,
      (loom_testbench_scenario_sample_coordinate_t){0}, work_item_index);
}

static void iree_benchmark_loom_append_scenario_trial_work(
    iree_benchmark_loom_work_plan_t* plan, iree_host_size_t selection_index,
    iree_host_size_t benchmark_sample_ordinal,
    loom_testbench_scenario_sample_coordinate_t coordinate) {
  const iree_host_size_t work_item_index = iree_benchmark_loom_append_work_item(
      plan, IREE_BENCHMARK_LOOM_WORK_ITEM_SCENARIO_TRIAL, selection_index,
      benchmark_sample_ordinal, benchmark_sample_ordinal + 1, coordinate,
      IREE_BENCHMARK_LOOM_WORK_PLAN_INDEX_INVALID);
  iree_benchmark_loom_append_logical_sample(
      plan, selection_index, benchmark_sample_ordinal,
      benchmark_sample_ordinal + 1, coordinate, work_item_index);
}

static void iree_benchmark_loom_append_scenario_work(
    iree_benchmark_loom_work_plan_t* plan, iree_host_size_t selection_index) {
  const iree_benchmark_loom_selected_benchmark_t* selection =
      &plan->selected_benchmarks[selection_index];
  const loom_testbench_scenario_plan_t* scenario = selection->scenario_plan;
  loom_testbench_scenario_sample_coordinate_t coordinate =
      loom_testbench_benchmark_sample_scenario_coordinate(
          scenario, selection->benchmark_plan,
          selection->begin_benchmark_sample);
  for (iree_host_size_t sample_ordinal = selection->begin_benchmark_sample;
       sample_ordinal < selection->end_benchmark_sample; ++sample_ordinal) {
    iree_benchmark_loom_append_scenario_trial_work(plan, selection_index,
                                                   sample_ordinal, coordinate);
    ++coordinate.trial_ordinal;
    if (coordinate.trial_ordinal ==
        scenario->trials[coordinate.trial_index].trial_count) {
      coordinate.trial_ordinal = 0;
      if (++coordinate.trial_index == scenario->trial_count) {
        coordinate.trial_index = 0;
        ++coordinate.configuration_ordinal;
      }
    }
  }
}

static void iree_benchmark_loom_append_selection_work(
    iree_benchmark_loom_work_plan_t* plan, iree_host_size_t selection_index,
    iree_host_size_t compile_item_index) {
  const iree_benchmark_loom_selected_benchmark_t* selection =
      &plan->selected_benchmarks[selection_index];
  if (selection->policy.measure_kind ==
      IREE_BENCHMARK_LOOM_MEASURE_CASE_END_TO_END) {
    iree_benchmark_loom_append_case_end_to_end_work(plan, selection_index,
                                                    compile_item_index);
    return;
  }
  if (selection->scenario_plan != NULL) {
    iree_benchmark_loom_append_scenario_work(plan, selection_index);
    return;
  }
  for (iree_host_size_t sample_ordinal = selection->begin_benchmark_sample;
       sample_ordinal < selection->end_benchmark_sample; ++sample_ordinal) {
    iree_benchmark_loom_append_dispatch_sample_work(
        plan, selection_index, sample_ordinal, compile_item_index);
  }
}

static iree_status_t iree_benchmark_loom_allocate_array(
    iree_allocator_t allocator, iree_host_size_t count,
    iree_host_size_t element_size, void** out_ptr) {
  *out_ptr = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  // Every plan array publishes only its completely initialized prefix.
  return iree_allocator_malloc_array_uninitialized(allocator, count,
                                                   element_size, out_ptr);
}

static iree_status_t iree_benchmark_loom_select_compare_benchmarks(
    const loom_testbench_module_plan_t* module_plan,
    const iree_benchmark_loom_options_t* options,
    iree_benchmark_loom_work_plan_t* plan) {
  iree_string_view_t compare = iree_string_view_trim(options->compare);
  const iree_host_size_t selection_count =
      iree_benchmark_loom_compare_token_count(compare);
  if (selection_count < 2) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--compare requires at least two comma-separated check.benchmark "
        "symbols");
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_allocate_array(
      plan->host_allocator, selection_count, sizeof(*plan->selected_benchmarks),
      (void**)&plan->selected_benchmarks));

  // Small comparison lists need at most eight linear scans. Larger lists
  // amortize a temporary name index; selections themselves stay in request
  // order and retain stable addresses for their candidate identifiers.
  iree_host_size_t* sorted_indices = NULL;
  if (selection_count > 8) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_allocate_array(
        plan->host_allocator, module_plan->benchmark_count,
        sizeof(*sorted_indices), (void**)&sorted_indices));
    for (iree_host_size_t i = 0; i < module_plan->benchmark_count; ++i) {
      sorted_indices[i] = i;
    }
    iree_benchmark_loom_sort_benchmark_names(module_plan, sorted_indices,
                                             module_plan->benchmark_count);
  }

  iree_status_t status = iree_ok_status();
  iree_host_size_t selection_index = 0;
  iree_string_view_t remaining = compare;
  while (iree_status_is_ok(status) && !iree_string_view_is_empty(remaining)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_split(remaining, ',', &token, &remaining);
    token = iree_benchmark_loom_normalize_selection_name(token);
    if (iree_string_view_is_empty(token)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "--compare contains an empty benchmark name");
      break;
    }
    iree_host_size_t benchmark_index = IREE_HOST_SIZE_MAX;
    const loom_testbench_benchmark_plan_t* benchmark = NULL;
    status = iree_benchmark_loom_find_benchmark_by_name(
        module_plan, token, sorted_indices, &benchmark_index, &benchmark);
    if (iree_status_is_ok(status)) {
      status = iree_benchmark_loom_selected_benchmark_initialize(
          module_plan, benchmark, options, benchmark_index, selection_index,
          &plan->selected_benchmarks[selection_index]);
    }
    ++selection_index;
    remaining = iree_string_view_trim(remaining);
  }
  iree_allocator_free(plan->host_allocator, sorted_indices);
  if (iree_status_is_ok(status)) {
    const bool compares_scenarios =
        plan->selected_benchmarks[0].scenario_plan != NULL;
    for (iree_host_size_t i = 1; i < selection_count; ++i) {
      if ((plan->selected_benchmarks[i].scenario_plan != NULL) !=
          compares_scenarios) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "--compare candidates must all reference check.case or all "
            "reference check.scenario");
        break;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    plan->selected_benchmark_count = selection_count;
  }
  return status;
}

static iree_status_t iree_benchmark_loom_select_matching_benchmarks(
    const loom_testbench_module_plan_t* module_plan,
    const iree_benchmark_loom_options_t* options,
    iree_benchmark_loom_work_plan_t* plan) {
  if (module_plan->benchmark_count == 0 &&
      iree_string_view_is_empty(options->selected_benchmark) &&
      iree_string_view_is_empty(options->selected_case)) {
    return iree_ok_status();
  }

  iree_host_size_t selection_count = 0;
  for (iree_host_size_t i = 0; i < module_plan->benchmark_count; ++i) {
    const loom_testbench_benchmark_plan_t* benchmark_plan =
        &module_plan->benchmarks[i];
    const bool is_case = benchmark_plan->case_index < module_plan->case_count;
    const bool is_scenario =
        benchmark_plan->scenario_index < module_plan->scenario_count;
    if (is_case == is_scenario) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "benchmark `%.*s` must reference exactly one planned check.case or "
          "check.scenario",
          (int)benchmark_plan->name.size, benchmark_plan->name.data);
    }
    const bool record_matches =
        is_case ? iree_benchmark_loom_case_matches_selection(
                      &module_plan->cases[benchmark_plan->case_index],
                      options->selected_case)
                : iree_benchmark_loom_scenario_matches_selection(
                      &module_plan->scenarios[benchmark_plan->scenario_index],
                      options->selected_case);
    if (iree_benchmark_loom_benchmark_matches_selection(
            benchmark_plan, options->selected_benchmark) &&
        record_matches) {
      ++selection_count;
    }
  }
  if (selection_count == 0) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "no check.benchmark matched '%.*s'",
                            (int)options->selected_benchmark.size,
                            options->selected_benchmark.data);
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_allocate_array(
      plan->host_allocator, selection_count, sizeof(*plan->selected_benchmarks),
      (void**)&plan->selected_benchmarks));

  iree_host_size_t selection_index = 0;
  for (iree_host_size_t i = 0; i < module_plan->benchmark_count; ++i) {
    const loom_testbench_benchmark_plan_t* benchmark_plan =
        &module_plan->benchmarks[i];
    const bool is_case = benchmark_plan->case_index < module_plan->case_count;
    const bool record_matches =
        is_case ? iree_benchmark_loom_case_matches_selection(
                      &module_plan->cases[benchmark_plan->case_index],
                      options->selected_case)
                : iree_benchmark_loom_scenario_matches_selection(
                      &module_plan->scenarios[benchmark_plan->scenario_index],
                      options->selected_case);
    if (!iree_benchmark_loom_benchmark_matches_selection(
            benchmark_plan, options->selected_benchmark) ||
        !record_matches) {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_selected_benchmark_initialize(
        module_plan, benchmark_plan, options, i, selection_index,
        &plan->selected_benchmarks[selection_index]));
    ++selection_index;
  }
  plan->selected_benchmark_count = selection_count;
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_work_plan_initialize(
    const loom_testbench_module_plan_t* module_plan,
    const iree_benchmark_loom_options_t* options, iree_allocator_t allocator,
    iree_benchmark_loom_work_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(module_plan);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = (iree_benchmark_loom_work_plan_t){0};

  iree_benchmark_loom_work_plan_t plan = {
      .host_allocator = allocator,
  };
  iree_status_t status = iree_ok_status();
  if (!iree_string_view_is_empty(options->compare)) {
    status = iree_benchmark_loom_select_compare_benchmarks(module_plan, options,
                                                           &plan);
  } else {
    status = iree_benchmark_loom_select_matching_benchmarks(module_plan,
                                                            options, &plan);
  }

  iree_host_size_t logical_sample_capacity = 0;
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < plan.selected_benchmark_count; ++i) {
    status = iree_benchmark_loom_count_logical_samples(
        &plan.selected_benchmarks[i], &logical_sample_capacity);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_allocate_array(
        allocator, logical_sample_capacity, sizeof(*plan.logical_samples),
        (void**)&plan.logical_samples);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_allocate_array(
        allocator,
        iree_min(module_plan->case_count, plan.selected_benchmark_count),
        sizeof(*plan.hal_compile_items), (void**)&plan.hal_compile_items);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_allocate_array(
        allocator, logical_sample_capacity, sizeof(*plan.work_items),
        (void**)&plan.work_items);
  }
  iree_host_size_t* case_compile_indices = NULL;
  iree_host_size_t* work_item_order = NULL;
  if (iree_status_is_ok(status) && plan.selected_benchmark_count != 0 &&
      module_plan->case_count != 0) {
    status = iree_allocator_malloc_array(allocator, module_plan->case_count,
                                         sizeof(*case_compile_indices),
                                         (void**)&case_compile_indices);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_allocate_array(
        allocator, logical_sample_capacity,
        sizeof(*work_item_order) + sizeof(iree_benchmark_loom_work_item_key_t),
        (void**)&work_item_order);
  }
  if (iree_status_is_ok(status) && logical_sample_capacity != 0) {
    for (iree_host_size_t i = 0; i < plan.selected_benchmark_count; ++i) {
      const iree_benchmark_loom_selected_benchmark_t* selection =
          &plan.selected_benchmarks[i];
      iree_host_size_t compile_item_index =
          IREE_BENCHMARK_LOOM_WORK_PLAN_INDEX_INVALID;
      if (selection->case_plan != NULL &&
          (selection->policy.measure_kind ==
               IREE_BENCHMARK_LOOM_MEASURE_DISPATCH_COMPLETE ||
           selection->case_plan->kernel_launch_count != 0)) {
        compile_item_index = iree_benchmark_loom_find_or_append_compile_item(
            &plan, i, case_compile_indices);
      }
      iree_benchmark_loom_append_selection_work(&plan, i, compile_item_index);
    }
    // The single scratch allocation holds sort indices followed by aligned
    // keys. Sorting moves indices, not the larger key records.
    iree_benchmark_loom_deduplicate_work_items(
        &plan, work_item_order,
        (iree_benchmark_loom_work_item_key_t*)(work_item_order +
                                               logical_sample_capacity));
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < plan.work_item_count; ++i) {
    iree_benchmark_loom_work_item_t* work_item = &plan.work_items[i];
    if (work_item->hal_compile_item_index ==
        IREE_BENCHMARK_LOOM_WORK_PLAN_INDEX_INVALID) {
      continue;
    }
    iree_benchmark_loom_hal_compile_item_t* compile_item =
        &plan.hal_compile_items[work_item->hal_compile_item_index];
    work_item->execution_sample_base = compile_item->execution_sample_count;
    if (!iree_host_size_checked_add(
            compile_item->execution_sample_count,
            work_item->end_benchmark_sample - work_item->begin_benchmark_sample,
            &compile_item->execution_sample_count)) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "selected execution sample count exceeds "
                                "host limits");
    }
  }
  iree_allocator_free(allocator, work_item_order);
  iree_allocator_free(allocator, case_compile_indices);

  if (iree_status_is_ok(status)) {
    *out_plan = plan;
  } else {
    iree_benchmark_loom_work_plan_deinitialize(&plan);
  }
  return status;
}

void iree_benchmark_loom_work_plan_deinitialize(
    iree_benchmark_loom_work_plan_t* plan) {
  IREE_ASSERT_ARGUMENT(plan);
  iree_allocator_t allocator = plan->host_allocator;
  iree_allocator_free(allocator, plan->work_items);
  iree_allocator_free(allocator, plan->hal_compile_items);
  iree_allocator_free(allocator, plan->logical_samples);
  iree_allocator_free(allocator, plan->selected_benchmarks);
  *plan = (iree_benchmark_loom_work_plan_t){0};
}
