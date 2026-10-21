// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-benchmark/event.h"

#include "loom/tools/loom-benchmark/profile_report.h"
#include "loom/tools/loom-benchmark/report.h"

iree_status_t loom_benchmark_event_sink_emit(
    const loom_benchmark_event_sink_t* sink,
    const loom_benchmark_event_t* event) {
  IREE_ASSERT_ARGUMENT(sink);
  IREE_ASSERT_ARGUMENT(event);
  if (sink->emit == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "benchmark event sink has no emit callback");
  }
  return sink->emit(sink->user_data, event);
}

iree_status_t loom_benchmark_event_sink_emit_run(
    const loom_benchmark_event_sink_t* sink,
    const loom_benchmark_run_identity_t* run, bool dry_run,
    const loom_sanitizer_options_t* sanitizer) {
  IREE_ASSERT_ARGUMENT(run);
  IREE_ASSERT_ARGUMENT(sanitizer);
  return loom_benchmark_event_sink_emit(sink,
                                        &(loom_benchmark_event_t){
                                            .kind = LOOM_BENCHMARK_EVENT_RUN,
                                            .run =
                                                {
                                                    .run = run,
                                                    .dry_run = dry_run,
                                                    .sanitizer = *sanitizer,
                                                },
                                        });
}

iree_status_t loom_benchmark_event_sink_emit_plan(
    const loom_benchmark_event_sink_t* sink,
    const loom_benchmark_run_identity_t* run, const loom_module_t* module,
    const loom_benchmark_selected_benchmark_t* selection,
    const loom_benchmark_options_t* options) {
  IREE_ASSERT_ARGUMENT(run);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(selection);
  IREE_ASSERT_ARGUMENT(options);
  return loom_benchmark_event_sink_emit(sink,
                                        &(loom_benchmark_event_t){
                                            .kind = LOOM_BENCHMARK_EVENT_PLAN,
                                            .plan =
                                                {
                                                    .run = run,
                                                    .module = module,
                                                    .selection = selection,
                                                    .options = options,
                                                },
                                        });
}

iree_status_t loom_benchmark_event_sink_emit_work_plan(
    const loom_benchmark_event_sink_t* sink,
    const loom_benchmark_run_identity_t* run, const loom_module_t* module,
    const loom_benchmark_work_plan_t* work_plan) {
  IREE_ASSERT_ARGUMENT(run);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(work_plan);
  return loom_benchmark_event_sink_emit(
      sink, &(loom_benchmark_event_t){
                .kind = LOOM_BENCHMARK_EVENT_WORK_PLAN,
                .work_plan =
                    {
                        .run = run,
                        .module = module,
                        .work_plan = work_plan,
                    },
            });
}

iree_status_t loom_benchmark_event_sink_emit_summary(
    const loom_benchmark_event_sink_t* sink,
    const loom_benchmark_run_identity_t* run,
    const loom_benchmark_artifact_bundle_t* artifact_bundle,
    iree_host_size_t planned_case_count,
    iree_host_size_t planned_benchmark_count,
    iree_host_size_t selected_benchmark_count,
    iree_host_size_t logical_sample_count, iree_host_size_t work_item_count,
    iree_host_size_t failure_count, iree_host_size_t failed_benchmark_count,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count, bool dry_run) {
  IREE_ASSERT_ARGUMENT(run);
  return loom_benchmark_event_sink_emit(
      sink, &(loom_benchmark_event_t){
                .kind = LOOM_BENCHMARK_EVENT_SUMMARY,
                .summary =
                    {
                        .run = run,
                        .artifact_bundle = artifact_bundle,
                        .planned_case_count = planned_case_count,
                        .planned_benchmark_count = planned_benchmark_count,
                        .selected_benchmark_count = selected_benchmark_count,
                        .logical_sample_count = logical_sample_count,
                        .work_item_count = work_item_count,
                        .failure_count = failure_count,
                        .failed_benchmark_count = failed_benchmark_count,
                        .correctness_sample_count = correctness_sample_count,
                        .correctness_failed_sample_count =
                            correctness_failed_sample_count,
                        .dry_run = dry_run,
                    },
            });
}

iree_status_t loom_benchmark_event_sink_emit_device(
    const loom_benchmark_event_sink_t* sink,
    const loom_benchmark_run_identity_t* run,
    const loom_benchmark_hal_context_t* context) {
  IREE_ASSERT_ARGUMENT(run);
  IREE_ASSERT_ARGUMENT(context);
  return loom_benchmark_event_sink_emit(sink,
                                        &(loom_benchmark_event_t){
                                            .kind = LOOM_BENCHMARK_EVENT_DEVICE,
                                            .device =
                                                {
                                                    .run = run,
                                                    .context = context,
                                                },
                                        });
}

iree_status_t loom_benchmark_event_sink_emit_compile(
    const loom_benchmark_event_sink_t* sink,
    const loom_benchmark_run_identity_t* run,
    const loom_benchmark_candidate_identity_t* candidate,
    const loom_benchmark_hal_compile_item_t* compile_item,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const loom_benchmark_hal_actual_provider_t* provider) {
  IREE_ASSERT_ARGUMENT(run);
  IREE_ASSERT_ARGUMENT(candidate);
  IREE_ASSERT_ARGUMENT(benchmark_plan);
  IREE_ASSERT_ARGUMENT(case_plan);
  IREE_ASSERT_ARGUMENT(provider);
  return loom_benchmark_event_sink_emit(
      sink, &(loom_benchmark_event_t){
                .kind = LOOM_BENCHMARK_EVENT_COMPILE,
                .compile =
                    {
                        .run = run,
                        .candidate = candidate,
                        .compile_item = compile_item,
                        .benchmark_plan = benchmark_plan,
                        .case_plan = case_plan,
                        .provider = provider,
                    },
            });
}

iree_status_t loom_benchmark_event_sink_emit_sample(
    const loom_benchmark_event_sink_t* sink,
    const loom_benchmark_run_identity_t* run,
    const loom_benchmark_candidate_identity_t* candidate,
    iree_host_size_t work_item_index, const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    iree_host_size_t benchmark_sample_ordinal,
    const loom_testbench_case_sample_result_t* sample_result) {
  IREE_ASSERT_ARGUMENT(run);
  IREE_ASSERT_ARGUMENT(candidate);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(benchmark_plan);
  IREE_ASSERT_ARGUMENT(case_plan);
  IREE_ASSERT_ARGUMENT(sample_result);
  return loom_benchmark_event_sink_emit(
      sink, &(loom_benchmark_event_t){
                .kind = LOOM_BENCHMARK_EVENT_SAMPLE,
                .sample =
                    {
                        .run = run,
                        .candidate = candidate,
                        .work_item_index = work_item_index,
                        .module = module,
                        .benchmark_plan = benchmark_plan,
                        .case_plan = case_plan,
                        .benchmark_sample_ordinal = benchmark_sample_ordinal,
                        .sample_result = sample_result,
                    },
            });
}

iree_status_t loom_benchmark_event_sink_emit_benchmark_result(
    const loom_benchmark_event_sink_t* sink,
    const loom_benchmark_run_identity_t* run,
    const loom_benchmark_candidate_identity_t* candidate,
    const loom_benchmark_work_item_t* work_item, const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_scenario_plan_t* scenario_plan,
    const loom_benchmark_benchmark_policy_t* policy,
    const loom_benchmark_benchmark_result_t* benchmark_result,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count) {
  IREE_ASSERT_ARGUMENT(run);
  IREE_ASSERT_ARGUMENT(candidate);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(benchmark_plan);
  IREE_ASSERT((case_plan != NULL) != (scenario_plan != NULL));
  IREE_ASSERT_ARGUMENT(policy);
  IREE_ASSERT_ARGUMENT(benchmark_result);
  return loom_benchmark_event_sink_emit(
      sink, &(loom_benchmark_event_t){
                .kind = LOOM_BENCHMARK_EVENT_BENCHMARK_RESULT,
                .benchmark_result =
                    {
                        .run = run,
                        .candidate = candidate,
                        .work_item = work_item,
                        .module = module,
                        .benchmark_plan = benchmark_plan,
                        .case_plan = case_plan,
                        .scenario_plan = scenario_plan,
                        .policy = policy,
                        .benchmark_result = benchmark_result,
                        .correctness_sample_count = correctness_sample_count,
                        .correctness_failed_sample_count =
                            correctness_failed_sample_count,
                    },
            });
}

iree_status_t loom_benchmark_event_sink_emit_profile_replay(
    const loom_benchmark_event_sink_t* sink,
    const loom_benchmark_run_identity_t* run,
    const loom_benchmark_candidate_identity_t* candidate,
    iree_host_size_t work_item_index, const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_scenario_plan_t* scenario_plan,
    const loom_benchmark_benchmark_policy_t* policy,
    const loom_benchmark_benchmark_result_t* benchmark_result) {
  IREE_ASSERT_ARGUMENT(run);
  IREE_ASSERT_ARGUMENT(candidate);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(benchmark_plan);
  IREE_ASSERT((case_plan != NULL) != (scenario_plan != NULL));
  IREE_ASSERT_ARGUMENT(policy);
  IREE_ASSERT_ARGUMENT(benchmark_result);
  return loom_benchmark_event_sink_emit(
      sink, &(loom_benchmark_event_t){
                .kind = LOOM_BENCHMARK_EVENT_PROFILE_REPLAY,
                .profile_replay =
                    {
                        .run = run,
                        .candidate = candidate,
                        .work_item_index = work_item_index,
                        .module = module,
                        .benchmark_plan = benchmark_plan,
                        .case_plan = case_plan,
                        .scenario_plan = scenario_plan,
                        .policy = policy,
                        .benchmark_result = benchmark_result,
                    },
            });
}

iree_status_t loom_benchmark_event_sink_emit_failure(
    const loom_benchmark_event_sink_t* sink,
    const loom_benchmark_run_identity_t* run, iree_string_view_t stage,
    iree_string_view_t kind, iree_string_view_t message,
    const loom_benchmark_diagnostic_capture_t* diagnostics) {
  IREE_ASSERT_ARGUMENT(run);
  return loom_benchmark_event_sink_emit(
      sink, &(loom_benchmark_event_t){
                .kind = LOOM_BENCHMARK_EVENT_FAILURE,
                .failure =
                    {
                        .run = run,
                        .stage = stage,
                        .kind = kind,
                        .message = message,
                        .diagnostics = diagnostics,
                    },
            });
}

iree_status_t loom_benchmark_event_sink_emit_planning_failure(
    const loom_benchmark_event_sink_t* sink,
    const loom_benchmark_run_identity_t* run, iree_string_view_t stage,
    iree_string_view_t kind, iree_string_view_t message,
    const loom_testbench_module_plan_t* testbench_plan,
    const loom_testbench_issue_t* planning_issues,
    iree_host_size_t planning_issue_count) {
  IREE_ASSERT_ARGUMENT(run);
  IREE_ASSERT_ARGUMENT(testbench_plan);
  IREE_ASSERT_ARGUMENT(planning_issues);
  return loom_benchmark_event_sink_emit(
      sink, &(loom_benchmark_event_t){
                .kind = LOOM_BENCHMARK_EVENT_FAILURE,
                .failure =
                    {
                        .run = run,
                        .stage = stage,
                        .kind = kind,
                        .message = message,
                        .testbench_plan = testbench_plan,
                        .planning_issues = planning_issues,
                        .planning_issue_count = planning_issue_count,
                    },
            });
}

iree_status_t loom_benchmark_event_sink_emit_benchmark_repetition(
    const loom_benchmark_event_sink_t* sink,
    const loom_benchmark_run_identity_t* run,
    const loom_benchmark_dispatch_comparison_candidate_t* candidate,
    const loom_benchmark_candidate_identity_t* baseline,
    iree_string_view_t comparison_group, iree_string_view_t method,
    iree_host_size_t order_index, iree_host_size_t repetition_index,
    char schedule_token, bool profile_suppressed,
    const loom_benchmark_benchmark_result_t* benchmark_result) {
  IREE_ASSERT_ARGUMENT(run);
  IREE_ASSERT_ARGUMENT(candidate);
  IREE_ASSERT_ARGUMENT(baseline);
  IREE_ASSERT_ARGUMENT(benchmark_result);
  return loom_benchmark_event_sink_emit(
      sink, &(loom_benchmark_event_t){
                .kind = LOOM_BENCHMARK_EVENT_BENCHMARK_REPETITION,
                .benchmark_repetition =
                    {
                        .run = run,
                        .candidate = candidate,
                        .baseline = baseline,
                        .comparison_group = comparison_group,
                        .method = method,
                        .order_index = order_index,
                        .repetition_index = repetition_index,
                        .schedule_token = schedule_token,
                        .profile_suppressed = profile_suppressed,
                        .benchmark_result = benchmark_result,
                    },
            });
}

iree_status_t loom_benchmark_event_sink_emit_comparison(
    const loom_benchmark_event_sink_t* sink,
    const loom_benchmark_run_identity_t* run,
    const loom_benchmark_dispatch_comparison_candidate_t* baseline,
    const loom_benchmark_dispatch_comparison_candidate_t* candidate,
    iree_string_view_t comparison_group, iree_string_view_t method) {
  IREE_ASSERT_ARGUMENT(run);
  IREE_ASSERT_ARGUMENT(baseline);
  IREE_ASSERT_ARGUMENT(candidate);
  return loom_benchmark_event_sink_emit(
      sink, &(loom_benchmark_event_t){
                .kind = LOOM_BENCHMARK_EVENT_COMPARISON,
                .comparison =
                    {
                        .run = run,
                        .baseline = baseline,
                        .candidate = candidate,
                        .comparison_group = comparison_group,
                        .method = method,
                    },
            });
}

static iree_status_t loom_benchmark_append_scenario_sample_row(
    const loom_benchmark_scenario_sample_event_t* event,
    iree_string_builder_t* output) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("row"), IREE_SV("sample")));
  IREE_RETURN_IF_ERROR(
      loom_benchmark_write_run_id_field_json(event->run, &object));
  IREE_RETURN_IF_ERROR(loom_benchmark_write_scenario_sample_fields_json(
      event->selection, event->work_item_index, event->benchmark_sample_ordinal,
      event->result, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  return loom_output_stream_write_char(&stream, '\n');
}

static iree_status_t loom_benchmark_jsonl_event_sink_emit(
    void* user_data, const loom_benchmark_event_t* event) {
  loom_benchmark_jsonl_event_sink_t* adapter =
      (loom_benchmark_jsonl_event_sink_t*)user_data;
  loom_benchmark_jsonl_sink_t* jsonl_sink = adapter->jsonl_sink;
  switch (event->kind) {
    case LOOM_BENCHMARK_EVENT_RUN:
      return loom_benchmark_jsonl_sink_end(
          jsonl_sink,
          loom_benchmark_append_run_row(
              event->run.run, event->run.dry_run, &event->run.sanitizer,
              loom_benchmark_jsonl_sink_begin(jsonl_sink)));
    case LOOM_BENCHMARK_EVENT_PLAN:
      return loom_benchmark_jsonl_sink_end(
          jsonl_sink,
          loom_benchmark_append_plan_row(
              event->plan.run, event->plan.module, event->plan.selection,
              event->plan.options, jsonl_sink->host_allocator,
              loom_benchmark_jsonl_sink_begin(jsonl_sink)));
    case LOOM_BENCHMARK_EVENT_SUMMARY:
      return loom_benchmark_jsonl_sink_end(
          jsonl_sink,
          loom_benchmark_append_summary_row(
              event->summary.run, event->summary.artifact_bundle,
              event->summary.planned_case_count,
              event->summary.planned_benchmark_count,
              event->summary.selected_benchmark_count,
              event->summary.logical_sample_count,
              event->summary.work_item_count, event->summary.failure_count,
              event->summary.failed_benchmark_count,
              event->summary.correctness_sample_count,
              event->summary.correctness_failed_sample_count,
              event->summary.dry_run,
              loom_benchmark_jsonl_sink_begin(jsonl_sink)));
    case LOOM_BENCHMARK_EVENT_DEVICE:
      return loom_benchmark_jsonl_sink_end(
          jsonl_sink, loom_benchmark_append_device_row(
                          event->device.run, event->device.context,
                          &adapter->device_row_state,
                          loom_benchmark_jsonl_sink_begin(jsonl_sink)));
    case LOOM_BENCHMARK_EVENT_COMPILE:
      return loom_benchmark_jsonl_sink_end(
          jsonl_sink,
          loom_benchmark_append_compile_row(
              event->compile.run, event->compile.candidate,
              event->compile.compile_item, event->compile.benchmark_plan,
              event->compile.case_plan, event->compile.provider,
              loom_benchmark_jsonl_sink_begin(jsonl_sink)));
    case LOOM_BENCHMARK_EVENT_SAMPLE:
      return loom_benchmark_jsonl_sink_end(
          jsonl_sink, loom_benchmark_append_sample_row(
                          event->sample.run, event->sample.candidate,
                          event->sample.work_item_index, event->sample.module,
                          event->sample.benchmark_plan, event->sample.case_plan,
                          event->sample.benchmark_sample_ordinal,
                          event->sample.sample_result,
                          loom_benchmark_jsonl_sink_begin(jsonl_sink)));
    case LOOM_BENCHMARK_EVENT_SCENARIO_SAMPLE:
      return loom_benchmark_jsonl_sink_end(
          jsonl_sink, loom_benchmark_append_scenario_sample_row(
                          &event->scenario_sample,
                          loom_benchmark_jsonl_sink_begin(jsonl_sink)));
    case LOOM_BENCHMARK_EVENT_BENCHMARK_RESULT:
      return loom_benchmark_jsonl_sink_end(
          jsonl_sink,
          loom_benchmark_append_benchmark_result(
              event->benchmark_result.run, event->benchmark_result.candidate,
              event->benchmark_result.work_item->work_item_index,
              event->benchmark_result.module,
              event->benchmark_result.benchmark_plan,
              event->benchmark_result.case_plan,
              event->benchmark_result.scenario_plan,
              event->benchmark_result.policy,
              event->benchmark_result.benchmark_result,
              event->benchmark_result.correctness_sample_count,
              event->benchmark_result.correctness_failed_sample_count,
              loom_benchmark_jsonl_sink_begin(jsonl_sink)));
    case LOOM_BENCHMARK_EVENT_PROFILE_REPLAY:
      return loom_benchmark_jsonl_sink_end(
          jsonl_sink,
          loom_benchmark_append_profile_replay_row(
              event->profile_replay.run, event->profile_replay.candidate,
              event->profile_replay.work_item_index,
              event->profile_replay.module,
              event->profile_replay.benchmark_plan,
              event->profile_replay.case_plan,
              event->profile_replay.scenario_plan, event->profile_replay.policy,
              event->profile_replay.benchmark_result,
              jsonl_sink->host_allocator,
              loom_benchmark_jsonl_sink_begin(jsonl_sink)));
    case LOOM_BENCHMARK_EVENT_FAILURE:
      return loom_benchmark_jsonl_sink_end(
          jsonl_sink,
          loom_benchmark_append_failure_row(
              event->failure.run, event->failure.stage, event->failure.kind,
              event->failure.message, event->failure.diagnostics,
              event->failure.testbench_plan, event->failure.planning_issues,
              event->failure.planning_issue_count,
              loom_benchmark_jsonl_sink_begin(jsonl_sink)));
    case LOOM_BENCHMARK_EVENT_BENCHMARK_REPETITION:
      return loom_benchmark_jsonl_sink_end(
          jsonl_sink, loom_benchmark_append_benchmark_repetition_row(
                          event->benchmark_repetition.run,
                          event->benchmark_repetition.candidate,
                          event->benchmark_repetition.baseline,
                          event->benchmark_repetition.comparison_group,
                          event->benchmark_repetition.method,
                          event->benchmark_repetition.order_index,
                          event->benchmark_repetition.repetition_index,
                          event->benchmark_repetition.schedule_token,
                          event->benchmark_repetition.profile_suppressed,
                          event->benchmark_repetition.benchmark_result,
                          loom_benchmark_jsonl_sink_begin(jsonl_sink)));
    case LOOM_BENCHMARK_EVENT_COMPARISON:
      return loom_benchmark_jsonl_sink_end(
          jsonl_sink,
          loom_benchmark_append_comparison_row(
              event->comparison.run, event->comparison.baseline,
              event->comparison.candidate, event->comparison.comparison_group,
              event->comparison.method,
              loom_benchmark_jsonl_sink_begin(jsonl_sink)));
    case LOOM_BENCHMARK_EVENT_WORK_PLAN:
      return iree_ok_status();
    case LOOM_BENCHMARK_EVENT_NONE:
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported benchmark event kind %d",
                              (int)event->kind);
  }
}

void loom_benchmark_jsonl_event_sink_initialize(
    loom_benchmark_jsonl_sink_t* jsonl_sink,
    loom_benchmark_jsonl_event_sink_t* out_adapter,
    loom_benchmark_event_sink_t* out_sink) {
  IREE_ASSERT_ARGUMENT(jsonl_sink);
  IREE_ASSERT_ARGUMENT(out_adapter);
  IREE_ASSERT_ARGUMENT(out_sink);
  *out_adapter = (loom_benchmark_jsonl_event_sink_t){
      .jsonl_sink = jsonl_sink,
  };
  *out_sink = (loom_benchmark_event_sink_t){
      .emit = loom_benchmark_jsonl_event_sink_emit,
      .user_data = out_adapter,
  };
}

void loom_benchmark_jsonl_event_sink_deinitialize(
    loom_benchmark_jsonl_event_sink_t* adapter) {
  IREE_ASSERT_ARGUMENT(adapter);
  *adapter = (loom_benchmark_jsonl_event_sink_t){0};
}
