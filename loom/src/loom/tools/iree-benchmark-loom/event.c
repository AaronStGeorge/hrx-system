// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/event.h"

#include "loom/tools/iree-benchmark-loom/report.h"

iree_status_t iree_benchmark_loom_event_sink_emit(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_event_t* event) {
  IREE_ASSERT_ARGUMENT(sink);
  IREE_ASSERT_ARGUMENT(event);
  if (sink->emit == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "benchmark event sink has no emit callback");
  }
  return sink->emit(sink->user_data, event);
}

iree_status_t iree_benchmark_loom_event_sink_emit_run(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_run_identity_t* run, bool dry_run,
    iree_benchmark_loom_sample_compilation_mode_t sample_compilation_mode) {
  IREE_ASSERT_ARGUMENT(run);
  return iree_benchmark_loom_event_sink_emit(
      sink, &(iree_benchmark_loom_event_t){
                .kind = IREE_BENCHMARK_LOOM_EVENT_RUN,
                .run =
                    {
                        .run = run,
                        .dry_run = dry_run,
                        .sample_compilation_mode = sample_compilation_mode,
                    },
            });
}

iree_status_t iree_benchmark_loom_event_sink_emit_plan(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_run_identity_t* run, const loom_module_t* module,
    const iree_benchmark_loom_selected_benchmark_t* selection,
    const iree_benchmark_loom_options_t* options,
    iree_benchmark_loom_sample_compilation_mode_t sample_compilation_mode) {
  IREE_ASSERT_ARGUMENT(run);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(selection);
  IREE_ASSERT_ARGUMENT(options);
  return iree_benchmark_loom_event_sink_emit(
      sink, &(iree_benchmark_loom_event_t){
                .kind = IREE_BENCHMARK_LOOM_EVENT_PLAN,
                .plan =
                    {
                        .run = run,
                        .module = module,
                        .selection = selection,
                        .options = options,
                        .sample_compilation_mode = sample_compilation_mode,
                    },
            });
}

iree_status_t iree_benchmark_loom_event_sink_emit_summary(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_artifact_bundle_t* artifact_bundle,
    iree_host_size_t planned_case_count,
    iree_host_size_t planned_benchmark_count,
    iree_host_size_t selected_benchmark_count, iree_host_size_t failure_count,
    iree_host_size_t failed_benchmark_count,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count, bool dry_run,
    iree_benchmark_loom_sample_compilation_mode_t sample_compilation_mode) {
  IREE_ASSERT_ARGUMENT(run);
  return iree_benchmark_loom_event_sink_emit(
      sink, &(iree_benchmark_loom_event_t){
                .kind = IREE_BENCHMARK_LOOM_EVENT_SUMMARY,
                .summary =
                    {
                        .run = run,
                        .artifact_bundle = artifact_bundle,
                        .planned_case_count = planned_case_count,
                        .planned_benchmark_count = planned_benchmark_count,
                        .selected_benchmark_count = selected_benchmark_count,
                        .failure_count = failure_count,
                        .failed_benchmark_count = failed_benchmark_count,
                        .correctness_sample_count = correctness_sample_count,
                        .correctness_failed_sample_count =
                            correctness_failed_sample_count,
                        .dry_run = dry_run,
                        .sample_compilation_mode = sample_compilation_mode,
                    },
            });
}

static iree_status_t iree_benchmark_loom_jsonl_event_sink_emit(
    void* user_data, const iree_benchmark_loom_event_t* event) {
  iree_benchmark_loom_jsonl_event_sink_t* adapter =
      (iree_benchmark_loom_jsonl_event_sink_t*)user_data;
  iree_benchmark_loom_jsonl_sink_t* jsonl_sink = adapter->jsonl_sink;
  switch (event->kind) {
    case IREE_BENCHMARK_LOOM_EVENT_RUN:
      return iree_benchmark_loom_jsonl_sink_end(
          jsonl_sink, iree_benchmark_loom_append_run_row(
                          event->run.run, event->run.dry_run,
                          event->run.sample_compilation_mode,
                          iree_benchmark_loom_jsonl_sink_begin(jsonl_sink)));
    case IREE_BENCHMARK_LOOM_EVENT_PLAN:
      return iree_benchmark_loom_jsonl_sink_end(
          jsonl_sink,
          iree_benchmark_loom_append_plan_row(
              event->plan.run, &event->plan.selection->identity,
              event->plan.module, event->plan.selection->benchmark_plan,
              event->plan.selection->case_plan, &event->plan.selection->policy,
              event->plan.options, event->plan.sample_compilation_mode,
              jsonl_sink->host_allocator,
              iree_benchmark_loom_jsonl_sink_begin(jsonl_sink)));
    case IREE_BENCHMARK_LOOM_EVENT_SUMMARY:
      return iree_benchmark_loom_jsonl_sink_end(
          jsonl_sink,
          iree_benchmark_loom_append_summary_row(
              event->summary.run, event->summary.artifact_bundle,
              event->summary.planned_case_count,
              event->summary.planned_benchmark_count,
              event->summary.selected_benchmark_count,
              event->summary.failure_count,
              event->summary.failed_benchmark_count,
              event->summary.correctness_sample_count,
              event->summary.correctness_failed_sample_count,
              event->summary.dry_run, event->summary.sample_compilation_mode,
              iree_benchmark_loom_jsonl_sink_begin(jsonl_sink)));
    case IREE_BENCHMARK_LOOM_EVENT_NONE:
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported benchmark event kind %d",
                              (int)event->kind);
  }
}

void iree_benchmark_loom_jsonl_event_sink_initialize(
    iree_benchmark_loom_jsonl_sink_t* jsonl_sink,
    iree_benchmark_loom_jsonl_event_sink_t* out_adapter,
    iree_benchmark_loom_event_sink_t* out_sink) {
  IREE_ASSERT_ARGUMENT(jsonl_sink);
  IREE_ASSERT_ARGUMENT(out_adapter);
  IREE_ASSERT_ARGUMENT(out_sink);
  *out_adapter = (iree_benchmark_loom_jsonl_event_sink_t){
      .jsonl_sink = jsonl_sink,
  };
  *out_sink = (iree_benchmark_loom_event_sink_t){
      .emit = iree_benchmark_loom_jsonl_event_sink_emit,
      .user_data = out_adapter,
  };
}

void iree_benchmark_loom_jsonl_event_sink_deinitialize(
    iree_benchmark_loom_jsonl_event_sink_t* adapter) {
  IREE_ASSERT_ARGUMENT(adapter);
  *adapter = (iree_benchmark_loom_jsonl_event_sink_t){0};
}
