// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// iree-benchmark-loom: correctness-gated benchmark runner for check.benchmark.

#include "loom/tools/iree-benchmark-loom/main.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "loom/ir/module.h"
#include "loom/tooling/execution/compile_report_capture.h"
#include "loom/tooling/io/file.h"
#include "loom/tooling/testbench/executor.h"
#include "loom/tools/iree-benchmark-loom/comparison_execution.h"
#include "loom/tools/iree-benchmark-loom/context.h"
#include "loom/tools/iree-benchmark-loom/diagnostics.h"
#include "loom/tools/iree-benchmark-loom/event.h"
#include "loom/tools/iree-benchmark-loom/help.h"
#include "loom/tools/iree-benchmark-loom/manifest.h"
#include "loom/tools/iree-benchmark-loom/model.h"
#include "loom/tools/iree-benchmark-loom/options.h"
#include "loom/tools/iree-benchmark-loom/output.h"
#include "loom/tools/iree-benchmark-loom/session.h"
#include "loom/tools/iree-benchmark-loom/snapshot.h"
#include "loom/tools/iree-benchmark-loom/work_execution.h"
#include "loom/tools/iree-benchmark-loom/work_plan.h"
#include "loom/verify/verify.h"

static iree_status_t iree_benchmark_loom_compile_report_options_initialize(
    const iree_benchmark_loom_options_t* options,
    loom_run_compile_report_capture_options_t* out_options) {
  loom_run_compile_report_capture_options_initialize(out_options);
  IREE_RETURN_IF_ERROR(loom_run_compile_report_capture_options_parse_request(
      options->compile_report, out_options));
  if (out_options->sink_format == LOOM_RUN_COMPILE_REPORT_SINK_FORMAT_TEXT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "iree-benchmark-loom emits structured JSON reports; use "
        "--compile_report=summary, details, json-summary, or json-details");
  }
  out_options->row_limit = options->compile_report_row_limit;
  return iree_ok_status();
}

int iree_benchmark_loom_main(
    int argc, char** argv,
    const iree_benchmark_loom_configuration_t* configuration) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t allocator = iree_allocator_system();
  iree_string_builder_t command_line_json;
  iree_string_builder_initialize(allocator, &command_line_json);
  iree_status_t status = iree_benchmark_loom_append_command_line_json(
      argc, argv, &command_line_json);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    iree_string_builder_deinitialize(&command_line_json);
    IREE_TRACE_ZONE_END(z0);
    IREE_TRACE_APP_EXIT(1);
    return 1;
  }

  iree_benchmark_loom_set_usage(configuration->tool_name);
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (iree_benchmark_loom_cli_flags_request_agents_md()) {
    iree_benchmark_loom_print_agents_md(stdout);
    iree_string_builder_deinitialize(&command_line_json);
    IREE_TRACE_ZONE_END(z0);
    IREE_TRACE_APP_EXIT(0);
    return 0;
  }

  iree_io_file_contents_t* contents = NULL;
  loom_run_session_t session = {0};
  loom_run_module_t run_module = {0};
  iree_benchmark_loom_artifact_bundle_t artifact_bundle = {0};
  iree_benchmark_loom_file_provider_t file_provider = {0};
  iree_benchmark_loom_hal_context_t hal_context = {0};
  iree_benchmark_loom_hal_context_initialize(configuration, allocator,
                                             &hal_context);
  iree_arena_allocator_t plan_arena;
  memset(&plan_arena, 0, sizeof(plan_arena));
  iree_arena_allocator_t execution_arena;
  memset(&execution_arena, 0, sizeof(execution_arena));
  iree_benchmark_loom_jsonl_sink_t jsonl_sink = {0};
  bool jsonl_sink_initialized = false;
  iree_benchmark_loom_jsonl_event_sink_t jsonl_event_sink = {0};
  iree_benchmark_loom_snapshot_sink_t snapshot_sink = {0};
  bool snapshot_sink_initialized = false;
  iree_benchmark_loom_event_sink_t event_sink = {0};
  iree_benchmark_loom_diagnostic_capture_t source_diagnostics = {0};
  iree_benchmark_loom_diagnostic_capture_initialize(allocator,
                                                    &source_diagnostics);
  iree_host_size_t planned_case_count = 0;
  iree_host_size_t planned_benchmark_count = 0;
  iree_host_size_t selected_benchmark_count = 0;
  iree_host_size_t logical_sample_count = 0;
  iree_host_size_t work_item_count = 0;
  iree_host_size_t failure_count = 0;
  iree_host_size_t failed_benchmark_count = 0;
  iree_host_size_t correctness_sample_count = 0;
  iree_host_size_t correctness_failed_sample_count = 0;
  int exit_code = 0;

  status = iree_ok_status();
  iree_benchmark_loom_options_t options = {0};
  status = iree_benchmark_loom_options_from_flags(&options);
  const bool compare_requested = !iree_string_view_is_empty(options.compare);
  if (argc > 2) {
    status = iree_status_join(
        status, iree_make_status(
                    IREE_STATUS_INVALID_ARGUMENT,
                    "iree-benchmark-loom accepts at most one input file or '-' "
                    "for stdin; got %d inputs",
                    argc - 1));
  }
  loom_run_compile_report_capture_options_t compile_report_options = {0};
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_compile_report_options_initialize(
        &options, &compile_report_options);
  }
  if (iree_status_is_ok(status)) {
    iree_benchmark_loom_artifact_bundle_options_t artifact_bundle_options = {
        .dir = options.artifact_bundle_dir,
        .policy = options.artifact_bundle_policy,
        .output_format = options.output_format,
    };
    status = iree_benchmark_loom_artifact_bundle_initialize(
        &artifact_bundle_options, allocator, &artifact_bundle);
    if (iree_status_is_ok(status)) {
      hal_context.artifact_bundle = &artifact_bundle;
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_session_initialize(configuration, allocator,
                                                    &session);
  }
  const iree_string_view_t input_path =
      argc < 2 ? iree_string_view_empty() : iree_make_cstring_view(argv[1]);
  const iree_string_view_t filename =
      (argc < 2 || iree_string_view_equal(input_path, IREE_SV("-")))
          ? IREE_SV("<stdin>")
          : input_path;
  char run_id_storage[32];
  snprintf(run_id_storage, sizeof(run_id_storage), "r%016" PRIx64,
           (uint64_t)iree_time_now());
  iree_string_view_t run_id = iree_make_cstring_view(run_id_storage);
  const iree_string_view_t results_output_path =
      iree_benchmark_loom_effective_results_output_path(options.output,
                                                        &artifact_bundle);
  const iree_string_view_t profile_artifacts_dir =
      iree_benchmark_loom_effective_profile_artifacts_dir(
          options.profile_artifacts_dir, &artifact_bundle);
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_file_provider_initialize(
        filename, run_id, options.file_output_dir,
        artifact_bundle.file_output_dir, &artifact_bundle, allocator,
        &file_provider);
  }
  const iree_benchmark_loom_run_identity_t run_identity = {
      .run_id = run_id,
      .source = filename,
      .results_path = iree_string_view_is_empty(results_output_path)
                          ? IREE_SV("-")
                          : results_output_path,
      .file_output_dir = file_provider.output_dir,
      .profile_artifacts_dir = profile_artifacts_dir,
      .artifact_bundle_dir = artifact_bundle.dir,
      .artifact_bundle_policy = iree_benchmark_loom_artifact_bundle_policy_name(
          artifact_bundle.policy),
  };
  if (iree_status_is_ok(status)) {
    switch (options.output_format) {
      case IREE_BENCHMARK_LOOM_OUTPUT_FORMAT_SNAPSHOT:
        status = iree_benchmark_loom_snapshot_sink_initialize(allocator,
                                                              &snapshot_sink);
        if (iree_status_is_ok(status)) {
          snapshot_sink_initialized = true;
          iree_benchmark_loom_snapshot_event_sink_initialize(&snapshot_sink,
                                                             &event_sink);
        }
        break;
      case IREE_BENCHMARK_LOOM_OUTPUT_FORMAT_JSONL:
        status = iree_benchmark_loom_jsonl_sink_initialize(
            results_output_path, allocator, &jsonl_sink);
        if (iree_status_is_ok(status)) {
          jsonl_sink_initialized = true;
          iree_benchmark_loom_jsonl_event_sink_initialize(
              &jsonl_sink, &jsonl_event_sink, &event_sink);
        }
        break;
      default:
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "unsupported benchmark output format %d",
                                  (int)options.output_format);
        break;
    }
    if (iree_status_is_ok(status)) {
      status = iree_benchmark_loom_event_sink_emit_run(
          &event_sink, &run_identity, options.dry_run,
          options.sample_compilation_mode);
    }
  }
  iree_string_view_t source = iree_string_view_empty();
  if (iree_status_is_ok(status)) {
    status = loom_tooling_read_input_file(input_path, allocator, &contents);
    if (iree_status_is_ok(status)) {
      source = loom_tooling_file_contents_string_view(contents);
    }
  }
  if (iree_status_is_ok(status)) {
    loom_run_module_parse_options_t parse_options = {0};
    loom_run_module_parse_options_initialize(&parse_options);
    parse_options.filename = filename;
    parse_options.source = source;
    parse_options.diagnostic_sink = (loom_diagnostic_sink_t){
        .fn = iree_benchmark_loom_diagnostic_capture_sink,
        .user_data = &source_diagnostics,
    };
    status = loom_run_module_parse(&session, &parse_options, &run_module);
    if (!iree_status_is_ok(status) && source_diagnostics.error_count != 0) {
      // The diagnostic sink owns the input rejection evidence; the status only
      // carries the same non-infrastructure failure.
      iree_status_free(status);
      status = iree_benchmark_loom_event_sink_emit_failure(
          &event_sink, &run_identity, IREE_SV("parse"), IREE_SV("diagnostics"),
          IREE_SV("input module has parse errors"), &source_diagnostics);
      ++failure_count;
      exit_code = 1;
    }
  }

  if (iree_status_is_ok(status) && failure_count == 0) {
    iree_benchmark_loom_diagnostic_capture_deinitialize(&source_diagnostics);
    iree_benchmark_loom_diagnostic_capture_initialize(allocator,
                                                      &source_diagnostics);
    loom_verify_options_t verify_options = {0};
    verify_options.sink = (loom_diagnostic_sink_t){
        .fn = iree_benchmark_loom_diagnostic_capture_sink,
        .user_data = &source_diagnostics,
    };
    verify_options.max_errors = 20;
    verify_options.source_resolver =
        loom_run_module_source_resolver(&run_module);
    loom_verify_result_t verify_result = {0};
    status =
        loom_verify_module(run_module.module, &verify_options, &verify_result);
    if (iree_status_is_ok(status) && verify_result.error_count != 0) {
      status = iree_benchmark_loom_event_sink_emit_failure(
          &event_sink, &run_identity, IREE_SV("verify"), IREE_SV("diagnostics"),
          IREE_SV("input module failed verification"), &source_diagnostics);
      ++failure_count;
      exit_code = 1;
    }
  }

  if (iree_status_is_ok(status) && failure_count == 0) {
    iree_arena_initialize(loom_run_session_block_pool(&session), &plan_arena);
    iree_arena_initialize(loom_run_session_block_pool(&session),
                          &execution_arena);
    loom_testbench_plan_options_t plan_options = {0};
    loom_testbench_plan_options_initialize(&plan_options);
    plan_options.max_samples_per_case = options.max_samples_per_case;
    loom_testbench_module_plan_t module_plan = {0};
    status = loom_testbench_plan_module(run_module.module, &plan_options,
                                        &plan_arena, &module_plan);
    if (iree_status_is_ok(status)) {
      planned_case_count = module_plan.case_count;
      planned_benchmark_count = module_plan.benchmark_count;
    }
    loom_testbench_case_execution_options_t execution_options = {0};
    loom_testbench_case_execution_options_initialize(&execution_options);
    execution_options.materializer.host_allocator = allocator;
    execution_options.materializer.open_read_file =
        (loom_testbench_file_open_callback_t){
            .fn = iree_benchmark_loom_open_file_for_read,
            .user_data = &file_provider,
        };
    execution_options.materializer.open_write_file =
        (loom_testbench_file_open_callback_t){
            .fn = iree_benchmark_loom_open_file_for_write,
            .user_data = &file_provider,
        };

    iree_benchmark_loom_work_plan_t work_plan = {0};
    bool work_plan_initialized = false;
    if (iree_status_is_ok(status)) {
      status = iree_benchmark_loom_work_plan_initialize(&module_plan, &options,
                                                        allocator, &work_plan);
      if (iree_status_is_ok(status)) {
        work_plan_initialized = true;
        selected_benchmark_count = work_plan.selected_benchmark_count;
        logical_sample_count = work_plan.logical_sample_count;
        work_item_count = work_plan.work_item_count;
      }
    }

    if (iree_status_is_ok(status) && compare_requested) {
      const iree_benchmark_loom_selected_benchmark_t* selections =
          work_plan.selected_benchmarks;
      const iree_host_size_t selection_count =
          work_plan.selected_benchmark_count;
      for (iree_host_size_t i = 0;
           iree_status_is_ok(status) && i < selection_count; ++i) {
        if (selections[i].policy.measure_kind !=
            IREE_BENCHMARK_LOOM_MEASURE_DISPATCH_COMPLETE) {
          status =
              iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                               "--compare benchmark `%.*s` must use measure = "
                               "\"dispatch_complete\"",
                               (int)selections[i].benchmark_plan->name.size,
                               selections[i].benchmark_plan->name.data);
          break;
        }
        status = iree_benchmark_loom_event_sink_emit_plan(
            &event_sink, &run_identity, module_plan.module, &selections[i],
            &options, options.sample_compilation_mode);
        if (iree_status_is_ok(status) && options.dry_run) {
          status = iree_benchmark_loom_event_sink_emit_device(
              &event_sink, &run_identity, &hal_context);
        }
      }
      if (iree_status_is_ok(status) && !options.dry_run) {
        const iree_benchmark_loom_comparison_execution_options_t
            comparison_execution_options = {
                .run = &run_identity,
                .selections = selections,
                .selection_count = selection_count,
                .module_plan = &module_plan,
                .benchmark_options = &options,
                .sample_compilation_mode = options.sample_compilation_mode,
                .interleave_mode = options.interleave_mode,
                .repetitions = options.repetitions,
                .hal_context = &hal_context,
                .session = &session,
                .filename = filename,
                .source = source,
                .compile_report_options = &compile_report_options,
                .case_execution_options = &execution_options,
                .execution_arena = &execution_arena,
                .host_allocator = allocator,
                .event_sink = &event_sink,
            };
        status = iree_benchmark_loom_run_dispatch_comparison(
            &comparison_execution_options, &correctness_sample_count,
            &correctness_failed_sample_count, &failed_benchmark_count);
      }
    }

    for (iree_host_size_t selection_index = 0;
         iree_status_is_ok(status) && !compare_requested &&
         selection_index < work_plan.selected_benchmark_count;
         ++selection_index) {
      const iree_benchmark_loom_selected_benchmark_t* selection =
          &work_plan.selected_benchmarks[selection_index];
      status = iree_benchmark_loom_event_sink_emit_plan(
          &event_sink, &run_identity, module_plan.module, selection, &options,
          options.sample_compilation_mode);
      if (iree_status_is_ok(status) && options.dry_run &&
          selection->policy.measure_kind ==
              IREE_BENCHMARK_LOOM_MEASURE_DISPATCH_COMPLETE) {
        status = iree_benchmark_loom_event_sink_emit_device(
            &event_sink, &run_identity, &hal_context);
      }
    }
    if (iree_status_is_ok(status) && options.dry_run) {
      status = iree_benchmark_loom_event_sink_emit_work_plan(
          &event_sink, &run_identity, module_plan.module, &work_plan);
    }
    if (iree_status_is_ok(status) && !compare_requested && !options.dry_run) {
      const iree_benchmark_loom_work_plan_execution_options_t
          work_execution_options = {
              .run = &run_identity,
              .module_plan = &module_plan,
              .work_plan = &work_plan,
              .benchmark_options = &options,
              .hal_context = &hal_context,
              .session = &session,
              .filename = filename,
              .source = source,
              .compile_report_options = &compile_report_options,
              .case_execution_options = &execution_options,
              .execution_arena = &execution_arena,
              .host_allocator = allocator,
              .event_sink = &event_sink,
          };
      status = iree_benchmark_loom_run_work_plan(
          &work_execution_options, &correctness_sample_count,
          &correctness_failed_sample_count, &failed_benchmark_count);
    }
    if (iree_status_is_ok(status) &&
        (failed_benchmark_count != 0 || correctness_failed_sample_count != 0)) {
      exit_code = 1;
    }
    if (work_plan_initialized) {
      iree_benchmark_loom_work_plan_deinitialize(&work_plan);
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_event_sink_emit_summary(
        &event_sink, &run_identity, &artifact_bundle, planned_case_count,
        planned_benchmark_count, selected_benchmark_count, logical_sample_count,
        work_item_count, failure_count, failed_benchmark_count,
        correctness_sample_count, correctness_failed_sample_count,
        options.dry_run, options.sample_compilation_mode);
  }
  if (iree_status_is_ok(status) && snapshot_sink_initialized) {
    status = iree_benchmark_loom_snapshot_sink_write(&snapshot_sink,
                                                     results_output_path);
  }
  if (iree_status_is_ok(status) && jsonl_sink_initialized) {
    status = iree_benchmark_loom_jsonl_sink_close(&jsonl_sink);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_write_artifact_bundle_manifest(
        &artifact_bundle, &run_identity, &hal_context, source,
        iree_string_builder_view(&command_line_json), options.dry_run,
        options.sample_compilation_mode, allocator);
  }
  if (iree_status_is_ok(status) && failure_count != 0) {
    exit_code = 1;
  }

  const bool had_error = !iree_status_is_ok(status);
  if (had_error) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = 1;
  }

  iree_benchmark_loom_diagnostic_capture_deinitialize(&source_diagnostics);
  if (jsonl_sink_initialized) {
    iree_benchmark_loom_jsonl_event_sink_deinitialize(&jsonl_event_sink);
  }
  if (snapshot_sink_initialized) {
    iree_benchmark_loom_snapshot_sink_deinitialize(&snapshot_sink);
  }
  if (jsonl_sink_initialized) {
    iree_benchmark_loom_jsonl_sink_deinitialize(&jsonl_sink);
  }
  iree_arena_deinitialize(&execution_arena);
  iree_arena_deinitialize(&plan_arena);
  loom_run_module_deinitialize(&run_module);
  iree_benchmark_loom_hal_context_deinitialize(&hal_context);
  iree_benchmark_loom_file_provider_deinitialize(&file_provider);
  iree_benchmark_loom_artifact_bundle_deinitialize(&artifact_bundle);
  iree_io_file_contents_free(contents);
  loom_run_session_deinitialize(&session);
  iree_string_builder_deinitialize(&command_line_json);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
