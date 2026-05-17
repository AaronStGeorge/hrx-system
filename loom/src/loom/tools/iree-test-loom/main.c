// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// iree-test-loom: executes check.case records from ordinary Loom modules.

#include "loom/tools/iree-test-loom/main.h"

#include <stdio.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "loom/ops/op_registry.h"
#include "loom/tooling/execution/hal/testbench_actual.h"
#include "loom/tooling/io/file.h"
#include "loom/tooling/testbench/executor.h"
#include "loom/util/stream.h"

IREE_FLAG(string, case, "",
          "Optional check.case symbol to execute, such as '@smoke'. Empty "
          "executes all cases in source order.");
IREE_FLAG(int32_t, sample, -1,
          "Optional concrete sample ordinal to execute for the selected case "
          "or cases. Negative executes all planned samples.");
IREE_FLAG(int32_t, max_samples_per_case,
          LOOM_TESTBENCH_DEFAULT_MAX_SAMPLES_PER_CASE,
          "Maximum number of samples planned per check.case.");
IREE_FLAG(string, pipeline, "default",
          "Pass pipeline used for HAL actual invocations. Use 'default', "
          "'none', '@symbol', or a comma-separated pass list.");

static iree_status_t iree_test_loom_register_context(void* user_data,
                                                     loom_context_t* context) {
  const iree_test_loom_configuration_t* configuration =
      (const iree_test_loom_configuration_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_op_registry_register_all_dialects(context));
  if (configuration->register_context.fn == NULL) {
    return iree_ok_status();
  }
  return configuration->register_context.fn(
      configuration->register_context.user_data, context);
}

static iree_string_view_t iree_test_loom_normalize_case_name(
    iree_string_view_t case_name) {
  case_name = iree_string_view_trim(case_name);
  if (iree_string_view_starts_with(case_name, IREE_SV("@"))) {
    return iree_string_view_substr(case_name, 1, IREE_HOST_SIZE_MAX);
  }
  return case_name;
}

static bool iree_test_loom_case_matches_selection(
    const loom_testbench_case_plan_t* case_plan,
    iree_string_view_t selected_case_name) {
  return iree_string_view_is_empty(selected_case_name) ||
         iree_string_view_equal(case_plan->name, selected_case_name);
}

static iree_status_t iree_test_loom_validate_sample_flag(
    iree_host_size_t sample_count, iree_host_size_t* out_sample_ordinal,
    bool* out_has_sample) {
  *out_sample_ordinal = 0;
  *out_has_sample = false;
  if (FLAG_sample < 0) {
    return iree_ok_status();
  }
  *out_sample_ordinal = (iree_host_size_t)FLAG_sample;
  *out_has_sample = true;
  if (*out_sample_ordinal >= sample_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "--sample=%" PRIhsz
                            " exceeds selected case sample count %" PRIhsz,
                            *out_sample_ordinal, sample_count);
  }
  return iree_ok_status();
}

static bool iree_test_loom_case_has_actual_invocation(
    const loom_testbench_case_plan_t* case_plan) {
  for (iree_host_size_t i = 0; i < case_plan->invocation_count; ++i) {
    if (case_plan->invocations[i].kind == LOOM_TESTBENCH_INVOCATION_ACTUAL) {
      return true;
    }
  }
  return false;
}

static iree_status_t iree_test_loom_configure_hal_actual_provider(
    const iree_test_loom_configuration_t* configuration,
    loom_run_session_t* session, const loom_run_module_t* run_module,
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_case_plan_t* case_plan,
    loom_run_hal_testbench_context_t* hal_context,
    loom_testbench_case_execution_options_t* execution_options,
    loom_run_hal_testbench_actual_provider_t* out_provider) {
  const loom_testbench_invocation_plan_t* actual_invocation = NULL;
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_select_actual_invocation(
      case_plan, &actual_invocation));
  IREE_RETURN_IF_ERROR(
      loom_run_hal_testbench_context_ensure_runtime(hal_context));

  execution_options->materializer.device = hal_context->runtime.device;
  execution_options->materializer.device_allocator =
      iree_hal_device_allocator(hal_context->runtime.device);
  execution_options->materializer.buffer_params =
      loom_run_hal_testbench_host_visible_buffer_params();

  const loom_run_hal_testbench_actual_provider_options_t provider_options = {
      .context = hal_context,
      .session = session,
      .target_environment = configuration->target_environment,
      .filename = run_module->filename,
      .source = run_module->source,
      .pipeline = iree_make_cstring_view(FLAG_pipeline),
      .test_module = module_plan->module,
      .actual_invocation = actual_invocation,
  };
  loom_run_hal_testbench_actual_provider_initialize(&provider_options,
                                                    out_provider);
  execution_options->invocation.invoke_actual =
      (loom_testbench_invocation_callback_t){
          .fn = loom_run_hal_testbench_actual_invoke,
          .user_data = out_provider,
      };
  return iree_ok_status();
}

static iree_status_t iree_test_loom_run_case_samples(
    const iree_test_loom_configuration_t* configuration,
    loom_run_session_t* session, const loom_run_module_t* run_module,
    const loom_testbench_module_plan_t* module_plan,
    iree_host_size_t case_index,
    const loom_testbench_case_execution_options_t* base_execution_options,
    loom_run_hal_testbench_context_t* hal_context,
    iree_arena_allocator_t* arena, iree_string_builder_t* sample_output,
    bool* inout_first_sample, iree_host_size_t* inout_sample_count,
    iree_host_size_t* inout_failed_sample_count) {
  const loom_testbench_case_plan_t* case_plan = &module_plan->cases[case_index];
  iree_host_size_t selected_sample_ordinal = 0;
  bool has_selected_sample = false;
  IREE_RETURN_IF_ERROR(iree_test_loom_validate_sample_flag(
      case_plan->sample_count, &selected_sample_ordinal, &has_selected_sample));

  iree_status_t status = iree_ok_status();
  loom_testbench_case_execution_options_t execution_options =
      *base_execution_options;
  loom_run_hal_testbench_actual_provider_t hal_actual_provider = {0};
  bool hal_actual_provider_initialized = false;
  if (iree_test_loom_case_has_actual_invocation(case_plan)) {
    status = iree_test_loom_configure_hal_actual_provider(
        configuration, session, run_module, module_plan, case_plan, hal_context,
        &execution_options, &hal_actual_provider);
    hal_actual_provider_initialized = iree_status_is_ok(status);
  }

  loom_testbench_prepared_case_t prepared_case = {0};
  if (iree_status_is_ok(status)) {
    status = loom_testbench_prepare_case_execution(
        &execution_options, module_plan, case_index, arena, &prepared_case);
  }

  bool executor_initialized = false;
  loom_testbench_case_executor_t executor = {0};
  if (iree_status_is_ok(status)) {
    status = loom_testbench_case_executor_initialize(
        &prepared_case, &execution_options, &executor);
    executor_initialized = iree_status_is_ok(status);
  }

  const iree_host_size_t begin_sample =
      has_selected_sample ? selected_sample_ordinal : 0;
  const iree_host_size_t end_sample = has_selected_sample
                                          ? selected_sample_ordinal + 1
                                          : case_plan->sample_count;
  for (iree_host_size_t sample_ordinal = begin_sample;
       iree_status_is_ok(status) && sample_ordinal < end_sample;
       ++sample_ordinal) {
    loom_testbench_case_sample_result_t sample_result = {0};
    status = loom_testbench_run_case_sample(&executor, sample_ordinal,
                                            &sample_result);
    if (iree_status_is_ok(status)) {
      if (!*inout_first_sample) {
        status = iree_string_builder_append_cstring(sample_output, ",");
      }
      if (iree_status_is_ok(status)) {
        loom_output_stream_t stream;
        loom_output_stream_for_builder(sample_output, &stream);
        status = loom_testbench_case_sample_result_write_json(&sample_result,
                                                              &stream);
      }
    }
    if (iree_status_is_ok(status)) {
      *inout_first_sample = false;
      ++*inout_sample_count;
      if (!sample_result.passed) {
        ++*inout_failed_sample_count;
      }
    }
  }

  if (executor_initialized) {
    loom_testbench_case_executor_deinitialize(&executor);
  }
  if (hal_actual_provider_initialized) {
    loom_run_hal_testbench_actual_provider_deinitialize(&hal_actual_provider);
  }
  iree_arena_reset(arena);
  return status;
}

static iree_status_t iree_test_loom_write_report(
    iree_host_size_t case_count, iree_host_size_t sample_count,
    iree_host_size_t failed_sample_count, iree_string_view_t samples,
    iree_string_builder_t* output) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      output, "{\"format\":\"loom.test.v0\""));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"case_count\":%" PRIhsz, case_count));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"sample_count\":%" PRIhsz, sample_count));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"failed_sample_count\":%" PRIhsz, failed_sample_count));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"samples\":["));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(output, samples));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "]}\n"));
  return iree_ok_status();
}

int iree_test_loom_main(int argc, char** argv,
                        const iree_test_loom_configuration_t* configuration) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_flags_set_usage(
      configuration->tool_name,
      "Executes check.case records from a normal Loom module.\n"
      "\n"
      "Usage:\n"
      "  iree-test-loom file.loom --case=@smoke\n"
      "  cat module.loom | iree-test-loom -\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_allocator_t allocator = iree_allocator_system();
  iree_io_file_contents_t* contents = NULL;
  loom_run_session_t session = {0};
  loom_run_module_t run_module = {0};
  loom_run_hal_testbench_context_t hal_context = {0};
  iree_arena_allocator_t plan_arena;
  memset(&plan_arena, 0, sizeof(plan_arena));
  iree_arena_allocator_t execution_arena;
  memset(&execution_arena, 0, sizeof(execution_arena));
  iree_string_builder_t sample_output;
  iree_string_builder_initialize(allocator, &sample_output);
  iree_string_builder_t report_output;
  iree_string_builder_initialize(allocator, &report_output);
  int exit_code = 0;

  iree_status_t status = iree_ok_status();
  if (argc > 2) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "iree-test-loom accepts at most one input file or '-' for stdin; got "
        "%d inputs",
        argc - 1);
  }
  if (iree_status_is_ok(status) && FLAG_max_samples_per_case <= 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--max_samples_per_case must be positive; got "
                              "%d",
                              (int)FLAG_max_samples_per_case);
  }

  if (iree_status_is_ok(status)) {
    loom_run_session_options_t session_options = {0};
    loom_run_session_options_initialize(&session_options);
    session_options.host_allocator = allocator;
    session_options.register_context = (loom_run_register_context_callback_t){
        .fn = iree_test_loom_register_context,
        .user_data = (void*)configuration,
    };
    session_options.initialize_low_descriptor_registry =
        configuration->initialize_low_descriptor_registry;
    status = loom_run_session_initialize(&session_options, &session);
  }
  if (iree_status_is_ok(status)) {
    loom_run_hal_testbench_context_initialize(
        configuration->hal_artifact_provider_registry, allocator, &hal_context);
  }

  const iree_string_view_t input_path =
      argc < 2 ? iree_string_view_empty() : iree_make_cstring_view(argv[1]);
  const iree_string_view_t filename =
      (argc < 2 || iree_string_view_equal(input_path, IREE_SV("-")))
          ? IREE_SV("<stdin>")
          : input_path;
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
    status = loom_run_module_parse(&session, &parse_options, &run_module);
  }

  if (iree_status_is_ok(status)) {
    iree_arena_initialize(loom_run_session_block_pool(&session), &plan_arena);
    iree_arena_initialize(loom_run_session_block_pool(&session),
                          &execution_arena);
    loom_testbench_plan_options_t plan_options = {0};
    loom_testbench_plan_options_initialize(&plan_options);
    plan_options.max_samples_per_case =
        (iree_host_size_t)FLAG_max_samples_per_case;
    loom_testbench_module_plan_t module_plan = {0};
    status = loom_testbench_plan_module(run_module.module, &plan_options,
                                        &plan_arena, &module_plan);
    const iree_string_view_t selected_case_name =
        iree_test_loom_normalize_case_name(iree_make_cstring_view(FLAG_case));
    loom_testbench_case_execution_options_t execution_options = {0};
    loom_testbench_case_execution_options_initialize(&execution_options);
    execution_options.materializer.host_allocator = allocator;

    iree_host_size_t selected_case_count = 0;
    iree_host_size_t sample_count = 0;
    iree_host_size_t failed_sample_count = 0;
    bool first_sample = true;
    for (iree_host_size_t case_index = 0;
         iree_status_is_ok(status) && case_index < module_plan.case_count;
         ++case_index) {
      const loom_testbench_case_plan_t* case_plan =
          &module_plan.cases[case_index];
      if (!iree_test_loom_case_matches_selection(case_plan,
                                                 selected_case_name)) {
        continue;
      }
      ++selected_case_count;
      status = iree_test_loom_run_case_samples(
          configuration, &session, &run_module, &module_plan, case_index,
          &execution_options, &hal_context, &execution_arena, &sample_output,
          &first_sample, &sample_count, &failed_sample_count);
    }
    if (iree_status_is_ok(status) && selected_case_count == 0) {
      status = iree_make_status(
          IREE_STATUS_NOT_FOUND, "no check.case matched '%.*s'",
          (int)selected_case_name.size, selected_case_name.data);
    }
    if (iree_status_is_ok(status)) {
      status = iree_test_loom_write_report(
          selected_case_count, sample_count, failed_sample_count,
          iree_string_builder_view(&sample_output), &report_output);
    }
    if (iree_status_is_ok(status)) {
      status =
          loom_tooling_write_stdout(iree_string_builder_view(&report_output));
    }
    if (iree_status_is_ok(status) && failed_sample_count != 0) {
      exit_code = 1;
    }
  }

  const bool had_error = !iree_status_is_ok(status);
  if (had_error) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = 1;
  }

  iree_string_builder_deinitialize(&report_output);
  iree_string_builder_deinitialize(&sample_output);
  iree_arena_deinitialize(&execution_arena);
  iree_arena_deinitialize(&plan_arena);
  loom_run_hal_testbench_context_deinitialize(&hal_context);
  loom_run_module_deinitialize(&run_module);
  iree_io_file_contents_free(contents);
  loom_run_session_deinitialize(&session);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
