// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "pass_run.h"

#include "compile_report.h"
#include "context.h"
#include "diagnostic.h"
#include "iree/io/vec_stream.h"
#include "loom/pass/interpreter.h"
#include "loom/pass/report.h"
#include "loom/target/predicate.h"
#include "loomc/iree.h"
#include "module.h"
#include "option_chain.h"
#include "pass_program.h"
#include "pass_trace.h"
#include "result.h"
#include "target.h"
#include "workspace.h"

static iree_status_t loomc_pass_report_write(void* user_data,
                                             iree_string_view_t text) {
  return iree_io_stream_write((iree_io_stream_t*)user_data, text.size,
                              text.data);
}

static loomc_status_t loomc_pass_report_add_artifact(
    const loom_pass_report_t* report,
    const loomc_pass_report_options_t* options, loomc_result_t* result) {
  iree_io_stream_t* storage = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_WRITABLE, 4096,
      iree_allocator_from_loomc(loomc_result_allocator(result)), &storage)));
  loom_output_stream_t stream = {
      .write = loomc_pass_report_write,
      .user_data = storage,
  };
  loomc_status_t status =
      loomc_status_from_iree(loom_pass_report_format_json(report, &stream));
  iree_byte_sequence_t* contents = NULL;
  if (loomc_status_is_ok(status)) {
    status = loomc_status_from_iree(
        iree_io_vec_stream_move_contents(storage, &contents));
  }
  if (loomc_status_is_ok(status)) {
    const loomc_artifact_t artifact = {
        .kind = LOOMC_ARTIFACT_KIND_REPORT,
        .format =
            loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_PASS_REPORT_JSON),
        .identifier = options->identifier,
        .contents = loomc_byte_sequence_from_iree(contents),
    };
    status = loomc_result_add_artifact(result, &artifact);
  }
  iree_byte_sequence_release(contents);
  iree_io_stream_release(storage);
  return status;
}

static loomc_status_t loomc_pass_program_execute_into_result(
    const loomc_pass_program_t* pass_program, loomc_workspace_t* workspace,
    loomc_module_t* module,
    const loom_pass_environment_capability_t* launch_config_capability,
    const loomc_option_chain_t* options, loomc_result_t* result) {
  loomc_context_t* context = loomc_module_context(module);
  loom_module_t* internal_module = loomc_module_loom_module(module);
  loom_source_resolver_t source_resolver = loomc_module_source_resolver(module);
  loom_function_version_owner_t* function_version_owner =
      loomc_module_function_version_owner(module);
  const loomc_target_pass_environment_t* target_pass_environment =
      loomc_context_target_pass_environment(context);
  loomc_diagnostic_capture_t capture;
  loomc_diagnostic_capture_initialize(
      result, /*source=*/NULL, internal_module, source_resolver,
      LOOM_EMITTER_PASS,
      target_pass_environment
          ? &target_pass_environment->diagnostic_type_print_options
          : NULL,
      &capture);
  loom_codegen_pass_environment_storage_t codegen_environment_storage = {0};
  const loom_pass_environment_t pass_environment =
      loomc_codegen_pass_environment_storage_initialize(
          target_pass_environment,
          loomc_context_cleanup_pattern_registry(context),
          function_version_owner, launch_config_capability,
          options->compile_report, &codegen_environment_storage);
  loom_target_pass_predicate_provider_storage_t predicate_storage = {0};
  loom_pass_predicate_provider_t predicate_provider = {0};
  if (target_pass_environment != NULL) {
    loom_target_pass_predicate_provider_storage_initialize(
        loomc_workspace_block_pool(workspace), &predicate_storage);
    predicate_provider =
        loom_target_pass_predicate_provider(&predicate_storage);
  }
  loomc_pass_trace_state_t pass_trace_state = {0};
  loom_pass_trace_t* pass_trace = NULL;
  if (options->pass_trace != NULL) {
    loomc_pass_trace_state_initialize(
        options->pass_trace, loomc_pass_program_trace_stage(pass_program),
        target_pass_environment, &function_version_owner->list,
        loomc_workspace_block_pool(workspace), &pass_trace_state);
    pass_trace = &pass_trace_state.trace;
  }
  const loom_pass_interpreter_options_t interpreter_options = {
      .block_pool = loomc_workspace_block_pool(workspace),
      .predicate_provider = predicate_provider,
      .diagnostic_emitter =
          {
              .fn = loomc_diagnostic_capture_emission,
              .user_data = &capture,
          },
      .environment = pass_environment,
      .function_versions = &function_version_owner->list,
      .trace = pass_trace,
      .report = options->pass_report,
  };
  loom_pass_run_result_t run_result = {0};
  loomc_status_t status =
      loomc_status_from_iree(loom_pass_interpreter_run_program(
          loomc_pass_program_loom_pass_program(pass_program), internal_module,
          &interpreter_options, &run_result));
  if (!loomc_status_is_ok(status) && !pass_trace_state.callback_failed &&
      loomc_status_is_result_diagnostic(status)) {
    if (run_result.error_count == 0) {
      status = loomc_result_fail_status_diagnostic_consume(
          result, /*source=*/NULL, LOOMC_DIAGNOSTIC_SEVERITY_ERROR,
          loomc_make_cstring_view("PASS_PROGRAM/EXECUTION"), status);
    } else {
      loomc_result_set_status_code(result, loomc_status_code(status));
      loomc_status_free(status);
      status = loomc_ok_status();
    }
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
      run_result.error_count != 0) {
    loomc_result_set_status_code(result, LOOMC_STATUS_FAILED_PRECONDITION);
  }
  loomc_module_invalidate_target_verification(module);
  return status;
}

static loomc_status_t loomc_pass_program_run_with_reports(
    const loomc_pass_program_t* pass_program, loomc_workspace_t* workspace,
    loomc_module_t* module,
    const loom_pass_environment_capability_t* launch_config_capability,
    const loomc_option_chain_t* options, loomc_result_t* result) {
  loom_pass_report_t pass_report = {0};
  loomc_compile_report_state_t compile_report = {0};
  loomc_option_chain_t report_options = *options;
  if (options->pass_report_options) {
    loom_pass_report_initialize(loomc_workspace_block_pool(workspace),
                                &pass_report);
    pass_report.include_timing = iree_all_bits_set(
        options->pass_report_options->flags, LOOMC_PASS_REPORT_FLAG_TIMING);
    report_options.pass_report = &pass_report;
  }
  if (options->compile_report_options &&
      options->compile_report_options->mode != LOOMC_COMPILE_REPORT_MODE_NONE) {
    loomc_compile_report_state_initialize(
        options->compile_report_options->mode,
        options->compile_report_options->format,
        loomc_workspace_block_pool(workspace), result, &compile_report);
    const loomc_target_pass_environment_t* target_pass_environment =
        loomc_context_target_pass_environment(loomc_module_context(module));
    compile_report.diagnostic_text_print_options =
        target_pass_environment
            ? &target_pass_environment->diagnostic_type_print_options
            : NULL;
    report_options.compile_report = &compile_report.report;
  }
  loomc_status_t status = loomc_pass_program_execute_into_result(
      pass_program, workspace, module, launch_config_capability,
      &report_options, result);
  if (loomc_status_is_ok(status) && options->pass_report_options) {
    status = loomc_pass_report_add_artifact(
        &pass_report, options->pass_report_options, result);
  }
  if (loomc_status_is_ok(status) && compile_report.mode) {
    loom_target_compile_report_record_status(
        &compile_report.report,
        (iree_status_code_t)loomc_result_status_code(result));
    status =
        loomc_compile_report_record_config_bindings(&compile_report, module);
    if (loomc_status_is_ok(status)) {
      status = loomc_compile_report_state_finish(
          &compile_report, options->compile_report_options->identifier);
    }
  }
  loomc_compile_report_state_deinitialize(&compile_report);
  if (options->pass_report_options) {
    loom_pass_report_deinitialize(&pass_report);
  }
  return status;
}

loomc_status_t loomc_pass_program_run_into_result(
    const loomc_pass_program_t* pass_program, loomc_workspace_t* workspace,
    loomc_module_t* module,
    const loom_pass_environment_capability_t* launch_config_capability,
    const loomc_option_chain_t* options, loomc_result_t* result) {
  if (options->pass_report_options || (options->compile_report_options &&
                                       options->compile_report_options->mode !=
                                           LOOMC_COMPILE_REPORT_MODE_NONE)) {
    return loomc_pass_program_run_with_reports(pass_program, workspace, module,
                                               launch_config_capability,
                                               options, result);
  }
  return loomc_pass_program_execute_into_result(pass_program, workspace, module,
                                                launch_config_capability,
                                                options, result);
}

static loomc_status_t loomc_pass_run_options_resolve(
    const loomc_pass_run_options_t* options,
    loomc_option_chain_t* out_option_chain) {
  if (options != NULL) {
    if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
        options->type != LOOMC_STRUCTURE_TYPE_PASS_RUN_OPTIONS) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "pass run options have an unknown structure type");
    }
    if (options->structure_size != 0 &&
        options->structure_size < sizeof(*options)) {
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "pass run options structure_size is too small");
    }
  }
  return loomc_option_chain_resolve(
      options ? options->next : NULL,
      LOOMC_OPTION_CHAIN_ALLOW_PASS_TRACE |
          LOOMC_OPTION_CHAIN_ALLOW_PASS_REPORT |
          LOOMC_OPTION_CHAIN_ALLOW_COMPILE_REPORT |
          LOOMC_OPTION_CHAIN_ALLOW_NATIVE_PASS_REPORT |
          LOOMC_OPTION_CHAIN_ALLOW_NATIVE_COMPILE_REPORT,
      out_option_chain);
}

loomc_status_t loomc_pass_program_run(const loomc_pass_program_t* pass_program,
                                      loomc_workspace_t* workspace,
                                      loomc_module_t* module,
                                      const loomc_pass_run_options_t* options,
                                      loomc_allocator_t allocator,
                                      loomc_result_t** out_result) {
  if (out_result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_result must not be NULL");
  }
  *out_result = NULL;
  if (pass_program == NULL || workspace == NULL || module == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "pass program, workspace, and module must not be NULL");
  }
  loomc_context_t* context = loomc_module_context(module);
  if (loomc_pass_program_context(pass_program) != context) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "pass program was created with another context");
  }
  loomc_option_chain_t option_chain = {0};
  LOOMC_RETURN_IF_ERROR(loomc_pass_run_options_resolve(options, &option_chain));
  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_result_create(
      LOOMC_STATUS_OK, loomc_context_source_retention(context), allocator,
      &result));
  loomc_status_t status = loomc_module_admit(
      module, loomc_context_target_environment(context), result);
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_pass_program_run_into_result(
        pass_program, workspace, module, /*launch_config_capability=*/NULL,
        &option_chain, result);
  }
  if (!loomc_status_is_ok(status) || !loomc_result_succeeded(result)) {
    loomc_module_invalidate_verification(module);
    loomc_module_invalidate_compilation(module);
  }
  if (loomc_status_is_ok(status)) {
    *out_result = result;
  } else {
    loomc_result_release(result);
  }
  return status;
}
