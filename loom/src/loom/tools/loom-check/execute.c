// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/execute.h"

#include "loom/codegen/low/allocation_pass.h"
#include "loom/codegen/low/lower.h"
#include "loom/codegen/low/lower_pass.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/codegen/low/verify.h"
#include "loom/error/diagnostic.h"
#include "loom/error/json_sink.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/pass/builtin_registry.h"
#include "loom/pass/pipeline.h"
#include "loom/pass/tooling.h"
#include "loom/target/presets.h"
#include "loom/testing/diff.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/requirements.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"
#include "loom/verify/verify.h"

void loom_check_result_initialize(iree_allocator_t allocator,
                                  loom_check_result_t* out_result) {
  memset(out_result, 0, sizeof(*out_result));
  iree_string_builder_initialize(allocator, &out_result->detail);
  iree_string_builder_initialize(allocator, &out_result->diff_hunk_json);
  iree_string_builder_initialize(allocator, &out_result->actual_output);
  iree_string_builder_initialize(allocator, &out_result->update_edit.text);
  iree_string_builder_initialize(allocator, &out_result->annotation_edits.json);
  iree_string_builder_initialize(allocator, &out_result->diagnostic_json);
}

void loom_check_result_deinitialize(loom_check_result_t* result) {
  iree_string_builder_deinitialize(&result->diagnostic_json);
  iree_string_builder_deinitialize(&result->annotation_edits.json);
  iree_string_builder_deinitialize(&result->update_edit.text);
  iree_string_builder_deinitialize(&result->actual_output);
  iree_string_builder_deinitialize(&result->diff_hunk_json);
  iree_string_builder_deinitialize(&result->detail);
}

static iree_status_t loom_check_write_diff_hunk_json(
    const loom_diff_result_t* diff, const loom_diff_hunk_t* hunk,
    loom_output_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(diff);
  IREE_ASSERT_ARGUMENT(hunk);
  IREE_ASSERT_ARGUMENT(stream);

  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      "\"expected_start_line\": %zu, \"expected_line_count\": %zu, "
      "\"actual_start_line\": %zu, \"actual_line_count\": %zu, \"lines\": [",
      hunk->expected_start_line, hunk->expected_line_count,
      hunk->actual_start_line, hunk->actual_line_count));
  for (iree_host_size_t i = 0; i < hunk->line_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
    }
    const loom_diff_hunk_line_t* line = &diff->lines[hunk->line_offset + i];
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, "{\"kind\": "));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
        stream, loom_diff_hunk_line_kind_name(line->kind)));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ", \"text\": "));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(stream, line->text));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]}"));
  return iree_ok_status();
}

static iree_status_t loom_check_write_diff_json_hunks(
    const loom_diff_result_t* diff, loom_check_result_t* result) {
  IREE_ASSERT_ARGUMENT(diff);
  IREE_ASSERT_ARGUMENT(result);

  loom_output_stream_t stream;
  loom_output_stream_for_builder(&result->diff_hunk_json, &stream);
  for (iree_host_size_t i = 0; i < diff->hunk_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ",\n"));
    }
    IREE_RETURN_IF_ERROR(
        loom_check_write_diff_hunk_json(diff, &diff->hunks[i], &stream));
  }
  result->diff_hunk_count = diff->hunk_count;
  return iree_ok_status();
}

iree_status_t loom_check_result_record_diff(iree_string_view_t expected,
                                            iree_string_view_t actual,
                                            iree_allocator_t allocator,
                                            loom_check_result_t* result) {
  IREE_ASSERT_ARGUMENT(result);

  loom_diff_result_t diff = {0};
  iree_status_t status = loom_diff_compute(
      expected, actual, LOOM_DIFF_DEFAULT_CONTEXT, allocator, &diff);
  if (iree_status_is_ok(status)) {
    status = loom_diff_format_result(&diff, &result->detail);
  }
  if (iree_status_is_ok(status)) {
    status = loom_check_write_diff_json_hunks(&diff, result);
  }
  loom_diff_result_deinitialize(allocator, &diff);
  return status;
}

static iree_status_t loom_check_append_diagnostic_json(
    loom_check_result_t* result, const loom_diagnostic_t* diagnostic) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&result->diagnostic_json, &stream);
  if (result->diagnostic_count > 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ",\n"));
  }
  IREE_RETURN_IF_ERROR(loom_diagnostic_json_write_object(
      &stream, diagnostic,
      (loom_type_formatter_t){loom_type_format_minimal, NULL}));
  ++result->diagnostic_count;
  return iree_ok_status();
}

iree_status_t loom_check_diagnostic_capture_sink(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  loom_check_diagnostic_capture_t* capture =
      (loom_check_diagnostic_capture_t*)user_data;
  if (capture->detail) {
    loom_output_stream_t stream;
    loom_output_stream_for_builder(capture->detail, &stream);
    IREE_RETURN_IF_ERROR(loom_diagnostic_format(diagnostic, &stream));
  }
  if (capture->result) {
    IREE_RETURN_IF_ERROR(
        loom_check_append_diagnostic_json(capture->result, diagnostic));
  }
  return iree_ok_status();
}

iree_status_t loom_check_context_initialize(
    const loom_check_environment_t* environment, loom_context_t* context) {
  if (environment == NULL || environment->register_context.fn == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "loom-check context registration is required");
  }
  IREE_RETURN_IF_ERROR(environment->register_context.fn(
      environment->register_context.user_data, context));
  return loom_context_finalize(context);
}

iree_status_t loom_check_environment_initialize_low_descriptor_registry(
    const loom_check_environment_t* environment,
    loom_target_low_descriptor_registry_t* out_registry) {
  if (out_registry == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor registry output is required");
  }
  *out_registry = (loom_target_low_descriptor_registry_t){0};
  if (environment == NULL ||
      environment->initialize_low_descriptor_registry.fn == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "loom-check low descriptor registry is required");
  }
  return environment->initialize_low_descriptor_registry.fn(
      environment->initialize_low_descriptor_registry.user_data, out_registry);
}

iree_status_t loom_check_environment_initialize_low_lower_policy_registry(
    const loom_check_environment_t* environment,
    loom_low_lower_policy_registry_t* out_registry) {
  if (out_registry == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low lower policy registry output is required");
  }
  *out_registry = (loom_low_lower_policy_registry_t){0};
  if (environment == NULL ||
      environment->initialize_low_lower_policy_registry.fn == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "loom-check low lower policy registry is required");
  }
  return environment->initialize_low_lower_policy_registry.fn(
      environment->initialize_low_lower_policy_registry.user_data,
      out_registry);
}

//===----------------------------------------------------------------------===//
// Execution
//===----------------------------------------------------------------------===//

typedef struct loom_check_pass_pipeline_config_t {
  // Low allocation pass configuration borrowed by matching pipeline entries.
  loom_low_materialize_allocation_pass_config_t* low_allocation_config;
  // Source-to-low pass configuration borrowed by matching pipeline entries.
  loom_low_source_to_low_pass_config_t* low_source_to_low_config;
} loom_check_pass_pipeline_config_t;

static iree_status_t loom_check_configure_pass_instruction(
    void* user_data, const loom_pass_program_instruction_t* instruction,
    void** out_pass_user_data) {
  IREE_ASSERT_ARGUMENT(user_data);
  IREE_ASSERT_ARGUMENT(instruction);
  IREE_ASSERT_ARGUMENT(out_pass_user_data);

  loom_check_pass_pipeline_config_t* config =
      (loom_check_pass_pipeline_config_t*)user_data;
  *out_pass_user_data = NULL;
  if (instruction->kind == LOOM_PASS_PROGRAM_INSTRUCTION_INVOKE &&
      iree_string_view_equal(instruction->invoke.descriptor->key,
                             IREE_SV("low-materialize-allocation"))) {
    *out_pass_user_data = config->low_allocation_config;
  } else if (instruction->kind == LOOM_PASS_PROGRAM_INSTRUCTION_INVOKE &&
             iree_string_view_equal(instruction->invoke.descriptor->key,
                                    IREE_SV("source-to-low"))) {
    *out_pass_user_data = config->low_source_to_low_config;
  }
  return iree_ok_status();
}

static bool loom_check_pass_requirement_is_satisfied(
    void* user_data, iree_string_view_t requirement) {
  loom_check_pass_pipeline_config_t* config =
      (loom_check_pass_pipeline_config_t*)user_data;
  return config &&
         (loom_low_materialize_allocation_pass_config_satisfies_requirement(
              config->low_allocation_config, requirement) ||
          loom_low_source_to_low_pass_config_satisfies_requirement(
              config->low_source_to_low_config, requirement));
}

static bool loom_check_pass_descriptor_declares_requirement(
    const loom_pass_descriptor_t* descriptor, iree_string_view_t requirement) {
  if (!descriptor) {
    return false;
  }
  for (uint16_t i = 0; i < descriptor->requirement_count; ++i) {
    if (iree_string_view_equal(descriptor->requirement_defs[i].key,
                               requirement)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_check_pipeline_requires_low_lower_policy_registry(
    iree_string_view_t pipeline, bool* out_required) {
  IREE_ASSERT_ARGUMENT(out_required);
  *out_required = false;
  const loom_pass_registry_t* pass_registry = loom_pass_builtin_registry();
  iree_string_view_t remaining = pipeline;
  while (!iree_string_view_is_empty(iree_string_view_trim(remaining))) {
    loom_pass_pipeline_entry_spec_t spec = {0};
    bool has_entry = false;
    IREE_RETURN_IF_ERROR(
        loom_pass_pipeline_consume_entry(&remaining, &spec, &has_entry));
    if (!has_entry) {
      return iree_ok_status();
    }
    const loom_pass_descriptor_t* descriptor = NULL;
    IREE_RETURN_IF_ERROR(
        loom_pass_registry_lookup(pass_registry, spec.name, &descriptor));
    if (loom_check_pass_descriptor_declares_requirement(
            descriptor,
            IREE_SV(
                LOOM_LOW_PASS_REQUIREMENT_TARGET_LOW_LOWER_POLICY_REGISTRY))) {
      *out_required = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

iree_status_t loom_check_execute_case(
    const loom_check_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    const loom_check_environment_t* environment, loom_context_t* context,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_check_result_t* result) {
  IREE_ASSERT_ARGUMENT(test_case);
  IREE_ASSERT_ARGUMENT(report);
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(result);

  bool continue_execution = true;
  IREE_RETURN_IF_ERROR(loom_check_preflight_requirements(
      test_case, environment, allocator, result, &continue_execution));
  if (!continue_execution) {
    return iree_ok_status();
  }

  switch (test_case->mode) {
    case LOOM_CHECK_MODE_ROUNDTRIP: {
      IREE_RETURN_IF_ERROR(loom_check_execute_roundtrip(
          test_case, filename, context, block_pool, allocator, result));
      break;
    }
    case LOOM_CHECK_MODE_VERIFY: {
      IREE_RETURN_IF_ERROR(loom_check_execute_verify(
          test_case, case_index, report, filename, environment, context,
          block_pool, allocator, result));
      break;
    }
    case LOOM_CHECK_MODE_PASS: {
      IREE_RETURN_IF_ERROR(loom_check_execute_pass(
          test_case, case_index, report, filename, environment, context,
          block_pool, allocator, result));
      break;
    }
    case LOOM_CHECK_MODE_FORMAT: {
      IREE_RETURN_IF_ERROR(loom_check_execute_format(
          test_case, filename, context, block_pool, allocator, result));
      break;
    }
    case LOOM_CHECK_MODE_EMIT: {
      IREE_RETURN_IF_ERROR(loom_check_execute_emit(
          test_case, case_index, report, filename, environment, context,
          block_pool, allocator, result));
      break;
    }
    case LOOM_CHECK_MODE_RUN: {
      IREE_RETURN_IF_ERROR(
          loom_check_execute_run(test_case, environment, allocator, result));
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown check mode: %d", (int)test_case->mode);
  }

  // XFAIL inversion: expected failure that fails is a pass, expected
  // failure that passes is a failure.
  if (test_case->xfail) {
    result->final_outcome = (result->raw_outcome == LOOM_CHECK_FAIL)
                                ? LOOM_CHECK_PASS
                                : LOOM_CHECK_FAIL;
  } else {
    result->final_outcome = result->raw_outcome;
  }

  return iree_ok_status();
}

// Runs a comma-separated pass pipeline on a module.
static iree_status_t loom_check_run_pipeline(
    iree_string_view_t pipeline, iree_arena_block_pool_t* block_pool,
    iree_diagnostic_emitter_t diagnostic_emitter,
    const loom_target_low_descriptor_registry_t* low_registry,
    const loom_low_lower_policy_registry_t* low_lower_policy_registry,
    loom_module_t* module) {
  const loom_pass_registry_t* pass_registry = loom_pass_builtin_registry();
  loom_low_materialize_allocation_pass_config_t
      low_materialize_allocation_config = {
          .descriptor_registry = &low_registry->registry,
      };
  loom_low_source_to_low_pass_config_t low_source_to_low_config = {
      .descriptor_registry = &low_registry->registry,
      .policy_registry = low_lower_policy_registry,
  };
  loom_check_pass_pipeline_config_t pipeline_config = {
      .low_allocation_config = &low_materialize_allocation_config,
      .low_source_to_low_config = &low_source_to_low_config,
  };

  loom_pass_tool_run_options_t options = {
      .registry = pass_registry,
      .requirement_provider =
          {
              .callback = loom_check_pass_requirement_is_satisfied,
              .user_data = &pipeline_config,
          },
      .block_pool = block_pool,
      .diagnostic_emitter = diagnostic_emitter,
      .configure =
          {
              .fn = loom_check_configure_pass_instruction,
              .user_data = &pipeline_config,
          },
  };
  return loom_pass_tool_run_flat_pipeline(module, pipeline, &options);
}

static iree_status_t loom_check_verify_pass_output(
    const loom_check_case_t* test_case, iree_string_view_t filename,
    const loom_check_environment_t* environment,
    loom_check_diagnostic_collector_t* diagnostic_collector,
    loom_module_t* module, bool* out_failed_verification) {
  *out_failed_verification = false;
  loom_source_entry_t source_entry = {0};
  loom_source_table_resolver_t resolver_data = {0};
  IREE_RETURN_IF_ERROR(loom_check_source_resolver_for_case(
      module->context, filename, test_case->input, &source_entry,
      &resolver_data));
  loom_verify_options_t verify_options = {
      .sink = {.fn = loom_check_diagnostic_collector_sink,
               .user_data = diagnostic_collector},
      .max_errors = 100,
      .source_resolver = {.fn = loom_source_table_resolve,
                          .user_data = &resolver_data},
  };
  loom_verify_result_t verify_result = {0};
  IREE_RETURN_IF_ERROR(
      loom_verify_module(module, &verify_options, &verify_result));
  *out_failed_verification = verify_result.error_count > 0;
  if (*out_failed_verification) {
    return iree_ok_status();
  }

  loom_target_low_descriptor_registry_t low_registry;
  IREE_RETURN_IF_ERROR(
      loom_check_environment_initialize_low_descriptor_registry(environment,
                                                                &low_registry));
  loom_check_diagnostic_emitter_capture_t low_diagnostic_capture = {
      .diagnostic_collector = diagnostic_collector,
      .module = module,
      .source_resolver = {.fn = loom_source_table_resolve,
                          .user_data = &resolver_data},
      .emitter = LOOM_EMITTER_VERIFIER,
  };
  loom_low_verify_options_t low_verify_options = {
      .flags = LOOM_LOW_VERIFY_FLAG_VERIFY_DESCRIPTOR_REGISTRY,
      .descriptor_registry = &low_registry.registry,
      .emitter =
          {
              .fn = loom_check_diagnostic_emitter_capture_emit,
              .user_data = &low_diagnostic_capture,
          },
      .max_errors = 100,
  };
  loom_low_verify_result_t low_verify_result = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_verify_module(module, &low_verify_options, &low_verify_result));
  *out_failed_verification = low_verify_result.error_count > 0;
  return iree_ok_status();
}

static iree_status_t loom_check_finish_diagnostics_if_needed(
    loom_check_diagnostic_collector_t* diagnostic_collector,
    const loom_check_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_allocator_t allocator,
    loom_check_result_t* result, bool* out_failed) {
  *out_failed = false;
  if (diagnostic_collector->count == 0 && test_case->annotation_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_check_diagnostic_collector_finish(
      diagnostic_collector, test_case, case_index, report, allocator, result));
  *out_failed = result->raw_outcome == LOOM_CHECK_FAIL;
  return iree_ok_status();
}

static bool loom_check_case_has_expected_output(
    const loom_check_case_t* test_case) {
  return !iree_string_view_is_empty(iree_string_view_trim(test_case->expected));
}

iree_status_t loom_check_execute_pass(
    const loom_check_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    const loom_check_environment_t* environment, loom_context_t* context,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_check_result_t* result) {
  loom_module_t* module = NULL;
  iree_arena_allocator_t diagnostic_arena;
  iree_arena_initialize(block_pool, &diagnostic_arena);
  loom_check_diagnostic_collector_t diagnostic_collector = {
      .arena = &diagnostic_arena,
      .host_allocator = allocator,
      .result = result,
  };
  loom_text_parse_options_t parse_options = {
      .diagnostic_sink = {.fn = loom_check_diagnostic_collector_sink,
                          .user_data = &diagnostic_collector},
      .max_errors = 20,
  };
  loom_target_low_descriptor_registry_t low_registry;
  iree_status_t status =
      loom_check_environment_initialize_low_descriptor_registry(environment,
                                                                &low_registry);
  loom_low_descriptor_text_asm_environment_initialize(
      &low_registry.registry, &parse_options.low_asm_environment);
  if (iree_status_is_ok(status)) {
    status = loom_text_parse(test_case->input, filename, context, block_pool,
                             &parse_options, &module);
  }
  diagnostic_collector.module = module;
  if (!module) {
    if (iree_status_is_ok(status)) {
      status = loom_check_diagnostic_collector_finish(
          &diagnostic_collector, test_case, case_index, report, allocator,
          result);
    }
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  if (iree_status_is_ok(status) && diagnostic_collector.count > 0) {
    status = loom_check_diagnostic_collector_finish(&diagnostic_collector,
                                                    test_case, case_index,
                                                    report, allocator, result);
  }
  if (!iree_status_is_ok(status) || diagnostic_collector.count > 0) {
    loom_module_free(module);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  const loom_target_preset_registry_t preset_registry =
      loom_target_low_descriptor_registry_presets(&low_registry);
  iree_host_size_t expanded_preset_count = 0;
  status = loom_target_expand_presets(module, &preset_registry,
                                      &expanded_preset_count);
  if (!iree_status_is_ok(status)) {
    loom_module_free(module);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  // Build and run the pass pipeline.
  loom_low_lower_policy_registry_t low_lower_policy_registry = {0};
  const loom_low_lower_policy_registry_t* low_lower_policy_registry_ref = NULL;
  loom_source_entry_t source_entry = {0};
  loom_source_table_resolver_t resolver_data = {0};
  status = loom_check_source_resolver_for_case(
      context, filename, test_case->input, &source_entry, &resolver_data);
  loom_check_diagnostic_emitter_capture_t pass_diagnostic_capture = {
      .diagnostic_collector = &diagnostic_collector,
      .module = module,
      .source_resolver = {.fn = loom_source_table_resolve,
                          .user_data = &resolver_data},
      .emitter = LOOM_EMITTER_PASS,
  };
  iree_diagnostic_emitter_t pass_diagnostic_emitter = {
      .fn = loom_check_diagnostic_emitter_capture_emit,
      .user_data = &pass_diagnostic_capture,
  };
  bool requires_low_lower_policy_registry = false;
  if (iree_status_is_ok(status)) {
    status = loom_check_pipeline_requires_low_lower_policy_registry(
        test_case->pipeline, &requires_low_lower_policy_registry);
  }
  if (iree_status_is_ok(status) && requires_low_lower_policy_registry) {
    status = loom_check_environment_initialize_low_lower_policy_registry(
        environment, &low_lower_policy_registry);
    if (iree_status_is_ok(status)) {
      low_lower_policy_registry_ref = &low_lower_policy_registry;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_check_run_pipeline(test_case->pipeline, block_pool,
                                     pass_diagnostic_emitter, &low_registry,
                                     low_lower_policy_registry_ref, module);
  }
  if (!iree_status_is_ok(status)) {
    if (pass_diagnostic_capture.emission_count > 0 ||
        diagnostic_collector.count > 0 || test_case->annotation_count > 0) {
      iree_status_free(status);
      status = loom_check_diagnostic_collector_finish(
          &diagnostic_collector, test_case, case_index, report, allocator,
          result);
    }
    loom_module_free(module);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  bool failed_verification = false;
  status = loom_check_verify_pass_output(test_case, filename, environment,
                                         &diagnostic_collector, module,
                                         &failed_verification);
  bool diagnostics_failed = false;
  if (iree_status_is_ok(status)) {
    status = loom_check_finish_diagnostics_if_needed(
        &diagnostic_collector, test_case, case_index, report, allocator, result,
        &diagnostics_failed);
  }
  bool has_expected_output = loom_check_case_has_expected_output(test_case);
  if (!iree_status_is_ok(status) || diagnostics_failed || failed_verification ||
      (diagnostic_collector.count > 0 && !has_expected_output)) {
    loom_module_free(module);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  // Print the result (stripping comments for comparison).
  status = loom_text_print_module_to_builder(module, &result->actual_output,
                                             LOOM_TEXT_PRINT_DEFAULT);
  loom_module_free(module);
  if (iree_status_is_ok(status)) {
    result->has_actual_output = true;
  }

  // Strip comments from the expected section.
  iree_string_builder_t stripped_expected;
  iree_string_builder_initialize(allocator, &stripped_expected);
  if (iree_status_is_ok(status)) {
    status = loom_check_strip_comments(test_case->expected, &stripped_expected);
  }
  if (iree_status_is_ok(status)) {
    iree_string_view_t actual_trimmed =
        iree_string_view_trim(iree_string_builder_view(&result->actual_output));
    iree_string_view_t expected_trimmed =
        iree_string_view_trim(iree_string_builder_view(&stripped_expected));
    if (iree_string_view_equal(actual_trimmed, expected_trimmed)) {
      result->raw_outcome = LOOM_CHECK_PASS;
    } else {
      result->raw_outcome = LOOM_CHECK_FAIL;
      status = loom_check_result_record_diff(expected_trimmed, actual_trimmed,
                                             allocator, result);
    }
  }

  iree_string_builder_deinitialize(&stripped_expected);
  iree_arena_deinitialize(&diagnostic_arena);
  return status;
}

iree_status_t loom_check_execute_format(const loom_check_case_t* test_case,
                                        iree_string_view_t filename,
                                        loom_context_t* context,
                                        iree_arena_block_pool_t* block_pool,
                                        iree_allocator_t allocator,
                                        loom_check_result_t* result) {
  (void)test_case;
  (void)filename;
  (void)context;
  (void)block_pool;
  (void)allocator;
  (void)result;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "format mode execution not yet implemented");
}
