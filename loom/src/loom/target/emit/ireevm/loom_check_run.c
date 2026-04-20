// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/ireevm/loom_check_run.h"

#include <string.h>

#include "loom/target/emit/ireevm/candidate.h"
#include "loom/tooling/execution/compile_report_capture.h"
#include "loom/tooling/execution/session.h"
#include "loom/tooling/execution/vm_invocation.h"

typedef struct loom_ireevm_loom_check_run_options_t {
  // Module-local target symbol selected by --loom_target.
  iree_string_view_t target_symbol;
  // Module name embedded in the emitted VM bytecode archive.
  iree_string_view_t module_name;
  // Exported VM function selected by --function.
  iree_string_view_t function_name;
  // Parsed --input values in VM function ABI order.
  iree_string_view_t* inputs;
  // Number of entries in |inputs|.
  iree_host_size_t input_count;
  // Parsed --output values in VM function ABI order.
  iree_string_view_t* outputs;
  // Number of entries in |outputs|.
  iree_host_size_t output_count;
  // Parsed --expected_output values in VM function ABI order.
  iree_string_view_t* expected_outputs;
  // Number of entries in |expected_outputs|.
  iree_host_size_t expected_output_count;
  // Maximum number of output elements to format.
  iree_host_size_t max_output_element_count;
  // Optional compile report capture/formatting options.
  loom_run_compile_report_capture_options_t compile_report_options;
} loom_ireevm_loom_check_run_options_t;

static void loom_ireevm_loom_check_run_options_deinitialize(
    loom_ireevm_loom_check_run_options_t* options, iree_allocator_t allocator) {
  if (options == NULL) {
    return;
  }
  iree_allocator_free(allocator, options->expected_outputs);
  iree_allocator_free(allocator, options->outputs);
  iree_allocator_free(allocator, options->inputs);
  *options = (loom_ireevm_loom_check_run_options_t){0};
}

static bool loom_ireevm_loom_check_run_argument_is_backend(
    iree_string_view_t argument, iree_string_view_t* out_value,
    bool* out_has_inline_value) {
  if (iree_string_view_equal(argument, IREE_SV("--loom_backend"))) {
    *out_value = iree_string_view_empty();
    *out_has_inline_value = false;
    return true;
  }
  if (iree_string_view_starts_with(argument, IREE_SV("--loom_backend="))) {
    *out_value = iree_string_view_substr(
        argument, IREE_SV("--loom_backend=").size, IREE_HOST_SIZE_MAX);
    *out_has_inline_value = true;
    return true;
  }
  return false;
}

static bool loom_ireevm_loom_check_run_provider_match(
    const loom_check_run_provider_t* provider,
    const loom_check_run_arguments_t* arguments) {
  (void)provider;
  if (arguments == NULL) {
    return false;
  }
  for (iree_host_size_t i = 0; i < arguments->count; ++i) {
    iree_string_view_t backend_name = iree_string_view_empty();
    bool has_inline_value = false;
    if (!loom_ireevm_loom_check_run_argument_is_backend(
            arguments->values[i], &backend_name, &has_inline_value)) {
      continue;
    }
    if (!has_inline_value) {
      if (i + 1 >= arguments->count) {
        return true;
      }
      backend_name = arguments->values[i + 1];
    }
    return iree_string_view_equal(backend_name, IREE_SV("vm"));
  }
  return true;
}

static iree_status_t loom_ireevm_loom_check_run_append_value(
    iree_string_view_t value, iree_string_view_t* values,
    iree_host_size_t capacity, iree_host_size_t* count,
    iree_string_view_t option_name) {
  IREE_ASSERT_ARGUMENT(values);
  IREE_ASSERT_ARGUMENT(count);
  if (*count >= capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many RUN: run --%.*s values",
                            (int)option_name.size, option_name.data);
  }
  values[(*count)++] = value;
  return iree_ok_status();
}

static iree_status_t loom_ireevm_loom_check_run_parse_uint32(
    iree_string_view_t value, iree_string_view_t option_name,
    uint32_t* out_value) {
  if (!iree_string_view_atoi_uint32(value, out_value)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT, "invalid RUN: run --%.*s value '%.*s'",
        (int)option_name.size, option_name.data, (int)value.size, value.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_ireevm_loom_check_run_options_initialize(
    const loom_check_run_arguments_t* arguments, iree_allocator_t allocator,
    loom_ireevm_loom_check_run_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(arguments);
  IREE_ASSERT_ARGUMENT(out_options);
  *out_options = (loom_ireevm_loom_check_run_options_t){
      .module_name = IREE_SVL("loom"),
      .max_output_element_count = 1024,
  };
  loom_run_compile_report_capture_options_initialize(
      &out_options->compile_report_options);

  iree_host_size_t values_size = 0;
  if (!iree_host_size_checked_mul(arguments->count, sizeof(iree_string_view_t),
                                  &values_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "RUN: run argument list is too large");
  }
  iree_status_t status = iree_allocator_malloc(allocator, values_size,
                                               (void**)&out_options->inputs);
  if (iree_status_is_ok(status)) {
    memset(out_options->inputs, 0, values_size);
    status = iree_allocator_malloc(allocator, values_size,
                                   (void**)&out_options->outputs);
  }
  if (iree_status_is_ok(status)) {
    memset(out_options->outputs, 0, values_size);
    status = iree_allocator_malloc(allocator, values_size,
                                   (void**)&out_options->expected_outputs);
  }
  if (iree_status_is_ok(status)) {
    memset(out_options->expected_outputs, 0, values_size);
  }
  if (!iree_status_is_ok(status)) {
    loom_ireevm_loom_check_run_options_deinitialize(out_options, allocator);
    return status;
  }

  for (iree_host_size_t i = 0;
       i < arguments->count && iree_status_is_ok(status); ++i) {
    iree_string_view_t value = iree_string_view_empty();
    bool matched = false;
    status = loom_check_run_arguments_take_option_value(
        arguments, &i, IREE_SV("loom_backend"), &value, &matched);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (matched) {
      if (!iree_string_view_equal(value, IREE_SV("vm"))) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "IREE VM run provider cannot execute --loom_backend='%.*s'",
            (int)value.size, value.data);
      }
      continue;
    }
    status = loom_check_run_arguments_take_option_value(
        arguments, &i, IREE_SV("loom_target"), &value, &matched);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (matched) {
      out_options->target_symbol = value;
      continue;
    }
    status = loom_check_run_arguments_take_option_value(
        arguments, &i, IREE_SV("loom_module_name"), &value, &matched);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (matched) {
      out_options->module_name = value;
      continue;
    }
    status = loom_check_run_arguments_take_option_value(
        arguments, &i, IREE_SV("function"), &value, &matched);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (matched) {
      out_options->function_name = value;
      continue;
    }
    status = loom_check_run_arguments_take_option_value(
        arguments, &i, IREE_SV("input"), &value, &matched);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (matched) {
      status = loom_ireevm_loom_check_run_append_value(
          value, out_options->inputs, arguments->count,
          &out_options->input_count, IREE_SV("input"));
      continue;
    }
    status = loom_check_run_arguments_take_option_value(
        arguments, &i, IREE_SV("output"), &value, &matched);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (matched) {
      status = loom_ireevm_loom_check_run_append_value(
          value, out_options->outputs, arguments->count,
          &out_options->output_count, IREE_SV("output"));
      continue;
    }
    status = loom_check_run_arguments_take_option_value(
        arguments, &i, IREE_SV("expected_output"), &value, &matched);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (matched) {
      status = loom_ireevm_loom_check_run_append_value(
          value, out_options->expected_outputs, arguments->count,
          &out_options->expected_output_count, IREE_SV("expected_output"));
      continue;
    }
    status = loom_check_run_arguments_take_option_value(
        arguments, &i, IREE_SV("output_max_element_count"), &value, &matched);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (matched) {
      uint32_t max_output_element_count = 0;
      status = loom_ireevm_loom_check_run_parse_uint32(
          value, IREE_SV("output_max_element_count"),
          &max_output_element_count);
      out_options->max_output_element_count = max_output_element_count;
      continue;
    }
    status = loom_check_run_arguments_take_option_value(
        arguments, &i, IREE_SV("compile_report"), &value, &matched);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (matched) {
      status = loom_run_compile_report_capture_options_parse_mode(
          value, &out_options->compile_report_options);
      continue;
    }
    status = loom_check_run_arguments_take_option_value(
        arguments, &i, IREE_SV("compile_report_row_limit"), &value, &matched);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (matched) {
      status = loom_run_compile_report_capture_options_parse_row_limit(
          value, &out_options->compile_report_options);
      continue;
    }
    if (iree_status_is_ok(status)) {
      iree_string_view_t argument = arguments->values[i];
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown IREE VM RUN: run argument '%.*s'",
                                (int)argument.size, argument.data);
    }
  }

  if (!iree_status_is_ok(status)) {
    loom_ireevm_loom_check_run_options_deinitialize(out_options, allocator);
  }
  return status;
}

static iree_status_t loom_ireevm_loom_check_run_register_context(
    void* user_data, loom_context_t* context) {
  const loom_check_environment_t* environment =
      (const loom_check_environment_t*)user_data;
  if (environment == NULL || environment->register_context.fn == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "loom-check run context registration is required");
  }
  return environment->register_context.fn(
      environment->register_context.user_data, context);
}

static iree_status_t loom_ireevm_loom_check_run_initialize_low_registry(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry) {
  const loom_check_environment_t* environment =
      (const loom_check_environment_t*)user_data;
  return loom_check_environment_initialize_low_descriptor_registry(
      environment, out_registry);
}

static iree_status_t loom_ireevm_loom_check_run_execute(
    const loom_check_run_provider_t* provider,
    const loom_check_run_provider_request_t* request) {
  (void)provider;
  IREE_ASSERT_ARGUMENT(request);
  IREE_ASSERT_ARGUMENT(request->environment);
  IREE_ASSERT_ARGUMENT(request->arguments);
  IREE_ASSERT_ARGUMENT(request->result);

  loom_ireevm_loom_check_run_options_t run_options = {0};
  loom_run_session_t session = {0};
  loom_run_module_t module = {0};
  loom_ireevm_run_candidate_t candidate = {0};
  loom_run_compile_report_capture_t compile_report_capture = {0};
  loom_run_vm_runtime_t runtime = {0};
  loom_run_vm_invocation_result_t invocation_result = {0};
  loom_run_vm_invocation_result_initialize(request->host_allocator,
                                           &invocation_result);

  iree_status_t status = loom_ireevm_loom_check_run_options_initialize(
      request->arguments, request->host_allocator, &run_options);
  if (iree_status_is_ok(status)) {
    loom_run_session_options_t session_options = {0};
    loom_run_session_options_initialize(&session_options);
    session_options.host_allocator = request->host_allocator;
    session_options.register_context = (loom_run_register_context_callback_t){
        .fn = loom_ireevm_loom_check_run_register_context,
        .user_data = (void*)request->environment,
    };
    session_options.initialize_low_descriptor_registry =
        (loom_run_initialize_low_descriptor_registry_callback_t){
            .fn = loom_ireevm_loom_check_run_initialize_low_registry,
            .user_data = (void*)request->environment,
        };
    status = loom_run_session_initialize(&session_options, &session);
  }
  if (iree_status_is_ok(status)) {
    loom_run_module_parse_options_t parse_options = {0};
    loom_run_module_parse_options_initialize(&parse_options);
    parse_options.filename = request->filename;
    parse_options.source = request->test_case->input;
    status = loom_run_module_parse(&session, &parse_options, &module);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_compile_report_capture_initialize(
        &run_options.compile_report_options, request->host_allocator,
        &compile_report_capture);
  }
  if (iree_status_is_ok(status)) {
    loom_run_candidate_compile_options_t compile_options = {0};
    loom_run_candidate_compile_options_initialize(&compile_options);
    compile_options.module_name = run_options.module_name;
    compile_options.target_symbol = run_options.target_symbol;
    compile_options.source_resolver = loom_run_module_source_resolver(&module);
    loom_run_compile_report_capture_configure_compile_options(
        &compile_report_capture, &compile_options);
    status = loom_ireevm_run_candidate_compile(
        &module, &compile_options, request->host_allocator, &candidate);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_vm_runtime_initialize(request->host_allocator, &runtime);
  }
  if (iree_status_is_ok(status)) {
    loom_run_vm_invocation_request_t invocation_request = {0};
    loom_run_vm_invocation_request_initialize(&invocation_request);
    invocation_request.runtime = &runtime;
    invocation_request.archive = &candidate.archive;
    invocation_request.options.function_name = run_options.function_name;
    invocation_request.options.inputs = (loom_run_vm_value_specs_t){
        .values = run_options.inputs,
        .count = run_options.input_count,
    };
    invocation_request.options.outputs = (loom_run_vm_value_specs_t){
        .values = run_options.outputs,
        .count = run_options.output_count,
    };
    invocation_request.options.expected_outputs = (loom_run_vm_value_specs_t){
        .values = run_options.expected_outputs,
        .count = run_options.expected_output_count,
    };
    invocation_request.options.max_output_element_count =
        run_options.max_output_element_count;
    status = loom_run_vm_invocation_run(
        &invocation_request, request->host_allocator, &invocation_result);
  }

  if (iree_status_is_ok(status)) {
    request->result->exit_code = invocation_result.exit_code;
    status = iree_string_builder_append_string(
        &request->result->stdout_text,
        iree_string_builder_view(&invocation_result.output));
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_compile_report_capture_append_text(
        &compile_report_capture, &request->result->stdout_text);
  } else {
    request->result->exit_code = 1;
    status = loom_check_run_result_append_status(status, request->result);
  }

  loom_run_compile_report_capture_deinitialize(&compile_report_capture);
  loom_run_vm_invocation_result_deinitialize(&invocation_result);
  loom_run_vm_runtime_deinitialize(&runtime);
  loom_ireevm_run_candidate_deinitialize(&candidate);
  loom_run_module_deinitialize(&module);
  loom_run_session_deinitialize(&session);
  loom_ireevm_loom_check_run_options_deinitialize(&run_options,
                                                  request->host_allocator);
  return status;
}

static iree_status_t loom_ireevm_loom_check_run_append_names(
    const loom_check_run_provider_t* provider, iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(builder, "iree-vm");
}

const loom_check_run_provider_t loom_ireevm_loom_check_run_provider = {
    .name = IREE_SVL("iree-vm"),
    .match = loom_ireevm_loom_check_run_provider_match,
    .execute = loom_ireevm_loom_check_run_execute,
    .append_names = loom_ireevm_loom_check_run_append_names,
};
