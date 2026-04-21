// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/hal_run_provider.h"

#include "iree/base/alignment.h"
#include "loom/tooling/execution/compile_report_capture.h"
#include "loom/tooling/execution/hal_candidate.h"
#include "loom/tooling/execution/hal_invocation.h"
#include "loom/tooling/execution/hal_runtime.h"
#include "loom/tooling/execution/session.h"

typedef struct loom_check_hal_run_options_t {
  // HAL backend name selected by --loom_backend.
  iree_string_view_t backend_name;
  // Module-local entry function selected by --loom_entry.
  iree_string_view_t entry_symbol;
  // HAL executable entry point ordinal selected by --entry_point.
  uint32_t entry_point;
  // Dispatch workgroup count selected by --workgroup_count.
  uint32_t workgroup_count[3];
  // Parsed --binding values in HAL binding ordinal order.
  iree_string_view_t binding_specs[LOOM_RUN_HAL_MAX_BINDING_COUNT];
  // Calling-convention character for each binding spec.
  char binding_conventions[LOOM_RUN_HAL_MAX_BINDING_COUNT];
  // Number of populated binding specs.
  iree_host_size_t binding_count;
  // Parsed --expected_binding values in HAL binding ordinal order.
  iree_string_view_t expected_binding_specs[LOOM_RUN_HAL_MAX_BINDING_COUNT];
  // Calling-convention character for each expected binding spec.
  char expected_binding_conventions[LOOM_RUN_HAL_MAX_BINDING_COUNT];
  // Number of populated expected binding specs.
  iree_host_size_t expected_binding_count;
  // Maximum number of output elements to format.
  iree_host_size_t max_output_element_count;
  // Optional compile report capture/formatting options.
  loom_run_compile_report_capture_options_t compile_report_options;
} loom_check_hal_run_options_t;

static const loom_check_hal_run_provider_t*
loom_check_hal_run_provider_from_base(
    const loom_check_run_provider_t* provider) {
  if (provider == NULL) {
    return NULL;
  }
  return iree_containerof(provider, loom_check_hal_run_provider_t, base);
}

static const loom_run_hal_backend_t* loom_check_hal_run_provider_lookup_backend(
    const loom_check_run_provider_t* provider,
    iree_string_view_t backend_name) {
  const loom_check_hal_run_provider_t* hal_run_provider =
      loom_check_hal_run_provider_from_base(provider);
  if (hal_run_provider == NULL) {
    return NULL;
  }
  return loom_run_hal_backend_registry_lookup(
      &hal_run_provider->backend_registry, backend_name);
}

bool loom_check_hal_run_provider_match(
    const loom_check_run_provider_t* provider,
    const loom_check_run_arguments_t* arguments) {
  if (arguments == NULL) {
    return false;
  }
  for (iree_host_size_t i = 0; i < arguments->count; ++i) {
    iree_string_view_t backend_name = iree_string_view_empty();
    if (iree_string_view_equal(arguments->values[i],
                               IREE_SV("--loom_backend"))) {
      if (i + 1 >= arguments->count) {
        return true;
      }
      backend_name = arguments->values[i + 1];
    } else if (iree_string_view_starts_with(arguments->values[i],
                                            IREE_SV("--loom_backend="))) {
      backend_name = iree_string_view_substr(arguments->values[i],
                                             IREE_SV("--loom_backend=").size,
                                             IREE_HOST_SIZE_MAX);
    } else {
      continue;
    }
    return loom_check_hal_run_provider_lookup_backend(provider, backend_name) !=
           NULL;
  }
  return false;
}

static iree_status_t loom_check_hal_run_parse_uint32(
    iree_string_view_t value, iree_string_view_t option_name,
    uint32_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  if (!iree_string_view_atoi_uint32(value, out_value)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT, "invalid RUN: run --%.*s value '%.*s'",
        (int)option_name.size, option_name.data, (int)value.size, value.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_check_hal_run_parse_workgroup_count(
    iree_string_view_t value, uint32_t* out_workgroup_count) {
  IREE_ASSERT_ARGUMENT(out_workgroup_count);
  iree_string_view_t remaining = value;
  iree_string_view_t x;
  iree_string_view_split(remaining, ',', &x, &remaining);
  iree_string_view_t y;
  iree_string_view_split(remaining, ',', &y, &remaining);
  iree_string_view_t z = remaining;
  if (!iree_string_view_atoi_uint32(x, &out_workgroup_count[0]) ||
      !iree_string_view_atoi_uint32(y, &out_workgroup_count[1]) ||
      !iree_string_view_atoi_uint32(z, &out_workgroup_count[2])) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "invalid RUN: run --workgroup_count='%.*s'; expected `x,y,z`",
        (int)value.size, value.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_check_hal_run_append_binding(
    iree_string_view_t value, iree_string_view_t* values, char* conventions,
    iree_host_size_t* count, iree_string_view_t option_name) {
  IREE_ASSERT_ARGUMENT(values);
  IREE_ASSERT_ARGUMENT(conventions);
  IREE_ASSERT_ARGUMENT(count);
  if (*count >= LOOM_RUN_HAL_MAX_BINDING_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many RUN: run --%.*s values; maximum is %d",
                            (int)option_name.size, option_name.data,
                            LOOM_RUN_HAL_MAX_BINDING_COUNT);
  }
  values[*count] = value;
  conventions[*count] = 'r';
  ++(*count);
  return iree_ok_status();
}

static iree_status_t loom_check_hal_run_options_initialize(
    const loom_check_run_arguments_t* arguments,
    loom_check_hal_run_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(arguments);
  IREE_ASSERT_ARGUMENT(out_options);
  *out_options = (loom_check_hal_run_options_t){
      .workgroup_count = {1, 1, 1},
      .max_output_element_count = 1024,
  };
  loom_run_compile_report_capture_options_initialize(
      &out_options->compile_report_options);

  iree_status_t status = iree_ok_status();
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
      out_options->backend_name = value;
      continue;
    }
    status = loom_check_run_arguments_take_option_value(
        arguments, &i, IREE_SV("loom_entry"), &value, &matched);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (matched) {
      out_options->entry_symbol = value;
      continue;
    }
    status = loom_check_run_arguments_take_option_value(
        arguments, &i, IREE_SV("entry_point"), &value, &matched);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (matched) {
      status = loom_check_hal_run_parse_uint32(value, IREE_SV("entry_point"),
                                               &out_options->entry_point);
      continue;
    }
    status = loom_check_run_arguments_take_option_value(
        arguments, &i, IREE_SV("workgroup_count"), &value, &matched);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (matched) {
      status = loom_check_hal_run_parse_workgroup_count(
          value, out_options->workgroup_count);
      continue;
    }
    status = loom_check_run_arguments_take_option_value(
        arguments, &i, IREE_SV("binding"), &value, &matched);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (matched) {
      status = loom_check_hal_run_append_binding(
          value, out_options->binding_specs, out_options->binding_conventions,
          &out_options->binding_count, IREE_SV("binding"));
      continue;
    }
    status = loom_check_run_arguments_take_option_value(
        arguments, &i, IREE_SV("expected_binding"), &value, &matched);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (matched) {
      status = loom_check_hal_run_append_binding(
          value, out_options->expected_binding_specs,
          out_options->expected_binding_conventions,
          &out_options->expected_binding_count, IREE_SV("expected_binding"));
      continue;
    }
    status = loom_check_run_arguments_take_option_value(
        arguments, &i, IREE_SV("output_max_element_count"), &value, &matched);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (matched) {
      uint32_t max_output_element_count = 0;
      status = loom_check_hal_run_parse_uint32(
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
                                "unknown HAL RUN: run argument '%.*s'",
                                (int)argument.size, argument.data);
    }
  }
  if (iree_status_is_ok(status) &&
      iree_string_view_is_empty(out_options->backend_name)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL RUN: run requires --loom_backend to name a linked HAL backend");
  }
  return status;
}

static iree_status_t loom_check_hal_run_register_context(
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

static iree_status_t loom_check_hal_run_initialize_low_registry(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry) {
  const loom_check_environment_t* environment =
      (const loom_check_environment_t*)user_data;
  return loom_check_environment_initialize_low_descriptor_registry(
      environment, out_registry);
}

iree_status_t loom_check_hal_run_provider_execute(
    const loom_check_run_provider_t* provider,
    const loom_check_run_provider_request_t* request) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(request);
  IREE_ASSERT_ARGUMENT(request->environment);
  IREE_ASSERT_ARGUMENT(request->arguments);
  IREE_ASSERT_ARGUMENT(request->result);

  loom_check_hal_run_options_t run_options = {0};
  loom_run_session_t session = {0};
  loom_run_module_t module = {0};
  loom_run_hal_runtime_t runtime = {0};
  loom_run_hal_candidate_t candidate = {0};
  loom_run_compile_report_capture_t compile_report_capture = {0};
  loom_run_hal_invocation_result_t invocation_result = {0};
  loom_run_hal_invocation_result_initialize(request->host_allocator,
                                            &invocation_result);

  const loom_run_hal_backend_t* backend = NULL;
  iree_status_t status =
      loom_check_hal_run_options_initialize(request->arguments, &run_options);
  if (iree_status_is_ok(status)) {
    backend = loom_check_hal_run_provider_lookup_backend(
        provider, run_options.backend_name);
    if (backend == NULL) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT, "unknown HAL RUN: run backend '%.*s'",
          (int)run_options.backend_name.size, run_options.backend_name.data);
    }
  }
  if (iree_status_is_ok(status)) {
    loom_run_session_options_t session_options = {0};
    loom_run_session_options_initialize(&session_options);
    session_options.host_allocator = request->host_allocator;
    session_options.register_context = (loom_run_register_context_callback_t){
        .fn = loom_check_hal_run_register_context,
        .user_data = (void*)request->environment,
    };
    session_options.initialize_low_descriptor_registry =
        (loom_run_initialize_low_descriptor_registry_callback_t){
            .fn = loom_check_hal_run_initialize_low_registry,
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
    status = loom_run_hal_runtime_initialize(backend, request->host_allocator,
                                             &runtime);
  }
  if (iree_status_is_ok(status)) {
    loom_run_candidate_compile_options_t compile_options = {0};
    loom_run_candidate_compile_options_initialize(&compile_options);
    compile_options.entry_symbol = run_options.entry_symbol;
    compile_options.source_resolver = loom_run_module_source_resolver(&module);
    loom_run_compile_report_capture_configure_compile_options(
        &compile_report_capture, &compile_options);
    status = loom_run_hal_candidate_compile(
        backend, &runtime, &module, &compile_options, request->host_allocator,
        &candidate);
  }
  if (iree_status_is_ok(status)) {
    loom_run_hal_invocation_request_t invocation_request = {0};
    loom_run_hal_invocation_request_initialize(&invocation_request);
    invocation_request.runtime = &runtime;
    invocation_request.executable = &candidate.executable;
    invocation_request.options.entry_point = run_options.entry_point;
    invocation_request.options.workgroup_count[0] =
        run_options.workgroup_count[0];
    invocation_request.options.workgroup_count[1] =
        run_options.workgroup_count[1];
    invocation_request.options.workgroup_count[2] =
        run_options.workgroup_count[2];
    invocation_request.bindings = (loom_run_hal_binding_specs_t){
        .values = run_options.binding_specs,
        .conventions = run_options.binding_conventions,
        .count = run_options.binding_count,
    };
    invocation_request.expected_bindings = (loom_run_hal_binding_specs_t){
        .values = run_options.expected_binding_specs,
        .conventions = run_options.expected_binding_conventions,
        .count = run_options.expected_binding_count,
    };
    invocation_request.max_output_element_count =
        run_options.max_output_element_count;
    status = loom_run_hal_invocation_run(
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
  loom_run_hal_invocation_result_deinitialize(&invocation_result);
  loom_run_hal_candidate_deinitialize(&candidate);
  loom_run_hal_runtime_deinitialize(&runtime);
  loom_run_module_deinitialize(&module);
  loom_run_session_deinitialize(&session);
  return status;
}

iree_status_t loom_check_hal_run_provider_append_names(
    const loom_check_run_provider_t* provider, iree_string_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(builder);
  const loom_check_hal_run_provider_t* hal_run_provider =
      loom_check_hal_run_provider_from_base(provider);
  return loom_run_hal_backend_registry_format_names(
      &hal_run_provider->backend_registry, builder);
}
