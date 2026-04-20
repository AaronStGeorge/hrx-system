// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "iree/io/file_contents.h"
#include "loom/target/tool/process.h"
#include "loom/tools/loom-check/execute.h"

typedef struct loom_check_run_argument_storage_t {
  // Public argument view passed to linked run providers.
  loom_check_run_arguments_t arguments;
  // Mutable argument views backed by |strings|.
  iree_string_view_t* values;
  // Allocated NUL-terminated argument strings.
  char** strings;
} loom_check_run_argument_storage_t;

static iree_string_view_t loom_check_iree_run_loom_path(
    const loom_check_environment_t* environment) {
  if (environment != NULL &&
      !iree_string_view_is_empty(environment->iree_run_loom_path)) {
    return environment->iree_run_loom_path;
  }
  return IREE_SV("iree-run-loom");
}

static bool loom_check_process_path_searches_path(iree_string_view_t path) {
  for (iree_host_size_t i = 0; i < path.size; ++i) {
    if (path.data[i] == '/' || path.data[i] == '\\') {
      return false;
    }
  }
  return true;
}

static void loom_check_run_arguments_deinitialize(
    loom_check_run_argument_storage_t* storage, iree_allocator_t allocator) {
  if (storage == NULL) {
    return;
  }
  if (storage->strings != NULL) {
    for (iree_host_size_t i = 0; i < storage->arguments.count; ++i) {
      iree_allocator_free(allocator, storage->strings[i]);
    }
  }
  iree_allocator_free(allocator, storage->strings);
  iree_allocator_free(allocator, storage->values);
  *storage = (loom_check_run_argument_storage_t){0};
}

static iree_status_t loom_check_run_count_arguments(
    iree_string_view_t text, iree_host_size_t* out_count) {
  IREE_ASSERT_ARGUMENT(out_count);
  *out_count = 0;
  text = iree_string_view_trim(text);
  iree_host_size_t index = 0;
  while (index < text.size) {
    while (index < text.size &&
           (text.data[index] == ' ' || text.data[index] == '\t')) {
      ++index;
    }
    if (index >= text.size) {
      break;
    }
    ++(*out_count);
    bool quoted = false;
    while (index < text.size) {
      char value = text.data[index++];
      if (value == '\\') {
        if (index >= text.size) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "RUN: run argument ends with '\\'");
        }
        ++index;
        continue;
      }
      if (value == '"') {
        quoted = !quoted;
        continue;
      }
      if (!quoted && (value == ' ' || value == '\t')) {
        break;
      }
    }
    if (quoted) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "RUN: run argument has unterminated quote");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_check_run_parse_one_argument(
    iree_string_view_t token, iree_allocator_t allocator,
    iree_string_view_t* out_value, char** out_storage) {
  IREE_ASSERT_ARGUMENT(out_value);
  IREE_ASSERT_ARGUMENT(out_storage);
  *out_value = iree_string_view_empty();
  *out_storage = NULL;

  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator, &builder);
  iree_status_t status = iree_ok_status();
  bool quoted = false;
  for (iree_host_size_t i = 0; i < token.size && iree_status_is_ok(status);
       ++i) {
    char value = token.data[i];
    if (value == '\\') {
      if (++i >= token.size) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "RUN: run argument ends with '\\'");
      } else {
        status = iree_string_builder_append_string(
            &builder, iree_make_string_view(&token.data[i], 1));
      }
      continue;
    }
    if (value == '"') {
      quoted = !quoted;
      continue;
    }
    status = iree_string_builder_append_string(
        &builder, iree_make_string_view(&value, 1));
  }
  if (iree_status_is_ok(status) && quoted) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "RUN: run argument has unterminated quote");
  }
  if (iree_status_is_ok(status)) {
    *out_storage = iree_string_builder_take_storage(&builder);
    *out_value = iree_make_cstring_view(*out_storage ? *out_storage : "");
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t loom_check_run_parse_arguments(
    iree_string_view_t text, iree_allocator_t allocator,
    loom_check_run_argument_storage_t* out_storage) {
  IREE_ASSERT_ARGUMENT(out_storage);
  *out_storage = (loom_check_run_argument_storage_t){0};

  iree_host_size_t argument_count = 0;
  IREE_RETURN_IF_ERROR(loom_check_run_count_arguments(text, &argument_count));
  if (argument_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "RUN: run requires runner arguments");
  }

  iree_host_size_t values_size = 0;
  iree_host_size_t strings_size = 0;
  if (!iree_host_size_checked_mul(argument_count, sizeof(iree_string_view_t),
                                  &values_size) ||
      !iree_host_size_checked_mul(argument_count, sizeof(char*),
                                  &strings_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "RUN: run argument array is too large");
  }
  iree_string_view_t* values = NULL;
  iree_status_t status =
      iree_allocator_malloc(allocator, values_size, (void**)&values);
  if (iree_status_is_ok(status)) {
    memset(values, 0, values_size);
    out_storage->values = values;
    out_storage->arguments.values = out_storage->values;
    status = iree_allocator_malloc(allocator, strings_size,
                                   (void**)&out_storage->strings);
  }
  if (iree_status_is_ok(status)) {
    memset(out_storage->strings, 0, strings_size);
  } else {
    loom_check_run_arguments_deinitialize(out_storage, allocator);
    return status;
  }
  out_storage->arguments.count = argument_count;

  text = iree_string_view_trim(text);
  iree_host_size_t argument_index = 0;
  iree_host_size_t token_start = 0;
  iree_host_size_t index = 0;
  status = iree_ok_status();
  while (index < text.size && iree_status_is_ok(status)) {
    while (index < text.size &&
           (text.data[index] == ' ' || text.data[index] == '\t')) {
      ++index;
    }
    if (index >= text.size) {
      break;
    }
    token_start = index;
    bool quoted = false;
    while (index < text.size) {
      char value = text.data[index++];
      if (value == '\\') {
        if (index < text.size) {
          ++index;
        }
        continue;
      }
      if (value == '"') {
        quoted = !quoted;
        continue;
      }
      if (!quoted && (value == ' ' || value == '\t')) {
        break;
      }
    }
    iree_host_size_t token_end = index;
    if (token_end > token_start &&
        (text.data[token_end - 1] == ' ' || text.data[token_end - 1] == '\t')) {
      --token_end;
    }
    iree_string_view_t token =
        iree_string_view_substr(text, token_start, token_end - token_start);
    status = loom_check_run_parse_one_argument(
        token, allocator, &out_storage->values[argument_index],
        &out_storage->strings[argument_index]);
    ++argument_index;
  }
  if (!iree_status_is_ok(status)) {
    loom_check_run_arguments_deinitialize(out_storage, allocator);
  }
  return status;
}

static iree_status_t loom_check_run_build_process_arguments(
    iree_string_view_t input_path,
    const loom_check_run_arguments_t* run_arguments, iree_allocator_t allocator,
    iree_string_view_t** out_arguments, iree_host_size_t* out_argument_count) {
  IREE_ASSERT_ARGUMENT(run_arguments);
  IREE_ASSERT_ARGUMENT(out_arguments);
  IREE_ASSERT_ARGUMENT(out_argument_count);
  *out_arguments = NULL;
  *out_argument_count = 0;

  iree_host_size_t argument_count = 0;
  if (!iree_host_size_checked_add(run_arguments->count, 1, &argument_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "RUN: run process argument count overflow");
  }
  iree_host_size_t argument_size = 0;
  if (!iree_host_size_checked_mul(argument_count, sizeof(iree_string_view_t),
                                  &argument_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "RUN: run process argument array is too large");
  }
  iree_string_view_t* arguments = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, argument_size, (void**)&arguments));
  arguments[0] = input_path;
  for (iree_host_size_t i = 0; i < run_arguments->count; ++i) {
    arguments[i + 1] = run_arguments->values[i];
  }
  *out_arguments = arguments;
  *out_argument_count = argument_count;
  return iree_ok_status();
}

static void loom_check_run_result_initialize(iree_allocator_t allocator,
                                             loom_check_run_result_t* result) {
  IREE_ASSERT_ARGUMENT(result);
  *result = (loom_check_run_result_t){0};
  iree_string_builder_initialize(allocator, &result->stdout_text);
  iree_string_builder_initialize(allocator, &result->stderr_text);
}

static void loom_check_run_result_deinitialize(
    loom_check_run_result_t* result) {
  if (result == NULL) {
    return;
  }
  iree_string_builder_deinitialize(&result->stderr_text);
  iree_string_builder_deinitialize(&result->stdout_text);
  *result = (loom_check_run_result_t){0};
}

static bool loom_check_run_case_has_requirement(
    const loom_check_case_t* test_case, iree_string_view_t requirement) {
  for (iree_host_size_t i = 0; i < test_case->requirement_count; ++i) {
    if (iree_string_view_equal(test_case->requirements[i], requirement)) {
      return true;
    }
  }
  return false;
}

static const loom_check_run_provider_t* loom_check_run_lookup_provider(
    const loom_check_environment_t* environment,
    const loom_check_run_arguments_t* arguments) {
  if (environment == NULL || environment->run_providers.providers == NULL) {
    return NULL;
  }
  for (iree_host_size_t i = 0; i < environment->run_providers.provider_count;
       ++i) {
    const loom_check_run_provider_t* provider =
        environment->run_providers.providers[i];
    if (provider == NULL || provider->match == NULL) {
      continue;
    }
    if (provider->match(provider, arguments)) {
      return provider;
    }
  }
  return NULL;
}

static iree_status_t loom_check_run_append_provider_names(
    const loom_check_environment_t* environment,
    iree_string_builder_t* output) {
  if (environment == NULL || environment->run_providers.providers == NULL ||
      environment->run_providers.provider_count == 0) {
    return iree_string_builder_append_cstring(output, "(none)");
  }
  bool appended_any = false;
  for (iree_host_size_t i = 0; i < environment->run_providers.provider_count;
       ++i) {
    const loom_check_run_provider_t* provider =
        environment->run_providers.providers[i];
    if (provider == NULL || provider->append_names == NULL) {
      continue;
    }
    if (appended_any) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, ", "));
    }
    IREE_RETURN_IF_ERROR(provider->append_names(provider, output));
    appended_any = true;
  }
  if (!appended_any) {
    return iree_string_builder_append_cstring(output, "(none)");
  }
  return iree_ok_status();
}

static iree_status_t loom_check_run_fail_no_provider(
    const loom_check_environment_t* environment, loom_check_result_t* result) {
  result->raw_outcome = LOOM_CHECK_FAIL;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      &result->detail,
      "RUN: run has no linked in-process provider for these arguments; linked "
      "providers are "));
  IREE_RETURN_IF_ERROR(
      loom_check_run_append_provider_names(environment, &result->detail));
  return iree_string_builder_append_cstring(
      &result->detail,
      "\nUse a target-owned loom-check runner or declare "
      "'// REQUIRES: iree-run-loom' to use the external compatibility "
      "runner.\n");
}

static iree_status_t loom_check_run_execute_provider(
    const loom_check_run_provider_t* provider,
    const loom_check_case_t* test_case, iree_string_view_t filename,
    const loom_check_environment_t* environment,
    const loom_check_run_arguments_t* arguments, iree_allocator_t allocator,
    loom_check_run_result_t* run_result) {
  if (provider->execute == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "RUN: run provider '%.*s' has no execute callback",
                            (int)provider->name.size, provider->name.data);
  }
  const loom_check_run_provider_request_t request = {
      .filename = filename,
      .test_case = test_case,
      .environment = environment,
      .arguments = arguments,
      .host_allocator = allocator,
      .result = run_result,
  };
  return provider->execute(provider, &request);
}

static iree_status_t loom_check_run_execute_external(
    const loom_check_case_t* test_case,
    const loom_check_environment_t* environment,
    const loom_check_run_arguments_t* run_arguments, iree_allocator_t allocator,
    loom_check_run_result_t* run_result) {
  loom_tool_temp_file_t input_file = {0};
  iree_string_view_t* process_arguments = NULL;
  iree_host_size_t process_argument_count = 0;
  loom_tool_process_result_t process_result = {0};

  iree_status_t status =
      loom_tool_temp_file_allocate(IREE_SV("loom_run"), &input_file);
  if (iree_status_is_ok(status)) {
    status = iree_io_file_contents_write(
        loom_tool_temp_file_path(&input_file),
        iree_make_const_byte_span((const uint8_t*)test_case->input.data,
                                  test_case->input.size),
        allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_check_run_build_process_arguments(
        loom_tool_temp_file_path(&input_file), run_arguments, allocator,
        &process_arguments, &process_argument_count);
  }
  if (iree_status_is_ok(status)) {
    iree_string_view_t executable_path =
        loom_check_iree_run_loom_path(environment);
    status = loom_tool_process_run(
        executable_path, loom_check_process_path_searches_path(executable_path),
        process_arguments, process_argument_count, allocator, &process_result);
  }
  if (iree_status_is_ok(status)) {
    run_result->exit_code = process_result.exit_code;
    status = iree_string_builder_append_string(
        &run_result->stdout_text,
        iree_make_string_view(process_result.stdout_text.data,
                              process_result.stdout_text.length));
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(
        &run_result->stderr_text,
        iree_make_string_view(process_result.stderr_text.data,
                              process_result.stderr_text.length));
  }

  loom_tool_process_result_deinitialize(&process_result, allocator);
  iree_allocator_free(allocator, process_arguments);
  loom_tool_temp_file_deinitialize(&input_file);
  return status;
}

static iree_status_t loom_check_run_append_output(
    iree_string_view_t stdout_text, iree_string_view_t stderr_text,
    int exit_code, loom_check_result_t* result) {
  IREE_ASSERT_ARGUMENT(result);
  const bool use_envelope = exit_code != 0 || stderr_text.size != 0;
  if (use_envelope) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        &result->actual_output, "exit_code: %d\nstdout:\n", exit_code));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(&result->actual_output, stdout_text));
  if (use_envelope) {
    if (stdout_text.size > 0 &&
        stdout_text.data[stdout_text.size - 1] != '\n') {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(&result->actual_output, "\n"));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        &result->actual_output, "stderr:\n"));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(&result->actual_output, stderr_text));
  }
  result->has_actual_output = true;
  return iree_ok_status();
}

static iree_status_t loom_check_run_record_nonzero_exit(
    int exit_code, loom_check_result_t* result) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      &result->detail, "RUN: run exited with code %d\n", exit_code));
  return iree_string_builder_append_string(
      &result->detail, iree_string_builder_view(&result->actual_output));
}

iree_status_t loom_check_execute_run(
    const loom_check_case_t* test_case, iree_string_view_t filename,
    const loom_check_environment_t* environment, iree_allocator_t allocator,
    loom_check_result_t* result) {
  IREE_ASSERT_ARGUMENT(test_case);
  IREE_ASSERT_ARGUMENT(result);

  loom_check_run_argument_storage_t run_arguments = {0};
  loom_check_run_result_t run_result = {0};
  loom_check_run_result_initialize(allocator, &run_result);
  bool has_run_result = false;

  iree_status_t status = loom_check_run_parse_arguments(
      test_case->run_arguments, allocator, &run_arguments);
  if (iree_status_is_ok(status)) {
    const loom_check_run_provider_t* provider =
        loom_check_run_lookup_provider(environment, &run_arguments.arguments);
    if (provider != NULL) {
      status = loom_check_run_execute_provider(
          provider, test_case, filename, environment, &run_arguments.arguments,
          allocator, &run_result);
      has_run_result = iree_status_is_ok(status);
    } else if (loom_check_run_case_has_requirement(test_case,
                                                   IREE_SV("iree-run-loom"))) {
      status = loom_check_run_execute_external(test_case, environment,
                                               &run_arguments.arguments,
                                               allocator, &run_result);
      has_run_result = iree_status_is_ok(status);
    } else {
      status = loom_check_run_fail_no_provider(environment, result);
    }
  }
  if (iree_status_is_ok(status) && has_run_result) {
    status = loom_check_run_append_output(
        iree_string_builder_view(&run_result.stdout_text),
        iree_string_builder_view(&run_result.stderr_text), run_result.exit_code,
        result);
  }
  if (iree_status_is_ok(status) && has_run_result &&
      run_result.exit_code != 0) {
    result->raw_outcome = LOOM_CHECK_FAIL;
    status = loom_check_run_record_nonzero_exit(run_result.exit_code, result);
  }
  if (iree_status_is_ok(status) && has_run_result &&
      run_result.exit_code == 0 && result->has_actual_output) {
    iree_string_view_t actual_trimmed =
        iree_string_view_trim(iree_string_builder_view(&result->actual_output));
    iree_string_view_t expected_trimmed =
        iree_string_view_trim(test_case->expected);
    if (iree_string_view_equal(actual_trimmed, expected_trimmed)) {
      result->raw_outcome = LOOM_CHECK_PASS;
    } else {
      result->raw_outcome = LOOM_CHECK_FAIL;
      status = loom_check_result_record_diff(expected_trimmed, actual_trimmed,
                                             allocator, result);
    }
  }

  loom_check_run_result_deinitialize(&run_result);
  loom_check_run_arguments_deinitialize(&run_arguments, allocator);
  return status;
}
