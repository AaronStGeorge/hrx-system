// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "diagnostic.h"

#include "iree/base/api.h"
#include "loom/error/error_defs.h"
#include "loom/error/renderer.h"
#include "loom/util/stream.h"
#include "loomc/iree.h"

static loomc_diagnostic_severity_t loomc_diagnostic_severity_from_loom(
    loom_diagnostic_severity_t severity) {
  switch (severity) {
    case LOOM_DIAGNOSTIC_ERROR:
      return LOOMC_DIAGNOSTIC_SEVERITY_ERROR;
    case LOOM_DIAGNOSTIC_WARNING:
      return LOOMC_DIAGNOSTIC_SEVERITY_WARNING;
    case LOOM_DIAGNOSTIC_REMARK:
      return LOOMC_DIAGNOSTIC_SEVERITY_NOTE;
    case LOOM_DIAGNOSTIC_COUNT_:
      break;
  }
  return LOOMC_DIAGNOSTIC_SEVERITY_ERROR;
}

static loomc_status_t loomc_format_loom_diagnostic_code(
    const loom_diagnostic_t* diagnostic, iree_string_builder_t* builder) {
  const char* domain = loom_error_domain_name(diagnostic->error->domain);
  return loomc_status_from_iree(iree_string_builder_append_format(
      builder, "%s/%03u", domain, diagnostic->error->code));
}

static loomc_status_t loomc_render_loom_diagnostic_message(
    const loom_diagnostic_t* diagnostic, iree_string_builder_t* builder) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  loom_type_formatter_t formatter = {loom_type_format_minimal, NULL};
  return loomc_status_from_iree(loom_diagnostic_render_message(
      diagnostic->error, diagnostic->params, diagnostic->param_count, formatter,
      &stream));
}

static loomc_source_range_t loomc_source_range_from_loom(
    const loomc_source_t* source, const loom_source_range_t* range) {
  if (source == NULL || range == NULL) {
    return (loomc_source_range_t){0};
  }
  return (loomc_source_range_t){
      .source = source,
      .start = range->start,
      .end = range->end,
      .start_line = range->start_line,
      .start_column = range->start_column,
      .end_line = range->end_line,
      .end_column = range->end_column,
  };
}

loomc_status_t loomc_result_add_loom_diagnostic(
    loomc_result_t* result, const loomc_source_t* source,
    const loom_diagnostic_t* diagnostic) {
  if (result == NULL || diagnostic == NULL || diagnostic->error == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "result and diagnostic must not be NULL");
  }
  iree_allocator_t allocator =
      iree_allocator_from_loomc(loomc_result_allocator(result));
  iree_string_builder_t code_builder;
  iree_string_builder_initialize(allocator, &code_builder);
  iree_string_builder_t message_builder;
  iree_string_builder_initialize(allocator, &message_builder);

  loomc_status_t status =
      loomc_format_loom_diagnostic_code(diagnostic, &code_builder);
  if (loomc_status_is_ok(status)) {
    status = loomc_render_loom_diagnostic_message(diagnostic, &message_builder);
  }
  if (loomc_status_is_ok(status)) {
    loomc_diagnostic_t public_diagnostic = {
        .severity = loomc_diagnostic_severity_from_loom(diagnostic->severity),
        .code = loomc_string_view_from_iree(
            iree_string_builder_view(&code_builder)),
        .message = loomc_string_view_from_iree(
            iree_string_builder_view(&message_builder)),
        .range =
            loomc_source_range_from_loom(source, &diagnostic->source_location),
    };
    status = loomc_result_add_diagnostic(result, &public_diagnostic);
  }

  iree_string_builder_deinitialize(&message_builder);
  iree_string_builder_deinitialize(&code_builder);
  return status;
}

loomc_status_t loomc_result_add_loom_diagnostic_emission(
    loomc_result_t* result, const loomc_source_t* source,
    loom_emitter_t emitter, const loom_diagnostic_emission_t* emission) {
  if (emission == NULL || emission->error == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "diagnostic emission must not be NULL");
  }
  loom_diagnostic_t diagnostic = {
      .severity = emission->error->severity,
      .error = emission->error,
      .params = emission->params,
      .param_count = emission->param_count,
      .emitter = emitter,
  };
  return loomc_result_add_loom_diagnostic(result, source, &diagnostic);
}

loomc_status_t loomc_result_add_status_diagnostic(
    loomc_result_t* result, const loomc_source_t* source,
    loomc_diagnostic_severity_t severity, loomc_string_view_t code,
    loomc_status_t status) {
  if (result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "result must not be NULL");
  }
  loomc_string_view_t message = loomc_status_message(status);
  if (loomc_string_view_is_empty(message)) {
    message = loomc_make_cstring_view(
        loomc_status_code_string(loomc_status_code(status)));
  }
  loomc_diagnostic_t diagnostic = {
      .severity = severity,
      .code = code,
      .message = message,
      .range =
          {
              .source = source,
          },
  };
  return loomc_result_add_diagnostic(result, &diagnostic);
}

loomc_status_t loomc_result_fail_status_diagnostic(
    loomc_result_t* result, const loomc_source_t* source,
    loomc_diagnostic_severity_t severity, loomc_string_view_t code,
    loomc_status_t status) {
  LOOMC_RETURN_IF_ERROR(loomc_result_add_status_diagnostic(
      result, source, severity, code, status));
  return loomc_result_set_state(result, LOOMC_RESULT_STATE_FAILED);
}
