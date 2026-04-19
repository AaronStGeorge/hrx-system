// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/report.h"

#include <inttypes.h>
#include <string.h>

#include "loom/pass/program.h"
#include "loom/pass/registry.h"
#include "loom/util/json.h"

void loom_pass_report_initialize(iree_allocator_t allocator,
                                 loom_pass_report_t* out_report) {
  *out_report = (loom_pass_report_t){
      .allocator = allocator,
  };
}

void loom_pass_report_deinitialize(loom_pass_report_t* report) {
  for (iree_host_size_t i = 0; i < report->invocation_count; ++i) {
    iree_allocator_free(report->allocator, report->invocations[i].statistics);
  }
  iree_allocator_free(report->allocator, report->invocations);
  memset(report, 0, sizeof(*report));
}

static iree_status_t loom_pass_report_ensure_invocation_capacity(
    loom_pass_report_t* report) {
  if (report->invocation_count < report->invocation_capacity) {
    return iree_ok_status();
  }
  iree_host_size_t new_capacity =
      report->invocation_capacity > 0 ? report->invocation_capacity * 2 : 16;
  IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
      report->allocator, new_capacity, sizeof(*report->invocations),
      (void**)&report->invocations));
  memset(report->invocations + report->invocation_capacity, 0,
         (new_capacity - report->invocation_capacity) *
             sizeof(*report->invocations));
  report->invocation_capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t loom_pass_report_copy_statistics(
    loom_pass_report_t* report, const loom_pass_info_t* pass_info,
    const int64_t* statistic_values,
    loom_pass_report_invocation_t* out_invocation) {
  out_invocation->statistic_count = pass_info->statistic_count;
  if (pass_info->statistic_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(report->allocator, pass_info->statistic_count,
                                  sizeof(*out_invocation->statistics),
                                  (void**)&out_invocation->statistics));
  for (uint16_t i = 0; i < pass_info->statistic_count; ++i) {
    out_invocation->statistics[i] = (loom_pass_report_statistic_t){
        .name = pass_info->statistic_defs[i].name,
        .value = statistic_values ? statistic_values[i] : 0,
    };
  }
  return iree_ok_status();
}

iree_status_t loom_pass_report_append_invocation(
    loom_pass_report_t* report,
    const loom_pass_report_invocation_options_t* options) {
  if (!report || !options || !options->instruction ||
      options->instruction->kind != LOOM_PASS_PROGRAM_INSTRUCTION_INVOKE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass report invocation requires an invoke instruction");
  }
  const loom_pass_program_invoke_t* invoke = &options->instruction->invoke;
  if (!invoke->descriptor || !invoke->info) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass report invocation requires descriptor metadata");
  }

  IREE_RETURN_IF_ERROR(loom_pass_report_ensure_invocation_capacity(report));
  loom_pass_report_invocation_t invocation = {
      .descriptor = invoke->descriptor,
      .pass_key = invoke->descriptor->key,
      .pass_kind = invoke->info->kind,
      .anchor_kind = options->anchor_kind,
      .pipeline_symbol = options->pipeline_symbol,
      .symbol_name = options->symbol_name,
      .instruction_index = options->instruction_index,
      .duration_nanoseconds = options->duration_nanoseconds,
      .changed = options->changed,
      .status_code = options->status_code,
  };
  IREE_RETURN_IF_ERROR(loom_pass_report_copy_statistics(
      report, invoke->info, options->statistics, &invocation));
  report->invocations[report->invocation_count++] = invocation;
  return iree_ok_status();
}

static iree_string_view_t loom_pass_report_kind_name(loom_pass_kind_t kind) {
  switch (kind) {
    case LOOM_PASS_MODULE:
      return IREE_SV("module");
    case LOOM_PASS_FUNCTION:
      return IREE_SV("func");
    default:
      return IREE_SV("unknown");
  }
}

static iree_status_t loom_pass_report_write_json_string_field(
    loom_output_stream_t* stream, const char* name, iree_string_view_t value) {
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_format(stream, "\"%s\":", name));
  return loom_json_write_escaped_string(stream, value);
}

iree_status_t loom_pass_report_format_json(const loom_pass_report_t* report,
                                           loom_output_stream_t* stream) {
  if (!report || !stream) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass report and output stream are required");
  }

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "{\n  \"invocations\": ["));
  for (iree_host_size_t i = 0; i < report->invocation_count; ++i) {
    const loom_pass_report_invocation_t* invocation = &report->invocations[i];
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        stream, i == 0 ? "\n    {" : ",\n    {"));
    IREE_RETURN_IF_ERROR(loom_pass_report_write_json_string_field(
        stream, "pass", invocation->pass_key));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    IREE_RETURN_IF_ERROR(loom_pass_report_write_json_string_field(
        stream, "kind", loom_pass_report_kind_name(invocation->pass_kind)));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    IREE_RETURN_IF_ERROR(loom_pass_report_write_json_string_field(
        stream, "anchor", loom_pass_report_kind_name(invocation->anchor_kind)));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    IREE_RETURN_IF_ERROR(loom_pass_report_write_json_string_field(
        stream, "pipeline", invocation->pipeline_symbol));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    IREE_RETURN_IF_ERROR(loom_pass_report_write_json_string_field(
        stream, "symbol", invocation->symbol_name));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream,
        ",\"instruction_index\":%" PRIhsz ",\"duration_ns\":%" PRId64
        ",\"changed\":%s,\"status\":",
        invocation->instruction_index, invocation->duration_nanoseconds,
        invocation->changed ? "true" : "false"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
        stream, iree_status_code_string(invocation->status_code)));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"statistics\":{"));
    for (uint16_t j = 0; j < invocation->statistic_count; ++j) {
      const loom_pass_report_statistic_t* statistic =
          &invocation->statistics[j];
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, j == 0 ? "" : ","));
      IREE_RETURN_IF_ERROR(
          loom_json_write_escaped_string(stream, statistic->name));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(stream, ":%" PRId64,
                                                           statistic->value));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}}"));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\n  ]\n}\n"));
  return iree_ok_status();
}
