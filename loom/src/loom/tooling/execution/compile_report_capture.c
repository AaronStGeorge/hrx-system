// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/compile_report_capture.h"

void loom_run_compile_report_capture_options_initialize(
    loom_run_compile_report_capture_options_t* out_options) {
  *out_options = (loom_run_compile_report_capture_options_t){
      .mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE,
      .row_limit = LOOM_RUN_COMPILE_REPORT_DEFAULT_ROW_LIMIT,
  };
}

iree_status_t loom_run_compile_report_capture_options_parse_mode(
    iree_string_view_t value,
    loom_run_compile_report_capture_options_t* options) {
  return loom_target_compile_report_format_mode_parse(value, &options->mode);
}

iree_status_t loom_run_compile_report_capture_options_parse_row_limit(
    iree_string_view_t value,
    loom_run_compile_report_capture_options_t* options) {
  uint64_t row_limit = 0;
  if (!iree_string_view_atoi_uint64(value, &row_limit) ||
      row_limit > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid compile report row limit '%.*s'",
                            (int)value.size, value.data);
  }
  options->row_limit = (iree_host_size_t)row_limit;
  return iree_ok_status();
}

static iree_status_t loom_run_compile_report_capture_allocate_rows(
    loom_run_compile_report_capture_t* capture) {
  if (capture->options.mode != LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS ||
      capture->options.row_limit == 0) {
    return iree_ok_status();
  }
  iree_host_size_t pressure_rows_size = 0;
  if (!iree_host_size_checked_mul(capture->options.row_limit,
                                  sizeof(*capture->pressure_rows),
                                  &pressure_rows_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "compile report pressure row storage is too large");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(capture->host_allocator,
                                             pressure_rows_size,
                                             (void**)&capture->pressure_rows));
  iree_host_size_t spill_rows_size = 0;
  if (!iree_host_size_checked_mul(capture->options.row_limit,
                                  sizeof(*capture->spill_rows),
                                  &spill_rows_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "compile report spill row storage is too large");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      capture->host_allocator, spill_rows_size, (void**)&capture->spill_rows));
  iree_host_size_t source_low_rows_size = 0;
  if (!iree_host_size_checked_mul(capture->options.row_limit,
                                  sizeof(*capture->source_low_rows),
                                  &source_low_rows_size)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "compile report source-low row storage is too large");
  }
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(capture->host_allocator, source_low_rows_size,
                            (void**)&capture->source_low_rows));
  iree_host_size_t target_legalization_rows_size = 0;
  if (!iree_host_size_checked_mul(capture->options.row_limit,
                                  sizeof(*capture->target_legalization_rows),
                                  &target_legalization_rows_size)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "compile report target-legalization row storage is too large");
  }
  return iree_allocator_malloc(capture->host_allocator,
                               target_legalization_rows_size,
                               (void**)&capture->target_legalization_rows);
}

iree_status_t loom_run_compile_report_capture_initialize(
    const loom_run_compile_report_capture_options_t* options,
    iree_allocator_t host_allocator,
    loom_run_compile_report_capture_t* out_capture) {
  *out_capture = (loom_run_compile_report_capture_t){
      .options = *options,
      .host_allocator = host_allocator,
  };
  loom_target_compile_report_initialize(&out_capture->report);
  iree_status_t status =
      loom_run_compile_report_capture_allocate_rows(out_capture);
  if (!iree_status_is_ok(status)) {
    loom_run_compile_report_capture_deinitialize(out_capture);
  }
  return status;
}

void loom_run_compile_report_capture_configure_compile_options(
    loom_run_compile_report_capture_t* capture,
    loom_run_candidate_compile_options_t* compile_options) {
  if (capture->options.mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE) {
    return;
  }
  compile_options->report = &capture->report;
  compile_options->report_row_storage =
      (loom_target_compile_report_row_storage_t){
          .pressure_rows = capture->pressure_rows,
          .pressure_row_capacity =
              capture->pressure_rows != NULL ? capture->options.row_limit : 0,
          .spill_rows = capture->spill_rows,
          .spill_row_capacity =
              capture->spill_rows != NULL ? capture->options.row_limit : 0,
          .source_low_rows = capture->source_low_rows,
          .source_low_row_capacity =
              capture->source_low_rows != NULL ? capture->options.row_limit : 0,
          .target_legalization_rows = capture->target_legalization_rows,
          .target_legalization_row_capacity =
              capture->target_legalization_rows != NULL
                  ? capture->options.row_limit
                  : 0,
      };
  loom_target_compile_report_set_row_storage(
      &capture->report, &compile_options->report_row_storage);
}

static iree_status_t loom_run_compile_report_capture_append_separator(
    iree_string_builder_t* builder) {
  iree_host_size_t builder_size = iree_string_builder_size(builder);
  if (builder_size == 0) {
    return iree_ok_status();
  }
  const char* buffer = iree_string_builder_buffer(builder);
  if (buffer[builder_size - 1] == '\n') {
    return iree_ok_status();
  }
  return iree_string_builder_append_string(builder, IREE_SV("\n"));
}

iree_status_t loom_run_compile_report_capture_append_text(
    const loom_run_compile_report_capture_t* capture,
    iree_string_builder_t* builder) {
  if (capture->options.mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_run_compile_report_capture_append_separator(builder));
  const loom_target_compile_report_format_options_t format_options = {
      .mode = capture->options.mode,
  };
  return loom_target_compile_report_format_text(&capture->report,
                                                &format_options, builder);
}

iree_status_t loom_run_compile_report_capture_append_json(
    const loom_run_compile_report_capture_t* capture,
    loom_output_stream_t* stream) {
  if (capture->options.mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE) {
    return iree_ok_status();
  }
  const loom_target_compile_report_format_options_t format_options = {
      .mode = capture->options.mode,
  };
  return loom_target_compile_report_format_json(&capture->report,
                                                &format_options, stream);
}

void loom_run_compile_report_capture_deinitialize(
    loom_run_compile_report_capture_t* capture) {
  if (capture == NULL) {
    return;
  }
  iree_allocator_free(capture->host_allocator, capture->spill_rows);
  iree_allocator_free(capture->host_allocator, capture->pressure_rows);
  iree_allocator_free(capture->host_allocator, capture->source_low_rows);
  iree_allocator_free(capture->host_allocator,
                      capture->target_legalization_rows);
  *capture = (loom_run_compile_report_capture_t){0};
}
