// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-benchmark/diagnostics.h"

#include "loom/error/json_sink.h"
#include "loom/error/renderer.h"
#include "loom/tooling/cli/loomc_diagnostic_json.h"

void loom_benchmark_diagnostic_capture_initialize(
    iree_allocator_t allocator, loom_benchmark_diagnostic_capture_t* capture) {
  *capture = (loom_benchmark_diagnostic_capture_t){
      .initialized = true,
  };
  loom_json_value_list_initialize(allocator, &capture->json_values);
}

void loom_benchmark_diagnostic_capture_deinitialize(
    loom_benchmark_diagnostic_capture_t* capture) {
  if (capture == NULL || !capture->initialized) {
    return;
  }
  loom_json_value_list_deinitialize(&capture->json_values);
  *capture = (loom_benchmark_diagnostic_capture_t){0};
}

iree_string_view_t loom_benchmark_diagnostic_capture_json(
    const loom_benchmark_diagnostic_capture_t* capture) {
  if (!capture || !capture->initialized) {
    return iree_string_view_empty();
  }
  return loom_json_value_list_body(&capture->json_values);
}

iree_status_t loom_benchmark_write_diagnostic_array_json(
    const loom_benchmark_diagnostic_capture_t* capture,
    loom_output_stream_t* stream) {
  if (!capture || !capture->initialized) {
    return loom_json_write_value_list_array(iree_string_view_empty(), stream);
  }
  return loom_json_value_list_write_array(&capture->json_values, stream);
}

iree_status_t loom_benchmark_diagnostic_capture_sink(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  loom_benchmark_diagnostic_capture_t* capture =
      (loom_benchmark_diagnostic_capture_t*)user_data;
  loom_output_stream_t stream;
  IREE_RETURN_IF_ERROR(
      loom_json_value_list_begin_value(&capture->json_values, &stream));
  IREE_RETURN_IF_ERROR(loom_diagnostic_json_write_object(
      &stream, diagnostic,
      (loom_type_formatter_t){loom_type_format_minimal, NULL}));
  switch (diagnostic->severity) {
    case LOOM_DIAGNOSTIC_ERROR:
      ++capture->error_count;
      break;
    case LOOM_DIAGNOSTIC_WARNING:
      ++capture->warning_count;
      break;
    case LOOM_DIAGNOSTIC_REMARK:
      ++capture->remark_count;
      break;
    default:
      break;
  }
  return iree_ok_status();
}

iree_status_t loom_benchmark_diagnostic_capture_loomc_result(
    loom_benchmark_diagnostic_capture_t* capture,
    const loomc_result_t* result) {
  for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result);
       ++i) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, i);
    loom_output_stream_t stream;
    IREE_RETURN_IF_ERROR(
        loom_json_value_list_begin_value(&capture->json_values, &stream));
    IREE_RETURN_IF_ERROR(
        loom_tooling_cli_write_loomc_diagnostic_json(&stream, diagnostic));
    switch (diagnostic->severity) {
      case LOOMC_DIAGNOSTIC_SEVERITY_NOTE:
        ++capture->remark_count;
        break;
      case LOOMC_DIAGNOSTIC_SEVERITY_WARNING:
        ++capture->warning_count;
        break;
      case LOOMC_DIAGNOSTIC_SEVERITY_ERROR:
        ++capture->error_count;
        break;
    }
  }
  return iree_ok_status();
}
