// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/json_output.h"

#include "loom/util/json.h"

static const char* loom_check_outcome_string(loom_check_outcome_t outcome) {
  return outcome == LOOM_CHECK_PASS ? "pass" : "fail";
}

iree_status_t loom_check_json_write_file_result(
    iree_string_view_t filename, const loom_check_file_t* file,
    const loom_check_result_t* results, iree_host_size_t pass_count,
    iree_host_size_t fail_count, iree_host_size_t skip_count,
    loom_output_stream_t* stream) {
  // Open the root object.
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{\n"));

  // "file": "<filename>"
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "  \"file\": "));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(stream, filename));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\n"));

  // "default_mode": "<mode>"
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "  \"default_mode\": \""));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      stream, loom_check_mode_name(file->default_mode)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\",\n"));

  // "cases": [...]
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "  \"cases\": [\n"));

  for (iree_host_size_t i = 0; i < file->case_count; ++i) {
    const loom_check_case_t* test_case = &file->cases[i];
    const loom_check_result_t* result = &results[i];

    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "    {\n"));

    // "index": N (1-based for human readability)
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, "      \"index\": %zu,\n", i + 1));

    // "mode": "<mode>"
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, "      \"mode\": \""));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        stream, loom_check_mode_name(test_case->mode)));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\",\n"));

    // "xfail": true/false
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, "      \"xfail\": %s,\n", test_case->xfail ? "true" : "false"));

    // "xfail_reason": "<reason>" (only when xfail is true)
    if (test_case->xfail) {
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, "      \"xfail_reason\": "));
      IREE_RETURN_IF_ERROR(
          loom_json_write_escaped_string(stream, test_case->xfail_reason));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\n"));
    }

    // "raw_outcome": "pass"/"fail"
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, "      \"raw_outcome\": \"%s\",\n",
        loom_check_outcome_string(result->raw_outcome)));

    // "final_outcome": "pass"/"fail"
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, "      \"final_outcome\": \"%s\",\n",
        loom_check_outcome_string(result->final_outcome)));

    // "detail": "<escaped detail>"
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, "      \"detail\": "));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, iree_string_builder_view(&result->detail)));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\n"));

    // Close the case object.
    if (i + 1 < file->case_count) {
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, "    },\n"));
    } else {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "    }\n"));
    }
  }

  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "  ],\n"));

  // "summary": {...}
  iree_host_size_t total = pass_count + fail_count + skip_count;
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "  \"summary\": {\n"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_format(stream, "    \"total\": %zu,\n", total));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "    \"passed\": %zu,\n", pass_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "    \"failed\": %zu,\n", fail_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "    \"skipped\": %zu\n", skip_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "  }\n"));

  // Close root object.
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}\n"));
  return iree_ok_status();
}
