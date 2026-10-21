// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-benchmark/output_sink.h"

#include <string.h>

iree_status_t loom_benchmark_output_sink_initialize(
    loom_benchmark_output_format_t format,
    iree_string_view_t results_output_path, iree_allocator_t allocator,
    loom_benchmark_output_sink_t* out_sink) {
  memset(out_sink, 0, sizeof(*out_sink));
  out_sink->format = format;
  switch (format) {
    case LOOM_BENCHMARK_OUTPUT_FORMAT_SNAPSHOT: {
      IREE_RETURN_IF_ERROR(loom_benchmark_snapshot_sink_initialize(
          allocator, &out_sink->snapshot_sink));
      out_sink->snapshot_sink_initialized = true;
      loom_benchmark_snapshot_event_sink_initialize(&out_sink->snapshot_sink,
                                                    &out_sink->event_sink);
      return iree_ok_status();
    }
    case LOOM_BENCHMARK_OUTPUT_FORMAT_JSONL: {
      IREE_RETURN_IF_ERROR(loom_benchmark_jsonl_sink_initialize(
          results_output_path, allocator, &out_sink->jsonl_sink));
      out_sink->jsonl_sink_initialized = true;
      loom_benchmark_jsonl_event_sink_initialize(&out_sink->jsonl_sink,
                                                 &out_sink->jsonl_event_sink,
                                                 &out_sink->event_sink);
      return iree_ok_status();
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported benchmark output format %d",
                              (int)format);
  }
}

iree_status_t loom_benchmark_output_sink_flush(
    loom_benchmark_output_sink_t* sink,
    iree_string_view_t results_output_path) {
  switch (sink->format) {
    case LOOM_BENCHMARK_OUTPUT_FORMAT_SNAPSHOT:
      return loom_benchmark_snapshot_sink_write(&sink->snapshot_sink,
                                                results_output_path);
    case LOOM_BENCHMARK_OUTPUT_FORMAT_JSONL:
      return loom_benchmark_jsonl_sink_close(&sink->jsonl_sink);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported benchmark output format %d",
                              (int)sink->format);
  }
}

void loom_benchmark_output_sink_deinitialize(
    loom_benchmark_output_sink_t* sink) {
  if (sink->jsonl_sink_initialized) {
    loom_benchmark_jsonl_event_sink_deinitialize(&sink->jsonl_event_sink);
  }
  if (sink->snapshot_sink_initialized) {
    loom_benchmark_snapshot_sink_deinitialize(&sink->snapshot_sink);
  }
  if (sink->jsonl_sink_initialized) {
    loom_benchmark_jsonl_sink_deinitialize(&sink->jsonl_sink);
  }
  *sink = (loom_benchmark_output_sink_t){0};
}
