// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structured benchmark runner entry points below the command-line adapter.

#ifndef LOOM_TOOLS_LOOM_BENCHMARK_RUN_H_
#define LOOM_TOOLS_LOOM_BENCHMARK_RUN_H_

#include "iree/base/api.h"
#include "loom/tools/loom-benchmark/configuration.h"
#include "loom/tools/loom-benchmark/event.h"
#include "loom/tools/loom-benchmark/options.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_benchmark_file_run_options_t {
  // Borrowed target-linked runner configuration.
  const loom_benchmark_configuration_t* configuration;
  // Borrowed parsed options; selected @symbols are normalized inside the run.
  const loom_benchmark_options_t* benchmark_options;
  // Input module path, empty for stdin, or "-" for stdin.
  iree_string_view_t input_path;
  // JSON command-line vector recorded in artifact manifests.
  iree_string_view_t command_line_json;
  // Optional caller-owned event sink; NULL selects configured result output.
  const loom_benchmark_event_sink_t* event_sink;
  // Host allocator used for run-owned storage.
  iree_allocator_t host_allocator;
} loom_benchmark_file_run_options_t;

typedef struct loom_benchmark_run_result_t {
  // Process-style exit code implied by correctness and infrastructure status.
  int exit_code;
} loom_benchmark_run_result_t;

// Runs one benchmark input file or stdin using structured benchmark options.
iree_status_t loom_benchmark_run_file(
    const loom_benchmark_file_run_options_t* options,
    loom_benchmark_run_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_BENCHMARK_RUN_H_
