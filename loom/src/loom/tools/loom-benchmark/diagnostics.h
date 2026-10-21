// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structured diagnostic capture used by parse, verify, and candidate compile.

#ifndef LOOM_TOOLS_LOOM_BENCHMARK_DIAGNOSTICS_H_
#define LOOM_TOOLS_LOOM_BENCHMARK_DIAGNOSTICS_H_

#include "iree/base/api.h"
#include "loom/error/diagnostic.h"
#include "loom/tools/loom-benchmark/model.h"
#include "loom/util/stream.h"
#include "loomc/result.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initializes |capture| to own a deferred diagnostic value list.
void loom_benchmark_diagnostic_capture_initialize(
    iree_allocator_t allocator, loom_benchmark_diagnostic_capture_t* capture);

// Releases storage owned by |capture|.
void loom_benchmark_diagnostic_capture_deinitialize(
    loom_benchmark_diagnostic_capture_t* capture);

// Returns the captured JSON array body without surrounding brackets.
iree_string_view_t loom_benchmark_diagnostic_capture_json(
    const loom_benchmark_diagnostic_capture_t* capture);

// Writes the captured diagnostics as a complete JSON array.
iree_status_t loom_benchmark_write_diagnostic_array_json(
    const loom_benchmark_diagnostic_capture_t* capture,
    loom_output_stream_t* stream);

// Diagnostic sink callback that appends one structured JSON diagnostic.
iree_status_t loom_benchmark_diagnostic_capture_sink(
    void* user_data, const loom_diagnostic_t* diagnostic);

// Appends every structured diagnostic in one public compiler result.
iree_status_t loom_benchmark_diagnostic_capture_loomc_result(
    loom_benchmark_diagnostic_capture_t* capture, const loomc_result_t* result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_BENCHMARK_DIAGNOSTICS_H_
