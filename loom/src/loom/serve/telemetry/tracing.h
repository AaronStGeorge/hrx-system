// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_TELEMETRY_TRACING_H_
#define LOOM_SERVE_TELEMETRY_TRACING_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Serving categories share the application's process-wide tracing namespace.
// These describe elapsed operations, independently of worker thread ownership.
typedef enum loom_serve_trace_section_e {
  LOOM_SERVE_TRACE_SECTION_PREPARE = 1,
  LOOM_SERVE_TRACE_SECTION_WARMUP = 2,
  LOOM_SERVE_TRACE_SECTION_DEACTIVATE = 3,
  LOOM_SERVE_TRACE_SECTION_RETIRE = 4,
  LOOM_SERVE_TRACE_SECTION_EXECUTE = 5,
} loom_serve_trace_section_t;

// Publishes category names when constructing a serving registry. Repeated
// setup preserves the same names across multiple registries in one process.
void loom_serve_trace_initialize(void);

// Enters an independently overlapping "model-<operation>:<name>" section.
// The provider consumes the deployment name during entry. The operation owner
// leaves the returned ID exactly once at its terminal result, including failure
// or cancellation; no trace-only waiter is needed. Execution sections span an
// accepted model unit through observed completion, not CPU preparation or
// response encoding. Callers guard section storage and entry with IREE_TRACE.
iree_section_id_t loom_serve_trace_section_enter(
    loom_serve_trace_section_t section, iree_string_view_t name);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_TELEMETRY_TRACING_H_
