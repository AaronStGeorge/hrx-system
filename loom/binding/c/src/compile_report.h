// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_COMPILE_REPORT_STATE_H_
#define LOOMC_COMPILE_REPORT_STATE_H_

#include "iree/base/internal/arena.h"
#include "iree/io/stream.h"
#include "loom/error/diagnostic.h"
#include "loom/format/text/printer.h"
#include "loom/target/reporting/format.h"
#include "loomc/compile_report.h"
#include "loomc/module.h"
#include "loomc/result.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Scoped observations for one invocation. Only formatted result bytes survive.
typedef struct loomc_compile_report_state_t {
  // Selected detail mode. NONE denotes an uninitialized, disabled state.
  loom_target_compile_report_format_mode_t mode;
  // Selected serialization format.
  loomc_compile_report_format_t format;
  // Borrowed result receiving diagnostics and the final report artifact.
  loomc_result_t* result;
  // Pooled storage for native report rows and captured diagnostic text.
  iree_arena_allocator_t arena;
  // Native facts collected while module and target identities remain alive.
  loom_target_compile_report_t report;
  // Canonical JSON objects, allocated lazily for detailed JSON reports.
  iree_io_stream_t* diagnostics;
  // Result sink capturing canonical diagnostics before their owners expire.
  loom_diagnostic_sink_t diagnostic_sink;
  // Borrowed target-aware printer options for diagnostic types.
  const loom_text_print_options_t* diagnostic_text_print_options;
} loomc_compile_report_state_t;

// Validates the public report descriptor at an invocation boundary.
LOOMC_API_PRIVATE loomc_status_t loomc_compile_report_options_validate(
    const loomc_compile_report_options_t* options);

// Initializes a validated, enabled report without allocating storage.
LOOMC_API_PRIVATE void loomc_compile_report_state_initialize(
    loomc_compile_report_mode_t mode, loomc_compile_report_format_t format,
    iree_arena_block_pool_t* block_pool, loomc_result_t* result,
    loomc_compile_report_state_t* out_state);

// Copies the module's retained configuration facts into the current report.
LOOMC_API_PRIVATE loomc_status_t loomc_compile_report_record_config_bindings(
    loomc_compile_report_state_t* state, const loomc_module_t* module);

// Serializes one immutable artifact. Called only after successful execution or
// a diagnosed compiler failure, never to recover from an infrastructure error.
LOOMC_API_PRIVATE loomc_status_t loomc_compile_report_state_finish(
    loomc_compile_report_state_t* state, loomc_string_view_t identifier);

// Releases scoped storage, including any retained target-specific objects.
// A zero-initialized disabled state requires no cleanup.
LOOMC_API_PRIVATE void loomc_compile_report_state_deinitialize(
    loomc_compile_report_state_t* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_COMPILE_REPORT_STATE_H_
