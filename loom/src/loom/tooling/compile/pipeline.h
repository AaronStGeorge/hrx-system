// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared Loom compile pass-pipeline execution.
//
// This is the production tooling boundary for applying the user-selected
// compile pipeline to a parsed module. Artifact emission and runtime invocation
// are separate phases layered on top of this.

#ifndef LOOM_TOOLING_COMPILE_PIPELINE_H_
#define LOOM_TOOLING_COMPILE_PIPELINE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/diagnostic.h"
#include "loom/ir/ir.h"
#include "loom/pass/interpreter.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/verify/verify.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_environment_t loom_target_environment_t;

typedef enum loom_compile_default_pipeline_e {
  // Build the shared source/kernel-to-target-low pipeline. This preserves the
  // current artifact-emission boundary where target emitters finish low
  // materialization until that remaining orchestration moves into pass IR.
  LOOM_COMPILE_DEFAULT_PIPELINE_SOURCE_LOW = 0,
  // Build the full prepared target-low pipeline including target ABI/resource
  // materialization and packetization preparation.
  LOOM_COMPILE_DEFAULT_PIPELINE_PREPARED_LOW = 1,
} loom_compile_default_pipeline_t;

typedef struct loom_compile_pipeline_options_t {
  // Pass pipeline spelling. Empty or "default" runs |default_pipeline|;
  // "none" skips pass execution; "@symbol" runs a module-local pass.pipeline;
  // otherwise the value is parsed as a comma-separated pass list.
  iree_string_view_t pipeline;
  // Default pipeline used when |pipeline| is empty or "default".
  loom_compile_default_pipeline_t default_pipeline;
  // Optional function symbol used for target-aware pass predicates and
  // diagnostics. A leading '@' is accepted.
  iree_string_view_t entry_symbol;
  // Target environment linked into this compile front door.
  const loom_target_environment_t* target_environment;
  // Target-low descriptor registry package initialized for this session.
  const loom_target_low_descriptor_registry_t* low_descriptor_registry;
  // Diagnostic sink used by pass execution.
  loom_diagnostic_sink_t diagnostic_sink;
  // Source resolver used to render source-attributed diagnostics.
  loom_source_resolver_t source_resolver;
  // Maximum pass diagnostics to emit before stopping. Zero uses a conservative
  // default.
  uint32_t max_errors;
} loom_compile_pipeline_options_t;

// Initializes compile pipeline options with the current artifact-front-door
// default: source/kernel-to-target-low, stderr diagnostics, and a small error
// cap.
void loom_compile_pipeline_options_initialize(
    loom_compile_pipeline_options_t* out_options);

// Returns true when |pipeline| disables pass execution.
bool loom_compile_pipeline_is_disabled(iree_string_view_t pipeline);

// Returns true when |pipeline| requests the configured default pipeline.
bool loom_compile_pipeline_is_default(iree_string_view_t pipeline);

// Runs the selected compile pipeline on |module|.
//
// Status is reserved for infrastructure failures. Pass-emitted errors are
// counted in |out_result| and converted to FAILED_PRECONDITION by this helper
// so command-line front doors share one failure policy.
iree_status_t loom_compile_run_pipeline(
    loom_module_t* module, const loom_compile_pipeline_options_t* options,
    iree_arena_block_pool_t* block_pool, loom_pass_run_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_COMPILE_PIPELINE_H_
