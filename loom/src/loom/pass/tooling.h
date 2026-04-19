// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tool-facing pass pipeline execution helpers.
//
// This is the cold boundary used by command-line tools and tests. It adapts
// user-facing selections such as named pass.pipeline symbols or shallow textual
// pass lists into the verifier/compiler/interpreter path. Concrete pass
// behavior still comes exclusively from descriptor registries.

#ifndef LOOM_PASS_TOOLING_H_
#define LOOM_PASS_TOOLING_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/pass/interpreter.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_pass_tool_run_options_t {
  // Registry used to resolve pass.run keys. Required.
  const loom_pass_registry_t* registry;
  // Optional descriptor requirement provider.
  loom_pass_requirement_provider_t requirement_provider;
  // Shared block pool used for compilation, execution, and scratch modules.
  iree_arena_block_pool_t* block_pool;
  // Optional structured diagnostic emitter copied into every pass instance.
  iree_diagnostic_emitter_t diagnostic_emitter;
  // Optional user-data callback for pass instances.
  loom_pass_interpreter_configure_callback_t configure;
} loom_pass_tool_run_options_t;

// Compiles and executes one pass.pipeline op on |module|. Module-root pipelines
// run once on the module. Function-root pipelines run once per bodyful
// function-like symbol using a deterministic symbol snapshot.
iree_status_t loom_pass_tool_run_pipeline_op(
    loom_module_t* module, const loom_op_t* pipeline_op,
    const loom_pass_tool_run_options_t* options);

// Looks up a module-local pass.pipeline symbol by name and executes it. The
// symbol may be spelled with or without a leading '@'.
iree_status_t loom_pass_tool_run_pipeline_symbol(
    loom_module_t* module, iree_string_view_t pipeline_symbol,
    const loom_pass_tool_run_options_t* options);

// Parses a shallow comma-separated pass list such as
// `canonicalize{max-iterations=20},dce`, converts it into a synthetic
// module-root pass.pipeline backed by descriptor options, and executes it.
// Function passes are wrapped in one pass.for<func> per entry so the execution
// order matches the legacy flat pass manager.
iree_status_t loom_pass_tool_run_flat_pipeline(
    loom_module_t* module, iree_string_view_t pipeline,
    const loom_pass_tool_run_options_t* options);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_PASS_TOOLING_H_
