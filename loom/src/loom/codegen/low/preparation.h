// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-low IR preparation before packetization.
//
// This is the production bridge between source-to-low lowering and
// packetization: it runs the fixed low-prep pass pipeline through the normal
// pass program/interpreter infrastructure on selected low functions. It does
// not parse tool pipeline strings, emit artifacts, or construct tables.

#ifndef LOOM_CODEGEN_LOW_PREPARATION_H_
#define LOOM_CODEGEN_LOW_PREPARATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/pass/registry.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_preparation_options_t {
  // Pass registry used to resolve the production low-prep pass program.
  const loom_pass_registry_t* pass_registry;
  // Low descriptor registry linked into the current compiler binary.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Structured diagnostic emitter forwarded to low-prep passes.
  iree_diagnostic_emitter_t diagnostic_emitter;
} loom_low_preparation_options_t;

// Runs the production low-prep pass pipeline over |low_func_ops|.
//
// All functions must be target-low function-like ops in |module|. The pipeline
// runs function-wise in the order provided so callers can prepare exactly the
// functions they will packetize without mutating unrelated source functions.
iree_status_t loom_low_prepare_functions_for_packetization(
    loom_module_t* module, loom_op_t* const* low_func_ops,
    iree_host_size_t low_func_count,
    const loom_low_preparation_options_t* options,
    iree_arena_block_pool_t* block_pool);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_PREPARATION_H_
