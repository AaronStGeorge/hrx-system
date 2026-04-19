// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Pass pipeline program interpreter.
//
// Executes the compact instruction program produced by program.h. The
// interpreter owns only transient execution state: per-invocation pass arenas,
// deterministic symbol snapshots, current anchor context, and diagnostic
// provenance. The compiled program remains immutable and reusable.

#ifndef LOOM_PASS_INTERPRETER_H_
#define LOOM_PASS_INTERPRETER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/pass/program.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_pass_interpreter_t loom_pass_interpreter_t;

// Supplies optional pass instance user data for one INVOKE instruction.
typedef iree_status_t (*loom_pass_interpreter_configure_fn_t)(
    void* user_data, const loom_pass_program_instruction_t* instruction,
    void** out_pass_user_data);

typedef struct loom_pass_interpreter_configure_callback_t {
  // Optional configure callback invoked before each pass instance is created.
  loom_pass_interpreter_configure_fn_t fn;
  // Opaque caller data passed to |fn|.
  void* user_data;
} loom_pass_interpreter_configure_callback_t;

typedef struct loom_pass_interpreter_options_t {
  // Shared block pool used for pass instance, scratch, and snapshot arenas.
  iree_arena_block_pool_t* block_pool;
  // Optional structured diagnostic emitter copied into every pass instance.
  iree_diagnostic_emitter_t diagnostic_emitter;
  // Optional user-data callback for pass instances.
  loom_pass_interpreter_configure_callback_t configure;
} loom_pass_interpreter_options_t;

// Executes a module-root compiled pass program.
iree_status_t loom_pass_interpreter_run_module(
    const loom_pass_program_t* program, loom_module_t* module,
    const loom_pass_interpreter_options_t* options);

// Executes a function-root compiled pass program on one function-like symbol.
iree_status_t loom_pass_interpreter_run_function(
    const loom_pass_program_t* program, loom_module_t* module,
    loom_func_like_t function, const loom_pass_interpreter_options_t* options);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_PASS_INTERPRETER_H_
