// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Pass manager: pipeline composition and execution.
//
// A pipeline is a flat sequence of passes. Each pass is either a module
// pass (runs once on the whole module) or a function pass (runs once per
// function-like symbol with a body). The pass manager iterates the
// pipeline, creating a fresh arena per pass invocation from a shared
// block pool.
//
// Pipeline strings for loom-opt:
//   --pass-pipeline='canonicalize,cse,dce'
//   --pass-pipeline='inline,canonicalize{max-iterations=20},cse'
//
// Pass lifecycle:
//   1. Arena created from shared block pool.
//   2. Pass create() called (if non-NULL) with option string.
//   3. Pass run() called once (module pass) or per-function.
//   4. Pass destroy() called (if non-NULL).
//   5. Arena deinitialized (all scratch memory freed).
//
// Reproducers: when a pass fails and DUMP_REPRODUCER is set, the pass
// manager writes the pre-pass IR and the remaining pipeline to a temp
// file, printing the path in the error status.

#ifndef LOOM_TRANSFORMS_PASS_H_
#define LOOM_TRANSFORMS_PASS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_module_t loom_module_t;
typedef struct loom_pass_t loom_pass_t;

//===----------------------------------------------------------------------===//
// Pass info
//===----------------------------------------------------------------------===//

typedef enum loom_pass_kind_e {
  LOOM_PASS_MODULE = 0,
  LOOM_PASS_FUNCTION = 1,
  LOOM_PASS_COUNT_,
} loom_pass_kind_t;

// Describes one named option a pass accepts.
typedef struct loom_pass_option_def_t {
  iree_string_view_t name;
  iree_string_view_t description;
} loom_pass_option_def_t;

// Describes one statistic a pass reports.
typedef struct loom_pass_statistic_def_t {
  iree_string_view_t name;
  iree_string_view_t description;
} loom_pass_statistic_def_t;

// Static metadata for a pass. One per pass kind, shared across
// all invocations. Lives in .rodata.
typedef struct loom_pass_info_t {
  iree_string_view_t name;
  iree_string_view_t description;
  loom_pass_kind_t kind;
  const loom_pass_option_def_t* option_defs;
  uint16_t option_count;
  const loom_pass_statistic_def_t* statistic_defs;
  uint16_t statistic_count;
} loom_pass_info_t;

//===----------------------------------------------------------------------===//
// Pass instance
//===----------------------------------------------------------------------===//

typedef iree_status_t (*loom_module_pass_fn_t)(loom_pass_t* pass,
                                               loom_module_t* module);
typedef iree_status_t (*loom_function_pass_fn_t)(loom_pass_t* pass,
                                                 loom_module_t* module,
                                                 loom_func_like_t function);
typedef iree_status_t (*loom_pass_create_fn_t)(loom_pass_t* pass,
                                               iree_string_view_t options);
typedef void (*loom_pass_destroy_fn_t)(loom_pass_t* pass);

// A pass instance. Created by the pass manager for each pipeline entry,
// destroyed after execution. The arena provides scratch memory that is
// freed automatically when the pass completes.
struct loom_pass_t {
  const loom_pass_info_t* info;
  union {
    loom_module_pass_fn_t module_run;
    loom_function_pass_fn_t function_run;
  };
  // Per-invocation scratch arena. All transient allocations (hash tables,
  // worklists, temporary IR) go here. Freed in one shot after the pass.
  iree_arena_allocator_t* arena;
  // Per-invocation statistics counters. Allocated from the arena, indexed
  // by the statistic_defs order in pass_info. Accumulated by the pass
  // manager after each invocation.
  int64_t* statistics;
  // Per-instance state from create(). The pass owns this memory (typically
  // allocated from the arena).
  void* state;
};

// Increments a statistic counter by |delta|.
static inline void loom_pass_statistic_add(loom_pass_t* pass,
                                           uint16_t statistic_index,
                                           int64_t delta) {
  pass->statistics[statistic_index] += delta;
}

//===----------------------------------------------------------------------===//
// Pass manager
//===----------------------------------------------------------------------===//

// One entry in the pipeline.
typedef struct loom_pipeline_entry_t {
  const loom_pass_info_t* info;
  union {
    loom_module_pass_fn_t module_run;
    loom_function_pass_fn_t function_run;
  };
  loom_pass_create_fn_t create;
  loom_pass_destroy_fn_t destroy;
  iree_string_view_t options;
} loom_pipeline_entry_t;

enum loom_pass_manager_flag_bits_e {
  LOOM_PASS_MANAGER_VERIFY_AFTER_EACH = 1u << 0,
  LOOM_PASS_MANAGER_DUMP_REPRODUCER = 1u << 1,
  LOOM_PASS_MANAGER_PRINT_BEFORE = 1u << 2,
  LOOM_PASS_MANAGER_PRINT_AFTER = 1u << 3,
  LOOM_PASS_MANAGER_TIMING = 1u << 4,
};
typedef uint32_t loom_pass_manager_flags_t;

typedef struct loom_pass_manager_t {
  loom_pipeline_entry_t* entries;
  iree_host_size_t count;
  iree_host_size_t capacity;
  iree_arena_block_pool_t* block_pool;
  loom_pass_manager_flags_t flags;
  iree_allocator_t allocator;
} loom_pass_manager_t;

// Initializes a pass manager. The block pool is shared across all pass
// arenas — blocks are borrowed during pass execution and returned after.
iree_status_t loom_pass_manager_initialize(iree_arena_block_pool_t* block_pool,
                                           loom_pass_manager_flags_t flags,
                                           iree_allocator_t allocator,
                                           loom_pass_manager_t* out_manager);

// Adds a module pass to the pipeline.
iree_status_t loom_pass_manager_add_module_pass(loom_pass_manager_t* manager,
                                                const loom_pass_info_t* info,
                                                loom_module_pass_fn_t run,
                                                loom_pass_create_fn_t create,
                                                loom_pass_destroy_fn_t destroy,
                                                iree_string_view_t options);

// Adds a function pass to the pipeline.
iree_status_t loom_pass_manager_add_function_pass(
    loom_pass_manager_t* manager, const loom_pass_info_t* info,
    loom_function_pass_fn_t run, loom_pass_create_fn_t create,
    loom_pass_destroy_fn_t destroy, iree_string_view_t options);

// Runs the full pipeline on a module. Each pass gets a fresh arena.
// Returns the first error encountered, with reproducer info if enabled.
iree_status_t loom_pass_manager_run(loom_pass_manager_t* manager,
                                    loom_module_t* module);

void loom_pass_manager_deinitialize(loom_pass_manager_t* manager);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_PASS_H_
