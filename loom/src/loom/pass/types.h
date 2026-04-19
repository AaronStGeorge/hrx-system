// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Core pass metadata and invocation contracts.
//
// Pass descriptors, pipeline programs, and interpreter execution all share this
// cold ABI. Concrete pass callbacks receive a loom_pass_t for their current
// invocation; ownership of the pass object, arenas, decoded options,
// statistics, diagnostics, and user data remains with the interpreter.

#ifndef LOOM_PASS_TYPES_H_
#define LOOM_PASS_TYPES_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_module_t loom_module_t;
typedef struct loom_pass_decoded_options_t loom_pass_decoded_options_t;
typedef struct loom_pass_t loom_pass_t;

typedef enum loom_pass_kind_e {
  LOOM_PASS_MODULE = 0,
  LOOM_PASS_FUNCTION = 1,
  LOOM_PASS_COUNT_,
} loom_pass_kind_t;

// Describes one named option a pass accepts.
typedef struct loom_pass_option_def_t {
  // Stable option key accepted by the pass.
  iree_string_view_t name;
  // Human-readable option description.
  iree_string_view_t description;
} loom_pass_option_def_t;

// Describes one statistic a pass reports.
typedef struct loom_pass_statistic_def_t {
  // Stable statistic key reported by the pass.
  iree_string_view_t name;
  // Human-readable statistic description.
  iree_string_view_t description;
} loom_pass_statistic_def_t;

// Static metadata for one pass kind, shared across all invocations.
typedef struct loom_pass_info_t {
  // Canonical pass name matching the descriptor key.
  iree_string_view_t name;
  // Human-readable pass description.
  iree_string_view_t description;
  // Anchor kind accepted by the pass callback.
  loom_pass_kind_t kind;
  // Descriptor-owned option definitions sorted by name.
  const loom_pass_option_def_t* option_defs;
  // Number of entries in option_defs.
  uint16_t option_count;
  // Descriptor-owned statistic definitions in report order.
  const loom_pass_statistic_def_t* statistic_defs;
  // Number of entries in statistic_defs.
  uint16_t statistic_count;
} loom_pass_info_t;

typedef iree_status_t (*loom_module_pass_fn_t)(loom_pass_t* pass,
                                               loom_module_t* module);
typedef iree_status_t (*loom_function_pass_fn_t)(loom_pass_t* pass,
                                                 loom_module_t* module,
                                                 loom_func_like_t function);
typedef iree_status_t (*loom_pass_create_fn_t)(loom_pass_t* pass,
                                               iree_string_view_t options);
typedef void (*loom_pass_destroy_fn_t)(loom_pass_t* pass);

// One pass callback invocation owned by the interpreter.
struct loom_pass_t {
  // Static pass metadata returned by the resolved descriptor.
  const loom_pass_info_t* info;
  // Callback selected for the resolved pass kind.
  union {
    // Module pass callback for LOOM_PASS_MODULE descriptors.
    loom_module_pass_fn_t module_run;
    // Function pass callback for LOOM_PASS_FUNCTION descriptors.
    loom_function_pass_fn_t function_run;
  };
  // Stable per-invocation arena for create()/destroy() state and statistics.
  iree_arena_allocator_t* instance_arena;
  // Scratch arena for the current run callback.
  iree_arena_allocator_t* arena;
  // Per-invocation statistic counters indexed by pass info statistic order.
  int64_t* statistics;
  // Per-invocation state produced by create() and consumed by destroy().
  void* state;
  // True when the pass callback explicitly records an IR or semantic change.
  bool changed;
  // Immutable descriptor-decoded options for pass.run-backed invocation.
  const loom_pass_decoded_options_t* decoded_options;
  // Optional structured diagnostic emitter for pass-specific failures.
  iree_diagnostic_emitter_t diagnostic_emitter;
  // Opaque caller data borrowed for this invocation.
  void* user_data;
};

// Increments a statistic counter by |delta|.
static inline void loom_pass_statistic_add(loom_pass_t* pass,
                                           uint16_t statistic_index,
                                           int64_t delta) {
  pass->statistics[statistic_index] += delta;
}

// Records that the current pass invocation changed IR or semantic module state.
static inline void loom_pass_mark_changed(loom_pass_t* pass) {
  pass->changed = true;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_PASS_TYPES_H_
