// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source module selection for source-to-low lowering.
//
// This is cold compilation setup: it resolves a func-owned target record
// through symbol facts, checks that the low lowering policy supports the
// resulting target contract, and returns the concrete inputs needed by the core
// lowerer. The selected target comes from each func contract, never from a
// target record pointing back at a source func.

#ifndef LOOM_CODEGEN_LOW_SOURCE_SELECTION_H_
#define LOOM_CODEGEN_LOW_SOURCE_SELECTION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/lower.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_source_selection_options_t {
  // Target lowering policies linked into the caller.
  const loom_low_lower_policy_registry_t* policy_registry;

  // Structured diagnostic emitter for invalid target contracts encountered
  // while selecting source functions.
  iree_diagnostic_emitter_t diagnostic_emitter;

  // User-facing lowering kind used in diagnostics.
  iree_string_view_t lowering_kind;
} loom_low_source_selection_options_t;

typedef struct loom_low_source_selection_t {
  // Source func selected for lowering.
  loom_func_like_t func;

  // Dense func facts backing |func|.
  const loom_func_symbol_facts_t* func_facts;

  // Module-local target record symbol referenced by |func|.
  loom_symbol_ref_t target_ref;

  // Storage for the effective target bundle selected by |func|.
  loom_target_bundle_storage_t target_bundle_storage;

  // Effective target bundle selected by |func|.
  const loom_target_bundle_t* target_bundle;

  // Lowering policy selected by |target_bundle|.
  const loom_low_lower_policy_t* policy;
} loom_low_source_selection_t;

typedef struct loom_low_source_selection_list_t {
  // Source funcs selected for lowering.
  loom_low_source_selection_t* values;

  // Number of source func selections in |values|.
  iree_host_size_t count;
} loom_low_source_selection_list_t;

// Selects all source funcs compatible with the injected target-low registries.
//
// Fact payloads and the returned selection array are allocated from |arena| and
// remain valid for the arena lifetime. A module with no compatible funcs
// succeeds with an empty list so module passes can be no-ops.
iree_status_t loom_low_select_source_funcs(
    const loom_module_t* module,
    const loom_low_source_selection_options_t* options,
    iree_arena_allocator_t* arena,
    loom_low_source_selection_list_t* out_selection_list);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SOURCE_SELECTION_H_
