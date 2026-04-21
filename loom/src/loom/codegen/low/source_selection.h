// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source func selection for source-to-low lowering.
//
// This is cold compilation setup: it resolves a func-owned target profile
// through symbol facts, checks that the low lowering policy supports the
// resulting target contract, and returns the concrete inputs needed by the core
// lowerer. The selected target comes from the func contract, never from a
// target record pointing back at a source func.

#ifndef LOOM_CODEGEN_LOW_SOURCE_SELECTION_H_
#define LOOM_CODEGEN_LOW_SOURCE_SELECTION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/lower.h"
#include "loom/ir/ir.h"
#include "loom/ops/func_symbol_facts.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_source_selection_options_t {
  // Optional module-local source func symbol name, with or without '@'.
  // Empty selects the only compatible func in the module.
  iree_string_view_t func_symbol_name;

  // Low descriptor registry linked into the caller.
  const loom_low_descriptor_registry_t* descriptor_registry;

  // Target lowering policies linked into the caller.
  const loom_low_lower_policy_registry_t* policy_registry;

  // User-facing lowering kind used in status messages.
  iree_string_view_t lowering_kind;
} loom_low_source_selection_options_t;

typedef struct loom_low_source_selection_t {
  // Source func selected for lowering.
  loom_func_like_t func;

  // Dense func facts backing |func|.
  const loom_func_symbol_facts_t* func_facts;

  // Module-local target.profile symbol referenced by |func|.
  loom_symbol_ref_t target_ref;

  // Effective target bundle from |func_facts|.
  const loom_target_bundle_t* target_bundle;

  // Lowering policy selected by |target_bundle|.
  const loom_low_lower_policy_t* policy;
} loom_low_source_selection_t;

// Selects a source func and target-low lowering policy.
//
// Fact payloads in |out_selection| are allocated from |arena| and remain valid
// for the arena lifetime. Malformed user IR returns status so command-line
// tools can report the failure through their normal diagnostic wrapper.
iree_status_t loom_low_select_source_func(
    const loom_module_t* module,
    const loom_low_source_selection_options_t* options,
    iree_arena_allocator_t* arena, loom_low_source_selection_t* out_selection);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SOURCE_SELECTION_H_
