// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared module-to-target compilation helpers.
//
// These utilities own target-neutral mechanics that every concrete backend
// needs: generic and low verification, func-entry selection, func-owned
// target profile resolution, and diagnostic emission with source ranges.

#ifndef LOOM_TARGET_MODULE_COMPILER_H_
#define LOOM_TARGET_MODULE_COMPILER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/diagnostic.h"
#include "loom/ir/ir.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/target/types.h"
#include "loom/verify/verify.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_module_compile_options_t {
  // Optional func symbol to compile. Empty requires exactly one compatible
  // func with a target profile. A leading '@' is accepted for command-line
  // ergonomics.
  iree_string_view_t entry_symbol;
  // Diagnostic sink used for verification, lowering, scheduling, and
  // allocation diagnostics. A NULL callback still counts diagnostics.
  loom_diagnostic_sink_t diagnostic_sink;
  // Source resolver used to render carets for op-backed diagnostics.
  loom_source_resolver_t source_resolver;
  // Maximum diagnostics to emit before the active subsystem stops walking.
  // Zero lets callers use a backend-defined default.
  uint32_t max_errors;
} loom_target_module_compile_options_t;

typedef struct loom_target_module_compile_entry_t {
  // Source or low entry func selected for this compilation.
  loom_func_like_t func;
  // Borrowed selected func symbol name.
  iree_string_view_t func_name;
  // Module-local symbol reference for |func|.
  loom_symbol_ref_t func_ref;
  // Module-local target.profile symbol referenced by |func|.
  loom_symbol_ref_t target_ref;
  // Materialized target bundle selected by |func|. The export plan is the
  // func-owned effective export plan, not a profile-global backreference.
  loom_target_bundle_storage_t bundle_storage;
} loom_target_module_compile_entry_t;

typedef struct loom_target_module_compile_diagnostic_emitter_t {
  // Module containing op locations referenced by emitted diagnostics.
  const loom_module_t* module;
  // Source resolver for original source-backed locations.
  loom_source_resolver_t source_resolver;
  // Final diagnostic sink that owns rendering or capture policy.
  loom_diagnostic_sink_t diagnostic_sink;
  // Subsystem identity stored in materialized diagnostics.
  loom_emitter_t emitter;
} loom_target_module_compile_diagnostic_emitter_t;

typedef bool(IREE_API_PTR* loom_target_module_compile_entry_predicate_fn_t)(
    void* user_data, const loom_target_module_compile_entry_t* entry);

// Returns |options->max_errors| when present, otherwise |default_max_errors|.
uint32_t loom_target_module_compile_max_errors(
    const loom_target_module_compile_options_t* options,
    uint32_t default_max_errors);

// Returns the normalized entry symbol name without a leading '@'.
iree_string_view_t loom_target_module_compile_entry_symbol_name(
    const loom_target_module_compile_options_t* options);

// Initializes a diagnostic emitter for compilation subsystems.
void loom_target_module_compile_diagnostic_emitter_initialize(
    const loom_module_t* module,
    const loom_target_module_compile_options_t* options, loom_emitter_t emitter,
    loom_target_module_compile_diagnostic_emitter_t* out_emitter);

// Returns a callback wrapper for |emitter|.
iree_diagnostic_emitter_t loom_target_module_compile_emitter(
    loom_target_module_compile_diagnostic_emitter_t* emitter);

// Runs generic module verification and fails if any diagnostics were emitted.
iree_status_t loom_target_module_compile_verify_module(
    const loom_module_t* module,
    const loom_target_module_compile_options_t* options,
    uint32_t default_max_errors);

// Runs target-low semantic verification and fails if any diagnostics were
// emitted.
iree_status_t loom_target_module_compile_verify_low_module(
    const loom_module_t* module,
    const loom_target_low_descriptor_registry_t* low_registry,
    loom_low_descriptor_requirement_flags_t descriptor_requirements,
    loom_target_module_compile_diagnostic_emitter_t* diagnostic_emitter,
    uint32_t max_errors);

// Finds a module-local symbol by spelling without a leading '@'.
iree_status_t loom_target_module_compile_find_symbol_by_name(
    const loom_module_t* module, iree_string_view_t symbol_name,
    uint16_t* out_symbol_id);

// Selects either the named func entry or the only compatible func entry
// according to |predicate|.
iree_status_t loom_target_module_compile_select_entry(
    const loom_module_t* module,
    const loom_target_module_compile_options_t* options,
    const loom_target_low_descriptor_registry_t* low_registry,
    loom_target_module_compile_entry_predicate_fn_t predicate,
    void* predicate_user_data, iree_string_view_t entry_kind,
    iree_arena_allocator_t* arena,
    loom_target_module_compile_entry_t* out_entry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_MODULE_COMPILER_H_
