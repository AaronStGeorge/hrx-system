// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Materialized module linker for in-memory Loom IR.
//
// This is the inner link step for modules that have already been parsed or read
// into loom_module_t. Larger library workflows should discover reachable
// symbols through a module index, materialize only those modules/functions, and
// then call this linker over that provided set. Symbol references are resolved
// by module-local symbol name, so harness modules can reference corpus
// definitions through ordinary func.call/func.decl mechanics instead of
// tool-specific include preprocessing.

#ifndef LOOM_LINK_LINKER_H_
#define LOOM_LINK_LINKER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Options controlling one link operation.
typedef struct loom_link_options_t {
  // Name assigned to the linked output module.
  iree_string_view_t module_name;
  // Root symbol names to materialize. An empty list links every materialized
  // source symbol.
  iree_string_view_list_t root_symbols;
} loom_link_options_t;

// Links already-materialized source modules into a freshly allocated output
// module.
//
// All source modules must share the same finalized context. The returned module
// is owned by the caller and must be released with loom_module_free().
//
// Symbol policy:
// - Module-local symbols with the same name resolve to one output symbol.
// - A func.decl may be superseded by a concrete definition with the same name;
//   compatible declaration-owned signature, target, ABI, export, import, and
//   modifier contracts merge into the selected symbol-defining op.
// - Multiple concrete definitions for the same name are rejected.
// - Unresolved references stay unresolved for the verifier to diagnose.
//
// When options.root_symbols is non-empty, the linker treats source modules as
// libraries: it materializes only those roots and the module-local symbols
// reachable from their attributes/regions. A declaration that is superseded by
// a reachable definition provides the structural insertion point for that
// definition, so small harness modules can replace func.decl placeholders with
// library definitions without concatenating the entire library.
iree_status_t loom_link_materialized_modules(
    const loom_module_t* const* source_modules,
    iree_host_size_t source_module_count, const loom_link_options_t* options,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_module_t** out_module);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_LINK_LINKER_H_
