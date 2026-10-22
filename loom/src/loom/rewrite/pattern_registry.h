// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable kind-rooted rewrite pattern registry.
//
// Compiler packages contribute static providers. A compiler invocation
// composes the providers it selected into one compact registry so per-op
// dispatch indexes directly by operation kind without scanning unrelated
// patterns or linking provider packages into the rewrite driver.

#ifndef LOOM_REWRITE_PATTERN_REGISTRY_H_
#define LOOM_REWRITE_PATTERN_REGISTRY_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_rewrite_pattern_t loom_rewrite_pattern_t;
typedef struct loom_value_fact_table_t loom_value_fact_table_t;
typedef struct loom_rewrite_action_t loom_rewrite_action_t;

// One prepared, meaning-preserving rewrite. Providers embed this header before
// their payload. Apply may replace only its own regionless root and must
// preserve result counts and types. It reads surviving operands from the live
// root, never from captured value IDs, and does not query facts. Other planned
// roots remain live until their own actions execute. Failure is limited to
// allocation or another infrastructure failure; matching has already made every
// decision.
struct loom_rewrite_action_t {
  // Next action in the temporary batch; assigned by the shared collector.
  loom_rewrite_action_t* next;
  // Live root to rewrite; assigned by the shared collector.
  loom_op_t* root;
  // Provider callback consuming this action and its trailing payload.
  iree_status_t (*apply)(const loom_rewrite_action_t* action,
                         loom_rewriter_t* rewriter);
};

// Matches against coherent read-only IR and facts. A match allocates its action
// and payload in |arena| and initializes the apply callback. A nonmatch returns
// NULL without allocating. No IR mutation occurs until all roots are prepared.
typedef iree_status_t (*loom_rewrite_pattern_prepare_fn_t)(
    const loom_rewrite_pattern_t* pattern, const loom_module_t* module,
    const loom_value_fact_table_t* facts, const loom_op_t* op,
    iree_arena_allocator_t* arena, loom_rewrite_action_t** out_action);

// Attempts one rewrite pattern. |context| is invocation-owned state shared by
// every pattern in the active registry. Pattern-specific immutable data is
// available through |pattern->user_data|. The callback sets |out_changed| only
// after it has rewritten |op|.
typedef iree_status_t (*loom_rewrite_pattern_fn_t)(
    const loom_rewrite_pattern_t* pattern, void* context, loom_op_t* op,
    loom_rewriter_t* rewriter, bool* out_changed);

// Provider-owned kind-rooted rewrite pattern.
struct loom_rewrite_pattern_t {
  // Operation kind this pattern can rewrite.
  loom_op_kind_t root_kind;

  // Callback that matches and optionally rewrites the root operation.
  loom_rewrite_pattern_fn_t match_and_rewrite;

  // Optional immutable pattern-specific data.
  const void* user_data;

  // Optional preparation of a root-local action from coherent facts. Selected
  // explicitly by the caller; ordinary registry application does not invoke it.
  loom_rewrite_pattern_prepare_fn_t prepare;
};

// Static pattern collection contributed by one compiler package.
typedef struct loom_rewrite_pattern_provider_t {
  // Stable provider name for composition diagnostics.
  iree_string_view_t name;

  // Provider-owned patterns in dispatch order.
  const loom_rewrite_pattern_t* patterns;

  // Number of patterns in |patterns|.
  uint16_t pattern_count;
} loom_rewrite_pattern_provider_t;

// Ordered borrowed provider pointers selected by a compiler composition.
typedef struct loom_rewrite_pattern_provider_list_t {
  // Number of provider pointers in |values|.
  iree_host_size_t count;

  // Borrowed provider pointer table.
  const loom_rewrite_pattern_provider_t* const* values;
} loom_rewrite_pattern_provider_list_t;

// Creates a borrowed rewrite pattern provider list.
static inline loom_rewrite_pattern_provider_list_t
loom_rewrite_pattern_provider_list_make(
    const loom_rewrite_pattern_provider_t* const* values,
    iree_host_size_t count) {
  return (loom_rewrite_pattern_provider_list_t){
      /*.count=*/count,
      /*.values=*/values,
  };
}

// Returns an empty rewrite pattern provider list.
static inline loom_rewrite_pattern_provider_list_t
loom_rewrite_pattern_provider_list_empty(void) {
  return (loom_rewrite_pattern_provider_list_t){0};
}

// Dense span metadata for one dialect-local operation kind.
typedef struct loom_rewrite_pattern_op_entry_t {
  // First pattern pointer in the registry's ordered pointer table.
  uint16_t pattern_start;

  // Number of patterns available for this operation kind.
  uint16_t pattern_count;
} loom_rewrite_pattern_op_entry_t;

// Dense dialect-local operation table.
typedef struct loom_rewrite_pattern_dialect_table_t {
  // Number of dialect-local operation entries.
  uint16_t op_count;

  // Dense entries indexed by loom_op_dialect_index.
  const loom_rewrite_pattern_op_entry_t* op_entries;
} loom_rewrite_pattern_dialect_table_t;

// Ordered pattern span returned for one operation kind.
typedef struct loom_rewrite_pattern_span_t {
  // Borrowed ordered pattern pointers.
  const loom_rewrite_pattern_t* const* patterns;

  // Number of pointers in |patterns|.
  uint16_t count;
} loom_rewrite_pattern_span_t;

// Immutable compact pattern registry composed for one compiler invocation.
typedef struct loom_rewrite_pattern_registry_t {
  // First dialect ID covered by |dialects|.
  uint8_t dialect_base_id;

  // Number of dense dialect slots.
  uint16_t dialect_count;

  // Dense dialect slots indexed by dialect ID minus |dialect_base_id|.
  const loom_rewrite_pattern_dialect_table_t* dialects;

  // Provider-owned pattern pointers grouped by root kind in dispatch order.
  const loom_rewrite_pattern_t* const* patterns;

  // Number of pointers in |patterns|.
  uint16_t pattern_count;
} loom_rewrite_pattern_registry_t;

// Owned storage for one composed rewrite pattern registry. Provider-owned
// descriptors, patterns, callbacks, names, and pattern data remain borrowed.
// One allocation backs every table and pointer array in |registry|.
typedef struct loom_rewrite_pattern_registry_storage_t {
  // Allocator used to release |allocation|.
  iree_allocator_t allocator;

  // Single allocation backing every pointer in |registry|.
  iree_byte_span_t allocation;

  // Immutable registry view over |allocation|.
  loom_rewrite_pattern_registry_t registry;
} loom_rewrite_pattern_registry_storage_t;

// Composes providers into a compact registry while preserving provider and
// pattern declaration order. An empty provider list produces an empty registry
// without allocating.
iree_status_t loom_rewrite_pattern_registry_storage_initialize(
    loom_rewrite_pattern_provider_list_t providers, iree_allocator_t allocator,
    loom_rewrite_pattern_registry_storage_t* out_storage);

// Releases registry storage. Borrowed provider data is never released.
void loom_rewrite_pattern_registry_storage_deinitialize(
    loom_rewrite_pattern_registry_storage_t* storage);

// Returns the immutable registry view owned by |storage|.
const loom_rewrite_pattern_registry_t*
loom_rewrite_pattern_registry_storage_registry(
    const loom_rewrite_pattern_registry_storage_t* storage);

// Looks up the ordered pattern span for |op_kind|.
static inline loom_rewrite_pattern_span_t
loom_rewrite_pattern_registry_lookup_kind(
    const loom_rewrite_pattern_registry_t* registry, loom_op_kind_t op_kind) {
  const uint8_t dialect_id = loom_op_dialect_id(op_kind);
  if (dialect_id < registry->dialect_base_id) {
    return (loom_rewrite_pattern_span_t){0};
  }
  const uint16_t dialect_index =
      (uint16_t)dialect_id - registry->dialect_base_id;
  if (dialect_index >= registry->dialect_count) {
    return (loom_rewrite_pattern_span_t){0};
  }
  const loom_rewrite_pattern_dialect_table_t* dialect =
      &registry->dialects[dialect_index];
  const uint8_t op_index = loom_op_dialect_index(op_kind);
  if (op_index >= dialect->op_count) {
    return (loom_rewrite_pattern_span_t){0};
  }
  const loom_rewrite_pattern_op_entry_t entry = dialect->op_entries[op_index];
  if (entry.pattern_count == 0) {
    return (loom_rewrite_pattern_span_t){0};
  }
  return (loom_rewrite_pattern_span_t){
      /*.patterns=*/&registry->patterns[entry.pattern_start],
      /*.count=*/entry.pattern_count,
  };
}

// Applies the patterns rooted at |op->kind| in registry order and stops after
// the first successful rewrite. Patterns for other operation kinds are never
// called.
iree_status_t loom_rewrite_pattern_registry_apply(
    const loom_rewrite_pattern_registry_t* registry, void* context,
    loom_op_t* op, loom_rewriter_t* rewriter, bool* out_changed);

// Prepares at most one action per root in |region|, taking the first matching
// preparation callback in registry order. The arena and returned linked list
// belong to the caller and must die after the complete batch is applied. No IR
// changes occur here, including on failure. Actions are returned in reverse
// preorder; their root-local contract makes application order independent.
iree_status_t loom_rewrite_pattern_registry_prepare_region(
    const loom_rewrite_pattern_registry_t* registry, loom_module_t* module,
    const loom_value_fact_table_t* facts, loom_region_t* region,
    iree_arena_allocator_t* arena, loom_rewrite_action_t** out_actions);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_REWRITE_PATTERN_REGISTRY_H_
