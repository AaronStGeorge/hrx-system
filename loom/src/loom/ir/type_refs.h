// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SSA references embedded in type values: structural traversal, remapping,
// equality, and hashing against the module that owns referenced identities.

#ifndef LOOM_IR_TYPE_REFS_H_
#define LOOM_IR_TYPE_REFS_H_

#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// One-way SSA value map used when comparing types across forwarding
// boundaries.
enum loom_type_value_remap_flag_bits_e {
  // source_values is a contiguous slice of block arguments, operation results,
  // or declaration arguments. Declaration arguments are indexed by their sole
  // ordinary operand use, which is their definition site. These are separate
  // index domains even when a declaration also owns results. Membership and
  // absence have bounded lookup independent of the span length.
  LOOM_TYPE_VALUE_REMAP_FLAG_SOURCE_DEFINITION_SLICE = 1u << 0,
};
typedef uint16_t loom_type_value_remap_flags_t;

typedef struct loom_type_value_remap_t {
  // Values appearing in the source-side type.
  const loom_value_id_t* source_values;
  // Values that the corresponding source values forward to.
  const loom_value_id_t* target_values;
  // Number of source/target value pairs.
  uint16_t count;
  // Structural promises enabling bounded source lookup.
  loom_type_value_remap_flags_t flags;
  // Additional discontiguous value pairs, or NULL when this is the last span.
  const struct loom_type_value_remap_t* next;
} loom_type_value_remap_t;

static_assert(sizeof(loom_type_value_remap_t) == (IREE_PTR_SIZE == 8 ? 32 : 16),
              "value remap spans must remain compact");

// Resolves one SSA value through |remap|. Values absent from the map preserve
// their identity. Source-definition slices use definition ordinals for bounded
// lookup; unflagged spans retain their declared first-match ordering.
loom_value_id_t loom_type_remap_value(const loom_module_t* module,
                                      const loom_type_value_remap_t* remap,
                                      loom_value_id_t value_id);

// Returns true if |source_type| equals |target_type| after applying |remap| to
// SSA value references embedded in |source_type|.
//
// This is used for region forwarding checks where a parent result type may
// reference a sibling result (`view<4xf32, %layout>`) while each region yields
// an equivalent branch-local value (`view<4xf32, %branch_layout>`). The helper
// is allocation-free and does not intern remapped types. The module and remap
// spans are trusted compiler-owned state and must be valid for the duration of
// the query.
bool loom_type_equal_after_value_remap(const loom_module_t* module,
                                       loom_type_t source_type,
                                       loom_type_t target_type,
                                       const loom_type_value_remap_t* remap);

// Returns a content-based hash after applying |remap| to embedded SSA value
// references. |module| owns type-valued parameter attributes reached from
// |type|. A NULL |remap| hashes the type without replacing value references.
//
// This hash is paired with loom_type_equal_after_value_remap(): when that
// comparison succeeds for source and target, hashing source with the same
// remap equals hashing target with a NULL remap. Its numeric result need not
// equal loom_type_hash(), even when |remap| is NULL.
uint32_t loom_type_hash_after_value_remap(const loom_module_t* module,
                                          loom_type_t type,
                                          const loom_type_value_remap_t* remap);

// Callback invoked for each SSA value reference embedded in a type.
typedef iree_status_t (*loom_type_value_ref_callback_t)(
    loom_value_id_t value_id, void* user_data);

// Walks SSA value references embedded in |type|.
//
// This includes dynamic shape and group dimensions, SSA
// encoding/layout attachments, and references in nested function, dialect, or
// register value types. References are reported in structural order and are
// not deduplicated: a type that mentions the same value twice emits two
// callbacks.
iree_status_t loom_type_walk_value_refs(const loom_module_t* module,
                                        loom_type_t type,
                                        loom_type_value_ref_callback_t callback,
                                        void* user_data);

// Returns true if |type| embeds a reference to |value_id|.
bool loom_type_references_value(const loom_module_t* module, loom_type_t type,
                                loom_value_id_t value_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_IR_TYPE_REFS_H_
