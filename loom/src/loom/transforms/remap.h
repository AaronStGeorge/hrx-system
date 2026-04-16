// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// IR value/type/attribute remapping for cloning and materialization.
//
// This is the source/target ownership boundary used by cloning, linking,
// inlining, and outlining helpers. It deliberately does not know about any
// callable policy or rewrite profitability: callers provide the SSA bindings
// and symbol policy, and this helper rewrites module-local IDs into the target
// module's tables.

#ifndef LOOM_TRANSFORMS_REMAP_H_
#define LOOM_TRANSFORMS_REMAP_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Remaps one source-module symbol reference to a target-module symbol
// reference.
//
// Cross-module materialization cannot infer symbol ownership safely. Callers
// that clone IR across modules must provide this callback when cloned
// attributes may reference symbols. Same-module remapping keeps symbol refs
// unchanged and does not call the callback.
typedef iree_status_t (*loom_ir_remap_symbol_fn_t)(
    void* user_data, const loom_module_t* source_module,
    loom_module_t* target_module, loom_symbol_ref_t source_ref,
    loom_symbol_ref_t* out_target_ref);

// Remap behavior knobs supplied at initialization.
typedef struct loom_ir_remap_options_t {
  // Allows unmapped SSA value references to remain unchanged when source and
  // target are the same module. Cross-module remaps still require an explicit
  // mapping for every SSA value reference.
  bool allow_unmapped_values;
  // Optional callback for cross-module symbol reference remapping.
  loom_ir_remap_symbol_fn_t remap_symbol;
  // Opaque caller data passed to remap_symbol.
  void* remap_symbol_user_data;
} loom_ir_remap_options_t;

// Dense SSA remap table for one source module and one target module.
typedef struct loom_ir_remap_t {
  // Module whose IDs appear in source IR payloads.
  const loom_module_t* source_module;
  // Module that will own remapped IR payloads.
  loom_module_t* target_module;
  // Scratch arena for remap tables and temporary recursive type arrays.
  iree_arena_allocator_t* arena;
  // Dense source value id -> target value id table.
  loom_value_id_t* value_map;
  // Number of source value IDs covered by value_map.
  iree_host_size_t value_map_count;
  // Source block pointer table for successor target remapping.
  const loom_block_t** block_map_sources;
  // Target block pointer table parallel to block_map_sources.
  loom_block_t** block_map_targets;
  // Number of source -> target block mappings installed.
  iree_host_size_t block_map_count;
  // Allocated capacity of block_map_sources and block_map_targets.
  iree_host_size_t block_map_capacity;
  // Allows same-module unmapped SSA refs to keep their original ID.
  bool allow_unmapped_values;
  // Optional callback for cross-module symbol reference remapping.
  loom_ir_remap_symbol_fn_t remap_symbol;
  // Opaque caller data passed to remap_symbol.
  void* remap_symbol_user_data;
  // Current recursive static-encoding remap depth. Internal recursion guard;
  // callers should treat this as owned by the remap helpers.
  uint16_t encoding_depth;
} loom_ir_remap_t;

// Initializes |out_remap| for source -> target materialization.
//
// The remap owns no teardown work; all allocations go through |arena|. The
// source module is treated as immutable for the lifetime of the remap so the
// dense value table can be indexed directly by source value ID.
iree_status_t loom_ir_remap_initialize(const loom_module_t* source_module,
                                       loom_module_t* target_module,
                                       iree_arena_allocator_t* arena,
                                       const loom_ir_remap_options_t* options,
                                       loom_ir_remap_t* out_remap);

// Adds or replaces one SSA value mapping.
iree_status_t loom_ir_remap_map_value(loom_ir_remap_t* remap,
                                      loom_value_id_t source_value,
                                      loom_value_id_t target_value);

// Adds or replaces a one-to-one SSA mapping for each source/target value pair.
//
// Use this before remapping result types for transforms that rebuild or fuse
// result lists. Dynamic type payloads can reference sibling/co-result values,
// so callers must install every old-result -> new-result mapping before asking
// loom_ir_remap_type to rewrite those result types.
iree_status_t loom_ir_remap_map_values(loom_ir_remap_t* remap,
                                       const loom_value_id_t* source_values,
                                       const loom_value_id_t* target_values,
                                       iree_host_size_t value_count);

// Looks up a mapped SSA value. Returns false when |source_value| is outside the
// source table or when no explicit mapping exists.
bool loom_ir_remap_try_lookup_value(const loom_ir_remap_t* remap,
                                    loom_value_id_t source_value,
                                    loom_value_id_t* out_target_value);

// Resolves one SSA value reference according to the remap's missing-value
// policy.
iree_status_t loom_ir_remap_resolve_value(const loom_ir_remap_t* remap,
                                          loom_value_id_t source_value,
                                          loom_value_id_t* out_target_value);

// Adds or replaces one block mapping used by successor edges during IR
// materialization.
iree_status_t loom_ir_remap_map_block(loom_ir_remap_t* remap,
                                      const loom_block_t* source_block,
                                      loom_block_t* target_block);

// Looks up a mapped block. Returns false when no explicit mapping exists.
bool loom_ir_remap_try_lookup_block(const loom_ir_remap_t* remap,
                                    const loom_block_t* source_block,
                                    loom_block_t** out_target_block);

// Resolves one successor block reference according to the remap's missing-block
// policy. Same-module remaps keep unmapped blocks unchanged; cross-module
// remaps require every successor target to have an explicit block mapping.
iree_status_t loom_ir_remap_resolve_block(const loom_ir_remap_t* remap,
                                          const loom_block_t* source_block,
                                          loom_block_t** out_target_block);

// Remaps one interned string ID into the target module. When |allow_invalid| is
// true, LOOM_STRING_ID_INVALID maps to itself.
iree_status_t loom_ir_remap_string_id(loom_ir_remap_t* remap,
                                      loom_string_id_t source_string_id,
                                      bool allow_invalid,
                                      loom_string_id_t* out_target_string_id);

// Remaps one source location ID into the target module.
iree_status_t loom_ir_remap_location_id(
    loom_ir_remap_t* remap, loom_location_id_t source_location_id,
    loom_location_id_t* out_target_location_id);

// Remaps all SSA and module-local references embedded in |source_type| and
// returns a target-module-owned equivalent type.
iree_status_t loom_ir_remap_type(loom_ir_remap_t* remap,
                                 loom_type_t source_type,
                                 loom_type_t* out_target_type);

// Remaps the type of each source value into a remap-arena-owned
// result type array.
//
// Callers that rebuild operation result lists should first map all old result
// values to their new values with loom_ir_remap_map_values. That makes
// co-result type references deterministic instead of relying on later RAUW to
// repair embedded dynamic dimensions or SSA encodings.
iree_status_t loom_ir_remap_value_types(loom_ir_remap_t* remap,
                                        const loom_value_id_t* source_values,
                                        iree_host_size_t value_count,
                                        loom_type_t** out_target_types);

// Remaps one static module encoding table ID into the target module.
iree_status_t loom_ir_remap_encoding_id(loom_ir_remap_t* remap,
                                        uint16_t source_encoding_id,
                                        uint16_t* out_target_encoding_id);

// Remaps one attribute payload into the target module.
iree_status_t loom_ir_remap_attribute(loom_ir_remap_t* remap,
                                      loom_attribute_t source_attr,
                                      loom_attribute_t* out_target_attr);

// Remaps a predicate list into target-module-owned storage.
iree_status_t loom_ir_remap_predicate_list(
    loom_ir_remap_t* remap, const loom_predicate_t* source_predicates,
    iree_host_size_t predicate_count, loom_predicate_t** out_target_predicates);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_REMAP_H_
