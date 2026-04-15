// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Module construction: create modules, define values, intern strings and
// types, create blocks and regions, insert ops into blocks.
//
// All allocations go through the module's arena (bump-pointer, O(1) bulk
// free on module destruction). Tables are pre-sized from capacity hints
// when available (bytecode reading, cloning) and grow by doubling when
// hints are absent (text parsing, test construction).
//
// Thread safety: modules are single-owner. No locks. Parallel compilation
// uses separate modules with separate arenas.

#ifndef LOOM_IR_MODULE_H_
#define LOOM_IR_MODULE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Capacity hints from bytecode header or source module (for cloning).
// NULL = use defaults (text parsing, test construction).
typedef struct loom_module_size_hints_t {
  iree_host_size_t value_count;
  iree_host_size_t string_count;
  iree_host_size_t type_count;
  iree_host_size_t symbol_count;
} loom_module_size_hints_t;

// Growth factor applied to size hints during module creation:
//   actual_capacity = (iree_host_size_t)(count * LOOM_MODULE_GROWTH_FACTOR)
// Provides headroom for passes that add values/ops during compilation.
// Tunable: profile real compilation pipelines to find the right value.
#define LOOM_MODULE_GROWTH_FACTOR 1.5f

// Creates a new empty module. The module owns an arena allocated from
// |block_pool|. All IR created through the module is arena-allocated
// and freed in O(1) when the module is destroyed. The module struct
// itself is allocated with |allocator|.
//
// |hints| may be NULL for default capacities (text parsing, tests).
// When non-NULL, tables are pre-allocated at hint * growth_factor.
iree_status_t loom_module_allocate(loom_context_t* context,
                                   iree_string_view_t name,
                                   iree_arena_block_pool_t* block_pool,
                                   const loom_module_size_hints_t* hints,
                                   iree_allocator_t allocator,
                                   loom_module_t** out_module);

// Destroys a module and frees all arena-allocated IR in O(1).
void loom_module_free(loom_module_t* module);

// Returns the module's body block. All top-level ops (function
// definitions, globals, executables) live in this block.
static inline loom_block_t* loom_module_block(loom_module_t* module) {
  return loom_region_entry_block(module->body);
}

// Returns a pointer to a value by ID.
static inline loom_value_t* loom_module_value(const loom_module_t* module,
                                              loom_value_id_t value_id) {
  IREE_ASSERT(value_id < module->values.count);
  return &module->values.entries[value_id];
}

// Returns the type of a value by ID.
static inline loom_type_t loom_module_value_type(const loom_module_t* module,
                                                 loom_value_id_t value_id) {
  IREE_ASSERT(value_id < module->values.count);
  return module->values.entries[value_id].type;
}

// Sets the type of a value by ID, updating SSA references carried by the old
// and new type payloads in the module's type-use side table.
iree_status_t loom_module_set_value_type(loom_module_t* module,
                                         loom_value_id_t value_id,
                                         loom_type_t type);

// Removes type-use records carried by |value_id|'s current type while leaving
// the stored type unchanged. Used when the carrier value is no longer live
// enough for its type to keep referenced values alive, such as op erasure.
void loom_module_drop_value_type_uses(loom_module_t* module,
                                      loom_value_id_t value_id);

// Rebuilds type-use records carried by one value's current type.
//
// References to values that do not exist yet are ignored; a later refresh after
// defining those values will add the records. Structural validation remains the
// verifier's job.
iree_status_t loom_module_refresh_value_type_uses(loom_module_t* module,
                                                  loom_value_id_t value_id);

// Returns the type of a block argument by index.
static inline loom_type_t loom_block_arg_type(const loom_module_t* module,
                                              const loom_block_t* block,
                                              uint16_t arg_index) {
  IREE_ASSERT(arg_index < block->arg_count);
  return loom_module_value_type(module, block->arg_ids[arg_index]);
}

// Defines a fresh SSA value in the module's value table with the given
// type. Returns the value ID (index into module->values.entries[]).
// The value's def pointer is unset — the builder fills it when
// finalizing the defining op (or loom_block_add_arg for block args).
iree_status_t loom_module_define_value(loom_module_t* module, loom_type_t type,
                                       loom_value_id_t* out_value_id);

// Rebuilds the dense type-use side table by walking all value types.
//
// Most construction paths maintain the table incrementally, but bulk readers
// and recovery paths can call this after setting value types directly.
iree_status_t loom_module_recompute_type_uses(loom_module_t* module);

// Returns true if |value_id| is referenced by any currently-active value type.
bool loom_module_value_has_type_uses(const loom_module_t* module,
                                     loom_value_id_t value_id);

// Replaces all SSA references to |old_id| embedded in value types with
// |new_id| and updates the module's type-use side table.
iree_status_t loom_module_replace_value_type_uses(loom_module_t* module,
                                                  loom_value_id_t old_id,
                                                  loom_value_id_t new_id);

// Interns a string in the module's string table. If an identical string
// already exists, returns its ID. Otherwise, arena-allocates a copy of
// the string data and appends a new entry.
iree_status_t loom_module_intern_string(loom_module_t* module,
                                        iree_string_view_t string,
                                        loom_string_id_t* out_string_id);

// Looks up a string in the module's intern table without inserting a new entry.
// Returns LOOM_STRING_ID_INVALID if |string| has not been interned.
loom_string_id_t loom_module_lookup_string(const loom_module_t* module,
                                           iree_string_view_t string);

// Builds a canonical DICT attribute in |module| from |entries|.
//
// The input entries may be in any order and may point to temporary storage.
// Each name_id must refer to a string interned in |module|. The builder
// recursively canonicalizes nested DICT values, sorts entries by key spelling,
// rejects duplicate keys, arena-copies the resulting immutable entry array,
// and stores the canonical wrapper in |out_attr|.
iree_status_t loom_module_make_canonical_attr_dict(
    loom_module_t* module, loom_named_attr_slice_t entries,
    loom_attribute_t* out_attr);

// Builds a fresh canonical DICT attribute from |base_entries| plus |updates|.
//
// |base_entries| is the existing dict content to patch, usually from a
// generated AttrDict accessor. |updates| may be in any order. Each update
// either inserts/replaces one key or removes it. Duplicate keys in |updates|
// are rejected so patch semantics stay order-independent. Replacement values
// are recursively canonicalized, unchanged base values are reused by value, and
// the resulting entry array is arena-owned by |module|.
iree_status_t loom_module_replace_canonical_attr_dict(
    loom_module_t* module, loom_named_attr_slice_t base_entries,
    loom_named_attr_update_slice_t updates, loom_attribute_t* out_attr);

// Verifies that |attr| is a canonical DICT attribute relative to |module|.
// Non-empty entries must be sorted by key spelling, duplicate-free, and
// recursively canonical. Returns INVALID_ARGUMENT for malformed dict attrs.
iree_status_t loom_module_verify_canonical_attr_dict(
    const loom_module_t* module, loom_attribute_t attr);

// Adds an encoding instance to the module's encoding table.
// Deduplicates by family name and parameter equality: if an identical encoding
// already exists, returns its 1-based ID without adding a new entry. Alias
// names are display-only, but they must remain unique across structurally
// distinct encodings so text output can emit unambiguous alias definitions.
// `name_id` and `alias_id` must be pre-interned in the module's string table.
// Parameters may point to temporary storage; they are recursively
// canonicalized into module-owned arena storage.
iree_status_t loom_module_add_encoding(loom_module_t* module,
                                       const loom_encoding_t* encoding,
                                       uint16_t* out_encoding_id);

// Returns the encoding at a 1-based index, or NULL if out of range.
static inline const loom_encoding_t* loom_module_encoding(
    const loom_module_t* module, uint16_t encoding_id) {
  if (encoding_id == 0 || encoding_id > module->encodings.count) return NULL;
  return &module->encodings.entries[encoding_id - 1];
}

// Returns the registered family vtable for |encoding_id|, or NULL if
// |encoding_id| is out of range, the module has no context, or the encoding
// family is not registered in that context.
const loom_encoding_vtable_t* loom_module_encoding_vtable(
    const loom_module_t* module, uint16_t encoding_id);

// Interns a type in the module's type table. If a structurally identical
// type already exists, returns the existing entry (by value). Otherwise,
// appends a new entry. Any heap-backed payload owned by |type| (overflow dims,
// function signatures, dialect params) is recursively copied into the module
// arena before storage, so callers may pass temporary or foreign-allocator
// payloads.
iree_status_t loom_module_intern_type(loom_module_t* module, loom_type_t type,
                                      loom_type_t* out_interned_type);

// Interns a function type directly from argument and result type arrays. If a
// structurally identical function type already exists, returns the canonical
// module-owned entry without cloning. Otherwise, recursively clones the
// signature payload into the module arena and appends a new interned type.
//
// |arg_types| and |result_types| may point to temporary parser scratch as long
// as they remain valid for the duration of this call.
iree_status_t loom_module_intern_function_type(loom_module_t* module,
                                               const loom_type_t* arg_types,
                                               uint16_t arg_count,
                                               const loom_type_t* result_types,
                                               uint16_t result_count,
                                               loom_type_t* out_interned_type);

// Adds a location entry to the module's location table and returns
// its ID. The entry is arena-allocated. Entry 0 is always
// LOOM_LOCATION_NONE (reserved automatically on first call).
iree_status_t loom_module_add_location(loom_module_t* module,
                                       loom_location_entry_t entry,
                                       loom_location_id_t* out_location_id);

// Attaches parser-captured field spans to a file location entry.
//
// |location_id| must reference an existing LOOM_LOCATION_FILE entry owned by
// |module|. |field_spans| are copied into module-owned arena storage, so the
// caller may pass parser scratch storage. Returns INVALID_ARGUMENT if the
// location is not a file range, the location ID is out of range, or spans were
// already attached to that location.
iree_status_t loom_module_attach_location_field_spans(
    loom_module_t* module, loom_location_id_t location_id,
    const loom_location_field_span_t* field_spans,
    iree_host_size_t field_span_count);

// Looks up a symbol by name. Returns the symbol index (0-based) or
// LOOM_SYMBOL_ID_INVALID if not found. O(n) linear scan — suitable
// for diagnostics and ad-hoc queries, not hot paths. For bulk
// lookups during parsing, use loom_symbol_map_t instead.
uint16_t loom_module_find_symbol(const loom_module_t* module,
                                 loom_string_id_t name_id);

// Adds a symbol to the module's symbol table. The name_id must be
// pre-interned via loom_module_intern_string. The symbol is zero-
// initialized except for the name_id. Returns the 0-based symbol
// index. Fails with RESOURCE_EXHAUSTED if the symbol count would
// exceed LOOM_SYMBOL_ID_INVALID (the symbol_id field width).
//
// Does NOT check for duplicates — callers should use
// loom_module_find_symbol first if deduplication is needed.
iree_status_t loom_module_add_symbol(loom_module_t* module,
                                     loom_string_id_t name_id,
                                     uint16_t* out_symbol_id);

// Allocates a block in the module's arena with initial op capacity.
iree_status_t loom_module_allocate_block(loom_module_t* module,
                                         loom_block_t** out_block);

// Allocates a region with |block_count| blocks in the module's arena.
// The entry block is embedded in the region object and additional blocks
// are arena-allocated individually. |block_count| may be 0; in that case the
// embedded entry block is initialized and becomes block 0 on the first append.
iree_status_t loom_module_allocate_region(loom_module_t* module,
                                          uint16_t block_count,
                                          loom_region_t** out_region);

// Appends a new block to |region| and returns it in |*out_block|. Existing
// block objects are never relocated; only the region's block pointer table may
// grow.
iree_status_t loom_region_append_block(loom_module_t* module,
                                       loom_region_t* region,
                                       loom_block_t** out_block);

// Adds a block argument. The value_id must already be defined in the
// module's value table (via loom_module_define_value). Sets
// LOOM_VALUE_FLAG_BLOCK_ARG on the value and records the block
// relationship.
iree_status_t loom_block_add_arg(loom_module_t* module, loom_block_t* block,
                                 loom_value_id_t value_id);

// Appends an op to the end of a block.
iree_status_t loom_block_append_op(loom_module_t* module, loom_block_t* block,
                                   loom_op_t* op);

// Inserts an op before |before_op| in |block|. |before_op| must be a live op
// in the block. Passing NULL appends.
iree_status_t loom_block_insert_before_op(loom_module_t* module,
                                          loom_block_t* block,
                                          loom_op_t* before_op, loom_op_t* op);

// Inserts an op at |index| in the block. This is a cold indexed helper for
// diagnostics and tests; hot mutation paths should carry an op pointer and use
// loom_block_insert_before_op.
// If index == block->op_count, equivalent to append.
iree_status_t loom_block_insert_op(loom_module_t* module, loom_block_t* block,
                                   iree_host_size_t index, loom_op_t* op);

// Records |op|'s direct effects in the transitive summaries on its containing
// region and ancestor regions. The op must be fully constructed: operands,
// results, attributes, and instance flags must already carry their final
// initial values.
void loom_module_record_op_effects(loom_module_t* module, loom_op_t* op);

// Removes |op|'s previously recorded direct effects and all nested op effects
// from the transitive summaries on their containing regions and ancestor
// regions.
void loom_module_drop_op_effects(loom_module_t* module, loom_op_t* op);

// Updates transitive summaries after an already-counted op changes the direct
// effective traits reported by its attributes or instance flags. Child region
// summaries are unchanged by this helper.
void loom_module_update_op_direct_effects(loom_op_t* op,
                                          loom_trait_flags_t old_traits,
                                          loom_trait_flags_t new_traits);

// Unlinks a live op from its parent block while preserving the op object for
// arena lifetime and diagnostics. The op's parent_block remains set so dead-op
// diagnostics can still report where it came from.
void loom_block_unlink_op(loom_module_t* module, loom_op_t* op);

// Finds the index of |op| in |block|'s op list by pointer comparison.
// Returns IREE_HOST_SIZE_MAX if not found. O(n) cold helper.
iree_host_size_t loom_block_find_op(const loom_block_t* block,
                                    const loom_op_t* op);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_IR_MODULE_H_
