// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/passes/cse.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

enum {
  LOOM_CSE_STAT_EXPRESSIONS_ELIMINATED = 0,
};

static const loom_pass_statistic_def_t kCSEStatistics[] = {
    {IREE_SVL("expressions-eliminated"),
     IREE_SVL("Number of redundant expressions removed.")},
};

static const loom_pass_info_t loom_cse_pass_info_storage = {
    .name = IREE_SVL("cse"),
    .description = IREE_SVL("Eliminate common subexpressions."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_defs = kCSEStatistics,
    .statistic_count = 1,
};

const loom_pass_info_t* loom_cse_pass_info(void) {
  return &loom_cse_pass_info_storage;
}

//===----------------------------------------------------------------------===//
// Op hashing and equality
//===----------------------------------------------------------------------===//

// FNV-1a hash over a byte range, folded into a running hash.
static uint32_t loom_cse_hash_bytes(const void* data, iree_host_size_t length,
                                    uint32_t hash) {
  const uint8_t* bytes = (const uint8_t*)data;
  for (iree_host_size_t i = 0; i < length; ++i) {
    hash ^= bytes[i];
    hash *= 16777619u;
  }
  return hash;
}

// Computes a content-aware hash for an op based on its kind, operands,
// result types, attributes, and instance flags. Uses loom_attribute_hash
// for each attribute so pointer-valued attribute kinds (I64_ARRAY,
// PREDICATE_LIST, DICT) are hashed by content rather than pointer value.
// Result types are included because they are not always derivable from
// (kind, operands, attributes) — cast and conversion ops can produce
// different result types from the same operands.
static uint32_t loom_cse_hash_op(const loom_module_t* module,
                                 const loom_op_t* op) {
  uint32_t hash = 2166136261u;
  hash = loom_cse_hash_bytes(&op->kind, sizeof(op->kind), hash);
  const loom_value_id_t* operands = loom_op_operands((loom_op_t*)op);
  hash = loom_cse_hash_bytes(
      operands, (iree_host_size_t)op->operand_count * sizeof(loom_value_id_t),
      hash);
  const loom_value_id_t* results = loom_op_results((loom_op_t*)op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] != LOOM_VALUE_ID_INVALID) {
      loom_type_t type = loom_module_value_type(module, results[i]);
      hash = loom_cse_hash_bytes(&type, sizeof(type), hash);
    }
  }
  if (op->attribute_count > 0) {
    const loom_attribute_t* attrs = loom_op_attrs((loom_op_t*)op);
    for (uint8_t i = 0; i < op->attribute_count; ++i) {
      uint32_t attribute_hash = loom_attribute_hash(&attrs[i]);
      hash = loom_cse_hash_bytes(&attribute_hash, sizeof(attribute_hash), hash);
    }
  }
  hash = loom_cse_hash_bytes(&op->instance_flags, sizeof(op->instance_flags),
                             hash);
  return hash;
}

// Structural equality: two ops are CSE-equivalent if they have the
// same kind, operands (same value IDs in same order), result types
// (checked via the module's value table), attributes (structurally
// equal via loom_attribute_equal), instance flags, and no regions.
static bool loom_cse_ops_equal(const loom_module_t* module, const loom_op_t* a,
                               const loom_op_t* b) {
  if (a->kind != b->kind) return false;
  if (a->operand_count != b->operand_count) return false;
  if (a->result_count != b->result_count) return false;
  if (a->attribute_count != b->attribute_count) return false;
  if (a->instance_flags != b->instance_flags) return false;
  if (a->region_count > 0) return false;
  const loom_value_id_t* a_operands = loom_op_operands((loom_op_t*)a);
  const loom_value_id_t* b_operands = loom_op_operands((loom_op_t*)b);
  if (memcmp(a_operands, b_operands,
             (iree_host_size_t)a->operand_count * sizeof(loom_value_id_t)) !=
      0) {
    return false;
  }
  const loom_value_id_t* a_results = loom_op_results((loom_op_t*)a);
  const loom_value_id_t* b_results = loom_op_results((loom_op_t*)b);
  for (uint16_t i = 0; i < a->result_count; ++i) {
    if (a_results[i] != LOOM_VALUE_ID_INVALID &&
        b_results[i] != LOOM_VALUE_ID_INVALID) {
      loom_type_t a_type = loom_module_value_type(module, a_results[i]);
      loom_type_t b_type = loom_module_value_type(module, b_results[i]);
      if (!loom_type_equal(a_type, b_type)) return false;
    }
  }
  if (a->attribute_count > 0) {
    const loom_attribute_t* a_attrs = loom_op_attrs((loom_op_t*)a);
    const loom_attribute_t* b_attrs = loom_op_attrs((loom_op_t*)b);
    for (uint8_t i = 0; i < a->attribute_count; ++i) {
      if (!loom_attribute_equal(&a_attrs[i], &b_attrs[i])) return false;
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Hash table with tombstone support
//===----------------------------------------------------------------------===//
//
// Open-addressed hash table using linear probing with tombstone entries.
// Tombstones are left when write barriers invalidate non-PURE entries.
// Without tombstones, invalidation breaks probe chains: if entries A, B, C
// all hash to slot 5 and B is evicted (set to NULL), lookups for C would
// stop at the NULL in B's slot and fail. Tombstones keep the probe chain
// intact — find skips them, insert can reuse them.

// Sentinel pointer value for tombstone entries. Distinguishable from
// NULL (empty slot) and any valid op pointer (which will be at least
// 4-byte aligned and thus > 1).
#define LOOM_CSE_TOMBSTONE ((loom_op_t*)(uintptr_t)1)

typedef struct loom_cse_entry_t {
  loom_op_t* op;              // NULL = empty, TOMBSTONE = deleted, else = live.
  uint32_t hash;              // FNV-1a hash of the op's identity.
  loom_trait_flags_t traits;  // Trait flags at insert time.
} loom_cse_entry_t;

typedef struct loom_cse_table_t {
  loom_cse_entry_t* entries;
  // Slots containing live non-PURE entries, used for O(non-pure) write
  // barriers.
  iree_host_size_t* non_pure_slots;
  // Always a power of 2.
  iree_host_size_t capacity;
  // Live entries only (excludes tombstones).
  iree_host_size_t count;
  // Count of live non-PURE slot entries.
  iree_host_size_t non_pure_slot_count;
  // Capacity of non_pure_slots.
  iree_host_size_t non_pure_slot_capacity;
} loom_cse_table_t;

static iree_status_t loom_cse_table_initialize(iree_arena_allocator_t* arena,
                                               iree_host_size_t capacity,
                                               iree_host_size_t max_entry_count,
                                               loom_cse_table_t* table) {
  table->entries = NULL;
  table->non_pure_slots = NULL;
  table->capacity = capacity;
  table->count = 0;
  table->non_pure_slot_count = 0;
  table->non_pure_slot_capacity = max_entry_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, capacity, sizeof(loom_cse_entry_t), (void**)&table->entries));
  memset(table->entries, 0, capacity * sizeof(loom_cse_entry_t));
  if (max_entry_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, max_entry_count, sizeof(iree_host_size_t),
        (void**)&table->non_pure_slots));
  }
  return iree_ok_status();
}

// Finds an equivalent op in this table only (no chain walk).
// Skips tombstones during probing. Stops at empty (NULL) slots.
static loom_op_t* loom_cse_table_find(const loom_cse_table_t* table,
                                      const loom_module_t* module,
                                      const loom_op_t* op, uint32_t hash) {
  iree_host_size_t mask = table->capacity - 1;
  iree_host_size_t slot = hash & mask;
  while (true) {
    const loom_cse_entry_t* entry = &table->entries[slot];
    if (!entry->op) return NULL;  // Empty — end of probe chain.
    if (entry->op != LOOM_CSE_TOMBSTONE && entry->hash == hash &&
        loom_cse_ops_equal(module, entry->op, op)) {
      return entry->op;
    }
    slot = (slot + 1) & mask;
  }
}

// Inserts an op into this table. Caller must verify no duplicate exists.
// Reuses tombstone slots when available to reclaim space.
static void loom_cse_table_insert(loom_cse_table_t* table, loom_op_t* op,
                                  uint32_t hash, loom_trait_flags_t traits) {
  iree_host_size_t mask = table->capacity - 1;
  iree_host_size_t slot = hash & mask;
  while (table->entries[slot].op &&
         table->entries[slot].op != LOOM_CSE_TOMBSTONE) {
    slot = (slot + 1) & mask;
  }
  table->entries[slot] = (loom_cse_entry_t){
      .op = op,
      .hash = hash,
      .traits = traits,
  };
  ++table->count;
  if (!(traits & LOOM_TRAIT_PURE)) {
    IREE_ASSERT(table->non_pure_slot_count < table->non_pure_slot_capacity);
    table->non_pure_slots[table->non_pure_slot_count++] = slot;
  }
}

// Evicts all non-PURE entries from the table by replacing them with
// tombstones. PURE entries (no memory effects) survive write barriers
// because their results don't depend on mutable resource state.
// The non_pure_slots side list avoids scanning the full hash table capacity on
// every write in large blocks that alternate reads and writes.
static void loom_cse_table_invalidate_reads(loom_cse_table_t* table) {
  if (table->non_pure_slot_count == 0) return;
  for (iree_host_size_t i = 0; i < table->non_pure_slot_count; ++i) {
    iree_host_size_t slot = table->non_pure_slots[i];
    loom_op_t* op = table->entries[slot].op;
    if (op && op != LOOM_CSE_TOMBSTONE &&
        !(table->entries[slot].traits & LOOM_TRAIT_PURE)) {
      table->entries[slot].op = LOOM_CSE_TOMBSTONE;
      --table->count;
    }
  }
  table->non_pure_slot_count = 0;
}

//===----------------------------------------------------------------------===//
// Scope chain
//===----------------------------------------------------------------------===//
//
// CSE operates on a scope chain that mirrors the region nesting tree.
// Each scope owns a hash table of CSE candidates from its block.
// Lookup walks up the chain: an inner op can be replaced by an
// equivalent outer op if the outer scope is reachable (non-isolated).
// Write barriers propagate upward: a write inside a nested region
// invalidates read-only entries in all ancestor scopes.
//
// For isolated-from-above regions, the child scope's parent is NULL —
// the chain is broken and no outer values are considered.
//
// Multi-block regions use entry block dominance: block 0 of a region
// dominates all other blocks (structured control flow guarantee). The
// entry block's scope serves as the parent for all sibling blocks,
// making the entry block's CSE candidates visible to all successors.
//
// Arena allocation: all scopes and tables are allocated from a
// dedicated scope arena that is reset between top-level blocks.
// This bounds peak memory to the largest single subtree rather than
// the sum of all subtrees across the function.

typedef struct loom_cse_scope_t {
  loom_cse_table_t table;
  struct loom_cse_scope_t* parent;
} loom_cse_scope_t;

// Allocates a new scope with a hash table sized for |block|.
static iree_status_t loom_cse_scope_allocate(iree_arena_allocator_t* arena,
                                             loom_cse_scope_t* parent,
                                             const loom_block_t* block,
                                             loom_cse_scope_t** out_scope) {
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(loom_cse_scope_t), (void**)out_scope));
  (*out_scope)->parent = parent;
  iree_host_size_t capacity = iree_host_size_next_power_of_two(
      iree_max((iree_host_size_t)block->op_count * 2, 16));
  return loom_cse_table_initialize(arena, capacity, block->op_count,
                                   &(*out_scope)->table);
}

// Walks the scope chain looking for an equivalent op.
static loom_op_t* loom_cse_scope_lookup(const loom_cse_scope_t* scope,
                                        const loom_module_t* module,
                                        const loom_op_t* op, uint32_t hash) {
  for (const loom_cse_scope_t* s = scope; s; s = s->parent) {
    loom_op_t* found = loom_cse_table_find(&s->table, module, op, hash);
    if (found) return found;
  }
  return NULL;
}

// Propagates write barrier up the entire scope chain.
static void loom_cse_scope_invalidate_reads(loom_cse_scope_t* scope) {
  for (loom_cse_scope_t* s = scope; s; s = s->parent) {
    loom_cse_table_invalidate_reads(&s->table);
  }
}

//===----------------------------------------------------------------------===//
// DFS stack
//===----------------------------------------------------------------------===//
//
// Iterative DFS over the region tree using an explicit stack of frames.
// Each frame represents a block being processed with a cursor into its
// ordered op list. When an op has nested regions, child frames are pushed
// onto the stack and processed before the parent frame resumes.
//
// The stack is allocated from the pass arena (pass lifetime) and grows
// by doubling via iree_arena_grow_array. Old arrays are abandoned in
// the arena and reclaimed when the pass completes.

typedef struct loom_cse_frame_t {
  loom_block_t* block;
  loom_op_t* next_op;
  loom_cse_scope_t* scope;
} loom_cse_frame_t;

typedef struct loom_cse_stack_t {
  loom_cse_frame_t* frames;
  iree_host_size_t count;
  iree_host_size_t capacity;
} loom_cse_stack_t;

#define LOOM_CSE_INITIAL_STACK_CAPACITY 32

static iree_status_t loom_cse_stack_initialize(iree_arena_allocator_t* arena,
                                               loom_cse_stack_t* stack) {
  stack->count = 0;
  stack->capacity = LOOM_CSE_INITIAL_STACK_CAPACITY;
  return iree_arena_allocate_array(
      arena, stack->capacity, sizeof(loom_cse_frame_t), (void**)&stack->frames);
}

// Ensures the stack has room for |additional| frames. Grows by doubling
// if needed. Must be called before pushing any frames.
static iree_status_t loom_cse_stack_reserve(loom_cse_stack_t* stack,
                                            iree_arena_allocator_t* arena,
                                            iree_host_size_t additional) {
  iree_host_size_t required = stack->count + additional;
  if (required <= stack->capacity) return iree_ok_status();
  return iree_arena_grow_array(arena, stack->count, required,
                               sizeof(loom_cse_frame_t), &stack->capacity,
                               (void**)&stack->frames);
}

static void loom_cse_stack_push(loom_cse_stack_t* stack, loom_block_t* block,
                                loom_cse_scope_t* scope) {
  IREE_ASSERT(stack->count < stack->capacity);
  stack->frames[stack->count++] = (loom_cse_frame_t){
      .block = block,
      .next_op = block->first_op,
      .scope = scope,
  };
}

//===----------------------------------------------------------------------===//
// Region child frame pushing
//===----------------------------------------------------------------------===//

// Counts the total number of blocks across all regions of an op.
// Used to reserve stack space before pushing child frames.
static iree_host_size_t loom_cse_total_block_count(const loom_op_t* op) {
  iree_host_size_t total = 0;
  loom_region_t** regions = loom_op_regions((loom_op_t*)op);
  for (uint8_t r = 0; r < op->region_count; ++r) {
    if (regions[r]) total += regions[r]->block_count;
  }
  return total;
}

// Pushes child frames for all nested regions of an op.
//
// For single-block regions (the common case), pushes one frame with
// a child scope whose parent is |parent_scope|.
//
// For structured multi-block regions, uses entry block dominance: block 0
// dominates all other blocks in structured control flow. The entry
// block's scope serves as the parent for all sibling blocks:
//   - bb0 gets entry_scope (parent = parent_scope).
//   - bb1..bbN-1 each get a fresh scope (parent = entry_scope).
//   - Frames are pushed in reverse order so bb0 is on top (processed
//     first). When bb0 finishes, entry_scope has its CSE entries.
//     Subsequent blocks can see them through their parent pointer.
//
// CFG regions are intentionally more conservative: each block gets a fresh
// scope parented by |parent_scope|. CSE within CFG regions can become
// dominance-aware later, but shared CSE must not infer cross-block visibility
// from block table order.
static iree_status_t loom_cse_push_region_frames(
    loom_cse_stack_t* stack, iree_arena_allocator_t* pass_arena,
    iree_arena_allocator_t* scope_arena, const loom_op_t* op,
    loom_cse_scope_t* parent_scope) {
  loom_region_t** regions = loom_op_regions((loom_op_t*)op);
  // Push regions in reverse order so the first region's first block
  // is on top of the stack and processed first.
  for (int32_t r = (int32_t)op->region_count - 1; r >= 0; --r) {
    loom_region_t* region = regions[r];
    if (!region || region->block_count == 0) continue;
    if (region->block_count == 1) {
      // Common case: single-block region.
      loom_block_t* entry_block = loom_region_entry_block(region);
      loom_cse_scope_t* child_scope = NULL;
      IREE_RETURN_IF_ERROR(loom_cse_scope_allocate(scope_arena, parent_scope,
                                                   entry_block, &child_scope));
      loom_cse_stack_push(stack, entry_block, child_scope);
    } else if (iree_any_bit_set(region->flags, LOOM_REGION_INSTANCE_FLAG_CFG)) {
      for (int32_t b = (int32_t)region->block_count - 1; b >= 0; --b) {
        loom_block_t* block = loom_region_block(region, (uint16_t)b);
        loom_cse_scope_t* block_scope = NULL;
        IREE_RETURN_IF_ERROR(loom_cse_scope_allocate(scope_arena, parent_scope,
                                                     block, &block_scope));
        loom_cse_stack_push(stack, block, block_scope);
      }
    } else {
      // Multi-block region: entry block dominates all siblings.
      loom_block_t* entry_block = loom_region_entry_block(region);
      loom_cse_scope_t* entry_scope = NULL;
      IREE_RETURN_IF_ERROR(loom_cse_scope_allocate(scope_arena, parent_scope,
                                                   entry_block, &entry_scope));
      // Push non-entry blocks in reverse order (parent = entry_scope).
      for (int32_t b = (int32_t)region->block_count - 1; b >= 1; --b) {
        loom_block_t* block = loom_region_block(region, (uint16_t)b);
        loom_cse_scope_t* sibling_scope = NULL;
        IREE_RETURN_IF_ERROR(loom_cse_scope_allocate(scope_arena, entry_scope,
                                                     block, &sibling_scope));
        loom_cse_stack_push(stack, block, sibling_scope);
      }
      // Push entry block on top — processed first.
      loom_cse_stack_push(stack, entry_block, entry_scope);
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// CSE eligibility
//===----------------------------------------------------------------------===//

// Returns true if the op must not be CSE'd. An op prevents CSE if it
// has observable side effects (writes, unknown effects, non-determinism)
// or produces a unique identity per execution (allocations).
static inline bool loom_cse_prevents_cse(loom_trait_flags_t traits) {
  return (traits &
          (LOOM_TRAIT_WRITES_MEMORY | LOOM_TRAIT_UNKNOWN_EFFECTS |
           LOOM_TRAIT_NON_DETERMINISTIC | LOOM_TRAIT_UNIQUE_IDENTITY)) != 0;
}

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

iree_status_t loom_cse_run(loom_pass_t* pass, loom_module_t* module,
                           loom_func_like_t function) {
  loom_region_t* body = loom_func_like_body(function);
  if (!body) return iree_ok_status();

  // Scope arena: holds all scope structs and hash table arrays.
  // Reset between top-level blocks to bound peak memory to the
  // largest single subtree. Shares the pass arena's block pool.
  iree_arena_allocator_t scope_arena;
  iree_arena_initialize(pass->arena->block_pool, &scope_arena);

  // DFS stack: allocated from the pass arena (pass lifetime).
  loom_cse_stack_t stack;
  iree_status_t status = loom_cse_stack_initialize(pass->arena, &stack);

  // Process each top-level block independently.
  loom_block_t* top_block = NULL;
  loom_region_for_each_block(body, top_block) {
    if (!iree_status_is_ok(status)) break;
    iree_arena_reset(&scope_arena);

    loom_cse_scope_t* root_scope = NULL;
    status =
        loom_cse_scope_allocate(&scope_arena, NULL, top_block, &root_scope);
    if (!iree_status_is_ok(status)) break;

    stack.count = 0;
    loom_cse_stack_push(&stack, top_block, root_scope);

    while (iree_status_is_ok(status) && stack.count > 0) {
      loom_cse_frame_t* frame = &stack.frames[stack.count - 1];

      // Block done — pop frame.
      if (!frame->next_op) {
        --stack.count;
        continue;
      }

      loom_op_t* op = frame->next_op;
      frame->next_op = op->next_op;
      if (op->flags & LOOM_OP_FLAG_DEAD) continue;

      const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
      if (!vtable) {
        // Unknown op kind — conservatively treat as a write barrier.
        // The verifier enforces that all ops have vtables, so this
        // should not occur in valid IR. But if it does, treating the
        // op as transparent would silently allow CSE across writes.
        loom_cse_scope_invalidate_reads(frame->scope);
        continue;
      }

      // Compute effective traits once per op. For most ops this is just
      // vtable->traits; for ops with an effective_traits callback (e.g.,
      // func.call with a pure callee) it incorporates instance flags.
      loom_trait_flags_t traits = vtable->effective_traits
                                      ? vtable->effective_traits(op)
                                      : vtable->traits;

      // Write barrier: a write or unknown-effect op invalidates all
      // non-PURE entries up the scope chain. This must happen before
      // any other checks because result-less writes still invalidate.
      if (loom_traits_may_write(traits) ||
          loom_op_regions_have_write_effects(op)) {
        loom_cse_scope_invalidate_reads(frame->scope);
      }

      // Push child frames for nested regions.
      if (op->region_count > 0) {
        loom_cse_scope_t* parent_scope =
            loom_traits_is_isolated(traits) ? NULL : frame->scope;

        iree_host_size_t child_block_count = loom_cse_total_block_count(op);
        status = loom_cse_stack_reserve(&stack, pass->arena, child_block_count);
        if (!iree_status_is_ok(status)) break;
        // Re-fetch frame pointer — reserve may have reallocated the array.
        frame = &stack.frames[stack.count - 1];

        status = loom_cse_push_region_frames(&stack, pass->arena, &scope_arena,
                                             op, parent_scope);
        if (!iree_status_is_ok(status)) break;
        continue;  // Ops with regions are never CSE candidates.
      }

      // CSE candidate check: must have results, no regions (handled
      // above), no writes, no unknown effects, and be deterministic.
      if (op->result_count == 0) continue;
      if (loom_cse_prevents_cse(traits)) continue;

      // Look up the scope chain for an equivalent op.
      uint32_t hash = loom_cse_hash_op(module, op);
      loom_op_t* existing =
          loom_cse_scope_lookup(frame->scope, module, op, hash);
      if (existing) {
        // Replace all uses and erase.
        loom_value_id_t* op_results = loom_op_results(op);
        loom_value_id_t* existing_results = loom_op_results(existing);
        for (uint16_t r = 0; r < op->result_count; ++r) {
          if (op_results[r] != LOOM_VALUE_ID_INVALID &&
              existing_results[r] != LOOM_VALUE_ID_INVALID) {
            status = loom_value_replace_all_uses_with(module, op_results[r],
                                                      existing_results[r]);
            if (!iree_status_is_ok(status)) break;
          }
        }
        if (!iree_status_is_ok(status)) break;
        status = loom_op_erase(module, op);
        if (!iree_status_is_ok(status)) break;
        loom_pass_mark_changed(pass);
        if (pass->statistics) {
          loom_pass_statistic_add(pass, LOOM_CSE_STAT_EXPRESSIONS_ELIMINATED,
                                  1);
        }
      } else {
        loom_cse_table_insert(&frame->scope->table, op, hash, traits);
      }
    }
  }

  iree_arena_deinitialize(&scope_arena);
  return status;
}
