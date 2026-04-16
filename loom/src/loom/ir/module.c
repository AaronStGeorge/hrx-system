// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/module.h"

#include <string.h>

#include "loom/ir/context.h"

//===----------------------------------------------------------------------===//
// Hash function
//===----------------------------------------------------------------------===//

// Extends an FNV-1a hash with one byte span.
static uint32_t loom_hash_bytes_extend(uint32_t hash, const void* data,
                                       iree_host_size_t length) {
  const uint8_t* bytes = (const uint8_t*)data;
  for (iree_host_size_t i = 0; i < length; ++i) {
    hash ^= bytes[i];
    hash *= 16777619u;
  }
  return hash;
}

// FNV-1a hash over a byte array.
static uint32_t loom_hash_bytes(const void* data, iree_host_size_t length) {
  return loom_hash_bytes_extend(2166136261u, data, length);
}

// Extends an FNV-1a hash with one uint16_t.
static uint32_t loom_hash_u16_extend(uint32_t hash, uint16_t value) {
  return loom_hash_bytes_extend(hash, &value, sizeof(value));
}

// Extends an FNV-1a hash with one uint32_t.
static uint32_t loom_hash_u32_extend(uint32_t hash, uint32_t value) {
  return loom_hash_bytes_extend(hash, &value, sizeof(value));
}

static uint32_t loom_hash_string_view(iree_string_view_t string) {
  return loom_hash_bytes(string.data, string.size);
}

//===----------------------------------------------------------------------===//
// Intern table
//===----------------------------------------------------------------------===//

// Returns the hash table capacity needed for |entry_capacity| entries
// at a maximum 0.75 load factor, rounded up to a power of two.
static iree_host_size_t loom_intern_capacity_for_entries(
    iree_host_size_t entry_capacity) {
  return iree_host_size_next_power_of_two((entry_capacity * 4 + 2) / 3);
}

static iree_status_t loom_intern_table_allocate(iree_arena_allocator_t* arena,
                                                iree_host_size_t capacity,
                                                loom_intern_table_t* table) {
  uint32_t* hashes = NULL;
  uint32_t* indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, capacity, sizeof(uint32_t), (void**)&hashes));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, capacity, sizeof(uint32_t), (void**)&indices));

  iree_host_size_t byte_size = capacity * sizeof(uint32_t);
  memset(hashes, 0, byte_size);
  memset(indices, 0xFF, byte_size);
  table->count = 0;
  table->capacity = capacity;
  table->hashes = hashes;
  table->indices = indices;
  return iree_ok_status();
}

static iree_status_t loom_intern_table_grow(iree_arena_allocator_t* arena,
                                            loom_intern_table_t* table) {
  iree_host_size_t old_capacity = table->capacity;
  uint32_t* old_hashes = table->hashes;
  uint32_t* old_indices = table->indices;

  iree_host_size_t new_capacity = old_capacity * 2;
  IREE_RETURN_IF_ERROR(loom_intern_table_allocate(arena, new_capacity, table));

  // Reinsert all entries from the old table.
  iree_host_size_t mask = new_capacity - 1;
  for (iree_host_size_t i = 0; i < old_capacity; ++i) {
    if (old_indices[i] == UINT32_MAX) continue;
    uint32_t hash = old_hashes[i];
    iree_host_size_t slot = hash & mask;
    while (table->indices[slot] != UINT32_MAX) {
      slot = (slot + 1) & mask;
    }
    table->hashes[slot] = hash;
    table->indices[slot] = old_indices[i];
    ++table->count;
  }

  return iree_ok_status();
}

typedef bool (*loom_intern_equal_fn_t)(const void* context, uint32_t index);

static uint32_t loom_intern_table_lookup(const loom_intern_table_t* table,
                                         uint32_t hash,
                                         loom_intern_equal_fn_t equal_fn,
                                         const void* equal_context) {
  if (table->capacity == 0) return UINT32_MAX;

  iree_host_size_t mask = table->capacity - 1;
  iree_host_size_t slot = hash & mask;
  while (true) {
    uint32_t index = table->indices[slot];
    if (index == UINT32_MAX) return UINT32_MAX;
    if (table->hashes[slot] == hash && equal_fn(equal_context, index)) {
      return index;
    }
    slot = (slot + 1) & mask;
  }
}

// Looks up or inserts an entry in the intern table.
// |hash|: pre-computed hash of the entry.
// |index|: the entry table index to insert if not found.
// |out_existing_index|: set to the existing index if found, or |index|
//   if newly inserted.
// Returns true if an existing entry was found, false if newly inserted.
// The caller must check for equality using the entry table — the intern
// table only stores hashes and indices.
static iree_status_t loom_intern_table_find_or_insert(
    iree_arena_allocator_t* arena, loom_intern_table_t* table, uint32_t hash,
    uint32_t new_index, loom_intern_equal_fn_t equal_fn,
    const void* equal_context, uint32_t* out_index) {
  // Lazy initialization.
  if (table->capacity == 0) {
    iree_host_size_t initial_capacity = 32;
    IREE_RETURN_IF_ERROR(
        loom_intern_table_allocate(arena, initial_capacity, table));
  }

  // Check load factor and grow if needed.
  if (table->count * 4 >= table->capacity * 3) {
    IREE_RETURN_IF_ERROR(loom_intern_table_grow(arena, table));
  }

  iree_host_size_t mask = table->capacity - 1;
  iree_host_size_t slot = hash & mask;

  while (true) {
    if (table->indices[slot] == UINT32_MAX) {
      // Empty slot: insert.
      table->hashes[slot] = hash;
      table->indices[slot] = new_index;
      ++table->count;
      *out_index = new_index;
      return iree_ok_status();
    }
    if (table->hashes[slot] == hash &&
        equal_fn(equal_context, table->indices[slot])) {
      // Found existing entry.
      *out_index = table->indices[slot];
      return iree_ok_status();
    }
    slot = (slot + 1) & mask;
  }
}

//===----------------------------------------------------------------------===//
// Growable table helpers
//===----------------------------------------------------------------------===//

static iree_status_t loom_value_table_ensure_capacity(
    iree_arena_allocator_t* arena, loom_value_table_t* table) {
  if (table->count < table->capacity) return iree_ok_status();
  iree_host_size_t new_capacity =
      table->capacity > 0 ? table->capacity * 2 : 2048;
  loom_value_t* new_entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array_aligned(
      arena, new_capacity, sizeof(loom_value_t), 64, (void**)&new_entries));
  memset(new_entries, 0, new_capacity * sizeof(loom_value_t));
  if (table->count > 0) {
    memcpy(new_entries, table->entries, table->count * sizeof(loom_value_t));
  }
  table->entries = new_entries;
  table->capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t loom_string_table_ensure_capacity(
    iree_arena_allocator_t* arena, loom_string_table_t* table) {
  if (table->count < table->capacity) return iree_ok_status();
  iree_host_size_t new_capacity =
      table->capacity > 0 ? table->capacity * 2 : 512;
  iree_string_view_t* new_entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, new_capacity, sizeof(iree_string_view_t), (void**)&new_entries));
  memset(new_entries, 0, new_capacity * sizeof(iree_string_view_t));
  if (table->count > 0) {
    memcpy(new_entries, table->entries,
           table->count * sizeof(iree_string_view_t));
  }
  table->entries = new_entries;
  table->capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t loom_type_table_ensure_capacity(
    iree_arena_allocator_t* arena, loom_type_table_t* table) {
  if (table->count < table->capacity) return iree_ok_status();
  iree_host_size_t new_capacity =
      table->capacity > 0 ? table->capacity * 2 : 64;
  loom_type_t* new_entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, new_capacity, sizeof(loom_type_t), (void**)&new_entries));
  memset(new_entries, 0, new_capacity * sizeof(loom_type_t));
  if (table->count > 0) {
    memcpy(new_entries, table->entries, table->count * sizeof(loom_type_t));
  }
  table->entries = new_entries;
  table->capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t loom_encoding_table_ensure_capacity(
    iree_arena_allocator_t* arena, loom_encoding_table_t* table) {
  if (table->count < table->capacity) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      arena, table->count, /*minimum_capacity=*/8, sizeof(loom_encoding_t),
      &table->capacity, (void**)&table->entries));
  return iree_ok_status();
}

static iree_status_t loom_symbol_table_ensure_capacity(
    iree_arena_allocator_t* arena, loom_symbol_table_t* table) {
  if (table->count < table->capacity) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      arena, table->count, /*minimum_capacity=*/8, sizeof(loom_symbol_t),
      &table->capacity, (void**)&table->entries));
  return iree_ok_status();
}

static iree_status_t loom_location_table_ensure_capacity(
    iree_arena_allocator_t* arena, loom_location_table_t* table) {
  if (table->count < table->capacity) return iree_ok_status();
  IREE_RETURN_IF_ERROR(
      iree_arena_grow_array(arena, table->count, /*minimum_capacity=*/16,
                            sizeof(loom_location_entry_t), &table->capacity,
                            (void**)&table->entries));
  return iree_ok_status();
}

static void loom_type_use_heads_initialize(loom_value_type_use_heads_t* heads,
                                           iree_host_size_t count) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    heads[i].first_incoming_use_id = LOOM_TYPE_USE_ID_INVALID;
    heads[i].first_outgoing_use_id = LOOM_TYPE_USE_ID_INVALID;
  }
}

static iree_status_t loom_type_use_table_ensure_value_capacity(
    iree_arena_allocator_t* arena, loom_type_use_table_t* table,
    iree_host_size_t minimum_capacity) {
  if (table->value_capacity >= minimum_capacity) return iree_ok_status();
  iree_host_size_t old_capacity = table->value_capacity;
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      arena, old_capacity, minimum_capacity,
      sizeof(loom_value_type_use_heads_t), &table->value_capacity,
      (void**)&table->value_heads));
  loom_type_use_heads_initialize(table->value_heads + old_capacity,
                                 table->value_capacity - old_capacity);
  return iree_ok_status();
}

static iree_status_t loom_type_use_table_ensure_record_capacity(
    iree_arena_allocator_t* arena, loom_type_use_table_t* table,
    iree_host_size_t additional_record_count) {
  if (table->free_count >= additional_record_count) return iree_ok_status();
  iree_host_size_t new_records_needed =
      additional_record_count - table->free_count;
  iree_host_size_t minimum_capacity = table->record_count + new_records_needed;
  if (minimum_capacity <= table->record_capacity) return iree_ok_status();
  if (minimum_capacity >= LOOM_TYPE_USE_ID_INVALID) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "type-use table needs %" PRIhsz " records, max id %u", minimum_capacity,
        (unsigned)(LOOM_TYPE_USE_ID_INVALID - 1));
  }
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      arena, table->record_count, minimum_capacity, sizeof(loom_type_use_t),
      &table->record_capacity, (void**)&table->records));
  return iree_ok_status();
}

static void loom_type_use_table_reset(loom_type_use_table_t* table) {
  loom_type_use_heads_initialize(table->value_heads, table->value_capacity);
  table->record_count = 0;
  table->active_count = 0;
  table->free_count = 0;
  table->first_free_use_id = LOOM_TYPE_USE_ID_INVALID;
}

static iree_status_t loom_module_initialize_block(loom_module_t* module,
                                                  loom_block_t* block) {
  (void)module;
  memset(block, 0, sizeof(*block));
  block->label_id = LOOM_STRING_ID_INVALID;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Region effect summaries
//===----------------------------------------------------------------------===//

static loom_trait_flags_t loom_module_effective_traits(
    const loom_module_t* module, const loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable) return LOOM_TRAIT_UNKNOWN_EFFECTS;
  return vtable->effective_traits ? vtable->effective_traits(op)
                                  : vtable->traits;
}

static void loom_region_adjust_effect_count(uint32_t* count, int32_t delta) {
  if (delta < 0) {
    uint32_t decrement = (uint32_t)(-delta);
    IREE_ASSERT(*count >= decrement);
    *count -= decrement;
  } else {
    *count += (uint32_t)delta;
  }
}

static void loom_region_adjust_effect_counts(loom_region_t* region,
                                             int32_t read_delta,
                                             int32_t write_delta) {
  if (read_delta == 0 && write_delta == 0) return;
  if (read_delta != 0) {
    loom_region_adjust_effect_count(&region->read_effect_count, read_delta);
  }
  if (write_delta != 0) {
    loom_region_adjust_effect_count(&region->write_effect_count, write_delta);
  }
}

static void loom_module_adjust_op_ancestor_effect_counts(loom_op_t* op,
                                                         int32_t read_delta,
                                                         int32_t write_delta) {
  if (read_delta == 0 && write_delta == 0) return;
  loom_region_t* region =
      op->parent_block ? op->parent_block->parent_region : NULL;
  loom_op_t* parent_op = op->parent_op;
  while (region) {
    loom_region_adjust_effect_counts(region, read_delta, write_delta);
    if (!parent_op) break;
    region =
        parent_op->parent_block ? parent_op->parent_block->parent_region : NULL;
    parent_op = parent_op->parent_op;
  }
}

static void loom_module_adjust_op_direct_effect_counts(
    loom_op_t* op, loom_trait_flags_t traits, int32_t direction) {
  int32_t read_delta = loom_traits_may_read(traits) ? direction : 0;
  int32_t write_delta = loom_traits_may_write(traits) ? direction : 0;
  loom_module_adjust_op_ancestor_effect_counts(op, read_delta, write_delta);
}

void loom_module_record_op_effects(loom_module_t* module, loom_op_t* op) {
  if (!op || iree_any_bit_set(
                 op->flags, LOOM_OP_FLAG_DEAD | LOOM_OP_FLAG_EFFECTS_COUNTED)) {
    return;
  }
  loom_module_adjust_op_direct_effect_counts(
      op, loom_module_effective_traits(module, op), +1);
  op->flags |= LOOM_OP_FLAG_EFFECTS_COUNTED;
}

void loom_module_drop_op_effects(loom_module_t* module, loom_op_t* op) {
  if (!op) return;
  if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_EFFECTS_COUNTED)) {
    loom_module_adjust_op_direct_effect_counts(
        op, loom_module_effective_traits(module, op), -1);
    op->flags &= ~LOOM_OP_FLAG_EFFECTS_COUNTED;
  }
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    loom_region_t* region = regions[i];
    if (!region) continue;
    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      loom_op_t* child_op = NULL;
      loom_block_for_each_op(block, child_op) {
        loom_module_drop_op_effects(module, child_op);
      }
    }
  }
}

void loom_module_update_op_direct_effects(loom_op_t* op,
                                          loom_trait_flags_t old_traits,
                                          loom_trait_flags_t new_traits) {
  if (!op || !iree_all_bits_set(op->flags, LOOM_OP_FLAG_EFFECTS_COUNTED)) {
    return;
  }
  int32_t old_read = loom_traits_may_read(old_traits) ? 1 : 0;
  int32_t old_write = loom_traits_may_write(old_traits) ? 1 : 0;
  int32_t new_read = loom_traits_may_read(new_traits) ? 1 : 0;
  int32_t new_write = loom_traits_may_write(new_traits) ? 1 : 0;
  loom_module_adjust_op_ancestor_effect_counts(op, new_read - old_read,
                                               new_write - old_write);
}

static iree_status_t loom_region_blocks_ensure_capacity(loom_module_t* module,
                                                        loom_region_t* region) {
  if (region->block_count < region->block_capacity) return iree_ok_status();
  if (region->block_capacity == UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "region block count exceeds UINT16_MAX");
  }

  iree_host_size_t old_capacity = region->block_capacity;
  iree_host_size_t new_capacity =
      old_capacity > 0 ? old_capacity * 2 : (iree_host_size_t)4;
  if (new_capacity > UINT16_MAX) new_capacity = UINT16_MAX;

  loom_block_t** new_blocks = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&module->arena, new_capacity,
                                                 sizeof(loom_block_t*),
                                                 (void**)&new_blocks));
  memset(new_blocks, 0, new_capacity * sizeof(loom_block_t*));
  if (region->block_count > 0) {
    memcpy(new_blocks, region->blocks,
           (iree_host_size_t)region->block_count * sizeof(loom_block_t*));
  }

  region->blocks = new_blocks;
  region->block_capacity = (uint16_t)new_capacity;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Module
//===----------------------------------------------------------------------===//

static iree_status_t loom_module_initialize_tables(
    loom_module_t* module, const loom_module_size_hints_t* hints) {
  iree_arena_allocator_t* arena = &module->arena;

  iree_host_size_t value_capacity = 2048;
  iree_host_size_t string_capacity = 512;
  iree_host_size_t type_capacity = 64;
  iree_host_size_t symbol_capacity = 32;

  if (hints) {
    value_capacity =
        (iree_host_size_t)(hints->value_count * LOOM_MODULE_GROWTH_FACTOR);
    string_capacity =
        (iree_host_size_t)(hints->string_count * LOOM_MODULE_GROWTH_FACTOR);
    type_capacity =
        (iree_host_size_t)(hints->type_count * LOOM_MODULE_GROWTH_FACTOR);
    symbol_capacity =
        (iree_host_size_t)(hints->symbol_count * LOOM_MODULE_GROWTH_FACTOR);
    if (value_capacity < 16) value_capacity = 16;
    if (string_capacity < 8) string_capacity = 8;
    if (type_capacity < 8) type_capacity = 8;
    if (symbol_capacity < 4) symbol_capacity = 4;
  }

  // Values: 64-byte aligned.
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array_aligned(
      arena, value_capacity, sizeof(loom_value_t), 64,
      (void**)&module->values.entries));
  module->values.capacity = value_capacity;
  memset(module->values.entries, 0, value_capacity * sizeof(loom_value_t));

  // Type-use heads: dense side metadata indexed by value ID.
  IREE_RETURN_IF_ERROR(loom_type_use_table_ensure_value_capacity(
      arena, &module->type_uses, value_capacity));
  module->type_uses.first_free_use_id = LOOM_TYPE_USE_ID_INVALID;

  // Strings.
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, string_capacity, sizeof(iree_string_view_t),
      (void**)&module->strings.entries));
  module->strings.capacity = string_capacity;
  memset(module->strings.entries, 0,
         string_capacity * sizeof(iree_string_view_t));

  // Types.
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, type_capacity, sizeof(loom_type_t),
                                (void**)&module->types.entries));
  module->types.capacity = type_capacity;
  memset(module->types.entries, 0, type_capacity * sizeof(loom_type_t));

  // Symbols.
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, symbol_capacity, sizeof(loom_symbol_t),
                                (void**)&module->symbols.entries));
  module->symbols.capacity = symbol_capacity;
  memset(module->symbols.entries, 0, symbol_capacity * sizeof(loom_symbol_t));

  // Intern tables sized to match entry capacity.
  iree_host_size_t string_intern_capacity =
      loom_intern_capacity_for_entries(string_capacity);
  IREE_RETURN_IF_ERROR(loom_intern_table_allocate(arena, string_intern_capacity,
                                                  &module->string_intern));
  iree_host_size_t type_intern_capacity =
      loom_intern_capacity_for_entries(type_capacity);
  IREE_RETURN_IF_ERROR(loom_intern_table_allocate(arena, type_intern_capacity,
                                                  &module->type_intern));

  return iree_ok_status();
}

iree_status_t loom_module_allocate(loom_context_t* context,
                                   iree_string_view_t name,
                                   iree_arena_block_pool_t* block_pool,
                                   const loom_module_size_hints_t* hints,
                                   iree_allocator_t allocator,
                                   loom_module_t** out_module) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_module = NULL;

  loom_module_t* module = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(allocator, sizeof(loom_module_t), (void**)&module));
  memset(module, 0, sizeof(loom_module_t));

  module->context = context;
  module->allocator = allocator;
  iree_arena_initialize(block_pool, &module->arena);

  iree_status_t status = loom_module_initialize_tables(module, hints);
  if (iree_status_is_ok(status)) {
    status = loom_module_intern_string(module, name, &module->name_id);
  }
  if (iree_status_is_ok(status)) {
    status = loom_module_allocate_region(module, 1, &module->body);
  }
  if (iree_status_is_ok(status)) {
    *out_module = module;
  } else {
    loom_module_free(module);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void loom_module_free(loom_module_t* module) {
  if (!module) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t allocator = module->allocator;
  iree_arena_deinitialize(&module->arena);
  iree_allocator_free(allocator, module);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Encoding table
//===----------------------------------------------------------------------===//

iree_status_t loom_module_add_encoding(loom_module_t* module,
                                       const loom_encoding_t* encoding,
                                       uint16_t* out_encoding_id) {
  if (encoding->name_id == LOOM_STRING_ID_INVALID ||
      encoding->name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "encoding family string id %u is out of range "
                            "(module has %" PRIhsz " strings)",
                            encoding->name_id, module->strings.count);
  }
  if (encoding->alias_id != LOOM_STRING_ID_INVALID &&
      encoding->alias_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "encoding alias string id %u is out of range "
                            "(module has %" PRIhsz " strings)",
                            encoding->alias_id, module->strings.count);
  }
  if (encoding->attribute_count > 0 && !encoding->attributes) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-empty encoding parameter list has a NULL entry pointer");
  }

  iree_string_view_t encoding_name = module->strings.entries[encoding->name_id];

  loom_attribute_t canonical_attr_dict = {0};
  IREE_RETURN_IF_ERROR(loom_module_make_canonical_attr_dict(
      module,
      loom_make_named_attr_slice(encoding->attributes,
                                 encoding->attribute_count),
      &canonical_attr_dict));
  if (canonical_attr_dict.count > UINT8_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "encoding '%.*s' has %u parameters, max %u",
                            (int)encoding_name.size, encoding_name.data,
                            (unsigned)canonical_attr_dict.count,
                            (unsigned)UINT8_MAX);
  }

  loom_encoding_t canonical_encoding = {
      .name_id = encoding->name_id,
      .alias_id = encoding->alias_id,
      .attribute_count = canonical_attr_dict.count,
      .attributes = canonical_attr_dict.dict_entries,
  };

  const loom_encoding_vtable_t* vtable =
      loom_context_lookup_encoding_vtable(module->context, encoding_name);
  if (!vtable && module->context->encoding_vtables.count > 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown encoding family '%.*s'",
                            (int)encoding_name.size, encoding_name.data);
  }
  if (vtable && vtable->verify) {
    IREE_RETURN_IF_ERROR(vtable->verify(module, &canonical_encoding));
  }

  // Check for an existing duplicate and reject alias collisions against
  // structurally different encodings. Alias names are file-local shorthand and
  // must resolve unambiguously when the module is printed back to text.
  iree_host_size_t existing_index = IREE_HOST_SIZE_MAX;
  for (iree_host_size_t i = 0; i < module->encodings.count; ++i) {
    if (loom_encoding_equal(&module->encodings.entries[i],
                            &canonical_encoding)) {
      existing_index = i;
      continue;
    }
    if (canonical_encoding.alias_id != LOOM_STRING_ID_INVALID &&
        module->encodings.entries[i].alias_id == canonical_encoding.alias_id) {
      iree_string_view_t alias_name =
          module->strings.entries[canonical_encoding.alias_id];
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "encoding alias '%.*s' already names a different encoding",
          (int)alias_name.size, alias_name.data);
    }
  }

  if (existing_index != IREE_HOST_SIZE_MAX) {
    if (module->encodings.entries[existing_index].alias_id ==
            LOOM_STRING_ID_INVALID &&
        canonical_encoding.alias_id != LOOM_STRING_ID_INVALID) {
      module->encodings.entries[existing_index].alias_id =
          canonical_encoding.alias_id;
    }
    *out_encoding_id = (uint16_t)(existing_index + 1);
    return iree_ok_status();
  }

  // Encoding IDs are 1-based uint16_t. ID 0 means "no encoding" and
  // UINT16_MAX is the maximum representable ID, so we can store at
  // most UINT16_MAX entries (IDs 1 through UINT16_MAX).
  if (module->encodings.count >= UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "encoding table full (%" PRIhsz " entries, max %u)",
                            module->encodings.count, (unsigned)UINT16_MAX);
  }

  IREE_RETURN_IF_ERROR(
      loom_encoding_table_ensure_capacity(&module->arena, &module->encodings));

  loom_encoding_t* entry = &module->encodings.entries[module->encodings.count];
  memset(entry, 0, sizeof(*entry));
  entry->name_id = canonical_encoding.name_id;
  entry->alias_id = canonical_encoding.alias_id;
  entry->attribute_count = canonical_encoding.attribute_count;
  entry->attributes = canonical_encoding.attributes;

  *out_encoding_id = (uint16_t)(module->encodings.count + 1);
  ++module->encodings.count;
  return iree_ok_status();
}

const loom_encoding_vtable_t* loom_module_encoding_vtable(
    const loom_module_t* module, uint16_t encoding_id) {
  if (!module || !module->context) return NULL;
  const loom_encoding_t* encoding = loom_module_encoding(module, encoding_id);
  if (!encoding || encoding->name_id >= module->strings.count) return NULL;
  return loom_context_lookup_encoding_vtable(
      module->context, module->strings.entries[encoding->name_id]);
}

//===----------------------------------------------------------------------===//
// Symbol table
//===----------------------------------------------------------------------===//

uint16_t loom_module_find_symbol(const loom_module_t* module,
                                 loom_string_id_t name_id) {
  // Linear scan — suitable for diagnostics and ad-hoc queries. For
  // bulk lookups during parsing, use loom_symbol_map_t instead.
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    if (module->symbols.entries[i].name_id == name_id) {
      return (uint16_t)i;
    }
  }
  return LOOM_SYMBOL_ID_INVALID;
}

iree_status_t loom_module_add_symbol(loom_module_t* module,
                                     loom_string_id_t name_id,
                                     uint16_t* out_symbol_id) {
  // Symbol IDs are 0-based uint16_t. LOOM_SYMBOL_ID_INVALID is the null
  // sentinel (loom_symbol_ref_null), so the maximum valid ID is
  // LOOM_SYMBOL_ID_INVALID - 1, giving a hard cap of 65535 symbols per module.
  if (module->symbols.count >= LOOM_SYMBOL_ID_INVALID) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "symbol table full (%" PRIhsz " entries, max %u)",
                            module->symbols.count,
                            (unsigned)(LOOM_SYMBOL_ID_INVALID - 1));
  }

  IREE_RETURN_IF_ERROR(
      loom_symbol_table_ensure_capacity(&module->arena, &module->symbols));

  uint16_t symbol_id = (uint16_t)module->symbols.count;
  loom_symbol_t* symbol = &module->symbols.entries[module->symbols.count++];
  memset(symbol, 0, sizeof(*symbol));
  symbol->name_id = name_id;

  *out_symbol_id = symbol_id;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Location table
//===----------------------------------------------------------------------===//

iree_status_t loom_module_add_location(loom_module_t* module,
                                       loom_location_entry_t entry,
                                       loom_location_id_t* out_location_id) {
  // Lazily initialize with entry 0 = LOOM_LOCATION_NONE.
  if (module->locations.count == 0) {
    IREE_RETURN_IF_ERROR(loom_location_table_ensure_capacity(
        &module->arena, &module->locations));
    module->locations.entries[0] = (loom_location_entry_t){
        .kind = LOOM_LOCATION_NONE,
    };
    module->locations.count = 1;
  }

  // Location IDs are 32-bit and ID 0 is reserved for LOOM_LOCATION_UNKNOWN.
  // IDs 1 through UINT32_MAX are representable, so once count advances past
  // UINT32_MAX the next cast would wrap to 0 and forge the null sentinel.
  if (module->locations.count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "location table full (%" PRIhsz
                            " entries, max id %u)",
                            module->locations.count, (unsigned)UINT32_MAX);
  }

  IREE_RETURN_IF_ERROR(
      loom_location_table_ensure_capacity(&module->arena, &module->locations));

  loom_location_id_t id = (loom_location_id_t)module->locations.count;
  module->locations.entries[id] = entry;
  module->locations.count++;
  *out_location_id = id;
  return iree_ok_status();
}

iree_status_t loom_module_attach_location_field_spans(
    loom_module_t* module, loom_location_id_t location_id,
    const loom_location_field_span_t* field_spans,
    iree_host_size_t field_span_count) {
  if (location_id == LOOM_LOCATION_UNKNOWN ||
      (iree_host_size_t)location_id >= module->locations.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "location_id %u out of range for field span "
                            "attachment (module has %" PRIhsz " locations)",
                            location_id, module->locations.count);
  }
  loom_location_entry_t* entry = &module->locations.entries[location_id];
  if (entry->kind != LOOM_LOCATION_FILE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "cannot attach field spans to non-file location kind %d",
        (int)entry->kind);
  }
  if (entry->file.field_span_count > 0 || entry->file.field_spans) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "location %u already has field spans attached",
                            location_id);
  }
  if (field_span_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "location field span count %" PRIhsz
                            " exceeds maximum %u",
                            field_span_count, (unsigned)UINT16_MAX);
  }
  if (field_span_count > 0 && !field_spans) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "field_span_count > 0 requires non-NULL spans");
  }

  loom_location_field_span_t* copied_spans = NULL;
  if (field_span_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &module->arena, field_span_count, sizeof(*copied_spans),
        (void**)&copied_spans));
    memcpy(copied_spans, field_spans, field_span_count * sizeof(*copied_spans));
  }
  entry->file.field_span_count = (uint16_t)field_span_count;
  entry->file.field_spans = copied_spans;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Value definition
//===----------------------------------------------------------------------===//

typedef struct loom_type_use_prepare_t {
  loom_module_t* module;
  iree_host_size_t reference_count;
} loom_type_use_prepare_t;

static iree_status_t loom_type_use_prepare_callback(loom_value_id_t value_id,
                                                    void* user_data) {
  loom_type_use_prepare_t* prepare = (loom_type_use_prepare_t*)user_data;
  if (value_id >= prepare->module->values.count) return iree_ok_status();
  ++prepare->reference_count;
  return iree_ok_status();
}

static iree_status_t loom_type_use_prepare_for_type(
    loom_module_t* module, loom_type_t type,
    iree_host_size_t* out_reference_count) {
  loom_type_use_prepare_t prepare = {
      .module = module,
      .reference_count = 0,
  };
  IREE_RETURN_IF_ERROR(loom_type_walk_value_refs(
      type, loom_type_use_prepare_callback, &prepare));
  IREE_RETURN_IF_ERROR(loom_type_use_table_ensure_record_capacity(
      &module->arena, &module->type_uses, prepare.reference_count));
  *out_reference_count = prepare.reference_count;
  return iree_ok_status();
}

static loom_type_use_id_t loom_type_use_table_allocate_record(
    loom_type_use_table_t* table) {
  loom_type_use_id_t use_id = LOOM_TYPE_USE_ID_INVALID;
  if (table->first_free_use_id != LOOM_TYPE_USE_ID_INVALID) {
    use_id = table->first_free_use_id;
    loom_type_use_t* record = &table->records[use_id];
    table->first_free_use_id = record->next_incoming_use_id;
    --table->free_count;
  } else {
    use_id = (loom_type_use_id_t)table->record_count++;
  }
  ++table->active_count;
  return use_id;
}

static void loom_type_use_table_link_record(loom_type_use_table_t* table,
                                            loom_type_use_id_t use_id,
                                            loom_value_id_t referenced_value_id,
                                            loom_value_id_t user_value_id) {
  loom_type_use_t* record = &table->records[use_id];
  *record = (loom_type_use_t){
      .referenced_value_id = referenced_value_id,
      .user_value_id = user_value_id,
      .next_incoming_use_id =
          table->value_heads[referenced_value_id].first_incoming_use_id,
      .previous_incoming_use_id = LOOM_TYPE_USE_ID_INVALID,
      .next_outgoing_use_id =
          table->value_heads[user_value_id].first_outgoing_use_id,
      .previous_outgoing_use_id = LOOM_TYPE_USE_ID_INVALID,
  };
  if (record->next_incoming_use_id != LOOM_TYPE_USE_ID_INVALID) {
    table->records[record->next_incoming_use_id].previous_incoming_use_id =
        use_id;
  }
  if (record->next_outgoing_use_id != LOOM_TYPE_USE_ID_INVALID) {
    table->records[record->next_outgoing_use_id].previous_outgoing_use_id =
        use_id;
  }
  table->value_heads[referenced_value_id].first_incoming_use_id = use_id;
  table->value_heads[user_value_id].first_outgoing_use_id = use_id;
}

static void loom_type_use_table_unlink_record(loom_type_use_table_t* table,
                                              loom_type_use_id_t use_id) {
  loom_type_use_t* record = &table->records[use_id];
  if (record->previous_incoming_use_id != LOOM_TYPE_USE_ID_INVALID) {
    table->records[record->previous_incoming_use_id].next_incoming_use_id =
        record->next_incoming_use_id;
  } else {
    table->value_heads[record->referenced_value_id].first_incoming_use_id =
        record->next_incoming_use_id;
  }
  if (record->next_incoming_use_id != LOOM_TYPE_USE_ID_INVALID) {
    table->records[record->next_incoming_use_id].previous_incoming_use_id =
        record->previous_incoming_use_id;
  }
  if (record->previous_outgoing_use_id != LOOM_TYPE_USE_ID_INVALID) {
    table->records[record->previous_outgoing_use_id].next_outgoing_use_id =
        record->next_outgoing_use_id;
  } else {
    table->value_heads[record->user_value_id].first_outgoing_use_id =
        record->next_outgoing_use_id;
  }
  if (record->next_outgoing_use_id != LOOM_TYPE_USE_ID_INVALID) {
    table->records[record->next_outgoing_use_id].previous_outgoing_use_id =
        record->previous_outgoing_use_id;
  }
}

static void loom_type_use_table_release_record(loom_type_use_table_t* table,
                                               loom_type_use_id_t use_id) {
  loom_type_use_t* record = &table->records[use_id];
  *record = (loom_type_use_t){
      .referenced_value_id = LOOM_VALUE_ID_INVALID,
      .user_value_id = LOOM_VALUE_ID_INVALID,
      .next_incoming_use_id = table->first_free_use_id,
      .previous_incoming_use_id = LOOM_TYPE_USE_ID_INVALID,
      .next_outgoing_use_id = LOOM_TYPE_USE_ID_INVALID,
      .previous_outgoing_use_id = LOOM_TYPE_USE_ID_INVALID,
  };
  table->first_free_use_id = use_id;
  --table->active_count;
  ++table->free_count;
}

static void loom_type_use_table_remove_outgoing_for_value(
    loom_type_use_table_t* table, loom_value_id_t user_value_id) {
  if (user_value_id >= table->value_capacity) return;
  loom_type_use_id_t use_id =
      table->value_heads[user_value_id].first_outgoing_use_id;
  while (use_id != LOOM_TYPE_USE_ID_INVALID) {
    loom_type_use_id_t next_use_id =
        table->records[use_id].next_outgoing_use_id;
    loom_type_use_table_unlink_record(table, use_id);
    loom_type_use_table_release_record(table, use_id);
    use_id = next_use_id;
  }
}

typedef struct loom_type_use_add_t {
  loom_module_t* module;
  loom_type_use_table_t* table;
  loom_value_id_t user_value_id;
} loom_type_use_add_t;

static iree_status_t loom_type_use_add_callback(loom_value_id_t value_id,
                                                void* user_data) {
  loom_type_use_add_t* add = (loom_type_use_add_t*)user_data;
  if (value_id >= add->module->values.count) return iree_ok_status();
  loom_type_use_id_t use_id = loom_type_use_table_allocate_record(add->table);
  loom_type_use_table_link_record(add->table, use_id, value_id,
                                  add->user_value_id);
  return iree_ok_status();
}

static iree_status_t loom_type_use_table_add_outgoing_for_value(
    loom_module_t* module, loom_value_id_t user_value_id, loom_type_t type) {
  loom_type_use_add_t add = {
      .module = module,
      .table = &module->type_uses,
      .user_value_id = user_value_id,
  };
  return loom_type_walk_value_refs(type, loom_type_use_add_callback, &add);
}

iree_status_t loom_module_define_value(loom_module_t* module, loom_type_t type,
                                       loom_value_id_t* out_value_id) {
  // Value IDs are 32-bit and LOOM_VALUE_ID_INVALID is the null sentinel.
  // Fail before count reaches the sentinel value so the cast below cannot
  // produce an invalid ID from user-controlled input size.
  if (module->values.count >= LOOM_VALUE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "value table full (%" PRIhsz " entries, max id %u)",
                            module->values.count,
                            (unsigned)(LOOM_VALUE_ID_INVALID - 1));
  }

  IREE_RETURN_IF_ERROR(
      loom_value_table_ensure_capacity(&module->arena, &module->values));
  IREE_RETURN_IF_ERROR(loom_type_use_table_ensure_value_capacity(
      &module->arena, &module->type_uses, module->values.capacity));
  iree_host_size_t reference_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_type_use_prepare_for_type(module, type, &reference_count));

  loom_value_id_t id = (loom_value_id_t)module->values.count;
  loom_value_t* value = &module->values.entries[id];
  value->type = type;
  value->name_id = LOOM_STRING_ID_INVALID;
  value->def = loom_value_def_make_none();
  module->values.count++;

  if (reference_count > 0) {
    IREE_RETURN_IF_ERROR(
        loom_type_use_table_add_outgoing_for_value(module, id, type));
  }

  *out_value_id = id;
  return iree_ok_status();
}

iree_status_t loom_module_set_value_type(loom_module_t* module,
                                         loom_value_id_t value_id,
                                         loom_type_t type) {
  if (value_id >= module->values.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value %%%u out of range (module has %" PRIhsz
                            " values)",
                            (unsigned)value_id, module->values.count);
  }
  loom_value_t* value = &module->values.entries[value_id];
  if (loom_type_equal(value->type, type)) {
    return loom_module_refresh_value_type_uses(module, value_id);
  }

  loom_type_use_table_remove_outgoing_for_value(&module->type_uses, value_id);
  value->type = type;
  return loom_module_refresh_value_type_uses(module, value_id);
}

iree_status_t loom_module_refresh_value_type_uses(loom_module_t* module,
                                                  loom_value_id_t value_id) {
  if (value_id >= module->values.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value %%%u out of range (module has %" PRIhsz
                            " values)",
                            (unsigned)value_id, module->values.count);
  }
  iree_host_size_t reference_count = 0;
  loom_type_t type = module->values.entries[value_id].type;
  IREE_RETURN_IF_ERROR(
      loom_type_use_prepare_for_type(module, type, &reference_count));
  loom_type_use_table_remove_outgoing_for_value(&module->type_uses, value_id);
  if (reference_count > 0) {
    IREE_RETURN_IF_ERROR(
        loom_type_use_table_add_outgoing_for_value(module, value_id, type));
  }
  return iree_ok_status();
}

static bool loom_module_value_tracks_type_uses(const loom_value_t* value);

iree_status_t loom_module_recompute_type_uses(loom_module_t* module) {
  iree_host_size_t reference_count = 0;
  // Rebuild in two passes so allocation failures leave the current table
  // intact. This is a bulk recovery path after structural reconstruction, not
  // the per-edit hot path.
  for (iree_host_size_t i = 0; i < module->values.count; ++i) {
    if (!loom_module_value_tracks_type_uses(&module->values.entries[i])) {
      continue;
    }
    iree_host_size_t value_reference_count = 0;
    IREE_RETURN_IF_ERROR(loom_type_use_prepare_for_type(
        module, module->values.entries[i].type, &value_reference_count));
    reference_count += value_reference_count;
  }
  IREE_RETURN_IF_ERROR(loom_type_use_table_ensure_record_capacity(
      &module->arena, &module->type_uses, reference_count));

  loom_type_use_table_reset(&module->type_uses);
  for (iree_host_size_t i = 0; i < module->values.count; ++i) {
    if (!loom_module_value_tracks_type_uses(&module->values.entries[i])) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_type_use_table_add_outgoing_for_value(
        module, (loom_value_id_t)i, module->values.entries[i].type));
  }
  return iree_ok_status();
}

bool loom_module_value_has_type_uses(const loom_module_t* module,
                                     loom_value_id_t value_id) {
  return value_id < module->values.count &&
         value_id < module->type_uses.value_capacity &&
         module->type_uses.value_heads[value_id].first_incoming_use_id !=
             LOOM_TYPE_USE_ID_INVALID;
}

static bool loom_module_value_tracks_type_uses(const loom_value_t* value) {
  if (loom_value_is_block_arg(value)) return true;
  loom_op_t* def_op = loom_value_def_op(value);
  return def_op && !iree_any_bit_set(def_op->flags, LOOM_OP_FLAG_DEAD);
}

void loom_module_drop_value_type_uses(loom_module_t* module,
                                      loom_value_id_t value_id) {
  loom_type_use_table_remove_outgoing_for_value(&module->type_uses, value_id);
}

//===----------------------------------------------------------------------===//
// String interning
//===----------------------------------------------------------------------===//

typedef struct loom_string_equal_context_t {
  const loom_module_t* module;
  iree_string_view_t string;
} loom_string_equal_context_t;

static bool loom_string_equal_fn(const void* context, uint32_t index) {
  const loom_string_equal_context_t* ctx =
      (const loom_string_equal_context_t*)context;
  return iree_string_view_equal(ctx->module->strings.entries[index],
                                ctx->string);
}

iree_status_t loom_module_intern_string(loom_module_t* module,
                                        iree_string_view_t string,
                                        loom_string_id_t* out_string_id) {
  uint32_t hash = loom_hash_string_view(string);
  loom_string_equal_context_t equal_context = {module, string};

  uint32_t existing_index = loom_intern_table_lookup(
      &module->string_intern, hash, loom_string_equal_fn, &equal_context);
  if (existing_index != UINT32_MAX) {
    *out_string_id = (loom_string_id_t)existing_index;
    return iree_ok_status();
  }

  // String IDs are 32-bit and LOOM_STRING_ID_INVALID is the null sentinel.
  // Check only after the duplicate probe so an already-interned spelling
  // still resolves when the table is full.
  if (module->strings.count >= LOOM_STRING_ID_INVALID) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "string table full (%" PRIhsz " entries, max id %u)",
        module->strings.count, (unsigned)(LOOM_STRING_ID_INVALID - 1));
  }

  // Ensure the string table has capacity before inserting into the
  // intern hash table. This guarantees the entry slot exists if the
  // hash table insert succeeds.
  IREE_RETURN_IF_ERROR(
      loom_string_table_ensure_capacity(&module->arena, &module->strings));

  uint32_t new_index = (uint32_t)module->strings.count;
  uint32_t result_index = 0;
  IREE_RETURN_IF_ERROR(loom_intern_table_find_or_insert(
      &module->arena, &module->string_intern, hash, new_index,
      loom_string_equal_fn, &equal_context, &result_index));

  if (result_index != new_index) {
    *out_string_id = (loom_string_id_t)result_index;
    return iree_ok_status();
  }

  // New entry: arena-allocate a copy of the string data.
  char* copy = NULL;
  if (string.size > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(&module->arena, string.size, (void**)&copy));
    memcpy(copy, string.data, string.size);
  }
  module->strings.entries[new_index] = iree_make_string_view(copy, string.size);
  module->strings.count++;

  *out_string_id = (loom_string_id_t)new_index;
  return iree_ok_status();
}

loom_string_id_t loom_module_lookup_string(const loom_module_t* module,
                                           iree_string_view_t string) {
  uint32_t hash = loom_hash_string_view(string);
  loom_string_equal_context_t equal_context = {module, string};
  return (loom_string_id_t)loom_intern_table_lookup(
      &module->string_intern, hash, loom_string_equal_fn, &equal_context);
}

//===----------------------------------------------------------------------===//
// Canonical dictionary attributes
//===----------------------------------------------------------------------===//

static iree_status_t loom_module_resolve_attr_dict_key_name(
    const loom_module_t* module, loom_string_id_t name_id,
    iree_string_view_t* out_name) {
  if (name_id == LOOM_STRING_ID_INVALID || name_id >= module->strings.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dict attribute key string id %u is out of range (module has %" PRIhsz
        " strings)",
        name_id, module->strings.count);
  }
  *out_name = module->strings.entries[name_id];
  return iree_ok_status();
}

static iree_status_t loom_module_canonicalize_attr_dict_i64_array_value(
    loom_module_t* module, loom_attribute_t value,
    loom_attribute_t* out_value) {
  if (value.count == 0) {
    value.i64_array = NULL;
    *out_value = value;
    return iree_ok_status();
  }
  if (!value.i64_array) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-empty dict attribute i64 array value has a NULL payload");
  }

  int64_t* values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &module->arena, value.count, sizeof(int64_t), (void**)&values));
  memcpy(values, value.i64_array,
         (iree_host_size_t)value.count * sizeof(int64_t));
  value.i64_array = values;
  *out_value = value;
  return iree_ok_status();
}

static iree_status_t loom_module_canonicalize_attr_dict_predicate_list_value(
    loom_module_t* module, loom_attribute_t value,
    loom_attribute_t* out_value) {
  if (value.count == 0) {
    value.predicate_list = NULL;
    *out_value = value;
    return iree_ok_status();
  }
  if (!value.predicate_list) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-empty dict attribute predicate list has a NULL payload");
  }

  loom_predicate_t* predicates = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&module->arena, value.count,
                                                 sizeof(loom_predicate_t),
                                                 (void**)&predicates));
  memcpy(predicates, value.predicate_list,
         (iree_host_size_t)value.count * sizeof(loom_predicate_t));
  value.predicate_list = predicates;
  *out_value = value;
  return iree_ok_status();
}

static iree_status_t loom_module_canonicalize_non_dict_attr_value(
    loom_module_t* module, loom_attribute_t value,
    loom_attribute_t* out_value) {
  value.reserved_0 = 0;
  value.reserved_1 = 0;

  switch ((loom_attr_kind_t)value.kind) {
    case LOOM_ATTR_ABSENT:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "dict attribute value is absent");
    case LOOM_ATTR_I64:
    case LOOM_ATTR_F64:
    case LOOM_ATTR_BOOL:
    case LOOM_ATTR_ENUM:
    case LOOM_ATTR_SYMBOL:
      *out_value = value;
      return iree_ok_status();
    case LOOM_ATTR_STRING:
      if (value.string_id == LOOM_STRING_ID_INVALID ||
          value.string_id >= module->strings.count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "dict attribute string value id %u is out of "
                                "range (module has %" PRIhsz " strings)",
                                value.string_id, module->strings.count);
      }
      *out_value = value;
      return iree_ok_status();
    case LOOM_ATTR_TYPE:
      if (value.type_id == LOOM_TYPE_ID_INVALID ||
          value.type_id >= module->types.count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "dict attribute type value id %u is out of "
                                "range (module has %" PRIhsz " types)",
                                value.type_id, module->types.count);
      }
      *out_value = value;
      return iree_ok_status();
    case LOOM_ATTR_ENCODING:
      if (value.encoding_id == 0 ||
          value.encoding_id > module->encodings.count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "dict attribute encoding value id %u is out of range "
            "(module has %" PRIhsz " encodings)",
            (unsigned)value.encoding_id, module->encodings.count);
      }
      *out_value = value;
      return iree_ok_status();
    case LOOM_ATTR_I64_ARRAY:
      return loom_module_canonicalize_attr_dict_i64_array_value(module, value,
                                                                out_value);
    case LOOM_ATTR_PREDICATE_LIST:
      return loom_module_canonicalize_attr_dict_predicate_list_value(
          module, value, out_value);
    case LOOM_ATTR_DICT:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "nested dict attribute must be canonicalized by the entry walker");
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "dict attribute value has unknown kind %u",
                              (unsigned)value.kind);
  }
}

static iree_status_t loom_module_make_canonical_attr_dict_entries(
    loom_module_t* module, const loom_named_attr_t* entries,
    iree_host_size_t count, iree_host_size_t depth,
    loom_attribute_t* out_attr) {
  if (depth >= LOOM_ATTR_DICT_MAX_NESTING_DEPTH) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dict attribute nesting exceeds max depth %u",
                            (unsigned)LOOM_ATTR_DICT_MAX_NESTING_DEPTH);
  }
  if (count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "dict attribute has %" PRIhsz " entries, max %u",
                            count, (unsigned)UINT16_MAX);
  }
  if (count > 0 && !entries) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-empty dict attribute has a NULL entry pointer");
  }

  if (count == 0) {
    *out_attr = loom_make_canonical_attr_dict(NULL, 0);
    return iree_ok_status();
  }

  loom_named_attr_t* canonical_entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&module->arena, count,
                                                 sizeof(loom_named_attr_t),
                                                 (void**)&canonical_entries));

  iree_host_size_t canonical_count = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_string_view_t key_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_module_resolve_attr_dict_key_name(
        module, entries[i].name_id, &key_name));

    loom_named_attr_t entry = {
        .name_id = entries[i].name_id,
        .value = {0},
    };
    if (entries[i].value.kind == LOOM_ATTR_DICT) {
      IREE_RETURN_IF_ERROR(loom_module_make_canonical_attr_dict_entries(
          module, entries[i].value.dict_entries, entries[i].value.count,
          depth + 1, &entry.value));
    } else {
      IREE_RETURN_IF_ERROR(loom_module_canonicalize_non_dict_attr_value(
          module, entries[i].value, &entry.value));
    }

    iree_host_size_t insert_index = canonical_count;
    while (insert_index > 0) {
      iree_string_view_t previous_key_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_module_resolve_attr_dict_key_name(
          module, canonical_entries[insert_index - 1].name_id,
          &previous_key_name));

      int comparison = iree_string_view_compare(key_name, previous_key_name);
      if (comparison == 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate dict attribute key '%.*s'",
                                (int)key_name.size, key_name.data);
      }
      if (comparison > 0) break;

      canonical_entries[insert_index] = canonical_entries[insert_index - 1];
      --insert_index;
    }

    canonical_entries[insert_index] = entry;
    ++canonical_count;
  }

  *out_attr = loom_make_canonical_attr_dict(canonical_entries, canonical_count);
  return iree_ok_status();
}

iree_status_t loom_module_make_canonical_attr_dict(
    loom_module_t* module, loom_named_attr_slice_t entries,
    loom_attribute_t* out_attr) {
  return loom_module_make_canonical_attr_dict_entries(
      module, entries.entries, entries.count, 0, out_attr);
}

iree_status_t loom_module_replace_canonical_attr_dict(
    loom_module_t* module, loom_named_attr_slice_t base_entries,
    loom_named_attr_update_slice_t updates, loom_attribute_t* out_attr) {
  if (base_entries.count > 0 && !base_entries.entries) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-empty base dict attribute has a NULL entry pointer");
  }
  if (updates.count > 0 && !updates.updates) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-empty dict attribute update list has a NULL update pointer");
  }
  if (base_entries.count > UINT16_MAX || updates.count > UINT16_MAX ||
      base_entries.count + updates.count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "dict attribute replacement would exceed %u "
                            "entries (%" PRIhsz " base + %" PRIhsz " updates)",
                            (unsigned)UINT16_MAX, base_entries.count,
                            updates.count);
  }

  loom_attribute_t base_attr =
      loom_make_canonical_attr_dict(base_entries.entries, base_entries.count);
  IREE_RETURN_IF_ERROR(
      loom_module_verify_canonical_attr_dict(module, base_attr));

  for (iree_host_size_t i = 0; i < updates.count; ++i) {
    iree_string_view_t key_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_module_resolve_attr_dict_key_name(
        module, updates.updates[i].name_id, &key_name));
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (updates.updates[j].name_id == updates.updates[i].name_id) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate dict attribute update key '%.*s'",
                                (int)key_name.size, key_name.data);
      }
    }
  }

  if (base_entries.count == 0 && updates.count == 0) {
    *out_attr = loom_make_canonical_attr_dict(NULL, 0);
    return iree_ok_status();
  }

  iree_host_size_t max_count = base_entries.count + updates.count;
  loom_named_attr_t* merged_entries = NULL;
  if (max_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&module->arena, max_count,
                                                   sizeof(loom_named_attr_t),
                                                   (void**)&merged_entries));
  }
  iree_host_size_t merged_count = base_entries.count;
  if (base_entries.count > 0) {
    memcpy(merged_entries, base_entries.entries,
           base_entries.count * sizeof(loom_named_attr_t));
  }

  for (iree_host_size_t update_index = 0; update_index < updates.count;
       ++update_index) {
    const loom_named_attr_update_t* update = &updates.updates[update_index];
    iree_string_view_t update_key_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_module_resolve_attr_dict_key_name(
        module, update->name_id, &update_key_name));

    iree_host_size_t entry_index = 0;
    bool found_existing = false;
    while (entry_index < merged_count) {
      iree_string_view_t entry_key_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_module_resolve_attr_dict_key_name(
          module, merged_entries[entry_index].name_id, &entry_key_name));
      int comparison =
          iree_string_view_compare(entry_key_name, update_key_name);
      if (comparison == 0) {
        found_existing = true;
        break;
      }
      if (comparison > 0) break;
      ++entry_index;
    }

    if (update->remove) {
      if (!found_existing) continue;
      for (iree_host_size_t i = entry_index + 1; i < merged_count; ++i) {
        merged_entries[i - 1] = merged_entries[i];
      }
      --merged_count;
      continue;
    }

    loom_attribute_t canonical_value = {0};
    if (update->value.kind == LOOM_ATTR_DICT) {
      IREE_RETURN_IF_ERROR(loom_module_make_canonical_attr_dict_entries(
          module, update->value.dict_entries, update->value.count,
          /*depth=*/1, &canonical_value));
    } else {
      IREE_RETURN_IF_ERROR(loom_module_canonicalize_non_dict_attr_value(
          module, update->value, &canonical_value));
    }
    if (found_existing) {
      merged_entries[entry_index].value = canonical_value;
      merged_entries[entry_index].reserved = 0;
      continue;
    }

    for (iree_host_size_t i = merged_count; i > entry_index; --i) {
      merged_entries[i] = merged_entries[i - 1];
    }
    merged_entries[entry_index] = (loom_named_attr_t){
        .name_id = update->name_id,
        .reserved = 0,
        .value = canonical_value,
    };
    ++merged_count;
  }

  *out_attr = loom_make_canonical_attr_dict(merged_entries, merged_count);
  return iree_ok_status();
}

static iree_status_t loom_module_verify_canonical_attr_header(
    const loom_attribute_t* attr, iree_string_view_t owner_name) {
  if (attr->reserved_0 != 0 || attr->reserved_1 != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s has non-zero reserved bits",
                            (int)owner_name.size, owner_name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_module_verify_canonical_attr_dict_i64_array_value(
    const loom_attribute_t* value) {
  if (value->count > 0 && !value->i64_array) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-empty dict attribute i64 array value has a NULL payload");
  }
  if (value->count == 0 && value->i64_array != NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "empty dict attribute i64 array value must use a NULL payload");
  }
  return iree_ok_status();
}

static iree_status_t
loom_module_verify_canonical_attr_dict_predicate_list_value(
    const loom_attribute_t* value) {
  if (value->count > 0 && !value->predicate_list) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-empty dict attribute predicate list has a NULL payload");
  }
  if (value->count == 0 && value->predicate_list != NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "empty dict attribute predicate list must use a NULL payload");
  }
  return iree_ok_status();
}

static iree_status_t loom_module_verify_non_dict_attr_value(
    const loom_module_t* module, const loom_attribute_t* value) {
  IREE_RETURN_IF_ERROR(loom_module_verify_canonical_attr_header(
      value, IREE_SV("dict attribute value")));

  switch ((loom_attr_kind_t)value->kind) {
    case LOOM_ATTR_ABSENT:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "dict attribute value is absent");
    case LOOM_ATTR_I64:
    case LOOM_ATTR_F64:
    case LOOM_ATTR_BOOL:
    case LOOM_ATTR_ENUM:
    case LOOM_ATTR_SYMBOL:
      return iree_ok_status();
    case LOOM_ATTR_STRING:
      if (value->string_id == LOOM_STRING_ID_INVALID ||
          value->string_id >= module->strings.count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "dict attribute string value id %u is out of "
                                "range (module has %" PRIhsz " strings)",
                                value->string_id, module->strings.count);
      }
      return iree_ok_status();
    case LOOM_ATTR_TYPE:
      if (value->type_id == LOOM_TYPE_ID_INVALID ||
          value->type_id >= module->types.count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "dict attribute type value id %u is out of "
                                "range (module has %" PRIhsz " types)",
                                value->type_id, module->types.count);
      }
      return iree_ok_status();
    case LOOM_ATTR_ENCODING:
      if (value->encoding_id == 0 ||
          value->encoding_id > module->encodings.count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "dict attribute encoding value id %u is out of range "
            "(module has %" PRIhsz " encodings)",
            (unsigned)value->encoding_id, module->encodings.count);
      }
      return iree_ok_status();
    case LOOM_ATTR_I64_ARRAY:
      return loom_module_verify_canonical_attr_dict_i64_array_value(value);
    case LOOM_ATTR_PREDICATE_LIST:
      return loom_module_verify_canonical_attr_dict_predicate_list_value(value);
    case LOOM_ATTR_DICT:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "nested dict attribute must be verified by the entry walker");
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "dict attribute value has unknown kind %u",
                              (unsigned)value->kind);
  }
}

static iree_status_t loom_module_verify_canonical_attr_dict_entries(
    const loom_module_t* module, const loom_attribute_t* attr,
    iree_host_size_t depth) {
  if (attr->kind != LOOM_ATTR_DICT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected a DICT attribute, got kind %u",
                            (unsigned)attr->kind);
  }
  IREE_RETURN_IF_ERROR(loom_module_verify_canonical_attr_header(
      attr, IREE_SV("dict attribute")));
  if (depth >= LOOM_ATTR_DICT_MAX_NESTING_DEPTH) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dict attribute nesting exceeds max depth %u",
                            (unsigned)LOOM_ATTR_DICT_MAX_NESTING_DEPTH);
  }
  if (attr->count == 0) {
    if (attr->dict_entries != NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "empty dict attribute must use a NULL entry pointer");
    }
    return iree_ok_status();
  }
  if (!attr->dict_entries) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-empty dict attribute has a NULL entry pointer");
  }

  iree_string_view_t previous_key_name = iree_string_view_empty();
  for (uint16_t i = 0; i < attr->count; ++i) {
    const loom_named_attr_t* entry = &attr->dict_entries[i];
    if (entry->reserved != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dict attribute entry %u has non-zero reserved bits", i);
    }

    iree_string_view_t key_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_module_resolve_attr_dict_key_name(
        module, entry->name_id, &key_name));
    if (i > 0) {
      int comparison = iree_string_view_compare(key_name, previous_key_name);
      if (comparison == 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate dict attribute key '%.*s'",
                                (int)key_name.size, key_name.data);
      }
      if (comparison < 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "dict attribute key '%.*s' appears after '%.*s' out of canonical "
            "order",
            (int)key_name.size, key_name.data, (int)previous_key_name.size,
            previous_key_name.data);
      }
    }

    if (entry->value.kind == LOOM_ATTR_DICT) {
      IREE_RETURN_IF_ERROR(loom_module_verify_canonical_attr_dict_entries(
          module, &entry->value, depth + 1));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_module_verify_non_dict_attr_value(module, &entry->value));
    }
    previous_key_name = key_name;
  }

  return iree_ok_status();
}

iree_status_t loom_module_verify_canonical_attr_dict(
    const loom_module_t* module, loom_attribute_t attr) {
  return loom_module_verify_canonical_attr_dict_entries(module, &attr, 0);
}

//===----------------------------------------------------------------------===//
// Type interning
//===----------------------------------------------------------------------===//

typedef struct loom_type_equal_context_t {
  const loom_module_t* module;
  loom_type_t type;
} loom_type_equal_context_t;

typedef struct loom_function_type_equal_context_t {
  const loom_module_t* module;
  const loom_type_t* arg_types;
  const loom_type_t* result_types;
  uint16_t arg_count;
  uint16_t result_count;
} loom_function_type_equal_context_t;

typedef iree_status_t (*loom_module_type_clone_fn_t)(loom_module_t* module,
                                                     const void* clone_context,
                                                     loom_type_t* out_type);

// Compares one interned module type against a temporary by-value candidate.
static bool loom_type_equal_fn(const void* context, uint32_t index) {
  const loom_type_equal_context_t* ctx =
      (const loom_type_equal_context_t*)context;
  return loom_type_equal(ctx->module->types.entries[index], ctx->type);
}

// Compares one interned module type against temporary arg/result arrays for a
// first-class function signature that has not been packed into a FAM payload.
static bool loom_function_type_equal_fn(const void* context, uint32_t index) {
  const loom_function_type_equal_context_t* ctx =
      (const loom_function_type_equal_context_t*)context;
  loom_type_t type = ctx->module->types.entries[index];
  if (loom_type_kind(type) != LOOM_TYPE_FUNCTION) return false;

  const loom_func_type_data_t* data = loom_type_func_data(type);
  if (!data) return false;
  if (data->arg_count != ctx->arg_count ||
      data->result_count != ctx->result_count) {
    return false;
  }

  for (uint16_t i = 0; i < ctx->arg_count; ++i) {
    if (!loom_type_equal(data->types[i], ctx->arg_types[i])) return false;
  }
  for (uint16_t i = 0; i < ctx->result_count; ++i) {
    if (!loom_type_equal(data->types[ctx->arg_count + i],
                         ctx->result_types[i])) {
      return false;
    }
  }
  return true;
}

// Computes the same structural hash as loom_type_hash() for a first-class
// function signature described by temporary arg/result arrays.
static uint32_t loom_function_type_hash(const loom_type_t* arg_types,
                                        uint16_t arg_count,
                                        const loom_type_t* result_types,
                                        uint16_t result_count) {
  uint32_t hash = 2166136261u;
  uint32_t header =
      loom_type_make_header(LOOM_TYPE_FUNCTION, (loom_scalar_type_t)0, 0, 0);
  hash = loom_hash_u32_extend(hash, header);
  hash = loom_hash_u16_extend(hash, 0);
  hash = loom_hash_u16_extend(hash, 0);
  hash = loom_hash_u16_extend(hash, arg_count);
  hash = loom_hash_u16_extend(hash, result_count);
  hash = loom_hash_u16_extend(hash, (uint16_t)(arg_count + result_count));
  for (uint16_t i = 0; i < arg_count; ++i) {
    hash = loom_hash_u32_extend(hash, loom_type_hash(arg_types[i]));
  }
  for (uint16_t i = 0; i < result_count; ++i) {
    hash = loom_hash_u32_extend(hash, loom_type_hash(result_types[i]));
  }
  return hash;
}

static iree_status_t loom_module_clone_type_payload(loom_module_t* module,
                                                    loom_type_t type,
                                                    loom_type_t* out_type);

// Clones temporary arg/result arrays into a module-owned FAM payload and
// returns the resulting first-class function type by value.
static iree_status_t loom_module_clone_function_type_payload(
    loom_module_t* module, const loom_type_t* arg_types, uint16_t arg_count,
    const loom_type_t* result_types, uint16_t result_count,
    loom_type_t* out_type) {
  iree_host_size_t type_count = (iree_host_size_t)arg_count + result_count;
  iree_host_size_t alloc_size = 0;
  IREE_RETURN_IF_ERROR(
      IREE_STRUCT_LAYOUT(sizeof(loom_func_type_data_t), &alloc_size,
                         IREE_STRUCT_FIELD_FAM(type_count, loom_type_t)));

  loom_func_type_data_t* cloned_data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(&module->arena, alloc_size, (void**)&cloned_data));
  cloned_data->arg_count = arg_count;
  cloned_data->result_count = result_count;
  cloned_data->reserved = 0;
  for (uint16_t i = 0; i < arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_module_clone_type_payload(
        module, arg_types[i], &cloned_data->types[i]));
  }
  for (uint16_t i = 0; i < result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_module_clone_type_payload(
        module, result_types[i], &cloned_data->types[arg_count + i]));
  }
  *out_type = loom_type_function(cloned_data);
  return iree_ok_status();
}

// Clones one temporary by-value type candidate into module-owned payload.
static iree_status_t loom_module_clone_type_from_context(
    loom_module_t* module, const void* clone_context, loom_type_t* out_type) {
  loom_type_t type = *(const loom_type_t*)clone_context;
  return loom_module_clone_type_payload(module, type, out_type);
}

// Clones one temporary first-class function signature described by
// arg/result arrays into module-owned payload.
static iree_status_t loom_module_clone_function_type_from_context(
    loom_module_t* module, const void* clone_context, loom_type_t* out_type) {
  const loom_function_type_equal_context_t* ctx =
      (const loom_function_type_equal_context_t*)clone_context;
  return loom_module_clone_function_type_payload(
      module, ctx->arg_types, ctx->arg_count, ctx->result_types,
      ctx->result_count, out_type);
}

// Recursively clones any pointer-backed payload referenced by |type| into the
// module arena and returns an equivalent by-value type that owns module-local
// payload.
static iree_status_t loom_module_clone_type_payload(loom_module_t* module,
                                                    loom_type_t type,
                                                    loom_type_t* out_type) {
  *out_type = type;

  switch (loom_type_kind(type)) {
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* func_data = loom_type_func_data(type);
      if (!func_data) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "function type has a NULL argument/result payload");
      }
      return loom_module_clone_function_type_payload(
          module, func_data->types, func_data->arg_count,
          func_data->types + func_data->arg_count, func_data->result_count,
          out_type);
    }

    case LOOM_TYPE_DIALECT: {
      uint16_t param_count = loom_type_dialect_param_count(type);
      loom_string_id_t name_id = loom_type_dialect_name_id(type);
      if (param_count == 0) {
        *out_type = loom_type_dialect_opaque(name_id);
        return iree_ok_status();
      }

      const loom_type_t* params = loom_type_dialect_params(type);
      if (!params) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "dialect type '%u' has %u params but a NULL payload",
            (unsigned)name_id, (unsigned)param_count);
      }

      loom_type_t* cloned_params = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          &module->arena, param_count, sizeof(loom_type_t),
          (void**)&cloned_params));
      for (uint16_t i = 0; i < param_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_module_clone_type_payload(module, params[i],
                                                            &cloned_params[i]));
      }
      *out_type = loom_type_dialect(name_id, param_count, cloned_params);
      return iree_ok_status();
    }

    default:
      break;
  }

  if (loom_type_has_inline_dims(type)) {
    return iree_ok_status();
  }

  uint8_t rank = loom_type_rank(type);
  if (rank == 0) {
    out_type->dims[0] = 0;
    out_type->dims[1] = 0;
    return iree_ok_status();
  }

  const loom_overflow_dim_t* src_dims =
      (const loom_overflow_dim_t*)(uintptr_t)type.dims[0];
  if (!src_dims) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "rank-%u type has a NULL overflow dim payload",
                            rank);
  }

  loom_overflow_dim_t* cloned_dims = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &module->arena, rank, sizeof(loom_overflow_dim_t), (void**)&cloned_dims));
  memcpy(cloned_dims, src_dims,
         (iree_host_size_t)rank * sizeof(loom_overflow_dim_t));
  out_type->dims[0] = (uint64_t)(uintptr_t)cloned_dims;
  out_type->dims[1] = 0;
  return iree_ok_status();
}

// Looks up or inserts one type candidate in module->type_intern. Candidate
// cloning only runs on a miss so repeated structural hits do not allocate new
// module payloads.
static iree_status_t loom_module_intern_type_impl(
    loom_module_t* module, uint32_t hash, loom_intern_equal_fn_t equal_fn,
    const void* equal_context, loom_module_type_clone_fn_t clone_fn,
    const void* clone_context, loom_type_t* out_interned_type,
    loom_type_id_t* out_type_id) {
  uint32_t existing_index = loom_intern_table_lookup(&module->type_intern, hash,
                                                     equal_fn, equal_context);
  if (existing_index != UINT32_MAX) {
    *out_interned_type = module->types.entries[existing_index];
    if (out_type_id) *out_type_id = (loom_type_id_t)existing_index;
    return iree_ok_status();
  }

  // Type interner slots use 32-bit indices with UINT32_MAX as the empty
  // sentinel in loom_intern_table_t. Reject a new unique type before that
  // sentinel value could be used as a real table index.
  if (module->types.count >= LOOM_TYPE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "type table full (%" PRIhsz " entries, max id %u)",
                            module->types.count,
                            (unsigned)(LOOM_TYPE_ID_INVALID - 1));
  }

  loom_type_t type = {0};
  IREE_RETURN_IF_ERROR(clone_fn(module, clone_context, &type));

  // Ensure the type table has capacity before inserting into the intern hash
  // table.
  IREE_RETURN_IF_ERROR(
      loom_type_table_ensure_capacity(&module->arena, &module->types));
  uint32_t new_index = (uint32_t)module->types.count;
  uint32_t result_index = 0;
  IREE_RETURN_IF_ERROR(loom_intern_table_find_or_insert(
      &module->arena, &module->type_intern, hash, new_index, equal_fn,
      equal_context, &result_index));

  if (result_index != new_index) {
    *out_interned_type = module->types.entries[result_index];
    if (out_type_id) *out_type_id = (loom_type_id_t)result_index;
    return iree_ok_status();
  }

  module->types.entries[new_index] = type;
  module->types.count++;
  *out_interned_type = type;
  if (out_type_id) *out_type_id = (loom_type_id_t)new_index;
  return iree_ok_status();
}

iree_status_t loom_module_intern_type(loom_module_t* module, loom_type_t type,
                                      loom_type_t* out_interned_type) {
  uint32_t hash = loom_type_hash(type);
  loom_type_equal_context_t equal_context = {module, type};

  return loom_module_intern_type_impl(
      module, hash, loom_type_equal_fn, &equal_context,
      loom_module_clone_type_from_context, &type, out_interned_type,
      /*out_type_id=*/NULL);
}

iree_status_t loom_module_intern_type_id(loom_module_t* module,
                                         loom_type_t type,
                                         loom_type_id_t* out_type_id) {
  if (!out_type_id) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "type id output is NULL");
  }
  *out_type_id = LOOM_TYPE_ID_INVALID;
  uint32_t hash = loom_type_hash(type);
  loom_type_equal_context_t equal_context = {module, type};
  loom_type_t interned_type = {0};
  return loom_module_intern_type_impl(
      module, hash, loom_type_equal_fn, &equal_context,
      loom_module_clone_type_from_context, &type, &interned_type, out_type_id);
}

iree_status_t loom_module_intern_function_type(loom_module_t* module,
                                               const loom_type_t* arg_types,
                                               uint16_t arg_count,
                                               const loom_type_t* result_types,
                                               uint16_t result_count,
                                               loom_type_t* out_interned_type) {
  if (arg_count > 0 && !arg_types) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "function type has %u args but a NULL arg type payload",
        (unsigned)arg_count);
  }
  if (result_count > 0 && !result_types) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "function type has %u results but a NULL result type payload",
        (unsigned)result_count);
  }

  loom_function_type_equal_context_t equal_context = {
      module, arg_types, result_types, arg_count, result_count,
  };
  uint32_t hash =
      loom_function_type_hash(arg_types, arg_count, result_types, result_count);

  return loom_module_intern_type_impl(
      module, hash, loom_function_type_equal_fn, &equal_context,
      loom_module_clone_function_type_from_context, &equal_context,
      out_interned_type, /*out_type_id=*/NULL);
}

//===----------------------------------------------------------------------===//
// Type-use replacement
//===----------------------------------------------------------------------===//

static bool loom_module_type_has_replaceable_dims(loom_type_t type) {
  return loom_type_is_shaped(type) || loom_type_is_pool(type);
}

static iree_status_t loom_module_replace_type_value_refs_impl(
    loom_module_t* module, loom_type_t type, loom_value_id_t old_id,
    loom_value_id_t new_id, loom_type_t* out_type, bool* out_changed);

static iree_status_t loom_module_replace_type_ref_sequence(
    loom_module_t* module, const loom_type_t* types, uint16_t type_count,
    loom_value_id_t old_id, loom_value_id_t new_id, loom_type_t** out_types,
    bool* out_changed) {
  *out_types = NULL;
  *out_changed = false;
  if (type_count == 0) return iree_ok_status();
  if (!types) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "type sequence has %u entries but a NULL payload",
                            (unsigned)type_count);
  }

  loom_type_t* replaced_types = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&module->arena, type_count,
                                                 sizeof(loom_type_t),
                                                 (void**)&replaced_types));
  for (uint16_t i = 0; i < type_count; ++i) {
    bool element_changed = false;
    IREE_RETURN_IF_ERROR(loom_module_replace_type_value_refs_impl(
        module, types[i], old_id, new_id, &replaced_types[i],
        &element_changed));
    *out_changed = *out_changed || element_changed;
  }
  *out_types = replaced_types;
  return iree_ok_status();
}

static iree_status_t loom_module_replace_type_value_refs_impl(
    loom_module_t* module, loom_type_t type, loom_value_id_t old_id,
    loom_value_id_t new_id, loom_type_t* out_type, bool* out_changed) {
  *out_type = type;
  *out_changed = false;

  loom_type_kind_t kind = loom_type_kind(type);
  if (!loom_type_kind_is_valid(kind)) return iree_ok_status();

  switch (kind) {
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* data = loom_type_func_data(type);
      if (!data) return iree_ok_status();
      uint16_t type_count = (uint16_t)(data->arg_count + data->result_count);
      loom_type_t* replaced_types = NULL;
      IREE_RETURN_IF_ERROR(loom_module_replace_type_ref_sequence(
          module, data->types, type_count, old_id, new_id, &replaced_types,
          out_changed));
      if (!*out_changed) return iree_ok_status();
      return loom_module_intern_function_type(
          module, replaced_types, data->arg_count,
          replaced_types + data->arg_count, data->result_count, out_type);
    }

    case LOOM_TYPE_DIALECT: {
      uint16_t param_count = loom_type_dialect_param_count(type);
      loom_type_t* replaced_params = NULL;
      IREE_RETURN_IF_ERROR(loom_module_replace_type_ref_sequence(
          module, loom_type_dialect_params(type), param_count, old_id, new_id,
          &replaced_params, out_changed));
      if (!*out_changed) return iree_ok_status();
      loom_type_t replaced_type = loom_type_dialect(
          loom_type_dialect_name_id(type), param_count, replaced_params);
      return loom_module_intern_type(module, replaced_type, out_type);
    }

    default:
      break;
  }

  loom_type_t replaced_type = type;
  if (loom_module_type_has_replaceable_dims(type)) {
    uint8_t rank = loom_type_rank(type);
    if (loom_type_has_inline_dims(type)) {
      for (uint8_t i = 0; i < rank; ++i) {
        if (!loom_dim_is_dynamic(replaced_type.dims[i])) continue;
        if (loom_dim_value_id(replaced_type.dims[i]) != old_id) continue;
        replaced_type.dims[i] = loom_dim_pack_dynamic(new_id);
        *out_changed = true;
      }
    } else if (rank > 0) {
      const loom_overflow_dim_t* old_dims =
          (const loom_overflow_dim_t*)(uintptr_t)type.dims[0];
      if (!old_dims) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "rank-%u type has a NULL overflow dim payload",
                                rank);
      }
      bool dims_changed = false;
      for (uint8_t i = 0; i < rank; ++i) {
        if (!loom_dim_is_dynamic(old_dims[i])) continue;
        if (loom_dim_value_id(old_dims[i]) == old_id) dims_changed = true;
      }
      if (dims_changed) {
        loom_overflow_dim_t* new_dims = NULL;
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            &module->arena, rank, sizeof(loom_overflow_dim_t),
            (void**)&new_dims));
        for (uint8_t i = 0; i < rank; ++i) {
          new_dims[i] = old_dims[i];
          if (!loom_dim_is_dynamic(new_dims[i])) continue;
          if (loom_dim_value_id(new_dims[i]) == old_id) {
            new_dims[i] = loom_dim_pack_dynamic(new_id);
          }
        }
        replaced_type.dims[0] = (uint64_t)(uintptr_t)new_dims;
        replaced_type.dims[1] = 0;
        *out_changed = true;
      }
    }
  }

  if (old_id <= UINT16_MAX && loom_type_has_ssa_encoding(type) &&
      loom_type_encoding_value_id(type) == old_id) {
    if (new_id > UINT16_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "cannot store value %%%u in a 16-bit SSA encoding reference",
          (unsigned)new_id);
    }
    replaced_type.encoding_id = (uint16_t)new_id;
    *out_changed = true;
  }

  if (!*out_changed) return iree_ok_status();
  return loom_module_intern_type(module, replaced_type, out_type);
}

iree_status_t loom_module_replace_type_value_references(
    loom_module_t* module, loom_type_t type, loom_value_id_t old_id,
    loom_value_id_t new_id, loom_type_t* out_type, bool* out_changed) {
  if (!module || !out_type || !out_changed) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "module, output type, and output changed flag are required");
  }
  *out_type = type;
  *out_changed = false;
  if (old_id == new_id) return iree_ok_status();
  if (old_id >= module->values.count || new_id >= module->values.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "cannot replace type references from %%%u to %%%u in a module with "
        "%" PRIhsz " values",
        (unsigned)old_id, (unsigned)new_id, module->values.count);
  }
  return loom_module_replace_type_value_refs_impl(module, type, old_id, new_id,
                                                  out_type, out_changed);
}

iree_status_t loom_module_replace_value_type_uses(loom_module_t* module,
                                                  loom_value_id_t old_id,
                                                  loom_value_id_t new_id) {
  if (old_id == new_id) return iree_ok_status();
  if (old_id >= module->values.count || new_id >= module->values.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "cannot replace type references from %%%u to %%%u in a module with "
        "%" PRIhsz " values",
        (unsigned)old_id, (unsigned)new_id, module->values.count);
  }

  // Each iteration rewrites one carrier value reached from the incoming-use
  // head. The helper removes all outgoing type-use records for that carrier
  // before inserting the replacement records, so the next head is always a
  // still-unprocessed carrier without rescanning the table.
  while (loom_module_value_has_type_uses(module, old_id)) {
    loom_type_use_id_t use_id =
        module->type_uses.value_heads[old_id].first_incoming_use_id;
    loom_value_id_t user_value_id =
        module->type_uses.records[use_id].user_value_id;
    loom_type_t old_type = module->values.entries[user_value_id].type;
    loom_type_t new_type = old_type;
    bool changed = false;
    IREE_RETURN_IF_ERROR(loom_module_replace_type_value_refs_impl(
        module, old_type, old_id, new_id, &new_type, &changed));
    if (!changed) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "type-use table says value %%%u references %%%u, but the type does "
          "not contain that reference",
          (unsigned)user_value_id, (unsigned)old_id);
    }
    IREE_RETURN_IF_ERROR(
        loom_module_set_value_type(module, user_value_id, new_type));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Block and region creation
//===----------------------------------------------------------------------===//

iree_status_t loom_module_allocate_block(loom_module_t* module,
                                         loom_block_t** out_block) {
  IREE_TRACE_ZONE_BEGIN(z0);
  loom_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&module->arena, sizeof(loom_block_t),
                              (void**)&block));
  iree_status_t status = loom_module_initialize_block(module, block);
  if (iree_status_is_ok(status)) {
    *out_block = block;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t loom_module_allocate_region(loom_module_t* module,
                                          uint16_t block_count,
                                          loom_region_t** out_region) {
  IREE_TRACE_ZONE_BEGIN(z0);
  loom_region_t* region = NULL;
  iree_status_t status = iree_arena_allocate(
      &module->arena, sizeof(loom_region_t), (void**)&region);
  if (iree_status_is_ok(status)) {
    memset(region, 0, sizeof(loom_region_t));
    region->blocks = region->inline_blocks;
    region->inline_blocks[0] = &region->entry_block;
    region->block_capacity = 1;
    status = loom_module_initialize_block(module, &region->entry_block);
    region->entry_block.parent_region = region;
  }
  if (iree_status_is_ok(status) && block_count > 1) {
    region->block_capacity = block_count;
    status = iree_arena_allocate_array(&module->arena, block_count,
                                       sizeof(loom_block_t*),
                                       (void**)&region->blocks);
    if (iree_status_is_ok(status)) {
      memset(region->blocks, 0,
             (iree_host_size_t)block_count * sizeof(loom_block_t*));
      region->blocks[0] = &region->entry_block;
    }
    for (uint16_t i = 1; i < block_count && iree_status_is_ok(status); ++i) {
      status = loom_module_allocate_block(module, &region->blocks[i]);
      if (iree_status_is_ok(status)) {
        region->blocks[i]->parent_region = region;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    region->block_count = block_count;
    *out_region = region;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t loom_region_append_block(loom_module_t* module,
                                       loom_region_t* region,
                                       loom_block_t** out_block) {
  IREE_RETURN_IF_ERROR(loom_region_blocks_ensure_capacity(module, region));
  loom_block_t* block = &region->entry_block;
  if (region->block_count > 0) {
    IREE_RETURN_IF_ERROR(loom_module_allocate_block(module, &block));
  }
  block->parent_region = region;
  region->blocks[region->block_count] = block;
  ++region->block_count;
  *out_block = block;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Block arguments
//===----------------------------------------------------------------------===//

iree_status_t loom_block_add_arg(loom_module_t* module, loom_block_t* block,
                                 loom_value_id_t value_id) {
  if (block->arg_count >= UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "block arg count exceeds UINT16_MAX");
  }
  if (value_id >= module->values.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value %%%u out of range (module has %" PRIhsz
                            " values)",
                            (unsigned)value_id, module->values.count);
  }
  if (block->arg_count >= block->arg_capacity) {
    iree_host_size_t capacity = block->arg_capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        &module->arena, block->arg_count, /*minimum_capacity=*/4,
        sizeof(loom_value_id_t), &capacity, (void**)&block->arg_ids));
    block->arg_capacity = (uint16_t)capacity;
  }
  uint16_t arg_index = block->arg_count++;
  block->arg_ids[arg_index] = value_id;

  loom_value_t* value = &module->values.entries[value_id];
  value->flags |= LOOM_VALUE_FLAG_BLOCK_ARG;
  value->def = loom_value_def_make_block(block, arg_index);
  return iree_ok_status();
}

iree_status_t loom_block_remove_arg(loom_module_t* module, loom_block_t* block,
                                    uint16_t arg_index) {
  if (arg_index >= block->arg_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "block argument index %u out of range for %u argument(s)",
        (unsigned)arg_index, (unsigned)block->arg_count);
  }

  loom_value_id_t value_id = block->arg_ids[arg_index];
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "block argument %u value %%%u is invalid",
                            (unsigned)arg_index, (unsigned)value_id);
  }

  loom_value_t* value = loom_module_value(module, value_id);
  if (!loom_value_is_block_arg(value) || loom_value_def_block(value) != block ||
      loom_value_def_index(value) != arg_index) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "value %%%u is not block argument %u of the target block",
        (unsigned)value_id, (unsigned)arg_index);
  }
  if (value->use_count != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cannot remove block argument %%%u with %u operand use(s)",
        (unsigned)value_id, (unsigned)value->use_count);
  }
  if (loom_module_value_has_type_uses(module, value_id)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cannot remove block argument %%%u with incoming type use(s)",
        (unsigned)value_id);
  }

  loom_module_drop_value_type_uses(module, value_id);
  value->flags &= ~LOOM_VALUE_FLAG_BLOCK_ARG;
  value->def = loom_value_def_make_none();

  for (uint16_t i = (uint16_t)(arg_index + 1); i < block->arg_count; ++i) {
    loom_value_id_t shifted_id = block->arg_ids[i];
    block->arg_ids[i - 1] = shifted_id;
    if (shifted_id == LOOM_VALUE_ID_INVALID ||
        shifted_id >= module->values.count) {
      continue;
    }
    loom_value_t* shifted = loom_module_value(module, shifted_id);
    if (loom_value_is_block_arg(shifted) &&
        loom_value_def_block(shifted) == block) {
      shifted->def = loom_value_def_make_block(block, (uint16_t)(i - 1));
    }
  }

  --block->arg_count;
  block->arg_ids[block->arg_count] = LOOM_VALUE_ID_INVALID;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Block op insertion
//===----------------------------------------------------------------------===//

#define LOOM_BLOCK_ORDINAL_STRIDE UINT64_C(0x100000000)

static iree_status_t loom_block_renumber_ordinals(loom_block_t* block) {
  uint64_t ordinal = LOOM_BLOCK_ORDINAL_STRIDE;
  loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    op->block_ordinal = ordinal;
    if (op->next_op && UINT64_MAX - ordinal < LOOM_BLOCK_ORDINAL_STRIDE) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "block has too many operations to assign sparse ordinals");
    }
    ordinal += LOOM_BLOCK_ORDINAL_STRIDE;
  }
  return iree_ok_status();
}

static iree_status_t loom_block_append_ordinal(loom_block_t* block,
                                               uint64_t* out_ordinal) {
  if (!block->last_op) {
    *out_ordinal = LOOM_BLOCK_ORDINAL_STRIDE;
    return iree_ok_status();
  }
  if (UINT64_MAX - block->last_op->block_ordinal > LOOM_BLOCK_ORDINAL_STRIDE) {
    *out_ordinal = block->last_op->block_ordinal + LOOM_BLOCK_ORDINAL_STRIDE;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_block_renumber_ordinals(block));
  if (UINT64_MAX - block->last_op->block_ordinal <= LOOM_BLOCK_ORDINAL_STRIDE) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "block has too many operations to append another ordinal");
  }
  *out_ordinal = block->last_op->block_ordinal + LOOM_BLOCK_ORDINAL_STRIDE;
  return iree_ok_status();
}

static iree_status_t loom_block_insert_ordinal(loom_block_t* block,
                                               loom_op_t* prev_op,
                                               loom_op_t* next_op,
                                               uint64_t* out_ordinal) {
  if (!next_op) return loom_block_append_ordinal(block, out_ordinal);

  uint64_t lower = prev_op ? prev_op->block_ordinal : 0;
  uint64_t upper = next_op->block_ordinal;
  if (lower + 1 < upper) {
    *out_ordinal = prev_op ? lower + 1 : lower + (upper - lower) / 2;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_block_renumber_ordinals(block));
  lower = prev_op ? prev_op->block_ordinal : 0;
  upper = next_op->block_ordinal;
  if (lower + 1 >= upper) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "block has too many operations to assign an insertion ordinal");
  }
  *out_ordinal = prev_op ? lower + 1 : lower + (upper - lower) / 2;
  return iree_ok_status();
}

static iree_status_t loom_block_link_op_between(loom_module_t* module,
                                                loom_block_t* block,
                                                loom_op_t* prev_op,
                                                loom_op_t* next_op,
                                                loom_op_t* op) {
  (void)module;
  if (block->op_count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "block op count exceeds UINT32_MAX");
  }
  if (op->parent_block && op->parent_block != block) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cannot insert op already owned by a different block");
  }
  if (op->prev_op || op->next_op || block->first_op == op ||
      block->last_op == op) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot insert op already linked into a block");
  }

  uint64_t ordinal = 0;
  IREE_RETURN_IF_ERROR(
      loom_block_insert_ordinal(block, prev_op, next_op, &ordinal));

  op->parent_block = block;
  op->block_ordinal = ordinal;
  op->prev_op = prev_op;
  op->next_op = next_op;

  if (prev_op) {
    prev_op->next_op = op;
  } else {
    block->first_op = op;
  }
  if (next_op) {
    next_op->prev_op = op;
  } else {
    block->last_op = op;
  }
  ++block->op_count;
  return iree_ok_status();
}

iree_status_t loom_block_append_op(loom_module_t* module, loom_block_t* block,
                                   loom_op_t* op) {
  return loom_block_link_op_between(module, block, block->last_op, NULL, op);
}

iree_status_t loom_block_insert_before_op(loom_module_t* module,
                                          loom_block_t* block,
                                          loom_op_t* before_op, loom_op_t* op) {
  if (!before_op) return loom_block_append_op(module, block, op);
  if (before_op->parent_block != block ||
      iree_any_bit_set(before_op->flags, LOOM_OP_FLAG_DEAD)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cannot insert before op that is not live in the target block");
  }
  return loom_block_link_op_between(module, block, before_op->prev_op,
                                    before_op, op);
}

iree_status_t loom_block_insert_op(loom_module_t* module, loom_block_t* block,
                                   iree_host_size_t index, loom_op_t* op) {
  if (index >= block->op_count) {
    return loom_block_append_op(module, block, op);
  }
  loom_op_t* before_op = block->first_op;
  for (iree_host_size_t i = 0; i < index; ++i) {
    before_op = before_op->next_op;
  }
  return loom_block_insert_before_op(module, block, before_op, op);
}

void loom_block_unlink_op(loom_module_t* module, loom_op_t* op) {
  loom_block_t* block = op->parent_block;
  if (!block || iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) return;
  loom_module_drop_op_effects(module, op);

  if (op->prev_op) {
    op->prev_op->next_op = op->next_op;
  } else if (block->first_op == op) {
    block->first_op = op->next_op;
  }

  if (op->next_op) {
    op->next_op->prev_op = op->prev_op;
  } else if (block->last_op == op) {
    block->last_op = op->prev_op;
  }

  op->prev_op = NULL;
  op->next_op = NULL;
  op->block_ordinal = 0;
  if (block->op_count > 0) --block->op_count;
}

iree_host_size_t loom_block_find_op(const loom_block_t* block,
                                    const loom_op_t* op) {
  iree_host_size_t index = 0;
  const loom_op_t* current = NULL;
  loom_block_for_each_op(block, current) {
    if (current == op) return index;
    ++index;
  }
  return IREE_HOST_SIZE_MAX;
}
