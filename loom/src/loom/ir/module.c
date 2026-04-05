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

// FNV-1a hash over a byte array.
static uint32_t loom_hash_bytes(const void* data, iree_host_size_t length) {
  uint32_t hash = 2166136261u;
  const uint8_t* bytes = (const uint8_t*)data;
  for (iree_host_size_t i = 0; i < length; ++i) {
    hash ^= bytes[i];
    hash *= 16777619u;
  }
  return hash;
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
  iree_status_t status = iree_arena_allocate_array(
      arena, capacity, sizeof(uint32_t), (void**)&hashes);
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(arena, capacity, sizeof(uint32_t),
                                       (void**)&indices);
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t byte_size = capacity * sizeof(uint32_t);
    memset(hashes, 0, byte_size);
    memset(indices, 0xFF, byte_size);
    table->count = 0;
    table->capacity = capacity;
    table->hashes = hashes;
    table->indices = indices;
  }
  return status;
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

static iree_status_t loom_block_ops_ensure_capacity(
    iree_arena_allocator_t* arena, loom_block_t* block) {
  if (block->op_count < block->op_capacity) return iree_ok_status();
  iree_host_size_t capacity = block->op_capacity;
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      arena, block->op_count, /*minimum_capacity=*/16, sizeof(loom_op_t*),
      &capacity, (void**)&block->ops));
  block->op_capacity = (uint16_t)capacity;
  return iree_ok_status();
}

static iree_status_t loom_module_initialize_block(loom_module_t* module,
                                                  loom_block_t* block) {
  memset(block, 0, sizeof(*block));
  block->label_id = LOOM_STRING_ID_INVALID;
  block->op_capacity = 16;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&module->arena, block->op_capacity,
                                sizeof(loom_op_t*), (void**)&block->ops));
  memset(block->ops, 0,
         (iree_host_size_t)block->op_capacity * sizeof(loom_op_t*));
  return iree_ok_status();
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

  iree_status_t status = iree_ok_status();

  // Values: 64-byte aligned.
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array_aligned(arena, value_capacity,
                                               sizeof(loom_value_t), 64,
                                               (void**)&module->values.entries);
    if (iree_status_is_ok(status)) {
      module->values.capacity = value_capacity;
      memset(module->values.entries, 0, value_capacity * sizeof(loom_value_t));
    }
  }

  // Strings.
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(arena, string_capacity,
                                       sizeof(iree_string_view_t),
                                       (void**)&module->strings.entries);
    if (iree_status_is_ok(status)) {
      module->strings.capacity = string_capacity;
      memset(module->strings.entries, 0,
             string_capacity * sizeof(iree_string_view_t));
    }
  }

  // Types.
  if (iree_status_is_ok(status)) {
    status =
        iree_arena_allocate_array(arena, type_capacity, sizeof(loom_type_t),
                                  (void**)&module->types.entries);
    if (iree_status_is_ok(status)) {
      module->types.capacity = type_capacity;
      memset(module->types.entries, 0, type_capacity * sizeof(loom_type_t));
    }
  }

  // Symbols.
  if (iree_status_is_ok(status)) {
    status =
        iree_arena_allocate_array(arena, symbol_capacity, sizeof(loom_symbol_t),
                                  (void**)&module->symbols.entries);
    if (iree_status_is_ok(status)) {
      module->symbols.capacity = symbol_capacity;
      memset(module->symbols.entries, 0,
             symbol_capacity * sizeof(loom_symbol_t));
    }
  }

  // Intern tables sized to match entry capacity.
  if (iree_status_is_ok(status)) {
    iree_host_size_t intern_capacity =
        loom_intern_capacity_for_entries(string_capacity);
    status = loom_intern_table_allocate(arena, intern_capacity,
                                        &module->string_intern);
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t intern_capacity =
        loom_intern_capacity_for_entries(type_capacity);
    status = loom_intern_table_allocate(arena, intern_capacity,
                                        &module->type_intern);
  }

  return status;
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

  // Grow the table if needed.
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      &module->arena, module->encodings.count,
      /*minimum_capacity=*/8, sizeof(loom_encoding_t),
      &module->encodings.capacity, (void**)&module->encodings.entries));

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

  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      &module->arena, module->symbols.count,
      /*minimum_capacity=*/8, sizeof(loom_symbol_t), &module->symbols.capacity,
      (void**)&module->symbols.entries));

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
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        &module->arena, /*existing_count=*/0, /*minimum_capacity=*/16,
        sizeof(loom_location_entry_t), &module->locations.capacity,
        (void**)&module->locations.entries));
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

  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      &module->arena, module->locations.count, /*minimum_capacity=*/16,
      sizeof(loom_location_entry_t), &module->locations.capacity,
      (void**)&module->locations.entries));

  loom_location_id_t id = (loom_location_id_t)module->locations.count;
  module->locations.entries[id] = entry;
  module->locations.count++;
  *out_location_id = id;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Value definition
//===----------------------------------------------------------------------===//

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

  loom_value_id_t id = (loom_value_id_t)module->values.count;
  loom_value_t* value = &module->values.entries[id];
  value->type = type;
  value->name_id = LOOM_STRING_ID_INVALID;
  value->def = loom_value_def_make_none();
  module->values.count++;

  *out_value_id = id;
  return iree_ok_status();
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

static iree_status_t loom_module_canonicalize_attr_dict_value_impl(
    loom_module_t* module, loom_attribute_t value, iree_host_size_t depth,
    loom_attribute_t* out_value);

static iree_status_t loom_module_make_canonical_attr_dict_impl(
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
    IREE_RETURN_IF_ERROR(loom_module_canonicalize_attr_dict_value_impl(
        module, entries[i].value, depth + 1, &entry.value));

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

static iree_status_t loom_module_canonicalize_attr_dict_value_impl(
    loom_module_t* module, loom_attribute_t value, iree_host_size_t depth,
    loom_attribute_t* out_value) {
  value.reserved_0 = 0;
  value.reserved_1 = 0;

  switch ((loom_attr_kind_t)value.kind) {
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
      if (value.count == 0) {
        value.i64_array = NULL;
      } else if (!value.i64_array) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "non-empty dict attribute i64 array value has a NULL payload");
      } else {
        int64_t* values = NULL;
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            &module->arena, value.count, sizeof(int64_t), (void**)&values));
        memcpy(values, value.i64_array,
               (iree_host_size_t)value.count * sizeof(int64_t));
        value.i64_array = values;
      }
      *out_value = value;
      return iree_ok_status();
    case LOOM_ATTR_PREDICATE_LIST:
      if (value.count == 0) {
        value.predicate_list = NULL;
      } else if (!value.predicate_list) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "non-empty dict attribute predicate list has a NULL payload");
      } else {
        loom_predicate_t* predicates = NULL;
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            &module->arena, value.count, sizeof(loom_predicate_t),
            (void**)&predicates));
        memcpy(predicates, value.predicate_list,
               (iree_host_size_t)value.count * sizeof(loom_predicate_t));
        value.predicate_list = predicates;
      }
      *out_value = value;
      return iree_ok_status();
    case LOOM_ATTR_DICT:
      return loom_module_make_canonical_attr_dict_impl(
          module, value.dict_entries, value.count, depth, out_value);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "dict attribute value has unknown kind %u",
                              (unsigned)value.kind);
  }
}

iree_status_t loom_module_make_canonical_attr_dict(
    loom_module_t* module, loom_named_attr_slice_t entries,
    loom_attribute_t* out_attr) {
  return loom_module_make_canonical_attr_dict_impl(module, entries.entries,
                                                   entries.count, 0, out_attr);
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
    IREE_RETURN_IF_ERROR(loom_module_canonicalize_attr_dict_value_impl(
        module, update->value, /*depth=*/1, &canonical_value));
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

static iree_status_t loom_module_verify_canonical_attr_dict_value_impl(
    const loom_module_t* module, const loom_attribute_t* value,
    iree_host_size_t depth);

static iree_status_t loom_module_verify_canonical_attr_dict_impl(
    const loom_module_t* module, const loom_attribute_t* attr,
    iree_host_size_t depth) {
  if (attr->kind != LOOM_ATTR_DICT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected a DICT attribute, got kind %u",
                            (unsigned)attr->kind);
  }
  if (attr->reserved_0 != 0 || attr->reserved_1 != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dict attribute has non-zero reserved bits");
  }
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

    IREE_RETURN_IF_ERROR(loom_module_verify_canonical_attr_dict_value_impl(
        module, &entry->value, depth + 1));
    previous_key_name = key_name;
  }

  return iree_ok_status();
}

static iree_status_t loom_module_verify_canonical_attr_dict_value_impl(
    const loom_module_t* module, const loom_attribute_t* value,
    iree_host_size_t depth) {
  if (value->reserved_0 != 0 || value->reserved_1 != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dict attribute value has non-zero reserved bits");
  }

  switch ((loom_attr_kind_t)value->kind) {
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
    case LOOM_ATTR_PREDICATE_LIST:
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
    case LOOM_ATTR_DICT:
      return loom_module_verify_canonical_attr_dict_impl(module, value, depth);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "dict attribute value has unknown kind %u",
                              (unsigned)value->kind);
  }
}

iree_status_t loom_module_verify_canonical_attr_dict(
    const loom_module_t* module, loom_attribute_t attr) {
  return loom_module_verify_canonical_attr_dict_impl(module, &attr, 0);
}

//===----------------------------------------------------------------------===//
// Type interning
//===----------------------------------------------------------------------===//

typedef struct loom_type_equal_context_t {
  const loom_module_t* module;
  loom_type_t type;
} loom_type_equal_context_t;

static bool loom_type_equal_fn(const void* context, uint32_t index) {
  const loom_type_equal_context_t* ctx =
      (const loom_type_equal_context_t*)context;
  return loom_type_equal(ctx->module->types.entries[index], ctx->type);
}

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

      iree_host_size_t type_count =
          (iree_host_size_t)func_data->arg_count + func_data->result_count;
      iree_host_size_t alloc_size = 0;
      IREE_RETURN_IF_ERROR(
          IREE_STRUCT_LAYOUT(sizeof(loom_func_type_data_t), &alloc_size,
                             IREE_STRUCT_FIELD_FAM(type_count, loom_type_t)));

      loom_func_type_data_t* cloned_data = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate(&module->arena, alloc_size,
                                               (void**)&cloned_data));
      cloned_data->arg_count = func_data->arg_count;
      cloned_data->result_count = func_data->result_count;
      cloned_data->reserved = 0;
      for (iree_host_size_t i = 0; i < type_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_module_clone_type_payload(
            module, func_data->types[i], &cloned_data->types[i]));
      }
      *out_type = loom_type_function(cloned_data);
      return iree_ok_status();
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

iree_status_t loom_module_intern_type(loom_module_t* module, loom_type_t type,
                                      loom_type_t* out_interned_type) {
  uint32_t hash = loom_type_hash(type);
  loom_type_equal_context_t equal_context = {module, type};

  uint32_t existing_index = loom_intern_table_lookup(
      &module->type_intern, hash, loom_type_equal_fn, &equal_context);
  if (existing_index != UINT32_MAX) {
    *out_interned_type = module->types.entries[existing_index];
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

  // Ensure the type table has capacity before inserting into the
  // intern hash table.
  IREE_RETURN_IF_ERROR(
      loom_type_table_ensure_capacity(&module->arena, &module->types));
  uint32_t new_index = (uint32_t)module->types.count;
  uint32_t result_index = 0;
  IREE_RETURN_IF_ERROR(loom_intern_table_find_or_insert(
      &module->arena, &module->type_intern, hash, new_index, loom_type_equal_fn,
      &equal_context, &result_index));

  if (result_index != new_index) {
    *out_interned_type = module->types.entries[result_index];
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_module_clone_type_payload(module, type, &type));

  module->types.entries[new_index] = type;
  module->types.count++;

  *out_interned_type = type;
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
                            "value_id %u out of range (module has %" PRIhsz
                            " values)",
                            value_id, module->values.count);
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

//===----------------------------------------------------------------------===//
// Block op insertion
//===----------------------------------------------------------------------===//

iree_status_t loom_block_append_op(loom_module_t* module, loom_block_t* block,
                                   loom_op_t* op) {
  if (block->op_count >= UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "block op count exceeds UINT16_MAX");
  }
  IREE_RETURN_IF_ERROR(loom_block_ops_ensure_capacity(&module->arena, block));
  block->ops[block->op_count++] = op;
  return iree_ok_status();
}

iree_status_t loom_block_insert_op(loom_module_t* module, loom_block_t* block,
                                   uint16_t index, loom_op_t* op) {
  if (index >= block->op_count) {
    return loom_block_append_op(module, block, op);
  }
  if (block->op_count >= UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "block op count exceeds UINT16_MAX");
  }
  IREE_RETURN_IF_ERROR(loom_block_ops_ensure_capacity(&module->arena, block));
  memmove(&block->ops[index + 1], &block->ops[index],
          (iree_host_size_t)(block->op_count - index) * sizeof(loom_op_t*));
  block->ops[index] = op;
  block->op_count++;
  return iree_ok_status();
}

uint16_t loom_block_find_op(const loom_block_t* block, const loom_op_t* op) {
  for (uint16_t i = 0; i < block->op_count; ++i) {
    if (block->ops[i] == op) return i;
  }
  return UINT16_MAX;
}
