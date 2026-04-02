// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/module.h"

#include <string.h>

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

// Hashes a loom_type_t. For inline-dim types (rank <= 2), hashes all
// 24 bytes. For overflow types, hashes the header fields and each
// overflow dim individually.
static uint32_t loom_hash_type(loom_type_t type) {
  if (loom_type_has_inline_dims(type)) {
    return loom_hash_bytes(&type, sizeof(loom_type_t));
  }
  // Overflow: hash header + encoding_id + encoding_flags + each dim.
  uint32_t hash = 2166136261u;
  hash ^= type.header;
  hash *= 16777619u;
  hash ^= type.encoding_id;
  hash *= 16777619u;
  hash ^= type.encoding_flags;
  hash *= 16777619u;
  uint8_t rank = loom_type_rank(type);
  const loom_overflow_dim_t* dims =
      (const loom_overflow_dim_t*)(uintptr_t)type.dims[0];
  for (uint8_t i = 0; i < rank; ++i) {
    uint64_t dim = dims[i];
    hash ^= (uint32_t)dim;
    hash *= 16777619u;
    hash ^= (uint32_t)(dim >> 32);
    hash *= 16777619u;
  }
  return hash;
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

// Returns true if two encodings have the same name and attributes.
static bool loom_encoding_equal(const loom_encoding_t* a,
                                const loom_encoding_t* b) {
  if (a->name_id != b->name_id) return false;
  if (a->attribute_count != b->attribute_count) return false;
  for (uint8_t i = 0; i < a->attribute_count; ++i) {
    if (a->attributes[i].name_id != b->attributes[i].name_id) return false;
    if (a->attributes[i].value.kind != b->attributes[i].value.kind) {
      return false;
    }
    if (a->attributes[i].value.raw != b->attributes[i].value.raw) {
      return false;
    }
  }
  return true;
}

iree_status_t loom_module_add_encoding(loom_module_t* module,
                                       const loom_encoding_t* encoding,
                                       uint16_t* out_encoding_id) {
  // Check for an existing duplicate.
  for (iree_host_size_t i = 0; i < module->encodings.count; ++i) {
    if (loom_encoding_equal(&module->encodings.entries[i], encoding)) {
      *out_encoding_id = (uint16_t)(i + 1);
      return iree_ok_status();
    }
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
  entry->name_id = encoding->name_id;
  entry->alias_id = encoding->alias_id;
  entry->attribute_count = encoding->attribute_count;

  // Arena-copy the attribute array (names are already interned string
  // IDs, attribute values are copied by value).
  if (encoding->attribute_count > 0) {
    loom_named_attr_t* attributes = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &module->arena, encoding->attribute_count, sizeof(loom_named_attr_t),
        (void**)&attributes));
    memcpy(attributes, encoding->attributes,
           encoding->attribute_count * sizeof(loom_named_attr_t));
    entry->attributes = attributes;
  }

  *out_encoding_id = (uint16_t)(module->encodings.count + 1);
  ++module->encodings.count;
  return iree_ok_status();
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

iree_status_t loom_module_intern_type(loom_module_t* module, loom_type_t type,
                                      loom_type_t* out_interned_type) {
  uint32_t hash = loom_hash_type(type);
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

  // For overflow types (rank > 2), arena-allocate the dim array.
  if (!loom_type_has_inline_dims(type)) {
    uint8_t rank = loom_type_rank(type);
    const loom_overflow_dim_t* src_dims =
        (const loom_overflow_dim_t*)(uintptr_t)type.dims[0];
    loom_overflow_dim_t* new_dims = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &module->arena, rank, sizeof(loom_overflow_dim_t), (void**)&new_dims));
    memcpy(new_dims, src_dims, rank * sizeof(loom_overflow_dim_t));
    type.dims[0] = (uint64_t)(uintptr_t)new_dims;
  }

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
  memset(block, 0, sizeof(loom_block_t));
  block->label_id = LOOM_STRING_ID_INVALID;

  block->op_capacity = 16;
  iree_status_t status =
      iree_arena_allocate_array(&module->arena, block->op_capacity,
                                sizeof(loom_op_t*), (void**)&block->ops);
  if (iree_status_is_ok(status)) {
    memset(block->ops, 0,
           (iree_host_size_t)block->op_capacity * sizeof(loom_op_t*));
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
    region->block_count = block_count;
  }
  if (iree_status_is_ok(status) && block_count > 0) {
    status = iree_arena_allocate_array(&module->arena, block_count,
                                       sizeof(loom_block_t),
                                       (void**)&region->blocks);
    if (iree_status_is_ok(status)) {
      memset(region->blocks, 0,
             (iree_host_size_t)block_count * sizeof(loom_block_t));
    }
    for (uint16_t i = 0; i < block_count && iree_status_is_ok(status); ++i) {
      loom_block_t* block = &region->blocks[i];
      block->label_id = LOOM_STRING_ID_INVALID;
      block->op_capacity = 16;
      status =
          iree_arena_allocate_array(&module->arena, block->op_capacity,
                                    sizeof(loom_op_t*), (void**)&block->ops);
      if (iree_status_is_ok(status)) {
        memset(block->ops, 0,
               (iree_host_size_t)block->op_capacity * sizeof(loom_op_t*));
      }
    }
  }
  if (iree_status_is_ok(status)) {
    *out_region = region;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
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
