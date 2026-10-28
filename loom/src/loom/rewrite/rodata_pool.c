// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/rewrite/rodata_pool.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/ir/intern_table.h"
#include "loom/ir/module.h"
#include "loom/ir/symbol_map.h"
#include "loom/ops/global/ops.h"

typedef struct loom_rodata_record_t {
  // Module-local symbol published during execution; null while only planned.
  loom_symbol_ref_t symbol;
  // First source location that requested the payload.
  loom_location_id_t location;
  // Copied immutable bytes and their explicit presentation form.
  loom_attribute_t contents;
  // Content hash retained for the symbol name assigned during execution.
  uint32_t hash;
  // Log2 of the maximum power-of-two alignment requested by equal payloads.
  uint8_t minimum_alignment_log2;
} loom_rodata_record_t;
static_assert(sizeof(loom_rodata_record_t) <= 32,
              "immutable payload records must fit in 32 bytes");

struct loom_rodata_pool_t {
  // Arena owning pool records, immutable payloads, and reservation indexes.
  iree_arena_allocator_t* arena;
  // Interned immutable payload records.
  loom_rodata_record_t* read_only_data_records;
  // Number of populated read_only_data_records entries.
  iree_host_size_t read_only_data_record_count;
  // Number of allocated read_only_data_records entries.
  iree_host_size_t read_only_data_record_capacity;
  // Content index over read_only_data_records.
  loom_intern_table_t read_only_data_index;
  // Module symbol names indexed for constant-time reservations.
  loom_symbol_map_t symbol_names;
  // Module symbol prefix already captured in symbol_names.
  iree_host_size_t indexed_symbol_count;
  // True after all interned payloads have been materialized.
  bool finalized;
};

iree_status_t loom_rodata_pool_allocate(iree_arena_allocator_t* arena,
                                        loom_rodata_pool_t** out_pool) {
  *out_pool = NULL;
  loom_rodata_pool_t* pool = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*pool), (void**)&pool));
  memset(pool, 0, sizeof(*pool));
  pool->arena = arena;
  IREE_RETURN_IF_ERROR(loom_intern_table_initialize(
      arena, /*capacity=*/0, &pool->read_only_data_index));
  *out_pool = pool;
  return iree_ok_status();
}

static uint32_t loom_rodata_hash(iree_const_byte_span_t contents) {
  uint32_t hash = (2166136261u ^ (uint32_t)contents.data_length) * 16777619u;
  for (iree_host_size_t i = 0; i < contents.data_length; ++i) {
    hash = (hash ^ contents.data[i]) * 16777619u;
  }
  return hash;
}

typedef struct loom_rodata_query_t {
  // Current record-array base, which may move between interning calls.
  const loom_rodata_record_t* records;
  // Borrowed bytes being compared with interned, arena-owned payloads.
  iree_const_byte_span_t contents;
} loom_rodata_query_t;

static bool loom_rodata_equal(const void* context, uint32_t record_index) {
  const loom_rodata_query_t* query = context;
  const iree_const_byte_span_t existing =
      loom_attr_as_bytes(query->records[record_index].contents);
  return existing.data_length == query->contents.data_length &&
         (existing.data_length == 0 ||
          memcmp(existing.data, query->contents.data,
                 query->contents.data_length) == 0);
}

static iree_status_t loom_rodata_pool_refresh_symbol_names(
    loom_rodata_pool_t* pool, const loom_module_t* module) {
  while (pool->indexed_symbol_count < module->symbols.count) {
    const iree_host_size_t symbol_id = pool->indexed_symbol_count;
    const loom_string_id_t name_id = module->symbols.entries[symbol_id].name_id;
    if (name_id != LOOM_STRING_ID_INVALID) {
      loom_symbol_id_t indexed_symbol_id = LOOM_SYMBOL_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_symbol_map_find_or_insert(
          &pool->symbol_names, pool->arena, name_id,
          (loom_symbol_id_t)symbol_id, &indexed_symbol_id));
    }
    ++pool->indexed_symbol_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_rodata_pool_reserve_symbol(
    loom_rodata_pool_t* pool, loom_module_t* module, uint32_t hash,
    loom_symbol_ref_t* out_symbol) {
  *out_symbol = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(loom_rodata_pool_refresh_symbol_names(pool, module));
  char name_storage[64];
  for (uint32_t discriminator = 0; discriminator < LOOM_SYMBOL_ID_INVALID;
       ++discriminator) {
    const int name_length =
        discriminator == 0 ? snprintf(name_storage, sizeof(name_storage),
                                      "__low_rodata_%08" PRIx32, hash)
                           : snprintf(name_storage, sizeof(name_storage),
                                      "__low_rodata_%08" PRIx32 "$%" PRIu32,
                                      hash, discriminator);
    if (name_length < 0 ||
        (iree_host_size_t)name_length >= sizeof(name_storage)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "read-only data symbol name overflow");
    }
    const iree_string_view_t name =
        iree_make_string_view(name_storage, (iree_host_size_t)name_length);
    const loom_string_id_t existing_name_id =
        loom_module_lookup_string(module, name);
    if (existing_name_id != LOOM_STRING_ID_INVALID &&
        loom_symbol_map_find(&pool->symbol_names, existing_name_id) !=
            LOOM_SYMBOL_ID_INVALID) {
      continue;
    }

    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_intern_string(module, name, &name_id));
    out_symbol->module_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_module_add_symbol(module, name_id, &out_symbol->symbol_id));
    IREE_RETURN_IF_ERROR(loom_symbol_map_insert(
        &pool->symbol_names, pool->arena, name_id, out_symbol->symbol_id));
    ++pool->indexed_symbol_count;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "read-only data symbol namespace exhausted");
}

iree_status_t loom_rodata_pool_intern(loom_rodata_pool_t* pool,
                                      loom_attribute_t contents_attr,
                                      uint64_t minimum_alignment,
                                      loom_location_id_t location,
                                      loom_rodata_id_t* out_id) {
  const iree_const_byte_span_t contents = loom_attr_as_bytes(contents_attr);
  IREE_ASSERT_FALSE(pool->finalized);

  const uint32_t hash = loom_rodata_hash(contents);
  const uint8_t minimum_alignment_log2 =
      (uint8_t)iree_math_count_trailing_zeros_u64(minimum_alignment);
  const loom_rodata_query_t query = {
      .records = pool->read_only_data_records,
      .contents = contents,
  };
  const loom_intern_probe_t probe = loom_intern_table_probe(
      &pool->read_only_data_index, hash, loom_rodata_equal, &query);
  if (probe.index != UINT32_MAX) {
    loom_rodata_record_t* record = &pool->read_only_data_records[probe.index];
    record->minimum_alignment_log2 =
        iree_max(record->minimum_alignment_log2, minimum_alignment_log2);
    record->contents.flags |= contents_attr.flags;
    *out_id = probe.index;
    return iree_ok_status();
  }

  if (pool->read_only_data_record_count + 1 > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "too many interned read-only data payloads");
  }
  iree_host_size_t slot = probe.slot;
  IREE_RETURN_IF_ERROR(loom_intern_table_reserve_insert(
      pool->arena, &pool->read_only_data_index, hash,
      /*insertion_count=*/1, &slot));
  if (pool->read_only_data_record_count ==
      pool->read_only_data_record_capacity) {
    const iree_host_size_t minimum_capacity =
        pool->read_only_data_record_count + 1;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        pool->arena, pool->read_only_data_record_count,
        iree_max(minimum_capacity, 8u), sizeof(*pool->read_only_data_records),
        &pool->read_only_data_record_capacity,
        (void**)&pool->read_only_data_records));
  }

  uint8_t* copied_data = NULL;
  if (contents.data_length) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(pool->arena, contents.data_length,
                                             (void**)&copied_data));
    memcpy(copied_data, contents.data, contents.data_length);
  }
  contents_attr.bytes = copied_data;
  const uint32_t record_index = (uint32_t)pool->read_only_data_record_count++;
  pool->read_only_data_records[record_index] = (loom_rodata_record_t){
      .symbol = loom_symbol_ref_null(),
      .location = location,
      .contents = contents_attr,
      .hash = hash,
      .minimum_alignment_log2 = minimum_alignment_log2,
  };
  loom_intern_table_insert(&pool->read_only_data_index, slot, hash,
                           record_index);
  *out_id = record_index;
  return iree_ok_status();
}

iree_status_t loom_rodata_pool_reference(loom_rodata_pool_t* pool,
                                         loom_module_t* module,
                                         loom_rodata_id_t id,
                                         loom_symbol_ref_t* out_symbol) {
  loom_rodata_record_t* record = &pool->read_only_data_records[id];
  if (!loom_symbol_ref_is_valid(record->symbol)) {
    IREE_RETURN_IF_ERROR(loom_rodata_pool_reserve_symbol(
        pool, module, record->hash, &record->symbol));
  }
  *out_symbol = record->symbol;
  return iree_ok_status();
}

iree_status_t loom_rodata_pool_finalize(loom_rodata_pool_t* pool,
                                        loom_module_t* module) {
  if (pool->finalized) {
    return iree_ok_status();
  }
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < pool->read_only_data_record_count;
       ++i) {
    const loom_rodata_record_t* record = &pool->read_only_data_records[i];
    loom_symbol_ref_t symbol = loom_symbol_ref_null();
    status =
        loom_rodata_pool_reference(pool, module, (loom_rodata_id_t)i, &symbol);
    if (iree_status_is_ok(status)) {
      loom_op_t* definition = NULL;
      status = loom_global_rodata_def_build(
          &builder, LOOM_GLOBAL_RODATA_DEF_BUILD_FLAG_HAS_ALIGNMENT,
          /*visibility=*/0, /*linkage=*/0, symbol,
          (int64_t)(UINT64_C(1) << record->minimum_alignment_log2),
          loom_symbol_ref_array_empty(), loom_attr_as_bytes(record->contents),
          record->location, &definition);
      if (iree_status_is_ok(status) && loom_attr_is_utf8(record->contents)) {
        loom_attribute_t contents =
            loom_global_rodata_def_contents_attr(definition);
        contents.flags = record->contents.flags;
        status =
            loom_global_rodata_def_set_contents(module, definition, contents);
      }
    }
  }
  pool->finalized = iree_status_is_ok(status);
  return status;
}
