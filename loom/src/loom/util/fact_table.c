// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_table.h"

#include <stdint.h>
#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

//===----------------------------------------------------------------------===//
// Capacity management
//===----------------------------------------------------------------------===//

typedef enum loom_value_fact_extension_kind_e {
  LOOM_VALUE_FACT_EXTENSION_UNIFORM_ELEMENT = 1,
  LOOM_VALUE_FACT_EXTENSION_VECTOR_IOTA = 2,
  LOOM_VALUE_FACT_EXTENSION_VECTOR_PREFIX_MASK = 3,
  LOOM_VALUE_FACT_EXTENSION_SMALL_STATIC_LANES = 4,
  LOOM_VALUE_FACT_EXTENSION_BUFFER_REFERENCE = 5,
  LOOM_VALUE_FACT_EXTENSION_VIEW_REFERENCE = 6,
  LOOM_VALUE_FACT_EXTENSION_ENCODING_SUMMARY = 7,
} loom_value_fact_extension_kind_t;

struct loom_value_fact_extension_entry_t {
  // Content hash for the extension kind and payload.
  uint32_t content_hash;
  // Collision-chain next extension ID, or zero for the end of the chain.
  loom_value_fact_extension_id_t next_id;
  // Extension payload kind.
  loom_value_fact_extension_kind_t kind;
  // Reserved for alignment and future flags. Always zero.
  uint32_t reserved;

  // Payload selected by kind.
  union {
    // Uniform-element vector payload.
    loom_value_fact_uniform_element_t uniform_element;
    // Small static vector lane payload.
    loom_value_fact_small_static_lanes_t small_static_lanes;
    // Vector iota payload.
    loom_value_fact_vector_iota_t vector_iota;
    // Vector prefix-mask payload.
    loom_value_fact_vector_prefix_mask_t vector_prefix_mask;
    // SSA encoding summary payload.
    loom_value_fact_encoding_summary_t encoding_summary;
    // Buffer storage-root payload.
    loom_value_fact_buffer_reference_t buffer_reference;
    // Typed view projection payload.
    loom_value_fact_view_reference_t view_reference;
  } payload;
};

static iree_status_t loom_value_fact_table_ensure_capacity(
    loom_value_fact_table_t* table, iree_host_size_t minimum_count) {
  if (minimum_count <= table->capacity) return iree_ok_status();

  iree_host_size_t new_capacity = table->capacity > 0 ? table->capacity : 256;
  while (new_capacity < minimum_count) {
    if (new_capacity > SIZE_MAX / 2) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "fact table capacity overflow");
    }
    new_capacity *= 2;
  }

  loom_value_facts_t* new_entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(table->arena, new_capacity,
                                                 sizeof(loom_value_facts_t),
                                                 (void**)&new_entries));

  // Zero-fill: known_divisor == 0 is the "undefined" sentinel.
  memset(new_entries, 0, new_capacity * sizeof(loom_value_facts_t));
  if (table->count > 0) {
    memcpy(new_entries, table->entries,
           table->count * sizeof(loom_value_facts_t));
  }

  table->entries = new_entries;
  table->capacity = new_capacity;
  return iree_ok_status();
}

static uint32_t loom_value_fact_hash_bytes(const void* data,
                                           iree_host_size_t length,
                                           uint32_t hash) {
  const uint8_t* bytes = (const uint8_t*)data;
  for (iree_host_size_t i = 0; i < length; ++i) {
    hash ^= bytes[i];
    hash *= 16777619u;
  }
  return hash;
}

static uint32_t loom_value_fact_hash_u32(uint32_t value, uint32_t hash) {
  return loom_value_fact_hash_bytes(&value, sizeof(value), hash);
}

static uint32_t loom_value_fact_hash_host_size(iree_host_size_t value,
                                               uint32_t hash) {
  return loom_value_fact_hash_bytes(&value, sizeof(value), hash);
}

static uint32_t loom_value_fact_hash_i64(int64_t value, uint32_t hash) {
  return loom_value_fact_hash_bytes(&value, sizeof(value), hash);
}

static uint32_t loom_value_fact_hash_u64(uint64_t value, uint32_t hash) {
  return loom_value_fact_hash_bytes(&value, sizeof(value), hash);
}

static uint32_t loom_value_fact_hash_facts(loom_value_facts_t facts,
                                           uint32_t hash) {
  return loom_value_fact_hash_bytes(&facts, sizeof(facts), hash);
}

static uint32_t loom_value_fact_hash_address_layout(
    loom_value_fact_address_layout_t layout, uint32_t hash) {
  hash = loom_value_fact_hash_u32((uint32_t)layout.kind, hash);
  hash = loom_value_fact_hash_u32(layout.rank, hash);
  if (layout.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED &&
      layout.rank > 0 && layout.strides) {
    hash = loom_value_fact_hash_bytes(
        layout.strides, layout.rank * sizeof(loom_value_facts_t), hash);
  }
  return hash;
}

static uint32_t loom_value_fact_hash_matrix_storage_schema(
    loom_value_fact_matrix_storage_schema_t schema, uint32_t hash) {
  hash = loom_value_fact_hash_u32((uint32_t)schema.format, hash);
  hash = loom_value_fact_hash_u32((uint32_t)schema.scale_kind, hash);
  hash = loom_value_fact_hash_u32((uint32_t)schema.scale_format, hash);
  hash = loom_value_fact_hash_u32((uint32_t)schema.scale_placement, hash);
  hash = loom_value_fact_hash_u32((uint32_t)schema.scale_conversion, hash);
  hash = loom_value_fact_hash_u32(schema.packed_register_count, hash);
  hash = loom_value_fact_hash_u32(schema.packed_element_count, hash);
  return loom_value_fact_hash_u32(schema.zero_scale_fallback ? 1u : 0u, hash);
}

static uint32_t loom_value_fact_hash_storage_schema(
    loom_value_fact_storage_schema_t schema, uint32_t hash) {
  hash = loom_value_fact_hash_u32(schema.static_spec_encoding_id, hash);
  return loom_value_fact_hash_matrix_storage_schema(schema.matrix, hash);
}

static uint32_t loom_value_fact_hash_encoding_summary(
    loom_value_fact_encoding_summary_t summary, uint32_t hash) {
  hash = loom_value_fact_hash_u32((uint32_t)summary.role, hash);
  hash = loom_value_fact_hash_u32(summary.static_spec_encoding_id, hash);
  hash = loom_value_fact_hash_address_layout(summary.address_layout, hash);
  return loom_value_fact_hash_storage_schema(summary.storage_schema, hash);
}

static uint32_t loom_value_fact_hash_buffer_reference(
    loom_value_fact_buffer_reference_t reference, uint32_t hash) {
  hash = loom_value_fact_hash_facts(reference.maximum_byte_extent, hash);
  hash = loom_value_fact_hash_u64(reference.minimum_alignment, hash);
  hash = loom_value_fact_hash_u32((uint32_t)reference.memory_space, hash);
  hash = loom_value_fact_hash_u32(reference.root_value_id, hash);
  return loom_value_fact_hash_u32(reference.nullability, hash);
}

static uint32_t loom_value_fact_hash_view_reference(
    loom_value_fact_view_reference_t reference, uint32_t hash) {
  hash = loom_value_fact_hash_facts(reference.base_byte_offset, hash);
  hash = loom_value_fact_hash_facts(reference.footprint_byte_length, hash);
  hash = loom_value_fact_hash_u64(reference.minimum_alignment, hash);
  hash = loom_value_fact_hash_u64(reference.root_minimum_alignment, hash);
  hash = loom_value_fact_hash_i64(reference.static_element_byte_count, hash);
  hash = loom_value_fact_hash_u32((uint32_t)reference.memory_space, hash);
  hash = loom_value_fact_hash_u32(reference.root_value_id, hash);
  return loom_value_fact_hash_u32(reference.nullability, hash);
}

static bool loom_value_fact_buffer_reference_equal(
    loom_value_fact_buffer_reference_t lhs,
    loom_value_fact_buffer_reference_t rhs) {
  return loom_value_facts_equal(lhs.maximum_byte_extent,
                                rhs.maximum_byte_extent) &&
         lhs.minimum_alignment == rhs.minimum_alignment &&
         lhs.memory_space == rhs.memory_space &&
         lhs.root_value_id == rhs.root_value_id &&
         lhs.nullability == rhs.nullability;
}

static bool loom_value_fact_view_reference_equal(
    loom_value_fact_view_reference_t lhs,
    loom_value_fact_view_reference_t rhs) {
  return loom_value_facts_equal(lhs.base_byte_offset, rhs.base_byte_offset) &&
         loom_value_facts_equal(lhs.footprint_byte_length,
                                rhs.footprint_byte_length) &&
         lhs.minimum_alignment == rhs.minimum_alignment &&
         lhs.root_minimum_alignment == rhs.root_minimum_alignment &&
         lhs.static_element_byte_count == rhs.static_element_byte_count &&
         lhs.memory_space == rhs.memory_space &&
         lhs.root_value_id == rhs.root_value_id &&
         lhs.nullability == rhs.nullability;
}

static bool loom_value_fact_address_layout_equal(
    loom_value_fact_address_layout_t lhs,
    loom_value_fact_address_layout_t rhs) {
  if (lhs.kind != rhs.kind || lhs.rank != rhs.rank) return false;
  if (lhs.kind != LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED) return true;
  if (lhs.rank == 0) return true;
  if (!lhs.strides || !rhs.strides) return lhs.strides == rhs.strides;
  return memcmp(lhs.strides, rhs.strides,
                lhs.rank * sizeof(loom_value_facts_t)) == 0;
}

static bool loom_value_fact_matrix_storage_schema_equal(
    loom_value_fact_matrix_storage_schema_t lhs,
    loom_value_fact_matrix_storage_schema_t rhs) {
  return lhs.format == rhs.format && lhs.scale_kind == rhs.scale_kind &&
         lhs.scale_format == rhs.scale_format &&
         lhs.scale_placement == rhs.scale_placement &&
         lhs.scale_conversion == rhs.scale_conversion &&
         lhs.packed_register_count == rhs.packed_register_count &&
         lhs.packed_element_count == rhs.packed_element_count &&
         lhs.zero_scale_fallback == rhs.zero_scale_fallback;
}

static bool loom_value_fact_matrix_storage_schema_is_unknown(
    loom_value_fact_matrix_storage_schema_t schema) {
  return schema.format == LOOM_VALUE_FACT_MATRIX_FORMAT_UNKNOWN &&
         schema.scale_kind == LOOM_VALUE_FACT_MATRIX_SCALE_UNKNOWN &&
         schema.scale_format == LOOM_VALUE_FACT_MATRIX_SCALE_FORMAT_UNKNOWN &&
         schema.scale_placement ==
             LOOM_VALUE_FACT_MATRIX_SCALE_PLACEMENT_UNKNOWN &&
         schema.scale_conversion ==
             LOOM_VALUE_FACT_MATRIX_SCALE_CONVERSION_UNKNOWN &&
         schema.packed_register_count == 0 &&
         schema.packed_element_count == 0 && !schema.zero_scale_fallback;
}

static bool loom_value_fact_storage_schema_equal(
    loom_value_fact_storage_schema_t lhs,
    loom_value_fact_storage_schema_t rhs) {
  return lhs.static_spec_encoding_id == rhs.static_spec_encoding_id &&
         loom_value_fact_matrix_storage_schema_equal(lhs.matrix, rhs.matrix);
}

static bool loom_value_fact_encoding_summary_equal(
    loom_value_fact_encoding_summary_t lhs,
    loom_value_fact_encoding_summary_t rhs) {
  return lhs.role == rhs.role &&
         lhs.static_spec_encoding_id == rhs.static_spec_encoding_id &&
         loom_value_fact_address_layout_equal(lhs.address_layout,
                                              rhs.address_layout) &&
         loom_value_fact_storage_schema_equal(lhs.storage_schema,
                                              rhs.storage_schema);
}

static uint32_t loom_value_fact_extension_hash(
    const loom_value_fact_extension_entry_t* entry) {
  uint32_t hash = 2166136261u;
  hash = loom_value_fact_hash_u32((uint32_t)entry->kind, hash);
  switch (entry->kind) {
    case LOOM_VALUE_FACT_EXTENSION_UNIFORM_ELEMENT:
      return loom_value_fact_hash_bytes(&entry->payload.uniform_element,
                                        sizeof(entry->payload.uniform_element),
                                        hash);
    case LOOM_VALUE_FACT_EXTENSION_SMALL_STATIC_LANES:
      hash = loom_value_fact_hash_host_size(
          entry->payload.small_static_lanes.count, hash);
      return loom_value_fact_hash_bytes(
          entry->payload.small_static_lanes.lanes,
          entry->payload.small_static_lanes.count * sizeof(loom_value_facts_t),
          hash);
    case LOOM_VALUE_FACT_EXTENSION_VECTOR_IOTA:
      return loom_value_fact_hash_bytes(&entry->payload.vector_iota,
                                        sizeof(entry->payload.vector_iota),
                                        hash);
    case LOOM_VALUE_FACT_EXTENSION_VECTOR_PREFIX_MASK:
      return loom_value_fact_hash_bytes(
          &entry->payload.vector_prefix_mask,
          sizeof(entry->payload.vector_prefix_mask), hash);
    case LOOM_VALUE_FACT_EXTENSION_ENCODING_SUMMARY:
      return loom_value_fact_hash_encoding_summary(
          entry->payload.encoding_summary, hash);
    case LOOM_VALUE_FACT_EXTENSION_BUFFER_REFERENCE:
      return loom_value_fact_hash_buffer_reference(
          entry->payload.buffer_reference, hash);
    case LOOM_VALUE_FACT_EXTENSION_VIEW_REFERENCE:
      return loom_value_fact_hash_view_reference(entry->payload.view_reference,
                                                 hash);
    default:
      return hash;
  }
}

static bool loom_value_fact_extension_content_equal(
    const loom_value_fact_extension_entry_t* lhs,
    const loom_value_fact_extension_entry_t* rhs) {
  if (lhs->kind != rhs->kind) return false;
  switch (lhs->kind) {
    case LOOM_VALUE_FACT_EXTENSION_UNIFORM_ELEMENT:
      return memcmp(&lhs->payload.uniform_element,
                    &rhs->payload.uniform_element,
                    sizeof(lhs->payload.uniform_element)) == 0;
    case LOOM_VALUE_FACT_EXTENSION_SMALL_STATIC_LANES:
      if (lhs->payload.small_static_lanes.count !=
          rhs->payload.small_static_lanes.count) {
        return false;
      }
      if (lhs->payload.small_static_lanes.count == 0) return true;
      return memcmp(lhs->payload.small_static_lanes.lanes,
                    rhs->payload.small_static_lanes.lanes,
                    lhs->payload.small_static_lanes.count *
                        sizeof(loom_value_facts_t)) == 0;
    case LOOM_VALUE_FACT_EXTENSION_VECTOR_IOTA:
      return memcmp(&lhs->payload.vector_iota, &rhs->payload.vector_iota,
                    sizeof(lhs->payload.vector_iota)) == 0;
    case LOOM_VALUE_FACT_EXTENSION_VECTOR_PREFIX_MASK:
      return memcmp(&lhs->payload.vector_prefix_mask,
                    &rhs->payload.vector_prefix_mask,
                    sizeof(lhs->payload.vector_prefix_mask)) == 0;
    case LOOM_VALUE_FACT_EXTENSION_ENCODING_SUMMARY:
      return loom_value_fact_encoding_summary_equal(
          lhs->payload.encoding_summary, rhs->payload.encoding_summary);
    case LOOM_VALUE_FACT_EXTENSION_BUFFER_REFERENCE:
      return loom_value_fact_buffer_reference_equal(
          lhs->payload.buffer_reference, rhs->payload.buffer_reference);
    case LOOM_VALUE_FACT_EXTENSION_VIEW_REFERENCE:
      return loom_value_fact_view_reference_equal(lhs->payload.view_reference,
                                                  rhs->payload.view_reference);
    default:
      return false;
  }
}

static iree_status_t loom_value_fact_table_clone_fact_array(
    loom_value_fact_table_t* table, const loom_value_facts_t* facts,
    iree_host_size_t count, const loom_value_facts_t** out_facts) {
  *out_facts = NULL;
  if (count == 0) return iree_ok_status();
  if (!facts) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "fact array pointer required");
  }
  loom_value_facts_t* cloned_facts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      table->arena, count, sizeof(loom_value_facts_t), (void**)&cloned_facts));
  memcpy(cloned_facts, facts, count * sizeof(loom_value_facts_t));
  *out_facts = cloned_facts;
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_materialize_extension_payload(
    loom_value_fact_table_t* table, loom_value_fact_extension_entry_t* entry) {
  if (entry->kind == LOOM_VALUE_FACT_EXTENSION_SMALL_STATIC_LANES) {
    const loom_value_facts_t* lanes = NULL;
    IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact_array(
        table, entry->payload.small_static_lanes.lanes,
        entry->payload.small_static_lanes.count, &lanes));
    entry->payload.small_static_lanes.lanes = lanes;
    return iree_ok_status();
  }
  if (entry->kind == LOOM_VALUE_FACT_EXTENSION_ENCODING_SUMMARY &&
      entry->payload.encoding_summary.address_layout.kind ==
          LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED) {
    loom_value_fact_address_layout_t* layout =
        &entry->payload.encoding_summary.address_layout;
    const loom_value_facts_t* strides = NULL;
    IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact_array(
        table, layout->strides, layout->rank, &strides));
    layout->strides = strides;
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_ensure_extension_capacity(
    loom_value_fact_table_t* table, iree_host_size_t minimum_count) {
  if (minimum_count <= table->extensions.capacity) return iree_ok_status();

  iree_host_size_t new_capacity =
      table->extensions.capacity > 0 ? table->extensions.capacity * 2 : 64;
  while (new_capacity < minimum_count) {
    if (new_capacity > SIZE_MAX / 2) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "fact extension capacity overflow");
    }
    new_capacity *= 2;
  }

  loom_value_fact_extension_entry_t* new_entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      table->arena, new_capacity, sizeof(loom_value_fact_extension_entry_t),
      (void**)&new_entries));
  memset(new_entries, 0,
         new_capacity * sizeof(loom_value_fact_extension_entry_t));
  if (table->extensions.count > 0) {
    memcpy(new_entries, table->extensions.entries,
           table->extensions.count * sizeof(loom_value_fact_extension_entry_t));
  }
  table->extensions.entries = new_entries;
  table->extensions.capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_rehash_extensions(
    loom_value_fact_table_t* table, iree_host_size_t new_bucket_count) {
  loom_value_fact_extension_id_t* new_buckets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      table->arena, new_bucket_count, sizeof(loom_value_fact_extension_id_t),
      (void**)&new_buckets));
  memset(new_buckets, 0,
         new_bucket_count * sizeof(loom_value_fact_extension_id_t));
  for (iree_host_size_t i = 0; i < table->extensions.count; ++i) {
    loom_value_fact_extension_entry_t* entry = &table->extensions.entries[i];
    loom_value_fact_extension_id_t id = (loom_value_fact_extension_id_t)(i + 1);
    iree_host_size_t bucket_index =
        (iree_host_size_t)entry->content_hash & (new_bucket_count - 1);
    entry->next_id = new_buckets[bucket_index];
    new_buckets[bucket_index] = id;
  }
  table->extensions.buckets = new_buckets;
  table->extensions.bucket_count = new_bucket_count;
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_ensure_extension_buckets(
    loom_value_fact_table_t* table, iree_host_size_t minimum_count) {
  iree_host_size_t new_bucket_count = table->extensions.bucket_count;
  if (new_bucket_count == 0) new_bucket_count = 64;
  while (minimum_count > new_bucket_count - new_bucket_count / 4) {
    if (new_bucket_count > SIZE_MAX / 2) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "fact extension bucket capacity overflow");
    }
    new_bucket_count *= 2;
  }
  if (new_bucket_count == table->extensions.bucket_count) {
    return iree_ok_status();
  }
  return loom_value_fact_table_rehash_extensions(table, new_bucket_count);
}

static iree_status_t loom_value_fact_context_require_table(
    loom_fact_context_t* context, loom_value_fact_table_t** out_table) {
  if (!context || !context->table || !context->table->arena) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "initialized fact context required");
  }
  *out_table = context->table;
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_intern_extension_impl(
    loom_value_fact_table_t* table,
    const loom_value_fact_extension_entry_t* candidate,
    bool materialize_payload, loom_value_fact_extension_id_t* out_id) {
  loom_value_fact_extension_entry_t entry = *candidate;
  entry.content_hash = loom_value_fact_extension_hash(&entry);
  entry.next_id = LOOM_VALUE_FACT_EXTENSION_ID_NONE;

  IREE_RETURN_IF_ERROR(loom_value_fact_table_ensure_extension_buckets(
      table, table->extensions.count));
  iree_host_size_t bucket_index = (iree_host_size_t)entry.content_hash &
                                  (table->extensions.bucket_count - 1);
  for (loom_value_fact_extension_id_t id =
           table->extensions.buckets[bucket_index];
       id != LOOM_VALUE_FACT_EXTENSION_ID_NONE;
       id = table->extensions.entries[id - 1].next_id) {
    const loom_value_fact_extension_entry_t* existing =
        &table->extensions.entries[id - 1];
    if (existing->content_hash == entry.content_hash &&
        loom_value_fact_extension_content_equal(existing, &entry)) {
      *out_id = id;
      return iree_ok_status();
    }
  }

  iree_host_size_t new_count = table->extensions.count + 1;
  if (new_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "fact extension ID capacity exceeded");
  }
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_ensure_extension_capacity(table, new_count));
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_ensure_extension_buckets(table, new_count));
  bucket_index = (iree_host_size_t)entry.content_hash &
                 (table->extensions.bucket_count - 1);
  loom_value_fact_extension_id_t id = (loom_value_fact_extension_id_t)new_count;
  if (materialize_payload) {
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_materialize_extension_payload(table, &entry));
  }
  entry.next_id = table->extensions.buckets[bucket_index];
  table->extensions.entries[table->extensions.count] = entry;
  table->extensions.buckets[bucket_index] = id;
  table->extensions.count = new_count;
  *out_id = id;
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_intern_extension(
    loom_value_fact_table_t* table,
    const loom_value_fact_extension_entry_t* candidate,
    loom_value_fact_extension_id_t* out_id) {
  return loom_value_fact_table_intern_extension_impl(table, candidate, true,
                                                     out_id);
}

static iree_status_t loom_value_facts_make_extension(
    loom_fact_context_t* context,
    const loom_value_fact_extension_entry_t* candidate,
    loom_value_facts_t* out) {
  if (!out) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "output fact pointer required");
  }
  loom_value_fact_table_t* table = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_context_require_table(context, &table));
  loom_value_fact_extension_id_t extension_id =
      LOOM_VALUE_FACT_EXTENSION_ID_NONE;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_intern_extension(table, candidate, &extension_id));
  *out = loom_value_facts_unknown();
  out->extension_id = extension_id;
  return iree_ok_status();
}

static const loom_value_fact_extension_entry_t*
loom_value_facts_lookup_extension(const loom_fact_context_t* context,
                                  loom_value_facts_t facts) {
  if (!context || !context->table ||
      facts.extension_id == LOOM_VALUE_FACT_EXTENSION_ID_NONE) {
    return NULL;
  }
  const loom_value_fact_table_t* table = context->table;
  if (facts.extension_id > table->extensions.count) return NULL;
  return &table->extensions.entries[facts.extension_id - 1];
}

//===----------------------------------------------------------------------===//
// Cross-table comparison
//===----------------------------------------------------------------------===//

static bool loom_value_fact_table_scalar_fields_equal(loom_value_facts_t lhs,
                                                      loom_value_facts_t rhs) {
  lhs.extension_id = LOOM_VALUE_FACT_EXTENSION_ID_NONE;
  rhs.extension_id = LOOM_VALUE_FACT_EXTENSION_ID_NONE;
  return loom_value_facts_equal(lhs, rhs);
}

static bool loom_value_fact_table_fact_array_equal(
    const loom_value_fact_table_t* lhs_table, const loom_value_facts_t* lhs,
    const loom_value_fact_table_t* rhs_table, const loom_value_facts_t* rhs,
    iree_host_size_t count) {
  if (count == 0) return true;
  if (!lhs || !rhs) return lhs == rhs;
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (!loom_value_fact_table_facts_equal(lhs_table, lhs[i], rhs_table,
                                           rhs[i])) {
      return false;
    }
  }
  return true;
}

static bool loom_value_fact_table_address_layout_equal(
    const loom_value_fact_table_t* lhs_table,
    loom_value_fact_address_layout_t lhs,
    const loom_value_fact_table_t* rhs_table,
    loom_value_fact_address_layout_t rhs) {
  if (lhs.kind != rhs.kind || lhs.rank != rhs.rank) return false;
  if (lhs.kind != LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED) return true;
  return loom_value_fact_table_fact_array_equal(
      lhs_table, lhs.strides, rhs_table, rhs.strides, lhs.rank);
}

static bool loom_value_fact_table_encoding_summary_equal(
    const loom_value_fact_table_t* lhs_table,
    loom_value_fact_encoding_summary_t lhs,
    const loom_value_fact_table_t* rhs_table,
    loom_value_fact_encoding_summary_t rhs) {
  return lhs.role == rhs.role &&
         lhs.static_spec_encoding_id == rhs.static_spec_encoding_id &&
         loom_value_fact_table_address_layout_equal(
             lhs_table, lhs.address_layout, rhs_table, rhs.address_layout) &&
         loom_value_fact_storage_schema_equal(lhs.storage_schema,
                                              rhs.storage_schema);
}

static bool loom_value_fact_table_buffer_reference_equal(
    const loom_value_fact_table_t* lhs_table,
    loom_value_fact_buffer_reference_t lhs,
    const loom_value_fact_table_t* rhs_table,
    loom_value_fact_buffer_reference_t rhs) {
  return loom_value_fact_table_facts_equal(lhs_table, lhs.maximum_byte_extent,
                                           rhs_table,
                                           rhs.maximum_byte_extent) &&
         lhs.minimum_alignment == rhs.minimum_alignment &&
         lhs.memory_space == rhs.memory_space &&
         lhs.root_value_id == rhs.root_value_id &&
         lhs.nullability == rhs.nullability;
}

static bool loom_value_fact_table_view_reference_equal(
    const loom_value_fact_table_t* lhs_table,
    loom_value_fact_view_reference_t lhs,
    const loom_value_fact_table_t* rhs_table,
    loom_value_fact_view_reference_t rhs) {
  return loom_value_fact_table_facts_equal(lhs_table, lhs.base_byte_offset,
                                           rhs_table, rhs.base_byte_offset) &&
         loom_value_fact_table_facts_equal(lhs_table, lhs.footprint_byte_length,
                                           rhs_table,
                                           rhs.footprint_byte_length) &&
         lhs.minimum_alignment == rhs.minimum_alignment &&
         lhs.root_minimum_alignment == rhs.root_minimum_alignment &&
         lhs.static_element_byte_count == rhs.static_element_byte_count &&
         lhs.memory_space == rhs.memory_space &&
         lhs.root_value_id == rhs.root_value_id &&
         lhs.nullability == rhs.nullability;
}

static bool loom_value_fact_table_extension_entries_equal(
    const loom_value_fact_table_t* lhs_table,
    const loom_value_fact_extension_entry_t* lhs,
    const loom_value_fact_table_t* rhs_table,
    const loom_value_fact_extension_entry_t* rhs) {
  if (!lhs || !rhs || lhs->kind != rhs->kind) return false;
  switch (lhs->kind) {
    case LOOM_VALUE_FACT_EXTENSION_UNIFORM_ELEMENT:
      return loom_value_fact_table_facts_equal(
          lhs_table, lhs->payload.uniform_element.element, rhs_table,
          rhs->payload.uniform_element.element);
    case LOOM_VALUE_FACT_EXTENSION_SMALL_STATIC_LANES:
      if (lhs->payload.small_static_lanes.count !=
          rhs->payload.small_static_lanes.count) {
        return false;
      }
      return loom_value_fact_table_fact_array_equal(
          lhs_table, lhs->payload.small_static_lanes.lanes, rhs_table,
          rhs->payload.small_static_lanes.lanes,
          lhs->payload.small_static_lanes.count);
    case LOOM_VALUE_FACT_EXTENSION_VECTOR_IOTA:
      return loom_value_fact_table_facts_equal(
                 lhs_table, lhs->payload.vector_iota.base, rhs_table,
                 rhs->payload.vector_iota.base) &&
             loom_value_fact_table_facts_equal(
                 lhs_table, lhs->payload.vector_iota.step, rhs_table,
                 rhs->payload.vector_iota.step);
    case LOOM_VALUE_FACT_EXTENSION_VECTOR_PREFIX_MASK:
      return loom_value_fact_table_facts_equal(
                 lhs_table, lhs->payload.vector_prefix_mask.lower_bound,
                 rhs_table, rhs->payload.vector_prefix_mask.lower_bound) &&
             loom_value_fact_table_facts_equal(
                 lhs_table, lhs->payload.vector_prefix_mask.upper_bound,
                 rhs_table, rhs->payload.vector_prefix_mask.upper_bound) &&
             loom_value_fact_table_facts_equal(
                 lhs_table, lhs->payload.vector_prefix_mask.step, rhs_table,
                 rhs->payload.vector_prefix_mask.step);
    case LOOM_VALUE_FACT_EXTENSION_ENCODING_SUMMARY:
      return loom_value_fact_table_encoding_summary_equal(
          lhs_table, lhs->payload.encoding_summary, rhs_table,
          rhs->payload.encoding_summary);
    case LOOM_VALUE_FACT_EXTENSION_BUFFER_REFERENCE:
      return loom_value_fact_table_buffer_reference_equal(
          lhs_table, lhs->payload.buffer_reference, rhs_table,
          rhs->payload.buffer_reference);
    case LOOM_VALUE_FACT_EXTENSION_VIEW_REFERENCE:
      return loom_value_fact_table_view_reference_equal(
          lhs_table, lhs->payload.view_reference, rhs_table,
          rhs->payload.view_reference);
    default:
      return false;
  }
}

bool loom_value_fact_table_extensions_equal(
    const loom_value_fact_table_t* lhs_table, loom_value_facts_t lhs,
    const loom_value_fact_table_t* rhs_table, loom_value_facts_t rhs) {
  if (lhs.extension_id == LOOM_VALUE_FACT_EXTENSION_ID_NONE ||
      rhs.extension_id == LOOM_VALUE_FACT_EXTENSION_ID_NONE) {
    return lhs.extension_id == rhs.extension_id;
  }
  const loom_value_fact_extension_entry_t* lhs_entry =
      loom_value_facts_lookup_extension(lhs_table ? &lhs_table->context : NULL,
                                        lhs);
  const loom_value_fact_extension_entry_t* rhs_entry =
      loom_value_facts_lookup_extension(rhs_table ? &rhs_table->context : NULL,
                                        rhs);
  return loom_value_fact_table_extension_entries_equal(lhs_table, lhs_entry,
                                                       rhs_table, rhs_entry);
}

bool loom_value_fact_table_facts_equal(const loom_value_fact_table_t* lhs_table,
                                       loom_value_facts_t lhs,
                                       const loom_value_fact_table_t* rhs_table,
                                       loom_value_facts_t rhs) {
  return loom_value_fact_table_scalar_fields_equal(lhs, rhs) &&
         loom_value_fact_table_extensions_equal(lhs_table, lhs, rhs_table, rhs);
}

//===----------------------------------------------------------------------===//
// Scratch buffers
//===----------------------------------------------------------------------===//

iree_status_t loom_value_fact_table_facts_scratch(
    loom_value_fact_table_t* table, iree_host_size_t count,
    loom_value_facts_t** out) {
  if (count <= table->scratch.facts.capacity) {
    *out = table->scratch.facts.values;
    return iree_ok_status();
  }
  loom_value_facts_t* new_scratch = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      table->arena, count, sizeof(loom_value_facts_t), (void**)&new_scratch));
  table->scratch.facts.values = new_scratch;
  table->scratch.facts.capacity = count;
  *out = new_scratch;
  return iree_ok_status();
}

iree_status_t loom_value_fact_table_value_id_scratch(
    loom_value_fact_table_t* table, iree_host_size_t count,
    loom_value_id_t** out) {
  if (count <= table->scratch.value_ids.capacity) {
    *out = table->scratch.value_ids.values;
    return iree_ok_status();
  }
  loom_value_id_t* new_scratch = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      table->arena, count, sizeof(loom_value_id_t), (void**)&new_scratch));
  table->scratch.value_ids.values = new_scratch;
  table->scratch.value_ids.capacity = count;
  *out = new_scratch;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Fact extensions
//===----------------------------------------------------------------------===//

iree_status_t loom_value_facts_make_uniform_element(
    loom_fact_context_t* context, loom_value_facts_t element,
    loom_value_facts_t* out) {
  loom_value_fact_extension_entry_t entry = {0};
  entry.kind = LOOM_VALUE_FACT_EXTENSION_UNIFORM_ELEMENT;
  entry.payload.uniform_element.element = element;
  return loom_value_facts_make_extension(context, &entry, out);
}

bool loom_value_facts_query_uniform_element(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_uniform_element_t* out) {
  const loom_value_fact_extension_entry_t* entry =
      loom_value_facts_lookup_extension(context, facts);
  if (!entry || entry->kind != LOOM_VALUE_FACT_EXTENSION_UNIFORM_ELEMENT) {
    return false;
  }
  if (out) *out = entry->payload.uniform_element;
  return true;
}

iree_status_t loom_value_facts_make_small_static_lanes(
    loom_fact_context_t* context, loom_value_fact_small_static_lanes_t lanes,
    loom_value_facts_t* out) {
  if (!out) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "output fact pointer required");
  }
  if (lanes.count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT) {
    *out = loom_value_facts_unknown();
    return iree_ok_status();
  }
  if (lanes.count > 0 && !lanes.lanes) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "lane fact pointer required");
  }
  loom_value_fact_extension_entry_t entry = {0};
  entry.kind = LOOM_VALUE_FACT_EXTENSION_SMALL_STATIC_LANES;
  entry.payload.small_static_lanes = lanes;
  return loom_value_facts_make_extension(context, &entry, out);
}

bool loom_value_facts_query_small_static_lanes(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_small_static_lanes_t* out) {
  const loom_value_fact_extension_entry_t* entry =
      loom_value_facts_lookup_extension(context, facts);
  if (!entry || entry->kind != LOOM_VALUE_FACT_EXTENSION_SMALL_STATIC_LANES) {
    return false;
  }
  if (out) *out = entry->payload.small_static_lanes;
  return true;
}

iree_status_t loom_value_facts_make_vector_iota(
    loom_fact_context_t* context, loom_value_fact_vector_iota_t iota,
    loom_value_facts_t* out) {
  loom_value_fact_extension_entry_t entry = {0};
  entry.kind = LOOM_VALUE_FACT_EXTENSION_VECTOR_IOTA;
  entry.payload.vector_iota = iota;
  return loom_value_facts_make_extension(context, &entry, out);
}

bool loom_value_facts_query_vector_iota(const loom_fact_context_t* context,
                                        loom_value_facts_t facts,
                                        loom_value_fact_vector_iota_t* out) {
  const loom_value_fact_extension_entry_t* entry =
      loom_value_facts_lookup_extension(context, facts);
  if (!entry || entry->kind != LOOM_VALUE_FACT_EXTENSION_VECTOR_IOTA) {
    return false;
  }
  if (out) *out = entry->payload.vector_iota;
  return true;
}

iree_status_t loom_value_facts_make_vector_prefix_mask(
    loom_fact_context_t* context, loom_value_fact_vector_prefix_mask_t mask,
    loom_value_facts_t* out) {
  loom_value_fact_extension_entry_t entry = {0};
  entry.kind = LOOM_VALUE_FACT_EXTENSION_VECTOR_PREFIX_MASK;
  entry.payload.vector_prefix_mask = mask;
  return loom_value_facts_make_extension(context, &entry, out);
}

bool loom_value_facts_query_vector_prefix_mask(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_vector_prefix_mask_t* out) {
  const loom_value_fact_extension_entry_t* entry =
      loom_value_facts_lookup_extension(context, facts);
  if (!entry || entry->kind != LOOM_VALUE_FACT_EXTENSION_VECTOR_PREFIX_MASK) {
    return false;
  }
  if (out) *out = entry->payload.vector_prefix_mask;
  return true;
}

iree_status_t loom_value_facts_make_encoding_summary(
    loom_fact_context_t* context, loom_value_fact_encoding_summary_t summary,
    loom_value_facts_t* out) {
  if (!out) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "output fact pointer required");
  }
  if (summary.role == LOOM_ENCODING_ROLE_UNKNOWN &&
      summary.static_spec_encoding_id == 0 &&
      summary.address_layout.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_UNKNOWN &&
      summary.storage_schema.static_spec_encoding_id == 0 &&
      loom_value_fact_matrix_storage_schema_is_unknown(
          summary.storage_schema.matrix)) {
    *out = loom_value_facts_unknown();
    return iree_ok_status();
  }
  if (summary.address_layout.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED &&
      summary.address_layout.rank > 0 && !summary.address_layout.strides) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "strided layout fact pointer required");
  }
  loom_value_fact_extension_entry_t entry = {0};
  entry.kind = LOOM_VALUE_FACT_EXTENSION_ENCODING_SUMMARY;
  entry.payload.encoding_summary = summary;
  return loom_value_facts_make_extension(context, &entry, out);
}

bool loom_value_facts_query_encoding_summary(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_encoding_summary_t* out) {
  const loom_value_fact_extension_entry_t* entry =
      loom_value_facts_lookup_extension(context, facts);
  if (!entry || entry->kind != LOOM_VALUE_FACT_EXTENSION_ENCODING_SUMMARY) {
    return false;
  }
  if (out) *out = entry->payload.encoding_summary;
  return true;
}

iree_status_t loom_value_facts_make_buffer_reference(
    loom_fact_context_t* context, loom_value_fact_buffer_reference_t reference,
    loom_value_facts_t* out) {
  loom_value_fact_extension_entry_t entry = {0};
  entry.kind = LOOM_VALUE_FACT_EXTENSION_BUFFER_REFERENCE;
  entry.payload.buffer_reference = reference;
  return loom_value_facts_make_extension(context, &entry, out);
}

bool loom_value_facts_query_buffer_reference(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_buffer_reference_t* out) {
  const loom_value_fact_extension_entry_t* entry =
      loom_value_facts_lookup_extension(context, facts);
  if (!entry || entry->kind != LOOM_VALUE_FACT_EXTENSION_BUFFER_REFERENCE) {
    return false;
  }
  if (out) *out = entry->payload.buffer_reference;
  return true;
}

iree_status_t loom_value_facts_make_view_reference(
    loom_fact_context_t* context, loom_value_fact_view_reference_t reference,
    loom_value_facts_t* out) {
  loom_value_fact_extension_entry_t entry = {0};
  entry.kind = LOOM_VALUE_FACT_EXTENSION_VIEW_REFERENCE;
  entry.payload.view_reference = reference;
  return loom_value_facts_make_extension(context, &entry, out);
}

bool loom_value_facts_query_view_reference(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_view_reference_t* out) {
  const loom_value_fact_extension_entry_t* entry =
      loom_value_facts_lookup_extension(context, facts);
  if (!entry || entry->kind != LOOM_VALUE_FACT_EXTENSION_VIEW_REFERENCE) {
    return false;
  }
  if (out) *out = entry->payload.view_reference;
  return true;
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t loom_value_fact_table_initialize(
    loom_value_fact_table_t* table, iree_arena_allocator_t* arena,
    iree_host_size_t initial_capacity) {
  memset(table, 0, sizeof(*table));
  table->arena = arena;
  table->context.table = table;
  if (initial_capacity == 0) return iree_ok_status();
  return loom_value_fact_table_ensure_capacity(table, initial_capacity);
}

iree_status_t loom_value_fact_table_define(loom_value_fact_table_t* table,
                                           loom_value_id_t value_id,
                                           loom_value_facts_t facts) {
  IREE_RETURN_IF_ERROR(loom_value_fact_table_ensure_capacity(
      table, (iree_host_size_t)value_id + 1));
  table->entries[value_id] = facts;
  if ((iree_host_size_t)value_id + 1 > table->count) {
    table->count = (iree_host_size_t)value_id + 1;
  }
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_clone_fact_array_between_tables(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* source,
    const loom_value_facts_t* source_facts, iree_host_size_t count,
    const loom_value_facts_t** out_facts) {
  *out_facts = NULL;
  if (count == 0) return iree_ok_status();
  if (!source_facts) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source fact array pointer required");
  }
  loom_value_facts_t* cloned_facts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      target->arena, count, sizeof(loom_value_facts_t), (void**)&cloned_facts));
  for (iree_host_size_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
        target, source, source_facts[i], &cloned_facts[i]));
  }
  *out_facts = cloned_facts;
  return iree_ok_status();
}

iree_status_t loom_value_fact_table_clone_fact(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* source,
    loom_value_facts_t facts, loom_value_facts_t* out_facts) {
  if (!target || !source || !out_facts) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target table, source table, and output facts are required");
  }
  if (facts.extension_id == LOOM_VALUE_FACT_EXTENSION_ID_NONE) {
    *out_facts = facts;
    return iree_ok_status();
  }

  const loom_value_fact_extension_entry_t* source_entry =
      loom_value_facts_lookup_extension(&source->context, facts);
  if (!source_entry) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source fact extension id %u is invalid",
                            (unsigned)facts.extension_id);
  }

  loom_value_fact_extension_entry_t target_entry = *source_entry;
  target_entry.content_hash = 0;
  target_entry.next_id = LOOM_VALUE_FACT_EXTENSION_ID_NONE;
  switch (source_entry->kind) {
    case LOOM_VALUE_FACT_EXTENSION_UNIFORM_ELEMENT: {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
          target, source, source_entry->payload.uniform_element.element,
          &target_entry.payload.uniform_element.element));
      break;
    }
    case LOOM_VALUE_FACT_EXTENSION_SMALL_STATIC_LANES: {
      IREE_RETURN_IF_ERROR(
          loom_value_fact_table_clone_fact_array_between_tables(
              target, source, source_entry->payload.small_static_lanes.lanes,
              source_entry->payload.small_static_lanes.count,
              &target_entry.payload.small_static_lanes.lanes));
      break;
    }
    case LOOM_VALUE_FACT_EXTENSION_VECTOR_IOTA: {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
          target, source, source_entry->payload.vector_iota.base,
          &target_entry.payload.vector_iota.base));
      IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
          target, source, source_entry->payload.vector_iota.step,
          &target_entry.payload.vector_iota.step));
      break;
    }
    case LOOM_VALUE_FACT_EXTENSION_VECTOR_PREFIX_MASK: {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
          target, source, source_entry->payload.vector_prefix_mask.lower_bound,
          &target_entry.payload.vector_prefix_mask.lower_bound));
      IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
          target, source, source_entry->payload.vector_prefix_mask.upper_bound,
          &target_entry.payload.vector_prefix_mask.upper_bound));
      IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
          target, source, source_entry->payload.vector_prefix_mask.step,
          &target_entry.payload.vector_prefix_mask.step));
      break;
    }
    case LOOM_VALUE_FACT_EXTENSION_ENCODING_SUMMARY: {
      if (source_entry->payload.encoding_summary.address_layout.kind ==
          LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED) {
        loom_value_fact_address_layout_t* target_layout =
            &target_entry.payload.encoding_summary.address_layout;
        IREE_RETURN_IF_ERROR(
            loom_value_fact_table_clone_fact_array_between_tables(
                target, source,
                source_entry->payload.encoding_summary.address_layout.strides,
                source_entry->payload.encoding_summary.address_layout.rank,
                &target_layout->strides));
      }
      break;
    }
    case LOOM_VALUE_FACT_EXTENSION_BUFFER_REFERENCE: {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
          target, source,
          source_entry->payload.buffer_reference.maximum_byte_extent,
          &target_entry.payload.buffer_reference.maximum_byte_extent));
      break;
    }
    case LOOM_VALUE_FACT_EXTENSION_VIEW_REFERENCE: {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
          target, source, source_entry->payload.view_reference.base_byte_offset,
          &target_entry.payload.view_reference.base_byte_offset));
      IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
          target, source,
          source_entry->payload.view_reference.footprint_byte_length,
          &target_entry.payload.view_reference.footprint_byte_length));
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown source fact extension kind %u",
                              (unsigned)source_entry->kind);
  }

  loom_value_fact_extension_id_t target_extension_id =
      LOOM_VALUE_FACT_EXTENSION_ID_NONE;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_intern_extension_impl(
      target, &target_entry, false, &target_extension_id));
  *out_facts = facts;
  out_facts->extension_id = target_extension_id;
  return iree_ok_status();
}

iree_status_t loom_value_fact_table_clone_defined_facts(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* source) {
  if (!target || !source) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target and source fact tables are required");
  }
  for (iree_host_size_t i = 0; i < source->count; ++i) {
    if (source->entries[i].known_divisor == 0) continue;
    loom_value_facts_t cloned_facts = loom_value_facts_unknown();
    IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
        target, source, source->entries[i], &cloned_facts));
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_define(target, (loom_value_id_t)i, cloned_facts));
  }
  return iree_ok_status();
}

static int64_t loom_value_fact_static_element_byte_count(loom_type_t type) {
  int32_t bit_count = loom_scalar_type_bitwidth(loom_type_element_type(type));
  if (bit_count <= 0 || (bit_count % 8) != 0) return -1;
  return bit_count / 8;
}

static iree_status_t loom_value_fact_table_seed_buffer_arg(
    loom_value_fact_table_t* table, loom_value_id_t value_id) {
  if (!loom_value_facts_is_unknown(
          loom_value_fact_table_lookup(table, value_id))) {
    return iree_ok_status();
  }
  loom_value_fact_buffer_reference_t reference = {
      .maximum_byte_extent = loom_value_facts_make(0, INT64_MAX, 1),
      .minimum_alignment = 1,
      .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN,
      .root_value_id = value_id,
      .nullability = LOOM_VALUE_FACT_REFERENCE_NULLABILITY_UNKNOWN,
  };
  loom_value_facts_t facts = loom_value_facts_unknown();
  IREE_RETURN_IF_ERROR(loom_value_facts_make_buffer_reference(
      &table->context, reference, &facts));
  return loom_value_fact_table_define(table, value_id, facts);
}

static iree_status_t loom_value_fact_table_seed_view_arg(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_type_t type) {
  if (!loom_value_facts_is_unknown(
          loom_value_fact_table_lookup(table, value_id))) {
    return iree_ok_status();
  }
  loom_value_fact_view_reference_t reference = {
      .base_byte_offset = loom_value_facts_exact_i64(0),
      .footprint_byte_length = loom_value_facts_make(0, INT64_MAX, 1),
      .minimum_alignment = 1,
      .root_minimum_alignment = 1,
      .static_element_byte_count =
          loom_value_fact_static_element_byte_count(type),
      .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN,
      .root_value_id = value_id,
      .nullability = LOOM_VALUE_FACT_REFERENCE_NULLABILITY_UNKNOWN,
  };
  loom_value_facts_t facts = loom_value_facts_unknown();
  IREE_RETURN_IF_ERROR(
      loom_value_facts_make_view_reference(&table->context, reference, &facts));
  return loom_value_fact_table_define(table, value_id, facts);
}

static iree_status_t loom_value_fact_table_define_if_changed(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_facts_t facts, bool* out_changed) {
  loom_value_facts_t old_facts = loom_value_fact_table_lookup(table, value_id);
  if (out_changed && !loom_value_facts_equal(old_facts, facts)) {
    *out_changed = true;
  }
  return loom_value_fact_table_define(table, value_id, facts);
}

static bool loom_value_fact_table_find_result(const loom_value_id_t* results,
                                              uint16_t result_count,
                                              loom_value_id_t value_id,
                                              uint16_t* out_result_index) {
  if (!results) return false;
  for (uint16_t i = 0; i < result_count; ++i) {
    if (results[i] != value_id) continue;
    *out_result_index = i;
    return true;
  }
  return false;
}

static bool loom_value_fact_table_dynamic_extent_type_supported(
    loom_type_t type) {
  if (!loom_type_is_scalar(type)) return false;
  loom_scalar_type_t scalar_type = loom_type_element_type(type);
  return scalar_type == LOOM_SCALAR_TYPE_INDEX ||
         scalar_type == LOOM_SCALAR_TYPE_OFFSET ||
         loom_scalar_type_is_integer(scalar_type);
}

static iree_status_t loom_value_fact_table_seed_dynamic_extent(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t value_id, const loom_value_id_t* result_ids,
    uint16_t result_count, loom_value_facts_t* result_facts,
    bool* out_changed) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return iree_ok_status();
  }
  if (!loom_value_fact_table_dynamic_extent_type_supported(
          loom_module_value_type(module, value_id))) {
    return iree_ok_status();
  }

  uint16_t result_index = 0;
  if (result_facts && loom_value_fact_table_find_result(
                          result_ids, result_count, value_id, &result_index)) {
    result_facts[result_index] =
        loom_value_facts_non_negative_extent(result_facts[result_index]);
    return iree_ok_status();
  }

  loom_value_facts_t current = loom_value_fact_table_lookup(table, value_id);
  loom_value_facts_t extent = loom_value_facts_non_negative_extent(current);
  return loom_value_fact_table_define_if_changed(table, value_id, extent,
                                                 out_changed);
}

static iree_status_t loom_value_fact_table_seed_type_extent_facts(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_type_t type, const loom_value_id_t* result_ids, uint16_t result_count,
    loom_value_facts_t* result_facts, bool* out_changed) {
  if (!loom_type_is_shaped(type)) return iree_ok_status();
  uint8_t rank = loom_type_rank(type);
  for (uint8_t i = 0; i < rank; ++i) {
    if (!loom_type_dim_is_dynamic_at(type, i)) continue;
    IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_dynamic_extent(
        table, module, loom_type_dim_value_id_at(type, i), result_ids,
        result_count, result_facts, out_changed));
  }
  return iree_ok_status();
}

static int64_t loom_value_fact_loop_iv_base_divisor(loom_value_facts_t facts) {
  if (!loom_value_facts_is_exact(facts) || facts.range_lo == INT64_MIN) {
    return facts.known_divisor;
  }
  // Preserve gcd(0, step) so a zero lower bound keeps step divisibility.
  return facts.range_lo >= 0 ? facts.range_lo : -facts.range_lo;
}

static loom_value_facts_t loom_value_fact_counted_loop_iv_facts(
    loom_value_facts_t lower_bound, loom_value_facts_t upper_bound,
    loom_value_facts_t step) {
  if (loom_value_facts_is_float(lower_bound) ||
      loom_value_facts_is_float(upper_bound) ||
      loom_value_facts_is_float(step) || !loom_value_facts_is_positive(step)) {
    return loom_value_facts_unknown();
  }

  int64_t lower_divisor = loom_value_fact_loop_iv_base_divisor(lower_bound);
  int64_t divisor = loom_gcd_i64(lower_divisor, step.known_divisor);

  if (loom_value_facts_is_exact(lower_bound) &&
      loom_value_facts_is_exact(upper_bound) &&
      loom_value_facts_is_exact(step)) {
    if (lower_bound.range_lo >= upper_bound.range_lo) {
      return loom_value_facts_unknown();
    }
    int64_t next = 0;
    if (!loom_checked_add_i64(lower_bound.range_lo, step.range_lo, &next) ||
        next >= upper_bound.range_lo) {
      return loom_value_facts_exact_i64(lower_bound.range_lo);
    }
  }

  int64_t hi = INT64_MAX;
  if (loom_checked_sub_i64(upper_bound.range_hi, 1, &hi)) {
    if (lower_bound.range_lo <= hi) {
      return loom_value_facts_make(lower_bound.range_lo, hi, divisor);
    }
  }
  return loom_value_facts_unknown();
}

static iree_status_t loom_value_fact_table_seed_loop_iv_arg(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_block_t* block, loom_op_t* parent_op) {
  loom_loop_like_t loop = loom_loop_like_cast(module, parent_op);
  if (!loom_loop_like_isa(loop) || !loom_loop_like_has_counted_range(loop)) {
    return iree_ok_status();
  }
  if (loop.vtable->iv_block_arg_index == LOOM_BLOCK_ARG_INDEX_NONE ||
      loop.vtable->iv_block_arg_index >= block->arg_count) {
    return iree_ok_status();
  }
  loom_region_t* body = loom_loop_like_body(loop);
  if (!body || body->block_count == 0 ||
      block != loom_region_const_entry_block(body)) {
    return iree_ok_status();
  }

  loom_value_facts_t lower_bound =
      loom_value_fact_table_lookup(table, loom_loop_like_lower_bound(loop));
  loom_value_facts_t upper_bound =
      loom_value_fact_table_lookup(table, loom_loop_like_upper_bound(loop));
  loom_value_facts_t step =
      loom_value_fact_table_lookup(table, loom_loop_like_step(loop));
  loom_value_facts_t iv_facts =
      loom_value_fact_counted_loop_iv_facts(lower_bound, upper_bound, step);
  loom_value_id_t iv_id =
      loom_block_arg_id(block, loop.vtable->iv_block_arg_index);
  if (!loom_value_facts_is_unknown(
          loom_value_fact_table_lookup(table, iv_id))) {
    return iree_ok_status();
  }
  return loom_value_fact_table_define(table, iv_id, iv_facts);
}

static iree_status_t loom_value_fact_table_seed_block_args(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_block_t* block, loom_op_t* parent_op) {
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    loom_value_id_t value_id = loom_block_arg_id(block, i);
    if (value_id >= module->values.count) continue;
    loom_type_t type = loom_module_value_type(module, value_id);
    if (loom_type_is_buffer(type)) {
      IREE_RETURN_IF_ERROR(
          loom_value_fact_table_seed_buffer_arg(table, value_id));
    } else if (loom_type_is_view(type)) {
      IREE_RETURN_IF_ERROR(
          loom_value_fact_table_seed_view_arg(table, value_id, type));
    }
    IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_type_extent_facts(
        table, module, type, /*result_ids=*/NULL, /*result_count=*/0,
        /*result_facts=*/NULL, /*out_changed=*/NULL));
  }
  return loom_value_fact_table_seed_loop_iv_arg(table, module, block,
                                                parent_op);
}

//===----------------------------------------------------------------------===//
// Forward pass: compute facts for ops
//===----------------------------------------------------------------------===//

iree_status_t loom_value_fact_table_compute_op(loom_value_fact_table_t* table,
                                               const loom_module_t* module,
                                               const loom_op_t* op) {
  return loom_value_fact_table_compute_op_and_report(table, module, op, NULL);
}

iree_status_t loom_value_fact_table_compute_op_and_report(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_op_t* op, bool* out_changed) {
  if (out_changed) *out_changed = false;
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable || !vtable->infer_facts) {
    const loom_value_id_t* results = loom_op_const_results(op);
    for (uint16_t i = 0; i < op->result_count; ++i) {
      if (results[i] == LOOM_VALUE_ID_INVALID ||
          results[i] >= module->values.count) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_type_extent_facts(
          table, module, loom_module_value_type(module, results[i]),
          /*result_ids=*/NULL, /*result_count=*/0, /*result_facts=*/NULL,
          out_changed));
    }
    return iree_ok_status();
  }

  // Get scratch for operand + result facts.
  iree_host_size_t total =
      (iree_host_size_t)op->operand_count + op->result_count;
  loom_value_facts_t* scratch = NULL;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_facts_scratch(table, total, &scratch));
  loom_value_facts_t* operand_facts = scratch;
  loom_value_facts_t* result_facts = scratch + op->operand_count;

  // Gather operand facts from the table.
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    operand_facts[i] = loom_value_fact_table_lookup(table, operands[i]);
  }

  // Call the fact inference function.
  IREE_RETURN_IF_ERROR(vtable->infer_facts(&table->context, module, op,
                                           operand_facts, result_facts));

  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] == LOOM_VALUE_ID_INVALID ||
        results[i] >= module->values.count) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_type_extent_facts(
        table, module, loom_module_value_type(module, results[i]), results,
        op->result_count, result_facts, out_changed));
  }

  // Store result facts.
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] != LOOM_VALUE_ID_INVALID) {
      if (out_changed) {
        loom_value_facts_t old_facts =
            loom_value_fact_table_lookup(table, results[i]);
        if (!loom_value_facts_equal(old_facts, result_facts[i])) {
          *out_changed = true;
        }
      }
      IREE_RETURN_IF_ERROR(
          loom_value_fact_table_define(table, results[i], result_facts[i]));
    }
  }

  return iree_ok_status();
}

#define LOOM_FACT_TABLE_INITIAL_REGION_STACK 8

typedef struct loom_value_fact_region_frame_t {
  // Region whose blocks still need fact propagation.
  loom_region_t* region;

  // Op that owns the region, or NULL for the root function body.
  loom_op_t* parent_op;
} loom_value_fact_region_frame_t;

iree_status_t loom_value_fact_table_compute(loom_value_fact_table_t* table,
                                            const loom_module_t* module,
                                            loom_func_like_t function) {
  loom_region_t* body = loom_func_like_body(function);
  if (!body) return iree_ok_status();

  // Iterative DFS over all regions in the function (same pattern as
  // loom_rewriter_seed_function). Visits ops in dominance order so
  // operand facts are computed before their users.
  iree_host_size_t stack_capacity = LOOM_FACT_TABLE_INITIAL_REGION_STACK;
  loom_value_fact_region_frame_t* region_stack = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(table->arena, stack_capacity,
                                                 sizeof(*region_stack),
                                                 (void**)&region_stack));
  iree_host_size_t stack_count = 0;
  region_stack[stack_count++] = (loom_value_fact_region_frame_t){
      .region = body,
      .parent_op = NULL,
  };

  while (stack_count > 0) {
    loom_value_fact_region_frame_t frame = region_stack[--stack_count];
    loom_block_t* block = NULL;
    loom_region_for_each_block(frame.region, block) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_block_args(
          table, module, block, frame.parent_op));
      loom_op_t* op = NULL;
      loom_block_for_each_op(block, op) {
        IREE_RETURN_IF_ERROR(
            loom_value_fact_table_compute_op(table, module, op));
        if (op->region_count == 0) continue;
        // Ensure space for nested regions.
        iree_host_size_t needed = stack_count + op->region_count;
        if (needed > stack_capacity) {
          IREE_RETURN_IF_ERROR(iree_arena_grow_array(
              table->arena, stack_count, needed, sizeof(*region_stack),
              &stack_capacity, (void**)&region_stack));
        }
        loom_region_t** regions = loom_op_regions(op);
        for (uint8_t r = 0; r < op->region_count; ++r) {
          if (regions[r]) {
            region_stack[stack_count++] = (loom_value_fact_region_frame_t){
                .region = regions[r],
                .parent_op = op,
            };
          }
        }
      }
    }
  }

  return iree_ok_status();
}
