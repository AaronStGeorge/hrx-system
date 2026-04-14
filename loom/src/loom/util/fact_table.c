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
    default:
      return false;
  }
}

static iree_status_t loom_value_fact_table_materialize_extension_payload(
    loom_value_fact_table_t* table, loom_value_fact_extension_entry_t* entry) {
  if (entry->kind != LOOM_VALUE_FACT_EXTENSION_SMALL_STATIC_LANES) {
    return iree_ok_status();
  }
  iree_host_size_t lane_count = entry->payload.small_static_lanes.count;
  if (lane_count == 0) {
    entry->payload.small_static_lanes.lanes = NULL;
    return iree_ok_status();
  }
  loom_value_facts_t* lanes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      table->arena, lane_count, sizeof(loom_value_facts_t), (void**)&lanes));
  memcpy(lanes, entry->payload.small_static_lanes.lanes,
         lane_count * sizeof(loom_value_facts_t));
  entry->payload.small_static_lanes.lanes = lanes;
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

static iree_status_t loom_value_fact_table_intern_extension(
    loom_value_fact_table_t* table,
    const loom_value_fact_extension_entry_t* candidate,
    loom_value_fact_extension_id_t* out_id) {
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
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_materialize_extension_payload(table, &entry));
  entry.next_id = table->extensions.buckets[bucket_index];
  table->extensions.entries[table->extensions.count] = entry;
  table->extensions.buckets[bucket_index] = id;
  table->extensions.count = new_count;
  *out_id = id;
  return iree_ok_status();
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

//===----------------------------------------------------------------------===//
// Forward pass: compute facts for ops
//===----------------------------------------------------------------------===//

iree_status_t loom_value_fact_table_compute_op(loom_value_fact_table_t* table,
                                               const loom_module_t* module,
                                               const loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable || !vtable->infer_facts) return iree_ok_status();

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

  // Store result facts.
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] != LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(
          loom_value_fact_table_define(table, results[i], result_facts[i]));
    }
  }

  return iree_ok_status();
}

#define LOOM_FACT_TABLE_INITIAL_REGION_STACK 8

iree_status_t loom_value_fact_table_compute(loom_value_fact_table_t* table,
                                            loom_module_t* module,
                                            loom_func_like_t function) {
  loom_region_t* body = loom_func_like_body(function);
  if (!body) return iree_ok_status();

  // Iterative DFS over all regions in the function (same pattern as
  // loom_rewriter_seed_function). Visits ops in dominance order so
  // operand facts are computed before their users.
  iree_host_size_t stack_capacity = LOOM_FACT_TABLE_INITIAL_REGION_STACK;
  loom_region_t** region_stack = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(table->arena, stack_capacity,
                                                 sizeof(loom_region_t*),
                                                 (void**)&region_stack));
  iree_host_size_t stack_count = 0;
  region_stack[stack_count++] = body;

  while (stack_count > 0) {
    loom_region_t* region = region_stack[--stack_count];
    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      loom_op_t* op = NULL;
      loom_block_for_each_op(block, op) {
        IREE_RETURN_IF_ERROR(
            loom_value_fact_table_compute_op(table, module, op));
        if (op->region_count == 0) continue;
        // Ensure space for nested regions.
        iree_host_size_t needed = stack_count + op->region_count;
        if (needed > stack_capacity) {
          IREE_RETURN_IF_ERROR(iree_arena_grow_array(
              table->arena, stack_count, needed, sizeof(loom_region_t*),
              &stack_capacity, (void**)&region_stack));
        }
        loom_region_t** regions = loom_op_regions(op);
        for (uint8_t r = 0; r < op->region_count; ++r) {
          if (regions[r]) {
            region_stack[stack_count++] = regions[r];
          }
        }
      }
    }
  }

  return iree_ok_status();
}
