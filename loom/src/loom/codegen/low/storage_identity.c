// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/storage_identity.h"

#include "loom/codegen/low/storage_relation.h"

typedef struct loom_low_storage_identity_builder_t {
  // Function-local value domain defining valid identity ordinals.
  const loom_local_value_domain_t* domain;
  // Domain values plus reserved preparation capacity.
  loom_value_ordinal_t capacity;
  // Function-lifetime arena owning the lazily allocated origin table.
  iree_arena_allocator_t* arena;
  // Union-find parents indexed by function-local value ordinal.
  loom_value_ordinal_t* origins;
} loom_low_storage_identity_builder_t;

static loom_value_ordinal_t loom_low_storage_identity_origin(
    loom_value_ordinal_t* origins, loom_value_ordinal_t ordinal) {
  loom_value_ordinal_t origin = ordinal;
  while (origins[origin] != origin) {
    origin = origins[origin];
  }
  while (origins[ordinal] != origin) {
    const loom_value_ordinal_t parent = origins[ordinal];
    origins[ordinal] = origin;
    ordinal = parent;
  }
  return origin;
}

static iree_status_t loom_low_storage_identity_require_origins(
    loom_low_storage_identity_builder_t* builder) {
  if (builder->origins != NULL) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, builder->capacity, sizeof(*builder->origins),
      (void**)&builder->origins));
  for (loom_value_ordinal_t i = 0; i < builder->capacity; ++i) {
    builder->origins[i] = i;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_storage_identity_union(
    loom_low_storage_identity_builder_t* builder,
    const loom_low_storage_relation_t* relation) {
  IREE_ASSERT_EQ(relation->destination_unit_offset, 0u);
  IREE_ASSERT_EQ(relation->source_unit_offset, 0u);

  IREE_RETURN_IF_ERROR(loom_low_storage_identity_require_origins(builder));
  const loom_value_ordinal_t destination = loom_low_storage_identity_origin(
      builder->origins, loom_local_value_domain_ordinal(
                            builder->domain, relation->destination_value_id));
  const loom_value_ordinal_t source = loom_low_storage_identity_origin(
      builder->origins, loom_local_value_domain_ordinal(
                            builder->domain, relation->source_value_id));
  if (destination != source) {
    builder->origins[destination] = source;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_storage_identity_visit_region(
    loom_low_storage_identity_builder_t* builder, const loom_region_t* region) {
  const loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (op->tied_result_count != 0 ||
          loom_traits_have_storage_relation(op->traits)) {
        loom_low_storage_relation_iterator_t iterator;
        loom_low_storage_relation_iterator_initialize(builder->domain->module,
                                                      op, &iterator);
        loom_low_storage_relation_t relation;
        while (loom_low_storage_relation_iterator_next(&iterator, &relation)) {
          if (relation.kind == LOOM_LOW_STORAGE_RELATION_SAME_STORAGE &&
              iree_any_bit_set(relation.flags,
                               LOOM_LOW_STORAGE_RELATION_FLAG_HARD)) {
            IREE_RETURN_IF_ERROR(
                loom_low_storage_identity_union(builder, &relation));
          }
        }
      }
      if (!iree_any_bit_set(builder->domain->flags,
                            LOOM_LOCAL_VALUE_DOMAIN_FLAG_REGION_TREE)) {
        continue;
      }
      loom_region_t* const* regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        if (regions[i] != NULL) {
          IREE_RETURN_IF_ERROR(
              loom_low_storage_identity_visit_region(builder, regions[i]));
        }
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_storage_identity_build(
    const loom_local_value_domain_t* domain,
    iree_host_size_t additional_value_capacity, iree_arena_allocator_t* arena,
    loom_value_ordinal_t** out_origins) {
  *out_origins = NULL;
  if (additional_value_capacity >
      LOOM_VALUE_ORDINAL_INVALID - domain->value_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "function preparation exceeds value ordinal range");
  }
  const loom_value_ordinal_t capacity =
      domain->value_count + (loom_value_ordinal_t)additional_value_capacity;
  loom_low_storage_identity_builder_t builder = {
      .domain = domain,
      .capacity = capacity,
      .arena = arena,
  };
  IREE_RETURN_IF_ERROR(
      loom_low_storage_identity_visit_region(&builder, domain->region));
  if (builder.origins != NULL) {
    for (loom_value_ordinal_t i = 0; i < domain->value_count; ++i) {
      builder.origins[i] = loom_low_storage_identity_origin(builder.origins, i);
    }
  }
  *out_origins = builder.origins;
  return iree_ok_status();
}
