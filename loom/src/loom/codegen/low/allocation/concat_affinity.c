// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/concat_affinity.h"

#include <string.h>

#include "loom/util/segmented_storage.h"

// One composed source span inside its terminal concat's storage.
typedef struct loom_low_allocation_concat_span_t {
  // Terminal concat result's local value ordinal.
  loom_value_ordinal_t result_ordinal;
  // First covered allocation unit in the indexed source value.
  uint32_t source_unit_offset;
  // First corresponding allocation unit in the terminal result.
  uint32_t result_unit_offset;
  // Covered unit count; zero means no unique aliasable concat chain.
  uint32_t unit_count;
} loom_low_allocation_concat_span_t;

#define LOOM_LOW_ALLOCATION_CONCAT_SPANS_PER_SEGMENT 1024u

struct loom_low_allocation_concat_affinity_t {
  // Composed spans indexed by local value ordinal in fixed pooled segments.
  loom_segmented_storage_t spans;
};

static const loom_low_allocation_concat_span_t*
loom_low_allocation_concat_affinity_span(
    const loom_low_allocation_concat_affinity_t* affinity,
    loom_value_ordinal_t value_ordinal) {
  const loom_low_allocation_concat_span_t* segment =
      loom_segmented_storage_const_segment(
          &affinity->spans,
          value_ordinal / LOOM_LOW_ALLOCATION_CONCAT_SPANS_PER_SEGMENT);
  return &segment[value_ordinal % LOOM_LOW_ALLOCATION_CONCAT_SPANS_PER_SEGMENT];
}

bool loom_low_allocation_concat_affinity_project(
    const loom_low_allocation_concat_affinity_t* affinity,
    loom_low_placement_relation_t* relation) {
  if (affinity == NULL) {
    return false;
  }
  const loom_low_allocation_concat_span_t* span =
      loom_low_allocation_concat_affinity_span(affinity,
                                               relation->result_ordinal);
  if (span->unit_count == 0) {
    return false;
  }
  const loom_low_placement_relation_t parent = {
      .source_ordinal = relation->result_ordinal,
      .result_ordinal = span->result_ordinal,
      .source_unit_offset = span->source_unit_offset,
      .result_unit_offset = span->result_unit_offset,
      .unit_count = span->unit_count,
  };
  loom_low_placement_relation_t composed;
  if (!loom_low_placement_relation_compose(relation, &parent, &composed)) {
    return false;
  }
  relation->result_ordinal = composed.result_ordinal;
  relation->source_unit_offset = composed.source_unit_offset;
  relation->result_unit_offset = composed.result_unit_offset;
  relation->unit_count = composed.unit_count;
  return true;
}

iree_status_t loom_low_allocation_concat_affinity_build(
    const loom_low_placement_table_t* placement, iree_arena_allocator_t* arena,
    loom_low_allocation_concat_affinity_t** out_affinity) {
  *out_affinity = NULL;
  if (placement->edge_relation_count == 0 ||
      !iree_any_bit_set(placement->storage.flags,
                        LOOM_LOW_PLACEMENT_STORAGE_FLAG_CONCAT)) {
    return iree_ok_status();
  }
  loom_low_allocation_concat_affinity_t* affinity = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*affinity), (void**)&affinity));
  loom_segmented_storage_initialize(
      iree_min(placement->value_count,
               LOOM_LOW_ALLOCATION_CONCAT_SPANS_PER_SEGMENT) *
          sizeof(loom_low_allocation_concat_span_t),
      iree_alignof(loom_low_allocation_concat_span_t), &affinity->spans);
  const uint32_t segment_count =
      placement->value_count / LOOM_LOW_ALLOCATION_CONCAT_SPANS_PER_SEGMENT +
      (placement->value_count % LOOM_LOW_ALLOCATION_CONCAT_SPANS_PER_SEGMENT !=
       0);
  for (uint32_t i = 0; i < segment_count; ++i) {
    void* segment = NULL;
    IREE_RETURN_IF_ERROR(
        loom_segmented_storage_append(&affinity->spans, arena, &segment));
    memset(segment, 0, affinity->spans.segment_size);
  }
  for (loom_value_ordinal_t cursor = 0;
       cursor < placement->storage_value_order_count; ++cursor) {
    const loom_value_ordinal_t value = placement->storage_value_order[cursor];
    const loom_low_placement_relation_range_t range =
        placement->ranges_by_source_ordinal[value];
    if (range.count != 1) {
      continue;
    }
    loom_low_placement_relation_t relation =
        placement->relations
            [placement->relation_indices_by_source_ordinal[range.start]];
    if (relation.cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT ||
        !loom_low_placement_relation_can_alias(&relation) ||
        iree_any_bit_set(relation.flags,
                         LOOM_LOW_PLACEMENT_RELATION_FLAG_CAPTURED_PART)) {
      continue;
    }
    if (!loom_low_allocation_concat_affinity_project(affinity, &relation)) {
      const loom_low_placement_relation_range_t outgoing =
          placement->ranges_by_source_ordinal[relation.result_ordinal];
      if (outgoing.count != 1) {
        continue;
      }
      const uint32_t edge_index =
          placement->relation_indices_by_source_ordinal[outgoing.start];
      if (!loom_low_placement_cause_is_edge(
              placement->relations[edge_index].cause)) {
        continue;
      }
    }
    loom_low_allocation_concat_span_t* segment = loom_segmented_storage_segment(
        &affinity->spans, value / LOOM_LOW_ALLOCATION_CONCAT_SPANS_PER_SEGMENT);
    segment[value % LOOM_LOW_ALLOCATION_CONCAT_SPANS_PER_SEGMENT] =
        (loom_low_allocation_concat_span_t){
            .result_ordinal = relation.result_ordinal,
            .source_unit_offset = relation.source_unit_offset,
            .result_unit_offset = relation.result_unit_offset,
            .unit_count = relation.unit_count,
        };
  }
  *out_affinity = affinity;
  return iree_ok_status();
}
