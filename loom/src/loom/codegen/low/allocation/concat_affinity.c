// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/concat_affinity.h"

#include <string.h>

#include "iree/base/bitmap.h"
#include "loom/util/segmented_storage.h"

// One composed source span inside its terminal concat's storage.
typedef struct loom_low_allocation_concat_span_t {
  // Terminal concat result's local value ordinal.
  loom_value_ordinal_t result_ordinal;
  // First covered allocation unit in the indexed source value.
  uint32_t source_unit_offset;
  // First corresponding allocation unit in the terminal result.
  uint32_t result_unit_offset;
  // Covered unit count; zero means no unique concat destination. A zero span
  // with an invalid result ordinal rejects later propagated candidates.
  uint32_t unit_count;
} loom_low_allocation_concat_span_t;

#define LOOM_LOW_ALLOCATION_CONCAT_SPANS_PER_SEGMENT 1024u

struct loom_low_allocation_concat_affinity_t {
  // Composed spans indexed by local value ordinal in fixed pooled segments.
  loom_segmented_storage_t spans;
  // Values that can trigger an early result reservation but whose storage
  // remains observable across the downstream concat result.
  iree_bitmap_t reservation_only_values;
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

static loom_low_allocation_concat_span_t*
loom_low_allocation_concat_affinity_mutable_span(
    loom_low_allocation_concat_affinity_t* affinity,
    loom_value_ordinal_t value_ordinal) {
  loom_low_allocation_concat_span_t* segment = loom_segmented_storage_segment(
      &affinity->spans,
      value_ordinal / LOOM_LOW_ALLOCATION_CONCAT_SPANS_PER_SEGMENT);
  return &segment[value_ordinal % LOOM_LOW_ALLOCATION_CONCAT_SPANS_PER_SEGMENT];
}

static void loom_low_allocation_concat_affinity_merge_span(
    loom_low_allocation_concat_affinity_t* affinity,
    loom_value_ordinal_t value_ordinal, loom_low_allocation_concat_span_t span,
    bool can_alias_storage) {
  loom_low_allocation_concat_span_t* retained =
      loom_low_allocation_concat_affinity_mutable_span(affinity, value_ordinal);
  if (retained->unit_count == 0) {
    if (retained->result_ordinal != LOOM_VALUE_ORDINAL_INVALID) {
      *retained = span;
      if (!can_alias_storage) {
        iree_bitmap_set(affinity->reservation_only_values, value_ordinal);
      }
    }
    return;
  }
  if (retained->result_ordinal == span.result_ordinal &&
      retained->source_unit_offset == span.source_unit_offset &&
      retained->result_unit_offset == span.result_unit_offset &&
      retained->unit_count == span.unit_count) {
    if (!can_alias_storage) {
      iree_bitmap_set(affinity->reservation_only_values, value_ordinal);
    }
    return;
  }
  *retained = (loom_low_allocation_concat_span_t){
      .result_ordinal = LOOM_VALUE_ORDINAL_INVALID,
  };
}

static void loom_low_allocation_concat_affinity_reject_span(
    loom_low_allocation_concat_affinity_t* affinity,
    loom_value_ordinal_t value_ordinal) {
  *loom_low_allocation_concat_affinity_mutable_span(affinity, value_ordinal) =
      (loom_low_allocation_concat_span_t){
          .result_ordinal = LOOM_VALUE_ORDINAL_INVALID,
      };
}

bool loom_low_allocation_concat_affinity_relation_for_source(
    const loom_low_allocation_concat_affinity_t* affinity,
    loom_value_ordinal_t source_ordinal,
    loom_low_placement_relation_t* out_relation) {
  if (affinity == NULL) {
    return false;
  }
  const loom_low_allocation_concat_span_t* span =
      loom_low_allocation_concat_affinity_span(affinity, source_ordinal);
  if (span->unit_count == 0) {
    return false;
  }
  *out_relation = (loom_low_placement_relation_t){
      .result_ordinal = span->result_ordinal,
      .source_ordinal = source_ordinal,
      .result_unit_offset = span->result_unit_offset,
      .source_unit_offset = span->source_unit_offset,
      .unit_count = span->unit_count,
      .kind = LOOM_LOW_PLACEMENT_RELATION_CONTIGUOUS_PART,
      .cause = LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT,
      .flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED,
      .priority = 1,
      .source_operand_index = LOOM_LOW_PLACEMENT_SOURCE_OPERAND_NONE,
  };
  if (!iree_bitmap_test(affinity->reservation_only_values, source_ordinal)) {
    out_relation->flags |= LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE;
  }
  return true;
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

static bool loom_low_allocation_concat_affinity_storage_handoff_is_disjoint(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    loom_value_ordinal_t source_ordinal, uint32_t source_unit_count,
    loom_value_ordinal_t result_ordinal) {
  const uint32_t source_point_start =
      unit_liveness->values[source_ordinal].unit_point_start;
  const uint32_t result_acquisition =
      unit_liveness->values[result_ordinal].acquisition_start_point;
  if (source_point_start == UINT32_MAX || result_acquisition == UINT32_MAX) {
    return false;
  }
  for (uint32_t unit = 0; unit < source_unit_count; ++unit) {
    if (unit_liveness->end_points[source_point_start + unit] >
        result_acquisition) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_low_allocation_concat_affinity_propagate_paths(
    loom_low_allocation_concat_affinity_t* affinity,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_placement_table_t* placement,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    iree_arena_allocator_t* arena) {
  if (liveness->operation_count == 0) {
    return iree_ok_status();
  }

  uint32_t max_operation_point = 0;
  for (iree_host_size_t i = 0; i < liveness->operation_count; ++i) {
    const loom_liveness_operation_point_t* operation =
        loom_liveness_operation_at(liveness, (uint32_t)i);
    max_operation_point = iree_max(max_operation_point, operation->end_point);
  }

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(arena->block_pool, &scratch_arena);
  uint32_t* result_heads_by_point = NULL;
  uint32_t* next_result_by_ordinal = NULL;
  uint8_t* direct_use_counts_by_ordinal = NULL;
  iree_status_t status = iree_arena_allocate_array(
      &scratch_arena, (iree_host_size_t)max_operation_point + 1,
      sizeof(*result_heads_by_point), (void**)&result_heads_by_point);
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(&scratch_arena, placement->value_count,
                                       sizeof(*next_result_by_ordinal),
                                       (void**)&next_result_by_ordinal);
  }
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(&scratch_arena, placement->value_count,
                                       sizeof(*direct_use_counts_by_ordinal),
                                       (void**)&direct_use_counts_by_ordinal);
  }
  if (!iree_status_is_ok(status)) {
    iree_arena_deinitialize(&scratch_arena);
    return status;
  }
  memset(result_heads_by_point, 0xFF,
         ((iree_host_size_t)max_operation_point + 1) *
             sizeof(*result_heads_by_point));
  memset(next_result_by_ordinal, 0xFF,
         placement->value_count * sizeof(*next_result_by_ordinal));
  memset(direct_use_counts_by_ordinal, 0,
         placement->value_count * sizeof(*direct_use_counts_by_ordinal));

  for (iree_host_size_t i = 0; i < liveness->operation_count; ++i) {
    const loom_liveness_operation_point_t* operation =
        loom_liveness_operation_at(liveness, (uint32_t)i);
    for (uint32_t use_index = 0; use_index < operation->direct_use_count;
         ++use_index) {
      const loom_value_ordinal_t ordinal = loom_liveness_operation_use_ordinal(
          liveness, operation->use_start + use_index);
      if (direct_use_counts_by_ordinal[ordinal] < 2) {
        ++direct_use_counts_by_ordinal[ordinal];
      }
    }
  }

  for (loom_value_ordinal_t ordinal = 0; ordinal < placement->value_count;
       ++ordinal) {
    const loom_liveness_interval_t* interval =
        loom_liveness_interval_for_value_ordinal(liveness, ordinal);
    if (interval == NULL || interval->definition_point == UINT32_MAX ||
        interval->definition_point > max_operation_point) {
      continue;
    }
    next_result_by_ordinal[ordinal] =
        result_heads_by_point[interval->definition_point];
    result_heads_by_point[interval->definition_point] = ordinal;
  }

  for (iree_host_size_t reverse_index = liveness->operation_count;
       reverse_index > 0; --reverse_index) {
    const loom_liveness_operation_point_t* operation =
        loom_liveness_operation_at(liveness, (uint32_t)(reverse_index - 1));
    loom_value_ordinal_t result_ordinal = LOOM_VALUE_ORDINAL_INVALID;
    const loom_low_allocation_concat_span_t* result_span = NULL;
    for (loom_value_ordinal_t ordinal =
             result_heads_by_point[operation->end_point];
         ordinal != LOOM_VALUE_ORDINAL_INVALID;
         ordinal = next_result_by_ordinal[ordinal]) {
      const loom_liveness_interval_t* interval =
          loom_liveness_interval_for_value_ordinal(liveness, ordinal);
      const loom_low_allocation_concat_span_t* span =
          loom_low_allocation_concat_affinity_span(affinity, ordinal);
      if (span->unit_count == 0 || span->source_unit_offset != 0 ||
          span->unit_count != interval->unit_count) {
        continue;
      }
      if (result_ordinal != LOOM_VALUE_ORDINAL_INVALID) {
        result_ordinal = LOOM_VALUE_ORDINAL_INVALID;
        result_span = NULL;
        break;
      }
      result_ordinal = ordinal;
      result_span = span;
    }
    if (result_span == NULL) {
      continue;
    }

    const loom_liveness_interval_t* result_interval =
        loom_liveness_interval_for_value_ordinal(liveness, result_ordinal);
    if (result_interval->value_class.type_kind != LOOM_TYPE_REGISTER ||
        !loom_low_reg_class_uses_explicit_physical_registers(
            &descriptor_set->reg_classes[result_interval->value_class
                                             .register_class_id])) {
      continue;
    }
    loom_value_ordinal_t source_ordinal = LOOM_VALUE_ORDINAL_INVALID;
    bool source_is_ambiguous = false;
    bool source_can_handoff_storage = false;
    for (uint32_t i = 0; i < operation->direct_use_count; ++i) {
      const loom_value_ordinal_t candidate =
          loom_liveness_operation_use_ordinal(liveness,
                                              operation->use_start + i);
      const loom_liveness_interval_t* source_interval =
          loom_liveness_interval_for_value_ordinal(liveness, candidate);
      if (source_interval == NULL ||
          source_interval->unit_count != result_interval->unit_count ||
          !loom_liveness_value_class_equal(source_interval->value_class,
                                           result_interval->value_class)) {
        continue;
      }
      if (source_ordinal != LOOM_VALUE_ORDINAL_INVALID) {
        source_is_ambiguous = true;
        continue;
      }
      source_ordinal = candidate;
      source_can_handoff_storage =
          loom_low_allocation_concat_affinity_storage_handoff_is_disjoint(
              unit_liveness, candidate, source_interval->unit_count,
              result_ordinal);
    }
    bool can_alias_storage = false;
    if (source_is_ambiguous || source_ordinal == LOOM_VALUE_ORDINAL_INVALID) {
      continue;
    }
    if (source_can_handoff_storage) {
      if (direct_use_counts_by_ordinal[source_ordinal] > 1) {
        can_alias_storage = !iree_bitmap_test(affinity->reservation_only_values,
                                              result_ordinal);
      } else {
        // A linear last-use chain has no competing placement to constrain.
        // Acting on every such path turns a copy-elision preference into an
        // early aggregate commitment and can make a feasible allocation fail.
        continue;
      }
    } else if (direct_use_counts_by_ordinal[source_ordinal] != 1) {
      continue;
    }
    loom_low_allocation_concat_affinity_merge_span(
        affinity, source_ordinal, *result_span, can_alias_storage);
  }

  iree_arena_deinitialize(&scratch_arena);
  return iree_ok_status();
}

iree_status_t loom_low_allocation_concat_affinity_build(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_placement_table_t* placement,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    iree_arena_allocator_t* arena,
    loom_low_allocation_concat_affinity_t** out_affinity) {
  *out_affinity = NULL;
  if (!iree_any_bit_set(placement->storage.flags,
                        LOOM_LOW_PLACEMENT_STORAGE_FLAG_CONCAT)) {
    return iree_ok_status();
  }
  loom_low_allocation_concat_affinity_t* affinity = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*affinity), (void**)&affinity));
  const iree_host_size_t reservation_word_count =
      iree_bitmap_calculate_words(placement->value_count);
  uint64_t* reservation_words = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, reservation_word_count,
                                                 sizeof(*reservation_words),
                                                 (void**)&reservation_words));
  memset(reservation_words, 0,
         reservation_word_count * sizeof(*reservation_words));
  affinity->reservation_only_values = (iree_bitmap_t){
      .bit_count = placement->value_count,
      .words = reservation_words,
  };
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
      for (uint32_t i = 0; i < range.count; ++i) {
        const uint32_t relation_index =
            placement->relation_indices_by_source_ordinal[range.start + i];
        if (placement->relations[relation_index].cause ==
            LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT) {
          // Authored concat policy remains authoritative even when fanout
          // prevents retaining one destination span.
          loom_low_allocation_concat_affinity_reject_span(affinity, value);
          break;
        }
      }
      continue;
    }
    loom_low_placement_relation_t relation =
        placement->relations
            [placement->relation_indices_by_source_ordinal[range.start]];
    if (relation.cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT ||
        !loom_low_placement_relation_can_alias(&relation) ||
        iree_any_bit_set(relation.flags,
                         LOOM_LOW_PLACEMENT_RELATION_FLAG_CAPTURED_PART)) {
      if (relation.cause == LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT) {
        loom_low_allocation_concat_affinity_reject_span(affinity, value);
      }
      continue;
    }
    loom_low_allocation_concat_affinity_project(affinity, &relation);
    loom_low_allocation_concat_affinity_merge_span(
        affinity, value,
        (loom_low_allocation_concat_span_t){
            .result_ordinal = relation.result_ordinal,
            .source_unit_offset = relation.source_unit_offset,
            .result_unit_offset = relation.result_unit_offset,
            .unit_count = relation.unit_count,
        },
        /*can_alias_storage=*/true);
  }
  IREE_RETURN_IF_ERROR(loom_low_allocation_concat_affinity_propagate_paths(
      affinity, descriptor_set, placement, liveness, unit_liveness, arena));
  *out_affinity = affinity;
  return iree_ok_status();
}
