// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/placement.h"

#include <string.h>

#include "loom/ops/low/ops.h"

typedef struct loom_low_placement_build_state_t {
  // Module containing the analyzed low region.
  loom_module_t* module;
  // Low function body region being analyzed.
  const loom_region_t* region;
  // Acquired local value domain for |region|.
  const loom_local_value_domain_t* value_domain;
  // Liveness analysis over |value_domain|.
  const loom_liveness_analysis_t* liveness;
  // Arena owning placement table storage.
  iree_arena_allocator_t* arena;
  // Mutable relation records being populated.
  loom_low_placement_relation_t* relations;
  // Relation ranges indexed by result value ordinal.
  loom_low_placement_relation_range_t* ranges_by_result_ordinal;
  // Relation indices grouped by source value ordinal.
  uint32_t* relation_indices_by_source_ordinal;
  // Relation ranges indexed by source value ordinal.
  loom_low_placement_relation_range_t* ranges_by_source_ordinal;
  // Number of relation records counted or populated.
  iree_host_size_t relation_count;
  // Number of relation records appended after range prefixing.
  iree_host_size_t appended_relation_count;
  // Number of source relation indices appended after range prefixing.
  iree_host_size_t appended_source_relation_count;
} loom_low_placement_build_state_t;

bool loom_low_placement_cause_can_alias(loom_low_placement_cause_t cause) {
  switch (cause) {
    case LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT:
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY:
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_SLICE:
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT:
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_BRANCH:
      return true;
    default:
      return false;
  }
}

static loom_value_ordinal_t loom_low_placement_value_ordinal(
    const loom_low_placement_build_state_t* state, loom_value_id_t value_id) {
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_try_ordinal(state->value_domain, value_id);
  IREE_ASSERT(value_ordinal != LOOM_VALUE_ORDINAL_INVALID,
              "verified low placement value must be inside the local value "
              "domain");
  return value_ordinal;
}

static const loom_liveness_interval_t* loom_low_placement_interval_for_ordinal(
    const loom_low_placement_build_state_t* state,
    loom_value_ordinal_t value_ordinal) {
  const loom_liveness_interval_t* interval =
      loom_liveness_interval_for_value_ordinal(state->liveness, value_ordinal);
  IREE_ASSERT(interval != NULL,
              "verified low placement value must have a liveness interval");
  return interval;
}

static iree_status_t loom_low_placement_increment_relation_count(
    loom_low_placement_build_state_t* state,
    loom_value_ordinal_t result_ordinal, loom_value_ordinal_t source_ordinal) {
  if (state->relation_count >= UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low placement relation count exceeds u32 range");
  }
  loom_low_placement_relation_range_t* result_range =
      &state->ranges_by_result_ordinal[result_ordinal];
  if (result_range->count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low placement result relation count exceeds u32 "
                            "range");
  }
  loom_low_placement_relation_range_t* source_range =
      &state->ranges_by_source_ordinal[source_ordinal];
  if (source_range->count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low placement source relation count exceeds u32 "
                            "range");
  }
  ++result_range->count;
  ++source_range->count;
  ++state->relation_count;
  return iree_ok_status();
}

static iree_status_t loom_low_placement_count_relation(
    loom_low_placement_build_state_t* state, loom_value_id_t result_value_id,
    loom_value_id_t source_value_id) {
  const loom_value_ordinal_t result_ordinal =
      loom_low_placement_value_ordinal(state, result_value_id);
  const loom_value_ordinal_t source_ordinal =
      loom_low_placement_value_ordinal(state, source_value_id);
  return loom_low_placement_increment_relation_count(state, result_ordinal,
                                                     source_ordinal);
}

static void loom_low_placement_prefix_range_array(
    loom_low_placement_relation_range_t* ranges,
    loom_value_ordinal_t range_count) {
  uint32_t relation_start = 0;
  for (loom_value_ordinal_t i = 0; i < range_count; ++i) {
    loom_low_placement_relation_range_t* range = &ranges[i];
    relation_start += range->count;
    range->start = relation_start - range->count;
    range->count = 0;
  }
}

static void loom_low_placement_prefix_ranges(
    loom_low_placement_build_state_t* state) {
  loom_low_placement_prefix_range_array(state->ranges_by_result_ordinal,
                                        state->value_domain->value_count);
  loom_low_placement_prefix_range_array(state->ranges_by_source_ordinal,
                                        state->value_domain->value_count);
}

static void loom_low_placement_append_relation(
    loom_low_placement_build_state_t* state,
    const loom_low_placement_relation_t* relation) {
  loom_low_placement_relation_range_t* result_range =
      &state->ranges_by_result_ordinal[relation->result_ordinal];
  const iree_host_size_t relation_index =
      (iree_host_size_t)result_range->start + result_range->count;
  IREE_ASSERT_LT(relation_index, state->relation_count);
  state->relations[relation_index] = *relation;
  ++result_range->count;

  loom_low_placement_relation_range_t* source_range =
      &state->ranges_by_source_ordinal[relation->source_ordinal];
  const iree_host_size_t source_index =
      (iree_host_size_t)source_range->start + source_range->count;
  IREE_ASSERT_LT(source_index, state->relation_count);
  state->relation_indices_by_source_ordinal[source_index] =
      (uint32_t)relation_index;
  ++source_range->count;
  ++state->appended_relation_count;
  ++state->appended_source_relation_count;
}

static void loom_low_placement_relation_unit_counts(
    const loom_low_placement_build_state_t* state,
    loom_value_ordinal_t result_ordinal, loom_value_ordinal_t source_ordinal,
    uint32_t* out_result_unit_count, uint32_t* out_source_unit_count) {
  const loom_liveness_interval_t* result_interval =
      loom_low_placement_interval_for_ordinal(state, result_ordinal);
  const loom_liveness_interval_t* source_interval =
      loom_low_placement_interval_for_ordinal(state, source_ordinal);
  *out_result_unit_count = result_interval->unit_count;
  *out_source_unit_count = source_interval->unit_count;
}

static iree_status_t loom_low_placement_append_same_storage_relation(
    loom_low_placement_build_state_t* state, const loom_op_t* op,
    loom_value_id_t result_value_id, loom_value_id_t source_value_id,
    loom_low_placement_cause_t cause,
    loom_low_placement_relation_flags_t flags) {
  const loom_value_ordinal_t result_ordinal =
      loom_low_placement_value_ordinal(state, result_value_id);
  const loom_value_ordinal_t source_ordinal =
      loom_low_placement_value_ordinal(state, source_value_id);
  uint32_t result_unit_count = 0;
  uint32_t source_unit_count = 0;
  loom_low_placement_relation_unit_counts(state, result_ordinal, source_ordinal,
                                          &result_unit_count,
                                          &source_unit_count);
  IREE_ASSERT_EQ(result_unit_count, source_unit_count,
                 "verified same-storage placement relation must use matching "
                 "unit counts");
  const loom_low_placement_relation_t relation = {
      .op = op,
      .result_ordinal = result_ordinal,
      .source_ordinal = source_ordinal,
      .result_unit_offset = 0,
      .source_unit_offset = 0,
      .unit_count = result_unit_count,
      .kind = LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE,
      .cause = cause,
      .flags = flags,
  };
  loom_low_placement_append_relation(state, &relation);
  return iree_ok_status();
}

static iree_status_t loom_low_placement_append_slice_relation(
    loom_low_placement_build_state_t* state, const loom_op_t* op) {
  const int64_t offset = loom_low_slice_offset(op);
  IREE_ASSERT(offset >= 0 && offset <= UINT32_MAX,
              "verified low.slice offset must fit in uint32_t");
  const loom_value_ordinal_t result_ordinal =
      loom_low_placement_value_ordinal(state, loom_low_slice_result(op));
  const loom_value_ordinal_t source_ordinal =
      loom_low_placement_value_ordinal(state, loom_low_slice_source(op));
  uint32_t result_unit_count = 0;
  uint32_t source_unit_count = 0;
  loom_low_placement_relation_unit_counts(state, result_ordinal, source_ordinal,
                                          &result_unit_count,
                                          &source_unit_count);
  const uint32_t source_offset = (uint32_t)offset;
  IREE_ASSERT(source_offset <= source_unit_count &&
                  result_unit_count <= source_unit_count - source_offset,
              "verified low.slice range must fit source unit count");
  const loom_low_placement_relation_t relation = {
      .op = op,
      .result_ordinal = result_ordinal,
      .source_ordinal = source_ordinal,
      .result_unit_offset = 0,
      .source_unit_offset = source_offset,
      .unit_count = result_unit_count,
      .kind = LOOM_LOW_PLACEMENT_RELATION_SUBRANGE,
      .cause = LOOM_LOW_PLACEMENT_CAUSE_LOW_SLICE,
      .flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED,
  };
  loom_low_placement_append_relation(state, &relation);
  return iree_ok_status();
}

static iree_status_t loom_low_placement_append_concat_relations(
    loom_low_placement_build_state_t* state, const loom_op_t* op) {
  const loom_value_ordinal_t result_ordinal =
      loom_low_placement_value_ordinal(state, loom_low_concat_result(op));
  const loom_liveness_interval_t* result_interval =
      loom_low_placement_interval_for_ordinal(state, result_ordinal);

  uint32_t result_offset = 0;
  loom_value_slice_t sources = loom_low_concat_sources(op);
  for (uint16_t i = 0; i < sources.count; ++i) {
    const loom_value_ordinal_t source_ordinal =
        loom_low_placement_value_ordinal(state, sources.values[i]);
    const loom_liveness_interval_t* source_interval =
        loom_low_placement_interval_for_ordinal(state, source_ordinal);
    IREE_ASSERT(source_interval->unit_count <= UINT32_MAX - result_offset,
                "verified low.concat range must fit uint32_t unit offsets");
    const loom_low_placement_relation_t relation = {
        .op = op,
        .result_ordinal = result_ordinal,
        .source_ordinal = source_ordinal,
        .result_unit_offset = result_offset,
        .source_unit_offset = 0,
        .unit_count = source_interval->unit_count,
        .kind = LOOM_LOW_PLACEMENT_RELATION_CONTIGUOUS_PART,
        .cause = LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT,
        .flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED,
    };
    loom_low_placement_append_relation(state, &relation);
    result_offset += source_interval->unit_count;
  }
  IREE_ASSERT_EQ(result_offset, result_interval->unit_count,
                 "verified low.concat source units must cover the result");
  return iree_ok_status();
}

static iree_status_t loom_low_placement_count_tied_results(
    loom_low_placement_build_state_t* state, const loom_op_t* op) {
  const loom_value_id_t* results = loom_op_const_results(op);
  const loom_value_id_t* operands = loom_op_const_operands(op);
  const loom_tied_result_t* tied_results = loom_op_tied_results(op);
  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    const loom_tied_result_t tied = tied_results[i];
    IREE_ASSERT(tied.result_index < op->result_count &&
                    tied.operand_index < op->operand_count,
                "verified tied result metadata must reference existing "
                "fields");
    IREE_RETURN_IF_ERROR(loom_low_placement_count_relation(
        state, results[tied.result_index], operands[tied.operand_index]));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_placement_append_tied_results(
    loom_low_placement_build_state_t* state, const loom_op_t* op) {
  const loom_value_id_t* results = loom_op_const_results(op);
  const loom_value_id_t* operands = loom_op_const_operands(op);
  const loom_tied_result_t* tied_results = loom_op_tied_results(op);
  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    const loom_tied_result_t tied = tied_results[i];
    IREE_ASSERT(tied.result_index < op->result_count &&
                    tied.operand_index < op->operand_count,
                "verified tied result metadata must reference existing "
                "fields");
    IREE_ASSERT(!tied.has_type_change,
                "low verification must reject type-changing tied results "
                "before placement analysis");
    IREE_RETURN_IF_ERROR(loom_low_placement_append_same_storage_relation(
        state, op, results[tied.result_index], operands[tied.operand_index],
        LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT,
        LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_placement_count_structural_relations(
    loom_low_placement_build_state_t* state, const loom_op_t* op) {
  if (loom_low_copy_isa(op)) {
    return loom_low_placement_count_relation(state, loom_low_copy_result(op),
                                             loom_low_copy_source(op));
  }
  if (loom_low_slice_isa(op)) {
    return loom_low_placement_count_relation(state, loom_low_slice_result(op),
                                             loom_low_slice_source(op));
  }
  if (loom_low_concat_isa(op)) {
    const loom_value_id_t result = loom_low_concat_result(op);
    loom_value_slice_t sources = loom_low_concat_sources(op);
    for (uint16_t i = 0; i < sources.count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_low_placement_count_relation(state, result, sources.values[i]));
    }
    return iree_ok_status();
  }
  if (loom_low_br_isa(op)) {
    const loom_block_t* dest = loom_low_br_dest(op);
    loom_value_slice_t args = loom_low_br_args(op);
    IREE_ASSERT_EQ(args.count, dest->arg_count,
                   "verified low.br payload must match destination block");
    for (uint16_t i = 0; i < args.count; ++i) {
      IREE_RETURN_IF_ERROR(loom_low_placement_count_relation(
          state, dest->arg_ids[i], args.values[i]));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_placement_append_structural_relations(
    loom_low_placement_build_state_t* state, const loom_op_t* op) {
  if (loom_low_copy_isa(op)) {
    return loom_low_placement_append_same_storage_relation(
        state, op, loom_low_copy_result(op), loom_low_copy_source(op),
        LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY,
        LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED);
  }
  if (loom_low_slice_isa(op)) {
    return loom_low_placement_append_slice_relation(state, op);
  }
  if (loom_low_concat_isa(op)) {
    return loom_low_placement_append_concat_relations(state, op);
  }
  if (loom_low_br_isa(op)) {
    const loom_block_t* dest = loom_low_br_dest(op);
    loom_value_slice_t args = loom_low_br_args(op);
    IREE_ASSERT_EQ(args.count, dest->arg_count,
                   "verified low.br payload must match destination block");
    for (uint16_t i = 0; i < args.count; ++i) {
      IREE_RETURN_IF_ERROR(loom_low_placement_append_same_storage_relation(
          state, op, dest->arg_ids[i], args.values[i],
          LOOM_LOW_PLACEMENT_CAUSE_LOW_BRANCH,
          LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_placement_visit_region_ops(
    loom_low_placement_build_state_t* state, const loom_region_t* region,
    iree_status_t (*visit)(loom_low_placement_build_state_t* state,
                           const loom_op_t* op)) {
  const loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      IREE_RETURN_IF_ERROR(visit(state, op));
      if (!iree_any_bit_set(state->value_domain->flags,
                            LOOM_LOCAL_VALUE_DOMAIN_FLAG_REGION_TREE)) {
        continue;
      }
      loom_region_t* const* regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        if (regions[i] == NULL) {
          continue;
        }
        IREE_RETURN_IF_ERROR(
            loom_low_placement_visit_region_ops(state, regions[i], visit));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_placement_visit_ops(
    loom_low_placement_build_state_t* state,
    iree_status_t (*visit)(loom_low_placement_build_state_t* state,
                           const loom_op_t* op)) {
  return loom_low_placement_visit_region_ops(state, state->region, visit);
}

static iree_status_t loom_low_placement_count_op_relations(
    loom_low_placement_build_state_t* state, const loom_op_t* op) {
  IREE_RETURN_IF_ERROR(loom_low_placement_count_tied_results(state, op));
  return loom_low_placement_count_structural_relations(state, op);
}

static iree_status_t loom_low_placement_append_op_relations(
    loom_low_placement_build_state_t* state, const loom_op_t* op) {
  IREE_RETURN_IF_ERROR(loom_low_placement_append_tied_results(state, op));
  return loom_low_placement_append_structural_relations(state, op);
}

iree_status_t loom_low_placement_analyze_region(
    loom_module_t* module, const loom_region_t* region,
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_analysis_t* liveness, iree_arena_allocator_t* arena,
    loom_low_placement_table_t* out_table) {
  IREE_ASSERT(loom_local_value_domain_is_acquired(value_domain));
  *out_table = (loom_low_placement_table_t){0};
  IREE_ASSERT(value_domain->value_count == liveness->value_count &&
                  value_domain->value_ids == liveness->value_ids,
              "low placement requires liveness over the same local value "
              "domain");

  loom_low_placement_build_state_t state = {
      .module = module,
      .region = region,
      .value_domain = value_domain,
      .liveness = liveness,
      .arena = arena,
  };
  if (value_domain->value_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, value_domain->value_count,
                                  sizeof(*state.ranges_by_result_ordinal),
                                  (void**)&state.ranges_by_result_ordinal));
    memset(state.ranges_by_result_ordinal, 0,
           value_domain->value_count * sizeof(*state.ranges_by_result_ordinal));
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, value_domain->value_count,
                                  sizeof(*state.ranges_by_source_ordinal),
                                  (void**)&state.ranges_by_source_ordinal));
    memset(state.ranges_by_source_ordinal, 0,
           value_domain->value_count * sizeof(*state.ranges_by_source_ordinal));
  }

  IREE_RETURN_IF_ERROR(loom_low_placement_visit_ops(
      &state, loom_low_placement_count_op_relations));
  const iree_host_size_t relation_count = state.relation_count;
  if (relation_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, relation_count,
                                                   sizeof(*state.relations),
                                                   (void**)&state.relations));
    memset(state.relations, 0, relation_count * sizeof(*state.relations));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, relation_count,
        sizeof(*state.relation_indices_by_source_ordinal),
        (void**)&state.relation_indices_by_source_ordinal));
    memset(state.relation_indices_by_source_ordinal, 0,
           relation_count * sizeof(*state.relation_indices_by_source_ordinal));
  }
  loom_low_placement_prefix_ranges(&state);
  IREE_RETURN_IF_ERROR(loom_low_placement_visit_ops(
      &state, loom_low_placement_append_op_relations));
  IREE_ASSERT_EQ(state.appended_relation_count, relation_count);
  IREE_ASSERT_EQ(state.appended_source_relation_count, relation_count);

  *out_table = (loom_low_placement_table_t){
      .module = module,
      .region = region,
      .value_ids = liveness->value_ids,
      .value_count = (loom_value_ordinal_t)liveness->value_count,
      .relations = state.relations,
      .relation_count = relation_count,
      .ranges_by_result_ordinal = state.ranges_by_result_ordinal,
      .relation_indices_by_source_ordinal =
          state.relation_indices_by_source_ordinal,
      .ranges_by_source_ordinal = state.ranges_by_source_ordinal,
  };
  return iree_ok_status();
}

loom_low_placement_relation_range_t
loom_low_placement_relation_range_for_value_ordinal(
    const loom_low_placement_table_t* table,
    loom_value_ordinal_t result_ordinal) {
  IREE_ASSERT_LT(result_ordinal, table->value_count);
  IREE_ASSERT(table->ranges_by_result_ordinal != NULL);
  return table->ranges_by_result_ordinal[result_ordinal];
}

loom_low_placement_relation_range_t
loom_low_placement_relation_range_for_source_value_ordinal(
    const loom_low_placement_table_t* table,
    loom_value_ordinal_t source_ordinal) {
  IREE_ASSERT_LT(source_ordinal, table->value_count);
  IREE_ASSERT(table->ranges_by_source_ordinal != NULL);
  return table->ranges_by_source_ordinal[source_ordinal];
}

loom_value_id_t loom_low_placement_value_id(
    const loom_low_placement_table_t* table,
    loom_value_ordinal_t value_ordinal) {
  IREE_ASSERT_LT(value_ordinal, table->value_count);
  return table->value_ids[value_ordinal];
}
