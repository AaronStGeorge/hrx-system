// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/register_class_map.h"
#include "loom/codegen/low/requirements.h"
#include "loom/error/error_defs.h"
#include "loom/ir/context.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"

#define LOOM_LOW_SCHEDULE_MAX_PRESSURE_CONTRIBUTORS 16

typedef struct loom_low_schedule_build_state_t {
  // Module containing the low function being scheduled.
  const loom_module_t* module;
  // Scheduler options provided by the caller.
  const loom_low_schedule_options_t* options;
  // Arena owning all sidecar storage produced by this schedule.
  iree_arena_allocator_t* arena;
  // low.func.def operation being scheduled.
  const loom_op_t* function_op;
  // Body region of function_op.
  loom_region_t* body;
  // Resolved target records and descriptor set for the low function.
  loom_low_resolved_target_t target;
  // Descriptor register-class lookup map for module register types.
  loom_low_register_class_map_t register_class_map;
  // Schedule block records indexed by region block ordinal.
  loom_low_schedule_block_t* blocks;
  // Schedule node records indexed by scheduler node ordinal.
  loom_low_schedule_node_t* nodes;
  // Dependency records accumulated while building the schedule DAG.
  loom_low_schedule_dependency_t* dependencies;
  // Node indices in final scheduled order.
  uint32_t* scheduled_node_indices;
  // Pressure-model steps in scheduled order for pressure strategy runs.
  loom_low_schedule_pressure_step_t* pressure_steps;
  // Pressure candidate decisions in scheduled order when requested.
  loom_low_schedule_candidate_decision_t* candidate_decisions;
  // Descriptor resource uses in scheduled order.
  loom_low_schedule_resource_use_t* resource_uses;
  // Descriptor effects in scheduled order.
  loom_low_schedule_effect_use_t* effect_uses;
  // Descriptor hazard rows in scheduled order.
  loom_low_schedule_hazard_use_t* hazard_uses;
  // Minimum-distance hazard gaps in scheduled order.
  loom_low_schedule_hazard_gap_t* hazard_gaps;
  // Schedule-class model quality summaries in schedule-class order.
  loom_low_schedule_model_summary_t* model_summaries;
  // Most recent producer state for each minimum-distance hazard key.
  struct loom_low_schedule_hazard_state_t* hazard_states;
  // Per-resource aggregate resource pressure, dense by descriptor resource id
  // until compacted after scheduling.
  loom_low_schedule_resource_summary_t* resource_summaries;
  // Producer node index for each module value, or none for block arguments.
  uint32_t* value_node_indices;
  // Number of populated dependency records.
  iree_host_size_t dependency_count;
  // Allocated dependency record capacity.
  iree_host_size_t dependency_capacity;
  // Number of populated scheduled_node_indices entries.
  iree_host_size_t scheduled_node_count;
  // Number of populated pressure_steps entries.
  iree_host_size_t pressure_step_count;
  // Number of populated candidate_decisions entries.
  iree_host_size_t candidate_decision_count;
  // Number of populated resource_uses entries.
  iree_host_size_t resource_use_count;
  // Number of populated effect_uses entries.
  iree_host_size_t effect_use_count;
  // Number of populated hazard_uses entries.
  iree_host_size_t hazard_use_count;
  // Number of populated hazard_gaps entries.
  iree_host_size_t hazard_gap_count;
  // Number of populated model_summaries entries.
  iree_host_size_t model_summary_count;
  // Number of populated hazard_states entries.
  iree_host_size_t hazard_state_count;
  // Number of populated resource_summaries entries after compaction.
  iree_host_size_t resource_summary_count;
  // Allocated resource-use record capacity.
  iree_host_size_t resource_use_capacity;
  // Allocated effect-use record capacity.
  iree_host_size_t effect_use_capacity;
  // Allocated hazard-use record capacity.
  iree_host_size_t hazard_use_capacity;
  // Allocated hazard-gap record capacity.
  iree_host_size_t hazard_gap_capacity;
  // Allocated hazard-state record capacity.
  iree_host_size_t hazard_state_capacity;
} loom_low_schedule_build_state_t;

typedef struct loom_low_schedule_hazard_state_t {
  // Hazard kind tracked by this state.
  loom_low_hazard_kind_t kind;
  // Interpretation of reference_id.
  loom_low_hazard_reference_kind_t reference_kind;
  // Resource, counter, or target-owned hazard identifier.
  uint16_t reference_id;
  // Producer stage published by the most recent matching scheduled hazard.
  uint16_t producer_stage;
  // Region block containing the most recent producer.
  uint32_t block_index;
  // Most recent producer node index.
  uint32_t node_index;
  // Scheduled ordinal of the most recent producer.
  uint32_t scheduled_ordinal;
  // Hazard row ordinal within the producer node's schedule class.
  uint16_t hazard_ordinal;
  // Required minimum distance published by the producer row.
  uint16_t distance;
  // Hazard flags published by the producer row.
  loom_low_hazard_flags_t hazard_flags;
} loom_low_schedule_hazard_state_t;

typedef struct loom_low_schedule_pressure_state_t {
  // Remaining operand uses for each module value in the current block.
  uint32_t* remaining_use_counts;
  // True when a module value is currently live in the simulated schedule.
  bool* live_values;
  // Current aggregate live register units in the simulated schedule.
  uint64_t current_live_units;
} loom_low_schedule_pressure_state_t;

typedef struct loom_low_schedule_candidate_score_t {
  // Aggregate live register units after scheduling the candidate.
  uint64_t projected_live_units;
  // Live register units whose last use is the candidate.
  uint64_t killed_live_units;
  // Register result units made live by the candidate.
  uint64_t produced_live_units;
  // Source-order tie breaker.
  uint32_t source_ordinal;
} loom_low_schedule_candidate_score_t;

static iree_status_t loom_low_schedule_emit(
    loom_low_schedule_build_state_t* state, const loom_op_t* op,
    const loom_error_def_t* error, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count) {
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(state->options->emitter, &emission);
}

static bool loom_low_schedule_op_is_descriptor_packet(const loom_op_t* op) {
  return loom_low_op_isa(op) || loom_low_const_isa(op);
}

static bool loom_low_schedule_op_is_terminator(const loom_module_t* module,
                                               const loom_op_t* op) {
  return iree_any_bit_set(loom_op_effective_traits(module, op),
                          LOOM_TRAIT_TERMINATOR);
}

static bool loom_low_schedule_node_has_effects(
    const loom_low_schedule_node_t* node,
    const loom_low_descriptor_t* descriptor) {
  if (descriptor) {
    return descriptor->effect_count != 0 ||
           iree_any_bit_set(descriptor->flags,
                            LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING |
                                LOOM_LOW_DESCRIPTOR_FLAG_TERMINATOR);
  }
  return iree_any_bit_set(node->traits, LOOM_TRAIT_READS_MEMORY |
                                            LOOM_TRAIT_WRITES_MEMORY |
                                            LOOM_TRAIT_NON_DETERMINISTIC |
                                            LOOM_TRAIT_UNKNOWN_EFFECTS);
}

static iree_status_t loom_low_schedule_emit_missing_descriptor(
    loom_low_schedule_build_state_t* state, const loom_op_t* op,
    iree_string_view_t opcode) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(
          loom_low_diagnostic_function_name(state->module, state->function_op)),
      loom_param_string(opcode),
      loom_param_string(state->target.descriptor_set_key),
  };
  return loom_low_schedule_emit(
      state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 4), params,
      IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_schedule_emit_missing_schedule_class(
    loom_low_schedule_build_state_t* state, const loom_op_t* op,
    iree_string_view_t opcode) {
  iree_string_view_t op_name = loom_op_name(state->module, op);
  loom_diagnostic_param_t params[] = {
      loom_param_string(op_name),
      loom_param_string(IREE_SV("low-schedule")),
      loom_param_string(IREE_SV("descriptor has no schedule class")),
  };
  (void)opcode;
  return loom_low_schedule_emit(
      state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 1), params,
      IREE_ARRAYSIZE(params));
}

static bool loom_low_schedule_interval_contains_point(
    const loom_liveness_interval_t* interval, uint32_t point) {
  return interval->start_point <= point && point < interval->end_point;
}

static iree_status_t loom_low_schedule_pressure_budget_for_class(
    const loom_low_schedule_build_state_t* state,
    loom_liveness_value_class_t value_class, uint32_t* out_budget,
    bool* out_has_budget) {
  *out_budget = 0;
  *out_has_budget = false;
  if (value_class.type_kind != LOOM_TYPE_REGISTER ||
      !state->target.descriptor_set) {
    return iree_ok_status();
  }
  uint16_t reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  const loom_low_reg_class_t* reg_class = NULL;
  bool found_reg_class = false;
  IREE_RETURN_IF_ERROR(loom_low_register_class_map_try_resolve_string_id(
      &state->register_class_map, value_class.register_class_id, &reg_class_id,
      &reg_class, &found_reg_class));
  if (!found_reg_class) {
    return iree_ok_status();
  }
  (void)reg_class_id;
  if (reg_class->physical_count == 0) {
    return iree_ok_status();
  }
  *out_budget = reg_class->physical_count;
  *out_has_budget = true;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_collect_pressure_contributors(
    loom_low_schedule_build_state_t* state,
    const loom_liveness_analysis_t* liveness,
    const loom_liveness_pressure_summary_t* summary,
    const iree_string_view_t** out_contributors,
    iree_host_size_t* out_contributor_count) {
  iree_string_view_t* contributors = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, LOOM_LOW_SCHEDULE_MAX_PRESSURE_CONTRIBUTORS,
      sizeof(*contributors), (void**)&contributors));
  iree_host_size_t contributor_count = 0;
  bool overflowed = false;
  for (uint32_t point_attempt = 0; point_attempt < 2; ++point_attempt) {
    if (point_attempt != 0 && contributor_count != 0) break;
    if (point_attempt != 0 && summary->peak_point == UINT32_MAX) break;
    uint32_t point = summary->peak_point + point_attempt;
    for (iree_host_size_t i = 0; i < liveness->interval_count; ++i) {
      const loom_liveness_interval_t* interval = &liveness->intervals[i];
      if (!loom_liveness_value_class_equal(interval->value_class,
                                           summary->value_class) ||
          !loom_low_schedule_interval_contains_point(interval, point)) {
        continue;
      }
      if (contributor_count < LOOM_LOW_SCHEDULE_MAX_PRESSURE_CONTRIBUTORS) {
        contributors[contributor_count++] =
            loom_low_diagnostic_value_name(state->module, interval->value_id);
      } else {
        overflowed = true;
      }
    }
  }
  if (overflowed &&
      contributor_count == LOOM_LOW_SCHEDULE_MAX_PRESSURE_CONTRIBUTORS) {
    contributors[contributor_count - 1] = IREE_SV("...");
  }
  if (contributor_count == 0) {
    contributors[contributor_count++] = IREE_SV("<none>");
  }
  *out_contributors = contributors;
  *out_contributor_count = contributor_count;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_emit_pressure_summary(
    loom_low_schedule_build_state_t* state,
    const loom_liveness_analysis_t* liveness,
    const loom_liveness_pressure_summary_t* summary) {
  uint32_t budget = 0;
  bool has_budget = false;
  IREE_RETURN_IF_ERROR(loom_low_schedule_pressure_budget_for_class(
      state, summary->value_class, &budget, &has_budget));
  if (!has_budget) {
    return iree_ok_status();
  }

  const iree_string_view_t* contributors = NULL;
  iree_host_size_t contributor_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_schedule_collect_pressure_contributors(
      state, liveness, summary, &contributors, &contributor_count));

  const loom_op_t* origin_op =
      summary->peak_op ? summary->peak_op : state->function_op;
  iree_string_view_t operation_name =
      summary->peak_op ? loom_op_name(state->module, summary->peak_op)
                       : IREE_SV("<block-boundary>");
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&state->target)),
      loom_param_string(loom_low_diagnostic_export_name(&state->target)),
      loom_param_string(loom_low_diagnostic_config_key(&state->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(state->module, state->function_op)),
      loom_param_string(loom_low_diagnostic_value_class_name(
          state->module, summary->value_class)),
      loom_param_u32(budget),
      loom_param_u32(summary->peak_live_units),
      loom_param_string(
          loom_low_diagnostic_block_name(state->module, summary->peak_block)),
      loom_param_string(operation_name),
      loom_param_string_list(contributors, contributor_count),
  };
  return loom_low_schedule_emit(
      state, origin_op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_BACKEND, 3),
      params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_schedule_emit_pressure_diagnostics(
    loom_low_schedule_build_state_t* state,
    const loom_liveness_analysis_t* liveness) {
  for (iree_host_size_t i = 0; i < liveness->pressure_summary_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_emit_pressure_summary(
        state, liveness, &liveness->pressure_summaries[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_node_diagnostic_label(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node, iree_string_view_t* out_label) {
  IREE_ASSERT_ARGUMENT(out_label);
  *out_label = IREE_SV("<unknown>");
  if (node == NULL) {
    return iree_ok_status();
  }
  if (node->descriptor_ordinal != LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    const loom_low_descriptor_t* descriptor =
        loom_low_descriptor_set_descriptor_at(state->target.descriptor_set,
                                              node->descriptor_ordinal);
    if (descriptor != NULL) {
      return loom_low_descriptor_set_string(state->target.descriptor_set,
                                            descriptor->key_string_offset,
                                            out_label);
    }
  }
  *out_label = loom_op_name(state->module, node->op);
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_emit_candidate_decision(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_candidate_decision_t* decision) {
  if (decision->rejected_node == LOOM_LOW_SCHEDULE_NODE_NONE) {
    return iree_ok_status();
  }
  const loom_low_schedule_node_t* chosen_node = NULL;
  if (decision->chosen_node < state->scheduled_node_count) {
    chosen_node = &state->nodes[decision->chosen_node];
  }
  const loom_low_schedule_node_t* rejected_node = NULL;
  if (decision->rejected_node < state->scheduled_node_count) {
    rejected_node = &state->nodes[decision->rejected_node];
  }
  const loom_block_t* block = NULL;
  if (decision->block_index < state->body->block_count) {
    block = state->blocks[decision->block_index].block;
  }
  iree_string_view_t chosen_label = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_schedule_node_diagnostic_label(
      state, chosen_node, &chosen_label));
  iree_string_view_t rejected_label = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_schedule_node_diagnostic_label(
      state, rejected_node, &rejected_label));
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&state->target)),
      loom_param_string(loom_low_diagnostic_export_name(&state->target)),
      loom_param_string(loom_low_diagnostic_config_key(&state->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(state->module, state->function_op)),
      loom_param_string(loom_low_diagnostic_block_name(state->module, block)),
      loom_param_u32(decision->scheduled_ordinal),
      loom_param_u32(decision->ready_candidate_count),
      loom_param_string(chosen_label),
      loom_param_string(rejected_label),
      loom_param_u64(decision->chosen_projected_live_units),
      loom_param_u64(decision->chosen_killed_live_units),
      loom_param_u64(decision->chosen_produced_live_units),
      loom_param_u64(decision->rejected_projected_live_units),
      loom_param_u64(decision->rejected_killed_live_units),
      loom_param_u64(decision->rejected_produced_live_units),
  };
  const loom_op_t* origin_op =
      chosen_node && chosen_node->op ? chosen_node->op : state->function_op;
  return loom_low_schedule_emit(
      state, origin_op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_BACKEND, 15),
      params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_schedule_emit_candidate_decision_diagnostics(
    loom_low_schedule_build_state_t* state) {
  for (iree_host_size_t i = 0; i < state->candidate_decision_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_emit_candidate_decision(
        state, &state->candidate_decisions[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_emit_model_summary(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_model_summary_t* summary) {
  if (summary->model_quality == LOOM_LOW_MODEL_QUALITY_EXACT) {
    return iree_ok_status();
  }
  const loom_low_schedule_node_t* first_node = NULL;
  if (summary->first_node < state->scheduled_node_count) {
    first_node = &state->nodes[summary->first_node];
  }
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&state->target)),
      loom_param_string(loom_low_diagnostic_export_name(&state->target)),
      loom_param_string(loom_low_diagnostic_config_key(&state->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(state->module, state->function_op)),
      loom_param_string(summary->schedule_class_name),
      loom_param_string(loom_low_model_quality_name(summary->model_quality)),
      loom_param_string(loom_low_latency_kind_name(summary->latency_kind)),
      loom_param_u32(summary->latency_cycles),
      loom_param_u32(summary->issue_use_count),
      loom_param_u32(summary->hazard_count),
      loom_param_u32(summary->use_count),
  };
  const loom_op_t* origin_op =
      first_node && first_node->op ? first_node->op : state->function_op;
  return loom_low_schedule_emit(
      state, origin_op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_BACKEND, 16),
      params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_schedule_emit_model_diagnostics(
    loom_low_schedule_build_state_t* state) {
  for (iree_host_size_t i = 0; i < state->model_summary_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_emit_model_summary(
        state, &state->model_summaries[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_emit_resource_bottleneck(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_resource_summary_t* summary) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&state->target)),
      loom_param_string(loom_low_diagnostic_export_name(&state->target)),
      loom_param_string(loom_low_diagnostic_config_key(&state->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(state->module, state->function_op)),
      loom_param_string(summary->resource_name),
      loom_param_u32(summary->capacity_per_cycle),
      loom_param_u32(summary->contention_group_id),
      loom_param_u32(summary->use_count),
      loom_param_u64(summary->total_unit_cycles),
      loom_param_u64(summary->estimated_min_cycles),
      loom_param_u32(summary->peak_units_per_cycle),
  };
  return loom_low_schedule_emit(
      state, state->function_op,
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_BACKEND, 13), params,
      IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_schedule_emit_resource_diagnostics(
    loom_low_schedule_build_state_t* state) {
  uint64_t maximum_estimated_min_cycles = 0;
  for (iree_host_size_t i = 0; i < state->resource_summary_count; ++i) {
    const loom_low_schedule_resource_summary_t* summary =
        &state->resource_summaries[i];
    if (summary->estimated_min_cycles > maximum_estimated_min_cycles) {
      maximum_estimated_min_cycles = summary->estimated_min_cycles;
    }
  }
  if (maximum_estimated_min_cycles == 0) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < state->resource_summary_count; ++i) {
    const loom_low_schedule_resource_summary_t* summary =
        &state->resource_summaries[i];
    if (summary->estimated_min_cycles != maximum_estimated_min_cycles) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_emit_resource_bottleneck(state, summary));
  }
  return iree_ok_status();
}

static uint32_t loom_low_schedule_hazard_gap_packet_index(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_hazard_gap_t* hazard_gap,
    uint32_t scheduled_ordinal) {
  if (hazard_gap->block_index >= state->body->block_count) {
    return UINT32_MAX;
  }
  const loom_low_schedule_block_t* block =
      &state->blocks[hazard_gap->block_index];
  const uint64_t packet_index =
      (uint64_t)block->scheduled_node_start + scheduled_ordinal;
  return packet_index <= UINT32_MAX ? (uint32_t)packet_index : UINT32_MAX;
}

static iree_string_view_t loom_low_schedule_hazard_reference_name(
    const loom_low_schedule_hazard_gap_t* hazard_gap, char* buffer,
    iree_host_size_t buffer_capacity) {
  if (!iree_string_view_is_empty(hazard_gap->resource_name)) {
    return hazard_gap->resource_name;
  }
  if (buffer_capacity == 0) {
    return IREE_SV("<unknown>");
  }
  iree_string_view_t reference_kind =
      loom_low_hazard_reference_kind_name(hazard_gap->reference_kind);
  int length = iree_snprintf(buffer, buffer_capacity, "%.*s:%" PRIu16,
                             (int)reference_kind.size, reference_kind.data,
                             hazard_gap->reference_id);
  if (length < 0) {
    return IREE_SV("<unknown>");
  }
  iree_host_size_t size = (iree_host_size_t)length;
  if (size >= buffer_capacity) {
    size = buffer_capacity - 1;
  }
  return iree_make_string_view(buffer, size);
}

static iree_status_t loom_low_schedule_emit_hazard_gap(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_hazard_gap_t* hazard_gap) {
  const loom_low_schedule_node_t* consumer_node = NULL;
  if (hazard_gap->consumer_node < state->scheduled_node_count) {
    consumer_node = &state->nodes[hazard_gap->consumer_node];
  }
  iree_string_view_t descriptor_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_schedule_node_diagnostic_label(
      state, consumer_node, &descriptor_key));
  char reference_buffer[32];
  iree_string_view_t reference_name = loom_low_schedule_hazard_reference_name(
      hazard_gap, reference_buffer, sizeof(reference_buffer));
  uint32_t producer_packet = loom_low_schedule_hazard_gap_packet_index(
      state, hazard_gap, hazard_gap->producer_scheduled_ordinal);
  uint32_t consumer_packet = loom_low_schedule_hazard_gap_packet_index(
      state, hazard_gap, hazard_gap->consumer_scheduled_ordinal);
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&state->target)),
      loom_param_string(loom_low_diagnostic_export_name(&state->target)),
      loom_param_string(loom_low_diagnostic_config_key(&state->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(state->module, state->function_op)),
      loom_param_string(loom_low_diagnostic_string_or_placeholder(
          descriptor_key, IREE_SV("<unknown>"))),
      loom_param_string(loom_low_hazard_kind_name(hazard_gap->kind)),
      loom_param_string(
          loom_low_hazard_reference_kind_name(hazard_gap->reference_kind)),
      loom_param_string(reference_name),
      loom_param_u32(hazard_gap->required_distance),
      loom_param_u32(hazard_gap->actual_distance),
      loom_param_u32(hazard_gap->required_delay),
      loom_param_u32(producer_packet),
      loom_param_u32(consumer_packet),
  };
  const loom_op_t* origin_op =
      consumer_node ? consumer_node->op : state->function_op;
  return loom_low_schedule_emit(
      state, origin_op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_BACKEND, 14),
      params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_schedule_emit_hazard_gap_diagnostics(
    loom_low_schedule_build_state_t* state) {
  for (iree_host_size_t i = 0; i < state->hazard_gap_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_emit_hazard_gap(state, &state->hazard_gaps[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_resolve_descriptor(
    loom_low_schedule_build_state_t* state, const loom_op_t* op,
    loom_low_schedule_node_t* node,
    const loom_low_descriptor_t** out_descriptor) {
  *out_descriptor = NULL;
  if (!loom_low_schedule_op_is_descriptor_packet(op)) {
    return iree_ok_status();
  }

  loom_low_resolved_descriptor_packet_t packet = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_descriptor_packet(
      state->module, &state->target, op, &packet));
  node->descriptor_id = packet.stable_id;
  if (packet.descriptor == NULL) {
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_emit_missing_descriptor(state, op, packet.key));
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low schedule descriptor '%.*s' is not available",
                            (int)packet.key.size, packet.key.data);
  }

  node->descriptor_ordinal = packet.descriptor_ordinal;
  node->effect_count = packet.descriptor->effect_count;
  node->schedule_class_id = packet.descriptor->schedule_class_id;
  if (packet.descriptor->schedule_class_id == LOOM_LOW_SCHEDULE_CLASS_NONE ||
      packet.descriptor->schedule_class_id >=
          state->target.descriptor_set->schedule_class_count) {
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_emit_missing_schedule_class(state, op, packet.key));
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low schedule descriptor '%.*s' has no usable schedule class",
        (int)packet.key.size, packet.key.data);
  }

  const loom_low_schedule_class_t* schedule_class =
      &state->target.descriptor_set
           ->schedule_classes[packet.descriptor->schedule_class_id];
  node->latency_cycles = schedule_class->latency_cycles;
  node->latency_kind = schedule_class->latency_kind;
  node->model_quality = schedule_class->model_quality;
  node->issue_use_count = schedule_class->issue_use_count;
  node->hazard_count = schedule_class->hazard_count;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
      state->target.descriptor_set, schedule_class->name_string_offset,
      &node->schedule_class_name));
  *out_descriptor = packet.descriptor;
  return iree_ok_status();
}

static bool loom_low_schedule_dependency_equal(
    const loom_low_schedule_dependency_t* dependency, uint32_t producer_node,
    uint32_t consumer_node, loom_low_schedule_dependency_kind_t kind,
    uint32_t operand_index) {
  return dependency->producer_node == producer_node &&
         dependency->consumer_node == consumer_node &&
         dependency->kind == kind && dependency->operand_index == operand_index;
}

static iree_status_t loom_low_schedule_add_dependency(
    loom_low_schedule_build_state_t* state, uint32_t producer_node,
    uint32_t consumer_node, loom_low_schedule_dependency_kind_t kind,
    uint32_t operand_index) {
  if (producer_node == consumer_node) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < state->dependency_count; ++i) {
    if (loom_low_schedule_dependency_equal(&state->dependencies[i],
                                           producer_node, consumer_node, kind,
                                           operand_index)) {
      return iree_ok_status();
    }
  }
  if (state->dependency_count >= state->dependency_capacity) {
    iree_host_size_t new_capacity =
        state->dependency_capacity == 0 ? 16 : state->dependency_capacity * 2;
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(state->arena, state->dependency_count,
                              new_capacity, sizeof(*state->dependencies),
                              &new_capacity, (void**)&state->dependencies));
    state->dependency_capacity = new_capacity;
  }
  state->dependencies[state->dependency_count++] =
      (loom_low_schedule_dependency_t){
          .producer_node = producer_node,
          .consumer_node = consumer_node,
          .kind = kind,
          .operand_index = operand_index,
      };
  return iree_ok_status();
}

static void loom_low_schedule_count_nodes(const loom_region_t* body,
                                          iree_host_size_t* out_node_count) {
  iree_host_size_t node_count = 0;
  const loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) { node_count += block->op_count; }
  *out_node_count = node_count;
}

static iree_status_t loom_low_schedule_initialize_storage(
    loom_low_schedule_build_state_t* state, iree_host_size_t node_count) {
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, state->body->block_count, sizeof(*state->blocks),
      (void**)&state->blocks));
  memset(state->blocks, 0, state->body->block_count * sizeof(*state->blocks));
  if (node_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(state->arena, node_count,
                                                   sizeof(*state->nodes),
                                                   (void**)&state->nodes));
    memset(state->nodes, 0, node_count * sizeof(*state->nodes));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, node_count, sizeof(*state->scheduled_node_indices),
        (void**)&state->scheduled_node_indices));
    if (state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          state->arena, node_count, sizeof(*state->pressure_steps),
          (void**)&state->pressure_steps));
    }
    if (state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE &&
        iree_any_bit_set(state->options->diagnostic_flags,
                         LOOM_LOW_SCHEDULE_DIAGNOSTIC_CANDIDATE_DECISIONS)) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          state->arena, node_count, sizeof(*state->candidate_decisions),
          (void**)&state->candidate_decisions));
    }
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, state->module->values.count,
      sizeof(*state->value_node_indices), (void**)&state->value_node_indices));
  for (iree_host_size_t i = 0; i < state->module->values.count; ++i) {
    state->value_node_indices[i] = LOOM_LOW_SCHEDULE_NODE_NONE;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_initialize_descriptor_sidecars(
    loom_low_schedule_build_state_t* state, iree_host_size_t node_count) {
  iree_host_size_t resource_use_capacity = 0;
  iree_host_size_t effect_use_capacity = 0;
  iree_host_size_t hazard_use_capacity = 0;
  for (iree_host_size_t node_index = 0; node_index < node_count; ++node_index) {
    if (!iree_host_size_checked_add(resource_use_capacity,
                                    state->nodes[node_index].issue_use_count,
                                    &resource_use_capacity)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "low schedule resource-use capacity exceeds host size");
    }
    if (!iree_host_size_checked_add(effect_use_capacity,
                                    state->nodes[node_index].effect_count,
                                    &effect_use_capacity)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "low schedule effect-use capacity exceeds host size");
    }
    if (!iree_host_size_checked_add(hazard_use_capacity,
                                    state->nodes[node_index].hazard_count,
                                    &hazard_use_capacity)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low schedule hazard-use capacity exceeds host "
                              "size");
    }
  }
  if (node_count != 0 &&
      state->target.descriptor_set->schedule_class_count != 0) {
    const uint32_t schedule_class_count =
        state->target.descriptor_set->schedule_class_count;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, schedule_class_count, sizeof(*state->model_summaries),
        (void**)&state->model_summaries));
    memset(state->model_summaries, 0,
           schedule_class_count * sizeof(*state->model_summaries));
  }
  if (resource_use_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, resource_use_capacity, sizeof(*state->resource_uses),
        (void**)&state->resource_uses));
    state->resource_use_capacity = resource_use_capacity;
    const iree_host_size_t resource_count =
        state->target.descriptor_set->resource_count;
    if (resource_count > UINT16_MAX) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low schedule resource summary count exceeds uint16_t");
    }
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, resource_count, sizeof(*state->resource_summaries),
        (void**)&state->resource_summaries));
    memset(state->resource_summaries, 0,
           resource_count * sizeof(*state->resource_summaries));
    for (iree_host_size_t i = 0; i < resource_count; ++i) {
      const loom_low_resource_t* resource =
          &state->target.descriptor_set->resources[i];
      if (resource->capacity_per_cycle == 0) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "low schedule resource summary cannot use zero capacity");
      }
      iree_string_view_t resource_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
          state->target.descriptor_set, resource->name_string_offset,
          &resource_name));
      state->resource_summaries[i] = (loom_low_schedule_resource_summary_t){
          .resource_id = (uint16_t)i,
          .resource_name = resource_name,
          .resource_kind = resource->kind,
          .resource_flags = resource->flags,
          .capacity_per_cycle = resource->capacity_per_cycle,
          .contention_group_id = resource->contention_group_id,
      };
    }
  }
  if (effect_use_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, effect_use_capacity, sizeof(*state->effect_uses),
        (void**)&state->effect_uses));
    state->effect_use_capacity = effect_use_capacity;
  }
  if (hazard_use_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, hazard_use_capacity, sizeof(*state->hazard_uses),
        (void**)&state->hazard_uses));
    state->hazard_use_capacity = hazard_use_capacity;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_fill_nodes(
    loom_low_schedule_build_state_t* state) {
  uint32_t next_node_index = 0;
  for (uint16_t block_index = 0; block_index < state->body->block_count;
       ++block_index) {
    loom_block_t* block = state->body->blocks[block_index];
    if (!block) {
      continue;
    }
    state->blocks[block_index] = (loom_low_schedule_block_t){
        .block = block,
        .node_start = next_node_index,
        .node_count = block->op_count,
        .scheduled_node_start = next_node_index,
        .scheduled_node_count = block->op_count,
    };

    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      loom_low_schedule_node_t* node = &state->nodes[next_node_index];
      *node = (loom_low_schedule_node_t){
          .op = op,
          .block = block,
          .block_index = block_index,
          .source_ordinal = next_node_index,
          .scheduled_ordinal = LOOM_LOW_SCHEDULE_NODE_NONE,
          .kind = LOOM_LOW_SCHEDULE_NODE_STRUCTURAL,
          .traits = loom_op_effective_traits(state->module, op),
          .descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE,
          .descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE,
          .schedule_class_id = LOOM_LOW_SCHEDULE_CLASS_NONE,
      };
      if (loom_low_schedule_op_is_terminator(state->module, op)) {
        node->kind = LOOM_LOW_SCHEDULE_NODE_TERMINATOR;
      } else if (loom_low_schedule_op_is_descriptor_packet(op)) {
        node->kind = LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR;
      }

      const loom_low_descriptor_t* descriptor = NULL;
      IREE_RETURN_IF_ERROR(
          loom_low_schedule_resolve_descriptor(state, op, node, &descriptor));

      const loom_value_id_t* results = loom_op_const_results(op);
      for (uint16_t result_index = 0; result_index < op->result_count;
           ++result_index) {
        loom_value_id_t value_id = results[result_index];
        if (value_id < state->module->values.count) {
          state->value_node_indices[value_id] = next_node_index;
        }
      }
      ++next_node_index;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_build_dependencies(
    loom_low_schedule_build_state_t* state) {
  for (iree_host_size_t block_index = 0; block_index < state->body->block_count;
       ++block_index) {
    const loom_low_schedule_block_t* block_record = &state->blocks[block_index];
    uint32_t last_effect_node = LOOM_LOW_SCHEDULE_NODE_NONE;
    uint32_t last_live_in_node = LOOM_LOW_SCHEDULE_NODE_NONE;
    bool live_in_preamble_open = true;
    const uint32_t block_node_end =
        block_record->node_start + block_record->node_count;
    for (uint32_t node_index = block_record->node_start;
         node_index < block_node_end; ++node_index) {
      const loom_low_schedule_node_t* node = &state->nodes[node_index];
      const loom_op_t* op = node->op;
      if (loom_low_live_in_isa(op)) {
        if (block_index != 0 || !live_in_preamble_open) {
          return iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "low schedule requires low.live_in packets in the entry "
              "preamble");
        }
        if (last_live_in_node != LOOM_LOW_SCHEDULE_NODE_NONE) {
          IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
              state, last_live_in_node, node_index,
              LOOM_LOW_SCHEDULE_DEPENDENCY_ANCHOR, UINT32_MAX));
        }
        last_live_in_node = node_index;
      } else {
        live_in_preamble_open = false;
        if (last_live_in_node != LOOM_LOW_SCHEDULE_NODE_NONE) {
          IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
              state, last_live_in_node, node_index,
              LOOM_LOW_SCHEDULE_DEPENDENCY_ANCHOR, UINT32_MAX));
        }
      }

      const loom_value_id_t* operands = loom_op_const_operands(op);
      for (uint16_t operand_index = 0; operand_index < op->operand_count;
           ++operand_index) {
        loom_value_id_t operand_value = operands[operand_index];
        if (operand_value >= state->module->values.count) continue;
        uint32_t producer_node = state->value_node_indices[operand_value];
        if (producer_node == LOOM_LOW_SCHEDULE_NODE_NONE) continue;
        if (state->nodes[producer_node].block != node->block) continue;
        IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
            state, producer_node, node_index, LOOM_LOW_SCHEDULE_DEPENDENCY_SSA,
            operand_index));
      }

      const loom_low_descriptor_t* descriptor = NULL;
      if (node->descriptor_ordinal != LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
        descriptor = loom_low_descriptor_set_descriptor_at(
            state->target.descriptor_set, node->descriptor_ordinal);
      }
      if (loom_low_schedule_node_has_effects(node, descriptor)) {
        if (last_effect_node != LOOM_LOW_SCHEDULE_NODE_NONE) {
          IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
              state, last_effect_node, node_index,
              LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, UINT32_MAX));
        }
        last_effect_node = node_index;
      }

      if (node->kind == LOOM_LOW_SCHEDULE_NODE_TERMINATOR) {
        for (uint32_t predecessor_node = block_record->node_start;
             predecessor_node < node_index; ++predecessor_node) {
          IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
              state, predecessor_node, node_index,
              LOOM_LOW_SCHEDULE_DEPENDENCY_CONTROL, UINT32_MAX));
        }
      }
    }
  }
  return iree_ok_status();
}

static bool loom_low_schedule_uses_pressure_strategy(
    const loom_low_schedule_build_state_t* state) {
  return state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE;
}

static bool loom_low_schedule_strategy_is_valid(
    loom_low_schedule_strategy_t strategy) {
  switch (strategy) {
    case LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY:
    case LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE:
      return true;
    default:
      return false;
  }
}

static uint32_t loom_low_schedule_register_unit_count(
    const loom_liveness_analysis_t* liveness, loom_value_id_t value_id) {
  const loom_liveness_interval_t* interval =
      loom_liveness_interval_for_value(liveness, value_id);
  if (!interval || interval->value_class.type_kind != LOOM_TYPE_REGISTER) {
    return 0;
  }
  return interval->unit_count;
}

static bool loom_low_schedule_operand_repeated_before(
    const loom_value_id_t* operands, uint16_t operand_index) {
  loom_value_id_t value_id = operands[operand_index];
  for (uint16_t previous_operand_index = 0;
       previous_operand_index < operand_index; ++previous_operand_index) {
    if (operands[previous_operand_index] == value_id) {
      return true;
    }
  }
  return false;
}

static uint32_t loom_low_schedule_candidate_operand_use_count(
    const loom_op_t* op, loom_value_id_t value_id) {
  uint32_t use_count = 0;
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t operand_index = 0; operand_index < op->operand_count;
       ++operand_index) {
    use_count += operands[operand_index] == value_id ? 1 : 0;
  }
  return use_count;
}

static iree_status_t loom_low_schedule_allocate_pressure_state(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* out_pressure_state) {
  *out_pressure_state = (loom_low_schedule_pressure_state_t){0};
  if (!loom_low_schedule_uses_pressure_strategy(state)) {
    return iree_ok_status();
  }
  if (state->module->values.count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, state->module->values.count,
      sizeof(*out_pressure_state->remaining_use_counts),
      (void**)&out_pressure_state->remaining_use_counts));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->arena, state->module->values.count,
                                sizeof(*out_pressure_state->live_values),
                                (void**)&out_pressure_state->live_values));
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_initialize_block_pressure(
    loom_low_schedule_build_state_t* state,
    const loom_liveness_analysis_t* liveness,
    const loom_low_schedule_block_t* block_record,
    loom_low_schedule_pressure_state_t* pressure_state) {
  pressure_state->current_live_units = 0;
  if (state->module->values.count == 0) {
    return iree_ok_status();
  }
  memset(pressure_state->remaining_use_counts, 0,
         state->module->values.count *
             sizeof(*pressure_state->remaining_use_counts));
  memset(pressure_state->live_values, 0,
         state->module->values.count * sizeof(*pressure_state->live_values));

  const uint32_t block_node_end =
      block_record->node_start + block_record->node_count;
  for (uint32_t node_index = block_record->node_start;
       node_index < block_node_end; ++node_index) {
    const loom_op_t* op = state->nodes[node_index].op;
    const loom_value_id_t* operands = loom_op_const_operands(op);
    for (uint16_t operand_index = 0; operand_index < op->operand_count;
         ++operand_index) {
      loom_value_id_t value_id = operands[operand_index];
      if (value_id >= state->module->values.count) continue;
      if (pressure_state->remaining_use_counts[value_id] == UINT32_MAX) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "low schedule pressure use count exceeds uint32_t");
      }
      ++pressure_state->remaining_use_counts[value_id];
    }
  }

  for (iree_host_size_t value_id = 0; value_id < state->module->values.count;
       ++value_id) {
    if (pressure_state->remaining_use_counts[value_id] == 0) continue;
    uint32_t producer_node = state->value_node_indices[value_id];
    if (producer_node != LOOM_LOW_SCHEDULE_NODE_NONE &&
        state->nodes[producer_node].block == block_record->block) {
      continue;
    }
    uint32_t unit_count = loom_low_schedule_register_unit_count(
        liveness, (loom_value_id_t)value_id);
    if (unit_count == 0) continue;
    pressure_state->live_values[value_id] = true;
    pressure_state->current_live_units += unit_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_pressure_score_candidate(
    const loom_low_schedule_build_state_t* state,
    const loom_liveness_analysis_t* liveness,
    const loom_low_schedule_pressure_state_t* pressure_state,
    uint32_t node_index, loom_low_schedule_candidate_score_t* out_score) {
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  uint64_t killed_live_units = 0;
  uint64_t produced_live_units = 0;

  const loom_value_id_t* operands = loom_op_const_operands(node->op);
  for (uint16_t operand_index = 0; operand_index < node->op->operand_count;
       ++operand_index) {
    loom_value_id_t value_id = operands[operand_index];
    if (value_id >= state->module->values.count) continue;
    if (loom_low_schedule_operand_repeated_before(operands, operand_index)) {
      continue;
    }
    if (!pressure_state->live_values[value_id]) continue;
    uint32_t candidate_use_count =
        loom_low_schedule_candidate_operand_use_count(node->op, value_id);
    if (pressure_state->remaining_use_counts[value_id] != candidate_use_count) {
      continue;
    }
    killed_live_units +=
        loom_low_schedule_register_unit_count(liveness, value_id);
  }

  const loom_value_id_t* results = loom_op_const_results(node->op);
  for (uint16_t result_index = 0; result_index < node->op->result_count;
       ++result_index) {
    loom_value_id_t value_id = results[result_index];
    if (value_id >= state->module->values.count) continue;
    if (pressure_state->remaining_use_counts[value_id] == 0) continue;
    if (pressure_state->live_values[value_id]) continue;
    produced_live_units +=
        loom_low_schedule_register_unit_count(liveness, value_id);
  }

  if (killed_live_units > pressure_state->current_live_units) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low schedule pressure candidate killed more units than are live");
  }
  uint64_t projected_live_units =
      pressure_state->current_live_units - killed_live_units;
  projected_live_units += produced_live_units;
  *out_score = (loom_low_schedule_candidate_score_t){
      .projected_live_units = projected_live_units,
      .killed_live_units = killed_live_units,
      .produced_live_units = produced_live_units,
      .source_ordinal = node->source_ordinal,
  };
  return iree_ok_status();
}

static bool loom_low_schedule_candidate_score_less(
    loom_low_schedule_candidate_score_t lhs,
    loom_low_schedule_candidate_score_t rhs) {
  if (lhs.projected_live_units != rhs.projected_live_units) {
    return lhs.projected_live_units < rhs.projected_live_units;
  }
  if (lhs.killed_live_units != rhs.killed_live_units) {
    return lhs.killed_live_units > rhs.killed_live_units;
  }
  if (lhs.produced_live_units != rhs.produced_live_units) {
    return lhs.produced_live_units < rhs.produced_live_units;
  }
  return lhs.source_ordinal < rhs.source_ordinal;
}

static void loom_low_schedule_record_candidate_decision(
    loom_low_schedule_build_state_t* state, uint32_t block_index,
    uint32_t scheduled_ordinal, uint32_t ready_candidate_count,
    uint32_t chosen_node, loom_low_schedule_candidate_score_t chosen_score,
    uint32_t rejected_node,
    loom_low_schedule_candidate_score_t rejected_score) {
  if (!state->candidate_decisions) {
    return;
  }
  if (rejected_node == LOOM_LOW_SCHEDULE_NODE_NONE) {
    return;
  }
  state->candidate_decisions[state->candidate_decision_count++] =
      (loom_low_schedule_candidate_decision_t){
          .block_index = block_index,
          .scheduled_ordinal = scheduled_ordinal,
          .ready_candidate_count = ready_candidate_count,
          .chosen_node = chosen_node,
          .rejected_node = rejected_node,
          .chosen_projected_live_units = chosen_score.projected_live_units,
          .chosen_killed_live_units = chosen_score.killed_live_units,
          .chosen_produced_live_units = chosen_score.produced_live_units,
          .rejected_projected_live_units = rejected_score.projected_live_units,
          .rejected_killed_live_units = rejected_score.killed_live_units,
          .rejected_produced_live_units = rejected_score.produced_live_units,
      };
}

static iree_status_t loom_low_schedule_note_pressure_node_scheduled(
    loom_low_schedule_build_state_t* state,
    const loom_liveness_analysis_t* liveness,
    loom_low_schedule_pressure_state_t* pressure_state, uint32_t node_index,
    loom_low_schedule_candidate_score_t score) {
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  uint64_t live_units_before = pressure_state->current_live_units;
  const loom_value_id_t* operands = loom_op_const_operands(node->op);
  for (uint16_t operand_index = 0; operand_index < node->op->operand_count;
       ++operand_index) {
    loom_value_id_t value_id = operands[operand_index];
    if (value_id >= state->module->values.count) continue;
    if (pressure_state->remaining_use_counts[value_id] == 0) continue;
    --pressure_state->remaining_use_counts[value_id];
    if (pressure_state->remaining_use_counts[value_id] == 0 &&
        pressure_state->live_values[value_id]) {
      pressure_state->live_values[value_id] = false;
      uint32_t unit_count =
          loom_low_schedule_register_unit_count(liveness, value_id);
      if (unit_count > pressure_state->current_live_units) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "low schedule pressure accounting underflow for value %u",
            value_id);
      }
      pressure_state->current_live_units -= unit_count;
    }
  }

  const loom_value_id_t* results = loom_op_const_results(node->op);
  for (uint16_t result_index = 0; result_index < node->op->result_count;
       ++result_index) {
    loom_value_id_t value_id = results[result_index];
    if (value_id >= state->module->values.count) continue;
    if (pressure_state->remaining_use_counts[value_id] == 0) continue;
    if (pressure_state->live_values[value_id]) continue;
    pressure_state->live_values[value_id] = true;
    pressure_state->current_live_units +=
        loom_low_schedule_register_unit_count(liveness, value_id);
  }
  if (pressure_state->current_live_units != score.projected_live_units) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low schedule pressure accounting did not match chosen candidate");
  }
  if (state->pressure_steps) {
    state->pressure_steps[state->pressure_step_count++] =
        (loom_low_schedule_pressure_step_t){
            .node_index = node_index,
            .block_index = node->block_index,
            .scheduled_ordinal = node->scheduled_ordinal,
            .live_units_before = live_units_before,
            .killed_live_units = score.killed_live_units,
            .produced_live_units = score.produced_live_units,
            .live_units_after = pressure_state->current_live_units,
        };
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_append_resource_use(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_resource_use_t resource_use) {
  if (state->resource_use_count >= state->resource_use_capacity) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "low schedule exceeded precomputed resource-use capacity");
  }
  if (!state->resource_summaries ||
      resource_use.resource_id >=
          state->target.descriptor_set->resource_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low schedule resource use cannot be summarized");
  }
  loom_low_schedule_resource_summary_t* summary =
      &state->resource_summaries[resource_use.resource_id];
  if (summary->use_count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "low schedule resource-use count overflows");
  }
  const uint64_t occupied_cycles = resource_use.cycles;
  const uint64_t unit_cycles =
      (uint64_t)resource_use.cycles * (uint64_t)resource_use.units;
  if (summary->total_occupied_cycles > UINT64_MAX - occupied_cycles ||
      summary->total_unit_cycles > UINT64_MAX - unit_cycles) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "low schedule resource summary counters overflow");
  }
  ++summary->use_count;
  summary->total_occupied_cycles += occupied_cycles;
  summary->total_unit_cycles += unit_cycles;
  summary->estimated_min_cycles =
      1 + (summary->total_unit_cycles - 1) / summary->capacity_per_cycle;
  if (resource_use.units > summary->peak_units_per_cycle) {
    summary->peak_units_per_cycle = resource_use.units;
  }
  state->resource_uses[state->resource_use_count++] = resource_use;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_append_effect_use(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_effect_use_t effect_use) {
  if (state->effect_use_count >= state->effect_use_capacity) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low schedule exceeded precomputed effect-use capacity");
  }
  state->effect_uses[state->effect_use_count++] = effect_use;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_append_hazard_use(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_hazard_use_t hazard_use) {
  if (state->hazard_use_count >= state->hazard_use_capacity) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "low schedule exceeded precomputed hazard-use capacity");
  }
  state->hazard_uses[state->hazard_use_count++] = hazard_use;
  return iree_ok_status();
}

static bool loom_low_schedule_hazard_state_matches(
    const loom_low_schedule_hazard_state_t* hazard_state,
    const loom_low_schedule_hazard_use_t* hazard_use, uint16_t producer_stage) {
  return hazard_state->kind == hazard_use->kind &&
         hazard_state->reference_kind == hazard_use->reference_kind &&
         hazard_state->reference_id == hazard_use->reference_id &&
         hazard_state->producer_stage == producer_stage &&
         hazard_state->block_index == hazard_use->block_index;
}

static iree_status_t loom_low_schedule_append_hazard_gap(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_hazard_gap_t hazard_gap) {
  if (state->hazard_gap_count >= state->hazard_gap_capacity) {
    iree_host_size_t new_capacity =
        state->hazard_gap_capacity == 0 ? 4 : state->hazard_gap_capacity * 2;
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(state->arena, state->hazard_gap_count,
                              new_capacity, sizeof(*state->hazard_gaps),
                              &new_capacity, (void**)&state->hazard_gaps));
    state->hazard_gap_capacity = new_capacity;
  }
  state->hazard_gaps[state->hazard_gap_count++] = hazard_gap;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_update_hazard_state(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_hazard_use_t* hazard_use) {
  for (iree_host_size_t i = 0; i < state->hazard_state_count; ++i) {
    loom_low_schedule_hazard_state_t* hazard_state = &state->hazard_states[i];
    if (!loom_low_schedule_hazard_state_matches(hazard_state, hazard_use,
                                                hazard_use->producer_stage)) {
      continue;
    }
    hazard_state->node_index = hazard_use->node_index;
    hazard_state->scheduled_ordinal = hazard_use->scheduled_ordinal;
    hazard_state->hazard_ordinal = hazard_use->hazard_ordinal;
    hazard_state->distance = hazard_use->distance;
    hazard_state->hazard_flags = hazard_use->hazard_flags;
    return iree_ok_status();
  }

  if (state->hazard_state_count >= state->hazard_state_capacity) {
    iree_host_size_t new_capacity = state->hazard_state_capacity == 0
                                        ? 4
                                        : state->hazard_state_capacity * 2;
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(state->arena, state->hazard_state_count,
                              new_capacity, sizeof(*state->hazard_states),
                              &new_capacity, (void**)&state->hazard_states));
    state->hazard_state_capacity = new_capacity;
  }
  state->hazard_states[state->hazard_state_count++] =
      (loom_low_schedule_hazard_state_t){
          .kind = hazard_use->kind,
          .reference_kind = hazard_use->reference_kind,
          .reference_id = hazard_use->reference_id,
          .producer_stage = hazard_use->producer_stage,
          .block_index = hazard_use->block_index,
          .node_index = hazard_use->node_index,
          .scheduled_ordinal = hazard_use->scheduled_ordinal,
          .hazard_ordinal = hazard_use->hazard_ordinal,
          .distance = hazard_use->distance,
          .hazard_flags = hazard_use->hazard_flags,
      };
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_note_min_distance_hazard(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_hazard_use_t* hazard_use) {
  if (hazard_use->kind != LOOM_LOW_HAZARD_KIND_MIN_DISTANCE) {
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < state->hazard_state_count; ++i) {
    const loom_low_schedule_hazard_state_t* hazard_state =
        &state->hazard_states[i];
    if (!loom_low_schedule_hazard_state_matches(hazard_state, hazard_use,
                                                hazard_use->consumer_stage)) {
      continue;
    }
    if (hazard_use->scheduled_ordinal < hazard_state->scheduled_ordinal) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low schedule hazard producer appears after consumer");
    }
    const uint32_t actual_distance =
        hazard_use->scheduled_ordinal - hazard_state->scheduled_ordinal;
    const uint16_t required_distance =
        hazard_state->distance > hazard_use->distance ? hazard_state->distance
                                                      : hazard_use->distance;
    if (actual_distance < required_distance) {
      IREE_RETURN_IF_ERROR(loom_low_schedule_append_hazard_gap(
          state,
          (loom_low_schedule_hazard_gap_t){
              .producer_node = hazard_state->node_index,
              .consumer_node = hazard_use->node_index,
              .block_index = hazard_use->block_index,
              .producer_scheduled_ordinal = hazard_state->scheduled_ordinal,
              .consumer_scheduled_ordinal = hazard_use->scheduled_ordinal,
              .producer_hazard_ordinal = hazard_state->hazard_ordinal,
              .consumer_hazard_ordinal = hazard_use->hazard_ordinal,
              .kind = hazard_use->kind,
              .reference_kind = hazard_use->reference_kind,
              .reference_id = hazard_use->reference_id,
              .resource_name = hazard_use->resource_name,
              .producer_stage = hazard_state->producer_stage,
              .consumer_stage = hazard_use->consumer_stage,
              .required_distance = required_distance,
              .actual_distance = actual_distance,
              .required_delay = (uint16_t)(required_distance - actual_distance),
              .hazard_flags =
                  hazard_state->hazard_flags | hazard_use->hazard_flags,
          }));
    }
    break;
  }
  return loom_low_schedule_update_hazard_state(state, hazard_use);
}

static void loom_low_schedule_compact_resource_summaries(
    loom_low_schedule_build_state_t* state) {
  if (!state->resource_summaries) {
    return;
  }
  iree_host_size_t write_index = 0;
  for (iree_host_size_t read_index = 0;
       read_index < state->target.descriptor_set->resource_count;
       ++read_index) {
    if (state->resource_summaries[read_index].use_count == 0) {
      continue;
    }
    state->resource_summaries[write_index++] =
        state->resource_summaries[read_index];
  }
  state->resource_summary_count = write_index;
}

static void loom_low_schedule_compact_model_summaries(
    loom_low_schedule_build_state_t* state) {
  if (!state->model_summaries) {
    return;
  }
  iree_host_size_t write_index = 0;
  for (iree_host_size_t read_index = 0;
       read_index < state->target.descriptor_set->schedule_class_count;
       ++read_index) {
    if (state->model_summaries[read_index].use_count == 0) {
      continue;
    }
    state->model_summaries[write_index++] = state->model_summaries[read_index];
  }
  state->model_summary_count = write_index;
}

static iree_status_t loom_low_schedule_note_descriptor_rows_for_node(
    loom_low_schedule_build_state_t* state, uint32_t node_index) {
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  if (node->schedule_class_id == LOOM_LOW_SCHEDULE_CLASS_NONE) {
    return iree_ok_status();
  }
  if (node->schedule_class_id >=
      state->target.descriptor_set->schedule_class_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low schedule node references invalid schedule class %" PRIu16,
        node->schedule_class_id);
  }
  const loom_low_schedule_class_t* schedule_class =
      &state->target.descriptor_set->schedule_classes[node->schedule_class_id];
  const loom_low_descriptor_t* descriptor = NULL;
  if (node->descriptor_ordinal != LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    descriptor = loom_low_descriptor_set_descriptor_at(
        state->target.descriptor_set, node->descriptor_ordinal);
    if (!descriptor) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low schedule node references invalid "
                              "descriptor ordinal %" PRIu32,
                              node->descriptor_ordinal);
    }
  }
  if (state->model_summaries) {
    loom_low_schedule_model_summary_t* model_summary =
        &state->model_summaries[node->schedule_class_id];
    if (model_summary->use_count == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "low schedule model summary counters overflow");
    }
    if (model_summary->use_count == 0) {
      iree_string_view_t schedule_class_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
          state->target.descriptor_set, schedule_class->name_string_offset,
          &schedule_class_name));
      *model_summary = (loom_low_schedule_model_summary_t){
          .first_node = node_index,
          .schedule_class_id = node->schedule_class_id,
          .schedule_class_name = schedule_class_name,
          .latency_cycles = schedule_class->latency_cycles,
          .latency_kind = schedule_class->latency_kind,
          .model_quality = schedule_class->model_quality,
          .issue_use_count = schedule_class->issue_use_count,
          .hazard_count = schedule_class->hazard_count,
      };
    }
    ++model_summary->use_count;
  }
  if (descriptor) {
    for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
      const loom_low_effect_t* effect =
          &state->target.descriptor_set->effects[descriptor->effect_start + i];
      IREE_RETURN_IF_ERROR(loom_low_schedule_append_effect_use(
          state, (loom_low_schedule_effect_use_t){
                     .node_index = node_index,
                     .block_index = node->block_index,
                     .scheduled_ordinal = node->scheduled_ordinal,
                     .effect_ordinal = i,
                     .kind = effect->kind,
                     .memory_space = effect->memory_space,
                     .scope_id = effect->scope_id,
                     .effect_flags = effect->flags,
                     .counter_id = effect->counter_id,
                     .width_bits = effect->width_bits,
                 }));
    }
  }
  for (uint16_t i = 0; i < schedule_class->issue_use_count; ++i) {
    const loom_low_issue_use_t* issue_use =
        &state->target.descriptor_set
             ->issue_uses[schedule_class->issue_use_start + i];
    if (issue_use->resource_id >=
        state->target.descriptor_set->resource_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low schedule issue-use references invalid resource %" PRIu16,
          issue_use->resource_id);
    }
    const loom_low_resource_t* resource =
        &state->target.descriptor_set->resources[issue_use->resource_id];
    iree_string_view_t resource_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
        state->target.descriptor_set, resource->name_string_offset,
        &resource_name));
    IREE_RETURN_IF_ERROR(loom_low_schedule_append_resource_use(
        state, (loom_low_schedule_resource_use_t){
                   .node_index = node_index,
                   .block_index = node->block_index,
                   .scheduled_ordinal = node->scheduled_ordinal,
                   .issue_use_ordinal = i,
                   .resource_id = issue_use->resource_id,
                   .resource_name = resource_name,
                   .resource_kind = resource->kind,
                   .resource_flags = resource->flags,
                   .capacity_per_cycle = resource->capacity_per_cycle,
                   .contention_group_id = resource->contention_group_id,
                   .stage = issue_use->stage,
                   .cycles = issue_use->cycles,
                   .units = issue_use->units,
               }));
  }
  for (uint16_t i = 0; i < schedule_class->hazard_count; ++i) {
    const loom_low_hazard_t* hazard =
        &state->target.descriptor_set
             ->hazards[schedule_class->hazard_start + i];
    iree_string_view_t resource_name = iree_string_view_empty();
    if (hazard->reference_kind == LOOM_LOW_HAZARD_REFERENCE_KIND_RESOURCE) {
      if (hazard->reference_id >=
          state->target.descriptor_set->resource_count) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "low schedule hazard references invalid resource %" PRIu16,
            hazard->reference_id);
      }
      const loom_low_resource_t* resource =
          &state->target.descriptor_set->resources[hazard->reference_id];
      IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
          state->target.descriptor_set, resource->name_string_offset,
          &resource_name));
    }
    IREE_RETURN_IF_ERROR(loom_low_schedule_append_hazard_use(
        state, (loom_low_schedule_hazard_use_t){
                   .node_index = node_index,
                   .block_index = node->block_index,
                   .scheduled_ordinal = node->scheduled_ordinal,
                   .hazard_ordinal = i,
                   .kind = hazard->kind,
                   .reference_kind = hazard->reference_kind,
                   .reference_id = hazard->reference_id,
                   .resource_name = resource_name,
                   .producer_stage = hazard->producer_stage,
                   .consumer_stage = hazard->consumer_stage,
                   .distance = hazard->distance,
                   .hazard_flags = hazard->flags,
               }));
    IREE_RETURN_IF_ERROR(loom_low_schedule_note_min_distance_hazard(
        state, &state->hazard_uses[state->hazard_use_count - 1]));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_run_list_scheduler(
    loom_low_schedule_build_state_t* state, iree_host_size_t node_count,
    const loom_liveness_analysis_t* liveness) {
  uint32_t* indegrees = NULL;
  bool* scheduled = NULL;
  loom_low_schedule_pressure_state_t pressure_state = {0};
  if (node_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, node_count, sizeof(*indegrees), (void**)&indegrees));
    memset(indegrees, 0, node_count * sizeof(*indegrees));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, node_count, sizeof(*scheduled), (void**)&scheduled));
    memset(scheduled, 0, node_count * sizeof(*scheduled));
  }
  IREE_RETURN_IF_ERROR(
      loom_low_schedule_allocate_pressure_state(state, &pressure_state));
  for (iree_host_size_t i = 0; i < state->dependency_count; ++i) {
    const loom_low_schedule_dependency_t* dependency = &state->dependencies[i];
    if (dependency->consumer_node < node_count) {
      ++indegrees[dependency->consumer_node];
    }
  }

  for (iree_host_size_t block_index = 0; block_index < state->body->block_count;
       ++block_index) {
    const loom_low_schedule_block_t* block_record = &state->blocks[block_index];
    const uint32_t block_node_end =
        block_record->node_start + block_record->node_count;
    if (loom_low_schedule_uses_pressure_strategy(state)) {
      IREE_RETURN_IF_ERROR(loom_low_schedule_initialize_block_pressure(
          state, liveness, block_record, &pressure_state));
    }
    uint32_t scheduled_in_block = 0;
    while (scheduled_in_block < block_record->node_count) {
      uint32_t chosen_node = LOOM_LOW_SCHEDULE_NODE_NONE;
      loom_low_schedule_candidate_score_t chosen_score = {0};
      uint32_t rejected_node = LOOM_LOW_SCHEDULE_NODE_NONE;
      loom_low_schedule_candidate_score_t rejected_score = {0};
      uint32_t ready_candidate_count = 0;
      for (uint32_t node_index = block_record->node_start;
           node_index < block_node_end; ++node_index) {
        if (scheduled[node_index] || indegrees[node_index] != 0) {
          continue;
        }
        if (!loom_low_schedule_uses_pressure_strategy(state)) {
          chosen_node = node_index;
          break;
        }
        ++ready_candidate_count;
        loom_low_schedule_candidate_score_t score = {0};
        IREE_RETURN_IF_ERROR(loom_low_schedule_pressure_score_candidate(
            state, liveness, &pressure_state, node_index, &score));
        if (chosen_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
            loom_low_schedule_candidate_score_less(score, chosen_score)) {
          if (chosen_node != LOOM_LOW_SCHEDULE_NODE_NONE) {
            rejected_node = chosen_node;
            rejected_score = chosen_score;
          }
          chosen_node = node_index;
          chosen_score = score;
        } else if (rejected_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
                   loom_low_schedule_candidate_score_less(score,
                                                          rejected_score)) {
          rejected_node = node_index;
          rejected_score = score;
        }
      }
      if (chosen_node == LOOM_LOW_SCHEDULE_NODE_NONE) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "low schedule dependency cycle in block %zu",
                                block_index);
      }

      scheduled[chosen_node] = true;
      state->nodes[chosen_node].scheduled_ordinal = scheduled_in_block++;
      state->scheduled_node_indices[state->scheduled_node_count++] =
          chosen_node;
      if (loom_low_schedule_uses_pressure_strategy(state)) {
        loom_low_schedule_record_candidate_decision(
            state, block_index, state->nodes[chosen_node].scheduled_ordinal,
            ready_candidate_count, chosen_node, chosen_score, rejected_node,
            rejected_score);
        IREE_RETURN_IF_ERROR(loom_low_schedule_note_pressure_node_scheduled(
            state, liveness, &pressure_state, chosen_node, chosen_score));
      }
      IREE_RETURN_IF_ERROR(
          loom_low_schedule_note_descriptor_rows_for_node(state, chosen_node));

      for (iree_host_size_t dependency_index = 0;
           dependency_index < state->dependency_count; ++dependency_index) {
        const loom_low_schedule_dependency_t* dependency =
            &state->dependencies[dependency_index];
        if (dependency->producer_node != chosen_node) {
          continue;
        }
        if (dependency->consumer_node < node_count) {
          --indegrees[dependency->consumer_node];
        }
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_schedule_function(
    const loom_module_t* module, const loom_op_t* low_func_op,
    const loom_low_schedule_options_t* options, iree_arena_allocator_t* arena,
    loom_low_schedule_sidecar_t* out_sidecar) {
  if (!module || !low_func_op || !options || !options->descriptor_registry ||
      !arena || !out_sidecar) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "module, low function op, options with descriptor registry, arena, and "
        "output sidecar are required");
  }
  if (!loom_low_func_def_isa(low_func_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected low.func.def");
  }
  if (!loom_low_schedule_strategy_is_valid(options->strategy)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown low schedule strategy %d",
                            (int)options->strategy);
  }
  *out_sidecar = (loom_low_schedule_sidecar_t){0};

  loom_low_schedule_build_state_t state = {
      .module = module,
      .options = options,
      .arena = arena,
      .function_op = low_func_op,
      .body = loom_low_func_def_body(low_func_op),
  };
  if (!state.body) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low.func.def body is required");
  }
  IREE_RETURN_IF_ERROR(loom_low_descriptor_registry_verify_requirements(
      options->descriptor_registry,
      LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION));
  IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
      module, low_func_op, options->descriptor_registry, options->emitter,
      &state.target));
  if (!state.target.descriptor_set) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low function target did not resolve");
  }
  IREE_RETURN_IF_ERROR(loom_low_register_class_map_initialize(
      module, state.target.descriptor_set, arena, &state.register_class_map));

  iree_host_size_t node_count = 0;
  loom_low_schedule_count_nodes(state.body, &node_count);
  IREE_RETURN_IF_ERROR(
      loom_low_schedule_initialize_storage(&state, node_count));
  IREE_RETURN_IF_ERROR(loom_low_schedule_fill_nodes(&state));
  IREE_RETURN_IF_ERROR(
      loom_low_schedule_initialize_descriptor_sidecars(&state, node_count));
  IREE_RETURN_IF_ERROR(loom_low_schedule_build_dependencies(&state));
  loom_liveness_analysis_t liveness = {0};
  IREE_RETURN_IF_ERROR(
      loom_liveness_analyze_region(module, state.body, arena, &liveness));
  IREE_RETURN_IF_ERROR(
      loom_low_schedule_run_list_scheduler(&state, node_count, &liveness));
  loom_low_schedule_compact_model_summaries(&state);
  loom_low_schedule_compact_resource_summaries(&state);
  if (iree_any_bit_set(options->diagnostic_flags,
                       LOOM_LOW_SCHEDULE_DIAGNOSTIC_PRESSURE_PEAKS)) {
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_emit_pressure_diagnostics(&state, &liveness));
  }
  if (iree_any_bit_set(options->diagnostic_flags,
                       LOOM_LOW_SCHEDULE_DIAGNOSTIC_CANDIDATE_DECISIONS)) {
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_emit_candidate_decision_diagnostics(&state));
  }
  if (iree_any_bit_set(options->diagnostic_flags,
                       LOOM_LOW_SCHEDULE_DIAGNOSTIC_MODEL_QUALITY)) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_emit_model_diagnostics(&state));
  }
  if (iree_any_bit_set(options->diagnostic_flags,
                       LOOM_LOW_SCHEDULE_DIAGNOSTIC_RESOURCE_BOTTLENECKS)) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_emit_resource_diagnostics(&state));
  }
  if (iree_any_bit_set(options->diagnostic_flags,
                       LOOM_LOW_SCHEDULE_DIAGNOSTIC_HAZARD_GAPS)) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_emit_hazard_gap_diagnostics(&state));
  }

  *out_sidecar = (loom_low_schedule_sidecar_t){
      .module = module,
      .function_op = low_func_op,
      .target = state.target,
      .liveness = liveness,
      .blocks = state.blocks,
      .block_count = state.body->block_count,
      .nodes = state.nodes,
      .node_count = node_count,
      .dependencies = state.dependencies,
      .dependency_count = state.dependency_count,
      .scheduled_node_indices = state.scheduled_node_indices,
      .scheduled_node_count = state.scheduled_node_count,
      .pressure_steps = state.pressure_steps,
      .pressure_step_count = state.pressure_step_count,
      .candidate_decisions = state.candidate_decisions,
      .candidate_decision_count = state.candidate_decision_count,
      .resource_uses = state.resource_uses,
      .resource_use_count = state.resource_use_count,
      .effect_uses = state.effect_uses,
      .effect_use_count = state.effect_use_count,
      .hazard_uses = state.hazard_uses,
      .hazard_use_count = state.hazard_use_count,
      .hazard_gaps = state.hazard_gaps,
      .hazard_gap_count = state.hazard_gap_count,
      .model_summaries = state.model_summaries,
      .model_summary_count = state.model_summary_count,
      .resource_summaries = state.resource_summaries,
      .resource_summary_count = state.resource_summary_count,
  };
  loom_target_ir_bundle_storage_rebind(&out_sidecar->target.bundle_storage);
  return iree_ok_status();
}
