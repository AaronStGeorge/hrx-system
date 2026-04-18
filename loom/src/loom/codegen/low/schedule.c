// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule.h"

#include <string.h>

#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/requirements.h"
#include "loom/error/error_defs.h"
#include "loom/ir/context.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"

#define LOOM_LOW_SCHEDULE_MAX_PRESSURE_CONTRIBUTORS 16

typedef struct loom_low_schedule_build_state_t {
  const loom_module_t* module;
  const loom_low_schedule_options_t* options;
  iree_arena_allocator_t* arena;
  const loom_op_t* function_op;
  loom_region_t* body;
  loom_low_resolved_target_t target;
  loom_low_schedule_block_t* blocks;
  loom_low_schedule_node_t* nodes;
  loom_low_schedule_dependency_t* dependencies;
  uint32_t* scheduled_node_indices;
  uint32_t* value_node_indices;
  iree_host_size_t dependency_count;
  iree_host_size_t dependency_capacity;
  iree_host_size_t scheduled_node_count;
} loom_low_schedule_build_state_t;

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

static iree_string_view_t loom_low_schedule_module_string(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
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
  iree_string_view_t value_class_name =
      loom_low_diagnostic_value_class_name(state->module, value_class);
  if (iree_string_view_equal(value_class_name, IREE_SV("<unknown>"))) {
    return iree_ok_status();
  }
  for (uint32_t i = 0; i < state->target.descriptor_set->reg_class_count; ++i) {
    const loom_low_reg_class_t* reg_class =
        &state->target.descriptor_set->reg_classes[i];
    iree_string_view_t reg_class_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
        state->target.descriptor_set, reg_class->name_string_offset,
        &reg_class_name));
    if (!iree_string_view_equal(reg_class_name, value_class_name)) continue;
    if (reg_class->physical_count == 0) return iree_ok_status();
    *out_budget = reg_class->physical_count;
    *out_has_budget = true;
    return iree_ok_status();
  }
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
  if (!has_budget) return iree_ok_status();

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

static iree_status_t loom_low_schedule_resolve_descriptor(
    loom_low_schedule_build_state_t* state, const loom_op_t* op,
    loom_low_schedule_node_t* node,
    const loom_low_descriptor_t** out_descriptor) {
  *out_descriptor = NULL;
  if (!loom_low_schedule_op_is_descriptor_packet(op)) return iree_ok_status();

  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  if (loom_low_op_isa(op)) {
    opcode_id = loom_low_op_opcode(op);
  } else {
    opcode_id = loom_low_const_opcode(op);
  }
  iree_string_view_t opcode =
      loom_low_schedule_module_string(state->module, opcode_id);
  node->descriptor_key = opcode;

  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  iree_status_t lookup_status = loom_low_descriptor_set_lookup_descriptor(
      state->target.descriptor_set, opcode, &descriptor_ordinal);
  if (!iree_status_is_ok(lookup_status)) {
    if (!iree_status_is_not_found(lookup_status)) {
      return iree_status_annotate_f(lookup_status,
                                    "failed to look up low descriptor '%.*s'",
                                    (int)opcode.size, opcode.data);
    }
    iree_status_free(lookup_status);
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_emit_missing_descriptor(state, op, opcode));
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low schedule descriptor '%.*s' is not available",
                            (int)opcode.size, opcode.data);
  }
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "low descriptor lookup returned no descriptor");
  }

  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(state->target.descriptor_set,
                                            descriptor_ordinal);
  if (!descriptor) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "resolved descriptor ordinal is out of bounds");
  }
  node->descriptor_ordinal = descriptor_ordinal;
  node->effect_count = descriptor->effect_count;
  node->schedule_class_id = descriptor->schedule_class_id;
  if (descriptor->schedule_class_id == LOOM_LOW_SCHEDULE_CLASS_NONE ||
      descriptor->schedule_class_id >=
          state->target.descriptor_set->schedule_class_count) {
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_emit_missing_schedule_class(state, op, opcode));
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low schedule descriptor '%.*s' has no usable schedule class",
        (int)opcode.size, opcode.data);
  }

  const loom_low_schedule_class_t* schedule_class =
      &state->target.descriptor_set
           ->schedule_classes[descriptor->schedule_class_id];
  node->latency_cycles = schedule_class->latency_cycles;
  node->latency_kind = schedule_class->latency_kind;
  node->model_quality = schedule_class->model_quality;
  node->issue_use_count = schedule_class->issue_use_count;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
      state->target.descriptor_set, schedule_class->name_string_offset,
      &node->schedule_class_name));
  *out_descriptor = descriptor;
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
  if (producer_node == consumer_node) return iree_ok_status();
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
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, state->module->values.count,
      sizeof(*state->value_node_indices), (void**)&state->value_node_indices));
  for (iree_host_size_t i = 0; i < state->module->values.count; ++i) {
    state->value_node_indices[i] = LOOM_LOW_SCHEDULE_NODE_NONE;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_fill_nodes(
    loom_low_schedule_build_state_t* state) {
  uint32_t next_node_index = 0;
  for (uint16_t block_index = 0; block_index < state->body->block_count;
       ++block_index) {
    loom_block_t* block = state->body->blocks[block_index];
    if (!block) continue;
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
    const uint32_t block_node_end =
        block_record->node_start + block_record->node_count;
    for (uint32_t node_index = block_record->node_start;
         node_index < block_node_end; ++node_index) {
      const loom_low_schedule_node_t* node = &state->nodes[node_index];
      const loom_op_t* op = node->op;
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

static iree_status_t loom_low_schedule_run_list_scheduler(
    loom_low_schedule_build_state_t* state, iree_host_size_t node_count) {
  uint32_t* indegrees = NULL;
  bool* scheduled = NULL;
  if (node_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, node_count, sizeof(*indegrees), (void**)&indegrees));
    memset(indegrees, 0, node_count * sizeof(*indegrees));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, node_count, sizeof(*scheduled), (void**)&scheduled));
    memset(scheduled, 0, node_count * sizeof(*scheduled));
  }
  for (iree_host_size_t i = 0; i < state->dependency_count; ++i) {
    const loom_low_schedule_dependency_t* dependency = &state->dependencies[i];
    if (dependency->consumer_node < node_count)
      ++indegrees[dependency->consumer_node];
  }

  for (iree_host_size_t block_index = 0; block_index < state->body->block_count;
       ++block_index) {
    const loom_low_schedule_block_t* block_record = &state->blocks[block_index];
    const uint32_t block_node_end =
        block_record->node_start + block_record->node_count;
    uint32_t scheduled_in_block = 0;
    while (scheduled_in_block < block_record->node_count) {
      uint32_t chosen_node = LOOM_LOW_SCHEDULE_NODE_NONE;
      for (uint32_t node_index = block_record->node_start;
           node_index < block_node_end; ++node_index) {
        if (!scheduled[node_index] && indegrees[node_index] == 0) {
          chosen_node = node_index;
          break;
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

      for (iree_host_size_t dependency_index = 0;
           dependency_index < state->dependency_count; ++dependency_index) {
        const loom_low_schedule_dependency_t* dependency =
            &state->dependencies[dependency_index];
        if (dependency->producer_node != chosen_node) continue;
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

  iree_host_size_t node_count = 0;
  loom_low_schedule_count_nodes(state.body, &node_count);
  IREE_RETURN_IF_ERROR(
      loom_low_schedule_initialize_storage(&state, node_count));
  IREE_RETURN_IF_ERROR(loom_low_schedule_fill_nodes(&state));
  IREE_RETURN_IF_ERROR(loom_low_schedule_build_dependencies(&state));
  IREE_RETURN_IF_ERROR(
      loom_low_schedule_run_list_scheduler(&state, node_count));

  loom_liveness_analysis_t liveness = {0};
  IREE_RETURN_IF_ERROR(
      loom_liveness_analyze_region(module, state.body, arena, &liveness));
  if (iree_any_bit_set(options->diagnostic_flags,
                       LOOM_LOW_SCHEDULE_DIAGNOSTIC_PRESSURE_PEAKS)) {
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_emit_pressure_diagnostics(&state, &liveness));
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
  };
  return iree_ok_status();
}
