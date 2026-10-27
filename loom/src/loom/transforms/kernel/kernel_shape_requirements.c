// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/kernel_shape_requirements.h"

#include <string.h>

#include "loom/analysis/value_relation.h"
#include "loom/ir/intern_table.h"
#include "loom/ir/module.h"
#include "loom/ir/structural_hash.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/template/ops.h"
#include "loom/util/walk.h"

// A monotone dependency equation. Conditional edges implement call argument
// substitution: source contributes only when the callee summary contains the
// named formal. Both inputs schedule the same equation when their set changes.
typedef struct loom_kernel_shape_edge_t {
  // Caller-local value whose dependencies contribute to destination.
  uint32_t source;
  // Value or callable summary receiving the dependencies.
  uint32_t destination;
  // Callee summary controlling substitution, or UINT32_MAX for a plain union.
  uint32_t condition;
  // Callee formal required in condition before source contributes.
  loom_value_id_t formal;
} loom_kernel_shape_edge_t;

typedef struct loom_kernel_shape_watch_t {
  // Equation to reevaluate when this input changes.
  uint32_t edge;
  // Next watcher for the same node, or UINT32_MAX.
  uint32_t next;
} loom_kernel_shape_watch_t;

typedef struct loom_kernel_shape_node_t {
  // Source identity, or INVALID for a callable summary node.
  loom_value_id_t value;
  // Canonical set of formal arguments supplying this value or summary.
  loom_value_set_id_t dependencies;
  // First equation watching this node, or UINT32_MAX.
  uint32_t first_watch;
  // Next queued node, or UINT32_MAX at the tail.
  uint32_t next_queued;
  // True while the node has pending propagation.
  bool queued;
} loom_kernel_shape_node_t;

struct loom_kernel_shape_values_t {
  // Sparse source-value identities indexed into nodes.
  loom_intern_table_t index;
  // Populated value and summary nodes; never sized by module value IDs.
  loom_kernel_shape_node_t* nodes;
  // Number of initialized nodes.
  iree_host_size_t count;
  // Allocated node capacity.
  iree_host_size_t capacity;
};

typedef struct loom_kernel_shape_value_probe_t {
  // Canonical values being searched.
  const loom_kernel_shape_values_t* values;
  // Source identity to locate.
  loom_value_id_t value;
} loom_kernel_shape_value_probe_t;

static bool loom_kernel_shape_value_equal(const void* context, uint32_t index) {
  const loom_kernel_shape_value_probe_t* probe = context;
  return probe->values->nodes[index].value == probe->value;
}

static uint32_t loom_kernel_shape_value_hash(loom_value_id_t value) {
  return loom_structural_hash_finalize(
      loom_structural_hash_mix_u32(loom_structural_hash_initialize(), value));
}

loom_value_set_id_t loom_kernel_shape_value_arguments(
    const loom_kernel_shape_requirements_t* requirements,
    loom_value_id_t value) {
  const loom_kernel_shape_value_probe_t key = {requirements->values, value};
  const loom_intern_probe_t probe = loom_intern_table_probe(
      &requirements->values->index, loom_kernel_shape_value_hash(value),
      loom_kernel_shape_value_equal, &key);
  return requirements->values->nodes[probe.index].dependencies;
}

typedef struct loom_kernel_shape_function_t {
  // Body and signature supplied by verified IR.
  loom_func_like_t function;
  // Body argument identities in formal order.
  const loom_value_id_t* arguments;
  // Number of arguments.
  uint16_t argument_count;
  // Node holding the function's structural argument dependencies.
  uint32_t requirement;
  // First result-summary node; function result ordinals are contiguous.
  uint32_t first_result;
} loom_kernel_shape_function_t;

typedef struct loom_kernel_shape_call_t {
  // Exact caller and callee symbol identities.
  loom_symbol_id_t caller;
  // Callee whose summary is substituted at this site.
  loom_symbol_id_t callee;
  // Actual arguments borrowed from the immutable source module.
  loom_value_slice_t arguments;
} loom_kernel_shape_call_t;

typedef struct loom_kernel_shape_builder_t {
  // Immutable module described by the analysis.
  const loom_module_t* module;
  // Header-owned family models; body dependencies never rank providers.
  const loom_template_decision_model_catalog_t* decision_models;
  // Storage for the equations and retained summaries.
  iree_arena_allocator_t* arena;
  // Published canonical sets and requirements.
  loom_kernel_shape_requirements_t* result;
  // One callable record per symbol.
  loom_kernel_shape_function_t* functions;
  // Sparse value identities and dense dependency nodes.
  loom_kernel_shape_values_t* values;
  // Dependency equations.
  loom_kernel_shape_edge_t* edges;
  // Number of initialized equations.
  iree_host_size_t edge_count;
  // Allocated equation capacity.
  iree_host_size_t edge_capacity;
  // Input-to-equation adjacency lists.
  loom_kernel_shape_watch_t* watches;
  // Number of initialized watchers.
  iree_host_size_t watch_count;
  // Allocated watcher capacity.
  iree_host_size_t watch_capacity;
  // Exact calls retained for publishing caller-local required values.
  loom_kernel_shape_call_t* calls;
  // Number of exact calls.
  iree_host_size_t call_count;
  // Allocated call capacity.
  iree_host_size_t call_capacity;
  // Function currently being structurally visited.
  loom_symbol_id_t current_symbol;
  // Head of the pending-node queue, or UINT32_MAX.
  uint32_t first_queued;
  // Tail of the pending-node queue, or UINT32_MAX.
  uint32_t last_queued;
} loom_kernel_shape_builder_t;

static iree_status_t loom_kernel_shape_append_node(
    loom_kernel_shape_builder_t* builder, loom_value_id_t value,
    uint32_t* out_node) {
  loom_kernel_shape_values_t* values = builder->values;
  if (values->count == UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "kernel shape dependency graph exceeds the node limit");
  }
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      builder->arena, values->count, values->count + 1, sizeof(*values->nodes),
      &values->capacity, (void**)&values->nodes));
  *out_node = (uint32_t)values->count++;
  values->nodes[*out_node] = (loom_kernel_shape_node_t){
      .value = value,
      .first_watch = UINT32_MAX,
  };
  return iree_ok_status();
}

static iree_status_t loom_kernel_shape_value_node(
    loom_kernel_shape_builder_t* builder, loom_value_id_t value,
    uint32_t* out_node) {
  loom_kernel_shape_values_t* values = builder->values;
  const loom_kernel_shape_value_probe_t key = {values, value};
  const uint32_t hash = loom_kernel_shape_value_hash(value);
  loom_intern_probe_t probe = loom_intern_table_probe(
      &values->index, hash, loom_kernel_shape_value_equal, &key);
  if (probe.index != UINT32_MAX) {
    *out_node = probe.index;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_intern_table_reserve_insert(
      builder->arena, &values->index, hash, 1, &probe.slot));
  IREE_RETURN_IF_ERROR(loom_kernel_shape_append_node(builder, value, out_node));
  loom_intern_table_insert(&values->index, probe.slot, hash, *out_node);
  return iree_ok_status();
}

static void loom_kernel_shape_enqueue(loom_kernel_shape_builder_t* builder,
                                      uint32_t node_id) {
  loom_kernel_shape_node_t* node = &builder->values->nodes[node_id];
  if (node->queued) {
    return;
  }
  node->queued = true;
  node->next_queued = UINT32_MAX;
  if (builder->last_queued != UINT32_MAX) {
    builder->values->nodes[builder->last_queued].next_queued = node_id;
  } else {
    builder->first_queued = node_id;
  }
  builder->last_queued = node_id;
}

static iree_status_t loom_kernel_shape_watch(
    loom_kernel_shape_builder_t* builder, uint32_t node_id, uint32_t edge) {
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      builder->arena, builder->watch_count, builder->watch_count + 1,
      sizeof(*builder->watches), &builder->watch_capacity,
      (void**)&builder->watches));
  builder->watches[builder->watch_count] = (loom_kernel_shape_watch_t){
      .edge = edge,
      .next = builder->values->nodes[node_id].first_watch,
  };
  builder->values->nodes[node_id].first_watch =
      (uint32_t)builder->watch_count++;
  return iree_ok_status();
}

static iree_status_t loom_kernel_shape_edge(
    loom_kernel_shape_builder_t* builder, uint32_t source, uint32_t destination,
    uint32_t condition, loom_value_id_t formal) {
  if (source == destination) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_arena_grow_array(builder->arena, builder->edge_count,
                            builder->edge_count + 1, sizeof(*builder->edges),
                            &builder->edge_capacity, (void**)&builder->edges));
  uint32_t edge = (uint32_t)builder->edge_count++;
  builder->edges[edge] = (loom_kernel_shape_edge_t){
      .source = source,
      .destination = destination,
      .condition = condition,
      .formal = formal,
  };
  IREE_RETURN_IF_ERROR(loom_kernel_shape_watch(builder, source, edge));
  if (condition != UINT32_MAX && condition != source) {
    IREE_RETURN_IF_ERROR(loom_kernel_shape_watch(builder, condition, edge));
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_shape_value_edge(
    loom_kernel_shape_builder_t* builder, loom_value_id_t source_value,
    loom_value_id_t destination_value) {
  uint32_t source, destination;
  IREE_RETURN_IF_ERROR(
      loom_kernel_shape_value_node(builder, source_value, &source));
  IREE_RETURN_IF_ERROR(
      loom_kernel_shape_value_node(builder, destination_value, &destination));
  return loom_kernel_shape_edge(builder, source, destination, UINT32_MAX,
                                LOOM_VALUE_ID_INVALID);
}

static iree_status_t loom_kernel_shape_require(
    loom_kernel_shape_builder_t* builder, loom_value_id_t value) {
  loom_kernel_shape_requirement_t* summary =
      &builder->result->symbols[builder->current_symbol];
  if (loom_value_set_index_contains(builder->result->sets, summary->values,
                                    value)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_value_set_index_add(
      builder->result->sets, summary->values, value, &summary->values));
  uint32_t node;
  IREE_RETURN_IF_ERROR(loom_kernel_shape_value_node(builder, value, &node));
  return loom_kernel_shape_edge(
      builder, node, builder->functions[builder->current_symbol].requirement,
      UINT32_MAX, LOOM_VALUE_ID_INVALID);
}

static iree_status_t loom_kernel_shape_type(
    loom_kernel_shape_builder_t* builder, loom_value_id_t value) {
  uint32_t node;
  IREE_RETURN_IF_ERROR(loom_kernel_shape_value_node(builder, value, &node));
  const loom_type_t type = loom_module_value_type(builder->module, value);
  if (!loom_type_is_vector(type) && !loom_type_is_tile(type)) {
    return iree_ok_status();
  }
  for (uint8_t i = 0; i < loom_type_rank(type); ++i) {
    const uint64_t dimension = loom_type_dim(type, i);
    if (loom_dim_is_dynamic(dimension)) {
      IREE_RETURN_IF_ERROR(
          loom_kernel_shape_require(builder, loom_dim_value_id(dimension)));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_shape_call(
    loom_kernel_shape_builder_t* builder, loom_symbol_ref_t callee_ref,
    loom_value_slice_t arguments, loom_value_slice_t results) {
  if (!loom_symbol_ref_is_valid(callee_ref) || callee_ref.module_id != 0 ||
      !loom_func_like_body(builder->functions[callee_ref.symbol_id].function)) {
    // An external call's result may depend on any input. There is no body
    // being compiled here from which to derive additional static requirements.
    for (uint16_t i = 0; i < results.count; ++i) {
      for (uint16_t j = 0; j < arguments.count; ++j) {
        IREE_RETURN_IF_ERROR(loom_kernel_shape_value_edge(
            builder, arguments.values[j], results.values[i]));
      }
    }
    return iree_ok_status();
  }
  const loom_kernel_shape_function_t* callee =
      &builder->functions[callee_ref.symbol_id];
  const loom_kernel_shape_function_t* caller =
      &builder->functions[builder->current_symbol];
  for (uint16_t j = 0; j < callee->argument_count; ++j) {
    uint32_t argument;
    IREE_RETURN_IF_ERROR(
        loom_kernel_shape_value_node(builder, arguments.values[j], &argument));
    IREE_RETURN_IF_ERROR(
        loom_kernel_shape_edge(builder, argument, caller->requirement,
                               callee->requirement, callee->arguments[j]));
    for (uint16_t i = 0; i < results.count; ++i) {
      uint32_t result;
      IREE_RETURN_IF_ERROR(
          loom_kernel_shape_value_node(builder, results.values[i], &result));
      IREE_RETURN_IF_ERROR(loom_kernel_shape_edge(builder, argument, result,
                                                  callee->first_result + i,
                                                  callee->arguments[j]));
    }
  }
  IREE_RETURN_IF_ERROR(
      iree_arena_grow_array(builder->arena, builder->call_count,
                            builder->call_count + 1, sizeof(*builder->calls),
                            &builder->call_capacity, (void**)&builder->calls));
  builder->calls[builder->call_count++] = (loom_kernel_shape_call_t){
      .caller = builder->current_symbol,
      .callee = callee_ref.symbol_id,
      .arguments = arguments,
  };
  return iree_ok_status();
}

static iree_status_t loom_kernel_shape_visit(void* user_data, loom_op_t* op,
                                             const loom_walk_context_t* context,
                                             loom_walk_result_t* out_result) {
  loom_kernel_shape_builder_t* builder = user_data;
  *out_result = LOOM_WALK_CONTINUE;
  (void)context;
  const loom_value_id_t* operands = loom_op_const_operands(op);
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_kernel_shape_type(builder, operands[i]));
  }
  for (uint16_t i = 0; i < op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_kernel_shape_type(builder, results[i]));
  }
  if (loom_index_constexpr_isa(op) || loom_scalar_constexpr_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_kernel_shape_require(builder, operands[0]));
  }
  if (loom_scf_for_isa(op)) {
    const loom_value_id_t induction =
        loom_region_entry_block(loom_scf_for_body(op))->arg_ids[0];
    for (uint16_t i = 0; i < 3; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_kernel_shape_value_edge(builder, operands[i], induction));
    }
    if (loom_scf_for_has_unroll_policy(op)) {
      for (uint16_t i = 0; i < 3; ++i) {
        IREE_RETURN_IF_ERROR(loom_kernel_shape_require(builder, operands[i]));
      }
    }
    if (loom_scf_for_pipeline_depth_is_present(op)) {
      IREE_RETURN_IF_ERROR(
          loom_kernel_shape_require(builder, loom_scf_for_pipeline_depth(op)));
      IREE_RETURN_IF_ERROR(
          loom_kernel_shape_require(builder, loom_scf_for_step(op)));
    }
    if (loom_scf_for_unroll_factor_is_present(op)) {
      IREE_RETURN_IF_ERROR(
          loom_kernel_shape_require(builder, loom_scf_for_unroll_factor(op)));
    }
  }
  const loom_kernel_shape_function_t* function =
      &builder->functions[builder->current_symbol];
  if (loom_func_like_op_is_body_exit(builder->module, function->function, op)) {
    for (uint16_t i = 0; i < op->operand_count; ++i) {
      uint32_t operand;
      IREE_RETURN_IF_ERROR(
          loom_kernel_shape_value_node(builder, operands[i], &operand));
      IREE_RETURN_IF_ERROR(
          loom_kernel_shape_edge(builder, operand, function->first_result + i,
                                 UINT32_MAX, LOOM_VALUE_ID_INVALID));
    }
    return iree_ok_status();
  }
  if (iree_all_bits_set(op->traits, LOOM_TRAIT_TERMINATOR)) {
    for (uint16_t i = 0; i < op->operand_count; ++i) {
      if (!loom_op_operand_has_role(builder->module, op, i,
                                    LOOM_OPERAND_ROLE_CONTROL_CONDITION)) {
        continue;
      }
      uint32_t operand;
      IREE_RETURN_IF_ERROR(
          loom_kernel_shape_value_node(builder, operands[i], &operand));
      for (uint16_t j = 0; j < function->function.op->result_count; ++j) {
        IREE_RETURN_IF_ERROR(
            loom_kernel_shape_edge(builder, operand, function->first_result + j,
                                   UINT32_MAX, LOOM_VALUE_ID_INVALID));
      }
    }
  }
  // Provider choice remains the header decision owner's responsibility. An
  // abstract application cannot make all of its alternatives simultaneously
  // impose their private structural requirements on the caller.
  if (loom_template_apply_isa(op)) {
    const loom_template_decision_model_t* model =
        loom_template_decision_model_lookup(builder->decision_models,
                                            loom_template_apply_family(op));
    loom_symbol_ref_t callee = loom_symbol_ref_null();
    if (model && model->providers.count == 1) {
      const loom_template_provider_summary_t* provider =
          loom_template_decision_model_provider(model, 0);
      if (provider->module == builder->module) {
        callee = provider->symbol;
      }
    }
    return loom_kernel_shape_call(builder, callee,
                                  loom_template_apply_operands(op),
                                  loom_template_apply_results(op));
  }
  const loom_call_like_t call = loom_call_like_cast(builder->module, op);
  if (loom_call_like_isa(call)) {
    return loom_kernel_shape_call(builder, loom_call_like_callee(call),
                                  loom_call_like_operands(call),
                                  loom_call_like_results(call));
  }
  const loom_value_relation_mask_t mask =
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_FACT_IDENTITY) |
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_VALUE_ALIAS) |
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_CFG_ARGUMENT) |
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_LOOP_CARRIED) |
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_LOOP_BYPASS) |
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_REGION_RESULT);
  loom_value_relation_iterator_t iterator;
  loom_value_relation_iterator_initialize(builder->module, op, mask, &iterator);
  loom_value_relation_t relation;
  while (loom_value_relation_iterator_next(&iterator, &relation)) {
    IREE_RETURN_IF_ERROR(loom_kernel_shape_value_edge(
        builder, relation.source_value_id, relation.destination_value_id));
  }
  // Fact identities transport ordinal pairs, not every argument into every
  // result. This matters for assumptions bundling independent dimensions.
  if (!iree_all_bits_set(op->traits, LOOM_TRAIT_FACT_IDENTITY)) {
    for (uint16_t i = 0; i < op->result_count; ++i) {
      for (uint16_t j = 0; j < op->operand_count; ++j) {
        IREE_RETURN_IF_ERROR(
            loom_kernel_shape_value_edge(builder, operands[j], results[i]));
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_kernel_shape_requirements_build(
    const loom_module_t* module,
    const loom_template_decision_model_catalog_t* decision_models,
    iree_arena_allocator_t* arena,
    loom_kernel_shape_requirements_t* out_requirements) {
  *out_requirements = (loom_kernel_shape_requirements_t){0};
  loom_kernel_shape_builder_t builder = {
      .module = module,
      .decision_models = decision_models,
      .arena = arena,
      .result = out_requirements,
      .first_queued = UINT32_MAX,
      .last_queued = UINT32_MAX,
  };
  IREE_RETURN_IF_ERROR(
      loom_value_set_index_allocate(arena, &out_requirements->sets));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module->symbols.count, sizeof(*out_requirements->symbols),
      (void**)&out_requirements->symbols));
  memset(out_requirements->symbols, 0,
         module->symbols.count * sizeof(*out_requirements->symbols));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, module->symbols.count,
                                                 sizeof(*builder.functions),
                                                 (void**)&builder.functions));
  memset(builder.functions, 0,
         module->symbols.count * sizeof(*builder.functions));
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, sizeof(*builder.values),
                                           (void**)&builder.values));
  *builder.values = (loom_kernel_shape_values_t){0};
  IREE_RETURN_IF_ERROR(
      loom_intern_table_initialize(arena, 0, &builder.values->index));
  out_requirements->values = builder.values;
  const loom_symbol_t* symbol = NULL;
  loom_module_for_each_symbol(module, symbol) {
    const loom_symbol_id_t id =
        (loom_symbol_id_t)(symbol - module->symbols.entries);
    loom_kernel_shape_function_t* function = &builder.functions[id];
    function->function = loom_func_like_cast(module, symbol->defining_op);
    if (!loom_func_like_body(function->function)) {
      continue;
    }
    function->arguments =
        loom_func_like_arg_ids(function->function, &function->argument_count);
    IREE_RETURN_IF_ERROR(loom_kernel_shape_append_node(
        &builder, LOOM_VALUE_ID_INVALID, &function->requirement));
    function->first_result = (uint32_t)builder.values->count;
    for (uint16_t i = 0; i < function->function.op->result_count; ++i) {
      uint32_t node;
      IREE_RETURN_IF_ERROR(loom_kernel_shape_append_node(
          &builder, LOOM_VALUE_ID_INVALID, &node));
    }
  }
  loom_module_for_each_symbol(module, symbol) {
    const loom_symbol_id_t id =
        (loom_symbol_id_t)(symbol - module->symbols.entries);
    loom_kernel_shape_function_t* function = &builder.functions[id];
    if (!loom_func_like_body(function->function)) {
      continue;
    }
    builder.current_symbol = id;
    for (uint16_t i = 0; i < function->argument_count; ++i) {
      const loom_value_id_t argument = function->arguments[i];
      uint32_t node;
      IREE_RETURN_IF_ERROR(
          loom_kernel_shape_value_node(&builder, argument, &node));
      IREE_RETURN_IF_ERROR(
          loom_value_set_index_add(out_requirements->sets, 0, argument,
                                   &builder.values->nodes[node].dependencies));
      loom_kernel_shape_enqueue(&builder, node);
      IREE_RETURN_IF_ERROR(loom_kernel_shape_type(&builder, argument));
    }
    loom_walk_result_t result;
    IREE_RETURN_IF_ERROR(loom_walk_region(
        module, loom_func_like_body(function->function), LOOM_WALK_PRE_ORDER,
        (loom_walk_callback_t){loom_kernel_shape_visit, &builder}, &result));
  }
  while (builder.first_queued != UINT32_MAX) {
    const uint32_t node_id = builder.first_queued;
    loom_kernel_shape_node_t* node = &builder.values->nodes[node_id];
    builder.first_queued = node->next_queued;
    if (builder.first_queued == UINT32_MAX) {
      builder.last_queued = UINT32_MAX;
    }
    node->queued = false;
    for (uint32_t watch = node->first_watch; watch != UINT32_MAX;
         watch = builder.watches[watch].next) {
      const loom_kernel_shape_edge_t* edge =
          &builder.edges[builder.watches[watch].edge];
      if (edge->condition != UINT32_MAX &&
          !loom_value_set_index_contains(
              out_requirements->sets,
              builder.values->nodes[edge->condition].dependencies,
              edge->formal)) {
        continue;
      }
      loom_kernel_shape_node_t* destination =
          &builder.values->nodes[edge->destination];
      loom_value_set_id_t combined = 0;
      IREE_RETURN_IF_ERROR(loom_value_set_index_union(
          out_requirements->sets, destination->dependencies,
          builder.values->nodes[edge->source].dependencies, &combined));
      if (combined != destination->dependencies) {
        destination->dependencies = combined;
        loom_kernel_shape_enqueue(&builder, edge->destination);
      }
    }
  }
  loom_module_for_each_symbol(module, symbol) {
    const loom_symbol_id_t id =
        (loom_symbol_id_t)(symbol - module->symbols.entries);
    const loom_kernel_shape_function_t* function = &builder.functions[id];
    if (loom_func_like_body(function->function)) {
      out_requirements->symbols[id].arguments =
          builder.values->nodes[function->requirement].dependencies;
    }
  }
  for (iree_host_size_t i = 0; i < builder.call_count; ++i) {
    const loom_kernel_shape_call_t* call = &builder.calls[i];
    const loom_kernel_shape_function_t* callee =
        &builder.functions[call->callee];
    loom_kernel_shape_requirement_t* caller =
        &out_requirements->symbols[call->caller];
    for (uint16_t j = 0; j < callee->argument_count; ++j) {
      if (loom_value_set_index_contains(
              out_requirements->sets,
              out_requirements->symbols[call->callee].arguments,
              callee->arguments[j])) {
        IREE_RETURN_IF_ERROR(loom_value_set_index_add(
            out_requirements->sets, caller->values, call->arguments.values[j],
            &caller->values));
      }
    }
  }
  return iree_ok_status();
}
