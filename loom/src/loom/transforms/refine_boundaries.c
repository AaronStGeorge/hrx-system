// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/refine_boundaries.h"

#include <stdint.h>
#include <string.h>

#include "loom/analysis/scc.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/special_values.h"
#include "loom/transforms/canonicalize.h"
#include "loom/util/fact_table.h"
#include "loom/util/walk.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

enum {
  LOOM_REFINE_BOUNDARIES_STAT_FUNCTIONS_VISITED = 0,
  LOOM_REFINE_BOUNDARIES_STAT_FUNCTIONS_CHANGED = 1,
  LOOM_REFINE_BOUNDARIES_STAT_BOUNDARY_FACTS_CHANGED = 2,
  LOOM_REFINE_BOUNDARIES_STAT_BOUNDARY_REPLACEMENTS_CHANGED = 3,
  LOOM_REFINE_BOUNDARIES_STAT_BOUNDARY_REPLACEMENTS_APPLIED = 4,
  LOOM_REFINE_BOUNDARIES_STAT_BOUNDARY_CONSTANTS_MATERIALIZED = 5,
  LOOM_REFINE_BOUNDARIES_STAT_BOUNDARY_ARGUMENTS_PRUNED = 6,
  LOOM_REFINE_BOUNDARIES_STAT_BOUNDARY_RESULTS_PRUNED = 7,
};

static const loom_pass_statistic_def_t kRefineBoundariesStatistics[] = {
    {IREE_SVL("functions-visited"),
     IREE_SVL("Number of function bodies canonicalized.")},
    {IREE_SVL("functions-changed"),
     IREE_SVL("Number of function canonicalizer runs that changed IR.")},
    {IREE_SVL("boundary-facts-changed"),
     IREE_SVL("Number of fixed-point rounds that changed boundary facts.")},
    {IREE_SVL("boundary-replacements-changed"),
     IREE_SVL(
         "Number of fixed-point rounds that changed boundary replacements.")},
    {IREE_SVL("boundary-replacements-applied"),
     IREE_SVL("Number of direct boundary value replacements applied.")},
    {IREE_SVL("boundary-constants-materialized"),
     IREE_SVL("Number of exact boundary constants materialized.")},
    {IREE_SVL("boundary-arguments-pruned"),
     IREE_SVL("Number of unused internal function arguments removed.")},
    {IREE_SVL("boundary-results-pruned"),
     IREE_SVL("Number of unused internal function results removed.")},
};

static const loom_pass_info_t loom_refine_boundaries_pass_info_storage = {
    .name = IREE_SVL("refine-boundaries"),
    .description =
        IREE_SVL("Propagate direct-call boundary facts and canonicalize."),
    .kind = LOOM_PASS_MODULE,
    .statistic_defs = kRefineBoundariesStatistics,
    .statistic_count = IREE_ARRAYSIZE(kRefineBoundariesStatistics),
};

const loom_pass_info_t* loom_refine_boundaries_pass_info(void) {
  return &loom_refine_boundaries_pass_info_storage;
}

//===----------------------------------------------------------------------===//
// Replacement summaries
//===----------------------------------------------------------------------===//

#define LOOM_REFINE_BOUNDARIES_DEFAULT_MAX_ITERATIONS 8

#define LOOM_REFINE_BOUNDARIES_FORWARD_UNSEEN ((int32_t)-2)
#define LOOM_REFINE_BOUNDARIES_FORWARD_NONE ((int32_t)-1)

typedef enum loom_refine_boundaries_replacement_state_e {
  LOOM_REFINE_BOUNDARIES_REPLACEMENT_NONE = 0,
  LOOM_REFINE_BOUNDARIES_REPLACEMENT_VALUE = 1,
  LOOM_REFINE_BOUNDARIES_REPLACEMENT_CONFLICT = 2,
} loom_refine_boundaries_replacement_state_t;

typedef struct loom_refine_boundaries_replacement_entry_t {
  // Current summary state for this value.
  loom_refine_boundaries_replacement_state_t state;

  // Replacement value when state is VALUE.
  loom_value_id_t replacement;
} loom_refine_boundaries_replacement_entry_t;

typedef struct loom_refine_boundaries_replacement_table_t {
  // Arena that owns dense replacement entries.
  iree_arena_allocator_t* arena;

  // Dense entries indexed by original value ID.
  loom_refine_boundaries_replacement_entry_t* entries;

  // Highest initialized value ID plus one.
  iree_host_size_t count;

  // Allocated entry count.
  iree_host_size_t capacity;
} loom_refine_boundaries_replacement_table_t;

static iree_status_t loom_refine_boundaries_replacement_table_ensure_capacity(
    loom_refine_boundaries_replacement_table_t* table,
    iree_host_size_t minimum_count) {
  if (minimum_count <= table->capacity) return iree_ok_status();

  iree_host_size_t new_capacity = table->capacity > 0 ? table->capacity : 256;
  while (new_capacity < minimum_count) {
    if (new_capacity > SIZE_MAX / 2) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "replacement table capacity overflow");
    }
    new_capacity *= 2;
  }

  loom_refine_boundaries_replacement_entry_t* new_entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      table->arena, new_capacity, sizeof(*new_entries), (void**)&new_entries));
  memset(new_entries, 0, new_capacity * sizeof(*new_entries));
  if (table->count > 0) {
    memcpy(new_entries, table->entries, table->count * sizeof(*new_entries));
  }

  table->entries = new_entries;
  table->capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_replacement_table_initialize(
    loom_refine_boundaries_replacement_table_t* table,
    iree_arena_allocator_t* arena, iree_host_size_t initial_capacity) {
  memset(table, 0, sizeof(*table));
  table->arena = arena;
  return loom_refine_boundaries_replacement_table_ensure_capacity(
      table, initial_capacity);
}

static bool loom_refine_boundaries_replacement_table_lookup(
    const loom_refine_boundaries_replacement_table_t* table,
    loom_value_id_t old_value, loom_value_id_t* out_replacement) {
  if (old_value >= table->count) return false;
  const loom_refine_boundaries_replacement_entry_t* entry =
      &table->entries[old_value];
  if (entry->state != LOOM_REFINE_BOUNDARIES_REPLACEMENT_VALUE) return false;
  *out_replacement = entry->replacement;
  return true;
}

static bool loom_refine_boundaries_replacement_table_resolve(
    const loom_refine_boundaries_replacement_table_t* table,
    loom_value_id_t old_value, loom_value_id_t* out_replacement) {
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  if (!loom_refine_boundaries_replacement_table_lookup(table, old_value,
                                                       &replacement)) {
    return false;
  }

  for (iree_host_size_t depth = 0; depth < table->count; ++depth) {
    loom_value_id_t next = LOOM_VALUE_ID_INVALID;
    if (!loom_refine_boundaries_replacement_table_lookup(table, replacement,
                                                         &next)) {
      *out_replacement = replacement;
      return true;
    }
    if (next == replacement || next == old_value) return false;
    replacement = next;
  }
  return false;
}

static iree_status_t loom_refine_boundaries_replacement_table_entry(
    loom_refine_boundaries_replacement_table_t* table,
    loom_value_id_t old_value,
    loom_refine_boundaries_replacement_entry_t** out_entry) {
  *out_entry = NULL;
  if (old_value == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_replacement_table_ensure_capacity(
      table, (iree_host_size_t)old_value + 1));
  if (old_value >= table->count) {
    table->count = (iree_host_size_t)old_value + 1;
  }

  *out_entry = &table->entries[old_value];
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_replacement_table_define(
    loom_refine_boundaries_replacement_table_t* table,
    loom_value_id_t old_value, loom_value_id_t replacement) {
  if (replacement == LOOM_VALUE_ID_INVALID || old_value == replacement) {
    return iree_ok_status();
  }

  loom_refine_boundaries_replacement_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      loom_refine_boundaries_replacement_table_entry(table, old_value, &entry));
  if (!entry) return iree_ok_status();
  if (entry->state == LOOM_REFINE_BOUNDARIES_REPLACEMENT_NONE) {
    entry->state = LOOM_REFINE_BOUNDARIES_REPLACEMENT_VALUE;
    entry->replacement = replacement;
    return iree_ok_status();
  }
  if (entry->state == LOOM_REFINE_BOUNDARIES_REPLACEMENT_CONFLICT) {
    return iree_ok_status();
  }
  if (entry->replacement != replacement) {
    entry->state = LOOM_REFINE_BOUNDARIES_REPLACEMENT_CONFLICT;
    entry->replacement = LOOM_VALUE_ID_INVALID;
  }
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_replacement_table_block(
    loom_refine_boundaries_replacement_table_t* table,
    loom_value_id_t old_value) {
  loom_refine_boundaries_replacement_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      loom_refine_boundaries_replacement_table_entry(table, old_value, &entry));
  if (!entry) return iree_ok_status();
  entry->state = LOOM_REFINE_BOUNDARIES_REPLACEMENT_CONFLICT;
  entry->replacement = LOOM_VALUE_ID_INVALID;
  return iree_ok_status();
}

static bool loom_refine_boundaries_replacement_tables_equal(
    const loom_refine_boundaries_replacement_table_t* lhs,
    const loom_refine_boundaries_replacement_table_t* rhs) {
  iree_host_size_t count = lhs->count > rhs->count ? lhs->count : rhs->count;
  for (iree_host_size_t i = 0; i < count; ++i) {
    loom_value_id_t lhs_replacement = LOOM_VALUE_ID_INVALID;
    loom_value_id_t rhs_replacement = LOOM_VALUE_ID_INVALID;
    bool lhs_has = loom_refine_boundaries_replacement_table_lookup(
        lhs, (loom_value_id_t)i, &lhs_replacement);
    bool rhs_has = loom_refine_boundaries_replacement_table_lookup(
        rhs, (loom_value_id_t)i, &rhs_replacement);
    if (lhs_has != rhs_has) return false;
    if (lhs_has && lhs_replacement != rhs_replacement) return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Function graph
//===----------------------------------------------------------------------===//

typedef struct loom_refine_boundaries_function_t {
  // Function-like wrapper for the bodyful definition.
  loom_func_like_t function;

  // Entry block argument ids.
  const loom_value_id_t* argument_ids;

  // Number of entry block arguments.
  uint16_t argument_count;

  // Number of function result slots.
  uint16_t result_count;

  // True when all possible callers are in this module.
  bool is_internal;

  // True after return facts have been computed in the current round.
  bool has_return_facts;

  // Joined return facts for each result slot in the current round.
  loom_value_facts_t* return_facts;

  // Per-result bit saying whether return_facts has an observed return value.
  bool* return_fact_defined;

  // Forwarded argument index for each result slot, or a negative sentinel.
  int32_t* return_forward_argument_indices;
} loom_refine_boundaries_function_t;

typedef struct loom_refine_boundaries_graph_t {
  // Module being refined.
  loom_module_t* module;

  // Reset before each graph walk; owns walker stacks only.
  iree_arena_allocator_t* walk_arena;

  // Dense function nodes.
  loom_refine_boundaries_function_t* functions;

  // Number of function nodes.
  iree_host_size_t function_count;

  // Symbol-id to function-node map.
  iree_host_size_t* symbol_to_node;

  // Number of entries in symbol_to_node.
  iree_host_size_t symbol_to_node_count;
} loom_refine_boundaries_graph_t;

static bool loom_refine_boundaries_read_call(const loom_op_t* op,
                                             loom_symbol_ref_t* out_callee,
                                             loom_value_slice_t* out_operands,
                                             loom_value_slice_t* out_results) {
  if (loom_func_call_isa(op)) {
    *out_callee = loom_func_call_callee(op);
    *out_operands = loom_func_call_operands(op);
    *out_results = loom_func_call_results(op);
    return true;
  }
  if (loom_func_apply_isa(op)) {
    *out_callee = loom_func_apply_callee(op);
    *out_operands = loom_func_apply_operands(op);
    *out_results = loom_func_apply_results(op);
    return true;
  }
  return false;
}

static bool loom_refine_boundaries_callee_node(
    const loom_refine_boundaries_graph_t* graph, loom_symbol_ref_t callee,
    iree_host_size_t* out_node) {
  if (!loom_symbol_ref_is_valid(callee) || callee.module_id != 0 ||
      callee.symbol_id >= graph->symbol_to_node_count) {
    return false;
  }
  iree_host_size_t node = graph->symbol_to_node[callee.symbol_id];
  if (node == IREE_HOST_SIZE_MAX) return false;
  *out_node = node;
  return true;
}

typedef struct loom_refine_boundaries_successor_walk_t {
  // Function graph adapter.
  const loom_refine_boundaries_graph_t* graph;

  // SCC successor visitor.
  loom_scc_successor_visitor_t visitor;

  // Opaque visitor state.
  void* visitor_user_data;
} loom_refine_boundaries_successor_walk_t;

static iree_status_t loom_refine_boundaries_visit_successor_call(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_refine_boundaries_successor_walk_t* walk =
      (loom_refine_boundaries_successor_walk_t*)user_data;
  loom_symbol_ref_t callee = loom_symbol_ref_null();
  loom_value_slice_t operands = {0};
  loom_value_slice_t results = {0};
  if (!loom_refine_boundaries_read_call(op, &callee, &operands, &results)) {
    return iree_ok_status();
  }

  iree_host_size_t callee_node = IREE_HOST_SIZE_MAX;
  if (!loom_refine_boundaries_callee_node(walk->graph, callee, &callee_node)) {
    return iree_ok_status();
  }
  return walk->visitor(callee_node, walk->visitor_user_data);
}

static iree_status_t loom_refine_boundaries_visit_successors(
    iree_host_size_t node, void* graph_user_data,
    loom_scc_successor_visitor_t visitor, void* visitor_user_data) {
  const loom_refine_boundaries_graph_t* graph =
      (const loom_refine_boundaries_graph_t*)graph_user_data;
  if (node >= graph->function_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function graph node %" PRIhsz
                            " out of range for %" PRIhsz " functions",
                            node, graph->function_count);
  }

  loom_refine_boundaries_successor_walk_t walk = {
      .graph = graph,
      .visitor = visitor,
      .visitor_user_data = visitor_user_data,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  iree_arena_reset(graph->walk_arena);
  return loom_walk_function(
      graph->module, graph->functions[node].function, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_refine_boundaries_visit_successor_call,
                             &walk},
      graph->walk_arena, &walk_result);
}

static iree_status_t loom_refine_boundaries_build_graph(
    loom_module_t* module, iree_arena_allocator_t* arena,
    iree_arena_allocator_t* walk_arena,
    loom_refine_boundaries_graph_t* out_graph, loom_scc_list_t* out_sccs) {
  memset(out_graph, 0, sizeof(*out_graph));
  out_graph->module = module;
  out_graph->walk_arena = walk_arena;
  out_graph->symbol_to_node_count = module->symbols.count;
  if (module->symbols.count == 0) {
    *out_sccs = (loom_scc_list_t){0};
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module->symbols.count, sizeof(*out_graph->symbol_to_node),
      (void**)&out_graph->symbol_to_node));
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    out_graph->symbol_to_node[i] = IREE_HOST_SIZE_MAX;
  }

  iree_host_size_t function_count = 0;
  loom_symbol_t* symbol = NULL;
  loom_module_for_each_symbol(module, symbol) {
    if (!loom_symbol_kind_is_function_like(symbol->kind) ||
        !symbol->defining_op) {
      continue;
    }
    loom_func_like_t function =
        loom_func_like_cast(module, symbol->defining_op);
    if (loom_func_like_body(function)) ++function_count;
  }

  if (function_count == 0) {
    *out_sccs = (loom_scc_list_t){0};
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, function_count, sizeof(*out_graph->functions),
      (void**)&out_graph->functions));
  memset(out_graph->functions, 0,
         function_count * sizeof(*out_graph->functions));
  out_graph->function_count = function_count;

  iree_host_size_t node = 0;
  loom_module_for_each_symbol(module, symbol) {
    if (!loom_symbol_kind_is_function_like(symbol->kind) ||
        !symbol->defining_op) {
      continue;
    }
    loom_func_like_t function =
        loom_func_like_cast(module, symbol->defining_op);
    if (!loom_func_like_body(function)) continue;

    loom_symbol_id_t symbol_id =
        (loom_symbol_id_t)(symbol - module->symbols.entries);
    loom_refine_boundaries_function_t* info = &out_graph->functions[node];
    info->function = function;
    info->argument_ids =
        loom_func_like_arg_ids(function, &info->argument_count);
    info->result_count = function.op->result_count;
    info->is_internal = loom_func_like_visibility(function) == 0;
    if (info->result_count > 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, info->result_count, sizeof(*info->return_facts),
          (void**)&info->return_facts));
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, info->result_count, sizeof(*info->return_fact_defined),
          (void**)&info->return_fact_defined));
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, info->result_count,
          sizeof(*info->return_forward_argument_indices),
          (void**)&info->return_forward_argument_indices));
      memset(info->return_fact_defined, 0,
             info->result_count * sizeof(*info->return_fact_defined));
      for (uint16_t i = 0; i < info->result_count; ++i) {
        info->return_forward_argument_indices[i] =
            LOOM_REFINE_BOUNDARIES_FORWARD_UNSEEN;
      }
    }
    out_graph->symbol_to_node[symbol_id] = node;
    ++node;
  }

  loom_scc_graph_t scc_graph = {
      .node_count = function_count,
      .visit_successors = loom_refine_boundaries_visit_successors,
      .user_data = out_graph,
  };
  return loom_scc_compute(&scc_graph, NULL, arena, out_sccs);
}

//===----------------------------------------------------------------------===//
// Fact summaries
//===----------------------------------------------------------------------===//

static loom_value_facts_t loom_refine_boundaries_stable_fact(
    loom_value_facts_t facts) {
  facts.extension_id = LOOM_VALUE_FACT_EXTENSION_ID_NONE;
  return facts;
}

static loom_value_facts_t loom_refine_boundaries_join_facts(
    loom_value_facts_t lhs, loom_value_facts_t rhs) {
  lhs = loom_refine_boundaries_stable_fact(lhs);
  rhs = loom_refine_boundaries_stable_fact(rhs);
  if (loom_value_facts_equal(lhs, rhs)) return lhs;
  if (loom_value_facts_is_float(lhs) || loom_value_facts_is_float(rhs)) {
    return loom_value_facts_unknown();
  }
  loom_value_facts_t joined = loom_value_facts_unknown();
  loom_value_facts_meet(&lhs, &rhs, &joined);
  return joined;
}

static bool loom_refine_boundaries_table_has_entry(
    const loom_value_fact_table_t* table, loom_value_id_t value_id) {
  return value_id < table->count && table->entries[value_id].known_divisor != 0;
}

static iree_status_t loom_refine_boundaries_merge_fact(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_facts_t facts) {
  facts = loom_refine_boundaries_stable_fact(facts);
  if (loom_value_facts_is_unknown(facts) &&
      !loom_refine_boundaries_table_has_entry(table, value_id)) {
    return iree_ok_status();
  }
  if (!loom_refine_boundaries_table_has_entry(table, value_id)) {
    return loom_value_fact_table_define(table, value_id, facts);
  }
  loom_value_facts_t existing = loom_value_fact_table_lookup(table, value_id);
  loom_value_facts_t joined =
      loom_refine_boundaries_join_facts(existing, facts);
  return loom_value_fact_table_define(table, value_id, joined);
}

static bool loom_refine_boundaries_fact_tables_equal(
    const loom_value_fact_table_t* lhs, const loom_value_fact_table_t* rhs) {
  iree_host_size_t count = lhs->count > rhs->count ? lhs->count : rhs->count;
  for (iree_host_size_t i = 0; i < count; ++i) {
    loom_value_facts_t lhs_facts =
        loom_value_fact_table_lookup(lhs, (loom_value_id_t)i);
    loom_value_facts_t rhs_facts =
        loom_value_fact_table_lookup(rhs, (loom_value_id_t)i);
    if (!loom_value_facts_equal(lhs_facts, rhs_facts)) return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Replacement application
//===----------------------------------------------------------------------===//

static bool loom_refine_boundaries_value_has_uses(const loom_module_t* module,
                                                  loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  return value->use_count > 0 ||
         loom_module_value_has_type_uses(module, value_id);
}

static iree_status_t loom_refine_boundaries_apply_value_replacement(
    loom_module_t* module,
    const loom_refine_boundaries_replacement_table_t* replacements,
    loom_value_id_t old_value, int64_t* applied_count) {
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  if (!loom_refine_boundaries_replacement_table_resolve(replacements, old_value,
                                                        &replacement)) {
    return iree_ok_status();
  }
  if (replacement == LOOM_VALUE_ID_INVALID ||
      replacement >= module->values.count ||
      old_value >= module->values.count || replacement == old_value) {
    return iree_ok_status();
  }
  if (!loom_refine_boundaries_value_has_uses(module, old_value)) {
    return iree_ok_status();
  }
  if (!loom_type_equal(loom_module_value_type(module, old_value),
                       loom_module_value_type(module, replacement))) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_value_replace_all_uses_with(module, old_value, replacement));
  *applied_count += 1;
  return iree_ok_status();
}

typedef struct loom_refine_boundaries_apply_t {
  // Module being rewritten.
  loom_module_t* module;

  // Replacement summary for the current fixed-point round.
  const loom_refine_boundaries_replacement_table_t* replacements;

  // Boundary facts from the current fixed-point round.
  const loom_value_fact_table_t* boundary_facts;

  // Number of replacements applied while walking this function.
  int64_t* applied_count;

  // Number of constants materialized while walking this function.
  int64_t* materialized_count;
} loom_refine_boundaries_apply_t;

static iree_status_t loom_refine_boundaries_materialize_exact_value(
    loom_module_t* module, const loom_value_fact_table_t* boundary_facts,
    loom_builder_t* builder, loom_value_id_t old_value,
    loom_location_id_t location, int64_t* materialized_count) {
  if (!loom_refine_boundaries_value_has_uses(module, old_value)) {
    return iree_ok_status();
  }
  if (!loom_refine_boundaries_table_has_entry(boundary_facts, old_value)) {
    return iree_ok_status();
  }
  loom_value_facts_t facts =
      loom_value_fact_table_lookup(boundary_facts, old_value);
  loom_type_t type = loom_module_value_type(module, old_value);
  if (!loom_value_facts_can_materialize_constant(facts, type)) {
    return iree_ok_status();
  }

  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_constant_build(builder, facts, type, location, &replacement));
  IREE_RETURN_IF_ERROR(
      loom_value_replace_all_uses_with(module, old_value, replacement));
  *materialized_count += 1;
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_apply_op_boundary_values(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  if (op->result_count == 0) return iree_ok_status();

  loom_refine_boundaries_apply_t* apply =
      (loom_refine_boundaries_apply_t*)user_data;
  loom_builder_t builder;
  loom_builder_initialize(apply->module, &apply->module->arena,
                          op->parent_block, &builder);
  loom_builder_set_before(&builder, op);

  loom_value_id_t* results = loom_op_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_refine_boundaries_apply_value_replacement(
        apply->module, apply->replacements, results[i], apply->applied_count));
    IREE_RETURN_IF_ERROR(loom_refine_boundaries_materialize_exact_value(
        apply->module, apply->boundary_facts, &builder, results[i],
        op->location, apply->materialized_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_apply_function_boundary_values(
    loom_module_t* module,
    const loom_refine_boundaries_replacement_table_t* replacements,
    const loom_value_fact_table_t* boundary_facts,
    loom_refine_boundaries_function_t* function_info,
    iree_arena_allocator_t* walk_arena, int64_t* out_applied_count,
    int64_t* out_materialized_count) {
  *out_applied_count = 0;
  *out_materialized_count = 0;
  loom_region_t* body = loom_func_like_body(function_info->function);
  if (!body) return iree_ok_status();

  loom_block_t* entry_block = loom_region_entry_block(body);
  loom_builder_t entry_builder;
  loom_builder_initialize(module, &module->arena, entry_block, &entry_builder);
  if (entry_block->first_op) {
    loom_builder_set_before(&entry_builder, entry_block->first_op);
  } else {
    entry_builder.ip.parent_op = function_info->function.op;
  }
  for (uint16_t i = 0; i < function_info->argument_count; ++i) {
    loom_value_id_t argument = function_info->argument_ids[i];
    IREE_RETURN_IF_ERROR(loom_refine_boundaries_apply_value_replacement(
        module, replacements, argument, out_applied_count));
    IREE_RETURN_IF_ERROR(loom_refine_boundaries_materialize_exact_value(
        module, boundary_facts, &entry_builder, argument,
        function_info->function.op->location, out_materialized_count));
  }

  loom_refine_boundaries_apply_t apply = {
      .module = module,
      .replacements = replacements,
      .boundary_facts = boundary_facts,
      .applied_count = out_applied_count,
      .materialized_count = out_materialized_count,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  iree_arena_reset(walk_arena);
  return loom_walk_function(
      module, function_info->function, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_refine_boundaries_apply_op_boundary_values,
                             &apply},
      walk_arena, &walk_result);
}

//===----------------------------------------------------------------------===//
// Boundary collection
//===----------------------------------------------------------------------===//

typedef struct loom_refine_boundaries_collect_t {
  // Function graph for resolving callees.
  loom_refine_boundaries_graph_t* graph;

  // Current function facts after canonicalization.
  const loom_value_fact_table_t* function_facts;

  // Boundary facts being produced for the next fixed-point round.
  loom_value_fact_table_t* next_boundary_facts;

  // Boundary replacements being produced for the next fixed-point round.
  loom_refine_boundaries_replacement_table_t* next_boundary_replacements;

  // Function whose body is being walked.
  loom_refine_boundaries_function_t* current_function;
} loom_refine_boundaries_collect_t;

static int32_t loom_refine_boundaries_find_argument_index(
    const loom_refine_boundaries_function_t* function,
    loom_value_id_t value_id) {
  for (uint16_t i = 0; i < function->argument_count; ++i) {
    if (function->argument_ids[i] == value_id) return (int32_t)i;
  }
  return LOOM_REFINE_BOUNDARIES_FORWARD_NONE;
}

static void loom_refine_boundaries_join_return_forward(
    loom_refine_boundaries_function_t* function, iree_host_size_t result_index,
    int32_t argument_index) {
  int32_t* slot = &function->return_forward_argument_indices[result_index];
  if (*slot == LOOM_REFINE_BOUNDARIES_FORWARD_UNSEEN) {
    *slot = argument_index;
    return;
  }
  if (*slot != argument_index) {
    *slot = LOOM_REFINE_BOUNDARIES_FORWARD_NONE;
  }
}

static iree_status_t loom_refine_boundaries_collect_return(
    loom_refine_boundaries_collect_t* collect, const loom_op_t* op) {
  loom_value_slice_t operands = loom_func_return_operands(op);
  loom_refine_boundaries_function_t* function = collect->current_function;
  iree_host_size_t count = operands.count < function->result_count
                               ? operands.count
                               : function->result_count;
  for (iree_host_size_t i = 0; i < count; ++i) {
    int32_t argument_index = loom_refine_boundaries_find_argument_index(
        function, operands.values[i]);
    loom_refine_boundaries_join_return_forward(function, i, argument_index);

    loom_value_facts_t facts = loom_value_fact_table_lookup(
        collect->function_facts, operands.values[i]);
    if (!function->return_fact_defined[i]) {
      function->return_facts[i] = loom_refine_boundaries_stable_fact(facts);
      function->return_fact_defined[i] = true;
    } else {
      function->return_facts[i] =
          loom_refine_boundaries_join_facts(function->return_facts[i], facts);
    }
  }
  for (iree_host_size_t i = count; i < function->result_count; ++i) {
    loom_refine_boundaries_join_return_forward(
        function, i, LOOM_REFINE_BOUNDARIES_FORWARD_NONE);
  }
  function->has_return_facts = true;
  return iree_ok_status();
}

static bool loom_refine_boundaries_values_have_equal_types(
    const loom_module_t* module, loom_value_id_t lhs, loom_value_id_t rhs) {
  if (lhs == LOOM_VALUE_ID_INVALID || rhs == LOOM_VALUE_ID_INVALID ||
      lhs >= module->values.count || rhs >= module->values.count) {
    return false;
  }
  return loom_type_equal(loom_module_value_type(module, lhs),
                         loom_module_value_type(module, rhs));
}

static iree_status_t loom_refine_boundaries_collect_argument_equality(
    loom_refine_boundaries_collect_t* collect,
    const loom_refine_boundaries_function_t* callee_info,
    loom_value_slice_t operands) {
  if (!callee_info->is_internal) return iree_ok_status();
  for (uint16_t i = 1; i < callee_info->argument_count; ++i) {
    loom_value_id_t old_argument = callee_info->argument_ids[i];
    if (i >= operands.count) {
      IREE_RETURN_IF_ERROR(loom_refine_boundaries_replacement_table_block(
          collect->next_boundary_replacements, old_argument));
      continue;
    }

    loom_value_id_t replacement_argument = LOOM_VALUE_ID_INVALID;
    for (uint16_t j = 0; j < i && j < operands.count; ++j) {
      if (operands.values[j] != operands.values[i]) continue;
      if (!loom_refine_boundaries_values_have_equal_types(
              collect->graph->module, old_argument,
              callee_info->argument_ids[j])) {
        continue;
      }
      replacement_argument = callee_info->argument_ids[j];
      break;
    }

    if (replacement_argument == LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_refine_boundaries_replacement_table_block(
          collect->next_boundary_replacements, old_argument));
    } else {
      IREE_RETURN_IF_ERROR(loom_refine_boundaries_replacement_table_define(
          collect->next_boundary_replacements, old_argument,
          replacement_argument));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_collect_return_forwarding(
    loom_refine_boundaries_collect_t* collect, const loom_op_t* call_op,
    const loom_refine_boundaries_function_t* callee_info,
    loom_value_slice_t operands, loom_value_slice_t results) {
  if (!callee_info->return_forward_argument_indices) return iree_ok_status();
  iree_host_size_t count = results.count < callee_info->result_count
                               ? results.count
                               : callee_info->result_count;
  const loom_tied_result_t* tied_results = loom_op_tied_results(call_op);
  for (iree_host_size_t i = 0; i < count; ++i) {
    bool result_is_tied = false;
    for (uint16_t j = 0; j < call_op->tied_result_count; ++j) {
      if (tied_results[j].result_index == i) {
        result_is_tied = true;
        break;
      }
    }
    if (result_is_tied) continue;

    int32_t argument_index = callee_info->return_forward_argument_indices[i];
    if (argument_index < 0 ||
        (iree_host_size_t)argument_index >= operands.count) {
      continue;
    }
    loom_value_id_t result = results.values[i];
    loom_value_id_t replacement = operands.values[argument_index];
    if (!loom_refine_boundaries_values_have_equal_types(collect->graph->module,
                                                        result, replacement)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_refine_boundaries_replacement_table_define(
        collect->next_boundary_replacements, result, replacement));
  }
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_collect_call(
    loom_refine_boundaries_collect_t* collect, const loom_op_t* op) {
  loom_symbol_ref_t callee = loom_symbol_ref_null();
  loom_value_slice_t operands = {0};
  loom_value_slice_t results = {0};
  if (!loom_refine_boundaries_read_call(op, &callee, &operands, &results)) {
    return iree_ok_status();
  }

  iree_host_size_t callee_node = IREE_HOST_SIZE_MAX;
  if (!loom_refine_boundaries_callee_node(collect->graph, callee,
                                          &callee_node)) {
    return iree_ok_status();
  }

  loom_refine_boundaries_function_t* callee_info =
      &collect->graph->functions[callee_node];
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_collect_argument_equality(
      collect, callee_info, operands));
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_collect_return_forwarding(
      collect, op, callee_info, operands, results));

  if (callee_info->is_internal) {
    iree_host_size_t count = operands.count < callee_info->argument_count
                                 ? operands.count
                                 : callee_info->argument_count;
    for (iree_host_size_t i = 0; i < count; ++i) {
      loom_value_facts_t facts = loom_value_fact_table_lookup(
          collect->function_facts, operands.values[i]);
      IREE_RETURN_IF_ERROR(loom_refine_boundaries_merge_fact(
          collect->next_boundary_facts, callee_info->argument_ids[i], facts));
    }
  }

  if (!callee_info->has_return_facts) return iree_ok_status();
  iree_host_size_t count = results.count < callee_info->result_count
                               ? results.count
                               : callee_info->result_count;
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (!callee_info->return_fact_defined[i]) continue;
    IREE_RETURN_IF_ERROR(loom_refine_boundaries_merge_fact(
        collect->next_boundary_facts, results.values[i],
        callee_info->return_facts[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_collect_op(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_refine_boundaries_collect_t* collect =
      (loom_refine_boundaries_collect_t*)user_data;
  if (loom_func_return_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_refine_boundaries_collect_return(collect, op));
  }
  return loom_refine_boundaries_collect_call(collect, op);
}

static iree_status_t loom_refine_boundaries_run_function(
    loom_pass_t* pass, loom_canonicalizer_t* canonicalizer,
    loom_refine_boundaries_graph_t* graph, loom_value_fact_table_t* seed_facts,
    const loom_refine_boundaries_replacement_table_t* seed_replacements,
    loom_value_fact_table_t* next_boundary_facts,
    loom_refine_boundaries_replacement_table_t* next_boundary_replacements,
    loom_refine_boundaries_function_t* function_info) {
  int64_t replacements_applied = 0;
  int64_t constants_materialized = 0;
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_apply_function_boundary_values(
      graph->module, seed_replacements, seed_facts, function_info,
      graph->walk_arena, &replacements_applied, &constants_materialized));

  loom_canonicalizer_options_t options = {
      .seed_facts = seed_facts,
  };
  loom_canonicalizer_result_t canonicalize_result = {0};
  IREE_RETURN_IF_ERROR(loom_canonicalizer_run_function(
      canonicalizer, function_info->function, &options, &canonicalize_result));
  if (pass->statistics) {
    loom_pass_statistic_add(pass, LOOM_REFINE_BOUNDARIES_STAT_FUNCTIONS_VISITED,
                            1);
    if (replacements_applied > 0 || constants_materialized > 0 ||
        canonicalize_result.changed) {
      loom_pass_statistic_add(pass,
                              LOOM_REFINE_BOUNDARIES_STAT_FUNCTIONS_CHANGED, 1);
    }
    if (replacements_applied > 0) {
      loom_pass_statistic_add(
          pass, LOOM_REFINE_BOUNDARIES_STAT_BOUNDARY_REPLACEMENTS_APPLIED,
          replacements_applied);
    }
    if (constants_materialized > 0) {
      loom_pass_statistic_add(
          pass, LOOM_REFINE_BOUNDARIES_STAT_BOUNDARY_CONSTANTS_MATERIALIZED,
          constants_materialized);
    }
  }

  const loom_value_fact_table_t* function_facts =
      loom_canonicalizer_fact_table(canonicalizer);
  if (!function_facts) return iree_ok_status();

  loom_refine_boundaries_collect_t collect = {
      .graph = graph,
      .function_facts = function_facts,
      .next_boundary_facts = next_boundary_facts,
      .next_boundary_replacements = next_boundary_replacements,
      .current_function = function_info,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  iree_arena_reset(graph->walk_arena);
  return loom_walk_function(
      graph->module, function_info->function, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_refine_boundaries_collect_op, &collect},
      graph->walk_arena, &walk_result);
}

//===----------------------------------------------------------------------===//
// Boundary pruning
//===----------------------------------------------------------------------===//

typedef struct loom_refine_boundaries_prune_plan_t {
  // Arguments to remove, indexed by the original callee argument ordinal.
  bool* prune_arguments;

  // Results to remove, indexed by the original callee result ordinal.
  bool* prune_results;

  // Original callee argument count for this plan.
  uint16_t argument_count;

  // Original callee result count for this plan.
  uint16_t result_count;

  // True when at least one argument is marked for pruning.
  bool has_prunable_arguments;

  // True when at least one result is marked for pruning.
  bool has_prunable_results;
} loom_refine_boundaries_prune_plan_t;

static bool loom_refine_boundaries_argument_is_prunable(
    const loom_module_t* module, loom_value_id_t argument) {
  if (argument == LOOM_VALUE_ID_INVALID || argument >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, argument);
  return value->use_count == 0 &&
         !loom_module_value_has_type_uses(module, argument);
}

static bool loom_refine_boundaries_result_is_tied(const loom_op_t* op,
                                                  uint16_t result_index) {
  const loom_tied_result_t* tied_results = loom_op_tied_results(op);
  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    if (tied_results[i].result_index == result_index) return true;
  }
  return false;
}

static bool loom_refine_boundaries_result_is_prunable(
    const loom_module_t* module, const loom_op_t* op, uint16_t result_index) {
  if (result_index >= op->result_count) return false;
  if (loom_refine_boundaries_result_is_tied(op, result_index)) return false;
  loom_value_id_t result = loom_op_const_results(op)[result_index];
  if (result == LOOM_VALUE_ID_INVALID || result >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, result);
  return value->use_count == 0 &&
         !loom_module_value_has_type_uses(module, result);
}

static iree_status_t loom_refine_boundaries_build_prune_plans(
    loom_module_t* module, const loom_refine_boundaries_graph_t* graph,
    iree_arena_allocator_t* arena,
    loom_refine_boundaries_prune_plan_t** out_plans) {
  *out_plans = NULL;
  if (graph->function_count == 0) return iree_ok_status();

  loom_refine_boundaries_prune_plan_t* plans = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, graph->function_count, sizeof(*plans), (void**)&plans));
  memset(plans, 0, graph->function_count * sizeof(*plans));

  for (iree_host_size_t node = 0; node < graph->function_count; ++node) {
    const loom_refine_boundaries_function_t* function_info =
        &graph->functions[node];
    if (!function_info->is_internal) continue;

    loom_region_t* body = loom_func_like_body(function_info->function);
    if (!body || body->block_count == 0) continue;
    loom_block_t* entry_block = loom_region_entry_block(body);

    loom_refine_boundaries_prune_plan_t* plan = &plans[node];
    if (entry_block->arg_count > 0) {
      plan->argument_count = entry_block->arg_count;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, entry_block->arg_count, sizeof(*plan->prune_arguments),
          (void**)&plan->prune_arguments));
      memset(plan->prune_arguments, 0,
             entry_block->arg_count * sizeof(*plan->prune_arguments));

      for (uint16_t i = 0; i < entry_block->arg_count; ++i) {
        loom_value_id_t argument = loom_block_arg_id(entry_block, i);
        if (!loom_refine_boundaries_argument_is_prunable(module, argument)) {
          continue;
        }
        plan->prune_arguments[i] = true;
        plan->has_prunable_arguments = true;
      }
    }

    if (function_info->result_count > 0) {
      plan->result_count = function_info->result_count;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, function_info->result_count, sizeof(*plan->prune_results),
          (void**)&plan->prune_results));
      memset(plan->prune_results, 0,
             function_info->result_count * sizeof(*plan->prune_results));

      for (uint16_t i = 0; i < function_info->result_count; ++i) {
        if (!loom_refine_boundaries_result_is_prunable(
                module, function_info->function.op, i)) {
          continue;
        }
        plan->prune_results[i] = true;
        plan->has_prunable_results = true;
      }
    }
  }

  *out_plans = plans;
  return iree_ok_status();
}

static void loom_refine_boundaries_recompute_prune_plan(
    loom_refine_boundaries_prune_plan_t* plan) {
  plan->has_prunable_arguments = false;
  for (uint16_t i = 0; i < plan->argument_count; ++i) {
    if (plan->prune_arguments[i]) {
      plan->has_prunable_arguments = true;
      break;
    }
  }
  plan->has_prunable_results = false;
  for (uint16_t i = 0; i < plan->result_count; ++i) {
    if (plan->prune_results[i]) {
      plan->has_prunable_results = true;
      break;
    }
  }
}

typedef struct loom_refine_boundaries_prune_call_walk_t {
  // Function graph for resolving direct callees.
  const loom_refine_boundaries_graph_t* graph;

  // Dense prune plans indexed by function node.
  loom_refine_boundaries_prune_plan_t* plans;
} loom_refine_boundaries_prune_call_walk_t;

static iree_status_t loom_refine_boundaries_preflight_pruned_call(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;

  loom_refine_boundaries_prune_call_walk_t* walk =
      (loom_refine_boundaries_prune_call_walk_t*)user_data;
  loom_symbol_ref_t callee = loom_symbol_ref_null();
  loom_value_slice_t operands = {0};
  loom_value_slice_t results = {0};
  if (!loom_refine_boundaries_read_call(op, &callee, &operands, &results)) {
    return iree_ok_status();
  }

  iree_host_size_t callee_node = IREE_HOST_SIZE_MAX;
  if (!loom_refine_boundaries_callee_node(walk->graph, callee, &callee_node)) {
    return iree_ok_status();
  }

  loom_refine_boundaries_prune_plan_t* plan = &walk->plans[callee_node];
  if (!plan->has_prunable_arguments && !plan->has_prunable_results) {
    return iree_ok_status();
  }
  if (plan->has_prunable_arguments && operands.count != plan->argument_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "direct call operand count %u"
        " does not match callee argument count %u during boundary pruning",
        (unsigned)operands.count, (unsigned)plan->argument_count);
  }
  if (plan->has_prunable_results && results.count != plan->result_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "direct call result count %u"
        " does not match callee result count %u during boundary pruning",
        (unsigned)results.count, (unsigned)plan->result_count);
  }

  const loom_tied_result_t* tied_results = loom_op_tied_results(op);
  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    uint16_t operand_index = tied_results[i].operand_index;
    if (plan->has_prunable_arguments && operand_index >= plan->argument_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "tied result operand index %u is out of range for %u argument(s)",
          (unsigned)operand_index, (unsigned)plan->argument_count);
    }
    if (plan->has_prunable_arguments) {
      plan->prune_arguments[operand_index] = false;
    }

    uint16_t result_index = tied_results[i].result_index;
    if (plan->has_prunable_results && result_index >= plan->result_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "tied result index %u is out of range for %u result(s)",
          (unsigned)result_index, (unsigned)plan->result_count);
    }
    if (plan->has_prunable_results) {
      plan->prune_results[result_index] = false;
    }
  }

  if (plan->has_prunable_results) {
    for (uint16_t i = 0; i < plan->result_count; ++i) {
      if (!plan->prune_results[i]) continue;
      if (!loom_refine_boundaries_result_is_prunable(walk->graph->module, op,
                                                     i)) {
        plan->prune_results[i] = false;
      }
    }
  }
  loom_refine_boundaries_recompute_prune_plan(plan);
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_copy_result_names(
    loom_module_t* module, const loom_op_t* old_op, loom_op_t* new_op,
    const uint16_t* old_to_new_result_indices) {
  const loom_value_id_t* old_results = loom_op_const_results(old_op);
  loom_value_id_t* new_results = loom_op_results(new_op);
  for (uint16_t i = 0; i < old_op->result_count; ++i) {
    uint16_t new_index = old_to_new_result_indices[i];
    if (new_index == UINT16_MAX) continue;
    if (new_index >= new_op->result_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "replacement call result index mismatch");
    }
    loom_value_id_t old_result = old_results[i];
    loom_value_id_t new_result = new_results[new_index];
    if (old_result == LOOM_VALUE_ID_INVALID ||
        new_result == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    if (old_result >= module->values.count ||
        new_result >= module->values.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "call result value out of range");
    }
    loom_string_id_t name_id = loom_module_value(module, old_result)->name_id;
    if (name_id == LOOM_STRING_ID_INVALID) continue;
    loom_value_t* new_value = loom_module_value(module, new_result);
    if (new_value->name_id == LOOM_STRING_ID_INVALID) {
      new_value->name_id = name_id;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_build_pruned_call(
    loom_module_t* module, loom_op_t* op,
    const loom_refine_boundaries_prune_plan_t* plan,
    iree_arena_allocator_t* arena, uint16_t** out_old_to_new_result_indices,
    loom_op_t** out_new_op) {
  *out_old_to_new_result_indices = NULL;
  *out_new_op = NULL;
  loom_symbol_ref_t callee = loom_symbol_ref_null();
  loom_value_slice_t operands = {0};
  loom_value_slice_t results = {0};
  if (!loom_refine_boundaries_read_call(op, &callee, &operands, &results)) {
    return iree_ok_status();
  }

  loom_value_id_t* kept_operands = NULL;
  uint16_t* old_to_new_operand_indices = NULL;
  if (operands.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, operands.count, sizeof(*kept_operands), (void**)&kept_operands));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, operands.count, sizeof(*old_to_new_operand_indices),
        (void**)&old_to_new_operand_indices));
    for (iree_host_size_t i = 0; i < operands.count; ++i) {
      old_to_new_operand_indices[i] = UINT16_MAX;
    }
  }
  iree_host_size_t kept_operand_count = 0;
  for (iree_host_size_t i = 0; i < operands.count; ++i) {
    if (i < plan->argument_count && plan->prune_arguments[i]) continue;
    old_to_new_operand_indices[i] = (uint16_t)kept_operand_count;
    kept_operands[kept_operand_count++] = operands.values[i];
  }

  uint16_t* old_to_new_result_indices = NULL;
  if (results.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, results.count, sizeof(*old_to_new_result_indices),
        (void**)&old_to_new_result_indices));
    for (iree_host_size_t i = 0; i < results.count; ++i) {
      old_to_new_result_indices[i] = UINT16_MAX;
    }
  }
  iree_host_size_t kept_result_count = 0;
  for (iree_host_size_t i = 0; i < results.count; ++i) {
    if (i < plan->result_count && plan->prune_results[i]) continue;
    old_to_new_result_indices[i] = (uint16_t)kept_result_count++;
  }

  loom_tied_result_t* tied_results = NULL;
  if (op->tied_result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, op->tied_result_count,
                                                   sizeof(*tied_results),
                                                   (void**)&tied_results));
    const loom_tied_result_t* old_tied_results = loom_op_tied_results(op);
    for (uint16_t i = 0; i < op->tied_result_count; ++i) {
      tied_results[i] = old_tied_results[i];
      uint16_t old_operand_index = old_tied_results[i].operand_index;
      if (old_operand_index >= operands.count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "tied result operand index %u is out of range for %u operand(s)",
            (unsigned)old_operand_index, (unsigned)operands.count);
      }
      uint16_t new_operand_index =
          old_to_new_operand_indices[old_operand_index];
      if (new_operand_index == UINT16_MAX) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "boundary pruning cannot remove tied operand %u",
            (unsigned)old_operand_index);
      }
      tied_results[i].operand_index = new_operand_index;
      uint16_t old_result_index = old_tied_results[i].result_index;
      if (old_result_index >= results.count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "tied result index %u is out of range for %u result(s)",
            (unsigned)old_result_index, (unsigned)results.count);
      }
      uint16_t new_result_index = old_to_new_result_indices[old_result_index];
      if (new_result_index == UINT16_MAX) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "boundary pruning cannot remove tied result %u",
                                (unsigned)old_result_index);
      }
      tied_results[i].result_index = new_result_index;
    }
  }

  loom_type_t* result_types = NULL;
  if (kept_result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, kept_result_count,
                                                   sizeof(*result_types),
                                                   (void**)&result_types));
    for (iree_host_size_t i = 0; i < results.count; ++i) {
      uint16_t new_index = old_to_new_result_indices[i];
      if (new_index == UINT16_MAX) continue;
      result_types[new_index] =
          loom_module_value_type(module, results.values[i]);
    }
  }
  *out_old_to_new_result_indices = old_to_new_result_indices;

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, op->parent_block, &builder);
  loom_builder_set_before(&builder, op);
  if (loom_func_call_isa(op)) {
    uint8_t purity = loom_func_call_purity(op);
    loom_func_call_build_flags_t build_flags =
        purity ? LOOM_FUNC_CALL_BUILD_FLAG_HAS_PURITY : 0;
    return loom_func_call_build(
        &builder, build_flags, purity, callee, kept_operands,
        kept_operand_count, result_types, kept_result_count, tied_results,
        op->tied_result_count, op->location, out_new_op);
  }
  if (loom_func_apply_isa(op)) {
    uint8_t purity = loom_func_apply_purity(op);
    loom_func_apply_build_flags_t build_flags =
        purity ? LOOM_FUNC_APPLY_BUILD_FLAG_HAS_PURITY : 0;
    return loom_func_apply_build(
        &builder, build_flags, purity, callee, kept_operands,
        kept_operand_count, result_types, kept_result_count, tied_results,
        op->tied_result_count, op->location, out_new_op);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "expected func.call or func.apply");
}

typedef struct loom_refine_boundaries_rewrite_call_walk_t {
  // Module being rewritten.
  loom_module_t* module;

  // Function graph for resolving direct callees.
  const loom_refine_boundaries_graph_t* graph;

  // Dense prune plans indexed by function node.
  loom_refine_boundaries_prune_plan_t* plans;

  // Scratch arena for filtered operand and result type arrays.
  iree_arena_allocator_t* arena;
} loom_refine_boundaries_rewrite_call_walk_t;

static iree_status_t loom_refine_boundaries_rewrite_pruned_call(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;

  loom_refine_boundaries_rewrite_call_walk_t* walk =
      (loom_refine_boundaries_rewrite_call_walk_t*)user_data;
  loom_symbol_ref_t callee = loom_symbol_ref_null();
  loom_value_slice_t operands = {0};
  loom_value_slice_t results = {0};
  if (!loom_refine_boundaries_read_call(op, &callee, &operands, &results)) {
    return iree_ok_status();
  }

  iree_host_size_t callee_node = IREE_HOST_SIZE_MAX;
  if (!loom_refine_boundaries_callee_node(walk->graph, callee, &callee_node)) {
    return iree_ok_status();
  }

  loom_refine_boundaries_prune_plan_t* plan = &walk->plans[callee_node];
  if (!plan->has_prunable_arguments && !plan->has_prunable_results) {
    return iree_ok_status();
  }

  uint16_t* old_to_new_result_indices = NULL;
  loom_op_t* new_op = NULL;
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_build_pruned_call(
      walk->module, op, plan, walk->arena, &old_to_new_result_indices,
      &new_op));
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_copy_result_names(
      walk->module, op, new_op, old_to_new_result_indices));

  const loom_value_id_t* old_results = loom_op_const_results(op);
  const loom_value_id_t* new_results = loom_op_const_results(new_op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    uint16_t new_index = old_to_new_result_indices[i];
    if (new_index == UINT16_MAX) continue;
    IREE_RETURN_IF_ERROR(loom_value_replace_all_uses_with(
        walk->module, old_results[i], new_results[new_index]));
  }
  return loom_op_erase(walk->module, op);
}

static iree_status_t loom_refine_boundaries_preflight_pruned_calls(
    loom_module_t* module, const loom_refine_boundaries_graph_t* graph,
    loom_refine_boundaries_prune_plan_t* plans,
    iree_arena_allocator_t* walk_arena) {
  loom_refine_boundaries_prune_call_walk_t walk = {
      .graph = graph,
      .plans = plans,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  iree_arena_reset(walk_arena);
  return loom_walk_region(module, module->body, LOOM_WALK_PRE_ORDER,
                          (loom_walk_callback_t){
                              loom_refine_boundaries_preflight_pruned_call,
                              &walk,
                          },
                          walk_arena, &walk_result);
}

static iree_status_t loom_refine_boundaries_rewrite_pruned_calls(
    loom_module_t* module, const loom_refine_boundaries_graph_t* graph,
    loom_refine_boundaries_prune_plan_t* plans,
    iree_arena_allocator_t* walk_arena) {
  loom_refine_boundaries_rewrite_call_walk_t walk = {
      .module = module,
      .graph = graph,
      .plans = plans,
      .arena = walk_arena,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  iree_arena_reset(walk_arena);
  return loom_walk_region(module, module->body, LOOM_WALK_PRE_ORDER,
                          (loom_walk_callback_t){
                              loom_refine_boundaries_rewrite_pruned_call,
                              &walk,
                          },
                          walk_arena, &walk_result);
}

typedef struct loom_refine_boundaries_return_list_t {
  // Function return ops discovered before mutation.
  loom_op_t** ops;

  // Number of return ops in |ops|.
  iree_host_size_t count;

  // Allocated op pointer capacity.
  iree_host_size_t capacity;

  // Arena owning |ops|.
  iree_arena_allocator_t* arena;
} loom_refine_boundaries_return_list_t;

static iree_status_t loom_refine_boundaries_append_return_op(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  if (!loom_func_return_isa(op)) return iree_ok_status();

  loom_refine_boundaries_return_list_t* list =
      (loom_refine_boundaries_return_list_t*)user_data;
  if (list->count >= list->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        list->arena, list->count, list->count + 1, sizeof(*list->ops),
        &list->capacity, (void**)&list->ops));
  }
  list->ops[list->count++] = op;
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_collect_return_ops(
    loom_module_t* module, loom_func_like_t function,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* walk_arena,
    loom_refine_boundaries_return_list_t* out_list) {
  memset(out_list, 0, sizeof(*out_list));
  out_list->arena = arena;
  out_list->capacity = 4;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, out_list->capacity,
                                                 sizeof(*out_list->ops),
                                                 (void**)&out_list->ops));

  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  iree_arena_reset(walk_arena);
  return loom_walk_function(
      module, function, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_refine_boundaries_append_return_op, out_list},
      walk_arena, &walk_result);
}

static iree_status_t loom_refine_boundaries_rewrite_pruned_return(
    loom_module_t* module, loom_op_t* return_op,
    const loom_refine_boundaries_prune_plan_t* plan,
    iree_arena_allocator_t* arena) {
  loom_value_slice_t operands = loom_func_return_operands(return_op);
  if (operands.count != plan->result_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "func.return operand count %u"
        " does not match callee result count %u during boundary pruning",
        (unsigned)operands.count, (unsigned)plan->result_count);
  }

  loom_value_id_t* kept_operands = NULL;
  uint16_t kept_count = 0;
  if (operands.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, operands.count, sizeof(*kept_operands), (void**)&kept_operands));
    for (uint16_t i = 0; i < operands.count; ++i) {
      if (plan->prune_results[i]) continue;
      kept_operands[kept_count++] = operands.values[i];
    }
  }

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, return_op->parent_block,
                          &builder);
  loom_builder_set_before(&builder, return_op);
  loom_op_t* new_return_op = NULL;
  IREE_RETURN_IF_ERROR(loom_func_return_build(&builder, kept_operands,
                                              kept_count, return_op->location,
                                              &new_return_op));
  return loom_op_erase(module, return_op);
}

static iree_status_t loom_refine_boundaries_rewrite_pruned_returns(
    loom_module_t* module, const loom_refine_boundaries_graph_t* graph,
    loom_refine_boundaries_prune_plan_t* plans, iree_arena_allocator_t* arena,
    iree_arena_allocator_t* walk_arena) {
  for (iree_host_size_t node = 0; node < graph->function_count; ++node) {
    const loom_refine_boundaries_prune_plan_t* plan = &plans[node];
    if (!plan->has_prunable_results) continue;

    loom_refine_boundaries_return_list_t returns = {0};
    IREE_RETURN_IF_ERROR(loom_refine_boundaries_collect_return_ops(
        module, graph->functions[node].function, arena, walk_arena, &returns));
    for (iree_host_size_t i = 0; i < returns.count; ++i) {
      IREE_RETURN_IF_ERROR(loom_refine_boundaries_rewrite_pruned_return(
          module, returns.ops[i], plan, arena));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_remove_pruned_results(
    loom_module_t* module, const loom_refine_boundaries_graph_t* graph,
    loom_refine_boundaries_prune_plan_t* plans, iree_arena_allocator_t* arena,
    int64_t* out_pruned_count) {
  *out_pruned_count = 0;
  for (iree_host_size_t node = 0; node < graph->function_count; ++node) {
    const loom_refine_boundaries_prune_plan_t* plan = &plans[node];
    if (!plan->has_prunable_results) continue;
    loom_op_t* function_op = graph->functions[node].function.op;
    if (function_op->result_count != plan->result_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "callee result count changed before boundary pruning");
    }
    uint16_t removed_count = 0;
    IREE_RETURN_IF_ERROR(loom_op_remove_results(
        module, function_op, plan->prune_results, arena, &removed_count));
    *out_pruned_count += removed_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_remove_pruned_arguments(
    loom_module_t* module, const loom_refine_boundaries_graph_t* graph,
    loom_refine_boundaries_prune_plan_t* plans, int64_t* out_pruned_count) {
  *out_pruned_count = 0;
  for (iree_host_size_t node = 0; node < graph->function_count; ++node) {
    const loom_refine_boundaries_prune_plan_t* plan = &plans[node];
    if (!plan->has_prunable_arguments) continue;
    loom_region_t* body = loom_func_like_body(graph->functions[node].function);
    if (!body || body->block_count == 0) continue;
    loom_block_t* entry_block = loom_region_entry_block(body);
    if (entry_block->arg_count != plan->argument_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "callee argument count changed before boundary pruning");
    }
    for (uint16_t i = plan->argument_count; i > 0; --i) {
      uint16_t argument_index = (uint16_t)(i - 1);
      if (!plan->prune_arguments[argument_index]) continue;
      IREE_RETURN_IF_ERROR(
          loom_block_remove_arg(module, entry_block, argument_index));
      *out_pruned_count += 1;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_prune_internal_boundaries(
    loom_module_t* module, const loom_refine_boundaries_graph_t* graph,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* walk_arena,
    int64_t* out_pruned_argument_count, int64_t* out_pruned_result_count) {
  *out_pruned_argument_count = 0;
  *out_pruned_result_count = 0;
  loom_refine_boundaries_prune_plan_t* plans = NULL;
  IREE_RETURN_IF_ERROR(
      loom_refine_boundaries_build_prune_plans(module, graph, arena, &plans));
  if (!plans) return iree_ok_status();

  IREE_RETURN_IF_ERROR(loom_refine_boundaries_preflight_pruned_calls(
      module, graph, plans, walk_arena));
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_rewrite_pruned_calls(
      module, graph, plans, walk_arena));
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_rewrite_pruned_returns(
      module, graph, plans, arena, walk_arena));
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_remove_pruned_results(
      module, graph, plans, arena, out_pruned_result_count));
  return loom_refine_boundaries_remove_pruned_arguments(
      module, graph, plans, out_pruned_argument_count);
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

iree_status_t loom_refine_boundaries_run(loom_pass_t* pass,
                                         loom_module_t* module) {
  iree_host_size_t initial_capacity = 0;
  if (!iree_host_size_checked_add(module->values.count, 64,
                                  &initial_capacity)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "initial fact table capacity overflow");
  }

  iree_arena_allocator_t facts_arena_a;
  iree_arena_allocator_t facts_arena_b;
  iree_arena_allocator_t iteration_arena;
  iree_arena_allocator_t walk_arena;
  iree_arena_initialize(pass->arena->block_pool, &facts_arena_a);
  iree_arena_initialize(pass->arena->block_pool, &facts_arena_b);
  iree_arena_initialize(pass->arena->block_pool, &iteration_arena);
  iree_arena_initialize(pass->arena->block_pool, &walk_arena);

  iree_arena_allocator_t* current_facts_arena = &facts_arena_a;
  iree_arena_allocator_t* next_facts_arena = &facts_arena_b;
  loom_value_fact_table_t boundary_facts_a = {0};
  loom_value_fact_table_t boundary_facts_b = {0};
  loom_value_fact_table_t* current_boundary_facts = &boundary_facts_a;
  loom_value_fact_table_t* next_boundary_facts = &boundary_facts_b;
  loom_refine_boundaries_replacement_table_t boundary_replacements_a = {0};
  loom_refine_boundaries_replacement_table_t boundary_replacements_b = {0};
  loom_refine_boundaries_replacement_table_t* current_boundary_replacements =
      &boundary_replacements_a;
  loom_refine_boundaries_replacement_table_t* next_boundary_replacements =
      &boundary_replacements_b;

  iree_status_t status = loom_value_fact_table_initialize(
      current_boundary_facts, current_facts_arena, initial_capacity);
  if (iree_status_is_ok(status)) {
    status = loom_refine_boundaries_replacement_table_initialize(
        current_boundary_replacements, current_facts_arena, initial_capacity);
  }

  loom_canonicalizer_t canonicalizer = {0};
  bool canonicalizer_initialized = false;
  if (iree_status_is_ok(status)) {
    status = loom_canonicalizer_initialize(module, pass->arena, &canonicalizer);
    canonicalizer_initialized = iree_status_is_ok(status);
  }

  bool converged = false;
  for (uint32_t iteration = 0;
       iree_status_is_ok(status) &&
       iteration < LOOM_REFINE_BOUNDARIES_DEFAULT_MAX_ITERATIONS;
       ++iteration) {
    iree_arena_reset(&iteration_arena);
    iree_arena_reset(next_facts_arena);

    loom_refine_boundaries_graph_t graph = {0};
    loom_scc_list_t sccs = {0};
    status = loom_refine_boundaries_build_graph(module, &iteration_arena,
                                                &walk_arena, &graph, &sccs);
    if (!iree_status_is_ok(status)) break;

    memset(next_boundary_facts, 0, sizeof(*next_boundary_facts));
    status = loom_value_fact_table_initialize(
        next_boundary_facts, next_facts_arena, initial_capacity);
    if (!iree_status_is_ok(status)) break;
    memset(next_boundary_replacements, 0, sizeof(*next_boundary_replacements));
    status = loom_refine_boundaries_replacement_table_initialize(
        next_boundary_replacements, next_facts_arena, initial_capacity);
    if (!iree_status_is_ok(status)) break;

    for (iree_host_size_t scc_index = 0;
         iree_status_is_ok(status) && scc_index < sccs.count; ++scc_index) {
      const loom_scc_t* scc = &sccs.values[scc_index];
      for (iree_host_size_t member = 0;
           iree_status_is_ok(status) && member < scc->node_count; ++member) {
        iree_host_size_t node = scc->nodes[member];
        status = loom_refine_boundaries_run_function(
            pass, &canonicalizer, &graph, current_boundary_facts,
            current_boundary_replacements, next_boundary_facts,
            next_boundary_replacements, &graph.functions[node]);
      }
    }
    if (!iree_status_is_ok(status)) break;

    bool boundary_facts_changed = !loom_refine_boundaries_fact_tables_equal(
        current_boundary_facts, next_boundary_facts);
    bool boundary_replacements_changed =
        !loom_refine_boundaries_replacement_tables_equal(
            current_boundary_replacements, next_boundary_replacements);
    if (!boundary_facts_changed && !boundary_replacements_changed) {
      int64_t pruned_argument_count = 0;
      int64_t pruned_result_count = 0;
      status = loom_refine_boundaries_prune_internal_boundaries(
          module, &graph, &iteration_arena, &walk_arena, &pruned_argument_count,
          &pruned_result_count);
      if (!iree_status_is_ok(status)) break;
      if (pruned_argument_count > 0 || pruned_result_count > 0) {
        if (pass->statistics) {
          if (pruned_argument_count > 0) {
            loom_pass_statistic_add(
                pass, LOOM_REFINE_BOUNDARIES_STAT_BOUNDARY_ARGUMENTS_PRUNED,
                pruned_argument_count);
          }
          if (pruned_result_count > 0) {
            loom_pass_statistic_add(
                pass, LOOM_REFINE_BOUNDARIES_STAT_BOUNDARY_RESULTS_PRUNED,
                pruned_result_count);
          }
        }
        continue;
      }
      converged = true;
      break;
    }
    if (pass->statistics) {
      if (boundary_facts_changed) {
        loom_pass_statistic_add(
            pass, LOOM_REFINE_BOUNDARIES_STAT_BOUNDARY_FACTS_CHANGED, 1);
      }
      if (boundary_replacements_changed) {
        loom_pass_statistic_add(
            pass, LOOM_REFINE_BOUNDARIES_STAT_BOUNDARY_REPLACEMENTS_CHANGED, 1);
      }
    }

    iree_arena_allocator_t* old_current_arena = current_facts_arena;
    current_facts_arena = next_facts_arena;
    next_facts_arena = old_current_arena;
    loom_value_fact_table_t* old_current_facts = current_boundary_facts;
    current_boundary_facts = next_boundary_facts;
    next_boundary_facts = old_current_facts;
    loom_refine_boundaries_replacement_table_t* old_current_replacements =
        current_boundary_replacements;
    current_boundary_replacements = next_boundary_replacements;
    next_boundary_replacements = old_current_replacements;
  }
  if (iree_status_is_ok(status) && !converged) {
    status =
        iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                         "refine-boundaries did not converge in %u iterations",
                         LOOM_REFINE_BOUNDARIES_DEFAULT_MAX_ITERATIONS);
  }

  if (canonicalizer_initialized) {
    loom_canonicalizer_deinitialize(&canonicalizer);
  }
  iree_arena_deinitialize(&walk_arena);
  iree_arena_deinitialize(&iteration_arena);
  iree_arena_deinitialize(&facts_arena_b);
  iree_arena_deinitialize(&facts_arena_a);
  return status;
}
