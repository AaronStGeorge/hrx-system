// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/refine_boundaries.h"

#include <string.h>

#include "loom/analysis/scc.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_defs.h"
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
};

static const loom_pass_statistic_def_t kRefineBoundariesStatistics[] = {
    {IREE_SVL("functions-visited"),
     IREE_SVL("Number of function bodies canonicalized.")},
    {IREE_SVL("functions-changed"),
     IREE_SVL("Number of function canonicalizer runs that changed IR.")},
    {IREE_SVL("boundary-facts-changed"),
     IREE_SVL("Number of fixed-point rounds that changed boundary facts.")},
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
// Function graph
//===----------------------------------------------------------------------===//

#define LOOM_REFINE_BOUNDARIES_DEFAULT_MAX_ITERATIONS 8

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
      memset(info->return_fact_defined, 0,
             info->result_count * sizeof(*info->return_fact_defined));
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
// Boundary collection
//===----------------------------------------------------------------------===//

typedef struct loom_refine_boundaries_collect_t {
  // Function graph for resolving callees.
  loom_refine_boundaries_graph_t* graph;

  // Current function facts after canonicalization.
  const loom_value_fact_table_t* function_facts;

  // Boundary facts being produced for the next fixed-point round.
  loom_value_fact_table_t* next_boundary_facts;

  // Function whose body is being walked.
  loom_refine_boundaries_function_t* current_function;
} loom_refine_boundaries_collect_t;

static iree_status_t loom_refine_boundaries_collect_return(
    loom_refine_boundaries_collect_t* collect, const loom_op_t* op) {
  loom_value_slice_t operands = loom_func_return_operands(op);
  loom_refine_boundaries_function_t* function = collect->current_function;
  iree_host_size_t count = operands.count < function->result_count
                               ? operands.count
                               : function->result_count;
  for (iree_host_size_t i = 0; i < count; ++i) {
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
  function->has_return_facts = true;
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
    loom_value_fact_table_t* next_boundary_facts,
    loom_refine_boundaries_function_t* function_info) {
  loom_canonicalizer_options_t options = {
      .seed_facts = seed_facts,
  };
  loom_canonicalizer_result_t canonicalize_result = {0};
  IREE_RETURN_IF_ERROR(loom_canonicalizer_run_function(
      canonicalizer, function_info->function, &options, &canonicalize_result));
  if (pass->statistics) {
    loom_pass_statistic_add(pass, LOOM_REFINE_BOUNDARIES_STAT_FUNCTIONS_VISITED,
                            1);
    if (canonicalize_result.changed) {
      loom_pass_statistic_add(pass,
                              LOOM_REFINE_BOUNDARIES_STAT_FUNCTIONS_CHANGED, 1);
    }
  }

  const loom_value_fact_table_t* function_facts =
      loom_canonicalizer_fact_table(canonicalizer);
  if (!function_facts) return iree_ok_status();

  loom_refine_boundaries_collect_t collect = {
      .graph = graph,
      .function_facts = function_facts,
      .next_boundary_facts = next_boundary_facts,
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

  iree_status_t status = loom_value_fact_table_initialize(
      current_boundary_facts, current_facts_arena, initial_capacity);

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

    for (iree_host_size_t scc_index = 0;
         iree_status_is_ok(status) && scc_index < sccs.count; ++scc_index) {
      const loom_scc_t* scc = &sccs.values[scc_index];
      for (iree_host_size_t member = 0;
           iree_status_is_ok(status) && member < scc->node_count; ++member) {
        iree_host_size_t node = scc->nodes[member];
        status = loom_refine_boundaries_run_function(
            pass, &canonicalizer, &graph, current_boundary_facts,
            next_boundary_facts, &graph.functions[node]);
      }
    }
    if (!iree_status_is_ok(status)) break;

    bool boundary_changed = !loom_refine_boundaries_fact_tables_equal(
        current_boundary_facts, next_boundary_facts);
    if (!boundary_changed) {
      converged = true;
      break;
    }
    if (pass->statistics) {
      loom_pass_statistic_add(
          pass, LOOM_REFINE_BOUNDARIES_STAT_BOUNDARY_FACTS_CHANGED, 1);
    }

    iree_arena_allocator_t* old_current_arena = current_facts_arena;
    current_facts_arena = next_facts_arena;
    next_facts_arena = old_current_arena;
    loom_value_fact_table_t* old_current_facts = current_boundary_facts;
    current_boundary_facts = next_boundary_facts;
    next_boundary_facts = old_current_facts;
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
