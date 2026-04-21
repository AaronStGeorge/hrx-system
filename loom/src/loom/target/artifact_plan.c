// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/artifact_plan.h"

#include <string.h>

#include "loom/ops/function_symbol_facts.h"

static bool loom_target_artifact_plan_symbol_ref_equal(loom_symbol_ref_t lhs,
                                                       loom_symbol_ref_t rhs) {
  return lhs.module_id == rhs.module_id && lhs.symbol_id == rhs.symbol_id;
}

static iree_status_t loom_target_artifact_plan_lookup_artifact(
    const loom_module_t* module, loom_symbol_ref_t artifact_symbol,
    loom_symbol_fact_table_t* fact_table,
    const loom_target_artifact_symbol_facts_t** out_artifact_facts) {
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
      fact_table, module, artifact_symbol, &base_facts));
  const loom_target_artifact_symbol_facts_t* artifact_facts =
      loom_target_artifact_symbol_facts_cast(base_facts);
  if (!artifact_facts) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "artifact plan root must resolve to target.artifact facts");
  }
  *out_artifact_facts = artifact_facts;
  return iree_ok_status();
}

static iree_status_t loom_target_artifact_plan_lookup_function(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    loom_symbol_id_t symbol_id,
    const loom_function_symbol_facts_t** out_function_facts) {
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup(fact_table, module,
                                                     symbol_id, &base_facts));
  *out_function_facts = loom_function_symbol_facts_cast(base_facts);
  return iree_ok_status();
}

static iree_status_t loom_target_artifact_plan_check_reachable_function(
    const loom_module_t* module, loom_symbol_ref_t artifact_symbol,
    loom_symbol_fact_table_t* fact_table, loom_symbol_id_t symbol_id,
    bool* out_has_body) {
  *out_has_body = false;
  const loom_function_symbol_facts_t* function_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_target_artifact_plan_lookup_function(
      module, fact_table, symbol_id, &function_facts));
  if (!function_facts) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "artifact plan closure reaches a non-function symbol");
  }
  if (function_facts->exports &&
      loom_symbol_ref_is_valid(function_facts->artifact_symbol) &&
      !loom_target_artifact_plan_symbol_ref_equal(
          function_facts->artifact_symbol, artifact_symbol)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "artifact plan closure reaches a function exported into another "
        "artifact");
  }
  *out_has_body = function_facts->has_body;
  return iree_ok_status();
}

static iree_status_t loom_target_artifact_plan_mark_closure(
    const loom_module_t* module, loom_symbol_ref_t artifact_symbol,
    loom_symbol_fact_table_t* fact_table, const loom_call_graph_t* call_graph,
    iree_arena_allocator_t* arena, uint8_t* function_marks,
    uint16_t* out_function_count) {
  loom_symbol_id_t* stack = NULL;
  if (call_graph->node_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, call_graph->node_count,
                                  sizeof(loom_symbol_id_t), (void**)&stack));
  }

  uint16_t stack_count = 0;
  for (uint16_t i = 0; i < call_graph->symbol_to_node_count; ++i) {
    if (!function_marks[i]) {
      continue;
    }
    const loom_call_graph_node_t* node = loom_call_graph_node(call_graph, i);
    if (!node) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "artifact plan entry function is missing from the call graph");
    }
    stack[stack_count++] = i;
  }

  while (stack_count > 0) {
    loom_symbol_id_t symbol_id = stack[--stack_count];
    const loom_call_graph_node_t* node =
        loom_call_graph_node(call_graph, symbol_id);
    if (!node) {
      continue;
    }
    for (uint16_t i = 0; i < node->callee_count; ++i) {
      loom_symbol_id_t callee_id = node->callees[i];
      bool callee_has_body = false;
      IREE_RETURN_IF_ERROR(loom_target_artifact_plan_check_reachable_function(
          module, artifact_symbol, fact_table, callee_id, &callee_has_body));
      if (!callee_has_body || function_marks[callee_id]) {
        continue;
      }
      function_marks[callee_id] = 1;
      ++(*out_function_count);
      stack[stack_count++] = callee_id;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_artifact_plan_allocate_ids(
    iree_arena_allocator_t* arena, uint16_t count, loom_symbol_id_t** out_ids) {
  *out_ids = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, count, sizeof(**out_ids), (void**)out_ids));
  return iree_ok_status();
}

iree_status_t loom_target_artifact_plan_build(
    const loom_module_t* module, loom_symbol_ref_t artifact_symbol,
    loom_symbol_fact_table_t* fact_table, const loom_call_graph_t* call_graph,
    iree_arena_allocator_t* arena, loom_target_artifact_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(fact_table);
  IREE_ASSERT_ARGUMENT(call_graph);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_plan);
  memset(out_plan, 0, sizeof(*out_plan));
  if (call_graph->module != module) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "artifact plan call graph belongs to another "
                            "module");
  }

  const loom_target_artifact_symbol_facts_t* artifact_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_target_artifact_plan_lookup_artifact(
      module, artifact_symbol, fact_table, &artifact_facts));

  uint8_t* entry_marks = NULL;
  uint8_t* function_marks = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, module->symbols.count,
                                                 sizeof(*entry_marks),
                                                 (void**)&entry_marks));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, module->symbols.count,
                                                 sizeof(*function_marks),
                                                 (void**)&function_marks));
  memset(entry_marks, 0, module->symbols.count * sizeof(*entry_marks));
  memset(function_marks, 0, module->symbols.count * sizeof(*function_marks));

  uint16_t entry_count = 0;
  uint16_t function_count = 0;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_function_symbol_facts_t* function_facts = NULL;
    IREE_RETURN_IF_ERROR(loom_target_artifact_plan_lookup_function(
        module, fact_table, (loom_symbol_id_t)i, &function_facts));
    if (!function_facts || !function_facts->exports ||
        !loom_target_artifact_plan_symbol_ref_equal(
            function_facts->artifact_symbol, artifact_symbol)) {
      continue;
    }
    if (!function_facts->has_body) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "artifact plan entry function must have a body");
    }
    entry_marks[i] = 1;
    function_marks[i] = 1;
    ++entry_count;
    ++function_count;
  }

  IREE_RETURN_IF_ERROR(loom_target_artifact_plan_mark_closure(
      module, artifact_symbol, fact_table, call_graph, arena, function_marks,
      &function_count));

  out_plan->artifact_symbol = artifact_symbol;
  out_plan->artifact_facts = artifact_facts;
  out_plan->entry_count = entry_count;
  out_plan->function_count = function_count;
  loom_symbol_id_t* entry_symbol_ids = NULL;
  IREE_RETURN_IF_ERROR(loom_target_artifact_plan_allocate_ids(
      arena, entry_count, &entry_symbol_ids));
  loom_symbol_id_t* function_symbol_ids = NULL;
  IREE_RETURN_IF_ERROR(loom_target_artifact_plan_allocate_ids(
      arena, function_count, &function_symbol_ids));
  out_plan->entry_symbol_ids = entry_symbol_ids;
  out_plan->function_symbol_ids = function_symbol_ids;

  uint16_t entry_index = 0;
  uint16_t function_index = 0;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    if (entry_marks[i]) {
      entry_symbol_ids[entry_index++] = (loom_symbol_id_t)i;
    }
    if (function_marks[i]) {
      function_symbol_ids[function_index++] = (loom_symbol_id_t)i;
    }
  }
  return iree_ok_status();
}
