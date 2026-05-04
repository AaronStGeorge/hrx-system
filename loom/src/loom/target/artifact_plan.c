// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/artifact_plan.h"

#include <string.h>

#include "loom/error/error_defs.h"
#include "loom/ops/func_symbol_facts.h"

static bool loom_target_artifact_plan_symbol_ref_equal(loom_symbol_ref_t lhs,
                                                       loom_symbol_ref_t rhs) {
  return lhs.module_id == rhs.module_id && lhs.symbol_id == rhs.symbol_id;
}

static iree_string_view_t loom_target_artifact_plan_symbol_name(
    const loom_module_t* module, loom_symbol_id_t symbol_id) {
  if (symbol_id >= module->symbols.count) {
    return IREE_SV("<unknown>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
  if (symbol->name_id >= module->strings.count) {
    return IREE_SV("<unknown>");
  }
  return module->strings.entries[symbol->name_id];
}

static iree_string_view_t loom_target_artifact_plan_symbol_ref_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (symbol_ref.module_id != 0) {
    return IREE_SV("<unknown>");
  }
  return loom_target_artifact_plan_symbol_name(module, symbol_ref.symbol_id);
}

static iree_status_t loom_target_artifact_plan_emit(
    iree_diagnostic_emitter_t diagnostic_emitter, const loom_op_t* op,
    uint16_t code, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count) {
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_TARGET, code),
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(diagnostic_emitter, &emission);
}

static iree_status_t loom_target_artifact_plan_reject(
    iree_diagnostic_emitter_t diagnostic_emitter, const loom_op_t* op,
    uint16_t code, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count, bool* out_valid) {
  *out_valid = false;
  return loom_target_artifact_plan_emit(diagnostic_emitter, op, code, params,
                                        param_count);
}

static iree_status_t loom_target_artifact_plan_lookup_artifact(
    const loom_module_t* module, loom_symbol_ref_t artifact_symbol,
    loom_symbol_fact_table_t* fact_table, bool* out_valid,
    const loom_target_artifact_symbol_facts_t** out_artifact_facts) {
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
      fact_table, module, artifact_symbol, &base_facts));
  const loom_target_artifact_symbol_facts_t* artifact_facts =
      loom_target_artifact_symbol_facts_cast(base_facts);
  if (!artifact_facts) {
    iree_string_view_t symbol_name =
        loom_target_artifact_plan_symbol_ref_name(module, artifact_symbol);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target artifact plan root '@%.*s' must name a target.artifact symbol",
        (int)symbol_name.size, symbol_name.data);
  }
  *out_artifact_facts = artifact_facts;
  *out_valid = true;
  return iree_ok_status();
}

static iree_status_t loom_target_artifact_plan_lookup_func(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    loom_symbol_id_t symbol_id,
    const loom_func_symbol_facts_t** out_func_facts) {
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup(fact_table, module,
                                                     symbol_id, &base_facts));
  *out_func_facts = loom_func_symbol_facts_cast(base_facts);
  return iree_ok_status();
}

static iree_status_t loom_target_artifact_plan_check_reachable_func(
    const loom_module_t* module, loom_symbol_ref_t artifact_symbol,
    const loom_target_artifact_symbol_facts_t* artifact_facts,
    loom_symbol_fact_table_t* fact_table,
    iree_diagnostic_emitter_t diagnostic_emitter, loom_symbol_id_t symbol_id,
    bool* out_valid, bool* out_has_body) {
  *out_has_body = false;
  const loom_func_symbol_facts_t* func_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_target_artifact_plan_lookup_func(
      module, fact_table, symbol_id, &func_facts));
  if (!func_facts) {
    iree_string_view_t symbol_name =
        loom_target_artifact_plan_symbol_name(module, symbol_id);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "verified artifact '%.*s' closure reached non-function symbol '@%.*s'",
        (int)artifact_facts->name.size, artifact_facts->name.data,
        (int)symbol_name.size, symbol_name.data);
  }
  if (func_facts->exports &&
      loom_symbol_ref_is_valid(func_facts->artifact_symbol) &&
      !loom_target_artifact_plan_symbol_ref_equal(func_facts->artifact_symbol,
                                                  artifact_symbol)) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(artifact_facts->name),
        loom_param_string(func_facts->name),
        loom_param_string(loom_target_artifact_plan_symbol_ref_name(
            module, func_facts->artifact_symbol)),
    };
    return loom_target_artifact_plan_reject(diagnostic_emitter,
                                            func_facts->func_op, 25, params,
                                            IREE_ARRAYSIZE(params), out_valid);
  }
  *out_has_body = func_facts->has_body;
  return iree_ok_status();
}

static iree_status_t loom_target_artifact_plan_mark_closure(
    const loom_module_t* module, loom_symbol_ref_t artifact_symbol,
    const loom_target_artifact_symbol_facts_t* artifact_facts,
    loom_symbol_fact_table_t* fact_table, const loom_call_graph_t* call_graph,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    uint8_t* func_marks, bool* out_valid, uint16_t* out_func_count) {
  loom_symbol_id_t* stack = NULL;
  if (call_graph->node_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, call_graph->node_count,
                                  sizeof(loom_symbol_id_t), (void**)&stack));
  }

  uint16_t stack_count = 0;
  for (uint16_t i = 0; i < call_graph->symbol_to_node_count; ++i) {
    if (!func_marks[i]) {
      continue;
    }
    const loom_call_graph_node_t* node = loom_call_graph_node(call_graph, i);
    if (!node) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "artifact plan entry func is missing from the call graph");
    }
    stack[stack_count++] = i;
  }

  while (*out_valid && stack_count > 0) {
    loom_symbol_id_t symbol_id = stack[--stack_count];
    const loom_call_graph_node_t* node =
        loom_call_graph_node(call_graph, symbol_id);
    if (!node) {
      continue;
    }
    for (uint16_t i = 0; i < node->callee_count; ++i) {
      loom_symbol_id_t callee_id = node->callees[i];
      bool callee_has_body = false;
      IREE_RETURN_IF_ERROR(loom_target_artifact_plan_check_reachable_func(
          module, artifact_symbol, artifact_facts, fact_table,
          diagnostic_emitter, callee_id, out_valid, &callee_has_body));
      if (!*out_valid) {
        return iree_ok_status();
      }
      if (!callee_has_body || func_marks[callee_id]) {
        continue;
      }
      func_marks[callee_id] = 1;
      ++(*out_func_count);
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

static iree_status_t loom_target_artifact_plan_assign_ordered_entries(
    const loom_module_t* module,
    const loom_target_artifact_symbol_facts_t* artifact_facts,
    loom_symbol_fact_table_t* fact_table,
    iree_diagnostic_emitter_t diagnostic_emitter, const uint8_t* entry_marks,
    uint16_t entry_count, uint16_t ordinal_count, bool* out_valid,
    loom_symbol_id_t* entry_symbol_ids) {
  if (entry_count == 0) {
    return iree_ok_status();
  }
  if (ordinal_count != 0 && ordinal_count != entry_count) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(artifact_facts->name),
        loom_param_u32(ordinal_count),
        loom_param_u32(entry_count),
    };
    return loom_target_artifact_plan_reject(
        diagnostic_emitter, artifact_facts->artifact_op, 28, params,
        IREE_ARRAYSIZE(params), out_valid);
  }

  if (ordinal_count == 0) {
    uint16_t entry_index = 0;
    for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
      if (entry_marks[i]) {
        entry_symbol_ids[entry_index++] = (loom_symbol_id_t)i;
      }
    }
    return iree_ok_status();
  }

  for (uint16_t i = 0; i < entry_count; ++i) {
    entry_symbol_ids[i] = LOOM_SYMBOL_ID_INVALID;
  }
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    if (!entry_marks[i]) {
      continue;
    }
    const loom_func_symbol_facts_t* func_facts = NULL;
    IREE_RETURN_IF_ERROR(loom_target_artifact_plan_lookup_func(
        module, fact_table, (loom_symbol_id_t)i, &func_facts));
    if (func_facts->export_ordinal >= entry_count) {
      const loom_diagnostic_param_t params[] = {
          loom_param_string(artifact_facts->name),
          loom_param_string(func_facts->name),
          loom_param_u32(func_facts->export_ordinal),
          loom_param_u32(entry_count),
      };
      return loom_target_artifact_plan_reject(
          diagnostic_emitter, func_facts->func_op, 29, params,
          IREE_ARRAYSIZE(params), out_valid);
    }
    loom_symbol_id_t* slot = &entry_symbol_ids[func_facts->export_ordinal];
    if (*slot != LOOM_SYMBOL_ID_INVALID) {
      const loom_func_symbol_facts_t* first_func_facts = NULL;
      IREE_RETURN_IF_ERROR(loom_target_artifact_plan_lookup_func(
          module, fact_table, *slot, &first_func_facts));
      const loom_diagnostic_param_t params[] = {
          loom_param_string(artifact_facts->name),
          loom_param_u32(func_facts->export_ordinal),
          loom_param_string(first_func_facts->name),
          loom_param_string(func_facts->name),
      };
      return loom_target_artifact_plan_reject(
          diagnostic_emitter, func_facts->func_op, 30, params,
          IREE_ARRAYSIZE(params), out_valid);
    }
    *slot = (loom_symbol_id_t)i;
  }
  for (uint16_t i = 0; i < entry_count; ++i) {
    if (entry_symbol_ids[i] == LOOM_SYMBOL_ID_INVALID) {
      const loom_diagnostic_param_t params[] = {
          loom_param_string(artifact_facts->name),
          loom_param_u32(i),
      };
      return loom_target_artifact_plan_reject(
          diagnostic_emitter, artifact_facts->artifact_op, 31, params,
          IREE_ARRAYSIZE(params), out_valid);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_target_artifact_plan_build(
    const loom_module_t* module, loom_symbol_ref_t artifact_symbol,
    loom_symbol_fact_table_t* fact_table, const loom_call_graph_t* call_graph,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    bool* out_valid, loom_target_artifact_plan_t* out_plan) {
  memset(out_plan, 0, sizeof(*out_plan));
  *out_valid = false;
  if (call_graph->module != module) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "artifact plan call graph belongs to another "
                            "module");
  }

  const loom_target_artifact_symbol_facts_t* artifact_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_target_artifact_plan_lookup_artifact(
      module, artifact_symbol, fact_table, out_valid, &artifact_facts));
  if (!*out_valid) {
    return iree_ok_status();
  }

  uint8_t* entry_marks = NULL;
  uint8_t* func_marks = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, module->symbols.count,
                                                 sizeof(*entry_marks),
                                                 (void**)&entry_marks));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module->symbols.count, sizeof(*func_marks), (void**)&func_marks));
  memset(entry_marks, 0, module->symbols.count * sizeof(*entry_marks));
  memset(func_marks, 0, module->symbols.count * sizeof(*func_marks));

  uint16_t entry_count = 0;
  uint16_t func_count = 0;
  uint16_t ordinal_count = 0;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_func_symbol_facts_t* func_facts = NULL;
    IREE_RETURN_IF_ERROR(loom_target_artifact_plan_lookup_func(
        module, fact_table, (loom_symbol_id_t)i, &func_facts));
    if (!func_facts || !func_facts->exports ||
        !loom_target_artifact_plan_symbol_ref_equal(func_facts->artifact_symbol,
                                                    artifact_symbol)) {
      continue;
    }
    if (!func_facts->has_body) {
      const loom_diagnostic_param_t params[] = {
          loom_param_string(artifact_facts->name),
          loom_param_string(func_facts->name),
      };
      return loom_target_artifact_plan_reject(
          diagnostic_emitter, func_facts->func_op, 26, params,
          IREE_ARRAYSIZE(params), out_valid);
    }
    if (!loom_target_artifact_plan_symbol_ref_equal(
            func_facts->target_symbol, artifact_facts->target_symbol)) {
      const loom_diagnostic_param_t params[] = {
          loom_param_string(artifact_facts->name),
          loom_param_string(loom_target_artifact_plan_symbol_ref_name(
              module, artifact_facts->target_symbol)),
          loom_param_string(func_facts->name),
          loom_param_string(loom_target_artifact_plan_symbol_ref_name(
              module, func_facts->target_symbol)),
      };
      return loom_target_artifact_plan_reject(
          diagnostic_emitter, func_facts->func_op, 27, params,
          IREE_ARRAYSIZE(params), out_valid);
    }
    entry_marks[i] = 1;
    func_marks[i] = 1;
    if (func_facts->has_export_ordinal) {
      ++ordinal_count;
    }
    ++entry_count;
    ++func_count;
  }

  IREE_RETURN_IF_ERROR(loom_target_artifact_plan_mark_closure(
      module, artifact_symbol, artifact_facts, fact_table, call_graph,
      diagnostic_emitter, arena, func_marks, out_valid, &func_count));
  if (!*out_valid) {
    return iree_ok_status();
  }

  out_plan->artifact_symbol = artifact_symbol;
  out_plan->artifact_facts = artifact_facts;
  out_plan->entry_count = entry_count;
  out_plan->func_count = func_count;
  loom_symbol_id_t* entry_symbol_ids = NULL;
  IREE_RETURN_IF_ERROR(loom_target_artifact_plan_allocate_ids(
      arena, entry_count, &entry_symbol_ids));
  loom_symbol_id_t* func_symbol_ids = NULL;
  IREE_RETURN_IF_ERROR(loom_target_artifact_plan_allocate_ids(
      arena, func_count, &func_symbol_ids));
  out_plan->entry_symbol_ids = entry_symbol_ids;
  out_plan->func_symbol_ids = func_symbol_ids;

  IREE_RETURN_IF_ERROR(loom_target_artifact_plan_assign_ordered_entries(
      module, artifact_facts, fact_table, diagnostic_emitter, entry_marks,
      entry_count, ordinal_count, out_valid, entry_symbol_ids));
  if (!*out_valid) {
    return iree_ok_status();
  }

  uint16_t func_index = 0;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    if (func_marks[i]) {
      func_symbol_ids[func_index++] = (loom_symbol_id_t)i;
    }
  }
  *out_valid = true;
  return iree_ok_status();
}
