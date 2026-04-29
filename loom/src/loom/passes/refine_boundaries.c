// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/passes/refine_boundaries.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "loom/analysis/scc.h"
#include "loom/analysis/type_refinement.h"
#include "loom/error/error_defs.h"
#include "loom/ir/context.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ir/type_refinement.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/special_values.h"
#include "loom/ops/type_registry.h"
#include "loom/pass/pipeline.h"
#include "loom/pass/registry.h"
#include "loom/passes/canonicalize.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/util/fact_table.h"
#include "loom/util/walk.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

enum {
  LOOM_REFINE_BOUNDARIES_STAT_FUNCTIONS_CANONICALIZED = 0,
  LOOM_REFINE_BOUNDARIES_STAT_FUNCTIONS_CHANGED = 1,
  LOOM_REFINE_BOUNDARIES_STAT_BOUNDARY_FACTS_CHANGED = 2,
  LOOM_REFINE_BOUNDARIES_STAT_BOUNDARY_REPLACEMENTS_CHANGED = 3,
  LOOM_REFINE_BOUNDARIES_STAT_BOUNDARY_REPLACEMENTS_APPLIED = 4,
  LOOM_REFINE_BOUNDARIES_STAT_BOUNDARY_CONSTANTS_MATERIALIZED = 5,
  LOOM_REFINE_BOUNDARIES_STAT_BOUNDARY_ARGUMENTS_PRUNED = 6,
  LOOM_REFINE_BOUNDARIES_STAT_BOUNDARY_RESULTS_PRUNED = 7,
  LOOM_REFINE_BOUNDARIES_STAT_BOUNDARY_SIGNATURE_TYPES_REFINED = 8,
  LOOM_REFINE_BOUNDARIES_STAT_BOUNDARY_SPECIALIZATIONS_CREATED = 9,
};

static const loom_pass_statistic_def_t kRefineBoundariesStatistics[] = {
    {IREE_SVL("functions-canonicalized"),
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
    {IREE_SVL("boundary-signature-types-refined"),
     IREE_SVL("Number of internal boundary value types refined.")},
    {IREE_SVL("boundary-specializations-created"),
     IREE_SVL("Number of private function specializations created.")},
};

static const loom_pass_option_def_t kRefineBoundariesOptions[] = {
    {IREE_SVL("max-iterations"),
     IREE_SVL("Maximum number of boundary fixed-point iterations.")},
};

static const loom_pass_info_t loom_refine_boundaries_pass_info_storage = {
    .name = IREE_SVL("refine-boundaries"),
    .description = IREE_SVL(
        "Propagate direct-call boundary facts and types and canonicalize."),
    .kind = LOOM_PASS_MODULE,
    .option_defs = kRefineBoundariesOptions,
    .option_count = IREE_ARRAYSIZE(kRefineBoundariesOptions),
    .statistic_defs = kRefineBoundariesStatistics,
    .statistic_count = IREE_ARRAYSIZE(kRefineBoundariesStatistics),
};

const loom_pass_info_t* loom_refine_boundaries_pass_info(void) {
  return &loom_refine_boundaries_pass_info_storage;
}

static iree_status_t loom_refine_boundaries_parse_option(
    void* user_data, iree_string_view_t name, iree_string_view_t value) {
  loom_refine_boundaries_options_t* options =
      (loom_refine_boundaries_options_t*)user_data;
  if (iree_string_view_equal(name, IREE_SV("max-iterations"))) {
    if (options->max_iterations != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "duplicate option 'max-iterations' for pass 'refine-boundaries'");
    }
    IREE_RETURN_IF_ERROR(loom_pass_option_parse_uint32(
        IREE_SV("refine-boundaries"), name, value, &options->max_iterations));
    if (options->max_iterations == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pass 'refine-boundaries' option "
                              "'max-iterations' must be greater than 0");
    }
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown option '%.*s' for pass 'refine-boundaries'",
                          (int)name.size, name.data);
}

iree_status_t loom_refine_boundaries_create(loom_pass_t* pass,
                                            iree_string_view_t options_string) {
  loom_refine_boundaries_options_t* options = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(pass->instance_arena,
                                           sizeof(*options), (void**)&options));
  memset(options, 0, sizeof(*options));
  if (pass->decoded_options) {
    for (uint16_t i = 0; i < pass->decoded_options->option_count; ++i) {
      const loom_pass_decoded_option_t* option =
          &pass->decoded_options->options[i];
      if (!option->present) {
        continue;
      }
      if (iree_string_view_equal(option->schema->name,
                                 IREE_SV("max-iterations"))) {
        options->max_iterations = option->uint32_value;
        continue;
      }
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "unknown decoded option '%.*s' for pass 'refine-boundaries'",
          (int)option->schema->name.size, option->schema->name.data);
    }
  } else {
    IREE_RETURN_IF_ERROR(
        loom_pass_options_parse(pass->info->name, options_string,
                                (loom_pass_option_parse_callback_t){
                                    .fn = loom_refine_boundaries_parse_option,
                                    .user_data = options,
                                }));
  }
  pass->state = options;
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_fail(
    loom_pass_t* pass, const loom_module_t* module, const loom_op_t* op,
    iree_string_view_t reason, iree_status_code_t status_code) {
  iree_string_view_t scope = op ? loom_op_name(module, op) : IREE_SV("module");
  loom_diagnostic_param_t params[] = {
      loom_param_string(pass->info->name),
      loom_param_string(scope),
      loom_param_string(reason),
  };
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 2),
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  IREE_RETURN_IF_ERROR(
      iree_diagnostic_emit(pass->diagnostic_emitter, &emission));
  return iree_make_status(status_code, "%.*s failed while refining %.*s: %.*s",
                          (int)pass->info->name.size, pass->info->name.data,
                          (int)scope.size, scope.data, (int)reason.size,
                          reason.data);
}

//===----------------------------------------------------------------------===//
// Replacement summaries
//===----------------------------------------------------------------------===//

#define LOOM_REFINE_BOUNDARIES_DEFAULT_MAX_ITERATIONS 32

#define LOOM_REFINE_BOUNDARIES_FORWARD_UNSEEN ((int32_t)-2)
#define LOOM_REFINE_BOUNDARIES_FORWARD_NONE ((int32_t)-1)

typedef enum loom_refine_boundaries_replacement_state_e {
  LOOM_REFINE_BOUNDARIES_REPLACEMENT_NONE = 0,
  LOOM_REFINE_BOUNDARIES_REPLACEMENT_VALUE = 1,
  LOOM_REFINE_BOUNDARIES_REPLACEMENT_CONFLICT = 2,
} loom_refine_boundaries_replacement_state_t;

typedef struct loom_refine_boundaries_replacement_entry_t {
  // Original value ID summarized by this entry.
  loom_value_id_t old_value;

  // Current summary state for this value.
  loom_refine_boundaries_replacement_state_t state;

  // Replacement value when state is VALUE.
  loom_value_id_t replacement;
} loom_refine_boundaries_replacement_entry_t;

typedef struct loom_refine_boundaries_replacement_table_t {
  // Arena that owns sparse replacement entries.
  iree_arena_allocator_t* arena;

  // Open-addressed replacement entries keyed by original value ID.
  loom_refine_boundaries_replacement_entry_t* entries;

  // Number of installed replacement entries.
  iree_host_size_t entry_count;

  // Allocated entry count.
  iree_host_size_t entry_capacity;
} loom_refine_boundaries_replacement_table_t;

static iree_host_size_t loom_refine_boundaries_value_hash(
    loom_value_id_t value_id) {
  uint32_t hash = value_id;
  hash ^= hash >> 16;
  hash *= 0x7feb352du;
  hash ^= hash >> 15;
  hash *= 0x846ca68bu;
  hash ^= hash >> 16;
  return (iree_host_size_t)hash;
}

static iree_status_t loom_refine_boundaries_replacement_capacity_for_count(
    iree_host_size_t entry_count, iree_host_size_t* out_entry_capacity) {
  iree_host_size_t minimum_capacity = 0;
  if (!iree_host_size_checked_mul(entry_count, 2, &minimum_capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replacement table capacity overflow");
  }
  minimum_capacity = iree_max(minimum_capacity, 16);
  iree_host_size_t entry_capacity =
      iree_host_size_next_power_of_two(minimum_capacity);
  if (entry_capacity < minimum_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replacement table capacity overflow");
  }
  *out_entry_capacity = entry_capacity;
  return iree_ok_status();
}

static void loom_refine_boundaries_replacement_entries_initialize(
    loom_refine_boundaries_replacement_entry_t* entries,
    iree_host_size_t entry_capacity) {
  for (iree_host_size_t i = 0; i < entry_capacity; ++i) {
    entries[i] = (loom_refine_boundaries_replacement_entry_t){
        .old_value = LOOM_VALUE_ID_INVALID,
        .state = LOOM_REFINE_BOUNDARIES_REPLACEMENT_NONE,
        .replacement = LOOM_VALUE_ID_INVALID,
    };
  }
}

static loom_refine_boundaries_replacement_entry_t*
loom_refine_boundaries_replacement_table_find_slot(
    loom_refine_boundaries_replacement_entry_t* entries,
    iree_host_size_t entry_capacity, loom_value_id_t old_value) {
  IREE_ASSERT(entry_capacity > 0);
  IREE_ASSERT(iree_host_size_is_power_of_two(entry_capacity));
  const iree_host_size_t mask = entry_capacity - 1;
  iree_host_size_t slot = loom_refine_boundaries_value_hash(old_value) & mask;
  while (true) {
    loom_refine_boundaries_replacement_entry_t* entry = &entries[slot];
    if (entry->old_value == LOOM_VALUE_ID_INVALID ||
        entry->old_value == old_value) {
      return entry;
    }
    slot = (slot + 1) & mask;
  }
}

static const loom_refine_boundaries_replacement_entry_t*
loom_refine_boundaries_replacement_table_find_const_slot(
    const loom_refine_boundaries_replacement_table_t* table,
    loom_value_id_t old_value) {
  if (table->entry_capacity == 0) {
    return NULL;
  }
  const loom_refine_boundaries_replacement_entry_t* entry =
      loom_refine_boundaries_replacement_table_find_slot(
          table->entries, table->entry_capacity, old_value);
  return entry->old_value == old_value ? entry : NULL;
}

static void loom_refine_boundaries_replacement_table_insert_entry(
    loom_refine_boundaries_replacement_entry_t* entries,
    iree_host_size_t entry_capacity,
    loom_refine_boundaries_replacement_entry_t source_entry) {
  loom_refine_boundaries_replacement_entry_t* entry =
      loom_refine_boundaries_replacement_table_find_slot(
          entries, entry_capacity, source_entry.old_value);
  IREE_ASSERT(entry->old_value == LOOM_VALUE_ID_INVALID);
  *entry = source_entry;
}

static iree_status_t loom_refine_boundaries_replacement_table_ensure_capacity(
    loom_refine_boundaries_replacement_table_t* table,
    iree_host_size_t required_entry_count) {
  if (required_entry_count <= table->entry_capacity / 2) {
    return iree_ok_status();
  }
  iree_host_size_t new_entry_capacity = 0;
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_replacement_capacity_for_count(
      required_entry_count, &new_entry_capacity));
  loom_refine_boundaries_replacement_entry_t* new_entries = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(table->arena, new_entry_capacity,
                                sizeof(*new_entries), (void**)&new_entries));
  loom_refine_boundaries_replacement_entries_initialize(new_entries,
                                                        new_entry_capacity);
  for (iree_host_size_t i = 0; i < table->entry_capacity; ++i) {
    loom_refine_boundaries_replacement_entry_t entry = table->entries[i];
    if (entry.old_value == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    loom_refine_boundaries_replacement_table_insert_entry(
        new_entries, new_entry_capacity, entry);
  }
  table->entries = new_entries;
  table->entry_capacity = new_entry_capacity;
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_replacement_table_initialize(
    loom_refine_boundaries_replacement_table_t* table,
    iree_arena_allocator_t* arena) {
  memset(table, 0, sizeof(*table));
  table->arena = arena;
  return iree_ok_status();
}

static bool loom_refine_boundaries_replacement_table_lookup(
    const loom_refine_boundaries_replacement_table_t* table,
    loom_value_id_t old_value, loom_value_id_t* out_replacement) {
  const loom_refine_boundaries_replacement_entry_t* entry =
      loom_refine_boundaries_replacement_table_find_const_slot(table,
                                                               old_value);
  if (!entry || entry->state != LOOM_REFINE_BOUNDARIES_REPLACEMENT_VALUE) {
    return false;
  }
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

  for (iree_host_size_t depth = 0; depth < table->entry_count; ++depth) {
    loom_value_id_t next = LOOM_VALUE_ID_INVALID;
    if (!loom_refine_boundaries_replacement_table_lookup(table, replacement,
                                                         &next)) {
      *out_replacement = replacement;
      return true;
    }
    if (next == replacement || next == old_value) {
      return false;
    }
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
  if (table->entry_capacity > 0) {
    loom_refine_boundaries_replacement_entry_t* entry =
        loom_refine_boundaries_replacement_table_find_slot(
            table->entries, table->entry_capacity, old_value);
    if (entry->old_value == old_value) {
      *out_entry = entry;
      return iree_ok_status();
    }
  }
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_replacement_table_ensure_capacity(
      table, table->entry_count + 1));
  loom_refine_boundaries_replacement_entry_t* entry =
      loom_refine_boundaries_replacement_table_find_slot(
          table->entries, table->entry_capacity, old_value);
  if (entry->old_value == LOOM_VALUE_ID_INVALID) {
    entry->old_value = old_value;
    entry->state = LOOM_REFINE_BOUNDARIES_REPLACEMENT_NONE;
    entry->replacement = LOOM_VALUE_ID_INVALID;
    ++table->entry_count;
  }
  *out_entry = entry;
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
  if (!entry) {
    return iree_ok_status();
  }
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
  if (!entry) {
    return iree_ok_status();
  }
  entry->state = LOOM_REFINE_BOUNDARIES_REPLACEMENT_CONFLICT;
  entry->replacement = LOOM_VALUE_ID_INVALID;
  return iree_ok_status();
}

static bool loom_refine_boundaries_replacement_tables_equal(
    const loom_refine_boundaries_replacement_table_t* lhs,
    const loom_refine_boundaries_replacement_table_t* rhs) {
  for (iree_host_size_t i = 0; i < lhs->entry_capacity; ++i) {
    const loom_refine_boundaries_replacement_entry_t* lhs_entry =
        &lhs->entries[i];
    if (lhs_entry->old_value == LOOM_VALUE_ID_INVALID ||
        lhs_entry->state != LOOM_REFINE_BOUNDARIES_REPLACEMENT_VALUE) {
      continue;
    }
    loom_value_id_t rhs_replacement = LOOM_VALUE_ID_INVALID;
    if (!loom_refine_boundaries_replacement_table_lookup(
            rhs, lhs_entry->old_value, &rhs_replacement)) {
      return false;
    }
    if (lhs_entry->replacement != rhs_replacement) {
      return false;
    }
  }
  for (iree_host_size_t i = 0; i < rhs->entry_capacity; ++i) {
    const loom_refine_boundaries_replacement_entry_t* rhs_entry =
        &rhs->entries[i];
    if (rhs_entry->old_value == LOOM_VALUE_ID_INVALID ||
        rhs_entry->state != LOOM_REFINE_BOUNDARIES_REPLACEMENT_VALUE) {
      continue;
    }
    loom_value_id_t lhs_replacement = LOOM_VALUE_ID_INVALID;
    if (!loom_refine_boundaries_replacement_table_lookup(
            lhs, rhs_entry->old_value, &lhs_replacement)) {
      return false;
    }
    if (rhs_entry->replacement != lhs_replacement) {
      return false;
    }
  }
  return true;
}

typedef struct loom_refine_boundaries_boundary_state_t {
  // Fixed-point facts known for function boundary values in one round.
  loom_value_fact_table_t facts;

  // Fixed-point value replacements known for boundary values in one round.
  loom_refine_boundaries_replacement_table_t replacements;
} loom_refine_boundaries_boundary_state_t;

static iree_status_t loom_refine_boundaries_boundary_state_initialize(
    loom_refine_boundaries_boundary_state_t* state,
    iree_arena_allocator_t* arena, iree_host_size_t fact_value_capacity) {
  memset(state, 0, sizeof(*state));
  IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize(&state->facts, arena,
                                                        fact_value_capacity));
  loom_type_registry_configure_fact_context(&state->facts.context);
  return loom_refine_boundaries_replacement_table_initialize(
      &state->replacements, arena);
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

  // Joined return facts for each result slot in the current round. Extension
  // IDs are owned by the next boundary fact table for that round.
  loom_value_facts_t* return_facts;

  // Per-result bit saying whether return_facts has an observed return value.
  bool* return_fact_defined;

  // Forwarded argument index for each result slot, or a negative sentinel.
  int32_t* return_forward_argument_indices;

  // Forwarded earlier result index for each result slot, or a negative
  // sentinel.
  int32_t* return_forward_result_indices;
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

static bool loom_refine_boundaries_call_kind_participates(
    loom_call_like_kind_t kind) {
  switch (kind) {
    case LOOM_CALL_LIKE_KIND_SEMANTIC:
    case LOOM_CALL_LIKE_KIND_TEMPLATE:
      return true;
    case LOOM_CALL_LIKE_KIND_LOW_INTERNAL:
    case LOOM_CALL_LIKE_KIND_LOW_INVOKE:
    case LOOM_CALL_LIKE_KIND_NONE:
    default:
      return false;
  }
}

static bool loom_refine_boundaries_read_call(const loom_module_t* module,
                                             loom_op_t* op,
                                             loom_call_like_t* out_call,
                                             loom_symbol_ref_t* out_callee,
                                             loom_value_slice_t* out_operands,
                                             loom_value_slice_t* out_results) {
  loom_call_like_t call = loom_call_like_cast(module, op);
  if (!loom_call_like_isa(call) ||
      !loom_refine_boundaries_call_kind_participates(
          loom_call_like_kind(call))) {
    if (out_call) *out_call = (loom_call_like_t){0};
    *out_callee = loom_symbol_ref_null();
    *out_operands = (loom_value_slice_t){0};
    *out_results = (loom_value_slice_t){0};
    return false;
  }
  if (out_call) *out_call = call;
  *out_callee = loom_call_like_callee(call);
  *out_operands = loom_call_like_operands(call);
  *out_results = loom_call_like_results(call);
  return true;
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
  loom_scc_successor_callback_t visitor;
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
  if (!loom_refine_boundaries_read_call(walk->graph->module, op, NULL, &callee,
                                        &operands, &results)) {
    return iree_ok_status();
  }

  iree_host_size_t callee_node = IREE_HOST_SIZE_MAX;
  if (!loom_refine_boundaries_callee_node(walk->graph, callee, &callee_node)) {
    return iree_ok_status();
  }
  return walk->visitor.fn(walk->visitor.user_data, callee_node);
}

static iree_status_t loom_refine_boundaries_visit_successors(
    void* user_data, iree_host_size_t node,
    loom_scc_successor_callback_t visitor) {
  const loom_refine_boundaries_graph_t* graph =
      (const loom_refine_boundaries_graph_t*)user_data;
  if (node >= graph->function_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function graph node %" PRIhsz
                            " out of range for %" PRIhsz " functions",
                            node, graph->function_count);
  }

  loom_refine_boundaries_successor_walk_t walk = {
      .graph = graph,
      .visitor = visitor,
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
    if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE) ||
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
    if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE) ||
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
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, info->result_count,
          sizeof(*info->return_forward_result_indices),
          (void**)&info->return_forward_result_indices));
      memset(info->return_fact_defined, 0,
             info->result_count * sizeof(*info->return_fact_defined));
      for (uint16_t i = 0; i < info->result_count; ++i) {
        info->return_forward_argument_indices[i] =
            LOOM_REFINE_BOUNDARIES_FORWARD_UNSEEN;
        info->return_forward_result_indices[i] =
            LOOM_REFINE_BOUNDARIES_FORWARD_UNSEEN;
      }
    }
    out_graph->symbol_to_node[symbol_id] = node;
    ++node;
  }

  loom_scc_graph_t scc_graph = {
      .node_count = function_count,
      .visit_successors = loom_scc_visit_successors_callback_make(
          loom_refine_boundaries_visit_successors, out_graph),
  };
  return loom_scc_compute(&scc_graph, NULL, arena, out_sccs);
}

//===----------------------------------------------------------------------===//
// Boundary type refinement
//===----------------------------------------------------------------------===//

static iree_status_t loom_refine_boundaries_refine_value_type_with_facts(
    loom_pass_t* pass, loom_module_t* module, const loom_op_t* owner_op,
    const loom_value_fact_table_t* facts, loom_value_id_t value_id,
    int64_t* changed_count) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return iree_ok_status();
  }
  loom_type_t current_type = loom_module_value_type(module, value_id);
  loom_type_t refined_type = current_type;
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_UNCHANGED;
  IREE_RETURN_IF_ERROR(loom_type_refine_with_value_facts(
      current_type, facts, &module->arena, &refined_type, &result));
  if (result == LOOM_TYPE_REFINEMENT_CONFLICT) {
    return loom_refine_boundaries_fail(
        pass, module, owner_op,
        IREE_SV("boundary facts conflict with a function boundary value type"),
        IREE_STATUS_FAILED_PRECONDITION);
  }
  if (result == LOOM_TYPE_REFINEMENT_UNCHANGED ||
      loom_type_equal(current_type, refined_type)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_module_set_value_type(module, value_id, refined_type));
  *changed_count += 1;
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_refine_function_signature(
    loom_pass_t* pass, loom_module_t* module,
    const loom_value_fact_table_t* facts,
    const loom_refine_boundaries_function_t* function_info,
    int64_t* changed_count) {
  if (!function_info->is_internal) return iree_ok_status();
  for (uint16_t i = 0; i < function_info->argument_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_refine_boundaries_refine_value_type_with_facts(
        pass, module, function_info->function.op, facts,
        function_info->argument_ids[i], changed_count));
  }
  const loom_value_id_t* results =
      loom_op_const_results(function_info->function.op);
  for (uint16_t i = 0; i < function_info->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_refine_boundaries_refine_value_type_with_facts(
        pass, module, function_info->function.op, facts, results[i],
        changed_count));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Fact summaries
//===----------------------------------------------------------------------===//

static loom_value_facts_t loom_refine_boundaries_scalar_fact(
    loom_value_facts_t facts) {
  facts.extension_id = LOOM_VALUE_FACT_EXTENSION_ID_NONE;
  return facts;
}

static bool loom_refine_boundaries_table_has_entry(
    const loom_value_fact_table_t* table, loom_value_id_t value_id) {
  return value_id < table->count && table->entries[value_id].known_divisor != 0;
}

static iree_status_t loom_refine_boundaries_join_facts(
    loom_value_fact_table_t* target_table,
    const loom_value_fact_table_t* existing_table,
    loom_value_facts_t existing_facts,
    const loom_value_fact_table_t* incoming_table,
    loom_value_facts_t incoming_facts, loom_value_facts_t* out_joined_facts) {
  if (loom_value_fact_table_facts_equal(existing_table, existing_facts,
                                        incoming_table, incoming_facts)) {
    if (existing_table == target_table) {
      *out_joined_facts = existing_facts;
      return iree_ok_status();
    }
    return loom_value_fact_table_clone_fact(target_table, existing_table,
                                            existing_facts, out_joined_facts);
  }

  loom_value_facts_t existing_scalar =
      loom_refine_boundaries_scalar_fact(existing_facts);
  loom_value_facts_t incoming_scalar =
      loom_refine_boundaries_scalar_fact(incoming_facts);
  if (loom_value_facts_is_float(existing_scalar) ||
      loom_value_facts_is_float(incoming_scalar)) {
    *out_joined_facts = loom_value_facts_unknown();
    return iree_ok_status();
  }

  loom_value_facts_meet(&existing_scalar, &incoming_scalar, out_joined_facts);
  if (loom_value_fact_table_extensions_equal(existing_table, existing_facts,
                                             incoming_table, incoming_facts)) {
    if (existing_table == target_table) {
      out_joined_facts->extension_id = existing_facts.extension_id;
    } else {
      loom_value_facts_t cloned_existing = loom_value_facts_unknown();
      IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
          target_table, existing_table, existing_facts, &cloned_existing));
      out_joined_facts->extension_id = cloned_existing.extension_id;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_merge_fact(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    const loom_value_fact_table_t* source_table, loom_value_facts_t facts) {
  if (loom_value_facts_is_unknown(facts) &&
      !loom_refine_boundaries_table_has_entry(table, value_id)) {
    return iree_ok_status();
  }
  if (!loom_refine_boundaries_table_has_entry(table, value_id)) {
    loom_value_facts_t cloned_facts = loom_value_facts_unknown();
    IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
        table, source_table, facts, &cloned_facts));
    return loom_value_fact_table_define(table, value_id, cloned_facts);
  }
  loom_value_facts_t existing = loom_value_fact_table_lookup(table, value_id);
  loom_value_facts_t joined = loom_value_facts_unknown();
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_join_facts(
      table, table, existing, source_table, facts, &joined));
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
    if (!loom_value_fact_table_facts_equal(lhs, lhs_facts, rhs, rhs_facts)) {
      return false;
    }
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

static bool loom_refine_boundaries_values_have_equal_types(
    const loom_module_t* module, loom_value_id_t lhs, loom_value_id_t rhs) {
  if (lhs == LOOM_VALUE_ID_INVALID || rhs == LOOM_VALUE_ID_INVALID ||
      lhs >= module->values.count || rhs >= module->values.count) {
    return false;
  }
  return loom_type_equal(loom_module_value_type(module, lhs),
                         loom_module_value_type(module, rhs));
}

static void loom_refine_boundaries_join_forward_index(
    int32_t* forward_indices, iree_host_size_t result_index,
    int32_t forward_index) {
  int32_t* slot = &forward_indices[result_index];
  if (*slot == LOOM_REFINE_BOUNDARIES_FORWARD_UNSEEN) {
    *slot = forward_index;
    return;
  }
  if (*slot != forward_index) {
    *slot = LOOM_REFINE_BOUNDARIES_FORWARD_NONE;
  }
}

static int32_t loom_refine_boundaries_find_prior_result_index(
    const loom_module_t* module,
    const loom_refine_boundaries_function_t* function,
    loom_value_slice_t operands, iree_host_size_t result_index) {
  if (result_index >= operands.count)
    return LOOM_REFINE_BOUNDARIES_FORWARD_NONE;
  const loom_value_id_t* function_results =
      loom_op_const_results(function->function.op);
  for (iree_host_size_t i = 0; i < result_index; ++i) {
    if (operands.values[i] != operands.values[result_index]) continue;
    if (!loom_refine_boundaries_values_have_equal_types(
            module, function_results[i], function_results[result_index])) {
      continue;
    }
    return (int32_t)i;
  }
  return LOOM_REFINE_BOUNDARIES_FORWARD_NONE;
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
    loom_refine_boundaries_join_forward_index(
        function->return_forward_argument_indices, i, argument_index);

    int32_t result_index = loom_refine_boundaries_find_prior_result_index(
        collect->graph->module, function, operands, i);
    loom_refine_boundaries_join_forward_index(
        function->return_forward_result_indices, i, result_index);

    loom_value_facts_t facts = loom_value_fact_table_lookup(
        collect->function_facts, operands.values[i]);
    if (!function->return_fact_defined[i]) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
          collect->next_boundary_facts, collect->function_facts, facts,
          &function->return_facts[i]));
      function->return_fact_defined[i] = true;
    } else {
      loom_value_facts_t joined_facts = loom_value_facts_unknown();
      IREE_RETURN_IF_ERROR(loom_refine_boundaries_join_facts(
          collect->next_boundary_facts, collect->next_boundary_facts,
          function->return_facts[i], collect->function_facts, facts,
          &joined_facts));
      function->return_facts[i] = joined_facts;
    }
  }
  for (iree_host_size_t i = count; i < function->result_count; ++i) {
    loom_refine_boundaries_join_forward_index(
        function->return_forward_argument_indices, i,
        LOOM_REFINE_BOUNDARIES_FORWARD_NONE);
    loom_refine_boundaries_join_forward_index(
        function->return_forward_result_indices, i,
        LOOM_REFINE_BOUNDARIES_FORWARD_NONE);
  }
  function->has_return_facts = true;
  return iree_ok_status();
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

static bool loom_refine_boundaries_call_result_is_tied(const loom_op_t* call_op,
                                                       iree_host_size_t index) {
  const loom_tied_result_t* tied_results = loom_op_tied_results(call_op);
  for (uint16_t i = 0; i < call_op->tied_result_count; ++i) {
    if (tied_results[i].result_index == index) return true;
  }
  return false;
}

static iree_status_t loom_refine_boundaries_collect_return_forwarding(
    loom_refine_boundaries_collect_t* collect, const loom_op_t* call_op,
    const loom_refine_boundaries_function_t* callee_info,
    loom_value_slice_t operands, loom_value_slice_t results) {
  if (!callee_info->return_forward_argument_indices) return iree_ok_status();
  iree_host_size_t count = results.count < callee_info->result_count
                               ? results.count
                               : callee_info->result_count;
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (loom_refine_boundaries_call_result_is_tied(call_op, i)) continue;

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
  if (!callee_info->return_forward_result_indices) return iree_ok_status();
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (callee_info->return_forward_argument_indices[i] >= 0) continue;
    if (loom_refine_boundaries_call_result_is_tied(call_op, i)) continue;

    int32_t result_index = callee_info->return_forward_result_indices[i];
    if (result_index < 0 || (iree_host_size_t)result_index >= results.count) {
      continue;
    }
    loom_value_id_t result = results.values[i];
    loom_value_id_t replacement = results.values[result_index];
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
    loom_refine_boundaries_collect_t* collect, loom_op_t* op) {
  loom_symbol_ref_t callee = loom_symbol_ref_null();
  loom_value_slice_t operands = {0};
  loom_value_slice_t results = {0};
  if (!loom_refine_boundaries_read_call(collect->graph->module, op, NULL,
                                        &callee, &operands, &results)) {
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
          collect->next_boundary_facts, callee_info->argument_ids[i],
          collect->function_facts, facts));
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
        collect->next_boundary_facts, callee_info->return_facts[i]));
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

static iree_status_t loom_refine_boundaries_collect_function(
    loom_refine_boundaries_graph_t* graph,
    const loom_value_fact_table_t* function_facts,
    loom_value_fact_table_t* next_boundary_facts,
    loom_refine_boundaries_replacement_table_t* next_boundary_replacements,
    loom_refine_boundaries_function_t* function_info) {
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
// Function refinement
//===----------------------------------------------------------------------===//

static iree_status_t loom_refine_boundaries_run_function(
    loom_pass_t* pass, loom_canonicalizer_t* canonicalizer,
    loom_refine_boundaries_graph_t* graph, loom_value_fact_table_t* seed_facts,
    const loom_refine_boundaries_replacement_table_t* seed_replacements,
    loom_value_fact_table_t* next_boundary_facts,
    loom_refine_boundaries_replacement_table_t* next_boundary_replacements,
    loom_refine_boundaries_function_t* function_info,
    int64_t* signature_type_changed_count) {
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
    loom_pass_statistic_add(
        pass, LOOM_REFINE_BOUNDARIES_STAT_FUNCTIONS_CANONICALIZED, 1);
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
  if (replacements_applied > 0 || constants_materialized > 0 ||
      canonicalize_result.changed) {
    loom_pass_mark_changed(pass);
  }

  const loom_value_fact_table_t* function_facts =
      loom_canonicalizer_fact_table(canonicalizer);
  if (!function_facts) return iree_ok_status();
  int64_t signature_type_changes_before = *signature_type_changed_count;
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_refine_function_signature(
      pass, graph->module, function_facts, function_info,
      signature_type_changed_count));
  if (signature_type_changes_before != *signature_type_changed_count) {
    loom_pass_mark_changed(pass);
  }

  return loom_refine_boundaries_collect_function(
      graph, function_facts, next_boundary_facts, next_boundary_replacements,
      function_info);
}

//===----------------------------------------------------------------------===//
// Boundary call result type refinement
//===----------------------------------------------------------------------===//

static iree_status_t loom_refine_boundaries_substitute_call_result_type(
    loom_module_t* module, const loom_refine_boundaries_function_t* callee_info,
    loom_value_slice_t operands, loom_value_slice_t results,
    loom_type_t callee_type, loom_type_t* out_type) {
  *out_type = callee_type;
  if (operands.count != callee_info->argument_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "direct call operand count %u does not match callee argument count %u "
        "during boundary type refinement",
        (unsigned)operands.count, (unsigned)callee_info->argument_count);
  }
  if (results.count != callee_info->result_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "direct call result count %u does not match callee result count %u "
        "during boundary type refinement",
        (unsigned)results.count, (unsigned)callee_info->result_count);
  }

  for (uint16_t i = 0; i < callee_info->argument_count; ++i) {
    bool changed = false;
    IREE_RETURN_IF_ERROR(loom_module_replace_type_value_references(
        module, *out_type, callee_info->argument_ids[i], operands.values[i],
        out_type, &changed));
  }

  const loom_value_id_t* callee_results =
      loom_op_const_results(callee_info->function.op);
  for (uint16_t i = 0; i < callee_info->result_count; ++i) {
    bool changed = false;
    IREE_RETURN_IF_ERROR(loom_module_replace_type_value_references(
        module, *out_type, callee_results[i], results.values[i], out_type,
        &changed));
  }
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_refine_call_result_type(
    loom_pass_t* pass, loom_module_t* module, const loom_op_t* call_op,
    const loom_refine_boundaries_function_t* callee_info,
    loom_value_slice_t operands, loom_value_slice_t results,
    iree_host_size_t result_index, int64_t* changed_count) {
  if (result_index >= results.count ||
      result_index >= callee_info->result_count) {
    return iree_ok_status();
  }

  const loom_value_id_t* callee_results =
      loom_op_const_results(callee_info->function.op);
  loom_type_t candidate_type =
      loom_module_value_type(module, callee_results[result_index]);
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_substitute_call_result_type(
      module, callee_info, operands, results, candidate_type, &candidate_type));

  loom_value_id_t result_value = results.values[result_index];
  loom_type_t current_type = loom_module_value_type(module, result_value);
  loom_type_t refined_type = current_type;
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_UNCHANGED;
  IREE_RETURN_IF_ERROR(loom_type_refine_with_candidate(
      current_type, candidate_type, &module->arena, &refined_type, &result));
  if (result == LOOM_TYPE_REFINEMENT_CONFLICT) {
    return loom_refine_boundaries_fail(
        pass, module, call_op,
        IREE_SV("callee result type conflicts with the call result type after "
                "boundary substitution"),
        IREE_STATUS_FAILED_PRECONDITION);
  }
  if (result == LOOM_TYPE_REFINEMENT_UNCHANGED ||
      loom_type_equal(current_type, refined_type)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_module_set_value_type(module, result_value, refined_type));
  *changed_count += 1;
  return iree_ok_status();
}

typedef struct loom_refine_boundaries_signature_call_walk_t {
  // Current pass instance used for structured diagnostics.
  loom_pass_t* pass;

  // Module being rewritten.
  loom_module_t* module;

  // Function graph for resolving direct callees.
  const loom_refine_boundaries_graph_t* graph;

  // Number of call result type changes applied while walking.
  int64_t* changed_count;
} loom_refine_boundaries_signature_call_walk_t;

static iree_status_t loom_refine_boundaries_refine_call_result_types(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;

  loom_refine_boundaries_signature_call_walk_t* walk =
      (loom_refine_boundaries_signature_call_walk_t*)user_data;
  loom_symbol_ref_t callee = loom_symbol_ref_null();
  loom_value_slice_t operands = {0};
  loom_value_slice_t results = {0};
  if (!loom_refine_boundaries_read_call(walk->module, op, NULL, &callee,
                                        &operands, &results)) {
    return iree_ok_status();
  }

  iree_host_size_t callee_node = IREE_HOST_SIZE_MAX;
  if (!loom_refine_boundaries_callee_node(walk->graph, callee, &callee_node)) {
    return iree_ok_status();
  }
  const loom_refine_boundaries_function_t* callee_info =
      &walk->graph->functions[callee_node];
  if (!callee_info->is_internal) return iree_ok_status();

  for (iree_host_size_t i = 0; i < results.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_refine_boundaries_refine_call_result_type(
        walk->pass, walk->module, op, callee_info, operands, results, i,
        walk->changed_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_refine_internal_signature_types(
    loom_pass_t* pass, loom_module_t* module,
    const loom_refine_boundaries_graph_t* graph,
    const loom_value_fact_table_t* boundary_facts,
    iree_arena_allocator_t* walk_arena, int64_t* out_changed_count) {
  *out_changed_count = 0;
  for (iree_host_size_t node = 0; node < graph->function_count; ++node) {
    IREE_RETURN_IF_ERROR(loom_refine_boundaries_refine_function_signature(
        pass, module, boundary_facts, &graph->functions[node],
        out_changed_count));
  }

  loom_refine_boundaries_signature_call_walk_t walk = {
      .pass = pass,
      .module = module,
      .graph = graph,
      .changed_count = out_changed_count,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  iree_arena_reset(walk_arena);
  return loom_walk_region(
      module, module->body, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_refine_boundaries_refine_call_result_types,
                             &walk},
      walk_arena, &walk_result);
}

//===----------------------------------------------------------------------===//
// Boundary specialization
//===----------------------------------------------------------------------===//

typedef struct loom_refine_boundaries_specialization_group_t {
  // Static result type tuple used as the specialization key.
  loom_type_t* result_types;

  // Calls whose current result types match |result_types|.
  loom_op_t** calls;

  // Number of entries in |calls|.
  iree_host_size_t call_count;

  // Allocated call pointer count.
  iree_host_size_t call_capacity;

  // Symbol created for this specialization.
  loom_symbol_ref_t symbol_ref;
} loom_refine_boundaries_specialization_group_t;

typedef struct loom_refine_boundaries_specialization_plan_t {
  // Specialization groups for one private function.
  loom_refine_boundaries_specialization_group_t* groups;

  // Number of entries in |groups|.
  iree_host_size_t group_count;

  // Allocated specialization group count.
  iree_host_size_t group_capacity;
} loom_refine_boundaries_specialization_plan_t;

typedef struct loom_refine_boundaries_specialization_call_walk_t {
  // Module being inspected.
  loom_module_t* module;

  // Function graph for resolving direct callees.
  const loom_refine_boundaries_graph_t* graph;

  // Dense specialization plans indexed by function node.
  loom_refine_boundaries_specialization_plan_t* plans;

  // Scratch arena for specialization plans and call lists.
  iree_arena_allocator_t* arena;
} loom_refine_boundaries_specialization_call_walk_t;

static bool loom_refine_boundaries_can_specialize_function(
    const loom_refine_boundaries_function_t* function_info) {
  return function_info->is_internal &&
         loom_func_def_isa(function_info->function.op) &&
         function_info->result_count > 0;
}

static bool loom_refine_boundaries_result_types_are_static(
    const loom_module_t* module, loom_value_slice_t results,
    uint16_t expected_result_count) {
  if (results.count != expected_result_count) return false;
  for (uint16_t i = 0; i < expected_result_count; ++i) {
    loom_value_id_t result = results.values[i];
    if (result == LOOM_VALUE_ID_INVALID || result >= module->values.count) {
      return false;
    }
    if (!loom_type_is_all_static(loom_module_value_type(module, result))) {
      return false;
    }
  }
  return true;
}

static bool loom_refine_boundaries_group_matches_call_results(
    const loom_module_t* module,
    const loom_refine_boundaries_specialization_group_t* group,
    loom_value_slice_t results, uint16_t result_count) {
  for (uint16_t i = 0; i < result_count; ++i) {
    if (!loom_type_equal(group->result_types[i],
                         loom_module_value_type(module, results.values[i]))) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_refine_boundaries_append_specialization_call(
    loom_refine_boundaries_specialization_group_t* group, loom_op_t* call_op,
    iree_arena_allocator_t* arena) {
  if (group->call_count >= group->call_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, group->call_count, group->call_count + 1, sizeof(*group->calls),
        &group->call_capacity, (void**)&group->calls));
  }
  group->calls[group->call_count++] = call_op;
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_add_specialization_group(
    const loom_module_t* module,
    loom_refine_boundaries_specialization_plan_t* plan,
    loom_value_slice_t results, uint16_t result_count,
    iree_arena_allocator_t* arena,
    loom_refine_boundaries_specialization_group_t** out_group) {
  *out_group = NULL;
  if (plan->group_count >= plan->group_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, plan->group_count, plan->group_count + 1, sizeof(*plan->groups),
        &plan->group_capacity, (void**)&plan->groups));
  }

  loom_refine_boundaries_specialization_group_t* group =
      &plan->groups[plan->group_count++];
  memset(group, 0, sizeof(*group));
  group->symbol_ref = loom_symbol_ref_null();
  if (result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, result_count, sizeof(*group->result_types),
        (void**)&group->result_types));
    for (uint16_t i = 0; i < result_count; ++i) {
      group->result_types[i] =
          loom_module_value_type(module, results.values[i]);
    }
  }

  *out_group = group;
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_record_specialization_call(
    const loom_module_t* module,
    const loom_refine_boundaries_function_t* callee_info,
    loom_refine_boundaries_specialization_plan_t* plan, loom_op_t* call_op,
    loom_value_slice_t results, iree_arena_allocator_t* arena) {
  if (!loom_refine_boundaries_result_types_are_static(
          module, results, callee_info->result_count)) {
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < plan->group_count; ++i) {
    loom_refine_boundaries_specialization_group_t* group = &plan->groups[i];
    if (!loom_refine_boundaries_group_matches_call_results(
            module, group, results, callee_info->result_count)) {
      continue;
    }
    return loom_refine_boundaries_append_specialization_call(group, call_op,
                                                             arena);
  }

  loom_refine_boundaries_specialization_group_t* group = NULL;
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_add_specialization_group(
      module, plan, results, callee_info->result_count, arena, &group));
  return loom_refine_boundaries_append_specialization_call(group, call_op,
                                                           arena);
}

static iree_status_t loom_refine_boundaries_collect_specialization_call(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;

  loom_refine_boundaries_specialization_call_walk_t* walk =
      (loom_refine_boundaries_specialization_call_walk_t*)user_data;
  loom_symbol_ref_t callee = loom_symbol_ref_null();
  loom_value_slice_t operands = {0};
  loom_value_slice_t results = {0};
  if (!loom_refine_boundaries_read_call(walk->module, op, NULL, &callee,
                                        &operands, &results)) {
    return iree_ok_status();
  }
  (void)operands;

  iree_host_size_t callee_node = IREE_HOST_SIZE_MAX;
  if (!loom_refine_boundaries_callee_node(walk->graph, callee, &callee_node)) {
    return iree_ok_status();
  }
  const loom_refine_boundaries_function_t* callee_info =
      &walk->graph->functions[callee_node];
  if (!loom_refine_boundaries_can_specialize_function(callee_info)) {
    return iree_ok_status();
  }

  return loom_refine_boundaries_record_specialization_call(
      walk->module, callee_info, &walk->plans[callee_node], op, results,
      walk->arena);
}

static iree_status_t loom_refine_boundaries_collect_specialization_plans(
    loom_module_t* module, const loom_refine_boundaries_graph_t* graph,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* walk_arena,
    loom_refine_boundaries_specialization_plan_t** out_plans) {
  *out_plans = NULL;
  if (graph->function_count == 0) return iree_ok_status();

  loom_refine_boundaries_specialization_plan_t* plans = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, graph->function_count, sizeof(*plans), (void**)&plans));
  memset(plans, 0, graph->function_count * sizeof(*plans));

  loom_refine_boundaries_specialization_call_walk_t walk = {
      .module = module,
      .graph = graph,
      .plans = plans,
      .arena = arena,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  iree_arena_reset(walk_arena);
  IREE_RETURN_IF_ERROR(loom_walk_region(
      module, module->body, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_refine_boundaries_collect_specialization_call,
                             &walk},
      walk_arena, &walk_result));

  *out_plans = plans;
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_make_specialization_symbol(
    loom_module_t* module, loom_symbol_ref_t source_ref,
    iree_host_size_t preferred_ordinal, iree_arena_allocator_t* arena,
    loom_symbol_ref_t* out_symbol_ref) {
  *out_symbol_ref = loom_symbol_ref_null();
  if (!loom_symbol_ref_is_valid(source_ref) || source_ref.module_id != 0 ||
      source_ref.symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source specialization symbol is invalid");
  }

  const loom_symbol_t* source_symbol =
      &module->symbols.entries[source_ref.symbol_id];
  if (source_symbol->name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source specialization symbol has no name");
  }
  iree_string_view_t source_name =
      module->strings.entries[source_symbol->name_id];

  for (iree_host_size_t ordinal = preferred_ordinal;
       ordinal < IREE_HOST_SIZE_MAX; ++ordinal) {
    char suffix[32] = {0};
    int suffix_length =
        snprintf(suffix, sizeof(suffix), "_spec%" PRIhsz, ordinal);
    if (suffix_length < 0 ||
        (iree_host_size_t)suffix_length >= sizeof(suffix)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "specialization symbol suffix overflow");
    }

    iree_host_size_t name_length = 0;
    if (!iree_host_size_checked_add(
            source_name.size, (iree_host_size_t)suffix_length, &name_length)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "specialization symbol name overflow");
    }

    char* name_storage = NULL;
    if (name_length > 0) {
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate(arena, name_length, (void**)&name_storage));
      memcpy(name_storage, source_name.data, source_name.size);
      memcpy(name_storage + source_name.size, suffix,
             (iree_host_size_t)suffix_length);
    }
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        module, iree_make_string_view(name_storage, name_length), &name_id));
    if (loom_module_find_symbol(module, name_id) != LOOM_SYMBOL_ID_INVALID) {
      continue;
    }

    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_add_symbol(module, name_id, &symbol_id));
    *out_symbol_ref =
        (loom_symbol_ref_t){.module_id = 0, .symbol_id = symbol_id};
    return iree_ok_status();
  }

  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "could not create a unique specialization symbol");
}

static iree_status_t loom_refine_boundaries_clone_function_specialization(
    loom_module_t* module,
    const loom_refine_boundaries_function_t* function_info,
    const loom_type_t* result_types, loom_symbol_ref_t target_ref,
    const loom_op_t* insertion_anchor, iree_arena_allocator_t* arena,
    loom_op_t** out_op) {
  *out_op = NULL;
  loom_op_t* source_op = function_info->function.op;
  loom_region_t* source_body = loom_func_like_body(function_info->function);
  if (!source_body) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "specialization source has no body");
  }

  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(module, module, arena,
                                                /*options=*/NULL, &remap));

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, source_op->parent_block,
                          &builder);
  loom_builder_set_after(&builder, insertion_anchor);

  loom_location_id_t target_location = LOOM_LOCATION_UNKNOWN;
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_location_id(&remap, source_op->location, &target_location));

  loom_op_t* target_op = NULL;
  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
      &builder, LOOM_OP_FUNC_DEF, 0, source_op->result_count, 1,
      source_op->tied_result_count, source_op->attribute_count, target_location,
      &target_op));
  target_op->instance_flags = source_op->instance_flags;
  target_op->traits = source_op->traits;

  loom_attribute_t* target_attrs = loom_op_attrs(target_op);
  const loom_attribute_t* source_attrs = loom_op_const_attrs(source_op);
  for (uint8_t i = 0; i < source_op->attribute_count; ++i) {
    target_attrs[i] = source_attrs[i];
  }
  target_attrs[0] = loom_attr_symbol(target_ref);
  if (function_info->function.vtable->predicates_attr_index !=
      LOOM_ATTR_INDEX_NONE) {
    target_attrs[function_info->function.vtable->predicates_attr_index] =
        (loom_attribute_t){0};
  }

  const loom_value_id_t* source_results = loom_op_const_results(source_op);
  loom_value_id_t* target_results = loom_op_results(target_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_module_define_value(module, loom_type_none(), &target_results[i]));
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_map_value(&remap, source_results[i], target_results[i]));
    loom_module_value(module, target_results[i])->name_id =
        loom_module_value(module, source_results[i])->name_id;
  }

  if (source_op->tied_result_count > 0) {
    memcpy(loom_op_tied_results(target_op), loom_op_tied_results(source_op),
           (iree_host_size_t)source_op->tied_result_count *
               sizeof(loom_tied_result_t));
  }

  loom_builder_ip_t saved_ip = loom_builder_save(&builder);
  builder.ip.parent_op = target_op;
  loom_region_t* target_body = NULL;
  iree_status_t status =
      loom_ir_clone_region(&builder, source_body, &remap, &target_body);
  loom_builder_restore(&builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);
  loom_op_regions(target_op)[0] = target_body;

  uint16_t predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(function_info->function, &predicate_count);
  if (predicate_count > 0) {
    loom_predicate_t* target_predicates = NULL;
    IREE_RETURN_IF_ERROR(loom_ir_remap_predicate_list(
        &remap, predicates, predicate_count, &target_predicates));
    target_attrs[function_info->function.vtable->predicates_attr_index] =
        loom_attr_predicate_list(target_predicates, predicate_count);
  }

  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_module_set_value_type(module, target_results[i], result_types[i]));
  }

  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(&builder, target_op));
  *out_op = target_op;
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_create_specialization_group(
    loom_module_t* module,
    const loom_refine_boundaries_function_t* function_info,
    loom_refine_boundaries_specialization_group_t* group,
    iree_host_size_t group_ordinal, const loom_op_t* insertion_anchor,
    iree_arena_allocator_t* arena, loom_op_t** out_op) {
  *out_op = NULL;
  loom_symbol_ref_t source_ref = loom_func_like_callee(function_info->function);
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_make_specialization_symbol(
      module, source_ref, group_ordinal, arena, &group->symbol_ref));

  return loom_refine_boundaries_clone_function_specialization(
      module, function_info, group->result_types, group->symbol_ref,
      insertion_anchor, arena, out_op);
}

static void loom_refine_boundaries_retarget_call(loom_module_t* module,
                                                 loom_op_t* call_op,
                                                 loom_symbol_ref_t callee) {
  loom_trait_flags_t old_traits = call_op->traits;
  loom_op_attrs(call_op)[0] = loom_attr_symbol(callee);
  loom_op_refresh_effective_traits(module, call_op);
  loom_module_update_op_direct_effects(call_op, old_traits, call_op->traits);
}

static iree_status_t loom_refine_boundaries_create_specializations(
    loom_module_t* module, const loom_refine_boundaries_graph_t* graph,
    loom_refine_boundaries_specialization_plan_t* plans,
    iree_arena_allocator_t* arena, int64_t* out_specialization_count) {
  *out_specialization_count = 0;
  for (iree_host_size_t node = 0; node < graph->function_count; ++node) {
    loom_refine_boundaries_specialization_plan_t* plan = &plans[node];
    if (plan->group_count < 2) continue;

    const loom_refine_boundaries_function_t* function_info =
        &graph->functions[node];
    loom_op_t* insertion_anchor = function_info->function.op;
    for (iree_host_size_t i = 0; i < plan->group_count; ++i) {
      loom_refine_boundaries_specialization_group_t* group = &plan->groups[i];
      loom_op_t* specialization_op = NULL;
      IREE_RETURN_IF_ERROR(loom_refine_boundaries_create_specialization_group(
          module, function_info, group, i, insertion_anchor, arena,
          &specialization_op));
      insertion_anchor = specialization_op;
      *out_specialization_count += 1;
    }
    for (iree_host_size_t i = 0; i < plan->group_count; ++i) {
      loom_refine_boundaries_specialization_group_t* group = &plan->groups[i];
      for (iree_host_size_t call_index = 0; call_index < group->call_count;
           ++call_index) {
        loom_refine_boundaries_retarget_call(module, group->calls[call_index],
                                             group->symbol_ref);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_specialize_internal_boundaries(
    loom_module_t* module, const loom_refine_boundaries_graph_t* graph,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* walk_arena,
    int64_t* out_specialization_count) {
  *out_specialization_count = 0;
  loom_refine_boundaries_specialization_plan_t* plans = NULL;
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_collect_specialization_plans(
      module, graph, arena, walk_arena, &plans));
  if (!plans) return iree_ok_status();
  return loom_refine_boundaries_create_specializations(
      module, graph, plans, arena, out_specialization_count);
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
  loom_call_like_t call = {0};
  loom_value_slice_t operands = {0};
  loom_value_slice_t results = {0};
  if (!loom_refine_boundaries_read_call(walk->graph->module, op, &call, &callee,
                                        &operands, &results)) {
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
  uint16_t operand_offset = loom_call_like_operand_offset(call);
  uint16_t result_offset = loom_call_like_result_offset(call);
  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    uint16_t operand_index = tied_results[i].operand_index;
    if (plan->has_prunable_arguments && operand_index >= operand_offset) {
      uint16_t argument_index = (uint16_t)(operand_index - operand_offset);
      if (argument_index >= plan->argument_count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "tied result operand index %u is out of range for %u argument(s)",
            (unsigned)argument_index, (unsigned)plan->argument_count);
      }
      plan->prune_arguments[argument_index] = false;
    }

    uint16_t result_index = tied_results[i].result_index;
    if (plan->has_prunable_results && result_index >= result_offset) {
      uint16_t call_result_index = (uint16_t)(result_index - result_offset);
      if (call_result_index >= plan->result_count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "tied result index %u is out of range for %u result(s)",
            (unsigned)call_result_index, (unsigned)plan->result_count);
      }
      plan->prune_results[call_result_index] = false;
    }
  }

  if (plan->has_prunable_results) {
    for (uint16_t i = 0; i < plan->result_count; ++i) {
      if (!plan->prune_results[i]) {
        continue;
      }
      if (!loom_refine_boundaries_result_is_prunable(walk->graph->module, op,
                                                     result_offset + i)) {
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
  loom_call_like_t call = {0};
  loom_value_slice_t operands = {0};
  loom_value_slice_t results = {0};
  if (!loom_refine_boundaries_read_call(module, op, &call, &callee, &operands,
                                        &results)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected direct call-like op");
  }
  (void)callee;

  if (op->successor_count != 0 || op->region_count != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "boundary pruning requires call-like ops without successors or "
        "regions");
  }
  uint16_t operand_offset = loom_call_like_operand_offset(call);
  uint16_t result_offset = loom_call_like_result_offset(call);
  if (operand_offset > op->operand_count || result_offset > op->result_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "call-like operand or result offset is out of range");
  }

  uint16_t* old_to_new_operand_indices = NULL;
  loom_value_id_t* new_operands = NULL;
  if (op->operand_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, op->operand_count, sizeof(*old_to_new_operand_indices),
        (void**)&old_to_new_operand_indices));
    for (uint16_t i = 0; i < op->operand_count; ++i) {
      old_to_new_operand_indices[i] = UINT16_MAX;
    }
  }
  uint16_t kept_call_operand_count = 0;
  for (uint16_t i = 0; i < operands.count; ++i) {
    if (i < plan->argument_count && plan->prune_arguments[i]) {
      continue;
    }
    old_to_new_operand_indices[operand_offset + i] =
        (uint16_t)(operand_offset + kept_call_operand_count);
    ++kept_call_operand_count;
  }
  uint16_t new_operand_count =
      (uint16_t)(operand_offset + kept_call_operand_count);
  if (new_operand_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, new_operand_count,
                                                   sizeof(*new_operands),
                                                   (void**)&new_operands));
    const loom_value_id_t* old_operands = loom_op_const_operands(op);
    for (uint16_t i = 0; i < operand_offset; ++i) {
      old_to_new_operand_indices[i] = i;
      new_operands[i] = old_operands[i];
    }
    uint16_t kept_index = operand_offset;
    for (uint16_t i = 0; i < operands.count; ++i) {
      uint16_t old_index = (uint16_t)(operand_offset + i);
      if (old_to_new_operand_indices[old_index] == UINT16_MAX) {
        continue;
      }
      new_operands[kept_index++] = operands.values[i];
    }
  }

  uint16_t* old_to_new_result_indices = NULL;
  if (op->result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, op->result_count, sizeof(*old_to_new_result_indices),
        (void**)&old_to_new_result_indices));
    for (uint16_t i = 0; i < op->result_count; ++i) {
      old_to_new_result_indices[i] = UINT16_MAX;
    }
  }
  uint16_t kept_call_result_count = 0;
  for (uint16_t i = 0; i < results.count; ++i) {
    if (i < plan->result_count && plan->prune_results[i]) {
      continue;
    }
    old_to_new_result_indices[result_offset + i] =
        (uint16_t)(result_offset + kept_call_result_count);
    ++kept_call_result_count;
  }
  uint16_t new_result_count =
      (uint16_t)(result_offset + kept_call_result_count);

  loom_tied_result_t* tied_results = NULL;
  if (op->tied_result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, op->tied_result_count,
                                                   sizeof(*tied_results),
                                                   (void**)&tied_results));
    const loom_tied_result_t* old_tied_results = loom_op_tied_results(op);
    for (uint16_t i = 0; i < op->tied_result_count; ++i) {
      tied_results[i] = old_tied_results[i];
      uint16_t old_operand_index = old_tied_results[i].operand_index;
      if (old_operand_index >= op->operand_count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "tied result operand index %u is out of range for %u operand(s)",
            (unsigned)old_operand_index, (unsigned)op->operand_count);
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
      if (old_result_index >= op->result_count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "tied result index %u is out of range for %u result(s)",
            (unsigned)old_result_index, (unsigned)op->result_count);
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
  if (new_result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, new_result_count, sizeof(*result_types), (void**)&result_types));
    const loom_value_id_t* old_results = loom_op_const_results(op);
    for (uint16_t i = 0; i < result_offset; ++i) {
      old_to_new_result_indices[i] = i;
      result_types[i] = loom_module_value_type(module, old_results[i]);
    }
    for (uint16_t i = 0; i < results.count; ++i) {
      uint16_t old_index = (uint16_t)(result_offset + i);
      uint16_t new_index = old_to_new_result_indices[old_index];
      if (new_index == UINT16_MAX) {
        continue;
      }
      result_types[new_index] =
          loom_module_value_type(module, results.values[i]);
    }
  }
  *out_old_to_new_result_indices = old_to_new_result_indices;

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, op->parent_block, &builder);
  loom_builder_set_before(&builder, op);
  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
      &builder, op->kind, new_operand_count, new_result_count,
      /*region_count=*/0, op->tied_result_count, op->attribute_count,
      op->location, out_new_op));
  if (new_operand_count > 0) {
    memcpy(loom_op_operands(*out_new_op), new_operands,
           (iree_host_size_t)new_operand_count * sizeof(*new_operands));
  }
  for (uint16_t i = 0; i < new_result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_builder_define_value(
        &builder, result_types[i], &loom_op_results(*out_new_op)[i]));
  }
  if (op->tied_result_count > 0) {
    memcpy(loom_op_tied_results(*out_new_op), tied_results,
           (iree_host_size_t)op->tied_result_count * sizeof(*tied_results));
  }
  if (op->attribute_count > 0) {
    memcpy(loom_op_attrs(*out_new_op), loom_op_const_attrs(op),
           (iree_host_size_t)op->attribute_count * sizeof(loom_attribute_t));
  }
  return loom_builder_finalize_op(&builder, *out_new_op);
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
  if (!loom_refine_boundaries_read_call(walk->module, op, NULL, &callee,
                                        &operands, &results)) {
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
    if (new_index == UINT16_MAX) {
      continue;
    }
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

iree_status_t loom_refine_boundaries_run_with_options(
    loom_pass_t* pass, loom_module_t* module,
    const loom_refine_boundaries_options_t* options) {
  uint32_t max_iterations = options && options->max_iterations > 0
                                ? options->max_iterations
                                : LOOM_REFINE_BOUNDARIES_DEFAULT_MAX_ITERATIONS;
  const iree_host_size_t boundary_fact_value_capacity = module->values.capacity;

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
  loom_refine_boundaries_boundary_state_t boundary_state_a = {0};
  loom_refine_boundaries_boundary_state_t boundary_state_b = {0};
  loom_refine_boundaries_boundary_state_t* current_boundary = &boundary_state_a;
  loom_refine_boundaries_boundary_state_t* next_boundary = &boundary_state_b;

  iree_status_t status = loom_refine_boundaries_boundary_state_initialize(
      current_boundary, current_facts_arena, boundary_fact_value_capacity);

  loom_canonicalizer_t canonicalizer = {0};
  bool canonicalizer_initialized = false;
  if (iree_status_is_ok(status)) {
    status = loom_canonicalizer_initialize(module, pass->arena,
                                           pass->value_facts, &canonicalizer);
    canonicalizer_initialized = iree_status_is_ok(status);
  }

  bool converged = false;
  for (uint32_t iteration = 0;
       iree_status_is_ok(status) && iteration < max_iterations; ++iteration) {
    iree_arena_reset(&iteration_arena);
    iree_arena_reset(next_facts_arena);

    loom_refine_boundaries_graph_t graph = {0};
    loom_scc_list_t sccs = {0};
    status = loom_refine_boundaries_build_graph(module, &iteration_arena,
                                                &walk_arena, &graph, &sccs);
    if (!iree_status_is_ok(status)) break;

    status = loom_refine_boundaries_boundary_state_initialize(
        next_boundary, next_facts_arena, boundary_fact_value_capacity);
    if (!iree_status_is_ok(status)) break;

    int64_t signature_type_changed_count = 0;
    for (iree_host_size_t scc_index = 0;
         iree_status_is_ok(status) && scc_index < sccs.count; ++scc_index) {
      const loom_scc_t* scc = &sccs.values[scc_index];
      for (iree_host_size_t member = 0;
           iree_status_is_ok(status) && member < scc->node_count; ++member) {
        iree_host_size_t node = scc->nodes[member];
        status = loom_refine_boundaries_run_function(
            pass, &canonicalizer, &graph, &current_boundary->facts,
            &current_boundary->replacements, &next_boundary->facts,
            &next_boundary->replacements, &graph.functions[node],
            &signature_type_changed_count);
      }
    }
    if (!iree_status_is_ok(status)) break;

    bool boundary_facts_changed = !loom_refine_boundaries_fact_tables_equal(
        &current_boundary->facts, &next_boundary->facts);
    bool boundary_replacements_changed =
        !loom_refine_boundaries_replacement_tables_equal(
            &current_boundary->replacements, &next_boundary->replacements);
    if (!boundary_facts_changed && !boundary_replacements_changed) {
      int64_t call_result_type_changed_count = 0;
      status = loom_refine_boundaries_refine_internal_signature_types(
          pass, module, &graph, &next_boundary->facts, &walk_arena,
          &call_result_type_changed_count);
      if (!iree_status_is_ok(status)) break;
      signature_type_changed_count += call_result_type_changed_count;
      if (signature_type_changed_count > 0) {
        loom_pass_mark_changed(pass);
        if (pass->statistics) {
          loom_pass_statistic_add(
              pass,
              LOOM_REFINE_BOUNDARIES_STAT_BOUNDARY_SIGNATURE_TYPES_REFINED,
              signature_type_changed_count);
        }
        continue;
      }

      int64_t specialization_count = 0;
      status = loom_refine_boundaries_specialize_internal_boundaries(
          module, &graph, &iteration_arena, &walk_arena, &specialization_count);
      if (!iree_status_is_ok(status)) break;
      if (specialization_count > 0) {
        loom_pass_mark_changed(pass);
        if (pass->statistics) {
          loom_pass_statistic_add(
              pass,
              LOOM_REFINE_BOUNDARIES_STAT_BOUNDARY_SPECIALIZATIONS_CREATED,
              specialization_count);
        }
        continue;
      }

      int64_t pruned_argument_count = 0;
      int64_t pruned_result_count = 0;
      status = loom_refine_boundaries_prune_internal_boundaries(
          module, &graph, &iteration_arena, &walk_arena, &pruned_argument_count,
          &pruned_result_count);
      if (!iree_status_is_ok(status)) break;
      if (pruned_argument_count > 0 || pruned_result_count > 0) {
        loom_pass_mark_changed(pass);
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
      if (signature_type_changed_count > 0) {
        loom_pass_statistic_add(
            pass, LOOM_REFINE_BOUNDARIES_STAT_BOUNDARY_SIGNATURE_TYPES_REFINED,
            signature_type_changed_count);
      }
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
    loom_refine_boundaries_boundary_state_t* old_current_boundary =
        current_boundary;
    current_boundary = next_boundary;
    next_boundary = old_current_boundary;
  }
  if (iree_status_is_ok(status) && !converged) {
    char reason_storage[192] = {0};
    int reason_length = snprintf(
        reason_storage, sizeof(reason_storage),
        "boundary facts did not converge in %u iteration(s); recursive/SCC "
        "summaries may require specialization or a wider pass limit",
        max_iterations);
    iree_string_view_t reason =
        reason_length > 0 &&
                (iree_host_size_t)reason_length < IREE_ARRAYSIZE(reason_storage)
            ? iree_make_string_view(reason_storage,
                                    (iree_host_size_t)reason_length)
            : IREE_SV("boundary facts did not converge");
    status = loom_refine_boundaries_fail(pass, module, NULL, reason,
                                         IREE_STATUS_RESOURCE_EXHAUSTED);
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

iree_status_t loom_refine_boundaries_run(loom_pass_t* pass,
                                         loom_module_t* module) {
  return loom_refine_boundaries_run_with_options(
      pass, module, (const loom_refine_boundaries_options_t*)pass->state);
}
