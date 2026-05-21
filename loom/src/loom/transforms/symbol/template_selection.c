// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/symbol/template_selection.h"

#include <stdint.h>
#include <string.h>

#include "loom/analysis/func_provider_catalog.h"
#include "loom/analysis/symbol_dependencies.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/analysis/symbol_liveness.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/pass/pipeline.h"
#include "loom/pass/registry.h"
#include "loom/rewrite/rewriter.h"
#include "loom/transforms/symbol/symbol_pruning.h"

//===----------------------------------------------------------------------===//
// Options and statistics
//===----------------------------------------------------------------------===//

typedef enum loom_template_selection_mode_e {
  LOOM_TEMPLATE_SELECTION_MODE_EARLY = 0,
  LOOM_TEMPLATE_SELECTION_MODE_FINAL = 1,
} loom_template_selection_mode_t;

typedef struct loom_template_selection_pass_state_t {
  // Early or final selection behavior.
  loom_template_selection_mode_t mode;

  // True when mode was explicitly provided.
  bool has_mode_option;
} loom_template_selection_pass_state_t;

static const loom_pass_option_def_t kTemplateSelectionOptions[] = {
    {IREE_SVL("mode"),
     IREE_SVL("Selection mode: early preserves unresolved applies, final "
              "emits diagnostics for every unresolved live apply.")},
};

enum {
  LOOM_TEMPLATE_SELECTION_STAT_APPLY_SITES = 0,
  LOOM_TEMPLATE_SELECTION_STAT_SELECTED_SITES = 1,
  LOOM_TEMPLATE_SELECTION_STAT_UNRESOLVED_SITES = 2,
  LOOM_TEMPLATE_SELECTION_STAT_PROVIDER_EDGES = 3,
  LOOM_TEMPLATE_SELECTION_STAT_SYMBOLS_PRUNED = 4,
};

static const loom_pass_statistic_def_t kTemplateSelectionStatistics[] = {
    {IREE_SVL("apply-sites"),
     IREE_SVL("Number of live func.apply sites analyzed.")},
    {IREE_SVL("selected-sites"),
     IREE_SVL("Number of func.apply sites resolved to inline calls.")},
    {IREE_SVL("unresolved-sites"),
     IREE_SVL("Number of live func.apply sites left unresolved.")},
    {IREE_SVL("provider-edges"),
     IREE_SVL("Number of apply-generated provider liveness edges.")},
    {IREE_SVL("symbols-pruned"),
     IREE_SVL("Number of unreachable private symbols pruned after selection.")},
};

static const loom_pass_info_t loom_template_selection_pass_info_storage = {
    .name = IREE_SVL("select-templates"),
    .description = IREE_SVL(
        "Select func.template providers for live func.apply contract demands."),
    .kind = LOOM_PASS_MODULE,
    .option_defs = kTemplateSelectionOptions,
    .option_count = IREE_ARRAYSIZE(kTemplateSelectionOptions),
    .statistic_defs = kTemplateSelectionStatistics,
    .statistic_count = IREE_ARRAYSIZE(kTemplateSelectionStatistics),
};

const loom_pass_info_t* loom_template_selection_pass_info(void) {
  return &loom_template_selection_pass_info_storage;
}

static iree_status_t loom_template_selection_parse_mode(
    iree_string_view_t value, loom_template_selection_pass_state_t* state) {
  if (state->has_mode_option) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "duplicate option 'mode' for pass 'select-templates'");
  }
  if (iree_string_view_equal(value, IREE_SV("early"))) {
    state->mode = LOOM_TEMPLATE_SELECTION_MODE_EARLY;
  } else if (iree_string_view_equal(value, IREE_SV("final"))) {
    state->mode = LOOM_TEMPLATE_SELECTION_MODE_FINAL;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "select-templates option 'mode' expected 'early' or 'final', got "
        "'%.*s'",
        (int)value.size, value.data);
  }
  state->has_mode_option = true;
  return iree_ok_status();
}

static iree_status_t loom_template_selection_parse_option(
    void* user_data, iree_string_view_t name, iree_string_view_t value) {
  loom_template_selection_pass_state_t* state =
      (loom_template_selection_pass_state_t*)user_data;
  if (iree_string_view_equal(name, IREE_SV("mode"))) {
    return loom_template_selection_parse_mode(value, state);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown option '%.*s' for pass 'select-templates'",
                          (int)name.size, name.data);
}

iree_status_t loom_template_selection_create(loom_pass_t* pass,
                                             iree_string_view_t options) {
  loom_template_selection_pass_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(pass->instance_arena, sizeof(*state),
                                           (void**)&state));
  memset(state, 0, sizeof(*state));
  state->mode = LOOM_TEMPLATE_SELECTION_MODE_EARLY;
  if (pass->decoded_options) {
    for (uint16_t i = 0; i < pass->decoded_options->option_count; ++i) {
      const loom_pass_decoded_option_t* option =
          &pass->decoded_options->options[i];
      if (!option->present) continue;
      if (iree_string_view_equal(option->schema->name, IREE_SV("mode"))) {
        IREE_RETURN_IF_ERROR(loom_template_selection_parse_mode(
            option->schema->enum_values[option->enum_value_index].value,
            state));
        continue;
      }
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown decoded option '%.*s' for pass "
                              "'select-templates'",
                              (int)option->schema->name.size,
                              option->schema->name.data);
    }
  } else {
    IREE_RETURN_IF_ERROR(
        loom_pass_options_parse(pass->info->name, options,
                                (loom_pass_option_parse_callback_t){
                                    .fn = loom_template_selection_parse_option,
                                    .user_data = state,
                                }));
  }
  pass->state = state;
  return iree_ok_status();
}

static loom_template_selection_mode_t loom_template_selection_mode(
    const loom_pass_t* pass) {
  if (pass->state) {
    const loom_template_selection_pass_state_t* state =
        (const loom_template_selection_pass_state_t*)pass->state;
    return state->mode;
  }
  if (!pass->decoded_options) return LOOM_TEMPLATE_SELECTION_MODE_EARLY;
  for (uint16_t i = 0; i < pass->decoded_options->option_count; ++i) {
    const loom_pass_decoded_option_t* option =
        &pass->decoded_options->options[i];
    if (!option->present) continue;
    if (!iree_string_view_equal(option->schema->name, IREE_SV("mode"))) {
      continue;
    }
    return option->enum_value_index == 1 ? LOOM_TEMPLATE_SELECTION_MODE_FINAL
                                         : LOOM_TEMPLATE_SELECTION_MODE_EARLY;
  }
  return LOOM_TEMPLATE_SELECTION_MODE_EARLY;
}

//===----------------------------------------------------------------------===//
// Plan model
//===----------------------------------------------------------------------===//

typedef enum loom_template_provider_feasibility_e {
  LOOM_TEMPLATE_PROVIDER_REJECT = 0,
  LOOM_TEMPLATE_PROVIDER_MAYBE = 1,
  LOOM_TEMPLATE_PROVIDER_MATCH = 2,
} loom_template_provider_feasibility_t;

typedef enum loom_template_selection_action_e {
  LOOM_TEMPLATE_SELECTION_ACTION_UNRESOLVED = 0,
  LOOM_TEMPLATE_SELECTION_ACTION_SELECT = 1,
} loom_template_selection_action_t;

typedef enum loom_template_selection_blocker_e {
  LOOM_TEMPLATE_SELECTION_BLOCKER_NONE = 0,
  LOOM_TEMPLATE_SELECTION_BLOCKER_NO_PROVIDER = 1,
  LOOM_TEMPLATE_SELECTION_BLOCKER_ALL_REJECTED = 2,
  LOOM_TEMPLATE_SELECTION_BLOCKER_MISSING_FACTS = 3,
  LOOM_TEMPLATE_SELECTION_BLOCKER_AMBIGUOUS = 4,
  LOOM_TEMPLATE_SELECTION_BLOCKER_MATERIALIZATION = 5,
} loom_template_selection_blocker_t;

typedef struct loom_template_selection_entry_t {
  // Live func.apply operation to rewrite or diagnose.
  loom_op_t* apply_op;

  // Interned contract key demanded by apply_op.
  loom_string_id_t contract_id;

  // Borrowed contract key text.
  iree_string_view_t contract;

  // Selected provider when action is SELECT or materialization is blocked.
  const loom_func_provider_summary_t* selected_provider;

  // Selection action for this apply.
  loom_template_selection_action_t action;

  // Reason an unresolved apply could not be selected.
  loom_template_selection_blocker_t blocker;
} loom_template_selection_entry_t;

typedef struct loom_template_selection_state_t {
  // Active pass invocation.
  loom_pass_t* pass;

  // Module being transformed.
  loom_module_t* module;

  // Early or final selection behavior.
  loom_template_selection_mode_t mode;

  // Symbol facts backing the provider catalog.
  loom_symbol_fact_table_t fact_table;

  // Local provider catalog keyed by func.apply contract.
  loom_func_provider_catalog_t catalog;

  // Concrete symbol dependencies for this module snapshot.
  loom_symbol_dependency_table_t dependencies;

  // Symbol-pruning policy shared with the liveness root classifier.
  loom_symbol_pruning_options_t pruning_options;

  // Liveness result after apply-generated provider edges.
  loom_symbol_liveness_t liveness;

  // Reachable apply-site selection entries.
  loom_template_selection_entry_t* entries;

  // Number of valid selection entries.
  iree_host_size_t entry_count;

  // Capacity of entries.
  iree_host_size_t entry_capacity;
} loom_template_selection_state_t;

static iree_string_view_t loom_template_selection_contract_name(
    const loom_module_t* module, loom_string_id_t contract_id) {
  if (contract_id < module->strings.count) {
    return module->strings.entries[contract_id];
  }
  return IREE_SV("<invalid>");
}

static iree_string_view_t loom_template_selection_blocker_reason(
    loom_template_selection_blocker_t blocker) {
  switch (blocker) {
    case LOOM_TEMPLATE_SELECTION_BLOCKER_NO_PROVIDER:
      return IREE_SV("no provider implements the requested contract");
    case LOOM_TEMPLATE_SELECTION_BLOCKER_ALL_REJECTED:
      return IREE_SV("all providers were rejected by signature constraints");
    case LOOM_TEMPLATE_SELECTION_BLOCKER_MISSING_FACTS:
      return IREE_SV(
          "selection depends on unresolved provider predicate facts");
    case LOOM_TEMPLATE_SELECTION_BLOCKER_AMBIGUOUS:
      return IREE_SV("multiple providers match at the highest priority");
    case LOOM_TEMPLATE_SELECTION_BLOCKER_MATERIALIZATION:
      return IREE_SV(
          "selected provider cannot be materialized as an inline func.call");
    case LOOM_TEMPLATE_SELECTION_BLOCKER_NONE:
    default:
      return IREE_SV("selection did not produce a provider");
  }
}

static bool loom_template_provider_is_materializable(
    const loom_func_provider_summary_t* provider) {
  return provider->origin == LOOM_FUNC_PROVIDER_ORIGIN_LOCAL &&
         provider->kind == LOOM_FUNC_PROVIDER_KIND_TEMPLATE &&
         provider->has_body && loom_symbol_ref_is_valid(provider->symbol);
}

//===----------------------------------------------------------------------===//
// Provider feasibility
//===----------------------------------------------------------------------===//

static iree_status_t loom_template_selection_types_match(
    const loom_template_selection_state_t* state, const loom_op_t* apply_op,
    const loom_func_provider_summary_t* provider, bool* out_match) {
  *out_match = false;
  loom_value_slice_t operands = loom_func_apply_operands(apply_op);
  loom_value_slice_t results = loom_func_apply_results(apply_op);
  if (operands.count != provider->argument_count ||
      results.count != provider->result_count) {
    return iree_ok_status();
  }
  loom_type_value_remap_t signature_remap = {
      .source_values =
          provider->func_facts ? provider->func_facts->argument_ids : NULL,
      .target_values = operands.values,
      .count = operands.count,
  };

  for (uint16_t i = 0; i < operands.count; ++i) {
    if (operands.values[i] >= state->module->values.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "func.apply operand value %u is outside the "
                              "module value table",
                              (uint32_t)operands.values[i]);
    }
    loom_type_t operand_type =
        loom_module_value_type(state->module, operands.values[i]);
    if (!loom_type_equal_after_value_remap(provider->argument_types[i],
                                           operand_type, &signature_remap)) {
      return iree_ok_status();
    }
  }

  for (uint16_t i = 0; i < results.count; ++i) {
    if (results.values[i] >= state->module->values.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "func.apply result value %u is outside the "
                              "module value table",
                              (uint32_t)results.values[i]);
    }
    loom_type_t result_type =
        loom_module_value_type(state->module, results.values[i]);
    if (!loom_type_equal_after_value_remap(provider->result_types[i],
                                           result_type, &signature_remap)) {
      return iree_ok_status();
    }
  }

  *out_match = true;
  return iree_ok_status();
}

static iree_status_t loom_template_selection_classify_provider(
    const loom_template_selection_state_t* state, const loom_op_t* apply_op,
    const loom_func_provider_summary_t* provider,
    loom_template_provider_feasibility_t* out_feasibility) {
  *out_feasibility = LOOM_TEMPLATE_PROVIDER_REJECT;
  bool types_match = false;
  IREE_RETURN_IF_ERROR(loom_template_selection_types_match(
      state, apply_op, provider, &types_match));
  if (!types_match) return iree_ok_status();

  if (provider->predicate_count > 0) {
    *out_feasibility = LOOM_TEMPLATE_PROVIDER_MAYBE;
    return iree_ok_status();
  }

  *out_feasibility = LOOM_TEMPLATE_PROVIDER_MATCH;
  return iree_ok_status();
}

static iree_status_t loom_template_selection_mark_provider_live(
    loom_template_selection_state_t* state,
    loom_symbol_liveness_contributor_context_t* context,
    const loom_func_provider_summary_t* provider) {
  loom_pass_statistic_add(state->pass,
                          LOOM_TEMPLATE_SELECTION_STAT_PROVIDER_EDGES, 1);
  return loom_symbol_liveness_mark_symbol_ref(context, provider->symbol);
}

static iree_status_t loom_template_selection_append_entry(
    loom_template_selection_state_t* state,
    loom_template_selection_entry_t** out_entry) {
  if (state->entry_count >= state->entry_capacity) {
    iree_host_size_t old_capacity = state->entry_capacity;
    iree_host_size_t new_capacity = old_capacity > 0 ? old_capacity * 2 : 16;
    if (new_capacity < old_capacity) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "template selection entry capacity overflow");
    }
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->pass->arena, old_capacity, new_capacity, sizeof(*state->entries),
        &state->entry_capacity, (void**)&state->entries));
  }
  loom_template_selection_entry_t* entry = &state->entries[state->entry_count];
  memset(entry, 0, sizeof(*entry));
  ++state->entry_count;
  *out_entry = entry;
  return iree_ok_status();
}

static iree_status_t loom_template_selection_mark_exact_priority(
    loom_template_selection_state_t* state,
    loom_symbol_liveness_contributor_context_t* context,
    const loom_op_t* apply_op, loom_func_provider_slice_t providers,
    int64_t priority) {
  for (iree_host_size_t i = 0; i < providers.count; ++i) {
    const loom_func_provider_summary_t* provider = &providers.providers[i];
    loom_template_provider_feasibility_t feasibility =
        LOOM_TEMPLATE_PROVIDER_REJECT;
    IREE_RETURN_IF_ERROR(loom_template_selection_classify_provider(
        state, apply_op, provider, &feasibility));
    if (feasibility != LOOM_TEMPLATE_PROVIDER_MATCH ||
        provider->priority != priority) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_template_selection_mark_provider_live(state, context, provider));
  }
  return iree_ok_status();
}

static iree_status_t loom_template_selection_mark_missing_fact_candidates(
    loom_template_selection_state_t* state,
    loom_symbol_liveness_contributor_context_t* context,
    const loom_op_t* apply_op, loom_func_provider_slice_t providers,
    bool has_exact, int64_t exact_priority) {
  for (iree_host_size_t i = 0; i < providers.count; ++i) {
    const loom_func_provider_summary_t* provider = &providers.providers[i];
    loom_template_provider_feasibility_t feasibility =
        LOOM_TEMPLATE_PROVIDER_REJECT;
    IREE_RETURN_IF_ERROR(loom_template_selection_classify_provider(
        state, apply_op, provider, &feasibility));
    if (feasibility == LOOM_TEMPLATE_PROVIDER_REJECT) continue;
    if (has_exact) {
      if (feasibility == LOOM_TEMPLATE_PROVIDER_MATCH &&
          provider->priority != exact_priority) {
        continue;
      }
      if (feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE &&
          provider->priority < exact_priority) {
        continue;
      }
    }
    IREE_RETURN_IF_ERROR(
        loom_template_selection_mark_provider_live(state, context, provider));
  }
  return iree_ok_status();
}

static iree_status_t loom_template_selection_analyze_apply(
    loom_template_selection_state_t* state,
    loom_symbol_liveness_contributor_context_t* context,
    const loom_op_t* apply_op) {
  loom_string_id_t contract_id = loom_func_apply_contract(apply_op);
  if (contract_id == LOOM_STRING_ID_INVALID ||
      contract_id >= state->module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "func.apply has an invalid contract string id");
  }

  loom_template_selection_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(loom_template_selection_append_entry(state, &entry));
  entry->apply_op = (loom_op_t*)apply_op;
  entry->contract_id = contract_id;
  entry->contract =
      loom_template_selection_contract_name(state->module, contract_id);
  entry->action = LOOM_TEMPLATE_SELECTION_ACTION_UNRESOLVED;

  loom_pass_statistic_add(state->pass, LOOM_TEMPLATE_SELECTION_STAT_APPLY_SITES,
                          1);

  loom_func_provider_slice_t providers =
      loom_func_provider_catalog_lookup(&state->catalog, contract_id);
  if (providers.count == 0) {
    entry->blocker = LOOM_TEMPLATE_SELECTION_BLOCKER_NO_PROVIDER;
    loom_pass_statistic_add(state->pass,
                            LOOM_TEMPLATE_SELECTION_STAT_UNRESOLVED_SITES, 1);
    return iree_ok_status();
  }

  bool has_exact = false;
  bool has_maybe = false;
  int64_t best_exact_priority = INT64_MIN;
  int64_t highest_maybe_priority = INT64_MIN;
  uint32_t best_exact_count = 0;
  uint32_t possible_count = 0;
  const loom_func_provider_summary_t* best_exact_provider = NULL;

  for (iree_host_size_t i = 0; i < providers.count; ++i) {
    const loom_func_provider_summary_t* provider = &providers.providers[i];
    loom_template_provider_feasibility_t feasibility =
        LOOM_TEMPLATE_PROVIDER_REJECT;
    IREE_RETURN_IF_ERROR(loom_template_selection_classify_provider(
        state, apply_op, provider, &feasibility));
    if (feasibility == LOOM_TEMPLATE_PROVIDER_REJECT) {
      continue;
    }
    ++possible_count;
    if (feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE) {
      has_maybe = true;
      if (provider->priority > highest_maybe_priority) {
        highest_maybe_priority = provider->priority;
      }
      continue;
    }

    if (!has_exact || provider->priority > best_exact_priority) {
      has_exact = true;
      best_exact_priority = provider->priority;
      best_exact_count = 1;
      best_exact_provider = provider;
    } else if (provider->priority == best_exact_priority) {
      ++best_exact_count;
    }
  }

  if (possible_count == 0) {
    entry->blocker = LOOM_TEMPLATE_SELECTION_BLOCKER_ALL_REJECTED;
    loom_pass_statistic_add(state->pass,
                            LOOM_TEMPLATE_SELECTION_STAT_UNRESOLVED_SITES, 1);
    return iree_ok_status();
  }

  if (!has_exact ||
      (has_maybe && highest_maybe_priority >= best_exact_priority)) {
    entry->blocker = LOOM_TEMPLATE_SELECTION_BLOCKER_MISSING_FACTS;
    IREE_RETURN_IF_ERROR(loom_template_selection_mark_missing_fact_candidates(
        state, context, apply_op, providers, has_exact, best_exact_priority));
    loom_pass_statistic_add(state->pass,
                            LOOM_TEMPLATE_SELECTION_STAT_UNRESOLVED_SITES, 1);
    return iree_ok_status();
  }

  if (best_exact_count > 1) {
    entry->blocker = LOOM_TEMPLATE_SELECTION_BLOCKER_AMBIGUOUS;
    IREE_RETURN_IF_ERROR(loom_template_selection_mark_exact_priority(
        state, context, apply_op, providers, best_exact_priority));
    loom_pass_statistic_add(state->pass,
                            LOOM_TEMPLATE_SELECTION_STAT_UNRESOLVED_SITES, 1);
    return iree_ok_status();
  }

  entry->selected_provider = best_exact_provider;
  IREE_RETURN_IF_ERROR(loom_template_selection_mark_provider_live(
      state, context, best_exact_provider));
  if (!loom_template_provider_is_materializable(best_exact_provider)) {
    entry->blocker = LOOM_TEMPLATE_SELECTION_BLOCKER_MATERIALIZATION;
    loom_pass_statistic_add(state->pass,
                            LOOM_TEMPLATE_SELECTION_STAT_UNRESOLVED_SITES, 1);
    return iree_ok_status();
  }

  entry->action = LOOM_TEMPLATE_SELECTION_ACTION_SELECT;
  entry->blocker = LOOM_TEMPLATE_SELECTION_BLOCKER_NONE;
  loom_pass_statistic_add(state->pass,
                          LOOM_TEMPLATE_SELECTION_STAT_SELECTED_SITES, 1);
  return iree_ok_status();
}

static iree_status_t loom_template_selection_visit_reachable_op(
    void* user_data, loom_symbol_liveness_contributor_context_t* context,
    const loom_op_t* op) {
  if (!loom_func_apply_isa(op)) return iree_ok_status();
  return loom_template_selection_analyze_apply(
      (loom_template_selection_state_t*)user_data, context, op);
}

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

static iree_status_t loom_template_selection_emit_blockers(
    loom_template_selection_state_t* state) {
  for (iree_host_size_t i = 0; i < state->entry_count; ++i) {
    const loom_template_selection_entry_t* entry = &state->entries[i];
    if (entry->action == LOOM_TEMPLATE_SELECTION_ACTION_SELECT) {
      continue;
    }
    loom_diagnostic_param_t params[] = {
        loom_param_string(loom_op_name(state->module, entry->apply_op)),
        loom_param_string(state->pass->info->name),
        loom_param_string(entry->contract),
        loom_param_string(
            loom_template_selection_blocker_reason(entry->blocker)),
    };
    loom_op_t* related_provider_op =
        entry->selected_provider ? entry->selected_provider->function.op : NULL;
    loom_diagnostic_related_op_t related_op = {
        .label = IREE_SV("selected provider"),
        .op = related_provider_op,
        .field_ref = loom_diagnostic_field_ref_none(),
    };
    loom_diagnostic_emission_t emission = {
        .op = entry->apply_op,
        .error = LOOM_ERR_LOWERING_045,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
        .related_ops = related_provider_op ? &related_op : NULL,
        .related_op_count = related_provider_op ? 1 : 0,
    };
    IREE_RETURN_IF_ERROR(
        iree_diagnostic_emit(state->pass->diagnostic_emitter, &emission));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Rewrite
//===----------------------------------------------------------------------===//

static iree_status_t loom_template_selection_copy_result_types(
    loom_template_selection_state_t* state, const loom_op_t* apply_op,
    loom_type_t** out_result_types) {
  *out_result_types = NULL;
  loom_value_slice_t results = loom_func_apply_results(apply_op);
  if (results.count == 0) return iree_ok_status();

  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->pass->arena, results.count,
                                sizeof(*result_types), (void**)&result_types));
  for (uint16_t i = 0; i < results.count; ++i) {
    if (results.values[i] >= state->module->values.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "func.apply result value %u is outside the "
                              "module value table",
                              (uint32_t)results.values[i]);
    }
    result_types[i] = loom_module_value_type(state->module, results.values[i]);
  }
  *out_result_types = result_types;
  return iree_ok_status();
}

static iree_status_t loom_template_selection_rewrite_entry(
    loom_template_selection_state_t* state, loom_rewriter_t* rewriter,
    const loom_template_selection_entry_t* entry) {
  loom_value_slice_t operands = loom_func_apply_operands(entry->apply_op);
  loom_value_slice_t results = loom_func_apply_results(entry->apply_op);
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_template_selection_copy_result_types(
      state, entry->apply_op, &result_types));

  loom_func_call_build_flags_t build_flags =
      LOOM_FUNC_CALL_BUILD_FLAG_HAS_INLINE_POLICY;
  uint8_t purity = loom_func_apply_purity(entry->apply_op);
  if (purity != 0) {
    build_flags |= LOOM_FUNC_CALL_BUILD_FLAG_HAS_PURITY;
  }
  uint8_t temperature = loom_func_apply_temperature(entry->apply_op);
  if (temperature != 0) {
    build_flags |= LOOM_FUNC_CALL_BUILD_FLAG_HAS_TEMPERATURE;
  }

  loom_builder_set_before(&rewriter->builder, entry->apply_op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* call_op = NULL;
  IREE_RETURN_IF_ERROR(loom_func_call_build(
      &rewriter->builder, build_flags, purity, temperature,
      LOOM_FUNC_INLINE_POLICY_INLINE, entry->selected_provider->symbol,
      operands.values, operands.count, result_types, results.count,
      loom_op_tied_results(entry->apply_op), entry->apply_op->tied_result_count,
      entry->apply_op->location, &call_op));
  loom_value_slice_t call_results = loom_func_call_results(call_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, entry->apply_op, call_results.values, call_results.count,
      value_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, entry->apply_op, call_results.values, call_results.count));
  loom_pass_mark_changed(state->pass);
  return iree_ok_status();
}

static iree_status_t loom_template_selection_execute_rewrites(
    loom_template_selection_state_t* state) {
  bool has_selected_entry = false;
  for (iree_host_size_t i = 0; i < state->entry_count; ++i) {
    if (state->entries[i].action == LOOM_TEMPLATE_SELECTION_ACTION_SELECT) {
      has_selected_entry = true;
      break;
    }
  }
  if (!has_selected_entry) return iree_ok_status();

  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, state->module, state->pass->arena));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < state->entry_count && iree_status_is_ok(status); ++i) {
    const loom_template_selection_entry_t* entry = &state->entries[i];
    if (entry->action != LOOM_TEMPLATE_SELECTION_ACTION_SELECT) continue;
    status = loom_template_selection_rewrite_entry(state, &rewriter, entry);
  }

  loom_rewriter_deinitialize(&rewriter);
  return status;
}

//===----------------------------------------------------------------------===//
// Pass entry
//===----------------------------------------------------------------------===//

static iree_status_t loom_template_selection_build_liveness(
    loom_template_selection_state_t* state) {
  loom_symbol_liveness_contributor_t contributor = {
      .visit_op = loom_template_selection_visit_reachable_op,
      .user_data = state,
  };
  loom_symbol_liveness_options_t options = {
      .flags = LOOM_SYMBOL_LIVENESS_INCLUDE_MODULE_EDGES,
      .root_query = loom_symbol_pruning_symbol_is_root,
      .root_query_user_data = &state->pruning_options,
      .contributors = &contributor,
      .contributor_count = 1,
  };
  return loom_symbol_liveness_compute(state->module, &state->dependencies,
                                      &options, state->pass->arena,
                                      &state->liveness);
}

iree_status_t loom_template_selection_run(loom_pass_t* pass,
                                          loom_module_t* module) {
  loom_template_selection_state_t state = {
      .pass = pass,
      .module = module,
      .mode = loom_template_selection_mode(pass),
      .pruning_options =
          {
              .flags = LOOM_SYMBOL_PRUNING_RETAIN_TARGET_SOURCE_ENTRIES,
          },
  };
  loom_symbol_fact_table_initialize(&state.fact_table, pass->arena);
  loom_func_provider_catalog_initialize(&state.catalog, pass->arena);

  IREE_RETURN_IF_ERROR(loom_func_provider_catalog_build_local(
      &state.catalog, module, &state.fact_table));
  IREE_RETURN_IF_ERROR(loom_symbol_dependency_table_build(module, pass->arena,
                                                          &state.dependencies));
  IREE_RETURN_IF_ERROR(loom_template_selection_build_liveness(&state));

  if (state.mode == LOOM_TEMPLATE_SELECTION_MODE_FINAL) {
    IREE_RETURN_IF_ERROR(loom_template_selection_emit_blockers(&state));
    if (loom_pass_has_error_diagnostics(pass)) {
      return iree_ok_status();
    }
  }

  IREE_RETURN_IF_ERROR(loom_template_selection_execute_rewrites(&state));

  loom_symbol_pruning_result_t pruning_result = {0};
  IREE_RETURN_IF_ERROR(loom_symbol_pruning_erase_unreachable(
      module, &state.liveness, &state.pruning_options, pass->arena,
      &pruning_result));
  if (pruning_result.symbol_count > 0) {
    loom_pass_mark_changed(pass);
    loom_pass_statistic_add(pass, LOOM_TEMPLATE_SELECTION_STAT_SYMBOLS_PRUNED,
                            pruning_result.symbol_count);
  }

  if (!pass->changed) {
    return iree_ok_status();
  }
  return loom_module_compact_symbols(module, pass->arena, NULL);
}
