// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/passes/target_legalize.h"

#include <string.h>

#include "loom/codegen/low/pass_environment.h"
#include "loom/codegen/low/source_selection.h"
#include "loom/pass/pipeline.h"
#include "loom/pass/registry.h"
#include "loom/pass/value_facts.h"
#include "loom/passes/vector/target_legalization.h"
#include "loom/rewrite/greedy.h"
#include "loom/target/legalization.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/target/low_legality.h"

typedef struct loom_low_target_legalize_pass_state_t {
  // Maximum number of fixed-point rewrite iterations.
  uint32_t max_iterations;
  // Maximum number of final legality diagnostics emitted before stopping.
  uint32_t max_errors;
  // Current target legalization phase.
  loom_target_legalization_mode_t mode;
  // True when max_iterations was explicitly provided.
  bool has_max_iterations_option;
  // True when max_errors was explicitly provided.
  bool has_max_errors_option;
  // True when mode was explicitly provided.
  bool has_mode_option;
} loom_low_target_legalize_pass_state_t;

typedef struct loom_low_target_legalize_parse_context_t {
  // Mutable pass state being populated.
  loom_low_target_legalize_pass_state_t* state;
} loom_low_target_legalize_parse_context_t;

static const loom_pass_option_def_t kLowTargetLegalizeOptions[] = {
    {IREE_SVL("max-errors"),
     IREE_SVL("Maximum number of final legality diagnostics to emit; zero "
              "means no limit.")},
    {IREE_SVL("max-iterations"),
     IREE_SVL("Maximum number of target legalization worklist iterations.")},
    {IREE_SVL("mode"), IREE_SVL("Legalization phase: eager or final.")},
};

enum {
  LOOM_LOW_TARGET_LEGALIZE_STAT_OPS_REWRITTEN = 0,
  LOOM_LOW_TARGET_LEGALIZE_STAT_LOOPS_CREATED = 1,
  LOOM_LOW_TARGET_LEGALIZE_STAT_LANES_MATERIALIZED = 2,
  LOOM_LOW_TARGET_LEGALIZE_STAT_OPS_LEGAL = 3,
  LOOM_LOW_TARGET_LEGALIZE_STAT_OPS_DEFERRED = 4,
  LOOM_LOW_TARGET_LEGALIZE_STAT_FUNCTIONS = 5,
  LOOM_LOW_TARGET_LEGALIZE_STAT_ERRORS = 6,
};

static const loom_pass_statistic_def_t kLowTargetLegalizeStatistics[] = {
    {IREE_SVL("ops-rewritten"),
     IREE_SVL("Number of source ops rewritten by target legalizers.")},
    {IREE_SVL("loops-created"),
     IREE_SVL("Number of structured loops created by reference lowerings.")},
    {IREE_SVL("lanes-materialized"),
     IREE_SVL("Number of scalar lane programs materialized by reference "
              "lowerings.")},
    {IREE_SVL("ops-legal"),
     IREE_SVL("Number of legalizer-rooted ops already accepted by the target "
              "contract.")},
    {IREE_SVL("ops-deferred"),
     IREE_SVL("Number of legalizer-rooted ops left for a later phase.")},
    {IREE_SVL("functions"),
     IREE_SVL("Number of target-bound source functions visited.")},
    {IREE_SVL("errors"), IREE_SVL("Number of final legality errors emitted.")},
};

static const loom_pass_info_t loom_low_target_legalize_pass_info_storage = {
    .name = IREE_SVL("target-legalize"),
    .description =
        IREE_SVL("Rewrite unsupported target-bound source ops toward a legal "
                 "target-low contract."),
    .kind = LOOM_PASS_MODULE,
    .option_defs = kLowTargetLegalizeOptions,
    .option_count = IREE_ARRAYSIZE(kLowTargetLegalizeOptions),
    .statistic_defs = kLowTargetLegalizeStatistics,
    .statistic_count = IREE_ARRAYSIZE(kLowTargetLegalizeStatistics),
};

const loom_pass_info_t* loom_low_target_legalize_pass_info(void) {
  return &loom_low_target_legalize_pass_info_storage;
}

static iree_status_t loom_low_target_legalize_parse_max_errors(
    uint32_t max_errors, loom_low_target_legalize_parse_context_t* context) {
  if (context->state->has_max_errors_option) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "duplicate option 'max-errors' for pass 'target-legalize'");
  }
  context->state->max_errors = max_errors;
  context->state->has_max_errors_option = true;
  return iree_ok_status();
}

static iree_status_t loom_low_target_legalize_parse_max_iterations(
    uint32_t max_iterations,
    loom_low_target_legalize_parse_context_t* context) {
  if (context->state->has_max_iterations_option) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "duplicate option 'max-iterations' for pass 'target-legalize'");
  }
  if (max_iterations == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass 'target-legalize' option "
                            "'max-iterations' must be greater than 0");
  }
  context->state->max_iterations = max_iterations;
  context->state->has_max_iterations_option = true;
  return iree_ok_status();
}

static iree_status_t loom_low_target_legalize_parse_mode(
    iree_string_view_t value,
    loom_low_target_legalize_parse_context_t* context) {
  if (context->state->has_mode_option) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate option 'mode' for pass "
                            "'target-legalize'");
  }
  if (iree_string_view_equal(value, IREE_SV("eager"))) {
    context->state->mode = LOOM_TARGET_LEGALIZATION_MODE_EAGER;
  } else if (iree_string_view_equal(value, IREE_SV("final"))) {
    context->state->mode = LOOM_TARGET_LEGALIZATION_MODE_FINAL;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-legalize option 'mode' expected 'eager' "
                            "or 'final', got '%.*s'",
                            (int)value.size, value.data);
  }
  context->state->has_mode_option = true;
  return iree_ok_status();
}

static iree_status_t loom_low_target_legalize_parse_option(
    void* user_data, iree_string_view_t name, iree_string_view_t value) {
  loom_low_target_legalize_parse_context_t* context =
      (loom_low_target_legalize_parse_context_t*)user_data;
  if (iree_string_view_equal(name, IREE_SV("max-errors"))) {
    uint32_t max_errors = 0;
    IREE_RETURN_IF_ERROR(loom_pass_option_parse_uint32(
        IREE_SV("target-legalize"), name, value, &max_errors));
    return loom_low_target_legalize_parse_max_errors(max_errors, context);
  }
  if (iree_string_view_equal(name, IREE_SV("max-iterations"))) {
    uint32_t max_iterations = 0;
    IREE_RETURN_IF_ERROR(loom_pass_option_parse_uint32(
        IREE_SV("target-legalize"), name, value, &max_iterations));
    return loom_low_target_legalize_parse_max_iterations(max_iterations,
                                                         context);
  }
  if (iree_string_view_equal(name, IREE_SV("mode"))) {
    return loom_low_target_legalize_parse_mode(value, context);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown option '%.*s' for pass 'target-legalize'",
                          (int)name.size, name.data);
}

iree_status_t loom_low_target_legalize_create(loom_pass_t* pass,
                                              iree_string_view_t options) {
  loom_low_target_legalize_pass_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(pass->instance_arena, sizeof(*state),
                                           (void**)&state));
  memset(state, 0, sizeof(*state));
  state->max_errors = 20;
  state->max_iterations = LOOM_GREEDY_REWRITE_DEFAULT_MAX_ITERATIONS;
  state->mode = LOOM_TARGET_LEGALIZATION_MODE_EAGER;

  loom_low_target_legalize_parse_context_t context = {
      .state = state,
  };
  if (pass->decoded_options) {
    for (uint16_t i = 0; i < pass->decoded_options->option_count; ++i) {
      const loom_pass_decoded_option_t* option =
          &pass->decoded_options->options[i];
      if (!option->present) {
        continue;
      }
      if (iree_string_view_equal(option->schema->name, IREE_SV("max-errors"))) {
        IREE_RETURN_IF_ERROR(loom_low_target_legalize_parse_max_errors(
            option->uint32_value, &context));
        continue;
      }
      if (iree_string_view_equal(option->schema->name,
                                 IREE_SV("max-iterations"))) {
        IREE_RETURN_IF_ERROR(loom_low_target_legalize_parse_max_iterations(
            option->uint32_value, &context));
        continue;
      }
      if (iree_string_view_equal(option->schema->name, IREE_SV("mode"))) {
        IREE_RETURN_IF_ERROR(loom_low_target_legalize_parse_mode(
            option->schema->enum_values[option->enum_value_index].value,
            &context));
        continue;
      }
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown decoded option '%.*s' for pass "
                              "'target-legalize'",
                              (int)option->schema->name.size,
                              option->schema->name.data);
    }
  } else {
    IREE_RETURN_IF_ERROR(
        loom_pass_options_parse(pass->info->name, options,
                                (loom_pass_option_parse_callback_t){
                                    .fn = loom_low_target_legalize_parse_option,
                                    .user_data = &context,
                                }));
  }

  pass->state = state;
  return iree_ok_status();
}

typedef struct loom_low_target_legalize_function_state_t {
  // Pass invocation being executed.
  loom_pass_t* pass;
  // Module being legalized.
  loom_module_t* module;
  // Selected target-bound source function.
  const loom_low_source_selection_t* selection;
  // Source-to-low query options shared by eager and final legality checks.
  loom_low_lower_options_t lower_options;
  // Low descriptor set selected by selection->target_bundle.
  const loom_low_descriptor_set_t* descriptor_set;
  // Active source-to-low contract query scope, rebuilt after IR mutation.
  loom_low_lower_source_query_scope_t* query_scope;
  // Fact table used to initialize query_scope.
  const loom_value_fact_table_t* query_scope_fact_table;
  // Arena receiving query_scope storage for this function run.
  iree_arena_allocator_t* query_scope_arena;
  // Target legalizer registry shared across functions in this pass run.
  const loom_target_legalizer_registry_t* legalizer_registry;
  // Target-low legality providers available to the final verifier.
  loom_target_low_legality_provider_list_t legality_provider_list;
  // Current legalization context passed to legalizer callbacks.
  loom_target_legalization_context_t legalization_context;
} loom_low_target_legalize_function_state_t;

static void loom_low_target_legalize_destroy_query_scope(
    loom_low_target_legalize_function_state_t* state) {
  if (state->query_scope == NULL) {
    return;
  }
  loom_low_lower_source_query_scope_destroy(state->query_scope);
  state->query_scope = NULL;
  state->query_scope_fact_table = NULL;
}

static iree_status_t loom_low_target_legalize_refresh_query_scope(
    loom_low_target_legalize_function_state_t* state,
    const loom_value_fact_table_t* fact_table) {
  loom_low_target_legalize_destroy_query_scope(state);
  state->lower_options.fact_table = (loom_value_fact_table_t*)fact_table;
  IREE_RETURN_IF_ERROR(loom_low_lower_source_query_scope_create(
      state->module, state->selection->func, &state->lower_options,
      state->query_scope_arena, &state->query_scope));
  state->query_scope_fact_table = fact_table;
  return iree_ok_status();
}

static iree_status_t loom_low_target_legalize_ensure_query_scope(
    loom_low_target_legalize_function_state_t* state,
    const loom_value_fact_table_t* fact_table) {
  if (state->query_scope != NULL &&
      state->query_scope_fact_table == fact_table) {
    return iree_ok_status();
  }
  return loom_low_target_legalize_refresh_query_scope(state, fact_table);
}

static iree_status_t loom_low_target_legalize_query_contract(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result) {
  loom_low_target_legalize_function_state_t* state =
      (loom_low_target_legalize_function_state_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_low_target_legalize_ensure_query_scope(
      state, environment->fact_table));
  const loom_target_contract_query_callback_t callback =
      loom_low_lower_source_query_scope_callback(state->query_scope);
  return callback.fn(callback.user_data, environment, source_op, out_result);
}

static iree_status_t loom_low_target_legalize_rewrite_op(
    void* user_data, loom_greedy_rewrite_driver_t* driver, loom_op_t* op,
    loom_greedy_rewrite_result_t* result, bool* out_changed) {
  *out_changed = false;
  loom_low_target_legalize_function_state_t* state =
      (loom_low_target_legalize_function_state_t*)user_data;
  const loom_target_legalizer_op_entry_t op_entry =
      loom_target_legalizer_registry_lookup_kind(state->legalizer_registry,
                                                 op->kind);
  if (loom_target_legalizer_op_entry_is_empty(op_entry)) {
    return iree_ok_status();
  }

  state->legalization_context.fact_table = driver->latest_facts;
  state->legalization_context.rewriter = &driver->rewriter;
  state->legalization_context.arena = driver->scratch_arena;
  loom_target_contract_query_result_t query_result =
      loom_target_contract_query_result_empty();
  IREE_RETURN_IF_ERROR(loom_target_legalization_query_contract(
      &state->legalization_context, op, &query_result));
  if (query_result.outcome == LOOM_TARGET_CONTRACT_QUERY_LEGAL) {
    loom_pass_statistic_add(state->pass,
                            LOOM_LOW_TARGET_LEGALIZE_STAT_OPS_LEGAL, 1);
    return iree_ok_status();
  }
  if (query_result.outcome == LOOM_TARGET_CONTRACT_QUERY_INVALID_IR) {
    loom_pass_statistic_add(state->pass,
                            LOOM_LOW_TARGET_LEGALIZE_STAT_OPS_DEFERRED, 1);
    return iree_ok_status();
  }

  for (uint16_t i = 0; i < op_entry.entry_count; ++i) {
    const loom_target_legalizer_entry_t* entry =
        &state->legalizer_registry->entries[op_entry.entry_start + i];
    driver->rewriter.flags = 0;
    loom_target_legalizer_result_t legalizer_result = {
        .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
    };
    IREE_RETURN_IF_ERROR(entry->legalize(entry, &state->legalization_context,
                                         op, &legalizer_result));
    switch (legalizer_result.action) {
      case LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT:
        continue;
      case LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN:
        loom_greedy_rewrite_result_record_change(
            result, &driver->rewriter,
            LOOM_GREEDY_REWRITE_CHANGE_FLAG_COUNT_MODIFIED_OP);
        loom_pass_statistic_add(state->pass,
                                LOOM_LOW_TARGET_LEGALIZE_STAT_OPS_REWRITTEN, 1);
        *out_changed = true;
        return iree_ok_status();
      case LOOM_TARGET_LEGALIZER_ACTION_DEFER:
      case LOOM_TARGET_LEGALIZER_ACTION_REJECT_INVALID_IR:
      case LOOM_TARGET_LEGALIZER_ACTION_REJECT_UNSUPPORTED_FINAL:
        loom_pass_statistic_add(state->pass,
                                LOOM_LOW_TARGET_LEGALIZE_STAT_OPS_DEFERRED, 1);
        return iree_ok_status();
      default:
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "target legalizer returned unknown action %d",
                                (int)legalizer_result.action);
    }
  }

  loom_pass_statistic_add(state->pass,
                          LOOM_LOW_TARGET_LEGALIZE_STAT_OPS_DEFERRED, 1);
  return iree_ok_status();
}

static void loom_low_target_legalize_changed(
    void* user_data, loom_greedy_rewrite_driver_t* driver) {
  (void)driver;
  loom_low_target_legalize_destroy_query_scope(
      (loom_low_target_legalize_function_state_t*)user_data);
}

static iree_status_t loom_low_target_legalize_verify_final(
    loom_module_t* module, loom_low_target_legalize_function_state_t* state,
    const loom_low_target_legalize_pass_state_t* pass_state,
    const loom_value_fact_table_t* fact_table, uint32_t* out_error_count) {
  *out_error_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_low_target_legalize_refresh_query_scope(state, fact_table));
  const loom_target_contract_query_callback_t contract_query =
      loom_low_lower_source_query_scope_callback(state->query_scope);
  loom_target_low_legality_result_t result = {0};
  const loom_target_low_legality_options_t legality_options = {
      .bundle = state->selection->target_bundle,
      .target_ref = state->selection->target_ref,
      .descriptor_registry = state->lower_options.descriptor_registry,
      .error_catalog = state->selection->policy->error_catalog,
      .provider_list = state->legality_provider_list,
      .contract_query = contract_query,
      .type_supported = state->selection->policy->source_type_supported,
      .fact_table = (loom_value_fact_table_t*)fact_table,
      .value_domain =
          loom_low_lower_source_query_scope_value_domain(state->query_scope),
      .emitter = state->pass->diagnostic_emitter,
      .max_errors = pass_state->max_errors,
  };
  iree_status_t status = loom_target_low_verify_function_legality(
      module, state->selection->func, &legality_options, &result);
  *out_error_count = result.error_count;
  return status;
}

static iree_status_t loom_low_target_legalize_function(
    loom_pass_t* pass, loom_module_t* module,
    const loom_low_target_legalize_pass_state_t* pass_state,
    const loom_low_source_selection_t* selection,
    const loom_target_legalizer_registry_t* legalizer_registry,
    loom_target_low_legality_provider_list_t legality_provider_list,
    iree_arena_allocator_t* arena, bool* out_emitted_error_diagnostics) {
  *out_emitted_error_diagnostics = false;
  const loom_low_pass_capability_t* low_capability =
      loom_low_pass_capability_from_pass(pass);
  const loom_low_descriptor_registry_t* descriptor_registry =
      loom_low_pass_capability_descriptor_registry(low_capability);

  loom_low_target_legalize_function_state_t state = {
      .pass = pass,
      .module = module,
      .selection = selection,
      .query_scope_arena = arena,
      .legalizer_registry = legalizer_registry,
      .legality_provider_list = legality_provider_list,
  };
  IREE_RETURN_IF_ERROR(loom_target_low_descriptor_set_select_for_bundle(
      descriptor_registry, selection->target_bundle, &state.descriptor_set));

  loom_value_fact_table_t* seed_facts = NULL;
  if (pass->value_facts != NULL) {
    IREE_RETURN_IF_ERROR(loom_pass_value_facts_acquire(
        pass, module,
        loom_pass_value_fact_scope_function_for_target(
            selection->func, selection->target_bundle),
        &seed_facts));
  }
  state.lower_options = (loom_low_lower_options_t){
      .target_ref = selection->target_ref,
      .bundle = selection->target_bundle,
      .descriptor_registry = descriptor_registry,
      .legality_provider_list = legality_provider_list,
      .policy = selection->policy,
      .fact_table = seed_facts,
      .emitter = pass->diagnostic_emitter,
      .max_errors = pass_state->max_errors,
  };

  loom_pass_value_fact_owner_t rewrite_value_facts = {0};
  loom_pass_value_fact_owner_initialize(module->arena.block_pool,
                                        &rewrite_value_facts);
  iree_arena_allocator_t rewrite_arena;
  iree_arena_initialize(module->arena.block_pool, &rewrite_arena);
  loom_greedy_rewrite_driver_t rewrite_driver;
  loom_greedy_rewrite_driver_initialize(module, &rewrite_arena,
                                        &rewrite_value_facts, &rewrite_driver);

  state.legalization_context = (loom_target_legalization_context_t){
      .pass = pass,
      .module = module,
      .function = selection->func,
      .bundle = selection->target_bundle,
      .target_ref = selection->target_ref,
      .descriptor_set = state.descriptor_set,
      .mode = pass_state->mode,
      .contract_query =
          {
              .fn = loom_low_target_legalize_query_contract,
              .user_data = &state,
          },
  };
  const loom_greedy_rewrite_options_t rewrite_options = {
      .max_iterations = pass_state->max_iterations,
      .seed_facts = seed_facts,
  };
  const loom_greedy_rewrite_callbacks_t rewrite_callbacks = {
      .user_data = &state,
      .rewrite_op = loom_low_target_legalize_rewrite_op,
      .changed = loom_low_target_legalize_changed,
  };
  loom_greedy_rewrite_result_t rewrite_result = {0};
  iree_status_t status = loom_greedy_rewrite_run_region(
      &rewrite_driver, selection->func, loom_func_like_body(selection->func),
      selection->func.op, &rewrite_options, &rewrite_callbacks,
      &rewrite_result);
  if (iree_status_is_ok(status) && rewrite_result.changed) {
    loom_pass_mark_changed(pass);
    if (pass->value_facts != NULL) {
      loom_pass_value_fact_owner_invalidate(pass->value_facts);
    }
  }

  uint32_t final_error_count = 0;
  if (iree_status_is_ok(status) &&
      pass_state->mode == LOOM_TARGET_LEGALIZATION_MODE_FINAL) {
    status = loom_low_target_legalize_verify_final(
        module, &state, pass_state,
        loom_greedy_rewrite_driver_fact_table(&rewrite_driver),
        &final_error_count);
  }
  loom_low_target_legalize_destroy_query_scope(&state);
  loom_greedy_rewrite_driver_deinitialize(&rewrite_driver);
  iree_arena_deinitialize(&rewrite_arena);
  loom_pass_value_fact_owner_deinitialize(&rewrite_value_facts);
  IREE_RETURN_IF_ERROR(status);

  loom_pass_statistic_add(pass, LOOM_LOW_TARGET_LEGALIZE_STAT_ERRORS,
                          final_error_count);
  *out_emitted_error_diagnostics = final_error_count != 0;
  return iree_ok_status();
}

static iree_status_t loom_low_target_legalize_compose_providers(
    const loom_target_legalizer_provider_list_t* target_provider_list,
    iree_arena_allocator_t* arena,
    const loom_target_legalizer_provider_t* const** out_providers,
    iree_host_size_t* out_provider_count) {
  const loom_target_legalizer_provider_t* vector_provider =
      loom_vector_target_legalizer_provider();
  const iree_host_size_t target_provider_count =
      target_provider_list ? target_provider_list->count : 0;
  const iree_host_size_t provider_count = target_provider_count + 1;
  const loom_target_legalizer_provider_t** providers = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, provider_count, sizeof(*providers), (void**)&providers));
  for (iree_host_size_t i = 0; i < target_provider_count; ++i) {
    providers[i] = target_provider_list->values[i];
  }
  providers[target_provider_count] = vector_provider;
  *out_providers = providers;
  *out_provider_count = provider_count;
  return iree_ok_status();
}

iree_status_t loom_low_target_legalize_run(loom_pass_t* pass,
                                           loom_module_t* module) {
  const loom_low_target_legalize_pass_state_t* state =
      (const loom_low_target_legalize_pass_state_t*)pass->state;
  const loom_low_pass_capability_t* low_capability =
      loom_low_pass_capability_from_pass(pass);
  const loom_low_lower_policy_registry_t* policy_registry =
      loom_low_pass_capability_lower_policy_registry(low_capability);
  const loom_target_low_legality_provider_list_t* legality_provider_list =
      loom_low_pass_capability_legality_provider_list(low_capability);
  const loom_target_legalizer_provider_list_t* target_legalizer_provider_list =
      loom_low_pass_capability_legalizer_provider_list(low_capability);

  iree_arena_allocator_t run_arena;
  iree_arena_initialize(module->arena.block_pool, &run_arena);

  const loom_target_legalizer_provider_t* const* legalizer_providers = NULL;
  iree_host_size_t legalizer_provider_count = 0;
  iree_status_t status = loom_low_target_legalize_compose_providers(
      target_legalizer_provider_list, &run_arena, &legalizer_providers,
      &legalizer_provider_count);

  loom_target_legalizer_registry_t legalizer_registry = {0};
  if (iree_status_is_ok(status)) {
    status = loom_target_legalizer_registry_compose(
        legalizer_providers, legalizer_provider_count, &legalizer_registry,
        &run_arena);
  }

  loom_low_source_selection_list_t selection_list = {0};
  if (iree_status_is_ok(status)) {
    const loom_low_source_selection_options_t selection_options = {
        .policy_registry = policy_registry,
        .diagnostic_emitter = pass->diagnostic_emitter,
        .lowering_kind = IREE_SV("target-legalize"),
    };
    status = loom_low_select_source_funcs(module, &selection_options,
                                          &run_arena, &selection_list);
  }

  bool emitted_error_diagnostics = false;
  uint32_t function_count = 0;
  const loom_target_low_legality_provider_list_t legality_list =
      legality_provider_list ? *legality_provider_list
                             : loom_target_low_legality_provider_list_empty();
  for (iree_host_size_t i = 0;
       i < selection_list.count && iree_status_is_ok(status) &&
       !emitted_error_diagnostics;
       ++i) {
    status = loom_low_target_legalize_function(
        pass, module, state, &selection_list.values[i], &legalizer_registry,
        legality_list, &run_arena, &emitted_error_diagnostics);
    if (iree_status_is_ok(status)) {
      ++function_count;
    }
  }

  iree_arena_deinitialize(&run_arena);
  IREE_RETURN_IF_ERROR(status);

  loom_pass_statistic_add(pass, LOOM_LOW_TARGET_LEGALIZE_STAT_FUNCTIONS,
                          function_count);
  return iree_ok_status();
}
