// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation_pass.h"

#include <string.h>

#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/allocation_materialization.h"
#include "loom/ops/low/ops.h"
#include "loom/pass/pipeline.h"
#include "loom/pass/registry.h"

typedef struct loom_low_materialize_allocation_pass_state_t {
  // Fixed register budget overrides parsed from the pass options.
  loom_low_allocation_budget_t* budgets;
  // Number of entries in |budgets|.
  iree_host_size_t budget_count;
  // True when the pass should emit spill materialization diagnostics.
  bool emit_spill_diagnostics;
  // True once the diagnostics option has been parsed.
  bool has_diagnostics_option;
} loom_low_materialize_allocation_pass_state_t;

typedef struct loom_low_materialize_allocation_parse_context_t {
  // Pass instance that owns option storage.
  loom_pass_t* pass;
  // Mutable pass state being populated.
  loom_low_materialize_allocation_pass_state_t* state;
} loom_low_materialize_allocation_parse_context_t;

static const loom_pass_option_def_t kLowMaterializeAllocationOptions[] = {
    {IREE_SVL("budgets"),
     IREE_SVL("Semicolon-separated register class budgets, such as "
              "vm.i32=2;x86.zmm=1.")},
    {IREE_SVL("diagnostics"),
     IREE_SVL("Diagnostic feedback to emit: none or spills.")},
};

enum {
  LOOM_LOW_MATERIALIZE_ALLOCATION_STAT_SLOTS = 0,
  LOOM_LOW_MATERIALIZE_ALLOCATION_STAT_SPILLS = 1,
  LOOM_LOW_MATERIALIZE_ALLOCATION_STAT_RELOADS = 2,
};

static const loom_pass_statistic_def_t kLowMaterializeAllocationStatistics[] = {
    {IREE_SVL("slots"), IREE_SVL("Number of low.slot records inserted.")},
    {IREE_SVL("spills"), IREE_SVL("Number of low.spill stores inserted.")},
    {IREE_SVL("reloads"), IREE_SVL("Number of low.reload ops inserted.")},
};

static const loom_pass_info_t
    loom_low_materialize_allocation_pass_info_storage = {
        .name = IREE_SVL("low-materialize-allocation"),
        .description =
            IREE_SVL("Allocate target-low values and materialize spills."),
        .kind = LOOM_PASS_FUNCTION,
        .option_defs = kLowMaterializeAllocationOptions,
        .option_count = IREE_ARRAYSIZE(kLowMaterializeAllocationOptions),
        .statistic_defs = kLowMaterializeAllocationStatistics,
        .statistic_count = IREE_ARRAYSIZE(kLowMaterializeAllocationStatistics),
};

const loom_pass_info_t* loom_low_materialize_allocation_pass_info(void) {
  return &loom_low_materialize_allocation_pass_info_storage;
}

bool loom_low_materialize_allocation_pass_config_satisfies_requirement(
    const loom_low_materialize_allocation_pass_config_t* config,
    iree_string_view_t requirement) {
  if (iree_string_view_equal(
          requirement,
          IREE_SV(LOOM_LOW_PASS_REQUIREMENT_TARGET_LOW_DESCRIPTOR_REGISTRY))) {
    return config && config->descriptor_registry;
  }
  return false;
}

static iree_status_t loom_low_materialize_allocation_count_budgets(
    iree_string_view_t text, iree_host_size_t* out_count) {
  iree_host_size_t count = 0;
  text = iree_string_view_trim(text);
  while (!iree_string_view_is_empty(text)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_t remaining = iree_string_view_empty();
    iree_string_view_split(text, ';', &token, &remaining);
    token = iree_string_view_trim(token);
    if (iree_string_view_is_empty(token)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass 'low-materialize-allocation' option 'budgets' contains an "
          "empty budget entry");
    }
    ++count;
    text = iree_string_view_trim(remaining);
  }
  *out_count = count;
  return iree_ok_status();
}

static iree_status_t loom_low_materialize_allocation_parse_budget(
    iree_string_view_t token, loom_low_allocation_budget_t* out_budget) {
  iree_string_view_t register_class = iree_string_view_empty();
  iree_string_view_t budget_text = iree_string_view_empty();
  iree_string_view_split(token, '=', &register_class, &budget_text);
  register_class = iree_string_view_trim(register_class);
  budget_text = iree_string_view_trim(budget_text);
  if (iree_string_view_is_empty(register_class) ||
      iree_string_view_is_empty(budget_text)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass 'low-materialize-allocation' option 'budgets' entries must have "
        "the form <register-class>=<units>");
  }
  uint32_t max_units = 0;
  if (!iree_string_view_atoi_uint32(budget_text, &max_units)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass 'low-materialize-allocation' budget expected a uint32 value, "
        "got '%.*s'",
        (int)budget_text.size, budget_text.data);
  }
  *out_budget = (loom_low_allocation_budget_t){
      .register_class = register_class,
      .max_units = max_units,
  };
  return iree_ok_status();
}

static iree_status_t loom_low_materialize_allocation_parse_budgets(
    iree_string_view_t text,
    loom_low_materialize_allocation_parse_context_t* context) {
  if (context->state->budgets) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "duplicate option 'budgets' for pass 'low-materialize-allocation'");
  }

  iree_host_size_t budget_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_low_materialize_allocation_count_budgets(text, &budget_count));
  if (budget_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass 'low-materialize-allocation' option 'budgets' must not be empty");
  }

  loom_low_allocation_budget_t* budgets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->pass->instance_arena,
                                                 budget_count, sizeof(*budgets),
                                                 (void**)&budgets));
  memset(budgets, 0, budget_count * sizeof(*budgets));

  iree_host_size_t index = 0;
  text = iree_string_view_trim(text);
  while (!iree_string_view_is_empty(text)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_t remaining = iree_string_view_empty();
    iree_string_view_split(text, ';', &token, &remaining);
    token = iree_string_view_trim(token);
    IREE_RETURN_IF_ERROR(
        loom_low_materialize_allocation_parse_budget(token, &budgets[index++]));
    text = iree_string_view_trim(remaining);
  }

  context->state->budgets = budgets;
  context->state->budget_count = budget_count;
  return iree_ok_status();
}

static iree_status_t loom_low_materialize_allocation_parse_diagnostics(
    iree_string_view_t text,
    loom_low_materialize_allocation_parse_context_t* context) {
  if (context->state->has_diagnostics_option) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "duplicate option 'diagnostics' for pass 'low-materialize-allocation'");
  }
  text = iree_string_view_trim(text);
  if (iree_string_view_equal(text, IREE_SV("none"))) {
    context->state->emit_spill_diagnostics = false;
  } else if (iree_string_view_equal(text, IREE_SV("spills"))) {
    context->state->emit_spill_diagnostics = true;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass 'low-materialize-allocation' option 'diagnostics' expected "
        "'none' or 'spills', got '%.*s'",
        (int)text.size, text.data);
  }
  context->state->has_diagnostics_option = true;
  return iree_ok_status();
}

static iree_status_t loom_low_materialize_allocation_parse_option(
    void* user_data, iree_string_view_t name, iree_string_view_t value) {
  loom_low_materialize_allocation_parse_context_t* context =
      (loom_low_materialize_allocation_parse_context_t*)user_data;
  if (iree_string_view_equal(name, IREE_SV("budgets"))) {
    return loom_low_materialize_allocation_parse_budgets(value, context);
  }
  if (iree_string_view_equal(name, IREE_SV("diagnostics"))) {
    return loom_low_materialize_allocation_parse_diagnostics(value, context);
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "unknown option '%.*s' for pass 'low-materialize-allocation'",
      (int)name.size, name.data);
}

iree_status_t loom_low_materialize_allocation_create(
    loom_pass_t* pass, iree_string_view_t options) {
  loom_low_materialize_allocation_pass_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(pass->instance_arena, sizeof(*state),
                                           (void**)&state));
  memset(state, 0, sizeof(*state));

  loom_low_materialize_allocation_parse_context_t context = {
      .pass = pass,
      .state = state,
  };
  if (pass->decoded_options) {
    for (uint16_t i = 0; i < pass->decoded_options->option_count; ++i) {
      const loom_pass_decoded_option_t* option =
          &pass->decoded_options->options[i];
      if (!option->present) {
        continue;
      }
      if (iree_string_view_equal(option->schema->name, IREE_SV("budgets"))) {
        IREE_RETURN_IF_ERROR(loom_low_materialize_allocation_parse_budgets(
            option->string_value, &context));
        continue;
      }
      if (iree_string_view_equal(option->schema->name,
                                 IREE_SV("diagnostics"))) {
        iree_string_view_t value =
            option->schema->enum_values[option->enum_value_index].value;
        IREE_RETURN_IF_ERROR(
            loom_low_materialize_allocation_parse_diagnostics(value, &context));
        continue;
      }
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown decoded option '%.*s' for pass "
                              "'low-materialize-allocation'",
                              (int)option->schema->name.size,
                              option->schema->name.data);
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_pass_options_parse(
        pass->info->name, options,
        (loom_pass_option_parse_callback_t){
            .fn = loom_low_materialize_allocation_parse_option,
            .user_data = &context,
        }));
  }
  pass->state = state;
  return iree_ok_status();
}

iree_status_t loom_low_materialize_allocation_run(loom_pass_t* pass,
                                                  loom_module_t* module,
                                                  loom_func_like_t function) {
  if (!loom_low_func_def_isa(function.op)) {
    return iree_ok_status();
  }

  loom_low_materialize_allocation_pass_state_t* state =
      (loom_low_materialize_allocation_pass_state_t*)pass->state;
  const loom_low_materialize_allocation_pass_config_t* config =
      (const loom_low_materialize_allocation_pass_config_t*)pass->user_data;
  if (!config || !config->descriptor_registry) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "pass 'low-materialize-allocation' requires an injected low "
        "descriptor registry");
  }
  loom_low_allocation_options_t allocation_options = {
      .descriptor_registry = config->descriptor_registry,
      .budgets = state ? state->budgets : NULL,
      .budget_count = state ? state->budget_count : 0,
      .emitter = pass->diagnostic_emitter,
  };
  loom_low_allocation_sidecar_t sidecar = {0};
  IREE_RETURN_IF_ERROR(loom_low_allocate_function(
      module, function.op, &allocation_options, pass->arena, &sidecar));

  loom_low_allocation_materialization_result_t result = {0};
  loom_low_allocation_materialization_options_t materialization_options = {
      .emitter = state && state->emit_spill_diagnostics
                     ? pass->diagnostic_emitter
                     : (iree_diagnostic_emitter_t){0},
  };
  IREE_RETURN_IF_ERROR(loom_low_allocation_materialize_spills(
      module, &sidecar, &materialization_options, pass->arena, &result));

  loom_pass_statistic_add(pass, LOOM_LOW_MATERIALIZE_ALLOCATION_STAT_SLOTS,
                          result.slot_count);
  loom_pass_statistic_add(pass, LOOM_LOW_MATERIALIZE_ALLOCATION_STAT_SPILLS,
                          result.spill_count);
  loom_pass_statistic_add(pass, LOOM_LOW_MATERIALIZE_ALLOCATION_STAT_RELOADS,
                          result.reload_count);
  if (result.slot_count > 0 || result.spill_count > 0 ||
      result.reload_count > 0) {
    loom_pass_mark_changed(pass);
  }
  return iree_ok_status();
}
