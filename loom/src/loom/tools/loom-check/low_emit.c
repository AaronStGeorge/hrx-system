// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/low_emit.h"

#include <inttypes.h>

#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"

iree_status_t loom_check_low_emit_parse_schedule_strategy(
    iree_string_view_t value, iree_string_view_t option_scope,
    loom_low_schedule_strategy_t* out_strategy) {
  IREE_ASSERT_ARGUMENT(out_strategy);
  if (iree_string_view_equal(value, IREE_SV("source"))) {
    *out_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("pressure"))) {
    *out_strategy = LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "%.*s option 'strategy' expected 'source' or 'pressure', got '%.*s'",
      (int)option_scope.size, option_scope.data, (int)value.size, value.data);
}

iree_status_t loom_check_low_emit_parse_allocation_budget(
    iree_string_view_t token, iree_string_view_t option_scope,
    loom_low_allocation_budget_t* budgets, iree_host_size_t budget_capacity,
    iree_host_size_t* budget_count) {
  IREE_ASSERT_ARGUMENT(budget_count);
  if (*budget_count >= budget_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many %.*s allocation budgets",
                            (int)option_scope.size, option_scope.data);
  }
  iree_string_view_t register_class = iree_string_view_empty();
  iree_string_view_t budget_text = iree_string_view_empty();
  iree_string_view_split(token, '=', &register_class, &budget_text);
  register_class = iree_string_view_trim(register_class);
  budget_text = iree_string_view_trim(budget_text);
  if (iree_string_view_is_empty(register_class) ||
      iree_string_view_is_empty(budget_text)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s allocation budget must have the form <register-class>=<units>",
        (int)option_scope.size, option_scope.data);
  }
  uint32_t max_units = 0;
  if (!iree_string_view_atoi_uint32(budget_text, &max_units)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid %.*s allocation budget '%.*s'",
                            (int)option_scope.size, option_scope.data,
                            (int)budget_text.size, budget_text.data);
  }
  budgets[(*budget_count)++] = (loom_low_allocation_budget_t){
      .register_class = register_class,
      .max_units = max_units,
  };
  return iree_ok_status();
}

iree_status_t loom_check_low_emit_find_low_func_def(
    loom_module_t* module, iree_string_view_t symbol_name,
    loom_op_t** out_low_function) {
  IREE_ASSERT_ARGUMENT(out_low_function);
  *out_low_function = NULL;
  if (module == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low emit module is required");
  }
  loom_string_id_t name_id = loom_module_lookup_string(module, symbol_name);
  if (name_id == LOOM_STRING_ID_INVALID) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "unknown low function '@%.*s'",
                            (int)symbol_name.size, symbol_name.data);
  }
  uint16_t symbol_id = loom_module_find_symbol(module, name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "unknown low function '@%.*s'",
                            (int)symbol_name.size, symbol_name.data);
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
  if (!loom_low_func_def_isa(symbol->defining_op) ||
      !loom_low_func_def_body(symbol->defining_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol '@%.*s' does not name a low function body",
                            (int)symbol_name.size, symbol_name.data);
  }
  *out_low_function = symbol->defining_op;
  return iree_ok_status();
}

iree_status_t loom_check_low_emit_packetize_function(
    const loom_check_emit_provider_request_t* request,
    iree_string_view_t function_symbol_name,
    loom_low_schedule_strategy_t schedule_strategy,
    const loom_low_allocation_budget_t* allocation_budgets,
    iree_host_size_t allocation_budget_count,
    loom_low_packetization_t* out_packetization) {
  IREE_ASSERT_ARGUMENT(out_packetization);
  if (request == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low emit provider request is required");
  }
  if (request->low_registry == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low emit descriptor registry is required");
  }

  loom_op_t* low_function = NULL;
  IREE_RETURN_IF_ERROR(loom_check_low_emit_find_low_func_def(
      request->module, function_symbol_name, &low_function));

  loom_low_packetization_options_t packetization_options = {
      .descriptor_registry = &request->low_registry->registry,
      .schedule_strategy = schedule_strategy,
      .allocation_budgets = allocation_budgets,
      .allocation_budget_count = allocation_budget_count,
  };
  *out_packetization = (loom_low_packetization_t){0};
  return loom_low_packetize_function(request->module, low_function,
                                     &packetization_options,
                                     request->case_arena, out_packetization);
}

static iree_string_view_t loom_check_low_emit_consume_line(
    iree_string_view_t* remaining) {
  iree_host_size_t newline_position =
      iree_string_view_find(*remaining, IREE_SV("\n"), 0);
  if (newline_position == IREE_STRING_VIEW_NPOS) {
    iree_string_view_t line = *remaining;
    *remaining = iree_string_view_empty();
    return line;
  }
  iree_string_view_t line =
      iree_string_view_substr(*remaining, 0, newline_position);
  *remaining = iree_string_view_substr(*remaining, newline_position + 1,
                                       IREE_HOST_SIZE_MAX);
  return line;
}

static bool loom_check_low_emit_assembly_line_is_label(
    iree_string_view_t line) {
  return iree_string_view_starts_with(line, IREE_SV(".")) &&
         iree_string_view_ends_with(line, IREE_SV(":"));
}

iree_status_t loom_check_low_emit_write_assembly_mnemonics(
    iree_string_view_t assembly, iree_string_builder_t* builder) {
  iree_string_view_t remaining = assembly;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t line =
        iree_string_view_trim(loom_check_low_emit_consume_line(&remaining));
    if (iree_string_view_is_empty(line) ||
        loom_check_low_emit_assembly_line_is_label(line)) {
      continue;
    }
    iree_string_view_t mnemonic = iree_string_view_empty();
    iree_string_view_split(line, ' ', &mnemonic, NULL);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(builder, mnemonic));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  }
  return iree_ok_status();
}
