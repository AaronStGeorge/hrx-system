// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/loom_check.h"

#include "loom/codegen/low/packetization.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/wait_packets.h"
#include "loom/target/arch/amdgpu/wait_plan.h"
#include "loom/target/emit/native/amdgpu/assembly.h"

#define LOOM_AMDGPU_LOOM_CHECK_MAX_ALLOCATION_BUDGETS 8

typedef enum loom_amdgpu_loom_check_emit_format_e {
  LOOM_AMDGPU_LOOM_CHECK_EMIT_ASSEMBLY = 0,
  LOOM_AMDGPU_LOOM_CHECK_EMIT_ASSEMBLY_MNEMONICS = 1,
} loom_amdgpu_loom_check_emit_format_t;

typedef enum loom_amdgpu_loom_check_wait_mode_e {
  LOOM_AMDGPU_LOOM_CHECK_WAIT_MODE_AUTO = 0,
  LOOM_AMDGPU_LOOM_CHECK_WAIT_MODE_NONE = 1,
} loom_amdgpu_loom_check_wait_mode_t;

typedef struct loom_amdgpu_loom_check_emit_options_t {
  // Output format selected by the emit target name.
  loom_amdgpu_loom_check_emit_format_t format;
  // Module-local low.func.def symbol selected by the RUN line.
  iree_string_view_t function_symbol_name;
  // Candidate selection strategy used by low packetization.
  loom_low_schedule_strategy_t schedule_strategy;
  // True once a strategy option has been parsed.
  bool has_schedule_strategy_option;
  // Wait-packet materialization mode for the emitted assembly.
  loom_amdgpu_loom_check_wait_mode_t wait_mode;
  // True once a waits option has been parsed.
  bool has_wait_mode_option;
  // Low allocation budget overrides parsed from target options.
  loom_low_allocation_budget_t
      allocation_budgets[LOOM_AMDGPU_LOOM_CHECK_MAX_ALLOCATION_BUDGETS];
  // Number of entries in |allocation_budgets|.
  iree_host_size_t allocation_budget_count;
} loom_amdgpu_loom_check_emit_options_t;

static bool loom_amdgpu_loom_check_emit_provider_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  (void)provider;
  return iree_string_view_equal(target_name, IREE_SV("amdgpu-assembly")) ||
         iree_string_view_equal(target_name, IREE_SV("amdgpu-asm")) ||
         iree_string_view_equal(target_name,
                                IREE_SV("amdgpu-assembly-mnemonics")) ||
         iree_string_view_equal(target_name, IREE_SV("amdgpu-asm-mnemonics"));
}

static iree_status_t loom_amdgpu_loom_check_parse_emit_format(
    iree_string_view_t target_name,
    loom_amdgpu_loom_check_emit_format_t* out_format) {
  IREE_ASSERT_ARGUMENT(out_format);
  if (iree_string_view_equal(target_name, IREE_SV("amdgpu-assembly")) ||
      iree_string_view_equal(target_name, IREE_SV("amdgpu-asm"))) {
    *out_format = LOOM_AMDGPU_LOOM_CHECK_EMIT_ASSEMBLY;
    return iree_ok_status();
  }
  if (iree_string_view_equal(target_name,
                             IREE_SV("amdgpu-assembly-mnemonics")) ||
      iree_string_view_equal(target_name, IREE_SV("amdgpu-asm-mnemonics"))) {
    *out_format = LOOM_AMDGPU_LOOM_CHECK_EMIT_ASSEMBLY_MNEMONICS;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown AMDGPU native emit target '%.*s'",
                          (int)target_name.size, target_name.data);
}

static iree_status_t loom_amdgpu_loom_check_parse_budget(
    iree_string_view_t token, loom_amdgpu_loom_check_emit_options_t* options) {
  if (options->allocation_budget_count >=
      LOOM_AMDGPU_LOOM_CHECK_MAX_ALLOCATION_BUDGETS) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many AMDGPU assembly allocation budgets");
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
        "AMDGPU assembly allocation budget must have the form "
        "<register-class>=<units>");
  }
  uint32_t max_units = 0;
  if (!iree_string_view_atoi_uint32(budget_text, &max_units)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid AMDGPU assembly allocation budget '%.*s'",
                            (int)budget_text.size, budget_text.data);
  }
  options->allocation_budgets[options->allocation_budget_count++] =
      (loom_low_allocation_budget_t){
          .register_class = register_class,
          .max_units = max_units,
      };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_loom_check_parse_key_value_option(
    iree_string_view_t token, loom_amdgpu_loom_check_emit_options_t* options,
    bool* out_matched) {
  IREE_ASSERT_ARGUMENT(out_matched);
  *out_matched = false;
  iree_string_view_t name = iree_string_view_empty();
  iree_string_view_t value = iree_string_view_empty();
  iree_string_view_split(token, '=', &name, &value);
  name = iree_string_view_trim(name);
  value = iree_string_view_trim(value);
  if (iree_string_view_equal(name, IREE_SV("strategy"))) {
    if (options->has_schedule_strategy_option) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate AMDGPU assembly option 'strategy'");
    }
    if (iree_string_view_equal(value, IREE_SV("source"))) {
      options->schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY;
    } else if (iree_string_view_equal(value, IREE_SV("pressure"))) {
      options->schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE;
    } else {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU assembly option 'strategy' expected 'source' or 'pressure', "
          "got '%.*s'",
          (int)value.size, value.data);
    }
    options->has_schedule_strategy_option = true;
    *out_matched = true;
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("waits"))) {
    if (options->has_wait_mode_option) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate AMDGPU assembly option 'waits'");
    }
    if (iree_string_view_equal(value, IREE_SV("auto"))) {
      options->wait_mode = LOOM_AMDGPU_LOOM_CHECK_WAIT_MODE_AUTO;
    } else if (iree_string_view_equal(value, IREE_SV("none"))) {
      options->wait_mode = LOOM_AMDGPU_LOOM_CHECK_WAIT_MODE_NONE;
    } else {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU assembly option 'waits' expected 'auto' or 'none', got "
          "'%.*s'",
          (int)value.size, value.data);
    }
    options->has_wait_mode_option = true;
    *out_matched = true;
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_loom_check_parse_option(
    iree_string_view_t token, loom_amdgpu_loom_check_emit_options_t* options) {
  bool matched = false;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_loom_check_parse_key_value_option(token, options, &matched));
  if (matched) {
    return iree_ok_status();
  }
  return loom_amdgpu_loom_check_parse_budget(token, options);
}

static iree_status_t loom_amdgpu_loom_check_parse_emit_options(
    const loom_check_emit_provider_request_t* request,
    loom_amdgpu_loom_check_emit_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  *out_options = (loom_amdgpu_loom_check_emit_options_t){
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
      .wait_mode = LOOM_AMDGPU_LOOM_CHECK_WAIT_MODE_AUTO,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_loom_check_parse_emit_format(
      request->target_name, &out_options->format));

  iree_string_view_t symbol_name = iree_string_view_empty();
  iree_string_view_t option_text = iree_string_view_empty();
  iree_string_view_split(request->target_options, ' ', &symbol_name,
                         &option_text);
  symbol_name = iree_string_view_trim(symbol_name);
  option_text = iree_string_view_trim(option_text);
  if (!iree_string_view_starts_with(symbol_name, IREE_SV("@"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU assembly requires a low function symbol "
                            "name");
  }
  out_options->function_symbol_name =
      iree_string_view_substr(symbol_name, 1, IREE_HOST_SIZE_MAX);
  if (iree_string_view_is_empty(out_options->function_symbol_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU assembly low function symbol name is "
                            "required");
  }

  while (!iree_string_view_is_empty(option_text)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_t remaining = iree_string_view_empty();
    iree_string_view_split(option_text, ' ', &token, &remaining);
    token = iree_string_view_trim(token);
    if (!iree_string_view_is_empty(token)) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_loom_check_parse_option(token, out_options));
    }
    option_text = iree_string_view_trim(remaining);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_loom_check_find_low_func_def(
    loom_module_t* module, iree_string_view_t symbol_name,
    loom_op_t** out_low_function) {
  IREE_ASSERT_ARGUMENT(out_low_function);
  *out_low_function = NULL;
  if (module == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU assembly module is required");
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

static iree_string_view_t loom_amdgpu_loom_check_consume_line(
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

static bool loom_amdgpu_loom_check_assembly_line_is_label(
    iree_string_view_t line) {
  return iree_string_view_starts_with(line, IREE_SV(".")) &&
         iree_string_view_ends_with(line, IREE_SV(":"));
}

static iree_status_t loom_amdgpu_loom_check_write_assembly_mnemonics(
    iree_string_view_t assembly, iree_string_builder_t* builder) {
  iree_string_view_t remaining = assembly;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t line =
        iree_string_view_trim(loom_amdgpu_loom_check_consume_line(&remaining));
    if (iree_string_view_is_empty(line) ||
        loom_amdgpu_loom_check_assembly_line_is_label(line)) {
      continue;
    }
    iree_string_view_t mnemonic = iree_string_view_empty();
    iree_string_view_split(line, ' ', &mnemonic, NULL);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(builder, mnemonic));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_loom_check_emit_assembly(
    const loom_low_packetization_t* packetization,
    const loom_amdgpu_loom_check_emit_options_t* options,
    iree_string_builder_t* builder, iree_arena_allocator_t* arena) {
  if (options->wait_mode == LOOM_AMDGPU_LOOM_CHECK_WAIT_MODE_NONE) {
    return loom_amdgpu_emit_assembly_fragment(
        &packetization->schedule, &packetization->allocation, builder);
  }

  loom_amdgpu_wait_plan_t wait_plan = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_plan_build(&packetization->schedule, arena, &wait_plan));
  loom_amdgpu_wait_packet_plan_t wait_packets = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_packet_plan_build(&wait_plan, arena, &wait_packets));
  return loom_amdgpu_emit_assembly_fragment_with_wait_packets(
      &packetization->schedule, &packetization->allocation, &wait_packets,
      builder);
}

static iree_status_t loom_amdgpu_loom_check_emit_provider_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  (void)provider;
  loom_amdgpu_loom_check_emit_options_t options;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_loom_check_parse_emit_options(request, &options));
  if (request->low_registry == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU assembly low descriptor registry is "
                            "required");
  }

  loom_op_t* low_function = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_loom_check_find_low_func_def(
      request->module, options.function_symbol_name, &low_function));

  loom_low_packetization_options_t packetization_options = {
      .descriptor_registry = &request->low_registry->registry,
      .schedule_strategy = options.schedule_strategy,
      .allocation_budgets = options.allocation_budgets,
      .allocation_budget_count = options.allocation_budget_count,
  };
  loom_low_packetization_t packetization = {0};
  IREE_RETURN_IF_ERROR(loom_low_packetize_function(
      request->module, low_function, &packetization_options,
      request->case_arena, &packetization));

  if (options.format == LOOM_AMDGPU_LOOM_CHECK_EMIT_ASSEMBLY) {
    return loom_amdgpu_loom_check_emit_assembly(&packetization, &options,
                                                &request->result->actual_output,
                                                request->case_arena);
  }

  iree_string_builder_t assembly_builder;
  iree_string_builder_initialize(request->host_allocator, &assembly_builder);
  iree_status_t status = loom_amdgpu_loom_check_emit_assembly(
      &packetization, &options, &assembly_builder, request->case_arena);
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_loom_check_write_assembly_mnemonics(
        iree_string_builder_view(&assembly_builder),
        &request->result->actual_output);
  }
  iree_string_builder_deinitialize(&assembly_builder);
  return status;
}

static iree_status_t loom_amdgpu_loom_check_emit_provider_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(
      builder,
      "amdgpu-assembly, amdgpu-asm, amdgpu-assembly-mnemonics, "
      "amdgpu-asm-mnemonics");
}

const loom_check_emit_provider_t loom_amdgpu_native_loom_check_emit_provider = {
    .name = IREE_SVL("amdgpu-native"),
    .match = loom_amdgpu_loom_check_emit_provider_matches,
    .execute = loom_amdgpu_loom_check_emit_provider_execute,
    .append_names = loom_amdgpu_loom_check_emit_provider_append_names,
};
