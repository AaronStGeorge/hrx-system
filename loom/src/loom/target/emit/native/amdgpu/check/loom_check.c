// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/check/loom_check.h"

#include "loom/codegen/low/frame.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/wait_packets.h"
#include "loom/target/arch/amdgpu/wait_plan.h"
#include "loom/target/emit/native/amdgpu/assembly.h"
#include "loom/tools/loom-check/low_emit.h"

typedef enum loom_amdgpu_loom_check_wait_mode_e {
  LOOM_AMDGPU_LOOM_CHECK_WAIT_MODE_AUTO = 0,
  LOOM_AMDGPU_LOOM_CHECK_WAIT_MODE_NONE = 1,
} loom_amdgpu_loom_check_wait_mode_t;

typedef struct loom_amdgpu_loom_check_emit_options_t {
  // Module-local target-low function symbol selected by the RUN line.
  iree_string_view_t function_symbol_name;
  // Candidate selection strategy used by low frame.
  loom_low_schedule_strategy_t schedule_strategy;
  // True once a strategy option has been parsed.
  bool has_schedule_strategy_option;
  // Wait-packet materialization mode for the emitted assembly.
  loom_amdgpu_loom_check_wait_mode_t wait_mode;
  // True once a waits option has been parsed.
  bool has_wait_mode_option;
  // Low allocation budget overrides parsed from target options.
  loom_low_allocation_budget_t
      allocation_budgets[LOOM_CHECK_LOW_EMIT_MAX_ALLOCATION_BUDGETS];
  // Number of entries in |allocation_budgets|.
  iree_host_size_t allocation_budget_count;
  // Fixed low allocation requests parsed from target options.
  loom_check_low_emit_fixed_value_spec_t allocation_fixed_value_specs
      [LOOM_CHECK_LOW_EMIT_MAX_ALLOCATION_FIXED_VALUES];
  // Number of entries in |allocation_fixed_value_specs|.
  iree_host_size_t allocation_fixed_value_spec_count;
} loom_amdgpu_loom_check_emit_options_t;

static bool loom_amdgpu_loom_check_emit_provider_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  (void)provider;
  return iree_string_view_equal(target_name, IREE_SV("amdgpu-assembly")) ||
         iree_string_view_equal(target_name, IREE_SV("amdgpu-asm"));
}

static iree_status_t loom_amdgpu_loom_check_parse_key_value_option(
    iree_string_view_t token, loom_amdgpu_loom_check_emit_options_t* options,
    bool* out_matched) {
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
    IREE_RETURN_IF_ERROR(loom_check_low_emit_parse_schedule_strategy(
        value, IREE_SV("AMDGPU assembly"), &options->schedule_strategy));
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
  return loom_check_low_emit_parse_allocation_option(
      token, IREE_SV("AMDGPU assembly"), options->allocation_budgets,
      IREE_ARRAYSIZE(options->allocation_budgets),
      &options->allocation_budget_count, options->allocation_fixed_value_specs,
      IREE_ARRAYSIZE(options->allocation_fixed_value_specs),
      &options->allocation_fixed_value_spec_count);
}

static iree_status_t loom_amdgpu_loom_check_parse_emit_options(
    const loom_check_emit_provider_request_t* request,
    loom_amdgpu_loom_check_emit_options_t* out_options) {
  *out_options = (loom_amdgpu_loom_check_emit_options_t){
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_LATENCY_HIDING,
      .wait_mode = LOOM_AMDGPU_LOOM_CHECK_WAIT_MODE_AUTO,
  };

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

static iree_status_t loom_amdgpu_loom_check_emit_assembly(
    const loom_low_emission_frame_t* frame,
    const loom_amdgpu_loom_check_emit_options_t* options,
    iree_string_builder_t* builder, iree_arena_allocator_t* arena) {
  if (options->wait_mode == LOOM_AMDGPU_LOOM_CHECK_WAIT_MODE_NONE) {
    return loom_amdgpu_emit_assembly_fragment(
        &frame->schedule, &frame->allocation, builder, arena);
  }

  loom_amdgpu_wait_plan_t wait_plan = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_plan_build(&frame->schedule, arena, &wait_plan));
  loom_amdgpu_wait_packet_plan_t wait_packets = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_packet_plan_build(&wait_plan, arena, &wait_packets));
  return loom_amdgpu_emit_assembly_fragment_with_wait_packets(
      &frame->schedule, &frame->allocation, &wait_packets, builder, arena);
}

static iree_status_t loom_amdgpu_loom_check_emit_provider_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  (void)provider;
  loom_amdgpu_loom_check_emit_options_t options;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_loom_check_parse_emit_options(request, &options));
  loom_low_emission_frame_t frame = {0};
  IREE_RETURN_IF_ERROR(loom_check_low_emit_packetize_function(
      request, options.function_symbol_name, options.schedule_strategy,
      options.allocation_budgets, options.allocation_budget_count,
      options.allocation_fixed_value_specs,
      options.allocation_fixed_value_spec_count, &frame));
  return loom_amdgpu_loom_check_emit_assembly(
      &frame, &options, &request->result->actual_output, request->case_arena);
}

static iree_status_t loom_amdgpu_loom_check_emit_provider_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(builder,
                                            "amdgpu-assembly, amdgpu-asm");
}

const loom_check_emit_provider_t loom_amdgpu_native_loom_check_emit_provider = {
    .name = IREE_SVL("amdgpu-native"),
    .match = loom_amdgpu_loom_check_emit_provider_matches,
    .execute = loom_amdgpu_loom_check_emit_provider_execute,
    .append_names = loom_amdgpu_loom_check_emit_provider_append_names,
};
