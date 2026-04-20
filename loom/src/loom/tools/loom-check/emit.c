// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/vec_stream.h"
#include "loom/analysis/liveness.h"
#include "loom/analysis/liveness_json.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/allocation_json.h"
#include "loom/codegen/low/packet_json.h"
#include "loom/codegen/low/packetization.h"
#include "loom/codegen/low/schedule.h"
#include "loom/codegen/low/schedule_json.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/codegen/low/verify.h"
#include "loom/format/text/parser.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/emit/llvmir/bitcode_writer.h"
#include "loom/target/emit/llvmir/legality.h"
#include "loom/target/emit/llvmir/lower.h"
#include "loom/target/emit/llvmir/target_registry.h"
#include "loom/target/emit/llvmir/text_writer.h"
#include "loom/target/emit/llvmir/verify.h"
#include "loom/target/ir_records.h"
#include "loom/target/presets.h"
#include "loom/target/tool/llvm.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/execute.h"
#include "loom/util/stream.h"
#include "loom/verify/verify.h"

typedef enum loom_check_emit_format_e {
  LOOM_CHECK_EMIT_LLVMIR_TEXT = 0,
  LOOM_CHECK_EMIT_LLVMIR_BODY_TEXT = 1,
  LOOM_CHECK_EMIT_LLVMIR_BITCODE_DISASSEMBLY = 2,
  LOOM_CHECK_EMIT_LLVMIR_OBJECT = 3,
  LOOM_CHECK_EMIT_LLVMIR_ASSEMBLY_MNEMONICS = 4,
  LOOM_CHECK_EMIT_LIVENESS_JSON = 5,
  LOOM_CHECK_EMIT_LOW_SCHEDULE_JSON = 6,
  LOOM_CHECK_EMIT_LOW_DESCRIPTOR_MANIFEST = 7,
  LOOM_CHECK_EMIT_TARGET_LOW_REGISTRY_MANIFEST = 8,
  LOOM_CHECK_EMIT_LOW_ALLOCATION_JSON = 9,
  LOOM_CHECK_EMIT_LOW_PACKET_JSON = 10,
} loom_check_emit_format_t;

#define LOOM_CHECK_LOW_ALLOCATION_MAX_BUDGETS 8

typedef struct loom_check_emit_request_t {
  // Serialized target form to produce before comparison.
  loom_check_emit_format_t format;
  // Canonical/user-facing emit target name used in diagnostics.
  iree_string_view_t emit_target_name;
  // Generic target bundle used by legality and LLVMIR profile derivation.
  const loom_target_bundle_t* target_bundle;
  // Module-local target.bundle symbol name to materialize after parsing.
  iree_string_view_t target_bundle_symbol_name;
  // Module-local function symbol name used by analysis dumps.
  iree_string_view_t analysis_symbol_name;
  // Low descriptor set used by descriptor-manifest dumps.
  const loom_low_descriptor_set_t* low_descriptor_set;
  // Low descriptor-set key used by descriptor-manifest dumps.
  iree_string_view_t low_descriptor_set_key;
  // Low allocation budget overrides parsed from the RUN line.
  loom_low_allocation_budget_t
      low_allocation_budgets[LOOM_CHECK_LOW_ALLOCATION_MAX_BUDGETS];
  // Number of entries in |low_allocation_budgets|.
  iree_host_size_t low_allocation_budget_count;
  // Low allocation diagnostic feedback requested by the RUN line.
  loom_low_allocation_diagnostic_flags_t low_allocation_diagnostic_flags;
  // True once a low allocation diagnostics option has been parsed.
  bool has_low_allocation_diagnostics_option;
  // True when the analysis pass should run without printing its JSON sidecar.
  bool suppress_output;
  // True once an output option has been parsed.
  bool has_output_option;
  // Low scheduler diagnostic feedback requested by the RUN line.
  loom_low_schedule_diagnostic_flags_t low_schedule_diagnostic_flags;
  // True once a low scheduler diagnostics option has been parsed.
  bool has_low_schedule_diagnostics_option;
  // Low scheduler candidate-selection strategy requested by the RUN line.
  loom_low_schedule_strategy_t low_schedule_strategy;
  // True once a low scheduler strategy option has been parsed.
  bool has_low_schedule_strategy_option;
} loom_check_emit_request_t;

typedef struct loom_check_emit_byte_buffer_t {
  // Allocated byte storage owned by this buffer.
  uint8_t* data;
  // Number of valid bytes in |data|.
  iree_host_size_t length;
} loom_check_emit_byte_buffer_t;

static iree_status_t loom_check_emit_parse_low_allocation_budget(
    iree_string_view_t token, loom_check_emit_request_t* request) {
  if (request->low_allocation_budget_count >=
      LOOM_CHECK_LOW_ALLOCATION_MAX_BUDGETS) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many low allocation budget overrides");
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
        "low allocation budget must have the form <register-class>=<units>");
  }
  uint32_t max_units = 0;
  if (!iree_string_view_atoi_uint32(budget_text, &max_units)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid low allocation budget '%.*s'",
                            (int)budget_text.size, budget_text.data);
  }
  request->low_allocation_budgets[request->low_allocation_budget_count++] =
      (loom_low_allocation_budget_t){
          .register_class = register_class,
          .max_units = max_units,
      };
  return iree_ok_status();
}

static iree_status_t loom_check_emit_parse_output_option(
    iree_string_view_t value, loom_check_emit_request_t* request) {
  if (request->has_output_option) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate emit option 'output'");
  }
  if (iree_string_view_equal(value, IREE_SV("json"))) {
    request->suppress_output = false;
  } else if (iree_string_view_equal(value, IREE_SV("none"))) {
    request->suppress_output = true;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "emit option 'output' expected 'json' or 'none', got '%.*s'",
        (int)value.size, value.data);
  }
  request->has_output_option = true;
  return iree_ok_status();
}

static iree_status_t loom_check_emit_parse_low_allocation_option(
    iree_string_view_t token, loom_check_emit_request_t* request) {
  iree_string_view_t name = iree_string_view_empty();
  iree_string_view_t value = iree_string_view_empty();
  iree_string_view_split(token, '=', &name, &value);
  name = iree_string_view_trim(name);
  value = iree_string_view_trim(value);
  if (iree_string_view_equal(name, IREE_SV("output"))) {
    return loom_check_emit_parse_output_option(value, request);
  }
  if (!iree_string_view_equal(name, IREE_SV("diagnostics"))) {
    return loom_check_emit_parse_low_allocation_budget(token, request);
  }
  if (request->has_low_allocation_diagnostics_option) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "duplicate low-allocation-json option 'diagnostics'");
  }
  if (iree_string_view_equal(value, IREE_SV("none"))) {
    request->low_allocation_diagnostic_flags = 0;
  } else if (iree_string_view_equal(value, IREE_SV("predicted-spills"))) {
    request->low_allocation_diagnostic_flags =
        LOOM_LOW_ALLOCATION_DIAGNOSTIC_PREDICTED_SPILLS;
  } else if (iree_string_view_equal(value, IREE_SV("copy-decisions"))) {
    request->low_allocation_diagnostic_flags =
        LOOM_LOW_ALLOCATION_DIAGNOSTIC_COPY_DECISIONS;
  } else if (iree_string_view_equal(value, IREE_SV("all"))) {
    request->low_allocation_diagnostic_flags =
        LOOM_LOW_ALLOCATION_DIAGNOSTIC_PREDICTED_SPILLS |
        LOOM_LOW_ALLOCATION_DIAGNOSTIC_COPY_DECISIONS;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low-allocation-json option 'diagnostics' expected 'none' or "
        "'predicted-spills', 'copy-decisions', or 'all', got '%.*s'",
        (int)value.size, value.data);
  }
  request->has_low_allocation_diagnostics_option = true;
  return iree_ok_status();
}

static iree_status_t loom_check_emit_parse_low_allocation_options(
    iree_string_view_t text, loom_check_emit_request_t* request) {
  text = iree_string_view_trim(text);
  while (!iree_string_view_is_empty(text)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_t remaining = iree_string_view_empty();
    iree_string_view_split(text, ' ', &token, &remaining);
    token = iree_string_view_trim(token);
    if (!iree_string_view_is_empty(token)) {
      IREE_RETURN_IF_ERROR(
          loom_check_emit_parse_low_allocation_option(token, request));
    }
    text = iree_string_view_trim(remaining);
  }
  return iree_ok_status();
}

static iree_status_t loom_check_emit_parse_low_schedule_option(
    iree_string_view_t token, loom_check_emit_request_t* request) {
  iree_string_view_t name = iree_string_view_empty();
  iree_string_view_t value = iree_string_view_empty();
  iree_string_view_split(token, '=', &name, &value);
  name = iree_string_view_trim(name);
  value = iree_string_view_trim(value);
  if (iree_string_view_equal(name, IREE_SV("output"))) {
    return loom_check_emit_parse_output_option(value, request);
  }
  if (iree_string_view_equal(name, IREE_SV("strategy"))) {
    if (request->has_low_schedule_strategy_option) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate low-schedule-json option 'strategy'");
    }
    if (iree_string_view_equal(value, IREE_SV("source"))) {
      request->low_schedule_strategy =
          LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY;
    } else if (iree_string_view_equal(value, IREE_SV("pressure"))) {
      request->low_schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE;
    } else {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low-schedule-json option 'strategy' expected 'source' or "
          "'pressure', got '%.*s'",
          (int)value.size, value.data);
    }
    request->has_low_schedule_strategy_option = true;
    return iree_ok_status();
  }
  if (!iree_string_view_equal(name, IREE_SV("diagnostics"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown low-schedule-json option '%.*s'",
                            (int)name.size, name.data);
  }
  if (request->has_low_schedule_diagnostics_option) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate low-schedule-json option 'diagnostics'");
  }
  if (iree_string_view_equal(value, IREE_SV("none"))) {
    request->low_schedule_diagnostic_flags = 0;
  } else if (iree_string_view_equal(value, IREE_SV("pressure"))) {
    request->low_schedule_diagnostic_flags =
        LOOM_LOW_SCHEDULE_DIAGNOSTIC_PRESSURE_PEAKS;
  } else if (iree_string_view_equal(value, IREE_SV("resources"))) {
    request->low_schedule_diagnostic_flags =
        LOOM_LOW_SCHEDULE_DIAGNOSTIC_RESOURCE_BOTTLENECKS;
  } else if (iree_string_view_equal(value, IREE_SV("hazards"))) {
    request->low_schedule_diagnostic_flags =
        LOOM_LOW_SCHEDULE_DIAGNOSTIC_HAZARD_GAPS;
  } else if (iree_string_view_equal(value, IREE_SV("candidates"))) {
    request->low_schedule_diagnostic_flags =
        LOOM_LOW_SCHEDULE_DIAGNOSTIC_CANDIDATE_DECISIONS;
  } else if (iree_string_view_equal(value, IREE_SV("model"))) {
    request->low_schedule_diagnostic_flags =
        LOOM_LOW_SCHEDULE_DIAGNOSTIC_MODEL_QUALITY;
  } else if (iree_string_view_equal(value, IREE_SV("all"))) {
    request->low_schedule_diagnostic_flags =
        LOOM_LOW_SCHEDULE_DIAGNOSTIC_PRESSURE_PEAKS |
        LOOM_LOW_SCHEDULE_DIAGNOSTIC_RESOURCE_BOTTLENECKS |
        LOOM_LOW_SCHEDULE_DIAGNOSTIC_HAZARD_GAPS |
        LOOM_LOW_SCHEDULE_DIAGNOSTIC_CANDIDATE_DECISIONS |
        LOOM_LOW_SCHEDULE_DIAGNOSTIC_MODEL_QUALITY;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low-schedule-json option 'diagnostics' expected 'none', 'pressure', "
        "'resources', 'hazards', 'candidates', 'model', or 'all', got '%.*s'",
        (int)value.size, value.data);
  }
  request->has_low_schedule_diagnostics_option = true;
  return iree_ok_status();
}

static iree_status_t loom_check_emit_parse_low_schedule_options(
    iree_string_view_t text, loom_check_emit_request_t* request) {
  text = iree_string_view_trim(text);
  while (!iree_string_view_is_empty(text)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_t remaining = iree_string_view_empty();
    iree_string_view_split(text, ' ', &token, &remaining);
    token = iree_string_view_trim(token);
    if (!iree_string_view_is_empty(token)) {
      IREE_RETURN_IF_ERROR(
          loom_check_emit_parse_low_schedule_option(token, request));
    }
    text = iree_string_view_trim(remaining);
  }
  return iree_ok_status();
}

static iree_status_t loom_check_emit_parse_low_packet_option(
    iree_string_view_t token, loom_check_emit_request_t* request) {
  iree_string_view_t name = iree_string_view_empty();
  iree_string_view_t value = iree_string_view_empty();
  iree_string_view_split(token, '=', &name, &value);
  name = iree_string_view_trim(name);
  value = iree_string_view_trim(value);
  if (iree_string_view_equal(name, IREE_SV("output"))) {
    return loom_check_emit_parse_output_option(value, request);
  }
  if (iree_string_view_equal(name, IREE_SV("strategy"))) {
    if (request->has_low_schedule_strategy_option) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate low-packet-json option 'strategy'");
    }
    if (iree_string_view_equal(value, IREE_SV("source"))) {
      request->low_schedule_strategy =
          LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY;
    } else if (iree_string_view_equal(value, IREE_SV("pressure"))) {
      request->low_schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE;
    } else {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low-packet-json option 'strategy' expected 'source' or 'pressure', "
          "got '%.*s'",
          (int)value.size, value.data);
    }
    request->has_low_schedule_strategy_option = true;
    return iree_ok_status();
  }
  return loom_check_emit_parse_low_allocation_budget(token, request);
}

static iree_status_t loom_check_emit_parse_low_packet_options(
    iree_string_view_t text, loom_check_emit_request_t* request) {
  text = iree_string_view_trim(text);
  while (!iree_string_view_is_empty(text)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_t remaining = iree_string_view_empty();
    iree_string_view_split(text, ' ', &token, &remaining);
    token = iree_string_view_trim(token);
    if (!iree_string_view_is_empty(token)) {
      IREE_RETURN_IF_ERROR(
          loom_check_emit_parse_low_packet_option(token, request));
    }
    text = iree_string_view_trim(remaining);
  }
  return iree_ok_status();
}

static iree_status_t loom_check_emit_parse_request(
    iree_string_view_t emit_target, loom_check_emit_request_t* out_request) {
  *out_request = (loom_check_emit_request_t){
      .format = LOOM_CHECK_EMIT_LLVMIR_TEXT,
      .emit_target_name = IREE_SV("emit"),
      .target_bundle = NULL,
      .target_bundle_symbol_name = iree_string_view_empty(),
      .analysis_symbol_name = iree_string_view_empty(),
      .low_descriptor_set = NULL,
      .low_descriptor_set_key = iree_string_view_empty(),
      .low_allocation_budget_count = 0,
      .low_allocation_diagnostic_flags = 0,
      .has_low_allocation_diagnostics_option = false,
      .suppress_output = false,
      .has_output_option = false,
      .low_schedule_diagnostic_flags = 0,
      .has_low_schedule_diagnostics_option = false,
      .low_schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
      .has_low_schedule_strategy_option = false,
  };
  emit_target = iree_string_view_trim(emit_target);
  iree_string_view_t target_name = iree_string_view_empty();
  iree_string_view_t profile_name = iree_string_view_empty();
  iree_string_view_split(emit_target, ' ', &target_name, &profile_name);
  target_name = iree_string_view_trim(target_name);
  profile_name = iree_string_view_trim(profile_name);
  if (!iree_string_view_is_empty(target_name)) {
    out_request->emit_target_name = target_name;
  }

  if (iree_string_view_equal(target_name, IREE_SV("llvmir")) ||
      iree_string_view_equal(target_name, IREE_SV("llvmir-text"))) {
    out_request->format = LOOM_CHECK_EMIT_LLVMIR_TEXT;
  } else if (iree_string_view_equal(target_name, IREE_SV("llvmir-body")) ||
             iree_string_view_equal(target_name, IREE_SV("llvmir-text-body"))) {
    out_request->format = LOOM_CHECK_EMIT_LLVMIR_BODY_TEXT;
  } else if (iree_string_view_equal(target_name, IREE_SV("llvmir-bitcode"))) {
    out_request->format = LOOM_CHECK_EMIT_LLVMIR_BITCODE_DISASSEMBLY;
  } else if (iree_string_view_equal(target_name, IREE_SV("llvmir-object"))) {
    out_request->format = LOOM_CHECK_EMIT_LLVMIR_OBJECT;
  } else if (iree_string_view_equal(target_name,
                                    IREE_SV("llvmir-assembly-mnemonics")) ||
             iree_string_view_equal(target_name,
                                    IREE_SV("llvmir-asm-mnemonics"))) {
    out_request->format = LOOM_CHECK_EMIT_LLVMIR_ASSEMBLY_MNEMONICS;
  } else if (iree_string_view_equal(target_name, IREE_SV("liveness-json")) ||
             iree_string_view_equal(target_name, IREE_SV("liveness"))) {
    if (!iree_string_view_starts_with(profile_name, IREE_SV("@"))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "liveness-json requires a function symbol name");
    }
    out_request->analysis_symbol_name =
        iree_string_view_substr(profile_name, 1, IREE_HOST_SIZE_MAX);
    if (iree_string_view_is_empty(out_request->analysis_symbol_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "function symbol name is required");
    }
    out_request->format = LOOM_CHECK_EMIT_LIVENESS_JSON;
    return iree_ok_status();
  } else if (iree_string_view_equal(target_name,
                                    IREE_SV("low-schedule-json")) ||
             iree_string_view_equal(target_name, IREE_SV("low-schedule"))) {
    iree_string_view_t symbol_name = iree_string_view_empty();
    iree_string_view_t option_text = iree_string_view_empty();
    iree_string_view_split(profile_name, ' ', &symbol_name, &option_text);
    symbol_name = iree_string_view_trim(symbol_name);
    option_text = iree_string_view_trim(option_text);
    if (!iree_string_view_starts_with(symbol_name, IREE_SV("@"))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low-schedule-json requires a low function "
                              "symbol name");
    }
    out_request->analysis_symbol_name =
        iree_string_view_substr(symbol_name, 1, IREE_HOST_SIZE_MAX);
    if (iree_string_view_is_empty(out_request->analysis_symbol_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low function symbol name is required");
    }
    IREE_RETURN_IF_ERROR(
        loom_check_emit_parse_low_schedule_options(option_text, out_request));
    out_request->format = LOOM_CHECK_EMIT_LOW_SCHEDULE_JSON;
    return iree_ok_status();
  } else if (iree_string_view_equal(target_name,
                                    IREE_SV("low-allocation-json")) ||
             iree_string_view_equal(target_name, IREE_SV("low-allocation"))) {
    iree_string_view_t symbol_name = iree_string_view_empty();
    iree_string_view_t option_text = iree_string_view_empty();
    iree_string_view_split(profile_name, ' ', &symbol_name, &option_text);
    symbol_name = iree_string_view_trim(symbol_name);
    option_text = iree_string_view_trim(option_text);
    if (!iree_string_view_starts_with(symbol_name, IREE_SV("@"))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low-allocation-json requires a low function "
                              "symbol name");
    }
    out_request->analysis_symbol_name =
        iree_string_view_substr(symbol_name, 1, IREE_HOST_SIZE_MAX);
    if (iree_string_view_is_empty(out_request->analysis_symbol_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low function symbol name is required");
    }
    IREE_RETURN_IF_ERROR(
        loom_check_emit_parse_low_allocation_options(option_text, out_request));
    out_request->format = LOOM_CHECK_EMIT_LOW_ALLOCATION_JSON;
    return iree_ok_status();
  } else if (iree_string_view_equal(target_name, IREE_SV("low-packet-json")) ||
             iree_string_view_equal(target_name, IREE_SV("low-packet"))) {
    iree_string_view_t symbol_name = iree_string_view_empty();
    iree_string_view_t option_text = iree_string_view_empty();
    iree_string_view_split(profile_name, ' ', &symbol_name, &option_text);
    symbol_name = iree_string_view_trim(symbol_name);
    option_text = iree_string_view_trim(option_text);
    if (!iree_string_view_starts_with(symbol_name, IREE_SV("@"))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low-packet-json requires a low function symbol "
                              "name");
    }
    out_request->analysis_symbol_name =
        iree_string_view_substr(symbol_name, 1, IREE_HOST_SIZE_MAX);
    if (iree_string_view_is_empty(out_request->analysis_symbol_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low function symbol name is required");
    }
    IREE_RETURN_IF_ERROR(
        loom_check_emit_parse_low_packet_options(option_text, out_request));
    out_request->format = LOOM_CHECK_EMIT_LOW_PACKET_JSON;
    return iree_ok_status();
  } else if (iree_string_view_equal(target_name,
                                    IREE_SV("low-descriptor-manifest")) ||
             iree_string_view_equal(target_name,
                                    IREE_SV("low-descriptor-json"))) {
    if (iree_string_view_is_empty(profile_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low descriptor set key is required");
    }
    out_request->format = LOOM_CHECK_EMIT_LOW_DESCRIPTOR_MANIFEST;
    out_request->low_descriptor_set_key = profile_name;
    return iree_ok_status();
  } else if (iree_string_view_equal(target_name,
                                    IREE_SV("target-low-registry-manifest")) ||
             iree_string_view_equal(target_name,
                                    IREE_SV("target-low-registry-json"))) {
    if (!iree_string_view_is_empty(profile_name)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "target-low registry manifest does not accept a profile");
    }
    out_request->format = LOOM_CHECK_EMIT_TARGET_LOW_REGISTRY_MANIFEST;
    return iree_ok_status();
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown emit target '%.*s'", (int)target_name.size,
                            target_name.data);
  }
  if (iree_string_view_starts_with(profile_name, IREE_SV("@"))) {
    out_request->target_bundle_symbol_name =
        iree_string_view_substr(profile_name, 1, IREE_HOST_SIZE_MAX);
    if (iree_string_view_is_empty(out_request->target_bundle_symbol_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "target bundle symbol name is required");
    }
    return iree_ok_status();
  }
  loom_llvmir_target_registry_t target_registry;
  loom_llvmir_target_registry_initialize(&target_registry);
  return loom_llvmir_target_registry_lookup_bundle(
      &target_registry, profile_name, &out_request->target_bundle);
}

static iree_string_view_t loom_check_emit_consume_line(
    iree_string_view_t* remaining) {
  iree_host_size_t newline_pos =
      iree_string_view_find(*remaining, IREE_SV("\n"), 0);
  if (newline_pos == IREE_STRING_VIEW_NPOS) {
    iree_string_view_t line = *remaining;
    *remaining = iree_string_view_empty();
    return line;
  }
  iree_string_view_t line = iree_string_view_substr(*remaining, 0, newline_pos);
  *remaining =
      iree_string_view_substr(*remaining, newline_pos + 1, IREE_HOST_SIZE_MAX);
  return line;
}

static bool loom_check_emit_line_has_ir(iree_string_view_t line) {
  line = iree_string_view_trim(line);
  return !iree_string_view_is_empty(line) &&
         !iree_string_view_starts_with(line, IREE_SV("//"));
}

static loom_source_range_t loom_check_emit_first_ir_source_range(
    const loom_check_case_t* test_case, iree_string_view_t filename) {
  if (iree_string_view_is_empty(test_case->input)) {
    return (loom_source_range_t){
        .provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE,
    };
  }

  iree_string_view_t remaining = test_case->input;
  uint32_t line_number = 1;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t line = loom_check_emit_consume_line(&remaining);
    if (loom_check_emit_line_has_ir(line)) {
      iree_host_size_t indentation_length = 0;
      while (indentation_length < line.size &&
             (line.data[indentation_length] == ' ' ||
              line.data[indentation_length] == '\t')) {
        ++indentation_length;
      }
      iree_host_size_t start =
          (iree_host_size_t)(line.data - test_case->input.data) +
          indentation_length;
      return (loom_source_range_t){
          .provenance = LOOM_SOURCE_PROVENANCE_EXACT_SOURCE,
          .filename = filename,
          .source = test_case->input,
          .start = start,
          .end = start + 1,
          .start_line = line_number,
          .start_column = (uint32_t)indentation_length + 1,
          .end_line = line_number,
          .end_column = (uint32_t)indentation_length + 2,
      };
    }
    ++line_number;
  }

  return (loom_source_range_t){
      .provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE,
  };
}

static iree_status_t loom_check_emit_collect_status_diagnostic(
    loom_check_diagnostic_collector_t* collector,
    const loom_check_case_t* test_case, iree_string_view_t filename,
    iree_string_view_t emit_target_name, iree_status_t failure_status) {
  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 1);
  if (!error) {
    iree_status_free(failure_status);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "LOWERING/001 diagnostic is not registered");
  }

  iree_string_view_t status_name = iree_make_cstring_view(
      iree_status_code_string(iree_status_code(failure_status)));
  loom_diagnostic_param_t params[3] = {
      loom_param_string(IREE_SV("module")),
      loom_param_string(emit_target_name),
      loom_param_string(status_name),
  };
  loom_source_range_t source_range =
      loom_check_emit_first_ir_source_range(test_case, filename);
  loom_diagnostic_t diagnostic = {
      .severity = error->severity,
      .error = error,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
      .emitter = LOOM_EMITTER_PASS,
      .origin = source_range,
      .source_location = source_range,
  };
  iree_status_t status =
      loom_check_diagnostic_collector_sink(collector, &diagnostic);
  iree_status_free(failure_status);
  return status;
}

static iree_status_t loom_check_emit_finish_status_failure(
    iree_status_t failure_status, loom_check_diagnostic_collector_t* collector,
    const loom_check_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    iree_string_view_t emit_target_name, iree_allocator_t allocator,
    loom_check_result_t* result) {
  IREE_RETURN_IF_ERROR(loom_check_emit_collect_status_diagnostic(
      collector, test_case, filename, emit_target_name, failure_status));
  return loom_check_diagnostic_collector_finish(
      collector, test_case, case_index, report, allocator, result);
}

static iree_status_t loom_check_emit_compare_output(
    const loom_check_case_t* test_case, iree_allocator_t allocator,
    loom_check_result_t* result) {
  iree_string_builder_t stripped_expected;
  iree_string_builder_initialize(allocator, &stripped_expected);

  iree_status_t status =
      loom_check_strip_comments(test_case->expected, &stripped_expected);
  if (iree_status_is_ok(status)) {
    iree_string_view_t actual_trimmed =
        iree_string_view_trim(iree_string_builder_view(&result->actual_output));
    iree_string_view_t expected_trimmed =
        iree_string_view_trim(iree_string_builder_view(&stripped_expected));
    if (iree_string_view_equal(actual_trimmed, expected_trimmed)) {
      result->raw_outcome = LOOM_CHECK_PASS;
    } else {
      result->raw_outcome = LOOM_CHECK_FAIL;
      status = loom_check_result_record_diff(expected_trimmed, actual_trimmed,
                                             allocator, result);
    }
  }

  iree_string_builder_deinitialize(&stripped_expected);
  return status;
}

static iree_status_t loom_check_emit_finish_diagnostics_and_compare_output(
    loom_check_diagnostic_collector_t* collector,
    const loom_check_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_allocator_t allocator,
    loom_check_result_t* result) {
  IREE_RETURN_IF_ERROR(loom_check_diagnostic_collector_finish(
      collector, test_case, case_index, report, allocator, result));
  if (result->raw_outcome != LOOM_CHECK_PASS || !result->has_actual_output) {
    return iree_ok_status();
  }
  return loom_check_emit_compare_output(test_case, allocator, result);
}

static iree_status_t loom_check_emit_strip_llvmir_comments(
    iree_string_view_t input, iree_string_builder_t* output) {
  iree_string_view_t remaining = input;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t line = loom_check_emit_consume_line(&remaining);
    iree_string_view_t trimmed = iree_string_view_trim(line);
    if (iree_string_view_starts_with(trimmed, IREE_SV(";"))) {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(output, line));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "\n"));
  }
  return iree_ok_status();
}

static bool loom_check_emit_is_llvmir_module_header_line(
    iree_string_view_t line) {
  line = iree_string_view_trim(line);
  return iree_string_view_starts_with(line, IREE_SV("source_filename =")) ||
         iree_string_view_starts_with(line, IREE_SV("target datalayout =")) ||
         iree_string_view_starts_with(line, IREE_SV("target triple ="));
}

static iree_status_t loom_check_emit_strip_llvmir_module_header(
    iree_string_view_t input, iree_string_builder_t* output) {
  bool stripped_header = false;
  bool emitted_body_line = false;
  iree_string_view_t remaining = input;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t line = loom_check_emit_consume_line(&remaining);
    if (loom_check_emit_is_llvmir_module_header_line(line)) {
      stripped_header = true;
      continue;
    }
    if (stripped_header && !emitted_body_line &&
        iree_string_view_is_empty(iree_string_view_trim(line))) {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(output, line));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "\n"));
    emitted_body_line = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_check_emit_write_llvmir_text(
    const loom_llvmir_module_t* lowered_module, loom_check_result_t* result) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&result->actual_output, &stream);
  return loom_llvmir_text_write_module(lowered_module, &stream);
}

static iree_status_t loom_check_emit_write_llvmir_body_text(
    const loom_llvmir_module_t* lowered_module, iree_allocator_t allocator,
    loom_check_result_t* result) {
  iree_string_builder_t module_text;
  iree_string_builder_initialize(allocator, &module_text);

  loom_output_stream_t stream;
  loom_output_stream_for_builder(&module_text, &stream);
  iree_status_t status = loom_llvmir_text_write_module(lowered_module, &stream);
  if (iree_status_is_ok(status)) {
    status = loom_check_emit_strip_llvmir_module_header(
        iree_string_builder_view(&module_text), &result->actual_output);
  }

  iree_string_builder_deinitialize(&module_text);
  return status;
}

static void loom_check_emit_byte_buffer_deinitialize(
    loom_check_emit_byte_buffer_t* buffer, iree_allocator_t allocator) {
  iree_allocator_free(allocator, buffer->data);
  *buffer = (loom_check_emit_byte_buffer_t){0};
}

static iree_status_t loom_check_emit_write_llvmir_bitcode_bytes(
    const loom_llvmir_module_t* lowered_module, iree_allocator_t allocator,
    loom_check_emit_byte_buffer_t* out_bitcode) {
  *out_bitcode = (loom_check_emit_byte_buffer_t){0};
  iree_io_stream_t* bitcode_stream = NULL;
  iree_status_t status = iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
          IREE_IO_STREAM_MODE_SEEKABLE,
      4096, allocator, &bitcode_stream);

  uint8_t* bitcode_data = NULL;
  iree_host_size_t bitcode_length = 0;
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_write_module(lowered_module, bitcode_stream);
  }
  if (iree_status_is_ok(status)) {
    iree_io_stream_pos_t stream_length = iree_io_stream_length(bitcode_stream);
    if (stream_length <= 0 ||
        (uint64_t)stream_length > (uint64_t)IREE_HOST_SIZE_MAX) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "LLVM bitcode output length is invalid");
    } else {
      bitcode_length = (iree_host_size_t)stream_length;
    }
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(allocator, bitcode_length, (void**)&bitcode_data);
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_stream_seek(bitcode_stream, IREE_IO_STREAM_SEEK_SET, 0);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_io_stream_read(bitcode_stream, bitcode_length, bitcode_data, NULL);
  }
  if (iree_status_is_ok(status)) {
    *out_bitcode = (loom_check_emit_byte_buffer_t){
        .data = bitcode_data,
        .length = bitcode_length,
    };
    bitcode_data = NULL;
  }

  iree_allocator_free(allocator, bitcode_data);
  iree_io_stream_release(bitcode_stream);
  return status;
}

static iree_status_t loom_check_emit_write_llvmir_bitcode_disassembly(
    const loom_llvmir_module_t* lowered_module, iree_allocator_t allocator,
    loom_check_result_t* result) {
  loom_check_emit_byte_buffer_t bitcode = {0};
  iree_status_t status = loom_check_emit_write_llvmir_bitcode_bytes(
      lowered_module, allocator, &bitcode);

  loom_llvm_tool_output_t disassembly = {0};
  if (iree_status_is_ok(status)) {
    loom_llvm_toolchain_t toolchain;
    loom_llvm_toolchain_initialize_from_environment(&toolchain);
    status = loom_llvm_tool_disassemble_bitcode(
        &toolchain, iree_make_const_byte_span(bitcode.data, bitcode.length),
        allocator, &disassembly);
  }
  if (iree_status_is_ok(status)) {
    status = loom_check_emit_strip_llvmir_comments(
        iree_make_string_view(disassembly.data, disassembly.length),
        &result->actual_output);
  }

  loom_llvm_tool_output_deinitialize(&disassembly, allocator);
  loom_check_emit_byte_buffer_deinitialize(&bitcode, allocator);
  return status;
}

static iree_status_t loom_check_emit_write_llvmir_object(
    const loom_llvmir_module_t* lowered_module,
    const loom_llvmir_target_profile_t* profile, iree_allocator_t allocator,
    loom_check_result_t* result) {
  loom_check_emit_byte_buffer_t bitcode = {0};
  iree_status_t status = loom_check_emit_write_llvmir_bitcode_bytes(
      lowered_module, allocator, &bitcode);

  loom_llvm_tool_output_t object = {0};
  if (iree_status_is_ok(status)) {
    loom_llvm_toolchain_t toolchain;
    loom_llvm_toolchain_initialize_from_environment(&toolchain);
    loom_llvmir_target_profile_llc_arguments_t llc_arguments = {0};
    status = loom_llvmir_target_profile_llc_arguments(profile, &llc_arguments);
    if (iree_status_is_ok(status)) {
      status = loom_llvm_tool_compile_object(
          &toolchain, iree_make_const_byte_span(bitcode.data, bitcode.length),
          llc_arguments.values, llc_arguments.count, allocator, &object);
    }
  }
  if (iree_status_is_ok(status) && object.length == 0) {
    status =
        iree_make_status(IREE_STATUS_DATA_LOSS, "LLVM object output is empty");
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_format(
        &result->actual_output, "object emitted: %.*s\n",
        (int)profile->name.size, profile->name.data);
  }

  loom_llvm_tool_output_deinitialize(&object, allocator);
  loom_check_emit_byte_buffer_deinitialize(&bitcode, allocator);
  return status;
}

static bool loom_check_emit_assembly_line_is_label(iree_string_view_t line) {
  return line.size > 0 && line.data[line.size - 1] == ':';
}

static iree_string_view_t loom_check_emit_strip_assembly_comment(
    iree_string_view_t line) {
  iree_host_size_t hash_position = iree_string_view_find(line, IREE_SV("#"), 0);
  iree_host_size_t semicolon_position =
      iree_string_view_find(line, IREE_SV(";"), 0);
  iree_host_size_t comment_position = IREE_STRING_VIEW_NPOS;
  if (hash_position != IREE_STRING_VIEW_NPOS) {
    comment_position = hash_position;
  }
  if (semicolon_position != IREE_STRING_VIEW_NPOS &&
      (comment_position == IREE_STRING_VIEW_NPOS ||
       semicolon_position < comment_position)) {
    comment_position = semicolon_position;
  }
  if (comment_position == IREE_STRING_VIEW_NPOS) {
    return line;
  }
  return iree_string_view_substr(line, 0, comment_position);
}

static iree_status_t loom_check_emit_write_assembly_mnemonics(
    iree_string_view_t assembly, iree_string_builder_t* output) {
  iree_string_view_t remaining = assembly;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t line = loom_check_emit_strip_assembly_comment(
        loom_check_emit_consume_line(&remaining));
    line = iree_string_view_trim(line);
    if (iree_string_view_is_empty(line) ||
        iree_string_view_starts_with_char(line, '.') ||
        iree_string_view_starts_with_char(line, '#') ||
        iree_string_view_starts_with_char(line, ';') ||
        loom_check_emit_assembly_line_is_label(line)) {
      continue;
    }
    while (iree_string_view_starts_with_char(line, '{')) {
      iree_host_size_t closing_brace =
          iree_string_view_find(line, IREE_SV("}"), 0);
      if (closing_brace == IREE_STRING_VIEW_NPOS) {
        break;
      }
      line = iree_string_view_trim(
          iree_string_view_substr(line, closing_brace + 1, IREE_HOST_SIZE_MAX));
    }
    if (iree_string_view_is_empty(line)) {
      continue;
    }
    iree_host_size_t mnemonic_length = 0;
    while (mnemonic_length < line.size && line.data[mnemonic_length] != ' ' &&
           line.data[mnemonic_length] != '\t') {
      ++mnemonic_length;
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        output, iree_string_view_substr(line, 0, mnemonic_length)));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "\n"));
  }
  return iree_ok_status();
}

static iree_status_t loom_check_emit_write_llvmir_assembly_mnemonics(
    const loom_llvmir_module_t* lowered_module,
    const loom_llvmir_target_profile_t* profile, iree_allocator_t allocator,
    loom_check_result_t* result) {
  loom_check_emit_byte_buffer_t bitcode = {0};
  iree_status_t status = loom_check_emit_write_llvmir_bitcode_bytes(
      lowered_module, allocator, &bitcode);

  loom_llvm_tool_output_t assembly = {0};
  if (iree_status_is_ok(status)) {
    loom_llvm_toolchain_t toolchain;
    loom_llvm_toolchain_initialize_from_environment(&toolchain);
    loom_llvmir_target_profile_llc_arguments_t llc_arguments = {0};
    status = loom_llvmir_target_profile_llc_arguments(profile, &llc_arguments);
    if (iree_status_is_ok(status)) {
      status = loom_llvm_tool_compile_assembly(
          &toolchain, iree_make_const_byte_span(bitcode.data, bitcode.length),
          llc_arguments.values, llc_arguments.count, allocator, &assembly);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_check_emit_write_assembly_mnemonics(
        iree_make_string_view(assembly.data, assembly.length),
        &result->actual_output);
  }

  loom_llvm_tool_output_deinitialize(&assembly, allocator);
  loom_check_emit_byte_buffer_deinitialize(&bitcode, allocator);
  return status;
}

static iree_status_t loom_check_emit_find_func_like(
    const loom_module_t* module, iree_string_view_t symbol_name,
    loom_func_like_t* out_function) {
  *out_function = (loom_func_like_t){0};
  loom_string_id_t name_id = loom_module_lookup_string(module, symbol_name);
  if (name_id == LOOM_STRING_ID_INVALID) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "unknown function '@%.*s'",
                            (int)symbol_name.size, symbol_name.data);
  }
  uint16_t symbol_id = loom_module_find_symbol(module, name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "unknown function '@%.*s'",
                            (int)symbol_name.size, symbol_name.data);
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
  loom_func_like_t function = loom_func_like_cast(module, symbol->defining_op);
  if (!loom_func_like_isa(function) || !loom_func_like_body(function)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol '@%.*s' does not name a function body",
                            (int)symbol_name.size, symbol_name.data);
  }
  *out_function = function;
  return iree_ok_status();
}

static iree_status_t loom_check_emit_write_liveness_json(
    const loom_module_t* module, iree_string_view_t symbol_name,
    iree_arena_allocator_t* analysis_arena, loom_check_result_t* result) {
  loom_func_like_t function = {0};
  IREE_RETURN_IF_ERROR(
      loom_check_emit_find_func_like(module, symbol_name, &function));
  loom_liveness_analysis_t analysis = {0};
  IREE_RETURN_IF_ERROR(loom_liveness_analyze_region(
      module, loom_func_like_body(function), analysis_arena, &analysis));
  return loom_liveness_format_json(&analysis, &result->actual_output);
}

static iree_status_t loom_check_emit_find_low_func_def(
    const loom_module_t* module, iree_string_view_t symbol_name,
    const loom_op_t** out_low_function) {
  *out_low_function = NULL;
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

static iree_status_t loom_check_emit_write_low_schedule_json(
    const loom_module_t* module, iree_string_view_t symbol_name,
    const loom_low_descriptor_registry_t* descriptor_registry,
    loom_low_schedule_diagnostic_flags_t diagnostic_flags,
    loom_low_schedule_strategy_t strategy, iree_diagnostic_emitter_t emitter,
    bool suppress_output, iree_arena_allocator_t* analysis_arena,
    loom_check_result_t* result) {
  const loom_op_t* low_function = NULL;
  IREE_RETURN_IF_ERROR(
      loom_check_emit_find_low_func_def(module, symbol_name, &low_function));
  loom_low_schedule_options_t options = {
      .descriptor_registry = descriptor_registry,
      .emitter = emitter,
      .diagnostic_flags = diagnostic_flags,
      .strategy = strategy,
  };
  loom_low_schedule_sidecar_t sidecar = {0};
  IREE_RETURN_IF_ERROR(loom_low_schedule_function(
      module, low_function, &options, analysis_arena, &sidecar));
  if (suppress_output) {
    return iree_ok_status();
  }
  return loom_low_schedule_format_json(&sidecar, &result->actual_output);
}

static iree_status_t loom_check_emit_write_low_allocation_json(
    const loom_module_t* module, iree_string_view_t symbol_name,
    const loom_low_descriptor_registry_t* descriptor_registry,
    const loom_low_allocation_budget_t* budgets, iree_host_size_t budget_count,
    loom_low_allocation_diagnostic_flags_t diagnostic_flags,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* analysis_arena,
    bool suppress_output, loom_check_result_t* result) {
  const loom_op_t* low_function = NULL;
  IREE_RETURN_IF_ERROR(
      loom_check_emit_find_low_func_def(module, symbol_name, &low_function));
  loom_low_allocation_options_t options = {
      .descriptor_registry = descriptor_registry,
      .budgets = budgets,
      .budget_count = budget_count,
      .emitter = emitter,
      .diagnostic_flags = diagnostic_flags,
  };
  loom_low_allocation_sidecar_t sidecar = {0};
  IREE_RETURN_IF_ERROR(loom_low_allocate_function(
      module, low_function, &options, analysis_arena, &sidecar));
  if (suppress_output) {
    return iree_ok_status();
  }
  return loom_low_allocation_format_json(&sidecar, &result->actual_output);
}

static iree_status_t loom_check_emit_write_low_packet_json(
    const loom_module_t* module, iree_string_view_t symbol_name,
    const loom_low_descriptor_registry_t* descriptor_registry,
    loom_low_schedule_strategy_t strategy,
    const loom_low_allocation_budget_t* budgets, iree_host_size_t budget_count,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* analysis_arena,
    bool suppress_output, loom_check_result_t* result) {
  const loom_op_t* low_function = NULL;
  IREE_RETURN_IF_ERROR(
      loom_check_emit_find_low_func_def(module, symbol_name, &low_function));
  loom_low_packetization_options_t packetization_options = {
      .descriptor_registry = descriptor_registry,
      .schedule_strategy = strategy,
      .allocation_budgets = budgets,
      .allocation_budget_count = budget_count,
      .emitter = emitter,
  };
  loom_low_packetization_t packetization = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_packetize_function(module, low_function, &packetization_options,
                                  analysis_arena, &packetization));
  if (suppress_output) {
    return iree_ok_status();
  }
  return loom_low_packet_format_json(&packetization.schedule,
                                     &packetization.allocation,
                                     &result->actual_output);
}

iree_status_t loom_check_execute_emit(
    const loom_check_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    const loom_check_environment_t* environment, loom_context_t* context,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_check_result_t* result) {
  iree_arena_allocator_t diagnostic_arena;
  iree_arena_initialize(block_pool, &diagnostic_arena);
  loom_check_diagnostic_collector_t diagnostic_collector = {
      .arena = &diagnostic_arena,
      .host_allocator = allocator,
      .result = result,
  };

  loom_check_emit_request_t request;
  iree_status_t status =
      loom_check_emit_parse_request(test_case->emit_target, &request);
  if (!iree_status_is_ok(status)) {
    status = loom_check_emit_finish_status_failure(
        status, &diagnostic_collector, test_case, case_index, report, filename,
        request.emit_target_name, allocator, result);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }
  if (request.format == LOOM_CHECK_EMIT_LOW_DESCRIPTOR_MANIFEST ||
      request.format == LOOM_CHECK_EMIT_TARGET_LOW_REGISTRY_MANIFEST) {
    loom_target_low_descriptor_registry_t registry = {0};
    status = loom_check_environment_initialize_low_descriptor_registry(
        environment, &registry);
    if (request.format == LOOM_CHECK_EMIT_LOW_DESCRIPTOR_MANIFEST) {
      if (iree_status_is_ok(status)) {
        status = loom_low_descriptor_registry_lookup(
            &registry.registry, request.low_descriptor_set_key,
            &request.low_descriptor_set);
      }
      if (iree_status_is_ok(status) && request.low_descriptor_set == NULL) {
        status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                  "unknown low descriptor set '%.*s'",
                                  (int)request.low_descriptor_set_key.size,
                                  request.low_descriptor_set_key.data);
      }
      if (iree_status_is_ok(status)) {
        status = loom_low_descriptor_set_format_manifest_json(
            request.low_descriptor_set, &result->actual_output);
      }
    } else if (iree_status_is_ok(status)) {
      status = loom_target_low_descriptor_registry_format_manifest_json(
          &registry, LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
          &result->actual_output);
    }
    if (!iree_status_is_ok(status)) {
      status = loom_check_emit_finish_status_failure(
          status, &diagnostic_collector, test_case, case_index, report,
          filename, request.emit_target_name, allocator, result);
      iree_arena_deinitialize(&diagnostic_arena);
      return status;
    }
    result->has_actual_output = true;
    if (test_case->annotation_count > 0) {
      status = loom_check_emit_finish_diagnostics_and_compare_output(
          &diagnostic_collector, test_case, case_index, report, allocator,
          result);
      iree_arena_deinitialize(&diagnostic_arena);
      return status;
    }
    status = loom_check_emit_compare_output(test_case, allocator, result);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  iree_string_builder_t stripped_input;
  iree_string_builder_initialize(allocator, &stripped_input);
  status = loom_check_strip_comments(test_case->input, &stripped_input);
  if (!iree_status_is_ok(status)) {
    iree_string_builder_deinitialize(&stripped_input);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  loom_module_t* module = NULL;
  loom_target_low_descriptor_registry_t low_registry = {0};
  status = loom_check_environment_initialize_low_descriptor_registry(
      environment, &low_registry);
  loom_text_parse_options_t parse_options = {
      .diagnostic_sink = {.fn = loom_check_diagnostic_collector_sink,
                          .user_data = &diagnostic_collector},
      .max_errors = 20,
  };
  loom_low_descriptor_text_asm_environment_initialize(
      &low_registry.registry, &parse_options.low_asm_environment);
  iree_string_view_t stripped_view = iree_string_builder_view(&stripped_input);
  if (iree_status_is_ok(status)) {
    status = loom_text_parse(stripped_view, filename, context, block_pool,
                             &parse_options, &module);
  }
  diagnostic_collector.module = module;
  if (!iree_status_is_ok(status)) {
    loom_module_free(module);
    iree_string_builder_deinitialize(&stripped_input);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }
  if (!module || diagnostic_collector.count > 0) {
    status = loom_check_diagnostic_collector_finish(&diagnostic_collector,
                                                    test_case, case_index,
                                                    report, allocator, result);
    loom_module_free(module);
    iree_string_builder_deinitialize(&stripped_input);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  const loom_target_preset_registry_t preset_registry =
      loom_target_low_descriptor_registry_presets(&low_registry);
  iree_host_size_t expanded_preset_count = 0;
  status = loom_target_expand_presets(module, &preset_registry,
                                      &expanded_preset_count);
  if (!iree_status_is_ok(status)) {
    loom_module_free(module);
    iree_string_builder_deinitialize(&stripped_input);
    status = loom_check_emit_finish_status_failure(
        status, &diagnostic_collector, test_case, case_index, report, filename,
        request.emit_target_name, allocator, result);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  if (request.format == LOOM_CHECK_EMIT_LIVENESS_JSON ||
      request.format == LOOM_CHECK_EMIT_LOW_SCHEDULE_JSON ||
      request.format == LOOM_CHECK_EMIT_LOW_ALLOCATION_JSON ||
      request.format == LOOM_CHECK_EMIT_LOW_PACKET_JSON) {
    loom_source_entry_t source_entry = {0};
    loom_source_table_resolver_t resolver_data = {0};
    status = loom_check_source_resolver_for_case(
        context, filename, stripped_view, &source_entry, &resolver_data);
    loom_verify_options_t verify_options = {
        .sink = {.fn = loom_check_diagnostic_collector_sink,
                 .user_data = &diagnostic_collector},
        .max_errors = 20,
        .source_resolver = {.fn = loom_source_table_resolve,
                            .user_data = &resolver_data},
    };
    loom_verify_result_t verify_result = {0};
    if (iree_status_is_ok(status)) {
      status = loom_verify_module(module, &verify_options, &verify_result);
    }
    if (!iree_status_is_ok(status)) {
      loom_module_free(module);
      iree_string_builder_deinitialize(&stripped_input);
      status = loom_check_emit_finish_status_failure(
          status, &diagnostic_collector, test_case, case_index, report,
          filename, request.emit_target_name, allocator, result);
      iree_arena_deinitialize(&diagnostic_arena);
      return status;
    }
    if (verify_result.error_count > 0 || diagnostic_collector.count > 0) {
      status = loom_check_diagnostic_collector_finish(
          &diagnostic_collector, test_case, case_index, report, allocator,
          result);
      loom_module_free(module);
      iree_string_builder_deinitialize(&stripped_input);
      iree_arena_deinitialize(&diagnostic_arena);
      return status;
    }
    iree_host_size_t low_diagnostic_count = 0;
    if (request.format == LOOM_CHECK_EMIT_LOW_SCHEDULE_JSON ||
        request.format == LOOM_CHECK_EMIT_LOW_ALLOCATION_JSON ||
        request.format == LOOM_CHECK_EMIT_LOW_PACKET_JSON) {
      loom_check_diagnostic_emitter_capture_t low_diagnostic_capture = {
          .diagnostic_collector = &diagnostic_collector,
          .module = module,
          .source_resolver = {.fn = loom_source_table_resolve,
                              .user_data = &resolver_data},
          .emitter = LOOM_EMITTER_VERIFIER,
      };
      loom_low_verify_options_t low_verify_options = {
          .flags = LOOM_LOW_VERIFY_FLAG_VERIFY_DESCRIPTOR_REGISTRY,
          .descriptor_registry = &low_registry.registry,
          .descriptor_requirements =
              LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
          .emitter =
              {
                  .fn = loom_check_diagnostic_emitter_capture_emit,
                  .user_data = &low_diagnostic_capture,
              },
          .max_errors = 20,
      };
      loom_low_verify_result_t low_verify_result = {0};
      status = loom_low_verify_module(module, &low_verify_options,
                                      &low_verify_result);
      low_diagnostic_count = low_diagnostic_capture.emission_count;
      if (iree_status_is_ok(status) && (low_verify_result.error_count > 0 ||
                                        diagnostic_collector.count > 0)) {
        status = loom_check_diagnostic_collector_finish(
            &diagnostic_collector, test_case, case_index, report, allocator,
            result);
        loom_module_free(module);
        iree_string_builder_deinitialize(&stripped_input);
        iree_arena_deinitialize(&diagnostic_arena);
        return status;
      }
    }
    loom_check_diagnostic_emitter_capture_t pass_diagnostic_capture = {
        .diagnostic_collector = &diagnostic_collector,
        .module = module,
        .source_resolver = {.fn = loom_source_table_resolve,
                            .user_data = &resolver_data},
        .emitter = LOOM_EMITTER_PASS,
    };
    if (iree_status_is_ok(status)) {
      if (request.format == LOOM_CHECK_EMIT_LIVENESS_JSON) {
        status = loom_check_emit_write_liveness_json(
            module, request.analysis_symbol_name, &diagnostic_arena, result);
      } else if (request.format == LOOM_CHECK_EMIT_LOW_SCHEDULE_JSON) {
        status = loom_check_emit_write_low_schedule_json(
            module, request.analysis_symbol_name, &low_registry.registry,
            request.low_schedule_diagnostic_flags,
            request.low_schedule_strategy,
            (iree_diagnostic_emitter_t){
                .fn = loom_check_diagnostic_emitter_capture_emit,
                .user_data = &pass_diagnostic_capture,
            },
            request.suppress_output, &diagnostic_arena, result);
      } else if (request.format == LOOM_CHECK_EMIT_LOW_ALLOCATION_JSON) {
        status = loom_check_emit_write_low_allocation_json(
            module, request.analysis_symbol_name, &low_registry.registry,
            request.low_allocation_budgets, request.low_allocation_budget_count,
            request.low_allocation_diagnostic_flags,
            (iree_diagnostic_emitter_t){
                .fn = loom_check_diagnostic_emitter_capture_emit,
                .user_data = &pass_diagnostic_capture,
            },
            &diagnostic_arena, request.suppress_output, result);
      } else {
        status = loom_check_emit_write_low_packet_json(
            module, request.analysis_symbol_name, &low_registry.registry,
            request.low_schedule_strategy, request.low_allocation_budgets,
            request.low_allocation_budget_count,
            (iree_diagnostic_emitter_t){
                .fn = loom_check_diagnostic_emitter_capture_emit,
                .user_data = &pass_diagnostic_capture,
            },
            &diagnostic_arena, request.suppress_output, result);
      }
    }
    if (iree_status_is_ok(status)) {
      result->has_actual_output = true;
    }
    loom_module_free(module);
    diagnostic_collector.module = NULL;
    if (!iree_status_is_ok(status)) {
      if (low_diagnostic_count > 0 ||
          pass_diagnostic_capture.emission_count > 0 ||
          diagnostic_collector.count > 0) {
        iree_status_free(status);
        status = loom_check_diagnostic_collector_finish(
            &diagnostic_collector, test_case, case_index, report, allocator,
            result);
      } else {
        status = loom_check_emit_finish_status_failure(
            status, &diagnostic_collector, test_case, case_index, report,
            filename, request.emit_target_name, allocator, result);
      }
      iree_string_builder_deinitialize(&stripped_input);
      iree_arena_deinitialize(&diagnostic_arena);
      return status;
    }
    if (test_case->annotation_count > 0 ||
        pass_diagnostic_capture.emission_count > 0 ||
        diagnostic_collector.count > 0) {
      status = loom_check_emit_finish_diagnostics_and_compare_output(
          &diagnostic_collector, test_case, case_index, report, allocator,
          result);
      iree_string_builder_deinitialize(&stripped_input);
      iree_arena_deinitialize(&diagnostic_arena);
      return status;
    }
    status = loom_check_emit_compare_output(test_case, allocator, result);
    iree_string_builder_deinitialize(&stripped_input);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }
  iree_string_builder_deinitialize(&stripped_input);

  loom_target_ir_bundle_storage_t target_bundle_storage = {0};
  if (!iree_string_view_is_empty(request.target_bundle_symbol_name)) {
    status = loom_target_ir_bundle_from_symbol_name(
        module, request.target_bundle_symbol_name, &target_bundle_storage);
    if (!iree_status_is_ok(status)) {
      loom_module_free(module);
      status = loom_check_emit_finish_status_failure(
          status, &diagnostic_collector, test_case, case_index, report,
          filename, request.emit_target_name, allocator, result);
      iree_arena_deinitialize(&diagnostic_arena);
      return status;
    }
    request.target_bundle = &target_bundle_storage.bundle;
  }

  loom_llvmir_target_registry_t target_registry;
  loom_llvmir_target_registry_initialize(&target_registry);
  loom_llvmir_target_legality_provider_list_t legality_providers;
  status = loom_llvmir_target_registry_select_legality_providers(
      &target_registry, request.target_bundle, &legality_providers);
  if (!iree_status_is_ok(status)) {
    loom_module_free(module);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }
  loom_llvmir_target_legality_options_t legality_options = {
      .snapshot = request.target_bundle->snapshot,
      .export_plan = request.target_bundle->export_plan,
      .config = request.target_bundle->config,
      .providers = legality_providers.providers,
      .provider_count = legality_providers.provider_count,
  };
  status = loom_llvmir_verify_target_legality(module, &legality_options, NULL);
  if (!iree_status_is_ok(status)) {
    loom_module_free(module);
    status = loom_check_emit_finish_status_failure(
        status, &diagnostic_collector, test_case, case_index, report, filename,
        request.emit_target_name, allocator, result);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  loom_llvmir_target_profile_storage_t profile_storage;
  status = loom_llvmir_target_profile_storage_initialize_from_bundle(
      request.target_bundle, &profile_storage);
  if (!iree_status_is_ok(status)) {
    loom_module_free(module);
    status = loom_check_emit_finish_status_failure(
        status, &diagnostic_collector, test_case, case_index, report, filename,
        request.emit_target_name, allocator, result);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }
  const loom_llvmir_target_profile_t* profile = &profile_storage.profile;
  loom_llvmir_lowering_provider_list_t lowering_providers;
  status = loom_llvmir_target_registry_select_lowering_providers(
      &target_registry, profile, &lowering_providers);
  if (!iree_status_is_ok(status)) {
    loom_module_free(module);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }
  loom_llvmir_lowering_options_t options = {
      .target_profile = profile,
      .source_name = filename,
      .providers = lowering_providers.providers,
      .provider_count = lowering_providers.provider_count,
  };
  loom_llvmir_module_t* lowered_module = NULL;
  status =
      loom_llvmir_lower_module(module, &options, allocator, &lowered_module);
  loom_module_free(module);
  diagnostic_collector.module = NULL;
  if (!iree_status_is_ok(status)) {
    loom_llvmir_module_free(lowered_module);
    status = loom_check_emit_finish_status_failure(
        status, &diagnostic_collector, test_case, case_index, report, filename,
        request.emit_target_name, allocator, result);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  status = loom_llvmir_verify_module(lowered_module);
  if (iree_status_is_ok(status)) {
    switch (request.format) {
      case LOOM_CHECK_EMIT_LLVMIR_TEXT:
        status = loom_check_emit_write_llvmir_text(lowered_module, result);
        break;
      case LOOM_CHECK_EMIT_LLVMIR_BODY_TEXT:
        status = loom_check_emit_write_llvmir_body_text(lowered_module,
                                                        allocator, result);
        break;
      case LOOM_CHECK_EMIT_LLVMIR_BITCODE_DISASSEMBLY:
        status = loom_check_emit_write_llvmir_bitcode_disassembly(
            lowered_module, allocator, result);
        break;
      case LOOM_CHECK_EMIT_LLVMIR_OBJECT:
        status = loom_check_emit_write_llvmir_object(lowered_module, profile,
                                                     allocator, result);
        break;
      case LOOM_CHECK_EMIT_LLVMIR_ASSEMBLY_MNEMONICS:
        status = loom_check_emit_write_llvmir_assembly_mnemonics(
            lowered_module, profile, allocator, result);
        break;
      case LOOM_CHECK_EMIT_LIVENESS_JSON:
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "liveness JSON emit should bypass LLVMIR lowering");
        break;
      case LOOM_CHECK_EMIT_LOW_SCHEDULE_JSON:
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "low schedule JSON emit should bypass LLVMIR lowering");
        break;
      case LOOM_CHECK_EMIT_LOW_ALLOCATION_JSON:
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "low allocation JSON emit should bypass LLVMIR lowering");
        break;
      case LOOM_CHECK_EMIT_LOW_PACKET_JSON:
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "low packet JSON emit should bypass LLVMIR lowering");
        break;
      case LOOM_CHECK_EMIT_LOW_DESCRIPTOR_MANIFEST:
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "low descriptor manifest emit should bypass LLVMIR lowering");
        break;
      case LOOM_CHECK_EMIT_TARGET_LOW_REGISTRY_MANIFEST:
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "target-low registry manifest emit should bypass LLVMIR lowering");
        break;
    }
  }
  loom_llvmir_module_free(lowered_module);
  if (!iree_status_is_ok(status)) {
    status = loom_check_emit_finish_status_failure(
        status, &diagnostic_collector, test_case, case_index, report, filename,
        request.emit_target_name, allocator, result);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }
  result->has_actual_output = true;

  if (test_case->annotation_count > 0) {
    status = loom_check_emit_finish_diagnostics_and_compare_output(
        &diagnostic_collector, test_case, case_index, report, allocator,
        result);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  status = loom_check_emit_compare_output(test_case, allocator, result);
  iree_arena_deinitialize(&diagnostic_arena);
  return status;
}
