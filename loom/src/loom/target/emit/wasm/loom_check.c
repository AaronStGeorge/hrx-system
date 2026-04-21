// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/wasm/loom_check.h"

#include <inttypes.h>
#include <string.h>

#include "loom/target/emit/wasm/module_binary.h"
#include "loom/tools/loom-check/low_emit.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

typedef struct loom_wasm_loom_check_emit_options_t {
  // Module-local low.func.def symbol selected by the RUN line.
  iree_string_view_t function_symbol_name;
  // Export name written into the single-function Wasm module.
  iree_string_view_t export_name;
  // True once an export option has been parsed.
  bool has_export_name_option;
  // Candidate selection strategy used by low packetization.
  loom_low_schedule_strategy_t schedule_strategy;
  // True once a strategy option has been parsed.
  bool has_schedule_strategy_option;
  // Low allocation budget overrides parsed from target options.
  loom_low_allocation_budget_t
      allocation_budgets[LOOM_CHECK_LOW_EMIT_MAX_ALLOCATION_BUDGETS];
  // Number of entries in |allocation_budgets|.
  iree_host_size_t allocation_budget_count;
} loom_wasm_loom_check_emit_options_t;

typedef struct loom_wasm_loom_check_reader_t {
  // Borrowed Wasm module bytes.
  const uint8_t* data;
  // Total byte length of |data|.
  iree_host_size_t data_length;
  // Current byte offset into |data|.
  iree_host_size_t offset;
} loom_wasm_loom_check_reader_t;

static bool loom_wasm_loom_check_emit_provider_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  (void)provider;
  return iree_string_view_equal(target_name, IREE_SV("wasm-module-summary")) ||
         iree_string_view_equal(target_name, IREE_SV("wasm-summary"));
}

static iree_status_t loom_wasm_loom_check_parse_key_value_option(
    iree_string_view_t token, loom_wasm_loom_check_emit_options_t* options,
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
                              "duplicate Wasm module option 'strategy'");
    }
    IREE_RETURN_IF_ERROR(loom_check_low_emit_parse_schedule_strategy(
        value, IREE_SV("Wasm module"), &options->schedule_strategy));
    options->has_schedule_strategy_option = true;
    *out_matched = true;
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("export"))) {
    if (options->has_export_name_option) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate Wasm module option 'export'");
    }
    if (iree_string_view_is_empty(value)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Wasm module option 'export' requires a value");
    }
    options->export_name = value;
    options->has_export_name_option = true;
    *out_matched = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_loom_check_parse_option(
    iree_string_view_t token, loom_wasm_loom_check_emit_options_t* options) {
  bool matched = false;
  IREE_RETURN_IF_ERROR(
      loom_wasm_loom_check_parse_key_value_option(token, options, &matched));
  if (matched) {
    return iree_ok_status();
  }
  return loom_check_low_emit_parse_allocation_budget(
      token, IREE_SV("Wasm module"), options->allocation_budgets,
      IREE_ARRAYSIZE(options->allocation_budgets),
      &options->allocation_budget_count);
}

static iree_status_t loom_wasm_loom_check_parse_emit_options(
    const loom_check_emit_provider_request_t* request,
    loom_wasm_loom_check_emit_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  *out_options = (loom_wasm_loom_check_emit_options_t){
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
  };

  iree_string_view_t symbol_name = iree_string_view_empty();
  iree_string_view_t option_text = iree_string_view_empty();
  iree_string_view_split(request->target_options, ' ', &symbol_name,
                         &option_text);
  symbol_name = iree_string_view_trim(symbol_name);
  option_text = iree_string_view_trim(option_text);
  if (!iree_string_view_starts_with(symbol_name, IREE_SV("@"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Wasm module summary requires a low function "
                            "symbol name");
  }
  out_options->function_symbol_name =
      iree_string_view_substr(symbol_name, 1, IREE_HOST_SIZE_MAX);
  if (iree_string_view_is_empty(out_options->function_symbol_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Wasm module summary low function symbol name is "
                            "required");
  }
  out_options->export_name = out_options->function_symbol_name;

  while (!iree_string_view_is_empty(option_text)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_t remaining = iree_string_view_empty();
    iree_string_view_split(option_text, ' ', &token, &remaining);
    token = iree_string_view_trim(token);
    if (!iree_string_view_is_empty(token)) {
      IREE_RETURN_IF_ERROR(
          loom_wasm_loom_check_parse_option(token, out_options));
    }
    option_text = iree_string_view_trim(remaining);
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_loom_check_read_u8(
    loom_wasm_loom_check_reader_t* reader, uint8_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  if (reader->offset >= reader->data_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "truncated Wasm module summary input");
  }
  *out_value = reader->data[reader->offset++];
  return iree_ok_status();
}

static iree_status_t loom_wasm_loom_check_read_u32_leb(
    loom_wasm_loom_check_reader_t* reader, uint32_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  uint32_t value = 0;
  uint32_t shift = 0;
  while (true) {
    uint8_t byte = 0;
    IREE_RETURN_IF_ERROR(loom_wasm_loom_check_read_u8(reader, &byte));
    value |= (uint32_t)(byte & 0x7Fu) << shift;
    if ((byte & 0x80u) == 0) {
      *out_value = value;
      return iree_ok_status();
    }
    shift += 7;
    if (shift >= 35) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Wasm module summary LEB value is too large");
    }
  }
}

static iree_status_t loom_wasm_loom_check_skip(
    loom_wasm_loom_check_reader_t* reader, iree_host_size_t byte_count) {
  if (byte_count > reader->data_length - reader->offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "truncated Wasm module section payload");
  }
  reader->offset += byte_count;
  return iree_ok_status();
}

static iree_string_view_t loom_wasm_loom_check_section_name(
    uint8_t section_id) {
  switch (section_id) {
    case 0:
      return IREE_SV("custom");
    case 1:
      return IREE_SV("type");
    case 2:
      return IREE_SV("import");
    case 3:
      return IREE_SV("function");
    case 4:
      return IREE_SV("table");
    case 5:
      return IREE_SV("memory");
    case 6:
      return IREE_SV("global");
    case 7:
      return IREE_SV("export");
    case 8:
      return IREE_SV("start");
    case 9:
      return IREE_SV("element");
    case 10:
      return IREE_SV("code");
    case 11:
      return IREE_SV("data");
    case 12:
      return IREE_SV("data-count");
    default:
      return IREE_SV("unknown");
  }
}

static iree_status_t loom_wasm_loom_check_write_section_array(
    const loom_wasm_module_binary_t* module, loom_output_stream_t* stream) {
  loom_wasm_loom_check_reader_t reader = {
      .data = module->data,
      .data_length = module->data_length,
  };
  const uint8_t expected_header[] = {
      0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00,
  };
  if (reader.data_length < sizeof(expected_header) ||
      memcmp(reader.data, expected_header, sizeof(expected_header)) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Wasm module summary input has an invalid header");
  }
  reader.offset = sizeof(expected_header);

  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
  bool needs_comma = false;
  while (reader.offset < reader.data_length) {
    uint8_t section_id = 0;
    IREE_RETURN_IF_ERROR(loom_wasm_loom_check_read_u8(&reader, &section_id));
    uint32_t payload_length = 0;
    IREE_RETURN_IF_ERROR(
        loom_wasm_loom_check_read_u32_leb(&reader, &payload_length));
    if (needs_comma) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
    }
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, loom_wasm_loom_check_section_name(section_id)));
    needs_comma = true;
    IREE_RETURN_IF_ERROR(loom_wasm_loom_check_skip(&reader, payload_length));
  }
  return loom_output_stream_write_cstring(stream, "]");
}

static iree_status_t loom_wasm_loom_check_write_module_summary(
    const loom_wasm_module_binary_t* module, iree_string_view_t export_name,
    iree_string_builder_t* builder) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      &stream,
      "{\"format\":\"loom.wasm.module.summary.v0\",\"version\":1,"
      "\"export\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(&stream, export_name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream,
      ",\"memory\":%s,\"sections\":", module->has_memory ? "true" : "false"));
  IREE_RETURN_IF_ERROR(
      loom_wasm_loom_check_write_section_array(module, &stream));
  return loom_output_stream_write_cstring(&stream, "}");
}

static iree_status_t loom_wasm_loom_check_emit_provider_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  (void)provider;
  loom_wasm_loom_check_emit_options_t options;
  IREE_RETURN_IF_ERROR(
      loom_wasm_loom_check_parse_emit_options(request, &options));

  loom_low_packetization_t packetization = {0};
  IREE_RETURN_IF_ERROR(loom_check_low_emit_packetize_function(
      request, options.function_symbol_name, options.schedule_strategy,
      options.allocation_budgets, options.allocation_budget_count,
      &packetization));

  loom_wasm_module_binary_t module = {0};
  iree_status_t status = loom_wasm_emit_single_function_module(
      &packetization.schedule, &packetization.allocation, options.export_name,
      request->host_allocator, &module);
  if (iree_status_is_ok(status)) {
    status = loom_wasm_loom_check_write_module_summary(
        &module, options.export_name, &request->result->actual_output);
  }
  loom_wasm_module_binary_deinitialize(&module, request->host_allocator);
  return status;
}

static iree_status_t loom_wasm_loom_check_emit_provider_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(
      builder, "wasm-module-summary, wasm-summary");
}

const loom_check_emit_provider_t loom_wasm_loom_check_emit_provider = {
    .name = IREE_SVL("wasm"),
    .match = loom_wasm_loom_check_emit_provider_matches,
    .execute = loom_wasm_loom_check_emit_provider_execute,
    .append_names = loom_wasm_loom_check_emit_provider_append_names,
};
