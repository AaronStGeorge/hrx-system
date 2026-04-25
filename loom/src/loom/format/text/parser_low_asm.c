// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "loom/format/text/parser_internal.h"

//===----------------------------------------------------------------------===//
// Target-low asm packet parsing
//===----------------------------------------------------------------------===//

typedef struct loom_low_asm_result_names_t {
  // Result-name tokens in parsed source order.
  loom_token_t* tokens;
  // Number of result-name tokens currently stored.
  iree_host_size_t count;
  // Number of result-name tokens that fit in |tokens|.
  iree_host_size_t capacity;
  // Inline storage for the common single-result asm packet.
  loom_token_t inline_tokens[4];
} loom_low_asm_result_names_t;

typedef struct loom_low_asm_value_list_t {
  // SSA value IDs in parsed source order.
  loom_value_id_t* values;
  // Number of value IDs currently stored.
  iree_host_size_t count;
  // Number of value IDs that fit in |values|.
  iree_host_size_t capacity;
  // Inline storage for the common small return packet.
  loom_value_id_t inline_values[4];
} loom_low_asm_value_list_t;

static void loom_low_asm_result_names_initialize(
    loom_low_asm_result_names_t* names) {
  names->tokens = names->inline_tokens;
  names->count = 0;
  names->capacity = IREE_ARRAYSIZE(names->inline_tokens);
}

static iree_status_t loom_low_asm_result_names_append(
    loom_parser_t* parser, loom_low_asm_result_names_t* names,
    loom_token_t token) {
  if (names->count >= names->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        &parser->parser_arena, names->count, /*minimum_capacity=*/0,
        sizeof(*names->tokens), &names->capacity, (void**)&names->tokens));
  }
  names->tokens[names->count++] = token;
  return iree_ok_status();
}

static void loom_low_asm_value_list_initialize(
    loom_low_asm_value_list_t* values) {
  values->values = values->inline_values;
  values->count = 0;
  values->capacity = IREE_ARRAYSIZE(values->inline_values);
}

static iree_status_t loom_low_asm_value_list_append(
    loom_parser_t* parser, loom_low_asm_value_list_t* values,
    loom_value_id_t value_id) {
  if (values->count >= values->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        &parser->parser_arena, values->count, /*minimum_capacity=*/0,
        sizeof(*values->values), &values->capacity, (void**)&values->values));
  }
  values->values[values->count++] = value_id;
  return iree_ok_status();
}

static bool loom_low_asm_token_is_name(loom_token_t token) {
  return token.kind == LOOM_TOKEN_BARE_IDENT ||
         token.kind == LOOM_TOKEN_OP_NAME;
}

static bool loom_low_asm_token_is_on_line(loom_token_t token, uint32_t line) {
  return token.kind != LOOM_TOKEN_EOF && token.kind != LOOM_TOKEN_RBRACE &&
         token.line == line;
}

static bool loom_low_asm_token_is_loc_keyword(loom_token_t token) {
  return token.kind == LOOM_TOKEN_BARE_IDENT &&
         iree_string_view_equal(token.text, IREE_SV("loc"));
}

static bool loom_low_asm_token_can_start_attr(loom_token_t token) {
  switch (token.kind) {
    case LOOM_TOKEN_INTEGER:
    case LOOM_TOKEN_FLOAT:
    case LOOM_TOKEN_STRING:
    case LOOM_TOKEN_SYMBOL:
    case LOOM_TOKEN_BARE_IDENT:
    case LOOM_TOKEN_LBRACKET:
    case LOOM_TOKEN_LBRACE:
    case LOOM_TOKEN_HASH_ATTR:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_parser_emit_low_asm_error(loom_parser_t* parser,
                                                    loom_token_t token,
                                                    iree_string_view_t detail) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(detail),
  };
  return loom_parser_emit(parser,
                          loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 34),
                          params, IREE_ARRAYSIZE(params), token);
}

static iree_status_t loom_parser_emit_low_asm_result_count_mismatch(
    loom_parser_t* parser, loom_token_t mnemonic_token, uint32_t expected_count,
    uint32_t actual_count) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(mnemonic_token.text),
      loom_param_u32(expected_count),
      loom_param_u32(actual_count),
  };
  return loom_parser_emit(parser,
                          loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 9),
                          params, IREE_ARRAYSIZE(params), mnemonic_token);
}

static iree_status_t loom_parser_emit_low_asm_operand_count_mismatch(
    loom_parser_t* parser, loom_token_t mnemonic_token, uint32_t expected_count,
    uint32_t actual_count) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(mnemonic_token.text),
      loom_param_u32(expected_count),
      loom_param_u32(actual_count),
  };
  return loom_parser_emit(parser,
                          loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 10),
                          params, IREE_ARRAYSIZE(params), mnemonic_token);
}

static iree_status_t loom_parse_low_asm_descriptor_set_key(
    loom_parser_t* parser, loom_token_t* out_key_token) {
  if (!loom_tokenizer_try_consume_keyword(&parser->tokenizer, IREE_SV("asm"))) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'asm'"));
  }
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LANGLE, NULL);

  loom_token_t key_token = loom_tokenizer_peek(&parser->tokenizer);
  if (key_token.kind != LOOM_TOKEN_STRING &&
      key_token.kind != LOOM_TOKEN_BARE_IDENT &&
      key_token.kind != LOOM_TOKEN_OP_NAME) {
    return loom_parser_emit_unexpected_token(parser, key_token,
                                             IREE_SV("low descriptor set key"));
  }
  *out_key_token = loom_tokenizer_next(&parser->tokenizer);
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RANGLE, NULL);
  return iree_ok_status();
}

static iree_status_t loom_parse_low_asm_descriptor_set(
    loom_parser_t* parser,
    const loom_text_low_asm_descriptor_set_t** out_descriptor_set) {
  *out_descriptor_set = NULL;

  uint32_t errors_before = parser->error_count;
  loom_token_t key_token = loom_token_none();
  IREE_RETURN_IF_ERROR(
      loom_parse_low_asm_descriptor_set_key(parser, &key_token));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  if (!loom_text_low_asm_environment_is_configured(
          &parser->low_asm_environment)) {
    return loom_parser_emit_low_asm_error(
        parser, key_token, IREE_SV("low asm environment is not configured"));
  }

  IREE_RETURN_IF_ERROR(
      parser->low_asm_environment.vtable->lookup_descriptor_set(
          parser->low_asm_environment.state, key_token.text,
          out_descriptor_set));
  if (*out_descriptor_set == NULL) {
    return loom_parser_emit_low_asm_error(
        parser, key_token, IREE_SV("unknown low descriptor set"));
  }
  return iree_ok_status();
}

static iree_status_t loom_parse_low_asm_result_types(
    loom_parser_t* parser, const loom_text_low_asm_packet_descriptor_t* packet,
    loom_token_t mnemonic_token, const loom_value_id_t* operands,
    iree_host_size_t operand_count, loom_type_t* result_types) {
  const uint16_t result_count = packet->result_count;
  if (result_count == 0) {
    if (loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_COLON)) {
      return loom_parser_emit_low_asm_result_count_mismatch(
          parser, mnemonic_token, /*expected_count=*/0,
          /*actual_count=*/1);
    }
    return iree_ok_status();
  }

  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COLON)) {
    uint32_t errors_before = parser->error_count;
    for (uint16_t i = 0; i < result_count; ++i) {
      iree_string_view_t diagnostic_detail = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(
          parser->low_asm_environment.vtable->infer_result_type(
              parser->low_asm_environment.state, packet, operands,
              operand_count, i, parser->module, &result_types[i],
              &diagnostic_detail));
      if (!iree_string_view_is_empty(diagnostic_detail)) {
        IREE_RETURN_IF_ERROR(loom_parser_emit_low_asm_error(
            parser, mnemonic_token, diagnostic_detail));
      }
      if (parser->error_count > errors_before) {
        return iree_ok_status();
      }
    }
    return iree_ok_status();
  }

  uint32_t errors_before = parser->error_count;
  for (uint16_t i = 0; i < result_count; ++i) {
    if (i > 0) {
      LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COMMA, NULL);
    }
    IREE_RETURN_IF_ERROR(
        loom_parse_type(parser, LOOM_TYPE_PARSE_BODY, &result_types[i]));
    if (parser->error_count > errors_before) {
      return iree_ok_status();
    }
    iree_string_view_t diagnostic_detail = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(
        parser->low_asm_environment.vtable->validate_result_type(
            parser->low_asm_environment.state, packet, operands, operand_count,
            i, parser->module, result_types[i], &diagnostic_detail));
    if (!iree_string_view_is_empty(diagnostic_detail)) {
      IREE_RETURN_IF_ERROR(loom_parser_emit_low_asm_error(
          parser, mnemonic_token, diagnostic_detail));
    }
    if (parser->error_count > errors_before) {
      return iree_ok_status();
    }
  }
  if (loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
    return loom_parser_emit_low_asm_result_count_mismatch(
        parser, mnemonic_token, result_count, (uint32_t)result_count + 1);
  }
  return iree_ok_status();
}

static iree_status_t loom_parse_low_asm_operands(
    loom_parser_t* parser, const loom_text_low_asm_packet_descriptor_t* packet,
    loom_token_t mnemonic_token, loom_value_id_t* operands) {
  const uint16_t operand_count = packet->operand_count;
  for (uint16_t i = 0; i < operand_count; ++i) {
    if (i > 0 &&
        !loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
      return loom_parser_emit_low_asm_operand_count_mismatch(
          parser, mnemonic_token, operand_count, i);
    }
    if (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_SSA_VALUE)) {
      return loom_parser_emit_low_asm_operand_count_mismatch(
          parser, mnemonic_token, operand_count, i);
    }
    loom_token_t operand_token = loom_tokenizer_next(&parser->tokenizer);
    uint32_t errors_before = parser->error_count;
    IREE_RETURN_IF_ERROR(
        loom_parser_resolve_value(parser, operand_token, &operands[i]));
    if (parser->error_count > errors_before) {
      return iree_ok_status();
    }
  }

  loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
  if (operand_count == 0 &&
      loom_low_asm_token_is_on_line(peek, mnemonic_token.line) &&
      peek.kind == LOOM_TOKEN_SSA_VALUE) {
    return loom_parser_emit_low_asm_operand_count_mismatch(
        parser, mnemonic_token, /*expected_count=*/0, /*actual_count=*/1);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_asm_immediate_descriptor(
    loom_parser_t* parser, const loom_text_low_asm_packet_descriptor_t* packet,
    uint16_t immediate_index,
    loom_text_low_asm_immediate_descriptor_t* out_immediate) {
  return parser->low_asm_environment.vtable->immediate_descriptor(
      parser->low_asm_environment.state, packet, immediate_index,
      out_immediate);
}

static iree_status_t loom_low_asm_append_immediate_attr(
    loom_parser_t* parser, iree_string_view_t field_name,
    loom_attribute_t value, loom_named_attr_t* attr_entry) {
  loom_string_id_t field_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(parser->module, field_name, &field_name_id));
  *attr_entry = (loom_named_attr_t){
      .name_id = field_name_id,
      .value = value,
  };
  return iree_ok_status();
}

static iree_status_t loom_parse_low_asm_named_immediates(
    loom_parser_t* parser, const loom_text_low_asm_packet_descriptor_t* packet,
    loom_token_t mnemonic_token, loom_named_attr_t* attrs,
    uint16_t* out_attr_count, loom_parsed_op_t* parsed_spans) {
  loom_token_t dict_token = loom_tokenizer_peek(&parser->tokenizer);
  if (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_LBRACE)) {
    return loom_parser_emit_low_asm_error(
        parser, dict_token, IREE_SV("expected named immediate dictionary"));
  }

  loom_attribute_t dict_attr = {0};
  uint32_t errors_before = parser->error_count;
  IREE_RETURN_IF_ERROR(loom_parse_attr_dict(parser, &dict_attr));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_parsed_op_add_field_span(
      parsed_spans, &parser->parser_arena, LOOM_LOCATION_FIELD_ATTRIBUTE,
      packet->immediate_attribute_field_index, dict_token,
      parser->tokenizer.consumed_end_line,
      parser->tokenizer.consumed_end_column));

  loom_named_attr_slice_t parsed_attrs = loom_attr_as_dict(dict_attr);
  for (iree_host_size_t i = 0; i < parsed_attrs.count; ++i) {
    iree_string_view_t parsed_name =
        parser->module->strings.entries[parsed_attrs.entries[i].name_id];
    bool found = false;
    for (uint16_t j = 0; j < packet->immediate_count; ++j) {
      loom_text_low_asm_immediate_descriptor_t immediate = {0};
      IREE_RETURN_IF_ERROR(
          loom_low_asm_immediate_descriptor(parser, packet, j, &immediate));
      if (iree_string_view_equal(parsed_name, immediate.spelling)) {
        found = true;
        break;
      }
    }
    if (!found) {
      return loom_parser_emit_low_asm_error(
          parser, dict_token, IREE_SV("unexpected named immediate"));
    }
  }

  for (uint16_t i = 0; i < packet->immediate_count; ++i) {
    loom_text_low_asm_immediate_descriptor_t immediate = {0};
    IREE_RETURN_IF_ERROR(
        loom_low_asm_immediate_descriptor(parser, packet, i, &immediate));
    const loom_named_attr_t* parsed_attr = NULL;
    for (iree_host_size_t j = 0; j < parsed_attrs.count; ++j) {
      iree_string_view_t parsed_name =
          parser->module->strings.entries[parsed_attrs.entries[j].name_id];
      if (iree_string_view_equal(parsed_name, immediate.spelling)) {
        parsed_attr = &parsed_attrs.entries[j];
        break;
      }
    }
    if (parsed_attr == NULL) {
      if (immediate.has_default_value) {
        continue;
      }
      return loom_parser_emit_low_asm_error(parser, mnemonic_token,
                                            IREE_SV("missing named immediate"));
    }
    IREE_RETURN_IF_ERROR(loom_low_asm_append_immediate_attr(
        parser, immediate.field_name, parsed_attr->value,
        &attrs[*out_attr_count]));
    ++*out_attr_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_parse_low_asm_positional_immediates(
    loom_parser_t* parser, const loom_text_low_asm_packet_descriptor_t* packet,
    loom_token_t mnemonic_token, loom_named_attr_t* attrs,
    uint16_t* out_attr_count, loom_parsed_op_t* parsed_spans) {
  for (uint16_t i = 0; i < packet->immediate_count; ++i) {
    if ((i > 0 || packet->operand_count > 0) &&
        !loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
      return loom_parser_emit_low_asm_error(parser, mnemonic_token,
                                            IREE_SV("missing immediate"));
    }

    loom_token_t immediate_token = loom_tokenizer_peek(&parser->tokenizer);
    if (!loom_low_asm_token_can_start_attr(immediate_token)) {
      return loom_parser_emit_low_asm_error(parser, immediate_token,
                                            IREE_SV("missing immediate"));
    }

    loom_attribute_t value = {0};
    uint32_t errors_before = parser->error_count;
    IREE_RETURN_IF_ERROR(
        loom_parse_generic_attr_value(parser, /*nesting_depth=*/0, &value));
    if (parser->error_count > errors_before) {
      return iree_ok_status();
    }

    loom_text_low_asm_immediate_descriptor_t immediate = {0};
    IREE_RETURN_IF_ERROR(
        loom_low_asm_immediate_descriptor(parser, packet, i, &immediate));
    IREE_RETURN_IF_ERROR(loom_low_asm_append_immediate_attr(
        parser, immediate.field_name, value, &attrs[*out_attr_count]));
    ++*out_attr_count;
    IREE_RETURN_IF_ERROR(loom_parsed_op_add_field_span(
        parsed_spans, &parser->parser_arena, LOOM_LOCATION_FIELD_ATTRIBUTE,
        packet->immediate_attribute_field_index, immediate_token,
        parser->tokenizer.consumed_end_line,
        parser->tokenizer.consumed_end_column));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_asm_required_immediate_count(
    loom_parser_t* parser, const loom_text_low_asm_packet_descriptor_t* packet,
    uint16_t* out_count) {
  *out_count = 0;
  for (uint16_t i = 0; i < packet->immediate_count; ++i) {
    loom_text_low_asm_immediate_descriptor_t immediate = {0};
    IREE_RETURN_IF_ERROR(
        loom_low_asm_immediate_descriptor(parser, packet, i, &immediate));
    if (!immediate.has_default_value) {
      ++*out_count;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_parse_low_asm_immediates(
    loom_parser_t* parser, const loom_text_low_asm_packet_descriptor_t* packet,
    loom_token_t mnemonic_token, loom_named_attr_t* attrs,
    uint16_t* out_attr_count, loom_parsed_op_t* parsed_spans) {
  *out_attr_count = 0;
  if (packet->immediate_count == 0) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    if (loom_low_asm_token_is_on_line(peek, mnemonic_token.line) &&
        (peek.kind == LOOM_TOKEN_LBRACE || peek.kind == LOOM_TOKEN_COMMA)) {
      return loom_parser_emit_low_asm_error(
          parser, peek, IREE_SV("unexpected immediate syntax"));
    }
    return iree_ok_status();
  }

  if (packet->has_named_immediates) {
    if (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_LBRACE)) {
      uint16_t required_immediate_count = 0;
      IREE_RETURN_IF_ERROR(loom_low_asm_required_immediate_count(
          parser, packet, &required_immediate_count));
      if (required_immediate_count == 0) {
        return iree_ok_status();
      }
    }
    return loom_parse_low_asm_named_immediates(
        parser, packet, mnemonic_token, attrs, out_attr_count, parsed_spans);
  }
  return loom_parse_low_asm_positional_immediates(
      parser, packet, mnemonic_token, attrs, out_attr_count, parsed_spans);
}

static iree_status_t loom_parse_low_asm_lhs(
    loom_parser_t* parser, loom_low_asm_result_names_t* result_names) {
  if (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_SSA_VALUE)) {
    return iree_ok_status();
  }

  for (;;) {
    if (result_names->count > 0) {
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
        break;
      }
    }
    if (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_SSA_VALUE)) {
      break;
    }
    IREE_RETURN_IF_ERROR(loom_low_asm_result_names_append(
        parser, result_names, loom_tokenizer_next(&parser->tokenizer)));
  }

  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_EQUALS)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'='"));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_asm_make_fallback_location(
    loom_parser_t* parser, loom_token_t start_token,
    loom_location_id_t* out_location) {
  *out_location = LOOM_LOCATION_UNKNOWN;
  if (parser->source_id == LOOM_SOURCE_ID_INVALID) {
    return iree_ok_status();
  }
  loom_location_entry_t entry =
      loom_location_file_range(parser->source_id, (uint16_t)start_token.line,
                               (uint16_t)start_token.column,
                               (uint16_t)parser->tokenizer.consumed_end_line,
                               (uint16_t)parser->tokenizer.consumed_end_column);
  return loom_module_add_location(parser->module, entry, out_location);
}

static iree_status_t loom_parse_low_asm_packet_location(
    loom_parser_t* parser, loom_token_t start_token,
    loom_token_t mnemonic_token, loom_parsed_op_t* parsed_spans,
    loom_location_id_t* out_location) {
  loom_location_id_t fallback_location = LOOM_LOCATION_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_low_asm_make_fallback_location(parser, start_token,
                                                           &fallback_location));
  *out_location = fallback_location;

  const uint32_t errors_before = parser->error_count;
  loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
  if (loom_low_asm_token_is_on_line(peek, mnemonic_token.line) &&
      loom_low_asm_token_is_loc_keyword(peek)) {
    IREE_RETURN_IF_ERROR(loom_parse_optional_op_location(
        parser, fallback_location, out_location));
    if (parser->error_count > errors_before) {
      return iree_ok_status();
    }
  }

  peek = loom_tokenizer_peek(&parser->tokenizer);
  if (loom_low_asm_token_is_on_line(peek, mnemonic_token.line)) {
    return loom_parser_emit_low_asm_error(
        parser, peek, IREE_SV("unexpected token after packet"));
  }

  if (*out_location == fallback_location &&
      fallback_location != LOOM_LOCATION_UNKNOWN &&
      parsed_spans->field_span_count > 0) {
    IREE_RETURN_IF_ERROR(loom_module_attach_location_field_spans(
        parser->module, fallback_location, parsed_spans->field_spans,
        parsed_spans->field_span_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_parse_low_asm_return(
    loom_parser_t* parser, const loom_low_asm_result_names_t* result_names,
    loom_token_t start_token, loom_token_t mnemonic_token,
    const iree_string_view_t* comments, iree_host_size_t comment_count,
    loom_parsed_op_t* parsed_spans) {
  const uint32_t errors_before = parser->error_count;
  if (result_names->count != 0) {
    IREE_RETURN_IF_ERROR(loom_parser_emit_low_asm_result_count_mismatch(
        parser, mnemonic_token, /*expected_count=*/0,
        (uint32_t)result_names->count));
    return iree_ok_status();
  }

  loom_low_asm_value_list_t values;
  loom_low_asm_value_list_initialize(&values);
  if (loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_SSA_VALUE)) {
    for (;;) {
      if (values.count > 0) {
        if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
          break;
        }
      }
      if (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_SSA_VALUE)) {
        break;
      }
      loom_token_t value_token = loom_tokenizer_next(&parser->tokenizer);
      loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_parser_resolve_value(parser, value_token, &value_id));
      if (parser->error_count > errors_before) {
        return iree_ok_status();
      }
      IREE_RETURN_IF_ERROR(
          loom_low_asm_value_list_append(parser, &values, value_id));
      IREE_RETURN_IF_ERROR(loom_parsed_op_add_field_span(
          parsed_spans, &parser->parser_arena, LOOM_LOCATION_FIELD_OPERAND,
          (uint16_t)(values.count - 1), value_token,
          parser->tokenizer.consumed_end_line,
          parser->tokenizer.consumed_end_column));
    }
  }

  if (loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COLON)) {
    if (values.count == 0) {
      return loom_parser_emit_low_asm_error(
          parser, mnemonic_token,
          IREE_SV("return type annotation count does not match value count"));
    }
    for (iree_host_size_t i = 0; i < values.count; ++i) {
      if (i > 0) {
        LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COMMA, NULL);
      }
      loom_type_t annotated_type = {0};
      IREE_RETURN_IF_ERROR(
          loom_parse_type(parser, LOOM_TYPE_PARSE_BODY, &annotated_type));
      if (parser->error_count > errors_before) {
        return iree_ok_status();
      }
      loom_type_t actual_type =
          loom_module_value_type(parser->module, values.values[i]);
      if (!loom_type_equal(actual_type, annotated_type)) {
        return loom_parser_emit_low_asm_error(
            parser, mnemonic_token,
            IREE_SV("return type annotation does not match value type"));
      }
    }
    if (loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
      return loom_parser_emit_low_asm_error(
          parser, mnemonic_token,
          IREE_SV("return type annotation count does not match value count"));
    }
  }

  loom_location_id_t location = LOOM_LOCATION_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_packet_location(
      parser, start_token, mnemonic_token, parsed_spans, &location));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(parser->low_asm_environment.vtable->build_return(
      parser->low_asm_environment.state, &parser->builder, values.values,
      values.count, location, &op));
  if (op == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm return builder returned no operation");
  }
  return loom_module_attach_op_comments(parser->module, op, comments,
                                        comment_count);
}

static iree_status_t loom_parse_low_asm_instruction(
    loom_parser_t* parser,
    const loom_text_low_asm_descriptor_set_t* descriptor_set,
    const loom_low_asm_result_names_t* result_names, loom_token_t start_token,
    loom_token_t mnemonic_token, const iree_string_view_t* comments,
    iree_host_size_t comment_count, loom_parsed_op_t* parsed_spans) {
  const uint32_t errors_before = parser->error_count;
  loom_text_low_asm_packet_descriptor_t packet = {0};
  IREE_RETURN_IF_ERROR(parser->low_asm_environment.vtable->lookup_packet(
      parser->low_asm_environment.state, descriptor_set, mnemonic_token.text,
      &packet));
  if (packet.descriptor == NULL) {
    return loom_parser_emit_low_asm_error(parser, mnemonic_token,
                                          IREE_SV("unknown low asm mnemonic"));
  }

  if (result_names->count != packet.result_count) {
    return loom_parser_emit_low_asm_result_count_mismatch(
        parser, mnemonic_token, packet.result_count,
        (uint32_t)result_names->count);
  }

  loom_value_id_t* operands = NULL;
  if (packet.operand_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(&parser->parser_arena, packet.operand_count,
                                  sizeof(*operands), (void**)&operands));
  }
  IREE_RETURN_IF_ERROR(
      loom_parse_low_asm_operands(parser, &packet, mnemonic_token, operands));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_named_attr_t* attrs = NULL;
  if (packet.immediate_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(&parser->parser_arena, packet.immediate_count,
                                  sizeof(*attrs), (void**)&attrs));
  }
  uint16_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_immediates(
      parser, &packet, mnemonic_token, attrs, &attr_count, parsed_spans));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_type_t* result_types = NULL;
  if (packet.result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &parser->parser_arena, packet.result_count, sizeof(*result_types),
        (void**)&result_types));
  }
  IREE_RETURN_IF_ERROR(
      loom_parse_low_asm_result_types(parser, &packet, mnemonic_token, operands,
                                      packet.operand_count, result_types));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_location_id_t location = LOOM_LOCATION_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_packet_location(
      parser, start_token, mnemonic_token, parsed_spans, &location));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_op_t* op = NULL;
  const loom_named_attr_slice_t attr_slice =
      loom_make_named_attr_slice(attrs, attr_count);
  IREE_RETURN_IF_ERROR(parser->low_asm_environment.vtable->build_packet(
      parser->low_asm_environment.state, &parser->builder, &packet, operands,
      packet.operand_count, attr_slice, result_types, packet.result_count,
      location, &op));
  if (op == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm packet builder returned no operation");
  }

  for (iree_host_size_t i = 0; i < result_names->count; ++i) {
    IREE_RETURN_IF_ERROR(loom_parser_define_value_name(
        parser, result_names->tokens[i], loom_op_results(op)[i]));
    if (parser->error_count > errors_before) {
      return iree_ok_status();
    }
  }
  return loom_module_attach_op_comments(parser->module, op, comments,
                                        comment_count);
}

static bool loom_low_asm_structural_kind_from_token(
    loom_token_t token, loom_text_low_asm_structural_kind_t* out_kind) {
  *out_kind = LOOM_TEXT_LOW_ASM_STRUCTURAL_UNKNOWN;
  if (token.kind != LOOM_TOKEN_BARE_IDENT) {
    return false;
  }
  if (iree_string_view_equal(token.text, IREE_SV("resource"))) {
    *out_kind = LOOM_TEXT_LOW_ASM_STRUCTURAL_RESOURCE;
    return true;
  }
  if (iree_string_view_equal(token.text, IREE_SV("live_in"))) {
    *out_kind = LOOM_TEXT_LOW_ASM_STRUCTURAL_LIVE_IN;
    return true;
  }
  if (iree_string_view_equal(token.text, IREE_SV("concat"))) {
    *out_kind = LOOM_TEXT_LOW_ASM_STRUCTURAL_CONCAT;
    return true;
  }
  if (iree_string_view_equal(token.text, IREE_SV("slice"))) {
    *out_kind = LOOM_TEXT_LOW_ASM_STRUCTURAL_SLICE;
    return true;
  }
  if (iree_string_view_equal(token.text, IREE_SV("frame_index"))) {
    *out_kind = LOOM_TEXT_LOW_ASM_STRUCTURAL_FRAME_INDEX;
    return true;
  }
  if (iree_string_view_equal(token.text, IREE_SV("copy"))) {
    *out_kind = LOOM_TEXT_LOW_ASM_STRUCTURAL_COPY;
    return true;
  }
  return false;
}

static iree_status_t loom_parse_low_asm_expect_single_result(
    loom_parser_t* parser, const loom_low_asm_result_names_t* result_names,
    loom_token_t mnemonic_token) {
  if (result_names->count != 1) {
    return loom_parser_emit_low_asm_result_count_mismatch(
        parser, mnemonic_token, /*expected_count=*/1,
        (uint32_t)result_names->count);
  }
  return iree_ok_status();
}

static iree_status_t loom_parse_low_asm_define_single_result(
    loom_parser_t* parser, const loom_low_asm_result_names_t* result_names,
    loom_op_t* op) {
  if (op->result_count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm structural builder returned %u results",
                            op->result_count);
  }
  return loom_parser_define_value_name(parser, result_names->tokens[0],
                                       loom_op_results(op)[0]);
}

static iree_status_t loom_parse_low_asm_angle_key(loom_parser_t* parser,
                                                  loom_token_t mnemonic_token,
                                                  loom_token_t* out_key_token) {
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LANGLE, NULL);
  loom_token_t key_token = loom_tokenizer_peek(&parser->tokenizer);
  if (key_token.kind != LOOM_TOKEN_STRING &&
      key_token.kind != LOOM_TOKEN_BARE_IDENT &&
      key_token.kind != LOOM_TOKEN_OP_NAME) {
    return loom_parser_emit_low_asm_error(
        parser, key_token, IREE_SV("expected structural intrinsic key"));
  }
  *out_key_token = loom_tokenizer_next(&parser->tokenizer);
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RANGLE)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'>'"));
  }
  (void)mnemonic_token;
  return iree_ok_status();
}

static iree_status_t loom_parse_low_asm_result_type(
    loom_parser_t* parser, loom_type_t* out_result_type) {
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COLON, NULL);
  return loom_parse_type(parser, LOOM_TYPE_PARSE_BODY, out_result_type);
}

static iree_status_t loom_parse_low_asm_arrow_result_type(
    loom_parser_t* parser, loom_type_t* out_result_type) {
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_ARROW, NULL);
  return loom_parse_type(parser, LOOM_TYPE_PARSE_BODY, out_result_type);
}

static iree_status_t loom_parse_low_asm_validate_operand_type(
    loom_parser_t* parser, loom_token_t mnemonic_token, loom_value_id_t value,
    loom_type_t annotated_type) {
  loom_type_t actual_type = loom_module_value_type(parser->module, value);
  if (!loom_type_equal(actual_type, annotated_type)) {
    return loom_parser_emit_low_asm_error(
        parser, mnemonic_token,
        IREE_SV("operand type annotation does not match value type"));
  }
  return iree_ok_status();
}

static iree_status_t loom_parse_low_asm_value_ref(
    loom_parser_t* parser, loom_token_t mnemonic_token,
    loom_parsed_op_t* parsed_spans, uint16_t operand_index,
    loom_value_id_t* out_value) {
  if (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_SSA_VALUE)) {
    return loom_parser_emit_low_asm_operand_count_mismatch(
        parser, mnemonic_token, /*expected_count=*/1, /*actual_count=*/0);
  }
  loom_token_t value_token = loom_tokenizer_next(&parser->tokenizer);
  uint32_t errors_before = parser->error_count;
  IREE_RETURN_IF_ERROR(
      loom_parser_resolve_value(parser, value_token, out_value));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }
  return loom_parsed_op_add_field_span(
      parsed_spans, &parser->parser_arena, LOOM_LOCATION_FIELD_OPERAND,
      operand_index, value_token, parser->tokenizer.consumed_end_line,
      parser->tokenizer.consumed_end_column);
}

static iree_status_t loom_parse_low_asm_value_list(
    loom_parser_t* parser, loom_token_t mnemonic_token,
    loom_parsed_op_t* parsed_spans, loom_low_asm_value_list_t* values) {
  const uint32_t errors_before = parser->error_count;
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LPAREN, NULL);
  if (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RPAREN)) {
    for (;;) {
      if (values->count > UINT16_MAX) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "low asm value list exceeds uint16_t range");
      }
      loom_value_id_t value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_parse_low_asm_value_ref(parser, mnemonic_token, parsed_spans,
                                       (uint16_t)values->count, &value));
      if (parser->error_count > errors_before) {
        return iree_ok_status();
      }
      IREE_RETURN_IF_ERROR(
          loom_low_asm_value_list_append(parser, values, value));
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
        break;
      }
    }
  }
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RPAREN, NULL);
  return iree_ok_status();
}

static iree_status_t loom_parse_low_asm_operand_type_list(
    loom_parser_t* parser, loom_token_t mnemonic_token,
    const loom_low_asm_value_list_t* values) {
  const uint32_t errors_before = parser->error_count;
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COLON, NULL);
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LPAREN, NULL);
  for (iree_host_size_t i = 0; i < values->count; ++i) {
    if (i > 0) {
      LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COMMA, NULL);
    }
    loom_type_t type = {0};
    IREE_RETURN_IF_ERROR(loom_parse_type(parser, LOOM_TYPE_PARSE_BODY, &type));
    if (parser->error_count > errors_before) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_parse_low_asm_validate_operand_type(
        parser, mnemonic_token, values->values[i], type));
  }
  if (values->count == 0 &&
      !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RPAREN)) {
    return loom_parser_emit_low_asm_operand_count_mismatch(
        parser, mnemonic_token, /*expected_count=*/0, /*actual_count=*/1);
  }
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RPAREN, NULL);
  return iree_ok_status();
}

static iree_status_t loom_parse_low_asm_structural_attr_dict(
    loom_parser_t* parser, loom_text_low_asm_structural_kind_t kind,
    loom_token_t mnemonic_token, loom_named_attr_slice_t* out_attrs) {
  *out_attrs = loom_named_attr_slice_empty();
  if (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_LBRACE)) {
    return iree_ok_status();
  }

  loom_token_t open_brace_token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LBRACE, &open_brace_token);
  (void)open_brace_token;
  uint32_t errors_before = parser->error_count;
  loom_parsed_attr_dict_entry_t stack_entries[16];
  uint16_t count = 0;

  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RBRACE) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (count > 0) {
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
        break;
      }
    }
    if (count >= IREE_ARRAYSIZE(stack_entries)) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_token_text_error(
          parser, loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 4), peek);
    }

    loom_token_t key_token = loom_token_none();
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_BARE_IDENT, &key_token);
    loom_string_id_t key_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_module_intern_string(parser->module, key_token.text, &key_id));
    for (uint16_t i = 0; i < count; ++i) {
      if (stack_entries[i].attr.name_id == key_id) {
        return loom_parser_emit_duplicate_attr_dict_key(
            parser, key_token, stack_entries[i].key_token);
      }
    }

    const loom_attr_descriptor_t* descriptor = NULL;
    IREE_RETURN_IF_ERROR(
        parser->low_asm_environment.vtable->structural_attr_descriptor(
            parser->low_asm_environment.state, kind, key_token.text,
            &descriptor));
    if (descriptor == NULL) {
      return loom_parser_emit_low_asm_error(
          parser, key_token, IREE_SV("unexpected structural attribute"));
    }

    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_EQUALS, NULL);

    loom_attribute_t value = {0};
    IREE_RETURN_IF_ERROR(loom_parse_attr_value(parser, descriptor, &value));
    if (parser->error_count > errors_before) {
      return iree_ok_status();
    }

    stack_entries[count].attr.name_id = key_id;
    stack_entries[count].attr.reserved = 0;
    stack_entries[count].attr.value = value;
    stack_entries[count].key_token = key_token;
    ++count;
  }

  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RBRACE, NULL);
  if (count == 0) {
    return iree_ok_status();
  }

  loom_parser_sort_attr_dict_entries(parser->module, stack_entries, count);
  loom_named_attr_t* attrs = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &parser->parser_arena, count, sizeof(*attrs), (void**)&attrs));
  for (uint16_t i = 0; i < count; ++i) {
    attrs[i] = stack_entries[i].attr;
  }
  *out_attrs = loom_make_named_attr_slice(attrs, count);
  (void)mnemonic_token;
  return iree_ok_status();
}

static iree_status_t loom_parse_low_asm_structural_location_and_build(
    loom_parser_t* parser, loom_text_low_asm_structural_kind_t kind,
    const loom_low_asm_result_names_t* result_names, loom_token_t start_token,
    loom_token_t mnemonic_token, const iree_string_view_t* comments,
    iree_host_size_t comment_count, const loom_value_id_t* operands,
    iree_host_size_t operand_count, loom_named_attr_slice_t attrs,
    iree_string_view_t key, int64_t offset, loom_type_t result_type,
    loom_parsed_op_t* parsed_spans) {
  const uint32_t errors_before = parser->error_count;
  loom_location_id_t location = LOOM_LOCATION_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_packet_location(
      parser, start_token, mnemonic_token, parsed_spans, &location));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(parser->low_asm_environment.vtable->build_structural(
      parser->low_asm_environment.state, &parser->builder, kind, key, operands,
      operand_count, attrs, offset, result_type, location, &op));
  if (op == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm structural builder returned no operation");
  }
  IREE_RETURN_IF_ERROR(
      loom_parse_low_asm_define_single_result(parser, result_names, op));
  return loom_module_attach_op_comments(parser->module, op, comments,
                                        comment_count);
}

static iree_status_t loom_parse_low_asm_structural_resource_or_live_in(
    loom_parser_t* parser, loom_text_low_asm_structural_kind_t kind,
    const loom_low_asm_result_names_t* result_names, loom_token_t start_token,
    loom_token_t mnemonic_token, const iree_string_view_t* comments,
    iree_host_size_t comment_count, loom_parsed_op_t* parsed_spans) {
  const uint32_t errors_before = parser->error_count;
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_expect_single_result(
      parser, result_names, mnemonic_token));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_token_t key_token = loom_token_none();
  IREE_RETURN_IF_ERROR(
      loom_parse_low_asm_angle_key(parser, mnemonic_token, &key_token));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_named_attr_slice_t attrs = loom_named_attr_slice_empty();
  if (kind == LOOM_TEXT_LOW_ASM_STRUCTURAL_RESOURCE) {
    IREE_RETURN_IF_ERROR(loom_parse_low_asm_structural_attr_dict(
        parser, kind, mnemonic_token, &attrs));
  } else {
    loom_attribute_t attr = {0};
    IREE_RETURN_IF_ERROR(loom_parse_attr_dict(parser, &attr));
    attrs = loom_attr_as_dict(attr);
  }
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_type_t result_type = {0};
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_result_type(parser, &result_type));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  return loom_parse_low_asm_structural_location_and_build(
      parser, kind, result_names, start_token, mnemonic_token, comments,
      comment_count, NULL, 0, attrs, key_token.text, 0, result_type,
      parsed_spans);
}

static iree_status_t loom_parse_low_asm_structural_concat(
    loom_parser_t* parser, const loom_low_asm_result_names_t* result_names,
    loom_token_t start_token, loom_token_t mnemonic_token,
    const iree_string_view_t* comments, iree_host_size_t comment_count,
    loom_parsed_op_t* parsed_spans) {
  const uint32_t errors_before = parser->error_count;
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_expect_single_result(
      parser, result_names, mnemonic_token));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_low_asm_value_list_t operands;
  loom_low_asm_value_list_initialize(&operands);
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_value_list(parser, mnemonic_token,
                                                     parsed_spans, &operands));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_parse_low_asm_operand_type_list(parser, mnemonic_token, &operands));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_type_t result_type = {0};
  IREE_RETURN_IF_ERROR(
      loom_parse_low_asm_arrow_result_type(parser, &result_type));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  return loom_parse_low_asm_structural_location_and_build(
      parser, LOOM_TEXT_LOW_ASM_STRUCTURAL_CONCAT, result_names, start_token,
      mnemonic_token, comments, comment_count, operands.values, operands.count,
      loom_named_attr_slice_empty(), iree_string_view_empty(), 0, result_type,
      parsed_spans);
}

static iree_status_t loom_parse_low_asm_structural_slice(
    loom_parser_t* parser, const loom_low_asm_result_names_t* result_names,
    loom_token_t start_token, loom_token_t mnemonic_token,
    const iree_string_view_t* comments, iree_host_size_t comment_count,
    loom_parsed_op_t* parsed_spans) {
  const uint32_t errors_before = parser->error_count;
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_expect_single_result(
      parser, result_names, mnemonic_token));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_value_id_t source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_value_ref(
      parser, mnemonic_token, parsed_spans, /*operand_index=*/0, &source));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LBRACKET, NULL);
  loom_token_t offset_token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_INTEGER, &offset_token);
  int64_t offset = 0;
  if (!iree_string_view_atoi_int64(offset_token.text, &offset)) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(offset_token.text),
    };
    return loom_parser_emit(parser,
                            loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 15),
                            params, IREE_ARRAYSIZE(params), offset_token);
  }
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RBRACKET, NULL);

  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COLON, NULL);
  loom_type_t source_type = {0};
  IREE_RETURN_IF_ERROR(
      loom_parse_type(parser, LOOM_TYPE_PARSE_BODY, &source_type));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_validate_operand_type(
      parser, mnemonic_token, source, source_type));

  loom_type_t result_type = {0};
  IREE_RETURN_IF_ERROR(
      loom_parse_low_asm_arrow_result_type(parser, &result_type));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  return loom_parse_low_asm_structural_location_and_build(
      parser, LOOM_TEXT_LOW_ASM_STRUCTURAL_SLICE, result_names, start_token,
      mnemonic_token, comments, comment_count, &source, 1,
      loom_named_attr_slice_empty(), iree_string_view_empty(), offset,
      result_type, parsed_spans);
}

static iree_status_t loom_parse_low_asm_structural_copy(
    loom_parser_t* parser, const loom_low_asm_result_names_t* result_names,
    loom_token_t start_token, loom_token_t mnemonic_token,
    const iree_string_view_t* comments, iree_host_size_t comment_count,
    loom_parsed_op_t* parsed_spans) {
  const uint32_t errors_before = parser->error_count;
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_expect_single_result(
      parser, result_names, mnemonic_token));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_value_id_t source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_value_ref(
      parser, mnemonic_token, parsed_spans, /*operand_index=*/0, &source));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COLON, NULL);
  loom_type_t source_type = {0};
  IREE_RETURN_IF_ERROR(
      loom_parse_type(parser, LOOM_TYPE_PARSE_BODY, &source_type));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_validate_operand_type(
      parser, mnemonic_token, source, source_type));

  loom_type_t result_type = {0};
  IREE_RETURN_IF_ERROR(
      loom_parse_low_asm_arrow_result_type(parser, &result_type));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  return loom_parse_low_asm_structural_location_and_build(
      parser, LOOM_TEXT_LOW_ASM_STRUCTURAL_COPY, result_names, start_token,
      mnemonic_token, comments, comment_count, &source, 1,
      loom_named_attr_slice_empty(), iree_string_view_empty(), 0, result_type,
      parsed_spans);
}

static iree_status_t loom_parse_low_asm_structural_frame_index(
    loom_parser_t* parser, const loom_low_asm_result_names_t* result_names,
    loom_token_t start_token, loom_token_t mnemonic_token,
    const iree_string_view_t* comments, iree_host_size_t comment_count,
    loom_parsed_op_t* parsed_spans) {
  const uint32_t errors_before = parser->error_count;
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_expect_single_result(
      parser, result_names, mnemonic_token));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_token_t slot_token = loom_tokenizer_peek(&parser->tokenizer);
  loom_attribute_t slot_attr = {0};
  IREE_RETURN_IF_ERROR(loom_parse_symbol_ref_attr(parser, &slot_attr));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_named_attr_slice_t offset_attrs = loom_named_attr_slice_empty();
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_structural_attr_dict(
      parser, LOOM_TEXT_LOW_ASM_STRUCTURAL_FRAME_INDEX, mnemonic_token,
      &offset_attrs));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_type_t result_type = {0};
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_result_type(parser, &result_type));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_named_attr_t* attrs = NULL;
  const iree_host_size_t attr_count = offset_attrs.count + 1;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &parser->parser_arena, attr_count, sizeof(*attrs), (void**)&attrs));
  loom_string_id_t slot_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      parser->module, IREE_SV("slot"), &slot_name_id));
  attrs[0] = (loom_named_attr_t){
      .name_id = slot_name_id,
      .reserved = 0,
      .value = slot_attr,
  };
  for (iree_host_size_t i = 0; i < offset_attrs.count; ++i) {
    attrs[i + 1] = offset_attrs.entries[i];
  }

  loom_parsed_attr_dict_entry_t stack_entries[2];
  for (iree_host_size_t i = 0; i < attr_count; ++i) {
    stack_entries[i] = (loom_parsed_attr_dict_entry_t){
        .attr = attrs[i],
        .key_token = i == 0 ? slot_token : loom_token_none(),
    };
  }
  loom_parser_sort_attr_dict_entries(parser->module, stack_entries,
                                     (uint16_t)attr_count);
  for (iree_host_size_t i = 0; i < attr_count; ++i) {
    attrs[i] = stack_entries[i].attr;
  }

  return loom_parse_low_asm_structural_location_and_build(
      parser, LOOM_TEXT_LOW_ASM_STRUCTURAL_FRAME_INDEX, result_names,
      start_token, mnemonic_token, comments, comment_count, NULL, 0,
      loom_make_named_attr_slice(attrs, attr_count), iree_string_view_empty(),
      0, result_type, parsed_spans);
}

static iree_status_t loom_parse_low_asm_structural(
    loom_parser_t* parser, loom_text_low_asm_structural_kind_t kind,
    const loom_low_asm_result_names_t* result_names, loom_token_t start_token,
    loom_token_t mnemonic_token, const iree_string_view_t* comments,
    iree_host_size_t comment_count, loom_parsed_op_t* parsed_spans) {
  switch (kind) {
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_RESOURCE:
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_LIVE_IN:
      return loom_parse_low_asm_structural_resource_or_live_in(
          parser, kind, result_names, start_token, mnemonic_token, comments,
          comment_count, parsed_spans);
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_CONCAT:
      return loom_parse_low_asm_structural_concat(
          parser, result_names, start_token, mnemonic_token, comments,
          comment_count, parsed_spans);
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_SLICE:
      return loom_parse_low_asm_structural_slice(
          parser, result_names, start_token, mnemonic_token, comments,
          comment_count, parsed_spans);
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_FRAME_INDEX:
      return loom_parse_low_asm_structural_frame_index(
          parser, result_names, start_token, mnemonic_token, comments,
          comment_count, parsed_spans);
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_COPY:
      return loom_parse_low_asm_structural_copy(
          parser, result_names, start_token, mnemonic_token, comments,
          comment_count, parsed_spans);
    default:
      return loom_parser_emit_low_asm_error(
          parser, mnemonic_token, IREE_SV("unknown structural intrinsic"));
  }
}

static iree_status_t loom_parse_low_asm_packet_impl(
    loom_parser_t* parser,
    const loom_text_low_asm_descriptor_set_t* descriptor_set,
    loom_token_t start_token, const iree_string_view_t* comments,
    iree_host_size_t comment_count, loom_parsed_op_t* parsed_spans) {
  const uint32_t errors_before = parser->error_count;
  loom_low_asm_result_names_t result_names;
  loom_low_asm_result_names_initialize(&result_names);
  IREE_RETURN_IF_ERROR(loom_parse_low_asm_lhs(parser, &result_names));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_token_t mnemonic_token = loom_tokenizer_peek(&parser->tokenizer);
  if (!loom_low_asm_token_is_name(mnemonic_token)) {
    return loom_parser_emit_unexpected_token(parser, mnemonic_token,
                                             IREE_SV("low asm mnemonic"));
  }
  mnemonic_token = loom_tokenizer_next(&parser->tokenizer);
  IREE_RETURN_IF_ERROR(loom_parsed_op_add_field_span(
      parsed_spans, &parser->parser_arena, LOOM_LOCATION_FIELD_ATTRIBUTE, 0,
      mnemonic_token, parser->tokenizer.consumed_end_line,
      parser->tokenizer.consumed_end_column));

  if (iree_string_view_equal(mnemonic_token.text, IREE_SV("return"))) {
    return loom_parse_low_asm_return(parser, &result_names, start_token,
                                     mnemonic_token, comments, comment_count,
                                     parsed_spans);
  }

  loom_text_low_asm_structural_kind_t structural_kind =
      LOOM_TEXT_LOW_ASM_STRUCTURAL_UNKNOWN;
  if (loom_low_asm_structural_kind_from_token(mnemonic_token,
                                              &structural_kind)) {
    return loom_parse_low_asm_structural(parser, structural_kind, &result_names,
                                         start_token, mnemonic_token, comments,
                                         comment_count, parsed_spans);
  }

  return loom_parse_low_asm_instruction(parser, descriptor_set, &result_names,
                                        start_token, mnemonic_token, comments,
                                        comment_count, parsed_spans);
}

static iree_status_t loom_parse_low_asm_packet(
    loom_parser_t* parser,
    const loom_text_low_asm_descriptor_set_t* descriptor_set) {
  loom_token_t start_token = loom_tokenizer_peek(&parser->tokenizer);
  const iree_string_view_t* comments = NULL;
  iree_host_size_t comment_count = 0;
  loom_tokenizer_take_pending_comments(&parser->tokenizer, &comments,
                                       &comment_count);

  loom_parsed_op_t* parsed_spans = NULL;
  IREE_RETURN_IF_ERROR(loom_parser_acquire_parsed_op(parser, &parsed_spans));
  iree_status_t status =
      loom_parse_low_asm_packet_impl(parser, descriptor_set, start_token,
                                     comments, comment_count, parsed_spans);
  loom_parser_release_parsed_op(parser, parsed_spans);
  return status;
}

static iree_status_t loom_parse_low_asm_region_body(
    loom_parser_t* parser, const loom_region_descriptor_t* region_descriptor,
    loom_region_t* region, const void* user_data,
    bool* out_region_end_consumed) {
  *out_region_end_consumed = false;
  const loom_text_low_asm_descriptor_set_t* descriptor_set =
      (const loom_text_low_asm_descriptor_set_t*)user_data;
  IREE_ASSERT_ARGUMENT(descriptor_set);

  IREE_RETURN_IF_ERROR(loom_parser_seed_region_entry_block(parser, region));

  loom_block_t* entry_block = loom_region_entry_block(region);
  loom_builder_set_block(&parser->builder, entry_block);
  const uint32_t block_errors_before = parser->error_count;
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RBRACE) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (loom_parser_at_error_limit(parser)) {
      break;
    }

    if (loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_BLOCK_LABEL)) {
      loom_token_t label_token = loom_tokenizer_next(&parser->tokenizer);
      IREE_RETURN_IF_ERROR(loom_parser_emit_low_asm_error(
          parser, label_token,
          IREE_SV("block labels are not supported in straight-line low asm")));
      loom_parser_sync_to_newline(parser);
      continue;
    }

    const uint32_t packet_errors_before = parser->error_count;
    IREE_RETURN_IF_ERROR(loom_parse_low_asm_packet(parser, descriptor_set));
    if (parser->error_count > packet_errors_before) {
      loom_parser_sync_to_newline(parser);
    }
  }

  if (parser->error_count == block_errors_before) {
    IREE_RETURN_IF_ERROR(loom_parser_append_implicit_terminator(
        parser, region_descriptor, entry_block));
  }

  loom_tokenizer_discard_pending_comments(&parser->tokenizer);
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RBRACE, NULL);
  *out_region_end_consumed = true;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Target-low asm region parsing
//===----------------------------------------------------------------------===//

iree_status_t loom_parse_low_asm_prefixed_region(
    loom_parser_t* parser, const loom_region_descriptor_t* region_descriptor,
    loom_region_t** out_region) {
  uint32_t errors_before = parser->error_count;

  const loom_text_low_asm_descriptor_set_t* descriptor_set = NULL;
  IREE_RETURN_IF_ERROR(
      loom_parse_low_asm_descriptor_set(parser, &descriptor_set));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }

  loom_parse_region_body_callback_t body = {
      .fn = loom_parse_low_asm_region_body,
      .user_data = descriptor_set,
  };
  return loom_parse_braced_region_with_body(parser, region_descriptor, body,
                                            out_region);
}
