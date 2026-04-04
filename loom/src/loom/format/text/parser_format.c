// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "loom/format/text/parser_internal.h"

//===----------------------------------------------------------------------===//
// Format element parsers
//===----------------------------------------------------------------------===//

// Builds the parser-owned result overlay so result type annotations can
// reference co-results by name. The overlay is reset by the caller once the
// current RESULT_TYPE* element has been parsed.
static iree_status_t loom_parse_format_prepare_result_scope(
    loom_parser_t* parser, const loom_parsed_op_t* parsed) {
  iree_host_size_t named_result_count = 0;
  for (uint16_t result_index = 0; result_index < parsed->result_count;
       ++result_index) {
    loom_value_id_t value_id = parsed->result_ids[result_index];
    loom_string_id_t name_id = parser->module->values.entries[value_id].name_id;
    if (name_id != LOOM_STRING_ID_INVALID &&
        name_id < parser->module->strings.count) {
      ++named_result_count;
    }
  }
  if (named_result_count == 0) {
    loom_parser_result_scope_reset(&parser->result_scope);
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_parser_result_scope_prepare(
      &parser->result_scope, named_result_count, &parser->parser_arena));
  for (uint16_t result_index = 0; result_index < parsed->result_count;
       ++result_index) {
    loom_value_id_t value_id = parsed->result_ids[result_index];
    loom_string_id_t name_id = parser->module->values.entries[value_id].name_id;
    if (name_id == LOOM_STRING_ID_INVALID ||
        name_id >= parser->module->strings.count) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_parser_result_scope_define(
        &parser->result_scope, name_id, value_id, /*out_duplicate=*/NULL));
  }
  return iree_ok_status();
}

static iree_status_t loom_parse_format_assign_lhs_result_type(
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
    loom_token_t op_name_token, loom_parsed_op_t* parsed, uint16_t result_index,
    loom_type_t type) {
  if (result_index >= parsed->result_count) {
    return loom_parser_emit_result_count_mismatch(parser, vtable, op_name_token,
                                                  (uint16_t)(result_index + 1),
                                                  parsed->result_count);
  }
  parser->module->values.entries[parsed->result_ids[result_index]].type = type;
  return iree_ok_status();
}

static iree_status_t loom_parse_format_append_symbol_result(
    loom_parser_t* parser, loom_parsed_op_t* parsed, loom_type_t type,
    loom_token_t name_token) {
  loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_parser_define_value(parser, name_token, type, &value_id));
  if (parser->error_count > 0) return iree_ok_status();
  return loom_parsed_op_add_result(parsed, &parser->parser_arena, value_id,
                                   name_token);
}

static iree_status_t loom_parse_format_resolve_tied_result_operand(
    loom_parser_t* parser, const loom_parsed_op_t* parsed,
    loom_token_t ssa_token, uint16_t* out_operand_index) {
  loom_value_id_t operand_id = LOOM_VALUE_ID_INVALID;
  LOOM_PARSE_RESOLVE_VALUE(parser, ssa_token, &operand_id);

  for (uint16_t operand_index = 0; operand_index < parsed->operand_count;
       ++operand_index) {
    if (parsed->operand_ids[operand_index] == operand_id) {
      *out_operand_index = operand_index;
      return iree_ok_status();
    }
  }

  loom_diagnostic_param_t params[] = {
      loom_param_string(ssa_token.text),
  };
  return loom_parser_emit(parser, &loom_err_parse_001, params,
                          IREE_ARRAYSIZE(params), ssa_token);
}

static iree_status_t loom_parse_format_resolve_symbol_tied_result_operand(
    loom_parser_t* parser, const loom_parsed_op_t* parsed,
    loom_token_t ssa_token, uint16_t* out_operand_index) {
  loom_value_id_t operand_id = LOOM_VALUE_ID_INVALID;
  LOOM_PARSE_RESOLVE_VALUE(parser, ssa_token, &operand_id);

  for (uint16_t operand_index = 0; operand_index < parsed->operand_count;
       ++operand_index) {
    if (parsed->operand_ids[operand_index] == operand_id) {
      *out_operand_index = operand_index;
      return iree_ok_status();
    }
  }
  for (uint16_t arg_index = 0; arg_index < parser->pending_block_args.count;
       ++arg_index) {
    if (parser->pending_block_args.entries[arg_index].value_id == operand_id) {
      *out_operand_index = (uint16_t)(parsed->operand_count + arg_index);
      return iree_ok_status();
    }
  }

  loom_diagnostic_param_t params[] = {
      loom_param_string(ssa_token.text),
  };
  return loom_parser_emit(parser, &loom_err_parse_001, params,
                          IREE_ARRAYSIZE(params), ssa_token);
}

// Parses a body-op result type list: (type, %operand as type, ...).
// Result names are preallocated on the LHS and must match the number of
// parsed type entries.
static iree_status_t loom_parse_format_lhs_result_type_list(
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
    loom_token_t op_name_token, const loom_format_element_t* element,
    loom_parsed_op_t* parsed) {
  uint32_t errors_before = parser->error_count;
  bool use_parens = (element->data & LOOM_RESULT_TYPE_LIST_PARENS) != 0;
  if (use_parens) {
    if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LPAREN)) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'('"));
    }
  }
  // Result name forward references are handled by parser->result_scope, not by
  // the type parse mode.
  // Body result type lists use BODY mode — unknown names are errors.
  loom_type_parse_mode_t type_mode = LOOM_TYPE_PARSE_BODY;
  uint16_t result_index = 0;
  while ((!use_parens ||
          !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RPAREN)) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (result_index > 0) {
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
        break;
      }
    }

    // Check for tied result: %operand as type.
    if (loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_SSA_VALUE)) {
      loom_token_t ssa_token = loom_tokenizer_peek(&parser->tokenizer);
      loom_tokenizer_next(&parser->tokenizer);
      if (loom_tokenizer_try_consume_keyword(&parser->tokenizer,
                                             IREE_SV("as"))) {
        // Tied result: %operand as type.
        uint16_t operand_index = UINT16_MAX;
        IREE_RETURN_IF_ERROR(loom_parse_format_resolve_tied_result_operand(
            parser, parsed, ssa_token, &operand_index));
        if (parser->error_count > errors_before) return iree_ok_status();

        loom_type_t type = {0};
        IREE_RETURN_IF_ERROR(loom_parse_type(parser, type_mode, &type));

        // Check if the result type differs from the operand type.
        loom_value_id_t operand_id = parsed->operand_ids[operand_index];
        loom_type_t operand_type =
            loom_module_value_type(parser->module, operand_id);
        loom_tied_result_t tied = {
            .result_index = result_index,
            .operand_index = operand_index,
            .has_type_change = !loom_type_equal(operand_type, type),
        };
        IREE_RETURN_IF_ERROR(loom_parsed_op_add_tied_result(
            parsed, &parser->parser_arena, tied));

        IREE_RETURN_IF_ERROR(loom_parse_format_assign_lhs_result_type(
            parser, vtable, op_name_token, parsed, result_index, type));
      } else {
        // Not tied — the SSA token was actually consumed. This
        // shouldn't normally happen in RESULT_TYPE_LIST (results
        // are types, not values), but handle gracefully.
        return loom_parser_emit_unexpected_token(
            parser, ssa_token, IREE_SV("a result type or '%operand as type'"));
      }
    } else {
      // Regular type.
      loom_type_t type = {0};
      IREE_RETURN_IF_ERROR(loom_parse_type(parser, type_mode, &type));
      IREE_RETURN_IF_ERROR(loom_parse_format_assign_lhs_result_type(
          parser, vtable, op_name_token, parsed, result_index, type));
    }
    if (parser->error_count > errors_before) return iree_ok_status();
    ++result_index;
  }
  // Consume the closing ')' if parenthesized.
  if (use_parens) {
    if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RPAREN)) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("')'"));
    }
  }
  if (result_index < parsed->result_count) {
    return loom_parser_emit_result_count_mismatch(
        parser, vtable, op_name_token, parsed->result_count, result_index);
  }
  return iree_ok_status();
}

// Parses a symbol-definition result type list:
//   (type, %name: type, %arg as type, ...)
//
// Result values are created as each type is parsed. Named result values are
// local to the surrounding Scope(...) and become visible only to subsequent
// result types / predicates in the same signature.
static iree_status_t loom_parse_format_symbol_result_type_list(
    loom_parser_t* parser, const loom_format_element_t* element,
    loom_parsed_op_t* parsed) {
  uint32_t errors_before = parser->error_count;
  bool use_parens = (element->data & LOOM_RESULT_TYPE_LIST_PARENS) != 0;
  if (use_parens) {
    if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LPAREN)) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'('"));
    }
  }

  bool first = true;
  while ((!use_parens ||
          !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RPAREN)) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (!first) {
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
        break;
      }
    }
    if (loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_SSA_VALUE)) {
      loom_token_t ssa_token = loom_tokenizer_next(&parser->tokenizer);
      if (loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COLON)) {
        loom_type_t type = {0};
        IREE_RETURN_IF_ERROR(
            loom_parse_type(parser, LOOM_TYPE_PARSE_ARG, &type));
        IREE_RETURN_IF_ERROR(loom_parse_format_append_symbol_result(
            parser, parsed, type, ssa_token));
      } else if (loom_tokenizer_try_consume_keyword(&parser->tokenizer,
                                                    IREE_SV("as"))) {
        uint16_t operand_index = UINT16_MAX;
        IREE_RETURN_IF_ERROR(
            loom_parse_format_resolve_symbol_tied_result_operand(
                parser, parsed, ssa_token, &operand_index));
        if (parser->error_count > errors_before) return iree_ok_status();

        loom_type_t type = {0};
        IREE_RETURN_IF_ERROR(
            loom_parse_type(parser, LOOM_TYPE_PARSE_BODY, &type));

        loom_value_id_t operand_id =
            operand_index < parsed->operand_count
                ? parsed->operand_ids[operand_index]
                : parser->pending_block_args
                      .entries[operand_index - parsed->operand_count]
                      .value_id;
        loom_type_t operand_type =
            loom_module_value_type(parser->module, operand_id);
        loom_tied_result_t tied = {
            .result_index = parsed->result_count,
            .operand_index = operand_index,
            .has_type_change = !loom_type_equal(operand_type, type),
        };
        IREE_RETURN_IF_ERROR(loom_parsed_op_add_tied_result(
            parsed, &parser->parser_arena, tied));
        IREE_RETURN_IF_ERROR(loom_parse_format_append_symbol_result(
            parser, parsed, type, loom_token_none()));
      } else {
        loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
        return loom_parser_emit_unexpected_token(parser, peek,
                                                 IREE_SV("':' or 'as'"));
      }
    } else {
      loom_type_t type = {0};
      loom_type_parse_mode_t type_mode = loom_parser_in_definition_scope(parser)
                                             ? LOOM_TYPE_PARSE_ARG
                                             : LOOM_TYPE_PARSE_BODY;
      IREE_RETURN_IF_ERROR(loom_parse_type(parser, type_mode, &type));
      IREE_RETURN_IF_ERROR(loom_parse_format_append_symbol_result(
          parser, parsed, type, loom_token_none()));
    }
    if (parser->error_count > errors_before) return iree_ok_status();
    first = false;
  }

  if (use_parens) {
    if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RPAREN)) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("')'"));
    }
  }
  return iree_ok_status();
}

// Parses a mixed static/dynamic index list: [0, %x, 4].
// Static values become an i64_array attribute. Dynamic values
// (INT64_MIN sentinels) generate operand references.
static iree_status_t loom_parse_format_index_list(
    loom_parser_t* parser, const loom_format_element_t* element,
    loom_parsed_op_t* parsed) {
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LBRACKET)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'['"));
  }

  int64_t static_values[32];
  uint16_t value_count = 0;
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RBRACKET) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (value_count > 0) {
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
        break;
      }
    }
    if (value_count >= 32) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_token_text_error(parser, &loom_err_parse_004,
                                               peek);
    }

    if (loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_SSA_VALUE)) {
      // Dynamic index.
      loom_token_t token = loom_tokenizer_next(&parser->tokenizer);
      loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
      LOOM_PARSE_RESOLVE_VALUE(parser, token, &value_id);
      static_values[value_count] = INT64_MIN;  // Sentinel.
      IREE_RETURN_IF_ERROR(
          loom_parsed_op_add_operand(parsed, &parser->parser_arena, value_id));
    } else {
      // Static index.
      loom_token_t token = loom_token_none();
      LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_INTEGER, &token);
      int64_t value = 0;
      if (!iree_string_view_atoi_int64(token.text, &value)) {
        loom_diagnostic_param_t params[] = {
            loom_param_string(token.text),
        };
        return loom_parser_emit(parser, &loom_err_parse_015, params,
                                IREE_ARRAYSIZE(params), token);
      }
      static_values[value_count] = value;
    }
    ++value_count;
  }

  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RBRACKET)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("']'"));
  }

  // Build the static i64 array attribute.
  int64_t* arena_values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&parser->module->arena,
                                                 value_count, sizeof(int64_t),
                                                 (void**)&arena_values));
  memcpy(arena_values, static_values, value_count * sizeof(int64_t));
  loom_attribute_t attr = loom_attr_i64_array(arena_values, value_count);
  return loom_parsed_op_set_attribute(parsed, &parser->parser_arena,
                                      element->data, attr);
}

// Parses a binding list: (%block_arg = %operand : type, ...).
// Creates block arg values and stores them as pending for the next REGION
// element. Names become visible in that region's child scope, not in the
// current parent scope.
static iree_status_t loom_parse_format_binding_list(
    loom_parser_t* parser, const loom_format_element_t* element,
    loom_parsed_op_t* parsed) {
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LPAREN)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'('"));
  }

  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RPAREN) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (parsed->operand_count > element->field_index) {
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
        break;
      }
    }

    // Block arg name.
    loom_token_t arg_token = loom_token_none();
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_SSA_VALUE, &arg_token);

    // '='.
    if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_EQUALS)) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'='"));
    }

    // Operand.
    loom_token_t op_token = loom_token_none();
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_SSA_VALUE, &op_token);
    loom_value_id_t operand_id = LOOM_VALUE_ID_INVALID;
    LOOM_PARSE_RESOLVE_VALUE(parser, op_token, &operand_id);

    // ':'.
    if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COLON)) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("':'"));
    }

    // Type.
    loom_type_t type = {0};
    IREE_RETURN_IF_ERROR(loom_parse_type(parser, LOOM_TYPE_PARSE_BODY, &type));

    // Add the operand.
    IREE_RETURN_IF_ERROR(
        loom_parsed_op_add_operand(parsed, &parser->parser_arena, operand_id));

    // Derive the block arg type from the binding kind.
    // ELEMENT bindings: block arg gets the element type of the
    // operand (scalar for shaped types). CAPTURE: same type.
    loom_type_t arg_type = {0};
    if (element->data == LOOM_BINDING_ELEMENT) {
      if (loom_type_is_shaped(type)) {
        arg_type = loom_type_scalar(loom_type_element_type(type));
      } else {
        arg_type = type;
      }
    } else {
      arg_type = type;
    }

    // Create the block arg value. REGION will define the child-scope name when
    // it attaches this pending arg to the entry block.
    loom_value_id_t arg_value_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_module_define_value(parser->module, arg_type, &arg_value_id));

    // Store as pending block arg for REGION to consume.
    IREE_RETURN_IF_ERROR(
        loom_parser_add_pending_block_arg(parser, arg_value_id, arg_token));
  }

  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RPAREN)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("')'"));
  }
  return iree_ok_status();
}

// Parses function arguments: (%name: type, type, ...).
// Creates SSA values in the surrounding signature scope and stores them as
// pending block args for the next REGION element. For bodyless func-like ops,
// loom_parse_op drains these values into op operands after the format walk.
static iree_status_t loom_parse_format_func_args(loom_parser_t* parser,
                                                 loom_parsed_op_t* parsed) {
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LPAREN)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'('"));
  }

  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RPAREN) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (parser->pending_block_args.count > 0) {
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
        break;
      }
    }

    loom_token_t name_token = loom_token_none();
    if (loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_SSA_VALUE)) {
      name_token = loom_tokenizer_next(&parser->tokenizer);
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COLON)) {
        loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
        return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("':'"));
      }
    }

    // type (ARG mode: [%M] creates new index value in scope)
    loom_type_t type = {0};
    IREE_RETURN_IF_ERROR(loom_parse_type(parser, LOOM_TYPE_PARSE_ARG, &type));

    // Create a fresh value or bind a local forward placeholder from an earlier
    // ARG-mode type reference in this declaration scope.
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_parser_define_value(parser, name_token, type, &value_id));
    if (parser->error_count > 0) return iree_ok_status();

    // Store as pending block arg for REGION to consume.
    IREE_RETURN_IF_ERROR(
        loom_parser_add_pending_block_arg(parser, value_id, name_token));
  }

  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RPAREN)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("')'"));
  }
  return iree_ok_status();
}

// Evaluates an optional group anchor and returns the number of
// format elements to skip if the group is absent. When the group
// is present, |*out_skip_count| is set to 0.
static iree_status_t loom_parse_format_optional_group(
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
    const loom_format_element_t* element, uint16_t element_index,
    const loom_parsed_op_t* parsed, uint16_t* out_skip_count) {
  uint16_t skip_count = element->data >> 2;
  uint8_t anchor_category = element->data & 3;
  bool present = false;

  switch (anchor_category) {
    case LOOM_ANCHOR_OPERAND: {
      // Check if the first inner element is a keyword (e.g., "iter_args"
      // before a binding list). In that case, probe for that specific
      // keyword rather than an SSA value — the operands come after the
      // keyword, not before it.
      const loom_format_element_t* first_inner =
          (element_index + 1 < vtable->format_element_count)
              ? &vtable->format_elements[element_index + 1]
              : NULL;
      if (first_inner && first_inner->kind == LOOM_FORMAT_KIND_KEYWORD) {
        loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
        loom_token_kind_t expected_kind =
            loom_keyword_token_kind(first_inner->data);
        if (expected_kind == LOOM_TOKEN_BARE_IDENT) {
          present = (peek.kind == LOOM_TOKEN_BARE_IDENT &&
                     loom_bstring_equal(
                         loom_keyword_bstrings[first_inner->data], peek.text));
        } else {
          present = (peek.kind == expected_kind);
        }
      } else {
        present = loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_SSA_VALUE);
      }
      break;
    }
    case LOOM_ANCHOR_ATTR: {
      // When the first inner element is a KEYWORD (e.g., "where"
      // before a predicate list), refine the check to match that
      // specific keyword. The generic token set would false-positive
      // on '{' (LBRACE), which is ambiguous between attr_dict and
      // region.
      //
      // When the first inner element is an ATTR_VALUE with ENUM kind,
      // refine the check to match a specific enum case name. Without
      // this, consecutive optional enum groups (visibility, purity)
      // all probe positive on any bare ident and the wrong group
      // consumes the token.
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      const loom_format_element_t* first_inner =
          (element_index + 1 < vtable->format_element_count)
              ? &vtable->format_elements[element_index + 1]
              : NULL;
      if (first_inner && first_inner->kind == LOOM_FORMAT_KIND_KEYWORD) {
        loom_token_kind_t expected_kind =
            loom_keyword_token_kind(first_inner->data);
        if (expected_kind == LOOM_TOKEN_BARE_IDENT) {
          // Text keyword (import, where, etc.): match by text.
          present = (peek.kind == LOOM_TOKEN_BARE_IDENT &&
                     loom_bstring_equal(
                         loom_keyword_bstrings[first_inner->data], peek.text));
        } else {
          // Punctuation keyword (comma, arrow, etc.): match by kind.
          present = (peek.kind == expected_kind);
        }
      } else if (first_inner &&
                 first_inner->kind == LOOM_FORMAT_KIND_ATTR_VALUE &&
                 peek.kind == LOOM_TOKEN_BARE_IDENT) {
        const loom_attr_descriptor_t* descriptor =
            &vtable->attr_descriptors[first_inner->field_index];
        if (descriptor->attr_kind == LOOM_ATTR_ENUM &&
            descriptor->enum_case_names) {
          present = false;
          for (uint8_t c = 0; c < descriptor->enum_case_count; ++c) {
            if (descriptor->enum_case_names[c] &&
                loom_bstring_equal(descriptor->enum_case_names[c], peek.text)) {
              present = true;
              break;
            }
          }
        } else {
          present = true;
        }
      } else {
        present =
            (peek.kind == LOOM_TOKEN_INTEGER || peek.kind == LOOM_TOKEN_FLOAT ||
             peek.kind == LOOM_TOKEN_STRING ||
             peek.kind == LOOM_TOKEN_BARE_IDENT ||
             peek.kind == LOOM_TOKEN_LBRACKET ||
             peek.kind == LOOM_TOKEN_LBRACE);
      }
      break;
    }
    case LOOM_ANCHOR_REGION:
      present = loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_LBRACE);
      break;
    case LOOM_ANCHOR_RESULTS:
      present = parsed->result_count > 0 ||
                loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_ARROW);
      break;
  }

  *out_skip_count = present ? 0 : skip_count;
  return iree_ok_status();
}

// Parses instance flags: <flag1|flag2>.
// Matches flag names against the vtable's flag case names.
static iree_status_t loom_parse_format_flags(loom_parser_t* parser,
                                             const loom_op_vtable_t* vtable,
                                             loom_parsed_op_t* parsed) {
  if (loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LANGLE)) {
    uint8_t flags = 0;
    bool first = true;
    while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RANGLE) &&
           !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
      if (!first) {
        if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_PIPE)) {
          break;
        }
      }
      loom_token_t flag_token = loom_token_none();
      LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_BARE_IDENT, &flag_token);
      // Look up the flag name.
      bool found = false;
      for (uint8_t bit = 0; bit < vtable->instance_flags_case_count; ++bit) {
        if (loom_bstring_equal(vtable->instance_flags_case_names[bit],
                               flag_token.text)) {
          flags |= (1u << bit);
          found = true;
          break;
        }
      }
      if (!found) {
        loom_diagnostic_param_t params[] = {
            loom_param_string(IREE_SV("instance flag")),
            loom_param_string(flag_token.text),
        };
        return loom_parser_emit(parser, &loom_err_parse_018, params,
                                IREE_ARRAYSIZE(params), flag_token);
      }
      first = false;
    }
    if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RANGLE)) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'>'"));
    }
    parsed->instance_flags = flags;
  }
  return iree_ok_status();
}

// Parses an op reference: <op.name>.
static iree_status_t loom_parse_format_op_ref(
    loom_parser_t* parser, const loom_format_element_t* element,
    loom_parsed_op_t* parsed) {
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LANGLE)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'<'"));
  }
  loom_token_t name_token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_OP_NAME, &name_token);
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RANGLE)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'>'"));
  }
  loom_string_id_t name_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(parser->module, name_token.text, &name_id));
  loom_attribute_t attr = loom_attr_string(name_id);
  return loom_parsed_op_set_attribute(parsed, &parser->parser_arena,
                                      element->field_index, attr);
}

//===----------------------------------------------------------------------===//
// Format walker
//===----------------------------------------------------------------------===//

iree_status_t loom_parser_walk_format(loom_parser_t* parser,
                                      const loom_op_vtable_t* vtable,
                                      loom_token_t op_name_token,
                                      loom_parsed_op_t* parsed) {
  const loom_format_element_t* elements = vtable->format_elements;
  uint16_t element_count = vtable->format_element_count;
  bool is_symbol_definition =
      iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE);

  uint32_t errors_before = parser->error_count;
  for (uint16_t i = 0; i < element_count; ++i) {
    const loom_format_element_t* element = &elements[i];
    switch (element->kind) {
      case LOOM_FORMAT_KIND_OPERAND_REF: {
        if (element->field_index == 0xFF) {
          // Induction variable reference (e.g., %iv in test.loop).
          // Consume the SSA token, create an index-typed value, and queue it
          // as a pending block arg. REGION defines the loop IV name in the
          // child scope when seeding the entry block.
          loom_token_t iv_token = loom_token_none();
          LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_SSA_VALUE, &iv_token);
          loom_value_id_t iv_value_id = 0;
          IREE_RETURN_IF_ERROR(loom_module_define_value(
              parser->module, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
              &iv_value_id));
          IREE_RETURN_IF_ERROR(
              loom_parser_add_pending_block_arg(parser, iv_value_id, iv_token));
          break;
        }
        loom_token_t token = loom_token_none();
        LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_SSA_VALUE, &token);
        loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
        LOOM_PARSE_RESOLVE_VALUE(parser, token, &value_id);
        IREE_RETURN_IF_ERROR(loom_parsed_op_add_operand(
            parsed, &parser->parser_arena, value_id));
        break;
      }

      case LOOM_FORMAT_KIND_OPERAND_REFS: {
        // Variadic operands: %a, %b, %c. Parse until we hit something
        // that isn't an SSA value or comma.
        bool first = true;
        while (loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_SSA_VALUE) ||
               (!first &&
                loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_COMMA))) {
          if (!first) {
            loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA);
          }
          if (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_SSA_VALUE)) {
            break;
          }
          loom_token_t token = loom_tokenizer_next(&parser->tokenizer);
          loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
          LOOM_PARSE_RESOLVE_VALUE(parser, token, &value_id);
          IREE_RETURN_IF_ERROR(loom_parsed_op_add_operand(
              parsed, &parser->parser_arena, value_id));
          first = false;
        }
        break;
      }

      case LOOM_FORMAT_KIND_ATTR_VALUE: {
        const loom_attr_descriptor_t* descriptor =
            &vtable->attr_descriptors[element->field_index];
        loom_attribute_t attr = {0};
        IREE_RETURN_IF_ERROR(loom_parse_attr_value(parser, descriptor, &attr));
        IREE_RETURN_IF_ERROR(loom_parsed_op_set_attribute(
            parsed, &parser->parser_arena, element->field_index, attr));
        break;
      }

      case LOOM_FORMAT_KIND_SYMBOL_REF: {
        loom_token_t token = loom_token_none();
        LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_SYMBOL, &token);
        loom_string_id_t name_id = 0;
        IREE_RETURN_IF_ERROR(
            loom_module_intern_string(parser->module, token.text, &name_id));
        loom_symbol_ref_t ref;
        ref.module_id = 0;
        uint16_t next_id = (uint16_t)parser->module->symbols.count;
        IREE_RETURN_IF_ERROR(loom_symbol_map_find_or_insert(
            &parser->symbol_lookup, &parser->parser_arena, name_id, next_id,
            &ref.symbol_id));
        if (ref.symbol_id == next_id) {
          uint16_t added_id = LOOM_SYMBOL_ID_INVALID;
          IREE_RETURN_IF_ERROR(
              loom_module_add_symbol(parser->module, name_id, &added_id));
        }
        loom_attribute_t attr = loom_attr_symbol(ref);
        IREE_RETURN_IF_ERROR(loom_parsed_op_set_attribute(
            parsed, &parser->parser_arena, element->field_index, attr));
        break;
      }

      case LOOM_FORMAT_KIND_OPERAND_TYPE: {
        // Type of an operand — parse and verify (or just consume).
        loom_type_t type = {0};
        IREE_RETURN_IF_ERROR(
            loom_parse_type(parser, LOOM_TYPE_PARSE_BODY, &type));
        break;
      }

      case LOOM_FORMAT_KIND_OPERAND_TYPES: {
        // Variadic operand types. Parse types separated by commas.
        bool first = true;
        for (;;) {
          loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
          if (peek.kind == LOOM_TOKEN_EOF || peek.kind == LOOM_TOKEN_RPAREN ||
              peek.kind == LOOM_TOKEN_RBRACE) {
            break;
          }
          if (!first) {
            if (!loom_tokenizer_try_consume(&parser->tokenizer,
                                            LOOM_TOKEN_COMMA)) {
              break;
            }
          }
          loom_type_t type = {0};
          IREE_RETURN_IF_ERROR(
              loom_parse_type(parser, LOOM_TYPE_PARSE_BODY, &type));
          first = false;
        }
        break;
      }

      case LOOM_FORMAT_KIND_RESULT_TYPE:
      case LOOM_FORMAT_KIND_RESULT_TYPE_SINGLE: {
        loom_type_parse_mode_t type_mode =
            loom_parser_in_definition_scope(parser) ? LOOM_TYPE_PARSE_ARG
                                                    : LOOM_TYPE_PARSE_BODY;
        if (!is_symbol_definition) {
          IREE_RETURN_IF_ERROR(
              loom_parse_format_prepare_result_scope(parser, parsed));
        }
        loom_type_t type = {0};
        IREE_RETURN_IF_ERROR(loom_parse_type(parser, type_mode, &type));
        if (is_symbol_definition) {
          IREE_RETURN_IF_ERROR(loom_parse_format_append_symbol_result(
              parser, parsed, type, loom_token_none()));
        } else {
          loom_parser_result_scope_reset(&parser->result_scope);
          IREE_RETURN_IF_ERROR(loom_parse_format_assign_lhs_result_type(
              parser, vtable, op_name_token, parsed, element->field_index,
              type));
        }
        break;
      }

      case LOOM_FORMAT_KIND_RESULT_TYPE_LIST: {
        if (is_symbol_definition) {
          IREE_RETURN_IF_ERROR(loom_parse_format_symbol_result_type_list(
              parser, element, parsed));
        } else {
          IREE_RETURN_IF_ERROR(
              loom_parse_format_prepare_result_scope(parser, parsed));
          IREE_RETURN_IF_ERROR(loom_parse_format_lhs_result_type_list(
              parser, vtable, op_name_token, element, parsed));
          loom_parser_result_scope_reset(&parser->result_scope);
        }
        break;
      }

      case LOOM_FORMAT_KIND_KEYWORD: {
        IREE_RETURN_IF_ERROR(loom_parse_keyword(parser, element->data));
        break;
      }

      case LOOM_FORMAT_KIND_REGION: {
        loom_region_t* region = NULL;
        IREE_RETURN_IF_ERROR(loom_parse_region(parser, &region));
        IREE_RETURN_IF_ERROR(
            loom_parsed_op_add_region(parsed, &parser->parser_arena, region));
        break;
      }

      case LOOM_FORMAT_KIND_INDEX_LIST: {
        IREE_RETURN_IF_ERROR(
            loom_parse_format_index_list(parser, element, parsed));
        break;
      }

      case LOOM_FORMAT_KIND_BINDING_LIST: {
        IREE_RETURN_IF_ERROR(
            loom_parse_format_binding_list(parser, element, parsed));
        break;
      }

      case LOOM_FORMAT_KIND_FUNC_ARGS: {
        IREE_RETURN_IF_ERROR(loom_parse_format_func_args(parser, parsed));
        break;
      }

      case LOOM_FORMAT_KIND_PREDICATE_LIST: {
        loom_attribute_t attr = {0};
        IREE_RETURN_IF_ERROR(loom_parse_predicate_list(parser, &attr));
        IREE_RETURN_IF_ERROR(loom_parsed_op_set_attribute(
            parsed, &parser->parser_arena, element->field_index, attr));
        break;
      }

      case LOOM_FORMAT_KIND_OPTIONAL_GROUP: {
        uint16_t skip_count = 0;
        IREE_RETURN_IF_ERROR(loom_parse_format_optional_group(
            parser, vtable, element, i, parsed, &skip_count));
        i += skip_count;
        break;
      }

      case LOOM_FORMAT_KIND_GLUE:
        // No-op for the parser. Spacing is irrelevant when reading.
        break;

      case LOOM_FORMAT_KIND_FLAGS: {
        IREE_RETURN_IF_ERROR(loom_parse_format_flags(parser, vtable, parsed));
        break;
      }

      case LOOM_FORMAT_KIND_OP_REF: {
        IREE_RETURN_IF_ERROR(loom_parse_format_op_ref(parser, element, parsed));
        break;
      }

      case LOOM_FORMAT_KIND_ATTR_DICT: {
        loom_attribute_t attr = {0};
        IREE_RETURN_IF_ERROR(loom_parse_attr_dict(parser, &attr));
        IREE_RETURN_IF_ERROR(loom_parsed_op_set_attribute(
            parsed, &parser->parser_arena, element->field_index, attr));
        break;
      }

      case LOOM_FORMAT_KIND_SCOPE: {
        IREE_RETURN_IF_ERROR(loom_parser_definition_scope_push(
            parser, (uint16_t)(i + element->data)));
        break;
      }
    }
    // Bail out of the format walk if any element emitted a parse error.
    // The diagnostic is already emitted — the caller (loom_parse_op)
    // handles recovery.
    if (parser->error_count > errors_before) return iree_ok_status();
    // Pop definition scope after the last child element has been processed.
    IREE_RETURN_IF_ERROR(loom_parser_definition_scope_pop_if_needed(parser, i));
    if (parser->error_count > errors_before) return iree_ok_status();
  }
  return iree_ok_status();
}
