// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "loom/format/text/parser_internal.h"
#include "loom/ir/context.h"

//===----------------------------------------------------------------------===//
// Encoding parameter parsing
//===----------------------------------------------------------------------===//

// Parses encoding parameters from the token stream. Called after LANGLE
// has been consumed. Consumes tokens through the closing RANGLE
// (inclusive). Grammar: key=value [, key=value]* >.
iree_status_t loom_parse_encoding_params(loom_parser_t* parser,
                                         loom_named_attr_t** out_attrs,
                                         uint8_t* out_count) {
  iree_host_size_t capacity = 8;
  loom_named_attr_t* attrs = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&parser->parser_arena, capacity,
                                sizeof(loom_named_attr_t), (void**)&attrs));
  uint8_t attr_count = 0;

  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RANGLE) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (attr_count > 0) {
      LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COMMA, NULL);
    }

    if (attr_count == UINT8_MAX) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_token_text_error(parser, &loom_err_parse_004,
                                               peek);
    }
    if (attr_count >= capacity) {
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(
          &parser->parser_arena, attr_count, /*minimum_capacity=*/8,
          sizeof(loom_named_attr_t), &capacity, (void**)&attrs));
    }

    // Parameter name.
    loom_token_t name_token = loom_token_none();
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_BARE_IDENT, &name_token);

    // Separator.
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_EQUALS, NULL);

    loom_string_id_t param_name_id = 0;
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        parser->module, name_token.text, &param_name_id));
    attrs[attr_count].name_id = param_name_id;
    attrs[attr_count].reserved = 0;

    uint32_t errors_before = parser->error_count;
    IREE_RETURN_IF_ERROR(loom_parse_generic_attr_value(
        parser, /*nesting_depth=*/0, &attrs[attr_count].value));
    if (parser->error_count > errors_before) {
      return iree_ok_status();
    }
    ++attr_count;
  }

  // Consume closing '>'.
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RANGLE, NULL);

  *out_attrs = attrs;
  *out_count = attr_count;
  return iree_ok_status();
}

iree_status_t loom_parse_static_encoding(loom_parser_t* parser,
                                         loom_string_id_t alias_id,
                                         uint16_t* out_encoding_id) {
  loom_token_t token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_HASH_ATTR, &token);

  uint16_t aliased_id = loom_alias_table_lookup(&parser->aliases, token.text);
  if (aliased_id != 0) {
    *out_encoding_id = aliased_id;
    return iree_ok_status();
  }

  loom_encoding_t encoding = {
      .name_id = LOOM_STRING_ID_INVALID,
      .alias_id = alias_id,
  };
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(parser->module, token.text, &encoding.name_id));

  if (loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LANGLE)) {
    loom_named_attr_t* attrs = NULL;
    uint8_t attr_count = 0;
    uint32_t errors_before = parser->error_count;
    IREE_RETURN_IF_ERROR(
        loom_parse_encoding_params(parser, &attrs, &attr_count));
    if (parser->error_count > errors_before) {
      return iree_ok_status();
    }
    encoding.attribute_count = attr_count;
    encoding.attributes = attrs;
  }

  if (parser->context->encoding_vtables.count > 0 &&
      !loom_context_lookup_encoding_vtable(parser->context, token.text)) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(token.text),
    };
    return loom_parser_emit(parser, &loom_err_parse_008, params,
                            IREE_ARRAYSIZE(params), token);
  }

  return loom_module_add_encoding(parser->module, &encoding, out_encoding_id);
}

//===----------------------------------------------------------------------===//
// Type reference resolution
//===----------------------------------------------------------------------===//

// Resolves an SSA value reference in a type context. |name_token| is
// an SSA_VALUE token whose text is the bare name (e.g., "M" for %M)
// and whose line/column point to the '%' sigil for diagnostics.
//
// If the name is already in scope, returns its value_id.
//
// In definition context (ARG mode), creates a new NONE-typed placeholder value
// and defines it in scope. The surrounding Scope(...) format element controls
// that mode for signatures/globals, so BODY-mode tied result annotations still
// reject unknown names even inside a signature scope. Type parsing still
// infers the placeholder's value type from each use (index for dims, encoding
// for SSA encodings), but scope-exit resolution is tracked separately so an
// inferred use without a matching declaration still diagnoses precisely.
//
// In body context (BODY mode), emits PARSE/001 for undefined names.
static iree_status_t loom_resolve_type_reference(
    loom_parser_t* parser, loom_token_t name_token, loom_type_parse_mode_t mode,
    loom_value_id_t* out_value_id) {
  loom_string_id_t name_id =
      loom_module_lookup_string(parser->module, name_token.text);
  loom_value_id_t value_id = loom_parser_lookup_value(parser, name_id);
  if (value_id != LOOM_VALUE_ID_INVALID) {
    *out_value_id = value_id;
    return iree_ok_status();
  }

  if (mode == LOOM_TYPE_PARSE_ARG) {
    // Forward reference in definition context — create placeholder.
    IREE_RETURN_IF_ERROR(
        loom_module_define_value(parser->module, loom_type_none(), &value_id));
    IREE_RETURN_IF_ERROR(
        loom_parser_define_value_name(parser, name_token, value_id));
    IREE_RETURN_IF_ERROR(
        loom_parser_add_unresolved_placeholder(parser, value_id, name_token));
    *out_value_id = value_id;
    return iree_ok_status();
  }

  // Body mode: undefined name is an error.
  return loom_parser_resolve_value(parser, name_token, out_value_id);
}

// Resolves a type-binding SSA reference and bails out of the enclosing parse
// function if BODY-mode resolution emits a parser diagnostic.
#define LOOM_PARSE_RESOLVE_TYPE_REFERENCE(parser, name_token, mode,            \
                                          out_value_id)                        \
  do {                                                                         \
    uint32_t _resolve_errors = (parser)->error_count;                          \
    IREE_RETURN_IF_ERROR(loom_resolve_type_reference((parser), (name_token),   \
                                                     (mode), (out_value_id))); \
    if ((parser)->error_count > _resolve_errors) return iree_ok_status();      \
  } while (0)

// Assigns types to NONE-typed values referenced by a parsed type's
// dim bindings and encoding binding. Dim binding values get index
// type, encoding binding values get encoding type. Values that
// already have a type are left untouched (they were defined elsewhere
// and their types are authoritative).
static bool loom_type_binding_is_unresolved_placeholder(
    const loom_parser_t* parser, loom_value_id_t value_id) {
  if (!loom_parser_in_definition_scope(parser)) return false;
  for (iree_host_size_t i = parser->definition_scope.placeholder_start;
       i < parser->unresolved_placeholders.count; ++i) {
    if (parser->unresolved_placeholders.entries[i].value_id == value_id) {
      return true;
    }
  }
  return false;
}

static void loom_assign_type_binding_types(loom_parser_t* parser,
                                           loom_type_t type) {
  // Dim bindings.
  uint8_t rank = loom_type_rank(type);
  for (uint8_t i = 0; i < rank; ++i) {
    if (loom_type_dim_is_dynamic_at(type, i)) {
      loom_value_id_t value_id = loom_type_dim_value_id_at(type, i);
      if (loom_type_binding_is_unresolved_placeholder(parser, value_id)) {
        continue;
      }
      loom_value_t* value = &parser->module->values.entries[value_id];
      if (loom_type_kind(value->type) == LOOM_TYPE_NONE) {
        value->type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
      }
    }
  }

  // Encoding binding (SSA encoding).
  if (loom_type_has_ssa_encoding(type)) {
    loom_value_id_t enc_id = loom_type_encoding_value_id(type);
    if (loom_type_binding_is_unresolved_placeholder(parser, enc_id)) return;
    loom_value_t* value = &parser->module->values.entries[enc_id];
    if (loom_type_kind(value->type) == LOOM_TYPE_NONE) {
      value->type = loom_type_encoding();
    }
  }
}

//===----------------------------------------------------------------------===//
// Dimension parsing
//===----------------------------------------------------------------------===//

// Parses a single dimension: INTEGER (static) or [ SSA_VALUE ] (dynamic).
// Returns the packed dim value. The caller has already peeked the token
// and confirmed it is INTEGER or LBRACKET.
static iree_status_t loom_parse_dim(loom_parser_t* parser,
                                    loom_type_parse_mode_t mode,
                                    uint64_t* out_dim) {
  loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);
  if (token.kind == LOOM_TOKEN_INTEGER) {
    loom_tokenizer_next(&parser->tokenizer);
    int64_t size = 0;
    if (!iree_string_view_atoi_int64(token.text, &size) || size < 0) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(token.text),
      };
      return loom_parser_emit(parser, &loom_err_parse_015, params,
                              IREE_ARRAYSIZE(params), token);
    }
    *out_dim = loom_dim_pack_static(size);
    return iree_ok_status();
  }
  // [SSA_VALUE]
  loom_tokenizer_next(&parser->tokenizer);  // consume [
  loom_token_t name_token;
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_SSA_VALUE, &name_token);
  loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
  LOOM_PARSE_RESOLVE_TYPE_REFERENCE(parser, name_token, mode, &value_id);
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RBRACKET, NULL);
  *out_dim = loom_dim_pack_dynamic(value_id);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Type encoding parsing
//===----------------------------------------------------------------------===//

// Parses a type encoding after the comma in a shaped type interior.
// Called after COMMA has been consumed, with in_dim_list already false.
// Handles SSA encodings (`%enc`), static encoding aliases (`#enc`), and
// canonical family spellings (`#q8_0` or `#q8_0<block=32>`).
static iree_status_t loom_parse_type_encoding(
    loom_parser_t* parser, loom_type_parse_mode_t mode,
    uint16_t* out_encoding_id, loom_encoding_flags_t* out_encoding_flags) {
  loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);
  if (token.kind == LOOM_TOKEN_SSA_VALUE) {
    // SSA encoding: %enc.
    loom_tokenizer_next(&parser->tokenizer);
    loom_value_id_t enc_value_id = LOOM_VALUE_ID_INVALID;
    LOOM_PARSE_RESOLVE_TYPE_REFERENCE(parser, token, mode, &enc_value_id);
    *out_encoding_id = (uint16_t)enc_value_id;
    *out_encoding_flags = LOOM_ENCODING_FLAG_SSA;
    return iree_ok_status();
  }

  if (token.kind == LOOM_TOKEN_HASH_ATTR) {
    IREE_RETURN_IF_ERROR(loom_parse_static_encoding(
        parser, LOOM_STRING_ID_INVALID, out_encoding_id));
    *out_encoding_flags = 0;
    return iree_ok_status();
  }

  if (token.kind == LOOM_TOKEN_RESULT_ORDINAL) {
    // '#' followed by a digit is not a valid encoding reference.
    loom_tokenizer_next(&parser->tokenizer);
    return loom_parser_emit_token_text_error(parser, &loom_err_parse_004,
                                             token);
  }

  loom_diagnostic_param_t params[] = {
      loom_param_string(token.text),
  };
  return loom_parser_emit(parser, &loom_err_parse_008, params,
                          IREE_ARRAYSIZE(params), token);
}

//===----------------------------------------------------------------------===//
// Shaped type construction
//===----------------------------------------------------------------------===//

// Constructs, interns, and assigns binding types for a shaped type from
// its parsed components. Handles inline dims (rank 0-2) and overflow
// allocation (rank > 2).
static iree_status_t loom_intern_shaped_type(
    loom_parser_t* parser, loom_type_kind_t kind,
    loom_scalar_type_t element_type, const uint64_t* dims, uint8_t rank,
    uint16_t encoding_id, loom_encoding_flags_t encoding_flags,
    loom_type_t* out_type) {
  loom_type_t type = {0};
  if (rank == 0) {
    type = loom_type_shaped_0d(kind, element_type, encoding_id);
  } else if (rank == 1) {
    type = loom_type_shaped_1d(kind, element_type, dims[0], encoding_id);
  } else if (rank == 2) {
    type =
        loom_type_shaped_2d(kind, element_type, dims[0], dims[1], encoding_id);
  } else {
    loom_overflow_dim_t* overflow = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&parser->module->arena, rank,
                                                   sizeof(loom_overflow_dim_t),
                                                   (void**)&overflow));
    memcpy(overflow, dims, rank * sizeof(uint64_t));

    uint8_t flags = 0;
    bool all_static = true;
    for (uint8_t i = 0; i < rank; ++i) {
      if (loom_dim_is_dynamic(dims[i])) {
        all_static = false;
        break;
      }
    }
    if (all_static) flags |= LOOM_TYPE_FLAG_ALL_STATIC;

    type.header = loom_type_make_header(kind, element_type, rank, flags);
    type.encoding_id = encoding_id;
    type.encoding_flags = encoding_flags;
    type.dims[0] = (uint64_t)(uintptr_t)overflow;
    // Precompute hash for fast inequality rejection.
    uint64_t hash = 0;
    for (uint8_t i = 0; i < rank; ++i) hash = hash * 31 + dims[i];
    type.dims[1] = hash;
  }

  if (rank <= 2) {
    type.encoding_flags = encoding_flags;
  }

  IREE_RETURN_IF_ERROR(loom_module_intern_type(parser->module, type, out_type));
  loom_assign_type_binding_types(parser, *out_type);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Shaped type parsing
//===----------------------------------------------------------------------===//

// Parses a shaped type (tile or tensor) from the token stream. Called
// after LANGLE has been consumed. Consumes tokens through RANGLE.
//
// Grammar: dim (x dim)* x element_type [, encoding] >
//
// The tokenizer's in_dim_list flag is managed here: set after the
// first dim is consumed, cleared before scanning what follows each
// DIM_X separator. This ensures element types and encoding params
// are always scanned in normal (non-dim) mode.
static iree_status_t loom_parse_shaped_type(loom_parser_t* parser,
                                            loom_type_kind_t kind,
                                            loom_type_parse_mode_t mode,
                                            loom_type_t* out_type) {
  uint64_t dims[16];
  uint8_t rank = 0;

  // Parse dimensions. in_dim_list must be true from the start so
  // that '0x' in '0xf32' is scanned as INTEGER(0) + DIM_X(x) +
  // BARE_IDENT(f32), not as hex INTEGER(0xf32).
  parser->tokenizer.in_dim_list = true;
  loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);
  if (token.kind == LOOM_TOKEN_INTEGER || token.kind == LOOM_TOKEN_LBRACKET) {
    IREE_RETURN_IF_ERROR(loom_parse_dim(parser, mode, &dims[rank++]));

    while (loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_DIM_X)) {
      loom_tokenizer_next(&parser->tokenizer);  // consume 'x'
      parser->tokenizer.in_dim_list = false;

      token = loom_tokenizer_peek(&parser->tokenizer);
      if (token.kind != LOOM_TOKEN_INTEGER &&
          token.kind != LOOM_TOKEN_LBRACKET) {
        break;  // element type follows
      }
      if (rank >= 16) {
        return loom_parser_emit_token_text_error(parser, &loom_err_parse_004,
                                                 token);
      }
      parser->tokenizer.in_dim_list = true;
      IREE_RETURN_IF_ERROR(loom_parse_dim(parser, mode, &dims[rank++]));
    }
  } else {
    // Rank 0 — no dims. Clear in_dim_list before element type.
    parser->tokenizer.in_dim_list = false;
  }

  // Parse element type.
  loom_token_t element_token;
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_BARE_IDENT, &element_token);
  loom_scalar_type_t element_type = 0;
  if (!loom_scalar_type_parse(element_token.text, &element_type)) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(element_token.text),
    };
    return loom_parser_emit(parser, &loom_err_parse_007, params,
                            IREE_ARRAYSIZE(params), element_token);
  }

  // Parse optional encoding.
  uint16_t encoding_id = 0;
  loom_encoding_flags_t encoding_flags = 0;
  if (loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
    IREE_RETURN_IF_ERROR(
        loom_parse_type_encoding(parser, mode, &encoding_id, &encoding_flags));
  }

  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RANGLE, NULL);

  return loom_intern_shaped_type(parser, kind, element_type, dims, rank,
                                 encoding_id, encoding_flags, out_type);
}

//===----------------------------------------------------------------------===//
// Pool type parsing
//===----------------------------------------------------------------------===//

// Parses a pool type from the token stream. Called after LANGLE has
// been consumed. Consumes tokens through RANGLE. Pool types have a
// single dimension and no element type or encoding.
static iree_status_t loom_parse_pool_type(loom_parser_t* parser,
                                          loom_type_parse_mode_t mode,
                                          loom_type_t* out_type) {
  loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);
  if (token.kind != LOOM_TOKEN_INTEGER && token.kind != LOOM_TOKEN_LBRACKET) {
    return loom_parser_emit_unexpected_token(parser, token,
                                             IREE_SV("integer or '['"));
  }
  uint64_t dim = 0;
  IREE_RETURN_IF_ERROR(loom_parse_dim(parser, mode, &dim));
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RANGLE, NULL);

  *out_type = loom_type_pool(dim);
  loom_assign_type_binding_types(parser, *out_type);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Group type parsing
//===----------------------------------------------------------------------===//

// Parses a group type from the token stream. Called after LANGLE has
// been consumed. Consumes tokens through RANGLE.
static iree_status_t loom_parse_group_type(loom_parser_t* parser,
                                           loom_type_t* out_type) {
  loom_token_t scope_token;
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_BARE_IDENT, &scope_token);
  loom_type_t type = {0};
  if (iree_string_view_equal(scope_token.text, IREE_SV("workgroup"))) {
    type.header = loom_type_make_header(
        LOOM_TYPE_GROUP, (loom_scalar_type_t)LOOM_GROUP_SCOPE_WORKGROUP, 0, 0);
  } else if (iree_string_view_equal(scope_token.text, IREE_SV("subgroup"))) {
    type.header = loom_type_make_header(
        LOOM_TYPE_GROUP, (loom_scalar_type_t)LOOM_GROUP_SCOPE_SUBGROUP, 0, 0);
  } else {
    loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV("group scope")),
        loom_param_string(scope_token.text),
    };
    return loom_parser_emit(parser, &loom_err_parse_018, params,
                            IREE_ARRAYSIZE(params), scope_token);
  }
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RANGLE, NULL);
  *out_type = type;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Dialect type parsing
//===----------------------------------------------------------------------===//

// Parses a dialect type (e.g., hal.buffer, vm.ref<i32>) from the
// token stream. Called after the OP_NAME token has been consumed.
static iree_status_t loom_parse_dialect_type(loom_parser_t* parser,
                                             loom_token_t name_token,
                                             loom_type_parse_mode_t mode,
                                             loom_type_t* out_type) {
  loom_string_id_t name_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(parser->module, name_token.text, &name_id));

  if (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_LANGLE)) {
    *out_type = loom_type_dialect_opaque(name_id);
    return iree_ok_status();
  }

  // Parameterized: name<type, type, ...>.
  loom_tokenizer_next(&parser->tokenizer);  // consume '<'
  loom_type_t param_types[8];
  uint16_t param_count = 0;
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RANGLE) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (param_count > 0) {
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
        break;
      }
    }
    if (param_count >= 8) {
      return loom_parser_emit_token_text_error(parser, &loom_err_parse_004,
                                               name_token);
    }
    IREE_RETURN_IF_ERROR(
        loom_parse_type(parser, mode, &param_types[param_count]));
    ++param_count;
  }
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RANGLE)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'>'"));
  }
  if (param_count > 0) {
    loom_type_t* arena_params = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(&parser->parser_arena, param_count,
                                  sizeof(loom_type_t), (void**)&arena_params));
    memcpy(arena_params, param_types, param_count * sizeof(loom_type_t));
    *out_type = loom_type_dialect(name_id, param_count, arena_params);
  } else {
    *out_type = loom_type_dialect_opaque(name_id);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Function type parsing
//===----------------------------------------------------------------------===//

// Parses a comma-separated type list terminated by RPAREN. Used for
// both argument and result type lists in function types. Caller has
// consumed the opening '('. This function consumes through ')'.
static iree_status_t loom_parse_type_list(loom_parser_t* parser,
                                          loom_type_parse_mode_t mode,
                                          loom_type_t* types, uint16_t capacity,
                                          uint16_t* out_count) {
  uint16_t count = 0;
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RPAREN) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (count > 0) {
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
        break;
      }
    }
    if (count >= capacity) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_token_text_error(parser, &loom_err_parse_004,
                                               peek);
    }
    IREE_RETURN_IF_ERROR(loom_parse_type(parser, mode, &types[count]));
    ++count;
  }
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RPAREN)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("')'"));
  }
  *out_count = count;
  return iree_ok_status();
}

// Parses a function type: (arg_types) -> (result_types). Called after
// the opening LPAREN has been consumed.
static iree_status_t loom_parse_function_type(loom_parser_t* parser,
                                              loom_type_parse_mode_t mode,
                                              loom_type_t* out_type) {
  loom_type_t arg_types[16];
  uint16_t arg_count = 0;
  IREE_RETURN_IF_ERROR(loom_parse_type_list(
      parser, mode, arg_types, IREE_ARRAYSIZE(arg_types), &arg_count));

  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_ARROW)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'->'"));
  }
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LPAREN)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'('"));
  }

  loom_type_t result_types[16];
  uint16_t result_count = 0;
  IREE_RETURN_IF_ERROR(loom_parse_type_list(
      parser, mode, result_types, IREE_ARRAYSIZE(result_types), &result_count));

  return loom_type_function_build(
      arg_types, arg_count, result_types, result_count,
      iree_arena_allocator(&parser->parser_arena), out_type);
}

//===----------------------------------------------------------------------===//
// Type dispatch
//===----------------------------------------------------------------------===//

// Consumes keyword, expects LANGLE, dispatches to the type-specific
// parser. Shared entry for tile, tensor, pool, and group.
static iree_status_t loom_parse_angle_bracketed_type(
    loom_parser_t* parser, loom_type_kind_t kind, loom_type_parse_mode_t mode,
    loom_type_t* out_type) {
  loom_tokenizer_next(&parser->tokenizer);  // consume type keyword
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LANGLE)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'<'"));
  }
  iree_status_t status;
  if (kind == LOOM_TYPE_POOL) {
    status = loom_parse_pool_type(parser, mode, out_type);
  } else if (kind == LOOM_TYPE_GROUP) {
    status = loom_parse_group_type(parser, out_type);
  } else {
    status = loom_parse_shaped_type(parser, kind, mode, out_type);
  }
  parser->tokenizer.in_dim_list = false;  // cleanup on error paths
  return status;
}

iree_status_t loom_parse_type(loom_parser_t* parser,
                              loom_type_parse_mode_t mode,
                              loom_type_t* out_type) {
  loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);

  if (token.kind == LOOM_TOKEN_BARE_IDENT) {
    // Scalar type keyword (f32, i8, index, ...).
    loom_scalar_type_t scalar_type = 0;
    if (loom_scalar_type_parse(token.text, &scalar_type)) {
      loom_tokenizer_next(&parser->tokenizer);
      *out_type = loom_type_scalar(scalar_type);
      return iree_ok_status();
    }

    // "encoding" keyword.
    if (iree_string_view_equal(token.text, IREE_SV("encoding"))) {
      loom_tokenizer_next(&parser->tokenizer);
      *out_type = loom_type_encoding();
      return iree_ok_status();
    }

    // Angle-bracketed types: tile, tensor, pool, group.
    if (iree_string_view_equal(token.text, IREE_SV("tile"))) {
      return loom_parse_angle_bracketed_type(parser, LOOM_TYPE_TILE, mode,
                                             out_type);
    }
    if (iree_string_view_equal(token.text, IREE_SV("tensor"))) {
      return loom_parse_angle_bracketed_type(parser, LOOM_TYPE_TENSOR, mode,
                                             out_type);
    }
    if (iree_string_view_equal(token.text, IREE_SV("pool"))) {
      return loom_parse_angle_bracketed_type(parser, LOOM_TYPE_POOL, mode,
                                             out_type);
    }
    if (iree_string_view_equal(token.text, IREE_SV("group"))) {
      return loom_parse_angle_bracketed_type(parser, LOOM_TYPE_GROUP, mode,
                                             out_type);
    }

    // Unknown bare ident.
    loom_diagnostic_param_t params[] = {
        loom_param_string(token.text),
    };
    return loom_parser_emit(parser, &loom_err_parse_007, params,
                            IREE_ARRAYSIZE(params), token);
  }

  // Dialect type: dotted name (e.g., hal.buffer).
  if (token.kind == LOOM_TOKEN_OP_NAME) {
    loom_tokenizer_next(&parser->tokenizer);
    return loom_parse_dialect_type(parser, token, mode, out_type);
  }

  // Function type: (types) -> (types).
  if (token.kind == LOOM_TOKEN_LPAREN) {
    loom_tokenizer_next(&parser->tokenizer);  // consume '('
    return loom_parse_function_type(parser, mode, out_type);
  }

  return loom_parser_emit_unexpected_token(parser, token, IREE_SV("a type"));
}
