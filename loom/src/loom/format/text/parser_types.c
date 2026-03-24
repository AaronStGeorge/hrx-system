// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "loom/format/text/parser_internal.h"

//===----------------------------------------------------------------------===//
// Encoding parameter parsing
//===----------------------------------------------------------------------===//

iree_status_t loom_parse_encoding_params(loom_parser_t* parser,
                                         iree_string_view_t params_text,
                                         loom_named_attr_t** out_attrs,
                                         uint8_t* out_count) {
  // Count commas to estimate param count.
  uint8_t estimated_count = 1;
  for (iree_host_size_t i = 0; i < params_text.size; ++i) {
    if (params_text.data[i] == ',') ++estimated_count;
  }
  loom_named_attr_t* attrs = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&parser->module->arena, estimated_count,
                                sizeof(loom_named_attr_t), (void**)&attrs));
  uint8_t attr_count = 0;

  iree_host_size_t position = 0;
  while (position < params_text.size) {
    // Skip whitespace and commas.
    while (position < params_text.size && (params_text.data[position] == ' ' ||
                                           params_text.data[position] == ',')) {
      ++position;
    }
    if (position >= params_text.size) break;

    // Find '='.
    iree_host_size_t equals_position = position;
    while (equals_position < params_text.size &&
           params_text.data[equals_position] != '=') {
      ++equals_position;
    }
    if (equals_position >= params_text.size) {
      iree_string_view_t remaining = iree_make_string_view(
          params_text.data + position, params_text.size - position);
      loom_token_t diag_token = loom_make_synthetic_token(remaining);
      loom_diagnostic_param_t params[] = {
          loom_param_string(remaining),
          loom_param_string(IREE_SV("'='")),
      };
      IREE_RETURN_IF_ERROR(loom_parser_emit(parser, &loom_err_parse_003, params,
                                            IREE_ARRAYSIZE(params),
                                            diag_token));
      break;
    }
    iree_string_view_t param_name = iree_string_view_trim(iree_make_string_view(
        params_text.data + position, equals_position - position));

    // Find end of value (comma or end of string).
    iree_host_size_t value_start = equals_position + 1;
    iree_host_size_t value_end = value_start;
    while (value_end < params_text.size && params_text.data[value_end] != ',') {
      ++value_end;
    }
    iree_string_view_t param_value =
        iree_string_view_trim(iree_make_string_view(
            params_text.data + value_start, value_end - value_start));

    loom_string_id_t param_name_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_module_intern_string(parser->module, param_name, &param_name_id));

    // Parse value: integer if possible, otherwise string.
    int64_t int_value = 0;
    if (iree_string_view_atoi_int64(param_value, &int_value)) {
      attrs[attr_count].name_id = param_name_id;
      attrs[attr_count].reserved = 0;
      attrs[attr_count].value = loom_attr_i64(int_value);
    } else {
      loom_string_id_t value_string_id = 0;
      IREE_RETURN_IF_ERROR(loom_module_intern_string(
          parser->module, param_value, &value_string_id));
      attrs[attr_count].name_id = param_name_id;
      attrs[attr_count].reserved = 0;
      attrs[attr_count].value = loom_attr_string(value_string_id);
    }
    ++attr_count;
    position = value_end;
  }

  *out_attrs = attrs;
  *out_count = attr_count;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Shaped type interior parsing
//===----------------------------------------------------------------------===//

// Parses the interior of a shaped type from a character-level string
// slice (the text between < and > after "tile" or "tensor").
//
// Grammar: dimxdimx...xelement_type[, encoding]
// Dims: integer (static), [%name] (dynamic SSA), [#N] (ordinal)
// Element type: scalar keyword (f32, i8, index, etc.)
// Encoding: #alias, #name<params>, #N (ordinal), %enc (SSA encoding)
//
// This is character-level because the main tokenizer can't handle
// "4x4xf32" — it tokenizes as INTEGER "4" + BARE_IDENT "x4xf32".
static iree_status_t loom_parse_shaped_interior(loom_parser_t* parser,
                                                loom_type_kind_t kind,
                                                iree_string_view_t interior,
                                                loom_type_parse_mode_t mode,
                                                loom_type_t* out_type) {
  const char* data = interior.data;
  iree_host_size_t length = interior.size;

  // Split on first ',' outside nested <> to separate shape from encoding.
  iree_host_size_t comma_position = length;  // No comma = entire string.
  {
    int angle_depth = 0;
    for (iree_host_size_t i = 0; i < length; ++i) {
      if (data[i] == '<')
        ++angle_depth;
      else if (data[i] == '>')
        --angle_depth;
      else if (data[i] == ',' && angle_depth == 0) {
        comma_position = i;
        break;
      }
    }
  }

  iree_string_view_t shape_part = iree_make_string_view(data, comma_position);

  // Split shape part on 'x' outside [...] brackets.
  // Collect segments into a stack array. Max rank is 15 (4 bits in header).
  iree_string_view_t segments[16];
  uint8_t segment_count = 0;
  {
    iree_host_size_t segment_start = 0;
    int bracket_depth = 0;
    for (iree_host_size_t i = 0; i < shape_part.size; ++i) {
      if (shape_part.data[i] == '[')
        ++bracket_depth;
      else if (shape_part.data[i] == ']')
        --bracket_depth;
      else if (shape_part.data[i] == 'x' && bracket_depth == 0) {
        if (segment_count >= 16) {
          loom_token_t diag_token = loom_make_synthetic_token(interior);
          loom_diagnostic_param_t params[] = {
              loom_param_string(interior),
          };
          return loom_parser_emit(parser, &loom_err_parse_004, params,
                                  IREE_ARRAYSIZE(params), diag_token);
        }
        segments[segment_count++] = iree_make_string_view(
            shape_part.data + segment_start, i - segment_start);
        segment_start = i + 1;
      }
    }
    // Last segment (element type or sole segment for 0-d).
    if (segment_start <= shape_part.size) {
      if (segment_count >= 16) {
        loom_token_t diag_token = loom_make_synthetic_token(interior);
        loom_diagnostic_param_t params[] = {
            loom_param_string(interior),
        };
        return loom_parser_emit(parser, &loom_err_parse_004, params,
                                IREE_ARRAYSIZE(params), diag_token);
      }
      segments[segment_count++] = iree_make_string_view(
          shape_part.data + segment_start, shape_part.size - segment_start);
    }
  }

  if (segment_count == 0) {
    loom_token_t diag_token = loom_make_synthetic_token(interior);
    loom_diagnostic_param_t params[] = {
        loom_param_string(interior),
    };
    return loom_parser_emit(parser, &loom_err_parse_004, params,
                            IREE_ARRAYSIZE(params), diag_token);
  }

  // Last segment is the element type.
  iree_string_view_t element_text =
      iree_string_view_trim(segments[segment_count - 1]);
  loom_scalar_type_t element_type = 0;
  if (!loom_scalar_type_parse(element_text, &element_type)) {
    loom_token_t diag_token = loom_make_synthetic_token(element_text);
    loom_diagnostic_param_t params[] = {
        loom_param_string(element_text),
    };
    return loom_parser_emit(parser, &loom_err_parse_007, params,
                            IREE_ARRAYSIZE(params), diag_token);
  }

  // Preceding segments are dimensions.
  uint8_t rank = segment_count - 1;
  uint64_t dims[16];
  for (uint8_t i = 0; i < rank; ++i) {
    iree_string_view_t segment = iree_string_view_trim(segments[i]);
    if (segment.size >= 2 && segment.data[0] == '[' &&
        segment.data[segment.size - 1] == ']') {
      // Dynamic dim: [%name] or [#N].
      iree_string_view_t inner = iree_string_view_trim(
          iree_make_string_view(segment.data + 1, segment.size - 2));
      if (inner.size > 0 && inner.data[0] == '%') {
        // SSA dim reference. Strip the leading '%' — names in the
        // scope and value table never include it.
        iree_string_view_t name =
            iree_make_string_view(inner.data + 1, inner.size - 1);
        loom_value_id_t value_id = loom_scope_lookup(parser->scope, name);
        if (value_id == LOOM_VALUE_ID_INVALID) {
          if (mode == LOOM_TYPE_PARSE_ARG ||
              parser->definition_scope_depth > 0) {
            // Create a new index value for this dim name.
            loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
            IREE_RETURN_IF_ERROR(loom_module_define_value(
                parser->module, index_type, &value_id));
            // Intern the name and set it on the value.
            loom_string_id_t name_id = 0;
            IREE_RETURN_IF_ERROR(
                loom_module_intern_string(parser->module, name, &name_id));
            parser->module->values.entries[value_id].name_id = name_id;
            IREE_RETURN_IF_ERROR(loom_scope_define(
                parser->scope, &parser->parser_arena, name, value_id,
                /*out_duplicate=*/NULL));
          } else {
            loom_token_t diag_token = loom_make_synthetic_token(inner);
            loom_diagnostic_param_t params[] = {
                loom_param_string(inner),
            };
            IREE_RETURN_IF_ERROR(
                loom_parser_emit(parser, &loom_err_parse_001, params,
                                 IREE_ARRAYSIZE(params), diag_token));
            return iree_ok_status();
          }
        }
        dims[i] = loom_dim_pack_dynamic(value_id);
      } else if (inner.size > 1 && inner.data[0] == '#') {
        // Ordinal dim reference: [#N].
        if (mode != LOOM_TYPE_PARSE_RETURN) {
          loom_token_t diag_token = loom_make_synthetic_token(segment);
          loom_diagnostic_param_t params[] = {
              loom_param_string(segment),
          };
          return loom_parser_emit(parser, &loom_err_parse_004, params,
                                  IREE_ARRAYSIZE(params), diag_token);
        }
        iree_string_view_t ordinal_text =
            iree_make_string_view(inner.data + 1, inner.size - 1);
        uint32_t ordinal = 0;
        if (!iree_string_view_atoi_uint32(ordinal_text, &ordinal)) {
          loom_token_t diag_token = loom_make_synthetic_token(ordinal_text);
          loom_diagnostic_param_t params[] = {
              loom_param_string(ordinal_text),
          };
          return loom_parser_emit(parser, &loom_err_parse_015, params,
                                  IREE_ARRAYSIZE(params), diag_token);
        }
        dims[i] = loom_dim_pack_ordinal(ordinal);
      } else {
        loom_token_t diag_token = loom_make_synthetic_token(segment);
        loom_diagnostic_param_t params[] = {
            loom_param_string(segment),
        };
        return loom_parser_emit(parser, &loom_err_parse_004, params,
                                IREE_ARRAYSIZE(params), diag_token);
      }
    } else {
      // Static dim: integer.
      int64_t size = 0;
      if (!iree_string_view_atoi_int64(segment, &size) || size < 0) {
        loom_token_t diag_token = loom_make_synthetic_token(segment);
        loom_diagnostic_param_t params[] = {
            loom_param_string(segment),
        };
        return loom_parser_emit(parser, &loom_err_parse_015, params,
                                IREE_ARRAYSIZE(params), diag_token);
      }
      dims[i] = loom_dim_pack_static(size);
    }
  }

  // Parse encoding if present (after the comma).
  uint16_t encoding_id = 0;
  loom_encoding_flags_t encoding_flags = 0;
  if (comma_position < length) {
    iree_string_view_t encoding_text =
        iree_string_view_trim(iree_make_string_view(
            data + comma_position + 1, length - comma_position - 1));
    if (encoding_text.size > 0 && encoding_text.data[0] == '%') {
      // SSA encoding reference. Strip the leading '%'.
      iree_string_view_t encoding_name =
          iree_make_string_view(encoding_text.data + 1, encoding_text.size - 1);
      loom_value_id_t enc_value_id =
          loom_scope_lookup(parser->scope, encoding_name);
      if (enc_value_id == LOOM_VALUE_ID_INVALID) {
        if (mode == LOOM_TYPE_PARSE_ARG || parser->definition_scope_depth > 0) {
          // Create a new encoding value for this name (type variable
          // at definition site or function arg).
          loom_type_t enc_type = loom_type_encoding();
          IREE_RETURN_IF_ERROR(loom_module_define_value(
              parser->module, enc_type, &enc_value_id));
          loom_string_id_t name_id = 0;
          IREE_RETURN_IF_ERROR(loom_module_intern_string(
              parser->module, encoding_name, &name_id));
          parser->module->values.entries[enc_value_id].name_id = name_id;
          IREE_RETURN_IF_ERROR(loom_scope_define(
              parser->scope, &parser->parser_arena, encoding_name, enc_value_id,
              /*out_duplicate=*/NULL));
        } else {
          loom_token_t diag_token = loom_make_synthetic_token(encoding_text);
          loom_diagnostic_param_t params[] = {
              loom_param_string(encoding_text),
          };
          return loom_parser_emit(parser, &loom_err_parse_001, params,
                                  IREE_ARRAYSIZE(params), diag_token);
        }
      }
      encoding_id = (uint16_t)enc_value_id;
      encoding_flags = LOOM_ENCODING_FLAG_SSA;
    } else if (encoding_text.size > 0 && encoding_text.data[0] == '#') {
      // Encoding starting with '#': either an ordinal (#N) or a static
      // encoding (#alias, #name<params>). Ordinals are purely numeric
      // after the '#'; encoding names always start with a letter.
      iree_string_view_t after_hash =
          iree_make_string_view(encoding_text.data + 1, encoding_text.size - 1);
      uint32_t ordinal = 0;
      if (after_hash.size > 0 && after_hash.data[0] >= '0' &&
          after_hash.data[0] <= '9') {
        // Encoding ordinal: #N.
        if (mode != LOOM_TYPE_PARSE_RETURN) {
          loom_token_t diag_token = loom_make_synthetic_token(encoding_text);
          loom_diagnostic_param_t params[] = {
              loom_param_string(encoding_text),
          };
          return loom_parser_emit(parser, &loom_err_parse_004, params,
                                  IREE_ARRAYSIZE(params), diag_token);
        }
        if (!iree_string_view_atoi_uint32(after_hash, &ordinal)) {
          loom_token_t diag_token = loom_make_synthetic_token(after_hash);
          loom_diagnostic_param_t params[] = {
              loom_param_string(after_hash),
          };
          return loom_parser_emit(parser, &loom_err_parse_015, params,
                                  IREE_ARRAYSIZE(params), diag_token);
        }
        encoding_id = (uint16_t)ordinal;
        encoding_flags = LOOM_ENCODING_FLAG_ORDINAL;
      } else {
        // Static encoding: #alias or #name<params>.
        iree_host_size_t angle_position = after_hash.size;
        for (iree_host_size_t i = 0; i < after_hash.size; ++i) {
          if (after_hash.data[i] == '<') {
            angle_position = i;
            break;
          }
        }
        iree_string_view_t name_part =
            iree_make_string_view(after_hash.data, angle_position);

        // First, check alias table for the full text (including #).
        uint16_t alias_id =
            loom_alias_table_lookup(&parser->aliases, encoding_text);
        if (alias_id != 0) {
          encoding_id = alias_id;
        } else {
          // Build an encoding instance and register it.
          loom_encoding_t encoding = {0};
          IREE_RETURN_IF_ERROR(loom_module_intern_string(
              parser->module, name_part, &encoding.name_id));
          encoding.alias_id = LOOM_STRING_ID_INVALID;

          if (angle_position < after_hash.size) {
            // Has params: #name<key=value, ...>.
            iree_string_view_t params_text = iree_make_string_view(
                after_hash.data + angle_position + 1,
                after_hash.size - angle_position - 2);  // Strip < and >.

            loom_named_attr_t* attrs = NULL;
            uint8_t attr_count = 0;
            IREE_RETURN_IF_ERROR(loom_parse_encoding_params(
                parser, params_text, &attrs, &attr_count));
            encoding.attribute_count = attr_count;
            encoding.attributes = attrs;
          }

          IREE_RETURN_IF_ERROR(loom_module_add_encoding(
              parser->module, &encoding, &encoding_id));
        }
      }
    } else if (encoding_text.size > 0) {
      loom_token_t diag_token = loom_make_synthetic_token(encoding_text);
      loom_diagnostic_param_t params[] = {
          loom_param_string(encoding_text),
      };
      return loom_parser_emit(parser, &loom_err_parse_008, params,
                              IREE_ARRAYSIZE(params), diag_token);
    }
  }

  // Construct the type.
  loom_type_t type = {0};
  if (rank == 0) {
    type = loom_type_shaped_0d(kind, element_type, encoding_id);
  } else if (rank == 1) {
    type = loom_type_shaped_1d(kind, element_type, dims[0], encoding_id);
  } else if (rank == 2) {
    type =
        loom_type_shaped_2d(kind, element_type, dims[0], dims[1], encoding_id);
  } else {
    // Rank > 2: allocate overflow dims in the module arena.
    loom_overflow_dim_t* overflow = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&parser->module->arena, rank,
                                                   sizeof(loom_overflow_dim_t),
                                                   (void**)&overflow));
    memcpy(overflow, dims, rank * sizeof(uint64_t));

    // Compute flags.
    uint8_t flags = 0;  // No LOOM_TYPE_FLAG_INLINE_DIMS for overflow.
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

  // Set encoding flags for inline types.
  if (rank <= 2) {
    type.encoding_flags = encoding_flags;
  }

  // Intern the type.
  return loom_module_intern_type(parser->module, type, out_type);
}

//===----------------------------------------------------------------------===//
// Pool type interior parsing
//===----------------------------------------------------------------------===//

// Parses a pool type interior: single dim, no element type.
static iree_status_t loom_parse_pool_interior(loom_parser_t* parser,
                                              iree_string_view_t interior,
                                              loom_type_parse_mode_t mode,
                                              loom_type_t* out_type) {
  iree_string_view_t trimmed = iree_string_view_trim(interior);
  uint64_t dim;
  if (trimmed.size >= 2 && trimmed.data[0] == '[' &&
      trimmed.data[trimmed.size - 1] == ']') {
    // Dynamic dim.
    iree_string_view_t inner = iree_string_view_trim(
        iree_make_string_view(trimmed.data + 1, trimmed.size - 2));
    if (inner.size > 0 && inner.data[0] == '%') {
      loom_value_id_t value_id = loom_scope_lookup(parser->scope, inner);
      if (value_id == LOOM_VALUE_ID_INVALID) {
        if (mode == LOOM_TYPE_PARSE_ARG) {
          loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
          IREE_RETURN_IF_ERROR(
              loom_module_define_value(parser->module, index_type, &value_id));
          loom_string_id_t name_id = 0;
          IREE_RETURN_IF_ERROR(
              loom_module_intern_string(parser->module, inner, &name_id));
          parser->module->values.entries[value_id].name_id = name_id;
          IREE_RETURN_IF_ERROR(loom_scope_define(
              parser->scope, &parser->parser_arena, inner, value_id,
              /*out_duplicate=*/NULL));
        } else {
          loom_token_t diag_token = loom_make_synthetic_token(inner);
          loom_diagnostic_param_t params[] = {
              loom_param_string(inner),
          };
          return loom_parser_emit(parser, &loom_err_parse_001, params,
                                  IREE_ARRAYSIZE(params), diag_token);
        }
      }
      dim = loom_dim_pack_dynamic(value_id);
    } else {
      loom_token_t diag_token = loom_make_synthetic_token(trimmed);
      loom_diagnostic_param_t params[] = {
          loom_param_string(trimmed),
      };
      return loom_parser_emit(parser, &loom_err_parse_004, params,
                              IREE_ARRAYSIZE(params), diag_token);
    }
  } else {
    // Static dim.
    int64_t size = 0;
    if (!iree_string_view_atoi_int64(trimmed, &size) || size < 0) {
      loom_token_t diag_token = loom_make_synthetic_token(trimmed);
      loom_diagnostic_param_t params[] = {
          loom_param_string(trimmed),
      };
      return loom_parser_emit(parser, &loom_err_parse_015, params,
                              IREE_ARRAYSIZE(params), diag_token);
    }
    dim = loom_dim_pack_static(size);
  }
  *out_type = loom_type_pool(dim);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Type dispatch
//===----------------------------------------------------------------------===//

iree_status_t loom_parse_type(loom_parser_t* parser,
                              loom_type_parse_mode_t mode,
                              loom_type_t* out_type) {
  loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);

  // Scalar type keyword.
  if (token.kind == LOOM_TOKEN_BARE_IDENT) {
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

    // Shaped types: tile, tensor, pool.
    if (iree_string_view_equal(token.text, IREE_SV("tile")) ||
        iree_string_view_equal(token.text, IREE_SV("tensor")) ||
        iree_string_view_equal(token.text, IREE_SV("pool"))) {
      loom_type_kind_t kind = 0;
      if (iree_string_view_equal(token.text, IREE_SV("tile"))) {
        kind = LOOM_TYPE_TILE;
      } else if (iree_string_view_equal(token.text, IREE_SV("tensor"))) {
        kind = LOOM_TYPE_TENSOR;
      } else {
        kind = LOOM_TYPE_POOL;
      }
      loom_tokenizer_next(&parser->tokenizer);
      // Consume '<', scan interior, consume '>'.
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LANGLE)) {
        loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
        loom_diagnostic_param_t params[] = {
            loom_param_string(peek.text),
            loom_param_string(IREE_SV("'<'")),
        };
        return loom_parser_emit(parser, &loom_err_parse_003, params,
                                IREE_ARRAYSIZE(params), peek);
      }
      iree_string_view_t interior = {0};
      IREE_RETURN_IF_ERROR(
          loom_tokenizer_scan_angle_interior(&parser->tokenizer, &interior));
      if (kind == LOOM_TYPE_POOL) {
        return loom_parse_pool_interior(parser, interior, mode, out_type);
      }
      return loom_parse_shaped_interior(parser, kind, interior, mode, out_type);
    }

    // Group type: group<scope>.
    if (iree_string_view_equal(token.text, IREE_SV("group"))) {
      loom_tokenizer_next(&parser->tokenizer);
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LANGLE)) {
        loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
        loom_diagnostic_param_t params[] = {
            loom_param_string(peek.text),
            loom_param_string(IREE_SV("'<'")),
        };
        return loom_parser_emit(parser, &loom_err_parse_003, params,
                                IREE_ARRAYSIZE(params), peek);
      }
      iree_string_view_t interior = {0};
      IREE_RETURN_IF_ERROR(
          loom_tokenizer_scan_angle_interior(&parser->tokenizer, &interior));
      iree_string_view_t scope_name = iree_string_view_trim(interior);
      loom_type_t type = {0};
      if (iree_string_view_equal(scope_name, IREE_SV("workgroup"))) {
        type.header = loom_type_make_header(
            LOOM_TYPE_GROUP, (loom_scalar_type_t)LOOM_GROUP_SCOPE_WORKGROUP, 0,
            0);
      } else if (iree_string_view_equal(scope_name, IREE_SV("subgroup"))) {
        type.header = loom_type_make_header(
            LOOM_TYPE_GROUP, (loom_scalar_type_t)LOOM_GROUP_SCOPE_SUBGROUP, 0,
            0);
      } else {
        loom_token_t diag_token = loom_make_synthetic_token(scope_name);
        loom_diagnostic_param_t params[] = {
            loom_param_string(IREE_SV("group scope")),
            loom_param_string(scope_name),
        };
        return loom_parser_emit(parser, &loom_err_parse_018, params,
                                IREE_ARRAYSIZE(params), diag_token);
      }
      *out_type = type;
      return iree_ok_status();
    }

    // Unknown bare ident — check for dialect types.
    loom_diagnostic_param_t params[] = {
        loom_param_string(token.text),
    };
    return loom_parser_emit(parser, &loom_err_parse_007, params,
                            IREE_ARRAYSIZE(params), token);
  }

  // Dialect types with dotted names (e.g., hal.buffer).
  if (token.kind == LOOM_TOKEN_OP_NAME) {
    loom_tokenizer_next(&parser->tokenizer);
    loom_string_id_t name_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_module_intern_string(parser->module, token.text, &name_id));
    // Check for parameterized dialect type: name<params>.
    if (loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_LANGLE)) {
      loom_tokenizer_next(&parser->tokenizer);  // Consume '<'.
      // Parse type parameters.
      loom_type_t param_types[8];
      uint16_t param_count = 0;
      while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RANGLE) &&
             !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
        if (param_count > 0) {
          if (!loom_tokenizer_try_consume(&parser->tokenizer,
                                          LOOM_TOKEN_COMMA)) {
            break;
          }
        }
        if (param_count >= 8) {
          loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
          loom_diagnostic_param_t params[] = {
              loom_param_string(token.text),
          };
          return loom_parser_emit(parser, &loom_err_parse_004, params,
                                  IREE_ARRAYSIZE(params), peek);
        }
        IREE_RETURN_IF_ERROR(
            loom_parse_type(parser, mode, &param_types[param_count]));
        ++param_count;
      }
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RANGLE)) {
        loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
        loom_diagnostic_param_t params[] = {
            loom_param_string(peek.text),
            loom_param_string(IREE_SV("'>'")),
        };
        return loom_parser_emit(parser, &loom_err_parse_003, params,
                                IREE_ARRAYSIZE(params), peek);
      }
      if (param_count > 0) {
        loom_type_t* arena_params = NULL;
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            &parser->module->arena, param_count, sizeof(loom_type_t),
            (void**)&arena_params));
        memcpy(arena_params, param_types, param_count * sizeof(loom_type_t));
        *out_type = loom_type_dialect(name_id, param_count, arena_params);
      } else {
        *out_type = loom_type_dialect_opaque(name_id);
      }
    } else {
      *out_type = loom_type_dialect_opaque(name_id);
    }
    return iree_ok_status();
  }

  // Function type: (types) -> (types).
  if (token.kind == LOOM_TOKEN_LPAREN) {
    loom_tokenizer_next(&parser->tokenizer);  // Consume '('.
    // Parse argument types.
    loom_type_t arg_types[16];
    uint16_t arg_count = 0;
    while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RPAREN) &&
           !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
      if (arg_count > 0) {
        if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
          break;
        }
      }
      if (arg_count >= 16) {
        loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
        loom_diagnostic_param_t params[] = {
            loom_param_string(peek.text),
        };
        return loom_parser_emit(parser, &loom_err_parse_004, params,
                                IREE_ARRAYSIZE(params), peek);
      }
      IREE_RETURN_IF_ERROR(
          loom_parse_type(parser, mode, &arg_types[arg_count]));
      ++arg_count;
    }
    if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RPAREN)) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      loom_diagnostic_param_t params[] = {
          loom_param_string(peek.text),
          loom_param_string(IREE_SV("')'")),
      };
      return loom_parser_emit(parser, &loom_err_parse_003, params,
                              IREE_ARRAYSIZE(params), peek);
    }
    // Expect '->'.
    if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_ARROW)) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      loom_diagnostic_param_t params[] = {
          loom_param_string(peek.text),
          loom_param_string(IREE_SV("'->'")),
      };
      return loom_parser_emit(parser, &loom_err_parse_003, params,
                              IREE_ARRAYSIZE(params), peek);
    }
    // Expect '('.
    if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LPAREN)) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      loom_diagnostic_param_t params[] = {
          loom_param_string(peek.text),
          loom_param_string(IREE_SV("'('")),
      };
      return loom_parser_emit(parser, &loom_err_parse_003, params,
                              IREE_ARRAYSIZE(params), peek);
    }
    loom_type_t result_types[16];
    uint16_t result_count = 0;
    while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RPAREN) &&
           !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
      if (result_count > 0) {
        if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
          break;
        }
      }
      if (result_count >= 16) {
        loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
        loom_diagnostic_param_t params[] = {
            loom_param_string(peek.text),
        };
        return loom_parser_emit(parser, &loom_err_parse_004, params,
                                IREE_ARRAYSIZE(params), peek);
      }
      IREE_RETURN_IF_ERROR(
          loom_parse_type(parser, mode, &result_types[result_count]));
      ++result_count;
    }
    if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RPAREN)) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      loom_diagnostic_param_t params[] = {
          loom_param_string(peek.text),
          loom_param_string(IREE_SV("')'")),
      };
      return loom_parser_emit(parser, &loom_err_parse_003, params,
                              IREE_ARRAYSIZE(params), peek);
    }
    // Build the function type.
    return loom_type_function_build(
        arg_types, arg_count, result_types, result_count,
        iree_arena_allocator(&parser->module->arena), out_type);
  }

  loom_diagnostic_param_t params[] = {
      loom_param_string(token.text),
      loom_param_string(IREE_SV("a type")),
  };
  return loom_parser_emit(parser, &loom_err_parse_003, params,
                          IREE_ARRAYSIZE(params), token);
}
