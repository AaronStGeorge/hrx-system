// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/printer.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/internal/unicode.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

//===----------------------------------------------------------------------===//
// Print context — spacing + indentation on top of output stream
//===----------------------------------------------------------------------===//

typedef struct loom_print_context_t {
  loom_output_stream_t* stream;
  const loom_module_t* module;
  loom_text_print_flags_t flags;
  bool has_previous_token;
  bool glue_next;
  char last_char;
  uint16_t indent;
  loom_print_field_callback_t field_callback;
} loom_print_context_t;

static bool loom_is_backward_glue(char c) {
  return c == ',' || c == ')' || c == ']' || c == '}';
}

static bool loom_is_forward_glue(char c) {
  return c == '(' || c == '[' || c == '{';
}

// Emits text with automatic spacing. Applies backward/forward glue
// rules and the explicit glue flag.
static iree_status_t loom_print_emit(loom_print_context_t* ctx,
                                     iree_string_view_t text, bool glue) {
  if (text.size == 0) return iree_ok_status();
  bool suppress_space = glue || ctx->glue_next || !ctx->has_previous_token ||
                        loom_is_backward_glue(text.data[0]) ||
                        loom_is_forward_glue(ctx->last_char);
  ctx->glue_next = false;
  if (!suppress_space) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, ' '));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write(ctx->stream, text));
  ctx->has_previous_token = true;
  ctx->last_char = text.data[text.size - 1];
  return iree_ok_status();
}

static iree_status_t loom_print_emit_cstr(loom_print_context_t* ctx,
                                          const char* text, bool glue) {
  return loom_print_emit(ctx, iree_make_cstring_view(text), glue);
}

static void loom_print_set_glue(loom_print_context_t* ctx) {
  ctx->glue_next = true;
}

// Emits a space if the spacing rules require one before the next token.
// Call before writing directly to the stream for content that is not
// backward-glue punctuation (types, attributes, integers — never start
// with ,)]}). After writing, call loom_print_did_write to update state.
static iree_status_t loom_print_space_if_needed(loom_print_context_t* ctx) {
  bool suppress = ctx->glue_next || !ctx->has_previous_token ||
                  loom_is_forward_glue(ctx->last_char);
  ctx->glue_next = false;
  if (!suppress) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, ' '));
  }
  return iree_ok_status();
}

// Updates spacing state after writing content directly to the stream.
static void loom_print_did_write(loom_print_context_t* ctx) {
  ctx->has_previous_token = true;
  ctx->last_char = ' ';
}

static iree_host_size_t loom_print_next_token_start_offset(
    const loom_print_context_t* ctx, bool glue, char first_char) {
  bool suppress_space = glue || ctx->glue_next || !ctx->has_previous_token ||
                        loom_is_backward_glue(first_char) ||
                        loom_is_forward_glue(ctx->last_char);
  return ctx->stream->offset + (suppress_space ? 0 : 1);
}

static void loom_print_report_field(loom_print_context_t* ctx,
                                    loom_print_field_ref_t field_ref,
                                    iree_host_size_t start,
                                    iree_host_size_t end) {
  if (!ctx->field_callback.fn) return;
  ctx->field_callback.fn(ctx->field_callback.user_data, field_ref, start, end);
}

// Writes indentation spaces directly to the stream.
static const char SPACES[] = "                                ";
static iree_status_t loom_print_indent(loom_print_context_t* ctx) {
  if (!iree_any_bit_set(ctx->flags, LOOM_TEXT_PRINT_INDENT)) {
    return iree_ok_status();
  }
  iree_host_size_t count = (iree_host_size_t)ctx->indent * 2;
  while (count > 0) {
    iree_host_size_t chunk =
        count < sizeof(SPACES) - 1 ? count : sizeof(SPACES) - 1;
    IREE_RETURN_IF_ERROR(loom_output_stream_write(
        ctx->stream, iree_make_string_view(SPACES, chunk)));
    count -= chunk;
  }
  return iree_ok_status();
}

// Forward declaration — defined after the type printing section.
static iree_status_t loom_print_attr(loom_output_stream_t* stream,
                                     const loom_attribute_t* attr,
                                     const loom_module_t* module,
                                     const loom_attr_descriptor_t* descriptor);

// Emits a canonical JSON-compatible string literal. Stored strings are expected
// to contain decoded UTF-8 payload bytes; this helper validates that invariant
// before writing so malformed IR never serializes as malformed text.
static iree_status_t loom_print_string_literal(loom_output_stream_t* stream,
                                               iree_string_view_t text) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '"'));
  iree_host_size_t position = 0;
  while (position < text.size) {
    iree_host_size_t codepoint_start = position;
    uint32_t codepoint = iree_unicode_utf8_decode(text, &position);
    if (codepoint == IREE_UNICODE_REPLACEMENT_CHAR) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid UTF-8 string literal");
    }
    switch (codepoint) {
      case '"': {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\\\""));
        break;
      }
      case '\\': {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\\\\"));
        break;
      }
      case '\b': {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\\b"));
        break;
      }
      case '\f': {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\\f"));
        break;
      }
      case '\n': {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\\n"));
        break;
      }
      case '\r': {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\\r"));
        break;
      }
      case '\t': {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\\t"));
        break;
      }
      default: {
        if (codepoint < 0x20) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
              stream, "\\u%04" PRIX32, codepoint));
        } else {
          IREE_RETURN_IF_ERROR(loom_output_stream_write(
              stream, iree_make_string_view(text.data + codepoint_start,
                                            position - codepoint_start)));
        }
        break;
      }
    }
  }
  return loom_output_stream_write_char(stream, '"');
}

//===----------------------------------------------------------------------===//
// Type printing — writes directly to stream
//===----------------------------------------------------------------------===//

static iree_status_t loom_print_scalar_type(loom_output_stream_t* stream,
                                            loom_scalar_type_t scalar) {
  const char* name = loom_scalar_type_name(scalar);
  if (!name) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown scalar type %d", (int)scalar);
  }
  return loom_output_stream_write_cstring(stream, name);
}

// Writes %name for a value_id to the stream. Named values use their
// string name (%foo). Unnamed values use their value_id (%0, %1, ...).
// These occupy separate syntactic namespaces (identifiers vs digits)
// so no collision is possible. Falls back to %? only when the module
// context is missing or the value_id is out of range.
static iree_status_t loom_print_value_ref(loom_output_stream_t* stream,
                                          const loom_module_t* module,
                                          loom_value_id_t value_id) {
  if (module && value_id < module->values.count) {
    loom_string_id_t name_id = module->values.entries[value_id].name_id;
    if (name_id != LOOM_STRING_ID_INVALID && name_id < module->strings.count) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '%'));
      return loom_output_stream_write(stream, module->strings.entries[name_id]);
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '%'));
    return loom_output_stream_write_format(stream, "%" PRIu32, value_id);
  }
  return loom_output_stream_write_cstring(stream, "%?");
}

static iree_status_t loom_print_canonical_encoding(
    loom_output_stream_t* stream, const loom_module_t* module,
    const loom_encoding_t* encoding) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(encoding);

  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '#'));
  if (encoding->name_id < module->strings.count) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write(
        stream, module->strings.entries[encoding->name_id]));
  }
  if (encoding->attribute_count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '<'));
  for (uint8_t i = 0; i < encoding->attribute_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
    }
    const loom_named_attr_t* param = &encoding->attributes[i];
    if (param->name_id < module->strings.count) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write(
          stream, module->strings.entries[param->name_id]));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '='));
    IREE_RETURN_IF_ERROR(loom_print_attr(stream, &param->value, module, NULL));
  }
  return loom_output_stream_write_char(stream, '>');
}

static iree_status_t loom_print_static_encoding(loom_output_stream_t* stream,
                                                const loom_module_t* module,
                                                uint16_t encoding_id) {
  if (module && encoding_id > 0 && encoding_id <= module->encodings.count) {
    const loom_encoding_t* encoding =
        &module->encodings.entries[encoding_id - 1];
    if (encoding->alias_id != LOOM_STRING_ID_INVALID &&
        encoding->alias_id < module->strings.count) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '#'));
      return loom_output_stream_write(
          stream, module->strings.entries[encoding->alias_id]);
    }
    return loom_print_canonical_encoding(stream, module, encoding);
  }

  return loom_output_stream_write_format(stream, "#encoding_%" PRIu16,
                                         encoding_id);
}

static iree_status_t loom_print_dim(loom_output_stream_t* stream,
                                    loom_type_t type,
                                    iree_host_size_t dim_index,
                                    const loom_module_t* module) {
  uint64_t packed = loom_type_dim(type, dim_index);
  if (loom_dim_is_dynamic(packed)) {
    // SSA value reference: [%name].
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '['));
    IREE_RETURN_IF_ERROR(
        loom_print_value_ref(stream, module, loom_dim_value_id(packed)));
    return loom_output_stream_write_char(stream, ']');
  }
  return loom_output_stream_write_format(stream, "%" PRId64,
                                         loom_dim_static_size(packed));
}

static iree_status_t loom_print_shaped_interior(loom_output_stream_t* stream,
                                                loom_type_t type,
                                                const loom_module_t* module) {
  uint8_t rank = loom_type_rank(type);
  for (uint8_t i = 0; i < rank; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "x"));
    }
    IREE_RETURN_IF_ERROR(loom_print_dim(stream, type, i, module));
  }
  if (rank > 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "x"));
  }
  IREE_RETURN_IF_ERROR(
      loom_print_scalar_type(stream, loom_type_element_type(type)));
  if (loom_type_has_encoding(type)) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
    if (loom_type_has_ssa_encoding(type)) {
      // SSA encoding: %name (or %N if unnamed).
      IREE_RETURN_IF_ERROR(loom_print_value_ref(
          stream, module, loom_type_encoding_value_id(type)));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_print_static_encoding(stream, module, type.encoding_id));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_print_shaped_type_prefix(loom_output_stream_t* stream,
                                                   loom_type_kind_t kind) {
  switch (kind) {
    case LOOM_TYPE_TILE:
      return loom_output_stream_write_cstring(stream, "tile<");
    case LOOM_TYPE_TENSOR:
      return loom_output_stream_write_cstring(stream, "tensor<");
    case LOOM_TYPE_VECTOR:
      return loom_output_stream_write_cstring(stream, "vector<");
    case LOOM_TYPE_VIEW:
      return loom_output_stream_write_cstring(stream, "view<");
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "type kind %d is not shaped", (int)kind);
  }
}

iree_status_t loom_text_print_type(loom_type_t type,
                                   const loom_module_t* module,
                                   loom_output_stream_t* stream) {
  switch (loom_type_kind(type)) {
    case LOOM_TYPE_NONE:
      return loom_output_stream_write_cstring(stream, "none");
    case LOOM_TYPE_SCALAR:
      return loom_print_scalar_type(stream, loom_type_element_type(type));
    case LOOM_TYPE_TILE:
    case LOOM_TYPE_TENSOR:
    case LOOM_TYPE_VECTOR:
    case LOOM_TYPE_VIEW: {
      IREE_RETURN_IF_ERROR(
          loom_print_shaped_type_prefix(stream, loom_type_kind(type)));
      IREE_RETURN_IF_ERROR(loom_print_shaped_interior(stream, type, module));
      return loom_output_stream_write_cstring(stream, ">");
    }
    case LOOM_TYPE_GROUP: {
      const char* scope_name =
          loom_group_scope_name(loom_type_group_scope(type));
      if (!scope_name) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown group scope %d",
                                (int)loom_type_group_scope(type));
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "group<"));
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, scope_name));
      return loom_output_stream_write_cstring(stream, ">");
    }
    case LOOM_TYPE_DIALECT: {
      loom_string_id_t name_id = loom_type_dialect_name_id(type);
      if (module && name_id < module->strings.count) {
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write(stream, module->strings.entries[name_id]));
      } else {
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write_cstring(stream, "?dialect"));
      }
      uint16_t param_count = loom_type_dialect_param_count(type);
      if (param_count > 0) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '<'));
        const loom_type_t* params = loom_type_dialect_params(type);
        for (uint16_t i = 0; i < param_count; ++i) {
          if (i > 0) {
            IREE_RETURN_IF_ERROR(
                loom_output_stream_write_cstring(stream, ", "));
          }
          IREE_RETURN_IF_ERROR(loom_text_print_type(params[i], module, stream));
        }
        IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '>'));
      }
      return iree_ok_status();
    }
    case LOOM_TYPE_ENCODING:
      return loom_output_stream_write_cstring(stream, "encoding");
    case LOOM_TYPE_BUFFER:
      return loom_output_stream_write_cstring(stream, "buffer");
    case LOOM_TYPE_POOL: {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "pool<"));
      IREE_RETURN_IF_ERROR(loom_print_dim(stream, type, 0, module));
      return loom_output_stream_write_cstring(stream, ">");
    }
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* func_data = loom_type_func_data(type);
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '('));
      for (uint16_t i = 0; i < func_data->arg_count; ++i) {
        if (i > 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
        }
        IREE_RETURN_IF_ERROR(
            loom_text_print_type(func_data->types[i], module, stream));
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ") -> ("));
      for (uint16_t i = 0; i < func_data->result_count; ++i) {
        if (i > 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
        }
        IREE_RETURN_IF_ERROR(loom_text_print_type(
            func_data->types[func_data->arg_count + i], module, stream));
      }
      return loom_output_stream_write_char(stream, ')');
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown type kind %d",
                              (int)loom_type_kind(type));
  }
}

static iree_status_t loom_print_result_type_dim(loom_output_stream_t* stream,
                                                loom_type_t type,
                                                iree_host_size_t dim_index,
                                                const loom_module_t* module) {
  return loom_print_dim(stream, type, dim_index, module);
}

static iree_status_t loom_text_print_result_type(loom_type_t type,
                                                 const loom_module_t* module,
                                                 loom_output_stream_t* stream) {
  switch (loom_type_kind(type)) {
    case LOOM_TYPE_TILE:
    case LOOM_TYPE_TENSOR:
    case LOOM_TYPE_VECTOR:
    case LOOM_TYPE_VIEW: {
      IREE_RETURN_IF_ERROR(
          loom_print_shaped_type_prefix(stream, loom_type_kind(type)));
      uint8_t rank = loom_type_rank(type);
      for (uint8_t i = 0; i < rank; ++i) {
        if (i > 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "x"));
        }
        IREE_RETURN_IF_ERROR(
            loom_print_result_type_dim(stream, type, i, module));
      }
      if (rank > 0) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "x"));
      }
      IREE_RETURN_IF_ERROR(
          loom_print_scalar_type(stream, loom_type_element_type(type)));
      if (loom_type_has_encoding(type)) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
        if (loom_type_has_ssa_encoding(type)) {
          IREE_RETURN_IF_ERROR(loom_print_value_ref(
              stream, module, loom_type_encoding_value_id(type)));
        } else {
          IREE_RETURN_IF_ERROR(
              loom_print_static_encoding(stream, module, type.encoding_id));
        }
      }
      return loom_output_stream_write_cstring(stream, ">");
    }
    default:
      return loom_text_print_type(type, module, stream);
  }
}

//===----------------------------------------------------------------------===//
// Location printing — writes directly to stream
//===----------------------------------------------------------------------===//

static iree_status_t loom_print_location_source(
    const loom_module_t* module, loom_source_id_t source_id,
    iree_string_view_t* out_source) {
  if (!module->context || source_id >= module->context->sources.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "location source_id %u out of range", source_id);
  }
  *out_source = module->context->sources.entries[source_id];
  return iree_ok_status();
}

static iree_status_t loom_print_location_body(loom_output_stream_t* stream,
                                              const loom_module_t* module,
                                              loom_location_id_t location_id) {
  if (location_id >= module->locations.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "location_id %u out of range (module has %" PRIhsz
                            " locations)",
                            location_id, module->locations.count);
  }
  const loom_location_entry_t* entry = &module->locations.entries[location_id];
  switch (entry->kind) {
    case LOOM_LOCATION_NONE:
      return iree_ok_status();
    case LOOM_LOCATION_FILE: {
      iree_string_view_t source = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(
          loom_print_location_source(module, entry->file.source_id, &source));
      IREE_RETURN_IF_ERROR(loom_print_string_literal(stream, source));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          stream, ":%u:%u", entry->file.start_line, entry->file.start_col));
      if (entry->file.end_line != entry->file.start_line ||
          entry->file.end_col != entry->file.start_col) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
            stream, " to %u:%u", entry->file.end_line, entry->file.end_col));
      }
      return iree_ok_status();
    }
    case LOOM_LOCATION_FUSED: {
      if (entry->fused.count > 0 && !entry->fused.children) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "fused location has count %u but NULL children",
                                entry->fused.count);
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "fused<"));
      for (uint32_t i = 0; i < entry->fused.count; ++i) {
        if (i > 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
        }
        IREE_RETURN_IF_ERROR(
            loom_print_location_body(stream, module, entry->fused.children[i]));
      }
      return loom_output_stream_write_char(stream, '>');
    }
    case LOOM_LOCATION_OPAQUE: {
      iree_string_view_t tag = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(
          loom_print_location_source(module, entry->opaque.source_id, &tag));
      if (entry->opaque.data_length > 0 && !entry->opaque.data) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "opaque location has data_length %u but NULL data",
            entry->opaque.data_length);
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "opaque<"));
      IREE_RETURN_IF_ERROR(loom_print_string_literal(stream, tag));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
      IREE_RETURN_IF_ERROR(loom_print_string_literal(
          stream, iree_make_string_view((const char*)entry->opaque.data,
                                        entry->opaque.data_length)));
      return loom_output_stream_write_char(stream, '>');
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown location kind %d", (int)entry->kind);
  }
}

// Prints a location annotation for an op. Emits nothing for
// LOOM_LOCATION_UNKNOWN (0). For all other locations, emits a
// trailing loc(...) annotation.
static iree_status_t loom_print_location(loom_output_stream_t* stream,
                                         const loom_module_t* module,
                                         loom_location_id_t location_id) {
  if (location_id == LOOM_LOCATION_UNKNOWN) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " loc("));
  IREE_RETURN_IF_ERROR(loom_print_location_body(stream, module, location_id));
  return loom_output_stream_write_char(stream, ')');
}

//===----------------------------------------------------------------------===//
// Attribute printing — writes directly to stream
//===----------------------------------------------------------------------===//

// Prints an attribute value. For enum attrs, |descriptor| provides the
// case name table for human-readable output. Pass NULL when no
// descriptor is available (the printer will return an error for enums).
static iree_status_t loom_print_attr(loom_output_stream_t* stream,
                                     const loom_attribute_t* attr,
                                     const loom_module_t* module,
                                     const loom_attr_descriptor_t* descriptor) {
  switch (attr->kind) {
    case LOOM_ATTR_I64:
      return loom_output_stream_write_format(stream, "%" PRId64, attr->i64);
    case LOOM_ATTR_F64: {
      char buffer[32];
      int length = iree_snprintf(buffer, sizeof(buffer), "%.17g", attr->f64);
      bool has_dot = false;
      bool has_exp = false;
      for (int i = 0; i < length; ++i) {
        if (buffer[i] == '.') has_dot = true;
        if (buffer[i] == 'e' || buffer[i] == 'E') has_exp = true;
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write(
          stream, iree_make_string_view(buffer, length)));
      if (!has_dot && !has_exp) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ".0"));
      }
      return iree_ok_status();
    }
    case LOOM_ATTR_STRING: {
      loom_string_id_t id = attr->string_id;
      iree_string_view_t attr_string = iree_string_view_empty();
      if (module && id < module->strings.count) {
        attr_string = module->strings.entries[id];
      }
      return loom_print_string_literal(stream, attr_string);
    }
    case LOOM_ATTR_BOOL:
      return loom_output_stream_write_cstring(stream,
                                              attr->raw ? "true" : "false");
    case LOOM_ATTR_ENUM: {
      uint8_t case_index = (uint8_t)attr->raw;
      if (descriptor && descriptor->enum_case_names &&
          case_index < descriptor->enum_case_count &&
          descriptor->enum_case_names[case_index]) {
        return loom_output_stream_write(
            stream, loom_bstring_view(descriptor->enum_case_names[case_index]));
      }
      // Print the raw value so invalid IR is inspectable without crashing.
      char fallback[16];
      iree_snprintf(fallback, sizeof(fallback), "<%u>", case_index);
      return loom_output_stream_write_cstring(stream, fallback);
    }
    case LOOM_ATTR_SYMBOL: {
      loom_symbol_ref_t ref = attr->symbol;
      if (module && ref.symbol_id < module->symbols.count) {
        loom_string_id_t name_id =
            module->symbols.entries[ref.symbol_id].name_id;
        if (name_id < module->strings.count) {
          iree_string_view_t name = module->strings.entries[name_id];
          IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '@'));
          return loom_output_stream_write(stream, name);
        }
      }
      return loom_output_stream_write_format(stream, "@<symbol:%" PRIu16 ">",
                                             ref.symbol_id);
    }
    case LOOM_ATTR_I64_ARRAY: {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '['));
      for (uint16_t i = 0; i < attr->count; ++i) {
        if (i > 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
        }
        IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
            stream, "%" PRId64, attr->i64_array[i]));
      }
      return loom_output_stream_write_char(stream, ']');
    }
    case LOOM_ATTR_TYPE:
      return loom_output_stream_write_format(stream, "type<%" PRIu32 ">",
                                             attr->type_id);
    case LOOM_ATTR_ENCODING:
      return loom_print_static_encoding(stream, module,
                                        loom_attr_as_encoding_id(*attr));
    case LOOM_ATTR_DICT: {
      if (attr->count > 0 && !attr->dict_entries) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "DICT attr has count %u but NULL entries",
                                attr->count);
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '{'));
      for (uint16_t i = 0; i < attr->count; ++i) {
        if (i > 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
        }
        const loom_named_attr_t* entry = &attr->dict_entries[i];
        if (module && entry->name_id < module->strings.count) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write(
              stream, module->strings.entries[entry->name_id]));
        } else {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
              stream, "<name:%" PRIu16 ">", entry->name_id));
        }
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " = "));
        IREE_RETURN_IF_ERROR(
            loom_print_attr(stream, &entry->value, module, NULL));
      }
      return loom_output_stream_write_char(stream, '}');
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown attribute kind %d", (int)attr->kind);
  }
}

//===----------------------------------------------------------------------===//
// Value name lookup
//===----------------------------------------------------------------------===//

// Stack buffer size for formatting auto-generated value names (%0, %1, ...).
// 32 bytes handles any value ID up to 2^64 with room to spare.
#define LOOM_VALUE_NAME_BUFFER_SIZE 32

// Returns the SSA name for a value, including the '%' prefix. User-assigned
// names are formatted as '%<name>' into |buffer|. Auto-generated names are
// formatted as '%<value_id>'. The returned view always points into |buffer|.
static iree_string_view_t loom_resolve_value_name(
    const loom_module_t* module, loom_value_id_t value_id, char* buffer,
    iree_host_size_t buffer_size) {
  if (value_id < module->values.count) {
    loom_string_id_t name_id = module->values.entries[value_id].name_id;
    if (name_id != LOOM_STRING_ID_INVALID && name_id < module->strings.count) {
      iree_string_view_t bare_name = module->strings.entries[name_id];
      int length = iree_snprintf(buffer, buffer_size, "%%%.*s",
                                 (int)bare_name.size, bare_name.data);
      return iree_make_string_view(buffer, length);
    }
  }
  int length = iree_snprintf(buffer, buffer_size, "%%%" PRIhsz,
                             (iree_host_size_t)value_id);
  return iree_make_string_view(buffer, length);
}

//===----------------------------------------------------------------------===//
// Predicate printing
//===----------------------------------------------------------------------===//

// Predicate kind names indexed by loom_predicate_kind_t.
static const char* const loom_predicate_kind_names[LOOM_PREDICATE_COUNT_] = {
    [LOOM_PREDICATE_EQ] = "eq",     [LOOM_PREDICATE_LT] = "lt",
    [LOOM_PREDICATE_LE] = "le",     [LOOM_PREDICATE_GT] = "gt",
    [LOOM_PREDICATE_GE] = "ge",     [LOOM_PREDICATE_MUL] = "mul",
    [LOOM_PREDICATE_MIN] = "min",   [LOOM_PREDICATE_MAX] = "max",
    [LOOM_PREDICATE_POW2] = "pow2", [LOOM_PREDICATE_RANGE] = "range",
};

// Prints a predicate argument based on its tag.
static iree_status_t loom_print_predicate_arg(loom_print_context_t* ctx,
                                              uint8_t tag, int64_t value) {
  switch (tag) {
    case LOOM_PRED_ARG_VALUE: {
      char buffer[LOOM_VALUE_NAME_BUFFER_SIZE];
      iree_string_view_t name = loom_resolve_value_name(
          ctx->module, (loom_value_id_t)value, buffer, sizeof(buffer));
      return loom_output_stream_write(ctx->stream, name);
    }
    case LOOM_PRED_ARG_CONST:
      return loom_output_stream_write_format(ctx->stream, "%" PRId64, value);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown predicate arg tag %d", (int)tag);
  }
}

// Prints a predicate list in the format: [pred(%name, 16), lt(%K, 1024)]
static iree_status_t loom_print_predicate_list(
    loom_print_context_t* ctx, const loom_predicate_t* predicates,
    uint16_t count) {
  if (count > 0 && !predicates) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "predicate list has count %u but NULL predicates",
                            count);
  }
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "[", false));
  for (uint16_t i = 0; i < count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ",", false));
    }
    const loom_predicate_t* predicate = &predicates[i];
    if (predicate->kind >= LOOM_PREDICATE_COUNT_) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown predicate kind %d",
                              (int)predicate->kind);
    }
    // Emit kind name and opening paren: "mul("
    IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        ctx->stream, loom_predicate_kind_names[predicate->kind]));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '('));
    // Emit arguments separated by ", ".
    for (uint8_t j = 0; j < predicate->arg_count; ++j) {
      if (j > 0) {
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write_cstring(ctx->stream, ", "));
      }
      IREE_RETURN_IF_ERROR(loom_print_predicate_arg(ctx, predicate->arg_tags[j],
                                                    predicate->args[j]));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, ')'));
    loom_print_did_write(ctx);
  }
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "]", false));
  return iree_ok_status();
}

static bool loom_print_optional_attr_present(const loom_op_vtable_t* vtable,
                                             const loom_op_t* op,
                                             uint16_t attr_index) {
  if (attr_index >= op->attribute_count) return false;
  loom_attribute_t attr = loom_op_attrs(op)[attr_index];
  if (loom_attr_is_absent(attr)) return false;
  if (!vtable->attr_descriptors || attr_index >= vtable->attribute_count ||
      !iree_any_bit_set(vtable->attr_descriptors[attr_index].flags,
                        LOOM_ATTR_OPTIONAL)) {
    return true;
  }
  switch ((loom_attr_kind_t)attr.kind) {
    case LOOM_ATTR_DICT:
    case LOOM_ATTR_I64_ARRAY:
    case LOOM_ATTR_PREDICATE_LIST:
      return attr.count > 0;
    default:
      return true;
  }
}

static bool loom_print_attr_is_optional(const loom_op_vtable_t* vtable,
                                        uint16_t attr_index) {
  return vtable->attr_descriptors && attr_index < vtable->attribute_count &&
         iree_any_bit_set(vtable->attr_descriptors[attr_index].flags,
                          LOOM_ATTR_OPTIONAL);
}

//===----------------------------------------------------------------------===//
// Format element walk — writes directly to stream via print context
//===----------------------------------------------------------------------===//

static iree_status_t loom_print_module_body(loom_print_context_t* ctx,
                                            const loom_region_t* region);
static iree_status_t loom_print_region_body(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor);

// Emits an SSA value name (%name or %N) through the spacing model.
static iree_status_t loom_print_value_name(loom_print_context_t* ctx,
                                           loom_value_id_t value_id) {
  char buffer[LOOM_VALUE_NAME_BUFFER_SIZE];
  return loom_print_emit(
      ctx,
      loom_resolve_value_name(ctx->module, value_id, buffer, sizeof(buffer)),
      false);
}

// Emits a value name and fires the field callback with the byte range.
// The spacing model may insert a space before the token, so we snapshot
// the offset AFTER emitting (which includes the space) and subtract
// the token length to get the true token start.
static iree_status_t loom_print_value_name_with_field(
    loom_print_context_t* ctx, loom_value_id_t value_id,
    loom_print_field_ref_t field_ref) {
  char buffer[LOOM_VALUE_NAME_BUFFER_SIZE];
  iree_string_view_t name =
      loom_resolve_value_name(ctx->module, value_id, buffer, sizeof(buffer));
  IREE_RETURN_IF_ERROR(loom_print_emit(ctx, name, false));
  iree_host_size_t end = ctx->stream->offset;
  iree_host_size_t start = end - name.size;
  loom_print_report_field(ctx, field_ref, start, end);
  return iree_ok_status();
}

static iree_status_t loom_print_attr_with_field(
    loom_print_context_t* ctx, const loom_attribute_t* attr,
    const loom_attr_descriptor_t* descriptor,
    loom_print_field_ref_t field_ref) {
  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  iree_host_size_t start = ctx->stream->offset;
  IREE_RETURN_IF_ERROR(
      loom_print_attr(ctx->stream, attr, ctx->module, descriptor));
  iree_host_size_t end = ctx->stream->offset;
  loom_print_did_write(ctx);
  loom_print_report_field(ctx, field_ref, start, end);
  return iree_ok_status();
}

static iree_status_t loom_print_value_type(loom_print_context_t* ctx,
                                           loom_value_id_t value_id) {
  if (value_id < ctx->module->values.count) {
    return loom_text_print_type(ctx->module->values.entries[value_id].type,
                                ctx->module, ctx->stream);
  }
  return loom_output_stream_write_cstring(ctx->stream, "<unknown>");
}

static iree_status_t loom_print_result_value_type(loom_print_context_t* ctx,
                                                  loom_value_id_t value_id) {
  if (value_id < ctx->module->values.count) {
    return loom_text_print_result_type(
        ctx->module->values.entries[value_id].type, ctx->module, ctx->stream);
  }
  return loom_output_stream_write_cstring(ctx->stream, "<unknown>");
}

// Returns the signature argument IDs for a FuncArgs element. Bodyful
// func-like ops use the entry block args; bodyless declarations store
// signature args as op operands.
static const loom_value_id_t* loom_print_func_arg_ids(
    const loom_op_t* op, const loom_op_vtable_t* vtable,
    uint16_t* out_arg_count) {
  *out_arg_count = 0;
  if (vtable->func_like) {
    if (vtable->func_like->args_as_operands) {
      *out_arg_count = op->operand_count;
      return loom_op_const_operands(op);
    }
    uint8_t body_index = vtable->func_like->body_region_index;
    if (body_index == LOOM_REGION_INDEX_NONE ||
        body_index >= op->region_count) {
      return NULL;
    }
    loom_region_t* body = loom_op_regions(op)[body_index];
    if (!body || body->block_count == 0) return NULL;
    const loom_block_t* block = loom_region_const_entry_block(body);
    *out_arg_count = block->arg_count;
    return block->arg_ids;
  }

  loom_region_t** regions = loom_op_regions(op);
  if (op->region_count > 0 && regions[0] && regions[0]->block_count > 0) {
    const loom_block_t* block = loom_region_const_entry_block(regions[0]);
    *out_arg_count = block->arg_count;
    return block->arg_ids;
  }
  if (op->operand_count > 0) {
    *out_arg_count = op->operand_count;
    return loom_op_const_operands(op);
  }
  return NULL;
}

// Returns the operand domain used for tied-result printing. Regular body ops
// tie to op operands; symbol-defining func-like ops tie to signature args.
static const loom_value_id_t* loom_print_tied_operand_ids(
    const loom_op_t* op, const loom_op_vtable_t* vtable,
    uint16_t* out_operand_count) {
  if (vtable->func_like &&
      iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE)) {
    return loom_print_func_arg_ids(op, vtable, out_operand_count);
  }
  *out_operand_count = op->operand_count;
  return loom_op_const_operands(op);
}

// Returns true if |value_id| has a user-assigned SSA name in the module's
// string table (as opposed to an autogenerated numeric fallback).
static bool loom_print_value_has_name(const loom_module_t* module,
                                      loom_value_id_t value_id) {
  if (value_id >= module->values.count) return false;
  loom_string_id_t name_id = module->values.entries[value_id].name_id;
  return name_id != LOOM_STRING_ID_INVALID && name_id < module->strings.count;
}

static iree_status_t loom_printer_walk_format(loom_print_context_t* ctx,
                                              const loom_op_t* op,
                                              const loom_op_vtable_t* vtable) {
  const loom_format_element_t* elements = vtable->format_elements;
  uint16_t element_count = vtable->format_element_count;

  for (uint16_t i = 0; i < element_count; ++i) {
    const loom_format_element_t* element = &elements[i];
    switch (element->kind) {
      case LOOM_FORMAT_KIND_OPERAND_REF: {
        loom_value_id_t value_id = 0;
        if (element->field_index == 0xFF) {
          loom_region_t** regions = loom_op_regions(op);
          const loom_block_t* entry_block =
              (op->region_count > 0 && regions[0] &&
               regions[0]->block_count > 0)
                  ? loom_region_const_entry_block(regions[0])
                  : NULL;
          value_id = (op->region_count > 0 && regions[0] &&
                      regions[0]->block_count > 0 && entry_block &&
                      entry_block->arg_count > 0)
                         ? loom_block_arg_id(entry_block, 0)
                         : LOOM_VALUE_ID_INVALID;
        } else if (element->field_index >= op->operand_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format OPERAND_REF field_index %u out of range (op has %u "
              "operands)",
              element->field_index, op->operand_count);
        } else {
          value_id = loom_op_const_operands(op)[element->field_index];
        }
        IREE_RETURN_IF_ERROR(loom_print_value_name_with_field(
            ctx, value_id,
            loom_print_field_ref(LOOM_PRINT_FIELD_OPERAND,
                                 element->field_index)));
        break;
      }
      case LOOM_FORMAT_KIND_OPERAND_REFS: {
        const loom_value_id_t* operands = loom_op_const_operands(op);
        uint16_t start = vtable->fixed_operand_count;
        for (uint16_t j = start; j < op->operand_count; ++j) {
          if (j > start) {
            IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ",", false));
          }
          IREE_RETURN_IF_ERROR(loom_print_value_name_with_field(
              ctx, operands[j],
              loom_print_field_ref(LOOM_PRINT_FIELD_OPERAND, j)));
        }
        break;
      }
      case LOOM_FORMAT_KIND_ATTR_VALUE: {
        if (element->field_index >= op->attribute_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format ATTR_VALUE field_index %u out of range (op has %u "
              "attributes)",
              element->field_index, op->attribute_count);
        }
        const loom_attr_descriptor_t* descriptor =
            (vtable->attr_descriptors &&
             element->field_index < vtable->attribute_count)
                ? &vtable->attr_descriptors[element->field_index]
                : NULL;
        IREE_RETURN_IF_ERROR(loom_print_attr_with_field(
            ctx, &loom_op_attrs(op)[element->field_index], descriptor,
            loom_print_field_ref(LOOM_PRINT_FIELD_ATTR, element->field_index)));
        break;
      }
      case LOOM_FORMAT_KIND_SYMBOL_REF: {
        if (element->field_index >= op->attribute_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format SYMBOL_REF field_index %u out of range (op has %u "
              "attributes)",
              element->field_index, op->attribute_count);
        }
        IREE_RETURN_IF_ERROR(loom_print_attr_with_field(
            ctx, &loom_op_attrs(op)[element->field_index], NULL,
            loom_print_field_ref(LOOM_PRINT_FIELD_ATTR, element->field_index)));
        break;
      }
      case LOOM_FORMAT_KIND_OPERAND_TYPE: {
        if (element->field_index >= op->operand_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format OPERAND_TYPE field_index %u out of range (op has %u "
              "operands)",
              element->field_index, op->operand_count);
        }
        IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
        IREE_RETURN_IF_ERROR(loom_print_value_type(
            ctx, loom_op_const_operands(op)[element->field_index]));
        loom_print_did_write(ctx);
        break;
      }
      case LOOM_FORMAT_KIND_RESULT_TYPE: {
        if (element->field_index >= op->result_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format RESULT_TYPE field_index %u out of range (op has %u "
              "results)",
              element->field_index, op->result_count);
        }
        IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
        IREE_RETURN_IF_ERROR(loom_print_result_value_type(
            ctx, loom_op_const_results(op)[element->field_index]));
        loom_print_did_write(ctx);
        break;
      }
      case LOOM_FORMAT_KIND_OPERAND_TYPES: {
        const loom_value_id_t* operands = loom_op_const_operands(op);
        uint16_t start = vtable->fixed_operand_count;
        for (uint16_t j = start; j < op->operand_count; ++j) {
          if (j > start) {
            IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ",", false));
          }
          IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
          IREE_RETURN_IF_ERROR(loom_print_value_type(ctx, operands[j]));
          loom_print_did_write(ctx);
        }
        break;
      }
      case LOOM_FORMAT_KIND_RESULT_TYPE_SINGLE: {
        if (element->field_index >= op->result_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format RESULT_TYPE_SINGLE field_index %u out of range (op has "
              "%u results)",
              element->field_index, op->result_count);
        }
        IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
        IREE_RETURN_IF_ERROR(loom_print_result_value_type(
            ctx, loom_op_const_results(op)[element->field_index]));
        loom_print_did_write(ctx);
        break;
      }
      case LOOM_FORMAT_KIND_RESULT_TYPE_LIST: {
        bool use_parens = (element->data & LOOM_RESULT_TYPE_LIST_PARENS) != 0;
        if (use_parens) {
          IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "(", false));
        }
        const bool is_symbol_definition =
            iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE);
        uint16_t tied_operand_count = 0;
        const loom_value_id_t* tied_operand_ids =
            loom_print_tied_operand_ids(op, vtable, &tied_operand_count);
        for (uint16_t j = 0; j < op->result_count; ++j) {
          if (j > 0) {
            IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ",", false));
          }
          // Check for tied result.
          const loom_tied_result_t* tied = loom_op_tied_results(op);
          bool is_tied = false;
          for (uint8_t t = 0; t < op->tied_result_count; ++t) {
            if (tied[t].result_index == j) {
              if (tied[t].operand_index >= tied_operand_count ||
                  !tied_operand_ids) {
                return iree_make_status(
                    IREE_STATUS_INVALID_ARGUMENT,
                    "tied result %u references operand %u but op has %u "
                    "operands",
                    tied[t].result_index, tied[t].operand_index,
                    tied_operand_count);
              }
              loom_value_id_t tied_operand_id =
                  tied_operand_ids[tied[t].operand_index];
              IREE_RETURN_IF_ERROR(loom_print_value_name(ctx, tied_operand_id));
              IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "as", false));
              is_tied = true;
              break;
            }
          }
          if (!is_tied && is_symbol_definition &&
              loom_print_value_has_name(ctx->module,
                                        loom_op_const_results(op)[j])) {
            IREE_RETURN_IF_ERROR(
                loom_print_value_name(ctx, loom_op_const_results(op)[j]));
            IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ":", true));
          }
          IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
          IREE_RETURN_IF_ERROR(
              loom_print_result_value_type(ctx, loom_op_const_results(op)[j]));
          loom_print_did_write(ctx);
        }
        if (use_parens) {
          IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ")", false));
        }
        break;
      }
      case LOOM_FORMAT_KIND_KEYWORD: {
        if (element->data >= LOOM_KW_COUNT_) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format KEYWORD data %u out of range (max %u)", element->data,
              (uint16_t)LOOM_KW_COUNT_);
        }
        IREE_RETURN_IF_ERROR(loom_print_emit(
            ctx, loom_bstring_view(loom_keyword_bstrings[element->data]),
            false));
        break;
      }
      case LOOM_FORMAT_KIND_REGION: {
        if (element->field_index >= op->region_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format REGION field_index %u out of range (op has %u regions)",
              element->field_index, op->region_count);
        }
        if (!vtable->region_descriptors ||
            element->field_index >= vtable->region_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format REGION field_index %u out of range (vtable has %u "
              "region descriptors)",
              element->field_index, vtable->region_count);
        }
        loom_region_t* region = loom_op_regions(op)[element->field_index];
        const loom_region_descriptor_t* region_descriptor =
            &vtable->region_descriptors[element->field_index];
        IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
        iree_host_size_t region_start = ctx->stream->offset;
        if (iree_any_bit_set(ctx->flags, LOOM_TEXT_PRINT_SKIP_REGIONS)) {
          // Declaration mode: print empty braces placeholder.
          IREE_RETURN_IF_ERROR(
              loom_output_stream_write_cstring(ctx->stream, "{ ... }"));
        } else {
          // Print region inline: " {\n" body "}" (indented).
          IREE_RETURN_IF_ERROR(
              loom_output_stream_write_cstring(ctx->stream, "{\n"));
          if (region) {
            ++ctx->indent;
            IREE_RETURN_IF_ERROR(
                loom_print_region_body(ctx, region, region_descriptor));
            --ctx->indent;
          }
          IREE_RETURN_IF_ERROR(loom_print_indent(ctx));
          IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '}'));
        }
        ctx->has_previous_token = true;
        ctx->last_char = '}';
        ctx->glue_next = false;
        loom_print_report_field(
            ctx,
            loom_print_field_ref(LOOM_PRINT_FIELD_REGION, element->field_index),
            region_start, ctx->stream->offset);
        break;
      }
      case LOOM_FORMAT_KIND_INDEX_LIST: {
        if (element->data >= op->attribute_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format INDEX_LIST attr index %u out of range (op has %u "
              "attributes)",
              element->data, op->attribute_count);
        }
        loom_attribute_t static_attr = loom_op_attrs(op)[element->data];
        const loom_value_id_t* operands = loom_op_const_operands(op);
        uint16_t dynamic_start = element->field_index;
        uint16_t dynamic_index = 0;
        IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "[", i > 0));
        for (uint16_t j = 0; j < static_attr.count; ++j) {
          if (j > 0) {
            IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ",", false));
          }
          int64_t static_value = static_attr.i64_array[j];
          if (static_value == INT64_MIN) {
            uint16_t operand_index = dynamic_start + dynamic_index++;
            if (operand_index >= op->operand_count) {
              return iree_make_status(
                  IREE_STATUS_INVALID_ARGUMENT,
                  "format INDEX_LIST dynamic operand %u out of range (op has "
                  "%u operands)",
                  operand_index, op->operand_count);
            }
            IREE_RETURN_IF_ERROR(loom_print_value_name_with_field(
                ctx, operands[operand_index],
                loom_print_field_ref(LOOM_PRINT_FIELD_OPERAND, operand_index)));
          } else {
            IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
            IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
                ctx->stream, "%" PRId64, static_value));
            loom_print_did_write(ctx);
          }
        }
        IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "]", false));
        break;
      }
      case LOOM_FORMAT_KIND_BINDING_LIST: {
        // (%block_arg = %operand : type, ...)
        // field_index = start of variadic operands in op's operand array.
        // data = binding kind (CAPTURE or ELEMENT).
        // The block args come from region 0's entry block. For CAPTURE
        // bindings, the first N block args are implicit (e.g., the IV in
        // a for-loop) and the binding covers the remaining args.
        const loom_value_id_t* operands = loom_op_const_operands(op);
        uint16_t start = element->field_index;
        uint16_t binding_count = 0;
        if (op->operand_count > start) {
          binding_count = op->operand_count - start;
        }
        loom_region_t** regions = loom_op_regions(op);
        const loom_block_t* block = NULL;
        if (op->region_count > 0 && regions[0] && regions[0]->block_count > 0) {
          block = loom_region_const_entry_block(regions[0]);
        }
        // Block arg offset: implicit args precede the bindings.
        uint16_t block_arg_offset = 0;
        if (block && block->arg_count > binding_count) {
          block_arg_offset = block->arg_count - binding_count;
        }
        IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "(", true));
        for (uint16_t j = 0; j < binding_count; ++j) {
          if (j > 0) {
            IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ",", false));
          }
          // Block arg name.
          if (block && (block_arg_offset + j) < block->arg_count) {
            IREE_RETURN_IF_ERROR(loom_print_value_name(
                ctx,
                loom_block_arg_id(block, (uint16_t)(block_arg_offset + j))));
          }
          IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "=", false));
          // Operand name.
          IREE_RETURN_IF_ERROR(loom_print_value_name_with_field(
              ctx, operands[start + j],
              loom_print_field_ref(LOOM_PRINT_FIELD_OPERAND,
                                   (uint16_t)(start + j))));
          IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ":", false));
          // Operand type.
          IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
          IREE_RETURN_IF_ERROR(loom_print_value_type(ctx, operands[start + j]));
          loom_print_did_write(ctx);
        }
        IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ")", false));
        break;
      }
      case LOOM_FORMAT_KIND_FUNC_ARGS: {
        // (%name: type, ...)
        uint16_t arg_count = 0;
        const loom_value_id_t* arg_ids =
            loom_print_func_arg_ids(op, vtable, &arg_count);
        IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "(", true));
        for (uint16_t j = 0; j < arg_count; ++j) {
          if (j > 0) {
            IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ",", false));
          }
          bool args_as_operands =
              vtable->func_like && vtable->func_like->args_as_operands;
          if (args_as_operands) {
            IREE_RETURN_IF_ERROR(loom_print_value_name_with_field(
                ctx, arg_ids[j],
                loom_print_field_ref(LOOM_PRINT_FIELD_OPERAND, j)));
          } else {
            IREE_RETURN_IF_ERROR(loom_print_value_name(ctx, arg_ids[j]));
          }
          IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ":", true));
          IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
          IREE_RETURN_IF_ERROR(loom_print_value_type(ctx, arg_ids[j]));
          loom_print_did_write(ctx);
        }
        IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ")", false));
        break;
      }
      case LOOM_FORMAT_KIND_ATTR_DICT: {
        // AttrDict reads a LOOM_ATTR_DICT attribute at field_index and
        // emits its entries as {key = value, key = value, ...}.
        bool optional =
            loom_print_attr_is_optional(vtable, element->field_index);
        if (element->field_index >= op->attribute_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format ATTR_DICT field_index %u out of range (op has %u "
              "attributes)",
              element->field_index, op->attribute_count);
        }
        loom_attribute_t dict_attr = loom_op_attrs(op)[element->field_index];
        if (loom_attr_is_absent(dict_attr)) {
          if (optional) break;
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "format ATTR_DICT field_index %u is absent",
                                  element->field_index);
        }
        if (dict_attr.kind != LOOM_ATTR_DICT) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format ATTR_DICT field_index %u expected DICT attr but found %d",
              element->field_index, (int)dict_attr.kind);
        }
        if (optional && dict_attr.count == 0) break;
        IREE_RETURN_IF_ERROR(loom_print_attr_with_field(
            ctx, &dict_attr, NULL,
            loom_print_field_ref(LOOM_PRINT_FIELD_ATTR, element->field_index)));
        break;
      }
      case LOOM_FORMAT_KIND_PREDICATE_LIST: {
        if (element->field_index >= op->attribute_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format PREDICATE_LIST field_index %u out of range (op has %u "
              "attributes)",
              element->field_index, op->attribute_count);
        }
        loom_attribute_t attr = loom_op_attrs(op)[element->field_index];
        if (attr.kind == LOOM_ATTR_PREDICATE_LIST) {
          if (attr.count > 0 && !attr.predicate_list) {
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "PREDICATE_LIST attr has count %u but NULL predicates",
                attr.count);
          }
          iree_host_size_t predicate_start =
              loom_print_next_token_start_offset(ctx, false, '[');
          IREE_RETURN_IF_ERROR(
              loom_print_predicate_list(ctx, attr.predicate_list, attr.count));
          loom_print_report_field(
              ctx,
              loom_print_field_ref(LOOM_PRINT_FIELD_ATTR, element->field_index),
              predicate_start, ctx->stream->offset);
        }
        break;
      }
      case LOOM_FORMAT_KIND_FLAGS: {
        // Per-instance flags in angle brackets, glued to the op name:
        // scalar.addi<nsw|nuw>. Walks set bits in instance_flags and
        // emits keywords from the vtable's instance_flags_case_names.
        uint8_t flags = op->instance_flags;
        if (flags != 0 && vtable->instance_flags_case_names != NULL) {
          IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "<", true));
          bool first = true;
          for (uint8_t bit = 0; bit < vtable->instance_flags_case_count;
               ++bit) {
            if (flags & (1u << bit)) {
              if (!first) {
                IREE_RETURN_IF_ERROR(
                    loom_output_stream_write_char(ctx->stream, '|'));
              }
              IREE_RETURN_IF_ERROR(loom_output_stream_write(
                  ctx->stream,
                  loom_bstring_view(vtable->instance_flags_case_names[bit])));
              first = false;
            }
          }
          IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ">", true));
          loom_print_did_write(ctx);
        }
        break;
      }
      case LOOM_FORMAT_KIND_OP_REF: {
        // Op kind reference in angle brackets, glued to the op name:
        // func.template<tile.contract>. The field_index references a
        // string attribute holding the target op name.
        if (element->field_index >= op->attribute_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format OP_REF field_index %u out of range (op has %u "
              "attributes)",
              element->field_index, op->attribute_count);
        }
        loom_attribute_t attr = loom_op_attrs(op)[element->field_index];
        if (attr.kind == LOOM_ATTR_STRING &&
            attr.string_id != LOOM_STRING_ID_INVALID &&
            attr.string_id < ctx->module->strings.count) {
          iree_string_view_t op_name =
              ctx->module->strings.entries[attr.string_id];
          iree_host_size_t op_ref_start = ctx->stream->offset;
          IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "<", true));
          IREE_RETURN_IF_ERROR(loom_print_emit(ctx, op_name, true));
          IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ">", true));
          loom_print_did_write(ctx);
          loom_print_report_field(
              ctx,
              loom_print_field_ref(LOOM_PRINT_FIELD_ATTR, element->field_index),
              op_ref_start, ctx->stream->offset);
        }
        break;
      }
      case LOOM_FORMAT_KIND_TEMPLATE_PARAM: {
        // Required compile-time op parameter in angle brackets, glued to
        // the op name: vector.reduce<addf>.
        if (element->field_index >= op->attribute_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format TEMPLATE_PARAM field_index %u out of range (op has %u "
              "attributes)",
              element->field_index, op->attribute_count);
        }
        const loom_attr_descriptor_t* descriptor =
            (vtable->attr_descriptors &&
             element->field_index < vtable->attribute_count)
                ? &vtable->attr_descriptors[element->field_index]
                : NULL;
        loom_attribute_t attr = loom_op_attrs(op)[element->field_index];
        iree_host_size_t param_start = ctx->stream->offset;
        IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "<", true));
        IREE_RETURN_IF_ERROR(
            loom_print_attr(ctx->stream, &attr, ctx->module, descriptor));
        IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ">", true));
        loom_print_did_write(ctx);
        loom_print_report_field(
            ctx,
            loom_print_field_ref(LOOM_PRINT_FIELD_ATTR, element->field_index),
            param_start, ctx->stream->offset);
        break;
      }
      case LOOM_FORMAT_KIND_GLUE:
        loom_print_set_glue(ctx);
        break;
      case LOOM_FORMAT_KIND_OPTIONAL_GROUP: {
        uint16_t skip_count = element->data >> 2;
        uint8_t anchor_category = element->data & 3;
        bool present = false;
        switch (anchor_category) {
          case LOOM_ANCHOR_OPERAND:
            present = op->operand_count > vtable->fixed_operand_count;
            break;
          case LOOM_ANCHOR_ATTR:
            present = loom_print_optional_attr_present(vtable, op,
                                                       element->field_index);
            break;
          case LOOM_ANCHOR_REGION: {
            if (element->field_index < op->region_count) {
              loom_region_t** regions = loom_op_regions(op);
              present = regions[element->field_index] != NULL &&
                        regions[element->field_index]->block_count > 0;
            }
            break;
          }
          case LOOM_ANCHOR_RESULTS:
            present = op->result_count > 0;
            break;
        }
        if (!present) i += skip_count;
        break;
      }
      case LOOM_FORMAT_KIND_SCOPE:
        // Scope is transparent for printing — children follow inline.
        // No scope state needed; the printer reads names from the value
        // table directly.
        break;
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Op and region printing
//===----------------------------------------------------------------------===//

static iree_status_t loom_print_op(loom_print_context_t* ctx,
                                   const loom_op_t* op) {
  const loom_op_vtable_t* vtable = NULL;
  if (ctx->module->context) {
    vtable = loom_context_resolve_op(ctx->module->context, op->kind);
  }
  if (!vtable) {
    IREE_RETURN_IF_ERROR(loom_print_indent(ctx));
    return loom_output_stream_write_cstring(ctx->stream, "<unknown op>\n");
  }

  // Reset token state for this op.
  ctx->has_previous_token = false;
  ctx->glue_next = false;
  ctx->last_char = 0;

  iree_status_t status = iree_ok_status();

  // Print results on the LHS. Symbol-defining ops (functions, globals)
  // carry result values for type information but don't produce SSA values.
  bool print_results =
      op->result_count > 0 &&
      !iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE);
  if (iree_status_is_ok(status) && print_results) {
    const loom_value_id_t* results = loom_op_const_results(op);
    for (uint16_t j = 0; j < op->result_count && iree_status_is_ok(status);
         ++j) {
      if (j > 0) status = loom_print_emit_cstr(ctx, ",", false);
      if (iree_status_is_ok(status)) {
        status = loom_print_value_name_with_field(
            ctx, results[j], loom_print_field_ref(LOOM_PRINT_FIELD_RESULT, j));
      }
    }
    if (iree_status_is_ok(status)) {
      status = loom_print_emit_cstr(ctx, "=", false);
    }
  }

  // Print op name.
  if (iree_status_is_ok(status)) {
    status = loom_print_emit(ctx, loom_op_vtable_name(vtable), false);
  }

  // Walk format elements. Regions are printed inline when their
  // REGION format element is encountered, properly interleaving
  // tokens with region bodies.
  if (iree_status_is_ok(status)) {
    status = loom_printer_walk_format(ctx, op, vtable);
  }

  // Location annotation (omitted for LOOM_LOCATION_UNKNOWN).
  if (iree_status_is_ok(status) && (ctx->flags & LOOM_TEXT_PRINT_LOCATIONS)) {
    status = loom_print_location(ctx->stream, ctx->module, op->location);
  }

  // Terminate the op line.
  if (iree_status_is_ok(status)) {
    status = loom_output_stream_write_char(ctx->stream, '\n');
  }

  return status;
}

static uint16_t loom_print_block_last_live_op_index(const loom_block_t* block) {
  for (uint16_t reverse_index = block->op_count; reverse_index > 0;
       --reverse_index) {
    uint16_t op_index = (uint16_t)(reverse_index - 1);
    const loom_op_t* op = loom_block_const_op(block, op_index);
    if ((op->flags & LOOM_OP_FLAG_DEAD) == 0) return op_index;
  }
  return UINT16_MAX;
}

static bool loom_print_should_elide_implicit_terminator(
    const loom_region_descriptor_t* region_descriptor, const loom_op_t* op) {
  IREE_ASSERT_ARGUMENT(region_descriptor);
  if (region_descriptor->implicit_terminator == LOOM_OP_KIND_UNKNOWN) {
    return false;
  }
  return op->kind == region_descriptor->implicit_terminator &&
         op->operand_count == 0 && op->result_count == 0 &&
         op->region_count == 0 && op->tied_result_count == 0 &&
         op->attribute_count == 0 && op->instance_flags == 0;
}

static iree_status_t loom_print_region_body(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor) {
  IREE_ASSERT_ARGUMENT(region_descriptor);
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(region, block_index);

    // Print block label if present: ^label(%arg: type, ...):
    if (block->label_id != LOOM_STRING_ID_INVALID &&
        block->label_id < ctx->module->strings.count) {
      IREE_RETURN_IF_ERROR(loom_print_indent(ctx));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '^'));
      IREE_RETURN_IF_ERROR(loom_output_stream_write(
          ctx->stream, ctx->module->strings.entries[block->label_id]));
      if (block->arg_count > 0) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '('));
        for (uint16_t arg_index = 0; arg_index < block->arg_count;
             ++arg_index) {
          if (arg_index > 0) {
            IREE_RETURN_IF_ERROR(
                loom_output_stream_write_cstring(ctx->stream, ", "));
          }
          char name_buffer[LOOM_VALUE_NAME_BUFFER_SIZE];
          loom_value_id_t arg_id = loom_block_arg_id(block, arg_index);
          IREE_RETURN_IF_ERROR(loom_output_stream_write(
              ctx->stream,
              loom_resolve_value_name(ctx->module, arg_id, name_buffer,
                                      sizeof(name_buffer))));
          IREE_RETURN_IF_ERROR(
              loom_output_stream_write_cstring(ctx->stream, " : "));
          IREE_RETURN_IF_ERROR(
              loom_text_print_type(loom_module_value_type(ctx->module, arg_id),
                                   ctx->module, ctx->stream));
        }
        IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, ')'));
      }
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(ctx->stream, ":\n"));
    }

    bool printed_any = false;
    uint16_t last_live_op_index = loom_print_block_last_live_op_index(block);
    for (uint16_t op_index = 0; op_index < block->op_count; ++op_index) {
      const loom_op_t* current_op = loom_block_const_op(block, op_index);
      if (current_op->flags & LOOM_OP_FLAG_DEAD) continue;
      if (op_index == last_live_op_index &&
          loom_print_should_elide_implicit_terminator(region_descriptor,
                                                      current_op)) {
        continue;
      }
      // Blank line between top-level symbol definitions (func.def,
      // func.decl, etc.) in the module body.
      if (printed_any && ctx->indent == 0) {
        const loom_op_vtable_t* current_vtable =
            loom_op_vtable(ctx->module, current_op);
        if (current_vtable &&
            (current_vtable->traits & LOOM_TRAIT_SYMBOL_DEFINE)) {
          IREE_RETURN_IF_ERROR(
              loom_output_stream_write_char(ctx->stream, '\n'));
        }
      }
      printed_any = true;
      IREE_RETURN_IF_ERROR(loom_print_indent(ctx));
      IREE_RETURN_IF_ERROR(loom_print_op(ctx, current_op));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_print_module_body(loom_print_context_t* ctx,
                                            const loom_region_t* region) {
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(region, block_index);

    if (block->label_id != LOOM_STRING_ID_INVALID &&
        block->label_id < ctx->module->strings.count) {
      IREE_RETURN_IF_ERROR(loom_print_indent(ctx));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '^'));
      IREE_RETURN_IF_ERROR(loom_output_stream_write(
          ctx->stream, ctx->module->strings.entries[block->label_id]));
      if (block->arg_count > 0) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '('));
        for (uint16_t arg_index = 0; arg_index < block->arg_count;
             ++arg_index) {
          if (arg_index > 0) {
            IREE_RETURN_IF_ERROR(
                loom_output_stream_write_cstring(ctx->stream, ", "));
          }
          char name_buffer[LOOM_VALUE_NAME_BUFFER_SIZE];
          loom_value_id_t arg_id = loom_block_arg_id(block, arg_index);
          IREE_RETURN_IF_ERROR(loom_output_stream_write(
              ctx->stream,
              loom_resolve_value_name(ctx->module, arg_id, name_buffer,
                                      sizeof(name_buffer))));
          IREE_RETURN_IF_ERROR(
              loom_output_stream_write_cstring(ctx->stream, " : "));
          IREE_RETURN_IF_ERROR(
              loom_text_print_type(loom_module_value_type(ctx->module, arg_id),
                                   ctx->module, ctx->stream));
        }
        IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, ')'));
      }
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(ctx->stream, ":\n"));
    }

    bool printed_any = false;
    for (uint16_t op_index = 0; op_index < block->op_count; ++op_index) {
      const loom_op_t* current_op = loom_block_const_op(block, op_index);
      if (current_op->flags & LOOM_OP_FLAG_DEAD) continue;
      if (printed_any) {
        const loom_op_vtable_t* current_vtable =
            loom_op_vtable(ctx->module, current_op);
        if (current_vtable &&
            (current_vtable->traits & LOOM_TRAIT_SYMBOL_DEFINE)) {
          IREE_RETURN_IF_ERROR(
              loom_output_stream_write_char(ctx->stream, '\n'));
        }
      }
      printed_any = true;
      IREE_RETURN_IF_ERROR(loom_print_indent(ctx));
      IREE_RETURN_IF_ERROR(loom_print_op(ctx, current_op));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_print_encoding_aliases(loom_print_context_t* ctx,
                                                 const loom_module_t* module) {
  for (uint16_t i = 0; i < module->encodings.count; ++i) {
    const loom_encoding_t* encoding = &module->encodings.entries[i];
    if (encoding->alias_id == LOOM_STRING_ID_INVALID ||
        encoding->alias_id >= module->strings.count) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '#'));
    IREE_RETURN_IF_ERROR(loom_output_stream_write(
        ctx->stream, module->strings.entries[encoding->alias_id]));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, " = "));
    IREE_RETURN_IF_ERROR(
        loom_print_canonical_encoding(ctx->stream, module, encoding));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '\n'));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t loom_text_print_module(const loom_module_t* module,
                                     loom_output_stream_t* stream,
                                     loom_text_print_flags_t flags) {
  if (!module || !module->body) return iree_ok_status();
  loom_print_context_t ctx = {
      .stream = stream,
      .module = module,
      .flags = flags,
  };
  IREE_RETURN_IF_ERROR(loom_print_encoding_aliases(&ctx, module));
  return loom_print_module_body(&ctx, module->body);
}

iree_status_t loom_text_print_operation(const loom_module_t* module,
                                        const loom_op_t* op,
                                        loom_output_stream_t* stream,
                                        loom_text_print_flags_t flags) {
  if (!module || !op) return iree_ok_status();
  loom_print_context_t ctx = {
      .stream = stream,
      .module = module,
      .flags = flags,
  };
  return loom_print_op(&ctx, op);
}

iree_status_t loom_text_print_module_to_builder(const loom_module_t* module,
                                                iree_string_builder_t* builder,
                                                loom_text_print_flags_t flags) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  return loom_text_print_module(module, &stream, flags);
}

iree_status_t loom_text_print_operation_to_builder(
    const loom_module_t* module, const loom_op_t* op,
    iree_string_builder_t* builder, loom_text_print_flags_t flags) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  return loom_text_print_operation(module, op, &stream, flags);
}

iree_status_t loom_text_print_operation_with_field_callback(
    const loom_module_t* module, const loom_op_t* op,
    iree_string_builder_t* builder, loom_text_print_flags_t flags,
    loom_print_field_callback_t callback) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  loom_print_context_t ctx = {
      .stream = &stream,
      .module = module,
      .flags = flags,
      .field_callback = callback,
  };
  return loom_print_op(&ctx, op);
}
