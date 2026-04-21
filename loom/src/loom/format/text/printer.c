// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/printer.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "iree/base/internal/unicode.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

//===----------------------------------------------------------------------===//
// Print context — spacing + indentation on top of output stream
//===----------------------------------------------------------------------===//

typedef struct loom_print_context_t {
  // Output stream receiving textual IR bytes.
  loom_output_stream_t* stream;
  // Module whose string, type, value, and location tables are printed.
  const loom_module_t* module;
  // Flag bitset controlling layout and optional annotations.
  loom_text_print_flags_t flags;
  // Optional descriptor-backed environment for low asm region syntax.
  loom_text_low_asm_environment_t low_asm_environment;
  // Descriptor-set key selected for low asm region syntax.
  iree_string_view_t low_asm_descriptor_set_key;
  // True when the current logical line already contains a printed token.
  bool has_previous_token;
  // True when the next token should be glued to the previous token.
  bool glue_next;
  // Last byte emitted through the token spacing model.
  char last_char;
  // Current indentation depth in logical two-space levels.
  uint16_t indent;
  // Optional callback receiving semantic field byte ranges.
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
static iree_status_t loom_print_indent_levels(loom_print_context_t* ctx,
                                              uint16_t indent) {
  if (!iree_any_bit_set(ctx->flags, LOOM_TEXT_PRINT_INDENT)) {
    return iree_ok_status();
  }
  iree_host_size_t count = (iree_host_size_t)indent * 2;
  while (count > 0) {
    iree_host_size_t chunk =
        count < sizeof(SPACES) - 1 ? count : sizeof(SPACES) - 1;
    IREE_RETURN_IF_ERROR(loom_output_stream_write(
        ctx->stream, iree_make_string_view(SPACES, chunk)));
    count -= chunk;
  }
  return iree_ok_status();
}

static iree_status_t loom_print_indent(loom_print_context_t* ctx) {
  return loom_print_indent_levels(ctx, ctx->indent);
}

static iree_status_t loom_print_leading_comments(
    loom_print_context_t* ctx, uint16_t indent,
    const iree_string_view_t* comments, iree_host_size_t comment_count) {
  for (iree_host_size_t i = 0; i < comment_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_print_indent_levels(ctx, indent));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, "//"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write(ctx->stream, comments[i]));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '\n'));
  }
  return iree_ok_status();
}

static iree_status_t loom_print_op_comments(loom_print_context_t* ctx,
                                            const loom_op_t* op) {
  iree_host_size_t comment_count = 0;
  const iree_string_view_t* comments =
      loom_module_op_comments(ctx->module, op, &comment_count);
  return loom_print_leading_comments(ctx, ctx->indent, comments, comment_count);
}

static iree_status_t loom_print_block_comments(loom_print_context_t* ctx,
                                               const loom_block_t* block,
                                               uint16_t indent) {
  iree_host_size_t comment_count = 0;
  const iree_string_view_t* comments =
      loom_module_block_comments(ctx->module, block, &comment_count);
  return loom_print_leading_comments(ctx, indent, comments, comment_count);
}

// Forward declaration — defined after the type printing section.
static iree_status_t loom_print_attr(loom_output_stream_t* stream,
                                     const loom_attribute_t* attr,
                                     const loom_module_t* module,
                                     const loom_attr_descriptor_t* descriptor);
static iree_status_t loom_print_value_type(loom_print_context_t* ctx,
                                           loom_value_id_t value_id);

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

static iree_status_t loom_print_encoding_type(loom_output_stream_t* stream,
                                              loom_type_t type) {
  loom_encoding_role_t role = loom_type_encoding_role(type);
  if (role == LOOM_ENCODING_ROLE_UNKNOWN) {
    return loom_output_stream_write_cstring(stream, "encoding");
  }
  const char* role_name = loom_encoding_role_name(role);
  if (!role_name) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown encoding role %d", (int)role);
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "encoding<"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, role_name));
  return loom_output_stream_write_char(stream, '>');
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

static bool loom_print_block_has_label(const loom_print_context_t* ctx,
                                       const loom_block_t* block) {
  return block->label_id != LOOM_STRING_ID_INVALID &&
         block->label_id < ctx->module->strings.count;
}

static bool loom_print_region_label_exists(const loom_print_context_t* ctx,
                                           const loom_region_t* region,
                                           iree_string_view_t label) {
  if (!region) return false;
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(region, block_index);
    if (!loom_print_block_has_label(ctx, block)) continue;
    if (iree_string_view_equal(ctx->module->strings.entries[block->label_id],
                               label)) {
      return true;
    }
  }
  return false;
}

static bool loom_print_region_find_block_index(const loom_region_t* region,
                                               const loom_block_t* block,
                                               uint16_t* out_block_index) {
  if (!region || !block) return false;
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    if (loom_region_const_block(region, block_index) == block) {
      *out_block_index = block_index;
      return true;
    }
  }
  return false;
}

static iree_status_t loom_print_block_label_view(
    const loom_print_context_t* ctx, const loom_region_t* region,
    const loom_block_t* block, char* synthetic_buffer,
    iree_host_size_t synthetic_buffer_capacity, iree_string_view_t* out_label) {
  if (loom_print_block_has_label(ctx, block)) {
    *out_label = ctx->module->strings.entries[block->label_id];
    return iree_ok_status();
  }
  uint16_t block_index = 0;
  if (!loom_print_region_find_block_index(region, block, &block_index)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "successor target block is not in the printed "
                            "region and has no explicit label");
  }
  for (uint32_t suffix = 0;; ++suffix) {
    int length =
        suffix == 0
            ? snprintf(synthetic_buffer, synthetic_buffer_capacity, "_bb%u",
                       (unsigned)block_index)
            : snprintf(synthetic_buffer, synthetic_buffer_capacity, "_bb%u_%u",
                       (unsigned)block_index, (unsigned)suffix);
    if (length < 0 || (iree_host_size_t)length >= synthetic_buffer_capacity) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "synthetic block label exceeds buffer capacity");
    }
    iree_string_view_t candidate =
        iree_make_string_view(synthetic_buffer, (iree_host_size_t)length);
    if (!loom_print_region_label_exists(ctx, region, candidate)) {
      *out_label = candidate;
      return iree_ok_status();
    }
  }
}

static bool loom_print_region_needs_synthetic_labels(
    const loom_print_context_t* ctx, const loom_region_t* region) {
  if (!region) return false;
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(region, block_index);
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      loom_block_t* const* successors = loom_op_const_successors(op);
      for (uint8_t i = 0; i < op->successor_count; ++i) {
        const loom_block_t* target = successors[i];
        if (target && !loom_print_block_has_label(ctx, target)) return true;
      }
    }
  }
  return false;
}

static iree_status_t loom_print_successor_ref(loom_print_context_t* ctx,
                                              const loom_op_t* op,
                                              uint8_t successor_index) {
  if (successor_index >= op->successor_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format SUCCESSOR_REF field_index %u out of range (op has %u "
        "successors)",
        successor_index, op->successor_count);
  }
  const loom_block_t* target = loom_op_const_successors(op)[successor_index];
  if (!target) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "successor %u target is NULL", successor_index);
  }
  const loom_region_t* region = op->parent_block
                                    ? op->parent_block->parent_region
                                    : target->parent_region;
  if (target->parent_region && region && target->parent_region != region) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "successor %u target belongs to a different region",
                            successor_index);
  }
  char label_buffer[64];
  iree_string_view_t label = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_print_block_label_view(
      ctx, region, target, label_buffer, sizeof(label_buffer), &label));

  iree_host_size_t successor_start =
      loom_print_next_token_start_offset(ctx, false, '^');
  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '^'));
  IREE_RETURN_IF_ERROR(loom_output_stream_write(ctx->stream, label));
  loom_print_did_write(ctx);
  loom_print_report_field(
      ctx, loom_print_field_ref(LOOM_PRINT_FIELD_SUCCESSOR, successor_index),
      successor_start, ctx->stream->offset);
  return iree_ok_status();
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
    case LOOM_TYPE_REGISTER: {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "reg<"));
      loom_string_id_t class_id = loom_type_register_class_id(type);
      if (module && class_id < module->strings.count) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write(
            stream, module->strings.entries[class_id]));
      } else {
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write_cstring(stream, "?register"));
      }
      uint32_t unit_count = loom_type_register_unit_count(type);
      if (unit_count != 1) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " x"));
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write_format(stream, "%u", unit_count));
      }
      return loom_output_stream_write_cstring(stream, ">");
    }
    case LOOM_TYPE_ENCODING:
      return loom_print_encoding_type(stream, type);
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

// Prints an attribute value. For enum attrs, |descriptor| provides the case
// name table for human-readable output. Pass NULL when no descriptor is
// available to print the raw enum value.
static iree_status_t loom_print_attr(loom_output_stream_t* stream,
                                     const loom_attribute_t* attr,
                                     const loom_module_t* module,
                                     const loom_attr_descriptor_t* descriptor) {
  switch (attr->kind) {
    case LOOM_ATTR_I64:
      return loom_output_stream_write_format(stream, "%" PRId64, attr->i64);
    case LOOM_ATTR_F64: {
      if (isnan(attr->f64)) {
        return loom_output_stream_write_cstring(stream, "nan");
      }
      if (isinf(attr->f64)) {
        return loom_output_stream_write_cstring(
            stream, attr->f64 < 0.0 ? "-inf" : "inf");
      }
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
      if (module && attr->type_id < module->types.count) {
        return loom_text_print_type(module->types.entries[attr->type_id],
                                    module, stream);
      }
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

iree_status_t loom_text_print_attribute(const loom_attribute_t* attr,
                                        const loom_module_t* module,
                                        loom_output_stream_t* stream) {
  if (attr == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "attribute to print is required");
  }
  if (stream == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "attribute output stream is required");
  }
  return loom_print_attr(stream, attr, module, NULL);
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
    const char* predicate_name = loom_predicate_kind_name(predicate->kind);
    if (!predicate_name) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown predicate kind %d",
                              (int)predicate->kind);
    }
    uint8_t expected_argument_count =
        loom_predicate_kind_argument_count(predicate->kind);
    if (predicate->arg_count != expected_argument_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "predicate kind %s expects %u arguments, got %u",
                              predicate_name, expected_argument_count,
                              predicate->arg_count);
    }
    // Emit kind name and opening paren: "mul("
    IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(ctx->stream, predicate_name));
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

static bool loom_print_format_element_covers_attr(
    const loom_format_element_t* element, uint16_t attr_index) {
  switch (element->kind) {
    case LOOM_FORMAT_KIND_ATTR_VALUE:
    case LOOM_FORMAT_KIND_SYMBOL_REF:
    case LOOM_FORMAT_KIND_OP_REF:
    case LOOM_FORMAT_KIND_TEMPLATE_PARAM:
    case LOOM_FORMAT_KIND_PREDICATE_LIST:
      return element->field_index == attr_index;
    case LOOM_FORMAT_KIND_DESCRIPTOR_REF:
      return element->field_index == attr_index || element->data == attr_index;
    case LOOM_FORMAT_KIND_INDEX_LIST:
      return element->data == attr_index;
    case LOOM_FORMAT_KIND_OPERAND_DICT:
      return element->data == attr_index;
    case LOOM_FORMAT_KIND_ATTR_TABLE:
      return element->data == attr_index;
    case LOOM_FORMAT_KIND_ATTR_DICT:
      if (iree_any_bit_set(element->data, LOOM_ATTR_DICT_FORMAT_INLINE_ATTRS)) {
        return false;
      }
      return element->field_index == attr_index;
    default:
      return false;
  }
}

static bool loom_print_inline_attr_dict_attr_present(
    const loom_op_t* op, const loom_op_vtable_t* vtable,
    const loom_format_element_t* inline_element, uint16_t attr_index) {
  if (attr_index >= op->attribute_count) return false;
  if (loom_attr_is_absent(loom_op_attrs(op)[attr_index])) return false;
  const loom_format_element_t* elements = vtable->format_elements;
  for (uint16_t i = 0; i < vtable->format_element_count; ++i) {
    const loom_format_element_t* element = &elements[i];
    if (element == inline_element) continue;
    if (loom_print_format_element_covers_attr(element, attr_index)) {
      return false;
    }
  }
  return true;
}

static bool loom_print_find_next_inline_attr(
    const loom_op_t* op, const loom_op_vtable_t* vtable,
    const loom_format_element_t* inline_element, bool has_previous_name,
    iree_string_view_t previous_name, uint16_t* out_attr_index,
    const loom_attr_descriptor_t** out_descriptor) {
  bool found = false;
  uint16_t best_index = 0;
  iree_string_view_t best_name = iree_string_view_empty();
  const loom_attr_descriptor_t* best_descriptor = NULL;

  uint16_t count = iree_min(vtable->attribute_count, op->attribute_count);
  for (uint16_t i = 0; i < count; ++i) {
    if (!loom_print_inline_attr_dict_attr_present(op, vtable, inline_element,
                                                  i)) {
      continue;
    }
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    iree_string_view_t name = loom_attr_descriptor_name(descriptor);
    if (has_previous_name &&
        iree_string_view_compare(name, previous_name) <= 0) {
      continue;
    }
    if (found && iree_string_view_compare(name, best_name) >= 0) continue;
    found = true;
    best_index = i;
    best_name = name;
    best_descriptor = descriptor;
  }

  if (!found) return false;
  *out_attr_index = best_index;
  *out_descriptor = best_descriptor;
  return true;
}

static iree_status_t loom_print_inline_attr_dict(
    loom_print_context_t* ctx, const loom_op_t* op,
    const loom_op_vtable_t* vtable,
    const loom_format_element_t* inline_element) {
  if (!vtable->attr_descriptors) return iree_ok_status();

  bool has_previous_name = false;
  iree_string_view_t previous_name = iree_string_view_empty();
  bool wrote_dict = false;
  for (;;) {
    uint16_t attr_index = 0;
    const loom_attr_descriptor_t* descriptor = NULL;
    if (!loom_print_find_next_inline_attr(op, vtable, inline_element,
                                          has_previous_name, previous_name,
                                          &attr_index, &descriptor)) {
      break;
    }

    if (!wrote_dict) {
      IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '{'));
      wrote_dict = true;
    } else {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, ", "));
    }

    iree_host_size_t attr_start = ctx->stream->offset;
    iree_string_view_t name = loom_attr_descriptor_name(descriptor);
    IREE_RETURN_IF_ERROR(loom_output_stream_write(ctx->stream, name));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, " = "));
    IREE_RETURN_IF_ERROR(loom_print_attr(
        ctx->stream, &loom_op_attrs(op)[attr_index], ctx->module, descriptor));
    loom_print_report_field(
        ctx, loom_print_field_ref(LOOM_PRINT_FIELD_ATTR, attr_index),
        attr_start, ctx->stream->offset);

    previous_name = name;
    has_previous_name = true;
  }

  if (wrote_dict) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '}'));
    loom_print_did_write(ctx);
  }
  return iree_ok_status();
}

static iree_status_t loom_print_operand_dict(
    loom_print_context_t* ctx, const loom_op_t* op,
    const loom_format_element_t* element) {
  uint16_t start = element->field_index;
  if (start > op->operand_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format OPERAND_DICT field_index %u out of range (op has %u operands)",
        start, op->operand_count);
  }
  uint16_t operand_count = (uint16_t)(op->operand_count - start);
  if (element->data >= op->attribute_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format OPERAND_DICT attr index %u out of range (op has %u attributes)",
        element->data, op->attribute_count);
  }

  loom_attribute_t names_attr = loom_op_attrs(op)[element->data];
  if (loom_attr_is_absent(names_attr)) {
    if (operand_count == 0) return iree_ok_status();
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format OPERAND_DICT has %u operands but names attr %u is absent",
        operand_count, element->data);
  }
  if (names_attr.kind != LOOM_ATTR_DICT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format OPERAND_DICT attr index %u expected DICT attr but found %d",
        element->data, (int)names_attr.kind);
  }
  if (names_attr.count == 0) {
    if (operand_count == 0) return iree_ok_status();
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format OPERAND_DICT has %u operands but names attr is empty",
        operand_count);
  }
  if (!names_attr.dict_entries) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "OPERAND_DICT names attr has count %u but NULL "
                            "entries",
                            names_attr.count);
  }
  if (names_attr.count != operand_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "format OPERAND_DICT has %u names but %u operands",
                            names_attr.count, operand_count);
  }

  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  iree_host_size_t dict_start = ctx->stream->offset;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '{'));
  for (uint16_t i = 0; i < names_attr.count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, ", "));
    }
    const loom_named_attr_t* entry = &names_attr.dict_entries[i];
    if (entry->name_id >= ctx->module->strings.count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "OPERAND_DICT key string id %u out of range (module has %" PRIhsz
          " strings)",
          entry->name_id, ctx->module->strings.count);
    }
    if (i > 0) {
      int comparison = iree_string_view_compare(
          ctx->module->strings.entries[names_attr.dict_entries[i - 1].name_id],
          ctx->module->strings.entries[entry->name_id]);
      if (comparison >= 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "OPERAND_DICT names attr keys are not "
                                "canonical sorted unique keys");
      }
    }
    if (entry->value.kind != LOOM_ATTR_I64) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "OPERAND_DICT key attr expected I64 ordinal but found %d",
          (int)entry->value.kind);
    }
    int64_t ordinal = entry->value.i64;
    if (ordinal < 0 || ordinal >= operand_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "OPERAND_DICT key ordinal %" PRId64
                              " out of range for %u operands",
                              ordinal, operand_count);
    }
    for (uint16_t j = 0; j < i; ++j) {
      if (names_attr.dict_entries[j].value.kind == LOOM_ATTR_I64 &&
          names_attr.dict_entries[j].value.i64 == ordinal) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "OPERAND_DICT key ordinal %" PRId64 " is duplicated", ordinal);
      }
    }

    IREE_RETURN_IF_ERROR(loom_output_stream_write(
        ctx->stream, ctx->module->strings.entries[entry->name_id]));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, " = "));
    uint16_t operand_index = (uint16_t)(start + (uint16_t)ordinal);
    iree_host_size_t value_start = ctx->stream->offset;
    IREE_RETURN_IF_ERROR(loom_print_value_ref(
        ctx->stream, ctx->module, loom_op_const_operands(op)[operand_index]));
    loom_print_report_field(
        ctx, loom_print_field_ref(LOOM_PRINT_FIELD_OPERAND, operand_index),
        value_start, ctx->stream->offset);
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, " : "));
    IREE_RETURN_IF_ERROR(
        loom_print_value_type(ctx, loom_op_const_operands(op)[operand_index]));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '}'));
  loom_print_did_write(ctx);
  loom_print_report_field(
      ctx, loom_print_field_ref(LOOM_PRINT_FIELD_ATTR, element->data),
      dict_start, ctx->stream->offset);
  return iree_ok_status();
}

static iree_status_t loom_print_attr_table_row_values(loom_print_context_t* ctx,
                                                      const loom_op_t* op,
                                                      uint16_t start,
                                                      uint16_t row_index,
                                                      uint16_t row_width) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '('));
  for (uint16_t column = 0; column < row_width; ++column) {
    if (column > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, ", "));
    }
    uint16_t operand_index = (uint16_t)(start + row_index * row_width + column);
    iree_host_size_t value_start = ctx->stream->offset;
    IREE_RETURN_IF_ERROR(loom_print_value_ref(ctx->stream, ctx->module,
                                              operands[operand_index]));
    loom_print_report_field(
        ctx, loom_print_field_ref(LOOM_PRINT_FIELD_OPERAND, operand_index),
        value_start, ctx->stream->offset);
  }
  return loom_output_stream_write_char(ctx->stream, ')');
}

static iree_status_t loom_print_attr_table(
    loom_print_context_t* ctx, const loom_op_t* op,
    const loom_format_element_t* element) {
  uint16_t start = element->field_index;
  if (start > op->operand_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format ATTR_TABLE field_index %u out of range (op has %u operands)",
        start, op->operand_count);
  }
  if (element->data >= op->attribute_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format ATTR_TABLE attr index %u out of range (op has %u attributes)",
        element->data, op->attribute_count);
  }

  loom_attribute_t keys_attr = loom_op_attrs(op)[element->data];
  if (keys_attr.kind != LOOM_ATTR_I64_ARRAY) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format ATTR_TABLE attr index %u expected I64_ARRAY attr but found %d",
        element->data, (int)keys_attr.kind);
  }
  if (keys_attr.count > 0 && !keys_attr.i64_array) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ATTR_TABLE keys attr has count %u but NULL values",
                            keys_attr.count);
  }
  uint16_t operand_count = (uint16_t)(op->operand_count - start);
  iree_host_size_t row_count = (iree_host_size_t)keys_attr.count + 1;
  if (operand_count % row_count != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "format ATTR_TABLE has %u operands for %" PRIhsz
                            " rows",
                            operand_count, row_count);
  }
  uint16_t row_width = (uint16_t)(operand_count / row_count);

  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  iree_host_size_t table_start = ctx->stream->offset;

  if (keys_attr.count == 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, "{}"));
  } else {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, "{\n"));
    ++ctx->indent;
    for (uint16_t row = 0; row < keys_attr.count; ++row) {
      IREE_RETURN_IF_ERROR(loom_print_indent(ctx));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          ctx->stream, "%" PRId64 " = ", keys_attr.i64_array[row]));
      IREE_RETURN_IF_ERROR(
          loom_print_attr_table_row_values(ctx, op, start, row, row_width));
      if (row + 1 < keys_attr.count) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, ','));
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '\n'));
    }
    --ctx->indent;
    IREE_RETURN_IF_ERROR(loom_print_indent(ctx));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '}'));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(ctx->stream, " default"));
  uint16_t default_row = keys_attr.count;
  IREE_RETURN_IF_ERROR(
      loom_print_attr_table_row_values(ctx, op, start, default_row, row_width));
  loom_print_did_write(ctx);
  loom_print_report_field(
      ctx, loom_print_field_ref(LOOM_PRINT_FIELD_ATTR, element->data),
      table_start, ctx->stream->offset);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Format element walk — writes directly to stream via print context
//===----------------------------------------------------------------------===//

static iree_status_t loom_print_module_body(loom_print_context_t* ctx,
                                            const loom_region_t* region);
static iree_status_t loom_print_op(loom_print_context_t* ctx,
                                   const loom_op_t* op);
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

static iree_status_t loom_print_low_asm_lookup_descriptor_set(
    loom_print_context_t* ctx, const void** out_descriptor_set) {
  *out_descriptor_set = NULL;
  if (!loom_text_low_asm_environment_supports_printing(
          &ctx->low_asm_environment)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "low asm region printing requires a descriptor-backed print "
        "environment");
  }
  if (iree_string_view_is_empty(ctx->low_asm_descriptor_set_key)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low asm region printing requires a descriptor-set key");
  }
  return ctx->low_asm_environment.vtable->lookup_descriptor_set(
      ctx->low_asm_environment.user_data, ctx->low_asm_descriptor_set_key,
      out_descriptor_set);
}

static bool loom_print_low_asm_is_requested(loom_print_context_t* ctx) {
  return !iree_string_view_is_empty(ctx->low_asm_descriptor_set_key);
}

static iree_status_t loom_print_low_asm_describe_operation(
    loom_print_context_t* ctx, const void* descriptor_set, const loom_op_t* op,
    loom_text_low_asm_statement_t* out_statement) {
  return ctx->low_asm_environment.vtable->describe_operation(
      ctx->low_asm_environment.user_data, descriptor_set, ctx->module, op,
      out_statement);
}

static iree_status_t loom_print_low_asm_attr_name(
    const loom_module_t* module, const loom_named_attr_t* attr,
    iree_string_view_t* out_name) {
  *out_name = iree_string_view_empty();
  if (attr->name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm immediate attribute name is out of range");
  }
  *out_name = module->strings.entries[attr->name_id];
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_find_immediate_attr(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement,
    uint16_t immediate_index,
    loom_text_low_asm_immediate_descriptor_t* out_immediate,
    const loom_named_attr_t** out_attr) {
  *out_immediate = (loom_text_low_asm_immediate_descriptor_t){0};
  *out_attr = NULL;
  IREE_RETURN_IF_ERROR(ctx->low_asm_environment.vtable->immediate_descriptor(
      ctx->low_asm_environment.user_data, &statement->packet, immediate_index,
      out_immediate));
  for (iree_host_size_t i = 0; i < statement->attributes.count; ++i) {
    iree_string_view_t attr_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_print_low_asm_attr_name(
        ctx->module, &statement->attributes.entries[i], &attr_name));
    if (iree_string_view_equal(attr_name, out_immediate->field_name)) {
      *out_attr = &statement->attributes.entries[i];
      return iree_ok_status();
    }
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "low asm packet '%.*s' is missing immediate "
      "attribute '%.*s'",
      (int)statement->packet.opcode_key.size, statement->packet.opcode_key.data,
      (int)out_immediate->field_name.size, out_immediate->field_name.data);
}

static iree_status_t loom_print_low_asm_validate_result_types(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  for (uint16_t i = 0; i < statement->result_count; ++i) {
    loom_value_id_t result = statement->results[i];
    if (result >= ctx->module->values.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low asm packet '%.*s' result %u is out of "
                              "range",
                              (int)statement->packet.opcode_key.size,
                              statement->packet.opcode_key.data, i);
    }
    bool annotation_required = false;
    iree_string_view_t diagnostic_detail = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(
        ctx->low_asm_environment.vtable->result_type_annotation_required(
            ctx->low_asm_environment.user_data, &statement->packet,
            statement->operands, statement->operand_count, i, ctx->module,
            ctx->module->values.entries[result].type, &annotation_required,
            &diagnostic_detail));
    (void)annotation_required;
    if (!iree_string_view_is_empty(diagnostic_detail)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low asm packet '%.*s' result %u cannot be "
                              "printed: %.*s",
                              (int)statement->packet.opcode_key.size,
                              statement->packet.opcode_key.data, i,
                              (int)diagnostic_detail.size,
                              diagnostic_detail.data);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_result_types_require_annotation(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement,
    bool* out_required) {
  *out_required = false;
  for (uint16_t i = 0; i < statement->result_count; ++i) {
    loom_value_id_t result = statement->results[i];
    bool annotation_required = false;
    iree_string_view_t diagnostic_detail = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(
        ctx->low_asm_environment.vtable->result_type_annotation_required(
            ctx->low_asm_environment.user_data, &statement->packet,
            statement->operands, statement->operand_count, i, ctx->module,
            ctx->module->values.entries[result].type, &annotation_required,
            &diagnostic_detail));
    if (!iree_string_view_is_empty(diagnostic_detail)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low asm packet '%.*s' result %u cannot be "
                              "printed: %.*s",
                              (int)statement->packet.opcode_key.size,
                              statement->packet.opcode_key.data, i,
                              (int)diagnostic_detail.size,
                              diagnostic_detail.data);
    }
    *out_required |= annotation_required;
  }
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_validate_statement(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  switch (statement->kind) {
    case LOOM_TEXT_LOW_ASM_STATEMENT_PACKET: {
      if (statement->result_count != statement->packet.result_count ||
          statement->operand_count != statement->packet.operand_count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "low asm packet '%.*s' statement shape does not match descriptor",
            (int)statement->packet.opcode_key.size,
            statement->packet.opcode_key.data);
      }
      if (statement->result_count > 0 && statement->results == NULL) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "low asm packet results are missing");
      }
      if (statement->operand_count > 0 && statement->operands == NULL) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "low asm packet operands are missing");
      }
      for (uint16_t i = 0; i < statement->operand_count; ++i) {
        if (statement->operands[i] >= ctx->module->values.count) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "low asm packet '%.*s' operand %u is out of "
                                  "range",
                                  (int)statement->packet.opcode_key.size,
                                  statement->packet.opcode_key.data, i);
        }
      }
      if (statement->attributes.count > 0 &&
          statement->attributes.entries == NULL) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "low asm packet immediates are missing");
      }
      if (statement->attributes.count != statement->packet.immediate_count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "low asm packet '%.*s' immediate count does not match descriptor",
            (int)statement->packet.opcode_key.size,
            statement->packet.opcode_key.data);
      }
      for (uint16_t i = 0; i < statement->packet.immediate_count; ++i) {
        loom_text_low_asm_immediate_descriptor_t immediate = {0};
        const loom_named_attr_t* attr = NULL;
        IREE_RETURN_IF_ERROR(loom_print_low_asm_find_immediate_attr(
            ctx, statement, i, &immediate, &attr));
        (void)attr;
      }
      IREE_RETURN_IF_ERROR(
          loom_print_low_asm_validate_result_types(ctx, statement));
      break;
    }
    case LOOM_TEXT_LOW_ASM_STATEMENT_RETURN: {
      if (statement->operand_count > 0 && statement->operands == NULL) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "low asm return values are missing");
      }
      for (uint16_t i = 0; i < statement->operand_count; ++i) {
        if (statement->operands[i] >= ctx->module->values.count) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "low asm return value %u is out of range", i);
        }
      }
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown low asm statement kind %u",
                              (uint32_t)statement->kind);
  }
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_region_preflight(
    loom_print_context_t* ctx, const loom_region_t* region,
    const void* descriptor_set, bool allow_entry_block_args) {
  if (!region || region->block_count == 0) return iree_ok_status();
  if (region->block_count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm region printing supports single-block "
                            "regions only");
  }
  const loom_block_t* block = loom_region_const_entry_block(region);
  if (block->arg_count != 0 && !allow_entry_block_args) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm region printing does not support block "
                            "arguments");
  }
  const loom_op_t* current_op = NULL;
  loom_block_for_each_op(block, current_op) {
    loom_text_low_asm_statement_t statement = {0};
    IREE_RETURN_IF_ERROR(loom_print_low_asm_describe_operation(
        ctx, descriptor_set, current_op, &statement));
    IREE_RETURN_IF_ERROR(
        loom_print_low_asm_validate_statement(ctx, &statement));
  }
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_result_list(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  for (uint16_t i = 0; i < statement->result_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ",", false));
    }
    IREE_RETURN_IF_ERROR(loom_print_value_name_with_field(
        ctx, statement->results[i],
        loom_print_field_ref(LOOM_PRINT_FIELD_RESULT, i)));
  }
  if (statement->result_count > 0) {
    IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "=", false));
  }
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_value_list(
    loom_print_context_t* ctx, const loom_value_id_t* values,
    uint16_t value_count, loom_print_field_kind_t field_kind) {
  for (uint16_t i = 0; i < value_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ",", false));
    }
    IREE_RETURN_IF_ERROR(loom_print_value_name_with_field(
        ctx, values[i], loom_print_field_ref(field_kind, i)));
  }
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_named_immediates(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  iree_host_size_t start = ctx->stream->offset;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '{'));
  for (uint16_t i = 0; i < statement->packet.immediate_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, ", "));
    }
    loom_text_low_asm_immediate_descriptor_t immediate = {0};
    const loom_named_attr_t* attr = NULL;
    IREE_RETURN_IF_ERROR(loom_print_low_asm_find_immediate_attr(
        ctx, statement, i, &immediate, &attr));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write(ctx->stream, immediate.spelling));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, " = "));
    IREE_RETURN_IF_ERROR(
        loom_print_attr(ctx->stream, &attr->value, ctx->module, NULL));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '}'));
  loom_print_did_write(ctx);
  if (statement->has_immediate_attribute_field) {
    loom_print_report_field(
        ctx,
        loom_print_field_ref(LOOM_PRINT_FIELD_ATTR,
                             statement->immediate_attribute_field_index),
        start, ctx->stream->offset);
  }
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_positional_immediates(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  for (uint16_t i = 0; i < statement->packet.immediate_count; ++i) {
    if (i > 0 || statement->operand_count > 0) {
      IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ",", false));
    }
    loom_text_low_asm_immediate_descriptor_t immediate = {0};
    const loom_named_attr_t* attr = NULL;
    IREE_RETURN_IF_ERROR(loom_print_low_asm_find_immediate_attr(
        ctx, statement, i, &immediate, &attr));
    IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
    iree_host_size_t start = ctx->stream->offset;
    IREE_RETURN_IF_ERROR(
        loom_print_attr(ctx->stream, &attr->value, ctx->module, NULL));
    loom_print_did_write(ctx);
    if (statement->has_immediate_attribute_field) {
      loom_print_report_field(
          ctx,
          loom_print_field_ref(LOOM_PRINT_FIELD_ATTR,
                               statement->immediate_attribute_field_index),
          start, ctx->stream->offset);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_immediates(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  if (statement->packet.immediate_count == 0) return iree_ok_status();
  if (statement->packet.has_named_immediates) {
    return loom_print_low_asm_named_immediates(ctx, statement);
  }
  return loom_print_low_asm_positional_immediates(ctx, statement);
}

static iree_status_t loom_print_low_asm_result_type_annotation(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  bool annotation_required = false;
  IREE_RETURN_IF_ERROR(loom_print_low_asm_result_types_require_annotation(
      ctx, statement, &annotation_required));
  if (!annotation_required) return iree_ok_status();

  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ":", false));
  for (uint16_t i = 0; i < statement->result_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ",", false));
    }
    IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
    IREE_RETURN_IF_ERROR(loom_text_print_type(
        ctx->module->values.entries[statement->results[i]].type, ctx->module,
        ctx->stream));
    loom_print_did_write(ctx);
  }
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_packet(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  IREE_RETURN_IF_ERROR(loom_print_low_asm_result_list(ctx, statement));
  IREE_RETURN_IF_ERROR(loom_print_emit(ctx, statement->packet.mnemonic, false));
  IREE_RETURN_IF_ERROR(loom_print_low_asm_value_list(ctx, statement->operands,
                                                     statement->operand_count,
                                                     LOOM_PRINT_FIELD_OPERAND));
  IREE_RETURN_IF_ERROR(loom_print_low_asm_immediates(ctx, statement));
  IREE_RETURN_IF_ERROR(
      loom_print_low_asm_result_type_annotation(ctx, statement));
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_return(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "return", false));
  return loom_print_low_asm_value_list(ctx, statement->operands,
                                       statement->operand_count,
                                       LOOM_PRINT_FIELD_OPERAND);
}

static iree_status_t loom_print_low_asm_statement(
    loom_print_context_t* ctx, const loom_text_low_asm_statement_t* statement) {
  ctx->has_previous_token = false;
  ctx->glue_next = false;
  ctx->last_char = 0;

  switch (statement->kind) {
    case LOOM_TEXT_LOW_ASM_STATEMENT_PACKET: {
      IREE_RETURN_IF_ERROR(loom_print_low_asm_packet(ctx, statement));
      break;
    }
    case LOOM_TEXT_LOW_ASM_STATEMENT_RETURN: {
      IREE_RETURN_IF_ERROR(loom_print_low_asm_return(ctx, statement));
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown low asm statement kind %u",
                              (uint32_t)statement->kind);
  }

  if (iree_any_bit_set(ctx->flags, LOOM_TEXT_PRINT_LOCATIONS)) {
    IREE_RETURN_IF_ERROR(
        loom_print_location(ctx->stream, ctx->module, statement->location));
  }
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_region_body(
    loom_print_context_t* ctx, const loom_region_t* region,
    const void* descriptor_set) {
  if (!region || region->block_count == 0) return iree_ok_status();
  const loom_block_t* block = loom_region_const_entry_block(region);
  const loom_op_t* current_op = NULL;
  loom_block_for_each_op(block, current_op) {
    IREE_RETURN_IF_ERROR(loom_print_op_comments(ctx, current_op));
    IREE_RETURN_IF_ERROR(loom_print_indent(ctx));
    loom_text_low_asm_statement_t statement = {0};
    IREE_RETURN_IF_ERROR(loom_print_low_asm_describe_operation(
        ctx, descriptor_set, current_op, &statement));
    IREE_RETURN_IF_ERROR(loom_print_low_asm_statement(ctx, &statement));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '\n'));
  }
  return iree_ok_status();
}

static iree_status_t loom_print_low_asm_region(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor,
    bool allow_entry_block_args) {
  (void)region_descriptor;
  const void* descriptor_set = NULL;
  IREE_RETURN_IF_ERROR(
      loom_print_low_asm_lookup_descriptor_set(ctx, &descriptor_set));

  if (!iree_any_bit_set(ctx->flags, LOOM_TEXT_PRINT_SKIP_REGIONS)) {
    IREE_RETURN_IF_ERROR(loom_print_low_asm_region_preflight(
        ctx, region, descriptor_set, allow_entry_block_args));
  }

  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "asm", false));
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "<", true));
  IREE_RETURN_IF_ERROR(
      loom_print_emit(ctx, ctx->low_asm_descriptor_set_key, true));
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ">", true));
  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  if (iree_any_bit_set(ctx->flags, LOOM_TEXT_PRINT_SKIP_REGIONS)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(ctx->stream, "{ ... }"));
  } else {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, "{\n"));
    ++ctx->indent;
    IREE_RETURN_IF_ERROR(
        loom_print_low_asm_region_body(ctx, region, descriptor_set));
    --ctx->indent;
    IREE_RETURN_IF_ERROR(loom_print_indent(ctx));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '}'));
  }
  ctx->has_previous_token = true;
  ctx->last_char = '}';
  ctx->glue_next = false;
  return iree_ok_status();
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

static const loom_value_id_t* loom_print_region_entry_arg_ids(
    const loom_op_t* op, uint8_t region_index, uint16_t* out_arg_count) {
  *out_arg_count = 0;
  if (region_index >= op->region_count) return NULL;
  loom_region_t* region = loom_op_regions(op)[region_index];
  if (!region || region->block_count == 0) return NULL;
  const loom_block_t* block = loom_region_const_entry_block(region);
  *out_arg_count = block->arg_count;
  return block->arg_ids;
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

static iree_status_t loom_print_braced_region(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, "{\n"));
  if (region) {
    ++ctx->indent;
    IREE_RETURN_IF_ERROR(
        loom_print_region_body(ctx, region, region_descriptor));
    --ctx->indent;
  }
  IREE_RETURN_IF_ERROR(loom_print_indent(ctx));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '}'));
  ctx->has_previous_token = true;
  ctx->last_char = '}';
  ctx->glue_next = false;
  return iree_ok_status();
}

static bool loom_print_pipeline_is_ident_start(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' ||
         c == '$';
}

static bool loom_print_pipeline_is_ident_continue(char c, bool allow_dot) {
  return loom_print_pipeline_is_ident_start(c) || (c >= '0' && c <= '9') ||
         c == '-' || (allow_dot && c == '.');
}

static bool loom_print_pipeline_is_printable_name(iree_string_view_t name,
                                                  bool allow_dot) {
  if (name.size == 0 || !loom_print_pipeline_is_ident_start(name.data[0])) {
    return false;
  }
  for (iree_host_size_t i = 1; i < name.size; ++i) {
    if (!loom_print_pipeline_is_ident_continue(name.data[i], allow_dot)) {
      return false;
    }
  }
  return true;
}

static bool loom_print_pipeline_is_elidable_terminator(
    const loom_region_descriptor_t* region_descriptor, const loom_op_t* op) {
  return region_descriptor->implicit_terminator != LOOM_OP_KIND_UNKNOWN && op &&
         op->kind == region_descriptor->implicit_terminator &&
         op->operand_count == 0 && op->result_count == 0 &&
         op->region_count == 0 && op->successor_count == 0 &&
         op->tied_result_count == 0 && op->attribute_count == 0 &&
         op->instance_flags == 0;
}

static const loom_op_vtable_t* loom_print_pipeline_op_vtable(
    const loom_print_context_t* ctx, const loom_op_t* op,
    iree_string_view_t expected_name) {
  const loom_op_vtable_t* vtable = loom_op_vtable(ctx->module, op);
  if (!vtable ||
      !iree_string_view_equal(loom_op_vtable_name(vtable), expected_name)) {
    return NULL;
  }
  if (op->operand_count != 0 || op->result_count != 0 ||
      op->successor_count != 0 || op->tied_result_count != 0 ||
      op->instance_flags != 0 ||
      op->attribute_count != vtable->attribute_count ||
      op->region_count != vtable->region_count) {
    return NULL;
  }
  if (vtable->attribute_count > 0 && !vtable->attr_descriptors) return NULL;
  return vtable;
}

static const loom_attribute_t* loom_print_pipeline_find_attr(
    const loom_op_t* op, const loom_op_vtable_t* vtable,
    iree_string_view_t attr_name,
    const loom_attr_descriptor_t** out_descriptor) {
  *out_descriptor = NULL;
  for (uint8_t i = 0; i < vtable->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    if (iree_string_view_equal(loom_attr_descriptor_name(descriptor),
                               attr_name)) {
      *out_descriptor = descriptor;
      return &loom_op_const_attrs(op)[i];
    }
  }
  return NULL;
}

static bool loom_print_pipeline_string_attr(const loom_print_context_t* ctx,
                                            const loom_op_t* op,
                                            const loom_op_vtable_t* vtable,
                                            iree_string_view_t attr_name,
                                            iree_string_view_t* out_value) {
  const loom_attr_descriptor_t* descriptor = NULL;
  const loom_attribute_t* attr =
      loom_print_pipeline_find_attr(op, vtable, attr_name, &descriptor);
  (void)descriptor;
  if (!attr || attr->kind != LOOM_ATTR_STRING ||
      attr->string_id >= ctx->module->strings.count) {
    return false;
  }
  *out_value = ctx->module->strings.entries[attr->string_id];
  return true;
}

static bool loom_print_pipeline_enum_attr(const loom_op_t* op,
                                          const loom_op_vtable_t* vtable,
                                          iree_string_view_t attr_name,
                                          iree_string_view_t* out_value) {
  const loom_attr_descriptor_t* descriptor = NULL;
  const loom_attribute_t* attr =
      loom_print_pipeline_find_attr(op, vtable, attr_name, &descriptor);
  if (!attr || attr->kind != LOOM_ATTR_ENUM || !descriptor ||
      descriptor->attr_kind != LOOM_ATTR_ENUM || !descriptor->enum_case_names ||
      attr->raw >= descriptor->enum_case_count) {
    return false;
  }
  *out_value =
      loom_bstring_view(descriptor->enum_case_names[(uint8_t)attr->raw]);
  return true;
}

static bool loom_print_pipeline_symbol_attr(const loom_print_context_t* ctx,
                                            const loom_op_t* op,
                                            const loom_op_vtable_t* vtable,
                                            iree_string_view_t attr_name) {
  const loom_attr_descriptor_t* descriptor = NULL;
  const loom_attribute_t* attr =
      loom_print_pipeline_find_attr(op, vtable, attr_name, &descriptor);
  (void)descriptor;
  if (!attr || attr->kind != LOOM_ATTR_SYMBOL || attr->symbol.module_id != 0 ||
      attr->symbol.symbol_id >= ctx->module->symbols.count) {
    return false;
  }
  loom_string_id_t name_id =
      ctx->module->symbols.entries[attr->symbol.symbol_id].name_id;
  return name_id < ctx->module->strings.count &&
         loom_print_pipeline_is_printable_name(
             ctx->module->strings.entries[name_id], /*allow_dot=*/false);
}

static bool loom_print_pipeline_attr_value_is_printable(
    const loom_print_context_t* ctx, const loom_attribute_t* attr,
    uint8_t nesting_depth) {
  switch ((loom_attr_kind_t)attr->kind) {
    case LOOM_ATTR_I64:
    case LOOM_ATTR_F64:
    case LOOM_ATTR_BOOL:
      return true;
    case LOOM_ATTR_STRING:
      return attr->string_id < ctx->module->strings.count;
    case LOOM_ATTR_I64_ARRAY:
      return attr->count == 0 || attr->i64_array != NULL;
    case LOOM_ATTR_SYMBOL: {
      if (attr->symbol.module_id != 0 ||
          attr->symbol.symbol_id >= ctx->module->symbols.count) {
        return false;
      }
      loom_string_id_t name_id =
          ctx->module->symbols.entries[attr->symbol.symbol_id].name_id;
      return name_id < ctx->module->strings.count &&
             loom_print_pipeline_is_printable_name(
                 ctx->module->strings.entries[name_id], /*allow_dot=*/false);
    }
    case LOOM_ATTR_TYPE:
      return attr->type_id < ctx->module->types.count;
    case LOOM_ATTR_ENCODING:
      return attr->encoding_id > 0 &&
             attr->encoding_id <= ctx->module->encodings.count;
    case LOOM_ATTR_DICT: {
      if (nesting_depth >= LOOM_ATTR_DICT_MAX_NESTING_DEPTH) return false;
      if (attr->count > 0 && !attr->dict_entries) return false;
      for (uint16_t i = 0; i < attr->count; ++i) {
        const loom_named_attr_t* entry = &attr->dict_entries[i];
        if (entry->name_id >= ctx->module->strings.count ||
            !loom_print_pipeline_is_printable_name(
                ctx->module->strings.entries[entry->name_id],
                /*allow_dot=*/false) ||
            !loom_print_pipeline_attr_value_is_printable(
                ctx, &entry->value, (uint8_t)(nesting_depth + 1))) {
          return false;
        }
      }
      return true;
    }
    case LOOM_ATTR_ABSENT:
    case LOOM_ATTR_ENUM:
    case LOOM_ATTR_PREDICATE_LIST:
    case LOOM_ATTR_ANY:
    case LOOM_ATTR_COUNT_:
      return false;
  }
  return false;
}

static bool loom_print_pipeline_dict_attr(const loom_print_context_t* ctx,
                                          const loom_op_t* op,
                                          const loom_op_vtable_t* vtable,
                                          iree_string_view_t attr_name,
                                          loom_named_attr_slice_t* out_attrs) {
  *out_attrs = loom_make_named_attr_slice(NULL, 0);
  const loom_attr_descriptor_t* descriptor = NULL;
  const loom_attribute_t* attr =
      loom_print_pipeline_find_attr(op, vtable, attr_name, &descriptor);
  if (!attr || loom_attr_is_absent(*attr)) return true;
  if (!descriptor || descriptor->attr_kind != LOOM_ATTR_DICT ||
      attr->kind != LOOM_ATTR_DICT ||
      !loom_print_pipeline_attr_value_is_printable(ctx, attr,
                                                   /*nesting_depth=*/0)) {
    return false;
  }
  *out_attrs = loom_attr_as_dict(*attr);
  return true;
}

static bool loom_print_pipeline_optional_i64_attr(
    const loom_op_t* op, const loom_op_vtable_t* vtable,
    iree_string_view_t attr_name, const loom_attribute_t** out_attr) {
  *out_attr = NULL;
  const loom_attr_descriptor_t* descriptor = NULL;
  const loom_attribute_t* attr =
      loom_print_pipeline_find_attr(op, vtable, attr_name, &descriptor);
  if (!attr || loom_attr_is_absent(*attr)) return true;
  if (!descriptor || descriptor->attr_kind != LOOM_ATTR_I64 ||
      attr->kind != LOOM_ATTR_I64) {
    return false;
  }
  *out_attr = attr;
  return true;
}

static const loom_region_descriptor_t*
loom_print_pipeline_body_region_descriptor(const loom_op_vtable_t* vtable) {
  return vtable->region_count == 1 ? loom_op_vtable_region_descriptor(vtable, 0)
                                   : NULL;
}

static bool loom_print_pipeline_region_is_friendly(
    const loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor);

static bool loom_print_pipeline_statement_is_friendly(
    const loom_print_context_t* ctx, const loom_op_t* op) {
  const loom_op_vtable_t* vtable = NULL;
  iree_string_view_t value = iree_string_view_empty();
  loom_named_attr_slice_t attrs = loom_make_named_attr_slice(NULL, 0);

  vtable = loom_print_pipeline_op_vtable(ctx, op, IREE_SV("pass.run"));
  if (vtable) {
    return loom_print_pipeline_string_attr(ctx, op, vtable, IREE_SV("key"),
                                           &value) &&
           loom_print_pipeline_is_printable_name(value, /*allow_dot=*/true) &&
           loom_print_pipeline_dict_attr(ctx, op, vtable, IREE_SV("options"),
                                         &attrs);
  }

  vtable = loom_print_pipeline_op_vtable(ctx, op, IREE_SV("pass.for"));
  if (vtable) {
    const loom_region_descriptor_t* body_descriptor =
        loom_print_pipeline_body_region_descriptor(vtable);
    return loom_print_pipeline_enum_attr(op, vtable, IREE_SV("anchor"),
                                         &value) &&
           loom_print_pipeline_is_printable_name(value, /*allow_dot=*/false) &&
           body_descriptor &&
           loom_print_pipeline_region_is_friendly(ctx, loom_op_regions(op)[0],
                                                  body_descriptor);
  }

  vtable = loom_print_pipeline_op_vtable(ctx, op, IREE_SV("pass.where"));
  if (vtable) {
    const loom_region_descriptor_t* body_descriptor =
        loom_print_pipeline_body_region_descriptor(vtable);
    return loom_print_pipeline_string_attr(ctx, op, vtable,
                                           IREE_SV("predicate"), &value) &&
           loom_print_pipeline_is_printable_name(value, /*allow_dot=*/true) &&
           loom_print_pipeline_dict_attr(ctx, op, vtable, IREE_SV("attrs"),
                                         &attrs) &&
           body_descriptor &&
           loom_print_pipeline_region_is_friendly(ctx, loom_op_regions(op)[0],
                                                  body_descriptor);
  }

  vtable = loom_print_pipeline_op_vtable(ctx, op, IREE_SV("pass.repeat"));
  if (vtable) {
    const loom_attribute_t* count_attr = NULL;
    const loom_attribute_t* max_iterations_attr = NULL;
    const loom_region_descriptor_t* body_descriptor =
        loom_print_pipeline_body_region_descriptor(vtable);
    return loom_print_pipeline_enum_attr(op, vtable, IREE_SV("mode"), &value) &&
           loom_print_pipeline_is_printable_name(value, /*allow_dot=*/false) &&
           loom_print_pipeline_optional_i64_attr(op, vtable, IREE_SV("count"),
                                                 &count_attr) &&
           loom_print_pipeline_optional_i64_attr(
               op, vtable, IREE_SV("max_iterations"), &max_iterations_attr) &&
           body_descriptor &&
           loom_print_pipeline_region_is_friendly(ctx, loom_op_regions(op)[0],
                                                  body_descriptor);
  }

  vtable = loom_print_pipeline_op_vtable(ctx, op, IREE_SV("pass.call"));
  if (vtable) {
    return loom_print_pipeline_symbol_attr(ctx, op, vtable, IREE_SV("callee"));
  }

  vtable = loom_print_pipeline_op_vtable(ctx, op, IREE_SV("pass.fail"));
  if (vtable) {
    return loom_print_pipeline_string_attr(ctx, op, vtable, IREE_SV("message"),
                                           &value);
  }

  vtable = loom_print_pipeline_op_vtable(ctx, op, IREE_SV("pass.halt"));
  if (vtable) {
    return loom_print_pipeline_string_attr(ctx, op, vtable, IREE_SV("message"),
                                           &value);
  }

  return false;
}

static bool loom_print_pipeline_region_is_friendly(
    const loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor) {
  if (iree_any_bit_set(ctx->flags, LOOM_TEXT_PRINT_LOCATIONS) || !region ||
      !region_descriptor ||
      region_descriptor->implicit_terminator == LOOM_OP_KIND_UNKNOWN ||
      region->block_count != 1 || region->flags != 0) {
    return false;
  }
  const loom_block_t* block = loom_region_const_entry_block(region);
  iree_host_size_t block_comment_count = 0;
  (void)loom_module_block_comments(ctx->module, block, &block_comment_count);
  if (block->arg_count != 0 || block->op_count == 0 ||
      block_comment_count != 0 || loom_print_block_has_label(ctx, block) ||
      !loom_print_pipeline_is_elidable_terminator(region_descriptor,
                                                  block->last_op)) {
    return false;
  }

  const loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (op == block->last_op) break;
    if (!loom_print_pipeline_statement_is_friendly(ctx, op)) return false;
  }
  return true;
}

static iree_status_t loom_print_pipeline_attr_parens(
    loom_print_context_t* ctx, loom_named_attr_slice_t attrs) {
  if (attrs.count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '('));
  for (uint16_t i = 0; i < attrs.count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, ", "));
    }
    const loom_named_attr_t* entry = &attrs.entries[i];
    IREE_RETURN_IF_ERROR(loom_output_stream_write(
        ctx->stream, ctx->module->strings.entries[entry->name_id]));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, " = "));
    IREE_RETURN_IF_ERROR(
        loom_print_attr(ctx->stream, &entry->value, ctx->module, NULL));
  }
  return loom_output_stream_write_char(ctx->stream, ')');
}

static iree_status_t loom_print_pipeline_string_attr_value(
    loom_print_context_t* ctx, const loom_op_t* op,
    const loom_op_vtable_t* vtable, iree_string_view_t attr_name) {
  const loom_attr_descriptor_t* descriptor = NULL;
  const loom_attribute_t* attr =
      loom_print_pipeline_find_attr(op, vtable, attr_name, &descriptor);
  (void)descriptor;
  if (!attr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "missing pass pipeline string attribute");
  }
  return loom_print_attr(ctx->stream, attr, ctx->module, NULL);
}

static iree_status_t loom_print_pipeline_symbol_attr_value(
    loom_print_context_t* ctx, const loom_op_t* op,
    const loom_op_vtable_t* vtable, iree_string_view_t attr_name) {
  const loom_attr_descriptor_t* descriptor = NULL;
  const loom_attribute_t* attr =
      loom_print_pipeline_find_attr(op, vtable, attr_name, &descriptor);
  (void)descriptor;
  if (!attr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "missing pass pipeline symbol attribute");
  }
  return loom_print_attr(ctx->stream, attr, ctx->module, NULL);
}

static iree_status_t loom_print_pipeline_nested_region(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor);

static iree_status_t loom_print_pipeline_statement(loom_print_context_t* ctx,
                                                   const loom_op_t* op) {
  const loom_op_vtable_t* vtable = NULL;
  iree_string_view_t value = iree_string_view_empty();
  loom_named_attr_slice_t attrs = loom_make_named_attr_slice(NULL, 0);

  vtable = loom_print_pipeline_op_vtable(ctx, op, IREE_SV("pass.run"));
  if (vtable) {
    (void)loom_print_pipeline_string_attr(ctx, op, vtable, IREE_SV("key"),
                                          &value);
    IREE_RETURN_IF_ERROR(loom_output_stream_write(ctx->stream, value));
    (void)loom_print_pipeline_dict_attr(ctx, op, vtable, IREE_SV("options"),
                                        &attrs);
    return loom_print_pipeline_attr_parens(ctx, attrs);
  }

  vtable = loom_print_pipeline_op_vtable(ctx, op, IREE_SV("pass.for"));
  if (vtable) {
    const loom_region_descriptor_t* body_descriptor =
        loom_print_pipeline_body_region_descriptor(vtable);
    (void)loom_print_pipeline_enum_attr(op, vtable, IREE_SV("anchor"), &value);
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, "for "));
    IREE_RETURN_IF_ERROR(loom_output_stream_write(ctx->stream, value));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, ' '));
    return loom_print_pipeline_nested_region(ctx, loom_op_regions(op)[0],
                                             body_descriptor);
  }

  vtable = loom_print_pipeline_op_vtable(ctx, op, IREE_SV("pass.where"));
  if (vtable) {
    const loom_region_descriptor_t* body_descriptor =
        loom_print_pipeline_body_region_descriptor(vtable);
    (void)loom_print_pipeline_string_attr(ctx, op, vtable, IREE_SV("predicate"),
                                          &value);
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(ctx->stream, "where "));
    IREE_RETURN_IF_ERROR(loom_output_stream_write(ctx->stream, value));
    (void)loom_print_pipeline_dict_attr(ctx, op, vtable, IREE_SV("attrs"),
                                        &attrs);
    IREE_RETURN_IF_ERROR(loom_print_pipeline_attr_parens(ctx, attrs));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, ' '));
    return loom_print_pipeline_nested_region(ctx, loom_op_regions(op)[0],
                                             body_descriptor);
  }

  vtable = loom_print_pipeline_op_vtable(ctx, op, IREE_SV("pass.repeat"));
  if (vtable) {
    const loom_region_descriptor_t* body_descriptor =
        loom_print_pipeline_body_region_descriptor(vtable);
    const loom_attribute_t* count_attr = NULL;
    const loom_attribute_t* max_iterations_attr = NULL;
    (void)loom_print_pipeline_enum_attr(op, vtable, IREE_SV("mode"), &value);
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(ctx->stream, "repeat "));
    IREE_RETURN_IF_ERROR(loom_output_stream_write(ctx->stream, value));
    (void)loom_print_pipeline_optional_i64_attr(op, vtable, IREE_SV("count"),
                                                &count_attr);
    (void)loom_print_pipeline_optional_i64_attr(
        op, vtable, IREE_SV("max_iterations"), &max_iterations_attr);
    if (count_attr || max_iterations_attr) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '('));
      bool wrote_attr = false;
      if (count_attr) {
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write_cstring(ctx->stream, "count = "));
        IREE_RETURN_IF_ERROR(
            loom_print_attr(ctx->stream, count_attr, ctx->module, NULL));
        wrote_attr = true;
      }
      if (max_iterations_attr) {
        if (wrote_attr) {
          IREE_RETURN_IF_ERROR(
              loom_output_stream_write_cstring(ctx->stream, ", "));
        }
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write_cstring(ctx->stream, "max_iterations = "));
        IREE_RETURN_IF_ERROR(loom_print_attr(ctx->stream, max_iterations_attr,
                                             ctx->module, NULL));
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, ')'));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, ' '));
    return loom_print_pipeline_nested_region(ctx, loom_op_regions(op)[0],
                                             body_descriptor);
  }

  vtable = loom_print_pipeline_op_vtable(ctx, op, IREE_SV("pass.call"));
  if (vtable) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(ctx->stream, "call "));
    return loom_print_pipeline_symbol_attr_value(ctx, op, vtable,
                                                 IREE_SV("callee"));
  }

  vtable = loom_print_pipeline_op_vtable(ctx, op, IREE_SV("pass.fail"));
  if (vtable) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(ctx->stream, "fail "));
    return loom_print_pipeline_string_attr_value(ctx, op, vtable,
                                                 IREE_SV("message"));
  }

  vtable = loom_print_pipeline_op_vtable(ctx, op, IREE_SV("pass.halt"));
  if (!vtable) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported pass pipeline statement");
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, "halt "));
  return loom_print_pipeline_string_attr_value(ctx, op, vtable,
                                               IREE_SV("message"));
}

static iree_status_t loom_print_pipeline_nested_region(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, "{\n"));
  ++ctx->indent;
  const loom_block_t* block = loom_region_const_entry_block(region);
  const loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (op == block->last_op &&
        loom_print_pipeline_is_elidable_terminator(region_descriptor, op)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_print_op_comments(ctx, op));
    IREE_RETURN_IF_ERROR(loom_print_indent(ctx));
    IREE_RETURN_IF_ERROR(loom_print_pipeline_statement(ctx, op));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '\n'));
  }
  --ctx->indent;
  IREE_RETURN_IF_ERROR(loom_print_indent(ctx));
  return loom_output_stream_write_char(ctx->stream, '}');
}

static iree_status_t loom_print_pipeline_region(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor) {
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "pipeline", false));
  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  IREE_RETURN_IF_ERROR(
      loom_print_pipeline_nested_region(ctx, region, region_descriptor));
  ctx->has_previous_token = true;
  ctx->last_char = '}';
  ctx->glue_next = false;
  return iree_ok_status();
}

static iree_status_t loom_print_pipeline_skipped_region(
    loom_print_context_t* ctx) {
  IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "pipeline", false));
  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(ctx->stream, "{ ... }"));
  ctx->has_previous_token = true;
  ctx->last_char = '}';
  ctx->glue_next = false;
  return iree_ok_status();
}

static char loom_print_region_syntax_first_char(loom_print_context_t* ctx,
                                                loom_region_syntax_t syntax) {
  switch (syntax) {
    case LOOM_REGION_SYNTAX_DEFAULT:
      return '{';
    case LOOM_REGION_SYNTAX_TEST_DO:
      return 'd';
    case LOOM_REGION_SYNTAX_LOW_ASM:
      return 'a';
    case LOOM_REGION_SYNTAX_LOW_ASM_OPTIONAL:
      return loom_print_low_asm_is_requested(ctx) ? 'a' : '{';
    case LOOM_REGION_SYNTAX_PIPELINE:
      return 'p';
    default:
      return '?';
  }
}

static iree_status_t loom_print_region_body_with_syntax(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor,
    loom_region_syntax_t syntax, bool allow_entry_block_args) {
  switch (syntax) {
    case LOOM_REGION_SYNTAX_DEFAULT: {
      IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
      break;
    }
    case LOOM_REGION_SYNTAX_TEST_DO: {
      IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "do", false));
      IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
      break;
    }
    case LOOM_REGION_SYNTAX_LOW_ASM:
      return loom_print_low_asm_region(ctx, region, region_descriptor,
                                       allow_entry_block_args);
    case LOOM_REGION_SYNTAX_LOW_ASM_OPTIONAL:
      if (loom_print_low_asm_is_requested(ctx)) {
        return loom_print_low_asm_region(ctx, region, region_descriptor,
                                         allow_entry_block_args);
      }
      IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
      break;
    case LOOM_REGION_SYNTAX_PIPELINE: {
      if (iree_any_bit_set(ctx->flags, LOOM_TEXT_PRINT_SKIP_REGIONS)) {
        return loom_print_pipeline_skipped_region(ctx);
      }
      if (!allow_entry_block_args && loom_print_pipeline_region_is_friendly(
                                         ctx, region, region_descriptor)) {
        return loom_print_pipeline_region(ctx, region, region_descriptor);
      }
      IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported region syntax %u", (uint32_t)syntax);
  }

  if (iree_any_bit_set(ctx->flags, LOOM_TEXT_PRINT_SKIP_REGIONS)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(ctx->stream, "{ ... }"));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_print_braced_region(ctx, region, region_descriptor));
  }
  ctx->has_previous_token = true;
  ctx->last_char = '}';
  ctx->glue_next = false;
  return iree_ok_status();
}

static iree_status_t loom_print_region_table(
    loom_print_context_t* ctx, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_format_element_t* element) {
  uint8_t keys_attr_index =
      LOOM_FORMAT_REGION_TABLE_KEYS_ATTR_INDEX(element->data);
  uint8_t default_region_index =
      LOOM_FORMAT_REGION_TABLE_DEFAULT_REGION_INDEX(element->data);
  uint8_t case_region_index = (uint8_t)element->field_index;
  if (keys_attr_index >= op->attribute_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format REGION_TABLE attr index %u out of range (op has %u "
        "attributes)",
        keys_attr_index, op->attribute_count);
  }
  loom_attribute_t keys_attr = loom_op_attrs(op)[keys_attr_index];
  if (keys_attr.kind != LOOM_ATTR_I64_ARRAY) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format REGION_TABLE attr index %u expected I64_ARRAY attr but found "
        "%d",
        keys_attr_index, (int)keys_attr.kind);
  }
  if (keys_attr.count > 0 && !keys_attr.i64_array) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "REGION_TABLE keys attr has count %u but NULL values", keys_attr.count);
  }
  if (default_region_index >= op->region_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format REGION_TABLE default region index %u out of range (op has %u "
        "regions)",
        default_region_index, op->region_count);
  }
  if (case_region_index > op->region_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format REGION_TABLE case region index %u out of range (op has %u "
        "regions)",
        case_region_index, op->region_count);
  }
  iree_host_size_t expected_region_count =
      (iree_host_size_t)case_region_index + keys_attr.count;
  if (expected_region_count != op->region_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format REGION_TABLE has %u case keys but %u case regions",
        keys_attr.count, (uint8_t)(op->region_count - case_region_index));
  }

  const loom_region_descriptor_t* default_descriptor =
      loom_op_vtable_region_descriptor(vtable, default_region_index);
  const loom_region_descriptor_t* case_descriptor =
      loom_op_vtable_region_descriptor(vtable, case_region_index);
  if (!default_descriptor || !case_descriptor) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format REGION_TABLE region descriptors are unavailable");
  }

  IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
  iree_host_size_t table_start = ctx->stream->offset;
  if (iree_any_bit_set(ctx->flags, LOOM_TEXT_PRINT_SKIP_REGIONS)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(ctx->stream, "{ ... }"));
    loom_print_did_write(ctx);
    loom_print_report_field(
        ctx, loom_print_field_ref(LOOM_PRINT_FIELD_ATTR, keys_attr_index),
        table_start, ctx->stream->offset);
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, "{\n"));
  ++ctx->indent;
  loom_region_t** regions = loom_op_regions(op);
  for (uint16_t row = 0; row < keys_attr.count; ++row) {
    uint8_t region_index = (uint8_t)(case_region_index + row);
    IREE_RETURN_IF_ERROR(loom_print_indent(ctx));
    iree_host_size_t region_start = ctx->stream->offset;
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        ctx->stream, "case %" PRId64 " ", keys_attr.i64_array[row]));
    IREE_RETURN_IF_ERROR(
        loom_print_braced_region(ctx, regions[region_index], case_descriptor));
    loom_print_report_field(
        ctx, loom_print_field_ref(LOOM_PRINT_FIELD_REGION, region_index),
        region_start, ctx->stream->offset);
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '\n'));
  }

  IREE_RETURN_IF_ERROR(loom_print_indent(ctx));
  iree_host_size_t default_start = ctx->stream->offset;
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(ctx->stream, "default "));
  IREE_RETURN_IF_ERROR(loom_print_braced_region(
      ctx, regions[default_region_index], default_descriptor));
  loom_print_report_field(
      ctx, loom_print_field_ref(LOOM_PRINT_FIELD_REGION, default_region_index),
      default_start, ctx->stream->offset);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '\n'));
  --ctx->indent;
  IREE_RETURN_IF_ERROR(loom_print_indent(ctx));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '}'));
  ctx->has_previous_token = true;
  ctx->last_char = '}';
  ctx->glue_next = false;
  loom_print_report_field(
      ctx, loom_print_field_ref(LOOM_PRINT_FIELD_ATTR, keys_attr_index),
      table_start, ctx->stream->offset);
  return iree_ok_status();
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
      case LOOM_FORMAT_KIND_SUCCESSOR_REF: {
        IREE_RETURN_IF_ERROR(
            loom_print_successor_ref(ctx, op, element->field_index));
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
        loom_bstring_t keyword_bstring =
            loom_keyword_bstring((loom_keyword_id_t)element->data);
        IREE_RETURN_IF_ERROR(
            loom_print_emit(ctx, loom_bstring_view(keyword_bstring), false));
        break;
      }
      case LOOM_FORMAT_KIND_REGION: {
        if (element->field_index >= op->region_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format REGION field_index %u out of range (op has %u regions)",
              element->field_index, op->region_count);
        }
        const loom_region_descriptor_t* region_descriptor =
            loom_op_vtable_region_descriptor(vtable, element->field_index);
        if (!region_descriptor) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format REGION field_index %u out of range (vtable has %u "
              "region descriptors)",
              element->field_index, vtable->region_count);
        }
        loom_region_t* region = loom_op_regions(op)[element->field_index];
        loom_region_syntax_t syntax = (loom_region_syntax_t)element->data;
        const bool region_args_declared_by_parent =
            vtable->func_like &&
            vtable->func_like->body_region_index == element->field_index;
        iree_host_size_t region_start = loom_print_next_token_start_offset(
            ctx, /*glue=*/false,
            loom_print_region_syntax_first_char(ctx, syntax));
        IREE_RETURN_IF_ERROR(loom_print_region_body_with_syntax(
            ctx, region, region_descriptor, syntax,
            region_args_declared_by_parent));
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
      case LOOM_FORMAT_KIND_BLOCK_ARGS: {
        uint16_t arg_count = 0;
        const loom_value_id_t* arg_ids = loom_print_region_entry_arg_ids(
            op, element->field_index, &arg_count);
        IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "(", true));
        for (uint16_t j = 0; j < arg_count; ++j) {
          if (j > 0) {
            IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ",", false));
          }
          IREE_RETURN_IF_ERROR(loom_print_value_name(ctx, arg_ids[j]));
          IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ":", true));
          IREE_RETURN_IF_ERROR(loom_print_space_if_needed(ctx));
          IREE_RETURN_IF_ERROR(loom_print_value_type(ctx, arg_ids[j]));
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
        if (iree_any_bit_set(element->data,
                             LOOM_ATTR_DICT_FORMAT_INLINE_ATTRS)) {
          IREE_RETURN_IF_ERROR(
              loom_print_inline_attr_dict(ctx, op, vtable, element));
          break;
        }
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
      case LOOM_FORMAT_KIND_OPERAND_DICT: {
        IREE_RETURN_IF_ERROR(loom_print_operand_dict(ctx, op, element));
        break;
      }
      case LOOM_FORMAT_KIND_ATTR_TABLE: {
        IREE_RETURN_IF_ERROR(loom_print_attr_table(ctx, op, element));
        break;
      }
      case LOOM_FORMAT_KIND_REGION_TABLE: {
        IREE_RETURN_IF_ERROR(loom_print_region_table(ctx, op, vtable, element));
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
      case LOOM_FORMAT_KIND_DESCRIPTOR_REF: {
        // Descriptor key reference in angle brackets, glued to the op name:
        // low.op<amdgpu.v_add_u32>. The field_index references the diagnostic
        // key spelling and data references the derived stable ID.
        if (element->field_index >= op->attribute_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format DESCRIPTOR_REF field_index %u out of range (op has %u "
              "attributes)",
              element->field_index, op->attribute_count);
        }
        if (element->data >= op->attribute_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "format DESCRIPTOR_REF stable-id field_index %u out of range "
              "(op has %u attributes)",
              element->data, op->attribute_count);
        }
        loom_attribute_t attr = loom_op_attrs(op)[element->field_index];
        if (attr.kind == LOOM_ATTR_STRING &&
            attr.string_id != LOOM_STRING_ID_INVALID &&
            attr.string_id < ctx->module->strings.count) {
          iree_string_view_t key = ctx->module->strings.entries[attr.string_id];
          iree_host_size_t ref_start = ctx->stream->offset;
          IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, "<", true));
          IREE_RETURN_IF_ERROR(loom_print_emit(ctx, key, true));
          IREE_RETURN_IF_ERROR(loom_print_emit_cstr(ctx, ">", true));
          loom_print_did_write(ctx);
          loom_print_report_field(
              ctx,
              loom_print_field_ref(LOOM_PRINT_FIELD_ATTR, element->field_index),
              ref_start, ctx->stream->offset);
          loom_print_report_field(
              ctx, loom_print_field_ref(LOOM_PRINT_FIELD_ATTR, element->data),
              ref_start, ctx->stream->offset);
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

static iree_status_t loom_print_block_label_line(loom_print_context_t* ctx,
                                                 const loom_region_t* region,
                                                 const loom_block_t* block) {
  uint16_t label_indent = ctx->indent > 0 ? (uint16_t)(ctx->indent - 1) : 0;
  char label_buffer[64];
  iree_string_view_t label = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_print_block_label_view(
      ctx, region, block, label_buffer, sizeof(label_buffer), &label));
  IREE_RETURN_IF_ERROR(loom_print_block_comments(ctx, block, label_indent));
  IREE_RETURN_IF_ERROR(loom_print_indent_levels(ctx, label_indent));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '^'));
  IREE_RETURN_IF_ERROR(loom_output_stream_write(ctx->stream, label));
  if (block->arg_count > 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '('));
    for (uint16_t arg_index = 0; arg_index < block->arg_count; ++arg_index) {
      if (arg_index > 0) {
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write_cstring(ctx->stream, ", "));
      }
      char name_buffer[LOOM_VALUE_NAME_BUFFER_SIZE];
      loom_value_id_t arg_id = loom_block_arg_id(block, arg_index);
      IREE_RETURN_IF_ERROR(loom_output_stream_write(
          ctx->stream, loom_resolve_value_name(ctx->module, arg_id, name_buffer,
                                               sizeof(name_buffer))));
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(ctx->stream, " : "));
      IREE_RETURN_IF_ERROR(
          loom_text_print_type(loom_module_value_type(ctx->module, arg_id),
                               ctx->module, ctx->stream));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, ')'));
  }
  return loom_output_stream_write_cstring(ctx->stream, ":\n");
}

static iree_status_t loom_print_region_body(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor) {
  IREE_ASSERT_ARGUMENT(region_descriptor);
  const bool needs_synthetic_labels =
      loom_print_region_needs_synthetic_labels(ctx, region);
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(region, block_index);

    // Print block label if present or needed by a successor reference.
    if (loom_print_block_has_label(ctx, block) || needs_synthetic_labels) {
      IREE_RETURN_IF_ERROR(loom_print_block_label_line(ctx, region, block));
    }

    bool printed_any = false;
    const loom_op_t* last_live_op = block->last_op;
    const loom_op_t* current_op = NULL;
    loom_block_for_each_op(block, current_op) {
      if (current_op == last_live_op &&
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
      IREE_RETURN_IF_ERROR(loom_print_op_comments(ctx, current_op));
      IREE_RETURN_IF_ERROR(loom_print_indent(ctx));
      IREE_RETURN_IF_ERROR(loom_print_op(ctx, current_op));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_print_module_body(loom_print_context_t* ctx,
                                            const loom_region_t* region) {
  const bool needs_synthetic_labels =
      loom_print_region_needs_synthetic_labels(ctx, region);
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(region, block_index);

    if (loom_print_block_has_label(ctx, block) || needs_synthetic_labels) {
      IREE_RETURN_IF_ERROR(loom_print_block_label_line(ctx, region, block));
    }

    bool printed_any = false;
    const loom_op_t* current_op = NULL;
    loom_block_for_each_op(block, current_op) {
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
      IREE_RETURN_IF_ERROR(loom_print_op_comments(ctx, current_op));
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

static loom_print_context_t loom_print_context_make(
    const loom_module_t* module, loom_output_stream_t* stream,
    const loom_text_print_options_t* options) {
  loom_text_print_flags_t flags =
      options ? options->flags : LOOM_TEXT_PRINT_DEFAULT;
  return (loom_print_context_t){
      .stream = stream,
      .module = module,
      .flags = flags,
      .low_asm_environment = options ? options->low_asm_environment
                                     : (loom_text_low_asm_environment_t){0},
      .low_asm_descriptor_set_key = options
                                        ? options->low_asm_descriptor_set_key
                                        : iree_string_view_empty(),
  };
}

iree_status_t loom_text_print_module(const loom_module_t* module,
                                     loom_output_stream_t* stream,
                                     loom_text_print_flags_t flags) {
  loom_text_print_options_t options = {
      .flags = flags,
  };
  return loom_text_print_module_with_options(module, stream, &options);
}

iree_status_t loom_text_print_module_with_options(
    const loom_module_t* module, loom_output_stream_t* stream,
    const loom_text_print_options_t* options) {
  if (!module || !module->body) return iree_ok_status();
  loom_print_context_t ctx = loom_print_context_make(module, stream, options);
  IREE_RETURN_IF_ERROR(loom_print_encoding_aliases(&ctx, module));
  return loom_print_module_body(&ctx, module->body);
}

iree_status_t loom_text_print_operation(const loom_module_t* module,
                                        const loom_op_t* op,
                                        loom_output_stream_t* stream,
                                        loom_text_print_flags_t flags) {
  loom_text_print_options_t options = {
      .flags = flags,
  };
  return loom_text_print_operation_with_options(module, op, stream, &options);
}

iree_status_t loom_text_print_operation_with_options(
    const loom_module_t* module, const loom_op_t* op,
    loom_output_stream_t* stream, const loom_text_print_options_t* options) {
  if (!module || !op) return iree_ok_status();
  loom_print_context_t ctx = loom_print_context_make(module, stream, options);
  IREE_RETURN_IF_ERROR(loom_print_op_comments(&ctx, op));
  return loom_print_op(&ctx, op);
}

iree_status_t loom_text_print_module_to_builder(const loom_module_t* module,
                                                iree_string_builder_t* builder,
                                                loom_text_print_flags_t flags) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  return loom_text_print_module(module, &stream, flags);
}

iree_status_t loom_text_print_module_to_builder_with_options(
    const loom_module_t* module, iree_string_builder_t* builder,
    const loom_text_print_options_t* options) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  return loom_text_print_module_with_options(module, &stream, options);
}

iree_status_t loom_text_print_operation_to_builder(
    const loom_module_t* module, const loom_op_t* op,
    iree_string_builder_t* builder, loom_text_print_flags_t flags) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  return loom_text_print_operation(module, op, &stream, flags);
}

iree_status_t loom_text_print_operation_to_builder_with_options(
    const loom_module_t* module, const loom_op_t* op,
    iree_string_builder_t* builder, const loom_text_print_options_t* options) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  return loom_text_print_operation_with_options(module, op, &stream, options);
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
  IREE_RETURN_IF_ERROR(loom_print_op_comments(&ctx, op));
  return loom_print_op(&ctx, op);
}
