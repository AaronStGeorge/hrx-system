// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/printer.h"

#include <inttypes.h>
#include <stdio.h>

#include "loom/format/text/printer_internal.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

static bool loom_is_backward_glue(char c) {
  return c == ',' || c == ')' || c == ']' || c == '}';
}

static bool loom_is_forward_glue(char c) {
  return c == '(' || c == '[' || c == '{';
}

// Emits text with automatic spacing. Applies backward/forward glue
// rules and the explicit glue flag.
iree_status_t loom_print_emit(loom_print_context_t* ctx,
                              iree_string_view_t text, bool glue) {
  if (text.size == 0) {
    return iree_ok_status();
  }
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

iree_status_t loom_print_emit_cstr(loom_print_context_t* ctx, const char* text,
                                   bool glue) {
  return loom_print_emit(ctx, iree_make_cstring_view(text), glue);
}

void loom_print_set_glue(loom_print_context_t* ctx) { ctx->glue_next = true; }

// Emits a space if the spacing rules require one before the next token.
// Call before writing directly to the stream for content that is not
// backward-glue punctuation (types, attributes, integers; never start
// with ,)]}). After writing, call loom_print_did_write to update state.
iree_status_t loom_print_space_if_needed(loom_print_context_t* ctx) {
  bool suppress = ctx->glue_next || !ctx->has_previous_token ||
                  loom_is_forward_glue(ctx->last_char);
  ctx->glue_next = false;
  if (!suppress) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, ' '));
  }
  return iree_ok_status();
}

// Updates spacing state after writing content directly to the stream.
void loom_print_did_write(loom_print_context_t* ctx) {
  ctx->has_previous_token = true;
  ctx->last_char = ' ';
}

iree_host_size_t loom_print_next_token_start_offset(
    const loom_print_context_t* ctx, bool glue, char first_char) {
  bool suppress_space = glue || ctx->glue_next || !ctx->has_previous_token ||
                        loom_is_backward_glue(first_char) ||
                        loom_is_forward_glue(ctx->last_char);
  return ctx->stream->offset + (suppress_space ? 0 : 1);
}

void loom_print_report_field(loom_print_context_t* ctx,
                             loom_print_field_ref_t field_ref,
                             iree_host_size_t start, iree_host_size_t end) {
  if (!ctx->field_callback.fn) {
    return;
  }
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

iree_status_t loom_print_indent(loom_print_context_t* ctx) {
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

iree_status_t loom_print_op_comments(loom_print_context_t* ctx,
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

//===----------------------------------------------------------------------===//
// Block and successor labels.
//===----------------------------------------------------------------------===//

bool loom_print_block_has_label(const loom_print_context_t* ctx,
                                const loom_block_t* block) {
  return block->label_id != LOOM_STRING_ID_INVALID &&
         block->label_id < ctx->module->strings.count;
}

static bool loom_print_region_label_exists(const loom_print_context_t* ctx,
                                           const loom_region_t* region,
                                           iree_string_view_t label) {
  if (!region) {
    return false;
  }
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(region, block_index);
    if (!loom_print_block_has_label(ctx, block)) {
      continue;
    }
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
  if (!region || !block) {
    return false;
  }
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
  if (!region) {
    return false;
  }
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(region, block_index);
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      loom_block_t* const* successors = loom_op_const_successors(op);
      for (uint8_t i = 0; i < op->successor_count; ++i) {
        const loom_block_t* target = successors[i];
        if (target && !loom_print_block_has_label(ctx, target)) {
          return true;
        }
      }
    }
  }
  return false;
}

iree_status_t loom_print_successor_ref(loom_print_context_t* ctx,
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
      if (j > 0) {
        status = loom_print_emit_cstr(ctx, ",", false);
      }
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
    status = loom_print_format_elements(ctx, op, vtable);
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
          ctx->stream,
          loom_print_resolve_value_name(ctx->module, arg_id, name_buffer,
                                        sizeof(name_buffer))));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, ": "));
      IREE_RETURN_IF_ERROR(
          loom_text_print_type(loom_module_value_type(ctx->module, arg_id),
                               ctx->module, ctx->stream));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, ')'));
  }
  return loom_output_stream_write_cstring(ctx->stream, ":\n");
}

iree_status_t loom_print_region_body(
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
  if (!module || !module->body) {
    return iree_ok_status();
  }
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
  if (!module || !op) {
    return iree_ok_status();
  }
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
