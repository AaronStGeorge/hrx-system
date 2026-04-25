// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Internal types and declarations shared across the printer implementation.
// Not a public API; do not include from outside this library.

#ifndef LOOM_FORMAT_TEXT_PRINTER_INTERNAL_H_
#define LOOM_FORMAT_TEXT_PRINTER_INTERNAL_H_

#include "iree/base/api.h"
#include "loom/format/text/printer.h"
#include "loom/ir/ir.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

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

// Emits text through the token spacing model.
iree_status_t loom_print_emit(loom_print_context_t* ctx,
                              iree_string_view_t text, bool glue);

// Emits a C string through the token spacing model.
iree_status_t loom_print_emit_cstr(loom_print_context_t* ctx, const char* text,
                                   bool glue);

// Emits a space if the token spacing model requires one before direct output.
iree_status_t loom_print_space_if_needed(loom_print_context_t* ctx);

// Updates token spacing state after direct output.
void loom_print_did_write(loom_print_context_t* ctx);

// Reports a semantic field byte range to the optional field callback.
void loom_print_report_field(loom_print_context_t* ctx,
                             loom_print_field_ref_t field_ref,
                             iree_host_size_t start, iree_host_size_t end);

// Writes indentation for the current logical line.
iree_status_t loom_print_indent(loom_print_context_t* ctx);

// Prints comments attached to |op| at the current indentation.
iree_status_t loom_print_op_comments(loom_print_context_t* ctx,
                                     const loom_op_t* op);

// Prints a canonical attribute payload.
iree_status_t loom_print_attr(loom_output_stream_t* stream,
                              const loom_attribute_t* attr,
                              const loom_module_t* module,
                              const loom_attr_descriptor_t* descriptor);

// Prints the canonical type of |value_id|, or <unknown> for malformed IR.
iree_status_t loom_print_value_type(loom_print_context_t* ctx,
                                    loom_value_id_t value_id);

// Prints the canonical result type of |value_id|, or <unknown> for malformed
// IR.
iree_status_t loom_print_result_value_type(loom_print_context_t* ctx,
                                           loom_value_id_t value_id);

// Prints a value name and reports its emitted field range.
iree_status_t loom_print_value_name_with_field(
    loom_print_context_t* ctx, loom_value_id_t value_id,
    loom_print_field_ref_t field_ref);

// Prints a source location annotation.
iree_status_t loom_print_location(loom_output_stream_t* stream,
                                  const loom_module_t* module,
                                  loom_location_id_t location_id);

// Returns true when low asm region syntax was requested by print options.
bool loom_print_low_asm_is_requested(loom_print_context_t* ctx);

// Prints a region using mandatory low asm syntax.
iree_status_t loom_print_low_asm_region(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor,
    bool allow_entry_block_args);

// Attempts to print a region using optional low asm syntax. Sets
// |out_printed| to false when the region has no lossless low asm spelling.
iree_status_t loom_print_low_asm_optional_region(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor,
    bool allow_entry_block_args, bool* out_printed);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_PRINTER_INTERNAL_H_
