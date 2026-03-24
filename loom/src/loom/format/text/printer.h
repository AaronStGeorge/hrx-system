// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Table-driven text printer for loom IR.
//
// Walks the format element arrays in each op's vtable (.rodata) to emit
// canonical textual IR. One generic function handles all ops — no per-op
// code. The printer is read-only over the IR (unless location capture
// is enabled, which updates op locations to point at the output).
//
// All output goes through a loom_output_stream_t — zero heap
// allocations in the print path.

#ifndef LOOM_FORMAT_TEXT_PRINTER_H_
#define LOOM_FORMAT_TEXT_PRINTER_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// Flags controlling printer behavior.
enum loom_text_print_flag_bits_e {
  // Update each op's location to point at its byte range in the output.
  LOOM_TEXT_PRINT_CAPTURE_LOCATIONS = 1u << 0,
  // Use encoding aliases (#enc) instead of inlining (#q8_0<block=32>).
  LOOM_TEXT_PRINT_USE_ALIASES = 1u << 1,
  // Pretty-print with 2-space indentation (default on).
  LOOM_TEXT_PRINT_INDENT = 1u << 2,
  // Omit region bodies. Regions print as empty braces: { ... }. Useful
  // for printing function/module declarations without their contents.
  LOOM_TEXT_PRINT_SKIP_REGIONS = 1u << 3,
  // Emit trailing loc() annotations on ops.
  LOOM_TEXT_PRINT_LOCATIONS = 1u << 4,
};
typedef uint32_t loom_text_print_flags_t;

// Default flags for canonical output.
#define LOOM_TEXT_PRINT_DEFAULT \
  (LOOM_TEXT_PRINT_INDENT | LOOM_TEXT_PRINT_USE_ALIASES)

// Prints a complete module to canonical text via the output stream.
iree_status_t loom_text_print_module(const loom_module_t* module,
                                     loom_output_stream_t* stream,
                                     loom_text_print_flags_t flags);

// Prints a single operation via the output stream.
iree_status_t loom_text_print_operation(const loom_module_t* module,
                                        const loom_op_t* op,
                                        loom_output_stream_t* stream,
                                        loom_text_print_flags_t flags);

// Prints a type in canonical form to the output stream.
iree_status_t loom_text_print_type(loom_type_t type,
                                   const loom_module_t* module,
                                   loom_output_stream_t* stream);

// Convenience: print module to an iree_string_builder_t.
iree_status_t loom_text_print_module_to_builder(const loom_module_t* module,
                                                iree_string_builder_t* builder,
                                                loom_text_print_flags_t flags);

// Convenience: print single op to an iree_string_builder_t.
iree_status_t loom_text_print_operation_to_builder(
    const loom_module_t* module, const loom_op_t* op,
    iree_string_builder_t* builder, loom_text_print_flags_t flags);

//===----------------------------------------------------------------------===//
// Field emission callback
//===----------------------------------------------------------------------===//

// Called by the printer for each field it emits in the output.
// Fires for operand %names, result %names, attribute values — every
// semantically meaningful token that corresponds to an op field.
// |field_ref| identifies the field via LOOM_FIELD_REF(category, index).
// |start| and |end| are byte offsets in the output stream.
typedef void (*loom_print_field_fn_t)(void* user_data,
                                      loom_field_ref_t field_ref,
                                      iree_host_size_t start,
                                      iree_host_size_t end);

// Bundles a field emission callback with its context.
typedef struct loom_print_field_callback_t {
  loom_print_field_fn_t fn;
  void* user_data;
} loom_print_field_callback_t;

// Prints a single operation to a builder, invoking |callback.fn| for
// each field emitted. Use the callback to record byte ranges of
// specific fields for diagnostic highlighting.
iree_status_t loom_text_print_operation_with_field_callback(
    const loom_module_t* module, const loom_op_t* op,
    iree_string_builder_t* builder, loom_text_print_flags_t flags,
    loom_print_field_callback_t callback);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_FORMAT_TEXT_PRINTER_H_
