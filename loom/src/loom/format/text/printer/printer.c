// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/printer/printer.h"

#include "loom/format/text/printer/atoms.h"
#include "loom/format/text/printer/context.h"
#include "loom/format/text/printer/regions.h"
#include "loom/ir/module.h"
#include "loom/ir/module_record.h"

static loom_print_context_t loom_print_context_make(
    loom_text_printer_t* printer, loom_output_stream_t* stream,
    const loom_text_print_options_t* options) {
  loom_print_context_t context = {0};
  context.stream = stream;
  context.module = printer->module;
  context.printer = printer;
  context.flags = options ? options->flags : LOOM_TEXT_PRINT_DEFAULT;
  if (options) {
    context.low_asm_environment = options->low_asm_environment;
  }
  return context;
}

void loom_text_printer_initialize(const loom_module_t* module,
                                  loom_text_printer_t* out_printer) {
  *out_printer = (loom_text_printer_t){.module = module};
}

void loom_text_printer_deinitialize(loom_text_printer_t* printer) {
  loom_print_name_plan_discard(printer);
  printer->module = NULL;
}

static bool loom_module_has_printable_content(const loom_module_t* module) {
  for (uint16_t i = 0; i < module->encodings.count; ++i) {
    if (module->encodings.entries[i].alias_id != LOOM_STRING_ID_INVALID) {
      return true;
    }
  }
  if (!module->body) {
    return false;
  }
  for (uint16_t i = 0; i < module->body->block_count; ++i) {
    const loom_block_t* block = loom_region_const_block(module->body, i);
    if (block->label_id != LOOM_STRING_ID_INVALID || block->first_op) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_print_file_header(const loom_module_t* module,
                                            loom_output_stream_t* stream) {
  iree_host_size_t line_count = 0;
  const iree_string_view_t* lines =
      loom_module_file_header(module, &line_count);
  for (iree_host_size_t i = 0; i < line_count; ++i) {
    iree_string_view_t line = lines[i];
    bool paragraph_break = iree_string_view_ends_with(line, IREE_SV("\n"));
    if (paragraph_break) {
      line = iree_string_view_remove_suffix(line, 1);
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "//"));
    if (!iree_string_view_is_empty(line)) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ' '));
      IREE_RETURN_IF_ERROR(loom_output_stream_write(stream, line));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '\n'));
    if (paragraph_break) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '\n'));
    }
  }
  return iree_ok_status();
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
  loom_text_printer_t printer;
  loom_text_printer_initialize(module, &printer);
  IREE_RETURN_IF_ERROR(loom_print_name_plan_prepare(&printer));
  loom_module_record_plan_t module_record_plan;
  iree_status_t status =
      loom_module_record_plan_initialize(module, &module_record_plan);
  if (!iree_status_is_ok(status)) {
    loom_text_printer_deinitialize(&printer);
    return status;
  }
  loom_print_context_t ctx = loom_print_context_make(&printer, stream, options);
  iree_host_size_t file_header_line_count = 0;
  (void)loom_module_file_header(module, &file_header_line_count);
  status = loom_print_file_header(module, stream);
  if (iree_status_is_ok(status) && file_header_line_count > 0 &&
      loom_module_has_printable_content(module)) {
    status = loom_output_stream_write_char(stream, '\n');
  }
  if (iree_status_is_ok(status)) {
    status = loom_print_encoding_aliases(&ctx, module);
  }
  if (iree_status_is_ok(status)) {
    status = loom_print_module_body(&ctx, module->body, &module_record_plan);
  }
  loom_module_record_plan_deinitialize(&module_record_plan);
  loom_text_printer_deinitialize(&printer);
  return status;
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
  loom_text_printer_t printer;
  loom_text_printer_initialize(module, &printer);
  iree_status_t status =
      loom_text_printer_write_operation(&printer, op, stream, options);
  loom_text_printer_deinitialize(&printer);
  return status;
}

iree_status_t loom_text_printer_write_operation(
    loom_text_printer_t* printer, const loom_op_t* op,
    loom_output_stream_t* stream, const loom_text_print_options_t* options) {
  if (!printer->module || !op) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_print_name_plan_prepare(printer));
  loom_print_context_t ctx = loom_print_context_make(printer, stream, options);
  iree_status_t status = loom_print_op_comments(&ctx, op);
  if (iree_status_is_ok(status)) {
    status = loom_print_op(&ctx, op, NULL);
  }
  return status;
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
  loom_text_printer_t printer;
  loom_text_printer_initialize(module, &printer);
  IREE_RETURN_IF_ERROR(loom_print_name_plan_prepare(&printer));
  loom_print_context_t ctx = {
      .stream = &stream,
      .module = module,
      .printer = &printer,
      .flags = flags,
      .field_callback = callback,
  };
  iree_status_t status = loom_print_op_comments(&ctx, op);
  if (iree_status_is_ok(status)) {
    status = loom_print_op(&ctx, op, NULL);
  }
  loom_text_printer_deinitialize(&printer);
  return status;
}
