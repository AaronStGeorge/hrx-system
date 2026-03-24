// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/diagnostic.h"

#include <stdio.h>

#include "loom/error/renderer.h"

//===----------------------------------------------------------------------===//
// Source line extraction
//===----------------------------------------------------------------------===//

// Finds the start of the line containing byte offset |position|.
static iree_host_size_t loom_find_line_start(iree_string_view_t source,
                                             iree_host_size_t position) {
  if (position == 0) return 0;
  iree_host_size_t i = position;
  while (i > 0 && source.data[i - 1] != '\n') {
    --i;
  }
  return i;
}

// Finds the end of the line containing byte offset |position|.
// Returns the offset of the newline (or source.size if at EOF).
static iree_host_size_t loom_find_line_end(iree_string_view_t source,
                                           iree_host_size_t position) {
  iree_host_size_t i = position;
  while (i < source.size && source.data[i] != '\n') {
    ++i;
  }
  return i;
}

//===----------------------------------------------------------------------===//
// Formatting
//===----------------------------------------------------------------------===//

iree_status_t loom_diagnostic_format(const loom_diagnostic_t* diagnostic,
                                     loom_output_stream_t* stream) {
  const loom_source_range_t* range = &diagnostic->origin;
  const char* severity_string =
      loom_diagnostic_severity_name(diagnostic->severity);
  bool has_source = range->source.size > 0 && range->start < range->source.size;
  loom_type_formatter_t type_formatter = {loom_type_format_minimal, NULL};

  // Clang-style first line:
  //   file:line:col: severity: DOMAIN/CODE: message
  //   file:line:col: severity: message          (unstructured)
  //   severity: message                          (no source)
  //   severity: DOMAIN/CODE: message             (no source, structured)
  if (has_source && range->filename.size > 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write(stream, range->filename));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, ":%" PRIu32 ":%" PRIu32 ": ", range->start_line,
        range->start_column));
  }

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, severity_string));

  // Error code.
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, " [%s/%03u]", loom_error_domain_name(diagnostic->error->domain),
      diagnostic->error->code));

  // Render the message from the error def's template and params.
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ": "));
  IREE_RETURN_IF_ERROR(loom_diagnostic_render_message(
      diagnostic->error, diagnostic->params, diagnostic->param_count,
      type_formatter, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\n"));

  // If we have no source, emit fix hint (if any) and stop.
  if (!has_source) {
    if (diagnostic->error && diagnostic->error->fix_hint_template) {
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, "  = help: "));
      IREE_RETURN_IF_ERROR(loom_diagnostic_render_fix_hint(
          diagnostic->error, diagnostic->params, diagnostic->param_count,
          type_formatter, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\n"));
    }
    return iree_ok_status();
  }

  // Extract the source line containing the diagnostic start.
  iree_host_size_t line_start =
      loom_find_line_start(range->source, range->start);
  iree_host_size_t line_end = loom_find_line_end(range->source, range->start);
  iree_string_view_t source_line = iree_make_string_view(
      range->source.data + line_start, line_end - line_start);

  // Compute the width of the line number for alignment.
  char line_number_buffer[16];
  int line_number_width =
      snprintf(line_number_buffer, sizeof(line_number_buffer), "%" PRIu32,
               range->start_line);

  // Source line with line number. Decomposed into separate writes because
  // the source line is already a string_view — no reason to format it.
  //   42 |   %r = test.addi %x, %y : i32
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " "));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, line_number_buffer));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " | "));
  IREE_RETURN_IF_ERROR(loom_output_stream_write(stream, source_line));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\n"));

  // Caret/underline line. Padding: " " + spaces(line_number_width) + " | ".
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " "));
  for (int i = 0; i < line_number_width; ++i) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ' '));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " | "));

  if (diagnostic->highlight_count > 0 && diagnostic->highlights) {
    // Multi-caret: draw carets at each highlighted range, spaces elsewhere.
    // Walk the source line column by column, checking if any highlight
    // covers this position.
    for (iree_host_size_t i = 0; i < source_line.size; ++i) {
      iree_host_size_t source_offset = line_start + i;
      bool highlighted = false;
      for (iree_host_size_t h = 0; h < diagnostic->highlight_count; ++h) {
        if (source_offset >= diagnostic->highlights[h].start &&
            source_offset < diagnostic->highlights[h].end) {
          highlighted = true;
          break;
        }
      }
      if (highlighted) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '^'));
      } else {
        // Preserve tabs for alignment.
        char c = (source_line.data[i] == '\t') ? '\t' : ' ';
        IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, c));
      }
    }
  } else {
    // Single range: spaces to reach the start column, then carets.
    iree_host_size_t column_offset = range->start - line_start;
    for (iree_host_size_t i = 0; i < column_offset; ++i) {
      char c =
          (i < source_line.size && source_line.data[i] == '\t') ? '\t' : ' ';
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, c));
    }
    iree_host_size_t underline_length = 1;
    if (range->end > range->start) {
      underline_length = range->end - range->start;
      if (range->start + underline_length > line_end) {
        underline_length = line_end - range->start;
      }
    }
    if (underline_length == 0) underline_length = 1;
    for (iree_host_size_t i = 0; i < underline_length; ++i) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '^'));
    }
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\n"));

  // Fix hint line (when structured diagnostic has one).
  if (diagnostic->error && diagnostic->error->fix_hint_template) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " "));
    for (int i = 0; i < line_number_width; ++i) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ' '));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " = help: "));
    IREE_RETURN_IF_ERROR(loom_diagnostic_render_fix_hint(
        diagnostic->error, diagnostic->params, diagnostic->param_count,
        type_formatter, stream));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\n"));
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Stderr sink
//===----------------------------------------------------------------------===//

iree_status_t loom_diagnostic_stderr_sink(void* user_data,
                                          const loom_diagnostic_t* diagnostic) {
  (void)user_data;
  loom_output_stream_t stream;
  loom_output_stream_for_file(stderr, &stream);
  return loom_diagnostic_format(diagnostic, &stream);
}
