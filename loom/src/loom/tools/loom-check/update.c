// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/update.h"

#include <string.h>

// Returns true if |text| ends with two consecutive newlines (\n\n),
// indicating an existing blank line at the end.
static bool loom_check_ends_with_blank_line(iree_string_view_t text) {
  return text.size >= 2 && text.data[text.size - 1] == '\n' &&
         text.data[text.size - 2] == '\n';
}

// Appends a blank line (\n) to |builder| if the content written so far
// does not already end with one. This ensures vertical whitespace around
// structural separators (// ====, // ----) in the reconstructed output.
static iree_status_t loom_check_ensure_blank_line(
    iree_string_builder_t* builder) {
  iree_string_view_t written = {
      .data = iree_string_builder_buffer(builder),
      .size = iree_string_builder_size(builder),
  };
  if (written.size > 0 && !loom_check_ends_with_blank_line(written)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  }
  return iree_ok_status();
}

// Ensures |text| has a trailing newline. Appends one if missing.
static iree_status_t loom_check_append_with_trailing_newline(
    iree_string_builder_t* builder, iree_string_view_t text) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(builder, text));
  if (text.size > 0 && text.data[text.size - 1] != '\n') {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  }
  return iree_ok_status();
}

iree_status_t loom_check_apply_updates(iree_string_view_t original_source,
                                       const loom_check_file_t* file,
                                       const loom_check_case_update_t* updates,
                                       iree_string_builder_t* new_source,
                                       iree_host_size_t* out_update_count) {
  const char* cursor = original_source.data;
  iree_host_size_t update_count = 0;

  for (iree_host_size_t i = 0; i < file->case_count; ++i) {
    if (!updates[i].needs_update) continue;
    ++update_count;

    const loom_check_case_t* test_case = &file->cases[i];

    if (test_case->has_expected_section) {
      // Emit everything from cursor to the start of the expected section.
      iree_host_size_t prefix_length =
          (iree_host_size_t)(updates[i].expected_start - cursor);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
          new_source, iree_make_string_view(cursor, prefix_length)));
      // Emit the actual output as the new expected section.
      IREE_RETURN_IF_ERROR(loom_check_append_with_trailing_newline(
          new_source, updates[i].actual_output));
      cursor = updates[i].expected_end;
    } else {
      // No expected section — insert separator + actual output after input.
      iree_host_size_t prefix_length =
          (iree_host_size_t)(updates[i].input_end - cursor);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
          new_source, iree_make_string_view(cursor, prefix_length)));
      // Blank line before the separator for visual separation from input.
      IREE_RETURN_IF_ERROR(loom_check_ensure_blank_line(new_source));
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(new_source, "// ----\n"));
      // Emit the actual output as the new expected section.
      IREE_RETURN_IF_ERROR(loom_check_append_with_trailing_newline(
          new_source, updates[i].actual_output));
      cursor = updates[i].input_end;
    }

    // If more source text follows, ensure a blank line between the updated
    // expected content and whatever comes next (typically // ====).
    const char* source_end = original_source.data + original_source.size;
    if (cursor < source_end) {
      IREE_RETURN_IF_ERROR(loom_check_ensure_blank_line(new_source));
    }
  }

  // Emit the tail of the original source after the last updated case.
  const char* source_end = original_source.data + original_source.size;
  if (cursor < source_end) {
    iree_host_size_t tail_length = (iree_host_size_t)(source_end - cursor);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        new_source, iree_make_string_view(cursor, tail_length)));
  }

  *out_update_count = update_count;
  return iree_ok_status();
}
