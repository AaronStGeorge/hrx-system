// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/update.h"

#include <string.h>

const char* loom_check_update_edit_kind_name(
    loom_check_update_edit_kind_t kind) {
  switch (kind) {
    case LOOM_CHECK_UPDATE_EDIT_REPLACE_EXPECTED_OUTPUT:
      return "replace_expected_output";
    case LOOM_CHECK_UPDATE_EDIT_INSERT_EXPECTED_OUTPUT:
      return "insert_expected_output";
    case LOOM_CHECK_UPDATE_EDIT_INSERT_DIAGNOSTIC_ANNOTATIONS:
      return "insert_diagnostic_annotations";
    case LOOM_CHECK_UPDATE_EDIT_DELETE_DIAGNOSTIC_ANNOTATION:
      return "delete_diagnostic_annotation";
  }
  return "unknown";
}

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

static bool loom_check_source_range_is_in_bounds(
    loom_check_source_range_t range, iree_host_size_t source_size) {
  return range.start_byte <= range.end_byte && range.end_byte <= source_size;
}

iree_status_t loom_check_build_update_edit(iree_string_view_t original_source,
                                           const loom_check_case_t* test_case,
                                           iree_string_view_t actual_output,
                                           iree_string_builder_t* new_text,
                                           loom_check_update_edit_t* out_edit) {
  IREE_ASSERT_ARGUMENT(test_case);
  IREE_ASSERT_ARGUMENT(new_text);
  IREE_ASSERT_ARGUMENT(out_edit);
  memset(out_edit, 0, sizeof(*out_edit));

  iree_host_size_t cursor_offset = 0;
  if (test_case->has_expected_section) {
    if (!loom_check_source_range_is_in_bounds(test_case->expected_range,
                                              original_source.size)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "expected section range out of bounds");
    }
    out_edit->kind = LOOM_CHECK_UPDATE_EDIT_REPLACE_EXPECTED_OUTPUT;
    out_edit->range = test_case->expected_range;
    cursor_offset = test_case->expected_range.end_byte;
    IREE_RETURN_IF_ERROR(
        loom_check_append_with_trailing_newline(new_text, actual_output));
  } else {
    if (!loom_check_source_range_is_in_bounds(test_case->input_range,
                                              original_source.size)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "input section range out of bounds");
    }
    out_edit->kind = LOOM_CHECK_UPDATE_EDIT_INSERT_EXPECTED_OUTPUT;
    out_edit->range = (loom_check_source_range_t){
        .start_byte = test_case->input_range.end_byte,
        .end_byte = test_case->input_range.end_byte,
    };
    cursor_offset = test_case->input_range.end_byte;

    iree_string_view_t prefix =
        iree_make_string_view(original_source.data, cursor_offset);
    if (prefix.size > 0 && !loom_check_ends_with_blank_line(prefix)) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(new_text, "\n"));
    }
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(new_text, "// ----\n"));
    IREE_RETURN_IF_ERROR(
        loom_check_append_with_trailing_newline(new_text, actual_output));
  }

  if (cursor_offset < original_source.size) {
    IREE_RETURN_IF_ERROR(loom_check_ensure_blank_line(new_text));
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
