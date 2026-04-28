// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/io/file.h"

#include <stdio.h>

bool loom_tooling_file_path_is_stdio(iree_string_view_t path) {
  return iree_string_view_is_empty(path) ||
         iree_string_view_equal(path, IREE_SV("-"));
}

iree_status_t loom_tooling_read_input_file(
    iree_string_view_t path, iree_allocator_t allocator,
    iree_io_file_contents_t** out_contents) {
  IREE_ASSERT_ARGUMENT(out_contents);
  if (loom_tooling_file_path_is_stdio(path)) {
    return iree_io_file_contents_read_stdin(allocator, out_contents);
  }
  return iree_io_file_contents_read(path, allocator, out_contents);
}

iree_string_view_t loom_tooling_file_contents_string_view(
    const iree_io_file_contents_t* contents) {
  IREE_ASSERT_ARGUMENT(contents);
  return iree_make_string_view((const char*)contents->const_buffer.data,
                               contents->const_buffer.data_length);
}

iree_status_t loom_tooling_write_output_file(iree_string_view_t path,
                                             iree_string_view_t contents,
                                             iree_allocator_t allocator) {
  iree_const_byte_span_t bytes =
      iree_make_const_byte_span(contents.data, contents.size);
  if (!loom_tooling_file_path_is_stdio(path)) {
    return iree_io_file_contents_write(path, bytes, allocator);
  }

  if (bytes.data_length > 0 &&
      fwrite(bytes.data, bytes.data_length, 1, stdout) != 1) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "failed to write %" PRIhsz " bytes to stdout",
                            bytes.data_length);
  }
  return fflush(stdout) == 0 ? iree_ok_status()
                             : iree_make_status(IREE_STATUS_DATA_LOSS,
                                                "failed to flush stdout");
}

iree_status_t loom_tooling_write_stdout(iree_string_view_t contents) {
  if (iree_string_view_is_empty(contents)) {
    return iree_ok_status();
  }
  return loom_tooling_write_output_file(IREE_SV("-"), contents,
                                        iree_allocator_null());
}
