// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// File input and output helpers shared by Loom command-line tools.

#ifndef LOOM_TOOLING_IO_FILE_H_
#define LOOM_TOOLING_IO_FILE_H_

#include "iree/base/api.h"
#include "iree/io/file_contents.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when |path| denotes stdin or stdout for command-line tools.
bool loom_tooling_file_path_is_stdio(iree_string_view_t path);

// Reads |path| into |out_contents|, treating empty and "-" paths as stdin.
iree_status_t loom_tooling_read_input_file(
    iree_string_view_t path, iree_allocator_t allocator,
    iree_io_file_contents_t** out_contents);

// Returns the contents buffer as a borrowed string view.
iree_string_view_t loom_tooling_file_contents_string_view(
    const iree_io_file_contents_t* contents);

// Writes |contents| to |path|, treating empty and "-" paths as stdout.
iree_status_t loom_tooling_write_output_file(iree_string_view_t path,
                                             iree_string_view_t contents,
                                             iree_allocator_t allocator);

// Writes |contents| to stdout and flushes the stream.
iree_status_t loom_tooling_write_stdout(iree_string_view_t contents);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_IO_FILE_H_
