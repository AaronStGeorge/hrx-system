// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Update logic for --update mode in loom-check.
//
// Reconstructs a test file source with updated expected sections. The
// reconstruction is pure computation (no file I/O) — the caller decides
// where to write the result. This separation keeps the logic testable
// without touching the filesystem.
//
// The typical call sequence:
//
//   1. Parse the file with loom_check_parse().
//   2. Execute each case with loom_check_execute_case().
//   3. Build loom_check_case_update_t entries from the results.
//   4. Call loom_check_apply_updates() to reconstruct the source.
//   5. Write the result wherever appropriate (file, stdout, etc.).

#ifndef LOOM_TOOLS_LOOM_CHECK_UPDATE_H_
#define LOOM_TOOLS_LOOM_CHECK_UPDATE_H_

#include "iree/base/api.h"
#include "loom/tools/loom-check/check.h"

#ifdef __cplusplus
extern "C" {
#endif

// Per-case update tracking. The caller populates one of these per case
// after execution, indicating whether the expected section should be
// rewritten and providing the actual output text.
typedef struct loom_check_case_update_t {
  // Whether this case's expected section should be rewritten.
  bool needs_update;
  // The actual output to write. Must remain valid until apply_updates
  // returns (typically a view into a result's actual_output builder).
  iree_string_view_t actual_output;
  // Pointer to the end of the input section in the original source.
  // Used to locate where to insert a // ---- separator when the case
  // has no existing expected section.
  const char* input_end;
  // Pointers to the start and end of the existing expected section in
  // the original source. Only valid when the case has_expected_section.
  const char* expected_start;
  const char* expected_end;
} loom_check_case_update_t;

// Reconstructs the source with updated expected sections. Writes the
// result into |new_source|. Does NOT write to disk — the caller decides
// whether and where to write. Returns the number of cases that were
// updated via |out_update_count|.
//
// The reconstruction walks the original source, copying unchanged
// regions verbatim and replacing or inserting expected sections for
// cases where needs_update is true. Trailing newlines are ensured
// after each inserted/replaced section.
iree_status_t loom_check_apply_updates(iree_string_view_t original_source,
                                       const loom_check_file_t* file,
                                       const loom_check_case_update_t* updates,
                                       iree_string_builder_t* new_source,
                                       iree_host_size_t* out_update_count);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_UPDATE_H_
