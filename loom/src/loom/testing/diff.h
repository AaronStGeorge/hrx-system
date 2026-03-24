// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Line-oriented unified diff of two text strings.
//
// Computes the shortest edit script between |expected| and |actual|,
// then formats the result as a unified diff with context lines.
// Output follows the standard unified diff format:
//
//   --- expected
//   +++ actual
//   @@ -1,4 +1,3 @@
//    context line
//   -removed line
//   +added line
//    context line
//
// When the inputs are identical, produces no output (builder is
// untouched). When either input is empty, every line in the other
// appears as an addition or removal.
//
// The algorithm is O(N*M) where N and M are the line counts of the
// two inputs. This is fine for IR text (typically tens to hundreds of
// lines). For very large inputs, consider a streaming approach.
//
// All output goes through an iree_string_builder_t — the caller owns
// the builder and its allocator.

#ifndef LOOM_TESTING_DIFF_H_
#define LOOM_TESTING_DIFF_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Number of unchanged context lines to show around each change hunk.
#define LOOM_DIFF_DEFAULT_CONTEXT 3

// Computes a unified diff between |expected| and |actual|.
//
// Both inputs are treated as line-oriented text (split on '\n').
// Trailing content after the last '\n' is treated as a final line.
//
// If the inputs are identical, nothing is appended to |builder| and
// the function returns iree_ok_status().
//
// The diff is formatted with |context_lines| lines of unchanged
// context around each change hunk. Adjacent hunks within context
// distance are merged into a single hunk.
//
// |allocator| is used for scratch allocations (line arrays, DP table).
// All scratch memory is freed before the function returns. An arena
// allocator is suitable when the caller already has one available.
iree_status_t loom_diff(iree_string_view_t expected, iree_string_view_t actual,
                        iree_host_size_t context_lines,
                        iree_allocator_t allocator,
                        iree_string_builder_t* builder);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TESTING_DIFF_H_
