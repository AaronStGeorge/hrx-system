// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Legalize-math recipe tables.

#ifndef LOOM_PASSES_MATH_PATTERNS_H_
#define LOOM_PASSES_MATH_PATTERNS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/rewrite/greedy.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_math_legalize_pattern_table_t {
  // Contiguous pattern array assembled from all enabled recipe shards.
  const loom_pattern_t* patterns;

  // Number of entries in patterns.
  iree_host_size_t pattern_count;
} loom_math_legalize_pattern_table_t;

// Collects all enabled math legalization recipes into one declaration-order
// table. The returned pattern storage is borrowed from |arena| when non-empty.
iree_status_t loom_math_legalize_collect_patterns(
    iree_arena_allocator_t* arena,
    loom_math_legalize_pattern_table_t* out_table);

// Collects activation-family recipes such as logistic, SiLU, GELU, and
// softplus. This shard is intentionally separate from the pass wrapper so the
// recipe set can grow by math family without creating a monolithic pass file.
iree_status_t loom_math_legalize_collect_activation_patterns(
    iree_arena_allocator_t* arena,
    loom_math_legalize_pattern_table_t* out_table);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_PASSES_MATH_PATTERNS_H_
