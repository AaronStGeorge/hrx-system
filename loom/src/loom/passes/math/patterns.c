// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/passes/math/patterns.h"

#include <string.h>

typedef struct loom_math_legalize_pattern_source_t {
  // Pattern array provided by one recipe shard.
  const loom_pattern_t* patterns;

  // Number of entries in patterns.
  iree_host_size_t pattern_count;
} loom_math_legalize_pattern_source_t;

static iree_status_t loom_math_legalize_collect_sources(
    iree_arena_allocator_t* arena, loom_math_legalize_pattern_source_t* sources,
    iree_host_size_t source_count,
    loom_math_legalize_pattern_table_t* out_table) {
  *out_table = (loom_math_legalize_pattern_table_t){0};
  iree_host_size_t total_pattern_count = 0;
  for (iree_host_size_t i = 0; i < source_count; ++i) {
    total_pattern_count += sources[i].pattern_count;
  }
  if (total_pattern_count == 0) {
    return iree_ok_status();
  }

  loom_pattern_t* patterns = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, total_pattern_count, sizeof(*patterns), (void**)&patterns));

  iree_host_size_t offset = 0;
  for (iree_host_size_t i = 0; i < source_count; ++i) {
    if (sources[i].pattern_count == 0) {
      continue;
    }
    memcpy(&patterns[offset], sources[i].patterns,
           sources[i].pattern_count * sizeof(*patterns));
    offset += sources[i].pattern_count;
  }

  *out_table = (loom_math_legalize_pattern_table_t){
      .patterns = patterns,
      .pattern_count = total_pattern_count,
  };
  return iree_ok_status();
}

iree_status_t loom_math_legalize_collect_patterns(
    iree_arena_allocator_t* arena,
    loom_math_legalize_pattern_table_t* out_table) {
  loom_math_legalize_pattern_table_t activation_patterns = {0};
  IREE_RETURN_IF_ERROR(loom_math_legalize_collect_activation_patterns(
      arena, &activation_patterns));

  loom_math_legalize_pattern_source_t sources[] = {
      {
          .patterns = activation_patterns.patterns,
          .pattern_count = activation_patterns.pattern_count,
      },
  };
  return loom_math_legalize_collect_sources(arena, sources,
                                            IREE_ARRAYSIZE(sources), out_table);
}
