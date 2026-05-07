// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/passes/math/patterns.h"

#include <string.h>

typedef struct loom_math_legalize_recipe_source_t {
  // Recipe array provided by one recipe shard.
  const loom_math_legalize_recipe_t* recipes;

  // Number of entries in recipes.
  iree_host_size_t recipe_count;
} loom_math_legalize_recipe_source_t;

static iree_status_t loom_math_legalize_collect_sources(
    iree_arena_allocator_t* arena, loom_math_legalize_recipe_source_t* sources,
    iree_host_size_t source_count,
    loom_math_legalize_recipe_table_t* out_table) {
  *out_table = (loom_math_legalize_recipe_table_t){0};
  iree_host_size_t total_recipe_count = 0;
  for (iree_host_size_t i = 0; i < source_count; ++i) {
    total_recipe_count += sources[i].recipe_count;
  }
  if (total_recipe_count == 0) {
    return iree_ok_status();
  }

  loom_math_legalize_recipe_t* recipes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, total_recipe_count, sizeof(*recipes), (void**)&recipes));

  iree_host_size_t offset = 0;
  for (iree_host_size_t i = 0; i < source_count; ++i) {
    if (sources[i].recipe_count == 0) {
      continue;
    }
    memcpy(&recipes[offset], sources[i].recipes,
           sources[i].recipe_count * sizeof(*recipes));
    offset += sources[i].recipe_count;
  }

  *out_table = (loom_math_legalize_recipe_table_t){
      .recipes = recipes,
      .recipe_count = total_recipe_count,
  };
  return iree_ok_status();
}

iree_status_t loom_math_legalize_collect_recipes(
    iree_arena_allocator_t* arena,
    loom_math_legalize_recipe_table_t* out_table) {
  loom_math_legalize_recipe_table_t elementwise_recipes = {0};
  IREE_RETURN_IF_ERROR(loom_math_legalize_collect_elementwise_recipes(
      arena, &elementwise_recipes));

  loom_math_legalize_recipe_source_t sources[] = {
      {
          .recipes = elementwise_recipes.recipes,
          .recipe_count = elementwise_recipes.recipe_count,
      },
  };
  return loom_math_legalize_collect_sources(arena, sources,
                                            IREE_ARRAYSIZE(sources), out_table);
}
