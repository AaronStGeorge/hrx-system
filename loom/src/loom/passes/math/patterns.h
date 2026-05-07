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
#include "loom/pass/types.h"
#include "loom/rewrite/greedy.h"
#include "loom/target/math_policy.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_math_legalize_recipe_t loom_math_legalize_recipe_t;

typedef struct loom_math_legalize_recipe_context_t {
  // Pass invocation using the recipe.
  loom_pass_t* pass;
  // Module being rewritten.
  loom_module_t* module;
  // Target-policy query that selected this recipe.
  loom_target_math_query_t query;
  // Target-policy decision that selected this recipe.
  loom_target_math_policy_decision_t decision;
} loom_math_legalize_recipe_context_t;

typedef iree_status_t (*loom_math_legalize_recipe_rewrite_fn_t)(
    const loom_math_legalize_recipe_t* recipe,
    const loom_math_legalize_recipe_context_t* context, loom_op_t* op,
    loom_rewriter_t* rewriter);

struct loom_math_legalize_recipe_t {
  // Target-policy recipe selected by a math policy.
  loom_target_math_recipe_t recipe;
  // Source op kind covered by this recipe row.
  loom_op_kind_t root_kind;
  // Recipe-owned rewrite callback.
  loom_math_legalize_recipe_rewrite_fn_t rewrite;
};

typedef struct loom_math_legalize_recipe_table_t {
  // Contiguous recipe array assembled from all enabled recipe shards.
  const loom_math_legalize_recipe_t* recipes;

  // Number of entries in recipes.
  iree_host_size_t recipe_count;
} loom_math_legalize_recipe_table_t;

// Collects all enabled math legalization recipes into one declaration-order
// table. The returned recipe storage is borrowed from |arena| when non-empty.
iree_status_t loom_math_legalize_collect_recipes(
    iree_arena_allocator_t* arena,
    loom_math_legalize_recipe_table_t* out_table);

// Collects elementwise scalar/vector recipes such as exp, erf, logistic, SiLU,
// GELU, and softplus. This shard is intentionally separate from the pass
// wrapper so the recipe set can grow by math family without creating a
// monolithic pass file.
iree_status_t loom_math_legalize_collect_elementwise_recipes(
    iree_arena_allocator_t* arena,
    loom_math_legalize_recipe_table_t* out_table);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_PASSES_MATH_PATTERNS_H_
