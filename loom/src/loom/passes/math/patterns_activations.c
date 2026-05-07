// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/passes/math/patterns.h"

iree_status_t loom_math_legalize_collect_activation_recipes(
    iree_arena_allocator_t* arena,
    loom_math_legalize_recipe_table_t* out_table) {
  *out_table = (loom_math_legalize_recipe_table_t){0};
  return iree_ok_status();
}
