// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Queries for linear value consumption.
//
// Tied low results consume their tied operands: after the consuming op executes
// along one dynamic path, later operations on that same path must observe the
// tied result, not the consumed value. CFG regions make that a path-sensitive
// question because a value can be re-created by a block argument or an earlier
// same-block definition on a later dynamic entry.

#ifndef LOOM_ANALYSIS_CONSUMPTION_H_
#define LOOM_ANALYSIS_CONSUMPTION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/util/cfg_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

// Operand occurrence that observes a consumed value.
typedef struct loom_consumption_use_t {
  // Operation containing the observing operand.
  const loom_op_t* op;
  // Operand index on |op| that observes the value.
  uint16_t operand_index;
} loom_consumption_use_t;

// Finds a use of |value_id| that can dynamically execute after
// |consuming_op|. If |cfg_graph| is non-NULL, it must describe
// |consuming_op|'s parent region and is reused without allocation. If
// |cfg_graph| is NULL and the parent region is CFG-shaped, this function builds
// a temporary graph in |scratch_arena|. Non-CFG regions only check later uses
// in the same block.
iree_status_t loom_consumption_find_use_after(
    const loom_module_t* module, const loom_cfg_graph_t* cfg_graph,
    const loom_op_t* consuming_op, loom_value_id_t value_id,
    iree_arena_allocator_t* scratch_arena, loom_consumption_use_t* out_use,
    bool* out_found);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_CONSUMPTION_H_
