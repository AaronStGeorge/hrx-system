// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Incremental value-fact maintenance and CFG snapshots during IR rewriting.

#ifndef LOOM_REWRITE_FACT_PROPAGATION_H_
#define LOOM_REWRITE_FACT_PROPAGATION_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/ops/op_defs.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Attaches caller-computed value facts to the rewriter for incremental
// maintenance. The table must already cover every value the rewrite policy may
// query and remains owned by the caller. NULL disables value-fact maintenance.
void loom_rewriter_attach_value_facts(loom_rewriter_t* rewriter,
                                      loom_value_fact_table_t* facts);

// Updates the structural fact snapshot after a completed CFG edit and queues
// users of changed block-argument joins. Call before querying facts or draining
// the worklist after changing successors, block membership, or block arguments.
// Ordinary value changes propagate through branch payloads in try_fold.
iree_status_t loom_rewriter_refresh_cfg_facts(loom_rewriter_t* rewriter,
                                              loom_region_t* region);

// Enables value analysis on this rewriter using caller-owned fact storage and
// runs the initial forward pass over |region| and its nested regions, seeding
// facts from fact inference functions. |function| supplies the logical
// function context for fact inference and may be empty for detached regions.
// Call once after initialize, before the worklist loop. Patterns can then query
// facts via loom_rewriter_value_facts.
iree_status_t loom_rewriter_enable_region_analysis(
    loom_rewriter_t* rewriter, loom_func_like_t function, loom_region_t* region,
    loom_op_t* parent_op, loom_value_fact_table_t* facts);

// Enables value analysis over a function-like body region.
iree_status_t loom_rewriter_enable_analysis(loom_rewriter_t* rewriter,
                                            loom_func_like_t function,
                                            loom_value_fact_table_t* facts);

typedef enum loom_rewriter_user_change_flag_bits_e {
  // Operand identities are changing, not just the facts of the same value.
  LOOM_REWRITER_USER_CHANGE_FLAG_REPLACED = 1u << 0,
} loom_rewriter_user_change_flag_bits_t;
typedef uint32_t loom_rewriter_user_change_flags_t;

// Returns whether inference owns a loop or region-branch summary.
bool loom_rewriter_op_summarizes_nested_regions(const loom_op_vtable_t* vtable);

// Returns whether every summarized region has its required terminator.
bool loom_rewriter_nested_region_summary_is_ready(
    const loom_rewriter_t* rewriter, loom_op_t* op,
    const loom_op_vtable_t* vtable);

// Recomputes direct result facts and records semantic fact changes. The
// caller owns scheduling users and enclosing summaries.
iree_status_t loom_rewriter_recompute_op_facts(loom_rewriter_t* rewriter,
                                               loom_op_t* op);

// Publishes target and execution context before nested bodies are built.
iree_status_t loom_rewriter_seed_nested_execution_facts(
    loom_rewriter_t* rewriter, loom_op_t* op);

// Withdraws owned structural snapshots and releases their arenas.
void loom_rewriter_release_cfg_facts(loom_rewriter_t* rewriter);

// Refreshes dirty CFG equations reached by a terminator visit.
iree_status_t loom_rewriter_update_successor_facts(loom_rewriter_t* rewriter,
                                                   loom_op_t* op);

// Invalidates retained forwarding before branch operand identities change.
void loom_rewriter_invalidate_cfg_forwarding(loom_rewriter_t* rewriter,
                                             const loom_op_t* op);

// Schedules enclosing structured and cyclic equation owners after an edit.
iree_status_t loom_rewriter_add_parent_summary_ops_to_worklist(
    loom_rewriter_t* rewriter, loom_op_t* op);

// Schedules local successors and enclosing equation owners after an edit.
iree_status_t loom_rewriter_add_summary_ops_to_worklist(
    loom_rewriter_t* rewriter, loom_op_t* op);

// Schedules retained operand, type, and attribute dependencies of a value.
iree_status_t loom_rewriter_add_users_to_worklist(
    loom_rewriter_t* rewriter, loom_value_id_t value_id,
    loom_rewriter_user_change_flags_t flags);

// Schedules retained dependencies of every result of an operation.
iree_status_t loom_rewriter_add_result_users_to_worklist(
    loom_rewriter_t* rewriter, loom_op_t* op);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_REWRITE_FACT_PROPAGATION_H_
