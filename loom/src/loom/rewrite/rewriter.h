// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Pattern rewriter: worklist-driven IR mutation with change tracking.
//
// The rewriter wraps a builder and tracks all IR mutations (creation,
// erasure, operand changes, RAUW). It maintains a worklist of ops that
// need revisiting after changes, using LOOM_OP_FLAG_ON_WORKLIST for
// O(1) dedup.
//
// The greedy rewrite driver iterates the worklist, applying patterns
// or vtable canonicalize functions until fixed point. This drives the
// canonicalize pass and any pass-specific pattern sets.
//
// Usage (canonicalize pass):
//   loom_rewriter_t rewriter;
//   loom_rewriter_initialize(&rewriter, module, pass->arena);
//   loom_rewriter_seed_function(&rewriter, function);
//   while (loom_op_t* op = loom_rewriter_pop(&rewriter)) {
//     const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
//     if (vtable && vtable->canonicalize) {
//       rewriter.flags = 0;
//       vtable->canonicalize(op, &rewriter);
//     }
//   }
//   loom_rewriter_deinitialize(&rewriter);

#ifndef LOOM_REWRITE_REWRITER_H_
#define LOOM_REWRITE_REWRITER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/ops/op_defs.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Rewriter
//===----------------------------------------------------------------------===//

enum loom_rewriter_flag_bits_e {
  // Set by any mutation (create, erase, RAUW, set_operand).
  // Cleared by the driver before each pattern invocation.
  LOOM_REWRITER_FLAG_CHANGED = 1u << 0,
  // Set when a mutation changes a value type.
  LOOM_REWRITER_FLAG_TYPE_CHANGED = 1u << 1,
  // Set when incremental fact recomputation changes a value's facts.
  LOOM_REWRITER_FLAG_FACTS_CHANGED = 1u << 2,
};
typedef uint8_t loom_rewriter_flags_t;

// Callback to materialize a constant from exact facts into IR.
// Called by try_fold when all results are exact, and by
// loom_rewriter_build_constant for pattern use. The implementation
// builds the appropriate constant op (e.g., scalar.constant or
// test.constant) and returns the result value ID.
typedef iree_status_t (*loom_materialize_constant_fn_t)(
    loom_builder_t* builder, loom_value_facts_t facts, loom_type_t result_type,
    loom_location_id_t location, loom_value_id_t* out_value_id);

// Callback to materialize a typed replacement value into IR.
typedef iree_status_t (*loom_materialize_value_fn_t)(
    loom_builder_t* builder, loom_type_t result_type,
    loom_location_id_t location, loom_value_id_t* out_value_id);

struct loom_rewriter_t {
  // Builder for creating new ops. The rewriter installs a finalize
  // callback that adds newly created ops to the worklist.
  loom_builder_t builder;

  // The module being transformed.
  loom_module_t* module;

  // Pass arena for scratch allocations (worklist growth, etc.).
  iree_arena_allocator_t* arena;

  // Mutation tracking. Set by any mutation, cleared by the driver
  // before each pattern invocation.
  loom_rewriter_flags_t flags;

  // Worklist of ops to revisit. Deduped via LOOM_OP_FLAG_ON_WORKLIST.
  loom_op_t** worklist;
  iree_host_size_t worklist_count;
  iree_host_size_t worklist_capacity;

  // Per-value analysis facts. Populated when a pass calls
  // loom_rewriter_enable_analysis. A zero-initialized table (entries
  // == NULL) means analysis is disabled.
  loom_value_fact_table_t fact_table;

  // Dialect-provided constant materialization. Set by the pass before
  // enabling analysis. NULL means try_fold cannot materialize
  // constants (facts are still computed and propagated).
  loom_materialize_constant_fn_t materialize_constant;
};

// Initializes the rewriter. Installs the on_op_finalized callback on
// the builder so newly created ops are added to the worklist.
iree_status_t loom_rewriter_initialize(loom_rewriter_t* rewriter,
                                       loom_module_t* module,
                                       iree_arena_allocator_t* arena);

void loom_rewriter_deinitialize(loom_rewriter_t* rewriter);

// Seeds the worklist with all live ops in a function.
iree_status_t loom_rewriter_seed_function(loom_rewriter_t* rewriter,
                                          loom_func_like_t function);

// Enables value analysis on this rewriter. Initializes the fact table
// and runs the initial forward pass over the function, seeding facts from fact
// inference functions. Call once after initialize, before the worklist
// loop. Patterns can then query facts via loom_rewriter_value_facts.
iree_status_t loom_rewriter_enable_analysis(loom_rewriter_t* rewriter,
                                            loom_func_like_t function);

// Enables value analysis and first clones caller-provided seed facts into the
// rewriter-owned fact table. Extension payloads are re-interned so seed facts
// can come from a different analysis context. Block-argument seeders preserve
// already-defined facts from |seed_facts|.
iree_status_t loom_rewriter_enable_analysis_with_seed_facts(
    loom_rewriter_t* rewriter, loom_func_like_t function,
    const loom_value_fact_table_t* seed_facts);

// Looks up facts from the rewriter's fact table. Returns unknown
// facts if analysis is not enabled or the value is not defined.
static inline loom_value_facts_t loom_rewriter_value_facts(
    const loom_rewriter_t* rewriter, loom_value_id_t value_id) {
  return loom_value_fact_table_lookup(&rewriter->fact_table, value_id);
}

// Materializes a constant value using the rewriter's callback.
// Convenience for patterns that need to create constants (e.g.,
// index values for shape dimensions) without depending on any
// specific dialect. Returns IREE_STATUS_UNIMPLEMENTED if no
// materialize_constant callback is set.
iree_status_t loom_rewriter_build_constant(loom_rewriter_t* rewriter,
                                           loom_value_facts_t facts,
                                           loom_type_t result_type,
                                           loom_location_id_t location,
                                           loom_value_id_t* out_value_id);

// Attempts to fold an op to constants using its vtable fact inference function.
// Gathers operand facts, updates stored facts in the table, and materializes
// constants via the rewriter's materialize_constant callback when all results
// produce exact facts. Replaces the original op via RAUW+erase.
// Sets |*out_folded| to true if the op was erased, false otherwise.
// No-op if analysis is not enabled or the op has no inference function.
iree_status_t loom_rewriter_try_fold(loom_rewriter_t* rewriter, loom_op_t* op,
                                     bool* out_folded);

// Captures the current high-water mark of the module value table.
//
// Use before materializing replacement IR. Pass the returned checkpoint to
// loom_rewriter_preserve_result_names_on_new_values so named result values
// keep their human/agent-facing spelling without accidentally renaming an
// existing operand or block argument used as a replacement.
loom_value_id_t loom_rewriter_value_checkpoint(const loom_rewriter_t* rewriter);

// Copies result names from |op| to replacement values created after
// |value_checkpoint|. Existing replacement values, already-named replacements,
// unnamed old results, and invalid result slots are left unchanged.
//
// This is intentionally separate from RAUW: many canonicalizations replace a
// named result with a pre-existing operand, and mutating that operand's name
// would make the IR less true, not more readable.
iree_status_t loom_rewriter_preserve_result_names_on_new_values(
    loom_rewriter_t* rewriter, const loom_op_t* op,
    const loom_value_id_t* replacements, uint16_t count,
    loom_value_id_t value_checkpoint);

// Returns true if the op has no uses, no side effects, and is not a
// terminator — safe to erase without affecting program semantics.
bool loom_rewriter_is_trivially_dead(const loom_rewriter_t* rewriter,
                                     const loom_op_t* op);

// If the op is trivially dead, erases it and adds its operand
// providers to the worklist for cascading DCE. Returns true via
// |*out_erased| if the op was erased. No-op if the op is live.
iree_status_t loom_rewriter_erase_if_dead(loom_rewriter_t* rewriter,
                                          loom_op_t* op, bool* out_erased);

//===----------------------------------------------------------------------===//
// Worklist
//===----------------------------------------------------------------------===//

// Adds an op to the worklist (no-op if dead or already on worklist).
iree_status_t loom_rewriter_add_to_worklist(loom_rewriter_t* rewriter,
                                            loom_op_t* op);

// Pops the next op from the worklist. Returns NULL when empty.
// Clears LOOM_OP_FLAG_ON_WORKLIST on the returned op.
loom_op_t* loom_rewriter_pop(loom_rewriter_t* rewriter);

//===----------------------------------------------------------------------===//
// Safe IR mutation
//===----------------------------------------------------------------------===//
//
// These functions perform the mutation AND update the worklist with
// affected ops. Always use these instead of direct module functions
// when inside a pattern or canonicalize callback.

// Replaces all uses of |op|'s results with |replacements| (one per
// result), adds all affected user ops to the worklist, then erases
// |op|. The replacements array must have op->result_count entries.
iree_status_t loom_rewriter_replace_all_uses_and_erase(
    loom_rewriter_t* rewriter, loom_op_t* op,
    const loom_value_id_t* replacements, uint16_t count);

// Replaces all uses and type uses of one value with another value and adds
// affected users to the worklist. Does not erase the defining op.
iree_status_t loom_rewriter_replace_all_uses_with(loom_rewriter_t* rewriter,
                                                  loom_value_id_t old_value,
                                                  loom_value_id_t new_value);

// Materializes one replacement value per result using |materialize_value|,
// preserves existing result names, replaces all uses, and erases |op|.
iree_status_t loom_rewriter_replace_results_with_materialized_values_and_erase(
    loom_rewriter_t* rewriter, loom_op_t* op,
    loom_materialize_value_fn_t materialize_value);

// Erases an op that has no remaining uses.
iree_status_t loom_rewriter_erase(loom_rewriter_t* rewriter, loom_op_t* op);

// Moves |op| before |before_op|, preserving operands/results/regions while
// retargeting ancestry, worklist membership, and region effect summaries.
// Both ops must already be live in the same module. Moving an op before itself
// or before its current immediate successor is a no-op.
iree_status_t loom_rewriter_move_before(loom_rewriter_t* rewriter,
                                        loom_op_t* op, loom_op_t* before_op);

// Replaces one operand of an op. Adds the op to the worklist.
iree_status_t loom_rewriter_set_operand(loom_rewriter_t* rewriter,
                                        loom_op_t* op, uint16_t operand_index,
                                        loom_value_id_t new_value);

// Changes the type of a value. Adds all users of the value to the
// worklist since they may be simplifiable with the new type (e.g.,
// encoding materialization resolving SSA encoding refs to static
// encoding table entries).
iree_status_t loom_rewriter_set_value_type(loom_rewriter_t* rewriter,
                                           loom_value_id_t value_id,
                                           loom_type_t new_type);

// Sets an attribute on an op by index. Adds all users of the op's
// results to the worklist since derived properties may change (e.g.,
// setting the purity attr on a func.call changes its effective traits).
iree_status_t loom_rewriter_set_attr(loom_rewriter_t* rewriter, loom_op_t* op,
                                     uint16_t attr_index,
                                     loom_attribute_t value);

// Applies key-level updates to a DICT attribute on |op| and stores the fresh
// canonical dict back into the same attribute slot. Adds all users of |op|'s
// results to the worklist, just like loom_rewriter_set_attr.
iree_status_t loom_rewriter_replace_attr_dict(
    loom_rewriter_t* rewriter, loom_op_t* op, uint16_t attr_index,
    loom_named_attr_update_slice_t updates);

// Sets the instance flags on an op. Adds all users of the op's results
// to the worklist since their effective traits may change.
iree_status_t loom_rewriter_set_instance_flags(loom_rewriter_t* rewriter,
                                               loom_op_t* op, uint8_t flags);

//===----------------------------------------------------------------------===//
// Greedy rewrite driver
//===----------------------------------------------------------------------===//

// A rewrite pattern: matches ops of a specific kind and attempts to
// transform them.
typedef struct loom_pattern_t loom_pattern_t;
struct loom_pattern_t {
  loom_op_kind_t root_kind;  // Op kind this pattern matches.
  uint16_t benefit;          // Higher = tried first.
  iree_status_t (*match_and_rewrite)(const loom_pattern_t* pattern,
                                     loom_op_t* op, loom_rewriter_t* rewriter);
};

typedef struct loom_rewrite_config_t {
  uint32_t max_iterations;  // 0 = default (10).
} loom_rewrite_config_t;

// Runs greedy pattern application on a function. Iterates the worklist
// until empty or max_iterations reached.
iree_status_t loom_greedy_rewrite(iree_arena_allocator_t* arena,
                                  loom_module_t* module,
                                  loom_func_like_t function,
                                  const loom_pattern_t* patterns,
                                  iree_host_size_t pattern_count,
                                  const loom_rewrite_config_t* config);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_REWRITE_REWRITER_H_
