// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Per-value fact table: dense array of loom_value_facts_t indexed by
// value_id. Arena-allocated. A zero-initialized table is valid (empty).
//
// Lookup always succeeds: returns unknown facts for undefined entries
// or out-of-range value IDs. Undefined entries are detected by
// known_divisor == 0 (valid facts always have known_divisor >= 1),
// allowing O(1) initialization via memset(0).
//
// Define stores facts for a value ID, growing the array as needed.
// Compute runs a forward pass over a function, calling each op's fold
// function to seed initial facts from constants and op semantics.
//
// The table is a reusable component: embedded in the rewriter for
// canonicalization, usable standalone for IPO or analysis tools.
//
// Typical lifecycle:
//
//   loom_value_fact_table_t table = {0};
//   IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize(
//       &table, arena, value_count));
//   IREE_RETURN_IF_ERROR(loom_value_fact_table_compute(
//       &table, arena, module, function));
//   loom_value_facts_t facts = loom_value_fact_table_lookup(&table, id);

#ifndef LOOM_UTIL_FACT_TABLE_H_
#define LOOM_UTIL_FACT_TABLE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/facts.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Fact table
//===----------------------------------------------------------------------===//

typedef struct loom_value_fact_table_t {
  // Arena for all allocations (entries, scratch). Stored at init time
  // so callers don't need to thread it through every operation.
  iree_arena_allocator_t* arena;

  loom_value_facts_t* entries;
  iree_host_size_t count;     // Highest defined value_id + 1.
  iree_host_size_t capacity;  // Allocated entry count.

  // Reusable scratch buffers for fold calls. Allocated on first use,
  // grown only when an op needs more slots. Never shrinks. Old
  // buffers are abandoned in the arena (freed in bulk with the arena).
  struct {
    struct {
      loom_value_facts_t* values;
      iree_host_size_t capacity;
    } facts;
    struct {
      loom_value_id_t* values;
      iree_host_size_t capacity;
    } value_ids;
  } scratch;
} loom_value_fact_table_t;

// Initializes the table with the given arena and pre-allocates for
// |initial_capacity| values. All entries are zero-initialized
// (known_divisor == 0 means undefined). The arena is stored and used
// for all subsequent allocations (entries, scratch growth).
iree_status_t loom_value_fact_table_initialize(
    loom_value_fact_table_t* table, iree_arena_allocator_t* arena,
    iree_host_size_t initial_capacity);

// Looks up facts for a value. Returns unknown facts if the value ID
// is out of range or the entry is undefined (known_divisor == 0).
static inline loom_value_facts_t loom_value_fact_table_lookup(
    const loom_value_fact_table_t* table, loom_value_id_t value_id) {
  if (value_id >= table->count || table->entries[value_id].known_divisor == 0) {
    return loom_value_facts_unknown();
  }
  return table->entries[value_id];
}

// Defines (or updates) facts for a value. Grows the table as needed.
iree_status_t loom_value_fact_table_define(loom_value_fact_table_t* table,
                                           loom_value_id_t value_id,
                                           loom_value_facts_t facts);

// Computes facts for a single op by calling its vtable fold function.
// Gathers operand facts from the table, calls fold, and defines
// result facts. No-op if the op has no fold function.
iree_status_t loom_value_fact_table_compute_op(loom_value_fact_table_t* table,
                                               const loom_module_t* module,
                                               const loom_op_t* op);

// Seeds the table by running a forward pass over all ops in a
// function. For each op with a fold function, calls compute_op.
// Visits ops in dominance order so operand facts are available
// before use.
iree_status_t loom_value_fact_table_compute(loom_value_fact_table_t* table,
                                            loom_module_t* module,
                                            loom_func_like_t function);

// Returns a facts scratch buffer with at least |count| entries.
// The returned pointer is valid until the next call. Grows the
// allocation if needed; never shrinks.
iree_status_t loom_value_fact_table_facts_scratch(
    loom_value_fact_table_t* table, iree_host_size_t count,
    loom_value_facts_t** out);

// Returns a value ID scratch buffer with at least |count| entries.
// Same lifetime and growth semantics as facts_scratch.
iree_status_t loom_value_fact_table_value_id_scratch(
    loom_value_fact_table_t* table, iree_host_size_t count,
    loom_value_id_t** out);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_FACT_TABLE_H_
