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
// Define stores facts for a value ID, growing the array as needed. Compute
// runs a forward pass over a function, calling each op's fact inference
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
//       &table, module, function));
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

typedef struct loom_value_fact_table_t loom_value_fact_table_t;
typedef struct loom_value_fact_extension_entry_t
    loom_value_fact_extension_entry_t;

// Maximum lane facts stored in a small static vector extension. Larger static
// vectors degrade to unknown facts instead of allocating per-lane analysis
// payloads.
#define LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT 16

// All lanes of a vector value share the same element facts.
typedef struct loom_value_fact_uniform_element_t {
  // Scalar facts that apply to every lane.
  loom_value_facts_t element;
} loom_value_fact_uniform_element_t;

// Per-lane facts for a small all-static vector, in logical lane order.
typedef struct loom_value_fact_small_static_lanes_t {
  // Borrowed lane facts. Query results point into the owning fact table.
  const loom_value_facts_t* lanes;
  // Number of lane facts in the slice.
  iree_host_size_t count;
} loom_value_fact_small_static_lanes_t;

// Vector value is a lane-coordinate sequence: base + lane_ordinal * step.
typedef struct loom_value_fact_vector_iota_t {
  // Facts for the first produced coordinate.
  loom_value_facts_t base;
  // Facts for the logical lane-ordinal delta.
  loom_value_facts_t step;
} loom_value_fact_vector_iota_t;

// Vector value is a prefix mask produced by vector.mask.range.
typedef struct loom_value_fact_vector_prefix_mask_t {
  // Facts for the first tested coordinate.
  loom_value_facts_t lower_bound;
  // Facts for the exclusive coordinate bound.
  loom_value_facts_t upper_bound;
  // Facts for the coordinate delta between adjacent logical lanes.
  loom_value_facts_t step;
} loom_value_fact_vector_prefix_mask_t;

// Per-analysis context passed to op fact inference callbacks.
struct loom_fact_context_t {
  // Table that owns the dense facts and any extension payloads allocated by
  // inference helpers.
  loom_value_fact_table_t* table;
};

struct loom_value_fact_table_t {
  // Arena for all allocations owned by the table.
  iree_arena_allocator_t* arena;

  // Dense fact entries indexed by value ID.
  loom_value_facts_t* entries;
  // Highest defined value ID plus one.
  iree_host_size_t count;
  // Allocated entry count.
  iree_host_size_t capacity;
  // Context object passed to op-specific fact inference callbacks.
  loom_fact_context_t context;

  // Interned fact extension payloads. Extension IDs stored in
  // loom_value_facts_t are one-based indexes into entries and are only valid
  // for this table/context.
  struct {
    // Extension entries indexed by one-based extension ID minus one.
    loom_value_fact_extension_entry_t* entries;
    // Allocated extension entry count.
    iree_host_size_t capacity;
    // Defined extension entry count.
    iree_host_size_t count;
    // Hash buckets storing one-based extension IDs, or zero for empty buckets.
    loom_value_fact_extension_id_t* buckets;
    // Allocated hash bucket count.
    iree_host_size_t bucket_count;
  } extensions;

  // Reusable scratch buffers for fact inference calls. Allocated on first use,
  // grown only when an op needs more slots. Never shrinks. Old buffers are
  // abandoned in the arena and freed in bulk with the arena.
  struct {
    // Fact scratch buffer state.
    struct {
      // Scratch fact values used for operand and result fact arrays.
      loom_value_facts_t* values;
      // Allocated fact scratch entry count.
      iree_host_size_t capacity;
    } facts;
    // Value ID scratch buffer state.
    struct {
      // Scratch value IDs used by transforms that materialize replacements.
      loom_value_id_t* values;
      // Allocated value ID scratch entry count.
      iree_host_size_t capacity;
    } value_ids;
  } scratch;
};

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

// Computes facts for a single op by calling its vtable fact inference function.
// Gathers operand facts from the table, calls the callback, and defines result
// facts. No-op if the op has no inference function.
iree_status_t loom_value_fact_table_compute_op(loom_value_fact_table_t* table,
                                               const loom_module_t* module,
                                               const loom_op_t* op);

// Seeds the table by running a forward pass over all ops in a
// function. For each op with an inference function, calls compute_op.
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

// Creates facts for a vector whose every lane has |element| facts.
iree_status_t loom_value_facts_make_uniform_element(
    loom_fact_context_t* context, loom_value_facts_t element,
    loom_value_facts_t* out);

// Returns true and populates |out| when |facts| is a uniform-element vector
// extension in |context|.
bool loom_value_facts_query_uniform_element(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_uniform_element_t* out);

// Creates facts for a small all-static vector with explicit per-lane facts.
// Vectors with more than LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT lanes degrade
// to unknown facts.
iree_status_t loom_value_facts_make_small_static_lanes(
    loom_fact_context_t* context, loom_value_fact_small_static_lanes_t lanes,
    loom_value_facts_t* out);

// Returns true and populates |out| when |facts| is a small-static-lanes vector
// extension in |context|.
bool loom_value_facts_query_small_static_lanes(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_small_static_lanes_t* out);

// Creates facts for a vector.iota-style lane-coordinate sequence.
iree_status_t loom_value_facts_make_vector_iota(
    loom_fact_context_t* context, loom_value_fact_vector_iota_t iota,
    loom_value_facts_t* out);

// Returns true and populates |out| when |facts| is a vector.iota extension in
// |context|.
bool loom_value_facts_query_vector_iota(const loom_fact_context_t* context,
                                        loom_value_facts_t facts,
                                        loom_value_fact_vector_iota_t* out);

// Creates facts for a vector.mask.range-style prefix mask.
iree_status_t loom_value_facts_make_vector_prefix_mask(
    loom_fact_context_t* context, loom_value_fact_vector_prefix_mask_t mask,
    loom_value_facts_t* out);

// Returns true and populates |out| when |facts| is a vector.mask.range
// extension in |context|.
bool loom_value_facts_query_vector_prefix_mask(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_vector_prefix_mask_t* out);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_FACT_TABLE_H_
