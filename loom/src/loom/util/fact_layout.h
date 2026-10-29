// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Scope-local SSA provenance for exact physical layouts. Numeric layout and
// storage-schema summaries are independently transferable facts. These origins
// name values in the current SSA scope and never travel with a fact extension.

#ifndef LOOM_UTIL_FACT_LAYOUT_H_
#define LOOM_UTIL_FACT_LAYOUT_H_

#include "iree/base/api.h"
#include "loom/ir/facts.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_value_fact_table_t loom_value_fact_table_t;
typedef struct loom_value_fact_layout_origins_t
    loom_value_fact_layout_origins_t;

// Exact recipe retained for one address layout in the current SSA scope.
typedef enum loom_value_fact_layout_origin_kind_e {
  // No scoped materialization recipe is available.
  LOOM_VALUE_FACT_LAYOUT_ORIGIN_NONE = 0,
  // One SSA stride value per axis, with INVALID for numerically exact axes.
  LOOM_VALUE_FACT_LAYOUT_ORIGIN_EXPLICIT_STRIDES = 1,
  // Packed static/SSA source dimensions defining dense row-major strides.
  LOOM_VALUE_FACT_LAYOUT_ORIGIN_DENSE_DIMENSIONS = 2,
} loom_value_fact_layout_origin_kind_t;

// Immutable physical-layout origin retained by the fact table. The compact
// representation remains 16 bytes because one entry may exist for every
// physical encoding in a function-local fact scope.
typedef struct loom_value_fact_layout_origin_t {
  union {
    // SSA stride per axis for EXPLICIT_STRIDES.
    const loom_value_id_t* explicit_strides;
    // Packed source dimension per axis for DENSE_DIMENSIONS.
    const uint64_t* dense_dimensions;
    // Untyped payload used only for generic storage management.
    const void* data;
  } values;
  // Root SSA value carrying the same complete physical encoding.
  loom_value_id_t physical_encoding_value_id;
  // Layout rank, or zero when no scoped recipe is available.
  uint16_t count;
  // Recipe family selecting the active payload member.
  uint8_t kind;
  // Whether physical_encoding_value_id is present. Value ID zero is valid.
  bool has_physical_encoding_value;
} loom_value_fact_layout_origin_t;

static_assert(sizeof(loom_value_fact_layout_origin_t) == sizeof(void*) + 8,
              "layout origins must remain compact");

// Availability of one materialized element-stride recipe.
typedef enum loom_value_fact_layout_axis_kind_e {
  // Numeric facts exist but no exact SSA recipe is retained.
  LOOM_VALUE_FACT_LAYOUT_AXIS_UNAVAILABLE = 0,
  // The complete element stride is an exact compile-time value.
  LOOM_VALUE_FACT_LAYOUT_AXIS_STATIC = 1,
  // Runtime SSA factors form the complete element-stride product.
  LOOM_VALUE_FACT_LAYOUT_AXIS_DYNAMIC = 2,
  // The retained recipe cannot be represented by the bounded product.
  LOOM_VALUE_FACT_LAYOUT_AXIS_INVALID = 3,
} loom_value_fact_layout_axis_kind_t;

// Bounded product materializing one physical element stride.
typedef struct loom_value_fact_layout_axis_t {
  // Availability and materialization class.
  loom_value_fact_layout_axis_kind_t kind;
  // Static coefficient multiplied by every dynamic factor.
  int64_t static_coefficient;
  // Runtime SSA values multiplied into the element stride.
  loom_value_id_t dynamic_factors[LOOM_TYPE_MAX_RANK];
  // Number of populated dynamic factors.
  uint8_t dynamic_factor_count;
} loom_value_fact_layout_axis_t;

// Retains a copy of explicit per-axis SSA stride bindings for |value_id|.
// Static axes use LOOM_VALUE_ID_INVALID; their values live in the numeric
// summary. An empty slice clears only the recipe. Recomputing a producer
// replaces its recipe; unchanged origins reuse their storage.
iree_status_t loom_value_fact_table_define_layout_explicit_strides(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    const loom_value_id_t* strides, uint16_t count);

// Retains a copy of packed dimensions whose row-major suffix products define
// the layout strides. Dynamic entries name SSA values in the current scope.
// An empty slice clears only the recipe.
iree_status_t loom_value_fact_table_define_layout_dense_dimensions(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    const uint64_t* dimensions, uint16_t count);

// Retains the root SSA value carrying the same address layout and storage
// schema as |value_id|. This identity remains useful when both values have
// opaque numeric facts at a callable boundary. INVALID clears only the
// physical-encoding identity.
iree_status_t loom_value_fact_table_define_layout_physical_encoding_value(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_id_t physical_encoding_value_id);

// Retains an origin copied from another fact scope. Payload arrays are copied
// into |table|; SSA identities must already be valid in the target scope.
iree_status_t loom_value_fact_table_define_layout_origin(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_layout_origin_t origin);

// Returns the retained exact recipe, or NONE when the encoding has no scoped
// materialization. The payload borrows immutable fact-table storage until the
// populated scope is cleared. This is an indexed query, not an IR traversal.
loom_value_fact_layout_origin_t loom_value_fact_table_query_layout_origin(
    const loom_value_fact_table_t* table, loom_value_id_t value_id);

// Compares recipe semantics, including payload contents rather than allocation
// addresses. Both recipes must remain live for the duration of the call.
bool loom_value_fact_layout_origin_equal(loom_value_fact_layout_origin_t lhs,
                                         loom_value_fact_layout_origin_t rhs);

// Shares the source's retained recipe with a value that has the same physical
// layout in this scope. An unavailable source clears the result. This does not
// map values across calls or select among control-flow predecessors.
iree_status_t loom_value_fact_table_forward_layout_origin(
    loom_value_fact_table_t* table, loom_value_id_t source_value_id,
    loom_value_id_t result_value_id);

// Withdraws a recipe when its producer is invalidated, without allocating.
void loom_value_fact_table_clear_layout_origin(loom_value_fact_table_t* table,
                                               loom_value_id_t value_id);

// Combines numeric stride facts with a retained exact recipe. Exact facts fold
// to a static coefficient. Dense recipes derive suffix products directly from
// their copied dimensions; explicit recipes use at most one SSA factor. The
// query is bounded by LOOM_TYPE_MAX_RANK and never inspects IR producers.
void loom_value_fact_layout_query_axis(const loom_value_fact_table_t* table,
                                       loom_value_fact_layout_origin_t origin,
                                       uint8_t axis,
                                       loom_value_facts_t numeric_stride_facts,
                                       loom_value_fact_layout_axis_t* out_axis);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_FACT_LAYOUT_H_
