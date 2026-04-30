// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target contract query ABI.
//
// Contract queries are the read-only selection layer shared by target
// legalization, target-low legality, and source-to-low emission. They answer
// whether a source op is already in a form accepted by the selected target
// bundle and descriptor set. Unsupported program forms are reported as compact
// query results; non-OK status is reserved for infrastructure failures.

#ifndef LOOM_TARGET_CONTRACT_H_
#define LOOM_TARGET_CONTRACT_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/ir.h"
#include "loom/ops/func/ops.h"
#include "loom/target/types.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_target_contract_query_outcome_e {
  // No linked contract table or provider has an opinion about the op.
  LOOM_TARGET_CONTRACT_QUERY_UNHANDLED = 0,
  // The op is already legal for the selected target contract.
  LOOM_TARGET_CONTRACT_QUERY_LEGAL = 1,
  // The op family is recognized but unsupported by the selected target
  // contract.
  LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED = 2,
  // The op violates the source contract required before target selection.
  LOOM_TARGET_CONTRACT_QUERY_INVALID_IR = 3,
} loom_target_contract_query_outcome_t;

typedef struct loom_target_contract_rejection_t {
  // Diagnostic subject category, such as "type", "attr", or "descriptor".
  iree_string_view_t subject_kind;
  // Diagnostic subject name within subject_kind.
  iree_string_view_t subject_name;
  // Human-readable rejection reason.
  iree_string_view_t reason;
} loom_target_contract_rejection_t;

typedef struct loom_target_contract_query_result_t {
  // Query outcome.
  loom_target_contract_query_outcome_t outcome;
  // Target contract table ordinal selected or rejected by the query.
  uint16_t table_index;
  // Target contract rule ordinal selected or rejected by the query.
  uint16_t rule_index;
  // Diagnostic row ordinal retained by the rejected rule.
  uint16_t diagnostic_index;
  // Number of guards matched before the selected rejected guard.
  uint16_t matched_guard_count;
  // Stable low descriptor ID selected by the accepted rule.
  uint64_t selected_descriptor_id;
  // Compact target-independent rejection flags.
  uint32_t source_rejection_bits;
  // Compact target-owned rejection flags.
  uint32_t target_rejection_bits;
  // Target feature bits missing from the selected bundle.
  uint32_t missing_feature_bits;
  // Value fact categories missing for the selected rule.
  uint32_t missing_fact_bits;
  // Optional rejection payload. Usually points into rodata or a scoped arena.
  const loom_target_contract_rejection_t* rejection;
} loom_target_contract_query_result_t;

// Returns an empty target contract query result.
static inline loom_target_contract_query_result_t
loom_target_contract_query_result_empty(void) {
  return (loom_target_contract_query_result_t){
      .outcome = LOOM_TARGET_CONTRACT_QUERY_UNHANDLED,
      .table_index = UINT16_MAX,
      .rule_index = UINT16_MAX,
      .diagnostic_index = UINT16_MAX,
      .matched_guard_count = 0,
      .selected_descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE,
      .source_rejection_bits = 0,
      .target_rejection_bits = 0,
      .missing_feature_bits = 0,
      .missing_fact_bits = 0,
      .rejection = NULL,
  };
}

typedef struct loom_target_contract_query_environment_t {
  // Source module being queried.
  const loom_module_t* module;
  // Source function containing the queried op.
  loom_func_like_t function;
  // Target bundle selected for this query.
  const loom_target_bundle_t* bundle;
  // Low descriptor set selected for this query.
  const loom_low_descriptor_set_t* descriptor_set;
  // Source value facts visible to the query.
  const loom_value_fact_table_t* fact_table;
  // Scoped arena available for rare query-side auxiliary records.
  iree_arena_allocator_t* arena;
} loom_target_contract_query_environment_t;

typedef iree_status_t (*loom_target_contract_query_op_fn_t)(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* op, loom_target_contract_query_result_t* out_result);

typedef struct loom_target_contract_query_callback_t {
  // Callback invoked to query one source op.
  loom_target_contract_query_op_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_target_contract_query_callback_t;

// Returns an empty target contract query callback.
static inline loom_target_contract_query_callback_t
loom_target_contract_query_callback_empty(void) {
  return (loom_target_contract_query_callback_t){0};
}

// Returns true when |callback| has no query function.
static inline bool loom_target_contract_query_callback_is_empty(
    loom_target_contract_query_callback_t callback) {
  return callback.fn == NULL;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_CONTRACT_H_
