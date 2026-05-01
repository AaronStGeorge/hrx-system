// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Read-only target contract queries over source-to-target-low rule tables.
//
// This layer owns the shared contract matching path used before mutation. It
// interprets rule-table guards against a selected target bundle and descriptor
// set, returning compact target contract query results without emitting
// diagnostics or lowering IR. Callers provide bridges for source-value register
// metadata and optional materialization predicates.

#ifndef LOOM_CODEGEN_LOW_CONTRACT_QUERY_H_
#define LOOM_CODEGEN_LOW_CONTRACT_QUERY_H_

#include "iree/base/api.h"
#include "loom/codegen/low/lower_rules.h"
#include "loom/ir/ir.h"
#include "loom/target/contract.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_lower_contract_query_options_t {
  // Optional compact contract table used for direct op-to-case lookup. When
  // present, descriptor-rule cases reference rows in |rule_sets|.
  const loom_target_contract_table_t* contract_table;
  // Rule sets to query in priority order.
  loom_low_lower_rule_set_list_t rule_sets;
  // Source-value to target-low register metadata mapper.
  loom_low_lower_rule_match_map_value_callback_t map_value;
  // Optional descriptor-register-class to module string mapper.
  loom_low_lower_rule_match_register_class_callback_t register_class;
  // Optional source value materializer predicate bridge.
  loom_low_lower_rule_match_can_materialize_value_callback_t can_materialize;
} loom_low_lower_contract_query_options_t;

// Queries source-to-target-low rule tables for one source op.
//
// A LEGAL result means one opt-in rule matched the selected target contract. An
// UNSUPPORTED result means at least one opt-in rule set covered the source op
// kind but all candidates rejected it. UNHANDLED means no opt-in rule set had
// an opinion. Non-OK status is reserved for malformed tables or allocation
// failures while preparing rare rejection payloads.
iree_status_t loom_low_lower_query_target_contract(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_contract_query_options_t* options,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_CONTRACT_QUERY_H_
