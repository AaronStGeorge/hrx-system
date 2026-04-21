// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Function-like symbol facts.
//
// These facts are the dense, direct-indexed function context shared by source
// functions and target-low functions. Dialects expose structure through the
// FuncLike interface; passes consume this payload instead of walking attrs or
// looking for companion global records.

#ifndef LOOM_OPS_FUNC_LIKE_FACTS_H_
#define LOOM_OPS_FUNC_LIKE_FACTS_H_

#include "iree/base/api.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/ir/ir.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_func_like_symbol_facts_t loom_func_like_symbol_facts_t;
typedef struct loom_target_profile_symbol_facts_t
    loom_target_profile_symbol_facts_t;

// Resolved function-like symbol payload.
typedef struct loom_func_like_symbol_facts_t {
  // Common symbol-fact header.
  loom_symbol_facts_base_t base;

  // Defining function-like op.
  loom_op_t* function_op;

  // Module-local symbol reference for the function definition.
  loom_symbol_ref_t symbol;

  // Borrowed function symbol name from the module string table.
  iree_string_view_t name;

  // Visibility enum value from the function-like interface.
  uint8_t visibility;

  // Calling convention enum value from the function-like interface.
  uint8_t calling_convention;

  // Purity enum value from the function-like interface.
  uint8_t purity;

  // True when the function-like op owns an implementation body.
  bool has_body;

  // Borrowed argument value IDs in signature order.
  const loom_value_id_t* argument_ids;

  // Number of argument value IDs.
  uint16_t argument_count;

  // Borrowed result value IDs in signature order.
  const loom_value_id_t* result_ids;

  // Number of result value IDs.
  uint16_t result_count;

  // Module-local target profile symbol, or null for target-independent funcs.
  loom_symbol_ref_t target_symbol;

  // Resolved target profile facts, or NULL for target-independent functions.
  const loom_target_profile_symbol_facts_t* target_profile;

  // Effective target bundle, or NULL for target-independent functions.
  const loom_target_bundle_t* target_bundle;

  // Effective ABI/export plan. Valid when target_bundle is non-NULL.
  loom_target_export_plan_t export_plan;
} loom_func_like_symbol_facts_t;

// Symbol fact domain used by generated function-like symbol descriptors.
extern const loom_symbol_fact_domain_t loom_func_like_symbol_fact_domain;

// Casts generic symbol facts to function-like facts when domains match.
const loom_func_like_symbol_facts_t* loom_func_like_symbol_facts_cast(
    const loom_symbol_facts_base_t* facts);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_FUNC_LIKE_FACTS_H_
