// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-low source legality verification.
//
// This verifier runs before source-to-low lowering. It checks the source
// function against the selected target bundle and linked low descriptor set,
// then gives target-family providers a hook to accept or reject explicit
// target contracts such as packed dot or matrix ops. User IR failures are
// reported through structured backend diagnostics; infrastructure and
// compiler-configuration failures are returned as status.

#ifndef LOOM_TARGET_LOW_LEGALITY_H_
#define LOOM_TARGET_LOW_LEGALITY_H_

#include "iree/base/api.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/ops/op_defs.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_low_legality_context_t
    loom_target_low_legality_context_t;
typedef struct loom_target_low_legality_provider_t
    loom_target_low_legality_provider_t;

typedef iree_status_t (*loom_target_low_legality_try_op_fn_t)(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

struct loom_target_low_legality_provider_t {
  // Stable provider name available to callbacks when emitting diagnostics.
  iree_string_view_t name;
  // Attempts to verify one source op. Sets |out_handled| false when the op does
  // not belong to this provider.
  loom_target_low_legality_try_op_fn_t try_verify_op;
};

typedef struct loom_target_low_legality_provider_list_t {
  // Total number of values in the list.
  iree_host_size_t count;
  // Value list or NULL if no values.
  const loom_target_low_legality_provider_t* const* values;
} loom_target_low_legality_provider_list_t;

// Returns an empty source legality provider list.
static inline loom_target_low_legality_provider_list_t
loom_target_low_legality_provider_list_empty(void) {
  return (loom_target_low_legality_provider_list_t){0, NULL};
}

// Returns true if |list| has no source legality providers.
static inline bool loom_target_low_legality_provider_list_is_empty(
    loom_target_low_legality_provider_list_t list) {
  return list.count == 0;
}

// Verifies that |list| is internally well-formed.
iree_status_t loom_target_low_legality_provider_list_verify(
    loom_target_low_legality_provider_list_t list);

typedef struct loom_target_low_legality_options_t {
  // Target bundle selected for this source-to-low lowering attempt.
  const loom_target_bundle_t* bundle;
  // Low descriptor registry linked into the current compiler binary.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Descriptor payload requirements needed by the upcoming lowering/emission
  // stage. Zero verifies only descriptor structural integrity needed to select
  // the set.
  loom_low_descriptor_requirement_flags_t descriptor_requirements;
  // Optional target-specific source legality providers.
  loom_target_low_legality_provider_list_t provider_list;
  // Structured diagnostic emitter for user legality failures and remarks.
  iree_diagnostic_emitter_t emitter;
  // Maximum number of errors to emit before aborting the walk. Zero means no
  // limit.
  uint32_t max_errors;
} loom_target_low_legality_options_t;

typedef struct loom_target_low_legality_result_t {
  // Number of error diagnostics emitted.
  uint32_t error_count;
  // Number of remark diagnostics emitted.
  uint32_t remark_count;
  // Descriptor set selected by options.bundle, or NULL when selection failed
  // before verification started.
  const loom_low_descriptor_set_t* descriptor_set;
} loom_target_low_legality_result_t;

// Verifies that |function| is legal as source IR for target-low lowering under
// |options|.
//
// User IR legality failures are counted in |out_result| and emitted through
// options.emitter. The function still returns OK unless an infrastructure error
// such as malformed options, invalid provider tables, or registry lookup
// failure occurs.
iree_status_t loom_target_low_verify_function_legality(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_low_legality_options_t* options,
    loom_target_low_legality_result_t* out_result);

// Returns the source module being checked.
const loom_module_t* loom_target_low_legality_module(
    const loom_target_low_legality_context_t* context);

// Returns the function being checked.
loom_func_like_t loom_target_low_legality_function(
    const loom_target_low_legality_context_t* context);

// Returns the selected target bundle.
const loom_target_bundle_t* loom_target_low_legality_bundle(
    const loom_target_low_legality_context_t* context);

// Returns the selected low descriptor set.
const loom_low_descriptor_set_t* loom_target_low_legality_descriptor_set(
    const loom_target_low_legality_context_t* context);

// Emits ERR_BACKEND_001 for an unsupported legality subject.
iree_status_t loom_target_low_legality_reject(
    loom_target_low_legality_context_t* context,
    const loom_target_low_legality_provider_t* provider, const loom_op_t* op,
    iree_string_view_t subject_kind, iree_string_view_t subject_name,
    iree_string_view_t reason);

// Emits ERR_BACKEND_002 for a target contract decision.
iree_status_t loom_target_low_legality_record_contract(
    loom_target_low_legality_context_t* context,
    const loom_target_low_legality_provider_t* provider, const loom_op_t* op,
    iree_string_view_t contract_key, iree_string_view_t decision,
    iree_string_view_t reason);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_LOW_LEGALITY_H_
