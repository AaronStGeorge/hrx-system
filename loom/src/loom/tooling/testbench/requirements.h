// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Provider-backed check.requires/check.skip_if evaluation.
//
// This layer is target-free. It interprets check requirement ops and delegates
// predicate answers to providers injected by the caller.

#ifndef LOOM_TOOLING_TESTBENCH_REQUIREMENTS_H_
#define LOOM_TOOLING_TESTBENCH_REQUIREMENTS_H_

#include "iree/base/api.h"
#include "loom/tooling/testbench/testbench.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_testbench_requirement_provider_t
    loom_testbench_requirement_provider_t;

// Evaluates one provider predicate.
//
// |attrs| is the check.requires/check.skip_if attribute dictionary. Providers
// return |out_satisfied| true when the predicate holds. |out_reason| may name a
// provider-specific reason for an unsatisfied requirement or active skip.
typedef iree_status_t (*loom_testbench_requirement_query_fn_t)(
    void* user_data, const loom_module_t* module, loom_named_attr_slice_t attrs,
    bool* out_satisfied, iree_string_view_t* out_reason);

struct loom_testbench_requirement_provider_t {
  // Provider name referenced by check.requires/check.skip_if.
  iree_string_view_t name;
  // Opaque provider state passed to |query|.
  void* user_data;
  // Predicate evaluator for this provider.
  loom_testbench_requirement_query_fn_t query;
};

typedef struct loom_testbench_requirement_provider_registry_t {
  // Borrowed provider entries visible to requirement evaluation.
  const loom_testbench_requirement_provider_t* providers;
  // Number of entries in |providers|.
  iree_host_size_t provider_count;
} loom_testbench_requirement_provider_registry_t;

typedef struct loom_testbench_requirement_result_t {
  // True when the selected case should be skipped.
  bool skipped;
  // Requirement op that caused the skip, or NULL when not skipped.
  const loom_op_t* op;
  // Provider name that caused the skip.
  iree_string_view_t provider;
  // Human-readable skip reason.
  iree_string_view_t reason;
} loom_testbench_requirement_result_t;

// Initializes a borrowed provider registry.
void loom_testbench_requirement_provider_registry_initialize(
    const loom_testbench_requirement_provider_t* providers,
    iree_host_size_t provider_count,
    loom_testbench_requirement_provider_registry_t* out_registry);

// Finds a named attribute in |attrs| by decoded string name.
const loom_named_attr_t* loom_testbench_requirement_find_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name);

// Reads a required string attribute from |attrs|.
iree_status_t loom_testbench_requirement_read_string_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name, iree_string_view_t* out_value);

// Reads an optional i64 attribute from |attrs|.
iree_status_t loom_testbench_requirement_read_optional_i64_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name, bool* out_present, int64_t* out_value);

// Evaluates requirement ops in source order for |case_plan|.
iree_status_t loom_testbench_evaluate_case_requirements(
    const loom_module_t* module, const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_requirement_provider_registry_t* registry,
    loom_testbench_requirement_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TESTBENCH_REQUIREMENTS_H_
