// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Prepared SPIR-V program consumed by binary emission.

#ifndef LOOM_TARGET_EMIT_SPIRV_PROGRAM_H_
#define LOOM_TARGET_EMIT_SPIRV_PROGRAM_H_

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/ir.h"
#include "loom/target/arch/spirv/value_types.h"
#include "loom/target/facts.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  // No emitted SPIR-V function is assigned to a module symbol.
  LOOM_SPIRV_PROGRAM_FUNCTION_INDEX_NONE = UINT32_MAX,
};

// One verified Low function and its target-specific representation binding.
typedef struct loom_spirv_function_plan_t {
  // Structured Low function definition emitted into the SPIR-V module.
  loom_op_t* function_op;
  // Function-specific target facts carrying its bundle and export plan.
  const loom_target_facts_t* target_facts;
  // Dense descriptor set selected by the function's Low representation.
  const loom_low_descriptor_set_t* descriptor_set;
  // Exact module-local symbol identity used by direct calls.
  loom_symbol_ref_t symbol_ref;
  // Semantic SPIR-V parameter types for an internal function.
  const loom_spirv_value_type_t* parameter_types;
  // Semantic SPIR-V result types for an internal function.
  const loom_spirv_value_type_t* result_types;
  // Logical source signature retained for HAL parameter reflection.
  // NULL when hand-authored Low IR did not declare a logical interface.
  const loom_func_type_data_t* logical_signature;
  // Number of entries in |parameter_types|.
  uint16_t parameter_count;
  // Number of entries in |result_types|.
  uint16_t result_count;
  // True when this function is an exported artifact entry. Internal functions
  // use native SPIR-V parameters/results and carry no entry-point metadata.
  bool is_entry;
} loom_spirv_function_plan_t;

// Returns the target bundle retained by |function_plan|.
static inline const loom_target_bundle_t*
loom_spirv_function_plan_target_bundle(
    const loom_spirv_function_plan_t* function_plan) {
  return loom_target_facts_bundle(function_plan->target_facts);
}

// Immutable SPIR-V program produced by compiler preparation.
//
// The function table contains only entries selected for this artifact. All
// entries have already resolved a concrete SPIR-V target, selected the
// spirv.logical.core descriptor set, and agreed on one module contract.
typedef struct loom_spirv_program_plan_t {
  // Module containing the prepared function bodies. Emission preserves
  // semantic IR and uses only module-owned value-domain scratch.
  loom_module_t* module;
  // Functions in callee-before-caller emission order. Recursion is rejected,
  // so each caller can consume the completed transitive summary of its callees.
  const loom_spirv_function_plan_t* functions;
  // Number of entries in |functions|. Prepared programs are never empty.
  iree_host_size_t function_count;
  // Function-plan index by module symbol ID.
  const uint32_t* function_indices_by_symbol;
  // Number of entries in |function_indices_by_symbol|.
  iree_host_size_t symbol_count;
  // Number of exported entry functions in |functions|.
  iree_host_size_t entry_count;
  // Index of the first exported entry in |functions|.
  uint32_t first_entry_index;
} loom_spirv_program_plan_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_SPIRV_PROGRAM_H_
