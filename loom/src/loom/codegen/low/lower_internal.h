// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Private state shared by source-to-target-low lowering implementation files.

#ifndef LOOM_CODEGEN_LOW_LOWER_INTERNAL_H_
#define LOOM_CODEGEN_LOW_LOWER_INTERNAL_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/lower.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_LOW_LOWER_VALUE_ID_ELIDED ((loom_value_id_t)(UINT32_MAX - 1))

typedef struct loom_low_lower_rule_t loom_low_lower_rule_t;

typedef struct loom_low_lower_selected_plan_t {
  // Source op this selected plan lowers.
  const loom_op_t* source_op;
  // Policy rule-set ordinal for table-driven selections.
  uint16_t rule_set_index;
  // Rule set owning |rule|, or NULL for target-owned callbacks.
  const loom_low_lower_rule_set_t* rule_set;
  // Table rule selected during planning, or NULL for target-owned callbacks.
  const loom_low_lower_rule_t* rule;
  // Target-owned plan selected during planning, or empty for table rules.
  loom_low_lower_plan_t plan;
} loom_low_lower_selected_plan_t;

struct loom_low_lower_context_t {
  // Module being mutated by this lowering run.
  loom_module_t* module;
  // Source function being lowered.
  loom_func_like_t source_function;
  // Caller-owned lowering options.
  const loom_low_lower_options_t* options;
  // Target-low lowering policy selected by the caller.
  const loom_low_lower_policy_t* policy;
  // Descriptor set selected by source legality.
  const loom_low_descriptor_set_t* descriptor_set;
  // Result object receiving counters and emitted low function metadata.
  loom_low_lower_result_t* result;
  // Scratch arena for transient maps and remapped operand lists.
  iree_arena_allocator_t arena;
  // Source value facts computed once before planning.
  loom_value_fact_table_t fact_table;
  // Builder used while emitting the low function.
  loom_builder_t builder;
  // Emitted low.func.def operation, or NULL before emission starts.
  loom_op_t* low_func_op;
  // Number of source values captured by |value_map|.
  iree_host_size_t value_map_count;
  // Source value id to emitted low value id map.
  loom_value_id_t* value_map;
  // Source block ordinal to emitted low block pointer map.
  loom_block_t** block_map;
  // Source function argument ABI mappings.
  loom_low_lower_abi_argument_t* argument_map;
  // Number of entries in |argument_map|.
  uint16_t argument_map_count;
  // Selected lowering plans for non-structural source ops.
  loom_low_lower_selected_plan_t* selected_plans;
  // Number of selected plan slots used during planning.
  iree_host_size_t selected_plan_count;
  // Number of selected plan slots allocated for planning.
  iree_host_size_t selected_plan_capacity;
  // Next selected plan consumed by the emission walk.
  iree_host_size_t selected_plan_emit_index;
};

// Returns the source function name used in source-to-low diagnostics/reports.
iree_string_view_t loom_low_lower_context_function_name(
    const loom_low_lower_context_t* context);

// Returns true when the lowering context has reached its diagnostic limit.
bool loom_low_lower_context_should_stop(
    const loom_low_lower_context_t* context);

// Copies a source SSA value display name onto a low SSA value.
iree_status_t loom_low_lower_copy_value_name(loom_low_lower_context_t* context,
                                             loom_value_id_t source_value_id,
                                             loom_value_id_t low_value_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_INTERNAL_H_
