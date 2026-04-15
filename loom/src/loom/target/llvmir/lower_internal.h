// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Internal lowering state and declarations shared by the LLVMIR lowering
// implementation files. Not a public API.

#ifndef LOOM_TARGET_LLVMIR_LOWER_INTERNAL_H_
#define LOOM_TARGET_LLVMIR_LOWER_INTERNAL_H_

#include "iree/base/api.h"
#include "loom/ir/module.h"
#include "loom/target/llvmir/builder.h"
#include "loom/target/llvmir/lower.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_llvmir_lowering_state_t {
  // Source Loom module being lowered.
  const loom_module_t* source_module;
  // Target structured LLVM IR module being populated.
  loom_llvmir_module_t* target_module;
  // Target/ABI profile selected for this lowering run.
  const loom_llvmir_target_profile_t* target_profile;
  // Host allocator used for temporary lowering maps.
  iree_allocator_t allocator;
  // Map from source Loom value id to target LLVMIR value id.
  loom_llvmir_value_id_t* value_map;
  // Number of entries in |value_map|.
  iree_host_size_t value_map_count;
  // Per-source-value pointer address space, or UINT32_MAX for non-pointers.
  uint32_t* value_pointer_address_spaces;
  // Per-source-value minimum pointer alignment, or zero when unknown.
  uint64_t* value_pointer_alignments;
  // Map from source module symbol id to target LLVMIR function object.
  loom_llvmir_function_t** symbol_functions;
  // Number of entries in |symbol_functions|.
  iree_host_size_t symbol_function_count;
  // Function-local fact table active while lowering a body, or NULL.
  const loom_value_fact_table_t* fact_table;
  // Cached profile kernel attribute group, or INVALID until materialized.
  loom_llvmir_attr_group_id_t kernel_attr_group_id;
  // Address spaces for cached llvm.prefetch declarations.
  uint32_t prefetch_function_address_spaces[8];
  // Cached llvm.prefetch declarations keyed by address space.
  loom_llvmir_function_t* prefetch_functions[8];
  // Number of cached llvm.prefetch declarations.
  iree_host_size_t prefetch_function_count;
} loom_llvmir_lowering_state_t;

iree_string_view_t loom_llvmir_lowering_value_name(
    const loom_llvmir_lowering_state_t* state, loom_value_id_t value_id);

iree_status_t loom_llvmir_lowering_unsupported_op(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    const char* detail);

iree_status_t loom_llvmir_lowering_unsupported_type(
    const loom_llvmir_lowering_state_t* state, loom_type_t type,
    const char* detail);

iree_status_t loom_llvmir_lowering_lower_scalar_type(
    loom_llvmir_lowering_state_t* state, loom_scalar_type_t scalar_type,
    loom_llvmir_type_id_t* out_type_id);

iree_status_t loom_llvmir_lowering_get_pointer_type(
    loom_llvmir_lowering_state_t* state, uint32_t address_space,
    loom_llvmir_type_id_t* out_type_id);

iree_status_t loom_llvmir_lowering_lower_type(
    loom_llvmir_lowering_state_t* state, loom_type_t type,
    loom_llvmir_type_id_t* out_type_id);

iree_status_t loom_llvmir_lowering_map_value(
    loom_llvmir_lowering_state_t* state, loom_value_id_t source_value_id,
    loom_llvmir_value_id_t target_value_id);

iree_status_t loom_llvmir_lowering_map_pointer_value(
    loom_llvmir_lowering_state_t* state, loom_value_id_t source_value_id,
    loom_llvmir_value_id_t target_value_id, uint32_t address_space,
    uint64_t minimum_alignment);

iree_status_t loom_llvmir_lowering_lookup_value(
    const loom_llvmir_lowering_state_t* state, loom_value_id_t source_value_id,
    loom_llvmir_value_id_t* out_target_value_id);

iree_status_t loom_llvmir_lowering_lookup_pointer(
    const loom_llvmir_lowering_state_t* state, loom_value_id_t source_value_id,
    loom_llvmir_value_id_t* out_target_value_id, uint32_t* out_address_space,
    uint64_t* out_minimum_alignment);

iree_status_t loom_llvmir_lowering_lookup_operands(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    loom_llvmir_value_id_t* target_operands);

iree_status_t loom_llvmir_lowering_map_single_result(
    loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    loom_llvmir_value_id_t target_value_id);

iree_status_t loom_llvmir_lowering_lower_binop(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_negf(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_constant(
    loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    loom_attribute_t value_attr);

iree_status_t loom_llvmir_lowering_lower_index_madd(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_icmp(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op, uint8_t source_predicate);

iree_status_t loom_llvmir_lowering_lower_fcmp(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_cast(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_index_cast(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_select(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_alloca(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_assume_memory_space(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_buffer_view(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_subview(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_refine(
    loom_llvmir_lowering_state_t* state, const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_view_load(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_view_store(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_view_prefetch(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_inline_asm(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_LLVMIR_LOWER_INTERNAL_H_
