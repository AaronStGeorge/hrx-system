// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lowering for structured control-flow Loom ops into LLVMIR blocks.

#include "loom/ops/scf/ops.h"
#include "loom/target/llvmir/lower_internal.h"

static iree_status_t loom_llvmir_lowering_scf_single_block(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    loom_region_t* region, const char* detail, loom_block_t** out_block) {
  if (!region || region->block_count != 1) {
    return loom_llvmir_lowering_unsupported_op(state, op, detail);
  }
  *out_block = loom_region_entry_block(region);
  return iree_ok_status();
}

static iree_status_t loom_llvmir_lowering_scf_yield_values(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    const loom_op_t* yield_op, loom_value_slice_t* out_values) {
  if (!yield_op || !loom_scf_yield_isa(yield_op)) {
    return loom_llvmir_lowering_unsupported_op(
        state, op, "SCF branch did not produce an scf.yield terminator");
  }
  loom_value_slice_t values = loom_scf_yield_values(yield_op);
  if (values.count != op->result_count) {
    return loom_llvmir_lowering_unsupported_op(
        state, op, "scf.yield operand count must match scf.if results");
  }
  *out_values = values;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_lowering_build_pointer_phi(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* join_block,
    const loom_op_t* op, iree_host_size_t result_ordinal,
    loom_value_id_t then_value_id, loom_value_id_t else_value_id,
    loom_llvmir_block_id_t then_predecessor,
    loom_llvmir_block_id_t else_predecessor) {
  loom_llvmir_value_id_t then_value = LOOM_LLVMIR_VALUE_ID_INVALID;
  uint32_t then_address_space = UINT32_MAX;
  uint64_t then_alignment = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lookup_pointer(
      state, then_value_id, &then_value, &then_address_space, &then_alignment));

  loom_llvmir_value_id_t else_value = LOOM_LLVMIR_VALUE_ID_INVALID;
  uint32_t else_address_space = UINT32_MAX;
  uint64_t else_alignment = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lookup_pointer(
      state, else_value_id, &else_value, &else_address_space, &else_alignment));
  if (then_address_space != else_address_space) {
    return loom_llvmir_lowering_unsupported_op(
        state, op, "scf.if pointer results must use one address space");
  }

  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_get_pointer_type(
      state, then_address_space, &result_type));
  loom_llvmir_phi_incoming_t incoming[2] = {
      {
          .value = then_value,
          .predecessor = then_predecessor,
      },
      {
          .value = else_value,
          .predecessor = else_predecessor,
      },
  };
  loom_value_id_t result_id = loom_op_const_results(op)[result_ordinal];
  loom_llvmir_value_id_t phi = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_phi(
      join_block,
      &(loom_llvmir_phi_desc_t){
          .result_name = loom_llvmir_lowering_value_name(state, result_id),
          .result_type = result_type,
          .incoming = incoming,
          .incoming_count = IREE_ARRAYSIZE(incoming),
      },
      &phi));
  uint64_t alignment =
      then_alignment < else_alignment ? then_alignment : else_alignment;
  return loom_llvmir_lowering_map_pointer_value(state, result_id, phi,
                                                then_address_space, alignment);
}

static iree_status_t loom_llvmir_lowering_build_value_phi(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* join_block,
    const loom_op_t* op, iree_host_size_t result_ordinal,
    loom_value_id_t then_value_id, loom_value_id_t else_value_id,
    loom_llvmir_block_id_t then_predecessor,
    loom_llvmir_block_id_t else_predecessor) {
  loom_llvmir_value_id_t then_value = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lookup_value(state, then_value_id, &then_value));
  loom_llvmir_value_id_t else_value = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lookup_value(state, else_value_id, &else_value));

  loom_value_id_t result_id = loom_op_const_results(op)[result_ordinal];
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_type(
      state, loom_module_value_type(state->source_module, result_id),
      &result_type));
  loom_llvmir_phi_incoming_t incoming[2] = {
      {
          .value = then_value,
          .predecessor = then_predecessor,
      },
      {
          .value = else_value,
          .predecessor = else_predecessor,
      },
  };
  loom_llvmir_value_id_t phi = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_phi(
      join_block,
      &(loom_llvmir_phi_desc_t){
          .result_name = loom_llvmir_lowering_value_name(state, result_id),
          .result_type = result_type,
          .incoming = incoming,
          .incoming_count = IREE_ARRAYSIZE(incoming),
      },
      &phi));
  return loom_llvmir_lowering_map_value(state, result_id, phi);
}

static iree_status_t loom_llvmir_lowering_build_scf_if_results(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* join_block,
    const loom_op_t* op, loom_value_slice_t then_values,
    loom_value_slice_t else_values, loom_llvmir_block_id_t then_predecessor,
    loom_llvmir_block_id_t else_predecessor) {
  for (iree_host_size_t i = 0; i < op->result_count; ++i) {
    loom_value_id_t result_id = loom_op_const_results(op)[i];
    loom_type_t result_type =
        loom_module_value_type(state->source_module, result_id);
    if (loom_type_is_buffer(result_type) || loom_type_is_view(result_type)) {
      IREE_RETURN_IF_ERROR(loom_llvmir_lowering_build_pointer_phi(
          state, join_block, op, i, then_values.values[i],
          else_values.values[i], then_predecessor, else_predecessor));
    } else {
      IREE_RETURN_IF_ERROR(loom_llvmir_lowering_build_value_phi(
          state, join_block, op, i, then_values.values[i],
          else_values.values[i], then_predecessor, else_predecessor));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_llvmir_lowering_lower_scf_if(
    loom_llvmir_lowering_state_t* state,
    loom_llvmir_function_t* target_function,
    loom_llvmir_block_t** current_block, const loom_op_t* op) {
  if (op->tied_result_count > 0) {
    return loom_llvmir_lowering_unsupported_op(
        state, op, "scf.if lowering does not support tied results");
  }

  loom_block_t* then_source_block = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_scf_single_block(
      state, op, loom_scf_if_then_region(op),
      "scf.if then region must contain one block", &then_source_block));
  loom_block_t* else_source_block = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_scf_single_block(
      state, op, loom_scf_if_else_region(op),
      "scf.if else region must contain one block", &else_source_block));

  loom_llvmir_value_id_t condition = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lookup_value(
      state, loom_scf_if_condition(op), &condition));

  loom_llvmir_block_t* then_block = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_function_add_block(
      target_function, iree_string_view_empty(), &then_block));
  loom_llvmir_block_t* else_block = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_function_add_block(
      target_function, iree_string_view_empty(), &else_block));
  loom_llvmir_block_t* join_block = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_function_add_block(
      target_function, iree_string_view_empty(), &join_block));

  IREE_RETURN_IF_ERROR(loom_llvmir_build_cond_br(
      *current_block, condition, loom_llvmir_block_id(then_block),
      loom_llvmir_block_id(else_block)));

  const loom_op_t* then_yield = NULL;
  loom_llvmir_block_t* then_current_block = then_block;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_source_block(
      state, target_function, then_source_block, &then_current_block,
      &then_yield));
  loom_llvmir_block_id_t then_predecessor =
      loom_llvmir_block_id(then_current_block);
  IREE_RETURN_IF_ERROR(loom_llvmir_build_br(then_current_block,
                                            loom_llvmir_block_id(join_block)));

  const loom_op_t* else_yield = NULL;
  loom_llvmir_block_t* else_current_block = else_block;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_source_block(
      state, target_function, else_source_block, &else_current_block,
      &else_yield));
  loom_llvmir_block_id_t else_predecessor =
      loom_llvmir_block_id(else_current_block);
  IREE_RETURN_IF_ERROR(loom_llvmir_build_br(else_current_block,
                                            loom_llvmir_block_id(join_block)));

  loom_value_slice_t then_values = {0};
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_scf_yield_values(
      state, op, then_yield, &then_values));
  loom_value_slice_t else_values = {0};
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_scf_yield_values(
      state, op, else_yield, &else_values));
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_build_scf_if_results(
      state, join_block, op, then_values, else_values, then_predecessor,
      else_predecessor));

  *current_block = join_block;
  return iree_ok_status();
}
