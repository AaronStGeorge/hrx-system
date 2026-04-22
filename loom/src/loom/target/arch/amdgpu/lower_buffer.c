// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ops/buffer/ops.h"
#include "loom/target/arch/amdgpu/lower_internal.h"
#include "loom/util/fact_table.h"

static bool loom_amdgpu_can_lower_buffer_view(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op) {
  int64_t unused_byte_offset = 0;
  return loom_amdgpu_value_is_byte_addressable_view(
             context, loom_buffer_view_result(source_op)) &&
         loom_amdgpu_module_value_as_exact_index_constant(
             loom_low_lower_context_module(context),
             loom_buffer_view_byte_offset(source_op), &unused_byte_offset) &&
         unused_byte_offset >= 0;
}

static bool loom_amdgpu_can_lower_buffer_alloca(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  if (loom_buffer_alloca_memory_space(source_op) !=
      LOOM_BUFFER_MEMORY_SPACE_WORKGROUP) {
    return false;
  }
  const int64_t base_alignment = loom_buffer_alloca_base_alignment(source_op);
  if (base_alignment <= 0 || base_alignment > UINT32_MAX ||
      !loom_amdgpu_u32_is_power_of_two((uint32_t)base_alignment)) {
    return false;
  }
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  int64_t byte_length = 0;
  return loom_amdgpu_value_facts_as_exact_non_negative_i64(
             loom_value_fact_table_lookup(
                 fact_table, loom_buffer_alloca_byte_length(source_op)),
             &byte_length) &&
         byte_length > 0;
}

iree_status_t loom_amdgpu_select_buffer_plan(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op,
                                             loom_low_lower_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = loom_low_lower_plan_empty();
  bool selected = false;
  switch (source_op->kind) {
    case LOOM_OP_BUFFER_ALLOCA:
      selected = loom_amdgpu_can_lower_buffer_alloca(context, source_op);
      break;
    case LOOM_OP_BUFFER_ASSUME_MEMORY_SPACE:
      selected = true;
      break;
    case LOOM_OP_BUFFER_VIEW:
      selected = loom_amdgpu_can_lower_buffer_view(context, source_op);
      break;
    default:
      return iree_ok_status();
  }
  if (selected) {
    loom_amdgpu_set_selected_plan(source_op->kind, 0, NULL, out_plan);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lower_buffer_alloca(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  IREE_ASSERT(loom_amdgpu_can_lower_buffer_alloca(context, source_op));

  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  int64_t byte_length = 0;
  const bool has_static_byte_length =
      loom_amdgpu_value_facts_as_exact_non_negative_i64(
          loom_value_fact_table_lookup(
              fact_table, loom_buffer_alloca_byte_length(source_op)),
          &byte_length);
  IREE_ASSERT(has_static_byte_length);
  IREE_ASSERT_GT(byte_length, 0);

  loom_symbol_ref_t slot_ref = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(loom_low_lower_create_function_symbol(
      context, IREE_SV("__lds"), /*append_index=*/true,
      loom_buffer_alloca_result(source_op), &slot_ref));

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_op_t* low_func_op = loom_low_lower_context_low_function(context);
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_after(builder, low_func_op);
  loom_op_t* slot_op = NULL;
  iree_status_t status = loom_low_slot_build(
      builder, slot_ref, loom_low_func_def_callee(low_func_op),
      LOOM_LOW_SLOT_SPACE_LDS, byte_length,
      loom_buffer_alloca_base_alignment(source_op), source_op->location,
      &slot_op);
  loom_builder_restore(builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_op_t* frame_index_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_frame_index_build(builder, slot_ref, /*offset=*/0, vgpr_type,
                                 source_op->location, &frame_index_op));
  return loom_low_lower_bind_value(context,
                                   loom_buffer_alloca_result(source_op),
                                   loom_low_frame_index_result(frame_index_op));
}

static iree_status_t loom_amdgpu_lower_buffer_view(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t low_buffer = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_buffer_view_buffer(source_op), &low_buffer));
  return loom_low_lower_bind_value(context, loom_buffer_view_result(source_op),
                                   low_buffer);
}

static iree_status_t loom_amdgpu_lower_buffer_assume_memory_space(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t low_buffer = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_buffer_assume_memory_space_buffer(source_op), &low_buffer));
  return loom_low_lower_bind_value(
      context, loom_buffer_assume_memory_space_result(source_op), low_buffer);
}

iree_status_t loom_amdgpu_lower_buffer_op(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op) {
  switch (source_op->kind) {
    case LOOM_OP_BUFFER_ALLOCA:
      return loom_amdgpu_lower_buffer_alloca(context, source_op);
    case LOOM_OP_BUFFER_ASSUME_MEMORY_SPACE:
      return loom_amdgpu_lower_buffer_assume_memory_space(context, source_op);
    case LOOM_OP_BUFFER_VIEW:
      return loom_amdgpu_lower_buffer_view(context, source_op);
    default:
      IREE_ASSERT_UNREACHABLE();
      return iree_ok_status();
  }
}

iree_status_t loom_amdgpu_low_legality_verify_buffer(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);
  if (!loom_amdgpu_type_is_byte_addressable_view(
          loom_module_value_type(module, loom_buffer_view_result(op)))) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("memory"), loom_op_name(module, op),
        IREE_SV("AMDGPU buffer memory lowering currently requires typed views "
                "over byte-addressable scalar elements"));
  }
  int64_t unused_byte_offset = 0;
  if (!loom_amdgpu_module_value_as_exact_index_constant(
          module, loom_buffer_view_byte_offset(op), &unused_byte_offset) ||
      unused_byte_offset < 0) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("memory"), loom_op_name(module, op),
        IREE_SV("AMDGPU HAL buffer views currently require exact non-negative "
                "static byte offsets"));
  }
  return iree_ok_status();
}
