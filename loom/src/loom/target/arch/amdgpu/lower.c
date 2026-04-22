// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
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

static iree_status_t loom_amdgpu_allocate_selected_plan_data(
    loom_low_lower_context_t* context, iree_host_size_t data_length,
    void** out_data) {
  IREE_ASSERT_GT(data_length, 0);
  IREE_ASSERT_ARGUMENT(out_data);
  *out_data = NULL;
  return loom_low_lower_allocate_scratch_array(context, 1, data_length,
                                               out_data);
}

static iree_status_t loom_amdgpu_select_plan_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
#define LOOM_AMDGPU_SELECT_IF(condition)                                 \
  do {                                                                   \
    if (condition) {                                                     \
      loom_amdgpu_set_selected_plan(source_op->kind, 0, NULL, out_plan); \
    }                                                                    \
    return iree_ok_status();                                             \
  } while (false)
#define LOOM_AMDGPU_SELECT_DESCRIPTOR(select_fn)                          \
  do {                                                                    \
    uint64_t descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE;                 \
    if (select_fn(context, source_op, &descriptor_id)) {                  \
      loom_amdgpu_set_selected_plan(source_op->kind, descriptor_id, NULL, \
                                    out_plan);                            \
    }                                                                     \
    return iree_ok_status();                                              \
  } while (false)
#define LOOM_AMDGPU_SELECT_DATA(plan_type, select_fn)                         \
  do {                                                                        \
    plan_type* plan_data = NULL;                                              \
    IREE_RETURN_IF_ERROR(loom_amdgpu_allocate_selected_plan_data(             \
        context, sizeof(*plan_data), (void**)&plan_data));                    \
    if (select_fn(context, source_op, plan_data)) {                           \
      loom_amdgpu_set_selected_plan(source_op->kind, 0, plan_data, out_plan); \
    }                                                                         \
    return iree_ok_status();                                                  \
  } while (false)
  switch (source_op->kind) {
    case LOOM_OP_INDEX_CONSTANT:
    case LOOM_OP_SCALAR_CONSTANT:
    case LOOM_OP_VECTOR_CONSTANT:
    case LOOM_OP_VECTOR_IOTA:
      return loom_amdgpu_select_value_plan(context, source_op, out_plan);
    case LOOM_OP_BUFFER_ALLOCA:
      LOOM_AMDGPU_SELECT_IF(
          loom_amdgpu_can_lower_buffer_alloca(context, source_op));
    case LOOM_OP_BUFFER_ASSUME_MEMORY_SPACE:
      loom_amdgpu_set_selected_plan(source_op->kind, 0, NULL, out_plan);
      return iree_ok_status();
    case LOOM_OP_BUFFER_VIEW:
      LOOM_AMDGPU_SELECT_IF(
          loom_amdgpu_can_lower_buffer_view(context, source_op));
    case LOOM_OP_KERNEL_WORKITEM_ID:
    case LOOM_OP_KERNEL_WORKGROUP_ID:
      return loom_amdgpu_select_preamble_plan(context, source_op, out_plan);
    case LOOM_OP_INDEX_ADD:
    case LOOM_OP_INDEX_SUB:
    case LOOM_OP_INDEX_MUL:
    case LOOM_OP_INDEX_MADD:
    case LOOM_OP_SCALAR_ADDI:
    case LOOM_OP_SCALAR_SUBI:
    case LOOM_OP_SCALAR_MULI:
    case LOOM_OP_SCALAR_ANDI:
    case LOOM_OP_SCALAR_ORI:
    case LOOM_OP_SCALAR_XORI:
    case LOOM_OP_SCALAR_SHLI:
    case LOOM_OP_SCALAR_SHRSI:
    case LOOM_OP_SCALAR_SHRUI:
      return loom_amdgpu_select_integer_plan(context, source_op, out_plan);
    case LOOM_OP_VECTOR_CMPI:
      LOOM_AMDGPU_SELECT_DESCRIPTOR(
          loom_amdgpu_select_vector_cmpi_descriptor_id);
    case LOOM_OP_VECTOR_CMPF:
      LOOM_AMDGPU_SELECT_DESCRIPTOR(
          loom_amdgpu_select_vector_cmpf_descriptor_id);
    case LOOM_OP_VECTOR_SELECT:
      LOOM_AMDGPU_SELECT_IF(
          loom_amdgpu_can_lower_vector_select(context, source_op));
    case LOOM_OP_VECTOR_TABLE_LOOKUP:
      LOOM_AMDGPU_SELECT_DATA(loom_amdgpu_table_lookup_plan_t,
                              loom_amdgpu_select_vector_table_lookup_plan);
    case LOOM_OP_VECTOR_BITFIELD_EXTRACTU:
    case LOOM_OP_VECTOR_BITFIELD_EXTRACTS:
      LOOM_AMDGPU_SELECT_DATA(loom_amdgpu_bitfield_extract_plan_t,
                              loom_amdgpu_select_vector_bitfield_extract_plan);
    case LOOM_OP_VECTOR_BITFIELD_INSERT:
      LOOM_AMDGPU_SELECT_DATA(loom_amdgpu_bitfield_insert_plan_t,
                              loom_amdgpu_select_vector_bitfield_insert_plan);
    case LOOM_OP_VECTOR_BITPACK:
      LOOM_AMDGPU_SELECT_DATA(loom_amdgpu_bitpack_plan_t,
                              loom_amdgpu_select_vector_bitpack_plan);
    case LOOM_OP_VECTOR_BITUNPACKS:
    case LOOM_OP_VECTOR_BITUNPACKU:
      LOOM_AMDGPU_SELECT_DATA(loom_amdgpu_bitunpack_plan_t,
                              loom_amdgpu_select_vector_bitunpack_plan);
    case LOOM_OP_VECTOR_BITCAST:
      LOOM_AMDGPU_SELECT_DATA(loom_amdgpu_vector_bitcast_plan_t,
                              loom_amdgpu_select_vector_bitcast_plan);
    case LOOM_OP_VECTOR_SLICE:
      LOOM_AMDGPU_SELECT_DATA(loom_amdgpu_vector_slice_plan_t,
                              loom_amdgpu_select_vector_slice_plan);
    case LOOM_OP_VECTOR_REDUCE:
      LOOM_AMDGPU_SELECT_DESCRIPTOR(
          loom_amdgpu_select_vector_reduce_descriptor_id);
    case LOOM_OP_VECTOR_DOTF:
      LOOM_AMDGPU_SELECT_IF(
          loom_amdgpu_can_lower_vector_dotf(context, source_op));
    case LOOM_OP_VECTOR_DOT4I:
      LOOM_AMDGPU_SELECT_IF(
          loom_amdgpu_can_lower_vector_dot4i(context, source_op));
    case LOOM_OP_VECTOR_DOT8I4:
      LOOM_AMDGPU_SELECT_IF(
          loom_amdgpu_can_lower_vector_dot8i4(context, source_op));
    case LOOM_OP_VECTOR_EXTRACT:
    case LOOM_OP_VECTOR_FROM_ELEMENTS:
      return loom_amdgpu_select_value_plan(context, source_op, out_plan);
    case LOOM_OP_VECTOR_LOAD:
      LOOM_AMDGPU_SELECT_DATA(loom_amdgpu_memory_access_plan_t,
                              loom_amdgpu_select_vector_load_plan);
    case LOOM_OP_VECTOR_STORE:
      LOOM_AMDGPU_SELECT_DATA(loom_amdgpu_memory_access_plan_t,
                              loom_amdgpu_select_vector_store_plan);
    default:
      return iree_ok_status();
  }
#undef LOOM_AMDGPU_SELECT_DATA
#undef LOOM_AMDGPU_SELECT_DESCRIPTOR
#undef LOOM_AMDGPU_SELECT_IF
}

static iree_status_t loom_amdgpu_select_op(void* user_data,
                                           loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_low_lower_plan_t* out_plan) {
  (void)user_data;
  IREE_ASSERT_ARGUMENT(out_plan);
  return loom_amdgpu_select_plan_id(context, source_op, out_plan);
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

static iree_status_t loom_amdgpu_emit_op(void* user_data,
                                         loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_low_lower_plan_t plan) {
  (void)user_data;
  switch (plan.id) {
    case LOOM_OP_INDEX_CONSTANT:
    case LOOM_OP_SCALAR_CONSTANT:
    case LOOM_OP_VECTOR_CONSTANT:
    case LOOM_OP_VECTOR_IOTA:
      return loom_amdgpu_lower_value_op(context, source_op, plan);
    case LOOM_OP_BUFFER_ALLOCA:
      return loom_amdgpu_lower_buffer_alloca(context, source_op);
    case LOOM_OP_BUFFER_ASSUME_MEMORY_SPACE:
      return loom_amdgpu_lower_buffer_assume_memory_space(context, source_op);
    case LOOM_OP_BUFFER_VIEW:
      return loom_amdgpu_lower_buffer_view(context, source_op);
    case LOOM_OP_KERNEL_WORKITEM_ID:
    case LOOM_OP_KERNEL_WORKGROUP_ID:
      return loom_amdgpu_lower_preamble_op(context, source_op);
    case LOOM_OP_INDEX_ADD:
    case LOOM_OP_INDEX_SUB:
    case LOOM_OP_INDEX_MUL:
    case LOOM_OP_INDEX_MADD:
    case LOOM_OP_SCALAR_ADDI:
    case LOOM_OP_SCALAR_SUBI:
    case LOOM_OP_SCALAR_MULI:
    case LOOM_OP_SCALAR_ANDI:
    case LOOM_OP_SCALAR_ORI:
    case LOOM_OP_SCALAR_XORI:
    case LOOM_OP_SCALAR_SHLI:
    case LOOM_OP_SCALAR_SHRSI:
    case LOOM_OP_SCALAR_SHRUI:
      return loom_amdgpu_lower_integer_op(context, source_op);
    case LOOM_OP_VECTOR_CMPI:
      return loom_amdgpu_lower_vector_cmpi(context, source_op, plan.payload);
    case LOOM_OP_VECTOR_CMPF:
      return loom_amdgpu_lower_vector_cmpf(context, source_op, plan.payload);
    case LOOM_OP_VECTOR_SELECT:
      return loom_amdgpu_lower_vector_select(context, source_op);
    case LOOM_OP_VECTOR_TABLE_LOOKUP:
      return loom_amdgpu_lower_vector_table_lookup(
          context, source_op,
          (const loom_amdgpu_table_lookup_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_BITFIELD_EXTRACTU:
    case LOOM_OP_VECTOR_BITFIELD_EXTRACTS:
      return loom_amdgpu_lower_vector_bitfield_extract(
          context, source_op,
          (const loom_amdgpu_bitfield_extract_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_BITFIELD_INSERT:
      return loom_amdgpu_lower_vector_bitfield_insert(
          context, source_op,
          (const loom_amdgpu_bitfield_insert_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_BITPACK:
      return loom_amdgpu_lower_vector_bitpack(
          context, source_op,
          (const loom_amdgpu_bitpack_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_BITUNPACKS:
    case LOOM_OP_VECTOR_BITUNPACKU:
      return loom_amdgpu_lower_vector_bitunpack(
          context, source_op,
          (const loom_amdgpu_bitunpack_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_BITCAST:
      return loom_amdgpu_lower_vector_bitcast(
          context, source_op,
          (const loom_amdgpu_vector_bitcast_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_SLICE:
      return loom_amdgpu_lower_vector_slice(
          context, source_op,
          (const loom_amdgpu_vector_slice_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_REDUCE:
      return loom_amdgpu_lower_vector_reduce(context, source_op, plan.payload);
    case LOOM_OP_VECTOR_DOTF:
      return loom_amdgpu_lower_vector_dotf(context, source_op);
    case LOOM_OP_VECTOR_DOT4I:
      return loom_amdgpu_lower_vector_dot4i(context, source_op);
    case LOOM_OP_VECTOR_DOT8I4:
      return loom_amdgpu_lower_vector_dot8i4(context, source_op);
    case LOOM_OP_VECTOR_EXTRACT:
    case LOOM_OP_VECTOR_FROM_ELEMENTS:
      return loom_amdgpu_lower_value_op(context, source_op, plan);
    case LOOM_OP_VECTOR_LOAD:
      return loom_amdgpu_lower_vector_load(
          context, source_op,
          (const loom_amdgpu_memory_access_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_STORE:
      return loom_amdgpu_lower_vector_store(
          context, source_op,
          (const loom_amdgpu_memory_access_plan_t*)plan.target_data);
    default:
      IREE_ASSERT_UNREACHABLE();
      return iree_ok_status();
  }
}

static iree_status_t loom_amdgpu_low_legality_verify_buffer_view(
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

static iree_status_t loom_amdgpu_low_legality_try_verify_op(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  *out_handled = false;
  switch (op->kind) {
    case LOOM_OP_BUFFER_VIEW:
      return loom_amdgpu_low_legality_verify_buffer_view(provider, context, op,
                                                         out_handled);
    case LOOM_OP_VECTOR_BITPACK:
    case LOOM_OP_VECTOR_BITUNPACKS:
    case LOOM_OP_VECTOR_BITUNPACKU:
      return loom_amdgpu_low_legality_verify_vector_bitstream(provider, context,
                                                              op, out_handled);
    case LOOM_OP_VECTOR_BITCAST:
    case LOOM_OP_VECTOR_SLICE:
      return loom_amdgpu_low_legality_verify_vector_structural(
          provider, context, op, out_handled);
    case LOOM_OP_VECTOR_TABLE_LOOKUP:
      return loom_amdgpu_low_legality_verify_vector_table(provider, context, op,
                                                          out_handled);
    case LOOM_OP_VECTOR_LOAD:
    case LOOM_OP_VECTOR_STORE:
      return loom_amdgpu_low_legality_verify_vector_memory(provider, context,
                                                           op, out_handled);
    default:
      if (loom_amdgpu_op_is_vector_dot(op->kind)) {
        return loom_amdgpu_low_legality_verify_vector_dot(provider, context, op,
                                                          out_handled);
      }
      return iree_ok_status();
  }
}

static const loom_low_lower_policy_t kAmdgpuLowLowerPolicy = {
    .name = IREE_SVL("amdgpu-register-lower"),
    .map_type = {.fn = loom_amdgpu_map_type, .user_data = NULL},
    .map_value = {.fn = loom_amdgpu_map_value, .user_data = NULL},
    .map_argument = {.fn = loom_amdgpu_map_argument, .user_data = NULL},
    .emit_preamble = {.fn = loom_amdgpu_emit_preamble, .user_data = NULL},
    .rule_set = &loom_amdgpu_arithmetic_rule_set,
    .select_op = {.fn = loom_amdgpu_select_op, .user_data = NULL},
    .emit_op = {.fn = loom_amdgpu_emit_op, .user_data = NULL},
};

const loom_target_low_legality_provider_t
    loom_amdgpu_low_legality_provider_storage = {
        .name = IREE_SVL("amdgpu"),
        .try_verify_op = loom_amdgpu_low_legality_try_verify_op,
};

const loom_low_lower_policy_t* loom_amdgpu_low_lower_policy(void) {
  return &kAmdgpuLowLowerPolicy;
}

const loom_target_low_legality_provider_t* loom_amdgpu_low_legality_provider(
    void) {
  return &loom_amdgpu_low_legality_provider_storage;
}

void loom_amdgpu_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("amdgpu.gfx950.core"),
          .policy = &kAmdgpuLowLowerPolicy,
      },
      {
          .contract_set_key = IREE_SVL("amdgpu.gfx11.core"),
          .policy = &kAmdgpuLowLowerPolicy,
      },
      {
          .contract_set_key = IREE_SVL("amdgpu.gfx12.core"),
          .policy = &kAmdgpuLowLowerPolicy,
      },
      {
          .contract_set_key = IREE_SVL("amdgpu.gfx1250.core"),
          .policy = &kAmdgpuLowLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
