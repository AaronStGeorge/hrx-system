// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/lower_internal.h"

static iree_status_t loom_amdgpu_select_plan_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
#define LOOM_AMDGPU_SELECT_IF(condition)                              \
  do {                                                                \
    if (condition) {                                                  \
      *out_plan = loom_low_lower_plan_make(source_op->kind, 0, NULL); \
    }                                                                 \
    return iree_ok_status();                                          \
  } while (false)
#define LOOM_AMDGPU_SELECT_DESCRIPTOR(select_fn)                          \
  do {                                                                    \
    uint64_t descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE;                 \
    if (select_fn(context, source_op, &descriptor_id)) {                  \
      *out_plan =                                                         \
          loom_low_lower_plan_make(source_op->kind, descriptor_id, NULL); \
    }                                                                     \
    return iree_ok_status();                                              \
  } while (false)
#define LOOM_AMDGPU_SELECT_DATA(plan_type, select_fn)                      \
  do {                                                                     \
    plan_type* plan_data = NULL;                                           \
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(                \
        context, sizeof(*plan_data), (void**)&plan_data));                 \
    if (select_fn(context, source_op, plan_data)) {                        \
      *out_plan = loom_low_lower_plan_make(source_op->kind, 0, plan_data); \
    }                                                                      \
    return iree_ok_status();                                               \
  } while (false)
  switch (source_op->kind) {
    case LOOM_OP_INDEX_CONSTANT:
    case LOOM_OP_SCALAR_CONSTANT:
    case LOOM_OP_VECTOR_CONSTANT:
    case LOOM_OP_VECTOR_IOTA:
      return loom_amdgpu_select_value_plan(context, source_op, out_plan);
    case LOOM_OP_BUFFER_ALLOCA:
    case LOOM_OP_BUFFER_ASSUME_MEMORY_SPACE:
    case LOOM_OP_BUFFER_VIEW:
      return loom_amdgpu_select_buffer_plan(context, source_op, out_plan);
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
      LOOM_AMDGPU_SELECT_DATA(loom_amdgpu_vector_compare_plan_t,
                              loom_amdgpu_select_vector_cmpi_plan);
    case LOOM_OP_VECTOR_CMPF:
      LOOM_AMDGPU_SELECT_DATA(loom_amdgpu_vector_compare_plan_t,
                              loom_amdgpu_select_vector_cmpf_plan);
    case LOOM_OP_VECTOR_SELECT:
      LOOM_AMDGPU_SELECT_DATA(loom_amdgpu_vector_select_plan_t,
                              loom_amdgpu_select_vector_select_plan);
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
    case LOOM_OP_VECTOR_DOT4I:
    case LOOM_OP_VECTOR_DOT8I4:
      LOOM_AMDGPU_SELECT_DATA(loom_amdgpu_dot_plan_t,
                              loom_amdgpu_select_vector_dot_plan);
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
    case LOOM_OP_BUFFER_ASSUME_MEMORY_SPACE:
    case LOOM_OP_BUFFER_VIEW:
      return loom_amdgpu_lower_buffer_op(context, source_op, plan);
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
      return loom_amdgpu_lower_vector_cmpi(
          context, source_op,
          (const loom_amdgpu_vector_compare_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_CMPF:
      return loom_amdgpu_lower_vector_cmpf(
          context, source_op,
          (const loom_amdgpu_vector_compare_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_SELECT:
      return loom_amdgpu_lower_vector_select(
          context, source_op,
          (const loom_amdgpu_vector_select_plan_t*)plan.target_data);
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
    case LOOM_OP_VECTOR_DOT4I:
    case LOOM_OP_VECTOR_DOT8I4:
      return loom_amdgpu_lower_vector_dot(
          context, source_op, (const loom_amdgpu_dot_plan_t*)plan.target_data);
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

static iree_status_t loom_amdgpu_low_legality_try_verify_op(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  *out_handled = false;
  switch (op->kind) {
    case LOOM_OP_BUFFER_VIEW:
      return loom_amdgpu_low_legality_verify_buffer(provider, context, op,
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
