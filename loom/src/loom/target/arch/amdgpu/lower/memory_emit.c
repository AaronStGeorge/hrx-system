// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/ops/cache.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/lower/internal.h"
#include "loom/target/arch/amdgpu/lower/memory_internal.h"

iree_status_t loom_amdgpu_emit_memory_vaddr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_plan_t* access,
    loom_value_id_t low_base_addr, loom_value_id_t* out_low_vaddr) {
  IREE_ASSERT_ARGUMENT(out_low_vaddr);
  *out_low_vaddr = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));

  loom_value_id_t low_accumulator = low_base_addr;
  for (uint8_t i = 0; i < access->source.dynamic_term_count; ++i) {
    switch (access->dynamic_term_kinds[i]) {
      case LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR:
        break;
      case LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET:
        continue;
      case LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_NONE:
        IREE_ASSERT_UNREACHABLE();
        continue;
    }
    const loom_low_source_memory_dynamic_term_t* term =
        &access->source.dynamic_terms[i];
    loom_value_id_t low_index = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_low_lower_lookup_value(context, term->index, &low_index));
    loom_value_id_t low_offset = low_index;
    if (term->byte_stride != 1) {
      const bool use_shift =
          term->byte_shift != LOOM_AMDGPU_MEMORY_ACCESS_BYTE_SHIFT_NONE;
      loom_value_id_t low_scale = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_MOV_B32,
          use_shift ? term->byte_shift : term->byte_stride, vgpr_type,
          &low_scale));
      loom_value_id_t operands[] = {
          use_shift ? low_scale : low_index,
          use_shift ? low_index : low_scale,
      };
      loom_op_t* low_offset_op = NULL;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
          context, source_op,
          use_shift ? LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHLREV_B32
                    : LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_LO_U32,
          operands, IREE_ARRAYSIZE(operands),
          loom_make_named_attr_slice(NULL, 0), &vgpr_type, 1, &low_offset_op));
      low_offset = loom_value_slice_get(loom_low_op_results(low_offset_op), 0);
    }
    if (low_accumulator == LOOM_VALUE_ID_INVALID) {
      low_accumulator = low_offset;
      continue;
    }
    loom_value_id_t add_operands[] = {low_accumulator, low_offset};
    loom_op_t* low_add_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32, add_operands,
        IREE_ARRAYSIZE(add_operands), loom_make_named_attr_slice(NULL, 0),
        &vgpr_type, 1, &low_add_op));
    low_accumulator = loom_value_slice_get(loom_low_op_results(low_add_op), 0);
  }

  if (low_accumulator == LOOM_VALUE_ID_INVALID) {
    return loom_amdgpu_emit_const_u32(context, source_op,
                                      LOOM_AMDGPU_DESCRIPTOR_ID_V_MOV_B32, 0,
                                      vgpr_type, out_low_vaddr);
  }
  *out_low_vaddr = low_accumulator;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_memory_soffset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_plan_t* access,
    loom_value_id_t* out_low_soffset) {
  IREE_ASSERT_ARGUMENT(out_low_soffset);
  return loom_amdgpu_emit_sgpr_byte_offset_terms(
      context, source_op, &access->source, access->dynamic_term_kinds,
      access->scalar_byte_offset, out_low_soffset);
}

iree_status_t loom_amdgpu_emit_memory_saddr(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op,
                                            loom_value_id_t low_resource,
                                            loom_value_id_t* out_low_saddr) {
  IREE_ASSERT_ARGUMENT(out_low_saddr);
  *out_low_saddr = LOOM_VALUE_ID_INVALID;
  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &sgpr_x2_type));
  return loom_amdgpu_emit_low_slice(context, source_op, low_resource,
                                    /*offset=*/0, sgpr_x2_type, out_low_saddr);
}

iree_status_t loom_amdgpu_emit_memory_flat_vaddr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_resource, loom_value_id_t* out_low_vaddr) {
  IREE_ASSERT_ARGUMENT(out_low_vaddr);
  *out_low_vaddr = LOOM_VALUE_ID_INVALID;

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_value_id_t low_base_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_resource, /*offset=*/0, sgpr_type, &low_base_lo));
  loom_value_id_t low_base_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_resource, /*offset=*/1, sgpr_type, &low_base_hi));

  loom_value_id_t low_vaddr_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_b32_copy(
      context, source_op, low_base_lo, &low_vaddr_lo));
  loom_value_id_t low_vaddr_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_b32_copy(
      context, source_op, low_base_hi, &low_vaddr_hi));

  loom_type_t vgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_vgpr_range_type(context, 2, &vgpr_x2_type));
  loom_value_id_t sources[] = {
      low_vaddr_lo,
      low_vaddr_hi,
  };
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), sources, IREE_ARRAYSIZE(sources),
      vgpr_x2_type, source_op->location, &concat_op));
  *out_low_vaddr = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_memory_cache_attrs(
    loom_low_lower_context_t* context,
    const loom_amdgpu_memory_access_plan_t* access, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* inout_attr_count) {
  const loom_vector_memory_cache_policy_t* policy =
      &access->source.cache_policy;
  if (!loom_amdgpu_memory_cache_policy_is_present(policy)) {
    return iree_ok_status();
  }
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  loom_amdgpu_memory_cache_policy_attrs_t cache_attrs;
  if (!loom_amdgpu_memory_cache_policy_encode(descriptor_set, access,
                                              &cache_attrs)) {
    return loom_amdgpu_memory_cache_policy_rejected_status(descriptor_set,
                                                           access, policy);
  }

  if (iree_any_bit_set(cache_attrs.flags,
                       LOOM_AMDGPU_MEMORY_CACHE_POLICY_ATTR_SCOPE)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
        context, IREE_SV("scope"), cache_attrs.scope, attrs, attr_capacity,
        inout_attr_count));
  }
  if (iree_any_bit_set(cache_attrs.flags,
                       LOOM_AMDGPU_MEMORY_CACHE_POLICY_ATTR_TH)) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_i64_attr(context, IREE_SV("th"), cache_attrs.th,
                                    attrs, attr_capacity, inout_attr_count));
  }
  if (iree_any_bit_set(cache_attrs.flags,
                       LOOM_AMDGPU_MEMORY_CACHE_POLICY_ATTR_NT)) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_i64_attr(context, IREE_SV("nt"), cache_attrs.nt,
                                    attrs, attr_capacity, inout_attr_count));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_make_memory_attrs(
    loom_low_lower_context_t* context,
    const loom_amdgpu_memory_access_plan_t* access, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* out_attr_count) {
  IREE_ASSERT_ARGUMENT(access);
  IREE_ASSERT_ARGUMENT(attrs);
  IREE_ASSERT_ARGUMENT(out_attr_count);
  *out_attr_count = 0;
  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_2ADDR) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
        context, IREE_SV("offset0"), access->immediate_offset, attrs,
        attr_capacity, out_attr_count));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
        context, IREE_SV("offset1"), access->secondary_immediate_offset, attrs,
        attr_capacity, out_attr_count));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
        context, IREE_SV("offset"), access->immediate_offset, attrs,
        attr_capacity, out_attr_count));
  }
  return loom_amdgpu_append_memory_cache_attrs(context, access, attrs,
                                               attr_capacity, out_attr_count);
}

iree_status_t loom_amdgpu_lower_vector_load(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_plan_t* access) {
  IREE_ASSERT_ARGUMENT(access);
  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_load_view(source_op), &low_resource));

  loom_value_id_t low_vaddr = LOOM_VALUE_ID_INVALID;
  if (access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO &&
      access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID) {
    const loom_value_id_t low_base_addr =
        access->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP
            ? low_resource
            : LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_vaddr(
        context, source_op, access, low_base_addr, &low_vaddr));
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, loom_vector_load_result(source_op), &result_type));

  loom_named_attr_t attrs[5] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_memory_attrs(
      context, access, attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  if (access->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID) {
      loom_value_id_t low_m0 = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_m0_live_in(context, &low_m0));
      loom_value_id_t operands[] = {low_m0};
      loom_op_t* low_op = NULL;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_low_op(context, source_op, access->descriptor_id,
                                  operands, IREE_ARRAYSIZE(operands),
                                  loom_make_named_attr_slice(attrs, attr_count),
                                  &result_type, 1, &low_op));
      return loom_low_lower_bind_value(
          context, loom_vector_load_result(source_op),
          loom_value_slice_get(loom_low_op_results(low_op), 0));
    }
    loom_value_id_t operands[] = {low_vaddr};
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, access->descriptor_id, operands,
        IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
        &result_type, 1, &low_op));
    return loom_low_lower_bind_value(
        context, loom_vector_load_result(source_op),
        loom_value_slice_get(loom_low_op_results(low_op), 0));
  }

  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR) {
    loom_value_id_t low_saddr = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_saddr(
        context, source_op, low_resource, &low_saddr));
    loom_value_id_t operands[] = {
        low_vaddr,
        low_saddr,
    };
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, access->descriptor_id, operands,
        IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
        &result_type, 1, &low_op));
    return loom_low_lower_bind_value(
        context, loom_vector_load_result(source_op),
        loom_value_slice_get(loom_low_op_results(low_op), 0));
  }

  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO) {
    loom_value_id_t operands[] = {low_resource};
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, access->descriptor_id, operands,
        IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
        &result_type, 1, &low_op));
    return loom_low_lower_bind_value(
        context, loom_vector_load_result(source_op),
        loom_value_slice_get(loom_low_op_results(low_op), 0));
  }

  loom_value_id_t low_soffset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_soffset(context, source_op,
                                                       access, &low_soffset));
  loom_value_id_t operands[] = {
      low_resource,
      low_vaddr,
      low_soffset,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, access->descriptor_id, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
      &result_type, 1, &low_op));
  return loom_low_lower_bind_value(
      context, loom_vector_load_result(source_op),
      loom_value_slice_get(loom_low_op_results(low_op), 0));
}

iree_status_t loom_amdgpu_lower_vector_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_plan_t* access) {
  IREE_ASSERT_ARGUMENT(access);
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_store_value(source_op), &low_value));
  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_store_view(source_op), &low_resource));

  loom_value_id_t low_vaddr = LOOM_VALUE_ID_INVALID;
  if (access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO &&
      access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID) {
    const loom_value_id_t low_base_addr =
        access->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP
            ? low_resource
            : LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_vaddr(
        context, source_op, access, low_base_addr, &low_vaddr));
  }

  loom_named_attr_t attrs[5] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_memory_attrs(
      context, access, attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  if (access->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID) {
      loom_value_id_t low_m0 = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_m0_live_in(context, &low_m0));
      loom_value_id_t operands[] = {
          low_value,
          low_m0,
      };
      loom_op_t* low_op = NULL;
      return loom_amdgpu_emit_low_op(
          context, source_op, access->descriptor_id, operands,
          IREE_ARRAYSIZE(operands),
          loom_make_named_attr_slice(attrs, attr_count),
          /*result_types=*/NULL, /*result_count=*/0, &low_op);
    }
    if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_2ADDR) {
      loom_type_t lane_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
      loom_value_id_t low_value0 = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_value, 0, lane_type, &low_value0));
      loom_value_id_t low_value1 = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_value, 1, lane_type, &low_value1));
      loom_value_id_t operands[] = {
          low_vaddr,
          low_value0,
          low_value1,
      };
      loom_op_t* low_op = NULL;
      return loom_amdgpu_emit_low_op(
          context, source_op, access->descriptor_id, operands,
          IREE_ARRAYSIZE(operands),
          loom_make_named_attr_slice(attrs, attr_count),
          /*result_types=*/NULL, /*result_count=*/0, &low_op);
    }
    loom_value_id_t operands[] = {
        low_vaddr,
        low_value,
    };
    loom_op_t* low_op = NULL;
    return loom_amdgpu_emit_low_op(
        context, source_op, access->descriptor_id, operands,
        IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
        /*result_types=*/NULL, /*result_count=*/0, &low_op);
  }

  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO) {
    loom_value_id_t operands[] = {
        low_value,
        low_resource,
    };
    loom_op_t* low_op = NULL;
    return loom_amdgpu_emit_low_op(
        context, source_op, access->descriptor_id, operands,
        IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
        /*result_types=*/NULL, /*result_count=*/0, &low_op);
  }

  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR) {
    loom_value_id_t low_saddr = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_saddr(
        context, source_op, low_resource, &low_saddr));
    loom_value_id_t operands[] = {
        low_vaddr,
        low_value,
        low_saddr,
    };
    loom_op_t* low_op = NULL;
    return loom_amdgpu_emit_low_op(
        context, source_op, access->descriptor_id, operands,
        IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
        /*result_types=*/NULL, /*result_count=*/0, &low_op);
  }

  loom_value_id_t low_soffset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_soffset(context, source_op,
                                                       access, &low_soffset));
  loom_value_id_t operands[] = {
      low_value,
      low_resource,
      low_vaddr,
      low_soffset,
  };
  loom_op_t* low_op = NULL;
  return loom_amdgpu_emit_low_op(
      context, source_op, access->descriptor_id, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
      /*result_types=*/NULL, /*result_count=*/0, &low_op);
}
