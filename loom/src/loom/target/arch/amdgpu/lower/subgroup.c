// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/subgroup.h"

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ops/kernel/ops.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/target_refs.h"

#define LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS 6u

static bool loom_amdgpu_subgroup_wavefront_size_is_supported(
    uint32_t wavefront_size) {
  return wavefront_size == 32 || wavefront_size == 64;
}

static uint32_t loom_amdgpu_subgroup_u32_log2(uint32_t value) {
  uint32_t log2 = 0;
  while (value > 1) {
    value >>= 1;
    ++log2;
  }
  return log2;
}

static bool loom_amdgpu_subgroup_exact_i32_value(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, int64_t* out_value) {
  *out_value = 0;

  int64_t fact_value = 0;
  if (loom_value_facts_as_exact_i64(
          loom_value_fact_table_lookup(fact_table, value_id), &fact_value) &&
      fact_value >= INT32_MIN && fact_value <= INT32_MAX) {
    *out_value = fact_value;
    return true;
  }

  return loom_amdgpu_module_value_as_i32_constant(module, value_id, out_value);
}

static loom_amdgpu_subgroup_payload_kind_t loom_amdgpu_subgroup_payload_kind(
    loom_type_t type, uint32_t* out_register_count) {
  *out_register_count = 0;
  if (loom_amdgpu_type_is_i32(type)) {
    *out_register_count = 1;
    return LOOM_AMDGPU_SUBGROUP_PAYLOAD_I32_SCALAR;
  }
  if (loom_amdgpu_type_is_f32(type)) {
    *out_register_count = 1;
    return LOOM_AMDGPU_SUBGROUP_PAYLOAD_F32_SCALAR;
  }
  const uint32_t i32_lane_count = loom_amdgpu_vector_i32_lane_count(type);
  if (i32_lane_count != 0) {
    *out_register_count = i32_lane_count;
    return LOOM_AMDGPU_SUBGROUP_PAYLOAD_I32_VECTOR;
  }
  const uint32_t f32_lane_count = loom_amdgpu_vector_f32_lane_count(type);
  if (f32_lane_count != 0) {
    *out_register_count = f32_lane_count;
    return LOOM_AMDGPU_SUBGROUP_PAYLOAD_F32_VECTOR;
  }
  return LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
}

static bool loom_amdgpu_subgroup_payload_is_supported(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_amdgpu_subgroup_payload_kind_t* out_kind,
    uint32_t* out_register_count) {
  *out_kind = LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  const loom_type_t type = loom_module_value_type(module, value_id);
  *out_kind = loom_amdgpu_subgroup_payload_kind(type, out_register_count);
  return *out_kind != LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
}

static bool loom_amdgpu_subgroup_payload_is_integer(
    loom_amdgpu_subgroup_payload_kind_t payload_kind) {
  return payload_kind == LOOM_AMDGPU_SUBGROUP_PAYLOAD_I32_SCALAR ||
         payload_kind == LOOM_AMDGPU_SUBGROUP_PAYLOAD_I32_VECTOR;
}

static bool loom_amdgpu_subgroup_payload_is_float(
    loom_amdgpu_subgroup_payload_kind_t payload_kind) {
  return payload_kind == LOOM_AMDGPU_SUBGROUP_PAYLOAD_F32_SCALAR ||
         payload_kind == LOOM_AMDGPU_SUBGROUP_PAYLOAD_F32_VECTOR;
}

static bool loom_amdgpu_subgroup_optional_attr_is_present(const loom_op_t* op,
                                                          uint16_t attr_index) {
  return attr_index < op->attribute_count &&
         !loom_attr_is_absent(loom_op_attrs(op)[attr_index]);
}

static bool loom_amdgpu_subgroup_reduce_has_cluster_attrs(const loom_op_t* op) {
  return loom_amdgpu_subgroup_optional_attr_is_present(
             op, loom_kernel_subgroup_reduce_cluster_size_ATTR_INDEX) ||
         loom_amdgpu_subgroup_optional_attr_is_present(
             op, loom_kernel_subgroup_reduce_cluster_stride_ATTR_INDEX);
}

static bool loom_amdgpu_subgroup_scan_has_cluster_attrs(const loom_op_t* op) {
  return loom_amdgpu_subgroup_optional_attr_is_present(
             op, loom_kernel_subgroup_scan_cluster_size_ATTR_INDEX) ||
         loom_amdgpu_subgroup_optional_attr_is_present(
             op, loom_kernel_subgroup_scan_cluster_stride_ATTR_INDEX);
}

static bool loom_amdgpu_subgroup_full_wave_workgroups(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle, uint32_t wavefront_size) {
  uint32_t flat_workgroup_size = 0;
  return loom_amdgpu_required_flat_workgroup_size(module, function, bundle,
                                                  &flat_workgroup_size) &&
         flat_workgroup_size >= wavefront_size &&
         (flat_workgroup_size % wavefront_size) == 0;
}

static bool loom_amdgpu_subgroup_reduce_active_lane_count(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle, uint32_t wavefront_size,
    uint32_t* out_active_lane_count) {
  *out_active_lane_count = 0;
  uint32_t flat_workgroup_size = 0;
  if (!loom_amdgpu_required_flat_workgroup_size(module, function, bundle,
                                                &flat_workgroup_size) ||
      flat_workgroup_size == 0) {
    return false;
  }
  if (flat_workgroup_size <= wavefront_size) {
    *out_active_lane_count = flat_workgroup_size;
    return true;
  }
  if ((flat_workgroup_size % wavefront_size) != 0) {
    return false;
  }
  *out_active_lane_count = wavefront_size;
  return true;
}

static bool loom_amdgpu_subgroup_combine_descriptor_ref(
    loom_combining_kind_t kind,
    loom_amdgpu_subgroup_payload_kind_t payload_kind,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref) {
  *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  switch (kind) {
    case LOOM_COMBINING_KIND_ADDI:
      if (!loom_amdgpu_subgroup_payload_is_integer(payload_kind)) {
        return false;
      }
      *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32;
      return true;
    case LOOM_COMBINING_KIND_MULI:
      if (!loom_amdgpu_subgroup_payload_is_integer(payload_kind)) {
        return false;
      }
      *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_LO_U32;
      return true;
    case LOOM_COMBINING_KIND_MINSI:
      if (!loom_amdgpu_subgroup_payload_is_integer(payload_kind)) {
        return false;
      }
      *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_I32;
      return true;
    case LOOM_COMBINING_KIND_MAXSI:
      if (!loom_amdgpu_subgroup_payload_is_integer(payload_kind)) {
        return false;
      }
      *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MAX_I32;
      return true;
    case LOOM_COMBINING_KIND_MINUI:
      if (!loom_amdgpu_subgroup_payload_is_integer(payload_kind)) {
        return false;
      }
      *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_U32;
      return true;
    case LOOM_COMBINING_KIND_MAXUI:
      if (!loom_amdgpu_subgroup_payload_is_integer(payload_kind)) {
        return false;
      }
      *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MAX_U32;
      return true;
    case LOOM_COMBINING_KIND_ANDI:
      if (!loom_amdgpu_subgroup_payload_is_integer(payload_kind)) {
        return false;
      }
      *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32;
      return true;
    case LOOM_COMBINING_KIND_ORI:
      if (!loom_amdgpu_subgroup_payload_is_integer(payload_kind)) {
        return false;
      }
      *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32;
      return true;
    case LOOM_COMBINING_KIND_XORI:
      if (!loom_amdgpu_subgroup_payload_is_integer(payload_kind)) {
        return false;
      }
      *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_XOR_B32;
      return true;
    case LOOM_COMBINING_KIND_ADDF:
      if (!loom_amdgpu_subgroup_payload_is_float(payload_kind)) {
        return false;
      }
      *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_F32;
      return true;
    case LOOM_COMBINING_KIND_MULF:
      if (!loom_amdgpu_subgroup_payload_is_float(payload_kind)) {
        return false;
      }
      *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32;
      return true;
    case LOOM_COMBINING_KIND_MINNUMF:
      if (!loom_amdgpu_subgroup_payload_is_float(payload_kind)) {
        return false;
      }
      *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_F32;
      return true;
    case LOOM_COMBINING_KIND_MAXNUMF:
      if (!loom_amdgpu_subgroup_payload_is_float(payload_kind)) {
        return false;
      }
      *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MAX_F32;
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_subgroup_combine_identity_bits(
    loom_combining_kind_t kind, uint32_t* out_bits) {
  *out_bits = 0;
  switch (kind) {
    case LOOM_COMBINING_KIND_ADDI:
    case LOOM_COMBINING_KIND_ORI:
    case LOOM_COMBINING_KIND_XORI:
    case LOOM_COMBINING_KIND_ADDF:
      *out_bits = 0u;
      return true;
    case LOOM_COMBINING_KIND_MULI:
      *out_bits = 1u;
      return true;
    case LOOM_COMBINING_KIND_MINSI:
      *out_bits = (uint32_t)INT32_MAX;
      return true;
    case LOOM_COMBINING_KIND_MAXSI:
      *out_bits = (uint32_t)INT32_MIN;
      return true;
    case LOOM_COMBINING_KIND_MINUI:
      *out_bits = UINT32_MAX;
      return true;
    case LOOM_COMBINING_KIND_MAXUI:
      *out_bits = 0u;
      return true;
    case LOOM_COMBINING_KIND_ANDI:
      *out_bits = UINT32_MAX;
      return true;
    case LOOM_COMBINING_KIND_MULF:
      *out_bits = 0x3f800000u;
      return true;
    case LOOM_COMBINING_KIND_MINNUMF:
      *out_bits = 0x7f800000u;
      return true;
    case LOOM_COMBINING_KIND_MAXNUMF:
      *out_bits = 0xff800000u;
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_descriptor_set_has_ref(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref) {
  return loom_amdgpu_descriptor_ref_ordinal(descriptor_set, descriptor_ref) !=
         LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
}

static bool loom_amdgpu_subgroup_mask_bit_count(const loom_module_t* module,
                                                loom_value_id_t value_id,
                                                uint32_t* out_bit_count) {
  *out_bit_count = 0;
  const loom_type_t type = loom_module_value_type(module, value_id);
  if (loom_amdgpu_type_is_i32(type)) {
    *out_bit_count = 32;
    return true;
  }
  if (loom_amdgpu_type_is_i64(type)) {
    *out_bit_count = 64;
    return true;
  }
  return false;
}

static bool loom_amdgpu_subgroup_mask_covers_wavefront(
    uint32_t mask_bit_count, uint32_t wavefront_size) {
  return mask_bit_count >= wavefront_size;
}

iree_status_t loom_amdgpu_select_kernel_subgroup_broadcast_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_broadcast_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_subgroup_broadcast_plan_t){0};
  *out_selected = false;
  if (!loom_kernel_subgroup_broadcast_isa(source_op)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t value = loom_kernel_subgroup_broadcast_value(source_op);
  loom_amdgpu_subgroup_payload_kind_t payload_kind =
      LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  uint32_t register_count = 0;
  if (!loom_amdgpu_subgroup_payload_is_supported(module, value, &payload_kind,
                                                 &register_count)) {
    return iree_ok_status();
  }

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_wavefront_size(
      loom_low_lower_context_bundle(context), &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size)) {
    return iree_ok_status();
  }

  int64_t source_lane = 0;
  if (!loom_amdgpu_subgroup_exact_i32_value(
          module, loom_low_lower_context_fact_table(context),
          loom_kernel_subgroup_broadcast_lane(source_op), &source_lane) ||
      source_lane < 0 || source_lane >= (int64_t)wavefront_size) {
    return iree_ok_status();
  }

  bool descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32,
      &out_plan->descriptor, &descriptor_present));
  if (!descriptor_present) {
    return iree_ok_status();
  }

  out_plan->value = value;
  out_plan->result = loom_kernel_subgroup_broadcast_result(source_op);
  out_plan->payload_kind = payload_kind;
  out_plan->register_count = register_count;
  out_plan->source_lane = (uint32_t)source_lane;
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_kernel_subgroup_broadcast_first_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_broadcast_first_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_subgroup_broadcast_first_plan_t){0};
  *out_selected = false;
  if (!loom_kernel_subgroup_broadcast_first_isa(source_op)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t value =
      loom_kernel_subgroup_broadcast_first_value(source_op);
  loom_amdgpu_subgroup_payload_kind_t payload_kind =
      LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  uint32_t register_count = 0;
  if (!loom_amdgpu_subgroup_payload_is_supported(module, value, &payload_kind,
                                                 &register_count)) {
    return iree_ok_status();
  }

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_wavefront_size(
      loom_low_lower_context_bundle(context), &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size)) {
    return iree_ok_status();
  }

  bool descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_READFIRSTLANE_B32,
      &out_plan->descriptor, &descriptor_present));
  if (!descriptor_present) {
    return iree_ok_status();
  }

  out_plan->value = value;
  out_plan->result = loom_kernel_subgroup_broadcast_first_result(source_op);
  out_plan->payload_kind = payload_kind;
  out_plan->register_count = register_count;
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_kernel_subgroup_shuffle_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_shuffle_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_subgroup_shuffle_plan_t){0};
  *out_selected = false;
  if (!loom_kernel_subgroup_shuffle_isa(source_op)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t value = loom_kernel_subgroup_shuffle_value(source_op);
  loom_amdgpu_subgroup_payload_kind_t payload_kind =
      LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  uint32_t register_count = 0;
  if (!loom_amdgpu_subgroup_payload_is_supported(module, value, &payload_kind,
                                                 &register_count)) {
    return iree_ok_status();
  }

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_wavefront_size(
      loom_low_lower_context_bundle(context), &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size)) {
    return iree_ok_status();
  }

  int64_t width = 0;
  if (!loom_amdgpu_subgroup_exact_i32_value(
          module, loom_low_lower_context_fact_table(context),
          loom_kernel_subgroup_shuffle_width(source_op), &width) ||
      width != (int64_t)wavefront_size) {
    return iree_ok_status();
  }

  int64_t offset = 0;
  if (!loom_amdgpu_subgroup_exact_i32_value(
          module, loom_low_lower_context_fact_table(context),
          loom_kernel_subgroup_shuffle_offset(source_op), &offset) ||
      offset < 0 || offset >= (int64_t)wavefront_size) {
    return iree_ok_status();
  }

  bool descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32,
      &out_plan->descriptor, &descriptor_present));
  if (!descriptor_present) {
    return iree_ok_status();
  }

  out_plan->value = value;
  out_plan->result = loom_kernel_subgroup_shuffle_result(source_op);
  out_plan->valid = loom_kernel_subgroup_shuffle_valid(source_op);
  out_plan->payload_kind = payload_kind;
  out_plan->register_count = register_count;
  out_plan->mode = loom_kernel_subgroup_shuffle_mode(source_op);
  out_plan->offset = (uint32_t)offset;
  out_plan->width = (uint32_t)width;
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_kernel_subgroup_reduce_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_reduce_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_subgroup_reduce_plan_t){0};
  *out_selected = false;
  if (!loom_kernel_subgroup_reduce_isa(source_op)) {
    return iree_ok_status();
  }
  if (loom_amdgpu_subgroup_reduce_has_cluster_attrs(source_op)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t value = loom_kernel_subgroup_reduce_value(source_op);
  loom_amdgpu_subgroup_payload_kind_t payload_kind =
      LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  uint32_t register_count = 0;
  if (!loom_amdgpu_subgroup_payload_is_supported(module, value, &payload_kind,
                                                 &register_count)) {
    return iree_ok_status();
  }

  const loom_combining_kind_t kind =
      loom_kernel_subgroup_reduce_kind(source_op);
  loom_amdgpu_descriptor_ref_t combine_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (!loom_amdgpu_subgroup_combine_descriptor_ref(kind, payload_kind,
                                                   &combine_descriptor_ref)) {
    return iree_ok_status();
  }

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_wavefront_size(
      loom_low_lower_context_bundle(context), &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size)) {
    return iree_ok_status();
  }
  uint32_t active_lane_count = 0;
  if (!loom_amdgpu_subgroup_reduce_active_lane_count(
          module, loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context), wavefront_size,
          &active_lane_count)) {
    return iree_ok_status();
  }

  bool bpermute_descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32,
      &out_plan->bpermute_descriptor, &bpermute_descriptor_present));
  if (!bpermute_descriptor_present) {
    return iree_ok_status();
  }

  bool combine_descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, combine_descriptor_ref, &out_plan->combine_descriptor,
      &combine_descriptor_present));
  if (!combine_descriptor_present) {
    return iree_ok_status();
  }

  uint32_t identity_bits = 0;
  if (!loom_amdgpu_u32_is_power_of_two(active_lane_count)) {
    if (!loom_amdgpu_subgroup_combine_identity_bits(kind, &identity_bits)) {
      return iree_ok_status();
    }
    bool guard_descriptor_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32,
        &out_plan->guard_descriptor, &guard_descriptor_present));
    if (!guard_descriptor_present) {
      return iree_ok_status();
    }

    bool select_descriptor_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
        &out_plan->select_descriptor, &select_descriptor_present));
    if (!select_descriptor_present) {
      return iree_ok_status();
    }
  }

  out_plan->value = value;
  out_plan->result = loom_kernel_subgroup_reduce_result(source_op);
  out_plan->payload_kind = payload_kind;
  out_plan->register_count = register_count;
  out_plan->wavefront_size = wavefront_size;
  out_plan->active_lane_count = active_lane_count;
  out_plan->identity_bits = identity_bits;
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_kernel_workgroup_reduce_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_workgroup_reduce_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_workgroup_reduce_plan_t){0};
  *out_selected = false;
  if (!loom_kernel_workgroup_reduce_isa(source_op)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t value = loom_kernel_workgroup_reduce_value(source_op);
  loom_amdgpu_subgroup_payload_kind_t payload_kind =
      LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  uint32_t register_count = 0;
  if (!loom_amdgpu_subgroup_payload_is_supported(module, value, &payload_kind,
                                                 &register_count)) {
    return iree_ok_status();
  }

  const loom_combining_kind_t kind =
      loom_kernel_workgroup_reduce_kind(source_op);
  loom_amdgpu_descriptor_ref_t combine_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (!loom_amdgpu_subgroup_combine_descriptor_ref(kind, payload_kind,
                                                   &combine_descriptor_ref)) {
    return iree_ok_status();
  }

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_wavefront_size(
      loom_low_lower_context_bundle(context), &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size)) {
    return iree_ok_status();
  }
  uint32_t flat_workgroup_size = 0;
  if (!loom_amdgpu_required_flat_workgroup_size(
          module, loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context), &flat_workgroup_size) ||
      flat_workgroup_size == 0) {
    return iree_ok_status();
  }
  const bool has_partial_tail = flat_workgroup_size > wavefront_size &&
                                (flat_workgroup_size % wavefront_size) != 0;
  const uint32_t wave_count =
      (flat_workgroup_size + wavefront_size - 1) / wavefront_size;
  if (flat_workgroup_size > wavefront_size && wave_count > wavefront_size) {
    return iree_ok_status();
  }
  const bool needs_cross_wave_identity =
      flat_workgroup_size > wavefront_size &&
      (has_partial_tail || !loom_amdgpu_u32_is_power_of_two(wave_count));
  const bool needs_subgroup_identity =
      flat_workgroup_size < wavefront_size &&
      !loom_amdgpu_u32_is_power_of_two(flat_workgroup_size);
  const bool needs_identity_guard =
      needs_subgroup_identity || needs_cross_wave_identity;
  const uint32_t scratch_slot_count =
      has_partial_tail ? flat_workgroup_size : wave_count;
  const uint64_t scratch_byte_length =
      (uint64_t)scratch_slot_count * register_count * 4u;
  if (scratch_byte_length > UINT32_MAX) {
    return iree_ok_status();
  }

  bool bpermute_descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32,
      &out_plan->bpermute_descriptor, &bpermute_descriptor_present));
  if (!bpermute_descriptor_present) {
    return iree_ok_status();
  }

  bool combine_descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, combine_descriptor_ref, &out_plan->combine_descriptor,
      &combine_descriptor_present));
  if (!combine_descriptor_present) {
    return iree_ok_status();
  }

  uint32_t identity_bits = 0;
  if (needs_identity_guard) {
    if (!loom_amdgpu_subgroup_combine_identity_bits(kind, &identity_bits)) {
      return iree_ok_status();
    }
  }
  if (needs_identity_guard || flat_workgroup_size > wavefront_size) {
    bool guard_descriptor_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32,
        &out_plan->guard_descriptor, &guard_descriptor_present));
    if (!guard_descriptor_present) {
      return iree_ok_status();
    }
  }

  if (needs_identity_guard) {
    bool select_descriptor_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
        &out_plan->select_descriptor, &select_descriptor_present));
    if (!select_descriptor_present) {
      return iree_ok_status();
    }
  }

  if (flat_workgroup_size > wavefront_size) {
    bool lds_read_descriptor_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B32,
        &out_plan->lds_read_descriptor, &lds_read_descriptor_present));
    if (!lds_read_descriptor_present) {
      return iree_ok_status();
    }

    bool lds_write_descriptor_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B32,
        &out_plan->lds_write_descriptor, &lds_write_descriptor_present));
    if (!lds_write_descriptor_present) {
      return iree_ok_status();
    }

    bool barrier_descriptor_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_S_BARRIER,
        &out_plan->barrier_descriptor, &barrier_descriptor_present));
    if (!barrier_descriptor_present) {
      return iree_ok_status();
    }

    bool saveexec_descriptor_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64,
        &out_plan->saveexec_descriptor, &saveexec_descriptor_present));
    if (!saveexec_descriptor_present) {
      return iree_ok_status();
    }

    bool restore_exec_descriptor_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
        &out_plan->restore_exec_descriptor, &restore_exec_descriptor_present));
    if (!restore_exec_descriptor_present) {
      return iree_ok_status();
    }
  }

  out_plan->value = value;
  out_plan->result = loom_kernel_workgroup_reduce_result(source_op);
  out_plan->payload_kind = payload_kind;
  out_plan->register_count = register_count;
  out_plan->wavefront_size = wavefront_size;
  out_plan->flat_workgroup_size = flat_workgroup_size;
  out_plan->identity_bits = identity_bits;
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_kernel_subgroup_scan_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_scan_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_subgroup_scan_plan_t){0};
  *out_selected = false;
  if (!loom_kernel_subgroup_scan_isa(source_op)) {
    return iree_ok_status();
  }
  if (loom_amdgpu_subgroup_scan_has_cluster_attrs(source_op)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t value = loom_kernel_subgroup_scan_value(source_op);
  loom_amdgpu_subgroup_payload_kind_t payload_kind =
      LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  uint32_t register_count = 0;
  if (!loom_amdgpu_subgroup_payload_is_supported(module, value, &payload_kind,
                                                 &register_count)) {
    return iree_ok_status();
  }

  const loom_combining_kind_t kind = loom_kernel_subgroup_scan_kind(source_op);
  loom_amdgpu_descriptor_ref_t combine_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (!loom_amdgpu_subgroup_combine_descriptor_ref(kind, payload_kind,
                                                   &combine_descriptor_ref)) {
    return iree_ok_status();
  }
  const loom_kernel_subgroup_scan_mode_t mode =
      loom_kernel_subgroup_scan_mode(source_op);
  switch (mode) {
    case LOOM_KERNEL_SUBGROUP_SCAN_MODE_INCLUSIVE:
      break;
    case LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE: {
      uint32_t unused_identity_bits = 0;
      if (!loom_amdgpu_subgroup_combine_identity_bits(kind,
                                                      &unused_identity_bits)) {
        return iree_ok_status();
      }
      break;
    }
    case LOOM_KERNEL_SUBGROUP_SCAN_MODE_COUNT_:
      return iree_ok_status();
  }

  loom_amdgpu_descriptor_ref_t guard_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  const loom_kernel_subgroup_scan_direction_t direction =
      loom_kernel_subgroup_scan_direction(source_op);
  switch (direction) {
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD:
      guard_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32;
      break;
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_REVERSE:
      guard_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32;
      break;
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_COUNT_:
      return iree_ok_status();
  }

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_wavefront_size(
      loom_low_lower_context_bundle(context), &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size) ||
      !loom_amdgpu_subgroup_full_wave_workgroups(
          module, loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context), wavefront_size)) {
    return iree_ok_status();
  }

  bool bpermute_descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32,
      &out_plan->bpermute_descriptor, &bpermute_descriptor_present));
  if (!bpermute_descriptor_present) {
    return iree_ok_status();
  }

  bool combine_descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, combine_descriptor_ref, &out_plan->combine_descriptor,
      &combine_descriptor_present));
  if (!combine_descriptor_present) {
    return iree_ok_status();
  }

  bool guard_descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, guard_descriptor_ref, &out_plan->guard_descriptor,
      &guard_descriptor_present));
  if (!guard_descriptor_present) {
    return iree_ok_status();
  }

  bool select_descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
      &out_plan->select_descriptor, &select_descriptor_present));
  if (!select_descriptor_present) {
    return iree_ok_status();
  }

  out_plan->value = value;
  out_plan->result = loom_kernel_subgroup_scan_result(source_op);
  out_plan->payload_kind = payload_kind;
  out_plan->register_count = register_count;
  out_plan->kind = kind;
  out_plan->mode = mode;
  out_plan->direction = direction;
  out_plan->wavefront_size = wavefront_size;
  out_plan->active_lane_count = wavefront_size;
  *out_selected = true;
  return iree_ok_status();
}

static bool loom_amdgpu_map_workgroup_scan_mode(
    loom_kernel_workgroup_scan_mode_t source_mode,
    loom_kernel_subgroup_scan_mode_t* out_mode) {
  switch (source_mode) {
    case LOOM_KERNEL_WORKGROUP_SCAN_MODE_INCLUSIVE:
      *out_mode = LOOM_KERNEL_SUBGROUP_SCAN_MODE_INCLUSIVE;
      return true;
    case LOOM_KERNEL_WORKGROUP_SCAN_MODE_EXCLUSIVE:
      *out_mode = LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE;
      return true;
    case LOOM_KERNEL_WORKGROUP_SCAN_MODE_COUNT_:
      return false;
  }
  return false;
}

static bool loom_amdgpu_map_workgroup_scan_direction(
    loom_kernel_workgroup_scan_direction_t source_direction,
    loom_kernel_subgroup_scan_direction_t* out_direction) {
  switch (source_direction) {
    case LOOM_KERNEL_WORKGROUP_SCAN_DIRECTION_FORWARD:
      *out_direction = LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD;
      return true;
    case LOOM_KERNEL_WORKGROUP_SCAN_DIRECTION_REVERSE:
      *out_direction = LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_REVERSE;
      return true;
    case LOOM_KERNEL_WORKGROUP_SCAN_DIRECTION_COUNT_:
      return false;
  }
  return false;
}

iree_status_t loom_amdgpu_select_kernel_workgroup_scan_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_scan_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_subgroup_scan_plan_t){0};
  *out_selected = false;
  if (!loom_kernel_workgroup_scan_isa(source_op)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t value = loom_kernel_workgroup_scan_value(source_op);
  loom_amdgpu_subgroup_payload_kind_t payload_kind =
      LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  uint32_t register_count = 0;
  if (!loom_amdgpu_subgroup_payload_is_supported(module, value, &payload_kind,
                                                 &register_count)) {
    return iree_ok_status();
  }

  const loom_combining_kind_t kind = loom_kernel_workgroup_scan_kind(source_op);
  loom_amdgpu_descriptor_ref_t combine_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (!loom_amdgpu_subgroup_combine_descriptor_ref(kind, payload_kind,
                                                   &combine_descriptor_ref)) {
    return iree_ok_status();
  }

  loom_kernel_subgroup_scan_mode_t mode = LOOM_KERNEL_SUBGROUP_SCAN_MODE_COUNT_;
  if (!loom_amdgpu_map_workgroup_scan_mode(
          loom_kernel_workgroup_scan_mode(source_op), &mode)) {
    return iree_ok_status();
  }
  if (mode == LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE) {
    uint32_t unused_identity_bits = 0;
    if (!loom_amdgpu_subgroup_combine_identity_bits(kind,
                                                    &unused_identity_bits)) {
      return iree_ok_status();
    }
  }

  loom_kernel_subgroup_scan_direction_t direction =
      LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_COUNT_;
  if (!loom_amdgpu_map_workgroup_scan_direction(
          loom_kernel_workgroup_scan_direction(source_op), &direction)) {
    return iree_ok_status();
  }
  loom_amdgpu_descriptor_ref_t guard_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  switch (direction) {
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD:
      guard_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32;
      break;
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_REVERSE:
      guard_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32;
      break;
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_COUNT_:
      return iree_ok_status();
  }

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_wavefront_size(
      loom_low_lower_context_bundle(context), &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size)) {
    return iree_ok_status();
  }

  uint32_t flat_workgroup_size = 0;
  if (!loom_amdgpu_required_flat_workgroup_size(
          module, loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context), &flat_workgroup_size) ||
      flat_workgroup_size == 0 || flat_workgroup_size > wavefront_size) {
    return iree_ok_status();
  }

  bool bpermute_descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32,
      &out_plan->bpermute_descriptor, &bpermute_descriptor_present));
  if (!bpermute_descriptor_present) {
    return iree_ok_status();
  }

  bool combine_descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, combine_descriptor_ref, &out_plan->combine_descriptor,
      &combine_descriptor_present));
  if (!combine_descriptor_present) {
    return iree_ok_status();
  }

  bool guard_descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, guard_descriptor_ref, &out_plan->guard_descriptor,
      &guard_descriptor_present));
  if (!guard_descriptor_present) {
    return iree_ok_status();
  }

  bool select_descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
      &out_plan->select_descriptor, &select_descriptor_present));
  if (!select_descriptor_present) {
    return iree_ok_status();
  }

  out_plan->value = value;
  out_plan->result = loom_kernel_workgroup_scan_result(source_op);
  out_plan->payload_kind = payload_kind;
  out_plan->register_count = register_count;
  out_plan->kind = kind;
  out_plan->mode = mode;
  out_plan->direction = direction;
  out_plan->wavefront_size = wavefront_size;
  out_plan->active_lane_count = flat_workgroup_size;
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_kernel_subgroup_active_mask_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_active_mask_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_subgroup_active_mask_plan_t){0};
  *out_selected = false;
  if (!loom_kernel_subgroup_active_mask_isa(source_op)) {
    return iree_ok_status();
  }

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_wavefront_size(
      loom_low_lower_context_bundle(context), &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t mask = loom_kernel_subgroup_active_mask_mask(source_op);
  uint32_t mask_bit_count = 0;
  if (!loom_amdgpu_subgroup_mask_bit_count(module, mask, &mask_bit_count) ||
      !loom_amdgpu_subgroup_mask_covers_wavefront(mask_bit_count,
                                                  wavefront_size)) {
    return iree_ok_status();
  }

  bool descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
      &out_plan->exec_read_descriptor, &descriptor_present));
  if (!descriptor_present) {
    return iree_ok_status();
  }

  out_plan->mask = mask;
  out_plan->mask_bit_count = mask_bit_count;
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_kernel_subgroup_ballot_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_ballot_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_subgroup_ballot_plan_t){0};
  *out_selected = false;
  if (!loom_kernel_subgroup_vote_ballot_isa(source_op)) {
    return iree_ok_status();
  }

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_wavefront_size(
      loom_low_lower_context_bundle(context), &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t predicate =
      loom_kernel_subgroup_vote_ballot_predicate(source_op);
  if (!loom_amdgpu_module_value_is_native_i1_mask(module, predicate)) {
    return iree_ok_status();
  }

  const loom_value_id_t mask = loom_kernel_subgroup_vote_ballot_mask(source_op);
  uint32_t mask_bit_count = 0;
  if (!loom_amdgpu_subgroup_mask_bit_count(module, mask, &mask_bit_count) ||
      !loom_amdgpu_subgroup_mask_covers_wavefront(mask_bit_count,
                                                  wavefront_size)) {
    return iree_ok_status();
  }

  out_plan->predicate = predicate;
  out_plan->mask = mask;
  out_plan->mask_bit_count = mask_bit_count;
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_kernel_subgroup_vote_any_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_vote_any_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_subgroup_vote_any_plan_t){0};
  *out_selected = false;
  if (!loom_kernel_subgroup_vote_any_isa(source_op)) {
    return iree_ok_status();
  }

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_wavefront_size(
      loom_low_lower_context_bundle(context), &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t predicate =
      loom_kernel_subgroup_vote_any_predicate(source_op);
  if (!loom_amdgpu_module_value_is_native_i1_mask(module, predicate)) {
    return iree_ok_status();
  }

  bool compare_descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_LG_U64,
      &out_plan->compare_descriptor, &compare_descriptor_present));
  if (!compare_descriptor_present) {
    return iree_ok_status();
  }

  bool zero_descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, &out_plan->zero_descriptor,
      &zero_descriptor_present));
  if (!zero_descriptor_present) {
    return iree_ok_status();
  }

  out_plan->predicate = predicate;
  out_plan->result = loom_kernel_subgroup_vote_any_result(source_op);
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_kernel_subgroup_vote_all_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_vote_all_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_subgroup_vote_all_plan_t){0};
  *out_selected = false;
  if (!loom_kernel_subgroup_vote_all_isa(source_op)) {
    return iree_ok_status();
  }

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_wavefront_size(
      loom_low_lower_context_bundle(context), &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t predicate =
      loom_kernel_subgroup_vote_all_predicate(source_op);
  if (!loom_amdgpu_module_value_is_native_i1_mask(module, predicate)) {
    return iree_ok_status();
  }

  bool compare_descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_EQ_U64,
      &out_plan->compare_descriptor, &compare_descriptor_present));
  if (!compare_descriptor_present) {
    return iree_ok_status();
  }

  bool exec_read_descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
      &out_plan->exec_read_descriptor, &exec_read_descriptor_present));
  if (!exec_read_descriptor_present) {
    return iree_ok_status();
  }

  out_plan->predicate = predicate;
  out_plan->result = loom_kernel_subgroup_vote_all_result(source_op);
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_subgroup_lookup_payload(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value, loom_amdgpu_subgroup_payload_kind_t payload_kind,
    loom_value_id_t* out_low_value) {
  switch (payload_kind) {
    case LOOM_AMDGPU_SUBGROUP_PAYLOAD_I32_SCALAR:
      return loom_amdgpu_lookup_or_materialize_vgpr_i32(context, source_op,
                                                        value, out_low_value);
    case LOOM_AMDGPU_SUBGROUP_PAYLOAD_F32_SCALAR:
    case LOOM_AMDGPU_SUBGROUP_PAYLOAD_I32_VECTOR:
    case LOOM_AMDGPU_SUBGROUP_PAYLOAD_F32_VECTOR:
      return loom_low_lower_lookup_value(context, value, out_low_value);
    case LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE:
      break;
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "AMDGPU subgroup lowering has no payload kind");
}

static iree_status_t loom_amdgpu_subgroup_payload_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint32_t register_count, loom_value_id_t low_value, uint32_t register_index,
    loom_type_t lane_type, loom_value_id_t* out_register) {
  *out_register = LOOM_VALUE_ID_INVALID;
  if (register_count == 1) {
    *out_register = low_value;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_low_slice(context, source_op, low_value,
                                    register_index, lane_type, out_register);
}

static iree_status_t loom_amdgpu_emit_subgroup_bpermute_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t low_source_byte_offset, loom_value_id_t low_source_value,
    loom_type_t lane_type, loom_value_id_t* out_low_result) {
  *out_low_result = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {
      low_source_byte_offset,
      low_source_value,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_low_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_readfirstlane_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t low_source_value, loom_type_t result_type,
    loom_value_id_t* out_low_result) {
  *out_low_result = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {
      low_source_value,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &result_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_low_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_bind_subgroup_payload_result(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, uint32_t register_count,
    const loom_value_id_t* result_registers) {
  if (register_count == 1) {
    return loom_low_lower_bind_value(context, source_result,
                                     result_registers[0]);
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), result_registers, register_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, source_result,
                                   loom_low_concat_result(concat_op));
}

iree_status_t loom_amdgpu_lower_kernel_subgroup_broadcast(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_broadcast_plan_t* plan) {
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));

  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_subgroup_lookup_payload(
      context, source_op, plan->value, plan->payload_kind, &low_value));

  loom_value_id_t low_source_byte_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
      plan->source_lane * 4u, lane_type, &low_source_byte_offset));

  loom_value_id_t result_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->register_count; ++i) {
    loom_value_id_t low_source_register = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_subgroup_payload_register(
        context, source_op, plan->register_count, low_value, i, lane_type,
        &low_source_register));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_bpermute_register(
        context, source_op, &plan->descriptor, low_source_byte_offset,
        low_source_register, lane_type, &result_registers[i]));
  }

  return loom_amdgpu_bind_subgroup_payload_result(
      context, source_op, plan->result, plan->register_count, result_registers);
}

iree_status_t loom_amdgpu_lower_kernel_subgroup_broadcast_first(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_broadcast_first_plan_t* plan) {
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_type_t read_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &read_type));

  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_subgroup_lookup_payload(
      context, source_op, plan->value, plan->payload_kind, &low_value));

  loom_value_id_t result_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->register_count; ++i) {
    loom_value_id_t low_source_register = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_subgroup_payload_register(
        context, source_op, plan->register_count, low_value, i, lane_type,
        &low_source_register));

    loom_value_id_t low_read_register = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_readfirstlane_register(
        context, source_op, &plan->descriptor, low_source_register, read_type,
        &low_read_register));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_b32_copy(
        context, source_op, low_read_register, &result_registers[i]));
  }

  return loom_amdgpu_bind_subgroup_payload_result(
      context, source_op, plan->result, plan->register_count, result_registers);
}

static iree_status_t loom_amdgpu_emit_subgroup_mask_compare(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t mask_type, loom_value_id_t* out_mask) {
  *out_mask = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {
      lhs,
      rhs,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_ref, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &mask_type, 1, &low_op));
  *out_mask = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_valid_true(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t lane_type, loom_type_t valid_type, loom_value_id_t* out_valid) {
  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0, lane_type,
      &zero));
  return loom_amdgpu_emit_subgroup_mask_compare(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32, zero, zero,
      valid_type, out_valid);
}

static iree_status_t loom_amdgpu_emit_subgroup_lane_byte_offset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t lane, loom_type_t lane_type,
    loom_value_id_t* out_byte_offset) {
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 2, lane,
      lane_type, out_byte_offset);
}

static iree_status_t loom_amdgpu_emit_subgroup_shuffle_source_byte_offset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_shuffle_plan_t* plan, loom_type_t lane_type,
    loom_type_t valid_type, loom_value_id_t* out_source_byte_offset,
    loom_value_id_t* out_valid) {
  *out_source_byte_offset = LOOM_VALUE_ID_INVALID;
  *out_valid = LOOM_VALUE_ID_INVALID;

  if (plan->mode == LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_INDEX) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_valid_true(
        context, source_op, lane_type, valid_type, out_valid));
    return loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        plan->offset * 4u, lane_type, out_source_byte_offset);
  }

  loom_value_id_t lane_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_current_subgroup_lane_id(
      context, source_op, lane_type, &lane_id));

  loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
  switch (plan->mode) {
    case LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_XOR:
      if (plan->offset == 0) {
        source_lane = lane_id;
      } else {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_XOR_B32_LIT,
            lane_id, plan->offset, lane_type, &source_lane));
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_valid_true(
          context, source_op, lane_type, valid_type, out_valid));
      break;
    case LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_UP: {
      loom_value_id_t offset = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
          plan->offset, lane_type, &offset));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_SUB_U32, lane_id,
          offset, lane_type, &source_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_mask_compare(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32, lane_id,
          offset, valid_type, out_valid));
      break;
    }
    case LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_DOWN: {
      loom_value_id_t offset = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
          plan->offset, lane_type, &offset));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32, lane_id,
          offset, lane_type, &source_lane));
      loom_value_id_t width = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, plan->width,
          lane_type, &width));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_mask_compare(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32,
          source_lane, width, valid_type, out_valid));
      break;
    }
    case LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_INDEX:
    case LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_COUNT_:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AMDGPU subgroup shuffle has invalid mode");
  }

  return loom_amdgpu_emit_subgroup_lane_byte_offset(
      context, source_op, source_lane, lane_type, out_source_byte_offset);
}

iree_status_t loom_amdgpu_lower_kernel_subgroup_shuffle(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_shuffle_plan_t* plan) {
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));

  loom_type_t valid_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->valid, &valid_type));

  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_subgroup_lookup_payload(
      context, source_op, plan->value, plan->payload_kind, &low_value));

  loom_value_id_t low_source_byte_offset = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_valid = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_shuffle_source_byte_offset(
      context, source_op, plan, lane_type, valid_type, &low_source_byte_offset,
      &low_valid));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_bind_value(context, plan->valid, low_valid));

  loom_value_id_t result_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->register_count; ++i) {
    loom_value_id_t low_source_register = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_subgroup_payload_register(
        context, source_op, plan->register_count, low_value, i, lane_type,
        &low_source_register));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_bpermute_register(
        context, source_op, &plan->descriptor, low_source_byte_offset,
        low_source_register, lane_type, &result_registers[i]));
  }

  return loom_amdgpu_bind_subgroup_payload_result(
      context, source_op, plan->result, plan->register_count, result_registers);
}

static iree_status_t loom_amdgpu_emit_subgroup_xor_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t lane_id, uint32_t offset, loom_type_t lane_type,
    loom_value_id_t* out_source_lane) {
  *out_source_lane = LOOM_VALUE_ID_INVALID;
  if (offset == 0) {
    *out_source_lane = lane_id;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_XOR_B32_LIT, lane_id,
      offset, lane_type, out_source_lane);
}

static iree_status_t loom_amdgpu_emit_subgroup_combine(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t lane_type, loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {
      lhs,
      rhs,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static uint32_t loom_amdgpu_subgroup_reduce_first_offset(
    uint32_t active_lane_count) {
  if (active_lane_count <= 1) {
    return 0;
  }
  uint32_t offset = 1;
  while ((offset << 1) < active_lane_count) {
    offset <<= 1;
  }
  return offset;
}

static iree_status_t loom_amdgpu_emit_subgroup_select_peer(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* select_descriptor,
    loom_value_id_t identity, loom_value_id_t peer, loom_value_id_t guard,
    loom_type_t lane_type, loom_value_id_t* out_selected_peer) {
  *out_selected_peer = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {
      identity,
      peer,
      guard,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, select_descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_selected_peer = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_lane_compare(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t mask_type, loom_value_id_t* out_guard) {
  *out_guard = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {
      lhs,
      rhs,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &mask_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_guard = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_reduce_xor_tree(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_reduce_plan_t* plan, loom_value_id_t lane_id,
    loom_type_t lane_type, loom_value_id_t* inout_registers) {
  const bool precompute_step_values = plan->register_count > 1;
  loom_value_id_t source_byte_offsets[LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS] = {
      0};
  uint32_t step_count = 0;
  const uint32_t first_offset =
      loom_amdgpu_subgroup_reduce_first_offset(plan->active_lane_count);
  if (precompute_step_values) {
    for (uint32_t offset = first_offset; offset != 0; offset >>= 1) {
      IREE_ASSERT_LT(step_count, IREE_ARRAYSIZE(source_byte_offsets));
      loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_xor_lane(
          context, source_op, lane_id, offset, lane_type, &source_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_byte_offset(
          context, source_op, source_lane, lane_type,
          &source_byte_offsets[step_count]));
      ++step_count;
    }
  }

  for (uint32_t i = 0; i < plan->register_count; ++i) {
    loom_value_id_t accumulator = inout_registers[i];
    uint32_t step_index = 0;
    for (uint32_t offset = first_offset; offset != 0; offset >>= 1) {
      loom_value_id_t low_source_byte_offset = LOOM_VALUE_ID_INVALID;
      if (precompute_step_values) {
        low_source_byte_offset = source_byte_offsets[step_index++];
      } else {
        loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_xor_lane(
            context, source_op, lane_id, offset, lane_type, &source_lane));
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_byte_offset(
            context, source_op, source_lane, lane_type,
            &low_source_byte_offset));
      }
      loom_value_id_t peer = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_bpermute_register(
          context, source_op, &plan->bpermute_descriptor,
          low_source_byte_offset, accumulator, lane_type, &peer));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_combine(
          context, source_op, &plan->combine_descriptor, accumulator, peer,
          lane_type, &accumulator));
    }
    inout_registers[i] = accumulator;
  }

  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_add_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t lane_id, uint32_t offset, loom_type_t lane_type,
    loom_value_id_t* out_source_lane) {
  *out_source_lane = LOOM_VALUE_ID_INVALID;
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT, lane_id,
      offset, lane_type, out_source_lane);
}

static iree_status_t loom_amdgpu_emit_subgroup_reduce_down_tree(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_reduce_plan_t* plan, loom_value_id_t lane_id,
    loom_type_t lane_type, loom_value_id_t* inout_registers) {
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));

  loom_value_id_t active_lane_count = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
      plan->active_lane_count, lane_type, &active_lane_count));
  loom_value_id_t identity = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
      plan->identity_bits, lane_type, &identity));
  loom_value_id_t first_lane_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0, lane_type,
      &first_lane_offset));

  const bool precompute_step_values = plan->register_count > 1;
  loom_value_id_t source_byte_offsets[LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS] = {
      0};
  loom_value_id_t guards[LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS] = {0};
  uint32_t step_count = 0;
  const uint32_t first_offset =
      loom_amdgpu_subgroup_reduce_first_offset(plan->active_lane_count);
  if (precompute_step_values) {
    for (uint32_t offset = first_offset; offset != 0; offset >>= 1) {
      IREE_ASSERT_LT(step_count, IREE_ARRAYSIZE(source_byte_offsets));
      loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_add_lane(
          context, source_op, lane_id, offset, lane_type, &source_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_byte_offset(
          context, source_op, source_lane, lane_type,
          &source_byte_offsets[step_count]));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_compare(
          context, source_op, &plan->guard_descriptor, source_lane,
          active_lane_count, mask_type, &guards[step_count]));
      ++step_count;
    }
  }

  for (uint32_t i = 0; i < plan->register_count; ++i) {
    loom_value_id_t accumulator = inout_registers[i];
    uint32_t step_index = 0;
    for (uint32_t offset = first_offset; offset != 0; offset >>= 1) {
      loom_value_id_t low_source_byte_offset = LOOM_VALUE_ID_INVALID;
      loom_value_id_t guard = LOOM_VALUE_ID_INVALID;
      if (precompute_step_values) {
        low_source_byte_offset = source_byte_offsets[step_index++];
        guard = guards[step_index - 1];
      } else {
        loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_add_lane(
            context, source_op, lane_id, offset, lane_type, &source_lane));
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_byte_offset(
            context, source_op, source_lane, lane_type,
            &low_source_byte_offset));
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_compare(
            context, source_op, &plan->guard_descriptor, source_lane,
            active_lane_count, mask_type, &guard));
      }
      loom_value_id_t peer = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_bpermute_register(
          context, source_op, &plan->bpermute_descriptor,
          low_source_byte_offset, accumulator, lane_type, &peer));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_select_peer(
          context, source_op, &plan->select_descriptor, identity, peer, guard,
          lane_type, &peer));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_combine(
          context, source_op, &plan->combine_descriptor, accumulator, peer,
          lane_type, &accumulator));
    }

    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_bpermute_register(
        context, source_op, &plan->bpermute_descriptor, first_lane_offset,
        accumulator, lane_type, &inout_registers[i]));
  }

  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_reduce_tree(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_reduce_plan_t* plan, loom_value_id_t lane_id,
    loom_type_t lane_type, loom_value_id_t* inout_registers) {
  if (plan->active_lane_count <= 1 ||
      loom_amdgpu_u32_is_power_of_two(plan->active_lane_count)) {
    return loom_amdgpu_emit_subgroup_reduce_xor_tree(
        context, source_op, plan, lane_id, lane_type, inout_registers);
  }
  return loom_amdgpu_emit_subgroup_reduce_down_tree(
      context, source_op, plan, lane_id, lane_type, inout_registers);
}

static iree_status_t loom_amdgpu_emit_subgroup_scan_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t false_value, loom_value_id_t true_value,
    loom_value_id_t guard, loom_type_t lane_type, loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {
      false_value,
      true_value,
      guard,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_scan_source(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_scan_plan_t* plan, loom_value_id_t lane_id,
    loom_value_id_t active_lane_count, uint32_t offset, loom_type_t lane_type,
    loom_type_t mask_type, loom_value_id_t* out_source_byte_offset,
    loom_value_id_t* out_guard) {
  *out_source_byte_offset = LOOM_VALUE_ID_INVALID;
  *out_guard = LOOM_VALUE_ID_INVALID;

  loom_value_id_t low_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, offset,
      lane_type, &low_offset));

  loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
  switch (plan->direction) {
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_SUB_U32, lane_id,
          low_offset, lane_type, &source_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_compare(
          context, source_op, &plan->guard_descriptor, lane_id, low_offset,
          mask_type, out_guard));
      break;
    }
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_REVERSE: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32, lane_id,
          low_offset, lane_type, &source_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_compare(
          context, source_op, &plan->guard_descriptor, source_lane,
          active_lane_count, mask_type, out_guard));
      break;
    }
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_COUNT_:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AMDGPU subgroup scan has invalid direction");
  }

  return loom_amdgpu_emit_subgroup_lane_byte_offset(
      context, source_op, source_lane, lane_type, out_source_byte_offset);
}

iree_status_t loom_amdgpu_lower_kernel_subgroup_reduce(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_reduce_plan_t* plan) {
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));

  loom_value_id_t lane_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_current_subgroup_lane_id(
      context, source_op, lane_type, &lane_id));

  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_subgroup_lookup_payload(
      context, source_op, plan->value, plan->payload_kind, &low_value));

  loom_value_id_t result_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_subgroup_payload_register(
        context, source_op, plan->register_count, low_value, i, lane_type,
        &result_registers[i]));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_reduce_tree(
      context, source_op, plan, lane_id, lane_type, result_registers));

  return loom_amdgpu_bind_subgroup_payload_result(
      context, source_op, plan->result, plan->register_count, result_registers);
}

static iree_status_t loom_amdgpu_emit_workgroup_reduce_scratch_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t scratch_base, loom_value_id_t dynamic_byte_offset,
    uint32_t static_byte_offset, loom_type_t lane_type,
    loom_value_id_t* out_address) {
  *out_address = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32, scratch_base,
      dynamic_byte_offset, lane_type, out_address));
  if (static_byte_offset == 0) {
    return iree_ok_status();
  }
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
      *out_address, static_byte_offset, lane_type, out_address);
}

static iree_status_t loom_amdgpu_emit_workgroup_reduce_scratch_write(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_reduce_plan_t* plan, loom_value_id_t address,
    loom_value_id_t value) {
  const loom_value_id_t operands[] = {
      address,
      value,
  };
  loom_op_t* write_op = NULL;
  return loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->lds_write_descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), /*result_types=*/NULL,
      /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, &write_op);
}

static iree_status_t loom_amdgpu_emit_workgroup_reduce_scratch_read(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_reduce_plan_t* plan, loom_value_id_t address,
    loom_type_t lane_type, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {address};
  loom_op_t* read_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->lds_read_descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &read_op));
  *out_value = loom_value_slice_get(loom_low_op_results(read_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_workgroup_reduce_barrier(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_reduce_plan_t* plan) {
  loom_op_t* barrier_op = NULL;
  return loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->barrier_descriptor, /*operands=*/NULL,
      /*operand_count=*/0, loom_make_named_attr_slice(NULL, 0),
      /*result_types=*/NULL, /*result_count=*/0, /*tied_results=*/NULL,
      /*tied_result_count=*/0, source_op->location, &barrier_op);
}

static iree_status_t loom_amdgpu_emit_workgroup_reduce_saveexec(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_reduce_plan_t* plan, loom_value_id_t guard,
    loom_type_t mask_type, loom_value_id_t* out_saved_exec) {
  *out_saved_exec = LOOM_VALUE_ID_INVALID;
  loom_type_t active_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &active_type));
  const loom_type_t result_types[] = {mask_type, active_type};
  loom_op_t* saveexec_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->saveexec_descriptor, &guard, 1,
      loom_make_named_attr_slice(NULL, 0), result_types,
      IREE_ARRAYSIZE(result_types), /*tied_results=*/NULL,
      /*tied_result_count=*/0, source_op->location, &saveexec_op));
  *out_saved_exec = loom_op_const_results(saveexec_op)[0];
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_workgroup_reduce_restore_exec(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_reduce_plan_t* plan,
    loom_value_id_t saved_exec) {
  loom_op_t* restore_op = NULL;
  return loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->restore_exec_descriptor, &saved_exec, 1,
      loom_make_named_attr_slice(NULL, 0), /*result_types=*/NULL,
      /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, &restore_op);
}

iree_status_t loom_amdgpu_lower_kernel_workgroup_reduce(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_reduce_plan_t* plan) {
  if (plan->flat_workgroup_size <= plan->wavefront_size) {
    const loom_amdgpu_subgroup_reduce_plan_t subgroup_plan = {
        .bpermute_descriptor = plan->bpermute_descriptor,
        .combine_descriptor = plan->combine_descriptor,
        .guard_descriptor = plan->guard_descriptor,
        .select_descriptor = plan->select_descriptor,
        .value = plan->value,
        .result = plan->result,
        .payload_kind = plan->payload_kind,
        .register_count = plan->register_count,
        .wavefront_size = plan->wavefront_size,
        .active_lane_count = plan->flat_workgroup_size,
        .identity_bits = plan->identity_bits,
    };
    return loom_amdgpu_lower_kernel_subgroup_reduce(context, source_op,
                                                    &subgroup_plan);
  }

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));

  loom_value_id_t linear_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_current_workitem_linear_id(
      context, source_op, lane_type, &linear_id));
  loom_value_id_t lane_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_current_subgroup_lane_id(
      context, source_op, lane_type, &lane_id));

  const uint32_t register_count = plan->register_count;
  const uint32_t flat_workgroup_size = plan->flat_workgroup_size;
  const uint32_t wavefront_size = plan->wavefront_size;
  const bool has_partial_tail = (flat_workgroup_size % wavefront_size) != 0;
  const uint32_t wave_count =
      (flat_workgroup_size + wavefront_size - 1) / wavefront_size;
  const uint32_t tail_lane_count = flat_workgroup_size % wavefront_size;
  const uint32_t scratch_slot_count =
      has_partial_tail ? flat_workgroup_size : wave_count;
  const int64_t scratch_byte_length =
      (int64_t)((uint64_t)scratch_slot_count * register_count * 4u);

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_op_t* storage_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_storage_reserve_build(
      builder, scratch_byte_length, /*byte_alignment=*/4,
      loom_type_storage(LOOM_STORAGE_SPACE_WORKGROUP), source_op->location,
      &storage_op));
  loom_op_t* storage_address_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_storage_address_build(
      builder, loom_low_storage_reserve_storage(storage_op), /*offset=*/0,
      lane_type, source_op->location, &storage_address_op));
  const loom_value_id_t scratch_base =
      loom_low_storage_address_result(storage_address_op);

  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_subgroup_lookup_payload(
      context, source_op, plan->value, plan->payload_kind, &low_value));

  if (!has_partial_tail) {
    loom_value_id_t result_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
    for (uint32_t i = 0; i < register_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_subgroup_payload_register(
          context, source_op, register_count, low_value, i, lane_type,
          &result_registers[i]));
    }

    const loom_amdgpu_subgroup_reduce_plan_t per_wave_plan = {
        .bpermute_descriptor = plan->bpermute_descriptor,
        .combine_descriptor = plan->combine_descriptor,
        .guard_descriptor = plan->guard_descriptor,
        .select_descriptor = plan->select_descriptor,
        .value = plan->value,
        .result = plan->result,
        .payload_kind = plan->payload_kind,
        .register_count = register_count,
        .wavefront_size = wavefront_size,
        .active_lane_count = wavefront_size,
        .identity_bits = plan->identity_bits,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_reduce_tree(
        context, source_op, &per_wave_plan, lane_id, lane_type,
        result_registers));

    loom_value_id_t subgroup_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
        loom_amdgpu_subgroup_u32_log2(wavefront_size), linear_id, lane_type,
        &subgroup_id));
    loom_value_id_t subgroup_byte_offset = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_vgpr_scale_u32(context, source_op, subgroup_id, 4,
                                        LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_NONE,
                                        lane_type, &subgroup_byte_offset));

    loom_type_t mask_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));

    loom_value_id_t lane_zero_count = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 1, lane_type,
        &lane_zero_count));
    loom_value_id_t lane_zero_guard = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_compare(
        context, source_op, &plan->guard_descriptor, lane_id, lane_zero_count,
        mask_type, &lane_zero_guard));
    loom_value_id_t saved_exec = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_saveexec(
        context, source_op, plan, lane_zero_guard, mask_type, &saved_exec));
    for (uint32_t i = 0; i < register_count; ++i) {
      const uint32_t register_byte_offset = i * wave_count * 4u;
      loom_value_id_t address = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_address(
          context, source_op, scratch_base, subgroup_byte_offset,
          register_byte_offset, lane_type, &address));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_write(
          context, source_op, plan, address, result_registers[i]));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_restore_exec(
        context, source_op, plan, saved_exec));

    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_workgroup_reduce_barrier(context, source_op, plan));

    loom_value_id_t wave_count_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, wave_count,
        lane_type, &wave_count_value));
    loom_value_id_t first_wave_guard = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_compare(
        context, source_op, &plan->guard_descriptor, linear_id,
        wave_count_value, mask_type, &first_wave_guard));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_saveexec(
        context, source_op, plan, first_wave_guard, mask_type, &saved_exec));

    loom_value_id_t lane_byte_offset = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_byte_offset(
        context, source_op, lane_id, lane_type, &lane_byte_offset));
    for (uint32_t i = 0; i < register_count; ++i) {
      const uint32_t register_byte_offset = i * wave_count * 4u;
      loom_value_id_t address = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_address(
          context, source_op, scratch_base, lane_byte_offset,
          register_byte_offset, lane_type, &address));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_read(
          context, source_op, plan, address, lane_type, &result_registers[i]));
    }

    const loom_amdgpu_subgroup_reduce_plan_t cross_wave_plan = {
        .bpermute_descriptor = plan->bpermute_descriptor,
        .combine_descriptor = plan->combine_descriptor,
        .guard_descriptor = plan->guard_descriptor,
        .select_descriptor = plan->select_descriptor,
        .value = plan->value,
        .result = plan->result,
        .payload_kind = plan->payload_kind,
        .register_count = register_count,
        .wavefront_size = wavefront_size,
        .active_lane_count = wave_count,
        .identity_bits = plan->identity_bits,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_reduce_tree(
        context, source_op, &cross_wave_plan, lane_id, lane_type,
        result_registers));

    loom_value_id_t publish_saved_exec = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_saveexec(
        context, source_op, plan, lane_zero_guard, mask_type,
        &publish_saved_exec));
    for (uint32_t i = 0; i < register_count; ++i) {
      const uint32_t register_byte_offset = i * wave_count * 4u;
      loom_value_id_t address = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_address(
          context, source_op, scratch_base, lane_byte_offset,
          register_byte_offset, lane_type, &address));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_write(
          context, source_op, plan, address, result_registers[i]));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_restore_exec(
        context, source_op, plan, publish_saved_exec));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_restore_exec(
        context, source_op, plan, saved_exec));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_workgroup_reduce_barrier(context, source_op, plan));

    loom_value_id_t zero_byte_offset = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0, lane_type,
        &zero_byte_offset));
    for (uint32_t i = 0; i < register_count; ++i) {
      const uint32_t register_byte_offset = i * wave_count * 4u;
      loom_value_id_t address = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_address(
          context, source_op, scratch_base, zero_byte_offset,
          register_byte_offset, lane_type, &address));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_read(
          context, source_op, plan, address, lane_type, &result_registers[i]));
    }

    return loom_amdgpu_bind_subgroup_payload_result(
        context, source_op, plan->result, register_count, result_registers);
  }

  loom_value_id_t linear_byte_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scale_u32(
      context, source_op, linear_id, 4, LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_NONE,
      lane_type, &linear_byte_offset));
  for (uint32_t i = 0; i < register_count; ++i) {
    loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_subgroup_payload_register(
        context, source_op, register_count, low_value, i, lane_type,
        &source_register));

    const uint32_t register_byte_offset = i * flat_workgroup_size * 4u;
    loom_value_id_t address = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_address(
        context, source_op, scratch_base, linear_byte_offset,
        register_byte_offset, lane_type, &address));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_write(
        context, source_op, plan, address, source_register));
  }

  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_workgroup_reduce_barrier(context, source_op, plan));

  loom_value_id_t lane_byte_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_byte_offset(
      context, source_op, lane_id, lane_type, &lane_byte_offset));

  loom_type_t mask_type = loom_type_none();
  loom_value_id_t tail_lane_guard = LOOM_VALUE_ID_INVALID;
  loom_value_id_t tail_lane_byte_offset = LOOM_VALUE_ID_INVALID;
  loom_value_id_t identity = LOOM_VALUE_ID_INVALID;
  if (has_partial_tail) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));
    loom_value_id_t tail_lane_count_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        tail_lane_count, lane_type, &tail_lane_count_value));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_compare(
        context, source_op, &plan->guard_descriptor, lane_id,
        tail_lane_count_value, mask_type, &tail_lane_guard));

    loom_value_id_t zero_byte_offset = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0, lane_type,
        &zero_byte_offset));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_select_peer(
        context, source_op, &plan->select_descriptor, zero_byte_offset,
        lane_byte_offset, tail_lane_guard, lane_type, &tail_lane_byte_offset));

    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        plan->identity_bits, lane_type, &identity));
  }

  loom_value_id_t result_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < register_count; ++i) {
    const uint32_t register_base_byte_offset = i * flat_workgroup_size * 4u;
    loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
    for (uint32_t chunk_base = 0; chunk_base < flat_workgroup_size;
         chunk_base += wavefront_size) {
      const bool is_tail_chunk =
          has_partial_tail &&
          (chunk_base + wavefront_size) > flat_workgroup_size;
      const uint32_t static_byte_offset =
          register_base_byte_offset + chunk_base * 4u;
      const loom_value_id_t chunk_lane_byte_offset =
          is_tail_chunk ? tail_lane_byte_offset : lane_byte_offset;
      loom_value_id_t address = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_address(
          context, source_op, scratch_base, chunk_lane_byte_offset,
          static_byte_offset, lane_type, &address));

      loom_value_id_t loaded = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_read(
          context, source_op, plan, address, lane_type, &loaded));
      if (is_tail_chunk) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_select_peer(
            context, source_op, &plan->select_descriptor, identity, loaded,
            tail_lane_guard, lane_type, &loaded));
      }
      if (accumulator == LOOM_VALUE_ID_INVALID) {
        accumulator = loaded;
      } else {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_combine(
            context, source_op, &plan->combine_descriptor, accumulator, loaded,
            lane_type, &accumulator));
      }
    }
    result_registers[i] = accumulator;
  }

  const loom_amdgpu_subgroup_reduce_plan_t subgroup_plan = {
      .bpermute_descriptor = plan->bpermute_descriptor,
      .combine_descriptor = plan->combine_descriptor,
      .guard_descriptor = plan->guard_descriptor,
      .select_descriptor = plan->select_descriptor,
      .value = plan->value,
      .result = plan->result,
      .payload_kind = plan->payload_kind,
      .register_count = register_count,
      .wavefront_size = wavefront_size,
      .active_lane_count = wavefront_size,
      .identity_bits = plan->identity_bits,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_reduce_tree(
      context, source_op, &subgroup_plan, lane_id, lane_type,
      result_registers));

  if (has_partial_tail) {
    loom_value_id_t wavefront_size_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        wavefront_size, lane_type, &wavefront_size_value));
    loom_value_id_t producer_wave_guard = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_compare(
        context, source_op, &plan->guard_descriptor, linear_id,
        wavefront_size_value, mask_type, &producer_wave_guard));

    loom_value_id_t saved_exec = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_saveexec(
        context, source_op, plan, producer_wave_guard, mask_type, &saved_exec));
    for (uint32_t i = 0; i < register_count; ++i) {
      const uint32_t register_base_byte_offset = i * flat_workgroup_size * 4u;
      loom_value_id_t address = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_address(
          context, source_op, scratch_base, lane_byte_offset,
          register_base_byte_offset, lane_type, &address));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_write(
          context, source_op, plan, address, result_registers[i]));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_restore_exec(
        context, source_op, plan, saved_exec));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_workgroup_reduce_barrier(context, source_op, plan));
    for (uint32_t i = 0; i < register_count; ++i) {
      const uint32_t register_base_byte_offset = i * flat_workgroup_size * 4u;
      loom_value_id_t address = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_address(
          context, source_op, scratch_base, lane_byte_offset,
          register_base_byte_offset, lane_type, &address));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_read(
          context, source_op, plan, address, lane_type, &result_registers[i]));
    }
  }

  return loom_amdgpu_bind_subgroup_payload_result(
      context, source_op, plan->result, register_count, result_registers);
}

iree_status_t loom_amdgpu_lower_kernel_subgroup_scan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_scan_plan_t* plan) {
  if (plan->active_lane_count == 0 ||
      plan->active_lane_count > plan->wavefront_size) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU subgroup scan has invalid active lane count");
  }
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));

  loom_value_id_t lane_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_current_subgroup_lane_id(
      context, source_op, lane_type, &lane_id));

  loom_value_id_t active_lane_count = LOOM_VALUE_ID_INVALID;
  if (plan->direction == LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_REVERSE) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        plan->active_lane_count, lane_type, &active_lane_count));
  }

  loom_value_id_t source_byte_offsets[LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS] = {
      0};
  loom_value_id_t guards[LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS] = {0};
  uint32_t step_count = 0;
  for (uint32_t offset = 1; offset < plan->active_lane_count; offset <<= 1) {
    IREE_ASSERT_LT(step_count, IREE_ARRAYSIZE(source_byte_offsets));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_scan_source(
        context, source_op, plan, lane_id, active_lane_count, offset, lane_type,
        mask_type, &source_byte_offsets[step_count], &guards[step_count]));
    ++step_count;
  }

  loom_value_id_t exclusive_byte_offset = LOOM_VALUE_ID_INVALID;
  loom_value_id_t exclusive_guard = LOOM_VALUE_ID_INVALID;
  uint32_t identity_bits = 0;
  const bool is_exclusive =
      plan->mode == LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE;
  if (is_exclusive) {
    if (!loom_amdgpu_subgroup_combine_identity_bits(plan->kind,
                                                    &identity_bits)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU subgroup scan has no identity for combining kind");
    }
    exclusive_byte_offset = source_byte_offsets[0];
    exclusive_guard = guards[0];
  } else if (plan->mode != LOOM_KERNEL_SUBGROUP_SCAN_MODE_INCLUSIVE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU subgroup scan has invalid mode");
  }

  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_subgroup_lookup_payload(
      context, source_op, plan->value, plan->payload_kind, &low_value));

  loom_value_id_t result_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->register_count; ++i) {
    loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_subgroup_payload_register(
        context, source_op, plan->register_count, low_value, i, lane_type,
        &accumulator));

    for (uint32_t step_index = 0; step_index < step_count; ++step_index) {
      loom_value_id_t peer = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_bpermute_register(
          context, source_op, &plan->bpermute_descriptor,
          source_byte_offsets[step_index], accumulator, lane_type, &peer));
      loom_value_id_t combined = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_combine(
          context, source_op, &plan->combine_descriptor, accumulator, peer,
          lane_type, &combined));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_scan_select(
          context, source_op, &plan->select_descriptor, accumulator, combined,
          guards[step_index], lane_type, &accumulator));
    }

    if (is_exclusive) {
      if (step_count == 0) {
        loom_value_id_t identity = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
            identity_bits, lane_type, &identity));
        accumulator = identity;
      } else {
        loom_value_id_t shifted = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_bpermute_register(
            context, source_op, &plan->bpermute_descriptor,
            exclusive_byte_offset, accumulator, lane_type, &shifted));
        loom_value_id_t identity = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
            identity_bits, lane_type, &identity));
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_scan_select(
            context, source_op, &plan->select_descriptor, identity, shifted,
            exclusive_guard, lane_type, &accumulator));
      }
    }

    result_registers[i] = accumulator;
  }

  return loom_amdgpu_bind_subgroup_payload_result(
      context, source_op, plan->result, plan->register_count, result_registers);
}

iree_status_t loom_amdgpu_lower_kernel_workgroup_scan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_scan_plan_t* plan) {
  return loom_amdgpu_lower_kernel_subgroup_scan(context, source_op, plan);
}

static iree_status_t loom_amdgpu_emit_subgroup_exec_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t* out_mask) {
  *out_mask = LOOM_VALUE_ID_INVALID;
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, /*unit_count=*/2, &mask_type));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, /*operands=*/NULL, /*operand_count=*/0,
      loom_make_named_attr_slice(NULL, 0), &mask_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_mask = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_bind_subgroup_lane_mask_result(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_mask, uint32_t mask_bit_count,
    loom_value_id_t low_mask) {
  if (mask_bit_count == 64) {
    return loom_low_lower_bind_value(context, source_mask, low_mask);
  }
  if (mask_bit_count != 32) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU subgroup mask has invalid bit width");
  }

  loom_type_t low_mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &low_mask_type));
  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_mask, /*offset=*/0, low_mask_type, &low_result));
  return loom_low_lower_bind_value(context, source_mask, low_result);
}

static iree_status_t loom_amdgpu_emit_subgroup_zero_lane_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t* out_mask) {
  *out_mask = LOOM_VALUE_ID_INVALID;

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_string_id_t imm32_attr_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_intern(context, IREE_SV("imm32"), &imm32_attr_name_id));

  loom_value_id_t zero_halves[2] = {0};
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(zero_halves); ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
        context, source_op, descriptor, imm32_attr_name_id, 0, sgpr_type,
        &zero_halves[i]));
  }

  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, /*unit_count=*/2, &mask_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), zero_halves,
      IREE_ARRAYSIZE(zero_halves), mask_type, source_op->location, &concat_op));
  *out_mask = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_resolved_mask_compare(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &result_type));
  const loom_value_id_t operands[] = {
      lhs,
      rhs,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &result_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_kernel_subgroup_active_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_active_mask_plan_t* plan) {
  loom_value_id_t low_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_exec_mask(
      context, source_op, &plan->exec_read_descriptor, &low_mask));
  return loom_amdgpu_bind_subgroup_lane_mask_result(
      context, source_op, plan->mask, plan->mask_bit_count, low_mask);
}

iree_status_t loom_amdgpu_lower_kernel_subgroup_ballot(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_ballot_plan_t* plan) {
  loom_value_id_t low_predicate = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->predicate, &low_predicate));
  return loom_amdgpu_bind_subgroup_lane_mask_result(
      context, source_op, plan->mask, plan->mask_bit_count, low_predicate);
}

iree_status_t loom_amdgpu_lower_kernel_subgroup_vote_any(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_vote_any_plan_t* plan) {
  loom_value_id_t low_predicate = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->predicate, &low_predicate));
  loom_value_id_t zero_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_zero_lane_mask(
      context, source_op, &plan->zero_descriptor, &zero_mask));
  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_resolved_mask_compare(
      context, source_op, &plan->compare_descriptor, low_predicate, zero_mask,
      &low_result));
  return loom_low_lower_bind_value(context, plan->result, low_result);
}

iree_status_t loom_amdgpu_lower_kernel_subgroup_vote_all(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_vote_all_plan_t* plan) {
  loom_value_id_t low_predicate = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->predicate, &low_predicate));
  loom_value_id_t exec_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_exec_mask(
      context, source_op, &plan->exec_read_descriptor, &exec_mask));
  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_resolved_mask_compare(
      context, source_op, &plan->compare_descriptor, low_predicate, exec_mask,
      &low_result));
  return loom_low_lower_bind_value(context, plan->result, low_result);
}

static iree_status_t loom_amdgpu_low_legality_verify_subgroup_wavefront(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    iree_string_view_t constraint_key, uint32_t* out_wavefront_size) {
  *out_wavefront_size = 0;
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_wavefront_size(bundle, out_wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(*out_wavefront_size)) {
    return loom_amdgpu_low_legality_reject(context, op, constraint_key);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_low_legality_verify_subgroup_native_predicate(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_value_id_t predicate, iree_string_view_t constraint_key) {
  if (!loom_amdgpu_module_value_is_native_i1_mask(
          loom_target_low_legality_module(context), predicate)) {
    return loom_amdgpu_low_legality_reject(context, op, constraint_key);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_low_legality_verify_subgroup_mask_result(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_value_id_t mask, uint32_t wavefront_size,
    uint32_t* out_mask_bit_count) {
  *out_mask_bit_count = 0;
  const loom_module_t* module = loom_target_low_legality_module(context);
  if (!loom_amdgpu_subgroup_mask_bit_count(module, mask, out_mask_bit_count)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_mask.result_type"));
  }
  if (!loom_amdgpu_subgroup_mask_covers_wavefront(*out_mask_bit_count,
                                                  wavefront_size)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_mask.wavefront_coverage"));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_low_legality_verify_subgroup_descriptor(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    iree_string_view_t constraint_key) {
  if (!loom_amdgpu_descriptor_set_has_ref(
          loom_target_low_legality_descriptor_set(context), descriptor_ref)) {
    return loom_amdgpu_low_legality_reject(context, op, constraint_key);
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_active_mask(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_subgroup_wavefront(
      context, op, IREE_SV("subgroup_active_mask.wavefront_size"),
      &wavefront_size));
  uint32_t unused_mask_bit_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_subgroup_mask_result(
      context, op, loom_kernel_subgroup_active_mask_mask(op), wavefront_size,
      &unused_mask_bit_count));
  return loom_amdgpu_low_legality_verify_subgroup_descriptor(
      context, op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
      IREE_SV("descriptor.s_mov_b64_exec_read"));
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_ballot(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_subgroup_wavefront(
      context, op, IREE_SV("subgroup_ballot.wavefront_size"), &wavefront_size));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_low_legality_verify_subgroup_native_predicate(
          context, op, loom_kernel_subgroup_vote_ballot_predicate(op),
          IREE_SV("subgroup_ballot.native_predicate")));
  uint32_t unused_mask_bit_count = 0;
  return loom_amdgpu_low_legality_verify_subgroup_mask_result(
      context, op, loom_kernel_subgroup_vote_ballot_mask(op), wavefront_size,
      &unused_mask_bit_count);
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_vote_any(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  uint32_t unused_wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_subgroup_wavefront(
      context, op, IREE_SV("subgroup_vote_any.wavefront_size"),
      &unused_wavefront_size));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_low_legality_verify_subgroup_native_predicate(
          context, op, loom_kernel_subgroup_vote_any_predicate(op),
          IREE_SV("subgroup_vote_any.native_predicate")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_subgroup_descriptor(
      context, op, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_LG_U64,
      IREE_SV("descriptor.s_cmp_lg_u64")));
  return loom_amdgpu_low_legality_verify_subgroup_descriptor(
      context, op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      IREE_SV("descriptor.s_mov_b32"));
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_vote_all(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  uint32_t unused_wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_subgroup_wavefront(
      context, op, IREE_SV("subgroup_vote_all.wavefront_size"),
      &unused_wavefront_size));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_low_legality_verify_subgroup_native_predicate(
          context, op, loom_kernel_subgroup_vote_all_predicate(op),
          IREE_SV("subgroup_vote_all.native_predicate")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_subgroup_descriptor(
      context, op, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_EQ_U64,
      IREE_SV("descriptor.s_cmp_eq_u64")));
  return loom_amdgpu_low_legality_verify_subgroup_descriptor(
      context, op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
      IREE_SV("descriptor.s_mov_b64_exec_read"));
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_shuffle(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);
  const loom_value_id_t value = loom_kernel_subgroup_shuffle_value(op);
  loom_amdgpu_subgroup_payload_kind_t unused_payload_kind =
      LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  uint32_t unused_register_count = 0;
  if (!loom_amdgpu_subgroup_payload_is_supported(
          module, value, &unused_payload_kind, &unused_register_count)) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("subgroup_shuffle.payload"));
  }

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_wavefront_size(bundle, &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_shuffle.wavefront_size"));
  }

  int64_t width = 0;
  if (!loom_amdgpu_subgroup_exact_i32_value(
          module, loom_target_low_legality_fact_table(context),
          loom_kernel_subgroup_shuffle_width(op), &width)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_shuffle.exact_width"));
  }
  if (width != (int64_t)wavefront_size) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_shuffle.full_wave_width"));
  }

  int64_t offset = 0;
  if (!loom_amdgpu_subgroup_exact_i32_value(
          module, loom_target_low_legality_fact_table(context),
          loom_kernel_subgroup_shuffle_offset(op), &offset)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_shuffle.exact_lane"));
  }
  if (offset < 0 || offset >= (int64_t)wavefront_size) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_shuffle.lane_range"));
  }

  const uint32_t descriptor_ordinal = loom_amdgpu_descriptor_ref_ordinal(
      loom_target_low_legality_descriptor_set(context),
      LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("descriptor.ds_bpermute_b32"));
  }

  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_reduce(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);
  if (loom_amdgpu_subgroup_reduce_has_cluster_attrs(op)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_reduce.full_subgroup"));
  }

  const loom_value_id_t value = loom_kernel_subgroup_reduce_value(op);
  loom_amdgpu_subgroup_payload_kind_t payload_kind =
      LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  uint32_t unused_register_count = 0;
  if (!loom_amdgpu_subgroup_payload_is_supported(module, value, &payload_kind,
                                                 &unused_register_count)) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("subgroup_reduce.payload"));
  }

  loom_amdgpu_descriptor_ref_t combine_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  const loom_combining_kind_t kind = loom_kernel_subgroup_reduce_kind(op);
  if (!loom_amdgpu_subgroup_combine_descriptor_ref(kind, payload_kind,
                                                   &combine_descriptor_ref)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_reduce.combining_kind"));
  }

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_wavefront_size(bundle, &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_reduce.wavefront_size"));
  }
  uint32_t active_lane_count = 0;
  if (!loom_amdgpu_subgroup_reduce_active_lane_count(
          module, loom_target_low_legality_function(context), bundle,
          wavefront_size, &active_lane_count)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_reduce.fixed_workgroup_wave_multiple"));
  }
  if (!loom_amdgpu_u32_is_power_of_two(active_lane_count)) {
    uint32_t unused_identity_bits = 0;
    if (!loom_amdgpu_subgroup_combine_identity_bits(kind,
                                                    &unused_identity_bits)) {
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("subgroup_reduce.identity"));
    }
  }

  const loom_low_descriptor_set_t* descriptor_set =
      loom_target_low_legality_descriptor_set(context);
  if (!loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("descriptor.ds_bpermute_b32"));
  }
  if (!loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                          combine_descriptor_ref)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("descriptor.reduce_combine"));
  }
  if (!loom_amdgpu_u32_is_power_of_two(active_lane_count)) {
    if (!loom_amdgpu_descriptor_set_has_ref(
            descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32)) {
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("descriptor.v_cmp_ult_u32"));
    }
    if (!loom_amdgpu_descriptor_set_has_ref(
            descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32)) {
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("descriptor.v_cndmask_b32"));
    }
  }

  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_workgroup_reduce(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);
  const loom_value_id_t value = loom_kernel_workgroup_reduce_value(op);
  loom_amdgpu_subgroup_payload_kind_t payload_kind =
      LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  uint32_t register_count = 0;
  if (!loom_amdgpu_subgroup_payload_is_supported(module, value, &payload_kind,
                                                 &register_count)) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("workgroup_reduce.payload"));
  }

  loom_amdgpu_descriptor_ref_t combine_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  const loom_combining_kind_t kind = loom_kernel_workgroup_reduce_kind(op);
  if (!loom_amdgpu_subgroup_combine_descriptor_ref(kind, payload_kind,
                                                   &combine_descriptor_ref)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("workgroup_reduce.combining_kind"));
  }

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_wavefront_size(bundle, &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("workgroup_reduce.wavefront_size"));
  }
  uint32_t flat_workgroup_size = 0;
  if (!loom_amdgpu_required_flat_workgroup_size(
          module, loom_target_low_legality_function(context), bundle,
          &flat_workgroup_size) ||
      flat_workgroup_size == 0) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("workgroup_reduce.fixed_workgroup_size"));
  }
  const bool has_partial_tail = flat_workgroup_size > wavefront_size &&
                                (flat_workgroup_size % wavefront_size) != 0;
  const uint32_t wave_count =
      (flat_workgroup_size + wavefront_size - 1) / wavefront_size;
  if (flat_workgroup_size > wavefront_size && wave_count > wavefront_size) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("workgroup_reduce.wave_count"));
  }
  const bool needs_cross_wave_identity =
      flat_workgroup_size > wavefront_size &&
      (has_partial_tail || !loom_amdgpu_u32_is_power_of_two(wave_count));
  const bool needs_subgroup_identity =
      flat_workgroup_size < wavefront_size &&
      !loom_amdgpu_u32_is_power_of_two(flat_workgroup_size);
  const bool needs_identity_guard =
      needs_subgroup_identity || needs_cross_wave_identity;
  if (needs_identity_guard) {
    uint32_t unused_identity_bits = 0;
    if (!loom_amdgpu_subgroup_combine_identity_bits(kind,
                                                    &unused_identity_bits)) {
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("workgroup_reduce.identity"));
    }
  }
  const uint32_t scratch_slot_count =
      has_partial_tail ? flat_workgroup_size : wave_count;
  const uint64_t scratch_byte_length =
      (uint64_t)scratch_slot_count * register_count * 4u;
  if (scratch_byte_length > UINT32_MAX) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("workgroup_reduce.scratch_byte_length"));
  }

  const loom_low_descriptor_set_t* descriptor_set =
      loom_target_low_legality_descriptor_set(context);
  if (!loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("descriptor.ds_bpermute_b32"));
  }
  if (!loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                          combine_descriptor_ref)) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("descriptor.combine"));
  }
  if (needs_identity_guard || flat_workgroup_size > wavefront_size) {
    if (!loom_amdgpu_descriptor_set_has_ref(
            descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32)) {
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("descriptor.v_cmp_ult_u32"));
    }
  }
  if (needs_identity_guard) {
    if (!loom_amdgpu_descriptor_set_has_ref(
            descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32)) {
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("descriptor.v_cndmask_b32"));
    }
  }
  if (flat_workgroup_size > wavefront_size) {
    if (!loom_amdgpu_descriptor_set_has_ref(
            descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B32)) {
      return loom_amdgpu_low_legality_reject(context, op,
                                             IREE_SV("descriptor.ds_read_b32"));
    }
    if (!loom_amdgpu_descriptor_set_has_ref(
            descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B32)) {
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("descriptor.ds_write_b32"));
    }
    if (!loom_amdgpu_descriptor_set_has_ref(
            descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_BARRIER)) {
      return loom_amdgpu_low_legality_reject(context, op,
                                             IREE_SV("descriptor.s_barrier"));
    }
    if (!loom_amdgpu_descriptor_set_has_ref(
            descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64)) {
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("descriptor.s_and_saveexec_b64"));
    }
    if (!loom_amdgpu_descriptor_set_has_ref(
            descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC)) {
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("descriptor.s_mov_b64_exec"));
    }
  }

  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_workgroup_scan(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);
  const loom_value_id_t value = loom_kernel_workgroup_scan_value(op);
  loom_amdgpu_subgroup_payload_kind_t payload_kind =
      LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  uint32_t unused_register_count = 0;
  if (!loom_amdgpu_subgroup_payload_is_supported(module, value, &payload_kind,
                                                 &unused_register_count)) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("workgroup_scan.payload"));
  }

  loom_amdgpu_descriptor_ref_t combine_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  const loom_combining_kind_t kind = loom_kernel_workgroup_scan_kind(op);
  if (!loom_amdgpu_subgroup_combine_descriptor_ref(kind, payload_kind,
                                                   &combine_descriptor_ref)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("workgroup_scan.combining_kind"));
  }

  loom_kernel_subgroup_scan_mode_t mode = LOOM_KERNEL_SUBGROUP_SCAN_MODE_COUNT_;
  if (!loom_amdgpu_map_workgroup_scan_mode(loom_kernel_workgroup_scan_mode(op),
                                           &mode)) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("workgroup_scan.mode"));
  }
  if (mode == LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE) {
    uint32_t unused_identity_bits = 0;
    if (!loom_amdgpu_subgroup_combine_identity_bits(kind,
                                                    &unused_identity_bits)) {
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("workgroup_scan.identity"));
    }
  }

  loom_kernel_subgroup_scan_direction_t direction =
      LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_COUNT_;
  if (!loom_amdgpu_map_workgroup_scan_direction(
          loom_kernel_workgroup_scan_direction(op), &direction)) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("workgroup_scan.direction"));
  }
  loom_amdgpu_descriptor_ref_t guard_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  switch (direction) {
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD:
      guard_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32;
      break;
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_REVERSE:
      guard_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32;
      break;
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_COUNT_:
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("workgroup_scan.direction"));
  }

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_wavefront_size(bundle, &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("workgroup_scan.wavefront_size"));
  }

  uint32_t flat_workgroup_size = 0;
  if (!loom_amdgpu_required_flat_workgroup_size(
          module, loom_target_low_legality_function(context), bundle,
          &flat_workgroup_size) ||
      flat_workgroup_size == 0) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("workgroup_scan.fixed_workgroup_size"));
  }
  if (flat_workgroup_size > wavefront_size) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("workgroup_scan.multi_wave"));
  }

  const loom_low_descriptor_set_t* descriptor_set =
      loom_target_low_legality_descriptor_set(context);
  if (!loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("descriptor.ds_bpermute_b32"));
  }
  if (!loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                          combine_descriptor_ref)) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("descriptor.scan_combine"));
  }
  if (!loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                          guard_descriptor_ref)) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("descriptor.scan_guard"));
  }
  if (!loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32)) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("descriptor.v_cndmask_b32"));
  }

  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_scan(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);
  if (loom_amdgpu_subgroup_scan_has_cluster_attrs(op)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_scan.full_subgroup"));
  }

  const loom_value_id_t value = loom_kernel_subgroup_scan_value(op);
  loom_amdgpu_subgroup_payload_kind_t payload_kind =
      LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  uint32_t unused_register_count = 0;
  if (!loom_amdgpu_subgroup_payload_is_supported(module, value, &payload_kind,
                                                 &unused_register_count)) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("subgroup_scan.payload"));
  }

  loom_amdgpu_descriptor_ref_t combine_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  const loom_combining_kind_t kind = loom_kernel_subgroup_scan_kind(op);
  if (!loom_amdgpu_subgroup_combine_descriptor_ref(kind, payload_kind,
                                                   &combine_descriptor_ref)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_scan.combining_kind"));
  }

  switch (loom_kernel_subgroup_scan_mode(op)) {
    case LOOM_KERNEL_SUBGROUP_SCAN_MODE_INCLUSIVE:
      break;
    case LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE: {
      uint32_t unused_identity_bits = 0;
      if (!loom_amdgpu_subgroup_combine_identity_bits(kind,
                                                      &unused_identity_bits)) {
        return loom_amdgpu_low_legality_reject(
            context, op, IREE_SV("subgroup_scan.identity"));
      }
      break;
    }
    case LOOM_KERNEL_SUBGROUP_SCAN_MODE_COUNT_:
      return loom_amdgpu_low_legality_reject(context, op,
                                             IREE_SV("subgroup_scan.mode"));
  }

  loom_amdgpu_descriptor_ref_t guard_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  switch (loom_kernel_subgroup_scan_direction(op)) {
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD:
      guard_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32;
      break;
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_REVERSE:
      guard_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32;
      break;
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_COUNT_:
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("subgroup_scan.direction"));
  }

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_wavefront_size(bundle, &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_scan.wavefront_size"));
  }
  if (!loom_amdgpu_subgroup_full_wave_workgroups(
          module, loom_target_low_legality_function(context), bundle,
          wavefront_size)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_scan.fixed_workgroup_wave_multiple"));
  }

  const loom_low_descriptor_set_t* descriptor_set =
      loom_target_low_legality_descriptor_set(context);
  if (!loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("descriptor.ds_bpermute_b32"));
  }
  if (!loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                          combine_descriptor_ref)) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("descriptor.scan_combine"));
  }
  if (!loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                          guard_descriptor_ref)) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("descriptor.scan_guard"));
  }
  if (!loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32)) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("descriptor.v_cndmask_b32"));
  }

  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_broadcast(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);
  const loom_value_id_t value = loom_kernel_subgroup_broadcast_value(op);
  loom_amdgpu_subgroup_payload_kind_t unused_payload_kind =
      LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  uint32_t unused_register_count = 0;
  if (!loom_amdgpu_subgroup_payload_is_supported(
          module, value, &unused_payload_kind, &unused_register_count)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_broadcast.payload"));
  }

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_wavefront_size(bundle, &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_broadcast.wavefront_size"));
  }

  int64_t source_lane = 0;
  if (!loom_amdgpu_subgroup_exact_i32_value(
          module, loom_target_low_legality_fact_table(context),
          loom_kernel_subgroup_broadcast_lane(op), &source_lane)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_broadcast.exact_lane"));
  }
  if (source_lane < 0 || source_lane >= (int64_t)wavefront_size) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_broadcast.lane_range"));
  }

  const uint32_t descriptor_ordinal = loom_amdgpu_descriptor_ref_ordinal(
      loom_target_low_legality_descriptor_set(context),
      LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("descriptor.ds_bpermute_b32"));
  }

  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_broadcast_first(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);
  const loom_value_id_t value = loom_kernel_subgroup_broadcast_first_value(op);
  loom_amdgpu_subgroup_payload_kind_t unused_payload_kind =
      LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  uint32_t unused_register_count = 0;
  if (!loom_amdgpu_subgroup_payload_is_supported(
          module, value, &unused_payload_kind, &unused_register_count)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_broadcast_first.payload"));
  }

  uint32_t unused_wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_subgroup_wavefront(
      context, op, IREE_SV("subgroup_broadcast_first.wavefront_size"),
      &unused_wavefront_size));
  return loom_amdgpu_low_legality_verify_subgroup_descriptor(
      context, op, LOOM_AMDGPU_DESCRIPTOR_REF_V_READFIRSTLANE_B32,
      IREE_SV("descriptor.v_readfirstlane_b32"));
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_match(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  return loom_amdgpu_low_legality_reject(
      context, op, IREE_SV("subgroup_match.target_legalization"));
}

#undef LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS
