// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ops/kernel/ops.h"
#include "loom/target/arch/amdgpu/lower/internal.h"
#include "loom/target/arch/amdgpu/target_refs.h"

#define LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS 6u

static bool loom_amdgpu_subgroup_wavefront_size_is_supported(
    uint32_t wavefront_size) {
  return wavefront_size == 32 || wavefront_size == 64;
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

  out_plan->value = value;
  out_plan->result = loom_kernel_subgroup_reduce_result(source_op);
  out_plan->payload_kind = payload_kind;
  out_plan->register_count = register_count;
  out_plan->wavefront_size = wavefront_size;
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
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_literal(
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

static iree_status_t loom_amdgpu_emit_subgroup_xor_lane_byte_offset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t lane_id, uint32_t offset, loom_type_t lane_type,
    loom_value_id_t* out_source_byte_offset) {
  loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_literal(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_XOR_B32_LIT, lane_id,
      offset, lane_type, &source_lane));
  return loom_amdgpu_emit_subgroup_lane_byte_offset(
      context, source_op, source_lane, lane_type, out_source_byte_offset);
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

static iree_status_t loom_amdgpu_emit_subgroup_scan_guard(
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
    loom_value_id_t wavefront_size, uint32_t offset, loom_type_t lane_type,
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
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_scan_guard(
          context, source_op, &plan->guard_descriptor, lane_id, low_offset,
          mask_type, out_guard));
      break;
    }
    case LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_REVERSE: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32, lane_id,
          low_offset, lane_type, &source_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_scan_guard(
          context, source_op, &plan->guard_descriptor, source_lane,
          wavefront_size, mask_type, out_guard));
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

  const bool precompute_byte_offsets = plan->register_count > 1;
  loom_value_id_t source_byte_offsets[LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS] = {
      0};
  uint32_t step_count = 0;
  if (precompute_byte_offsets) {
    for (uint32_t offset = plan->wavefront_size / 2u; offset != 0;
         offset >>= 1) {
      if (step_count >= IREE_ARRAYSIZE(source_byte_offsets)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AMDGPU subgroup reduce lowering has too many native tree steps");
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_xor_lane_byte_offset(
          context, source_op, lane_id, offset, lane_type,
          &source_byte_offsets[step_count++]));
    }
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

    uint32_t step_index = 0;
    for (uint32_t offset = plan->wavefront_size / 2u; offset != 0;
         offset >>= 1) {
      loom_value_id_t low_source_byte_offset = LOOM_VALUE_ID_INVALID;
      if (precompute_byte_offsets) {
        low_source_byte_offset = source_byte_offsets[step_index++];
      } else {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_xor_lane_byte_offset(
            context, source_op, lane_id, offset, lane_type,
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

    result_registers[i] = accumulator;
  }

  return loom_amdgpu_bind_subgroup_payload_result(
      context, source_op, plan->result, plan->register_count, result_registers);
}

iree_status_t loom_amdgpu_lower_kernel_subgroup_scan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_scan_plan_t* plan) {
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));

  loom_value_id_t lane_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_current_subgroup_lane_id(
      context, source_op, lane_type, &lane_id));

  loom_value_id_t wavefront_size = LOOM_VALUE_ID_INVALID;
  if (plan->direction == LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_REVERSE) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        plan->wavefront_size, lane_type, &wavefront_size));
  }

  loom_value_id_t source_byte_offsets[LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS] = {
      0};
  loom_value_id_t guards[LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS] = {0};
  uint32_t step_count = 0;
  for (uint32_t offset = 1; offset < plan->wavefront_size; offset <<= 1) {
    if (step_count >= IREE_ARRAYSIZE(source_byte_offsets)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU subgroup scan lowering has too many native tree steps");
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_scan_source(
        context, source_op, plan, lane_id, wavefront_size, offset, lane_type,
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
      loom_value_id_t shifted = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_bpermute_register(
          context, source_op, &plan->bpermute_descriptor, exclusive_byte_offset,
          accumulator, lane_type, &shifted));
      loom_value_id_t identity = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
          identity_bits, lane_type, &identity));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_scan_select(
          context, source_op, &plan->select_descriptor, identity, shifted,
          exclusive_guard, lane_type, &accumulator));
    }

    result_registers[i] = accumulator;
  }

  return loom_amdgpu_bind_subgroup_payload_result(
      context, source_op, plan->result, plan->register_count, result_registers);
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
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    iree_string_view_t message, uint32_t* out_wavefront_size) {
  *out_wavefront_size = 0;
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_wavefront_size(bundle, out_wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(*out_wavefront_size)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("subgroup"),
        loom_op_name(loom_target_low_legality_module(context), op), message);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_low_legality_verify_subgroup_native_predicate(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_value_id_t predicate, iree_string_view_t message) {
  if (!loom_amdgpu_module_value_is_native_i1_mask(
          loom_target_low_legality_module(context), predicate)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("predicate"),
        loom_op_name(loom_target_low_legality_module(context), op), message);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_low_legality_verify_subgroup_mask_result(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_value_id_t mask, uint32_t wavefront_size,
    uint32_t* out_mask_bit_count) {
  *out_mask_bit_count = 0;
  const loom_module_t* module = loom_target_low_legality_module(context);
  if (!loom_amdgpu_subgroup_mask_bit_count(module, mask, out_mask_bit_count)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("mask"), loom_op_name(module, op),
        IREE_SV("AMDGPU subgroup mask lowering requires an i32 or i64 integer "
                "mask result"));
  }
  if (!loom_amdgpu_subgroup_mask_covers_wavefront(*out_mask_bit_count,
                                                  wavefront_size)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("mask"), loom_op_name(module, op),
        IREE_SV("AMDGPU subgroup mask lowering requires an i64 mask result on "
                "wave64 targets"));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_low_legality_verify_subgroup_descriptor(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    iree_string_view_t descriptor_name, iree_string_view_t message) {
  if (!loom_amdgpu_descriptor_set_has_ref(
          loom_target_low_legality_descriptor_set(context), descriptor_ref)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("descriptor"), descriptor_name, message);
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
      provider, context, op,
      IREE_SV("AMDGPU subgroup active.mask lowering requires a wave32 or "
              "wave64 target subgroup"),
      &wavefront_size));
  uint32_t unused_mask_bit_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_subgroup_mask_result(
      provider, context, op, loom_kernel_subgroup_active_mask_mask(op),
      wavefront_size, &unused_mask_bit_count));
  return loom_amdgpu_low_legality_verify_subgroup_descriptor(
      provider, context, op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
      IREE_SV("amdgpu.s_mov_b64_exec_read"),
      IREE_SV("selected descriptor set does not provide a native EXEC mask "
              "read packet"));
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
      provider, context, op,
      IREE_SV("AMDGPU subgroup vote.ballot lowering requires a wave32 or "
              "wave64 target subgroup"),
      &wavefront_size));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_low_legality_verify_subgroup_native_predicate(
          provider, context, op, loom_kernel_subgroup_vote_ballot_predicate(op),
          IREE_SV("AMDGPU subgroup vote.ballot lowering requires a divergent "
                  "i1 predicate that maps to a native EXEC-width mask")));
  uint32_t unused_mask_bit_count = 0;
  return loom_amdgpu_low_legality_verify_subgroup_mask_result(
      provider, context, op, loom_kernel_subgroup_vote_ballot_mask(op),
      wavefront_size, &unused_mask_bit_count);
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
      provider, context, op,
      IREE_SV("AMDGPU subgroup vote.any lowering requires a wave32 or wave64 "
              "target subgroup"),
      &unused_wavefront_size));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_low_legality_verify_subgroup_native_predicate(
          provider, context, op, loom_kernel_subgroup_vote_any_predicate(op),
          IREE_SV("AMDGPU subgroup vote.any lowering requires a divergent i1 "
                  "predicate that maps to a native EXEC-width mask")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_subgroup_descriptor(
      provider, context, op, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_LG_U64,
      IREE_SV("amdgpu.s_cmp_lg_u64"),
      IREE_SV("selected descriptor set does not provide a native 64-bit SALU "
              "mask compare packet")));
  return loom_amdgpu_low_legality_verify_subgroup_descriptor(
      provider, context, op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      IREE_SV("amdgpu.s_mov_b32"),
      IREE_SV("selected descriptor set does not provide a native SGPR zero "
              "materialization packet"));
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
      provider, context, op,
      IREE_SV("AMDGPU subgroup vote.all lowering requires a wave32 or wave64 "
              "target subgroup"),
      &unused_wavefront_size));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_low_legality_verify_subgroup_native_predicate(
          provider, context, op, loom_kernel_subgroup_vote_all_predicate(op),
          IREE_SV("AMDGPU subgroup vote.all lowering requires a divergent i1 "
                  "predicate that maps to a native EXEC-width mask")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_subgroup_descriptor(
      provider, context, op, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_EQ_U64,
      IREE_SV("amdgpu.s_cmp_eq_u64"),
      IREE_SV("selected descriptor set does not provide a native 64-bit SALU "
              "mask compare packet")));
  return loom_amdgpu_low_legality_verify_subgroup_descriptor(
      provider, context, op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
      IREE_SV("amdgpu.s_mov_b64_exec_read"),
      IREE_SV("selected descriptor set does not provide a native EXEC mask "
              "read packet"));
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
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("collective"), loom_op_name(module, op),
        IREE_SV("AMDGPU subgroup shuffle requires an i32 or f32 scalar, or a "
                "rank-1 static i32/f32 vector payload"));
  }

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_wavefront_size(bundle, &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("subgroup"), loom_op_name(module, op),
        IREE_SV("AMDGPU subgroup shuffle lowering requires a wave32 or wave64 "
                "target subgroup"));
  }

  int64_t width = 0;
  if (!loom_amdgpu_subgroup_exact_i32_value(
          module, loom_target_low_legality_fact_table(context),
          loom_kernel_subgroup_shuffle_width(op), &width)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("collective"), loom_op_name(module, op),
        IREE_SV("AMDGPU subgroup shuffle requires an exact width so validity "
                "maps to native wave32/wave64 lane masks"));
  }
  if (width != (int64_t)wavefront_size) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("collective"), loom_op_name(module, op),
        IREE_SV("AMDGPU subgroup shuffle currently requires full selected "
                "wave32 or wave64 width"));
  }

  int64_t offset = 0;
  if (!loom_amdgpu_subgroup_exact_i32_value(
          module, loom_target_low_legality_fact_table(context),
          loom_kernel_subgroup_shuffle_offset(op), &offset)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("collective"), loom_op_name(module, op),
        IREE_SV("AMDGPU subgroup shuffle requires an exact lane offset or "
                "index so each 32-bit payload register maps to one native "
                "ds_bpermute packet"));
  }
  if (offset < 0 || offset >= (int64_t)wavefront_size) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("collective"), loom_op_name(module, op),
        IREE_SV("AMDGPU subgroup shuffle offset/index must be within the "
                "selected wave32 or wave64 subgroup"));
  }

  const uint32_t descriptor_ordinal = loom_amdgpu_descriptor_ref_ordinal(
      loom_target_low_legality_descriptor_set(context),
      LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("descriptor"),
        IREE_SV("amdgpu.ds_bpermute_b32"),
        IREE_SV("selected descriptor set does not provide a native subgroup "
                "shuffle cross-lane packet"));
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
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("cluster"), loom_op_name(module, op),
        IREE_SV("AMDGPU subgroup reduce lowering requires full subgroup "
                "reductions with cluster_size and cluster_stride absent"));
  }

  const loom_value_id_t value = loom_kernel_subgroup_reduce_value(op);
  loom_amdgpu_subgroup_payload_kind_t payload_kind =
      LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  uint32_t unused_register_count = 0;
  if (!loom_amdgpu_subgroup_payload_is_supported(module, value, &payload_kind,
                                                 &unused_register_count)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("collective"), loom_op_name(module, op),
        IREE_SV("AMDGPU subgroup reduce requires an i32 or f32 scalar, or a "
                "rank-1 static i32/f32 vector payload"));
  }

  loom_amdgpu_descriptor_ref_t combine_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (!loom_amdgpu_subgroup_combine_descriptor_ref(
          loom_kernel_subgroup_reduce_kind(op), payload_kind,
          &combine_descriptor_ref)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("kind"), loom_op_name(module, op),
        IREE_SV("AMDGPU subgroup reduce requires an i32 add/mul/and/or/xor "
                "or f32 add/mul/minnum/maxnum combining kind"));
  }

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_wavefront_size(bundle, &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("subgroup"), loom_op_name(module, op),
        IREE_SV("AMDGPU subgroup reduce lowering requires a wave32 or wave64 "
                "target subgroup"));
  }
  if (!loom_amdgpu_subgroup_full_wave_workgroups(
          module, loom_target_low_legality_function(context), bundle,
          wavefront_size)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("workgroup"), loom_op_name(module, op),
        IREE_SV("AMDGPU subgroup reduce lowering requires a fixed workgroup "
                "size that is an exact multiple of the selected wave32 or "
                "wave64 subgroup"));
  }

  const loom_low_descriptor_set_t* descriptor_set =
      loom_target_low_legality_descriptor_set(context);
  if (!loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("descriptor"),
        IREE_SV("amdgpu.ds_bpermute_b32"),
        IREE_SV("selected descriptor set does not provide a native subgroup "
                "reduce cross-lane packet"));
  }
  if (!loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                          combine_descriptor_ref)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("descriptor"),
        IREE_SV("amdgpu.reduce.combine"),
        IREE_SV("selected descriptor set does not provide the native subgroup "
                "reduce combine packet"));
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
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("cluster"), loom_op_name(module, op),
        IREE_SV("AMDGPU subgroup scan lowering requires full subgroup scans "
                "with cluster_size and cluster_stride absent"));
  }

  const loom_value_id_t value = loom_kernel_subgroup_scan_value(op);
  loom_amdgpu_subgroup_payload_kind_t payload_kind =
      LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  uint32_t unused_register_count = 0;
  if (!loom_amdgpu_subgroup_payload_is_supported(module, value, &payload_kind,
                                                 &unused_register_count)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("collective"), loom_op_name(module, op),
        IREE_SV("AMDGPU subgroup scan requires an i32 or f32 scalar, or a "
                "rank-1 static i32/f32 vector payload"));
  }

  loom_amdgpu_descriptor_ref_t combine_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  const loom_combining_kind_t kind = loom_kernel_subgroup_scan_kind(op);
  if (!loom_amdgpu_subgroup_combine_descriptor_ref(kind, payload_kind,
                                                   &combine_descriptor_ref)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("kind"), loom_op_name(module, op),
        IREE_SV("AMDGPU subgroup scan requires an i32 add/mul/and/or/xor or "
                "f32 add/mul/minnum/maxnum combining kind"));
  }

  switch (loom_kernel_subgroup_scan_mode(op)) {
    case LOOM_KERNEL_SUBGROUP_SCAN_MODE_INCLUSIVE:
      break;
    case LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE: {
      uint32_t unused_identity_bits = 0;
      if (!loom_amdgpu_subgroup_combine_identity_bits(kind,
                                                      &unused_identity_bits)) {
        return loom_target_low_legality_reject(
            context, provider, op, IREE_SV("kind"), loom_op_name(module, op),
            IREE_SV("AMDGPU subgroup exclusive scan requires a native identity "
                    "for the combining kind"));
      }
      break;
    }
    case LOOM_KERNEL_SUBGROUP_SCAN_MODE_COUNT_:
      return loom_target_low_legality_reject(
          context, provider, op, IREE_SV("mode"), loom_op_name(module, op),
          IREE_SV("AMDGPU subgroup scan requires inclusive or exclusive mode"));
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
      return loom_target_low_legality_reject(
          context, provider, op, IREE_SV("direction"), loom_op_name(module, op),
          IREE_SV("AMDGPU subgroup scan requires forward or reverse "
                  "direction"));
  }

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_wavefront_size(bundle, &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("subgroup"), loom_op_name(module, op),
        IREE_SV("AMDGPU subgroup scan lowering requires a wave32 or wave64 "
                "target subgroup"));
  }
  if (!loom_amdgpu_subgroup_full_wave_workgroups(
          module, loom_target_low_legality_function(context), bundle,
          wavefront_size)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("workgroup"), loom_op_name(module, op),
        IREE_SV("AMDGPU subgroup scan lowering requires a fixed workgroup size "
                "that is an exact multiple of the selected wave32 or wave64 "
                "subgroup"));
  }

  const loom_low_descriptor_set_t* descriptor_set =
      loom_target_low_legality_descriptor_set(context);
  if (!loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("descriptor"),
        IREE_SV("amdgpu.ds_bpermute_b32"),
        IREE_SV("selected descriptor set does not provide a native subgroup "
                "scan cross-lane packet"));
  }
  if (!loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                          combine_descriptor_ref)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("descriptor"),
        IREE_SV("amdgpu.scan.combine"),
        IREE_SV("selected descriptor set does not provide the native subgroup "
                "scan combine packet"));
  }
  if (!loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                          guard_descriptor_ref)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("descriptor"),
        IREE_SV("amdgpu.scan.guard"),
        IREE_SV("selected descriptor set does not provide the native subgroup "
                "scan lane-bound compare packet"));
  }
  if (!loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("descriptor"),
        IREE_SV("amdgpu.v_cndmask_b32"),
        IREE_SV("selected descriptor set does not provide the native subgroup "
                "scan guarded-select packet"));
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
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("collective"), loom_op_name(module, op),
        IREE_SV("AMDGPU subgroup broadcast requires an i32 or f32 scalar, or a "
                "rank-1 static i32/f32 vector payload"));
  }

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_wavefront_size(bundle, &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("subgroup"), loom_op_name(module, op),
        IREE_SV("AMDGPU subgroup broadcast lowering requires a wave32 or "
                "wave64 target subgroup"));
  }

  int64_t source_lane = 0;
  if (!loom_amdgpu_subgroup_exact_i32_value(
          module, loom_target_low_legality_fact_table(context),
          loom_kernel_subgroup_broadcast_lane(op), &source_lane)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("collective"), loom_op_name(module, op),
        IREE_SV("AMDGPU subgroup broadcast requires an exact source lane so "
                "each 32-bit payload register maps to one native ds_bpermute "
                "packet"));
  }
  if (source_lane < 0 || source_lane >= (int64_t)wavefront_size) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("collective"), loom_op_name(module, op),
        IREE_SV("AMDGPU subgroup broadcast source lane must be within the "
                "selected wave32 or wave64 subgroup"));
  }

  const uint32_t descriptor_ordinal = loom_amdgpu_descriptor_ref_ordinal(
      loom_target_low_legality_descriptor_set(context),
      LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("descriptor"),
        IREE_SV("amdgpu.ds_bpermute_b32"),
        IREE_SV("selected descriptor set does not provide a native subgroup "
                "broadcast cross-lane packet"));
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
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("collective"), loom_op_name(module, op),
        IREE_SV("AMDGPU subgroup broadcast.first requires an i32 or f32 "
                "scalar, or a rank-1 static i32/f32 vector payload"));
  }

  uint32_t unused_wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_subgroup_wavefront(
      provider, context, op,
      IREE_SV("AMDGPU subgroup broadcast.first lowering requires a wave32 or "
              "wave64 target subgroup"),
      &unused_wavefront_size));
  return loom_amdgpu_low_legality_verify_subgroup_descriptor(
      provider, context, op, LOOM_AMDGPU_DESCRIPTOR_REF_V_READFIRSTLANE_B32,
      IREE_SV("amdgpu.v_readfirstlane_b32"),
      IREE_SV("selected descriptor set does not provide a native first-active "
              "lane read packet"));
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

  return loom_target_low_legality_reject(
      context, provider, op, IREE_SV("collective"),
      loom_op_name(loom_target_low_legality_module(context), op),
      IREE_SV("AMDGPU subgroup match requires target legalization before "
              "source-to-low because equality masks are lane-varying and do "
              "not map to uniform EXEC masks"));
}

#undef LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS
