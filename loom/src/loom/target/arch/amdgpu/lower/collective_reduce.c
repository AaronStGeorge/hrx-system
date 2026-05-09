// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ops/kernel/ops.h"
#include "loom/target/arch/amdgpu/lower/collective_combine.h"
#include "loom/target/arch/amdgpu/lower/collective_payload.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/subgroup.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/lower/workgroup.h"
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

static bool loom_amdgpu_descriptor_set_has_ref(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref) {
  return loom_amdgpu_descriptor_ref_ordinal(descriptor_set, descriptor_ref) !=
         LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
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
  if (!loom_amdgpu_collective_payload_is_supported(module, value, &payload_kind,
                                                   &register_count)) {
    return iree_ok_status();
  }

  const loom_combining_kind_t kind =
      loom_kernel_subgroup_reduce_kind(source_op);
  loom_amdgpu_descriptor_ref_t combine_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (!loom_amdgpu_collective_combine_descriptor_ref(kind, payload_kind,
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
    if (!loom_amdgpu_collective_combine_identity_bits(kind, &identity_bits)) {
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
  if (!loom_amdgpu_collective_payload_is_supported(module, value, &payload_kind,
                                                   &register_count)) {
    return iree_ok_status();
  }

  const loom_combining_kind_t kind =
      loom_kernel_workgroup_reduce_kind(source_op);
  loom_amdgpu_descriptor_ref_t combine_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (!loom_amdgpu_collective_combine_descriptor_ref(kind, payload_kind,
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
    if (!loom_amdgpu_collective_combine_identity_bits(kind, &identity_bits)) {
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

static iree_status_t loom_amdgpu_emit_subgroup_lane_byte_offset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t lane, loom_type_t lane_type,
    loom_value_id_t* out_byte_offset) {
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 2, lane,
      lane_type, out_byte_offset);
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
iree_status_t loom_amdgpu_lower_kernel_subgroup_reduce(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_reduce_plan_t* plan) {
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));

  loom_value_id_t lane_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_current_subgroup_lane_id(
      context, source_op, lane_type, &lane_id));

  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_collective_lookup_payload(
      context, source_op, plan->value, plan->payload_kind, &low_value));

  loom_value_id_t result_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_collective_payload_register(
        context, source_op, plan->register_count, low_value, i, lane_type,
        &result_registers[i]));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_reduce_tree(
      context, source_op, plan, lane_id, lane_type, result_registers));

  return loom_amdgpu_collective_bind_payload_result(
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
  IREE_RETURN_IF_ERROR(loom_amdgpu_collective_lookup_payload(
      context, source_op, plan->value, plan->payload_kind, &low_value));

  if (!has_partial_tail) {
    loom_value_id_t result_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
    for (uint32_t i = 0; i < register_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_collective_payload_register(
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

    return loom_amdgpu_collective_bind_payload_result(
        context, source_op, plan->result, register_count, result_registers);
  }

  loom_value_id_t linear_byte_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scale_u32(
      context, source_op, linear_id, 4, LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_NONE,
      lane_type, &linear_byte_offset));
  for (uint32_t i = 0; i < register_count; ++i) {
    loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_collective_payload_register(
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

  return loom_amdgpu_collective_bind_payload_result(
      context, source_op, plan->result, register_count, result_registers);
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
  if (!loom_amdgpu_collective_payload_is_supported(module, value, &payload_kind,
                                                   &unused_register_count)) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("subgroup_reduce.payload"));
  }

  loom_amdgpu_descriptor_ref_t combine_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  const loom_combining_kind_t kind = loom_kernel_subgroup_reduce_kind(op);
  if (!loom_amdgpu_collective_combine_descriptor_ref(kind, payload_kind,
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
    if (!loom_amdgpu_collective_combine_identity_bits(kind,
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
  if (!loom_amdgpu_collective_payload_is_supported(module, value, &payload_kind,
                                                   &register_count)) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("workgroup_reduce.payload"));
  }

  loom_amdgpu_descriptor_ref_t combine_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  const loom_combining_kind_t kind = loom_kernel_workgroup_reduce_kind(op);
  if (!loom_amdgpu_collective_combine_descriptor_ref(kind, payload_kind,
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
    if (!loom_amdgpu_collective_combine_identity_bits(kind,
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

#undef LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS
