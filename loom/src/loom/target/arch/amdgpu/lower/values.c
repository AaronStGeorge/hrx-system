// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/values.h"

#include <stdint.h>

#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/target_refs.h"

typedef struct loom_amdgpu_constant_plan_t {
  // Source result value receiving the emitted low constant.
  loom_value_id_t result;
  // Descriptor row selected for the constant move packet.
  loom_low_lower_resolved_descriptor_t descriptor;
  // Module string ID for the descriptor's imm32 attribute.
  loom_string_id_t imm32_attr_name_id;
  // Number of 32-bit registers receiving immediate bit patterns.
  uint32_t register_count;
  // Immediate bit patterns emitted into selected result registers.
  uint32_t bit_patterns[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
} loom_amdgpu_constant_plan_t;

typedef struct loom_amdgpu_vector_iota_plan_t {
  // Descriptor row selected for each lane constant packet.
  loom_low_lower_resolved_descriptor_t descriptor;
  // Module string ID for the descriptor's imm32 attribute.
  loom_string_id_t imm32_attr_name_id;
  // Source scalar base value used by dynamic lane materialization.
  loom_value_id_t base;
  // Source scalar step value used by dynamic lane materialization.
  loom_value_id_t step;
  // Result vector receiving the generated i32 lane constants.
  loom_value_id_t result;
  // Static number of generated lanes.
  uint32_t lane_count;
  // Precomputed lane bit patterns emitted as VGPR constants.
  uint32_t lane_bit_patterns[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  // True when one or more iota operands must be materialized dynamically.
  bool is_dynamic;
} loom_amdgpu_vector_iota_plan_t;

typedef struct loom_amdgpu_vector_from_elements_plan_t {
  // Result vector assembled from the selected source elements.
  loom_value_id_t result;
  // Physical storage selected for the result vector.
  loom_amdgpu_vector_storage_kind_t storage_kind;
  // Static source element count.
  uint32_t element_count;
  // Static result register count after source elements are packed.
  uint32_t register_count;
  // Static 32-bit register count occupied by one source element.
  uint32_t element_register_count;
  // Static payload bit count occupied by one source element.
  uint32_t element_bit_count;
  // Source and result scalar element type.
  loom_scalar_type_t element_type;
  // Source scalar values in result lane order.
  loom_value_id_t elements[LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES];
} loom_amdgpu_vector_from_elements_plan_t;

typedef struct loom_amdgpu_vector_insert_plan_t {
  // Scalar value inserted into the destination vector.
  loom_value_id_t value;
  // Destination vector whose lanes are copied except at the selected index.
  loom_value_id_t dest;
  // Optional dynamic destination lane index, or invalid for static insertion.
  loom_value_id_t dynamic_index;
  // Result vector receiving the updated lane payload.
  loom_value_id_t result;
  // Static destination lane offset.
  uint32_t lane_offset;
  // Static logical destination lane count.
  uint32_t lane_count;
  // Static 32-bit backing register count for the destination vector.
  uint32_t register_count;
  // Number of payload bits occupied by each logical destination lane.
  uint32_t lane_bit_count;
  // Source and result scalar element type.
  loom_scalar_type_t element_type;
  // True when insertion uses |dynamic_index| instead of |lane_offset|.
  bool is_dynamic;
} loom_amdgpu_vector_insert_plan_t;

typedef enum loom_amdgpu_vector_bf16_conversion_kind_e {
  LOOM_AMDGPU_VECTOR_BF16_CONVERSION_KIND_NONE = 0,
  LOOM_AMDGPU_VECTOR_BF16_CONVERSION_KIND_EXTF = 1,
  LOOM_AMDGPU_VECTOR_BF16_CONVERSION_KIND_FPTRUNC = 2,
} loom_amdgpu_vector_bf16_conversion_kind_t;

typedef struct loom_amdgpu_vector_bf16_conversion_plan_t {
  // Conversion operation selected for the source/result type pair.
  loom_amdgpu_vector_bf16_conversion_kind_t kind;
  // Source vector value being converted.
  loom_value_id_t source;
  // Result vector value receiving the converted lane payload.
  loom_value_id_t result;
  // Static vector lane count.
  uint32_t lane_count;
  // Number of 32-bit source registers occupied by the source vector.
  uint32_t source_register_count;
  // Number of 32-bit result registers occupied by the result vector.
  uint32_t result_register_count;
} loom_amdgpu_vector_bf16_conversion_plan_t;

typedef struct loom_amdgpu_cast_alias_plan_t {
  // Source value whose existing low mapping can represent the cast result.
  loom_value_id_t source;
  // Result value receiving the low source alias.
  loom_value_id_t result;
} loom_amdgpu_cast_alias_plan_t;

typedef struct loom_amdgpu_scalar_trunci_plan_t {
  // Source integer value being truncated.
  loom_value_id_t source;
  // Result integer value receiving the low 32 bits of source.
  loom_value_id_t result;
} loom_amdgpu_scalar_trunci_plan_t;

typedef struct loom_amdgpu_scalar_extsi_plan_t {
  // Source 32-bit integer value being sign-extended.
  loom_value_id_t source;
  // Result 64-bit integer value receiving the sign-extended pair.
  loom_value_id_t result;
} loom_amdgpu_scalar_extsi_plan_t;

static bool loom_amdgpu_descriptor_present(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref) {
  return loom_amdgpu_descriptor_ref_ordinal(descriptor_set, descriptor_ref) !=
         LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
}

static bool loom_amdgpu_iota_i32_lane_value(int64_t base, int64_t step,
                                            uint32_t lane, int64_t* out_value) {
  *out_value = 0;
  int64_t scaled_step = 0;
  if (!iree_checked_mul_i64((int64_t)lane, step, &scaled_step)) {
    return false;
  }
  int64_t value = 0;
  if (!iree_checked_add_i64(base, scaled_step, &value) || value < INT32_MIN ||
      value > INT32_MAX) {
    return false;
  }
  *out_value = value;
  return true;
}

static bool loom_amdgpu_value_type_can_materialize_as_vgpr_i32(
    const loom_module_t* module, loom_value_id_t value_id) {
  return loom_amdgpu_type_is_i32(loom_module_value_type(module, value_id));
}

static bool loom_amdgpu_vector_iota_has_lane_offsets_in_i32_range(
    uint32_t lane_count, int64_t step) {
  for (uint32_t i = 1; i < lane_count; ++i) {
    int64_t lane_offset = 0;
    if (!iree_checked_mul_i64((int64_t)i, step, &lane_offset) ||
        lane_offset < INT32_MIN || lane_offset > INT32_MAX) {
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_vector_iota_needs_dynamic_add(uint32_t lane_count,
                                                      int64_t step) {
  return lane_count > 1 && step != 0;
}

static bool loom_amdgpu_vector_iota_needs_dynamic_step_shift(
    uint32_t lane_count) {
  for (uint32_t i = 2; i < lane_count; ++i) {
    if (loom_amdgpu_u32_is_power_of_two(i)) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_vector_iota_needs_dynamic_step_multiply(
    uint32_t lane_count) {
  for (uint32_t i = 2; i < lane_count; ++i) {
    if (!loom_amdgpu_u32_is_power_of_two(i)) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_vector_iota_source_supported(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* source_op,
    iree_string_view_t* out_constraint_key) {
  *out_constraint_key = IREE_SV("vector_iota.i32_static_elements");
  const loom_value_id_t result = loom_vector_iota_result(source_op);
  const uint32_t element_count = loom_amdgpu_vector_i32_register_count(
      loom_module_value_type(module, result));
  if (element_count == 0) {
    return false;
  }

  *out_constraint_key = IREE_SV("vector_iota.i32_operands");
  const loom_value_id_t base = loom_vector_iota_base(source_op);
  const loom_value_id_t step = loom_vector_iota_step(source_op);
  if (!loom_amdgpu_value_type_can_materialize_as_vgpr_i32(module, base) ||
      !loom_amdgpu_value_type_can_materialize_as_vgpr_i32(module, step)) {
    return false;
  }

  int64_t base_value = 0;
  int64_t step_value = 0;
  const bool has_static_base =
      loom_amdgpu_module_value_as_i32_constant(module, base, &base_value);
  const bool has_static_step =
      loom_amdgpu_module_value_as_i32_constant(module, step, &step_value);
  if (has_static_base && has_static_step) {
    *out_constraint_key = IREE_SV("vector_iota.i32_lane_range");
    for (uint32_t i = 0; i < element_count; ++i) {
      int64_t lane_value = 0;
      if (!loom_amdgpu_iota_i32_lane_value(base_value, step_value, i,
                                           &lane_value)) {
        return false;
      }
    }
    *out_constraint_key = IREE_SV("descriptor.v_mov_b32");
    return loom_amdgpu_descriptor_present(descriptor_set,
                                          LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32);
  }

  *out_constraint_key = IREE_SV("descriptor.v_mov_b32");
  if (!loom_amdgpu_descriptor_present(descriptor_set,
                                      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32)) {
    return false;
  }
  *out_constraint_key = IREE_SV("descriptor.v_mov_b32_copy");
  if (!loom_amdgpu_descriptor_present(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY)) {
    return false;
  }

  if (has_static_step) {
    *out_constraint_key = IREE_SV("vector_iota.i32_lane_range");
    if (!loom_amdgpu_vector_iota_has_lane_offsets_in_i32_range(element_count,
                                                               step_value)) {
      return false;
    }
    if (loom_amdgpu_vector_iota_needs_dynamic_add(element_count, step_value)) {
      *out_constraint_key = IREE_SV("descriptor.v_add_u32_lit");
      if (!loom_amdgpu_descriptor_present(
              descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT)) {
        return false;
      }
    }
    return true;
  }

  *out_constraint_key = IREE_SV("descriptor.v_add_u32");
  if (!loom_amdgpu_descriptor_present(descriptor_set,
                                      LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32)) {
    return false;
  }
  if (loom_amdgpu_vector_iota_needs_dynamic_step_shift(element_count)) {
    *out_constraint_key = IREE_SV("descriptor.v_lshlrev_b32_lit");
    if (!loom_amdgpu_descriptor_present(
            descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT)) {
      return false;
    }
  }
  if (loom_amdgpu_vector_iota_needs_dynamic_step_multiply(element_count)) {
    *out_constraint_key = IREE_SV("descriptor.v_mul_lo_u32");
    if (!loom_amdgpu_descriptor_present(
            descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_LO_U32)) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_amdgpu_resolve_imm32_descriptor(
    loom_low_lower_context_t* context,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_low_lower_resolved_descriptor_t* out_descriptor,
    loom_string_id_t* out_imm32_attr_name_id, bool* out_present) {
  *out_imm32_attr_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, descriptor_ref, out_descriptor, out_present));
  if (!*out_present) {
    return iree_ok_status();
  }
  return loom_amdgpu_intern(context, IREE_SV("imm32"), out_imm32_attr_name_id);
}

static iree_status_t loom_amdgpu_select_u32_bit_pattern_constant_plan(
    loom_low_lower_context_t* context, uint32_t bit_pattern,
    loom_value_id_t result, loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_selected = false;
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  loom_string_id_t imm32_attr_name_id = LOOM_STRING_ID_INVALID;
  bool descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_imm32_descriptor(
      context, descriptor_ref, &descriptor, &imm32_attr_name_id,
      &descriptor_present));
  if (!descriptor_present) {
    return iree_ok_status();
  }
  *out_plan = (loom_amdgpu_constant_plan_t){
      .result = result,
      .descriptor = descriptor,
      .imm32_attr_name_id = imm32_attr_name_id,
      .register_count = 1,
  };
  out_plan->bit_patterns[0] = bit_pattern;
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_i32_constant_plan(
    loom_low_lower_context_t* context, loom_attribute_t value,
    loom_value_id_t result, loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  if (!loom_amdgpu_attr_is_i32_immediate(value)) {
    *out_selected = false;
    return iree_ok_status();
  }
  return loom_amdgpu_select_u32_bit_pattern_constant_plan(
      context, (uint32_t)(int32_t)value.i64, result, descriptor_ref, out_plan,
      out_selected);
}

static void loom_amdgpu_repeat_first_constant_bit_pattern(
    loom_amdgpu_constant_plan_t* plan, uint32_t register_count) {
  IREE_ASSERT_GT(register_count, 0);
  IREE_ASSERT_LE(register_count, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);
  const uint32_t bit_pattern = plan->bit_patterns[0];
  plan->register_count = register_count;
  for (uint32_t i = 1; i < register_count; ++i) {
    plan->bit_patterns[i] = bit_pattern;
  }
}

static iree_status_t loom_amdgpu_select_f32_constant_plan(
    loom_low_lower_context_t* context, loom_attribute_t value,
    loom_value_id_t result, uint32_t register_count,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_selected = false;
  if (!loom_amdgpu_attr_is_f32_immediate(value)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_u32_bit_pattern_constant_plan(
      context, loom_amdgpu_attr_f32_bit_pattern(value), result,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, out_plan, out_selected));
  if (!*out_selected) {
    return iree_ok_status();
  }
  loom_amdgpu_repeat_first_constant_bit_pattern(out_plan, register_count);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_packed_16bit_float_constant_plan(
    loom_low_lower_context_t* context, loom_type_t result_type,
    loom_attribute_t value, loom_value_id_t result,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_selected = false;
  uint32_t unused_payload_bit_count = 0;
  uint32_t register_count = 0;
  if (!loom_amdgpu_type_packed_16bit_float_storage(
          result_type, &unused_payload_bit_count, &register_count) ||
      !loom_amdgpu_attr_is_16bit_float_immediate(value)) {
    return iree_ok_status();
  }
  const uint32_t lane_bit_pattern = loom_amdgpu_attr_16bit_float_bit_pattern(
      loom_type_element_type(result_type), value);
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_u32_bit_pattern_constant_plan(
      context, lane_bit_pattern | (lane_bit_pattern << 16), result,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, out_plan, out_selected));
  if (!*out_selected) {
    return iree_ok_status();
  }
  loom_amdgpu_repeat_first_constant_bit_pattern(out_plan, register_count);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_index_constant_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_constant_plan_t){0};
  *out_selected = false;
  const loom_value_id_t result = loom_index_constant_result(source_op);
  const loom_attribute_t value = loom_index_constant_value(source_op);
  if (!loom_amdgpu_value_is_address_scalar(context, result) ||
      !loom_amdgpu_attr_is_u32_address_immediate(value)) {
    return iree_ok_status();
  }
  const loom_amdgpu_descriptor_ref_t descriptor_ref =
      loom_amdgpu_value_prefers_vgpr(context, result)
          ? LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32
          : LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32;
  return loom_amdgpu_select_u32_bit_pattern_constant_plan(
      context, (uint32_t)value.i64, result, descriptor_ref, out_plan,
      out_selected);
}

static iree_status_t loom_amdgpu_select_i64_constant_plan(
    loom_low_lower_context_t* context, loom_attribute_t value,
    loom_value_id_t result, loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  const uint64_t bit_pattern = (uint64_t)value.i64;
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_u32_bit_pattern_constant_plan(
      context, (uint32_t)bit_pattern, result, descriptor_ref, out_plan,
      out_selected));
  if (!*out_selected) {
    return iree_ok_status();
  }
  out_plan->register_count = 2;
  out_plan->bit_patterns[1] = (uint32_t)(bit_pattern >> 32);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_scalar_constant_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_constant_plan_t){0};
  *out_selected = false;
  const loom_value_id_t result = loom_scalar_constant_result(source_op);
  const loom_attribute_t value = loom_scalar_constant_value(source_op);
  if (loom_amdgpu_value_is_f32(context, result)) {
    return loom_amdgpu_select_f32_constant_plan(
        context, value, result, /*register_count=*/1, out_plan, out_selected);
  }
  if (loom_amdgpu_value_is_16bit_float(context, result)) {
    if (!loom_amdgpu_attr_is_16bit_float_immediate(value)) {
      return iree_ok_status();
    }
    const loom_type_t result_type =
        loom_module_value_type(loom_low_lower_context_module(context), result);
    return loom_amdgpu_select_u32_bit_pattern_constant_plan(
        context,
        loom_amdgpu_attr_16bit_float_bit_pattern(
            loom_type_element_type(result_type), value),
        result, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, out_plan, out_selected);
  }
  const loom_type_t result_type =
      loom_module_value_type(loom_low_lower_context_module(context), result);
  if (loom_amdgpu_type_is_i64(result_type)) {
    const loom_amdgpu_descriptor_ref_t descriptor_ref =
        loom_amdgpu_value_prefers_vgpr(context, result)
            ? LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32
            : LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32;
    return loom_amdgpu_select_i64_constant_plan(
        context, value, result, descriptor_ref, out_plan, out_selected);
  }
  if (!loom_amdgpu_type_is_i32(result_type)) {
    return iree_ok_status();
  }
  const loom_amdgpu_descriptor_ref_t descriptor_ref =
      loom_amdgpu_value_prefers_vgpr(context, result)
          ? LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32
          : LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32;
  return loom_amdgpu_select_i32_constant_plan(
      context, value, result, descriptor_ref, out_plan, out_selected);
}

static iree_status_t loom_amdgpu_select_vector_constant_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_constant_plan_t){0};
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t result = loom_vector_constant_result(source_op);
  const loom_type_t result_type = loom_module_value_type(module, result);
  const loom_attribute_t value = loom_vector_constant_value(source_op);
  const uint32_t i32_register_count =
      loom_amdgpu_vector_i32_register_count(result_type);
  if (i32_register_count != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_select_i32_constant_plan(
        context, value, result, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, out_plan,
        out_selected));
    if (!*out_selected) {
      return iree_ok_status();
    }
    loom_amdgpu_repeat_first_constant_bit_pattern(out_plan, i32_register_count);
    return iree_ok_status();
  }
  const uint32_t f32_register_count =
      loom_amdgpu_vector_f32_register_count(result_type);
  if (f32_register_count != 0) {
    return loom_amdgpu_select_f32_constant_plan(
        context, value, result, f32_register_count, out_plan, out_selected);
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_packed_16bit_float_constant_plan(
      context, result_type, value, result, out_plan, out_selected));
  if (*out_selected) {
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_vector_iota_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_iota_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_vector_iota_plan_t){0};
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t result = loom_vector_iota_result(source_op);
  const uint32_t element_count = loom_amdgpu_vector_i32_register_count(
      loom_module_value_type(module, result));
  if (element_count == 0) {
    return iree_ok_status();
  }
  const loom_value_id_t base_id = loom_vector_iota_base(source_op);
  const loom_value_id_t step_id = loom_vector_iota_step(source_op);
  int64_t base = 0;
  int64_t step = 0;
  const bool has_static_base =
      loom_amdgpu_value_as_i32_constant(context, base_id, &base);
  const bool has_static_step =
      loom_amdgpu_value_as_i32_constant(context, step_id, &step);
  out_plan->base = base_id;
  out_plan->step = step_id;
  out_plan->result = result;
  out_plan->lane_count = element_count;

  iree_string_view_t constraint_key = iree_string_view_empty();
  if (!loom_amdgpu_vector_iota_source_supported(
          module, loom_low_lower_context_descriptor_set(context), source_op,
          &constraint_key)) {
    return iree_ok_status();
  }

  if (!has_static_base || !has_static_step) {
    out_plan->is_dynamic = true;
    *out_selected = true;
    return iree_ok_status();
  }

  for (uint32_t i = 0; i < element_count; ++i) {
    int64_t lane_value = 0;
    if (!loom_amdgpu_iota_i32_lane_value(base, step, i, &lane_value)) {
      return iree_ok_status();
    }
    out_plan->lane_bit_patterns[i] = (uint32_t)(int32_t)lane_value;
  }
  bool descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_imm32_descriptor(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, &out_plan->descriptor,
      &out_plan->imm32_attr_name_id, &descriptor_present));
  if (!descriptor_present) {
    return iree_ok_status();
  }
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_vector_iota(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);
  iree_string_view_t constraint_key = iree_string_view_empty();
  if (loom_amdgpu_vector_iota_source_supported(
          module, loom_target_low_legality_descriptor_set(context), op,
          &constraint_key)) {
    return iree_ok_status();
  }
  return loom_amdgpu_low_legality_reject(context, op, constraint_key);
}

static bool loom_amdgpu_select_vector_extract_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_extract_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_vector_extract_plan_t){0};
  loom_attribute_t static_indices =
      loom_vector_extract_static_indices(source_op);
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY) {
    return false;
  }
  const loom_value_slice_t indices = loom_vector_extract_indices(source_op);

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t source = loom_vector_extract_source(source_op);
  const loom_value_id_t result = loom_vector_extract_result(source_op);
  const loom_type_t source_type = loom_module_value_type(module, source);
  const loom_type_t result_type = loom_module_value_type(module, result);

  uint32_t lane_count = loom_amdgpu_vector_32bit_lane_count(source_type);
  uint32_t register_count = lane_count;
  uint32_t result_register_count = 1;
  uint32_t lane_bit_count = 32;
  if (lane_count != 0) {
    if (loom_type_is_scalar(result_type)) {
      if (loom_type_element_type(source_type) == LOOM_SCALAR_TYPE_I32 &&
          !loom_amdgpu_type_is_i32(result_type)) {
        return false;
      }
      if (loom_type_element_type(source_type) == LOOM_SCALAR_TYPE_F32 &&
          !loom_amdgpu_type_is_f32(result_type)) {
        return false;
      }
    } else {
      result_register_count =
          loom_amdgpu_vector_32bit_register_count(result_type);
      if (result_register_count == 0 ||
          !loom_type_element_type_equals(source_type, result_type)) {
        return false;
      }
    }
  } else {
    register_count = loom_amdgpu_vector_32bit_register_count(source_type);
    if (register_count != 0) {
      if (loom_type_is_scalar(result_type)) {
        const loom_scalar_type_t element_type =
            loom_type_element_type(source_type);
        if ((element_type == LOOM_SCALAR_TYPE_I32 &&
             !loom_amdgpu_type_is_i32(result_type)) ||
            (element_type == LOOM_SCALAR_TYPE_F32 &&
             !loom_amdgpu_type_is_f32(result_type))) {
          return false;
        }
      } else {
        result_register_count =
            loom_amdgpu_vector_32bit_register_count(result_type);
        if (result_register_count == 0 ||
            !loom_type_element_type_equals(source_type, result_type)) {
          return false;
        }
      }
      lane_count = register_count;
    } else {
      uint32_t payload_bit_count = 0;
      if (!loom_amdgpu_type_packed_16bit_float_storage(
              source_type, &payload_bit_count, &register_count)) {
        return false;
      }
      lane_count = payload_bit_count / 16u;
      lane_bit_count = 16;
      if (!loom_type_is_scalar(result_type) ||
          loom_type_element_type(result_type) !=
              loom_type_element_type(source_type)) {
        return false;
      }
    }
  }
  if (lane_count == 0 || register_count == 0 || result_register_count == 0) {
    return false;
  }

  if (static_indices.count > loom_type_rank(source_type)) {
    return false;
  }
  if (loom_type_is_scalar(result_type)) {
    if (static_indices.count != loom_type_rank(source_type)) {
      return false;
    }
  } else if (static_indices.count + loom_type_rank(result_type) !=
             loom_type_rank(source_type)) {
    return false;
  }

  bool is_dynamic = false;
  uint32_t lane_offset = 0;
  loom_value_id_t dynamic_index = LOOM_VALUE_ID_INVALID;
  if (static_indices.count == 1 && static_indices.i64_array[0] == INT64_MIN) {
    if (indices.count != 1 || !loom_type_is_scalar(result_type)) {
      return false;
    }
    is_dynamic = true;
    dynamic_index = indices.values[0];
  } else {
    if (indices.count != 0) {
      return false;
    }
    int64_t source_indices[LOOM_TYPE_MAX_RANK] = {0};
    for (uint16_t i = 0; i < static_indices.count; ++i) {
      const int64_t index = static_indices.i64_array[i];
      if (index < 0 || index == INT64_MIN) {
        return false;
      }
      source_indices[i] = index;
    }
    const uint32_t source_lane_limit =
        lane_bit_count == 16 ? lane_count : register_count;
    const uint32_t result_lane_count =
        lane_bit_count == 16 ? 1u : result_register_count;
    if (!loom_amdgpu_static_vector_flat_register_from_indices(
            source_type, source_indices, &lane_offset) ||
        (uint64_t)lane_offset + result_lane_count >
            (uint64_t)source_lane_limit) {
      return false;
    }
  }

  *out_plan = (loom_amdgpu_vector_extract_plan_t){
      .source = source,
      .dynamic_index = dynamic_index,
      .result = result,
      .lane_offset = lane_offset,
      .lane_count = lane_count,
      .register_count = register_count,
      .result_register_count = result_register_count,
      .lane_bit_count = lane_bit_count,
      .is_dynamic = is_dynamic,
  };
  return true;
}

static bool loom_amdgpu_select_vector_from_elements_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_from_elements_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_vector_from_elements_plan_t){0};
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_slice_t elements =
      loom_vector_from_elements_elements(source_op);
  if (elements.count == 0) {
    return false;
  }
  const loom_value_id_t result = loom_vector_from_elements_result(source_op);
  const loom_type_t result_type = loom_module_value_type(module, result);
  loom_amdgpu_vector_storage_t storage = {0};
  if (!loom_amdgpu_type_vector_storage(result_type, &storage) ||
      elements.count != storage.element_count ||
      elements.count > IREE_ARRAYSIZE(out_plan->elements)) {
    return false;
  }
  for (uint32_t i = 0; i < elements.count; ++i) {
    const loom_value_id_t element = elements.values[i];
    const loom_type_t source_type = loom_module_value_type(module, element);
    if (!loom_type_is_scalar(source_type) ||
        loom_type_element_type(source_type) != storage.element_type) {
      return false;
    }
    if (storage.element_type == LOOM_SCALAR_TYPE_I32 &&
        !loom_amdgpu_value_can_materialize_as_vgpr_i32(context, element)) {
      return false;
    }
    out_plan->elements[i] = element;
  }
  out_plan->result = result;
  out_plan->element_count = elements.count;
  out_plan->storage_kind = storage.kind;
  out_plan->register_count = storage.register_count;
  out_plan->element_register_count = storage.element_register_count;
  out_plan->element_bit_count = storage.element_bit_count;
  out_plan->element_type = storage.element_type;
  return true;
}

static bool loom_amdgpu_select_vector_splat_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_from_elements_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_vector_from_elements_plan_t){0};
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t result = loom_vector_splat_result(source_op);
  const loom_type_t result_type = loom_module_value_type(module, result);
  loom_amdgpu_vector_storage_t storage = {0};
  if (!loom_amdgpu_type_vector_storage(result_type, &storage) ||
      storage.element_count > IREE_ARRAYSIZE(out_plan->elements)) {
    return false;
  }
  const loom_value_id_t scalar = loom_vector_splat_scalar(source_op);
  const loom_type_t scalar_type = loom_module_value_type(module, scalar);
  if (!loom_type_is_scalar(scalar_type) ||
      loom_type_element_type(scalar_type) != storage.element_type) {
    return false;
  }
  if (storage.element_type == LOOM_SCALAR_TYPE_I32 &&
      !loom_amdgpu_value_can_materialize_as_vgpr_i32(context, scalar)) {
    return false;
  }
  for (uint32_t i = 0; i < storage.element_count; ++i) {
    out_plan->elements[i] = scalar;
  }
  out_plan->result = result;
  out_plan->storage_kind = storage.kind;
  out_plan->element_count = storage.element_count;
  out_plan->register_count = storage.register_count;
  out_plan->element_register_count = storage.element_register_count;
  out_plan->element_bit_count = storage.element_bit_count;
  out_plan->element_type = storage.element_type;
  return true;
}

static bool loom_amdgpu_select_vector_insert_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_insert_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_vector_insert_plan_t){0};
  loom_attribute_t static_indices =
      loom_vector_insert_static_indices(source_op);
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY || static_indices.count != 1) {
    return false;
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t value = loom_vector_insert_value(source_op);
  const loom_value_id_t dest = loom_vector_insert_dest(source_op);
  const loom_value_id_t result = loom_vector_insert_result(source_op);
  const loom_type_t value_type = loom_module_value_type(module, value);
  const loom_type_t dest_type = loom_module_value_type(module, dest);
  const loom_type_t result_type = loom_module_value_type(module, result);
  if (!loom_type_equal(dest_type, result_type)) {
    return false;
  }

  uint32_t lane_count = loom_amdgpu_vector_32bit_lane_count(dest_type);
  uint32_t register_count = lane_count;
  uint32_t lane_bit_count = 32;
  if (lane_count == 0) {
    uint32_t payload_bit_count = 0;
    if (!loom_amdgpu_type_packed_16bit_float_storage(
            dest_type, &payload_bit_count, &register_count)) {
      return false;
    }
    lane_count = payload_bit_count / 16u;
    lane_bit_count = 16;
  }
  if (lane_count == 0 || register_count == 0 ||
      !loom_type_is_scalar(value_type)) {
    return false;
  }
  const loom_scalar_type_t element_type = loom_type_element_type(dest_type);
  if (loom_type_element_type(value_type) != element_type) {
    return false;
  }
  if (element_type != LOOM_SCALAR_TYPE_I32 &&
      element_type != LOOM_SCALAR_TYPE_F32 &&
      element_type != LOOM_SCALAR_TYPE_F16 &&
      element_type != LOOM_SCALAR_TYPE_BF16) {
    return false;
  }
  if (element_type == LOOM_SCALAR_TYPE_I32 &&
      !loom_amdgpu_value_can_materialize_as_vgpr_i32(context, value)) {
    return false;
  }

  bool is_dynamic = false;
  uint32_t lane_offset = 0;
  loom_value_id_t dynamic_index = LOOM_VALUE_ID_INVALID;
  const loom_value_slice_t indices = loom_vector_insert_indices(source_op);
  if (static_indices.i64_array[0] == INT64_MIN) {
    if (indices.count != 1) {
      return false;
    }
    is_dynamic = true;
    dynamic_index = indices.values[0];
  } else {
    if (indices.count != 0 || static_indices.i64_array[0] < 0 ||
        static_indices.i64_array[0] > UINT32_MAX) {
      return false;
    }
    lane_offset = (uint32_t)static_indices.i64_array[0];
    if (lane_offset >= lane_count) {
      return false;
    }
  }

  *out_plan = (loom_amdgpu_vector_insert_plan_t){
      .value = value,
      .dest = dest,
      .dynamic_index = dynamic_index,
      .result = result,
      .lane_offset = lane_offset,
      .lane_count = lane_count,
      .register_count = register_count,
      .lane_bit_count = lane_bit_count,
      .element_type = element_type,
      .is_dynamic = is_dynamic,
  };
  return true;
}

static bool loom_amdgpu_type_is_bf16_packed_vector(
    loom_type_t type, uint32_t* out_lane_count, uint32_t* out_register_count) {
  *out_lane_count = 0;
  *out_register_count = 0;
  if (loom_type_element_type(type) != LOOM_SCALAR_TYPE_BF16) {
    return false;
  }
  uint32_t payload_bit_count = 0;
  if (!loom_amdgpu_type_packed_16bit_float_storage(type, &payload_bit_count,
                                                   out_register_count)) {
    return false;
  }
  *out_lane_count = payload_bit_count / 16u;
  return true;
}

static bool loom_amdgpu_select_vector_bf16_conversion_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_bf16_conversion_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_vector_bf16_conversion_plan_t){0};
  const loom_module_t* module = loom_low_lower_context_module(context);

  loom_value_id_t source = LOOM_VALUE_ID_INVALID;
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_vector_bf16_conversion_kind_t kind =
      LOOM_AMDGPU_VECTOR_BF16_CONVERSION_KIND_NONE;
  switch (source_op->kind) {
    case LOOM_OP_VECTOR_EXTF:
      source = loom_vector_extf_input(source_op);
      result = loom_vector_extf_result(source_op);
      kind = LOOM_AMDGPU_VECTOR_BF16_CONVERSION_KIND_EXTF;
      break;
    case LOOM_OP_VECTOR_FPTRUNC:
      source = loom_vector_fptrunc_input(source_op);
      result = loom_vector_fptrunc_result(source_op);
      kind = LOOM_AMDGPU_VECTOR_BF16_CONVERSION_KIND_FPTRUNC;
      break;
    default:
      return false;
  }

  const loom_type_t source_type = loom_module_value_type(module, source);
  const loom_type_t result_type = loom_module_value_type(module, result);
  uint32_t source_lane_count = 0;
  uint32_t source_register_count = 0;
  uint32_t result_lane_count = 0;
  uint32_t result_register_count = 0;
  if (kind == LOOM_AMDGPU_VECTOR_BF16_CONVERSION_KIND_EXTF) {
    if (!loom_amdgpu_type_is_bf16_packed_vector(source_type, &source_lane_count,
                                                &source_register_count)) {
      return false;
    }
    result_lane_count = loom_amdgpu_vector_f32_register_count(result_type);
    result_register_count = result_lane_count;
  } else {
    source_lane_count = loom_amdgpu_vector_f32_register_count(source_type);
    source_register_count = source_lane_count;
    if (!loom_amdgpu_type_is_bf16_packed_vector(result_type, &result_lane_count,
                                                &result_register_count)) {
      return false;
    }
  }
  if (source_lane_count == 0 || source_lane_count != result_lane_count) {
    return false;
  }

  *out_plan = (loom_amdgpu_vector_bf16_conversion_plan_t){
      .kind = kind,
      .source = source,
      .result = result,
      .lane_count = source_lane_count,
      .source_register_count = source_register_count,
      .result_register_count = result_register_count,
  };
  return true;
}

static iree_status_t loom_amdgpu_select_index_cast_alias_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_cast_alias_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_cast_alias_plan_t){0};
  *out_selected = false;
  const loom_value_id_t source = loom_index_cast_input(source_op);
  const loom_value_id_t result = loom_index_cast_result(source_op);

  loom_type_t source_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, source, &source_low_type));
  if (!loom_low_type_is_register(source_low_type)) {
    return iree_ok_status();
  }
  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op, result,
                                                   &result_low_type));
  if (!loom_type_equal(source_low_type, result_low_type)) {
    return iree_ok_status();
  }

  *out_plan = (loom_amdgpu_cast_alias_plan_t){
      .source = source,
      .result = result,
  };
  *out_selected = true;
  return iree_ok_status();
}

static bool loom_amdgpu_select_scalar_trunci_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_scalar_trunci_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_scalar_trunci_plan_t){0};
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t source = loom_scalar_trunci_input(source_op);
  const loom_value_id_t result = loom_scalar_trunci_result(source_op);
  const loom_type_t source_type = loom_module_value_type(module, source);
  const loom_type_t result_type = loom_module_value_type(module, result);
  if (!loom_amdgpu_type_is_i64(source_type) ||
      !loom_amdgpu_type_is_i32(result_type)) {
    return false;
  }
  *out_plan = (loom_amdgpu_scalar_trunci_plan_t){
      .source = source,
      .result = result,
  };
  return true;
}

static iree_status_t loom_amdgpu_select_scalar_extsi_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_scalar_extsi_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_scalar_extsi_plan_t){0};
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t source = loom_scalar_extsi_input(source_op);
  const loom_value_id_t result = loom_scalar_extsi_result(source_op);
  const loom_type_t source_type = loom_module_value_type(module, source);
  const loom_type_t result_type = loom_module_value_type(module, result);
  if (!loom_amdgpu_type_is_i32(source_type) ||
      !loom_amdgpu_type_is_i64(result_type)) {
    return iree_ok_status();
  }

  loom_low_lower_resolved_descriptor_t unused_descriptor = {0};
  bool descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT, &unused_descriptor,
      &descriptor_present));
  if (!descriptor_present) {
    return iree_ok_status();
  }

  *out_plan = (loom_amdgpu_scalar_extsi_plan_t){
      .source = source,
      .result = result,
  };
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_value_plan(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op,
                                            loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  switch (source_op->kind) {
    case LOOM_OP_INDEX_CONSTANT:
    case LOOM_OP_SCALAR_CONSTANT:
    case LOOM_OP_VECTOR_CONSTANT: {
      loom_amdgpu_constant_plan_t* plan_data = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
          context, sizeof(*plan_data), (void**)&plan_data));
      bool selected = false;
      if (source_op->kind == LOOM_OP_INDEX_CONSTANT) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_select_index_constant_plan(
            context, source_op, plan_data, &selected));
      } else if (source_op->kind == LOOM_OP_SCALAR_CONSTANT) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_select_scalar_constant_plan(
            context, source_op, plan_data, &selected));
      } else {
        IREE_RETURN_IF_ERROR(loom_amdgpu_select_vector_constant_plan(
            context, source_op, plan_data, &selected));
      }
      if (selected) {
        *out_plan = loom_low_lower_plan_make(source_op->kind, plan_data);
      }
      return iree_ok_status();
    }
    case LOOM_OP_INDEX_CAST: {
      loom_amdgpu_cast_alias_plan_t* plan_data = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
          context, sizeof(*plan_data), (void**)&plan_data));
      bool selected = false;
      IREE_RETURN_IF_ERROR(loom_amdgpu_select_index_cast_alias_plan(
          context, source_op, plan_data, &selected));
      if (selected) {
        *out_plan = loom_low_lower_plan_make(source_op->kind, plan_data);
      }
      return iree_ok_status();
    }
    case LOOM_OP_SCALAR_TRUNCI: {
      loom_amdgpu_scalar_trunci_plan_t* plan_data = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
          context, sizeof(*plan_data), (void**)&plan_data));
      if (loom_amdgpu_select_scalar_trunci_plan(context, source_op,
                                                plan_data)) {
        *out_plan = loom_low_lower_plan_make(source_op->kind, plan_data);
      }
      return iree_ok_status();
    }
    case LOOM_OP_SCALAR_EXTSI: {
      loom_amdgpu_scalar_extsi_plan_t* plan_data = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
          context, sizeof(*plan_data), (void**)&plan_data));
      bool selected = false;
      IREE_RETURN_IF_ERROR(loom_amdgpu_select_scalar_extsi_plan(
          context, source_op, plan_data, &selected));
      if (selected) {
        *out_plan = loom_low_lower_plan_make(source_op->kind, plan_data);
      }
      return iree_ok_status();
    }
    case LOOM_OP_VECTOR_IOTA: {
      loom_amdgpu_vector_iota_plan_t* plan_data = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
          context, sizeof(*plan_data), (void**)&plan_data));
      bool selected = false;
      IREE_RETURN_IF_ERROR(loom_amdgpu_select_vector_iota_plan(
          context, source_op, plan_data, &selected));
      if (selected) {
        *out_plan = loom_low_lower_plan_make(source_op->kind, plan_data);
      }
      return iree_ok_status();
    }
    case LOOM_OP_VECTOR_EXTRACT: {
      loom_amdgpu_vector_extract_plan_t* plan_data = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
          context, sizeof(*plan_data), (void**)&plan_data));
      if (loom_amdgpu_select_vector_extract_plan(context, source_op,
                                                 plan_data)) {
        *out_plan = loom_low_lower_plan_make(source_op->kind, plan_data);
      }
      return iree_ok_status();
    }
    case LOOM_OP_VECTOR_FROM_ELEMENTS: {
      loom_amdgpu_vector_from_elements_plan_t* plan_data = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
          context, sizeof(*plan_data), (void**)&plan_data));
      if (loom_amdgpu_select_vector_from_elements_plan(context, source_op,
                                                       plan_data)) {
        *out_plan = loom_low_lower_plan_make(source_op->kind, plan_data);
      }
      return iree_ok_status();
    }
    case LOOM_OP_VECTOR_SPLAT: {
      loom_amdgpu_vector_from_elements_plan_t* plan_data = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
          context, sizeof(*plan_data), (void**)&plan_data));
      if (loom_amdgpu_select_vector_splat_plan(context, source_op, plan_data)) {
        *out_plan = loom_low_lower_plan_make(source_op->kind, plan_data);
      }
      return iree_ok_status();
    }
    case LOOM_OP_VECTOR_INSERT: {
      loom_amdgpu_vector_insert_plan_t* plan_data = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
          context, sizeof(*plan_data), (void**)&plan_data));
      if (loom_amdgpu_select_vector_insert_plan(context, source_op,
                                                plan_data)) {
        *out_plan = loom_low_lower_plan_make(source_op->kind, plan_data);
      }
      return iree_ok_status();
    }
    case LOOM_OP_VECTOR_EXTF:
    case LOOM_OP_VECTOR_FPTRUNC: {
      loom_amdgpu_vector_bf16_conversion_plan_t* plan_data = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
          context, sizeof(*plan_data), (void**)&plan_data));
      if (loom_amdgpu_select_vector_bf16_conversion_plan(context, source_op,
                                                         plan_data)) {
        *out_plan = loom_low_lower_plan_make(source_op->kind, plan_data);
      }
      return iree_ok_status();
    }
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_amdgpu_bind_register_u32_lane_constants(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_string_id_t imm32_attr_name_id, const uint32_t* lane_bit_patterns,
    uint32_t lane_count) {
  IREE_ASSERT_GT(lane_count, 0);
  IREE_ASSERT_LE(lane_count, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));
  IREE_ASSERT(loom_low_type_is_register(result_type));
  IREE_ASSERT_EQ(loom_low_register_type_unit_count(result_type), lane_count);
  const loom_type_t lane_type =
      loom_low_register_type_with_unit_count(result_type, 1);

  loom_value_id_t low_lane_values[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
        context, source_op, descriptor, imm32_attr_name_id,
        lane_bit_patterns[i], lane_type, &low_lane_values[i]));
  }

  if (lane_count == 1) {
    return loom_low_lower_bind_value(context, source_result,
                                     low_lane_values[0]);
  }

  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), low_lane_values, lane_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, source_result,
                                   loom_low_concat_result(concat_op));
}

static iree_status_t loom_amdgpu_lower_u32_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_string_id_t imm32_attr_name_id, uint32_t bit_pattern,
    loom_value_id_t source_result) {
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));

  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
      context, source_op, descriptor, imm32_attr_name_id, bit_pattern,
      result_type, &low_result));
  return loom_low_lower_bind_value(context, source_result, low_result);
}

static iree_status_t loom_amdgpu_lower_constant_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_constant_plan_t* plan) {
  if (plan->register_count == 1) {
    return loom_amdgpu_lower_u32_constant(context, source_op, &plan->descriptor,
                                          plan->imm32_attr_name_id,
                                          plan->bit_patterns[0], plan->result);
  }
  return loom_amdgpu_bind_register_u32_lane_constants(
      context, source_op, plan->result, &plan->descriptor,
      plan->imm32_attr_name_id, plan->bit_patterns, plan->register_count);
}

static iree_status_t loom_amdgpu_lower_vector_iota(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_iota_plan_t* plan) {
  if (!plan->is_dynamic) {
    return loom_amdgpu_bind_register_u32_lane_constants(
        context, source_op, plan->result, &plan->descriptor,
        plan->imm32_attr_name_id, plan->lane_bit_patterns, plan->lane_count);
  }

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_value_id_t low_base = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
      context, source_op, plan->base, &low_base));

  int64_t exact_step = 0;
  const bool has_exact_step =
      loom_amdgpu_value_as_i32_constant(context, plan->step, &exact_step);
  loom_value_id_t low_step = LOOM_VALUE_ID_INVALID;
  if (!has_exact_step) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
        context, source_op, plan->step, &low_step));
  }

  loom_value_id_t lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES] = {0};
  lanes[0] = low_base;
  for (uint32_t i = 1; i < plan->lane_count; ++i) {
    if (has_exact_step) {
      int64_t lane_offset = 0;
      const bool lane_offset_in_range =
          iree_checked_mul_i64((int64_t)i, exact_step, &lane_offset) &&
          lane_offset >= INT32_MIN && lane_offset <= INT32_MAX;
      IREE_ASSERT(lane_offset_in_range);
      if (lane_offset == 0) {
        lanes[i] = low_base;
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
          low_base, (uint32_t)(int32_t)lane_offset, lane_type, &lanes[i]));
      continue;
    }

    loom_value_id_t scaled_step = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scale_u32(
        context, source_op, low_step, i, LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_NONE,
        lane_type, &scaled_step));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32, low_base,
        scaled_step, lane_type, &lanes[i]));
  }

  if (plan->lane_count == 1) {
    return loom_low_lower_bind_value(context, plan->result, lanes[0]);
  }
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), lanes, plan->lane_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->result,
                                   loom_low_concat_result(concat_op));
}

static iree_status_t loom_amdgpu_extract_register_unit(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t register_count,
    uint32_t register_offset, loom_type_t unit_type,
    loom_value_id_t* out_register_unit) {
  *out_register_unit = LOOM_VALUE_ID_INVALID;
  if (register_count == 1) {
    *out_register_unit = low_source;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_low_slice(context, source_op, low_source,
                                    register_offset, unit_type,
                                    out_register_unit);
}

static iree_status_t loom_amdgpu_extract_packed_register_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, const loom_amdgpu_vector_extract_plan_t* plan,
    uint32_t lane_offset, loom_type_t lane_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT(plan->lane_bit_count == 16 || plan->lane_bit_count == 32);
  const uint32_t lanes_per_register = 32u / plan->lane_bit_count;
  const uint32_t register_offset = lane_offset / lanes_per_register;
  const uint32_t register_bit_offset =
      (lane_offset % lanes_per_register) * plan->lane_bit_count;
  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_register_unit(
      context, source_op, low_source, plan->register_count, register_offset,
      lane_type, &source_register));
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      register_bit_offset, source_register, lane_type, out_lane);
}

static loom_type_t loom_amdgpu_low_register_lane_type(
    const loom_module_t* module, loom_value_id_t low_value) {
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  if (!loom_low_type_is_register(low_type)) {
    return loom_type_none();
  }
  return loom_low_register_type_with_unit_count(low_type, 1);
}

static iree_status_t loom_amdgpu_lower_static_vector_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_extract_plan_t* plan) {
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));
  if (plan->lane_offset == 0 &&
      plan->result_register_count == plan->register_count) {
    return loom_low_lower_bind_value(context, plan->result, low_source);
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_type_t register_type =
      loom_amdgpu_low_register_lane_type(module, low_source);
  if (loom_type_kind(register_type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &register_type));
  }

  if (plan->result_register_count == 1) {
    loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_packed_register_lane(
        context, source_op, low_source, plan, plan->lane_offset, register_type,
        &low_result));
    return loom_low_lower_bind_value(context, plan->result, low_result);
  }

  loom_value_id_t registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->result_register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_packed_register_lane(
        context, source_op, low_source, plan, plan->lane_offset + i,
        register_type, &registers[i]));
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(loom_low_lower_context_builder(context), registers,
                            plan->result_register_count, result_type,
                            source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->result,
                                   loom_low_concat_result(concat_op));
}

static iree_status_t loom_amdgpu_lower_dynamic_vector_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_extract_plan_t* plan) {
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));
  if (plan->lane_count == 1) {
    return loom_low_lower_bind_value(context, plan->result, low_source);
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_type_t source_lane_type =
      loom_amdgpu_low_register_lane_type(module, low_source);
  if (loom_type_kind(source_lane_type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_vgpr_type(context, &source_lane_type));
  }
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_type_t mask_lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_lane_type));

  loom_value_id_t selected_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_packed_register_lane(
      context, source_op, low_source, plan, 0, source_lane_type,
      &selected_lane));
  if (!loom_type_equal(source_lane_type, lane_type)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_b32_copy(
        context, source_op, selected_lane, &selected_lane));
  }

  loom_value_id_t index_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
      context, source_op, plan->dynamic_index, &index_lane));
  for (uint32_t i = 1; i < plan->lane_count; ++i) {
    loom_value_id_t ordinal = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, i, lane_type,
        &ordinal));

    const loom_value_id_t compare_operands[] = {
        index_lane,
        ordinal,
    };
    loom_op_t* compare_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
        compare_operands, IREE_ARRAYSIZE(compare_operands),
        loom_make_named_attr_slice(NULL, 0), &mask_lane_type, 1, &compare_op));

    loom_value_id_t table_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_packed_register_lane(
        context, source_op, low_source, plan, i, source_lane_type,
        &table_lane));
    if (!loom_type_equal(source_lane_type, lane_type)) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_b32_copy(
          context, source_op, table_lane, &table_lane));
    }
    const loom_value_id_t select_operands[] = {
        selected_lane,
        table_lane,
        loom_value_slice_get(loom_low_op_results(compare_op), 0),
    };
    loom_op_t* select_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
        select_operands, IREE_ARRAYSIZE(select_operands),
        loom_make_named_attr_slice(NULL, 0), &lane_type, 1, &select_op));
    selected_lane = loom_value_slice_get(loom_low_op_results(select_op), 0);
  }
  return loom_low_lower_bind_value(context, plan->result, selected_lane);
}

static iree_status_t loom_amdgpu_lower_vector_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_extract_plan_t* plan) {
  return plan->is_dynamic ? loom_amdgpu_lower_dynamic_vector_extract(
                                context, source_op, plan)
                          : loom_amdgpu_lower_static_vector_extract(
                                context, source_op, plan);
}

static iree_status_t loom_amdgpu_lookup_or_materialize_vgpr_16bit_float(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_type_t lane_type,
    loom_value_id_t* out_low_value) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, out_low_value));
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, *out_low_value, out_low_value));
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      *out_low_value, UINT32_C(0xFFFF), lane_type, out_low_value);
}

static iree_status_t loom_amdgpu_lower_vector_from_16bit_elements(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_from_elements_plan_t* plan) {
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));

  loom_value_id_t registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t register_index = 0; register_index < plan->register_count;
       ++register_index) {
    const uint32_t lane_base = register_index * 2u;
    loom_value_id_t packed = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_16bit_float(
        context, source_op, plan->elements[lane_base], lane_type, &packed));
    if (lane_base + 1u < plan->element_count) {
      loom_value_id_t high_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_16bit_float(
          context, source_op, plan->elements[lane_base + 1u], lane_type,
          &high_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 16,
          high_lane, lane_type, &high_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, packed,
          high_lane, lane_type, &packed));
    }
    registers[register_index] = packed;
  }

  if (plan->register_count == 1) {
    return loom_low_lower_bind_value(context, plan->result, registers[0]);
  }
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), registers, plan->register_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->result,
                                   loom_low_concat_result(concat_op));
}

static iree_status_t loom_amdgpu_lower_vector_from_packed_integer_elements(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_from_elements_plan_t* plan) {
  const uint32_t element_bit_count = plan->element_bit_count;
  IREE_ASSERT_TRUE(element_bit_count == 8 || element_bit_count == 16);
  const uint32_t element_mask = (UINT32_C(1) << element_bit_count) - 1u;
  const uint32_t elements_per_register = 32u / element_bit_count;

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_value_id_t registers[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
  for (uint32_t register_index = 0; register_index < plan->register_count;
       ++register_index) {
    loom_value_id_t packed = LOOM_VALUE_ID_INVALID;
    const uint32_t lane_base = register_index * elements_per_register;
    for (uint32_t lane_index = 0; lane_index < elements_per_register;
         ++lane_index) {
      const uint32_t element_index = lane_base + lane_index;
      if (element_index >= plan->element_count) {
        break;
      }

      loom_value_id_t low_element = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
          context, plan->elements[element_index], &low_element));
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
          context, source_op, low_element, &low_element));

      loom_value_id_t low_bits = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
          low_element, element_mask, lane_type, &low_bits));
      loom_value_id_t shifted = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
          lane_index * element_bit_count, low_bits, lane_type, &shifted));
      if (packed == LOOM_VALUE_ID_INVALID) {
        packed = shifted;
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, packed,
          shifted, lane_type, &packed));
    }
    registers[register_index] = packed;
  }

  if (plan->register_count == 1) {
    return loom_low_lower_bind_value(context, plan->result, registers[0]);
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), registers, plan->register_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->result,
                                   loom_low_concat_result(concat_op));
}

static iree_status_t loom_amdgpu_lower_vector_from_register_elements(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_from_elements_plan_t* plan) {
  loom_value_id_t elements[LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES];
  for (uint32_t i = 0; i < plan->element_count; ++i) {
    bool reused = false;
    for (uint32_t j = 0; j < i; ++j) {
      if (plan->elements[j] == plan->elements[i]) {
        elements[i] = elements[j];
        reused = true;
        break;
      }
    }
    if (reused) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_low_lower_lookup_value(context, plan->elements[i], &elements[i]));
    if (plan->storage_kind == LOOM_AMDGPU_VECTOR_STORAGE_KIND_I1_MASK) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
        context, source_op, elements[i], &elements[i]));
  }

  if (plan->element_count == 1) {
    return loom_low_lower_bind_value(context, plan->result, elements[0]);
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), elements, plan->element_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->result,
                                   loom_low_concat_result(concat_op));
}

static iree_status_t loom_amdgpu_lower_vector_from_elements(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_from_elements_plan_t* plan) {
  switch (plan->storage_kind) {
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT:
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_64BIT:
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_I1_MASK:
      return loom_amdgpu_lower_vector_from_register_elements(context, source_op,
                                                             plan);
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_16BIT_FLOAT:
      return loom_amdgpu_lower_vector_from_16bit_elements(context, source_op,
                                                          plan);
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER:
      return loom_amdgpu_lower_vector_from_packed_integer_elements(
          context, source_op, plan);
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_NONE:
    default:
      IREE_ASSERT_UNREACHABLE("unsupported vector element plan");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_amdgpu_lookup_vector_insert_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_insert_plan_t* plan, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t lane_type = loom_type_none();
  switch (plan->element_type) {
    case LOOM_SCALAR_TYPE_I32:
      return loom_amdgpu_lookup_or_materialize_vgpr_i32(context, source_op,
                                                        plan->value, out_value);
    case LOOM_SCALAR_TYPE_F32: {
      IREE_RETURN_IF_ERROR(
          loom_low_lower_lookup_value(context, plan->value, out_value));
      return loom_amdgpu_materialize_low_vgpr_b32(context, source_op,
                                                  *out_value, out_value);
    }
    case LOOM_SCALAR_TYPE_F16:
    case LOOM_SCALAR_TYPE_BF16: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
      return loom_amdgpu_lookup_or_materialize_vgpr_16bit_float(
          context, source_op, plan->value, lane_type, out_value);
    }
    default:
      IREE_ASSERT_UNREACHABLE("unsupported vector insert element plan");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_amdgpu_select_dynamic_insert_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t old_lane, loom_value_id_t new_lane,
    loom_value_id_t index_lane, uint32_t lane_ordinal, loom_type_t lane_type,
    loom_type_t mask_lane_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;

  loom_value_id_t ordinal = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, lane_ordinal,
      lane_type, &ordinal));

  const loom_value_id_t compare_operands[] = {
      index_lane,
      ordinal,
  };
  loom_op_t* compare_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
      compare_operands, IREE_ARRAYSIZE(compare_operands),
      loom_make_named_attr_slice(NULL, 0), &mask_lane_type, 1, &compare_op));

  const loom_value_id_t select_operands[] = {
      old_lane,
      new_lane,
      loom_value_slice_get(loom_low_op_results(compare_op), 0),
  };
  loom_op_t* select_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
      select_operands, IREE_ARRAYSIZE(select_operands),
      loom_make_named_attr_slice(NULL, 0), &lane_type, 1, &select_op));
  *out_lane = loom_value_slice_get(loom_low_op_results(select_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_replace_16bit_vector_register_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t old_register, loom_value_id_t low_value,
    uint32_t lane_ordinal, loom_type_t register_type,
    loom_value_id_t* out_register) {
  *out_register = LOOM_VALUE_ID_INVALID;
  const bool is_high_lane = (lane_ordinal & 1u) != 0;
  const uint32_t preserved_mask =
      is_high_lane ? UINT32_C(0x0000FFFF) : UINT32_C(0xFFFF0000);

  loom_value_id_t preserved = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      old_register, preserved_mask, register_type, &preserved));

  loom_value_id_t inserted = low_value;
  if (is_high_lane) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 16,
        low_value, register_type, &inserted));
  }
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, preserved,
      inserted, register_type, out_register);
}

static iree_status_t loom_amdgpu_lower_16bit_vector_insert(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_insert_plan_t* plan, loom_value_id_t low_value) {
  loom_value_id_t low_dest = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->dest, &low_dest));
  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_type_t register_type =
      loom_amdgpu_low_register_lane_type(module, low_dest);
  if (loom_type_kind(register_type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &register_type));
  }

  loom_value_id_t index_lane = LOOM_VALUE_ID_INVALID;
  loom_type_t mask_lane_type = loom_type_none();
  if (plan->is_dynamic) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
        context, source_op, plan->dynamic_index, &index_lane));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_sgpr_range_type(context, 2, &mask_lane_type));
  }

  loom_value_id_t registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t register_index = 0; register_index < plan->register_count;
       ++register_index) {
    loom_value_id_t old_register = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_register_unit(
        context, source_op, low_dest, plan->register_count, register_index,
        register_type, &old_register));

    loom_value_id_t selected_register = old_register;
    const uint32_t first_lane = register_index * 2u;
    const uint32_t end_lane = iree_min(first_lane + 2u, plan->lane_count);
    for (uint32_t lane_ordinal = first_lane; lane_ordinal < end_lane;
         ++lane_ordinal) {
      if (!plan->is_dynamic && lane_ordinal != plan->lane_offset) {
        continue;
      }

      loom_value_id_t replacement_register = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_replace_16bit_vector_register_lane(
          context, source_op, old_register, low_value, lane_ordinal,
          register_type, &replacement_register));
      if (!plan->is_dynamic) {
        selected_register = replacement_register;
        break;
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_select_dynamic_insert_lane(
          context, source_op, selected_register, replacement_register,
          index_lane, lane_ordinal, register_type, mask_lane_type,
          &selected_register));
    }
    registers[register_index] = selected_register;
  }

  if (plan->register_count == 1) {
    return loom_low_lower_bind_value(context, plan->result, registers[0]);
  }
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), registers, plan->register_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->result,
                                   loom_low_concat_result(concat_op));
}

static iree_status_t loom_amdgpu_lower_vector_insert(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_insert_plan_t* plan) {
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_vector_insert_value(
      context, source_op, plan, &low_value));
  if (plan->lane_count == 1) {
    return loom_low_lower_bind_value(context, plan->result, low_value);
  }
  if (plan->lane_bit_count == 16) {
    return loom_amdgpu_lower_16bit_vector_insert(context, source_op, plan,
                                                 low_value);
  }

  loom_value_id_t low_dest = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->dest, &low_dest));
  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_type_t lane_type = loom_amdgpu_low_register_lane_type(module, low_dest);
  if (loom_type_kind(lane_type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  }

  loom_value_id_t index_lane = LOOM_VALUE_ID_INVALID;
  loom_type_t mask_lane_type = loom_type_none();
  if (plan->is_dynamic) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
        context, source_op, plan->dynamic_index, &index_lane));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_sgpr_range_type(context, 2, &mask_lane_type));
  }

  loom_value_id_t lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->lane_count; ++i) {
    loom_value_id_t old_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_register_unit(
        context, source_op, low_dest, plan->register_count, i, lane_type,
        &old_lane));
    if (!plan->is_dynamic) {
      lanes[i] = i == plan->lane_offset ? low_value : old_lane;
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_select_dynamic_insert_lane(
        context, source_op, old_lane, low_value, index_lane, i, lane_type,
        mask_lane_type, &lanes[i]));
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), lanes, plan->lane_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->result,
                                   loom_low_concat_result(concat_op));
}

static iree_status_t loom_amdgpu_extract_16bit_float_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t source_register_count,
    uint32_t lane_index, loom_type_t source_lane_type,
    loom_type_t result_lane_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  const uint32_t register_index = lane_index / 2u;
  const uint32_t register_bit_offset = (lane_index % 2u) * 16u;
  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_register_unit(
      context, source_op, low_source, source_register_count, register_index,
      source_lane_type, &source_register));
  if (register_bit_offset == 0) {
    return loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 16,
        source_register, result_lane_type, out_lane);
  }
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      source_register, UINT32_C(0xFFFF0000), result_lane_type, out_lane);
}

iree_status_t loom_amdgpu_emit_f32_to_bf16_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_lane, loom_type_t lane_type,
    loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, source_lane, &source_lane));

  loom_value_id_t upper = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT, 16,
      source_lane, lane_type, &upper));
  loom_value_id_t lsb = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, upper, 1,
      lane_type, &lsb));
  loom_value_id_t bias = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT, lsb,
      UINT32_C(0x7FFF), lane_type, &bias));
  loom_value_id_t rounded = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32, source_lane,
      bias, lane_type, &rounded));
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT, 16,
      rounded, lane_type, out_lane);
}

static iree_status_t loom_amdgpu_emit_bf16_pack_descriptor(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t low_lane, loom_value_id_t high_lane, loom_type_t lane_type,
    loom_value_id_t* out_packed) {
  *out_packed = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {low_lane, high_lane};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_packed = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_native_f32_to_packed_bf16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t low_source_lane, loom_value_id_t high_source_lane,
    loom_type_t lane_type, loom_value_id_t* out_packed) {
  *out_packed = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, low_source_lane, &low_source_lane));
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, high_source_lane, &high_source_lane));
  return loom_amdgpu_emit_bf16_pack_descriptor(
      context, source_op, descriptor, low_source_lane, high_source_lane,
      lane_type, out_packed);
}

static iree_status_t loom_amdgpu_lower_vector_bf16_extf(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_bf16_conversion_plan_t* plan) {
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));

  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_type_t source_lane_type =
      loom_amdgpu_low_register_lane_type(module, low_source);
  if (loom_type_kind(source_lane_type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_vgpr_type(context, &source_lane_type));
  }
  loom_type_t result_lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &result_lane_type));

  loom_value_id_t lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->lane_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_16bit_float_lane(
        context, source_op, low_source, plan->source_register_count, i,
        source_lane_type, result_lane_type, &lanes[i]));
  }

  if (plan->lane_count == 1) {
    return loom_low_lower_bind_value(context, plan->result, lanes[0]);
  }
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), lanes, plan->lane_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->result,
                                   loom_low_concat_result(concat_op));
}

static iree_status_t loom_amdgpu_lower_vector_bf16_fptrunc(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_bf16_conversion_plan_t* plan) {
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));

  loom_low_lower_resolved_descriptor_t native_descriptor = {0};
  bool has_native_descriptor = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_BF16_F32, &native_descriptor,
      &has_native_descriptor));
  loom_low_lower_resolved_descriptor_t pack_descriptor = {0};
  bool has_pack_descriptor = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_U16_U32, &pack_descriptor,
      &has_pack_descriptor));

  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_type_t source_lane_type =
      loom_amdgpu_low_register_lane_type(module, low_source);
  if (loom_type_kind(source_lane_type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_vgpr_type(context, &source_lane_type));
  }
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));

  loom_value_id_t packed_registers[LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES];
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    const uint32_t lane_base = register_index * 2u;
    loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_register_unit(
        context, source_op, low_source, plan->source_register_count, lane_base,
        source_lane_type, &source_lane));
    loom_value_id_t packed = LOOM_VALUE_ID_INVALID;

    if (lane_base + 1u < plan->lane_count) {
      loom_value_id_t high_source_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_register_unit(
          context, source_op, low_source, plan->source_register_count,
          lane_base + 1u, source_lane_type, &high_source_lane));
      if (has_native_descriptor) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_native_f32_to_packed_bf16(
            context, source_op, &native_descriptor, source_lane,
            high_source_lane, lane_type, &packed));
        packed_registers[register_index] = packed;
        continue;
      }

      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_f32_to_bf16_lane(
          context, source_op, source_lane, lane_type, &packed));
      loom_value_id_t high_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_f32_to_bf16_lane(
          context, source_op, high_source_lane, lane_type, &high_lane));
      if (has_pack_descriptor) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_bf16_pack_descriptor(
            context, source_op, &pack_descriptor, packed, high_lane, lane_type,
            &packed));
      } else {
        loom_value_id_t high_bits = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
            16, high_lane, lane_type, &high_bits));
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, packed,
            high_bits, lane_type, &packed));
      }
    } else if (has_native_descriptor) {
      loom_value_id_t zero_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
          lane_type, &zero_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_native_f32_to_packed_bf16(
          context, source_op, &native_descriptor, source_lane, zero_lane,
          lane_type, &packed));
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_f32_to_bf16_lane(
          context, source_op, source_lane, lane_type, &packed));
    }
    packed_registers[register_index] = packed;
  }

  if (plan->result_register_count == 1) {
    return loom_low_lower_bind_value(context, plan->result,
                                     packed_registers[0]);
  }
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(loom_low_lower_context_builder(context),
                            packed_registers, plan->result_register_count,
                            result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->result,
                                   loom_low_concat_result(concat_op));
}

static iree_status_t loom_amdgpu_lower_vector_bf16_conversion(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_bf16_conversion_plan_t* plan) {
  switch (plan->kind) {
    case LOOM_AMDGPU_VECTOR_BF16_CONVERSION_KIND_EXTF:
      return loom_amdgpu_lower_vector_bf16_extf(context, source_op, plan);
    case LOOM_AMDGPU_VECTOR_BF16_CONVERSION_KIND_FPTRUNC:
      return loom_amdgpu_lower_vector_bf16_fptrunc(context, source_op, plan);
    default:
      IREE_ASSERT_UNREACHABLE("unknown BF16 conversion plan");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_amdgpu_lower_cast_alias(
    loom_low_lower_context_t* context,
    const loom_amdgpu_cast_alias_plan_t* plan) {
  return loom_low_lower_bind_value_alias(context, plan->source, plan->result);
}

static iree_status_t loom_amdgpu_lower_scalar_trunci(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_scalar_trunci_plan_t* plan) {
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(context, source_op,
                                                  low_source, /*lane_offset=*/0,
                                                  result_type, &low_result));
  return loom_low_lower_bind_value(context, plan->result, low_result);
}

static iree_status_t loom_amdgpu_lower_scalar_extsi(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_scalar_extsi_plan_t* plan) {
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
      context, source_op, plan->source, &low_source));

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_value_id_t high_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT,
      /*shift=*/31, low_source, lane_type, &high_bits));

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_vgpr_range_type(context, 2, &result_type));
  const loom_value_id_t lanes[] = {
      low_source,
      high_bits,
  };
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), lanes, IREE_ARRAYSIZE(lanes),
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->result,
                                   loom_low_concat_result(concat_op));
}

iree_status_t loom_amdgpu_lower_value_op(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_low_lower_plan_t plan) {
  switch (source_op->kind) {
    case LOOM_OP_INDEX_CONSTANT:
    case LOOM_OP_SCALAR_CONSTANT:
    case LOOM_OP_VECTOR_CONSTANT:
      return loom_amdgpu_lower_constant_plan(
          context, source_op,
          (const loom_amdgpu_constant_plan_t*)plan.target_data);
    case LOOM_OP_INDEX_CAST:
      return loom_amdgpu_lower_cast_alias(
          context, (const loom_amdgpu_cast_alias_plan_t*)plan.target_data);
    case LOOM_OP_SCALAR_TRUNCI:
      return loom_amdgpu_lower_scalar_trunci(
          context, source_op,
          (const loom_amdgpu_scalar_trunci_plan_t*)plan.target_data);
    case LOOM_OP_SCALAR_EXTSI:
      return loom_amdgpu_lower_scalar_extsi(
          context, source_op,
          (const loom_amdgpu_scalar_extsi_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_IOTA:
      return loom_amdgpu_lower_vector_iota(
          context, source_op,
          (const loom_amdgpu_vector_iota_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_EXTRACT:
      return loom_amdgpu_lower_vector_extract(
          context, source_op,
          (const loom_amdgpu_vector_extract_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_FROM_ELEMENTS:
    case LOOM_OP_VECTOR_SPLAT:
      return loom_amdgpu_lower_vector_from_elements(
          context, source_op,
          (const loom_amdgpu_vector_from_elements_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_INSERT:
      return loom_amdgpu_lower_vector_insert(
          context, source_op,
          (const loom_amdgpu_vector_insert_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_EXTF:
    case LOOM_OP_VECTOR_FPTRUNC:
      return loom_amdgpu_lower_vector_bf16_conversion(
          context, source_op,
          (const loom_amdgpu_vector_bf16_conversion_plan_t*)plan.target_data);
    default:
      IREE_ASSERT_UNREACHABLE("AMDGPU value plan selected unknown op kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}
