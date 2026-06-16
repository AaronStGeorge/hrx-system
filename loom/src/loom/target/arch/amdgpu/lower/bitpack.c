// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/bitpack.h"

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

typedef enum loom_amdgpu_bitstream_recipe_kind_e {
  LOOM_AMDGPU_BITSTREAM_RECIPE_PACK_I8_FROM_I32 = 0,
  LOOM_AMDGPU_BITSTREAM_RECIPE_UNPACKU = 1,
  LOOM_AMDGPU_BITSTREAM_RECIPE_UNPACKS = 2,
} loom_amdgpu_bitstream_recipe_kind_t;

typedef enum loom_amdgpu_bitstream_rejection_e {
  LOOM_AMDGPU_BITSTREAM_REJECTION_NONE = 0,
  LOOM_AMDGPU_BITSTREAM_REJECTION_WIDTH = 1,
  LOOM_AMDGPU_BITSTREAM_REJECTION_WIDTH_RANGE = 2,
  LOOM_AMDGPU_BITSTREAM_REJECTION_WIDTH_DWORD_DIVISOR = 3,
  LOOM_AMDGPU_BITSTREAM_REJECTION_SOURCE_TYPE = 4,
  LOOM_AMDGPU_BITSTREAM_REJECTION_RESULT_TYPE = 5,
  LOOM_AMDGPU_BITSTREAM_REJECTION_LANE_COUNT = 6,
  LOOM_AMDGPU_BITSTREAM_REJECTION_LANE_GROUP = 7,
  LOOM_AMDGPU_BITSTREAM_REJECTION_SOURCE_STORAGE = 8,
  LOOM_AMDGPU_BITSTREAM_REJECTION_PAYLOAD_DIVISIBILITY = 9,
} loom_amdgpu_bitstream_rejection_t;

typedef struct loom_amdgpu_bitstream_diagnostic_t {
  // Stable lowering recipe family selected from the source op kind.
  loom_amdgpu_bitstream_recipe_kind_t recipe_kind;
  // First user-IR constraint that rejected the recipe.
  loom_amdgpu_bitstream_rejection_t rejection;
  // Source op bit width attribute, or zero before an op-specific width exists.
  int64_t width;
  // Type of the source operand seen by the recipe matcher.
  loom_type_t source_type;
  // Type of the result value seen by the recipe matcher.
  loom_type_t result_type;
  // Logical source lane count proven by the matcher.
  uint32_t source_lane_count;
  // Logical result lane count proven by the matcher.
  uint32_t result_lane_count;
  // Payload bit count represented by the source storage shape.
  uint32_t source_payload_bit_count;
  // Payload bit count represented by the result storage shape.
  uint32_t result_payload_bit_count;
  // Number of 32-bit source registers represented by the storage shape.
  uint32_t source_register_count;
  // Number of 32-bit result registers represented by the storage shape.
  uint32_t result_register_count;
} loom_amdgpu_bitstream_diagnostic_t;

static void loom_amdgpu_bitstream_diagnostic_initialize(
    loom_amdgpu_bitstream_diagnostic_t* out_diagnostic,
    loom_amdgpu_bitstream_recipe_kind_t recipe_kind, int64_t width) {
  if (out_diagnostic == NULL) {
    return;
  }
  *out_diagnostic = (loom_amdgpu_bitstream_diagnostic_t){
      .recipe_kind = recipe_kind,
      .rejection = LOOM_AMDGPU_BITSTREAM_REJECTION_NONE,
      .width = width,
      .source_type = loom_type_none(),
      .result_type = loom_type_none(),
  };
}

static bool loom_amdgpu_bitstream_reject(
    loom_amdgpu_bitstream_diagnostic_t* diagnostic,
    loom_amdgpu_bitstream_rejection_t rejection) {
  if (diagnostic != NULL) {
    diagnostic->rejection = rejection;
  }
  return false;
}

static bool loom_amdgpu_bitpack_plan_from_op(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_amdgpu_bitpack_plan_t* out_plan,
    loom_amdgpu_bitstream_diagnostic_t* out_diagnostic) {
  *out_plan = (loom_amdgpu_bitpack_plan_t){0};
  IREE_ASSERT(loom_vector_bitpack_isa(source_op));
  const int64_t width = loom_vector_bitpack_width(source_op);
  loom_amdgpu_bitstream_diagnostic_initialize(
      out_diagnostic, LOOM_AMDGPU_BITSTREAM_RECIPE_PACK_I8_FROM_I32, width);
  const loom_value_id_t source = loom_vector_bitpack_source(source_op);
  const loom_value_id_t result = loom_vector_bitpack_result(source_op);
  const loom_type_t source_type = loom_module_value_type(module, source);
  const loom_type_t result_type = loom_module_value_type(module, result);
  const uint32_t source_lane_count =
      loom_amdgpu_vector_i32_lane_count(source_type);
  const uint32_t result_lane_count =
      loom_amdgpu_vector_i8_lane_count(result_type);
  if (out_diagnostic != NULL) {
    out_diagnostic->source_type = source_type;
    out_diagnostic->result_type = result_type;
    out_diagnostic->source_lane_count = source_lane_count;
    out_diagnostic->result_lane_count = result_lane_count;
    out_diagnostic->source_payload_bit_count = source_lane_count * 32u;
    out_diagnostic->source_register_count = source_lane_count;
    (void)loom_amdgpu_type_packed_integer_storage(
        result_type, &out_diagnostic->result_payload_bit_count,
        &out_diagnostic->result_register_count);
  }
  if (width != 8) {
    return loom_amdgpu_bitstream_reject(out_diagnostic,
                                        LOOM_AMDGPU_BITSTREAM_REJECTION_WIDTH);
  }
  if (source_lane_count == 0) {
    return loom_amdgpu_bitstream_reject(
        out_diagnostic, LOOM_AMDGPU_BITSTREAM_REJECTION_SOURCE_TYPE);
  }
  if (result_lane_count == 0) {
    return loom_amdgpu_bitstream_reject(
        out_diagnostic, LOOM_AMDGPU_BITSTREAM_REJECTION_RESULT_TYPE);
  }
  if (result_lane_count != source_lane_count) {
    return loom_amdgpu_bitstream_reject(
        out_diagnostic, LOOM_AMDGPU_BITSTREAM_REJECTION_LANE_COUNT);
  }
  if ((result_lane_count % 4) != 0) {
    return loom_amdgpu_bitstream_reject(
        out_diagnostic, LOOM_AMDGPU_BITSTREAM_REJECTION_LANE_GROUP);
  }

  out_plan->source = source;
  out_plan->result = result;
  out_plan->result_register_count = result_lane_count / 4u;
  return true;
}

static bool loom_amdgpu_bitunpack_plan_from_op(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_amdgpu_bitunpack_plan_t* out_plan,
    loom_amdgpu_bitstream_diagnostic_t* out_diagnostic) {
  *out_plan = (loom_amdgpu_bitunpack_plan_t){0};

  int64_t width = 0;
  if (loom_vector_bitunpacku_isa(source_op)) {
    width = loom_vector_bitunpacku_width(source_op);
    loom_amdgpu_bitstream_diagnostic_initialize(
        out_diagnostic, LOOM_AMDGPU_BITSTREAM_RECIPE_UNPACKU, width);
    out_plan->source = loom_vector_bitunpacku_source(source_op);
    out_plan->result = loom_vector_bitunpacku_result(source_op);
    out_plan->is_signed = false;
  } else if (loom_vector_bitunpacks_isa(source_op)) {
    width = loom_vector_bitunpacks_width(source_op);
    loom_amdgpu_bitstream_diagnostic_initialize(
        out_diagnostic, LOOM_AMDGPU_BITSTREAM_RECIPE_UNPACKS, width);
    out_plan->source = loom_vector_bitunpacks_source(source_op);
    out_plan->result = loom_vector_bitunpacks_result(source_op);
    out_plan->is_signed = true;
  } else {
    IREE_ASSERT_UNREACHABLE(
        "bitstream plan selector called for unsupported source op");
    IREE_BUILTIN_UNREACHABLE();
  }

  const loom_type_t source_type =
      loom_module_value_type(module, out_plan->source);
  const loom_type_t result_type =
      loom_module_value_type(module, out_plan->result);
  if (out_diagnostic != NULL) {
    out_diagnostic->source_type = source_type;
    out_diagnostic->result_type = result_type;
    (void)loom_amdgpu_type_packed_integer_storage(
        source_type, &out_diagnostic->source_payload_bit_count,
        &out_diagnostic->source_register_count);
    const uint32_t i32_lane_count =
        loom_amdgpu_vector_i32_lane_count(result_type);
    if (i32_lane_count != 0) {
      out_diagnostic->result_lane_count = i32_lane_count;
      out_diagnostic->result_payload_bit_count = i32_lane_count * 32u;
      out_diagnostic->result_register_count = i32_lane_count;
    } else {
      out_diagnostic->result_lane_count =
          loom_amdgpu_vector_i8_lane_count(result_type);
      (void)loom_amdgpu_type_packed_integer_storage(
          result_type, &out_diagnostic->result_payload_bit_count,
          &out_diagnostic->result_register_count);
    }
  }
  if (width < 1 || width > 32) {
    return loom_amdgpu_bitstream_reject(
        out_diagnostic, LOOM_AMDGPU_BITSTREAM_REJECTION_WIDTH_RANGE);
  }
  if ((32 % width) != 0) {
    return loom_amdgpu_bitstream_reject(
        out_diagnostic, LOOM_AMDGPU_BITSTREAM_REJECTION_WIDTH_DWORD_DIVISOR);
  }

  uint32_t source_payload_bit_count = 0;
  uint32_t source_register_count = 0;
  if (!loom_amdgpu_type_packed_integer_storage(
          source_type, &source_payload_bit_count, &source_register_count)) {
    if (out_diagnostic != NULL) {
      out_diagnostic->source_type = source_type;
    }
    return loom_amdgpu_bitstream_reject(
        out_diagnostic, LOOM_AMDGPU_BITSTREAM_REJECTION_SOURCE_STORAGE);
  }
  if (out_diagnostic != NULL) {
    out_diagnostic->source_type = source_type;
    out_diagnostic->source_payload_bit_count = source_payload_bit_count;
    out_diagnostic->source_register_count = source_register_count;
  }
  if ((source_payload_bit_count % (uint32_t)width) != 0) {
    return loom_amdgpu_bitstream_reject(
        out_diagnostic, LOOM_AMDGPU_BITSTREAM_REJECTION_PAYLOAD_DIVISIBILITY);
  }

  const uint32_t lane_count = source_payload_bit_count / (uint32_t)width;
  if (out_diagnostic != NULL) {
    out_diagnostic->source_lane_count = lane_count;
  }
  if (lane_count == 0 || lane_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
    return loom_amdgpu_bitstream_reject(
        out_diagnostic, LOOM_AMDGPU_BITSTREAM_REJECTION_LANE_COUNT);
  }

  const uint32_t i32_lane_count =
      loom_amdgpu_vector_i32_lane_count(result_type);
  if (out_diagnostic != NULL) {
    out_diagnostic->result_type = result_type;
    out_diagnostic->result_lane_count = i32_lane_count;
    out_diagnostic->result_payload_bit_count = i32_lane_count * 32u;
    out_diagnostic->result_register_count = i32_lane_count;
  }
  if (i32_lane_count == lane_count) {
    out_plan->result_kind = LOOM_AMDGPU_BITUNPACK_RESULT_KIND_I32_LANES;
    out_plan->result_register_count = lane_count;
  } else {
    uint32_t result_payload_bit_count = 0;
    uint32_t result_register_count = 0;
    const uint32_t i8_lane_count =
        loom_amdgpu_vector_i8_lane_count(result_type);
    const bool result_has_packed_storage =
        loom_amdgpu_type_packed_integer_storage(
            result_type, &result_payload_bit_count, &result_register_count);
    if (out_diagnostic != NULL) {
      out_diagnostic->result_lane_count = i8_lane_count;
      out_diagnostic->result_payload_bit_count = result_payload_bit_count;
      out_diagnostic->result_register_count = result_register_count;
    }
    if (width > 8 || i8_lane_count != lane_count ||
        !result_has_packed_storage ||
        result_payload_bit_count != lane_count * 8u) {
      return loom_amdgpu_bitstream_reject(
          out_diagnostic, LOOM_AMDGPU_BITSTREAM_REJECTION_RESULT_TYPE);
    }
    out_plan->result_kind = LOOM_AMDGPU_BITUNPACK_RESULT_KIND_PACKED_I8;
    out_plan->result_register_count = result_register_count;
  }
  if (out_plan->result_kind == LOOM_AMDGPU_BITUNPACK_RESULT_KIND_NONE) {
    return loom_amdgpu_bitstream_reject(
        out_diagnostic, LOOM_AMDGPU_BITSTREAM_REJECTION_RESULT_TYPE);
  }

  out_plan->width = (uint32_t)width;
  out_plan->source_register_count = source_register_count;
  out_plan->lane_count = lane_count;
  return true;
}

iree_status_t loom_amdgpu_select_vector_bitpack_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_bitpack_plan_t* out_plan, bool* out_selected) {
  *out_selected = loom_amdgpu_bitpack_plan_from_op(
      loom_low_lower_context_module(context), source_op, out_plan, NULL);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_vector_bitunpack_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_bitunpack_plan_t* out_plan, bool* out_selected) {
  *out_selected = loom_amdgpu_bitunpack_plan_from_op(
      loom_low_lower_context_module(context), source_op, out_plan, NULL);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_pack_i8_bits_into_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_bits, uint32_t register_lane, loom_type_t lane_type,
    loom_value_id_t* inout_packed) {
  if (*inout_packed == LOOM_VALUE_ID_INVALID) {
    loom_value_id_t shifted = low_bits;
    if (register_lane != 0) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
          register_lane * 8u, low_bits, lane_type, &shifted));
    }
    *inout_packed = shifted;
    return iree_ok_status();
  }

  if (register_lane != 0) {
    loom_value_id_t packed = LOOM_VALUE_ID_INVALID;
    bool selected_lshl_add = false;
    // Packed bytes occupy disjoint bit ranges, so add and OR are equivalent
    // after shifting the incoming low byte into its target lane.
    IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_vgpr_lshl_add_u32(
        context, source_op, low_bits, *inout_packed, register_lane * 8u,
        lane_type, &packed, &selected_lshl_add));
    if (selected_lshl_add) {
      *inout_packed = packed;
      return iree_ok_status();
    }
  }

  loom_value_id_t shifted = low_bits;
  if (register_lane != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
        register_lane * 8u, low_bits, lane_type, &shifted));
  }
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, *inout_packed,
      shifted, lane_type, inout_packed);
}

static iree_status_t loom_amdgpu_pack_i8_lane_into_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_lane, uint32_t register_lane, loom_type_t lane_type,
    loom_value_id_t* inout_packed) {
  loom_value_id_t low_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, source_lane,
      UINT32_C(255), lane_type, &low_bits));
  return loom_amdgpu_pack_i8_bits_into_register(
      context, source_op, low_bits, register_lane, lane_type, inout_packed);
}

iree_status_t loom_amdgpu_lower_vector_bitpack(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitpack_plan_t* plan) {
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));

  loom_value_id_t packed_registers[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    loom_value_id_t packed = LOOM_VALUE_ID_INVALID;
    const uint32_t lane_base = register_index * 4u;
    for (uint32_t lane_index = 0; lane_index < 4; ++lane_index) {
      loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_source, lane_base + lane_index, lane_type,
          &source_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_pack_i8_lane_into_register(
          context, source_op, source_lane, lane_index, lane_type, &packed));
    }
    packed_registers[register_index] = packed;
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             packed_registers,
                                             plan->result_register_count);
}

static uint32_t loom_amdgpu_low_mask(uint32_t width) {
  return width == 32 ? UINT32_MAX : (UINT32_C(1) << width) - 1u;
}

static bool loom_amdgpu_signed_bitunpack_lane_prefers_bfe(
    const loom_amdgpu_bitunpack_plan_t* plan, uint32_t source_bit_offset) {
  // The high byte of a 32-bit word already sign-extends with one ASHR.
  return plan->width == 8 && source_bit_offset < 24;
}

static bool loom_amdgpu_unsigned_bitunpack_lane_prefers_bfe(
    uint32_t source_bit_offset) {
  // Offset zero already extracts with a single mask packet.
  return source_bit_offset != 0;
}

static iree_status_t loom_amdgpu_emit_bitunpacku_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitunpack_plan_t* plan, loom_value_id_t low_source,
    uint32_t source_bit_offset, loom_type_t lane_type,
    loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  if (source_bit_offset == 0 && plan->width == 32) {
    *out_lane = low_source;
    return iree_ok_status();
  }

  bool selected_sdwa = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_vgpr_b32_sdwa_extract(
      context, source_op, low_source, source_bit_offset, plan->width,
      LOOM_AMDGPU_VGPR_SDWA_EXTRACT_FLAG_NONE, lane_type, out_lane,
      &selected_sdwa));
  if (selected_sdwa) {
    return iree_ok_status();
  }

  if (loom_amdgpu_unsigned_bitunpack_lane_prefers_bfe(source_bit_offset)) {
    bool selected_bfe = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_vgpr_b32_bfe_extract(
        context, source_op, low_source, source_bit_offset, plan->width,
        LOOM_AMDGPU_VGPR_BFE_EXTRACT_FLAG_NONE, lane_type, out_lane,
        &selected_bfe));
    if (selected_bfe) {
      return iree_ok_status();
    }
  }

  loom_value_id_t shifted = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      source_bit_offset, low_source, lane_type, &shifted));

  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, shifted,
      loom_amdgpu_low_mask(plan->width), lane_type, out_lane);
}

static iree_status_t loom_amdgpu_emit_bitunpacks_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitunpack_plan_t* plan, loom_value_id_t low_source,
    uint32_t source_bit_offset, loom_type_t lane_type,
    loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  if (source_bit_offset == 0 && plan->width == 32) {
    *out_lane = low_source;
    return iree_ok_status();
  }

  bool selected_sdwa = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_vgpr_b32_sdwa_extract(
      context, source_op, low_source, source_bit_offset, plan->width,
      LOOM_AMDGPU_VGPR_SDWA_EXTRACT_FLAG_SIGN_EXTEND, lane_type, out_lane,
      &selected_sdwa));
  if (selected_sdwa) {
    return iree_ok_status();
  }

  if (loom_amdgpu_signed_bitunpack_lane_prefers_bfe(plan, source_bit_offset)) {
    bool selected_bfe = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_vgpr_b32_bfe_extract(
        context, source_op, low_source, source_bit_offset, plan->width,
        LOOM_AMDGPU_VGPR_BFE_EXTRACT_FLAG_SIGN_EXTEND, lane_type, out_lane,
        &selected_bfe));
    if (selected_bfe) {
      return iree_ok_status();
    }
  }

  loom_value_id_t shifted_left = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      32u - source_bit_offset - plan->width, low_source, lane_type,
      &shifted_left));
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT,
      32u - plan->width, shifted_left, lane_type, out_lane);
}

static iree_status_t loom_amdgpu_emit_bitunpack_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitunpack_plan_t* plan, loom_value_id_t low_source,
    uint32_t source_bit_offset, loom_type_t lane_type,
    loom_value_id_t* out_lane) {
  if (plan->is_signed) {
    return loom_amdgpu_emit_bitunpacks_lane(context, source_op, plan,
                                            low_source, source_bit_offset,
                                            lane_type, out_lane);
  }
  return loom_amdgpu_emit_bitunpacku_lane(context, source_op, plan, low_source,
                                          source_bit_offset, lane_type,
                                          out_lane);
}

static iree_status_t loom_amdgpu_source_bitunpack_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitunpack_plan_t* plan, loom_value_id_t low_source,
    uint32_t source_bit_offset, loom_type_t lane_type,
    loom_value_id_t* out_source_register,
    uint32_t* out_source_register_bit_offset) {
  const uint32_t source_register_index = source_bit_offset / 32u;
  *out_source_register_bit_offset = source_bit_offset & 31u;
  *out_source_register = low_source;
  if (plan->source_register_count == 1) {
    return iree_ok_status();
  }
  return loom_amdgpu_emit_low_slice(context, source_op, low_source,
                                    source_register_index, lane_type,
                                    out_source_register);
}

static iree_status_t loom_amdgpu_lower_vector_bitunpack_i32_lanes(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitunpack_plan_t* plan, loom_value_id_t low_source,
    loom_type_t lane_type) {
  loom_value_id_t lane_results[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->lane_count; ++i) {
    const uint32_t source_bit_offset = i * plan->width;
    loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
    uint32_t source_register_bit_offset = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_source_bitunpack_register(
        context, source_op, plan, low_source, source_bit_offset, lane_type,
        &source_register, &source_register_bit_offset));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_bitunpack_lane(
        context, source_op, plan, source_register, source_register_bit_offset,
        lane_type, &lane_results[i]));
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             lane_results, plan->lane_count);
}

static iree_status_t loom_amdgpu_lower_vector_bitunpack_packed_i8(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitunpack_plan_t* plan, loom_value_id_t low_source,
    loom_type_t lane_type) {
  loom_value_id_t result_registers[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    loom_value_id_t packed = LOOM_VALUE_ID_INVALID;
    const uint32_t lane_base = register_index * 4u;
    for (uint32_t register_lane = 0; register_lane < 4; ++register_lane) {
      const uint32_t lane_ordinal = lane_base + register_lane;
      if (lane_ordinal >= plan->lane_count) {
        break;
      }
      const uint32_t source_bit_offset = lane_ordinal * plan->width;
      loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
      uint32_t source_register_bit_offset = 0;
      IREE_RETURN_IF_ERROR(loom_amdgpu_source_bitunpack_register(
          context, source_op, plan, low_source, source_bit_offset, lane_type,
          &source_register, &source_register_bit_offset));
      loom_value_id_t lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_bitunpack_lane(
          context, source_op, plan, source_register, source_register_bit_offset,
          lane_type, &lane));
      if (plan->is_signed) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_pack_i8_lane_into_register(
            context, source_op, lane, register_lane, lane_type, &packed));
      } else {
        IREE_RETURN_IF_ERROR(loom_amdgpu_pack_i8_bits_into_register(
            context, source_op, lane, register_lane, lane_type, &packed));
      }
    }
    result_registers[register_index] = packed;
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             result_registers,
                                             plan->result_register_count);
}

iree_status_t loom_amdgpu_lower_vector_bitunpack(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitunpack_plan_t* plan) {
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  switch (plan->result_kind) {
    case LOOM_AMDGPU_BITUNPACK_RESULT_KIND_I32_LANES:
      return loom_amdgpu_lower_vector_bitunpack_i32_lanes(
          context, source_op, plan, low_source, lane_type);
    case LOOM_AMDGPU_BITUNPACK_RESULT_KIND_PACKED_I8:
      return loom_amdgpu_lower_vector_bitunpack_packed_i8(
          context, source_op, plan, low_source, lane_type);
    case LOOM_AMDGPU_BITUNPACK_RESULT_KIND_NONE:
    default:
      IREE_ASSERT_UNREACHABLE("unsupported bitunpack result kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_string_view_t loom_amdgpu_bitstream_recipe_key(
    loom_amdgpu_bitstream_recipe_kind_t recipe_kind) {
  switch (recipe_kind) {
    case LOOM_AMDGPU_BITSTREAM_RECIPE_PACK_I8_FROM_I32:
      return IREE_SV("amdgpu.bitstream.pack_i8_from_i32");
    case LOOM_AMDGPU_BITSTREAM_RECIPE_UNPACKU:
      return IREE_SV("amdgpu.bitstream.unpacku");
    case LOOM_AMDGPU_BITSTREAM_RECIPE_UNPACKS:
      return IREE_SV("amdgpu.bitstream.unpacks");
    default:
      IREE_ASSERT_UNREACHABLE("unsupported bitstream recipe kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_string_view_t loom_amdgpu_bitstream_constraint_key(
    loom_amdgpu_bitstream_rejection_t rejection) {
  switch (rejection) {
    case LOOM_AMDGPU_BITSTREAM_REJECTION_WIDTH:
      return IREE_SV("width");
    case LOOM_AMDGPU_BITSTREAM_REJECTION_WIDTH_RANGE:
      return IREE_SV("width_range");
    case LOOM_AMDGPU_BITSTREAM_REJECTION_WIDTH_DWORD_DIVISOR:
      return IREE_SV("width_dword_divisor");
    case LOOM_AMDGPU_BITSTREAM_REJECTION_SOURCE_TYPE:
      return IREE_SV("source_type");
    case LOOM_AMDGPU_BITSTREAM_REJECTION_RESULT_TYPE:
      return IREE_SV("result_type");
    case LOOM_AMDGPU_BITSTREAM_REJECTION_LANE_COUNT:
      return IREE_SV("lane_count");
    case LOOM_AMDGPU_BITSTREAM_REJECTION_LANE_GROUP:
      return IREE_SV("lane_group");
    case LOOM_AMDGPU_BITSTREAM_REJECTION_SOURCE_STORAGE:
      return IREE_SV("source_storage");
    case LOOM_AMDGPU_BITSTREAM_REJECTION_PAYLOAD_DIVISIBILITY:
      return IREE_SV("payload_divisibility");
    case LOOM_AMDGPU_BITSTREAM_REJECTION_NONE:
    default:
      return IREE_SV("shape");
  }
}

static iree_status_t loom_amdgpu_emit_bitstream_rejection_diagnostic(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_amdgpu_bitstream_diagnostic_t* diagnostic) {
  IREE_ASSERT_NE(diagnostic->rejection, LOOM_AMDGPU_BITSTREAM_REJECTION_NONE);
  loom_diagnostic_param_t
      params[LOOM_AMDGPU_LOW_LEGALITY_CONTEXT_PARAM_COUNT + 11];
  loom_amdgpu_low_legality_make_context_params(context, op, params);
  params[LOOM_AMDGPU_LOW_LEGALITY_CONTEXT_PARAM_COUNT + 0] = loom_param_string(
      loom_amdgpu_bitstream_recipe_key(diagnostic->recipe_kind));
  params[LOOM_AMDGPU_LOW_LEGALITY_CONTEXT_PARAM_COUNT + 1] = loom_param_string(
      loom_amdgpu_bitstream_constraint_key(diagnostic->rejection));
  params[LOOM_AMDGPU_LOW_LEGALITY_CONTEXT_PARAM_COUNT + 2] =
      loom_param_i64(diagnostic->width);
  params[LOOM_AMDGPU_LOW_LEGALITY_CONTEXT_PARAM_COUNT + 3] =
      loom_param_type(diagnostic->source_type);
  params[LOOM_AMDGPU_LOW_LEGALITY_CONTEXT_PARAM_COUNT + 4] =
      loom_param_type(diagnostic->result_type);
  params[LOOM_AMDGPU_LOW_LEGALITY_CONTEXT_PARAM_COUNT + 5] =
      loom_param_u32(diagnostic->source_lane_count);
  params[LOOM_AMDGPU_LOW_LEGALITY_CONTEXT_PARAM_COUNT + 6] =
      loom_param_u32(diagnostic->result_lane_count);
  params[LOOM_AMDGPU_LOW_LEGALITY_CONTEXT_PARAM_COUNT + 7] =
      loom_param_u32(diagnostic->source_payload_bit_count);
  params[LOOM_AMDGPU_LOW_LEGALITY_CONTEXT_PARAM_COUNT + 8] =
      loom_param_u32(diagnostic->result_payload_bit_count);
  params[LOOM_AMDGPU_LOW_LEGALITY_CONTEXT_PARAM_COUNT + 9] =
      loom_param_u32(diagnostic->source_register_count);
  params[LOOM_AMDGPU_LOW_LEGALITY_CONTEXT_PARAM_COUNT + 10] =
      loom_param_u32(diagnostic->result_register_count);
  return loom_target_low_legality_emit_error_ref(
      context, op, LOOM_ERR_AMDGPU_040_REF, params, IREE_ARRAYSIZE(params));
}

iree_status_t loom_amdgpu_low_legality_verify_vector_bitstream(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);
  switch (op->kind) {
    case LOOM_OP_VECTOR_BITPACK: {
      loom_amdgpu_bitpack_plan_t unused_plan = {0};
      loom_amdgpu_bitstream_diagnostic_t diagnostic = {0};
      if (loom_amdgpu_bitpack_plan_from_op(module, op, &unused_plan,
                                           &diagnostic)) {
        return iree_ok_status();
      }
      return loom_amdgpu_emit_bitstream_rejection_diagnostic(context, op,
                                                             &diagnostic);
    }
    case LOOM_OP_VECTOR_BITUNPACKS:
    case LOOM_OP_VECTOR_BITUNPACKU: {
      loom_amdgpu_bitunpack_plan_t unused_plan = {0};
      loom_amdgpu_bitstream_diagnostic_t diagnostic = {0};
      if (loom_amdgpu_bitunpack_plan_from_op(module, op, &unused_plan,
                                             &diagnostic)) {
        return iree_ok_status();
      }
      return loom_amdgpu_emit_bitstream_rejection_diagnostic(context, op,
                                                             &diagnostic);
    }
    default:
      *out_handled = false;
      return iree_ok_status();
  }
}
