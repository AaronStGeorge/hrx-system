// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/lower_internal.h"

typedef struct loom_amdgpu_bitpack_t {
  // Source vector value containing unpacked i32 lanes.
  loom_value_id_t source;
  // Result vector value containing packed i8 lanes.
  loom_value_id_t result;
  // Number of packed 32-bit registers in the result.
  uint32_t result_register_count;
} loom_amdgpu_bitpack_t;

typedef enum loom_amdgpu_bitpack_rejection_e {
  LOOM_AMDGPU_BITPACK_REJECTION_NONE = 0,
  LOOM_AMDGPU_BITPACK_REJECTION_OP = 1,
  LOOM_AMDGPU_BITPACK_REJECTION_WIDTH = 2,
  LOOM_AMDGPU_BITPACK_REJECTION_SOURCE_TYPE = 3,
  LOOM_AMDGPU_BITPACK_REJECTION_RESULT_TYPE = 4,
  LOOM_AMDGPU_BITPACK_REJECTION_LANE_COUNT = 5,
  LOOM_AMDGPU_BITPACK_REJECTION_LANE_GROUP = 6,
} loom_amdgpu_bitpack_rejection_t;

typedef struct loom_amdgpu_bitunpack_t {
  // Source vector value containing packed integer bitstream storage.
  loom_value_id_t source;
  // Result vector value containing unpacked i32 lanes.
  loom_value_id_t result;
  // Number of bits unpacked into each result lane.
  uint32_t width;
  // Number of packed 32-bit source registers.
  uint32_t source_register_count;
  // Number of unpacked result lanes.
  uint32_t lane_count;
  // True when unpacked lanes are sign-extended.
  bool is_signed;
} loom_amdgpu_bitunpack_t;

typedef enum loom_amdgpu_bitunpack_rejection_e {
  LOOM_AMDGPU_BITUNPACK_REJECTION_NONE = 0,
  LOOM_AMDGPU_BITUNPACK_REJECTION_OP = 1,
  LOOM_AMDGPU_BITUNPACK_REJECTION_WIDTH_RANGE = 2,
  LOOM_AMDGPU_BITUNPACK_REJECTION_WIDTH_DWORD_DIVISOR = 3,
  LOOM_AMDGPU_BITUNPACK_REJECTION_SOURCE_STORAGE = 4,
  LOOM_AMDGPU_BITUNPACK_REJECTION_PAYLOAD_DIVISIBILITY = 5,
  LOOM_AMDGPU_BITUNPACK_REJECTION_LANE_COUNT = 6,
  LOOM_AMDGPU_BITUNPACK_REJECTION_RESULT_TYPE = 7,
} loom_amdgpu_bitunpack_rejection_t;

static bool loom_amdgpu_bitpack_reject(
    loom_amdgpu_bitpack_rejection_t rejection,
    loom_amdgpu_bitpack_rejection_t* out_rejection) {
  if (out_rejection) {
    *out_rejection = rejection;
  }
  return false;
}

static bool loom_amdgpu_bitunpack_reject(
    loom_amdgpu_bitunpack_rejection_t rejection,
    loom_amdgpu_bitunpack_rejection_t* out_rejection) {
  if (out_rejection) {
    *out_rejection = rejection;
  }
  return false;
}

static bool loom_amdgpu_bitpack_select(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_amdgpu_bitpack_t* out_select,
    loom_amdgpu_bitpack_rejection_t* out_rejection) {
  IREE_ASSERT_ARGUMENT(out_select);
  *out_select = (loom_amdgpu_bitpack_t){0};
  if (out_rejection) {
    *out_rejection = LOOM_AMDGPU_BITPACK_REJECTION_NONE;
  }
  if (!loom_vector_bitpack_isa(source_op)) {
    return loom_amdgpu_bitpack_reject(LOOM_AMDGPU_BITPACK_REJECTION_OP,
                                      out_rejection);
  }
  if (loom_vector_bitpack_width(source_op) != 8) {
    return loom_amdgpu_bitpack_reject(LOOM_AMDGPU_BITPACK_REJECTION_WIDTH,
                                      out_rejection);
  }

  const loom_value_id_t source = loom_vector_bitpack_source(source_op);
  const loom_value_id_t result = loom_vector_bitpack_result(source_op);
  const loom_type_t source_type = loom_module_value_type(module, source);
  const loom_type_t result_type = loom_module_value_type(module, result);
  const uint32_t source_lane_count =
      loom_amdgpu_vector_i32_lane_count(source_type);
  const uint32_t result_lane_count =
      loom_amdgpu_vector_i8_lane_count(result_type);
  if (source_lane_count == 0) {
    return loom_amdgpu_bitpack_reject(LOOM_AMDGPU_BITPACK_REJECTION_SOURCE_TYPE,
                                      out_rejection);
  }
  if (result_lane_count == 0) {
    return loom_amdgpu_bitpack_reject(LOOM_AMDGPU_BITPACK_REJECTION_RESULT_TYPE,
                                      out_rejection);
  }
  if (result_lane_count != source_lane_count) {
    return loom_amdgpu_bitpack_reject(LOOM_AMDGPU_BITPACK_REJECTION_LANE_COUNT,
                                      out_rejection);
  }
  if ((result_lane_count % 4) != 0) {
    return loom_amdgpu_bitpack_reject(LOOM_AMDGPU_BITPACK_REJECTION_LANE_GROUP,
                                      out_rejection);
  }

  out_select->source = source;
  out_select->result = result;
  out_select->result_register_count = result_lane_count / 4u;
  return true;
}

static bool loom_amdgpu_bitunpack_select(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_amdgpu_bitunpack_t* out_select,
    loom_amdgpu_bitunpack_rejection_t* out_rejection) {
  IREE_ASSERT_ARGUMENT(out_select);
  *out_select = (loom_amdgpu_bitunpack_t){0};
  if (out_rejection) {
    *out_rejection = LOOM_AMDGPU_BITUNPACK_REJECTION_NONE;
  }

  int64_t width = 0;
  if (loom_vector_bitunpacku_isa(source_op)) {
    width = loom_vector_bitunpacku_width(source_op);
    out_select->source = loom_vector_bitunpacku_source(source_op);
    out_select->result = loom_vector_bitunpacku_result(source_op);
    out_select->is_signed = false;
  } else if (loom_vector_bitunpacks_isa(source_op)) {
    width = loom_vector_bitunpacks_width(source_op);
    out_select->source = loom_vector_bitunpacks_source(source_op);
    out_select->result = loom_vector_bitunpacks_result(source_op);
    out_select->is_signed = true;
  } else {
    return loom_amdgpu_bitunpack_reject(LOOM_AMDGPU_BITUNPACK_REJECTION_OP,
                                        out_rejection);
  }

  if (width < 1 || width > 32) {
    return loom_amdgpu_bitunpack_reject(
        LOOM_AMDGPU_BITUNPACK_REJECTION_WIDTH_RANGE, out_rejection);
  }
  if ((32 % width) != 0) {
    return loom_amdgpu_bitunpack_reject(
        LOOM_AMDGPU_BITUNPACK_REJECTION_WIDTH_DWORD_DIVISOR, out_rejection);
  }

  const loom_type_t source_type =
      loom_module_value_type(module, out_select->source);
  uint32_t source_payload_bit_count = 0;
  uint32_t source_register_count = 0;
  if (!loom_amdgpu_type_packed_integer_storage(
          source_type, &source_payload_bit_count, &source_register_count)) {
    return loom_amdgpu_bitunpack_reject(
        LOOM_AMDGPU_BITUNPACK_REJECTION_SOURCE_STORAGE, out_rejection);
  }
  if ((source_payload_bit_count % (uint32_t)width) != 0) {
    return loom_amdgpu_bitunpack_reject(
        LOOM_AMDGPU_BITUNPACK_REJECTION_PAYLOAD_DIVISIBILITY, out_rejection);
  }

  const uint32_t lane_count = source_payload_bit_count / (uint32_t)width;
  if (lane_count == 0 || lane_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
    return loom_amdgpu_bitunpack_reject(
        LOOM_AMDGPU_BITUNPACK_REJECTION_LANE_COUNT, out_rejection);
  }

  const loom_type_t result_type =
      loom_module_value_type(module, out_select->result);
  if (loom_amdgpu_vector_i32_lane_count(result_type) != lane_count) {
    return loom_amdgpu_bitunpack_reject(
        LOOM_AMDGPU_BITUNPACK_REJECTION_RESULT_TYPE, out_rejection);
  }

  out_select->width = (uint32_t)width;
  out_select->source_register_count = source_register_count;
  out_select->lane_count = lane_count;
  return true;
}

bool loom_amdgpu_can_lower_vector_bitpack(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op) {
  loom_amdgpu_bitpack_t select = {0};
  return loom_amdgpu_bitpack_select(loom_low_lower_context_module(context),
                                    source_op, &select, NULL);
}

bool loom_amdgpu_can_lower_vector_bitunpack(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  loom_amdgpu_bitunpack_t select = {0};
  return loom_amdgpu_bitunpack_select(loom_low_lower_context_module(context),
                                      source_op, &select, NULL);
}

iree_status_t loom_amdgpu_lower_vector_bitpack(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_amdgpu_bitpack_t select = {0};
  if (!loom_amdgpu_bitpack_select(loom_low_lower_context_module(context),
                                  source_op, &select, NULL)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "planning accepted unsupported AMDGPU "
                            "vector.bitpack");
  }

  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, select.source, &low_source));

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_value_id_t mask_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_MOV_B32, UINT32_C(255),
      lane_type, &mask_value));

  loom_value_id_t packed_registers[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
  for (uint32_t register_index = 0;
       register_index < select.result_register_count; ++register_index) {
    loom_value_id_t packed = LOOM_VALUE_ID_INVALID;
    const uint32_t lane_base = register_index * 4u;
    for (uint32_t lane_index = 0; lane_index < 4; ++lane_index) {
      loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_source, lane_base + lane_index, lane_type,
          &source_lane));
      loom_value_id_t low_bits = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_AND_B32, source_lane,
          mask_value, lane_type, &low_bits));
      loom_value_id_t shifted = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHLREV_B32,
          lane_index * 8u, low_bits, lane_type, &shifted));
      if (packed == LOOM_VALUE_ID_INVALID) {
        packed = shifted;
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_OR_B32, packed,
          shifted, lane_type, &packed));
    }
    packed_registers[register_index] = packed;
  }

  if (select.result_register_count == 1) {
    return loom_low_lower_bind_value(context, select.result,
                                     packed_registers[0]);
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, select.result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(loom_low_lower_context_builder(context),
                            packed_registers, select.result_register_count,
                            result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, select.result,
                                   loom_low_concat_result(concat_op));
}

static uint32_t loom_amdgpu_low_mask(uint32_t width) {
  return width == 32 ? UINT32_MAX : (UINT32_C(1) << width) - 1u;
}

static iree_status_t loom_amdgpu_emit_bitunpacku_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitunpack_t* select, loom_value_id_t low_source,
    uint32_t source_bit_offset, loom_value_id_t mask_value,
    loom_type_t lane_type, loom_value_id_t* out_lane) {
  IREE_ASSERT_ARGUMENT(out_lane);
  *out_lane = LOOM_VALUE_ID_INVALID;

  loom_value_id_t shifted = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHRREV_B32,
      source_bit_offset, low_source, lane_type, &shifted));
  if (select->width == 32) {
    *out_lane = shifted;
    return iree_ok_status();
  }

  return loom_amdgpu_emit_vgpr_binary(context, source_op,
                                      LOOM_AMDGPU_DESCRIPTOR_ID_V_AND_B32,
                                      shifted, mask_value, lane_type, out_lane);
}

static iree_status_t loom_amdgpu_emit_bitunpacks_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitunpack_t* select, loom_value_id_t low_source,
    uint32_t source_bit_offset, loom_type_t lane_type,
    loom_value_id_t* out_lane) {
  IREE_ASSERT_ARGUMENT(out_lane);
  *out_lane = LOOM_VALUE_ID_INVALID;
  if (source_bit_offset == 0 && select->width == 32) {
    *out_lane = low_source;
    return iree_ok_status();
  }

  loom_value_id_t shifted_left = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHLREV_B32,
      32u - source_bit_offset - select->width, low_source, lane_type,
      &shifted_left));
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_ASHRREV_I32,
      32u - select->width, shifted_left, lane_type, out_lane);
}

static iree_status_t loom_amdgpu_emit_bitunpack_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitunpack_t* select, loom_value_id_t low_source,
    uint32_t source_bit_offset, loom_value_id_t mask_value,
    loom_type_t lane_type, loom_value_id_t* out_lane) {
  if (select->is_signed) {
    return loom_amdgpu_emit_bitunpacks_lane(context, source_op, select,
                                            low_source, source_bit_offset,
                                            lane_type, out_lane);
  }
  return loom_amdgpu_emit_bitunpacku_lane(context, source_op, select,
                                          low_source, source_bit_offset,
                                          mask_value, lane_type, out_lane);
}

iree_status_t loom_amdgpu_lower_vector_bitunpack(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_amdgpu_bitunpack_t select = {0};
  if (!loom_amdgpu_bitunpack_select(loom_low_lower_context_module(context),
                                    source_op, &select, NULL)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "planning accepted unsupported AMDGPU "
                            "vector.bitunpack");
  }

  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, select.source, &low_source));

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_value_id_t mask_value = LOOM_VALUE_ID_INVALID;
  if (!select.is_signed && select.width < 32) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_MOV_B32,
        loom_amdgpu_low_mask(select.width), lane_type, &mask_value));
  }
  if (select.lane_count == 1) {
    loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_bitunpack_lane(context, source_op, &select, low_source,
                                        0, mask_value, lane_type, &low_result));
    return loom_low_lower_bind_value(context, select.result, low_result);
  }

  loom_value_id_t lane_results[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < select.lane_count; ++i) {
    const uint32_t source_bit_offset = i * select.width;
    const uint32_t source_register_index = source_bit_offset / 32u;
    const uint32_t source_register_bit_offset = source_bit_offset & 31u;
    loom_value_id_t source_register = low_source;
    if (select.source_register_count != 1) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_source, source_register_index, lane_type,
          &source_register));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_bitunpack_lane(
        context, source_op, &select, source_register,
        source_register_bit_offset, mask_value, lane_type, &lane_results[i]));
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, select.result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), lane_results, select.lane_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, select.result,
                                   loom_low_concat_result(concat_op));
}

static iree_string_view_t loom_amdgpu_bitpack_rejection_detail(
    loom_amdgpu_bitpack_rejection_t rejection) {
  switch (rejection) {
    case LOOM_AMDGPU_BITPACK_REJECTION_WIDTH:
      return IREE_SV(
          "AMDGPU vector.bitpack currently lowers byte packing width 8");
    case LOOM_AMDGPU_BITPACK_REJECTION_SOURCE_TYPE:
      return IREE_SV(
          "AMDGPU vector.bitpack requires a static rank-1 i32 "
          "source vector with at most eight lanes");
    case LOOM_AMDGPU_BITPACK_REJECTION_RESULT_TYPE:
      return IREE_SV(
          "AMDGPU vector.bitpack requires a static rank-1 i8 "
          "result vector that fits packed 32-bit registers");
    case LOOM_AMDGPU_BITPACK_REJECTION_LANE_COUNT:
      return IREE_SV(
          "AMDGPU vector.bitpack source and result lane counts "
          "must match");
    case LOOM_AMDGPU_BITPACK_REJECTION_LANE_GROUP:
      return IREE_SV(
          "AMDGPU vector.bitpack requires result lanes grouped "
          "into complete 32-bit packed registers");
    case LOOM_AMDGPU_BITPACK_REJECTION_OP:
    case LOOM_AMDGPU_BITPACK_REJECTION_NONE:
      return IREE_SV("AMDGPU vector.bitpack unsupported bitstream shape");
  }
  return IREE_SV("AMDGPU vector.bitpack unsupported bitstream shape");
}

static iree_string_view_t loom_amdgpu_bitunpack_rejection_detail(
    loom_amdgpu_bitunpack_rejection_t rejection) {
  switch (rejection) {
    case LOOM_AMDGPU_BITUNPACK_REJECTION_WIDTH_RANGE:
      return IREE_SV(
          "AMDGPU vector.bitunpack width must be in the range [1, 32]");
    case LOOM_AMDGPU_BITUNPACK_REJECTION_WIDTH_DWORD_DIVISOR:
      return IREE_SV(
          "AMDGPU vector.bitunpack width must evenly partition a "
          "32-bit backing register");
    case LOOM_AMDGPU_BITUNPACK_REJECTION_SOURCE_STORAGE:
      return IREE_SV(
          "AMDGPU vector.bitunpack requires static integer source "
          "payload storage fitting one to four packed 32-bit "
          "registers");
    case LOOM_AMDGPU_BITUNPACK_REJECTION_PAYLOAD_DIVISIBILITY:
      return IREE_SV(
          "AMDGPU vector.bitunpack source payload bit count must "
          "be divisible by the unpack width");
    case LOOM_AMDGPU_BITUNPACK_REJECTION_LANE_COUNT:
      return IREE_SV(
          "AMDGPU vector.bitunpack requires one to eight unpacked "
          "result lanes");
    case LOOM_AMDGPU_BITUNPACK_REJECTION_RESULT_TYPE:
      return IREE_SV(
          "AMDGPU vector.bitunpack requires a static rank-1 i32 "
          "result vector matching the unpacked lane count");
    case LOOM_AMDGPU_BITUNPACK_REJECTION_OP:
    case LOOM_AMDGPU_BITUNPACK_REJECTION_NONE:
      return IREE_SV("AMDGPU vector.bitunpack unsupported bitstream shape");
  }
  return IREE_SV("AMDGPU vector.bitunpack unsupported bitstream shape");
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
      loom_amdgpu_bitpack_t unused_select = {0};
      loom_amdgpu_bitpack_rejection_t rejection =
          LOOM_AMDGPU_BITPACK_REJECTION_NONE;
      if (loom_amdgpu_bitpack_select(module, op, &unused_select, &rejection)) {
        return iree_ok_status();
      }
      return loom_target_low_legality_reject(
          context, provider, op, IREE_SV("bitstream"), loom_op_name(module, op),
          loom_amdgpu_bitpack_rejection_detail(rejection));
    }
    case LOOM_OP_VECTOR_BITUNPACKS:
    case LOOM_OP_VECTOR_BITUNPACKU: {
      loom_amdgpu_bitunpack_t unused_select = {0};
      loom_amdgpu_bitunpack_rejection_t rejection =
          LOOM_AMDGPU_BITUNPACK_REJECTION_NONE;
      if (loom_amdgpu_bitunpack_select(module, op, &unused_select,
                                       &rejection)) {
        return iree_ok_status();
      }
      return loom_target_low_legality_reject(
          context, provider, op, IREE_SV("bitstream"), loom_op_name(module, op),
          loom_amdgpu_bitunpack_rejection_detail(rejection));
    }
    default:
      *out_handled = false;
      return iree_ok_status();
  }
}
