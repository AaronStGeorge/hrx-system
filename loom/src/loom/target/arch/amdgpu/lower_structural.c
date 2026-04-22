// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/lower_internal.h"

static bool loom_amdgpu_vector_bitcast_storage_shape(
    loom_type_t type, uint32_t* out_payload_bit_count,
    uint32_t* out_register_count) {
  IREE_ASSERT_ARGUMENT(out_payload_bit_count);
  IREE_ASSERT_ARGUMENT(out_register_count);
  *out_payload_bit_count = 0;
  *out_register_count = 0;

  const uint32_t lane_count = loom_amdgpu_vector_32bit_lane_count(type);
  if (lane_count != 0) {
    *out_payload_bit_count = 32u * lane_count;
    *out_register_count = lane_count;
    return true;
  }

  return loom_amdgpu_type_packed_integer_storage(type, out_payload_bit_count,
                                                 out_register_count);
}

bool loom_amdgpu_can_lower_vector_bitcast(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op) {
  if (!loom_vector_bitcast_isa(source_op)) {
    return false;
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t input_type =
      loom_module_value_type(module, loom_vector_bitcast_input(source_op));
  const loom_type_t result_type =
      loom_module_value_type(module, loom_vector_bitcast_result(source_op));

  uint32_t input_payload_bit_count = 0;
  uint32_t input_register_count = 0;
  uint32_t result_payload_bit_count = 0;
  uint32_t result_register_count = 0;
  return loom_amdgpu_vector_bitcast_storage_shape(
             input_type, &input_payload_bit_count, &input_register_count) &&
         loom_amdgpu_vector_bitcast_storage_shape(
             result_type, &result_payload_bit_count, &result_register_count) &&
         input_payload_bit_count == result_payload_bit_count &&
         input_register_count == result_register_count;
}

iree_status_t loom_amdgpu_lower_vector_bitcast(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  if (!loom_amdgpu_can_lower_vector_bitcast(context, source_op)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU "
                            "vector.bitcast");
  }

  loom_value_id_t low_input = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_bitcast_input(source_op), &low_input));

  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, loom_vector_bitcast_result(source_op),
      &result_low_type));
  const loom_type_t input_low_type =
      loom_module_value_type(loom_low_lower_context_module(context), low_input);
  if (!loom_type_equal(input_low_type, result_low_type)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU vector.bitcast source and result map to different low types");
  }

  return loom_low_lower_bind_value(
      context, loom_vector_bitcast_result(source_op), low_input);
}
