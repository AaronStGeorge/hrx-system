// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lowering for x86-native Loom vector contracts into structured LLVMIR ops.

#include "loom/target/emit/llvmir/x86/lower.h"

#include "loom/ops/vector/ops.h"
#include "loom/target/arch/x86/packed_dot_vector.h"
#include "loom/target/emit/llvmir/lower_internal.h"

static iree_status_t loom_llvmir_lowering_x86_packed_dot_scalar_type(
    loom_llvmir_lowering_state_t* state,
    loom_x86_packed_dot_numeric_type_t numeric_type,
    loom_llvmir_type_id_t* out_type) {
  switch (numeric_type) {
    case LOOM_X86_PACKED_DOT_NUMERIC_I8:
    case LOOM_X86_PACKED_DOT_NUMERIC_U8:
      return loom_llvmir_module_get_integer_type(state->target_module, 8,
                                                 out_type);
    case LOOM_X86_PACKED_DOT_NUMERIC_I16:
    case LOOM_X86_PACKED_DOT_NUMERIC_U16:
      return loom_llvmir_module_get_integer_type(state->target_module, 16,
                                                 out_type);
    case LOOM_X86_PACKED_DOT_NUMERIC_I32:
      return loom_llvmir_module_get_integer_type(state->target_module, 32,
                                                 out_type);
    case LOOM_X86_PACKED_DOT_NUMERIC_F16:
      return loom_llvmir_module_get_float_type(state->target_module,
                                               LOOM_LLVMIR_FLOAT_F16, out_type);
    case LOOM_X86_PACKED_DOT_NUMERIC_BF16:
      return loom_llvmir_module_get_float_type(
          state->target_module, LOOM_LLVMIR_FLOAT_BF16, out_type);
    case LOOM_X86_PACKED_DOT_NUMERIC_F32:
      return loom_llvmir_module_get_float_type(state->target_module,
                                               LOOM_LLVMIR_FLOAT_F32, out_type);
    case LOOM_X86_PACKED_DOT_NUMERIC_UNKNOWN:
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown x86 packed-dot numeric type");
  }
}

static iree_status_t loom_llvmir_lowering_x86_packed_dot_vector_type(
    loom_llvmir_lowering_state_t* state,
    loom_x86_packed_dot_numeric_type_t numeric_type, uint16_t lane_count,
    loom_llvmir_type_id_t* out_type) {
  loom_llvmir_type_id_t element_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_x86_packed_dot_scalar_type(
      state, numeric_type, &element_type));
  return loom_llvmir_module_get_vector_type(state->target_module, lane_count,
                                            element_type, out_type);
}

static iree_status_t
loom_llvmir_lowering_x86_packed_dot_intrinsic_source_vector_type(
    loom_llvmir_lowering_state_t* state,
    const loom_x86_packed_dot_descriptor_t* descriptor,
    loom_x86_packed_dot_numeric_type_t logical_numeric_type,
    loom_llvmir_type_id_t* out_type) {
  switch (descriptor->llvm_source_abi) {
    case LOOM_X86_PACKED_DOT_LLVM_SOURCE_ABI_PAYLOAD:
      return loom_llvmir_lowering_x86_packed_dot_vector_type(
          state, logical_numeric_type, descriptor->shape.input_lane_count,
          out_type);
    case LOOM_X86_PACKED_DOT_LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR:
      return loom_llvmir_lowering_x86_packed_dot_vector_type(
          state, descriptor->accumulator_numeric_type,
          descriptor->shape.result_lane_count, out_type);
    case LOOM_X86_PACKED_DOT_LLVM_SOURCE_ABI_UNKNOWN:
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown x86 packed-dot LLVM source ABI");
  }
}

static iree_status_t loom_llvmir_lowering_adapt_x86_packed_dot_source_operand(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_x86_packed_dot_descriptor_t* descriptor,
    loom_llvmir_value_id_t source_value,
    loom_x86_packed_dot_numeric_type_t logical_numeric_type,
    loom_llvmir_value_id_t* out_value) {
  if (descriptor->llvm_source_abi ==
      LOOM_X86_PACKED_DOT_LLVM_SOURCE_ABI_PAYLOAD) {
    *out_value = source_value;
    return iree_ok_status();
  }
  loom_llvmir_type_id_t abi_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_x86_packed_dot_intrinsic_source_vector_type(
          state, descriptor, logical_numeric_type, &abi_type));
  return loom_llvmir_build_cast(target_block,
                                &(loom_llvmir_cast_desc_t){
                                    .result_name = iree_string_view_empty(),
                                    .result_type = abi_type,
                                    .op = LOOM_LLVMIR_CAST_BITCAST,
                                    .value = source_value,
                                },
                                out_value);
}

static iree_status_t loom_llvmir_lowering_declare_x86_packed_dot_intrinsic(
    loom_llvmir_lowering_state_t* state,
    const loom_x86_packed_dot_descriptor_t* descriptor,
    loom_llvmir_function_t** out_function) {
  for (iree_host_size_t i = 0; i < state->provider_intrinsic_function_count;
       ++i) {
    if (state->provider_intrinsic_keys[i] == descriptor) {
      *out_function = state->provider_intrinsic_functions[i];
      return iree_ok_status();
    }
  }
  if (state->provider_intrinsic_function_count >=
      IREE_ARRAYSIZE(state->provider_intrinsic_functions)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "LLVM target-provider intrinsic cache capacity exceeded");
  }

  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t accumulator_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t lhs_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t rhs_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_x86_packed_dot_vector_type(
      state, descriptor->result_numeric_type,
      descriptor->shape.result_lane_count, &result_type));
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_x86_packed_dot_vector_type(
      state, descriptor->accumulator_numeric_type,
      descriptor->shape.result_lane_count, &accumulator_type));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_x86_packed_dot_intrinsic_source_vector_type(
          state, descriptor, descriptor->lhs_numeric_type, &lhs_type));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_x86_packed_dot_intrinsic_source_vector_type(
          state, descriptor, descriptor->rhs_numeric_type, &rhs_type));

  loom_llvmir_function_t* function = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_function(
      state->target_module,
      &(loom_llvmir_function_desc_t){
          .kind = LOOM_LLVMIR_FUNCTION_DECLARATION,
          .name = descriptor->llvm_intrinsic_name,
          .return_type = result_type,
          .linkage = LOOM_LLVMIR_LINKAGE_DEFAULT,
          .calling_convention = LOOM_LLVMIR_CALLING_CONVENTION_DEFAULT,
          .attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID,
      },
      &function));
  loom_llvmir_type_id_t parameter_types[3] = {
      accumulator_type,
      lhs_type,
      rhs_type,
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(parameter_types); ++i) {
    loom_llvmir_value_id_t parameter = LOOM_LLVMIR_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_llvmir_function_add_parameter(function,
                                           &(loom_llvmir_parameter_desc_t){
                                               .type_id = parameter_types[i],
                                               .name = iree_string_view_empty(),
                                           },
                                           &parameter));
  }

  iree_host_size_t cache_ordinal = state->provider_intrinsic_function_count++;
  state->provider_intrinsic_keys[cache_ordinal] = descriptor;
  state->provider_intrinsic_functions[cache_ordinal] = function;
  *out_function = function;
  return iree_ok_status();
}

static const char* loom_llvmir_lowering_x86_packed_dot_rejection_detail(
    loom_x86_packed_dot_rejection_bits_t rejection_bits) {
  if (iree_any_bit_set(rejection_bits,
                       LOOM_X86_PACKED_DOT_REJECTION_FEATURES)) {
    return "target profile does not enable a matching x86 packed-dot feature";
  }
  if (iree_any_bit_set(rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_PAYLOAD)) {
    return "no x86 packed-dot descriptor matches the vector dot payload types";
  }
  if (iree_any_bit_set(rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_SHAPE)) {
    return "no x86 packed-dot descriptor matches the vector dot shape";
  }
  return "no x86 packed-dot descriptor matches this vector dot op";
}

static iree_status_t loom_llvmir_lowering_lower_vector_x86_packed_dot(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op, loom_x86_packed_dot_match_request_t request) {
  request.feature_bits = (loom_x86_packed_dot_feature_bits_t)
                             state->target_profile->x86_packed_dot_feature_bits;

  loom_x86_packed_dot_match_diagnostic_t diagnostic = {0};
  const loom_x86_packed_dot_descriptor_t* descriptor =
      loom_x86_packed_dot_select(&request, &diagnostic);
  if (descriptor == NULL) {
    return loom_llvmir_lowering_unsupported_op(
        state, op,
        loom_llvmir_lowering_x86_packed_dot_rejection_detail(
            diagnostic.rejection_bits));
  }

  const loom_value_id_t* operands = loom_op_const_operands(op);
  if (op->operand_count != 3 || op->result_count != 1) {
    return loom_llvmir_lowering_unsupported_op(
        state, op,
        "x86 packed-dot vector ops require three operands and one "
        "result");
  }
  loom_llvmir_value_id_t args[3] = {
      LOOM_LLVMIR_VALUE_ID_INVALID,
      LOOM_LLVMIR_VALUE_ID_INVALID,
      LOOM_LLVMIR_VALUE_ID_INVALID,
  };
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lookup_value(state, operands[2], &args[0]));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lookup_value(state, operands[0], &args[1]));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lookup_value(state, operands[1], &args[2]));
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_adapt_x86_packed_dot_source_operand(
      state, target_block, descriptor, args[1], descriptor->lhs_numeric_type,
      &args[1]));
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_adapt_x86_packed_dot_source_operand(
      state, target_block, descriptor, args[2], descriptor->rhs_numeric_type,
      &args[2]));

  loom_llvmir_function_t* intrinsic = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_declare_x86_packed_dot_intrinsic(
      state, descriptor, &intrinsic));
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_call(target_block,
                             &(loom_llvmir_call_desc_t){
                                 .result_name = loom_llvmir_lowering_value_name(
                                     state, loom_op_const_results(op)[0]),
                                 .callee = loom_llvmir_function_id(intrinsic),
                                 .args = args,
                                 .arg_count = IREE_ARRAYSIZE(args),
                             },
                             &result));
  return loom_llvmir_lowering_map_single_result(state, op, result);
}

static iree_status_t loom_llvmir_x86_try_lower_op(
    const loom_llvmir_lowering_provider_t* provider,
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op, bool* out_handled) {
  (void)provider;
  *out_handled = false;
  switch (op->kind) {
    case LOOM_OP_VECTOR_DOT2F:
    case LOOM_OP_VECTOR_DOT4I: {
      loom_x86_packed_dot_match_request_t request = {0};
      if (!loom_x86_packed_dot_match_request_from_vector_op(
              state->source_module, op, &request)) {
        return iree_ok_status();
      }
      *out_handled = true;
      return loom_llvmir_lowering_lower_vector_x86_packed_dot(
          state, target_block, op, request);
    }
    default:
      return iree_ok_status();
  }
}

static const loom_llvmir_lowering_provider_t kX86LoweringProvider = {
    .name = IREE_SVL("x86"),
    .try_lower_op = loom_llvmir_x86_try_lower_op,
};

const loom_llvmir_lowering_provider_t* loom_llvmir_x86_lowering_provider(void) {
  return &kX86LoweringProvider;
}
