// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/lower.h"

#include "loom/codegen/low/descriptors.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/x86/avx512_descriptors.h"
#include "loom/target/arch/x86/packed_dot_contract.h"
#include "loom/target/arch/x86/packed_dot_vector.h"

#define LOOM_X86_CONTRACT_SET_AVX512_CORE IREE_SV("x86.avx512.core")
#define LOOM_X86_CONTRACT_SET_PACKED_DOT_CORE IREE_SV("x86.packed_dot.core")

static bool loom_x86_contract_set_key_is_avx512_core(
    iree_string_view_t contract_set_key) {
  return iree_string_view_equal(contract_set_key,
                                LOOM_X86_CONTRACT_SET_AVX512_CORE);
}

static bool loom_x86_contract_set_key_is_packed_dot_core(
    iree_string_view_t contract_set_key) {
  return iree_string_view_equal(contract_set_key,
                                LOOM_X86_CONTRACT_SET_PACKED_DOT_CORE);
}

static iree_string_view_t loom_x86_lower_contract_set_key(
    loom_low_lower_context_t* context) {
  const loom_target_bundle_t* bundle = loom_low_lower_context_bundle(context);
  if (bundle == NULL || bundle->config == NULL) {
    return iree_string_view_empty();
  }
  return bundle->config->contract_set_key;
}

static iree_string_view_t loom_x86_legality_contract_set_key(
    const loom_target_low_legality_context_t* context) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (bundle == NULL || bundle->config == NULL) {
    return iree_string_view_empty();
  }
  return bundle->config->contract_set_key;
}

static bool loom_x86_contract_set_key_is_x86(
    iree_string_view_t contract_set_key) {
  return loom_x86_contract_set_key_is_avx512_core(contract_set_key) ||
         loom_x86_contract_set_key_is_packed_dot_core(contract_set_key);
}

static bool loom_x86_scalar_type_has_packed_dot_register_width(
    loom_scalar_type_t scalar_type, uint32_t* out_bit_width) {
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_I8:
      *out_bit_width = 8;
      return true;
    case LOOM_SCALAR_TYPE_I16:
    case LOOM_SCALAR_TYPE_F16:
    case LOOM_SCALAR_TYPE_BF16:
      *out_bit_width = 16;
      return true;
    case LOOM_SCALAR_TYPE_I32:
    case LOOM_SCALAR_TYPE_F32:
      *out_bit_width = 32;
      return true;
    default:
      *out_bit_width = 0;
      return false;
  }
}

static bool loom_x86_type_static_vector_bit_width(loom_type_t type,
                                                  uint32_t* out_bit_width) {
  *out_bit_width = 0;
  if (!loom_type_is_vector(type) || loom_type_rank(type) != 1 ||
      !loom_type_is_all_static(type)) {
    return false;
  }
  uint32_t element_bit_width = 0;
  if (!loom_x86_scalar_type_has_packed_dot_register_width(
          loom_type_element_type(type), &element_bit_width)) {
    return false;
  }
  const int64_t lane_count = loom_type_dim_static_size_at(type, 0);
  if (lane_count <= 0 ||
      (uint64_t)lane_count > UINT32_MAX / element_bit_width) {
    return false;
  }
  *out_bit_width = (uint32_t)lane_count * element_bit_width;
  return true;
}

static bool loom_x86_type_is_vector_16xi32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32 &&
         loom_type_dim_static_size_at(type, 0) == 16;
}

static bool loom_x86_type_is_vector_16xf32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32 &&
         loom_type_dim_static_size_at(type, 0) == 16;
}

static bool loom_x86_value_is_vector_16xi32(loom_low_lower_context_t* context,
                                            loom_value_id_t value_id) {
  return loom_x86_type_is_vector_16xi32(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

static bool loom_x86_value_is_vector_16xf32(loom_low_lower_context_t* context,
                                            loom_value_id_t value_id) {
  return loom_x86_type_is_vector_16xf32(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

static iree_status_t loom_x86_make_register_type(
    loom_low_lower_context_t* context, iree_string_view_t register_class,
    loom_type_t* out_type) {
  loom_string_id_t register_class_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(loom_low_lower_context_module(context),
                                register_class, &register_class_id));
  *out_type = loom_type_register(register_class_id, 1);
  return iree_ok_status();
}

static iree_status_t loom_x86_make_vector_register_type(
    loom_low_lower_context_t* context, uint32_t vector_bit_width,
    loom_type_t* out_type) {
  switch (vector_bit_width) {
    case 128:
      return loom_x86_make_register_type(context, IREE_SV("x86.xmm"), out_type);
    case 256:
      return loom_x86_make_register_type(context, IREE_SV("x86.ymm"), out_type);
    case 512:
      return loom_x86_make_register_type(context, IREE_SV("x86.zmm"), out_type);
    default:
      *out_type = loom_type_none();
      return iree_ok_status();
  }
}

static iree_status_t loom_x86_map_type(void* user_data,
                                       loom_low_lower_context_t* context,
                                       const loom_op_t* source_op,
                                       loom_type_t source_type,
                                       loom_type_t* out_low_type) {
  (void)user_data;
  const iree_string_view_t contract_set_key =
      loom_x86_lower_contract_set_key(context);
  if (loom_x86_contract_set_key_is_avx512_core(contract_set_key)) {
    if (loom_x86_type_is_vector_16xi32(source_type) ||
        loom_x86_type_is_vector_16xf32(source_type)) {
      return loom_x86_make_vector_register_type(context, 512, out_low_type);
    }
    return loom_low_lower_emit_reject(
        context, source_op, IREE_SV("type"), IREE_SV("source"),
        IREE_SV("x86 AVX512 lowering currently supports only vector<16xi32> "
                "and vector<16xf32> values"));
  }
  if (loom_x86_contract_set_key_is_packed_dot_core(contract_set_key)) {
    uint32_t vector_bit_width = 0;
    if (loom_x86_type_static_vector_bit_width(source_type, &vector_bit_width)) {
      IREE_RETURN_IF_ERROR(loom_x86_make_vector_register_type(
          context, vector_bit_width, out_low_type));
      if (loom_type_kind(*out_low_type) != LOOM_TYPE_NONE) {
        return iree_ok_status();
      }
    }
    return loom_low_lower_emit_reject(
        context, source_op, IREE_SV("type"), IREE_SV("source"),
        IREE_SV("x86 packed-dot lowering requires a static 128-, 256-, or "
                "512-bit i8/i16/f16/bf16/i32/f32 vector"));
  }
  return loom_low_lower_emit_reject(
      context, source_op, IREE_SV("target"),
      iree_string_view_is_empty(contract_set_key) ? IREE_SV("<missing>")
                                                  : contract_set_key,
      IREE_SV("x86 source-to-low lowering has no policy for this contract "
              "set"));
}

static bool loom_x86_can_lower_vector_addi(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op) {
  return loom_x86_value_is_vector_16xi32(context,
                                         loom_vector_addi_lhs(source_op)) &&
         loom_x86_value_is_vector_16xi32(context,
                                         loom_vector_addi_rhs(source_op)) &&
         loom_x86_value_is_vector_16xi32(context,
                                         loom_vector_addi_result(source_op));
}

static bool loom_x86_can_lower_vector_subi(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op) {
  return loom_x86_value_is_vector_16xi32(context,
                                         loom_vector_subi_lhs(source_op)) &&
         loom_x86_value_is_vector_16xi32(context,
                                         loom_vector_subi_rhs(source_op)) &&
         loom_x86_value_is_vector_16xi32(context,
                                         loom_vector_subi_result(source_op));
}

static bool loom_x86_can_lower_vector_muli(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op) {
  return loom_x86_value_is_vector_16xi32(context,
                                         loom_vector_muli_lhs(source_op)) &&
         loom_x86_value_is_vector_16xi32(context,
                                         loom_vector_muli_rhs(source_op)) &&
         loom_x86_value_is_vector_16xi32(context,
                                         loom_vector_muli_result(source_op));
}

static bool loom_x86_can_lower_vector_addf(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op) {
  return loom_x86_value_is_vector_16xf32(context,
                                         loom_vector_addf_lhs(source_op)) &&
         loom_x86_value_is_vector_16xf32(context,
                                         loom_vector_addf_rhs(source_op)) &&
         loom_x86_value_is_vector_16xf32(context,
                                         loom_vector_addf_result(source_op));
}

static bool loom_x86_can_lower_vector_subf(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op) {
  return loom_x86_value_is_vector_16xf32(context,
                                         loom_vector_subf_lhs(source_op)) &&
         loom_x86_value_is_vector_16xf32(context,
                                         loom_vector_subf_rhs(source_op)) &&
         loom_x86_value_is_vector_16xf32(context,
                                         loom_vector_subf_result(source_op));
}

static bool loom_x86_can_lower_vector_mulf(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op) {
  return loom_x86_value_is_vector_16xf32(context,
                                         loom_vector_mulf_lhs(source_op)) &&
         loom_x86_value_is_vector_16xf32(context,
                                         loom_vector_mulf_rhs(source_op)) &&
         loom_x86_value_is_vector_16xf32(context,
                                         loom_vector_mulf_result(source_op));
}

static bool loom_x86_can_lower_vector_fmaf(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op) {
  return loom_x86_value_is_vector_16xf32(context,
                                         loom_vector_fmaf_a(source_op)) &&
         loom_x86_value_is_vector_16xf32(context,
                                         loom_vector_fmaf_b(source_op)) &&
         loom_x86_value_is_vector_16xf32(context,
                                         loom_vector_fmaf_c(source_op)) &&
         loom_x86_value_is_vector_16xf32(context,
                                         loom_vector_fmaf_result(source_op));
}

static iree_string_view_t loom_x86_packed_dot_rejection_name(
    loom_x86_packed_dot_rejection_bits_t rejection_bits) {
  if (iree_any_bit_set(rejection_bits,
                       LOOM_X86_PACKED_DOT_REJECTION_FEATURES)) {
    return IREE_SV("features");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_PAYLOAD)) {
    return IREE_SV("payload");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_SHAPE)) {
    return IREE_SV("shape");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_FAMILY)) {
    return IREE_SV("family");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_FLAGS)) {
    return IREE_SV("flags");
  }
  return IREE_SV("invalid-request");
}

static iree_string_view_t loom_x86_packed_dot_rejection_detail(
    loom_x86_packed_dot_rejection_bits_t rejection_bits) {
  if (iree_any_bit_set(rejection_bits,
                       LOOM_X86_PACKED_DOT_REJECTION_FEATURES)) {
    return IREE_SV(
        "target profile does not enable a matching x86 packed-dot feature");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_PAYLOAD)) {
    return IREE_SV(
        "no x86 packed-dot descriptor matches the vector dot payload types");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_SHAPE)) {
    return IREE_SV("no x86 packed-dot descriptor matches the vector dot shape");
  }
  return IREE_SV("no x86 packed-dot descriptor matches this vector dot op");
}

static iree_status_t loom_x86_select_packed_dot_descriptor(
    const loom_module_t* module, const loom_target_bundle_t* bundle,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* op,
    loom_x86_packed_dot_match_diagnostic_t* out_diagnostic,
    const loom_x86_packed_dot_descriptor_t** out_descriptor) {
  *out_descriptor = NULL;
  loom_x86_packed_dot_match_request_t request = {0};
  if (!loom_x86_packed_dot_match_request_from_vector_op(module, op, &request)) {
    if (out_diagnostic != NULL) {
      *out_diagnostic = (loom_x86_packed_dot_match_diagnostic_t){
          .rejection_bits = LOOM_X86_PACKED_DOT_REJECTION_INVALID_REQUEST,
      };
    }
    return iree_ok_status();
  }
  if (bundle == NULL || bundle->config == NULL || descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "x86 packed-dot selection requires a target bundle "
                            "and descriptor set");
  }
  request.feature_bits =
      (loom_x86_packed_dot_feature_bits_t)bundle->config->contract_feature_bits;

  loom_x86_packed_dot_match_diagnostic_t diagnostic = {0};
  const loom_x86_packed_dot_descriptor_t* descriptor =
      loom_x86_packed_dot_select(&request, &diagnostic);
  if (out_diagnostic != NULL) {
    *out_diagnostic = diagnostic;
  }
  if (descriptor == NULL) {
    return iree_ok_status();
  }

  uint32_t descriptor_ordinal = loom_low_descriptor_set_lookup_descriptor_by_id(
      descriptor_set, descriptor->stable_id);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "x86 packed-dot descriptor '%.*s' is missing",
                            (int)descriptor->name.size, descriptor->name.data);
  }
  *out_descriptor = descriptor;
  return iree_ok_status();
}

static bool loom_x86_op_is_vector_dot(loom_op_kind_t kind) {
  switch (kind) {
    case LOOM_OP_VECTOR_DOT2F:
    case LOOM_OP_VECTOR_DOT4F8:
    case LOOM_OP_VECTOR_DOT4I:
    case LOOM_OP_VECTOR_DOT8I4:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_x86_can_lower_vector_dot(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_handled) {
  *out_handled = false;
  if (!loom_x86_contract_set_key_is_packed_dot_core(
          loom_x86_lower_contract_set_key(context))) {
    return iree_ok_status();
  }
  const loom_x86_packed_dot_descriptor_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_x86_select_packed_dot_descriptor(
      loom_low_lower_context_module(context),
      loom_low_lower_context_bundle(context),
      loom_low_lower_context_descriptor_set(context), source_op,
      /*out_diagnostic=*/NULL, &descriptor));
  *out_handled = descriptor != NULL;
  return iree_ok_status();
}

static iree_status_t loom_x86_can_lower_op(void* user_data,
                                           loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           bool* out_handled) {
  (void)user_data;
  const iree_string_view_t contract_set_key =
      loom_x86_lower_contract_set_key(context);
  switch (source_op->kind) {
    case LOOM_OP_VECTOR_ADDI:
      *out_handled =
          loom_x86_contract_set_key_is_avx512_core(contract_set_key) &&
          loom_x86_can_lower_vector_addi(context, source_op);
      return iree_ok_status();
    case LOOM_OP_VECTOR_SUBI:
      *out_handled =
          loom_x86_contract_set_key_is_avx512_core(contract_set_key) &&
          loom_x86_can_lower_vector_subi(context, source_op);
      return iree_ok_status();
    case LOOM_OP_VECTOR_MULI:
      *out_handled =
          loom_x86_contract_set_key_is_avx512_core(contract_set_key) &&
          loom_x86_can_lower_vector_muli(context, source_op);
      return iree_ok_status();
    case LOOM_OP_VECTOR_ADDF:
      *out_handled =
          loom_x86_contract_set_key_is_avx512_core(contract_set_key) &&
          loom_x86_can_lower_vector_addf(context, source_op);
      return iree_ok_status();
    case LOOM_OP_VECTOR_SUBF:
      *out_handled =
          loom_x86_contract_set_key_is_avx512_core(contract_set_key) &&
          loom_x86_can_lower_vector_subf(context, source_op);
      return iree_ok_status();
    case LOOM_OP_VECTOR_MULF:
      *out_handled =
          loom_x86_contract_set_key_is_avx512_core(contract_set_key) &&
          loom_x86_can_lower_vector_mulf(context, source_op);
      return iree_ok_status();
    case LOOM_OP_VECTOR_FMAF:
      *out_handled =
          loom_x86_contract_set_key_is_avx512_core(contract_set_key) &&
          loom_x86_can_lower_vector_fmaf(context, source_op);
      return iree_ok_status();
    case LOOM_OP_VECTOR_DOT2F:
    case LOOM_OP_VECTOR_DOT4I:
      return loom_x86_can_lower_vector_dot(context, source_op, out_handled);
    case LOOM_OP_VECTOR_DOT4F8:
    case LOOM_OP_VECTOR_DOT8I4:
      *out_handled = false;
      return iree_ok_status();
    default:
      *out_handled = false;
      return iree_ok_status();
  }
}

static iree_status_t loom_x86_low_result_type(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              loom_value_id_t source_result,
                                              loom_type_t* out_low_type) {
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(context, source_op,
                                                source_result, out_low_type));
  if (!loom_type_is_register(*out_low_type)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "x86 source type did not map to a register");
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_lower_binary_op(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              uint64_t descriptor_id,
                                              loom_value_id_t source_lhs,
                                              loom_value_id_t source_rhs,
                                              loom_value_id_t source_result) {
  loom_value_id_t low_operands[2] = {LOOM_VALUE_ID_INVALID,
                                     LOOM_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_lhs, &low_operands[0]));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_rhs, &low_operands[1]));

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_x86_low_result_type(context, source_op,
                                                source_result, &result_type));

  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_descriptor_op(
      context, descriptor_id, low_operands, IREE_ARRAYSIZE(low_operands),
      loom_make_named_attr_slice(NULL, 0), &result_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  return loom_low_lower_bind_value(
      context, source_result,
      loom_value_slice_get(loom_low_op_results(low_op), 0));
}

static iree_status_t loom_x86_lower_vector_addi(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_x86_lower_binary_op(
      context, source_op, X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VPADDD_ZMM,
      loom_vector_addi_lhs(source_op), loom_vector_addi_rhs(source_op),
      loom_vector_addi_result(source_op));
}

static iree_status_t loom_x86_lower_vector_subi(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_x86_lower_binary_op(
      context, source_op, X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VPSUBD_ZMM,
      loom_vector_subi_lhs(source_op), loom_vector_subi_rhs(source_op),
      loom_vector_subi_result(source_op));
}

static iree_status_t loom_x86_lower_vector_muli(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_x86_lower_binary_op(
      context, source_op, X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VPMULLD_ZMM,
      loom_vector_muli_lhs(source_op), loom_vector_muli_rhs(source_op),
      loom_vector_muli_result(source_op));
}

static iree_status_t loom_x86_lower_vector_addf(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_x86_lower_binary_op(
      context, source_op, X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VADDPS_ZMM,
      loom_vector_addf_lhs(source_op), loom_vector_addf_rhs(source_op),
      loom_vector_addf_result(source_op));
}

static iree_status_t loom_x86_lower_vector_subf(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_x86_lower_binary_op(
      context, source_op, X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VSUBPS_ZMM,
      loom_vector_subf_lhs(source_op), loom_vector_subf_rhs(source_op),
      loom_vector_subf_result(source_op));
}

static iree_status_t loom_x86_lower_vector_mulf(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_x86_lower_binary_op(
      context, source_op, X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VMULPS_ZMM,
      loom_vector_mulf_lhs(source_op), loom_vector_mulf_rhs(source_op),
      loom_vector_mulf_result(source_op));
}

static iree_status_t loom_x86_lower_tied_ternary_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t descriptor_id, loom_value_id_t source_lhs,
    loom_value_id_t source_rhs, loom_value_id_t source_accumulator,
    loom_value_id_t source_result) {
  loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_lhs, &low_lhs));
  loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_rhs, &low_rhs));
  loom_value_id_t low_accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(context, source_accumulator,
                                                   &low_accumulator));

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_x86_low_result_type(context, source_op,
                                                source_result, &result_type));

  loom_op_t* accumulator_copy_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_copy_build(
      loom_low_lower_context_builder(context), low_accumulator, result_type,
      source_op->location, &accumulator_copy_op));
  const loom_value_id_t copied_accumulator =
      loom_low_copy_result(accumulator_copy_op);

  loom_value_id_t low_operands[3] = {copied_accumulator, low_lhs, low_rhs};
  loom_tied_result_t tied_result = {
      .result_index = 0,
      .operand_index = 0,
      .has_type_change = false,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_descriptor_op(
      context, descriptor_id, low_operands, IREE_ARRAYSIZE(low_operands),
      loom_make_named_attr_slice(NULL, 0), &result_type, 1, &tied_result, 1,
      source_op->location, &low_op));
  return loom_low_lower_bind_value(
      context, source_result,
      loom_value_slice_get(loom_low_op_results(low_op), 0));
}

static iree_status_t loom_x86_lower_vector_fmaf(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_x86_lower_tied_ternary_op(
      context, source_op,
      X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VFMADD231PS_ZMM,
      loom_vector_fmaf_a(source_op), loom_vector_fmaf_b(source_op),
      loom_vector_fmaf_c(source_op), loom_vector_fmaf_result(source_op));
}

static iree_status_t loom_x86_lower_packed_dot_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_lhs, loom_value_id_t source_rhs,
    loom_value_id_t source_accumulator, loom_value_id_t source_result) {
  const loom_x86_packed_dot_descriptor_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_x86_select_packed_dot_descriptor(
      loom_low_lower_context_module(context),
      loom_low_lower_context_bundle(context),
      loom_low_lower_context_descriptor_set(context), source_op,
      /*out_diagnostic=*/NULL, &descriptor));
  if (descriptor == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "x86 packed-dot lowering reached an unselected descriptor");
  }
  return loom_x86_lower_tied_ternary_op(
      context, source_op, descriptor->stable_id, source_lhs, source_rhs,
      source_accumulator, source_result);
}

static iree_status_t loom_x86_lower_vector_dot2f(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_x86_lower_packed_dot_op(
      context, source_op, loom_vector_dot2f_lhs(source_op),
      loom_vector_dot2f_rhs(source_op), loom_vector_dot2f_acc(source_op),
      loom_vector_dot2f_result(source_op));
}

static iree_status_t loom_x86_lower_vector_dot4i(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_x86_lower_packed_dot_op(
      context, source_op, loom_vector_dot4i_lhs(source_op),
      loom_vector_dot4i_rhs(source_op), loom_vector_dot4i_acc(source_op),
      loom_vector_dot4i_result(source_op));
}

static iree_status_t loom_x86_try_lower_op(void* user_data,
                                           loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           bool* out_handled) {
  IREE_RETURN_IF_ERROR(
      loom_x86_can_lower_op(user_data, context, source_op, out_handled));
  if (!*out_handled) {
    return iree_ok_status();
  }

  switch (source_op->kind) {
    case LOOM_OP_VECTOR_ADDI:
      return loom_x86_lower_vector_addi(context, source_op);
    case LOOM_OP_VECTOR_SUBI:
      return loom_x86_lower_vector_subi(context, source_op);
    case LOOM_OP_VECTOR_MULI:
      return loom_x86_lower_vector_muli(context, source_op);
    case LOOM_OP_VECTOR_ADDF:
      return loom_x86_lower_vector_addf(context, source_op);
    case LOOM_OP_VECTOR_SUBF:
      return loom_x86_lower_vector_subf(context, source_op);
    case LOOM_OP_VECTOR_MULF:
      return loom_x86_lower_vector_mulf(context, source_op);
    case LOOM_OP_VECTOR_FMAF:
      return loom_x86_lower_vector_fmaf(context, source_op);
    case LOOM_OP_VECTOR_DOT2F:
      return loom_x86_lower_vector_dot2f(context, source_op);
    case LOOM_OP_VECTOR_DOT4I:
      return loom_x86_lower_vector_dot4i(context, source_op);
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_x86_low_legality_verify_packed_dot(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const iree_string_view_t contract_set_key =
      loom_x86_legality_contract_set_key(context);
  if (!loom_x86_contract_set_key_is_x86(contract_set_key)) {
    return iree_ok_status();
  }
  *out_handled = true;

  if (!loom_x86_contract_set_key_is_packed_dot_core(contract_set_key)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("contract"), contract_set_key,
        IREE_SV("x86 vector dot ops require the x86.packed_dot.core "
                "target-low contract set"));
  }

  loom_x86_packed_dot_match_diagnostic_t diagnostic = {0};
  const loom_x86_packed_dot_descriptor_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_x86_select_packed_dot_descriptor(
      loom_target_low_legality_module(context),
      loom_target_low_legality_bundle(context),
      loom_target_low_legality_descriptor_set(context), op, &diagnostic,
      &descriptor));
  if (descriptor == NULL) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("contract"),
        loom_x86_packed_dot_rejection_name(diagnostic.rejection_bits),
        loom_x86_packed_dot_rejection_detail(diagnostic.rejection_bits));
  }

  return loom_target_low_legality_record_contract(
      context, provider, op, descriptor->name, IREE_SV("selected"),
      IREE_SV("selected x86 packed-dot descriptor"));
}

static iree_status_t loom_x86_low_legality_try_verify_op(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  *out_handled = false;
  if (loom_x86_op_is_vector_dot(op->kind)) {
    return loom_x86_low_legality_verify_packed_dot(provider, context, op,
                                                   out_handled);
  }
  return iree_ok_status();
}

static const loom_low_lower_policy_t kX86LowLowerPolicy = {
    .name = IREE_SVL("x86-low-lower"),
    .map_type = {.fn = loom_x86_map_type, .user_data = NULL},
    .can_lower_op = {.fn = loom_x86_can_lower_op, .user_data = NULL},
    .try_lower_op = {.fn = loom_x86_try_lower_op, .user_data = NULL},
};

const loom_target_low_legality_provider_t
    loom_x86_low_legality_provider_storage = {
        .name = IREE_SVL("x86"),
        .try_verify_op = loom_x86_low_legality_try_verify_op,
};

const loom_low_lower_policy_t* loom_x86_low_lower_policy(void) {
  return &kX86LowLowerPolicy;
}

const loom_target_low_legality_provider_t* loom_x86_low_legality_provider(
    void) {
  return &loom_x86_low_legality_provider_storage;
}

void loom_x86_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("x86.avx512.core"),
          .policy = &kX86LowLowerPolicy,
      },
      {
          .contract_set_key = IREE_SVL("x86.packed_dot.core"),
          .policy = &kX86LowLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
