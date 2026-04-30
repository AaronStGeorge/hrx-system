// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/ir/module.h"
#include "loom/target/arch/x86/avx512_descriptors.h"
#include "loom/target/arch/x86/avx512_packed_dot_descriptors.h"
#include "loom/target/arch/x86/lower/internal.h"

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

static bool loom_x86_type_is_vector_16xi1(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1 &&
         loom_type_dim_static_size_at(type, 0) == 16;
}

static bool loom_x86_type_is_vector_4xi32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32 &&
         loom_type_dim_static_size_at(type, 0) == 4;
}

static bool loom_x86_type_is_vector_4xf32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32 &&
         loom_type_dim_static_size_at(type, 0) == 4;
}

static bool loom_x86_type_is_vector_4xi1(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1 &&
         loom_type_dim_static_size_at(type, 0) == 4;
}

static bool loom_x86_type_is_scalar_i32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32;
}

static bool loom_x86_type_is_scalar_f32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32;
}

static bool loom_x86_type_is_address_gpr64(loom_type_t type) {
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  switch (loom_type_element_type(type)) {
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_OFFSET:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_x86_make_gpr64_register_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, X86_AVX512_CORE_REG_CLASS_ID_X86_GPR64, 1, out_type);
}

static iree_status_t loom_x86_make_gpr32_register_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, X86_AVX512_CORE_REG_CLASS_ID_X86_GPR32, 1, out_type);
}

iree_status_t loom_x86_make_xmm_register_type(loom_low_lower_context_t* context,
                                              loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, X86_AVX512_CORE_REG_CLASS_ID_X86_XMM, 1, out_type);
}

iree_status_t loom_x86_make_zmm_register_type(loom_low_lower_context_t* context,
                                              loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, X86_AVX512_CORE_REG_CLASS_ID_X86_ZMM, 1, out_type);
}

static iree_status_t loom_x86_make_avx512_packed_dot_xmm_register_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, X86_AVX512_PACKED_DOT_CORE_REG_CLASS_ID_X86_XMM, 1, out_type);
}

static iree_status_t loom_x86_make_avx512_packed_dot_ymm_register_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, X86_AVX512_PACKED_DOT_CORE_REG_CLASS_ID_X86_YMM, 1, out_type);
}

static iree_status_t loom_x86_make_avx512_packed_dot_zmm_register_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, X86_AVX512_PACKED_DOT_CORE_REG_CLASS_ID_X86_ZMM, 1, out_type);
}

static iree_status_t loom_x86_make_k_register_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, X86_AVX512_CORE_REG_CLASS_ID_X86_K, 1, out_type);
}

static iree_status_t loom_x86_map_avx512_type(void* user_data,
                                              loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              loom_type_t source_type,
                                              loom_type_t* out_low_type) {
  (void)user_data;
  if (loom_x86_type_is_address_gpr64(source_type)) {
    return loom_x86_make_gpr64_register_type(context, out_low_type);
  }
  if (loom_x86_type_is_scalar_i32(source_type)) {
    return loom_x86_make_gpr32_register_type(context, out_low_type);
  }
  if (loom_x86_type_is_scalar_f32(source_type)) {
    return loom_x86_make_xmm_register_type(context, out_low_type);
  }
  if (loom_x86_type_is_vector_4xi32(source_type) ||
      loom_x86_type_is_vector_4xf32(source_type)) {
    return loom_x86_make_xmm_register_type(context, out_low_type);
  }
  if (loom_x86_type_is_vector_4xi1(source_type)) {
    return loom_x86_make_k_register_type(context, out_low_type);
  }
  if (loom_x86_type_is_vector_16xi32(source_type) ||
      loom_x86_type_is_vector_16xf32(source_type)) {
    return loom_x86_make_zmm_register_type(context, out_low_type);
  }
  if (loom_x86_type_is_vector_16xi1(source_type)) {
    return loom_x86_make_k_register_type(context, out_low_type);
  }
  return loom_low_lower_emit_reject(
      context, source_op, IREE_SV("type"), IREE_SV("source"),
      IREE_SV("x86 AVX512 lowering currently supports only index/offset "
              "address values, i32/f32 scalar splat values, and "
              "vector<4xi1>/vector<4xi32>/vector<4xf32>/vector<16xi1>/"
              "vector<16xi32>/vector<16xf32> values"));
}

static iree_status_t loom_x86_map_avx512_argument(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_function_op, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument) {
  (void)source_argument_index;
  IREE_ASSERT_ARGUMENT(out_argument);
  const loom_type_t source_type = loom_module_value_type(
      loom_low_lower_context_module(context), source_argument_id);
  if (loom_type_is_buffer(source_type)) {
    loom_type_t address_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_x86_make_gpr64_register_type(context, &address_type));
    *out_argument = (loom_low_lower_abi_argument_t){
        .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
        .abi_type = address_type,
        .resource_semantic_type = loom_type_none(),
    };
    return iree_ok_status();
  }

  *out_argument = (loom_low_lower_abi_argument_t){
      .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
      .abi_type = loom_type_none(),
      .resource_semantic_type = loom_type_none(),
  };
  return loom_x86_map_avx512_type(user_data, context, source_function_op,
                                  source_type, &out_argument->abi_type);
}

static iree_status_t loom_x86_map_avx512_packed_dot_type(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_type_t source_type,
    loom_type_t* out_low_type) {
  uint32_t vector_bit_width = 0;
  if (loom_x86_packed_dot_type_static_vector_bit_width(source_type,
                                                       &vector_bit_width)) {
    switch (vector_bit_width) {
      case 128:
        return loom_x86_make_avx512_packed_dot_xmm_register_type(context,
                                                                 out_low_type);
      case 256:
        return loom_x86_make_avx512_packed_dot_ymm_register_type(context,
                                                                 out_low_type);
      case 512:
        return loom_x86_make_avx512_packed_dot_zmm_register_type(context,
                                                                 out_low_type);
      default:
        break;
    }
  }
  return loom_x86_map_avx512_type(user_data, context, source_op, source_type,
                                  out_low_type);
}

static iree_status_t loom_x86_map_avx512_packed_dot_argument(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_function_op, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument) {
  (void)source_argument_index;
  IREE_ASSERT_ARGUMENT(out_argument);
  const loom_type_t source_type = loom_module_value_type(
      loom_low_lower_context_module(context), source_argument_id);
  if (loom_type_is_buffer(source_type)) {
    loom_type_t address_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_x86_make_gpr64_register_type(context, &address_type));
    *out_argument = (loom_low_lower_abi_argument_t){
        .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
        .abi_type = address_type,
        .resource_semantic_type = loom_type_none(),
    };
    return iree_ok_status();
  }

  *out_argument = (loom_low_lower_abi_argument_t){
      .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
      .abi_type = loom_type_none(),
      .resource_semantic_type = loom_type_none(),
  };
  return loom_x86_map_avx512_packed_dot_type(user_data, context,
                                             source_function_op, source_type,
                                             &out_argument->abi_type);
}

static const loom_low_lower_rule_set_t* const kX86Avx512RuleSets[] = {
    &loom_x86_avx512_rule_set,
    &loom_x86_avx512_reduce_rule_set,
};

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

static iree_status_t loom_x86_select_avx512_packed_dot_op(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_low_lower_plan_t* out_plan) {
  if (loom_x86_op_is_vector_dot(source_op->kind)) {
    return loom_x86_select_packed_dot_op(user_data, context, source_op,
                                         out_plan);
  }
  return loom_x86_select_avx512_op(user_data, context, source_op, out_plan);
}

static iree_status_t loom_x86_emit_avx512_packed_dot_op(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_low_lower_plan_t plan) {
  if (loom_x86_op_is_vector_dot(source_op->kind)) {
    return loom_x86_emit_packed_dot_op(user_data, context, source_op, plan);
  }
  return loom_x86_emit_avx512_op(user_data, context, source_op, plan);
}

static const loom_low_lower_policy_t kX86Avx512LowLowerPolicy = {
    .name = IREE_SVL("x86-avx512-low-lower"),
    .map_type = {.fn = loom_x86_map_avx512_type, .user_data = NULL},
    .map_argument = {.fn = loom_x86_map_avx512_argument, .user_data = NULL},
    .rule_sets =
        {
            .count = IREE_ARRAYSIZE(kX86Avx512RuleSets),
            .values = kX86Avx512RuleSets,
        },
    .select_op = {.fn = loom_x86_select_avx512_op, .user_data = NULL},
    .emit_op = {.fn = loom_x86_emit_avx512_op, .user_data = NULL},
};

static const loom_low_lower_policy_t kX86PackedDotLowLowerPolicy = {
    .name = IREE_SVL("x86-packed-dot-low-lower"),
    .map_type = {.fn = loom_x86_map_packed_dot_type, .user_data = NULL},
    .select_op = {.fn = loom_x86_select_packed_dot_op, .user_data = NULL},
    .emit_op = {.fn = loom_x86_emit_packed_dot_op, .user_data = NULL},
};

static const loom_low_lower_policy_t kX86Avx512PackedDotLowLowerPolicy = {
    .name = IREE_SVL("x86-avx512-packed-dot-low-lower"),
    .map_type = {.fn = loom_x86_map_avx512_packed_dot_type, .user_data = NULL},
    .map_argument = {.fn = loom_x86_map_avx512_packed_dot_argument,
                     .user_data = NULL},
    .rule_sets =
        {
            .count = IREE_ARRAYSIZE(kX86Avx512RuleSets),
            .values = kX86Avx512RuleSets,
        },
    .select_op = {.fn = loom_x86_select_avx512_packed_dot_op,
                  .user_data = NULL},
    .emit_op = {.fn = loom_x86_emit_avx512_packed_dot_op, .user_data = NULL},
};

const loom_target_low_legality_provider_t
    loom_x86_low_legality_provider_storage = {
        .name = IREE_SVL("x86"),
        .builtin_dialect_bits = 1u << LOOM_DIALECT_VECTOR,
        .try_verify_op = loom_x86_low_legality_try_verify_op,
};

const loom_low_lower_policy_t* loom_x86_avx512_low_lower_policy(void) {
  return &kX86Avx512LowLowerPolicy;
}

const loom_low_lower_policy_t* loom_x86_packed_dot_low_lower_policy(void) {
  return &kX86PackedDotLowLowerPolicy;
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
          .policy = &kX86Avx512LowLowerPolicy,
      },
      {
          .contract_set_key = IREE_SVL("x86.packed_dot.core"),
          .policy = &kX86PackedDotLowLowerPolicy,
      },
      {
          .contract_set_key = IREE_SVL("x86.avx512_packed_dot.core"),
          .policy = &kX86Avx512PackedDotLowLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
