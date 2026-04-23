// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/lower_rules.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/x86/avx512_descriptors.h"
#include "loom/target/arch/x86/lower_internal.h"

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

static iree_status_t loom_x86_make_zmm_register_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, X86_AVX512_CORE_REG_CLASS_ID_X86_ZMM, 1, out_type);
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
  if (loom_x86_type_is_vector_16xi32(source_type) ||
      loom_x86_type_is_vector_16xf32(source_type)) {
    return loom_x86_make_zmm_register_type(context, out_low_type);
  }
  return loom_low_lower_emit_reject(
      context, source_op, IREE_SV("type"), IREE_SV("source"),
      IREE_SV("x86 AVX512 lowering currently supports only index/offset "
              "address values and vector<16xi32>/vector<16xf32> values"));
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

static const loom_low_lower_rule_set_t* const kX86Avx512RuleSets[] = {
    &loom_x86_avx512_rule_set,
};

static bool loom_x86_memory_space_is_object_memory(
    loom_value_fact_memory_space_t memory_space) {
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
      return true;
    default:
      return false;
  }
}

static bool loom_x86_i64_fits_disp32(int64_t value) {
  return value >= INT32_MIN && value <= INT32_MAX;
}

static bool loom_x86_memory_access_shape_is_zmm32(
    const loom_low_source_memory_access_plan_t* plan) {
  return plan->element_byte_count == 4 && plan->vector_lane_count == 16 &&
         plan->vector_lane_byte_stride == 4;
}

typedef enum loom_x86_memory_address_kind_e {
  LOOM_X86_MEMORY_ADDRESS_STATIC = 0,
  LOOM_X86_MEMORY_ADDRESS_INDEXED = 1,
} loom_x86_memory_address_kind_t;

typedef struct loom_x86_memory_access_plan_t {
  // Target-independent memory decomposition shared across targets.
  loom_low_source_memory_access_plan_t source;
  // Selected x86 addressing form.
  loom_x86_memory_address_kind_t address_kind;
  // x86 index scale for LOOM_X86_MEMORY_ADDRESS_INDEXED.
  uint8_t index_scale;
} loom_x86_memory_access_plan_t;

static bool loom_x86_source_value_is_block_argument(const loom_module_t* module,
                                                    loom_value_id_t value_id) {
  if (value_id >= module->values.count) {
    return false;
  }
  return loom_value_is_block_arg(loom_module_value(module, value_id));
}

static bool loom_x86_dynamic_stride_as_address_scale(int64_t byte_stride,
                                                     uint8_t* out_scale) {
  IREE_ASSERT_ARGUMENT(out_scale);
  switch (byte_stride) {
    case 1:
    case 2:
    case 4:
    case 8:
      *out_scale = (uint8_t)byte_stride;
      return true;
    default:
      *out_scale = 0;
      return false;
  }
}

static bool loom_x86_select_memory_access(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_x86_memory_access_plan_t* out_plan) {
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_low_source_memory_access_plan_build(
          module, loom_low_lower_context_fact_table(context), source_op,
          &out_plan->source, &diagnostic)) {
    return false;
  }
  if (!loom_x86_memory_access_shape_is_zmm32(&out_plan->source) ||
      !loom_x86_memory_space_is_object_memory(out_plan->source.memory_space) ||
      !loom_x86_i64_fits_disp32(out_plan->source.static_byte_offset) ||
      !loom_x86_source_value_is_block_argument(
          module, out_plan->source.root_value_id)) {
    return false;
  }
  if (!loom_low_source_memory_access_is_dynamic(&out_plan->source)) {
    out_plan->address_kind = LOOM_X86_MEMORY_ADDRESS_STATIC;
    out_plan->index_scale = 0;
    return true;
  }
  if (out_plan->source.dynamic_index_source !=
      LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE) {
    return false;
  }
  out_plan->address_kind = LOOM_X86_MEMORY_ADDRESS_INDEXED;
  return loom_x86_dynamic_stride_as_address_scale(
      out_plan->source.dynamic_index_byte_stride, &out_plan->index_scale);
}

static iree_status_t loom_x86_select_avx512_op(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_low_lower_plan_t* out_plan) {
  (void)user_data;
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = loom_low_lower_plan_empty();
  switch (source_op->kind) {
    case LOOM_OP_BUFFER_ASSUME_MEMORY_SPACE:
    case LOOM_OP_BUFFER_VIEW:
      *out_plan = loom_low_lower_plan_make(source_op->kind, NULL);
      return iree_ok_status();
    case LOOM_OP_VECTOR_LOAD:
    case LOOM_OP_VECTOR_STORE: {
      loom_x86_memory_access_plan_t* plan = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
          context, sizeof(*plan), (void**)&plan));
      if (loom_x86_select_memory_access(context, source_op, plan)) {
        *out_plan = loom_low_lower_plan_make(source_op->kind, plan);
      }
      return iree_ok_status();
    }
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_x86_make_disp32_attr(
    loom_low_lower_context_t* context, int64_t value,
    loom_named_attr_t* out_attr) {
  IREE_ASSERT_ARGUMENT(out_attr);
  IREE_ASSERT(loom_x86_i64_fits_disp32(value));
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      loom_low_lower_context_module(context), IREE_SV("disp32"), &name_id));
  *out_attr = (loom_named_attr_t){
      .name_id = name_id,
      .value = loom_attr_i64(value),
  };
  return iree_ok_status();
}

static iree_status_t loom_x86_make_scale_attr(loom_low_lower_context_t* context,
                                              uint8_t value,
                                              loom_named_attr_t* out_attr) {
  IREE_ASSERT_ARGUMENT(out_attr);
  IREE_ASSERT(value == 1 || value == 2 || value == 4 || value == 8);
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      loom_low_lower_context_module(context), IREE_SV("scale"), &name_id));
  *out_attr = (loom_named_attr_t){
      .name_id = name_id,
      .value = loom_attr_i64(value),
  };
  return iree_ok_status();
}

static iree_status_t loom_x86_lower_buffer_alias(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    loom_value_id_t result_value_id) {
  loom_value_id_t low_value_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value_id, &low_value_id));
  return loom_low_lower_bind_value(context, result_value_id, low_value_id);
}

static iree_status_t loom_x86_lower_vector_load(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_x86_memory_access_plan_t* plan) {
  loom_value_id_t base = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source.root_value_id, &base));

  loom_named_attr_t attrs[2] = {0};
  IREE_RETURN_IF_ERROR(loom_x86_make_disp32_attr(
      context, plan->source.static_byte_offset, &attrs[0]));
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_x86_make_zmm_register_type(context, &result_type));
  uint64_t descriptor_id =
      X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VMOVDQU32_LOAD_ZMM;
  loom_value_id_t operands[2] = {base, LOOM_VALUE_ID_INVALID};
  iree_host_size_t operand_count = 1;
  iree_host_size_t attr_count = 1;
  if (plan->address_kind == LOOM_X86_MEMORY_ADDRESS_INDEXED) {
    descriptor_id =
        X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VMOVDQU32_LOAD_INDEXED_ZMM;
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, plan->source.dynamic_index, &operands[1]));
    IREE_RETURN_IF_ERROR(
        loom_x86_make_scale_attr(context, plan->index_scale, &attrs[1]));
    operand_count = 2;
    attr_count = 2;
  }
  loom_op_t* load_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_descriptor_op(
      context, descriptor_id, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), &result_type, 1, NULL, 0,
      source_op->location, &load_op));
  return loom_low_lower_bind_value(
      context, loom_vector_load_result(source_op),
      loom_value_slice_get(loom_low_op_results(load_op), 0));
}

static iree_status_t loom_x86_lower_vector_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_x86_memory_access_plan_t* plan) {
  loom_value_id_t base = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source.root_value_id, &base));
  loom_value_id_t value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_store_value(source_op), &value));

  loom_named_attr_t attrs[2] = {0};
  IREE_RETURN_IF_ERROR(loom_x86_make_disp32_attr(
      context, plan->source.static_byte_offset, &attrs[0]));
  uint64_t descriptor_id =
      X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VMOVDQU32_STORE_ZMM;
  loom_value_id_t operands[3] = {value, base, LOOM_VALUE_ID_INVALID};
  iree_host_size_t operand_count = 2;
  iree_host_size_t attr_count = 1;
  if (plan->address_kind == LOOM_X86_MEMORY_ADDRESS_INDEXED) {
    descriptor_id =
        X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VMOVDQU32_STORE_INDEXED_ZMM;
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, plan->source.dynamic_index, &operands[2]));
    IREE_RETURN_IF_ERROR(
        loom_x86_make_scale_attr(context, plan->index_scale, &attrs[1]));
    operand_count = 3;
    attr_count = 2;
  }
  loom_op_t* store_op = NULL;
  return loom_low_lower_emit_descriptor_op(
      context, descriptor_id, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), NULL, 0, NULL, 0,
      source_op->location, &store_op);
}

static iree_status_t loom_x86_emit_avx512_op(void* user_data,
                                             loom_low_lower_context_t* context,
                                             const loom_op_t* source_op,
                                             loom_low_lower_plan_t plan) {
  (void)user_data;
  switch (plan.id) {
    case LOOM_OP_BUFFER_ASSUME_MEMORY_SPACE:
      return loom_x86_lower_buffer_alias(
          context, loom_buffer_assume_memory_space_buffer(source_op),
          loom_buffer_assume_memory_space_result(source_op));
    case LOOM_OP_BUFFER_VIEW:
      return loom_x86_lower_buffer_alias(context,
                                         loom_buffer_view_buffer(source_op),
                                         loom_buffer_view_result(source_op));
    case LOOM_OP_VECTOR_LOAD:
      return loom_x86_lower_vector_load(
          context, source_op,
          (const loom_x86_memory_access_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_STORE:
      return loom_x86_lower_vector_store(
          context, source_op,
          (const loom_x86_memory_access_plan_t*)plan.target_data);
    default:
      IREE_ASSERT_UNREACHABLE();
      return iree_ok_status();
  }
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

const loom_target_low_legality_provider_t
    loom_x86_low_legality_provider_storage = {
        .name = IREE_SVL("x86"),
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
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
