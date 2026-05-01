// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/wasm/descriptors.h"
#include "loom/target/emit/wasm/contracts/core_simd128.h"
#include "loom/target/emit/wasm/core_simd128_lower_rules.h"
#include "loom/target/emit/wasm/lower.h"

static bool loom_wasm_type_is_address_i32(loom_type_t type) {
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  switch (loom_type_element_type(type)) {
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_OFFSET:
    case LOOM_SCALAR_TYPE_I32:
      return true;
    default:
      return false;
  }
}

static bool loom_wasm_type_is_scalar_f32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32;
}

static bool loom_wasm_type_is_vector_4xi32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32 &&
         loom_type_dim_static_size_at(type, 0) == 4;
}

static bool loom_wasm_type_is_vector_4xi1(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1 &&
         loom_type_dim_static_size_at(type, 0) == 4;
}

static bool loom_wasm_type_is_vector_4xf32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32 &&
         loom_type_dim_static_size_at(type, 0) == 4;
}

static iree_status_t loom_wasm_make_i32_register_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, WASM_CORE_SIMD128_REG_CLASS_ID_I32, 1, out_type);
}

static iree_status_t loom_wasm_make_f32_register_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, WASM_CORE_SIMD128_REG_CLASS_ID_F32, 1, out_type);
}

static iree_status_t loom_wasm_make_v128_register_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, WASM_CORE_SIMD128_REG_CLASS_ID_V128, 1, out_type);
}

static iree_status_t loom_wasm_map_type(void* user_data,
                                        loom_low_lower_context_t* context,
                                        const loom_op_t* source_op,
                                        loom_type_t source_type,
                                        loom_type_t* out_low_type) {
  (void)user_data;
  if (loom_wasm_type_is_address_i32(source_type)) {
    return loom_wasm_make_i32_register_type(context, out_low_type);
  }
  if (loom_wasm_type_is_scalar_f32(source_type)) {
    return loom_wasm_make_f32_register_type(context, out_low_type);
  }
  if (loom_wasm_type_is_vector_4xi32(source_type)) {
    return loom_wasm_make_v128_register_type(context, out_low_type);
  }
  if (loom_wasm_type_is_vector_4xi1(source_type)) {
    return loom_wasm_make_v128_register_type(context, out_low_type);
  }
  if (loom_wasm_type_is_vector_4xf32(source_type)) {
    return loom_wasm_make_v128_register_type(context, out_low_type);
  }
  return loom_low_lower_emit_reject(
      context, source_op, IREE_SV("type"), IREE_SV("source"),
      IREE_SV("Wasm lowering currently supports only i32/index/offset scalar "
              "values, f32 scalar values, and vector<4xi1>/vector<4xi32>/"
              "vector<4xf32> SIMD values"));
}

static iree_status_t loom_wasm_map_argument(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_function_op, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument) {
  (void)source_argument_index;
  loom_type_t source_type = loom_module_value_type(
      loom_low_lower_context_module(context), source_argument_id);
  if (loom_type_is_buffer(source_type)) {
    loom_type_t address_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_wasm_make_i32_register_type(context, &address_type));
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
  return loom_wasm_map_type(user_data, context, source_function_op, source_type,
                            &out_argument->abi_type);
}

static const loom_low_lower_rule_set_t* const kWasmRuleSets[] = {
    &loom_wasm_core_simd128_lower_rule_set,
};

static const loom_target_contract_binding_t kWasmContractBindings[] = {
    {&loom_wasm_core_simd128_contract_fragment, 0},
};

typedef struct loom_wasm_memory_access_plan_t {
  // Target-independent source memory decomposition.
  loom_low_source_memory_access_plan_t source;
  // Descriptor row selected for the load/store memory packet.
  loom_low_lower_resolved_descriptor_t memory_descriptor;
  // Descriptor row selected for i32 address constants.
  loom_low_lower_resolved_descriptor_t i32_const_descriptor;
  // Descriptor row selected for i32 address additions.
  loom_low_lower_resolved_descriptor_t i32_add_descriptor;
  // Descriptor row selected for i32 address multiplies.
  loom_low_lower_resolved_descriptor_t i32_mul_descriptor;
  // Low i32 register type used by address arithmetic packets.
  loom_type_t i32_type;
  // Low v128 register type produced by load packets, or none for stores.
  loom_type_t load_result_type;
  // Module string ID for i32 const immediate attributes.
  loom_string_id_t i32_value_attr_name_id;
} loom_wasm_memory_access_plan_t;

static bool loom_wasm_memory_space_is_linear(
    loom_value_fact_memory_space_t memory_space) {
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
      return true;
    default:
      return false;
  }
}

static bool loom_wasm_i64_fits_i32(int64_t value) {
  return value >= INT32_MIN && value <= INT32_MAX;
}

static bool loom_wasm_memory_access_shape_is_v128(
    const loom_low_source_memory_access_plan_t* plan) {
  return plan->element_byte_count == 4 && plan->vector_lane_count == 4 &&
         plan->vector_lane_byte_stride == 4;
}

static iree_status_t loom_wasm_select_memory_access(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_source_memory_operation_kind_t operation_kind,
    uint64_t memory_descriptor_id, loom_wasm_memory_access_plan_t* out_plan,
    bool* out_selected) {
  *out_plan = (loom_wasm_memory_access_plan_t){
      .i32_type = loom_type_none(),
      .load_result_type = loom_type_none(),
      .i32_value_attr_name_id = LOOM_STRING_ID_INVALID,
  };
  *out_selected = false;

  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_low_source_memory_access_plan_build(
          module, loom_low_lower_context_fact_table(context), source_op,
          &out_plan->source, &diagnostic)) {
    return iree_ok_status();
  }
  if (out_plan->source.operation_kind != operation_kind ||
      !loom_wasm_memory_access_shape_is_v128(&out_plan->source) ||
      !loom_wasm_memory_space_is_linear(out_plan->source.memory_space) ||
      !loom_wasm_i64_fits_i32(out_plan->source.static_byte_offset) ||
      !loom_low_source_memory_value_is_block_argument(
          module, out_plan->source.root_value_id)) {
    return iree_ok_status();
  }
  if (!loom_low_source_memory_dynamic_offset_fits_unsigned_bit_count(
          &out_plan->source, out_plan->source.static_byte_offset, 32)) {
    return iree_ok_status();
  }
  for (uint8_t i = 0; i < out_plan->source.dynamic_term_count; ++i) {
    const loom_low_source_memory_dynamic_term_t* term =
        &out_plan->source.dynamic_terms[i];
    if (term->byte_stride <= 0 || !loom_wasm_i64_fits_i32(term->byte_stride)) {
      return iree_ok_status();
    }
  }

  IREE_RETURN_IF_ERROR(loom_low_lower_resolve_descriptor(
      context, memory_descriptor_id, &out_plan->memory_descriptor));
  if (operation_kind == LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD) {
    IREE_RETURN_IF_ERROR(loom_wasm_make_v128_register_type(
        context, &out_plan->load_result_type));
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_resolve_descriptor(
      context, WASM_CORE_SIMD128_DESCRIPTOR_ID_I32_CONST,
      &out_plan->i32_const_descriptor));
  IREE_RETURN_IF_ERROR(loom_low_lower_resolve_descriptor(
      context, WASM_CORE_SIMD128_DESCRIPTOR_ID_I32_ADD,
      &out_plan->i32_add_descriptor));
  IREE_RETURN_IF_ERROR(loom_low_lower_resolve_descriptor(
      context, WASM_CORE_SIMD128_DESCRIPTOR_ID_I32_MUL,
      &out_plan->i32_mul_descriptor));
  IREE_RETURN_IF_ERROR(
      loom_wasm_make_i32_register_type(context, &out_plan->i32_type));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("i32_value"), &out_plan->i32_value_attr_name_id));
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_wasm_select_op(void* user_data,
                                         loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_low_lower_plan_t* out_plan) {
  (void)user_data;
  *out_plan = loom_low_lower_plan_empty();
  switch (source_op->kind) {
    case LOOM_OP_BUFFER_VIEW:
      *out_plan = loom_low_lower_plan_make(source_op->kind, NULL);
      return iree_ok_status();
    case LOOM_OP_VECTOR_LOAD:
    case LOOM_OP_VECTOR_STORE: {
      loom_wasm_memory_access_plan_t selected_plan = {0};
      bool selected = false;
      const bool is_load = source_op->kind == LOOM_OP_VECTOR_LOAD;
      IREE_RETURN_IF_ERROR(loom_wasm_select_memory_access(
          context, source_op,
          is_load ? LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD
                  : LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE,
          is_load ? WASM_CORE_SIMD128_DESCRIPTOR_ID_V128_LOAD
                  : WASM_CORE_SIMD128_DESCRIPTOR_ID_V128_STORE,
          &selected_plan, &selected));
      if (selected) {
        loom_wasm_memory_access_plan_t* plan = NULL;
        IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
            context, sizeof(*plan), (void**)&plan));
        *plan = selected_plan;
        *out_plan = loom_low_lower_plan_make(source_op->kind, plan);
      }
      return iree_ok_status();
    }
    default:
      return iree_ok_status();
  }
}

static loom_named_attr_t loom_wasm_make_i64_attr(loom_string_id_t name_id,
                                                 int64_t value) {
  IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
  return (loom_named_attr_t){
      .name_id = name_id,
      .value = loom_attr_i64(value),
  };
}

static loom_named_attr_t loom_wasm_make_i32_value_attr(loom_string_id_t name_id,
                                                       int64_t value) {
  IREE_ASSERT(loom_wasm_i64_fits_i32(value));
  return loom_wasm_make_i64_attr(name_id, value);
}

static iree_status_t loom_wasm_emit_resolved_i32_const(
    loom_low_lower_context_t* context,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_string_id_t value_attr_name_id, loom_type_t result_type, int64_t value,
    loom_location_id_t location, loom_value_id_t* out_value_id) {
  *out_value_id = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attr =
      loom_wasm_make_i32_value_attr(value_attr_name_id, value);
  loom_op_t* const_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_const(
      context, descriptor, loom_make_named_attr_slice(&attr, 1), result_type,
      location, &const_op));
  *out_value_id = loom_low_const_result(const_op);
  return iree_ok_status();
}

static iree_status_t loom_wasm_emit_resolved_typed_binary(
    loom_low_lower_context_t* context,
    const loom_low_lower_resolved_descriptor_t* descriptor, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_value_id) {
  *out_value_id = LOOM_VALUE_ID_INVALID;
  loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &result_type, 1, NULL, 0, location, &op));
  *out_value_id = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_wasm_emit_address_offset(
    loom_low_lower_context_t* context,
    const loom_wasm_memory_access_plan_t* plan, loom_location_id_t location,
    loom_value_id_t* inout_address) {
  if (plan->source.static_byte_offset == 0) {
    return iree_ok_status();
  }
  loom_value_id_t offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_wasm_emit_resolved_i32_const(
      context, &plan->i32_const_descriptor, plan->i32_value_attr_name_id,
      plan->i32_type, plan->source.static_byte_offset, location, &offset));
  return loom_wasm_emit_resolved_typed_binary(
      context, &plan->i32_add_descriptor, *inout_address, offset,
      plan->i32_type, location, inout_address);
}

static iree_status_t loom_wasm_emit_dynamic_address_offset(
    loom_low_lower_context_t* context,
    const loom_wasm_memory_access_plan_t* plan, loom_location_id_t location,
    loom_value_id_t* inout_address) {
  if (!loom_low_source_memory_access_is_dynamic(&plan->source)) {
    return iree_ok_status();
  }
  for (uint8_t i = 0; i < plan->source.dynamic_term_count; ++i) {
    const loom_low_source_memory_dynamic_term_t* term =
        &plan->source.dynamic_terms[i];
    loom_value_id_t dynamic_index = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_low_lower_lookup_value(context, term->index, &dynamic_index));
    loom_value_id_t dynamic_offset = dynamic_index;
    if (term->byte_stride != 1) {
      loom_value_id_t stride = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_wasm_emit_resolved_i32_const(
          context, &plan->i32_const_descriptor, plan->i32_value_attr_name_id,
          plan->i32_type, term->byte_stride, location, &stride));
      IREE_RETURN_IF_ERROR(loom_wasm_emit_resolved_typed_binary(
          context, &plan->i32_mul_descriptor, dynamic_index, stride,
          plan->i32_type, location, &dynamic_offset));
    }
    IREE_RETURN_IF_ERROR(loom_wasm_emit_resolved_typed_binary(
        context, &plan->i32_add_descriptor, *inout_address, dynamic_offset,
        plan->i32_type, location, inout_address));
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_emit_memory_address(
    loom_low_lower_context_t* context,
    const loom_wasm_memory_access_plan_t* plan, loom_location_id_t location,
    loom_value_id_t* out_address) {
  *out_address = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, plan->source.root_value_id, out_address));
  IREE_RETURN_IF_ERROR(loom_wasm_emit_dynamic_address_offset(
      context, plan, location, out_address));
  return loom_wasm_emit_address_offset(context, plan, location, out_address);
}

static iree_status_t loom_wasm_lower_vector_load(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_wasm_memory_access_plan_t* plan) {
  loom_value_id_t address = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_wasm_emit_memory_address(
      context, plan, source_op->location, &address));
  loom_op_t* load_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->memory_descriptor, &address, 1,
      loom_named_attr_slice_empty(), &plan->load_result_type, 1, NULL, 0,
      source_op->location, &load_op));
  IREE_RETURN_IF_ERROR(loom_low_lower_record_source_memory_access(
      context, load_op, &plan->source));
  return loom_low_lower_bind_value(
      context, loom_vector_load_result(source_op),
      loom_value_slice_get(loom_low_op_results(load_op), 0));
}

static iree_status_t loom_wasm_lower_vector_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_wasm_memory_access_plan_t* plan) {
  loom_value_id_t address = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_wasm_emit_memory_address(
      context, plan, source_op->location, &address));
  loom_value_id_t value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_store_value(source_op), &value));
  loom_value_id_t operands[] = {address, value};
  loom_op_t* store_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->memory_descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), NULL, 0, NULL, 0, source_op->location,
      &store_op));
  return loom_low_lower_record_source_memory_access(context, store_op,
                                                    &plan->source);
}

static iree_status_t loom_wasm_emit_op(void* user_data,
                                       loom_low_lower_context_t* context,
                                       const loom_op_t* source_op,
                                       loom_low_lower_plan_t plan) {
  (void)user_data;
  switch (plan.id) {
    case LOOM_OP_BUFFER_VIEW:
      return loom_low_lower_bind_value_alias(
          context, loom_buffer_view_buffer(source_op),
          loom_buffer_view_result(source_op));
    case LOOM_OP_VECTOR_LOAD:
      return loom_wasm_lower_vector_load(
          context, source_op,
          (const loom_wasm_memory_access_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_STORE:
      return loom_wasm_lower_vector_store(
          context, source_op,
          (const loom_wasm_memory_access_plan_t*)plan.target_data);
    default:
      IREE_CHECK_UNREACHABLE();
  }
}

static const loom_low_lower_policy_t kWasmLowLowerPolicy = {
    .name = IREE_SVL("wasm-lower"),
    .map_type = {.fn = loom_wasm_map_type, .user_data = NULL},
    .map_argument = {.fn = loom_wasm_map_argument, .user_data = NULL},
    .rule_sets =
        {
            .count = IREE_ARRAYSIZE(kWasmRuleSets),
            .values = kWasmRuleSets,
        },
    .contract_bindings = kWasmContractBindings,
    .contract_binding_count = IREE_ARRAYSIZE(kWasmContractBindings),
    .select_op = {.fn = loom_wasm_select_op, .user_data = NULL},
    .emit_op = {.fn = loom_wasm_emit_op, .user_data = NULL},
};

const loom_low_lower_policy_t* loom_wasm_low_lower_policy(void) {
  return &kWasmLowLowerPolicy;
}

void loom_wasm_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("wasm.core.simd128"),
          .policy = &kWasmLowLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
