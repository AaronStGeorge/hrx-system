// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/test/contract_table.h"
#include "loom/target/test/descriptors.h"
#include "loom/target/test/lower.h"
#include "loom/target/test/lower_rules.h"

//===----------------------------------------------------------------------===//
// Type mapping
//===----------------------------------------------------------------------===//

static bool loom_test_low_is_i32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32;
}

static bool loom_test_low_is_i8(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I8;
}

static bool loom_test_low_is_f32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32;
}

static bool loom_test_low_is_index_like(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         (loom_type_element_type(type) == LOOM_SCALAR_TYPE_INDEX ||
          loom_type_element_type(type) == LOOM_SCALAR_TYPE_OFFSET);
}

static bool loom_test_low_is_i1(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1;
}

static bool loom_test_low_is_vector_4xi32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32 &&
         loom_type_dim_static_size_at(type, 0) == 4;
}

static bool loom_test_low_is_vector_4xf32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32 &&
         loom_type_dim_static_size_at(type, 0) == 4;
}

static bool loom_test_low_is_vector_4xi1(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1 &&
         loom_type_dim_static_size_at(type, 0) == 4;
}

static bool loom_test_low_is_vector_16xi8(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I8 &&
         loom_type_dim_static_size_at(type, 0) == 16;
}

static iree_status_t loom_test_low_make_register_type(
    loom_low_lower_context_t* context, iree_string_view_t register_class,
    uint32_t unit_count, loom_type_t* out_type) {
  IREE_ASSERT_ARGUMENT(out_type);
  loom_string_id_t register_class_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(loom_low_lower_context_module(context),
                                register_class, &register_class_id));
  *out_type = loom_type_register(register_class_id, unit_count);
  return iree_ok_status();
}

iree_status_t loom_test_low_lower_map_type(void* user_data,
                                           loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_type_t source_type,
                                           loom_type_t* out_low_type) {
  (void)user_data;
  IREE_ASSERT_ARGUMENT(out_low_type);
  if (loom_test_low_is_i32(source_type) || loom_test_low_is_i1(source_type) ||
      loom_test_low_is_index_like(source_type)) {
    return loom_test_low_make_register_type(context, IREE_SV("test.i32"), 1,
                                            out_low_type);
  }
  if (loom_test_low_is_i8(source_type)) {
    return loom_test_low_make_register_type(context, IREE_SV("test.i8"), 1,
                                            out_low_type);
  }
  if (loom_test_low_is_f32(source_type)) {
    return loom_test_low_make_register_type(context, IREE_SV("test.f32"), 1,
                                            out_low_type);
  }
  if (loom_test_low_is_vector_4xi32(source_type)) {
    return loom_test_low_make_register_type(context, IREE_SV("test.i32"), 4,
                                            out_low_type);
  }
  if (loom_test_low_is_vector_4xf32(source_type)) {
    return loom_test_low_make_register_type(context, IREE_SV("test.f32"), 4,
                                            out_low_type);
  }
  if (loom_test_low_is_vector_4xi1(source_type)) {
    return loom_test_low_make_register_type(context, IREE_SV("test.i32"), 4,
                                            out_low_type);
  }
  if (loom_test_low_is_vector_16xi8(source_type)) {
    return loom_test_low_make_register_type(context, IREE_SV("test.i8"), 16,
                                            out_low_type);
  }
  return loom_low_lower_emit_reject(
      context, source_op, IREE_SV("type"), IREE_SV("source"),
      IREE_SV("test lowering only maps i1, i8, i32, f32, index, offset, "
              "vector<4xi1>, vector<4xi32>, vector<4xf32>, and "
              "vector<16xi8>"));
}

iree_status_t loom_test_low_lower_map_argument(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_function_op, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument) {
  IREE_ASSERT_ARGUMENT(out_argument);
  loom_type_t source_type = loom_module_value_type(
      loom_low_lower_context_module(context), source_argument_id);
  if (loom_type_is_buffer(source_type)) {
    loom_type_t resource_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_test_low_make_register_type(
        context, IREE_SV("test.ptr"), 1, &resource_type));
    *out_argument = (loom_low_lower_abi_argument_t){
        .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_RESOURCE,
        .abi_type = resource_type,
        .resource_import_kind = LOOM_LOW_RESOURCE_IMPORT_KIND_NATIVE_POINTER,
        .resource_index = source_argument_index,
        .resource_semantic_type = source_type,
    };
    return iree_ok_status();
  }

  *out_argument = (loom_low_lower_abi_argument_t){
      .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
      .abi_type = loom_type_none(),
      .resource_semantic_type = loom_type_none(),
  };
  return loom_test_low_lower_map_type(user_data, context, source_function_op,
                                      source_type, &out_argument->abi_type);
}

iree_status_t loom_test_low_lower_rule_match_map_value(
    void* user_data, const loom_low_lower_rule_match_context_t* context,
    const loom_op_t* source_op, loom_value_id_t source_value_id,
    loom_low_lower_rule_mapped_value_t* out_mapped_value) {
  (void)user_data;
  (void)source_op;
  IREE_ASSERT_ARGUMENT(out_mapped_value);
  const loom_target_contract_query_environment_t environment = {
      .module = context->module,
      .descriptor_set = context->descriptor_set,
  };
  return loom_test_low_lower_map_contract_value(
      NULL, &environment, source_op, source_value_id, out_mapped_value);
}

iree_status_t loom_test_low_lower_map_contract_value(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op, loom_value_id_t source_value_id,
    loom_low_lower_rule_mapped_value_t* out_mapped_value) {
  (void)user_data;
  (void)source_op;
  IREE_ASSERT_ARGUMENT(out_mapped_value);
  *out_mapped_value = loom_low_lower_rule_mapped_value_none();
  loom_type_t source_type =
      loom_module_value_type(environment->module, source_value_id);
  if (loom_test_low_is_i32(source_type) || loom_test_low_is_i1(source_type) ||
      loom_test_low_is_index_like(source_type)) {
    *out_mapped_value = loom_low_lower_rule_mapped_value_register(
        TEST_LOW_CORE_REG_CLASS_ID_TEST_I32, 1);
  } else if (loom_test_low_is_i8(source_type)) {
    *out_mapped_value = loom_low_lower_rule_mapped_value_register(
        TEST_LOW_CORE_REG_CLASS_ID_TEST_I8, 1);
  } else if (loom_test_low_is_f32(source_type)) {
    *out_mapped_value = loom_low_lower_rule_mapped_value_register(
        TEST_LOW_CORE_REG_CLASS_ID_TEST_F32, 1);
  } else if (loom_test_low_is_vector_4xi32(source_type)) {
    *out_mapped_value = loom_low_lower_rule_mapped_value_register(
        TEST_LOW_CORE_REG_CLASS_ID_TEST_I32, 4);
  } else if (loom_test_low_is_vector_4xf32(source_type)) {
    *out_mapped_value = loom_low_lower_rule_mapped_value_register(
        TEST_LOW_CORE_REG_CLASS_ID_TEST_F32, 4);
  } else if (loom_test_low_is_vector_4xi1(source_type)) {
    *out_mapped_value = loom_low_lower_rule_mapped_value_register(
        TEST_LOW_CORE_REG_CLASS_ID_TEST_I32, 4);
  } else if (loom_test_low_is_vector_16xi8(source_type)) {
    *out_mapped_value = loom_low_lower_rule_mapped_value_register(
        TEST_LOW_CORE_REG_CLASS_ID_TEST_I8, 16);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Source memory lowering callbacks
//===----------------------------------------------------------------------===//

static bool loom_test_low_value_facts_are_exact_zero(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id) {
  loom_value_facts_t facts = loom_value_fact_table_lookup(fact_table, value_id);
  return loom_value_facts_is_exact(facts) &&
         !loom_value_facts_is_float(facts) && facts.range_lo == 0;
}

static bool loom_test_low_can_lower_buffer_view(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  if (!loom_type_is_view(
          loom_module_value_type(loom_low_lower_context_module(context),
                                 loom_buffer_view_result(source_op)))) {
    return false;
  }
  return loom_test_low_value_facts_are_exact_zero(
      loom_low_lower_context_fact_table(context),
      loom_buffer_view_byte_offset(source_op));
}

static bool loom_test_low_source_memory_access_is_supported(
    const loom_low_source_memory_access_plan_t* plan) {
  const bool supported_memory_space =
      plan->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN ||
      plan->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC ||
      plan->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL;
  if (!supported_memory_space || plan->root_value_id == LOOM_VALUE_ID_INVALID ||
      plan->element_byte_count != 4 || plan->vector_lane_count != 4 ||
      plan->vector_lane_byte_stride != 4 || plan->static_byte_offset != 0 ||
      plan->cache_policy.build_flags != 0) {
    return false;
  }
  if (!loom_low_source_memory_access_is_dynamic(plan)) {
    return true;
  }
  const loom_low_source_memory_dynamic_term_t* term =
      loom_low_source_memory_access_single_dynamic_term(plan);
  return term &&
         term->source == LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE &&
         term->byte_stride == 4;
}

typedef struct loom_test_low_memory_access_plan_t {
  // Target-independent source memory decomposition.
  loom_low_source_memory_access_plan_t access;
  // Test-low memory descriptor selected during source op selection.
  loom_low_lower_resolved_descriptor_t descriptor;
} loom_test_low_memory_access_plan_t;

static bool loom_test_low_memory_access_descriptor_id(
    const loom_module_t* module,
    const loom_low_source_memory_access_plan_t* access,
    const loom_op_t* source_op, uint64_t* out_descriptor_id) {
  IREE_ASSERT_ARGUMENT(out_descriptor_id);
  *out_descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE;
  const bool dynamic = loom_low_source_memory_access_is_dynamic(access);
  switch (source_op->kind) {
    case LOOM_OP_VECTOR_LOAD: {
      const loom_type_t result_type =
          loom_module_value_type(module, loom_vector_load_result(source_op));
      if (loom_test_low_is_vector_4xi32(result_type)) {
        *out_descriptor_id =
            dynamic ? TEST_LOW_CORE_DESCRIPTOR_ID_TEST_LOAD_INDEX_V4I32
                    : TEST_LOW_CORE_DESCRIPTOR_ID_TEST_LOAD_V4I32;
        return true;
      }
      if (loom_test_low_is_vector_4xf32(result_type)) {
        *out_descriptor_id =
            dynamic ? TEST_LOW_CORE_DESCRIPTOR_ID_TEST_LOAD_INDEX_V4F32
                    : TEST_LOW_CORE_DESCRIPTOR_ID_TEST_LOAD_V4F32;
        return true;
      }
      return false;
    }
    case LOOM_OP_VECTOR_STORE: {
      const loom_type_t value_type =
          loom_module_value_type(module, loom_vector_store_value(source_op));
      if (loom_test_low_is_vector_4xi32(value_type)) {
        *out_descriptor_id =
            dynamic ? TEST_LOW_CORE_DESCRIPTOR_ID_TEST_STORE_INDEX_V4I32
                    : TEST_LOW_CORE_DESCRIPTOR_ID_TEST_STORE_V4I32;
        return true;
      }
      if (loom_test_low_is_vector_4xf32(value_type)) {
        *out_descriptor_id =
            dynamic ? TEST_LOW_CORE_DESCRIPTOR_ID_TEST_STORE_INDEX_V4F32
                    : TEST_LOW_CORE_DESCRIPTOR_ID_TEST_STORE_V4F32;
        return true;
      }
      return false;
    }
    default:
      return false;
  }
}

static iree_status_t loom_test_low_select_memory_access(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_source_memory_operation_kind_t operation_kind,
    loom_low_lower_plan_t* out_plan) {
  loom_low_source_memory_access_plan_t access = {0};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  if (!loom_low_source_memory_access_plan_build(
          loom_low_lower_context_module(context),
          loom_low_lower_context_fact_table(context), source_op, &access,
          &diagnostic)) {
    return iree_ok_status();
  }
  if (access.operation_kind != operation_kind ||
      !loom_test_low_source_memory_access_is_supported(&access)) {
    return iree_ok_status();
  }
  uint64_t descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE;
  if (!loom_test_low_memory_access_descriptor_id(
          loom_low_lower_context_module(context), &access, source_op,
          &descriptor_id)) {
    return iree_ok_status();
  }

  loom_test_low_memory_access_plan_t* plan_data = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sizeof(*plan_data), (void**)&plan_data));
  *plan_data = (loom_test_low_memory_access_plan_t){
      .access = access,
  };
  IREE_RETURN_IF_ERROR(loom_low_lower_resolve_descriptor(
      context, descriptor_id, &plan_data->descriptor));
  *out_plan = loom_low_lower_plan_make(source_op->kind, plan_data);
  return iree_ok_status();
}

static iree_status_t loom_test_low_select_op(void* user_data,
                                             loom_low_lower_context_t* context,
                                             const loom_op_t* source_op,
                                             loom_low_lower_plan_t* out_plan) {
  (void)user_data;
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = loom_low_lower_plan_empty();
  switch (source_op->kind) {
    case LOOM_OP_BUFFER_VIEW:
      if (loom_test_low_can_lower_buffer_view(context, source_op)) {
        *out_plan = loom_low_lower_plan_make(source_op->kind, NULL);
      }
      return iree_ok_status();
    case LOOM_OP_VECTOR_LOAD:
      return loom_test_low_select_memory_access(
          context, source_op, LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD, out_plan);
    case LOOM_OP_VECTOR_STORE:
      return loom_test_low_select_memory_access(
          context, source_op, LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE, out_plan);
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_test_low_emit_vector_load(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_test_low_memory_access_plan_t* plan) {
  IREE_ASSERT_ARGUMENT(plan);
  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, plan->access.view_value_id, &low_resource));
  loom_value_id_t operands[2] = {
      low_resource,
      LOOM_VALUE_ID_INVALID,
  };
  iree_host_size_t operand_count = 1;
  if (loom_low_source_memory_access_is_dynamic(&plan->access)) {
    const loom_low_source_memory_dynamic_term_t* term =
        loom_low_source_memory_access_single_dynamic_term(&plan->access);
    IREE_ASSERT(term);
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, term->index, &operands[operand_count++]));
  }
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(
      context, source_op, loom_vector_load_result(source_op), &result_type));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->descriptor, operands, operand_count,
      loom_named_attr_slice_empty(), &result_type, 1, NULL, 0,
      source_op->location, &low_op));
  IREE_RETURN_IF_ERROR(loom_low_lower_record_source_memory_access(
      context, low_op, &plan->access));
  return loom_low_lower_bind_value(
      context, loom_vector_load_result(source_op),
      loom_value_slice_get(loom_low_op_results(low_op), 0));
}

static iree_status_t loom_test_low_emit_vector_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_test_low_memory_access_plan_t* plan) {
  IREE_ASSERT_ARGUMENT(plan);
  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, plan->access.view_value_id, &low_resource));
  loom_value_id_t operands[3] = {
      low_resource,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  iree_host_size_t operand_count = 1;
  if (loom_low_source_memory_access_is_dynamic(&plan->access)) {
    const loom_low_source_memory_dynamic_term_t* term =
        loom_low_source_memory_access_single_dynamic_term(&plan->access);
    IREE_ASSERT(term);
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, term->index, &operands[operand_count++]));
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_store_value(source_op), &operands[operand_count++]));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->descriptor, operands, operand_count,
      loom_named_attr_slice_empty(), /*result_types=*/NULL,
      /*result_count=*/0, NULL, 0, source_op->location, &low_op));
  return loom_low_lower_record_source_memory_access(context, low_op,
                                                    &plan->access);
}

static iree_status_t loom_test_low_emit_op(void* user_data,
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
      return loom_test_low_emit_vector_load(
          context, source_op,
          (const loom_test_low_memory_access_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_STORE:
      return loom_test_low_emit_vector_store(
          context, source_op,
          (const loom_test_low_memory_access_plan_t*)plan.target_data);
    default:
      IREE_CHECK_UNREACHABLE();
  }
}

static const loom_low_lower_rule_set_t* const kTestLowRuleSets[] = {
    &loom_test_low_core_lower_rule_set,
};

static const loom_low_lower_policy_t kTestLowLowerPolicy = {
    .name = IREE_SVL("test-low-lower-policy"),
    .map_type = {.fn = loom_test_low_lower_map_type, .user_data = NULL},
    .map_contract_value = {.fn = loom_test_low_lower_map_contract_value,
                           .user_data = NULL},
    .map_argument = {.fn = loom_test_low_lower_map_argument, .user_data = NULL},
    .rule_sets =
        {
            .count = IREE_ARRAYSIZE(kTestLowRuleSets),
            .values = kTestLowRuleSets,
        },
    .contract_table = &loom_test_low_core_contract_table,
    .select_op = {.fn = loom_test_low_select_op, .user_data = NULL},
    .emit_op = {.fn = loom_test_low_emit_op, .user_data = NULL},
};

const loom_low_lower_policy_t* loom_test_low_lower_policy(void) {
  return &kTestLowLowerPolicy;
}

void loom_test_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  IREE_ASSERT_ARGUMENT(out_registry);
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("test.low.core"),
          .policy = &kTestLowLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
