// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/test/lower.h"

#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/test/descriptors.h"

//===----------------------------------------------------------------------===//
// Type mapping
//===----------------------------------------------------------------------===//

static bool loom_test_low_is_i32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32;
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
  if (loom_test_low_is_vector_4xi32(source_type)) {
    return loom_test_low_make_register_type(context, IREE_SV("test.i32"), 4,
                                            out_low_type);
  }
  return loom_low_lower_emit_reject(
      context, source_op, IREE_SV("type"), IREE_SV("source"),
      IREE_SV("test lowering only maps i1, i32, index, offset, and "
              "vector<4xi32>"));
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
  IREE_ASSERT_LT(source_value_id, context->module->values.count);
  *out_mapped_value = loom_low_lower_rule_mapped_value_none();
  loom_type_t source_type =
      loom_module_value_type(context->module, source_value_id);
  if (loom_test_low_is_i32(source_type) || loom_test_low_is_i1(source_type) ||
      loom_test_low_is_index_like(source_type)) {
    *out_mapped_value = loom_low_lower_rule_mapped_value_register(
        TEST_LOW_CORE_REG_CLASS_ID_TEST_I32, 1);
  } else if (loom_test_low_is_vector_4xi32(source_type)) {
    *out_mapped_value = loom_low_lower_rule_mapped_value_register(
        TEST_LOW_CORE_REG_CLASS_ID_TEST_I32, 4);
  }
  return iree_ok_status();
}

bool loom_test_low_lower_rule_match_can_materialize(
    void* user_data, const loom_low_lower_rule_match_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, loom_value_id_t source_value_id) {
  (void)user_data;
  (void)context;
  (void)rule_set;
  (void)source_op;
  (void)value_ref_index;
  (void)source_value_id;
  return true;
}

static bool loom_test_low_can_materialize_copy(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id) {
  (void)context;
  (void)source_op;
  (void)source_value_id;
  return true;
}

static iree_status_t loom_test_low_materialize_copy(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id, loom_value_id_t* out_low_value_id) {
  IREE_ASSERT_ARGUMENT(out_low_value_id);
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_value_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value_id, &low_value_id));
  loom_type_t low_type = loom_module_value_type(
      loom_low_lower_context_module(context), low_value_id);
  loom_op_t* copy_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_copy_build(loom_low_lower_context_builder(context), low_value_id,
                          low_type, source_op->location, &copy_op));
  *out_low_value_id = loom_low_copy_result(copy_op);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Rule table
//===----------------------------------------------------------------------===//

enum {
  LOOM_TEST_LOW_MATERIALIZER_COPY = 1,
};

static const loom_low_lower_value_materializer_t kTestLowMaterializers[] = {
    {
        .can_materialize = loom_test_low_can_materialize_copy,
        .materialize = loom_test_low_materialize_copy,
    },
};

enum {
  LOOM_TEST_LOW_TYPE_I32,
  LOOM_TEST_LOW_TYPE_V4I32,
  LOOM_TEST_LOW_TYPE_INDEX,
};

static const loom_low_lower_type_pattern_t kTestLowTypePatterns[] = {
    {
        .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                 LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT,
        .type_kind = LOOM_TYPE_SCALAR,
        .element_type_mask =
            LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_I32),
    },
    {
        .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                 LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT |
                 LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_RANK |
                 LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM0_RANGE,
        .type_kind = LOOM_TYPE_VECTOR,
        .element_type_mask =
            LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_I32),
        .rank = 1,
        .static_dim0_min = 4,
        .static_dim0_max = 4,
    },
    {
        .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                 LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT,
        .type_kind = LOOM_TYPE_SCALAR,
        .element_type_mask =
            LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_INDEX) |
            LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_OFFSET),
    },
};

enum {
  LOOM_TEST_LOW_OPERAND0,
  LOOM_TEST_LOW_OPERAND1,
  LOOM_TEST_LOW_RESULT0,
  LOOM_TEST_LOW_OPERAND2,
  LOOM_TEST_LOW_TEMPORARY0,
  LOOM_TEST_LOW_MATERIALIZED_OPERAND2,
};

static const loom_low_lower_value_ref_t kTestLowValueRefs[] = {
    {
        .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
        .index = 0,
    },
    {
        .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
        .index = 1,
    },
    {
        .kind = LOOM_LOW_LOWER_VALUE_REF_RESULT,
        .index = 0,
    },
    {
        .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
        .index = 2,
    },
    {
        .kind = LOOM_LOW_LOWER_VALUE_REF_TEMPORARY,
        .index = 0,
    },
    {
        .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
        .index = 2,
        .materializer_index = LOOM_TEST_LOW_MATERIALIZER_COPY,
    },
};

enum {
  LOOM_TEST_LOW_ATTR_COPY_I64,
  LOOM_TEST_LOW_ATTR_COPY_STATIC_LANE0,
  LOOM_TEST_LOW_ATTR_COPY_SHUFFLE_CONTROL,
};

static const loom_low_lower_attr_copy_t kTestLowAttrCopies[] = {
    [LOOM_TEST_LOW_ATTR_COPY_I64] =
        {
            .target_name = IREE_SVL("i32_value"),
            .source_attr_index = 0,
        },
    [LOOM_TEST_LOW_ATTR_COPY_STATIC_LANE0] =
        {
            .kind = LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_ELEMENT,
            .target_name = IREE_SVL("i32_value"),
            .source_attr_index = 0,
            .source_element_index = 0,
        },
    [LOOM_TEST_LOW_ATTR_COPY_SHUFFLE_CONTROL] =
        {
            .kind = LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_PACK_ELEMENTS,
            .target_name = IREE_SVL("shuffle_control"),
            .source_attr_index = 0,
            .source_element_index = 0,
            .source_element_count = 4,
            .source_element_bit_width = 2,
        },
};

enum {
  LOOM_TEST_LOW_DIAGNOSTIC_I32,
  LOOM_TEST_LOW_DIAGNOSTIC_V4I32,
  LOOM_TEST_LOW_DIAGNOSTIC_I64_ATTR,
  LOOM_TEST_LOW_DIAGNOSTIC_INDEX,
  LOOM_TEST_LOW_DIAGNOSTIC_MATERIALIZED,
  LOOM_TEST_LOW_DIAGNOSTIC_STATIC_LANE,
  LOOM_TEST_LOW_DIAGNOSTIC_SHUFFLE_LANES,
};

static const loom_low_lower_diagnostic_t kTestLowDiagnostics[] = {
    {
        .subject_kind = IREE_SVL("type"),
        .subject_name = IREE_SVL("i32"),
        .reason = IREE_SVL("test lowering requires i32 scalar values"),
    },
    {
        .subject_kind = IREE_SVL("type"),
        .subject_name = IREE_SVL("vector<4xi32>"),
        .reason =
            IREE_SVL("test lowering requires vector<4xi32> vector values"),
    },
    {
        .subject_kind = IREE_SVL("attr"),
        .subject_name = IREE_SVL("value"),
        .reason = IREE_SVL("test constant lowering requires an i64 value"),
    },
    {
        .subject_kind = IREE_SVL("type"),
        .subject_name = IREE_SVL("index"),
        .reason = IREE_SVL("test lowering requires index or offset values"),
    },
    {
        .subject_kind = IREE_SVL("materializer"),
        .subject_name = IREE_SVL("copy"),
        .reason = IREE_SVL("test lowering requires copy-materializable values"),
    },
    {
        .subject_kind = IREE_SVL("attr"),
        .subject_name = IREE_SVL("static_indices"),
        .reason = IREE_SVL("test lowering requires one static lane in [0, 3]"),
    },
    {
        .subject_kind = IREE_SVL("attr"),
        .subject_name = IREE_SVL("source_lanes"),
        .reason =
            IREE_SVL("test lowering requires four shuffle lanes in [0, 3]"),
    },
};

enum {
  LOOM_TEST_LOW_CONST_VALUE_GUARD,
  LOOM_TEST_LOW_CONST_RESULT_GUARD,
  LOOM_TEST_LOW_SCALAR_LHS_GUARD,
  LOOM_TEST_LOW_SCALAR_RHS_GUARD,
  LOOM_TEST_LOW_SCALAR_RESULT_GUARD,
  LOOM_TEST_LOW_VECTOR_EXTRACT_STATIC_INDICES_KIND_GUARD,
  LOOM_TEST_LOW_VECTOR_EXTRACT_STATIC_INDICES_COUNT_GUARD,
  LOOM_TEST_LOW_VECTOR_EXTRACT_STATIC_LANE_RANGE_GUARD,
  LOOM_TEST_LOW_VECTOR_EXTRACT_SOURCE_GUARD,
  LOOM_TEST_LOW_VECTOR_EXTRACT_RESULT_GUARD,
  LOOM_TEST_LOW_VECTOR_SHUFFLE_SOURCE_LANES_KIND_GUARD,
  LOOM_TEST_LOW_VECTOR_SHUFFLE_SOURCE_LANES_COUNT_GUARD,
  LOOM_TEST_LOW_VECTOR_SHUFFLE_SOURCE_LANES_RANGE_GUARD,
  LOOM_TEST_LOW_VECTOR_SHUFFLE_SOURCE_GUARD,
  LOOM_TEST_LOW_VECTOR_SHUFFLE_RESULT_GUARD,
  LOOM_TEST_LOW_VECTOR_LHS_GUARD,
  LOOM_TEST_LOW_VECTOR_LHS_DIM0_MULTIPLE_GUARD,
  LOOM_TEST_LOW_VECTOR_RHS_GUARD,
  LOOM_TEST_LOW_VECTOR_RHS_UNIT_COUNT_GUARD,
  LOOM_TEST_LOW_VECTOR_RESULT_GUARD,
  LOOM_TEST_LOW_VECTOR_RESULT_UNIT_COUNT_GUARD,
  LOOM_TEST_LOW_INDEX_LHS_GUARD,
  LOOM_TEST_LOW_INDEX_RHS_GUARD,
  LOOM_TEST_LOW_INDEX_ACC_GUARD,
  LOOM_TEST_LOW_INDEX_ACC_MATERIALIZE_GUARD,
  LOOM_TEST_LOW_INDEX_RESULT_GUARD,
  LOOM_TEST_LOW_INDEX_CONST_VALUE_GUARD,
  LOOM_TEST_LOW_INDEX_CONST_RESULT_GUARD,
};

static const loom_low_lower_guard_t kTestLowGuards[] = {
    {
        .kind = LOOM_LOW_LOWER_GUARD_ATTR_KIND,
        .attr_index = 0,
        .diagnostic_index = LOOM_TEST_LOW_DIAGNOSTIC_I64_ATTR,
        .attr_kind = LOOM_ATTR_I64,
    },
    {
        .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
        .value_ref_index = LOOM_TEST_LOW_RESULT0,
        .type_pattern_index = LOOM_TEST_LOW_TYPE_I32,
        .diagnostic_index = LOOM_TEST_LOW_DIAGNOSTIC_I32,
    },
    {
        .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
        .value_ref_index = LOOM_TEST_LOW_OPERAND0,
        .type_pattern_index = LOOM_TEST_LOW_TYPE_I32,
        .diagnostic_index = LOOM_TEST_LOW_DIAGNOSTIC_I32,
    },
    {
        .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
        .value_ref_index = LOOM_TEST_LOW_OPERAND1,
        .type_pattern_index = LOOM_TEST_LOW_TYPE_I32,
        .diagnostic_index = LOOM_TEST_LOW_DIAGNOSTIC_I32,
    },
    {
        .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
        .value_ref_index = LOOM_TEST_LOW_RESULT0,
        .type_pattern_index = LOOM_TEST_LOW_TYPE_I32,
        .diagnostic_index = LOOM_TEST_LOW_DIAGNOSTIC_I32,
    },
    {
        .kind = LOOM_LOW_LOWER_GUARD_ATTR_KIND,
        .attr_index = 0,
        .diagnostic_index = LOOM_TEST_LOW_DIAGNOSTIC_STATIC_LANE,
        .attr_kind = LOOM_ATTR_I64_ARRAY,
    },
    {
        .kind = LOOM_LOW_LOWER_GUARD_ATTR_I64_ARRAY_COUNT_EQ,
        .attr_index = 0,
        .diagnostic_index = LOOM_TEST_LOW_DIAGNOSTIC_STATIC_LANE,
        .u64 = 1,
    },
    {
        .kind = LOOM_LOW_LOWER_GUARD_ATTR_I64_ARRAY_ELEMENT_RANGE,
        .attr_index = 0,
        .diagnostic_index = LOOM_TEST_LOW_DIAGNOSTIC_STATIC_LANE,
        .u64 = 0,
        .minimum_i64 = 0,
        .maximum_i64 = 3,
    },
    {
        .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
        .value_ref_index = LOOM_TEST_LOW_OPERAND0,
        .type_pattern_index = LOOM_TEST_LOW_TYPE_V4I32,
        .diagnostic_index = LOOM_TEST_LOW_DIAGNOSTIC_V4I32,
    },
    {
        .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
        .value_ref_index = LOOM_TEST_LOW_RESULT0,
        .type_pattern_index = LOOM_TEST_LOW_TYPE_I32,
        .diagnostic_index = LOOM_TEST_LOW_DIAGNOSTIC_I32,
    },
    {
        .kind = LOOM_LOW_LOWER_GUARD_ATTR_KIND,
        .attr_index = 0,
        .diagnostic_index = LOOM_TEST_LOW_DIAGNOSTIC_SHUFFLE_LANES,
        .attr_kind = LOOM_ATTR_I64_ARRAY,
    },
    {
        .kind = LOOM_LOW_LOWER_GUARD_ATTR_I64_ARRAY_COUNT_EQ,
        .attr_index = 0,
        .diagnostic_index = LOOM_TEST_LOW_DIAGNOSTIC_SHUFFLE_LANES,
        .u64 = 4,
    },
    {
        .kind = LOOM_LOW_LOWER_GUARD_ATTR_I64_ARRAY_ELEMENTS_RANGE,
        .attr_index = 0,
        .diagnostic_index = LOOM_TEST_LOW_DIAGNOSTIC_SHUFFLE_LANES,
        .minimum_i64 = 0,
        .maximum_i64 = 3,
    },
    {
        .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
        .value_ref_index = LOOM_TEST_LOW_OPERAND0,
        .type_pattern_index = LOOM_TEST_LOW_TYPE_V4I32,
        .diagnostic_index = LOOM_TEST_LOW_DIAGNOSTIC_V4I32,
    },
    {
        .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
        .value_ref_index = LOOM_TEST_LOW_RESULT0,
        .type_pattern_index = LOOM_TEST_LOW_TYPE_V4I32,
        .diagnostic_index = LOOM_TEST_LOW_DIAGNOSTIC_V4I32,
    },
    {
        .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
        .value_ref_index = LOOM_TEST_LOW_OPERAND0,
        .type_pattern_index = LOOM_TEST_LOW_TYPE_V4I32,
        .diagnostic_index = LOOM_TEST_LOW_DIAGNOSTIC_V4I32,
    },
    {
        .kind = LOOM_LOW_LOWER_GUARD_VALUE_STATIC_DIM0_MULTIPLE,
        .value_ref_index = LOOM_TEST_LOW_OPERAND0,
        .diagnostic_index = LOOM_TEST_LOW_DIAGNOSTIC_V4I32,
        .u64 = 4,
    },
    {
        .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
        .value_ref_index = LOOM_TEST_LOW_OPERAND1,
        .type_pattern_index = LOOM_TEST_LOW_TYPE_V4I32,
        .diagnostic_index = LOOM_TEST_LOW_DIAGNOSTIC_V4I32,
    },
    {
        .kind = LOOM_LOW_LOWER_GUARD_LOW_VALUE_REGISTER_UNIT_COUNT_EQ,
        .value_ref_index = LOOM_TEST_LOW_OPERAND0,
        .other_value_ref_index = LOOM_TEST_LOW_OPERAND1,
        .diagnostic_index = LOOM_TEST_LOW_DIAGNOSTIC_V4I32,
    },
    {
        .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
        .value_ref_index = LOOM_TEST_LOW_RESULT0,
        .type_pattern_index = LOOM_TEST_LOW_TYPE_V4I32,
        .diagnostic_index = LOOM_TEST_LOW_DIAGNOSTIC_V4I32,
    },
    {
        .kind = LOOM_LOW_LOWER_GUARD_LOW_VALUE_REGISTER_UNIT_COUNT_EQ,
        .value_ref_index = LOOM_TEST_LOW_OPERAND0,
        .other_value_ref_index = LOOM_TEST_LOW_RESULT0,
        .diagnostic_index = LOOM_TEST_LOW_DIAGNOSTIC_V4I32,
    },
    {
        .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
        .value_ref_index = LOOM_TEST_LOW_OPERAND0,
        .type_pattern_index = LOOM_TEST_LOW_TYPE_INDEX,
        .diagnostic_index = LOOM_TEST_LOW_DIAGNOSTIC_INDEX,
    },
    {
        .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
        .value_ref_index = LOOM_TEST_LOW_OPERAND1,
        .type_pattern_index = LOOM_TEST_LOW_TYPE_INDEX,
        .diagnostic_index = LOOM_TEST_LOW_DIAGNOSTIC_INDEX,
    },
    {
        .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
        .value_ref_index = LOOM_TEST_LOW_OPERAND2,
        .type_pattern_index = LOOM_TEST_LOW_TYPE_INDEX,
        .diagnostic_index = LOOM_TEST_LOW_DIAGNOSTIC_INDEX,
    },
    {
        .kind = LOOM_LOW_LOWER_GUARD_VALUE_MATERIALIZABLE,
        .value_ref_index = LOOM_TEST_LOW_MATERIALIZED_OPERAND2,
        .diagnostic_index = LOOM_TEST_LOW_DIAGNOSTIC_MATERIALIZED,
    },
    {
        .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
        .value_ref_index = LOOM_TEST_LOW_RESULT0,
        .type_pattern_index = LOOM_TEST_LOW_TYPE_INDEX,
        .diagnostic_index = LOOM_TEST_LOW_DIAGNOSTIC_INDEX,
    },
    {
        .kind = LOOM_LOW_LOWER_GUARD_ATTR_KIND,
        .attr_index = 0,
        .diagnostic_index = LOOM_TEST_LOW_DIAGNOSTIC_I64_ATTR,
        .attr_kind = LOOM_ATTR_I64,
    },
    {
        .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
        .value_ref_index = LOOM_TEST_LOW_RESULT0,
        .type_pattern_index = LOOM_TEST_LOW_TYPE_INDEX,
        .diagnostic_index = LOOM_TEST_LOW_DIAGNOSTIC_INDEX,
    },
};

enum {
  LOOM_TEST_LOW_EMIT_CONST_I32,
  LOOM_TEST_LOW_EMIT_ADD_I32,
  LOOM_TEST_LOW_EMIT_TIED_I32,
  LOOM_TEST_LOW_EMIT_PROJECT_LANE_I32,
  LOOM_TEST_LOW_EMIT_SHUFFLE_V4I32,
  LOOM_TEST_LOW_EMIT_ADD_V4I32,
  LOOM_TEST_LOW_EMIT_SWAPPED_ADD_V4I32,
  LOOM_TEST_LOW_EMIT_INDEX_PRODUCT,
  LOOM_TEST_LOW_EMIT_INDEX_ADDEND,
};

static const loom_tied_result_t kTestLowTiedResults[] = {
    {
        .result_index = 0,
        .operand_index = 0,
        .has_type_change = false,
    },
};

static const loom_low_lower_emit_t kTestLowEmits[] = {
    {
        .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_CONST,
        .descriptor_id = TEST_LOW_CORE_DESCRIPTOR_ID_TEST_CONST_I32,
        .result_ref_start = LOOM_TEST_LOW_RESULT0,
        .result_ref_count = 1,
        .attr_copy_start = LOOM_TEST_LOW_ATTR_COPY_I64,
        .attr_copy_count = 1,
    },
    {
        .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
        .descriptor_id = TEST_LOW_CORE_DESCRIPTOR_ID_TEST_ADD_I32,
        .operand_ref_start = LOOM_TEST_LOW_OPERAND0,
        .operand_ref_count = 2,
        .result_ref_start = LOOM_TEST_LOW_RESULT0,
        .result_ref_count = 1,
    },
    {
        .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
        .descriptor_id = TEST_LOW_CORE_DESCRIPTOR_ID_TEST_TIED_ANY,
        .operand_ref_start = LOOM_TEST_LOW_OPERAND0,
        .operand_ref_count = 1,
        .copy_operand_mask = 0x1,
        .result_ref_start = LOOM_TEST_LOW_RESULT0,
        .result_ref_count = 1,
        .tied_result_start = 0,
        .tied_result_count = 1,
    },
    {
        .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_CONST,
        .descriptor_id = TEST_LOW_CORE_DESCRIPTOR_ID_TEST_CONST_I32,
        .result_ref_start = LOOM_TEST_LOW_RESULT0,
        .result_ref_count = 1,
        .attr_copy_start = LOOM_TEST_LOW_ATTR_COPY_STATIC_LANE0,
        .attr_copy_count = 1,
    },
    {
        .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
        .descriptor_id = TEST_LOW_CORE_DESCRIPTOR_ID_TEST_SHUFFLE_V4I32,
        .operand_ref_start = LOOM_TEST_LOW_OPERAND0,
        .operand_ref_count = 1,
        .result_ref_start = LOOM_TEST_LOW_RESULT0,
        .result_ref_count = 1,
        .attr_copy_start = LOOM_TEST_LOW_ATTR_COPY_SHUFFLE_CONTROL,
        .attr_copy_count = 1,
    },
    {
        .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE,
        .descriptor_id = TEST_LOW_CORE_DESCRIPTOR_ID_TEST_ADD_I32,
        .operand_ref_start = LOOM_TEST_LOW_OPERAND0,
        .operand_ref_count = 2,
        .result_ref_start = LOOM_TEST_LOW_RESULT0,
        .result_ref_count = 1,
    },
    {
        .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE,
        .flags = LOOM_LOW_LOWER_EMIT_FLAG_SWAP_OPERANDS_0_1,
        .descriptor_id = TEST_LOW_CORE_DESCRIPTOR_ID_TEST_ADD_I32,
        .operand_ref_start = LOOM_TEST_LOW_OPERAND0,
        .operand_ref_count = 2,
        .result_ref_start = LOOM_TEST_LOW_RESULT0,
        .result_ref_count = 1,
    },
    {
        .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
        .flags = LOOM_LOW_LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS,
        .descriptor_id = TEST_LOW_CORE_DESCRIPTOR_ID_TEST_ADD_I32,
        .operand_ref_start = LOOM_TEST_LOW_OPERAND0,
        .operand_ref_count = 2,
        .result_ref_start = LOOM_TEST_LOW_RESULT0,
        .result_ref_count = 1,
        .result_bind_ref_start = LOOM_TEST_LOW_TEMPORARY0,
    },
    {
        .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
        .descriptor_id = TEST_LOW_CORE_DESCRIPTOR_ID_TEST_ADD_I32,
        .operand_ref_start = LOOM_TEST_LOW_TEMPORARY0,
        .operand_ref_count = 2,
        .result_ref_start = LOOM_TEST_LOW_RESULT0,
        .result_ref_count = 1,
    },
};

enum {
  LOOM_TEST_LOW_RULE_SCALAR_ADDI,
  LOOM_TEST_LOW_RULE_SCALAR_SUBI,
  LOOM_TEST_LOW_RULE_SCALAR_CONSTANT,
  LOOM_TEST_LOW_RULE_INDEX_CONSTANT,
  LOOM_TEST_LOW_RULE_VECTOR_EXTRACT,
  LOOM_TEST_LOW_RULE_VECTOR_SHUFFLE,
  LOOM_TEST_LOW_RULE_VECTOR_ADDI,
  LOOM_TEST_LOW_RULE_VECTOR_SUBI,
  LOOM_TEST_LOW_RULE_INDEX_MADD,
};

static const loom_low_lower_rule_t kTestLowRules[] = {
    [LOOM_TEST_LOW_RULE_SCALAR_ADDI] =
        {
            .source_op_kind = LOOM_OP_SCALAR_ADDI,
            .guard_start = LOOM_TEST_LOW_SCALAR_LHS_GUARD,
            .guard_count = 3,
            .emit_start = LOOM_TEST_LOW_EMIT_ADD_I32,
            .emit_count = 1,
        },
    [LOOM_TEST_LOW_RULE_SCALAR_SUBI] =
        {
            .source_op_kind = LOOM_OP_SCALAR_SUBI,
            .guard_start = LOOM_TEST_LOW_SCALAR_LHS_GUARD,
            .guard_count = 3,
            .emit_start = LOOM_TEST_LOW_EMIT_TIED_I32,
            .emit_count = 1,
        },
    [LOOM_TEST_LOW_RULE_SCALAR_CONSTANT] =
        {
            .source_op_kind = LOOM_OP_SCALAR_CONSTANT,
            .guard_start = LOOM_TEST_LOW_CONST_VALUE_GUARD,
            .guard_count = 2,
            .emit_start = LOOM_TEST_LOW_EMIT_CONST_I32,
            .emit_count = 1,
        },
    [LOOM_TEST_LOW_RULE_INDEX_CONSTANT] =
        {
            .source_op_kind = LOOM_OP_INDEX_CONSTANT,
            .guard_start = LOOM_TEST_LOW_INDEX_CONST_VALUE_GUARD,
            .guard_count = 2,
            .emit_start = LOOM_TEST_LOW_EMIT_CONST_I32,
            .emit_count = 1,
        },
    [LOOM_TEST_LOW_RULE_VECTOR_EXTRACT] =
        {
            .source_op_kind = LOOM_OP_VECTOR_EXTRACT,
            .guard_start =
                LOOM_TEST_LOW_VECTOR_EXTRACT_STATIC_INDICES_KIND_GUARD,
            .guard_count = 5,
            .emit_start = LOOM_TEST_LOW_EMIT_PROJECT_LANE_I32,
            .emit_count = 1,
        },
    [LOOM_TEST_LOW_RULE_VECTOR_SHUFFLE] =
        {
            .source_op_kind = LOOM_OP_VECTOR_SHUFFLE,
            .guard_start = LOOM_TEST_LOW_VECTOR_SHUFFLE_SOURCE_LANES_KIND_GUARD,
            .guard_count = 5,
            .emit_start = LOOM_TEST_LOW_EMIT_SHUFFLE_V4I32,
            .emit_count = 1,
        },
    [LOOM_TEST_LOW_RULE_VECTOR_ADDI] =
        {
            .source_op_kind = LOOM_OP_VECTOR_ADDI,
            .guard_start = LOOM_TEST_LOW_VECTOR_LHS_GUARD,
            .guard_count = 6,
            .emit_start = LOOM_TEST_LOW_EMIT_ADD_V4I32,
            .emit_count = 1,
        },
    [LOOM_TEST_LOW_RULE_VECTOR_SUBI] =
        {
            .source_op_kind = LOOM_OP_VECTOR_SUBI,
            .guard_start = LOOM_TEST_LOW_VECTOR_LHS_GUARD,
            .guard_count = 6,
            .emit_start = LOOM_TEST_LOW_EMIT_SWAPPED_ADD_V4I32,
            .emit_count = 1,
        },
    [LOOM_TEST_LOW_RULE_INDEX_MADD] =
        {
            .source_op_kind = LOOM_OP_INDEX_MADD,
            .temporary_count = 1,
            .guard_start = LOOM_TEST_LOW_INDEX_LHS_GUARD,
            .guard_count = 5,
            .emit_start = LOOM_TEST_LOW_EMIT_INDEX_PRODUCT,
            .emit_count = 2,
        },
};

static const loom_low_lower_rule_span_t kTestLowSpans[] = {
    {
        .source_op_kind = LOOM_OP_SCALAR_ADDI,
        .rule_start = LOOM_TEST_LOW_RULE_SCALAR_ADDI,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_SUBI,
        .rule_start = LOOM_TEST_LOW_RULE_SCALAR_SUBI,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_CONSTANT,
        .rule_start = LOOM_TEST_LOW_RULE_SCALAR_CONSTANT,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_EXTRACT,
        .rule_start = LOOM_TEST_LOW_RULE_VECTOR_EXTRACT,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_SHUFFLE,
        .rule_start = LOOM_TEST_LOW_RULE_VECTOR_SHUFFLE,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_ADDI,
        .rule_start = LOOM_TEST_LOW_RULE_VECTOR_ADDI,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_SUBI,
        .rule_start = LOOM_TEST_LOW_RULE_VECTOR_SUBI,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_INDEX_CONSTANT,
        .rule_start = LOOM_TEST_LOW_RULE_INDEX_CONSTANT,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_INDEX_MADD,
        .rule_start = LOOM_TEST_LOW_RULE_INDEX_MADD,
        .rule_count = 1,
    },
};

const loom_low_lower_rule_set_t loom_test_low_lower_rule_set = {
    .spans = kTestLowSpans,
    .span_count = IREE_ARRAYSIZE(kTestLowSpans),
    .rules = kTestLowRules,
    .rule_count = IREE_ARRAYSIZE(kTestLowRules),
    .type_patterns = kTestLowTypePatterns,
    .type_pattern_count = IREE_ARRAYSIZE(kTestLowTypePatterns),
    .value_refs = kTestLowValueRefs,
    .value_ref_count = IREE_ARRAYSIZE(kTestLowValueRefs),
    .materializers = kTestLowMaterializers,
    .materializer_count = IREE_ARRAYSIZE(kTestLowMaterializers),
    .guards = kTestLowGuards,
    .guard_count = IREE_ARRAYSIZE(kTestLowGuards),
    .attr_copies = kTestLowAttrCopies,
    .attr_copy_count = IREE_ARRAYSIZE(kTestLowAttrCopies),
    .tied_results = kTestLowTiedResults,
    .tied_result_count = IREE_ARRAYSIZE(kTestLowTiedResults),
    .emits = kTestLowEmits,
    .emit_count = IREE_ARRAYSIZE(kTestLowEmits),
    .diagnostics = kTestLowDiagnostics,
    .diagnostic_count = IREE_ARRAYSIZE(kTestLowDiagnostics),
};

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
      plan->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL;
  return supported_memory_space &&
         plan->root_value_id != LOOM_VALUE_ID_INVALID &&
         plan->element_byte_count == 4 && plan->vector_lane_count == 4 &&
         plan->vector_lane_byte_stride == 4 && plan->static_byte_offset == 0 &&
         plan->dynamic_index == LOOM_VALUE_ID_INVALID &&
         plan->cache_policy.build_flags == 0;
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

  loom_low_source_memory_access_plan_t* plan_data = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sizeof(*plan_data), (void**)&plan_data));
  *plan_data = access;
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

static iree_status_t loom_test_low_emit_buffer_view(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t low_buffer = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_buffer_view_buffer(source_op), &low_buffer));
  return loom_low_lower_bind_value(context, loom_buffer_view_result(source_op),
                                   low_buffer);
}

static iree_status_t loom_test_low_emit_vector_load(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_access_plan_t* access) {
  IREE_ASSERT_ARGUMENT(access);
  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, access->view_value_id, &low_resource));
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(
      context, source_op, loom_vector_load_result(source_op), &result_type));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_descriptor_op(
      context, TEST_LOW_CORE_DESCRIPTOR_ID_TEST_LOAD_V4I32, &low_resource, 1,
      loom_named_attr_slice_empty(), &result_type, 1, NULL, 0,
      source_op->location, &low_op));
  return loom_low_lower_bind_value(
      context, loom_vector_load_result(source_op),
      loom_value_slice_get(loom_low_op_results(low_op), 0));
}

static iree_status_t loom_test_low_emit_vector_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_access_plan_t* access) {
  IREE_ASSERT_ARGUMENT(access);
  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, access->view_value_id, &low_resource));
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_store_value(source_op), &low_value));
  loom_value_id_t operands[] = {
      low_resource,
      low_value,
  };
  loom_op_t* low_op = NULL;
  return loom_low_lower_emit_descriptor_op(
      context, TEST_LOW_CORE_DESCRIPTOR_ID_TEST_STORE_V4I32, operands,
      IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(),
      /*result_types=*/NULL, /*result_count=*/0, NULL, 0, source_op->location,
      &low_op);
}

static iree_status_t loom_test_low_emit_op(void* user_data,
                                           loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_low_lower_plan_t plan) {
  (void)user_data;
  switch (plan.id) {
    case LOOM_OP_BUFFER_VIEW:
      return loom_test_low_emit_buffer_view(context, source_op);
    case LOOM_OP_VECTOR_LOAD:
      return loom_test_low_emit_vector_load(
          context, source_op,
          (const loom_low_source_memory_access_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_STORE:
      return loom_test_low_emit_vector_store(
          context, source_op,
          (const loom_low_source_memory_access_plan_t*)plan.target_data);
    default:
      IREE_ASSERT_UNREACHABLE();
      return iree_ok_status();
  }
}

static const loom_low_lower_rule_set_t* const kTestLowRuleSets[] = {
    &loom_test_low_lower_rule_set,
};

static const loom_low_lower_policy_t kTestLowLowerPolicy = {
    .name = IREE_SVL("test-low-lower-policy"),
    .map_type = {.fn = loom_test_low_lower_map_type, .user_data = NULL},
    .map_argument = {.fn = loom_test_low_lower_map_argument, .user_data = NULL},
    .rule_sets =
        {
            .count = IREE_ARRAYSIZE(kTestLowRuleSets),
            .values = kTestLowRuleSets,
        },
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
