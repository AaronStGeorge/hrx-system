// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/contract_storage.h"
#include "loom/format/text/parser.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/func/ops.h"
#include "loom/target/arch/amdgpu/matrix_contract.h"
#include "loom/target/arch/amdgpu/matrix_contract_projection.h"
#include "loom/target/arch/amdgpu/matrix_facts.h"
#include "loom/testing/context.h"
#include "loom/util/fact_table.h"
#include "loom/verify/verify.h"

namespace {

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

loom_amdgpu_matrix_payload_shape_t PayloadShape(
    loom_amdgpu_matrix_numeric_type_t numeric_type) {
  loom_amdgpu_matrix_payload_shape_t payload = {};
  payload.numeric_type = numeric_type;
  return payload;
}

enum class ViewPayloadKind {
  kPlainElement,
  kMatrixStorageSchema,
  kUnsupportedStorageSchema,
};

struct ModuleDeleter {
  void operator()(loom_module_t* module) const { loom_module_free(module); }
};

using ModulePtr = std::unique_ptr<loom_module_t, ModuleDeleter>;

struct ViewPayload {
  // Origin of the target payload interpretation.
  ViewPayloadKind kind = ViewPayloadKind::kPlainElement;
  // Generic contract operand facts used before target projection.
  loom_contract_operand_t operand = {};
  // Generic scale operand shape required by encoded matrix storage, if any.
  loom_contract_scale_kind_t scale_kind = LOOM_CONTRACT_SCALE_NONE;
  // Generic capability classes proven available from the view type/facts.
  loom_contract_capability_flags_t available_flags = 0;
  // Target-independent storage-schema facts recovered from the view type.
  loom_value_fact_storage_schema_t storage_schema = {};
};

bool PlainElementPayload(loom_scalar_type_t element_type,
                         loom_contract_operand_t* out_operand) {
  *out_operand = {};
  loom_contract_numeric_type_t numeric_type = LOOM_CONTRACT_NUMERIC_UNKNOWN;
  if (!loom_contract_numeric_type_from_scalar(element_type, false,
                                              &numeric_type)) {
    return false;
  }
  out_operand->numeric_type = numeric_type;
  return true;
}

bool QueryViewPayload(const loom_value_fact_table_t* fact_table,
                      const loom_module_t* module, loom_type_t view_type,
                      ViewPayload* out_payload) {
  *out_payload = {};
  if (!loom_type_is_view(view_type)) {
    return false;
  }

  loom_value_fact_storage_schema_t storage_schema = {};
  if (loom_encoding_query_type_storage_schema(&fact_table->context, module,
                                              view_type, &storage_schema)) {
    out_payload->kind = ViewPayloadKind::kUnsupportedStorageSchema;
    out_payload->storage_schema = storage_schema;
    if (!loom_contract_operand_from_storage_schema(
            LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN, storage_schema,
            &out_payload->operand)) {
      return true;
    }
    if (!loom_contract_scale_kind_from_storage_schema(
            storage_schema, &out_payload->scale_kind)) {
      return false;
    }
    out_payload->kind = ViewPayloadKind::kMatrixStorageSchema;
    out_payload->available_flags =
        loom_contract_capability_flags_from_storage_schema(storage_schema);
    return true;
  }

  out_payload->kind = ViewPayloadKind::kPlainElement;
  return PlainElementPayload(loom_type_element_type(view_type),
                             &out_payload->operand);
}

loom_contract_operand_t OperandWithRole(loom_contract_operand_role_t role,
                                        loom_contract_operand_t operand) {
  operand.role = role;
  return operand;
}

loom_contract_request_t MatrixContractRequest(
    int64_t m, int64_t n, int64_t k, loom_contract_operand_t lhs,
    loom_contract_operand_t rhs,
    loom_contract_numeric_type_t accumulator_numeric_type,
    loom_contract_numeric_type_t result_numeric_type,
    loom_contract_scale_kind_t scale_kind,
    loom_contract_capability_flags_t available_flags,
    loom_contract_capability_flags_t required_flags) {
  loom_contract_request_t request = {};
  loom_contract_request_initialize(&request);
  request.kind = LOOM_CONTRACT_KIND_MATRIX_MULTIPLY;
  request.arithmetic = LOOM_CONTRACT_ARITHMETIC_MIXED_DOT;
  request.shape = {
      .m = m,
      .n = n,
      .k = k,
  };
  request.k_group_size = 1;
  request.lhs = OperandWithRole(LOOM_CONTRACT_OPERAND_ROLE_LHS, lhs);
  request.rhs = OperandWithRole(LOOM_CONTRACT_OPERAND_ROLE_RHS, rhs);
  request.accumulator = {
      .role = LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR,
      .numeric_type = accumulator_numeric_type,
  };
  request.result = {
      .role = LOOM_CONTRACT_OPERAND_ROLE_RESULT,
      .numeric_type = result_numeric_type,
  };
  request.fragment = {
      .atom_bits = LOOM_CONTRACT_FRAGMENT_SUBGROUP_LANE,
      .subgroup_size = 64,
  };
  request.capability_class = LOOM_CONTRACT_CAPABILITY_CLASS_GPU_MATRIX;
  request.available_capability_flags = available_flags;
  request.required_capability_flags = required_flags;
  request.scale_kind = scale_kind;
  request.policy = LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED;
  return request;
}

class MatrixContractFixtureTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(loom_testing_context_initialize_all(iree_allocator_system(),
                                                       &context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_status_t ParseAndVerify(const char* source, loom_module_t** out_module) {
    *out_module = nullptr;
    loom_text_parse_options_t parse_options = {};
    loom_module_t* module = nullptr;
    IREE_RETURN_IF_ERROR(loom_text_parse(
        iree_make_cstring_view(source), IREE_SV("matrix_fixture.loom"),
        &context_, &block_pool_, &parse_options, &module));
    if (!module) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "parser did not produce a module");
    }

    loom_verify_options_t verify_options = {};
    loom_verify_result_t verify_result = {};
    iree_status_t status =
        loom_verify_module(module, &verify_options, &verify_result);
    if (iree_status_is_ok(status) && verify_result.error_count != 0) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "module verification produced %u errors",
                                verify_result.error_count);
    }
    if (!iree_status_is_ok(status)) {
      loom_module_free(module);
      return status;
    }
    *out_module = module;
    return iree_ok_status();
  }

  loom_func_like_t FirstFunctionWithBody(const loom_module_t* module) {
    loom_symbol_t* symbol = nullptr;
    loom_module_for_each_symbol(module, symbol) {
      loom_func_like_t function =
          loom_func_like_cast(module, symbol->defining_op);
      if (loom_func_like_body(function)) {
        return function;
      }
    }
    return loom_func_like_t{};
  }

  std::vector<loom_type_t> CollectBufferViewTypes(const loom_module_t* module,
                                                  loom_func_like_t function) {
    std::vector<loom_type_t> view_types;
    loom_region_t* body = loom_func_like_body(function);
    if (!body) {
      return view_types;
    }

    loom_block_t* block = nullptr;
    loom_region_for_each_block(body, block) {
      loom_op_t* op = nullptr;
      loom_block_for_each_op(block, op) {
        if (!loom_buffer_view_isa(op)) {
          continue;
        }
        view_types.push_back(
            loom_module_value_type(module, loom_buffer_view_result(op)));
      }
    }
    return view_types;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(MatrixContractFixtureTest, ParsedI8ViewsSelectGfx908Mfma) {
  static const char kSource[] = R"(
func.def @i8_mfma(%buffer: buffer, %base: offset) {
  %layout = encoding.layout.dense : encoding<layout>
  %lhs = buffer.view %buffer[%base] : buffer -> view<16x16xi8, %layout>
  %rhs = buffer.view %buffer[%base] : buffer -> view<16x16xi8, %layout>
  test.use %lhs, %rhs : view<16x16xi8, %layout>, view<16x16xi8, %layout>
  func.return
}
)";

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseAndVerify(kSource, &module));
  ModulePtr module_ptr(module);
  loom_func_like_t function = FirstFunctionWithBody(module_ptr.get());
  ASSERT_TRUE(loom_func_like_isa(function));

  loom_value_fact_table_t fact_table = {};
  IREE_ASSERT_OK(
      loom_value_fact_table_initialize(&fact_table, &module_ptr->arena, 0));
  IREE_ASSERT_OK(
      loom_value_fact_table_compute(&fact_table, module_ptr.get(), function));

  std::vector<loom_type_t> view_types =
      CollectBufferViewTypes(module_ptr.get(), function);
  ASSERT_EQ(view_types.size(), 2u);

  ViewPayload lhs = {};
  ViewPayload rhs = {};
  ASSERT_TRUE(
      QueryViewPayload(&fact_table, module_ptr.get(), view_types[0], &lhs));
  ASSERT_TRUE(
      QueryViewPayload(&fact_table, module_ptr.get(), view_types[1], &rhs));
  EXPECT_EQ(lhs.kind, ViewPayloadKind::kPlainElement);
  EXPECT_EQ(rhs.kind, ViewPayloadKind::kPlainElement);
  EXPECT_EQ(lhs.operand.numeric_type, LOOM_CONTRACT_NUMERIC_I8);
  EXPECT_EQ(rhs.operand.numeric_type, LOOM_CONTRACT_NUMERIC_I8);

  loom_amdgpu_matrix_feature_bits_t feature_bits = 0;
  IREE_ASSERT_OK(loom_amdgpu_matrix_feature_bits_for_processor(
      IREE_SV("gfx908"), &feature_bits));
  loom_contract_request_t contract = MatrixContractRequest(
      16, 16, 16, lhs.operand, rhs.operand, LOOM_CONTRACT_NUMERIC_I32,
      LOOM_CONTRACT_NUMERIC_I32, LOOM_CONTRACT_SCALE_NONE, 0, 0);
  loom_contract_diagnostic_t contract_diagnostic = {};
  ASSERT_TRUE(loom_contract_request_validate(&contract, &contract_diagnostic));
  loom_amdgpu_matrix_contract_match_request_t request = {};
  ASSERT_TRUE(loom_amdgpu_matrix_contract_match_request_from_contract(
      &contract, feature_bits, 64, &request, nullptr));
  request.family = LOOM_AMDGPU_MATRIX_FAMILY_MFMA;

  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      loom_amdgpu_matrix_contract_select(&request, nullptr);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(ToString(descriptor->name), "mfma.i32.16x16x16.i8");
}

TEST_F(MatrixContractFixtureTest, ParsedScaledFp6StorageFactsSelectGfx950Mfma) {
  static const char kSource[] = R"(
func.def @scaled_fp6_mfma(%buffer: buffer, %base: offset) {
  %layout = encoding.layout.dense : encoding<layout>
  %lhs_schema = encoding.define #amdgpu_matrix_operand<format="fp6", packed_elements=32, packed_registers=6, scale="scale32", scale_conversion="convergent", scale_format="none", scale_placement="explicit", zero_scale_fallback=true> : encoding<schema>
  %rhs_schema = encoding.define #amdgpu_matrix_operand<format="bf6", packed_elements=32, packed_registers=6, scale="scale32", scale_conversion="convergent", scale_format="none", scale_placement="explicit", zero_scale_fallback=true> : encoding<schema>
  %lhs_storage = encoding.define #physical_storage {layout = %layout : encoding<layout>, schema = %lhs_schema : encoding<schema>} : encoding<storage>
  %rhs_storage = encoding.define #physical_storage {layout = %layout : encoding<layout>, schema = %rhs_schema : encoding<schema>} : encoding<storage>
  %lhs = buffer.view %buffer[%base] : buffer -> view<16x16xi8, %lhs_storage>
  %rhs = buffer.view %buffer[%base] : buffer -> view<16x16xi8, %rhs_storage>
  test.use %lhs, %rhs : view<16x16xi8, %lhs_storage>, view<16x16xi8, %rhs_storage>
  func.return
}
)";

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseAndVerify(kSource, &module));
  ModulePtr module_ptr(module);
  loom_func_like_t function = FirstFunctionWithBody(module_ptr.get());
  ASSERT_TRUE(loom_func_like_isa(function));

  loom_value_fact_table_t fact_table = {};
  IREE_ASSERT_OK(
      loom_value_fact_table_initialize(&fact_table, &module_ptr->arena, 0));
  IREE_ASSERT_OK(
      loom_value_fact_table_compute(&fact_table, module_ptr.get(), function));

  std::vector<loom_type_t> view_types =
      CollectBufferViewTypes(module_ptr.get(), function);
  ASSERT_EQ(view_types.size(), 2u);

  loom_value_facts_t stride_storage[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {};
  loom_value_fact_address_layout_t layout = {};
  ASSERT_TRUE(loom_encoding_query_type_address_layout(
      &fact_table.context, module_ptr.get(), view_types[0], stride_storage,
      IREE_ARRAYSIZE(stride_storage), &layout));
  EXPECT_EQ(layout.kind, LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE);

  ViewPayload lhs = {};
  ViewPayload rhs = {};
  ASSERT_TRUE(
      QueryViewPayload(&fact_table, module_ptr.get(), view_types[0], &lhs));
  ASSERT_TRUE(
      QueryViewPayload(&fact_table, module_ptr.get(), view_types[1], &rhs));
  EXPECT_EQ(lhs.kind, ViewPayloadKind::kMatrixStorageSchema);
  EXPECT_EQ(rhs.kind, ViewPayloadKind::kMatrixStorageSchema);
  EXPECT_EQ(lhs.operand.numeric_type, LOOM_CONTRACT_NUMERIC_FP6);
  EXPECT_EQ(rhs.operand.numeric_type, LOOM_CONTRACT_NUMERIC_BF6);
  EXPECT_EQ(lhs.scale_kind, LOOM_CONTRACT_SCALE_32);
  EXPECT_EQ(rhs.scale_kind, LOOM_CONTRACT_SCALE_32);

  loom_amdgpu_matrix_feature_bits_t feature_bits = 0;
  IREE_ASSERT_OK(loom_amdgpu_matrix_feature_bits_for_processor(
      IREE_SV("gfx950"), &feature_bits));
  loom_contract_capability_flags_t available_flags =
      lhs.available_flags | rhs.available_flags;
  loom_contract_capability_flags_t required_flags =
      LOOM_CONTRACT_CAPABILITY_SCALE_OPERANDS |
      LOOM_CONTRACT_CAPABILITY_FORMAT_SELECTORS;
  loom_contract_request_t contract = MatrixContractRequest(
      16, 16, 128, lhs.operand, rhs.operand, LOOM_CONTRACT_NUMERIC_F32,
      LOOM_CONTRACT_NUMERIC_F32, lhs.scale_kind, available_flags,
      required_flags);
  loom_contract_diagnostic_t contract_diagnostic = {};
  ASSERT_TRUE(loom_contract_request_validate(&contract, &contract_diagnostic));
  loom_amdgpu_matrix_contract_match_request_t request = {};
  ASSERT_TRUE(loom_amdgpu_matrix_contract_match_request_from_contract(
      &contract, feature_bits, 64, &request, nullptr));
  request.family = LOOM_AMDGPU_MATRIX_FAMILY_MFMA;

  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      loom_amdgpu_matrix_contract_select(&request, nullptr);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(ToString(descriptor->name), "mfma.scale.f32.16x16x128.f8f6f4");
}

TEST_F(MatrixContractFixtureTest, ParsedUnsupportedSchemaStaysFallbackOnly) {
  static const char kSource[] = R"(
func.def @ggml_q4_reference(%buffer: buffer, %base: offset) {
  %layout = encoding.layout.dense : encoding<layout>
  %schema = encoding.define #ggml_q4_0<block_elems=32, storage_bytes=18> : encoding<schema>
  %storage = encoding.define #physical_storage {layout = %layout : encoding<layout>, schema = %schema : encoding<schema>} : encoding<storage>
  %weights = buffer.view %buffer[%base] : buffer -> view<32xi8, %storage>
  test.use %weights : view<32xi8, %storage>
  func.return
}
)";

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseAndVerify(kSource, &module));
  ModulePtr module_ptr(module);
  loom_func_like_t function = FirstFunctionWithBody(module_ptr.get());
  ASSERT_TRUE(loom_func_like_isa(function));

  loom_value_fact_table_t fact_table = {};
  IREE_ASSERT_OK(
      loom_value_fact_table_initialize(&fact_table, &module_ptr->arena, 0));
  IREE_ASSERT_OK(
      loom_value_fact_table_compute(&fact_table, module_ptr.get(), function));

  std::vector<loom_type_t> view_types =
      CollectBufferViewTypes(module_ptr.get(), function);
  ASSERT_EQ(view_types.size(), 1u);

  loom_value_facts_t stride_storage[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {};
  loom_value_fact_address_layout_t layout = {};
  ASSERT_TRUE(loom_encoding_query_type_address_layout(
      &fact_table.context, module_ptr.get(), view_types[0], stride_storage,
      IREE_ARRAYSIZE(stride_storage), &layout));
  EXPECT_EQ(layout.kind, LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE);

  ViewPayload weights = {};
  ASSERT_TRUE(
      QueryViewPayload(&fact_table, module_ptr.get(), view_types[0], &weights));
  EXPECT_EQ(weights.kind, ViewPayloadKind::kUnsupportedStorageSchema);
  EXPECT_NE(weights.storage_schema.static_spec_encoding_id, 0u);
  EXPECT_EQ(weights.storage_schema.matrix.format,
            LOOM_VALUE_FACT_MATRIX_FORMAT_UNKNOWN);
  EXPECT_EQ(weights.operand.numeric_type, LOOM_CONTRACT_NUMERIC_UNKNOWN);
}

TEST_F(MatrixContractFixtureTest, ContractRejectionReasonsStayStructural) {
  loom_value_fact_storage_schema_t schema = {};
  schema.matrix.format = LOOM_VALUE_FACT_MATRIX_FORMAT_FP6;
  schema.matrix.scale_kind = LOOM_VALUE_FACT_MATRIX_SCALE_32;
  schema.matrix.scale_format = LOOM_VALUE_FACT_MATRIX_SCALE_FORMAT_NONE;
  schema.matrix.scale_placement =
      LOOM_VALUE_FACT_MATRIX_SCALE_PLACEMENT_EXPLICIT;
  schema.matrix.scale_conversion =
      LOOM_VALUE_FACT_MATRIX_SCALE_CONVERSION_CONVERGENT;
  schema.matrix.packed_register_count = 6;
  schema.matrix.packed_element_count = 32;
  schema.matrix.zero_scale_fallback = true;

  loom_amdgpu_matrix_payload_shape_t payload = {};
  ASSERT_TRUE(loom_amdgpu_matrix_payload_from_storage_schema(schema, &payload));

  loom_amdgpu_matrix_contract_match_request_t request = {};
  request.family = LOOM_AMDGPU_MATRIX_FAMILY_MFMA;
  request.tile_shape.result_row_count = 16;
  request.tile_shape.result_column_count = 16;
  request.tile_shape.reduction_count = 128;
  request.lhs_payload = payload;
  request.rhs_payload = payload;
  request.accumulator_payload = PayloadShape(LOOM_AMDGPU_MATRIX_NUMERIC_F32);
  request.result_payload = PayloadShape(LOOM_AMDGPU_MATRIX_NUMERIC_F32);
  request.scale_kind = LOOM_AMDGPU_MATRIX_SCALE_32;
  request.wave_size = 64;
  request.available_flags =
      loom_amdgpu_matrix_available_flags_from_storage_schema(schema);
  request.required_flags = LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED |
                           LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS;

  loom_amdgpu_matrix_contract_match_diagnostic_t diagnostic = {};
  EXPECT_EQ(loom_amdgpu_matrix_contract_select(&request, &diagnostic), nullptr);
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_FEATURES);

  IREE_ASSERT_OK(loom_amdgpu_matrix_feature_bits_for_processor(
      IREE_SV("gfx950"), &request.feature_bits));
  request.tile_shape.result_row_count = 17;
  diagnostic = {};
  EXPECT_EQ(loom_amdgpu_matrix_contract_select(&request, &diagnostic), nullptr);
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_TILE_SHAPE);

  request.tile_shape.result_row_count = 16;
  request.wave_size = 48;
  diagnostic = {};
  EXPECT_EQ(loom_amdgpu_matrix_contract_select(&request, &diagnostic), nullptr);
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_WAVE_SIZE);

  request.wave_size = 64;
  request.available_flags &= ~LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS;
  diagnostic = {};
  EXPECT_EQ(loom_amdgpu_matrix_contract_select(&request, &diagnostic), nullptr);
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_MATRIX_FORMATS);
}

}  // namespace
