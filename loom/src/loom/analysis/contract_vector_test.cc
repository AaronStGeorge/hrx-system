// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/contract_vector.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/testing/module_ptr.h"
#include "loom/verify/verify.h"

namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class ContractVectorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* func_vtables =
        loom_func_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_FUNC,
                                                 func_vtables, count));
    const loom_op_vtable_t* const* vector_vtables =
        loom_vector_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_VECTOR,
                                                 vector_vtables, count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
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
        iree_make_cstring_view(source), IREE_SV("contract_vector.loom"),
        &context_, &block_pool_, &parse_options, &module));
    if (module == nullptr) {
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

  const loom_op_t* FirstVectorDotOp(const loom_module_t* module) {
    loom_symbol_t* symbol = nullptr;
    loom_module_for_each_symbol(module, symbol) {
      const loom_func_like_t function =
          loom_func_like_cast(module, symbol->defining_op);
      if (!loom_func_like_isa(function)) {
        continue;
      }
      loom_region_t* body = loom_func_like_body(function);
      if (body == nullptr) {
        continue;
      }
      loom_block_t* block = nullptr;
      loom_region_for_each_block(body, block) {
        loom_op_t* op = nullptr;
        loom_block_for_each_op(block, op) {
          if (loom_vector_dot2f_isa(op) || loom_vector_dot4i_isa(op) ||
              loom_vector_dot8i4_isa(op)) {
            return op;
          }
        }
      }
    }
    return nullptr;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(ContractVectorTest, BuildsGenericU8S8Dot4Contract) {
  static const char kSource[] = R"(
func.def @u8s8_dot4(%lhs: vector<32xi8>, %rhs: vector<32xi8>, %acc: vector<8xi32>) -> (vector<8xi32>) {
  %r = vector.dot4i<u8s8> %lhs, %rhs, %acc : vector<32xi8>, vector<32xi8>, vector<8xi32>
  func.return %r : vector<8xi32>
}
)";

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseAndVerify(kSource, &module));
  ModulePtr module_ptr(module);

  const loom_op_t* op = FirstVectorDotOp(module_ptr.get());
  ASSERT_NE(op, nullptr);
  loom_contract_request_t request = {};
  ASSERT_TRUE(loom_contract_request_from_vector_dot_op(
      module_ptr.get(), op, LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED,
      &request, nullptr));

  EXPECT_EQ(request.kind, LOOM_CONTRACT_KIND_VECTOR_DOT);
  EXPECT_EQ(request.arithmetic, LOOM_CONTRACT_ARITHMETIC_INTEGER_DOT);
  EXPECT_EQ(request.shape.m, 8);
  EXPECT_EQ(request.shape.n, 1);
  EXPECT_EQ(request.shape.k, 32);
  EXPECT_EQ(request.k_group_size, 4);
  EXPECT_EQ(request.lhs.numeric_type, LOOM_CONTRACT_NUMERIC_U8);
  EXPECT_EQ(request.rhs.numeric_type, LOOM_CONTRACT_NUMERIC_I8);
  EXPECT_EQ(request.accumulator.numeric_type, LOOM_CONTRACT_NUMERIC_I32);
  EXPECT_EQ(request.result.numeric_type, LOOM_CONTRACT_NUMERIC_I32);
  EXPECT_EQ(request.fragment.atom_bits, LOOM_CONTRACT_FRAGMENT_VECTOR_LANE);
  EXPECT_EQ(request.fragment.vector_bit_width, 256);
  EXPECT_EQ(request.fragment.source_lane_count, 32);
  EXPECT_EQ(request.fragment.result_lane_count, 8);
  EXPECT_EQ(request.capability_class,
            LOOM_CONTRACT_CAPABILITY_CLASS_CPU_PACKED_DOT);
  EXPECT_EQ(request.policy, LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED);
}

TEST_F(ContractVectorTest, DynamicDot4ReportsShapeRejection) {
  static const char kSource[] = R"(
func.def @dynamic_dot4(%n: index, %m: index, %lhs: vector<[%n]xi8>, %rhs: vector<[%n]xi8>, %acc: vector<[%m]xi32>) -> (vector<[%m]xi32>) {
  %r = vector.dot4i<u8s8> %lhs, %rhs, %acc : vector<[%n]xi8>, vector<[%n]xi8>, vector<[%m]xi32>
  func.return %r : vector<[%m]xi32>
}
)";

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseAndVerify(kSource, &module));
  ModulePtr module_ptr(module);

  const loom_op_t* op = FirstVectorDotOp(module_ptr.get());
  ASSERT_NE(op, nullptr);
  loom_contract_request_t request = {};
  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_contract_request_from_vector_dot_op(
      module_ptr.get(), op, LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED,
      &request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_SHAPE);
}

TEST_F(ContractVectorTest, PackedI4DotDoesNotPretendToBeByteDotContract) {
  static const char kSource[] = R"(
func.def @packed_i4_dot(%lhs: vector<4xi32>, %rhs: vector<4xi32>, %acc: vector<4xi32>) -> (vector<4xi32>) {
  %r = vector.dot8i4<u4s4> %lhs, %rhs, %acc : vector<4xi32>
  func.return %r : vector<4xi32>
}
)";

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseAndVerify(kSource, &module));
  ModulePtr module_ptr(module);

  const loom_op_t* op = FirstVectorDotOp(module_ptr.get());
  ASSERT_NE(op, nullptr);
  loom_contract_request_t request = {};
  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_contract_request_from_vector_dot_op(
      module_ptr.get(), op, LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED,
      &request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_INVALID_REQUEST);
}

}  // namespace
