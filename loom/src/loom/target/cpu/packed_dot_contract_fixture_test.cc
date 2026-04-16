// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/cpu/packed_dot_contract.h"
#include "loom/target/cpu/packed_dot_vector.h"
#include "loom/testing/context.h"
#include "loom/transforms/verify.h"

namespace {

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

struct ModuleDeleter {
  void operator()(loom_module_t* module) const { loom_module_free(module); }
};

using ModulePtr = std::unique_ptr<loom_module_t, ModuleDeleter>;

class PackedDotContractFixtureTest : public ::testing::Test {
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
        iree_make_cstring_view(source), IREE_SV("packed_dot_fixture.loom"),
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
      if (!loom_func_like_isa(function)) continue;
      loom_region_t* body = loom_func_like_body(function);
      if (body == nullptr) continue;
      loom_block_t* block = nullptr;
      loom_region_for_each_block(body, block) {
        loom_op_t* op = nullptr;
        loom_block_for_each_op(block, op) {
          if (loom_vector_dot2f_isa(op) || loom_vector_dot4i_isa(op) ||
              loom_vector_dot8i4_isa(op) || loom_vector_dot4f8_isa(op)) {
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

TEST_F(PackedDotContractFixtureTest, ParsedF16Dot2SelectsAvx10WideForm) {
  static const char kSource[] = R"(
func.def @f16_dot2(%lhs: vector<32xf16>, %rhs: vector<32xf16>, %acc: vector<16xf32>) -> (vector<16xf32>) {
  %r = vector.dot2f %lhs, %rhs, %acc : vector<32xf16>, vector<32xf16>, vector<16xf32>
  func.return %r : vector<16xf32>
}
)";

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseAndVerify(kSource, &module));
  ModulePtr module_ptr(module);

  const loom_op_t* op = FirstVectorDotOp(module_ptr.get());
  ASSERT_NE(op, nullptr);
  loom_cpu_packed_dot_match_request_t request = {};
  ASSERT_TRUE(loom_cpu_packed_dot_match_request_from_vector_op(module_ptr.get(),
                                                               op, &request));
  IREE_ASSERT_OK(loom_cpu_packed_dot_feature_bits_for_name(
      IREE_SV("x86-avx10.2"), &request.feature_bits));

  const loom_cpu_packed_dot_descriptor_t* descriptor =
      loom_cpu_packed_dot_select(&request, nullptr);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(ToString(descriptor->name), "x86.avx10.2.vdpphps.512");
}

TEST_F(PackedDotContractFixtureTest, ParsedBf16Dot2SelectsAvx512Bf16) {
  static const char kSource[] = R"(
func.def @bf16_dot2(%lhs: vector<16xbf16>, %rhs: vector<16xbf16>, %acc: vector<8xf32>) -> (vector<8xf32>) {
  %r = vector.dot2f %lhs, %rhs, %acc : vector<16xbf16>, vector<16xbf16>, vector<8xf32>
  func.return %r : vector<8xf32>
}
)";

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseAndVerify(kSource, &module));
  ModulePtr module_ptr(module);

  const loom_op_t* op = FirstVectorDotOp(module_ptr.get());
  ASSERT_NE(op, nullptr);
  loom_cpu_packed_dot_match_request_t request = {};
  ASSERT_TRUE(loom_cpu_packed_dot_match_request_from_vector_op(module_ptr.get(),
                                                               op, &request));
  IREE_ASSERT_OK(loom_cpu_packed_dot_feature_bits_for_name(
      IREE_SV("x86-avx512-bf16-vl"), &request.feature_bits));

  const loom_cpu_packed_dot_descriptor_t* descriptor =
      loom_cpu_packed_dot_select(&request, nullptr);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(ToString(descriptor->name), "x86.avx512-bf16.vdpbf16ps.256");
}

TEST_F(PackedDotContractFixtureTest, ParsedU8S8Dot4SelectsAvxVnni256) {
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
  loom_cpu_packed_dot_match_request_t request = {};
  ASSERT_TRUE(loom_cpu_packed_dot_match_request_from_vector_op(module_ptr.get(),
                                                               op, &request));
  IREE_ASSERT_OK(loom_cpu_packed_dot_feature_bits_for_name(
      IREE_SV("x86-avx-vnni"), &request.feature_bits));

  const loom_cpu_packed_dot_descriptor_t* descriptor =
      loom_cpu_packed_dot_select(&request, nullptr);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(ToString(descriptor->name), "x86.avx-vnni.vpdpbusd.256");
}

TEST_F(PackedDotContractFixtureTest, ParsedS8S8Dot4NeedsInt8Feature) {
  static const char kSource[] = R"(
func.def @s8s8_dot4(%lhs: vector<32xi8>, %rhs: vector<32xi8>, %acc: vector<8xi32>) -> (vector<8xi32>) {
  %r = vector.dot4i<s8s8> %lhs, %rhs, %acc : vector<32xi8>, vector<32xi8>, vector<8xi32>
  func.return %r : vector<8xi32>
}
)";

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseAndVerify(kSource, &module));
  ModulePtr module_ptr(module);

  const loom_op_t* op = FirstVectorDotOp(module_ptr.get());
  ASSERT_NE(op, nullptr);
  loom_cpu_packed_dot_match_request_t request = {};
  ASSERT_TRUE(loom_cpu_packed_dot_match_request_from_vector_op(module_ptr.get(),
                                                               op, &request));

  IREE_ASSERT_OK(loom_cpu_packed_dot_feature_bits_for_name(
      IREE_SV("x86-avx-vnni"), &request.feature_bits));
  loom_cpu_packed_dot_match_diagnostic_t diagnostic = {};
  EXPECT_EQ(loom_cpu_packed_dot_select(&request, &diagnostic), nullptr);
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CPU_PACKED_DOT_REJECTION_FEATURES);

  IREE_ASSERT_OK(loom_cpu_packed_dot_feature_bits_for_name(
      IREE_SV("x86-avx-vnni-int8"), &request.feature_bits));
  const loom_cpu_packed_dot_descriptor_t* descriptor =
      loom_cpu_packed_dot_select(&request, nullptr);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(ToString(descriptor->name), "x86.avx-vnni-int8.vpdpbssd.256");
}

TEST_F(PackedDotContractFixtureTest, ParsedS8S8Dot4SelectsAvx10WideForm) {
  static const char kSource[] = R"(
func.def @s8s8_dot4_wide(%lhs: vector<64xi8>, %rhs: vector<64xi8>, %acc: vector<16xi32>) -> (vector<16xi32>) {
  %r = vector.dot4i<s8s8> %lhs, %rhs, %acc : vector<64xi8>, vector<64xi8>, vector<16xi32>
  func.return %r : vector<16xi32>
}
)";

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseAndVerify(kSource, &module));
  ModulePtr module_ptr(module);

  const loom_op_t* op = FirstVectorDotOp(module_ptr.get());
  ASSERT_NE(op, nullptr);
  loom_cpu_packed_dot_match_request_t request = {};
  ASSERT_TRUE(loom_cpu_packed_dot_match_request_from_vector_op(module_ptr.get(),
                                                               op, &request));
  request.family = LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX10_2;
  IREE_ASSERT_OK(loom_cpu_packed_dot_feature_bits_for_name(
      IREE_SV("x86-avx10.2"), &request.feature_bits));

  const loom_cpu_packed_dot_descriptor_t* descriptor =
      loom_cpu_packed_dot_select(&request, nullptr);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(ToString(descriptor->name), "x86.avx10.2.vpdpbssd.512");
}

TEST_F(PackedDotContractFixtureTest, ParsedDot8I4DoesNotInferCpuRequest) {
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
  loom_cpu_packed_dot_match_request_t request = {};
  EXPECT_FALSE(loom_cpu_packed_dot_match_request_from_vector_op(
      module_ptr.get(), op, &request));
}

TEST_F(PackedDotContractFixtureTest,
       ParsedDot8I4DoesNotInferCpuRequestForByteDotShape) {
  static const char kSource[] = R"(
func.def @packed_i4_dot_vnni_shaped(%lhs: vector<8xi32>, %rhs: vector<8xi32>, %acc: vector<8xi32>) -> (vector<8xi32>) {
  %r = vector.dot8i4<s4s4> %lhs, %rhs, %acc : vector<8xi32>
  func.return %r : vector<8xi32>
}
)";

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseAndVerify(kSource, &module));
  ModulePtr module_ptr(module);

  const loom_op_t* op = FirstVectorDotOp(module_ptr.get());
  ASSERT_NE(op, nullptr);
  loom_cpu_packed_dot_match_request_t request = {};
  EXPECT_FALSE(loom_cpu_packed_dot_match_request_from_vector_op(
      module_ptr.get(), op, &request));
}

TEST_F(PackedDotContractFixtureTest, DynamicDot4DoesNotInferNativeRequest) {
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
  loom_cpu_packed_dot_match_request_t request = {};
  request.feature_bits = UINT64_MAX;
  EXPECT_FALSE(loom_cpu_packed_dot_match_request_from_vector_op(
      module_ptr.get(), op, &request));
  EXPECT_EQ(request.feature_bits, 0u);
}

TEST_F(PackedDotContractFixtureTest, Dot4F8DoesNotInferCpuRequest) {
  static const char kSource[] = R"(
func.def @dot4f8(%lhs: vector<4xi32>, %rhs: vector<4xi32>, %acc: vector<4xf32>) -> (vector<4xf32>) {
  %r = vector.dot4f8<fp8bf8> %lhs, %rhs, %acc : vector<4xi32>, vector<4xf32>
  func.return %r : vector<4xf32>
}
)";

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseAndVerify(kSource, &module));
  ModulePtr module_ptr(module);

  const loom_op_t* op = FirstVectorDotOp(module_ptr.get());
  ASSERT_NE(op, nullptr);
  loom_cpu_packed_dot_match_request_t request = {};
  EXPECT_FALSE(loom_cpu_packed_dot_match_request_from_vector_op(
      module_ptr.get(), op, &request));
}

TEST_F(PackedDotContractFixtureTest, Dot4VerifierRejectsBadDivisibility) {
  static const char kSource[] = R"(
func.def @bad_dot4(%lhs: vector<10xi8>, %rhs: vector<10xi8>, %acc: vector<2xi32>) {
  %r = vector.dot4i<s8s8> %lhs, %rhs, %acc : vector<10xi8>, vector<10xi8>, vector<2xi32>
  func.return
}
)";

  loom_module_t* module = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        ParseAndVerify(kSource, &module));
}

}  // namespace
