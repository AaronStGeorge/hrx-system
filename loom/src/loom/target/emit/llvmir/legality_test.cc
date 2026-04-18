// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/legality.h"

#include <cstdint>
#include <cstring>
#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/llvmir/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/emit/llvmir/test_target.h"

namespace loom {
namespace {

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

class LlvmIrLegalityTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LLVMIR, loom_llvmir_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCF, loom_scf_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_VECTOR, loom_vector_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));

    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("legality_test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  typedef const loom_op_vtable_t* const* (*DialectVtablesFn)(
      iree_host_size_t* out_count);

  void RegisterDialect(loom_dialect_id_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  loom_string_id_t InternString(iree_string_view_t string) {
    loom_string_id_t string_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(module_, string, &string_id));
    return string_id;
  }

  loom_symbol_ref_t MakeSymbol(iree_string_view_t name) {
    loom_string_id_t name_id = InternString(name);
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    loom_symbol_ref_t ref;
    std::memset(&ref, 0, sizeof(ref));
    ref.symbol_id = symbol_id;
    return ref;
  }

  loom_builder_t BodyBuilder(loom_op_t* func_op) {
    loom_func_like_t func = loom_func_like_cast(module_, func_op);
    loom_region_t* body = loom_func_like_body(func);
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_region_entry_block(body), &builder);
    return builder;
  }

  iree_status_t VerifyTestObject(
      loom_llvmir_target_legality_diagnostic_t* diagnostic) {
    const loom_target_bundle_t* bundle =
        loom_llvmir_target_bundle_test_object();
    loom_llvmir_target_legality_options_t options;
    std::memset(&options, 0, sizeof(options));
    options.snapshot = bundle->snapshot;
    options.export_plan = bundle->export_plan;
    options.config = bundle->config;
    return loom_llvmir_verify_target_legality(module_, &options, diagnostic);
  }

  void BuildAddI32Function() {
    loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    loom_type_t arg_types[2] = {i32, i32};
    loom_type_t result_types[1] = {i32};
    loom_symbol_ref_t symbol = MakeSymbol(IREE_SV("add_i32"));
    loom_op_t* func_op = NULL;
    IREE_ASSERT_OK(loom_func_def_build(
        &module_builder_, LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY,
        LOOM_FUNC_VISIBILITY_PUBLIC, 0, 0, symbol, arg_types, 2, result_types,
        1, NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &func_op));
    loom_func_like_t func = loom_func_like_cast(module_, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
    ASSERT_EQ(arg_count, 2);

    loom_builder_t body_builder = BodyBuilder(func_op);
    loom_op_t* add_op = NULL;
    IREE_ASSERT_OK(loom_scalar_addi_build(&body_builder, 0, args[0], args[1],
                                          i32, LOOM_LOCATION_UNKNOWN, &add_op));
    loom_value_id_t sum = loom_scalar_addi_result(add_op);
    loom_op_t* return_op = NULL;
    IREE_ASSERT_OK(loom_func_return_build(&body_builder, &sum, 1,
                                          LOOM_LOCATION_UNKNOWN, &return_op));
  }

  void BuildTargetRecords() {
    loom_symbol_ref_t snapshot_symbol = MakeSymbol(IREE_SV("snapshot"));
    loom_symbol_ref_t export_symbol = MakeSymbol(IREE_SV("export"));
    loom_symbol_ref_t config_symbol = MakeSymbol(IREE_SV("config"));
    loom_symbol_ref_t bundle_symbol = MakeSymbol(IREE_SV("bundle"));
    loom_op_t* snapshot_op = NULL;
    IREE_ASSERT_OK(loom_target_snapshot_build(
        &module_builder_, snapshot_symbol,
        LOOM_TARGET_SNAPSHOT_CODEGEN_FORMAT_LLVMIR,
        InternString(IREE_SV("loom-test64-unknown-none")),
        InternString(IREE_SV("e-p:64:64-i64:64-n8:16:32:64-S128")),
        LOOM_TARGET_SNAPSHOT_ARTIFACT_FORMAT_ELF, InternString(IREE_SV("")),
        InternString(IREE_SV("")), 64, 64, 64, 0, 0, 0, 0, 0, 0, UINT32_MAX,
        LOOM_LOCATION_UNKNOWN, &snapshot_op));
    ASSERT_NE(snapshot_op, nullptr);
    loom_op_t* export_op = NULL;
    IREE_ASSERT_OK(loom_target_export_build(
        &module_builder_, 0, export_symbol, loom_symbol_ref_null(),
        InternString(IREE_SV("add_i32")),
        LOOM_TARGET_EXPORT_ABI_OBJECT_FUNCTION,
        LOOM_TARGET_EXPORT_LINKAGE_DEFAULT, 0, 0, 0, 0, 0, 0, 0,
        LOOM_LOCATION_UNKNOWN, &export_op));
    ASSERT_NE(export_op, nullptr);
    loom_op_t* config_op = NULL;
    IREE_ASSERT_OK(loom_target_config_build(
        &module_builder_, config_symbol, InternString(IREE_SV("")),
        /*contract_feature_bits=*/0, LOOM_LOCATION_UNKNOWN, &config_op));
    ASSERT_NE(config_op, nullptr);
    loom_op_t* bundle_op = NULL;
    IREE_ASSERT_OK(loom_target_bundle_build(
        &module_builder_, bundle_symbol, snapshot_symbol, export_symbol,
        config_symbol, LOOM_LOCATION_UNKNOWN, &bundle_op));
    ASSERT_NE(bundle_op, nullptr);
  }

  void BuildStructuredIfFunction() {
    loom_type_t i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
    loom_symbol_ref_t symbol = MakeSymbol(IREE_SV("structured_if"));
    loom_type_t arg_types[1] = {i1};
    loom_op_t* func_op = NULL;
    IREE_ASSERT_OK(loom_func_def_build(
        &module_builder_, LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY,
        LOOM_FUNC_VISIBILITY_PUBLIC, 0, 0, symbol, arg_types,
        IREE_ARRAYSIZE(arg_types), NULL, 0, NULL, 0, NULL, 0,
        LOOM_LOCATION_UNKNOWN, &func_op));
    loom_func_like_t func = loom_func_like_cast(module_, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
    ASSERT_EQ(arg_count, 1);

    loom_builder_t body_builder = BodyBuilder(func_op);
    loom_op_t* if_op = NULL;
    IREE_ASSERT_OK(loom_scf_if_build(&body_builder, args[0], NULL, 0, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &if_op));
    loom_builder_t then_builder;
    loom_builder_initialize(
        module_, &module_->arena,
        loom_region_entry_block(loom_scf_if_then_region(if_op)), &then_builder);
    loom_op_t* then_yield = NULL;
    IREE_ASSERT_OK(loom_scf_yield_build(&then_builder, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &then_yield));
    loom_builder_t else_builder;
    loom_builder_initialize(
        module_, &module_->arena,
        loom_region_entry_block(loom_scf_if_else_region(if_op)), &else_builder);
    loom_op_t* else_yield = NULL;
    IREE_ASSERT_OK(loom_scf_yield_build(&else_builder, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &else_yield));
    loom_op_t* return_op = NULL;
    IREE_ASSERT_OK(loom_func_return_build(&body_builder, NULL, 0,
                                          LOOM_LOCATION_UNKNOWN, &return_op));
  }

  void BuildUnknownIntrinsicFunction() {
    loom_symbol_ref_t symbol = MakeSymbol(IREE_SV("unknown_intrinsic"));
    loom_op_t* func_op = NULL;
    IREE_ASSERT_OK(loom_func_def_build(&module_builder_, 0, 0, 0, 0, symbol,
                                       NULL, 0, NULL, 0, NULL, 0, NULL, 0,
                                       LOOM_LOCATION_UNKNOWN, &func_op));
    loom_builder_t body_builder = BodyBuilder(func_op);
    loom_op_t* intrinsic_op = NULL;
    IREE_ASSERT_OK(loom_llvmir_intrinsic_build(
        &body_builder, InternString(IREE_SV("llvm.imaginary")), NULL, 0, NULL,
        0, NULL, 0, LOOM_LOCATION_UNKNOWN, &intrinsic_op));
    loom_op_t* return_op = NULL;
    IREE_ASSERT_OK(loom_func_return_build(&body_builder, NULL, 0,
                                          LOOM_LOCATION_UNKNOWN, &return_op));
  }

  void BuildF8Function() {
    loom_type_t f8 = loom_type_scalar(LOOM_SCALAR_TYPE_F8E4M3);
    loom_symbol_ref_t symbol = MakeSymbol(IREE_SV("f8_identity"));
    loom_op_t* func_op = NULL;
    IREE_ASSERT_OK(loom_func_def_build(&module_builder_, 0, 0, 0, 0, symbol,
                                       &f8, 1, &f8, 1, NULL, 0, NULL, 0,
                                       LOOM_LOCATION_UNKNOWN, &func_op));
    loom_func_like_t func = loom_func_like_cast(module_, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
    ASSERT_EQ(arg_count, 1);
    loom_builder_t body_builder = BodyBuilder(func_op);
    loom_op_t* return_op = NULL;
    IREE_ASSERT_OK(loom_func_return_build(&body_builder, args, 1,
                                          LOOM_LOCATION_UNKNOWN, &return_op));
  }

  void BuildDot4S8S8Function() {
    loom_type_t input_type = loom_type_shaped_1d(
        LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I8, loom_dim_pack_static(32), 0);
    loom_type_t result_type = loom_type_shaped_1d(
        LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(8), 0);
    loom_symbol_ref_t symbol = MakeSymbol(IREE_SV("dot4_s8s8"));
    loom_type_t arg_types[3] = {input_type, input_type, result_type};
    loom_op_t* func_op = NULL;
    IREE_ASSERT_OK(loom_func_def_build(&module_builder_, 0, 0, 0, 0, symbol,
                                       arg_types, IREE_ARRAYSIZE(arg_types),
                                       &result_type, 1, NULL, 0, NULL, 0,
                                       LOOM_LOCATION_UNKNOWN, &func_op));
    loom_func_like_t func = loom_func_like_cast(module_, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
    ASSERT_EQ(arg_count, 3);

    loom_builder_t body_builder = BodyBuilder(func_op);
    loom_op_t* dot_op = NULL;
    IREE_ASSERT_OK(loom_vector_dot4i_build(
        &body_builder, LOOM_VECTOR_DOT4I_KIND_S8S8, args[0], args[1], args[2],
        result_type, LOOM_LOCATION_UNKNOWN, &dot_op));
    loom_value_id_t dot = loom_vector_dot4i_result(dot_op);
    loom_op_t* return_op = NULL;
    IREE_ASSERT_OK(loom_func_return_build(&body_builder, &dot, 1,
                                          LOOM_LOCATION_UNKNOWN, &return_op));
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = NULL;
  loom_builder_t module_builder_;
};

TEST_F(LlvmIrLegalityTest, AcceptsObjectArithmetic) {
  BuildAddI32Function();

  loom_llvmir_target_legality_diagnostic_t diagnostic;
  IREE_ASSERT_OK(VerifyTestObject(&diagnostic));
  EXPECT_EQ(diagnostic.code, LOOM_LLVMIR_TARGET_LEGALITY_OK);
}

TEST_F(LlvmIrLegalityTest, AcceptsModuleTargetRecordsAsMetadata) {
  BuildTargetRecords();
  BuildAddI32Function();

  loom_llvmir_target_legality_diagnostic_t diagnostic;
  IREE_ASSERT_OK(VerifyTestObject(&diagnostic));
  EXPECT_EQ(diagnostic.code, LOOM_LLVMIR_TARGET_LEGALITY_OK);
}

TEST_F(LlvmIrLegalityTest, RejectsStructuredScfBeforeLowering) {
  BuildStructuredIfFunction();

  loom_llvmir_target_legality_diagnostic_t diagnostic;
  iree_status_t status = VerifyTestObject(&diagnostic);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, status);
  EXPECT_EQ(diagnostic.code,
            LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_CONTROL_FLOW);
  EXPECT_EQ(ToString(diagnostic.op_name), "scf.if");
  EXPECT_NE(ToString(diagnostic.detail).find("lowered to CFG"),
            std::string::npos);
}

TEST_F(LlvmIrLegalityTest, RejectsUnknownIntrinsicKind) {
  BuildUnknownIntrinsicFunction();

  loom_llvmir_target_legality_diagnostic_t diagnostic;
  iree_status_t status = VerifyTestObject(&diagnostic);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, status);
  EXPECT_EQ(diagnostic.code, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_INTRINSIC);
  EXPECT_EQ(ToString(diagnostic.target_detail), "llvm.imaginary");
}

TEST_F(LlvmIrLegalityTest, RejectsFp8TypeBeforeLowering) {
  BuildF8Function();

  loom_llvmir_target_legality_diagnostic_t diagnostic;
  iree_status_t status = VerifyTestObject(&diagnostic);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, status);
  EXPECT_EQ(diagnostic.code, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_TYPE);
  EXPECT_EQ(ToString(diagnostic.target_detail), "fp8");
}

TEST_F(LlvmIrLegalityTest, RejectsTargetContractWithoutProvider) {
  BuildDot4S8S8Function();

  loom_llvmir_target_legality_diagnostic_t diagnostic;
  iree_status_t status = VerifyTestObject(&diagnostic);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, status);
  EXPECT_EQ(diagnostic.code,
            LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_TARGET_CONTRACT);
  EXPECT_NE(ToString(diagnostic.detail).find("target legality provider"),
            std::string::npos);
}

}  // namespace
}  // namespace loom
