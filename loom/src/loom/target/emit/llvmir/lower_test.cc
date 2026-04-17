// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/lower.h"

#include <cstring>
#include <memory>
#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/emit/llvmir/test_target.h"
#include "loom/target/emit/llvmir/verify.h"

namespace loom {
namespace {

using LlvmModulePtr =
    std::unique_ptr<loom_llvmir_module_t, void (*)(loom_llvmir_module_t*)>;

std::string StatusToString(iree_status_t status) {
  iree_allocator_t allocator = iree_allocator_system();
  char* buffer = NULL;
  iree_host_size_t length = 0;
  if (iree_status_to_string(status, &allocator, &buffer, &length)) {
    std::string result(buffer, length);
    iree_allocator_free(allocator, buffer);
    return result;
  }
  return std::string("status code ") +
         std::to_string(static_cast<int>(iree_status_code(status)));
}

class LlvmIrLowerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_VECTOR, loom_vector_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));

    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("lower_test"),
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

  void SetValueName(loom_value_id_t value_id, iree_string_view_t name) {
    IREE_CHECK_OK(
        loom_module_set_value_name(module_, value_id, InternString(name)));
  }

  loom_builder_t BodyBuilder(loom_op_t* func_op) {
    loom_func_like_t func = loom_func_like_cast(module_, func_op);
    loom_region_t* body = loom_func_like_body(func);
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_region_entry_block(body), &builder);
    return builder;
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
    SetValueName(args[0], IREE_SV("lhs"));
    SetValueName(args[1], IREE_SV("rhs"));

    loom_builder_t body_builder = BodyBuilder(func_op);
    loom_op_t* add_op = NULL;
    IREE_ASSERT_OK(loom_scalar_addi_build(
        &body_builder, LOOM_SCALAR_INTOVERFLOWFLAGS_NSW, args[0], args[1], i32,
        LOOM_LOCATION_UNKNOWN, &add_op));
    loom_value_id_t sum = loom_scalar_addi_result(add_op);
    SetValueName(sum, IREE_SV("sum"));
    loom_op_t* return_op = NULL;
    IREE_ASSERT_OK(loom_func_return_build(&body_builder, &sum, 1,
                                          LOOM_LOCATION_UNKNOWN, &return_op));
  }

  void BuildDot4S8S8Function() {
    loom_type_t input_type = loom_type_shaped_1d(
        LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I8, loom_dim_pack_static(32), 0);
    loom_type_t result_type = loom_type_shaped_1d(
        LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(8), 0);
    loom_type_t arg_types[3] = {input_type, input_type, result_type};
    loom_symbol_ref_t symbol = MakeSymbol(IREE_SV("dot4_s8s8"));
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

  iree_status_t LowerModule(const loom_llvmir_target_profile_t* profile,
                            const loom_llvmir_lowering_provider_t* provider,
                            loom_llvmir_module_t** out_lowered) {
    const loom_llvmir_lowering_provider_t* providers[1];
    loom_llvmir_lowering_options_t options;
    std::memset(&options, 0, sizeof(options));
    options.target_profile = profile;
    options.source_name = IREE_SV("lower_test");
    if (provider) {
      providers[0] = provider;
      options.providers = providers;
      options.provider_count = IREE_ARRAYSIZE(providers);
    }
    return loom_llvmir_lower_module(module_, &options, iree_allocator_system(),
                                    out_lowered);
  }

  iree_status_t LowerModule(loom_llvmir_module_t** out_lowered) {
    return LowerModule(loom_llvmir_target_profile_test_object(), NULL,
                       out_lowered);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = NULL;
  loom_builder_t module_builder_;
};

TEST_F(LlvmIrLowerTest, LoweringRequiresTargetProfile) {
  BuildAddI32Function();

  loom_llvmir_module_t* lowered = NULL;
  loom_llvmir_lowering_options_t options;
  std::memset(&options, 0, sizeof(options));
  iree_status_t status = loom_llvmir_lower_module(
      module_, &options, iree_allocator_system(), &lowered);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(lowered, nullptr);
  iree_status_free(status);
}

TEST_F(LlvmIrLowerTest, LoweringRejectsInvalidProvider) {
  BuildAddI32Function();

  loom_llvmir_lowering_provider_t invalid_provider;
  std::memset(&invalid_provider, 0, sizeof(invalid_provider));
  loom_llvmir_module_t* lowered = NULL;
  iree_status_t status = LowerModule(loom_llvmir_target_profile_test_object(),
                                     &invalid_provider, &lowered);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(lowered, nullptr);
  iree_status_free(status);
}

TEST_F(LlvmIrLowerTest, LowersCoreOpsWithTestTarget) {
  BuildAddI32Function();

  loom_llvmir_module_t* lowered = NULL;
  IREE_ASSERT_OK(LowerModule(&lowered));
  LlvmModulePtr lowered_ptr(lowered, loom_llvmir_module_free);
  IREE_ASSERT_OK(loom_llvmir_verify_module(lowered_ptr.get()));
}

TEST_F(LlvmIrLowerTest, TargetContractOpsRequireProvider) {
  BuildDot4S8S8Function();

  loom_llvmir_module_t* lowered = NULL;
  iree_status_t status = LowerModule(&lowered);
  std::string message = StatusToString(status);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_UNIMPLEMENTED);
  EXPECT_EQ(lowered, nullptr);
  EXPECT_NE(message.find("explicit target lowering provider"),
            std::string::npos)
      << message;
  iree_status_free(status);
}

}  // namespace
}  // namespace loom
