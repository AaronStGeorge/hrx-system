// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/x86/legality.h"

#include <cstring>
#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/emit/llvmir/x86/target_env.h"

namespace loom {
namespace {

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

class LlvmIrX86LegalityTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_VECTOR, loom_vector_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));

    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("x86_legality_test"),
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

  bool VerifyBundle(const loom_target_bundle_t* bundle,
                    loom_llvmir_target_legality_diagnostic_t* diagnostic) {
    const loom_llvmir_target_legality_provider_t* providers[] = {
        loom_llvmir_x86_legality_provider(),
    };
    const loom_llvmir_target_profile_t* profile =
        bundle == loom_llvmir_target_bundle_x86_64_packed_dot_object()
            ? loom_llvmir_target_profile_x86_64_packed_dot_object()
            : loom_llvmir_target_profile_x86_64_object();
    loom_llvmir_target_legality_options_t options;
    std::memset(&options, 0, sizeof(options));
    options.snapshot = bundle->snapshot;
    options.export_plan = bundle->export_plan;
    options.config = bundle->config;
    options.profile = profile;
    options.providers = providers;
    options.provider_count = IREE_ARRAYSIZE(providers);
    return loom_llvmir_verify_target_legality(module_, &options, diagnostic);
  }

  void BuildDot4S8S8Function() {
    loom_type_t input_type = loom_type_shaped_1d(
        LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I8, loom_dim_pack_static(32), 0);
    loom_type_t result_type = loom_type_shaped_1d(
        LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(8), 0);
    loom_symbol_ref_t symbol = MakeSymbol(IREE_SV("dot4_s8s8"));
    loom_type_t arg_types[3] = {input_type, input_type, result_type};
    loom_op_t* func_op = NULL;
    IREE_ASSERT_OK(loom_func_def_build(
        &module_builder_, 0, 0, 0, 0, loom_symbol_ref_null(), 0,
        loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
        loom_named_attr_slice_empty(), symbol, arg_types,
        IREE_ARRAYSIZE(arg_types), &result_type, 1, NULL, 0, NULL, 0,
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

TEST_F(LlvmIrX86LegalityTest, ReportsPackedDotFeatureRejection) {
  BuildDot4S8S8Function();

  loom_llvmir_target_legality_diagnostic_t diagnostic;
  EXPECT_FALSE(
      VerifyBundle(loom_llvmir_target_bundle_x86_64_object(), &diagnostic));
  EXPECT_EQ(diagnostic.code,
            LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_TARGET_CONTRACT);
  EXPECT_EQ(ToString(diagnostic.provider_name), "x86");
  EXPECT_EQ(ToString(diagnostic.constraint_key), "x86-packed-dot-descriptor");
  EXPECT_EQ(ToString(diagnostic.subject_key), "features");
}

TEST_F(LlvmIrX86LegalityTest, AcceptsPackedDotWithMatchingFeatures) {
  BuildDot4S8S8Function();

  loom_llvmir_target_legality_diagnostic_t diagnostic;
  ASSERT_TRUE(VerifyBundle(loom_llvmir_target_bundle_x86_64_packed_dot_object(),
                           &diagnostic));
  EXPECT_EQ(diagnostic.code, LOOM_LLVMIR_TARGET_LEGALITY_OK);
}

}  // namespace
}  // namespace loom
