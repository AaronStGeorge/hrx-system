// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/amdgpu/legality.h"

#include <cstring>
#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/llvmir/ops.h"
#include "loom/target/emit/llvmir/amdgpu/target_env.h"

namespace loom {
namespace {

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

class LlvmIrAmdgpuLegalityTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LLVMIR, loom_llvmir_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));

    IREE_ASSERT_OK(loom_module_allocate(
        &context_, IREE_SV("amdgpu_legality_test"), &block_pool_, NULL,
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

  iree_status_t VerifyAmdgpuHal(
      loom_llvmir_target_legality_diagnostic_t* diagnostic) {
    const loom_llvmir_target_legality_provider_t* providers[] = {
        loom_llvmir_amdgpu_legality_provider(),
    };
    const loom_target_bundle_t* bundle = loom_llvmir_target_bundle_amdgpu_hal();
    loom_llvmir_target_legality_options_t options;
    std::memset(&options, 0, sizeof(options));
    options.snapshot = bundle->snapshot;
    options.export_plan = bundle->export_plan;
    options.config = bundle->config;
    options.providers = providers;
    options.provider_count = IREE_ARRAYSIZE(providers);
    return loom_llvmir_verify_target_legality(module_, &options, diagnostic);
  }

  void BuildWorkitemKernel() {
    loom_symbol_ref_t symbol = MakeSymbol(IREE_SV("dispatch"));
    loom_op_t* func_op = NULL;
    IREE_ASSERT_OK(loom_func_def_build(
        &module_builder_,
        LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY |
            LOOM_FUNC_DEF_BUILD_FLAG_HAS_CC,
        LOOM_FUNC_VISIBILITY_PUBLIC, LOOM_FUNC_CC_DEVICE, 0,
        loom_symbol_ref_null(), 0, loom_named_attr_slice_empty(),
        LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), symbol, NULL, 0,
        NULL, 0, NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &func_op));

    loom_builder_t body_builder = BodyBuilder(func_op);
    loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    loom_op_t* intrinsic_op = NULL;
    IREE_ASSERT_OK(loom_llvmir_intrinsic_build(
        &body_builder, InternString(IREE_SV("llvm.amdgcn.workitem.id.x")), NULL,
        0, &i32, 1, NULL, 0, LOOM_LOCATION_UNKNOWN, &intrinsic_op));
    loom_op_t* return_op = NULL;
    IREE_ASSERT_OK(loom_func_return_build(&body_builder, NULL, 0,
                                          LOOM_LOCATION_UNKNOWN, &return_op));
  }

  void BuildViewParameterKernel() {
    loom_type_t view_type = loom_type_shaped_1d(
        LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(1), 0);
    loom_symbol_ref_t symbol = MakeSymbol(IREE_SV("bad_view_dispatch"));
    loom_op_t* func_op = NULL;
    IREE_ASSERT_OK(loom_func_def_build(
        &module_builder_,
        LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY |
            LOOM_FUNC_DEF_BUILD_FLAG_HAS_CC,
        LOOM_FUNC_VISIBILITY_PUBLIC, LOOM_FUNC_CC_DEVICE, 0,
        loom_symbol_ref_null(), 0, loom_named_attr_slice_empty(),
        LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), symbol,
        &view_type, 1, NULL, 0, NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN,
        &func_op));

    loom_builder_t body_builder = BodyBuilder(func_op);
    loom_op_t* return_op = NULL;
    IREE_ASSERT_OK(loom_func_return_build(&body_builder, NULL, 0,
                                          LOOM_LOCATION_UNKNOWN, &return_op));
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = NULL;
  loom_builder_t module_builder_;
};

TEST_F(LlvmIrAmdgpuLegalityTest, AcceptsWorkitemKernel) {
  BuildWorkitemKernel();

  loom_llvmir_target_legality_diagnostic_t diagnostic;
  IREE_ASSERT_OK(VerifyAmdgpuHal(&diagnostic));
  EXPECT_EQ(diagnostic.code, LOOM_LLVMIR_TARGET_LEGALITY_OK);
}

TEST_F(LlvmIrAmdgpuLegalityTest, RejectsHalViewParameter) {
  BuildViewParameterKernel();

  loom_llvmir_target_legality_diagnostic_t diagnostic;
  iree_status_t status = VerifyAmdgpuHal(&diagnostic);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, status);
  EXPECT_EQ(diagnostic.code, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_ABI);
  EXPECT_NE(ToString(diagnostic.detail).find("view parameters"),
            std::string::npos);
}

}  // namespace
}  // namespace loom
