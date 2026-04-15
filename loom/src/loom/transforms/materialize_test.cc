// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/materialize.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

class MaterializeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_test_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TEST, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("source"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &source_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("target"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &target_));
    loom_builder_initialize(source_, &source_->arena,
                            loom_module_block(source_), &source_builder_);
    loom_builder_initialize(target_, &target_->arena,
                            loom_module_block(target_), &target_builder_);
    iree_arena_initialize(&block_pool_, &remap_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&remap_arena_);
    loom_module_free(target_);
    loom_module_free(source_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_ir_remap_t InitializeRemap(
      bool allow_unmapped_values = false,
      const loom_ir_remap_options_t* options = nullptr) {
    loom_ir_remap_options_t local_options = {};
    if (options) local_options = *options;
    local_options.allow_unmapped_values = allow_unmapped_values;
    loom_ir_remap_t remap = {};
    IREE_CHECK_OK(loom_ir_remap_initialize(source_, target_, &remap_arena_,
                                           &local_options, &remap));
    return remap;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* source_ = nullptr;
  loom_module_t* target_ = nullptr;
  loom_builder_t source_builder_ = {};
  loom_builder_t target_builder_ = {};
  iree_arena_allocator_t remap_arena_;
};

TEST_F(MaterializeTest, ClonesCoResultDynamicTypeReferences) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t input_type = loom_type_shaped_1d(
      LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_op_t* input_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&source_builder_, loom_attr_i64(0),
                                          input_type, LOOM_LOCATION_UNKNOWN,
                                          &input_op));
  loom_value_id_t source_input = loom_test_constant_result(input_op);

  loom_value_id_t reserved_results[2] = {};
  IREE_ASSERT_OK(loom_builder_reserve_results(
      &source_builder_, IREE_ARRAYSIZE(reserved_results), reserved_results));
  loom_type_t result_types[] = {
      loom_type_shaped_1d(LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(reserved_results[1]), 0),
      index_type,
  };
  loom_op_t* deflate_op = nullptr;
  IREE_ASSERT_OK(
      loom_test_deflate_build(&source_builder_, source_input, result_types,
                              IREE_ARRAYSIZE(result_types), nullptr, 0,
                              LOOM_LOCATION_UNKNOWN, &deflate_op));

  loom_op_t* target_input_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&target_builder_, loom_attr_i64(0),
                                          input_type, LOOM_LOCATION_UNKNOWN,
                                          &target_input_op));
  loom_value_id_t target_input = loom_test_constant_result(target_input_op);

  loom_ir_remap_t remap = InitializeRemap();
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, source_input, target_input));
  loom_op_t* cloned_op = nullptr;
  IREE_ASSERT_OK(
      loom_ir_clone_op(&target_builder_, deflate_op, &remap, &cloned_op));

  ASSERT_TRUE(loom_test_deflate_isa(cloned_op));
  loom_value_slice_t cloned_results = loom_test_deflate_results(cloned_op);
  ASSERT_EQ(cloned_results.count, 2u);
  loom_type_t cloned_output_type =
      loom_module_value_type(target_, cloned_results.values[0]);
  ASSERT_TRUE(loom_type_dim_is_dynamic_at(cloned_output_type, 0));
  EXPECT_EQ(loom_type_dim_value_id_at(cloned_output_type, 0),
            cloned_results.values[1]);
}

TEST_F(MaterializeTest, ClonesNestedRegionsAndBlockArguments) {
  loom_type_t tile_type = loom_type_shaped_1d(
      LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_op_t* input_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&source_builder_, loom_attr_i64(0),
                                          tile_type, LOOM_LOCATION_UNKNOWN,
                                          &input_op));
  loom_value_id_t source_input = loom_test_constant_result(input_op);

  loom_op_t* map_op = nullptr;
  IREE_ASSERT_OK(loom_test_map_build(&source_builder_, &source_input, 1,
                                     tile_type, nullptr, 0,
                                     LOOM_LOCATION_UNKNOWN, &map_op));
  loom_region_t* source_body = loom_test_map_body(map_op);
  loom_builder_ip_t saved_source_ip =
      loom_builder_enter_region(&source_builder_, map_op, source_body);
  loom_value_id_t element = loom_region_entry_arg_id(source_body, 0);
  loom_op_t* neg_op = nullptr;
  IREE_ASSERT_OK(loom_test_neg_build(&source_builder_, element, f32_type,
                                     LOOM_LOCATION_UNKNOWN, &neg_op));
  loom_value_id_t negated = loom_test_neg_result(neg_op);
  loom_op_t* yield_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&source_builder_, &negated, 1,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));
  loom_builder_restore(&source_builder_, saved_source_ip);

  loom_op_t* target_input_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&target_builder_, loom_attr_i64(0),
                                          tile_type, LOOM_LOCATION_UNKNOWN,
                                          &target_input_op));
  loom_value_id_t target_input = loom_test_constant_result(target_input_op);

  loom_ir_remap_t remap = InitializeRemap();
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, source_input, target_input));
  loom_op_t* cloned_map_op = nullptr;
  IREE_ASSERT_OK(
      loom_ir_clone_op(&target_builder_, map_op, &remap, &cloned_map_op));

  ASSERT_TRUE(loom_test_map_isa(cloned_map_op));
  EXPECT_EQ(loom_test_map_inputs(cloned_map_op).values[0], target_input);
  loom_region_t* cloned_body = loom_test_map_body(cloned_map_op);
  ASSERT_NE(cloned_body, nullptr);
  ASSERT_EQ(loom_region_entry_arg_count(cloned_body), 1u);
  loom_block_t* cloned_block = loom_region_entry_block(cloned_body);
  ASSERT_EQ(cloned_block->op_count, 2u);
  ASSERT_TRUE(loom_test_neg_isa(loom_block_op(cloned_block, 0)));
  ASSERT_TRUE(loom_test_yield_isa(loom_block_op(cloned_block, 1)));
  EXPECT_EQ(loom_block_op(cloned_block, 0)->parent_op, cloned_map_op);
  EXPECT_EQ(loom_block_op(cloned_block, 1)->parent_op, cloned_map_op);
  EXPECT_EQ(loom_test_neg_input(loom_block_op(cloned_block, 0)),
            loom_region_entry_arg_id(cloned_body, 0));
}

TEST_F(MaterializeTest, ClonesOperandDictOps) {
  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_op_t* source_input_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&source_builder_, loom_attr_i64(0),
                                          f32_type, LOOM_LOCATION_UNKNOWN,
                                          &source_input_op));
  loom_value_id_t source_input = loom_test_constant_result(source_input_op);
  loom_op_t* source_alpha_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&source_builder_, loom_attr_i64(1),
                                          i32_type, LOOM_LOCATION_UNKNOWN,
                                          &source_alpha_op));
  loom_value_id_t source_alpha = loom_test_constant_result(source_alpha_op);
  loom_op_t* source_beta_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&source_builder_, loom_attr_i64(2),
                                          f32_type, LOOM_LOCATION_UNKNOWN,
                                          &source_beta_op));
  loom_value_id_t source_beta = loom_test_constant_result(source_beta_op);

  loom_string_id_t source_beta_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(source_, IREE_SV("beta"), &source_beta_name));
  loom_string_id_t source_alpha_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(source_, IREE_SV("alpha"), &source_alpha_name));
  loom_named_value_t source_params[] = {
      {
          .name_id = source_beta_name,
          .reserved = 0,
          .value_id = source_beta,
      },
      {
          .name_id = source_alpha_name,
          .reserved = 0,
          .value_id = source_alpha,
      },
  };
  loom_op_t* source_dict_op = nullptr;
  IREE_ASSERT_OK(loom_test_operand_dict_build(
      &source_builder_, source_input, source_params,
      IREE_ARRAYSIZE(source_params), f32_type, LOOM_LOCATION_UNKNOWN,
      &source_dict_op));

  loom_op_t* target_input_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&target_builder_, loom_attr_i64(0),
                                          f32_type, LOOM_LOCATION_UNKNOWN,
                                          &target_input_op));
  loom_value_id_t target_input = loom_test_constant_result(target_input_op);
  loom_op_t* target_alpha_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&target_builder_, loom_attr_i64(1),
                                          i32_type, LOOM_LOCATION_UNKNOWN,
                                          &target_alpha_op));
  loom_value_id_t target_alpha = loom_test_constant_result(target_alpha_op);
  loom_op_t* target_beta_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&target_builder_, loom_attr_i64(2),
                                          f32_type, LOOM_LOCATION_UNKNOWN,
                                          &target_beta_op));
  loom_value_id_t target_beta = loom_test_constant_result(target_beta_op);

  loom_ir_remap_t remap = InitializeRemap();
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, source_input, target_input));
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, source_alpha, target_alpha));
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, source_beta, target_beta));
  loom_op_t* cloned_op = nullptr;
  IREE_ASSERT_OK(
      loom_ir_clone_op(&target_builder_, source_dict_op, &remap, &cloned_op));

  ASSERT_TRUE(loom_test_operand_dict_isa(cloned_op));
  EXPECT_EQ(loom_test_operand_dict_input(cloned_op), target_input);
  loom_value_slice_t cloned_params = loom_test_operand_dict_params(cloned_op);
  ASSERT_EQ(cloned_params.count, 2u);
  EXPECT_EQ(cloned_params.values[0], target_alpha);
  EXPECT_EQ(cloned_params.values[1], target_beta);

  loom_named_attr_slice_t names = loom_test_operand_dict_param_names(cloned_op);
  ASSERT_EQ(names.count, 2u);
  EXPECT_TRUE(iree_string_view_equal(
      target_->strings.entries[names.entries[0].name_id], IREE_SV("alpha")));
  EXPECT_EQ(loom_attr_as_i64(names.entries[0].value), 0);
  EXPECT_TRUE(iree_string_view_equal(
      target_->strings.entries[names.entries[1].name_id], IREE_SV("beta")));
  EXPECT_EQ(loom_attr_as_i64(names.entries[1].value), 1);
}

TEST_F(MaterializeTest, ClonesBlockOpsCanOmitTerminators) {
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_region_t* source_region = nullptr;
  IREE_ASSERT_OK(loom_module_allocate_region(source_, 1, &source_region));
  loom_builder_t source_region_builder = {};
  loom_builder_initialize(source_, &source_->arena,
                          loom_region_entry_block(source_region),
                          &source_region_builder);
  loom_op_t* constant_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&source_region_builder,
                                          loom_attr_i64(1), i32_type,
                                          LOOM_LOCATION_UNKNOWN, &constant_op));
  loom_value_id_t value = loom_test_constant_result(constant_op);
  loom_op_t* yield_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&source_region_builder, &value, 1,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));

  loom_ir_remap_t remap = InitializeRemap();
  loom_ir_clone_block_options_t options = {
      .omit_terminators = true,
  };
  IREE_ASSERT_OK(loom_ir_clone_block_ops(&target_builder_,
                                         loom_region_entry_block(source_region),
                                         &remap, &options));

  loom_block_t* target_block = loom_module_block(target_);
  ASSERT_EQ(target_block->op_count, 1u);
  ASSERT_TRUE(loom_test_constant_isa(loom_block_op(target_block, 0)));
}

}  // namespace
}  // namespace loom
