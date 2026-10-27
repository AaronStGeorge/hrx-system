// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/storage_identity.h"

#include <array>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/registers.h"

namespace loom {
namespace {

class LowStorageIdentityTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_low_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_LOW, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(
        &context_, IREE_SV("storage_identity_test"), &pool_, nullptr,
        iree_allocator_system(), &module_));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_value_id_t AddArgument(loom_block_t* block) {
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_module_define_value(
        module_, loom_low_register_type(1, 0, 1), &value_id));
    IREE_CHECK_OK(loom_block_add_arg(module_, block, value_id));
    return value_id;
  }

  iree_arena_block_pool_t pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
};

TEST_F(LowStorageIdentityTest, FlattensIdentityBranchCycle) {
  loom_region_t* region = nullptr;
  IREE_ASSERT_OK(loom_module_allocate_region(module_, 3, &region));
  region->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;

  std::array<std::array<loom_value_id_t, 3>, 3> arguments;
  for (uint16_t block_index = 0; block_index < 3; ++block_index) {
    loom_block_t* block = loom_region_block(region, block_index);
    for (loom_value_id_t& argument : arguments[block_index]) {
      argument = AddArgument(block);
    }
  }
  for (uint16_t block_index = 0; block_index < 3; ++block_index) {
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_region_block(region, block_index), &builder);
    loom_op_t* branch = nullptr;
    IREE_ASSERT_OK(loom_low_br_identity_build(
        &builder, /*start=*/1, /*count=*/1,
        loom_region_block(region, (block_index + 1) % 3),
        arguments[block_index].data(), arguments[block_index].size(),
        LOOM_LOCATION_UNKNOWN, &branch));
  }

  loom_local_value_domain_t domain = {};
  IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region(
      module_, region, &module_->arena, &domain));
  loom_value_ordinal_t* origins = nullptr;
  IREE_ASSERT_OK(loom_low_storage_identity_build(
      &domain, /*additional_value_capacity=*/0, &module_->arena, &origins));
  ASSERT_NE(origins, nullptr);

  const loom_value_ordinal_t identity_origin =
      origins[loom_local_value_domain_ordinal(&domain, arguments[0][1])];
  for (const auto& block_arguments : arguments) {
    const loom_value_ordinal_t prefix =
        loom_local_value_domain_ordinal(&domain, block_arguments[0]);
    const loom_value_ordinal_t identity =
        loom_local_value_domain_ordinal(&domain, block_arguments[1]);
    const loom_value_ordinal_t suffix =
        loom_local_value_domain_ordinal(&domain, block_arguments[2]);
    EXPECT_EQ(origins[prefix], prefix);
    EXPECT_EQ(origins[identity], identity_origin);
    EXPECT_EQ(origins[suffix], suffix);
  }
  loom_local_value_domain_release(&domain);
}

}  // namespace
}  // namespace loom
