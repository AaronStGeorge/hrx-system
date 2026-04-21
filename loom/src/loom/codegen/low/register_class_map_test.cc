// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/register_class_map.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/target/test/descriptors.h"

namespace loom {
namespace {

std::string DescriptorString(const loom_low_descriptor_set_t* descriptor_set,
                             loom_bstring_table_offset_t string_offset) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(
      loom_low_descriptor_set_string(descriptor_set, string_offset, &value));
  return std::string(value.data, value.size);
}

class RegisterClassMapTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* AllocateModule() {
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module));
    EXPECT_NE(module, nullptr);
    return module;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(RegisterClassMapTest, ResolvesModuleStringIdsToDescriptorIds) {
  loom_module_t* module = AllocateModule();
  ASSERT_NE(module, nullptr);
  loom_string_id_t i32_string_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("test.i32"), &i32_string_id));

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_register_class_map_t map = {0};
  const loom_low_descriptor_set_t* descriptor_set =
      loom_test_low_core_descriptor_set();
  IREE_ASSERT_OK(loom_low_register_class_map_initialize(module, descriptor_set,
                                                        &arena, &map));

  uint16_t descriptor_register_class_id = LOOM_LOW_REG_CLASS_NONE;
  const loom_low_reg_class_t* descriptor_register_class = nullptr;
  bool found = false;
  IREE_ASSERT_OK(loom_low_register_class_map_try_resolve_string_id(
      &map, i32_string_id, &descriptor_register_class_id,
      &descriptor_register_class, &found));
  ASSERT_TRUE(found);
  ASSERT_NE(descriptor_register_class, nullptr);
  EXPECT_NE(descriptor_register_class_id, LOOM_LOW_REG_CLASS_NONE);
  EXPECT_EQ(DescriptorString(descriptor_set,
                             descriptor_register_class->name_string_offset),
            "test.i32");

  iree_arena_deinitialize(&arena);
  loom_module_free(module);
}

TEST_F(RegisterClassMapTest, RejectsUnknownAndNonRegisterTypes) {
  loom_module_t* module = AllocateModule();
  ASSERT_NE(module, nullptr);
  loom_string_id_t unknown_string_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("test.unknown"),
                                           &unknown_string_id));

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_register_class_map_t map = {0};
  const loom_low_descriptor_set_t* descriptor_set =
      loom_test_low_core_descriptor_set();
  IREE_ASSERT_OK(loom_low_register_class_map_initialize(module, descriptor_set,
                                                        &arena, &map));

  uint16_t descriptor_register_class_id = LOOM_LOW_REG_CLASS_NONE;
  const loom_low_reg_class_t* descriptor_register_class = nullptr;
  bool found = true;
  IREE_ASSERT_OK(loom_low_register_class_map_try_resolve_type(
      &map, loom_type_register(unknown_string_id, 1),
      &descriptor_register_class_id, &descriptor_register_class, &found));
  EXPECT_FALSE(found);
  EXPECT_EQ(descriptor_register_class_id, LOOM_LOW_REG_CLASS_NONE);
  EXPECT_EQ(descriptor_register_class, nullptr);

  found = true;
  IREE_ASSERT_OK(loom_low_register_class_map_try_resolve_type(
      &map, loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      &descriptor_register_class_id, &descriptor_register_class, &found));
  EXPECT_FALSE(found);
  EXPECT_EQ(descriptor_register_class_id, LOOM_LOW_REG_CLASS_NONE);
  EXPECT_EQ(descriptor_register_class, nullptr);

  iree_arena_deinitialize(&arena);
  loom_module_free(module);
}

TEST(RegisterClassLookupTest, LooksUpDescriptorNamesAtConfigurationBoundary) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_test_low_core_descriptor_set();
  uint16_t descriptor_register_class_id = LOOM_LOW_REG_CLASS_NONE;
  const loom_low_reg_class_t* descriptor_register_class = nullptr;
  bool found = false;
  IREE_ASSERT_OK(loom_low_register_class_try_lookup_name(
      descriptor_set, IREE_SV("test.v4i32"), &descriptor_register_class_id,
      &descriptor_register_class, &found));
  ASSERT_TRUE(found);
  ASSERT_NE(descriptor_register_class, nullptr);
  EXPECT_NE(descriptor_register_class_id, LOOM_LOW_REG_CLASS_NONE);
  EXPECT_EQ(DescriptorString(descriptor_set,
                             descriptor_register_class->name_string_offset),
            "test.v4i32");

  found = true;
  IREE_ASSERT_OK(loom_low_register_class_try_lookup_name(
      descriptor_set, IREE_SV("test.missing"), &descriptor_register_class_id,
      &descriptor_register_class, &found));
  EXPECT_FALSE(found);
  EXPECT_EQ(descriptor_register_class_id, LOOM_LOW_REG_CLASS_NONE);
  EXPECT_EQ(descriptor_register_class, nullptr);
}

}  // namespace
}  // namespace loom
