// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/global/rodata_pool.h"

#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/global/ops.h"

namespace loom {
namespace {

class RodataPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    IREE_ASSERT_OK(loom_rodata_pool_allocate(&arena_, &pool_));
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_global_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_GLOBAL,
                                                 vtables, (uint16_t)count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("constants"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Storage for the destination module and temporary interning arena.
  iree_arena_block_pool_t block_pool_;
  // Temporary pool storage released independently of the destination module.
  iree_arena_allocator_t arena_;
  // Pool owned by arena_.
  loom_rodata_pool_t* pool_ = nullptr;
  // Registry for emitted global definitions.
  loom_context_t context_;
  // Destination whose payloads must outlive pool_.
  loom_module_t* module_ = nullptr;
};

TEST_F(RodataPoolTest, MergesFormsAndAlignmentAcrossGrowth) {
  loom_rodata_id_t ids[40];
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(ids); ++i) {
    char bytes[] = {'a', (char)('A' + i), '\0', 'z'};
    loom_attribute_t contents =
        i & 1 ? loom_attr_utf8(iree_make_string_view(bytes, sizeof(bytes)))
              : loom_attr_bytes((const uint8_t*)bytes, sizeof(bytes));
    IREE_ASSERT_OK(loom_rodata_pool_intern(pool_, contents, 1,
                                           LOOM_LOCATION_UNKNOWN, &ids[i]));
    memset(bytes, 0, sizeof(bytes));
  }
  EXPECT_EQ(module_->symbols.count, 0u);
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(ids); ++i) {
    char bytes[] = {'a', (char)('A' + i), '\0', 'z'};
    loom_attribute_t contents =
        i & 1 ? loom_attr_bytes((const uint8_t*)bytes, sizeof(bytes))
              : loom_attr_utf8(iree_make_string_view(bytes, sizeof(bytes)));
    loom_rodata_id_t duplicate;
    IREE_ASSERT_OK(loom_rodata_pool_intern(pool_, contents, 32,
                                           LOOM_LOCATION_UNKNOWN, &duplicate));
    EXPECT_EQ(duplicate, ids[i]);
  }

  loom_symbol_ref_t symbols[IREE_ARRAYSIZE(ids)];
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(ids); ++i) {
    IREE_ASSERT_OK(
        loom_rodata_pool_reference(pool_, module_, ids[i], &symbols[i]));
  }
  IREE_ASSERT_OK(loom_rodata_pool_finalize(pool_, module_));
  // Emitted attributes own their data; no pass-local bytes survive in the IR.
  iree_arena_reset(&arena_);
  pool_ = nullptr;
  EXPECT_EQ(loom_module_block(module_)->op_count, IREE_ARRAYSIZE(ids));
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(ids); ++i) {
    const loom_op_t* definition =
        module_->symbols.entries[symbols[i].symbol_id].defining_op;
    EXPECT_EQ(loom_global_rodata_def_alignment(definition), 32);
    const loom_attribute_t contents =
        loom_global_rodata_def_contents_attr(definition);
    EXPECT_TRUE(loom_attr_is_utf8(contents));
    const auto bytes = loom_attr_as_bytes(contents);
    ASSERT_EQ(bytes.data_length, 4u);
    EXPECT_EQ(bytes.data[0], 'a');
    EXPECT_EQ(bytes.data[1], 'A' + i);
    EXPECT_EQ(bytes.data[2], 0);
    EXPECT_EQ(bytes.data[3], 'z');
  }
}

TEST_F(RodataPoolTest, EmptyPayloadKeepsExactExtentAndUtf8Form) {
  loom_rodata_id_t first;
  IREE_ASSERT_OK(loom_rodata_pool_intern(pool_, loom_attr_bytes(nullptr, 0), 1,
                                         LOOM_LOCATION_UNKNOWN, &first));
  loom_rodata_id_t duplicate;
  IREE_ASSERT_OK(
      loom_rodata_pool_intern(pool_, loom_attr_utf8(iree_string_view_empty()),
                              16, LOOM_LOCATION_UNKNOWN, &duplicate));
  EXPECT_EQ(first, duplicate);
  loom_symbol_ref_t symbol;
  IREE_ASSERT_OK(loom_rodata_pool_reference(pool_, module_, first, &symbol));
  IREE_ASSERT_OK(loom_rodata_pool_finalize(pool_, module_));
  const loom_op_t* definition =
      module_->symbols.entries[symbol.symbol_id].defining_op;
  EXPECT_EQ(loom_global_rodata_def_alignment(definition), 16);
  EXPECT_TRUE(
      loom_attr_is_utf8(loom_global_rodata_def_contents_attr(definition)));
  EXPECT_EQ(loom_global_rodata_def_contents(definition).data_length, 0u);
}

}  // namespace
}  // namespace loom
