// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/register_class_map.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/target/arch/wasm/descriptors.h"

namespace loom {
namespace {

std::string DescriptorString(const loom_low_descriptor_set_t* descriptor_set,
                             loom_bstring_table_offset_t string_offset) {
  iree_string_view_t value =
      loom_low_descriptor_set_string(descriptor_set, string_offset);
  return std::string(value.data, value.size);
}

TEST(WasmRegisterClassMapTest, ResolvesRealDescriptorIds) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  loom_context_t context;
  loom_context_initialize(iree_allocator_system(), &context);
  IREE_ASSERT_OK(loom_context_finalize(&context));

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(loom_module_allocate(&context, IREE_SV("wasm-test"),
                                      &block_pool, nullptr,
                                      iree_allocator_system(), &module));
  ASSERT_NE(module, nullptr);
  loom_string_id_t v128_string_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module, IREE_SV("wasm.v128"), &v128_string_id));

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_wasm_core_simd128_descriptor_set();
  loom_low_register_class_map_t map = {0};
  IREE_ASSERT_OK(loom_low_register_class_map_initialize(module, descriptor_set,
                                                        &arena, &map));

  uint16_t descriptor_register_class_id = LOOM_LOW_REG_CLASS_NONE;
  const loom_low_reg_class_t* descriptor_register_class = nullptr;
  bool found = false;
  IREE_ASSERT_OK(loom_low_register_class_map_try_resolve_string_id(
      &map, v128_string_id, &descriptor_register_class_id,
      &descriptor_register_class, &found));
  ASSERT_TRUE(found);
  EXPECT_EQ(descriptor_register_class_id,
            WASM_CORE_SIMD128_REG_CLASS_ID_WASM_V128);
  ASSERT_NE(descriptor_register_class, nullptr);
  EXPECT_EQ(DescriptorString(descriptor_set,
                             descriptor_register_class->name_string_offset),
            "wasm.v128");

  iree_arena_deinitialize(&arena);
  loom_module_free(module);
  loom_context_deinitialize(&context);
  iree_arena_block_pool_deinitialize(&block_pool);
}

}  // namespace
}  // namespace loom
