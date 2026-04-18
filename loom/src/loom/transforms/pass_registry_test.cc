// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/pass_registry.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/transforms/dce.h"

namespace loom {
namespace {

TEST(PassRegistryTest, BuiltinRegistryVerifies) {
  IREE_ASSERT_OK(loom_pass_registry_verify(loom_pass_builtin_registry()));
}

TEST(PassRegistryTest, BuiltinRegistryIsSorted) {
  const loom_pass_registry_t* registry = loom_pass_builtin_registry();
  ASSERT_GT(registry->descriptor_count, 0u);
  for (iree_host_size_t i = 1; i < registry->descriptor_count; ++i) {
    EXPECT_LT(iree_string_view_compare(registry->descriptors[i - 1].key,
                                       registry->descriptors[i].key),
              0)
        << "pass registry entry " << i << " is out of order";
  }
}

TEST(PassRegistryTest, LookupKnownAndUnknownPasses) {
  const loom_pass_descriptor_t* descriptor = nullptr;
  IREE_ASSERT_OK(loom_pass_registry_lookup(
      loom_pass_builtin_registry(), IREE_SV("canonicalize"), &descriptor));
  ASSERT_NE(descriptor, nullptr);
  EXPECT_TRUE(iree_string_view_equal(descriptor->key, IREE_SV("canonicalize")));
  ASSERT_NE(descriptor->info, nullptr);
  EXPECT_EQ(descriptor->info()->kind, LOOM_PASS_FUNCTION);
  EXPECT_NE(descriptor->function_run, nullptr);
  EXPECT_NE(descriptor->create, nullptr);

  descriptor = reinterpret_cast<const loom_pass_descriptor_t*>(0x1);
  IREE_ASSERT_OK(loom_pass_registry_lookup(loom_pass_builtin_registry(),
                                           IREE_SV("definitely-not-a-pass"),
                                           &descriptor));
  EXPECT_EQ(descriptor, nullptr);
}

TEST(PassRegistryTest, UnavailableDescriptorCannotBeAdded) {
  loom_pass_descriptor_t descriptor = {
      .key = IREE_SVL("dce"),
      .info = loom_dce_pass_info,
      .function_run = loom_dce_run,
      .flags = LOOM_PASS_DESCRIPTOR_UNAVAILABLE,
      .unavailable_reason = IREE_SVL("disabled for test"),
  };

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  loom_pass_manager_t manager;
  IREE_ASSERT_OK(loom_pass_manager_initialize(
      &block_pool, 0, iree_allocator_system(), &manager));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_pass_manager_add_descriptor(
                            &manager, &descriptor, iree_string_view_empty(),
                            /*user_data=*/nullptr));

  loom_pass_manager_deinitialize(&manager);
  iree_arena_block_pool_deinitialize(&block_pool);
}

}  // namespace
}  // namespace loom
