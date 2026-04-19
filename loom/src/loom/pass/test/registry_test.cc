// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/test/registry.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

namespace loom {
namespace {

static const loom_pass_descriptor_t* LookupTestPass(iree_string_view_t name) {
  const loom_pass_descriptor_t* descriptor = nullptr;
  IREE_EXPECT_OK(
      loom_pass_registry_lookup(loom_test_pass_registry(), name, &descriptor));
  return descriptor;
}

static iree_status_t configure_trace(
    void* user_data, const loom_pass_pipeline_descriptor_entry_t* entry,
    void** out_pass_user_data) {
  (void)entry;
  *out_pass_user_data = user_data;
  return iree_ok_status();
}

TEST(PassTestRegistryTest, Verifies) {
  IREE_ASSERT_OK(loom_pass_registry_verify(loom_test_pass_registry()));
}

TEST(PassTestRegistryTest, ContainsExpectedPasses) {
  const loom_pass_registry_t* registry = loom_test_pass_registry();
  const iree_string_view_t expected_keys[] = {
      IREE_SV("test.fail"),        IREE_SV("test.mark-changed"),
      IREE_SV("test.module-noop"), IREE_SV("test.noop"),
      IREE_SV("test.options"),     IREE_SV("test.required"),
      IREE_SV("test.unavailable"),
  };
  ASSERT_EQ(registry->descriptor_count, IREE_ARRAYSIZE(expected_keys));
  for (iree_host_size_t i = 0; i < registry->descriptor_count; ++i) {
    EXPECT_TRUE(
        iree_string_view_equal(registry->descriptors[i].key, expected_keys[i]))
        << "unexpected test pass registry entry " << i;
  }
}

TEST(PassTestRegistryTest, ValidatesOptions) {
  const loom_pass_descriptor_t* descriptor =
      LookupTestPass(IREE_SV("test.options"));
  ASSERT_NE(descriptor, nullptr);

  IREE_ASSERT_OK(loom_pass_descriptor_validate_options(
      descriptor, IREE_SV("count=4,mode=beta,string=payload")));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_descriptor_validate_options(descriptor, IREE_SV("count=0")));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_descriptor_validate_options(descriptor, IREE_SV("mode=gamma")));
}

TEST(PassTestRegistryTest, ValidatesRequiredOptions) {
  const loom_pass_descriptor_t* descriptor =
      LookupTestPass(IREE_SV("test.required"));
  ASSERT_NE(descriptor, nullptr);

  IREE_ASSERT_OK(loom_pass_descriptor_validate_options(
      descriptor, IREE_SV("required=value")));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_descriptor_validate_options(descriptor,
                                            iree_string_view_empty()));
}

TEST(PassTestRegistryTest, DecodesOptions) {
  const loom_pass_descriptor_t* descriptor =
      LookupTestPass(IREE_SV("test.options"));
  ASSERT_NE(descriptor, nullptr);

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  loom_pass_decoded_options_t decoded_options = {};
  IREE_ASSERT_OK(loom_pass_descriptor_decode_options(
      descriptor, IREE_SV("count=7,mode=beta,string=payload"), &arena,
      &decoded_options));

  ASSERT_EQ(decoded_options.option_count, 3u);
  EXPECT_EQ(decoded_options.options[0].uint32_value, 7u);
  EXPECT_EQ(decoded_options.options[1].enum_value_index, 1u);
  EXPECT_TRUE(iree_string_view_equal(decoded_options.options[2].string_value,
                                     IREE_SV("payload")));

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(PassTestRegistryTest, RunsThroughPassManager) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  loom_context_t context;
  loom_context_initialize(iree_allocator_system(), &context);
  IREE_ASSERT_OK(loom_context_finalize(&context));
  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(loom_module_allocate(&context, IREE_SV("test"), &block_pool,
                                      nullptr, iree_allocator_system(),
                                      &module));

  loom_pass_manager_t manager;
  IREE_ASSERT_OK(loom_pass_manager_initialize(
      &block_pool, 0, iree_allocator_system(), &manager));
  loom_test_pass_trace_t trace = {};
  IREE_ASSERT_OK(loom_pass_manager_add_pipeline(
      &manager, loom_test_pass_registry(), IREE_SV("test.module-noop"),
      (loom_pass_pipeline_configure_callback_t){
          .fn = configure_trace,
          .user_data = &trace,
      }));

  IREE_ASSERT_OK(loom_pass_manager_run(&manager, module));
  EXPECT_EQ(trace.module_noop_invocation_count, 1);

  loom_pass_manager_deinitialize(&manager);
  loom_module_free(module);
  loom_context_deinitialize(&context);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(PassTestRegistryTest, FailingPassPropagatesStatus) {
  const loom_pass_descriptor_t* descriptor =
      LookupTestPass(IREE_SV("test.fail"));
  ASSERT_NE(descriptor, nullptr);

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  loom_pass_manager_t manager;
  IREE_ASSERT_OK(loom_pass_manager_initialize(
      &block_pool, 0, iree_allocator_system(), &manager));

  IREE_ASSERT_OK(loom_pass_manager_add_descriptor(
      &manager, descriptor, iree_string_view_empty(), nullptr));
  loom_context_t context;
  loom_context_initialize(iree_allocator_system(), &context);
  IREE_ASSERT_OK(loom_context_finalize(&context));
  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(loom_module_allocate(&context, IREE_SV("test"), &block_pool,
                                      nullptr, iree_allocator_system(),
                                      &module));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL,
                        loom_pass_manager_run(&manager, module));

  loom_module_free(module);
  loom_context_deinitialize(&context);
  loom_pass_manager_deinitialize(&manager);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(PassTestRegistryTest, UnavailablePassIsKnownButCannotBeAdded) {
  const loom_pass_descriptor_t* descriptor =
      LookupTestPass(IREE_SV("test.unavailable"));
  ASSERT_NE(descriptor, nullptr);
  EXPECT_FALSE(loom_pass_descriptor_is_available(descriptor));

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  loom_pass_manager_t manager;
  IREE_ASSERT_OK(loom_pass_manager_initialize(
      &block_pool, 0, iree_allocator_system(), &manager));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_pass_manager_add_descriptor(
                            &manager, descriptor, iree_string_view_empty(),
                            nullptr));

  loom_pass_manager_deinitialize(&manager);
  iree_arena_block_pool_deinitialize(&block_pool);
}

}  // namespace
}  // namespace loom
