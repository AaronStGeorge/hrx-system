// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/test/registry.h"

#include <string.h>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/pass/ops.h"
#include "loom/pass/tooling.h"

namespace loom {
namespace {

static const loom_pass_descriptor_t* LookupTestPass(iree_string_view_t name) {
  const loom_pass_descriptor_t* descriptor = nullptr;
  IREE_EXPECT_OK(
      loom_pass_registry_lookup(loom_test_pass_registry(), name, &descriptor));
  return descriptor;
}

static iree_status_t configure_trace(
    void* user_data, const loom_pass_program_instruction_t* instruction,
    void** out_pass_user_data) {
  (void)instruction;
  *out_pass_user_data = user_data;
  return iree_ok_status();
}

typedef struct test_module_storage_t {
  // Block pool used by module allocation and pass execution.
  iree_arena_block_pool_t block_pool;
  // Context containing the pass dialect required by synthetic pipelines.
  loom_context_t context;
  // Empty target module used by module-pass tests.
  loom_module_t* module;
} test_module_storage_t;

static void initialize_test_module_storage(test_module_storage_t* storage) {
  memset(storage, 0, sizeof(*storage));
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &storage->block_pool);
  loom_context_initialize(iree_allocator_system(), &storage->context);
  iree_host_size_t pass_count = 0;
  const loom_op_vtable_t* const* pass_vtables =
      loom_pass_dialect_vtables(&pass_count);
  IREE_ASSERT_OK(loom_context_register_dialect(&storage->context,
                                               LOOM_DIALECT_PASS, pass_vtables,
                                               (uint16_t)pass_count));
  IREE_ASSERT_OK(loom_context_finalize(&storage->context));
  IREE_ASSERT_OK(loom_module_allocate(
      &storage->context, IREE_SV("test"), &storage->block_pool, nullptr,
      iree_allocator_system(), &storage->module));
}

static void deinitialize_test_module_storage(test_module_storage_t* storage) {
  if (storage->module) {
    loom_module_free(storage->module);
  }
  loom_context_deinitialize(&storage->context);
  iree_arena_block_pool_deinitialize(&storage->block_pool);
}

TEST(PassTestRegistryTest, Verifies) {
  IREE_ASSERT_OK(loom_pass_registry_verify(loom_test_pass_registry()));
}

TEST(PassTestRegistryTest, ContainsExpectedPasses) {
  const loom_pass_registry_t* registry = loom_test_pass_registry();
  const iree_string_view_t expected_keys[] = {
      IREE_SV("test.fail"),
      IREE_SV("test.mark-changed"),
      IREE_SV("test.module-noop"),
      IREE_SV("test.noop"),
      IREE_SV("test.options"),
      IREE_SV("test.required"),
      IREE_SV("test.requires-target"),
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
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_descriptor_validate_options(
                            descriptor, iree_string_view_empty()));
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

TEST(PassTestRegistryTest, RunsThroughToolingInterpreter) {
  test_module_storage_t storage;
  initialize_test_module_storage(&storage);
  loom_test_pass_trace_t trace = {};
  loom_pass_tool_run_options_t options = {
      .registry = loom_test_pass_registry(),
      .block_pool = &storage.block_pool,
      .configure =
          {
              .fn = configure_trace,
              .user_data = &trace,
          },
  };
  IREE_ASSERT_OK(loom_pass_tool_run_flat_pipeline(
      storage.module, IREE_SV("test.module-noop"), &options));

  EXPECT_EQ(trace.module_noop_invocation_count, 1);

  deinitialize_test_module_storage(&storage);
}

TEST(PassTestRegistryTest, FailingPassPropagatesStatusThroughTooling) {
  test_module_storage_t storage;
  initialize_test_module_storage(&storage);
  loom_pass_tool_run_options_t options = {
      .registry = loom_test_pass_registry(),
      .block_pool = &storage.block_pool,
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL,
                        loom_pass_tool_run_flat_pipeline(
                            storage.module, IREE_SV("test.fail"), &options));

  deinitialize_test_module_storage(&storage);
}

TEST(PassTestRegistryTest, UnavailablePassFailsThroughTooling) {
  const loom_pass_descriptor_t* descriptor =
      LookupTestPass(IREE_SV("test.unavailable"));
  ASSERT_NE(descriptor, nullptr);
  EXPECT_FALSE(loom_pass_descriptor_is_available(descriptor));

  test_module_storage_t storage;
  initialize_test_module_storage(&storage);
  loom_pass_tool_run_options_t options = {
      .registry = loom_test_pass_registry(),
      .block_pool = &storage.block_pool,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_pass_tool_run_flat_pipeline(storage.module,
                                       IREE_SV("test.unavailable"), &options));

  deinitialize_test_module_storage(&storage);
}

}  // namespace
}  // namespace loom
