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

static const loom_pass_descriptor_t* LookupBuiltinPass(
    iree_string_view_t name) {
  const loom_pass_descriptor_t* descriptor = nullptr;
  IREE_EXPECT_OK(loom_pass_registry_lookup(loom_pass_builtin_registry(), name,
                                           &descriptor));
  return descriptor;
}

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

TEST(PassRegistryTest, ValidatesUint32OptionSchema) {
  const loom_pass_descriptor_t* descriptor =
      LookupBuiltinPass(IREE_SV("canonicalize"));
  ASSERT_NE(descriptor, nullptr);

  IREE_ASSERT_OK(loom_pass_descriptor_validate_options(
      descriptor, IREE_SV("max-iterations=4")));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_descriptor_validate_options(
                            descriptor, IREE_SV("max-iterations=0")));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_descriptor_validate_options(
                            descriptor, IREE_SV("max-iterations=many")));
}

TEST(PassRegistryTest, RejectsUnknownAndDuplicateOptions) {
  const loom_pass_descriptor_t* descriptor =
      LookupBuiltinPass(IREE_SV("canonicalize"));
  ASSERT_NE(descriptor, nullptr);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_descriptor_validate_options(descriptor, IREE_SV("unknown=1")));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_descriptor_validate_options(
          descriptor, IREE_SV("max-iterations=1,max-iterations=2")));
}

TEST(PassRegistryTest, ValidatesEnumOptionSchema) {
  const loom_pass_descriptor_t* descriptor =
      LookupBuiltinPass(IREE_SV("low-materialize-allocation"));
  ASSERT_NE(descriptor, nullptr);

  IREE_ASSERT_OK(loom_pass_descriptor_validate_options(
      descriptor, IREE_SV("diagnostics=spills")));
  IREE_ASSERT_OK(loom_pass_descriptor_validate_options(
      descriptor, IREE_SV("budgets=vm.i32=2;vm.ref=1")));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_descriptor_validate_options(
                            descriptor, IREE_SV("diagnostics=verbose")));
}

TEST(PassRegistryTest, DecodesTypedOptions) {
  const loom_pass_descriptor_t* descriptor =
      LookupBuiltinPass(IREE_SV("canonicalize"));
  ASSERT_NE(descriptor, nullptr);

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  loom_pass_decoded_options_t decoded_options = {};
  IREE_ASSERT_OK(loom_pass_descriptor_decode_options(
      descriptor, IREE_SV("max-iterations=7"), &arena, &decoded_options));

  EXPECT_EQ(decoded_options.descriptor, descriptor);
  ASSERT_EQ(decoded_options.option_count, 1u);
  ASSERT_NE(decoded_options.options, nullptr);
  EXPECT_TRUE(decoded_options.options[0].present);
  EXPECT_EQ(decoded_options.options[0].schema->kind,
            LOOM_PASS_OPTION_SCHEMA_UINT32);
  EXPECT_EQ(decoded_options.options[0].uint32_value, 7u);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(PassRegistryTest, DecodesStringAndEnumOptions) {
  const loom_pass_descriptor_t* descriptor =
      LookupBuiltinPass(IREE_SV("low-materialize-allocation"));
  ASSERT_NE(descriptor, nullptr);

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  loom_pass_decoded_options_t decoded_options = {};
  IREE_ASSERT_OK(loom_pass_descriptor_decode_options(
      descriptor, IREE_SV("budgets=vm.i32=2;vm.ref=1,diagnostics=spills"),
      &arena, &decoded_options));

  ASSERT_EQ(decoded_options.option_count, 2u);
  ASSERT_NE(decoded_options.options, nullptr);
  EXPECT_TRUE(decoded_options.options[0].present);
  EXPECT_TRUE(iree_string_view_equal(decoded_options.options[0].string_value,
                                     IREE_SV("vm.i32=2;vm.ref=1")));
  EXPECT_TRUE(decoded_options.options[1].present);
  EXPECT_EQ(decoded_options.options[1].enum_value_index, 1u);
  EXPECT_TRUE(iree_string_view_equal(
      decoded_options.options[1]
          .schema->enum_values[decoded_options.options[1].enum_value_index]
          .value,
      IREE_SV("spills")));

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(PassRegistryTest, DecodedOptionalOptionsTrackAbsence) {
  const loom_pass_descriptor_t* descriptor =
      LookupBuiltinPass(IREE_SV("canonicalize"));
  ASSERT_NE(descriptor, nullptr);

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  loom_pass_decoded_options_t decoded_options = {};
  IREE_ASSERT_OK(loom_pass_descriptor_decode_options(
      descriptor, iree_string_view_empty(), &arena, &decoded_options));

  ASSERT_EQ(decoded_options.option_count, 1u);
  ASSERT_NE(decoded_options.options, nullptr);
  EXPECT_FALSE(decoded_options.options[0].present);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(PassRegistryTest, DecodeRejectsInvalidOptions) {
  const loom_pass_descriptor_t* descriptor =
      LookupBuiltinPass(IREE_SV("canonicalize"));
  ASSERT_NE(descriptor, nullptr);

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  loom_pass_decoded_options_t decoded_options = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_descriptor_decode_options(
          descriptor, IREE_SV("max-iterations=1,max-iterations=2"), &arena,
          &decoded_options));

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(PassRegistryTest, RejectsOptionsForDescriptorWithoutCreateCallback) {
  const loom_pass_descriptor_t* descriptor = LookupBuiltinPass(IREE_SV("dce"));
  ASSERT_NE(descriptor, nullptr);

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  loom_pass_manager_t manager;
  IREE_ASSERT_OK(loom_pass_manager_initialize(
      &block_pool, 0, iree_allocator_system(), &manager));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_manager_add_descriptor(
                            &manager, descriptor, IREE_SV("unknown=1"),
                            /*user_data=*/nullptr));

  loom_pass_manager_deinitialize(&manager);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(PassRegistryTest, AddPipelineResolvesDescriptors) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  loom_pass_manager_t manager;
  IREE_ASSERT_OK(loom_pass_manager_initialize(
      &block_pool, 0, iree_allocator_system(), &manager));

  loom_pass_pipeline_configure_callback_t no_config = {};
  IREE_ASSERT_OK(loom_pass_manager_add_pipeline(
      &manager, loom_pass_builtin_registry(),
      IREE_SV("canonicalize{max-iterations=2},dce"), no_config));

  ASSERT_EQ(manager.count, 2u);
  EXPECT_TRUE(iree_string_view_equal(manager.entries[0].info->name,
                                     IREE_SV("canonicalize")));
  EXPECT_TRUE(iree_string_view_equal(manager.entries[0].options,
                                     IREE_SV("max-iterations=2")));
  EXPECT_TRUE(
      iree_string_view_equal(manager.entries[1].info->name, IREE_SV("dce")));

  loom_pass_manager_deinitialize(&manager);
  iree_arena_block_pool_deinitialize(&block_pool);
}

typedef struct test_pipeline_config_t {
  // User data pointer to attach to matching pipeline entries.
  void* entry_user_data;
} test_pipeline_config_t;

static iree_status_t configure_test_pipeline_entry(
    void* user_data, const loom_pass_pipeline_descriptor_entry_t* entry,
    void** out_pass_user_data) {
  test_pipeline_config_t* config = (test_pipeline_config_t*)user_data;
  *out_pass_user_data = nullptr;
  if (iree_string_view_equal(entry->descriptor->key, IREE_SV("dce"))) {
    *out_pass_user_data = config->entry_user_data;
  }
  return iree_ok_status();
}

TEST(PassRegistryTest, AddPipelineAppliesConfigureCallback) {
  int sentinel = 0;
  test_pipeline_config_t config = {
      .entry_user_data = &sentinel,
  };
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  loom_pass_manager_t manager;
  IREE_ASSERT_OK(loom_pass_manager_initialize(
      &block_pool, 0, iree_allocator_system(), &manager));

  IREE_ASSERT_OK(loom_pass_manager_add_pipeline(
      &manager, loom_pass_builtin_registry(), IREE_SV("dce"),
      (loom_pass_pipeline_configure_callback_t){
          .fn = configure_test_pipeline_entry,
          .user_data = &config,
      }));

  ASSERT_EQ(manager.count, 1u);
  EXPECT_EQ(manager.entries[0].user_data, &sentinel);

  loom_pass_manager_deinitialize(&manager);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(PassRegistryTest, AddPipelineRejectsUnknownPass) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  loom_pass_manager_t manager;
  IREE_ASSERT_OK(loom_pass_manager_initialize(
      &block_pool, 0, iree_allocator_system(), &manager));

  loom_pass_pipeline_configure_callback_t no_config = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_manager_add_pipeline(
                            &manager, loom_pass_builtin_registry(),
                            IREE_SV("definitely-not-a-pass"), no_config));

  loom_pass_manager_deinitialize(&manager);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(PassRegistryTest, AddPipelineRollsBackOnFailure) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  loom_pass_manager_t manager;
  IREE_ASSERT_OK(loom_pass_manager_initialize(
      &block_pool, 0, iree_allocator_system(), &manager));

  loom_pass_pipeline_configure_callback_t no_config = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_manager_add_pipeline(
                            &manager, loom_pass_builtin_registry(),
                            IREE_SV("dce,definitely-not-a-pass"), no_config));
  EXPECT_EQ(manager.count, 0u);

  loom_pass_manager_deinitialize(&manager);
  iree_arena_block_pool_deinitialize(&block_pool);
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
