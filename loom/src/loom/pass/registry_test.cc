// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/registry.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/pass/environment.h"
#include "loom/pass/registry_verify.h"
#include "loom/pass/report.h"
#include "loom/pass/test/registry.h"

namespace loom {
namespace {

static const loom_pass_descriptor_t* LookupTestPass(iree_string_view_t name) {
  const loom_pass_descriptor_t* descriptor = nullptr;
  IREE_EXPECT_OK(
      loom_pass_registry_lookup(loom_test_pass_registry(), name, &descriptor));
  return descriptor;
}

static const loom_pass_info_t* BrokenStatisticsPassInfo() {
  static const loom_pass_info_t kInfo = {
      .name = IREE_SVL("test.broken-statistics"),
      .description = IREE_SVL("Synthetic pass with malformed statistics."),
      .kind = LOOM_PASS_MODULE,
      .statistic_count = 1,
  };
  return &kInfo;
}

static bool SatisfyTestRequirement(
    const loom_pass_environment_capability_t* capability,
    iree_string_view_t requirement) {
  (void)capability;
  (void)requirement;
  return true;
}

static const loom_pass_environment_capability_type_t kTestRequirementType = {
    .name = IREE_SVL("test.requirements"),
    .satisfies_requirement = SatisfyTestRequirement,
};

TEST(PassRegistryCoreTest, SyntheticRegistryVerifies) {
  IREE_ASSERT_OK(loom_pass_registry_verify(loom_test_pass_registry()));
}

TEST(PassRegistryCoreTest, FormatsRegistryMetadataJson) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);

  IREE_EXPECT_OK(loom_pass_report_format_registry_json(
      loom_test_pass_registry(), &stream));

  iree_string_view_t json = iree_string_builder_view(&builder);
  std::string text(json.data, json.size);
  EXPECT_NE(text.find("\"key\":\"test.options\""), std::string::npos);
  EXPECT_NE(text.find("\"kind\":\"uint32\""), std::string::npos);
  EXPECT_NE(text.find("\"minimum\":1"), std::string::npos);
  EXPECT_NE(text.find("\"values\":[\"alpha\",\"beta\"]"), std::string::npos);
  EXPECT_NE(text.find("\"required\":true"), std::string::npos);
  EXPECT_NE(text.find("\"key\":\"test.unavailable\""), std::string::npos);
  EXPECT_NE(text.find("\"available\":false"), std::string::npos);
  EXPECT_NE(text.find("\"unavailable_reason\":\"disabled for test\""),
            std::string::npos);
  EXPECT_NE(text.find("\"requirements\":[{\"key\":\"target.profile\""),
            std::string::npos);
  EXPECT_NE(text.find("\"capability\":\"test.target-profile\""),
            std::string::npos);

  iree_string_builder_deinitialize(&builder);
}

TEST(PassRegistryCoreTest, RejectsMissingStatisticMetadata) {
  loom_pass_descriptor_t descriptor =
      *LookupTestPass(IREE_SV("test.module-noop"));
  descriptor.key = IREE_SVL("test.broken-statistics");
  descriptor.info = BrokenStatisticsPassInfo;
  const loom_pass_registry_t registry = {
      .descriptors = &descriptor,
      .descriptor_count = 1,
  };

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_registry_verify(&registry));
}

TEST(PassRegistryCoreTest, VerifiesRequirementMetadata) {
  const loom_pass_requirement_def_t requirements[] = {
      {
          .capability_type = &kTestRequirementType,
          .key = IREE_SVL("analysis.liveness"),
          .description = IREE_SVL("Requires precomputed liveness."),
      },
      {
          .capability_type = &kTestRequirementType,
          .key = IREE_SVL("target.low-descriptor-registry"),
          .description = IREE_SVL("Requires a target-low descriptor registry."),
      },
  };
  loom_pass_descriptor_t descriptor =
      *LookupTestPass(IREE_SV("test.module-noop"));
  descriptor.requirement_defs = requirements;
  descriptor.requirement_count = IREE_ARRAYSIZE(requirements);
  const loom_pass_registry_t registry = {
      .descriptors = &descriptor,
      .descriptor_count = 1,
  };

  IREE_ASSERT_OK(loom_pass_registry_verify(&registry));
}

TEST(PassRegistryCoreTest, RejectsRequirementWithoutCapabilityType) {
  const loom_pass_requirement_def_t requirements[] = {
      {
          .key = IREE_SVL("analysis.liveness"),
          .description = IREE_SVL("Requires precomputed liveness."),
      },
  };
  loom_pass_descriptor_t descriptor =
      *LookupTestPass(IREE_SV("test.module-noop"));
  descriptor.requirement_defs = requirements;
  descriptor.requirement_count = IREE_ARRAYSIZE(requirements);
  const loom_pass_registry_t registry = {
      .descriptors = &descriptor,
      .descriptor_count = 1,
  };

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_registry_verify(&registry));
}

TEST(PassRegistryCoreTest, RejectsUnsortedRequirementMetadata) {
  const loom_pass_requirement_def_t requirements[] = {
      {
          .capability_type = &kTestRequirementType,
          .key = IREE_SVL("target.low-descriptor-registry"),
          .description = IREE_SVL("Requires a target-low descriptor registry."),
      },
      {
          .capability_type = &kTestRequirementType,
          .key = IREE_SVL("analysis.liveness"),
          .description = IREE_SVL("Requires precomputed liveness."),
      },
  };
  loom_pass_descriptor_t descriptor =
      *LookupTestPass(IREE_SV("test.module-noop"));
  descriptor.requirement_defs = requirements;
  descriptor.requirement_count = IREE_ARRAYSIZE(requirements);
  const loom_pass_registry_t registry = {
      .descriptors = &descriptor,
      .descriptor_count = 1,
  };

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_registry_verify(&registry));
}

TEST(PassRegistryCoreTest, LookupKnownAndUnknownPasses) {
  const loom_pass_descriptor_t* descriptor = nullptr;
  IREE_ASSERT_OK(loom_pass_registry_lookup(
      loom_test_pass_registry(), IREE_SV("test.options"), &descriptor));
  ASSERT_NE(descriptor, nullptr);
  EXPECT_TRUE(iree_string_view_equal(descriptor->key, IREE_SV("test.options")));
  ASSERT_NE(descriptor->info, nullptr);
  EXPECT_EQ(descriptor->info()->kind, LOOM_PASS_FUNCTION);
  EXPECT_NE(descriptor->function_run, nullptr);
  EXPECT_NE(descriptor->create, nullptr);

  descriptor = reinterpret_cast<const loom_pass_descriptor_t*>(0x1);
  IREE_ASSERT_OK(loom_pass_registry_lookup(loom_test_pass_registry(),
                                           IREE_SV("definitely-not-a-pass"),
                                           &descriptor));
  EXPECT_EQ(descriptor, nullptr);
}

TEST(PassRegistryCoreTest, ValidatesUint32OptionSchema) {
  const loom_pass_descriptor_t* descriptor =
      LookupTestPass(IREE_SV("test.options"));
  ASSERT_NE(descriptor, nullptr);

  IREE_ASSERT_OK(
      loom_pass_descriptor_validate_options(descriptor, IREE_SV("count=4")));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_descriptor_validate_options(descriptor, IREE_SV("count=0")));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_descriptor_validate_options(descriptor, IREE_SV("count=many")));
}

TEST(PassRegistryCoreTest, ValidatesEnumAndStringOptionSchema) {
  const loom_pass_descriptor_t* descriptor =
      LookupTestPass(IREE_SV("test.options"));
  ASSERT_NE(descriptor, nullptr);

  IREE_ASSERT_OK(loom_pass_descriptor_validate_options(
      descriptor, IREE_SV("mode=alpha,string=payload")));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_descriptor_validate_options(descriptor, IREE_SV("mode=gamma")));
}

TEST(PassRegistryCoreTest, ValidatesRequiredOptionSchema) {
  const loom_pass_descriptor_t* descriptor =
      LookupTestPass(IREE_SV("test.required"));
  ASSERT_NE(descriptor, nullptr);

  IREE_ASSERT_OK(loom_pass_descriptor_validate_options(
      descriptor, IREE_SV("required=value")));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_descriptor_validate_options(
                            descriptor, iree_string_view_empty()));
}

TEST(PassRegistryCoreTest, RejectsUnknownAndDuplicateOptions) {
  const loom_pass_descriptor_t* descriptor =
      LookupTestPass(IREE_SV("test.options"));
  ASSERT_NE(descriptor, nullptr);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_descriptor_validate_options(descriptor, IREE_SV("unknown=1")));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_descriptor_validate_options(
                            descriptor, IREE_SV("count=1,count=2")));
}

TEST(PassRegistryCoreTest, DecodesTypedOptions) {
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

  EXPECT_EQ(decoded_options.descriptor, descriptor);
  ASSERT_EQ(decoded_options.option_count, 3u);
  ASSERT_NE(decoded_options.options, nullptr);
  EXPECT_TRUE(decoded_options.options[0].present);
  EXPECT_EQ(decoded_options.options[0].uint32_value, 7u);
  EXPECT_TRUE(decoded_options.options[1].present);
  EXPECT_EQ(decoded_options.options[1].enum_value_index, 1u);
  EXPECT_TRUE(decoded_options.options[2].present);
  EXPECT_TRUE(iree_string_view_equal(decoded_options.options[2].string_value,
                                     IREE_SV("payload")));

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(PassRegistryCoreTest, DecodedOptionalOptionsTrackAbsence) {
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
      descriptor, iree_string_view_empty(), &arena, &decoded_options));

  ASSERT_EQ(decoded_options.option_count, 3u);
  ASSERT_NE(decoded_options.options, nullptr);
  EXPECT_FALSE(decoded_options.options[0].present);
  EXPECT_FALSE(decoded_options.options[1].present);
  EXPECT_FALSE(decoded_options.options[2].present);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(PassRegistryCoreTest, DecodeRejectsInvalidOptions) {
  const loom_pass_descriptor_t* descriptor =
      LookupTestPass(IREE_SV("test.options"));
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
          descriptor, IREE_SV("count=1,count=2"), &arena, &decoded_options));

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(PassRegistryCoreTest, RejectsOptionsForDescriptorWithoutCreateCallback) {
  const loom_pass_descriptor_t* descriptor =
      LookupTestPass(IREE_SV("test.module-noop"));
  ASSERT_NE(descriptor, nullptr);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_descriptor_validate_options(descriptor, IREE_SV("unknown=1")));
}

TEST(PassRegistryCoreTest, UnavailableDescriptorKeepsAvailabilityMetadata) {
  const loom_pass_descriptor_t* descriptor =
      LookupTestPass(IREE_SV("test.unavailable"));
  ASSERT_NE(descriptor, nullptr);
  EXPECT_FALSE(loom_pass_descriptor_is_available(descriptor));
  EXPECT_TRUE(iree_string_view_equal(descriptor->unavailable_reason,
                                     IREE_SV("disabled for test")));
}

}  // namespace
}  // namespace loom
