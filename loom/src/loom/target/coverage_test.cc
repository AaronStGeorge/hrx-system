// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/coverage.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

constexpr loom_target_coverage_phase_flags_t kExecutablePhases =
    LOOM_TARGET_COVERAGE_PHASE_DESCRIPTOR |
    LOOM_TARGET_COVERAGE_PHASE_LOW_PARSE |
    LOOM_TARGET_COVERAGE_PHASE_SOURCE_LOWER |
    LOOM_TARGET_COVERAGE_PHASE_SCHEDULE | LOOM_TARGET_COVERAGE_PHASE_ALLOCATE |
    LOOM_TARGET_COVERAGE_PHASE_TEXT_EMIT |
    LOOM_TARGET_COVERAGE_PHASE_BINARY_ENCODE |
    LOOM_TARGET_COVERAGE_PHASE_ARTIFACT_WRAP |
    LOOM_TARGET_COVERAGE_PHASE_INSPECT | LOOM_TARGET_COVERAGE_PHASE_RUN;

const loom_target_coverage_row_t kRows[] = {
    {
        .target_key = IREE_SVL("sample"),
        .descriptor_set_key = IREE_SVL("sample.core"),
        .category = IREE_SVL("vector.arithmetic"),
        .semantic_tag = IREE_SVL("i32x4.add"),
        .expected_phases = kExecutablePhases,
        .supported_phases = kExecutablePhases,
    },
    {
        .target_key = IREE_SVL("sample"),
        .descriptor_set_key = IREE_SVL("sample.core"),
        .category = IREE_SVL("artifact"),
        .semantic_tag = IREE_SVL("object"),
        .expected_phases = LOOM_TARGET_COVERAGE_PHASE_BINARY_ENCODE |
                           LOOM_TARGET_COVERAGE_PHASE_ARTIFACT_WRAP |
                           LOOM_TARGET_COVERAGE_PHASE_RUN,
        .supported_phases = LOOM_TARGET_COVERAGE_PHASE_BINARY_ENCODE,
        .gap_key = IREE_SVL("runner-missing"),
    },
};

const loom_target_coverage_provider_t kProvider = {
    .name = IREE_SVL("sample"),
    .rows = kRows,
    .row_count = IREE_ARRAYSIZE(kRows),
};

const loom_target_coverage_provider_t* const kProviders[] = {
    &kProvider,
};

const loom_target_coverage_provider_set_t kProviderSet = {
    .providers = kProviders,
    .provider_count = IREE_ARRAYSIZE(kProviders),
};

TEST(TargetCoverageTest, VerifiesProviderRows) {
  IREE_EXPECT_OK(loom_target_coverage_provider_set_verify(&kProviderSet));
}

TEST(TargetCoverageTest, FormatsManifestJson) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_target_coverage_provider_set_format_manifest_json(
      &kProviderSet, &builder));

  const std::string json(iree_string_builder_buffer(&builder),
                         iree_string_builder_size(&builder));
  EXPECT_NE(json.find("\"provider_count\":1"), std::string::npos);
  EXPECT_NE(json.find("\"row_count\":2"), std::string::npos);
  EXPECT_NE(json.find("\"category\":\"vector.arithmetic\""), std::string::npos);
  EXPECT_NE(json.find("\"missing\":[\"artifact-wrap\",\"run\"]"),
            std::string::npos);
  EXPECT_NE(json.find("\"gap\":\"runner-missing\""), std::string::npos);
  iree_string_builder_deinitialize(&builder);
}

TEST(TargetCoverageTest, RejectsMissingGapKey) {
  loom_target_coverage_row_t row = kRows[1];
  row.gap_key = IREE_SV("");
  const loom_target_coverage_provider_t provider = {
      .name = IREE_SVL("sample"),
      .rows = &row,
      .row_count = 1,
  };
  const loom_target_coverage_provider_t* const providers[] = {
      &provider,
  };
  const loom_target_coverage_provider_set_t provider_set = {
      .providers = providers,
      .provider_count = IREE_ARRAYSIZE(providers),
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_coverage_provider_set_verify(&provider_set));
}

TEST(TargetCoverageTest, RejectsUnsupportedPhaseBits) {
  loom_target_coverage_row_t row = kRows[0];
  row.supported_phases = LOOM_TARGET_COVERAGE_PHASE_RUN;
  row.expected_phases = LOOM_TARGET_COVERAGE_PHASE_DESCRIPTOR;
  const loom_target_coverage_provider_t provider = {
      .name = IREE_SVL("sample"),
      .rows = &row,
      .row_count = 1,
  };
  const loom_target_coverage_provider_t* const providers[] = {
      &provider,
  };
  const loom_target_coverage_provider_set_t provider_set = {
      .providers = providers,
      .provider_count = IREE_ARRAYSIZE(providers),
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_coverage_provider_set_verify(&provider_set));
}

}  // namespace
}  // namespace loom
