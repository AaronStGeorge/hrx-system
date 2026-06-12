// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/runtime_requirements.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/sanitizer/options.h"

namespace {

TEST(AmdgpuRuntimeRequirementsTest, ValidateKnownRequirements) {
  IREE_EXPECT_OK(loom_amdgpu_runtime_requirements_validate(
      LOOM_AMDGPU_RUNTIME_REQUIREMENT_NONE));
  IREE_EXPECT_OK(loom_amdgpu_runtime_requirements_validate(
      LOOM_AMDGPU_RUNTIME_REQUIREMENT_FEEDBACK |
      LOOM_AMDGPU_RUNTIME_REQUIREMENT_ASAN_SHADOW));
}

TEST(AmdgpuRuntimeRequirementsTest, RejectUnknownRequirements) {
  iree_status_t status = loom_amdgpu_runtime_requirements_validate(
      (loom_amdgpu_runtime_requirements_t)0x80000000u);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_free(status);
}

TEST(AmdgpuRuntimeRequirementsTest, NullOptionsRequireNothing) {
  EXPECT_EQ(loom_amdgpu_runtime_requirements_from_target_pipeline_options(
                /*options=*/nullptr),
            LOOM_AMDGPU_RUNTIME_REQUIREMENT_NONE);
}

TEST(AmdgpuRuntimeRequirementsTest, DisabledSanitizersRequireNothing) {
  const loom_target_pipeline_options_t options = {0};
  EXPECT_EQ(
      loom_amdgpu_runtime_requirements_from_target_pipeline_options(&options),
      LOOM_AMDGPU_RUNTIME_REQUIREMENT_NONE);
}

TEST(AmdgpuRuntimeRequirementsTest, ValueSanitizerRequiresFeedbackOnly) {
  const loom_target_pipeline_options_t options = {
      .sanitizer =
          {
              .checks = LOOM_SANITIZER_CHECK_VALUE,
          },
  };
  EXPECT_EQ(
      loom_amdgpu_runtime_requirements_from_target_pipeline_options(&options),
      LOOM_AMDGPU_RUNTIME_REQUIREMENT_FEEDBACK);
}

TEST(AmdgpuRuntimeRequirementsTest, AccessSanitizerRequiresAsanShadow) {
  const loom_target_pipeline_options_t options = {
      .sanitizer =
          {
              .checks = LOOM_SANITIZER_CHECK_ACCESS,
          },
  };
  EXPECT_EQ(
      loom_amdgpu_runtime_requirements_from_target_pipeline_options(&options),
      LOOM_AMDGPU_RUNTIME_REQUIREMENT_FEEDBACK |
          LOOM_AMDGPU_RUNTIME_REQUIREMENT_ASAN_SHADOW);
}

}  // namespace
