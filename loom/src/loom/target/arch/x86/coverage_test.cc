// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/coverage.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

TEST(X86CoverageTest, VerifiesRows) {
  const loom_target_coverage_provider_t* const providers[] = {
      &loom_x86_target_coverage_provider,
  };
  const loom_target_coverage_provider_set_t provider_set = {
      .providers = providers,
      .provider_count = IREE_ARRAYSIZE(providers),
  };
  IREE_EXPECT_OK(loom_target_coverage_provider_set_verify(&provider_set));
}

TEST(X86CoverageTest, FormatsRepresentativeRows) {
  const loom_target_coverage_provider_t* const providers[] = {
      &loom_x86_target_coverage_provider,
  };
  const loom_target_coverage_provider_set_t provider_set = {
      .providers = providers,
      .provider_count = IREE_ARRAYSIZE(providers),
  };
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_target_coverage_provider_set_format_manifest_json(
      &provider_set, &builder));

  const std::string json(iree_string_builder_buffer(&builder),
                         iree_string_builder_size(&builder));
  EXPECT_NE(json.find("\"target\":\"x86-packed-dot\""), std::string::npos);
  EXPECT_NE(json.find("\"semantic\":\"vnni-bf16-f16\""), std::string::npos);
  EXPECT_NE(json.find("\"gap\":\"cpu-object-abi-missing\""), std::string::npos);
  iree_string_builder_deinitialize(&builder);
}

}  // namespace
}  // namespace loom
