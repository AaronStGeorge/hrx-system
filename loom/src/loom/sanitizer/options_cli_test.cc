// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/sanitizer/options_cli.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

TEST(SanitizerOptionsCliTest, ParsesNamedCheckSets) {
  loom_sanitizer_options_t options = {};
  IREE_ASSERT_OK(loom_sanitizer_options_parse_checks(
      IREE_SV("access | operation"), IREE_SV("--sanitizer"), &options));
  EXPECT_EQ(options.checks,
            LOOM_SANITIZER_CHECK_ACCESS | LOOM_SANITIZER_CHECK_OPERATION);
  EXPECT_EQ(options.flags, 0u);
  EXPECT_EQ(options.reporting_mode, LOOM_SANITIZER_REPORTING_MODE_DEFAULT);

  IREE_ASSERT_OK(loom_sanitizer_options_parse_checks(
      IREE_SV("all"), IREE_SV("--sanitizer"), &options));
  EXPECT_EQ(options.checks, LOOM_SANITIZER_CHECKS_KNOWN);

  IREE_ASSERT_OK(loom_sanitizer_options_parse_checks(
      IREE_SV("none"), IREE_SV("--sanitizer"), &options));
  EXPECT_EQ(options.checks, 0u);
}

TEST(SanitizerOptionsCliTest, RejectsMalformedCheckSets) {
  loom_sanitizer_checks_t checks = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_sanitizer_checks_parse(IREE_SV("none|access"),
                                  IREE_SV("--sanitizer"), &checks));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_sanitizer_checks_parse(IREE_SV("access|"), IREE_SV("--sanitizer"),
                                  &checks));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_sanitizer_checks_parse(
                            IREE_SV("bogus"), IREE_SV("--sanitizer"), &checks));
}

TEST(SanitizerOptionsCliTest, FormatsCanonicalCheckSets) {
  iree_string_view_t formatted = iree_string_view_empty();
  IREE_ASSERT_OK(loom_sanitizer_checks_format(
      LOOM_SANITIZER_CHECK_OPERATION | LOOM_SANITIZER_CHECK_VALUE, &formatted));
  EXPECT_TRUE(iree_string_view_equal(formatted, IREE_SV("value|operation")));

  IREE_ASSERT_OK(
      loom_sanitizer_checks_format(LOOM_SANITIZER_CHECKS_KNOWN, &formatted));
  EXPECT_TRUE(
      iree_string_view_equal(formatted, IREE_SV("access|value|operation")));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_sanitizer_checks_format(1ull << 63, &formatted));
}

TEST(SanitizerOptionsCliTest, ParsesReportingModes) {
  loom_sanitizer_reporting_mode_t mode = LOOM_SANITIZER_REPORTING_MODE_DEFAULT;
  IREE_ASSERT_OK(loom_sanitizer_reporting_mode_parse(
      IREE_SV("trap"), IREE_SV("--sanitizer-reporting"), &mode));
  EXPECT_EQ(mode, LOOM_SANITIZER_REPORTING_MODE_TRAP);

  IREE_ASSERT_OK(loom_sanitizer_reporting_mode_parse(
      IREE_SV("default"), IREE_SV("--sanitizer-reporting"), &mode));
  EXPECT_EQ(mode, LOOM_SANITIZER_REPORTING_MODE_DEFAULT);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_sanitizer_reporting_mode_parse(
          IREE_SV("printf"), IREE_SV("--sanitizer-reporting"), &mode));
}

TEST(SanitizerOptionsCliTest, FormatsReportingModes) {
  iree_string_view_t formatted = iree_string_view_empty();
  IREE_ASSERT_OK(loom_sanitizer_reporting_mode_format(
      LOOM_SANITIZER_REPORTING_MODE_TRAP, &formatted));
  EXPECT_TRUE(iree_string_view_equal(formatted, IREE_SV("trap")));

  IREE_ASSERT_OK(loom_sanitizer_reporting_mode_format(
      LOOM_SANITIZER_REPORTING_MODE_DEFAULT, &formatted));
  EXPECT_TRUE(iree_string_view_equal(formatted, IREE_SV("default")));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_sanitizer_reporting_mode_format(
                            (loom_sanitizer_reporting_mode_t)99, &formatted));
}

}  // namespace
}  // namespace loom
