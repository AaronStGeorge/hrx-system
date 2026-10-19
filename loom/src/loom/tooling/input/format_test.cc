// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/input/format.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(InputFormatTest, SelectsMetadataWithoutImporters) {
  const iree_string_view_t suffixes[] = {IREE_SV(".foreign"),
                                         IREE_SV(".foreign-test")};
  const loom_input_format_t foreign = {IREE_SV("foreign"), {2, suffixes}};
  const loom_input_format_t* const formats[] = {&foreign};
  const loom_input_format_list_t list = {formats, 1};
  struct Case {
    // Physical source name used for suffix selection.
    const char* path;
    // Explicit format override, or empty.
    const char* spelling;
    // Selected immutable descriptor.
    const loom_input_format_t* expected;
    // Position in the builtin-plus-optional format domain.
    iree_host_size_t ordinal;
  };
  const Case cases[] = {
      {"input.loom", "", &loom_input_text_format, 0},
      {"input.loom-test", "", &loom_input_text_format, 0},
      {"input.loombc", "", &loom_input_bytecode_format, 1},
      {"input.foreign", "", &foreign, 2},
      {"input.foreign-test", "", &foreign, 2},
      {"<stdin>", "", &loom_input_text_format, 0},
      {"arbitrary.suffix", "loom", &loom_input_text_format, 0},
      {"input.loom", "foreign", &foreign, 2},
      {"<stdin>", "loombc", &loom_input_bytecode_format, 1},
  };
  for (const auto& test_case : cases) {
    const loom_input_format_t* selected = nullptr;
    iree_host_size_t ordinal = IREE_HOST_SIZE_MAX;
    IREE_ASSERT_OK(loom_input_format_select(
        list, iree_make_cstring_view(test_case.spelling),
        iree_make_cstring_view(test_case.path), &selected, &ordinal));
    EXPECT_EQ(selected, test_case.expected);
    EXPECT_EQ(ordinal, test_case.ordinal);
  }
  const loom_input_format_t* selected = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      loom_input_format_select({}, {}, IREE_SV("input.foreign"), &selected,
                               nullptr));
  EXPECT_EQ(selected, nullptr);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      loom_input_format_select(list, IREE_SV("missing"), IREE_SV("input.loom"),
                               &selected, nullptr));
}

TEST(InputOptionsTest, KeepsLanguageOptionsIndependent) {
  const iree_string_view_t entries[] = {IREE_SV("cxx:std=c++20 D=VALUE=7"),
                                        IREE_SV("other:mode=fast")};
  iree_string_view_t options;
  IREE_ASSERT_OK(loom_input_options_for_provider(
      {IREE_ARRAYSIZE(entries), entries}, IREE_SV("cxx"), &options));
  EXPECT_TRUE(iree_string_view_equal(options, IREE_SV("std=c++20 D=VALUE=7")));
  IREE_ASSERT_OK(loom_input_options_for_provider(
      {IREE_ARRAYSIZE(entries), entries}, IREE_SV("loom"), &options));
  EXPECT_TRUE(iree_string_view_is_empty(options));
  const iree_string_view_t duplicates[] = {IREE_SV("cxx:"),
                                           IREE_SV("cxx:std=c++20")};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_input_options_for_provider({IREE_ARRAYSIZE(duplicates), duplicates},
                                      IREE_SV("cxx"), &options));
  const iree_string_view_t malformed[] = {IREE_SV("std=c++20")};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_input_options_for_provider({IREE_ARRAYSIZE(malformed), malformed},
                                      IREE_SV("cxx"), &options));
}

}  // namespace
