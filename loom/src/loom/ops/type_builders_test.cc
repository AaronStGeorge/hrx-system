// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/type_builders.h"

#include <initializer_list>

#include "iree/testing/gtest.h"

namespace {

void CheckEnumSpellings(iree_string_view_t (*name)(uint8_t),
                        bool (*parse)(iree_string_view_t, uint8_t*),
                        std::initializer_list<iree_string_view_t> spellings) {
  for (uint32_t value = 0; value <= UINT8_MAX; ++value) {
    SCOPED_TRACE(value);
    iree_string_view_t expected = value < spellings.size()
                                      ? spellings.begin()[value]
                                      : iree_string_view_empty();
    EXPECT_TRUE(iree_string_view_equal(name((uint8_t)value), expected));
    uint8_t parsed = UINT8_MAX;
    if (iree_string_view_is_empty(expected)) {
      EXPECT_FALSE(parse(expected, &parsed));
      EXPECT_EQ(parsed, UINT8_MAX);
    } else {
      EXPECT_TRUE(parse(expected, &parsed));
      EXPECT_EQ(parsed, value);
    }
  }
  uint8_t parsed = UINT8_MAX;
  EXPECT_FALSE(parse(IREE_SV("not_declared"), &parsed));
  EXPECT_EQ(parsed, UINT8_MAX);
}

TEST(TypeBuilders, EncodingRoleSpellings) {
  CheckEnumSpellings(loom_encoding_type_role_name,
                     loom_encoding_type_role_parse,
                     {IREE_SV(""), IREE_SV("layout"), IREE_SV("schema"),
                      IREE_SV("storage"), IREE_SV("transform")});
}

TEST(TypeBuilders, StorageSpaceSpellings) {
  CheckEnumSpellings(loom_low_storage_type_space_name,
                     loom_low_storage_type_space_parse,
                     {IREE_SV("stack"), IREE_SV("scratch"), IREE_SV("private"),
                      IREE_SV("workgroup")});
}

}  // namespace
