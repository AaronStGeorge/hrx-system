// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// NOTE: FlatCC common headers must be included before generated schema headers.
// clang-format off
#include "iree/base/internal/flatcc/parsing.h"
#include "flatcc_test_library_schema_reader.h"
#include "flatcc_test_library_schema_verifier.h"
// clang-format on

#include "gtest/gtest.h"

namespace {

TEST(FlatccCompileTest, GeneratedHeadersExposeRootType) {
  EXPECT_NE(iree_test_LibraryRoot_type_hash,
            static_cast<flatbuffers_thash_t>(0));
}

}  // namespace
