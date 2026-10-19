// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>

extern "C" int source_increment(int value);

int imported_increment(int value) { return source_increment(value); }

LOOM_CHECK_CASE(imported_dependency) {
  const auto actual = imported_increment(7);
  loom::check::expect_equal(actual, 8);
}
