// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>

static constexpr unsigned table[2] = {1, 2};

unsigned immutable_left_lookup(unsigned index) { return table[index & 1u]; }
unsigned immutable_right_lookup(unsigned index);

static unsigned immutable_combined_lookup(unsigned index) {
  return immutable_left_lookup(index) + immutable_right_lookup(index);
}

LOOM_CHECK_CASE(immutable_object_link_identity) {
  const auto first = immutable_combined_lookup(0);
  const auto second = immutable_combined_lookup(1);
  loom::check::expect_equal(first, 11u);
  loom::check::expect_equal(second, 22u);
}
