// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>

#include "immutable_object_link_tables.h"

static constexpr unsigned local_table[2] = {1, 2};

unsigned immutable_left_lookup(unsigned index) {
  const unsigned lane = index & 1u;
  return local_table[lane] + immutable_link::bias[lane] +
         immutable_link::weights[lane];
}
unsigned immutable_right_lookup(unsigned index);

static unsigned immutable_combined_lookup(unsigned index) {
  return immutable_left_lookup(index) + immutable_right_lookup(index);
}

LOOM_CHECK_CASE(immutable_object_link_identity) {
  const auto first = immutable_combined_lookup(0);
  const auto second = immutable_combined_lookup(1);
  loom::check::expect_equal(first, 117u);
  loom::check::expect_equal(second, 232u);
}
