// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>

namespace detail {
struct Pair {
  unsigned first;
  unsigned second;
};

static constexpr unsigned table[4] = {11, 22, 33, 44};
static constexpr unsigned nested[2][3] = {{1, 2, 3}, {4, 5, 6}};
static constexpr Pair pairs[2] = {{10, 20}, {30, 40}};
static constexpr unsigned scalar = 7;
}  // namespace detail

static unsigned lookup(unsigned index) {
  return detail::table[index & 3u] + detail::nested[index & 1u][index % 3u] +
         detail::pairs[index & 1u].second;
}

static unsigned addressed_scalar() {
  const unsigned* pointer = &detail::scalar;
  return *pointer;
}

LOOM_CHECK_CASE(immutable_object_values) {
  const auto first = lookup(0);
  const auto second = lookup(1);
  const auto third = lookup(2);
  const auto fourth = lookup(3);
  const auto scalar = addressed_scalar();
  loom::check::expect_equal(first, 32u);
  loom::check::expect_equal(second, 67u);
  loom::check::expect_equal(third, 56u);
  loom::check::expect_equal(fourth, 88u);
  loom::check::expect_equal(scalar, 7u);
}
