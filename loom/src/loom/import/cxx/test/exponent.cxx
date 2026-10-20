// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/scalar.h>

template <class Float>
static bool check_pair(Float input) {
  auto original = input++;
  --input;
  auto pair = loom::scalar::frexpf(input);
  auto copy = pair;
  auto rebuilt = loom::scalar::ldexpf(copy.fraction, copy.exponent);
  return original == input && copy.fraction == Float(0.625) &&
         copy.exponent == 2 && rebuilt == input;
}

template <class Float>
static Float fraction(Float input) {
  return loom::scalar::frexpf(input).fraction;
}

static bool builtin_evaluation() {
  double input = 8.0;
  int exponents[2] = {};
  int index = 0;
  double fraction = __builtin_frexp(input++, &exponents[index++]);
  // The discarded fraction still leaves the standard exponent store.
  __builtin_frexpf(4.0f, &exponents[1]);
  const auto incremented_input = input;
  const auto incremented_index = index;
  const auto scaled = __builtin_ldexp(input++, index++);
  const auto scaled_float = __builtin_ldexpf(-0.75f, -3);
  return incremented_input == 9.0 && incremented_index == 1 &&
         fraction == 0.5 && exponents[0] == 4 && exponents[1] == 3 &&
         scaled == 18.0 && input == 10.0 && index == 2 &&
         scaled_float == -0.09375f;
}

LOOM_CHECK_CASE(exponent_values) {
  const auto double_ok = check_pair(double(2.5));
  const auto single_ok = check_pair(float(2.5));
  const auto half_ok = check_pair(_Float16(2.5));
  const auto bfloat_ok = check_pair(__bf16(2.5));
  const auto e4m3_ok = check_pair(__float8_e4m3fn(2.5));
  const auto e5m2_ok = check_pair(__float8_e5m2(2.5));
  const auto bfloat_fraction = fraction(__bf16(2.5));
  const auto e4m3_fraction = fraction(__float8_e4m3fn(2.5));
  const auto e5m2_fraction = fraction(__float8_e5m2(2.5));
  const auto builtin_ok = builtin_evaluation();
  loom::check::expect_equal(double_ok, true);
  loom::check::expect_equal(single_ok, true);
  loom::check::expect_equal(half_ok, true);
  loom::check::expect_equal(bfloat_ok, true);
  loom::check::expect_equal(e4m3_ok, true);
  loom::check::expect_equal(e5m2_ok, true);
  loom::check::expect_equal(bfloat_fraction, __bf16(0.625));
  loom::check::expect_equal(e4m3_fraction, __float8_e4m3fn(0.625));
  loom::check::expect_equal(e5m2_fraction, __float8_e5m2(0.625));
  loom::check::expect_equal(builtin_ok, true);
}
