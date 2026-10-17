// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMCXX_NUMERIC_H_
#define LOOMCXX_NUMERIC_H_

namespace loom::type {
// One-byte E4M3 with signed finite saturation and NaN preservation.
using float8_e4m3fn_t = __float8_e4m3fn;
// One-byte IEEE E5M2 with infinities and NaNs.
using float8_e5m2_t = __float8_e5m2;

template <class Shape, class T>
class vector;
}  // namespace loom::type

namespace loom {

// Converts one scalar or each lane of a native fixed vector. The reference
// body preserves ordinary C++ constant evaluation; runtime imports bind the
// specialization directly to Loom's numeric conversion operations.
template <class To, class From>
[[loom::op("numeric.convert")]] constexpr To convert(From value) {
  if constexpr (__is_arithmetic(To) && __is_arithmetic(From)) {
    return static_cast<To>(value);
  } else {
    return __builtin_convertvector(value, To);
  }
}

// Converts each element of a semantic vector without changing its logical
// shape. Dynamic extent identities are carried from source to result in IR.
template <class To, class Shape, class T>
[[loom::op("numeric.convert")]] constexpr To convert(type::vector<Shape, T>) {
  return {};
}

}  // namespace loom

#endif  // LOOMCXX_NUMERIC_H_
