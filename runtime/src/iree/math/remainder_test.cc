// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/math/remainder.h"

#include "iree/math/elementary_test_util.h"

namespace iree::math {
namespace {
using testing::ElementaryTest;
using testing::F32FromBits;
using testing::F32ToBits;
using testing::F64FromBits;
using testing::F64ToBits;

template <typename Bits>
struct RemainderCase {
  // Positive dividend payload; both signs are checked.
  Bits numerator;
  // Positive divisor payload; both signs are checked.
  Bits denominator;
  // Exact integer/rational oracle result before applying the dividend sign.
  Bits expected;
};

template <typename Bits, typename Float, size_t N>
void CheckRemainders(const RemainderCase<Bits> (&cases)[N], Bits sign_mask,
                     Bits infinity, Float (*from_bits)(Bits),
                     Bits (*to_bits)(Float), Float (*function)(Float, Float)) {
  for (const auto& row : cases) {
    for (Bits numerator_sign : {Bits{0}, sign_mask}) {
      for (Bits denominator_sign : {Bits{0}, sign_mask}) {
        SCOPED_TRACE(::testing::Message()
                     << std::hex << "numerator=" << row.numerator
                     << " denominator=" << row.denominator
                     << " signs=" << numerator_sign << "," << denominator_sign);
        const Bits expected = row.expected >= infinity
                                  ? row.expected
                                  : row.expected | numerator_sign;
        EXPECT_EQ(
            expected,
            to_bits(function(from_bits(row.numerator | numerator_sign),
                             from_bits(row.denominator | denominator_sign))));
      }
    }
  }
}

TEST_F(ElementaryTest, FmodF32ExactBoundaries) {
  // Equal/neighboring values, normal and subnormal transitions, reduction
  // chunk boundaries, the largest exponent gap, and exceptional operands.
  const RemainderCase<uint32_t> cases[] = {
      {0x00000000u, 0x3f800000u, 0x00000000u},
      {0x3f800000u, 0x00000000u, 0x7fc00000u},
      {0x3f800000u, 0x3f800000u, 0x00000000u},
      {0x3f800000u, 0x3f800001u, 0x3f800000u},
      {0x3f800001u, 0x3f800000u, 0x34000000u},
      {0x007fffffu, 0x00000003u, 0x00000001u},
      {0x00800001u, 0x00800000u, 0x00000001u},
      {0x00800000u, 0x00000003u, 0x00000002u},
      {0x7f7fffffu, 0x00000003u, 0x00000000u},
      {0x7f7fffffu, 0x3fc00003u, 0x3f140002u},
      {0x7f7fffffu, 0x7f7ffffeu, 0x73800000u},
      {0x0b800003u, 0x00800001u, 0x00800000u},
      {0x0c000003u, 0x0b800001u, 0x01000000u},
      {0x407fffffu, 0x3fc00003u, 0x3f7ffff0u},
      {0x43ffffffu, 0x3fc00003u, 0x3effec04u},
      {0x447fffffu, 0x3fc00003u, 0x3f7fec04u},
      {0x53ffffffu, 0x3fc00003u, 0x3f940002u},
      {0x547fffffu, 0x3fc00003u, 0x3f500002u},
      {0x3f800000u, 0x7f800000u, 0x3f800000u},
      {0x7f800000u, 0x3f800000u, 0x7fc00000u},
      {0x7f800000u, 0x7f800000u, 0x7fc00000u},
      {0x7f800001u, 0x3f800000u, 0x7fc00000u},
      {0x3f800000u, 0x7f800001u, 0x7fc00000u},
      {0x7fc00000u, 0x00000000u, 0x7fc00000u},
  };
  CheckRemainders(cases, 0x80000000u, 0x7f800000u, F32FromBits, F32ToBits,
                  iree_math_fmod_f32);
  EnableFlushDenormalsToZero();
  CheckRemainders(cases, 0x80000000u, 0x7f800000u, F32FromBits, F32ToBits,
                  iree_math_fmod_f32);
}

TEST_F(ElementaryTest, FmodF64ExactBoundaries) {
  // Equal/neighboring values, normal and subnormal transitions, reduction
  // chunk boundaries, the largest exponent gap, and exceptional operands.
  const RemainderCase<uint64_t> cases[] = {
      {UINT64_C(0x0000000000000000), UINT64_C(0x3ff0000000000000),
       UINT64_C(0x0000000000000000)},
      {UINT64_C(0x3ff0000000000000), UINT64_C(0x0000000000000000),
       UINT64_C(0x7ff8000000000000)},
      {UINT64_C(0x3ff0000000000000), UINT64_C(0x3ff0000000000000),
       UINT64_C(0x0000000000000000)},
      {UINT64_C(0x3ff0000000000000), UINT64_C(0x3ff0000000000001),
       UINT64_C(0x3ff0000000000000)},
      {UINT64_C(0x3ff0000000000001), UINT64_C(0x3ff0000000000000),
       UINT64_C(0x3cb0000000000000)},
      {UINT64_C(0x000fffffffffffff), UINT64_C(0x0000000000000003),
       UINT64_C(0x0000000000000000)},
      {UINT64_C(0x0010000000000001), UINT64_C(0x0010000000000000),
       UINT64_C(0x0000000000000001)},
      {UINT64_C(0x0010000000000000), UINT64_C(0x0000000000000003),
       UINT64_C(0x0000000000000001)},
      {UINT64_C(0x7fefffffffffffff), UINT64_C(0x0000000000000003),
       UINT64_C(0x0000000000000002)},
      {UINT64_C(0x7fefffffffffffff), UINT64_C(0x3ff8000000000003),
       UINT64_C(0x3fdfffffffffff64)},
      {UINT64_C(0x7fefffffffffffff), UINT64_C(0x7feffffffffffffe),
       UINT64_C(0x7ca0000000000000)},
      {UINT64_C(0x0340000000000003), UINT64_C(0x0010000000000001),
       UINT64_C(0x0010000000000000)},
      {UINT64_C(0x0350000000000003), UINT64_C(0x0340000000000001),
       UINT64_C(0x0020000000000000)},
      {UINT64_C(0x400fffffffffffff), UINT64_C(0x3ff8000000000003),
       UINT64_C(0x3feffffffffffff0)},
      {UINT64_C(0x40afffffffffffff), UINT64_C(0x3ff8000000000003),
       UINT64_C(0x3fefffffffffb004)},
      {UINT64_C(0x40bfffffffffffff), UINT64_C(0x3ff8000000000003),
       UINT64_C(0x3fdffffffffec004)},
      {UINT64_C(0x3ff0000000000000), UINT64_C(0x7ff0000000000000),
       UINT64_C(0x3ff0000000000000)},
      {UINT64_C(0x7ff0000000000000), UINT64_C(0x3ff0000000000000),
       UINT64_C(0x7ff8000000000000)},
      {UINT64_C(0x7ff0000000000000), UINT64_C(0x7ff0000000000000),
       UINT64_C(0x7ff8000000000000)},
      {UINT64_C(0x7ff0000000000001), UINT64_C(0x3ff0000000000000),
       UINT64_C(0x7ff8000000000000)},
      {UINT64_C(0x3ff0000000000000), UINT64_C(0x7ff0000000000001),
       UINT64_C(0x7ff8000000000000)},
      {UINT64_C(0x7ff8000000000000), UINT64_C(0x0000000000000000),
       UINT64_C(0x7ff8000000000000)},
  };
  CheckRemainders(cases, UINT64_C(0x8000000000000000),
                  UINT64_C(0x7ff0000000000000), F64FromBits, F64ToBits,
                  iree_math_fmod_f64);
  EnableFlushDenormalsToZero();
  CheckRemainders(cases, UINT64_C(0x8000000000000000),
                  UINT64_C(0x7ff0000000000000), F64FromBits, F64ToBits,
                  iree_math_fmod_f64);
}

}  // namespace
}  // namespace iree::math
