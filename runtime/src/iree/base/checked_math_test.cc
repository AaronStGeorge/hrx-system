// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/checked_math.h"

#include "iree/testing/gtest.h"

namespace {

//===----------------------------------------------------------------------===//
// Checked arithmetic tests - iree_host_size_t
//===----------------------------------------------------------------------===//

TEST(CheckedArithmetic, AddNoOverflow) {
  iree_host_size_t result;
  EXPECT_TRUE(iree_host_size_checked_add(100, 200, &result));
  EXPECT_EQ(result, 300);
}

TEST(CheckedArithmetic, AddOverflowMax) {
  iree_host_size_t result;
  EXPECT_FALSE(iree_host_size_checked_add(IREE_HOST_SIZE_MAX, 1, &result));
}

TEST(CheckedArithmetic, AddOverflowLarge) {
  iree_host_size_t result;
  EXPECT_FALSE(iree_host_size_checked_add(IREE_HOST_SIZE_MAX - 10,
                                          IREE_HOST_SIZE_MAX - 10, &result));
}

TEST(CheckedArithmetic, AddZero) {
  iree_host_size_t result;
  EXPECT_TRUE(iree_host_size_checked_add(0, 0, &result));
  EXPECT_EQ(result, 0);
}

TEST(CheckedArithmetic, AddMaxPlusZero) {
  iree_host_size_t result;
  EXPECT_TRUE(iree_host_size_checked_add(IREE_HOST_SIZE_MAX, 0, &result));
  EXPECT_EQ(result, IREE_HOST_SIZE_MAX);
}

TEST(CheckedArithmetic, MulNoOverflow) {
  iree_host_size_t result;
  EXPECT_TRUE(iree_host_size_checked_mul(100, 200, &result));
  EXPECT_EQ(result, 20000);
}

TEST(CheckedArithmetic, MulOverflowMaxTimesTwo) {
  iree_host_size_t result;
  EXPECT_FALSE(iree_host_size_checked_mul(IREE_HOST_SIZE_MAX, 2, &result));
}

TEST(CheckedArithmetic, MulOverflowLarge) {
  iree_host_size_t result;
  EXPECT_FALSE(iree_host_size_checked_mul(IREE_HOST_SIZE_MAX / 2 + 1,
                                          IREE_HOST_SIZE_MAX / 2 + 1, &result));
}

TEST(CheckedArithmetic, MulZeroFirst) {
  iree_host_size_t result;
  EXPECT_TRUE(iree_host_size_checked_mul(0, IREE_HOST_SIZE_MAX, &result));
  EXPECT_EQ(result, 0);
}

TEST(CheckedArithmetic, MulZeroSecond) {
  iree_host_size_t result;
  EXPECT_TRUE(iree_host_size_checked_mul(IREE_HOST_SIZE_MAX, 0, &result));
  EXPECT_EQ(result, 0);
}

TEST(CheckedArithmetic, MulOneFirst) {
  iree_host_size_t result;
  EXPECT_TRUE(iree_host_size_checked_mul(1, IREE_HOST_SIZE_MAX, &result));
  EXPECT_EQ(result, IREE_HOST_SIZE_MAX);
}

TEST(CheckedArithmetic, MulOneSecond) {
  iree_host_size_t result;
  EXPECT_TRUE(iree_host_size_checked_mul(IREE_HOST_SIZE_MAX, 1, &result));
  EXPECT_EQ(result, IREE_HOST_SIZE_MAX);
}

TEST(CheckedArithmetic, MulAddNoOverflow) {
  iree_host_size_t result;
  // 100 + 10 * 20 = 300
  EXPECT_TRUE(iree_host_size_checked_mul_add(100, 10, 20, &result));
  EXPECT_EQ(result, 300);
}

TEST(CheckedArithmetic, MulAddMulOverflow) {
  iree_host_size_t result;
  // 0 + MAX * 2 overflows in multiplication
  EXPECT_FALSE(
      iree_host_size_checked_mul_add(0, IREE_HOST_SIZE_MAX, 2, &result));
}

TEST(CheckedArithmetic, MulAddAddOverflow) {
  iree_host_size_t result;
  // MAX + 1 * 1 overflows in addition
  EXPECT_FALSE(
      iree_host_size_checked_mul_add(IREE_HOST_SIZE_MAX, 1, 1, &result));
}

TEST(CheckedArithmetic, MulAddZeroCount) {
  iree_host_size_t result;
  // 100 + 0 * 1000 = 100
  EXPECT_TRUE(iree_host_size_checked_mul_add(100, 0, 1000, &result));
  EXPECT_EQ(result, 100);
}

TEST(CheckedArithmetic, AlignNoOverflow) {
  iree_host_size_t result;
  EXPECT_TRUE(iree_host_size_checked_align(100, 16, &result));
  EXPECT_EQ(result, 112);
}

TEST(CheckedArithmetic, AlignAlreadyAligned) {
  iree_host_size_t result;
  EXPECT_TRUE(iree_host_size_checked_align(128, 16, &result));
  EXPECT_EQ(result, 128);
}

TEST(CheckedArithmetic, AlignZero) {
  iree_host_size_t result;
  EXPECT_TRUE(iree_host_size_checked_align(0, 16, &result));
  EXPECT_EQ(result, 0);
}

TEST(CheckedArithmetic, AlignOverflow) {
  iree_host_size_t result;
  // MAX - 5 + 15 (alignment-1) would overflow.
  EXPECT_FALSE(
      iree_host_size_checked_align(IREE_HOST_SIZE_MAX - 5, 16, &result));
}

TEST(CheckedArithmetic, AlignNearMaxNoOverflow) {
  iree_host_size_t result;
  // Value that is already aligned at MAX boundary should work.
  iree_host_size_t max_aligned = IREE_HOST_SIZE_MAX & ~(iree_host_size_t)15;
  EXPECT_TRUE(iree_host_size_checked_align(max_aligned, 16, &result));
  EXPECT_EQ(result, max_aligned);
}

//===----------------------------------------------------------------------===//
// Checked arithmetic tests - fixed-width integers
//===----------------------------------------------------------------------===//

TEST(CheckedArithmeticUnsignedI64, AddNoOverflow) {
  uint64_t result = 0;
  EXPECT_TRUE(iree_checked_add_u64(UINT64_MAX - 1, 1, &result));
  EXPECT_EQ(result, UINT64_MAX);
}

TEST(CheckedArithmeticUnsignedI64, AddOverflow) {
  uint64_t result = 0;
  EXPECT_FALSE(iree_checked_add_u64(UINT64_MAX, 1, &result));
}

TEST(CheckedArithmeticUnsignedI64, Align) {
  uint64_t result = 0;
  EXPECT_TRUE(iree_checked_align_u64(0, 16, &result));
  EXPECT_EQ(result, 0u);
  EXPECT_TRUE(iree_checked_align_u64(100, 16, &result));
  EXPECT_EQ(result, 112u);
  EXPECT_TRUE(iree_checked_align_u64(112, 16, &result));
  EXPECT_EQ(result, 112u);
  EXPECT_TRUE(iree_checked_align_u64(UINT64_MAX & ~(uint64_t)15, 16, &result));
  EXPECT_EQ(result, UINT64_MAX & ~(uint64_t)15);
  EXPECT_FALSE(iree_checked_align_u64(UINT64_MAX - 5, 16, &result));
}

TEST(CheckedArithmeticUnsignedI64, MulNoOverflow) {
  uint64_t result = 0;
  EXPECT_TRUE(iree_checked_mul_u64(UINT64_MAX, 1, &result));
  EXPECT_EQ(result, UINT64_MAX);
}

TEST(CheckedArithmeticUnsignedI64, MulOverflow) {
  uint64_t result = 0;
  EXPECT_FALSE(iree_checked_mul_u64(UINT64_MAX, 2, &result));
}

TEST(CheckedArithmeticSignedI32, AddNoOverflow) {
  int32_t result = 0;
  EXPECT_TRUE(iree_checked_add_i32(-100, 250, &result));
  EXPECT_EQ(result, 150);
}

TEST(CheckedArithmeticSignedI32, AddOverflowPositive) {
  int32_t result = 0;
  EXPECT_FALSE(iree_checked_add_i32(INT32_MAX, 1, &result));
}

TEST(CheckedArithmeticSignedI32, AddOverflowNegative) {
  int32_t result = 0;
  EXPECT_FALSE(iree_checked_add_i32(INT32_MIN, -1, &result));
}

TEST(CheckedArithmeticSignedI32, MulNoOverflow) {
  int32_t result = 0;
  EXPECT_TRUE(iree_checked_mul_i32(-100, 200, &result));
  EXPECT_EQ(result, -20000);
}

TEST(CheckedArithmeticSignedI32, MulOverflowPositive) {
  int32_t result = 0;
  EXPECT_FALSE(iree_checked_mul_i32(INT32_MAX, 2, &result));
}

TEST(CheckedArithmeticSignedI32, MulOverflowNegative) {
  int32_t result = 0;
  EXPECT_FALSE(iree_checked_mul_i32(INT32_MIN, -1, &result));
}

TEST(CheckedArithmeticSignedI32, MulAddNoOverflow) {
  int32_t result = 0;
  EXPECT_TRUE(iree_checked_mul_add_i32(100, -10, 20, &result));
  EXPECT_EQ(result, -100);
}

TEST(CheckedArithmeticSignedI32, MulAddMulOverflow) {
  int32_t result = 0;
  EXPECT_FALSE(iree_checked_mul_add_i32(0, INT32_MAX, 2, &result));
}

TEST(CheckedArithmeticSignedI32, MulAddAddOverflow) {
  int32_t result = 0;
  EXPECT_FALSE(iree_checked_mul_add_i32(INT32_MAX, 1, 1, &result));
}

TEST(CheckedArithmeticSignedI64, AddNoOverflow) {
  int64_t result = 0;
  EXPECT_TRUE(iree_checked_add_i64(-100, 250, &result));
  EXPECT_EQ(result, 150);
}

TEST(CheckedArithmeticSignedI64, AddOverflowPositive) {
  int64_t result = 0;
  EXPECT_FALSE(iree_checked_add_i64(INT64_MAX, 1, &result));
}

TEST(CheckedArithmeticSignedI64, AddOverflowNegative) {
  int64_t result = 0;
  EXPECT_FALSE(iree_checked_add_i64(INT64_MIN, -1, &result));
}

TEST(CheckedArithmeticSignedI64, SubNoOverflow) {
  int64_t result = 0;
  EXPECT_TRUE(iree_checked_sub_i64(-100, 250, &result));
  EXPECT_EQ(result, -350);
}

TEST(CheckedArithmeticSignedI64, SubOverflowPositive) {
  int64_t result = 0;
  EXPECT_FALSE(iree_checked_sub_i64(INT64_MAX, -1, &result));
}

TEST(CheckedArithmeticSignedI64, SubOverflowNegative) {
  int64_t result = 0;
  EXPECT_FALSE(iree_checked_sub_i64(INT64_MIN, 1, &result));
}

TEST(CheckedArithmeticSignedI64, MulNoOverflow) {
  int64_t result = 0;
  EXPECT_TRUE(iree_checked_mul_i64(-100, 200, &result));
  EXPECT_EQ(result, -20000);
}

TEST(CheckedArithmeticSignedI64, MulOverflowPositive) {
  int64_t result = 0;
  EXPECT_FALSE(iree_checked_mul_i64(INT64_MAX, 2, &result));
}

TEST(CheckedArithmeticSignedI64, MulOverflowNegative) {
  int64_t result = 0;
  EXPECT_FALSE(iree_checked_mul_i64(INT64_MIN, -1, &result));
}

TEST(CheckedArithmeticSignedI64, MulOverflowBothNegative) {
  int64_t result = 0;
  EXPECT_FALSE(iree_checked_mul_i64(INT64_MIN, -2, &result));
}

TEST(CheckedArithmeticSignedI64, MulAddNoOverflow) {
  int64_t result = 0;
  EXPECT_TRUE(iree_checked_mul_add_i64(100, -10, 20, &result));
  EXPECT_EQ(result, -100);
}

TEST(CheckedArithmeticSignedI64, MulAddMulOverflow) {
  int64_t result = 0;
  EXPECT_FALSE(iree_checked_mul_add_i64(0, INT64_MAX, 2, &result));
}

TEST(CheckedArithmeticSignedI64, MulAddAddOverflow) {
  int64_t result = 0;
  EXPECT_FALSE(iree_checked_mul_add_i64(INT64_MAX, 1, 1, &result));
}

//===----------------------------------------------------------------------===//
// Checked arithmetic tests - iree_device_size_t
//===----------------------------------------------------------------------===//

TEST(CheckedArithmeticDevice, AddNoOverflow) {
  iree_device_size_t result;
  EXPECT_TRUE(iree_device_size_checked_add(100, 200, &result));
  EXPECT_EQ(result, 300);
}

TEST(CheckedArithmeticDevice, AddOverflow) {
  iree_device_size_t result;
  EXPECT_FALSE(iree_device_size_checked_add(IREE_DEVICE_SIZE_MAX, 1, &result));
}

TEST(CheckedArithmeticDevice, MulNoOverflow) {
  iree_device_size_t result;
  EXPECT_TRUE(iree_device_size_checked_mul(100, 200, &result));
  EXPECT_EQ(result, 20000);
}

TEST(CheckedArithmeticDevice, MulOverflow) {
  iree_device_size_t result;
  EXPECT_FALSE(iree_device_size_checked_mul(IREE_DEVICE_SIZE_MAX, 2, &result));
}

TEST(CheckedArithmeticDevice, MulAddNoOverflow) {
  iree_device_size_t result;
  EXPECT_TRUE(iree_device_size_checked_mul_add(100, 10, 20, &result));
  EXPECT_EQ(result, 300);
}

}  // namespace
