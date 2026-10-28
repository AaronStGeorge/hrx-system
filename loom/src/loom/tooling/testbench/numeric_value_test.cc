// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/numeric_value.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

TEST(NumericValueTest, ResolvesStorageAndScalarTypes) {
  const auto* f8e4m3 =
      loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_F8E4M3);
  ASSERT_NE(f8e4m3, nullptr);
  EXPECT_STREQ(f8e4m3->name, "f8e4m3");
  EXPECT_EQ(f8e4m3->byte_count, 1);
  EXPECT_TRUE(loom_testbench_numeric_type_is_float(f8e4m3));
  EXPECT_FALSE(loom_testbench_numeric_type_is_integer(f8e4m3));
  EXPECT_EQ(loom_testbench_numeric_type_parse(IREE_SV("f8e4m3")), f8e4m3);
  EXPECT_EQ(loom_testbench_numeric_type_from_hal_element_type(
                IREE_HAL_ELEMENT_TYPE_FLOAT_8_E4M3_FN),
            f8e4m3);
  EXPECT_EQ(
      loom_testbench_numeric_type_from_scalar_type(LOOM_SCALAR_TYPE_F8E4M3),
      f8e4m3);

  const auto* i32 = loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_I32);
  ASSERT_NE(i32, nullptr);
  EXPECT_TRUE(loom_testbench_numeric_type_is_integer(i32));
  EXPECT_EQ(i32->minimum_i64, INT32_MIN);
  EXPECT_EQ(i32->maximum_i64, INT32_MAX);
  EXPECT_TRUE(loom_testbench_numeric_type_matches_storage(
      i32, IREE_HAL_ELEMENT_TYPE_INT_32));
  EXPECT_TRUE(loom_testbench_numeric_type_matches_storage(
      i32, IREE_HAL_ELEMENT_TYPE_UINT_32));

  EXPECT_EQ(loom_testbench_numeric_type_parse(IREE_SV("unknown")), nullptr);
  EXPECT_EQ(
      loom_testbench_numeric_type_from_scalar_type(LOOM_SCALAR_TYPE_INDEX),
      nullptr);
}

TEST(NumericValueTest, LoadsFloatingElementsWithFixedDecoders) {
  const uint8_t f8e4m3[] = {0x28};
  EXPECT_EQ(loom_testbench_numeric_load_f64_at(
                loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_F8E4M3),
                f8e4m3, 0),
            0.25);
  const uint8_t f8e5m2[] = {0x34};
  EXPECT_EQ(loom_testbench_numeric_load_f64_at(
                loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_F8E5M2),
                f8e5m2, 0),
            0.25);

  const uint16_t f16[] = {0x3E00};
  EXPECT_EQ(loom_testbench_numeric_load_f64_at(
                loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_F16),
                reinterpret_cast<const uint8_t*>(f16), 0),
            1.5);
  const uint16_t bf16[] = {0x3FC0};
  EXPECT_EQ(loom_testbench_numeric_load_f64_at(
                loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_BF16),
                reinterpret_cast<const uint8_t*>(bf16), 0),
            1.5);

  const float f32[] = {-3.25f, 7.5f};
  EXPECT_EQ(loom_testbench_numeric_load_f64_at(
                loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_F32),
                reinterpret_cast<const uint8_t*>(f32), 1),
            7.5);
  const double f64[] = {-3.25, 9.125};
  EXPECT_EQ(loom_testbench_numeric_load_f64_at(
                loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_F64),
                reinterpret_cast<const uint8_t*>(f64), 1),
            9.125);
}

TEST(NumericValueTest, LoadsIntegerElementsExactly) {
  const int8_t i8[] = {-128, 127};
  EXPECT_EQ(loom_testbench_numeric_load_i64_at(
                loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_I8),
                reinterpret_cast<const uint8_t*>(i8), 0),
            -128);
  const uint16_t u16[] = {0, UINT16_MAX};
  EXPECT_EQ(loom_testbench_numeric_load_i64_at(
                loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_U16),
                reinterpret_cast<const uint8_t*>(u16), 1),
            UINT16_MAX);
  const uint32_t u32[] = {0, UINT32_MAX};
  EXPECT_EQ(loom_testbench_numeric_load_i64_at(
                loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_U32),
                reinterpret_cast<const uint8_t*>(u32), 1),
            static_cast<int64_t>(UINT32_MAX));
  const int64_t i64[] = {INT64_MIN, INT64_MAX};
  EXPECT_EQ(loom_testbench_numeric_load_i64_at(
                loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_I64),
                reinterpret_cast<const uint8_t*>(i64), 0),
            INT64_MIN);
  EXPECT_TRUE(loom_testbench_numeric_type_contains_i64(
      loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_I8), INT8_MAX));
  EXPECT_FALSE(loom_testbench_numeric_type_contains_i64(
      loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_I8),
      static_cast<int64_t>(INT8_MAX) + 1));
}

}  // namespace
}  // namespace loom
