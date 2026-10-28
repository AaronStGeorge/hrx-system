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
  EXPECT_EQ(loom_testbench_numeric_type_parse(IREE_SV("u64")), nullptr);
  EXPECT_EQ(loom_testbench_numeric_type_from_hal_element_type(
                IREE_HAL_ELEMENT_TYPE_UINT_64),
            nullptr);
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
  const uint64_t u64[] = {UINT64_C(9007199254740993), UINT64_MAX};
  EXPECT_EQ(loom_testbench_numeric_load_u64_at(
                loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_U64),
                reinterpret_cast<const uint8_t*>(u64), 0),
            UINT64_C(9007199254740993));
  EXPECT_EQ(loom_testbench_numeric_load_u64_at(
                loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_U64),
                reinterpret_cast<const uint8_t*>(u64), 1),
            UINT64_MAX);
  EXPECT_TRUE(loom_testbench_numeric_type_contains_i64(
      loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_I8), INT8_MAX));
  EXPECT_FALSE(loom_testbench_numeric_type_contains_i64(
      loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_I8),
      static_cast<int64_t>(INT8_MAX) + 1));
}

TEST(NumericValueTest, MapsEveryFixedWidthIntegerInterpretation) {
  struct IntegerFormat {
    // Authored signless Loom scalar type.
    loom_scalar_type_t scalar_type;
    // Signed numerical interpretation.
    loom_testbench_numeric_kind_t signed_kind;
    // Unsigned numerical interpretation.
    loom_testbench_numeric_kind_t unsigned_kind;
    // Signed HAL storage spelling.
    iree_hal_element_type_t signed_storage;
    // Unsigned HAL storage spelling.
    iree_hal_element_type_t unsigned_storage;
    // Signless HAL storage spelling.
    iree_hal_element_type_t signless_storage;
  };
  const IntegerFormat formats[] = {
      {LOOM_SCALAR_TYPE_I8, LOOM_TESTBENCH_NUMERIC_I8,
       LOOM_TESTBENCH_NUMERIC_U8, IREE_HAL_ELEMENT_TYPE_SINT_8,
       IREE_HAL_ELEMENT_TYPE_UINT_8, IREE_HAL_ELEMENT_TYPE_INT_8},
      {LOOM_SCALAR_TYPE_I16, LOOM_TESTBENCH_NUMERIC_I16,
       LOOM_TESTBENCH_NUMERIC_U16, IREE_HAL_ELEMENT_TYPE_SINT_16,
       IREE_HAL_ELEMENT_TYPE_UINT_16, IREE_HAL_ELEMENT_TYPE_INT_16},
      {LOOM_SCALAR_TYPE_I32, LOOM_TESTBENCH_NUMERIC_I32,
       LOOM_TESTBENCH_NUMERIC_U32, IREE_HAL_ELEMENT_TYPE_SINT_32,
       IREE_HAL_ELEMENT_TYPE_UINT_32, IREE_HAL_ELEMENT_TYPE_INT_32},
      {LOOM_SCALAR_TYPE_I64, LOOM_TESTBENCH_NUMERIC_I64,
       LOOM_TESTBENCH_NUMERIC_U64, IREE_HAL_ELEMENT_TYPE_SINT_64,
       IREE_HAL_ELEMENT_TYPE_UINT_64, IREE_HAL_ELEMENT_TYPE_INT_64},
  };
  for (iree_host_size_t format_index = 0;
       format_index < IREE_ARRAYSIZE(formats); ++format_index) {
    const auto& format = formats[format_index];
    const auto* signed_type =
        loom_testbench_numeric_type_from_signed_scalar_type(format.scalar_type);
    const auto* unsigned_type =
        loom_testbench_numeric_type_from_unsigned_scalar_type(
            format.scalar_type);
    ASSERT_NE(signed_type, nullptr);
    ASSERT_NE(unsigned_type, nullptr);
    EXPECT_EQ(signed_type->kind, format.signed_kind);
    EXPECT_EQ(unsigned_type->kind, format.unsigned_kind);
    for (const auto* numeric_type : {signed_type, unsigned_type}) {
      EXPECT_TRUE(loom_testbench_numeric_type_matches_storage(
          numeric_type, format.signed_storage));
      EXPECT_TRUE(loom_testbench_numeric_type_matches_storage(
          numeric_type, format.unsigned_storage));
      EXPECT_TRUE(loom_testbench_numeric_type_matches_storage(
          numeric_type, format.signless_storage));
      const auto& other = formats[(format_index + 1) % IREE_ARRAYSIZE(formats)];
      EXPECT_FALSE(loom_testbench_numeric_type_matches_storage(
          numeric_type, other.signless_storage));
    }
  }
}

}  // namespace
}  // namespace loom
