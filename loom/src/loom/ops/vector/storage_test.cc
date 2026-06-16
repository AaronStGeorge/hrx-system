// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/vector/storage.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

TEST(VectorStorageTest, StaticRank1LaneCountAcceptsMatchingVector) {
  loom_type_t type = loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                                         loom_dim_pack_static(4), 0);

  EXPECT_EQ(loom_vector_static_rank1_lane_count(type, LOOM_SCALAR_TYPE_I32,
                                                /*maximum_lane_count=*/8),
            4u);
}

TEST(VectorStorageTest, StaticRank1LaneCountRejectsWrongShape) {
  loom_type_t dynamic_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_dynamic(0), 0);
  loom_type_t rank2_type =
      loom_type_shaped_2d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                          loom_dim_pack_static(2), loom_dim_pack_static(2), 0);
  loom_type_t f32_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_type_t over_limit_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(9), 0);

  EXPECT_EQ(loom_vector_static_rank1_lane_count(dynamic_type,
                                                LOOM_SCALAR_TYPE_I32, 8),
            0u);
  EXPECT_EQ(
      loom_vector_static_rank1_lane_count(rank2_type, LOOM_SCALAR_TYPE_I32, 8),
      0u);
  EXPECT_EQ(
      loom_vector_static_rank1_lane_count(f32_type, LOOM_SCALAR_TYPE_I32, 8),
      0u);
  EXPECT_EQ(loom_vector_static_rank1_lane_count(over_limit_type,
                                                LOOM_SCALAR_TYPE_I32, 8),
            0u);
}

TEST(VectorStorageTest, PackedIntegerStorageShapeRoundsToStorageUnits) {
  loom_type_t type = loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I8,
                                         loom_dim_pack_static(5), 0);

  loom_vector_packed_integer_storage_shape_t shape;
  ASSERT_TRUE(loom_vector_packed_integer_storage_shape(
      type, /*storage_unit_bit_count=*/32, /*maximum_storage_unit_count=*/2,
      &shape));
  EXPECT_TRUE(loom_type_equal(shape.type, type));
  EXPECT_EQ(shape.element_type, LOOM_SCALAR_TYPE_I8);
  EXPECT_EQ(shape.lane_count, 5u);
  EXPECT_EQ(shape.element_bit_count, 8u);
  EXPECT_EQ(shape.payload_bit_count, 40u);
  EXPECT_EQ(shape.storage_unit_bit_count, 32u);
  EXPECT_EQ(shape.storage_unit_count, 2u);
}

TEST(VectorStorageTest, PackedIntegerStorageShapeRejectsInvalidShapes) {
  loom_vector_packed_integer_storage_shape_t shape;
  loom_type_t f32_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_type_t dynamic_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I8, loom_dim_pack_dynamic(0), 0);
  loom_type_t over_limit_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I8, loom_dim_pack_static(9), 0);

  EXPECT_FALSE(
      loom_vector_packed_integer_storage_shape(f32_type, 32, 2, &shape));
  EXPECT_FALSE(
      loom_vector_packed_integer_storage_shape(dynamic_type, 32, 2, &shape));
  EXPECT_FALSE(
      loom_vector_packed_integer_storage_shape(over_limit_type, 32, 2, &shape));
  EXPECT_FALSE(
      loom_vector_packed_integer_storage_shape(over_limit_type, 0, 2, &shape));
  EXPECT_FALSE(
      loom_vector_packed_integer_storage_shape(over_limit_type, 32, 0, &shape));
}

}  // namespace
}  // namespace loom
