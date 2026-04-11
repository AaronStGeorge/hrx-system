// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/vector/memory.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/view/ops.h"

namespace loom {
namespace {

class VectorMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t view_vtable_count = 0;
    const loom_op_vtable_t* const* view_vtables =
        loom_view_dialect_vtables(&view_vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_VIEW,
                                                 view_vtables,
                                                 (uint16_t)view_vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void BuildDenseLayout(loom_value_id_t* out_layout) {
    loom_op_t* layout = nullptr;
    IREE_ASSERT_OK(loom_view_layout_dense_build(
        &builder_, loom_type_encoding(), LOOM_LOCATION_UNKNOWN, &layout));
    *out_layout = loom_view_layout_dense_result(layout);
  }

  void BuildStridedLayout(const loom_value_id_t* dynamic_strides,
                          iree_host_size_t dynamic_stride_count,
                          const int64_t* static_strides,
                          iree_host_size_t static_stride_count,
                          loom_value_id_t* out_layout) {
    loom_op_t* layout = nullptr;
    IREE_ASSERT_OK(loom_view_layout_strided_build(
        &builder_, dynamic_strides, dynamic_stride_count, static_strides,
        static_stride_count, loom_type_encoding(), LOOM_LOCATION_UNKNOWN,
        &layout));
    *out_layout = loom_view_layout_strided_result(layout);
  }

  static loom_type_t ViewWithLayout(loom_type_t view_type,
                                    loom_value_id_t layout_id) {
    view_type.encoding_id = (uint16_t)layout_id;
    view_type.encoding_flags = LOOM_ENCODING_FLAG_SSA;
    return view_type;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t builder_;
};

TEST_F(VectorMemoryTest, DenseLayoutComputesLaneOffsets) {
  loom_value_id_t layout = LOOM_VALUE_ID_INVALID;
  BuildDenseLayout(&layout);
  loom_type_t view_type = ViewWithLayout(
      loom_type_shaped_2d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(8), loom_dim_pack_static(16),
                          /*encoding_id=*/0),
      layout);
  loom_type_t vector_type =
      loom_type_shaped_2d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(2), loom_dim_pack_static(4),
                          /*encoding_id=*/0);

  loom_vector_memory_access_t access;
  ASSERT_TRUE(loom_vector_memory_access_describe(module_, view_type,
                                                 vector_type, &access));
  EXPECT_EQ(access.layout_kind, LOOM_VECTOR_MEMORY_LAYOUT_DENSE);

  int64_t row_stride = 0;
  int64_t column_stride = 0;
  EXPECT_TRUE(
      loom_vector_memory_access_static_axis_stride(&access, 0, &row_stride));
  EXPECT_TRUE(
      loom_vector_memory_access_static_axis_stride(&access, 1, &column_stride));
  EXPECT_EQ(row_stride, 16);
  EXPECT_EQ(column_stride, 1);

  int64_t static_indices[] = {3, 5};
  loom_attribute_t static_index_attr = loom_attr_i64_array(
      static_indices, (uint16_t)IREE_ARRAYSIZE(static_indices));
  int64_t lane_indices[] = {1, 3};
  int64_t element_offset = 0;
  int64_t byte_offset = 0;
  EXPECT_TRUE(loom_vector_memory_access_static_lane_element_offset(
      &access, static_index_attr, lane_indices,
      (uint8_t)IREE_ARRAYSIZE(lane_indices), &element_offset));
  EXPECT_TRUE(loom_vector_memory_access_static_lane_byte_offset(
      &access, static_index_attr, lane_indices,
      (uint8_t)IREE_ARRAYSIZE(lane_indices), &byte_offset));
  EXPECT_EQ(element_offset, 72);
  EXPECT_EQ(byte_offset, 288);
}

TEST_F(VectorMemoryTest, StridedLayoutComputesPaddedLaneOffsets) {
  int64_t static_strides[] = {64, 1};
  loom_value_id_t layout = LOOM_VALUE_ID_INVALID;
  BuildStridedLayout(
      /*dynamic_strides=*/nullptr, /*dynamic_stride_count=*/0, static_strides,
      IREE_ARRAYSIZE(static_strides), &layout);
  loom_type_t view_type = ViewWithLayout(
      loom_type_shaped_2d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(8), loom_dim_pack_static(16),
                          /*encoding_id=*/0),
      layout);
  loom_type_t vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(4), /*encoding_id=*/0);

  loom_vector_memory_access_t access;
  ASSERT_TRUE(loom_vector_memory_access_describe(module_, view_type,
                                                 vector_type, &access));
  EXPECT_EQ(access.layout_kind, LOOM_VECTOR_MEMORY_LAYOUT_STRIDED);

  int64_t row_extent = 0;
  int64_t column_extent = 0;
  EXPECT_TRUE(
      loom_vector_memory_access_static_axis_extent(&access, 0, &row_extent));
  EXPECT_TRUE(
      loom_vector_memory_access_static_axis_extent(&access, 1, &column_extent));
  EXPECT_EQ(row_extent, 1);
  EXPECT_EQ(column_extent, 4);

  int64_t static_indices[] = {2, 3};
  loom_attribute_t static_index_attr = loom_attr_i64_array(
      static_indices, (uint16_t)IREE_ARRAYSIZE(static_indices));
  int64_t lane_indices[] = {2};
  int64_t element_offset = 0;
  EXPECT_TRUE(loom_vector_memory_access_static_lane_element_offset(
      &access, static_index_attr, lane_indices,
      (uint8_t)IREE_ARRAYSIZE(lane_indices), &element_offset));
  EXPECT_EQ(element_offset, 133);
}

TEST_F(VectorMemoryTest, DenseLayoutReportsUnknownStaticSuffixStride) {
  loom_value_id_t layout = LOOM_VALUE_ID_INVALID;
  BuildDenseLayout(&layout);
  loom_type_t view_type = ViewWithLayout(
      loom_type_shaped_2d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(8), loom_dim_pack_dynamic(42),
                          /*encoding_id=*/0),
      layout);
  loom_type_t vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(4), /*encoding_id=*/0);

  loom_vector_memory_access_t access;
  ASSERT_TRUE(loom_vector_memory_access_describe(module_, view_type,
                                                 vector_type, &access));

  int64_t row_stride = 0;
  int64_t column_stride = 0;
  EXPECT_FALSE(
      loom_vector_memory_access_static_axis_stride(&access, 0, &row_stride));
  EXPECT_TRUE(
      loom_vector_memory_access_static_axis_stride(&access, 1, &column_stride));
  EXPECT_EQ(column_stride, 1);
}

TEST_F(VectorMemoryTest, StridedLayoutSupportsDynamicStride) {
  int64_t static_strides[] = {INT64_MIN, 1};
  loom_value_id_t dynamic_stride = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_module_block(module_),
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &dynamic_stride));
  loom_value_id_t layout = LOOM_VALUE_ID_INVALID;
  BuildStridedLayout(&dynamic_stride, /*dynamic_stride_count=*/1,
                     static_strides, IREE_ARRAYSIZE(static_strides), &layout);
  loom_type_t view_type = ViewWithLayout(
      loom_type_shaped_2d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(8), loom_dim_pack_static(16),
                          /*encoding_id=*/0),
      layout);
  loom_type_t vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(4), /*encoding_id=*/0);

  loom_vector_memory_access_t access;
  ASSERT_TRUE(loom_vector_memory_access_describe(module_, view_type,
                                                 vector_type, &access));
  EXPECT_EQ(access.layout_kind, LOOM_VECTOR_MEMORY_LAYOUT_STRIDED);

  int64_t row_stride = 0;
  int64_t column_stride = 0;
  EXPECT_FALSE(
      loom_vector_memory_access_static_axis_stride(&access, 0, &row_stride));
  EXPECT_TRUE(
      loom_vector_memory_access_static_axis_stride(&access, 1, &column_stride));
  EXPECT_EQ(column_stride, 1);

  int64_t static_indices[] = {1, 0};
  loom_attribute_t static_index_attr = loom_attr_i64_array(
      static_indices, (uint16_t)IREE_ARRAYSIZE(static_indices));
  int64_t lane_indices[] = {0};
  int64_t element_offset = 0;
  EXPECT_FALSE(loom_vector_memory_access_static_lane_element_offset(
      &access, static_index_attr, lane_indices,
      (uint8_t)IREE_ARRAYSIZE(lane_indices), &element_offset));

  static_indices[0] = 0;
  EXPECT_TRUE(loom_vector_memory_access_static_lane_element_offset(
      &access, static_index_attr, lane_indices,
      (uint8_t)IREE_ARRAYSIZE(lane_indices), &element_offset));
  EXPECT_EQ(element_offset, 0);
}

TEST_F(VectorMemoryTest, ByteOffsetRejectsSubByteElementType) {
  loom_value_id_t layout = LOOM_VALUE_ID_INVALID;
  BuildDenseLayout(&layout);
  loom_type_t view_type = ViewWithLayout(
      loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I1,
                          loom_dim_pack_static(8), /*encoding_id=*/0),
      layout);
  loom_type_t vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I1,
                          loom_dim_pack_static(4), /*encoding_id=*/0);

  loom_vector_memory_access_t access;
  ASSERT_TRUE(loom_vector_memory_access_describe(module_, view_type,
                                                 vector_type, &access));

  int64_t static_indices[] = {2};
  loom_attribute_t static_index_attr = loom_attr_i64_array(
      static_indices, (uint16_t)IREE_ARRAYSIZE(static_indices));
  int64_t lane_indices[] = {1};
  int64_t element_offset = 0;
  int64_t byte_offset = 0;
  EXPECT_TRUE(loom_vector_memory_access_static_lane_element_offset(
      &access, static_index_attr, lane_indices,
      (uint8_t)IREE_ARRAYSIZE(lane_indices), &element_offset));
  EXPECT_FALSE(loom_vector_memory_access_static_lane_byte_offset(
      &access, static_index_attr, lane_indices,
      (uint8_t)IREE_ARRAYSIZE(lane_indices), &byte_offset));
  EXPECT_EQ(element_offset, 3);
}

}  // namespace
}  // namespace loom
