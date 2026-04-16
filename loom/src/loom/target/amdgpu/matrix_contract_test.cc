// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/amdgpu/matrix_contract.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

const loom_amdgpu_matrix_contract_descriptor_t* FindDescriptor(
    const char* name) {
  return loom_amdgpu_matrix_contract_find_by_name(iree_make_cstring_view(name));
}

TEST(MatrixContractTest, DescriptorNamesAreUnique) {
  const iree_host_size_t count = loom_amdgpu_matrix_contract_descriptor_count();
  ASSERT_GT(count, 0u);
  for (iree_host_size_t i = 0; i < count; ++i) {
    const loom_amdgpu_matrix_contract_descriptor_t* lhs =
        loom_amdgpu_matrix_contract_descriptor_at(i);
    ASSERT_NE(lhs, nullptr);
    EXPECT_FALSE(iree_string_view_is_empty(lhs->name));
    EXPECT_FALSE(iree_string_view_is_empty(lhs->llvm_intrinsic_name));
    for (iree_host_size_t j = i + 1; j < count; ++j) {
      const loom_amdgpu_matrix_contract_descriptor_t* rhs =
          loom_amdgpu_matrix_contract_descriptor_at(j);
      ASSERT_NE(rhs, nullptr);
      EXPECT_FALSE(iree_string_view_equal(lhs->name, rhs->name))
          << ToString(lhs->name);
    }
  }
  EXPECT_EQ(loom_amdgpu_matrix_contract_descriptor_at(count), nullptr);
}

TEST(MatrixContractTest, LookupRejectsMissingDescriptor) {
  EXPECT_EQ(FindDescriptor("mfma.f32.4x4x4.imaginary"), nullptr);
}

TEST(MatrixContractTest, Gfx942Fp8MfmaDescriptor) {
  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      FindDescriptor("mfma.f32.16x16x32.fp8.fp8");
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->family, LOOM_AMDGPU_MATRIX_FAMILY_MFMA);
  EXPECT_EQ(ToString(descriptor->llvm_intrinsic_name),
            "llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8");
  EXPECT_EQ(descriptor->tile_shape.result_row_count, 16);
  EXPECT_EQ(descriptor->tile_shape.result_column_count, 16);
  EXPECT_EQ(descriptor->tile_shape.reduction_count, 32);
  EXPECT_EQ(descriptor->lhs_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_FP8);
  EXPECT_EQ(descriptor->rhs_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_FP8);
  EXPECT_EQ(descriptor->accumulator_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32);
  EXPECT_EQ(descriptor->result_payload.register_count, 4);
  EXPECT_EQ(descriptor->scale_kind, LOOM_AMDGPU_MATRIX_SCALE_NONE);
}

TEST(MatrixContractTest, Gfx90aBf16OneKMfmaDescriptor) {
  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      FindDescriptor("mfma.f32.16x16x16.bf16.1k");
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->family, LOOM_AMDGPU_MATRIX_FAMILY_MFMA);
  EXPECT_EQ(ToString(descriptor->llvm_intrinsic_name),
            "llvm.amdgcn.mfma.f32.16x16x16.bf16.1k");
  EXPECT_EQ(descriptor->tile_shape.result_row_count, 16);
  EXPECT_EQ(descriptor->tile_shape.result_column_count, 16);
  EXPECT_EQ(descriptor->tile_shape.reduction_count, 16);
  EXPECT_EQ(descriptor->lhs_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16);
  EXPECT_EQ(descriptor->lhs_payload.register_count, 2);
  EXPECT_EQ(descriptor->accumulator_payload.register_count, 4);
}

TEST(MatrixContractTest, Gfx950ScaledMfmaDescriptor) {
  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      FindDescriptor("mfma.scale.f32.32x32x64.f8f6f4");
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->family, LOOM_AMDGPU_MATRIX_FAMILY_MFMA);
  EXPECT_EQ(descriptor->tile_shape.result_row_count, 32);
  EXPECT_EQ(descriptor->tile_shape.result_column_count, 32);
  EXPECT_EQ(descriptor->tile_shape.reduction_count, 64);
  EXPECT_EQ(descriptor->lhs_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4);
  EXPECT_EQ(descriptor->rhs_payload.register_count, 0);
  EXPECT_EQ(descriptor->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED);
  EXPECT_EQ(descriptor->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS);
  EXPECT_EQ(
      descriptor->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_ZERO_SCALE_FALLBACK,
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_ZERO_SCALE_FALLBACK);
  EXPECT_EQ(descriptor->scale_kind, LOOM_AMDGPU_MATRIX_SCALE_32);
}

TEST(MatrixContractTest, Gfx1250WmmaScale16Descriptor) {
  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      FindDescriptor("wmma.scale16.f32.16x16x128.f8f6f4");
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->family, LOOM_AMDGPU_MATRIX_FAMILY_WMMA);
  EXPECT_EQ(descriptor->tile_shape.reduction_count, 128);
  EXPECT_EQ(descriptor->scale_kind, LOOM_AMDGPU_MATRIX_SCALE_16);
  EXPECT_EQ(descriptor->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE);
}

TEST(MatrixContractTest, SparseDescriptorsCarrySparseFlag) {
  const loom_amdgpu_matrix_contract_descriptor_t* smfmac =
      FindDescriptor("smfmac.f32.16x16x128.fp8.fp8");
  ASSERT_NE(smfmac, nullptr);
  EXPECT_EQ(smfmac->family, LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC);
  EXPECT_EQ(smfmac->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE);

  const loom_amdgpu_matrix_contract_descriptor_t* swmmac =
      FindDescriptor("swmmac.i32.16x16x64.iu4");
  ASSERT_NE(swmmac, nullptr);
  EXPECT_EQ(swmmac->family, LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC);
  EXPECT_EQ(swmmac->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE);
  EXPECT_EQ(swmmac->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT);
}

TEST(MatrixContractTest, SparseFp8CrossProductDescriptors) {
  const loom_amdgpu_matrix_contract_descriptor_t* smfmac =
      FindDescriptor("smfmac.f32.16x16x64.bf8.fp8");
  ASSERT_NE(smfmac, nullptr);
  EXPECT_EQ(smfmac->family, LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC);
  EXPECT_EQ(smfmac->tile_shape.reduction_count, 64);
  EXPECT_EQ(smfmac->lhs_payload.numeric_type, LOOM_AMDGPU_MATRIX_NUMERIC_BF8);
  EXPECT_EQ(smfmac->rhs_payload.numeric_type, LOOM_AMDGPU_MATRIX_NUMERIC_FP8);
  EXPECT_EQ(smfmac->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE);

  const loom_amdgpu_matrix_contract_descriptor_t* swmmac =
      FindDescriptor("swmmac.f32.16x16x32.fp8.bf8");
  ASSERT_NE(swmmac, nullptr);
  EXPECT_EQ(swmmac->family, LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC);
  EXPECT_EQ(swmmac->tile_shape.reduction_count, 32);
  EXPECT_EQ(swmmac->lhs_payload.numeric_type, LOOM_AMDGPU_MATRIX_NUMERIC_FP8);
  EXPECT_EQ(swmmac->rhs_payload.numeric_type, LOOM_AMDGPU_MATRIX_NUMERIC_BF8);
  EXPECT_EQ(swmmac->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE);
}

TEST(MatrixContractTest, WmmaFp8CrossProductDescriptors) {
  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      FindDescriptor("wmma.f32.16x16x16.bf8.fp8");
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->family, LOOM_AMDGPU_MATRIX_FAMILY_WMMA);
  EXPECT_EQ(descriptor->tile_shape.reduction_count, 16);
  EXPECT_EQ(descriptor->lhs_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF8);
  EXPECT_EQ(descriptor->rhs_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_FP8);
}

TEST(MatrixContractTest, ProcessorFeatureBitsGateAvailability) {
  loom_amdgpu_matrix_feature_bits_t gfx942_features = 0;
  IREE_ASSERT_OK(loom_amdgpu_matrix_feature_bits_for_processor(
      IREE_SV("gfx942"), &gfx942_features));
  loom_amdgpu_matrix_feature_bits_t gfx950_features = 0;
  IREE_ASSERT_OK(loom_amdgpu_matrix_feature_bits_for_processor(
      IREE_SV("gfx950"), &gfx950_features));
  loom_amdgpu_matrix_feature_bits_t gfx1250_features = 0;
  IREE_ASSERT_OK(loom_amdgpu_matrix_feature_bits_for_processor(
      IREE_SV("gfx1250"), &gfx1250_features));

  const loom_amdgpu_matrix_contract_descriptor_t* fp8_mfma =
      FindDescriptor("mfma.f32.16x16x32.fp8.fp8");
  ASSERT_NE(fp8_mfma, nullptr);
  EXPECT_TRUE(
      loom_amdgpu_matrix_contract_is_available(fp8_mfma, gfx942_features, 64));

  const loom_amdgpu_matrix_contract_descriptor_t* scaled_mfma =
      FindDescriptor("mfma.scale.f32.16x16x128.f8f6f4");
  ASSERT_NE(scaled_mfma, nullptr);
  EXPECT_FALSE(loom_amdgpu_matrix_contract_is_available(scaled_mfma,
                                                        gfx942_features, 64));
  EXPECT_TRUE(loom_amdgpu_matrix_contract_is_available(scaled_mfma,
                                                       gfx950_features, 64));

  const loom_amdgpu_matrix_contract_descriptor_t* scaled_wmma =
      FindDescriptor("wmma.scale.f32.16x16x128.f8f6f4");
  ASSERT_NE(scaled_wmma, nullptr);
  EXPECT_FALSE(loom_amdgpu_matrix_contract_is_available(scaled_wmma,
                                                        gfx950_features, 64));
  EXPECT_TRUE(loom_amdgpu_matrix_contract_is_available(scaled_wmma,
                                                       gfx1250_features, 32));
}

TEST(MatrixContractTest, ScaleFeatureDoesNotGateUnscaledDescriptors) {
  const loom_amdgpu_matrix_contract_descriptor_t* unscaled_mfma =
      FindDescriptor("mfma.f32.16x16x32.f16");
  ASSERT_NE(unscaled_mfma, nullptr);
  EXPECT_TRUE(loom_amdgpu_matrix_contract_is_available(
      unscaled_mfma, LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950, 64));

  const loom_amdgpu_matrix_contract_descriptor_t* scaled_mfma =
      FindDescriptor("mfma.scale.f32.16x16x128.f8f6f4");
  ASSERT_NE(scaled_mfma, nullptr);
  EXPECT_FALSE(loom_amdgpu_matrix_contract_is_available(
      scaled_mfma, LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950, 64));

  const loom_amdgpu_matrix_contract_descriptor_t* unscaled_wmma =
      FindDescriptor("wmma.f32.16x16x32.f16");
  ASSERT_NE(unscaled_wmma, nullptr);
  EXPECT_TRUE(loom_amdgpu_matrix_contract_is_available(
      unscaled_wmma, LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250, 32));

  const loom_amdgpu_matrix_contract_descriptor_t* unscaled_f8_wmma =
      FindDescriptor("wmma.f32.16x16x128.f8f6f4");
  ASSERT_NE(unscaled_f8_wmma, nullptr);
  EXPECT_TRUE(loom_amdgpu_matrix_contract_is_available(
      unscaled_f8_wmma, LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250, 32));

  const loom_amdgpu_matrix_contract_descriptor_t* scaled_wmma =
      FindDescriptor("wmma.scale.f32.16x16x128.f8f6f4");
  ASSERT_NE(scaled_wmma, nullptr);
  EXPECT_FALSE(loom_amdgpu_matrix_contract_is_available(
      scaled_wmma, LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250, 32));
}

TEST(MatrixContractTest, ProcessorFeatureBitsRejectUnknownProcessor) {
  loom_amdgpu_matrix_feature_bits_t feature_bits = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_amdgpu_matrix_feature_bits_for_processor(
                            IREE_SV("gfx9999"), &feature_bits));
}

TEST(MatrixContractTest, NamesAreStable) {
  EXPECT_EQ(ToString(loom_amdgpu_matrix_family_name(
                LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC)),
            "swmmac");
  EXPECT_EQ(ToString(loom_amdgpu_matrix_numeric_type_name(
                LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4)),
            "f8f6f4");
  EXPECT_EQ(
      ToString(loom_amdgpu_matrix_scale_kind_name(LOOM_AMDGPU_MATRIX_SCALE_16)),
      "scale16");
}

}  // namespace
