// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/descriptor.h"

#include <array>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

loom_amdgpu_metadata_kernel_t MinimalMetadataKernel() {
  return loom_amdgpu_metadata_kernel_t{
      .name = IREE_SV("loom_kernel"),
      .descriptor_symbol = IREE_SV("loom_kernel.kd"),
      .kernarg_segment_size = 0,
      .kernarg_segment_alignment = 8,
      .wavefront_size = 32,
      .group_segment_fixed_size = 0,
      .private_segment_fixed_size = 0,
      .sgpr_count = 0,
      .vgpr_count = 0,
      .max_flat_workgroup_size = 64,
      .has_required_workgroup_size = false,
  };
}

uint16_t LoadLeU16(const std::array<uint8_t, 65>& bytes, size_t offset) {
  return (uint16_t)bytes[offset] | ((uint16_t)bytes[offset + 1] << 8);
}

uint32_t LoadLeU32(const std::array<uint8_t, 65>& bytes, size_t offset) {
  return (uint32_t)bytes[offset] | ((uint32_t)bytes[offset + 1] << 8) |
         ((uint32_t)bytes[offset + 2] << 16) |
         ((uint32_t)bytes[offset + 3] << 24);
}

int64_t LoadLeI64(const std::array<uint8_t, 65>& bytes, size_t offset) {
  uint64_t value = 0;
  for (size_t i = 0; i < 8; ++i) {
    value |= (uint64_t)bytes[offset + i] << (8 * i);
  }
  return (int64_t)value;
}

void ExpectZeroRange(const std::array<uint8_t, 65>& bytes, size_t begin,
                     size_t end) {
  for (size_t i = begin; i < end; ++i) {
    EXPECT_EQ(bytes[i], 0u) << "offset " << i;
  }
}

TEST(AmdgpuDescriptorTest, WritesNoArgGfx1100Descriptor) {
  loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
  loom_amdgpu_kernel_descriptor_t descriptor = {};
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
      IREE_SV("gfx1100"), &metadata, -64, &descriptor));
  IREE_ASSERT_OK(
      loom_amdgpu_kernel_descriptor_validate_metadata(&descriptor, &metadata));

  std::array<uint8_t, 65> bytes;
  bytes.fill(0xcc);
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_write(
      &descriptor, iree_make_byte_span(bytes.data(), bytes.size())));

  EXPECT_EQ(LoadLeU32(bytes, 0), 0u);
  EXPECT_EQ(LoadLeU32(bytes, 4), 0u);
  EXPECT_EQ(LoadLeU32(bytes, 8), 0u);
  ExpectZeroRange(bytes, 12, 16);
  EXPECT_EQ(LoadLeI64(bytes, 16), -64);
  ExpectZeroRange(bytes, 24, 44);
  EXPECT_EQ(LoadLeU32(bytes, 44), 0u);
  EXPECT_EQ(LoadLeU32(bytes, 48), 0xe0ac0000u);
  EXPECT_EQ(LoadLeU32(bytes, 52), 0u);
  EXPECT_EQ(LoadLeU16(bytes, 56), 0x0400u);
  EXPECT_EQ(LoadLeU16(bytes, 58), 0u);
  ExpectZeroRange(bytes, 60, 64);
  EXPECT_EQ(bytes[64], 0xccu);
}

TEST(AmdgpuDescriptorTest, EnablesKernargSegmentPointerFromMetadata) {
  loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
  metadata.kernarg_segment_size = 8;
  metadata.sgpr_count = 2;

  loom_amdgpu_kernel_descriptor_t descriptor = {};
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
      IREE_SV("gfx1100"), &metadata, -64, &descriptor));
  IREE_ASSERT_OK(
      loom_amdgpu_kernel_descriptor_validate_metadata(&descriptor, &metadata));

  std::array<uint8_t, 65> bytes;
  bytes.fill(0);
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_write(
      &descriptor, iree_make_byte_span(bytes.data(), bytes.size())));

  EXPECT_EQ(LoadLeU32(bytes, 8), 8u);
  EXPECT_EQ(LoadLeU32(bytes, 52), 0x00000004u);
  EXPECT_EQ(LoadLeU16(bytes, 56), 0x0408u);
}

TEST(AmdgpuDescriptorTest, EncodesResourceAndAbiFields) {
  loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
  metadata.group_segment_fixed_size = 128;
  metadata.private_segment_fixed_size = 16;
  metadata.kernarg_segment_size = 24;
  metadata.sgpr_count = 20;
  metadata.vgpr_count = 9;

  loom_amdgpu_kernel_descriptor_t descriptor = {};
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
      IREE_SV("gfx1100"), &metadata, 256, &descriptor));
  descriptor.user_sgpr_count = 2;
  descriptor.flags |=
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_KERNARG_SEGMENT_PTR |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_X |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_X |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Y |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Z |
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_USES_DYNAMIC_STACK;

  std::array<uint8_t, 65> bytes;
  bytes.fill(0);
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_write(
      &descriptor, iree_make_byte_span(bytes.data(), bytes.size())));

  EXPECT_EQ(LoadLeU32(bytes, 0), 128u);
  EXPECT_EQ(LoadLeU32(bytes, 4), 16u);
  EXPECT_EQ(LoadLeU32(bytes, 8), 24u);
  EXPECT_EQ(LoadLeI64(bytes, 16), 256);
  EXPECT_EQ(LoadLeU32(bytes, 48), 0xe0ac0001u);
  EXPECT_EQ(LoadLeU32(bytes, 52), 0x00001084u);
  EXPECT_EQ(LoadLeU16(bytes, 56), 0x0c08u);
}

TEST(AmdgpuDescriptorTest, RejectsSparseWorkitemIdFlags) {
  loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
  loom_amdgpu_kernel_descriptor_t descriptor = {};
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
      IREE_SV("gfx1100"), &metadata, 0, &descriptor));

  std::array<uint8_t, 64> bytes;
  descriptor.flags |= LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Y;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_kernel_descriptor_write(
          &descriptor, iree_make_byte_span(bytes.data(), bytes.size())));

  descriptor.flags = LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_X |
                     LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Z |
                     LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_WAVEFRONT_SIZE32;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_kernel_descriptor_write(
          &descriptor, iree_make_byte_span(bytes.data(), bytes.size())));
}

TEST(AmdgpuDescriptorTest, SupportsGfx11ProcessorVariants) {
  static const iree_string_view_t processors[] = {
      IREE_SVL("gfx1100"), IREE_SVL("gfx1101"), IREE_SVL("gfx1102"),
      IREE_SVL("gfx1103"), IREE_SVL("gfx1150"), IREE_SVL("gfx1151"),
      IREE_SVL("gfx1152"), IREE_SVL("gfx1153"), IREE_SVL("gfx1170"),
  };
  loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(processors); ++i) {
    loom_amdgpu_kernel_descriptor_t descriptor = {};
    IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
        processors[i], &metadata, -64, &descriptor));
    IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_validate_metadata(&descriptor,
                                                                   &metadata));
  }
}

TEST(AmdgpuDescriptorTest, RejectsMetadataMismatch) {
  loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
  loom_amdgpu_kernel_descriptor_t descriptor = {};
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
      IREE_SV("gfx1100"), &metadata, 0, &descriptor));
  descriptor.kernarg_size = 4;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_amdgpu_kernel_descriptor_validate_metadata(&descriptor, &metadata));
}

TEST(AmdgpuDescriptorTest, RejectsTooFewUserSgprs) {
  loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
  loom_amdgpu_kernel_descriptor_t descriptor = {};
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
      IREE_SV("gfx1100"), &metadata, 0, &descriptor));
  descriptor.user_sgpr_count = 1;
  descriptor.flags |=
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_KERNARG_SEGMENT_PTR;

  std::array<uint8_t, 64> bytes;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_kernel_descriptor_write(
          &descriptor, iree_make_byte_span(bytes.data(), bytes.size())));
}

TEST(AmdgpuDescriptorTest, RejectsLegacyFlatScratchUserSgprsOnGfx1100) {
  loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
  loom_amdgpu_kernel_descriptor_t descriptor = {};
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
      IREE_SV("gfx1100"), &metadata, 0, &descriptor));
  descriptor.user_sgpr_count = 4;
  descriptor.flags |=
      LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER;

  std::array<uint8_t, 64> bytes;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_kernel_descriptor_write(
          &descriptor, iree_make_byte_span(bytes.data(), bytes.size())));
}

TEST(AmdgpuDescriptorTest, RejectsUnsupportedTarget) {
  loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
  loom_amdgpu_kernel_descriptor_t descriptor = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        loom_amdgpu_kernel_descriptor_initialize_from_metadata(
                            IREE_SV("gfx950"), &metadata, 0, &descriptor));
}

TEST(AmdgpuDescriptorTest, RejectsSmallOutputBuffer) {
  loom_amdgpu_metadata_kernel_t metadata = MinimalMetadataKernel();
  loom_amdgpu_kernel_descriptor_t descriptor = {};
  IREE_ASSERT_OK(loom_amdgpu_kernel_descriptor_initialize_from_metadata(
      IREE_SV("gfx1100"), &metadata, 0, &descriptor));

  std::array<uint8_t, 63> bytes;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_kernel_descriptor_write(
          &descriptor, iree_make_byte_span(bytes.data(), bytes.size())));
}

}  // namespace
}  // namespace loom
