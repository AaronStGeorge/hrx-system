// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/hsaco_kernarg_layout.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "iree/hal/drivers/amdgpu/abi/kernel_args.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

static iree_hal_amdgpu_hsaco_metadata_arg_t MakeArg(
    iree_string_view_t name, uint32_t offset, uint32_t size,
    iree_hal_amdgpu_hsaco_metadata_arg_kind_t kind,
    iree_string_view_t value_kind) {
  return iree_hal_amdgpu_hsaco_metadata_arg_t{
      .name = name,
      .offset = offset,
      .size = size,
      .alignment = size >= 8 ? 8u : 4u,
      .kind = kind,
      .value_kind = value_kind,
  };
}

static iree_hal_amdgpu_hsaco_metadata_kernel_t MakeKernel(
    uint32_t kernarg_segment_size,
    const std::vector<iree_hal_amdgpu_hsaco_metadata_arg_t>& args) {
  return iree_hal_amdgpu_hsaco_metadata_kernel_t{
      .symbol_name = IREE_SV("test.kd"),
      .reflection_name = IREE_SV("test"),
      .kernarg_segment_size = kernarg_segment_size,
      .kernarg_segment_alignment = 8,
      .arg_count = args.size(),
      .args = args.data(),
  };
}

TEST(HsacoKernargLayoutTest, DerivesInterleavedVisibleParameterLayout) {
  std::vector<iree_hal_amdgpu_hsaco_metadata_arg_t> args = {
      MakeArg(IREE_SV("lhs"), 0, 8,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_GLOBAL_BUFFER,
              IREE_SV("global_buffer")),
      MakeArg(IREE_SV("scale"), 8, 2,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_BY_VALUE,
              IREE_SV("by_value")),
      MakeArg(IREE_SV("rhs"), 16, 8,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_GLOBAL_BUFFER,
              IREE_SV("global_buffer")),
      MakeArg(IREE_SV("bias"), 24, 6,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_BY_VALUE,
              IREE_SV("by_value")),
  };
  iree_hal_amdgpu_hsaco_metadata_kernel_t kernel =
      MakeKernel(/*kernarg_segment_size=*/32, args);

  iree_hal_amdgpu_hsaco_kernarg_layout_t layout;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_hsaco_kernarg_layout_calculate(&kernel, &layout));
  EXPECT_EQ(layout.flags, IREE_HAL_AMDGPU_HSACO_KERNARG_LAYOUT_FLAG_NONE);
  EXPECT_EQ(layout.parameter_count, 4);
  EXPECT_EQ(layout.binding_count, 2);
  EXPECT_EQ(layout.constant_byte_length, 8);
  EXPECT_EQ(layout.explicit_kernarg_size, 30);
  EXPECT_EQ(layout.implicit_args_offset, UINT16_MAX);
  EXPECT_EQ(layout.total_kernarg_size, 32);
  EXPECT_EQ(layout.kernarg_alignment, 8);

  std::vector<iree_hal_amdgpu_hsaco_kernarg_parameter_t> parameters(
      layout.parameter_count);
  IREE_ASSERT_OK(iree_hal_amdgpu_hsaco_kernarg_layout_populate_parameters(
      &kernel, parameters.size(), parameters.data()));

  EXPECT_EQ(parameters[0].kind,
            IREE_HAL_AMDGPU_HSACO_KERNARG_PARAMETER_KIND_BINDING);
  EXPECT_EQ(parameters[0].kernarg_offset, 0);
  EXPECT_EQ(parameters[0].source_offset, 0);
  EXPECT_EQ(parameters[0].byte_length, 8);

  EXPECT_EQ(parameters[1].kind,
            IREE_HAL_AMDGPU_HSACO_KERNARG_PARAMETER_KIND_CONSTANT);
  EXPECT_EQ(parameters[1].kernarg_offset, 8);
  EXPECT_EQ(parameters[1].source_offset, 0);
  EXPECT_EQ(parameters[1].byte_length, 2);

  EXPECT_EQ(parameters[2].kind,
            IREE_HAL_AMDGPU_HSACO_KERNARG_PARAMETER_KIND_BINDING);
  EXPECT_EQ(parameters[2].kernarg_offset, 16);
  EXPECT_EQ(parameters[2].source_offset, 1);
  EXPECT_EQ(parameters[2].byte_length, 8);

  EXPECT_EQ(parameters[3].kind,
            IREE_HAL_AMDGPU_HSACO_KERNARG_PARAMETER_KIND_CONSTANT);
  EXPECT_EQ(parameters[3].kernarg_offset, 24);
  EXPECT_EQ(parameters[3].source_offset, 2);
  EXPECT_EQ(parameters[3].byte_length, 6);
}

TEST(HsacoKernargLayoutTest, DerivesImplicitArgsSuffixLayout) {
  std::vector<iree_hal_amdgpu_hsaco_metadata_arg_t> args = {
      MakeArg(IREE_SV("buffer"), 0, 8,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_GLOBAL_BUFFER,
              IREE_SV("global_buffer")),
      MakeArg(IREE_SV("value"), 8, 4,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_BY_VALUE,
              IREE_SV("by_value")),
      MakeArg(IREE_SV("grid_x"), 16, 8,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_HIDDEN,
              IREE_SV("hidden_global_offset_x")),
  };
  iree_hal_amdgpu_hsaco_metadata_kernel_t kernel = MakeKernel(
      /*kernarg_segment_size=*/16 + IREE_AMDGPU_KERNEL_IMPLICIT_ARGS_SIZE,
      args);

  iree_hal_amdgpu_hsaco_kernarg_layout_t layout;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_hsaco_kernarg_layout_calculate(&kernel, &layout));
  EXPECT_EQ(layout.flags,
            IREE_HAL_AMDGPU_HSACO_KERNARG_LAYOUT_FLAG_IMPLICIT_ARGS);
  EXPECT_EQ(layout.parameter_count, 2);
  EXPECT_EQ(layout.binding_count, 1);
  EXPECT_EQ(layout.constant_byte_length, 4);
  EXPECT_EQ(layout.explicit_kernarg_size, 12);
  EXPECT_EQ(layout.implicit_args_offset, 16);
  EXPECT_EQ(layout.total_kernarg_size,
            16 + IREE_AMDGPU_KERNEL_IMPLICIT_ARGS_SIZE);
}

TEST(HsacoKernargLayoutTest, RejectsHiddenArgsInterleavedWithVisibleArgs) {
  std::vector<iree_hal_amdgpu_hsaco_metadata_arg_t> args = {
      MakeArg(IREE_SV("buffer"), 0, 8,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_GLOBAL_BUFFER,
              IREE_SV("global_buffer")),
      MakeArg(IREE_SV("grid_x"), 8, 8,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_HIDDEN,
              IREE_SV("hidden_global_offset_x")),
      MakeArg(IREE_SV("value"), 16, 4,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_BY_VALUE,
              IREE_SV("by_value")),
  };
  iree_hal_amdgpu_hsaco_metadata_kernel_t kernel =
      MakeKernel(/*kernarg_segment_size=*/24, args);

  iree_hal_amdgpu_hsaco_kernarg_layout_t layout;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_hsaco_kernarg_layout_calculate(&kernel, &layout));
}

TEST(HsacoKernargLayoutTest, RejectsUnalignedImplicitArgsSuffix) {
  std::vector<iree_hal_amdgpu_hsaco_metadata_arg_t> args = {
      MakeArg(IREE_SV("buffer"), 0, 8,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_GLOBAL_BUFFER,
              IREE_SV("global_buffer")),
      MakeArg(IREE_SV("grid_x"), 12, 8,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_HIDDEN,
              IREE_SV("hidden_global_offset_x")),
  };
  iree_hal_amdgpu_hsaco_metadata_kernel_t kernel = MakeKernel(
      /*kernarg_segment_size=*/12 + IREE_AMDGPU_KERNEL_IMPLICIT_ARGS_SIZE,
      args);

  iree_hal_amdgpu_hsaco_kernarg_layout_t layout;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_hsaco_kernarg_layout_calculate(&kernel, &layout));
}

TEST(HsacoKernargLayoutTest, RejectsUnsupportedVisibleArgumentKind) {
  std::vector<iree_hal_amdgpu_hsaco_metadata_arg_t> args = {
      MakeArg(IREE_SV("image"), 0, 8,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_IMAGE, IREE_SV("image")),
  };
  iree_hal_amdgpu_hsaco_metadata_kernel_t kernel =
      MakeKernel(/*kernarg_segment_size=*/8, args);

  iree_hal_amdgpu_hsaco_kernarg_layout_t layout;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_hsaco_kernarg_layout_calculate(&kernel, &layout));
}

TEST(HsacoKernargLayoutTest, RejectsKernargSegmentsBeyondLayoutLimit) {
  std::vector<iree_hal_amdgpu_hsaco_metadata_arg_t> args;
  iree_hal_amdgpu_hsaco_metadata_kernel_t kernel =
      MakeKernel(/*kernarg_segment_size=*/UINT16_MAX + 1u, args);

  iree_hal_amdgpu_hsaco_kernarg_layout_t layout;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_amdgpu_hsaco_kernarg_layout_calculate(&kernel, &layout));
}

TEST(HsacoKernargLayoutTest, RejectsParameterOutputCapacityMismatch) {
  std::vector<iree_hal_amdgpu_hsaco_metadata_arg_t> args = {
      MakeArg(IREE_SV("buffer"), 0, 8,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_GLOBAL_BUFFER,
              IREE_SV("global_buffer")),
  };
  iree_hal_amdgpu_hsaco_metadata_kernel_t kernel =
      MakeKernel(/*kernarg_segment_size=*/8, args);
  std::vector<iree_hal_amdgpu_hsaco_kernarg_parameter_t> parameters(1);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_amdgpu_hsaco_kernarg_layout_populate_parameters(
          &kernel, /*parameter_capacity=*/0, parameters.data()));
}

}  // namespace
}  // namespace iree::hal::amdgpu
