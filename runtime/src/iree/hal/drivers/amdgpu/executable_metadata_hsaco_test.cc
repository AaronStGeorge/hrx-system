// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/executable_metadata_hsaco.h"

#include <cstdint>
#include <vector>

#include "iree/hal/drivers/amdgpu/abi/kernel_args.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

using iree::testing::status::StatusIs;

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
    iree_string_view_t name, iree_string_view_t symbol_name,
    uint32_t kernarg_segment_size,
    const std::vector<iree_hal_amdgpu_hsaco_metadata_arg_t>& args) {
  return iree_hal_amdgpu_hsaco_metadata_kernel_t{
      .name = name,
      .symbol_name = symbol_name,
      .reflection_name = name,
      .kernarg_segment_size = kernarg_segment_size,
      .kernarg_segment_alignment = 8,
      .group_segment_fixed_size = 16,
      .private_segment_fixed_size = 32,
      .arg_count = args.size(),
      .args = args.data(),
  };
}

static iree_hal_amdgpu_executable_metadata_t* AllocateAndPopulate(
    const iree_hal_amdgpu_hsaco_metadata_t* hsaco_metadata) {
  iree_hal_amdgpu_executable_metadata_counts_t counts;
  IREE_CHECK_OK(iree_hal_amdgpu_executable_metadata_calculate_hsaco_counts(
      hsaco_metadata, &counts));
  iree_hal_amdgpu_executable_metadata_t* metadata = nullptr;
  IREE_CHECK_OK(iree_hal_amdgpu_executable_metadata_allocate(
      &counts, iree_allocator_system(), &metadata));
  static const uint8_t code_object_data[] = {0x7F, 'E', 'L', 'F'};
  IREE_CHECK_OK(iree_hal_amdgpu_executable_metadata_populate_from_hsaco(
      hsaco_metadata,
      iree_make_const_byte_span(code_object_data, sizeof(code_object_data)),
      metadata));
  return metadata;
}

TEST(ExecutableMetadataHsacoTest, PopulatesSparseInterleavedKernelLayout) {
  std::vector<iree_hal_amdgpu_hsaco_metadata_arg_t> args = {
      MakeArg(IREE_SV("scale"), 0, 2,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_BY_VALUE,
              IREE_SV("by_value")),
      MakeArg(IREE_SV("lhs"), 8, 8,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_GLOBAL_BUFFER,
              IREE_SV("global_buffer")),
      MakeArg(IREE_SV("bias"), 20, 6,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_BY_VALUE,
              IREE_SV("by_value")),
      MakeArg(IREE_SV("rhs"), 32, 8,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_GLOBAL_BUFFER,
              IREE_SV("global_buffer")),
  };
  iree_hal_amdgpu_hsaco_metadata_kernel_t kernel =
      MakeKernel(IREE_SV("test"), IREE_SV("test.kd"),
                 /*kernarg_segment_size=*/40, args);
  kernel.has_required_workgroup_size = true;
  kernel.required_workgroup_size[0] = 4;
  kernel.required_workgroup_size[1] = 2;
  kernel.required_workgroup_size[2] = 1;
  iree_hal_amdgpu_hsaco_metadata_t hsaco_metadata = {
      .target = IREE_SV("amdgcn-amd-amdhsa--gfx942"),
      .kernel_count = 1,
      .kernels = &kernel,
  };

  iree_hal_amdgpu_executable_metadata_counts_t counts;
  IREE_ASSERT_OK(iree_hal_amdgpu_executable_metadata_calculate_hsaco_counts(
      &hsaco_metadata, &counts));
  EXPECT_EQ(counts.export_count, 1);
  EXPECT_EQ(counts.parameter_count, 4);
  EXPECT_GT(counts.layout_blob_byte_length, 0);

  iree_hal_amdgpu_executable_metadata_t* metadata =
      AllocateAndPopulate(&hsaco_metadata);
  EXPECT_EQ(metadata->source,
            IREE_HAL_AMDGPU_EXECUTABLE_METADATA_SOURCE_HSACO_MESSAGEPACK);
  EXPECT_EQ(metadata->export_count, 1);
  EXPECT_EQ(metadata->target.data, hsaco_metadata.target.data);
  EXPECT_EQ(metadata->code_object_data.data_length, 4);

  const iree_hal_amdgpu_executable_export_t& export_info = metadata->exports[0];
  EXPECT_EQ(export_info.flags, IREE_HAL_AMDGPU_EXECUTABLE_EXPORT_FLAG_NONE);
  EXPECT_EQ(export_info.workgroup_size[0], 4);
  EXPECT_EQ(export_info.workgroup_size[1], 2);
  EXPECT_EQ(export_info.workgroup_size[2], 1);
  EXPECT_EQ(export_info.fixed_group_segment_size, 16);
  EXPECT_EQ(export_info.fixed_private_segment_size, 32);

  const iree_hal_amdgpu_kernarg_layout_t* layout = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_executable_metadata_resolve_layout(
      metadata, export_info.kernarg_layout, &layout));
  ASSERT_NE(layout, nullptr);
  EXPECT_EQ(layout->kernarg_byte_length, 40);
  EXPECT_EQ(layout->binding_count, 2);
  EXPECT_EQ(layout->constant_span_count, 2);
  EXPECT_EQ(layout->constant_byte_length, 8);
  EXPECT_FALSE(iree_any_bit_set(
      layout->flags,
      IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_PACKED_BINDING_PREFIX));
  EXPECT_TRUE(iree_all_bits_set(
      layout->flags,
      IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_REQUIRES_ZERO_FILL |
          IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_CONTIGUOUS_CONSTANTS));

  const iree_hal_amdgpu_kernarg_binding_slot_t* binding_slots =
      iree_hal_amdgpu_kernarg_layout_binding_slots(layout);
  EXPECT_EQ(binding_slots[0].target_qword_index, 1);
  EXPECT_EQ(binding_slots[1].target_qword_index, 4);

  const iree_hal_amdgpu_kernarg_constant_span_t* constant_spans =
      iree_hal_amdgpu_kernarg_layout_constant_spans(layout);
  EXPECT_EQ(constant_spans[0].target_byte_offset, 0);
  EXPECT_EQ(constant_spans[0].source_byte_offset, 0);
  EXPECT_EQ(constant_spans[0].byte_length, 2);
  EXPECT_EQ(constant_spans[1].target_byte_offset, 20);
  EXPECT_EQ(constant_spans[1].source_byte_offset, 2);
  EXPECT_EQ(constant_spans[1].byte_length, 6);

  EXPECT_EQ(metadata->reflection[0].name.data, kernel.reflection_name.data);
  EXPECT_EQ(metadata->reflection[0].symbol_name.data, kernel.symbol_name.data);
  EXPECT_EQ(metadata->reflection[0].parameter_offset, 0);
  EXPECT_EQ(metadata->reflection[0].parameter_count, 4);
  EXPECT_EQ(metadata->parameters[0].type,
            IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT);
  EXPECT_EQ(metadata->parameters[0].offset, 0);
  EXPECT_EQ(metadata->parameters[0].size, 2);
  EXPECT_EQ(metadata->parameters[1].type,
            IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING);
  EXPECT_EQ(metadata->parameters[1].offset, 0);
  EXPECT_EQ(metadata->parameters[2].type,
            IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT);
  EXPECT_EQ(metadata->parameters[2].offset, 2);
  EXPECT_EQ(metadata->parameters[2].size, 6);
  EXPECT_EQ(metadata->parameters[3].type,
            IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING);
  EXPECT_EQ(metadata->parameters[3].offset, 1);

  iree_hal_amdgpu_executable_metadata_free(metadata);
}

TEST(ExecutableMetadataHsacoTest, PopulatesImplicitArgsSuffixLayout) {
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
  iree_hal_amdgpu_hsaco_metadata_kernel_t kernel =
      MakeKernel(IREE_SV("implicit"), IREE_SV("implicit.kd"),
                 16 + IREE_AMDGPU_KERNEL_IMPLICIT_ARGS_SIZE, args);
  iree_hal_amdgpu_hsaco_metadata_t hsaco_metadata = {
      .kernel_count = 1,
      .kernels = &kernel,
  };

  iree_hal_amdgpu_executable_metadata_t* metadata =
      AllocateAndPopulate(&hsaco_metadata);
  EXPECT_TRUE(iree_any_bit_set(
      metadata->exports[0].flags,
      IREE_HAL_AMDGPU_EXECUTABLE_EXPORT_FLAG_REQUIRES_DISPATCH_WORKGROUP_SIZE));

  const iree_hal_amdgpu_kernarg_layout_t* layout = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_executable_metadata_resolve_layout(
      metadata, metadata->exports[0].kernarg_layout, &layout));
  EXPECT_EQ(layout->implicit_args_byte_offset, 16);
  EXPECT_TRUE(iree_all_bits_set(
      layout->flags,
      IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_IMPLICIT_ARGS |
          IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_REQUIRES_ZERO_FILL));
  EXPECT_EQ(layout->binding_count, 1);
  EXPECT_EQ(layout->constant_byte_length, 4);

  iree_hal_amdgpu_executable_metadata_free(metadata);
}

TEST(ExecutableMetadataHsacoTest, PopulatesElfOnlyCustomDirectExport) {
  iree_hal_amdgpu_hsaco_metadata_elf_kernel_symbol_t symbol = {
      .name = IREE_SV("direct"),
      .symbol_name = IREE_SV("direct.kd"),
  };
  iree_hal_amdgpu_hsaco_metadata_t hsaco_metadata = {
      .elf_kernel_symbol_count = 1,
      .elf_kernel_symbols = &symbol,
  };

  iree_hal_amdgpu_executable_metadata_counts_t counts;
  IREE_ASSERT_OK(iree_hal_amdgpu_executable_metadata_calculate_hsaco_counts(
      &hsaco_metadata, &counts));
  EXPECT_EQ(counts.export_count, 1);
  EXPECT_EQ(counts.parameter_count, 0);
  EXPECT_EQ(counts.layout_blob_byte_length, 0);

  iree_hal_amdgpu_executable_metadata_t* metadata =
      AllocateAndPopulate(&hsaco_metadata);
  EXPECT_EQ(metadata->source,
            IREE_HAL_AMDGPU_EXECUTABLE_METADATA_SOURCE_ELF_SYMBOLS);
  EXPECT_EQ(metadata->reflection[0].name.data, symbol.name.data);
  EXPECT_EQ(metadata->reflection[0].symbol_name.data, symbol.symbol_name.data);
  EXPECT_TRUE(iree_all_bits_set(
      metadata->exports[0].flags,
      IREE_HAL_AMDGPU_EXECUTABLE_EXPORT_FLAG_CUSTOM_DIRECT_ONLY |
          IREE_HAL_AMDGPU_EXECUTABLE_EXPORT_FLAG_REQUIRES_DISPATCH_WORKGROUP_SIZE));
  EXPECT_FALSE(iree_hal_amdgpu_kernarg_layout_ref_is_valid(
      metadata->exports[0].kernarg_layout));

  iree_hal_amdgpu_executable_metadata_free(metadata);
}

TEST(ExecutableMetadataHsacoTest, RejectsUnsupportedVisibleArgumentKind) {
  std::vector<iree_hal_amdgpu_hsaco_metadata_arg_t> args = {
      MakeArg(IREE_SV("image"), 0, 8,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_IMAGE, IREE_SV("image")),
  };
  iree_hal_amdgpu_hsaco_metadata_kernel_t kernel =
      MakeKernel(IREE_SV("bad"), IREE_SV("bad.kd"), 8, args);
  iree_hal_amdgpu_hsaco_metadata_t hsaco_metadata = {
      .kernel_count = 1,
      .kernels = &kernel,
  };
  iree_hal_amdgpu_executable_metadata_counts_t counts;

  EXPECT_THAT(Status(iree_hal_amdgpu_executable_metadata_calculate_hsaco_counts(
                  &hsaco_metadata, &counts)),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST(ExecutableMetadataHsacoTest, RejectsMisalignedGlobalBufferArgument) {
  std::vector<iree_hal_amdgpu_hsaco_metadata_arg_t> args = {
      MakeArg(IREE_SV("buffer"), 4, 8,
              IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_GLOBAL_BUFFER,
              IREE_SV("global_buffer")),
  };
  iree_hal_amdgpu_hsaco_metadata_kernel_t kernel =
      MakeKernel(IREE_SV("bad"), IREE_SV("bad.kd"), 16, args);
  iree_hal_amdgpu_hsaco_metadata_t hsaco_metadata = {
      .kernel_count = 1,
      .kernels = &kernel,
  };
  iree_hal_amdgpu_executable_metadata_counts_t counts;

  EXPECT_THAT(Status(iree_hal_amdgpu_executable_metadata_calculate_hsaco_counts(
                  &hsaco_metadata, &counts)),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST(ExecutableMetadataHsacoTest, RejectsHiddenArgsBeforeVisibleArgsEnd) {
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
      MakeKernel(IREE_SV("bad"), IREE_SV("bad.kd"), 24, args);
  iree_hal_amdgpu_hsaco_metadata_t hsaco_metadata = {
      .kernel_count = 1,
      .kernels = &kernel,
  };
  iree_hal_amdgpu_executable_metadata_counts_t counts;

  EXPECT_THAT(Status(iree_hal_amdgpu_executable_metadata_calculate_hsaco_counts(
                  &hsaco_metadata, &counts)),
              StatusIs(StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace iree::hal::amdgpu
