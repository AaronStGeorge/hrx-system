// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/hal/executable.h"

#include <string>

#include "iree/hal/utils/executable_header.h"
#include "iree/schemas/amdgpu_executable_def_reader.h"
#include "iree/schemas/amdgpu_executable_def_verifier.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

std::string FlatbufferString(flatbuffers_string_t string) {
  return std::string(string, flatbuffers_string_len(string));
}

TEST(AmdgpuHalExecutableTest, WrapsHsacoInHalExecutableContainer) {
  static const uint8_t kHsaco[] = {
      0x7f, 'E', 'L', 'F', 'l', 'o',  'o',  'm',
      'h',  's', 'a', 'c', 'o', '\0', '\1', '\2',
  };
  const loom_amdgpu_hal_kernel_binding_flags_t binding_flags[] = {
      LOOM_AMDGPU_HAL_KERNEL_BINDING_READ_ONLY,
      LOOM_AMDGPU_HAL_KERNEL_BINDING_READ_ONLY |
          LOOM_AMDGPU_HAL_KERNEL_BINDING_INDIRECT,
  };
  const loom_amdgpu_hal_kernel_export_t export_def = {
      .symbol_name = IREE_SV("loom_kernel.kd"),
      .workgroup_size = {.x = 64, .y = 2, .z = 1},
      .constant_count = 7,
      .binding_flags = binding_flags,
      .binding_count = IREE_ARRAYSIZE(binding_flags),
  };

  loom_amdgpu_hal_executable_t executable = {};
  IREE_ASSERT_OK(loom_amdgpu_package_hal_executable(
      IREE_SV("amdgcn-amd-amdhsa--gfx1100"),
      iree_make_const_byte_span(kHsaco, sizeof(kHsaco)), &export_def, 1,
      iree_allocator_system(), &executable));

  iree_const_byte_span_t flatbuffer_data = iree_const_byte_span_empty();
  IREE_ASSERT_OK(iree_hal_read_executable_flatbuffer_header(
      iree_make_const_byte_span(executable.data, executable.data_length),
      /*unsafe_infer_size=*/false,
      iree_hal_amdgpu_ExecutableDef_file_identifier, &flatbuffer_data));
  const int verify_ret = iree_hal_amdgpu_ExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  ASSERT_EQ(verify_ret, flatcc_verify_ok)
      << flatcc_verify_error_string(verify_ret);

  iree_hal_amdgpu_ExecutableDef_table_t executable_def =
      iree_hal_amdgpu_ExecutableDef_as_root(flatbuffer_data.data);
  EXPECT_EQ(
      FlatbufferString(iree_hal_amdgpu_ExecutableDef_isa_get(executable_def)),
      "amdgcn-amd-amdhsa--gfx1100");

  iree_hal_amdgpu_ModuleDef_vec_t modules =
      iree_hal_amdgpu_ExecutableDef_modules_get(executable_def);
  ASSERT_EQ(iree_hal_amdgpu_ModuleDef_vec_len(modules), 1u);
  iree_hal_amdgpu_ModuleDef_table_t module =
      iree_hal_amdgpu_ModuleDef_vec_at(modules, 0);
  flatbuffers_string_t image = iree_hal_amdgpu_ModuleDef_image_get(module);
  ASSERT_EQ(flatbuffers_string_len(image), sizeof(kHsaco));
  EXPECT_EQ(std::string(image, flatbuffers_string_len(image)),
            std::string(reinterpret_cast<const char*>(kHsaco), sizeof(kHsaco)));

  iree_hal_amdgpu_ExportDef_vec_t exports =
      iree_hal_amdgpu_ExecutableDef_exports_get(executable_def);
  ASSERT_EQ(iree_hal_amdgpu_ExportDef_vec_len(exports), 1u);
  iree_hal_amdgpu_ExportDef_table_t export_table =
      iree_hal_amdgpu_ExportDef_vec_at(exports, 0);
  EXPECT_EQ(
      FlatbufferString(iree_hal_amdgpu_ExportDef_symbol_name_get(export_table)),
      "loom_kernel.kd");
  iree_hal_amdgpu_Dims_struct_t workgroup_size =
      iree_hal_amdgpu_ExportDef_workgroup_size_get(export_table);
  ASSERT_NE(workgroup_size, nullptr);
  EXPECT_EQ(workgroup_size->x, 64u);
  EXPECT_EQ(workgroup_size->y, 2u);
  EXPECT_EQ(workgroup_size->z, 1u);
  EXPECT_EQ(iree_hal_amdgpu_ExportDef_constant_count_get(export_table), 7u);

  iree_hal_amdgpu_BindingBits_vec_t emitted_binding_flags =
      iree_hal_amdgpu_ExportDef_binding_flags_get(export_table);
  ASSERT_EQ(iree_hal_amdgpu_BindingBits_vec_len(emitted_binding_flags), 2u);
  EXPECT_EQ(iree_hal_amdgpu_BindingBits_vec_at(emitted_binding_flags, 0),
            iree_hal_amdgpu_BindingBits_READ_ONLY);
  EXPECT_EQ(iree_hal_amdgpu_BindingBits_vec_at(emitted_binding_flags, 1),
            iree_hal_amdgpu_BindingBits_READ_ONLY |
                iree_hal_amdgpu_BindingBits_INDIRECT);

  loom_amdgpu_hal_executable_deinitialize(&executable, iree_allocator_system());
}

TEST(AmdgpuHalExecutableTest, RejectsUnknownBindingFlags) {
  static const uint8_t kHsaco[] = {0x7f, 'E', 'L', 'F'};
  const loom_amdgpu_hal_kernel_binding_flags_t binding_flags[] = {
      UINT64_C(1) << 63,
  };
  const loom_amdgpu_hal_kernel_export_t export_def = {
      .symbol_name = IREE_SV("loom_kernel.kd"),
      .workgroup_size = {.x = 64, .y = 1, .z = 1},
      .constant_count = 0,
      .binding_flags = binding_flags,
      .binding_count = IREE_ARRAYSIZE(binding_flags),
  };

  loom_amdgpu_hal_executable_t executable = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_package_hal_executable(
          IREE_SV("amdgcn-amd-amdhsa--gfx1100"),
          iree_make_const_byte_span(kHsaco, sizeof(kHsaco)), &export_def, 1,
          iree_allocator_system(), &executable));
  loom_amdgpu_hal_executable_deinitialize(&executable, iree_allocator_system());
}

}  // namespace
}  // namespace loom
