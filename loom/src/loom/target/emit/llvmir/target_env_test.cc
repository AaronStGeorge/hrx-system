// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/target_env.h"

#include <memory>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/emit/llvmir/amdgpu/target_env.h"
#include "loom/target/emit/llvmir/builder.h"
#include "loom/target/emit/llvmir/target_presets.h"
#include "loom/target/emit/llvmir/text_writer.h"
#include "loom/target/emit/llvmir/verify.h"
#include "loom/target/emit/llvmir/x86/target_env.h"
#include "loom/util/stream.h"

namespace loom {
namespace {

using ModulePtr =
    std::unique_ptr<loom_llvmir_module_t, void (*)(loom_llvmir_module_t*)>;

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

std::string WriteText(const loom_llvmir_module_t* module) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_CHECK_OK(loom_llvmir_text_write_module(module, &stream));
  std::string text(iree_string_builder_buffer(&builder),
                   iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);
  return text;
}

TEST(LlvmIrTargetEnvTest, X86ObjectProfileNamesObjectAbi) {
  const loom_llvmir_target_profile_t* profile =
      loom_llvmir_target_profile_x86_64_object();
  ASSERT_NE(profile, nullptr);
  ASSERT_NE(profile->target_env, nullptr);

  EXPECT_EQ(ToString(profile->name), "x86_64-object");
  EXPECT_EQ(profile->kind, LOOM_LLVMIR_TARGET_PROFILE_HOST_OBJECT);
  EXPECT_EQ(profile->target_env->object_format, LOOM_LLVMIR_OBJECT_FORMAT_ELF);
  EXPECT_EQ(ToString(profile->target_env->target_triple),
            "x86_64-unknown-linux-gnu");
  EXPECT_EQ(ToString(profile->target_env->data_layout),
            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-"
            "f80:128-n8:16:32:64-S128");
  EXPECT_EQ(profile->target_env->address_spaces.generic, 0u);
  EXPECT_EQ(profile->target_env->address_spaces.global, 0u);
  EXPECT_EQ(profile->target_env->address_spaces.buffer_resource, UINT32_MAX);
  EXPECT_EQ(profile->exported_linkage, LOOM_LLVMIR_LINKAGE_DSO_LOCAL);

  loom_llvmir_target_config_t config = {};
  IREE_ASSERT_OK(loom_llvmir_target_profile_module_config(
      profile, IREE_SV("x86-source"), &config));
  EXPECT_EQ(ToString(config.source_name), "x86-source");
  EXPECT_EQ(ToString(config.target_triple), "x86_64-unknown-linux-gnu");
  EXPECT_EQ(ToString(config.data_layout),
            ToString(profile->target_env->data_layout));
  EXPECT_EQ(config.default_pointer_bitwidth, 64u);
  EXPECT_EQ(config.index_bitwidth, 64u);
  EXPECT_EQ(config.offset_bitwidth, 64u);

  loom_llvmir_target_profile_t profile_copy = {};
  IREE_ASSERT_OK(
      loom_llvmir_target_profile_initialize_x86_64_object(&profile_copy));
  EXPECT_EQ(ToString(profile_copy.name), ToString(profile->name));
  EXPECT_EQ(profile_copy.target_env, profile->target_env);
}

TEST(LlvmIrTargetEnvTest, AmdgpuHalProfileNamesKernelAbi) {
  const loom_llvmir_target_profile_t* profile =
      loom_llvmir_target_profile_amdgpu_hal();
  ASSERT_NE(profile, nullptr);
  ASSERT_NE(profile->target_env, nullptr);

  EXPECT_EQ(ToString(profile->name), "amdgpu-hal");
  EXPECT_EQ(profile->kind, LOOM_LLVMIR_TARGET_PROFILE_HAL_KERNEL);
  EXPECT_EQ(ToString(profile->target_env->target_triple), "amdgcn-amd-amdhsa");
  EXPECT_EQ(ToString(profile->target_env->data_layout),
            "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-"
            "p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:"
            "256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-"
            "v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-"
            "S32-A5-G1-ni:7:8:9");
  EXPECT_EQ(profile->target_env->address_spaces.generic, 0u);
  EXPECT_EQ(profile->target_env->address_spaces.global, 1u);
  EXPECT_EQ(profile->target_env->address_spaces.local, 3u);
  EXPECT_EQ(profile->target_env->address_spaces.constant, 4u);
  EXPECT_EQ(profile->target_env->address_spaces.private_memory, 5u);
  EXPECT_EQ(profile->target_env->address_spaces.buffer_resource, 7u);
  EXPECT_EQ(profile->kernel_calling_convention,
            LOOM_LLVMIR_CALLING_CONVENTION_AMDGPU_KERNEL);
  EXPECT_EQ(profile->amdgpu_hal.binding_alignment, 16u);
  EXPECT_EQ(profile->amdgpu_hal.required_workgroup_size.x, 64u);
  EXPECT_EQ(profile->amdgpu_hal.required_workgroup_size.y, 1u);
  EXPECT_EQ(profile->amdgpu_hal.required_workgroup_size.z, 1u);
  EXPECT_EQ(profile->amdgpu_hal.flat_workgroup_size_min, 64u);
  EXPECT_EQ(profile->amdgpu_hal.flat_workgroup_size_max, 64u);
  EXPECT_EQ(profile->amdgpu_hal.buffer_resource_flags, 159744u);

  loom_llvmir_attr_t
      binding_attrs[LOOM_LLVMIR_TARGET_PROFILE_MAX_KERNEL_BINDING_ATTR_COUNT];
  iree_host_size_t binding_attr_count = 0;
  IREE_ASSERT_OK(loom_llvmir_target_profile_kernel_binding_attrs(
      profile, binding_attrs, IREE_ARRAYSIZE(binding_attrs),
      &binding_attr_count));
  EXPECT_EQ(binding_attr_count, 5u);
  EXPECT_EQ(binding_attrs[4].kind, LOOM_LLVMIR_ATTR_ALIGN);
  EXPECT_EQ(binding_attrs[4].value, 16u);
}

TEST(LlvmIrTargetEnvTest, LooksUpBuiltinProfilesByName) {
  const loom_llvmir_target_profile_t* profile = nullptr;
  IREE_ASSERT_OK(loom_llvmir_target_profile_lookup(IREE_SV(""), &profile));
  EXPECT_EQ(profile, loom_llvmir_target_profile_x86_64_object());

  profile = nullptr;
  IREE_ASSERT_OK(
      loom_llvmir_target_profile_lookup(IREE_SV("x86_64-object"), &profile));
  EXPECT_EQ(profile, loom_llvmir_target_profile_x86_64_object());

  profile = nullptr;
  IREE_ASSERT_OK(
      loom_llvmir_target_profile_lookup(IREE_SV("amdgpu-hal"), &profile));
  EXPECT_EQ(profile, loom_llvmir_target_profile_amdgpu_hal());
}

TEST(LlvmIrTargetEnvTest, RejectsUnknownBuiltinProfileName) {
  const loom_llvmir_target_profile_t* profile = nullptr;
  iree_status_t status =
      loom_llvmir_target_profile_lookup(IREE_SV("spirv-vulkan"), &profile);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(profile, nullptr);
  iree_status_ignore(status);
}

TEST(LlvmIrTargetEnvTest, AmdgpuHalProfileMaterializesKernelDecorations) {
  const loom_llvmir_target_profile_t* profile =
      loom_llvmir_target_profile_amdgpu_hal();
  loom_llvmir_target_config_t config = {};
  IREE_ASSERT_OK(loom_llvmir_target_profile_module_config(
      profile, IREE_SV("kernel-source"), &config));

  loom_llvmir_module_t* module = nullptr;
  IREE_ASSERT_OK(
      loom_llvmir_module_allocate(&config, iree_allocator_system(), &module));
  ModulePtr module_ptr(module, loom_llvmir_module_free);

  loom_llvmir_type_id_t void_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_llvmir_module_get_void_type(module_ptr.get(), &void_type));
  loom_llvmir_attr_group_id_t attr_group = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID;
  IREE_ASSERT_OK(loom_llvmir_target_profile_add_kernel_attr_group(
      module_ptr.get(), profile, &attr_group));

  loom_llvmir_function_desc_t function_desc = {};
  function_desc.kind = LOOM_LLVMIR_FUNCTION_DEFINITION;
  function_desc.name = IREE_SV("dispatch");
  function_desc.return_type = void_type;
  function_desc.linkage = profile->exported_linkage;
  function_desc.calling_convention = profile->kernel_calling_convention;
  function_desc.attr_group_id = attr_group;
  loom_llvmir_function_t* function = nullptr;
  IREE_ASSERT_OK(loom_llvmir_module_add_function(module_ptr.get(),
                                                 &function_desc, &function));
  IREE_ASSERT_OK(
      loom_llvmir_target_profile_attach_kernel_metadata(function, profile));

  loom_llvmir_block_t* entry = nullptr;
  IREE_ASSERT_OK(
      loom_llvmir_function_add_block(function, IREE_SV("entry"), &entry));
  IREE_ASSERT_OK(loom_llvmir_build_ret_void(entry));
  IREE_ASSERT_OK(loom_llvmir_verify_module(module_ptr.get()));
}

TEST(LlvmIrTargetEnvTest, AmdgpuHalProfileCopyControlsKernelDecorations) {
  loom_llvmir_target_profile_t profile = {};
  IREE_ASSERT_OK(loom_llvmir_target_profile_initialize_amdgpu_hal(&profile));
  profile.name = IREE_SV("amdgpu-hal-variant");
  profile.target_cpu = IREE_SV("gfx1100");
  profile.target_features = IREE_SV("+wavefrontsize64");
  profile.amdgpu_hal.binding_alignment = 32;
  profile.amdgpu_hal.required_workgroup_size.x = 128;
  profile.amdgpu_hal.required_workgroup_size.y = 2;
  profile.amdgpu_hal.required_workgroup_size.z = 1;
  profile.amdgpu_hal.flat_workgroup_size_min = 128;
  profile.amdgpu_hal.flat_workgroup_size_max = 256;

  loom_llvmir_target_config_t config = {};
  IREE_ASSERT_OK(loom_llvmir_target_profile_module_config(
      &profile, IREE_SV("kernel-variant-source"), &config));
  loom_llvmir_module_t* module = nullptr;
  IREE_ASSERT_OK(
      loom_llvmir_module_allocate(&config, iree_allocator_system(), &module));
  ModulePtr module_ptr(module, loom_llvmir_module_free);

  loom_llvmir_type_id_t void_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_llvmir_module_get_void_type(module_ptr.get(), &void_type));
  loom_llvmir_attr_group_id_t attr_group = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID;
  IREE_ASSERT_OK(loom_llvmir_target_profile_add_kernel_attr_group(
      module_ptr.get(), &profile, &attr_group));

  loom_llvmir_function_desc_t function_desc = {};
  function_desc.kind = LOOM_LLVMIR_FUNCTION_DEFINITION;
  function_desc.name = IREE_SV("dispatch");
  function_desc.return_type = void_type;
  function_desc.linkage = profile.exported_linkage;
  function_desc.calling_convention = profile.kernel_calling_convention;
  function_desc.attr_group_id = attr_group;
  loom_llvmir_function_t* function = nullptr;
  IREE_ASSERT_OK(loom_llvmir_module_add_function(module_ptr.get(),
                                                 &function_desc, &function));

  loom_llvmir_type_id_t global_pointer_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_llvmir_module_get_pointer_type(
      module_ptr.get(), profile.target_env->address_spaces.global,
      &global_pointer_type));
  loom_llvmir_attr_t
      binding_attrs[LOOM_LLVMIR_TARGET_PROFILE_MAX_KERNEL_BINDING_ATTR_COUNT];
  iree_host_size_t binding_attr_count = 0;
  IREE_ASSERT_OK(loom_llvmir_target_profile_kernel_binding_attrs(
      &profile, binding_attrs, IREE_ARRAYSIZE(binding_attrs),
      &binding_attr_count));
  loom_llvmir_value_id_t parameter = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_parameter_desc_t parameter_desc = {};
  parameter_desc.type_id = global_pointer_type;
  parameter_desc.name = IREE_SV("input");
  parameter_desc.attrs = binding_attrs;
  parameter_desc.attr_count = binding_attr_count;
  IREE_ASSERT_OK(loom_llvmir_function_add_parameter(function, &parameter_desc,
                                                    &parameter));
  IREE_ASSERT_OK(
      loom_llvmir_target_profile_attach_kernel_metadata(function, &profile));

  loom_llvmir_block_t* entry = nullptr;
  IREE_ASSERT_OK(
      loom_llvmir_function_add_block(function, IREE_SV("entry"), &entry));
  IREE_ASSERT_OK(loom_llvmir_build_ret_void(entry));
  IREE_ASSERT_OK(loom_llvmir_verify_module(module_ptr.get()));

  std::string text = WriteText(module_ptr.get());
  EXPECT_NE(text.find("ptr addrspace(1) inreg noalias noundef nonnull align "
                      "32 %input"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("\"amdgpu-flat-work-group-size\"=\"128,256\""),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("!0 = !{i32 128, i32 2, i32 1}\n"), std::string::npos)
      << text;
}

TEST(LlvmIrTargetEnvTest, BuildsLlcArgumentsFromTargetProfile) {
  loom_llvmir_target_profile_llc_arguments_t arguments = {};
  const loom_llvmir_target_profile_t* profile =
      loom_llvmir_target_profile_x86_64_object();
  IREE_ASSERT_OK(loom_llvmir_target_profile_llc_arguments(profile, &arguments));
  ASSERT_EQ(arguments.count, 1u);
  EXPECT_EQ(ToString(arguments.values[0]), "-mtriple=x86_64-unknown-linux-gnu");

  IREE_ASSERT_OK(loom_llvmir_target_profile_lookup(
      IREE_SV("x86_64-packed-dot-object"), &profile));
  IREE_ASSERT_OK(loom_llvmir_target_profile_llc_arguments(profile, &arguments));
  ASSERT_EQ(arguments.count, 2u);
  EXPECT_EQ(ToString(arguments.values[0]), "-mtriple=x86_64-unknown-linux-gnu");
  EXPECT_EQ(ToString(arguments.values[1]),
            "-mattr=+avx512bf16,+avx512vl,+avxvnni,+avxvnniint8");
}

TEST(LlvmIrTargetEnvTest, RejectsInvalidAmdgpuHalProfileValues) {
  loom_llvmir_target_profile_t profile = {};
  IREE_ASSERT_OK(loom_llvmir_target_profile_initialize_amdgpu_hal(&profile));
  loom_llvmir_attr_t
      binding_attrs[LOOM_LLVMIR_TARGET_PROFILE_MAX_KERNEL_BINDING_ATTR_COUNT];
  iree_host_size_t binding_attr_count = 0;

  profile.amdgpu_hal.binding_alignment = 0;
  iree_status_t status = loom_llvmir_target_profile_kernel_binding_attrs(
      &profile, binding_attrs, IREE_ARRAYSIZE(binding_attrs),
      &binding_attr_count);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);

  IREE_ASSERT_OK(loom_llvmir_target_profile_initialize_amdgpu_hal(&profile));
  loom_llvmir_target_config_t config = {};
  IREE_ASSERT_OK(loom_llvmir_target_profile_module_config(
      &profile, IREE_SV("bad-kernel-profile"), &config));
  loom_llvmir_module_t* module = nullptr;
  IREE_ASSERT_OK(
      loom_llvmir_module_allocate(&config, iree_allocator_system(), &module));
  ModulePtr module_ptr(module, loom_llvmir_module_free);

  profile.amdgpu_hal.flat_workgroup_size_min = 256;
  profile.amdgpu_hal.flat_workgroup_size_max = 128;
  loom_llvmir_attr_group_id_t attr_group = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID;
  status = loom_llvmir_target_profile_add_kernel_attr_group(
      module_ptr.get(), &profile, &attr_group);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);

  IREE_ASSERT_OK(loom_llvmir_target_profile_initialize_amdgpu_hal(&profile));
  loom_llvmir_type_id_t void_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_llvmir_module_get_void_type(module_ptr.get(), &void_type));
  loom_llvmir_function_desc_t function_desc = {};
  function_desc.kind = LOOM_LLVMIR_FUNCTION_DEFINITION;
  function_desc.name = IREE_SV("dispatch");
  function_desc.return_type = void_type;
  function_desc.linkage = profile.exported_linkage;
  function_desc.calling_convention = profile.kernel_calling_convention;
  function_desc.attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID;
  loom_llvmir_function_t* function = nullptr;
  IREE_ASSERT_OK(loom_llvmir_module_add_function(module_ptr.get(),
                                                 &function_desc, &function));
  profile.amdgpu_hal.required_workgroup_size.x = 0;
  status =
      loom_llvmir_target_profile_attach_kernel_metadata(function, &profile);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

}  // namespace
}  // namespace loom
