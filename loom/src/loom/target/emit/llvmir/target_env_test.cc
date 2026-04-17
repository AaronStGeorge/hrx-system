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
#include "loom/target/records.h"
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

iree_status_t LookupRegisteredProfile(
    iree_string_view_t profile_name,
    const loom_llvmir_target_profile_t** out_profile) {
  const loom_llvmir_target_profile_provider_t* providers[] = {
      loom_llvmir_x86_target_profile_provider(),
      loom_llvmir_amdgpu_target_profile_provider(),
  };
  loom_llvmir_target_profile_registry_t registry = {};
  registry.default_profile = loom_llvmir_target_profile_x86_64_object();
  registry.providers = providers;
  registry.provider_count = IREE_ARRAYSIZE(providers);
  return loom_llvmir_target_profile_registry_lookup(&registry, profile_name,
                                                    out_profile);
}

void ExpectSnapshotMatchesLlvmEnv(const loom_target_snapshot_t* snapshot,
                                  const loom_llvmir_target_env_t* target_env) {
  ASSERT_NE(snapshot, nullptr);
  ASSERT_NE(target_env, nullptr);
  EXPECT_EQ(snapshot->codegen_format, LOOM_TARGET_CODEGEN_FORMAT_LLVMIR);
  EXPECT_EQ(snapshot->artifact_format, LOOM_TARGET_ARTIFACT_FORMAT_ELF);
  EXPECT_EQ(ToString(snapshot->target_triple),
            ToString(target_env->target_triple));
  EXPECT_EQ(ToString(snapshot->data_layout), ToString(target_env->data_layout));
  EXPECT_EQ(snapshot->default_pointer_bitwidth,
            target_env->default_pointer_bitwidth);
  EXPECT_EQ(snapshot->index_bitwidth, target_env->index_bitwidth);
  EXPECT_EQ(snapshot->offset_bitwidth, target_env->offset_bitwidth);
  EXPECT_EQ(snapshot->memory_spaces.generic,
            target_env->address_spaces.generic);
  EXPECT_EQ(snapshot->memory_spaces.global, target_env->address_spaces.global);
  EXPECT_EQ(snapshot->memory_spaces.workgroup,
            target_env->address_spaces.local);
  EXPECT_EQ(snapshot->memory_spaces.constant,
            target_env->address_spaces.constant);
  EXPECT_EQ(snapshot->memory_spaces.private_memory,
            target_env->address_spaces.private_memory);
  EXPECT_EQ(snapshot->memory_spaces.descriptor,
            target_env->address_spaces.buffer_resource);
}

void ExpectBundleFingerprint(const loom_target_bundle_t* bundle) {
  uint64_t fingerprint = 0;
  IREE_ASSERT_OK(loom_target_bundle_fingerprint(bundle, &fingerprint));
  EXPECT_NE(fingerprint, 0u);
}

void ExpectDerivedProfileMatchesStatic(
    const loom_llvmir_target_profile_t* derived_profile,
    const loom_llvmir_target_profile_t* static_profile) {
  ASSERT_NE(derived_profile, nullptr);
  ASSERT_NE(static_profile, nullptr);
  ASSERT_NE(derived_profile->target_env, nullptr);
  ASSERT_NE(static_profile->target_env, nullptr);
  EXPECT_EQ(ToString(derived_profile->name), ToString(static_profile->name));
  EXPECT_EQ(derived_profile->kind, static_profile->kind);
  EXPECT_EQ(ToString(derived_profile->target_env->target_triple),
            ToString(static_profile->target_env->target_triple));
  EXPECT_EQ(ToString(derived_profile->target_env->data_layout),
            ToString(static_profile->target_env->data_layout));
  EXPECT_EQ(derived_profile->target_env->object_format,
            static_profile->target_env->object_format);
  EXPECT_EQ(derived_profile->target_env->default_pointer_bitwidth,
            static_profile->target_env->default_pointer_bitwidth);
  EXPECT_EQ(derived_profile->target_env->index_bitwidth,
            static_profile->target_env->index_bitwidth);
  EXPECT_EQ(derived_profile->target_env->offset_bitwidth,
            static_profile->target_env->offset_bitwidth);
  EXPECT_EQ(derived_profile->target_env->address_spaces.generic,
            static_profile->target_env->address_spaces.generic);
  EXPECT_EQ(derived_profile->target_env->address_spaces.global,
            static_profile->target_env->address_spaces.global);
  EXPECT_EQ(derived_profile->target_env->address_spaces.local,
            static_profile->target_env->address_spaces.local);
  EXPECT_EQ(derived_profile->target_env->address_spaces.constant,
            static_profile->target_env->address_spaces.constant);
  EXPECT_EQ(derived_profile->target_env->address_spaces.private_memory,
            static_profile->target_env->address_spaces.private_memory);
  EXPECT_EQ(derived_profile->target_env->address_spaces.buffer_resource,
            static_profile->target_env->address_spaces.buffer_resource);
  EXPECT_EQ(ToString(derived_profile->target_cpu),
            ToString(static_profile->target_cpu));
  EXPECT_EQ(ToString(derived_profile->target_features),
            ToString(static_profile->target_features));
  EXPECT_EQ(derived_profile->x86_packed_dot_feature_bits,
            static_profile->x86_packed_dot_feature_bits);
  EXPECT_EQ(derived_profile->exported_linkage,
            static_profile->exported_linkage);
  EXPECT_EQ(derived_profile->kernel_calling_convention,
            static_profile->kernel_calling_convention);
  EXPECT_EQ(ToString(derived_profile->required_workgroup_size_metadata_name),
            ToString(static_profile->required_workgroup_size_metadata_name));
  EXPECT_EQ(derived_profile->amdgpu_hal.binding_alignment,
            static_profile->amdgpu_hal.binding_alignment);
  EXPECT_EQ(derived_profile->amdgpu_hal.required_workgroup_size.x,
            static_profile->amdgpu_hal.required_workgroup_size.x);
  EXPECT_EQ(derived_profile->amdgpu_hal.required_workgroup_size.y,
            static_profile->amdgpu_hal.required_workgroup_size.y);
  EXPECT_EQ(derived_profile->amdgpu_hal.required_workgroup_size.z,
            static_profile->amdgpu_hal.required_workgroup_size.z);
  EXPECT_EQ(derived_profile->amdgpu_hal.flat_workgroup_size_min,
            static_profile->amdgpu_hal.flat_workgroup_size_min);
  EXPECT_EQ(derived_profile->amdgpu_hal.flat_workgroup_size_max,
            static_profile->amdgpu_hal.flat_workgroup_size_max);
  EXPECT_EQ(derived_profile->amdgpu_hal.buffer_resource_flags,
            static_profile->amdgpu_hal.buffer_resource_flags);
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

TEST(LlvmIrTargetEnvTest, X86ObjectProfileHasGenericTargetBundle) {
  const loom_llvmir_target_profile_t* profile =
      loom_llvmir_target_profile_x86_64_object();
  const loom_target_bundle_t* bundle =
      loom_llvmir_target_bundle_x86_64_object();
  ASSERT_NE(bundle, nullptr);
  EXPECT_EQ(ToString(bundle->name), ToString(profile->name));
  ExpectSnapshotMatchesLlvmEnv(bundle->snapshot, profile->target_env);
  EXPECT_TRUE(iree_string_view_is_empty(bundle->snapshot->target_features));
  EXPECT_EQ(bundle->snapshot->memory_spaces.host, 0u);
  ASSERT_NE(bundle->export_plan, nullptr);
  EXPECT_EQ(ToString(bundle->export_plan->name), ToString(profile->name));
  EXPECT_EQ(bundle->export_plan->abi_kind, LOOM_TARGET_ABI_OBJECT_FUNCTION);
  EXPECT_EQ(bundle->export_plan->linkage, LOOM_TARGET_LINKAGE_DSO_LOCAL);
  ASSERT_NE(bundle->config, nullptr);
  EXPECT_TRUE(iree_string_view_is_empty(bundle->config->contract_set_key));
  ExpectBundleFingerprint(bundle);
}

TEST(LlvmIrTargetEnvTest, X86PackedDotProfileHasGenericTargetBundle) {
  const loom_llvmir_target_profile_t* profile =
      loom_llvmir_target_profile_x86_64_packed_dot_object();
  const loom_target_bundle_t* bundle =
      loom_llvmir_target_bundle_x86_64_packed_dot_object();
  ASSERT_NE(bundle, nullptr);
  EXPECT_EQ(ToString(bundle->name), ToString(profile->name));
  ExpectSnapshotMatchesLlvmEnv(bundle->snapshot, profile->target_env);
  EXPECT_EQ(ToString(bundle->snapshot->target_features),
            ToString(profile->target_features));
  ASSERT_NE(bundle->export_plan, nullptr);
  EXPECT_EQ(bundle->export_plan->abi_kind, LOOM_TARGET_ABI_OBJECT_FUNCTION);
  EXPECT_EQ(bundle->export_plan->linkage, LOOM_TARGET_LINKAGE_DSO_LOCAL);
  ASSERT_NE(bundle->config, nullptr);
  EXPECT_EQ(ToString(bundle->config->contract_set_key),
            "x86.packed_dot.avx512bf16-avx512vl-avxvnni-avxvnniint8");

  uint64_t object_fingerprint = 0;
  IREE_ASSERT_OK(loom_target_bundle_fingerprint(
      loom_llvmir_target_bundle_x86_64_object(), &object_fingerprint));
  uint64_t packed_dot_fingerprint = 0;
  IREE_ASSERT_OK(
      loom_target_bundle_fingerprint(bundle, &packed_dot_fingerprint));
  EXPECT_NE(object_fingerprint, packed_dot_fingerprint);
}

TEST(LlvmIrTargetEnvTest, DerivesX86ObjectProfileFromGenericBundle) {
  loom_llvmir_target_profile_storage_t storage = {};
  IREE_ASSERT_OK(loom_llvmir_target_profile_storage_initialize_from_bundle(
      loom_llvmir_target_bundle_x86_64_object(), &storage));
  ExpectDerivedProfileMatchesStatic(&storage.profile,
                                    loom_llvmir_target_profile_x86_64_object());

  loom_llvmir_target_config_t config = {};
  IREE_ASSERT_OK(loom_llvmir_target_profile_module_config(
      &storage.profile, IREE_SV("derived-x86-source"), &config));
  EXPECT_EQ(ToString(config.source_name), "derived-x86-source");
  EXPECT_EQ(ToString(config.target_triple), "x86_64-unknown-linux-gnu");
}

TEST(LlvmIrTargetEnvTest, DerivesX86PackedDotProfileFromGenericBundle) {
  loom_llvmir_target_profile_storage_t storage = {};
  IREE_ASSERT_OK(loom_llvmir_target_profile_storage_initialize_from_bundle(
      loom_llvmir_target_bundle_x86_64_packed_dot_object(), &storage));
  ExpectDerivedProfileMatchesStatic(
      &storage.profile, loom_llvmir_target_profile_x86_64_packed_dot_object());
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

TEST(LlvmIrTargetEnvTest, AmdgpuHalProfileHasGenericTargetBundle) {
  const loom_llvmir_target_profile_t* profile =
      loom_llvmir_target_profile_amdgpu_hal();
  const loom_target_bundle_t* bundle = loom_llvmir_target_bundle_amdgpu_hal();
  ASSERT_NE(bundle, nullptr);
  EXPECT_EQ(ToString(bundle->name), ToString(profile->name));
  ExpectSnapshotMatchesLlvmEnv(bundle->snapshot, profile->target_env);
  EXPECT_EQ(bundle->snapshot->memory_spaces.host, UINT32_MAX);
  ASSERT_NE(bundle->export_plan, nullptr);
  EXPECT_EQ(ToString(bundle->export_plan->name), ToString(profile->name));
  EXPECT_EQ(bundle->export_plan->abi_kind, LOOM_TARGET_ABI_HAL_KERNEL);
  EXPECT_EQ(bundle->export_plan->linkage, LOOM_TARGET_LINKAGE_DEFAULT);
  EXPECT_EQ(bundle->export_plan->hal_kernel.binding_alignment,
            profile->amdgpu_hal.binding_alignment);
  EXPECT_EQ(bundle->export_plan->hal_kernel.required_workgroup_size.x,
            profile->amdgpu_hal.required_workgroup_size.x);
  EXPECT_EQ(bundle->export_plan->hal_kernel.required_workgroup_size.y,
            profile->amdgpu_hal.required_workgroup_size.y);
  EXPECT_EQ(bundle->export_plan->hal_kernel.required_workgroup_size.z,
            profile->amdgpu_hal.required_workgroup_size.z);
  EXPECT_EQ(bundle->export_plan->hal_kernel.flat_workgroup_size_min,
            profile->amdgpu_hal.flat_workgroup_size_min);
  EXPECT_EQ(bundle->export_plan->hal_kernel.flat_workgroup_size_max,
            profile->amdgpu_hal.flat_workgroup_size_max);
  EXPECT_EQ(bundle->export_plan->hal_kernel.buffer_resource_flags,
            profile->amdgpu_hal.buffer_resource_flags);
  ASSERT_NE(bundle->config, nullptr);
  EXPECT_TRUE(iree_string_view_is_empty(bundle->config->contract_set_key));
  ExpectBundleFingerprint(bundle);
}

TEST(LlvmIrTargetEnvTest, DerivesAmdgpuHalProfileFromGenericBundle) {
  loom_llvmir_target_profile_storage_t storage = {};
  IREE_ASSERT_OK(loom_llvmir_target_profile_storage_initialize_from_bundle(
      loom_llvmir_target_bundle_amdgpu_hal(), &storage));
  ExpectDerivedProfileMatchesStatic(&storage.profile,
                                    loom_llvmir_target_profile_amdgpu_hal());
}

TEST(LlvmIrTargetEnvTest, RejectsMalformedDerivedAmdgpuHalProfile) {
  const loom_target_bundle_t* fixture_bundle =
      loom_llvmir_target_bundle_amdgpu_hal();
  loom_target_snapshot_t snapshot = *fixture_bundle->snapshot;
  loom_target_export_plan_t export_plan = *fixture_bundle->export_plan;
  loom_target_config_t config = *fixture_bundle->config;
  export_plan.hal_kernel.required_workgroup_size.x = 0;

  loom_target_bundle_t broken_bundle = {};
  broken_bundle.name = fixture_bundle->name;
  broken_bundle.snapshot = &snapshot;
  broken_bundle.export_plan = &export_plan;
  broken_bundle.config = &config;

  loom_llvmir_target_profile_storage_t storage = {};
  iree_status_t status =
      loom_llvmir_target_profile_storage_initialize_from_bundle(&broken_bundle,
                                                                &storage);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LlvmIrTargetEnvTest, LooksUpRegisteredProfilesByName) {
  const loom_llvmir_target_profile_t* profile = nullptr;
  IREE_ASSERT_OK(LookupRegisteredProfile(IREE_SV(""), &profile));
  EXPECT_EQ(profile, loom_llvmir_target_profile_x86_64_object());

  profile = nullptr;
  IREE_ASSERT_OK(LookupRegisteredProfile(IREE_SV("x86_64-object"), &profile));
  EXPECT_EQ(profile, loom_llvmir_target_profile_x86_64_object());

  profile = nullptr;
  IREE_ASSERT_OK(LookupRegisteredProfile(IREE_SV("amdgpu-hal"), &profile));
  EXPECT_EQ(profile, loom_llvmir_target_profile_amdgpu_hal());
}

TEST(LlvmIrTargetEnvTest, RejectsUnknownRegisteredProfileName) {
  const loom_llvmir_target_profile_t* profile = nullptr;
  iree_status_t status =
      LookupRegisteredProfile(IREE_SV("spirv-vulkan"), &profile);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(profile, nullptr);
}

TEST(LlvmIrTargetEnvTest, RegistryOnlySeesExplicitProviders) {
  const loom_llvmir_target_profile_provider_t* providers[] = {
      loom_llvmir_x86_target_profile_provider(),
  };
  loom_llvmir_target_profile_registry_t registry = {};
  registry.default_profile = loom_llvmir_target_profile_x86_64_object();
  registry.providers = providers;
  registry.provider_count = IREE_ARRAYSIZE(providers);

  const loom_llvmir_target_profile_t* profile = nullptr;
  IREE_ASSERT_OK(loom_llvmir_target_profile_registry_lookup(
      &registry, IREE_SV("x86_64-object"), &profile));
  EXPECT_EQ(profile, loom_llvmir_target_profile_x86_64_object());

  profile = nullptr;
  iree_status_t status = loom_llvmir_target_profile_registry_lookup(
      &registry, IREE_SV("amdgpu-hal"), &profile);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(profile, nullptr);
}

TEST(LlvmIrTargetEnvTest, AmdgpuHalProfileMaterializesKernelDecorations) {
  loom_llvmir_target_profile_storage_t storage = {};
  IREE_ASSERT_OK(loom_llvmir_target_profile_storage_initialize_from_bundle(
      loom_llvmir_target_bundle_amdgpu_hal(), &storage));
  const loom_llvmir_target_profile_t* profile = &storage.profile;
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

  IREE_ASSERT_OK(
      LookupRegisteredProfile(IREE_SV("x86_64-packed-dot-object"), &profile));
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
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);

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
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);

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
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

}  // namespace
}  // namespace loom
