// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/amdgpu.h"

#include <cstring>
#include <string>

#include "iree/testing/gtest.h"
#include "loomc/context.h"
#include "loomc/emit.h"
#include "loomc/module.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/status.h"
#include "loomc/target.h"
#include "loomc/workspace.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using TargetEnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using TargetProfilePtr =
    HandlePtr<loomc_target_profile_t, loomc_target_profile_release>;
using TargetSelectionPtr =
    HandlePtr<loomc_target_selection_t, loomc_target_selection_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

std::string ToString(loomc_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
}

std::string ToString(loomc_byte_span_t value) {
  return value.data ? std::string(reinterpret_cast<const char*>(value.data),
                                  value.data_length)
                    : std::string();
}

void ExpectSucceededResult(const loomc_result_t* result) {
  ASSERT_NE(result, nullptr);
  if (!loomc_result_succeeded(result) &&
      loomc_result_diagnostic_count(result) != 0) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, 0);
    ASSERT_NE(diagnostic, nullptr);
    ADD_FAILURE() << ToString(diagnostic->message);
  }
  EXPECT_TRUE(loomc_result_succeeded(result));
}

TargetEnvironmentPtr CreateAmdgpuTargetEnvironment() {
  loomc_target_environment_t* target_environment = nullptr;
  loomc_status_t status = loomc_target_environment_create_amdgpu(
      loomc_allocator_system(), &target_environment);
  LOOMC_EXPECT_OK(status);
  return TargetEnvironmentPtr(target_environment);
}

ContextPtr CreateAmdgpuContext(loomc_target_environment_t* target_environment) {
  loomc_context_target_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_environment=*/target_environment,
  };
  loomc_context_options_t context_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      /*.structure_size=*/sizeof(context_options),
      /*.next=*/&target_options,
  };
  loomc_context_t* context = nullptr;
  loomc_status_t status = loomc_context_create(
      &context_options, loomc_allocator_system(), &context);
  LOOMC_EXPECT_OK(status);
  return ContextPtr(context);
}

WorkspacePtr CreateWorkspace() {
  loomc_workspace_t* workspace = nullptr;
  loomc_status_t status =
      loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace);
  LOOMC_EXPECT_OK(status);
  return WorkspacePtr(workspace);
}

SourcePtr CreateTextSource(const char* identifier, const char* contents) {
  loomc_source_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
      /*.identifier=*/loomc_make_cstring_view(identifier),
      /*.contents=*/loomc_make_byte_span(contents, strlen(contents)),
      /*.storage=*/LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* source = nullptr;
  loomc_status_t status =
      loomc_source_create(&options, loomc_allocator_system(), &source);
  LOOMC_EXPECT_OK(status);
  return SourcePtr(source);
}

ModulePtr DeserializeModule(loomc_context_t* context,
                            loomc_workspace_t* workspace,
                            const loomc_source_t* source) {
  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_module_deserialize_from_source(
      context, workspace, source, nullptr, loomc_allocator_system(), &module,
      &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  return ModulePtr(module);
}

ModulePtr CreatePreparedArithmeticModule(loomc_context_t* context,
                                         loomc_workspace_t* workspace) {
  SourcePtr source = CreateTextSource("amdgpu_prepared_arithmetic.loom", R"(
amdgpu.target<gfx1100> @gfx_target

low.kernel.def target(@gfx_target) workgroup_size(64, 1, 1) @loom_kernel() {
  %zero = low.const<amdgpu.v_mov_b32> {imm32 = 0} : reg<amdgpu.vgpr>
  %one = low.const<amdgpu.v_mov_b32> {imm32 = 1} : reg<amdgpu.vgpr>
  %sum = low.op<amdgpu.v_add_u32>(%zero, %one) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  low.return
}
)");
  return DeserializeModule(context, workspace, source.get());
}

TargetSelectionPtr CreateGfx1100Selection(
    loomc_target_environment_t* target_environment) {
  loomc_amdgpu_profile_options_t profile_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(profile_options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("gfx1100-test"),
      /*.processor=*/loomc_make_cstring_view("gfx1100"),
  };
  loomc_target_profile_t* profile = nullptr;
  loomc_status_t status = loomc_target_profile_create_amdgpu(
      target_environment, &profile_options, loomc_allocator_system(), &profile);
  LOOMC_EXPECT_OK(status);
  TargetProfilePtr profile_ptr(profile);

  loomc_target_selection_t* selection = nullptr;
  status = loomc_target_selection_create_from_profile(
      profile_ptr.get(), loomc_allocator_system(), &selection);
  LOOMC_EXPECT_OK(status);
  return TargetSelectionPtr(selection);
}

ResultPtr EmitModule(loomc_target_environment_t* target_environment,
                     loomc_workspace_t* workspace, loomc_module_t* module,
                     loomc_target_selection_t* selection,
                     loomc_amdgpu_runtime_global_flags_t runtime_globals) {
  loomc_target_selection_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_selection=*/selection,
  };
  loomc_amdgpu_emit_options_t amdgpu_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_AMDGPU_EMIT_OPTIONS,
      /*.structure_size=*/sizeof(amdgpu_options),
      /*.next=*/&target_options,
      /*.runtime_globals=*/runtime_globals,
  };
  loomc_emit_options_t emit_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      /*.structure_size=*/sizeof(emit_options),
      /*.next=*/&amdgpu_options,
      /*.artifact_format=*/
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
      /*.identifier=*/loomc_make_cstring_view("loom_kernel.hsaco"),
      /*.artifact_flags=*/LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
  };
  loomc_result_t* result = nullptr;
  loomc_status_t status =
      loomc_emit_module(target_environment, workspace, module, &emit_options,
                        loomc_allocator_system(), &result);
  LOOMC_EXPECT_OK(status);
  return ResultPtr(result);
}

TEST(AmdgpuTargetTest, EmitRuntimeGlobalsFromTargetOptions) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  ContextPtr context = CreateAmdgpuContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module =
      CreatePreparedArithmeticModule(context.get(), workspace.get());
  TargetSelectionPtr selection =
      CreateGfx1100Selection(target_environment.get());

  ResultPtr result = EmitModule(target_environment.get(), workspace.get(),
                                module.get(), selection.get(),
                                LOOMC_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG |
                                    LOOMC_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG);
  ExpectSucceededResult(result.get());

  ASSERT_EQ(loomc_result_artifact_count(result.get()), 1u);
  const loomc_artifact_t* artifact = loomc_result_artifact_at(result.get(), 0);
  ASSERT_NE(artifact, nullptr);
  EXPECT_EQ(artifact->kind, LOOMC_ARTIFACT_KIND_EXECUTABLE);
  EXPECT_EQ(ToString(artifact->format), LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO);

  const std::string hsaco = ToString(artifact->contents);
  EXPECT_NE(hsaco.find("iree_asan_config"), std::string::npos);
  EXPECT_NE(hsaco.find("iree_feedback_config"), std::string::npos);
}

}  // namespace
