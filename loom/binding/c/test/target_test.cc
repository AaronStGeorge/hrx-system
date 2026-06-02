// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target.h"

#include <cstring>
#include <memory>
#include <string>

#include "iree/testing/gtest.h"
#include "loomc/compile.h"
#include "loomc/context.h"
#include "loomc/link.h"
#include "loomc/module.h"
#include "loomc/pass.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/status.h"
#include "loomc/target/spirv/base.h"
#include "loomc/workspace.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using BuilderPtr =
    HandlePtr<loomc_link_index_builder_t, loomc_link_index_builder_release>;
using CompilerPtr = HandlePtr<loomc_compiler_t, loomc_compiler_release>;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using LinkIndexPtr = HandlePtr<loomc_link_index_t, loomc_link_index_release>;
using LinkerPtr = HandlePtr<loomc_linker_t, loomc_linker_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using PassProgramPtr =
    HandlePtr<loomc_pass_program_t, loomc_pass_program_release>;
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

ContextPtr CreateContext() {
  loomc_context_t* context = nullptr;
  loomc_status_t status =
      loomc_context_create(nullptr, loomc_allocator_system(), &context);
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

CompilerPtr CreateCompiler(loomc_context_t* context) {
  loomc_compiler_t* compiler = nullptr;
  loomc_status_t status = loomc_compiler_create(
      context, nullptr, loomc_allocator_system(), &compiler);
  LOOMC_EXPECT_OK(status);
  return CompilerPtr(compiler);
}

LinkerPtr CreateLinker(loomc_context_t* context) {
  loomc_linker_t* linker = nullptr;
  loomc_status_t status =
      loomc_linker_create(context, nullptr, loomc_allocator_system(), &linker);
  LOOMC_EXPECT_OK(status);
  return LinkerPtr(linker);
}

BuilderPtr CreateLinkIndexBuilder(loomc_context_t* context) {
  loomc_link_index_builder_t* builder = nullptr;
  loomc_status_t status = loomc_link_index_builder_create(
      context, nullptr, loomc_allocator_system(), &builder);
  LOOMC_EXPECT_OK(status);
  return BuilderPtr(builder);
}

TargetEnvironmentPtr CreateSpirvTargetEnvironment() {
  loomc_target_environment_t* target_environment = nullptr;
  loomc_status_t status = loomc_target_environment_create_spirv(
      loomc_allocator_system(), &target_environment);
  LOOMC_EXPECT_OK(status);
  return TargetEnvironmentPtr(target_environment);
}

ContextPtr CreateSpirvContext(loomc_target_environment_t* target_environment) {
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

TargetProfilePtr CreateEmptyProfile(
    loomc_target_environment_t* target_environment) {
  loomc_target_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("test-profile"),
  };
  loomc_target_profile_t* profile = nullptr;
  loomc_status_t status = loomc_target_profile_create_empty(
      target_environment, &options, loomc_allocator_system(), &profile);
  LOOMC_EXPECT_OK(status);
  return TargetProfilePtr(profile);
}

TargetSelectionPtr CreateSelectionFromProfile(loomc_target_profile_t* profile) {
  loomc_target_selection_t* selection = nullptr;
  loomc_status_t status = loomc_target_selection_create_from_profile(
      profile, loomc_allocator_system(), &selection);
  LOOMC_EXPECT_OK(status);
  return TargetSelectionPtr(selection);
}

loomc_pass_program_options_t PassOptionsWithSelection(
    loomc_target_selection_options_t* target_options) {
  return loomc_pass_program_options_t{
      /*.type=*/LOOMC_STRUCTURE_TYPE_PASS_PROGRAM_OPTIONS,
      /*.structure_size=*/sizeof(loomc_pass_program_options_t),
      /*.next=*/target_options,
      /*.identifier=*/loomc_make_cstring_view("selected-pass-program"),
  };
}

PassProgramPtr CreateEmptyPassProgramWithSelection(
    loomc_context_t* context, loomc_target_selection_t* target_selection) {
  loomc_target_selection_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_selection=*/target_selection,
  };
  loomc_pass_program_options_t pass_options =
      PassOptionsWithSelection(&target_options);
  loomc_pass_program_t* pass_program = nullptr;
  loomc_status_t status = loomc_pass_program_create_empty(
      context, &pass_options, loomc_allocator_system(), &pass_program);
  LOOMC_EXPECT_OK(status);
  return PassProgramPtr(pass_program);
}

TargetSelectionPtr CreateEmptySelection() {
  loomc_target_selection_t* selection = nullptr;
  loomc_status_t status =
      loomc_target_selection_create_empty(loomc_allocator_system(), &selection);
  LOOMC_EXPECT_OK(status);
  return TargetSelectionPtr(selection);
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

ModulePtr DeserializeModule(loomc_context_t* context,
                            const loomc_source_t* source) {
  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_module_deserialize_from_source(
      context, source, nullptr, loomc_allocator_system(), &module, &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  return ModulePtr(module);
}

ModulePtr CreateIdentityModule(loomc_context_t* context, const char* symbol) {
  std::string contents = "func.def public @";
  contents.append(symbol);
  contents.append(R"((%x: i32) -> (i32) {
  func.return %x : i32
}
)");
  SourcePtr source = CreateTextSource("identity.loom", contents.c_str());
  return DeserializeModule(context, source.get());
}

LinkIndexPtr CreateSingleSourceLinkIndex(loomc_context_t* context) {
  BuilderPtr builder = CreateLinkIndexBuilder(context);
  SourcePtr source = CreateTextSource("link-input.loom", R"(
func.def public @entry(%x: i32) -> (i32) {
  func.return %x : i32
}
)");
  loomc_link_index_source_options_t source_options = {
      /*.provider_name=*/loomc_make_cstring_view("jit-input"),
      /*.role=*/LOOMC_LINK_PROVIDER_ROLE_INPUT,
  };
  loomc_status_t status = loomc_link_index_builder_add_source(
      builder.get(), source.get(), &source_options, nullptr);
  LOOMC_EXPECT_OK(status);

  loomc_link_index_t* link_index = nullptr;
  loomc_result_t* result = nullptr;
  status = loomc_link_index_builder_finish(builder.get(), &link_index, &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  return LinkIndexPtr(link_index);
}

TEST(TargetTest, RetainReleaseProfileAndSelection) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  TargetProfilePtr profile = CreateEmptyProfile(target_environment.get());
  loomc_target_profile_retain(profile.get());
  loomc_target_profile_release(profile.get());

  TargetSelectionPtr selection = CreateSelectionFromProfile(profile.get());
  loomc_target_selection_retain(selection.get());
  loomc_target_selection_release(selection.get());

  TargetSelectionPtr empty_selection = CreateEmptySelection();
  loomc_target_selection_retain(empty_selection.get());
  loomc_target_selection_release(empty_selection.get());
}

TEST(TargetTest, AcceptsExplicitTargetlessSelectionWithoutTargetEnvironment) {
  ContextPtr context = CreateContext();
  TargetSelectionPtr selection = CreateEmptySelection();
  loomc_target_selection_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_selection=*/selection.get(),
  };
  loomc_pass_program_options_t pass_options =
      PassOptionsWithSelection(&target_options);

  loomc_pass_program_t* pass_program = nullptr;
  loomc_status_t status = loomc_pass_program_create_empty(
      context.get(), &pass_options, loomc_allocator_system(), &pass_program);
  LOOMC_EXPECT_OK(status);
  PassProgramPtr pass_program_ptr(pass_program);
  EXPECT_NE(pass_program_ptr.get(), nullptr);
}

TEST(TargetTest, RejectsProfileSelectionWithoutTargetEnvironment) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  TargetProfilePtr profile = CreateEmptyProfile(target_environment.get());
  TargetSelectionPtr selection = CreateSelectionFromProfile(profile.get());
  ContextPtr context = CreateContext();
  loomc_target_selection_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_selection=*/selection.get(),
  };
  loomc_pass_program_options_t pass_options =
      PassOptionsWithSelection(&target_options);

  loomc_pass_program_t* pass_program = nullptr;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_pass_program_create_empty(context.get(), &pass_options,
                                      loomc_allocator_system(), &pass_program));
  EXPECT_EQ(pass_program, nullptr);
}

TEST(TargetTest, AcceptsProfileSelectionWithCompatibleTargetEnvironment) {
  TargetEnvironmentPtr profile_environment = CreateSpirvTargetEnvironment();
  TargetEnvironmentPtr context_environment = CreateSpirvTargetEnvironment();
  TargetProfilePtr profile = CreateEmptyProfile(profile_environment.get());
  TargetSelectionPtr selection = CreateSelectionFromProfile(profile.get());
  ContextPtr context = CreateSpirvContext(context_environment.get());
  loomc_target_selection_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_selection=*/selection.get(),
  };
  loomc_pass_program_options_t pass_options =
      PassOptionsWithSelection(&target_options);

  loomc_pass_program_t* pass_program = nullptr;
  loomc_status_t status = loomc_pass_program_create_empty(
      context.get(), &pass_options, loomc_allocator_system(), &pass_program);
  LOOMC_EXPECT_OK(status);
  PassProgramPtr pass_program_ptr(pass_program);
  EXPECT_NE(pass_program_ptr.get(), nullptr);
}

TEST(TargetTest, RejectsDuplicateTargetSelectionOptions) {
  ContextPtr context = CreateContext();
  TargetSelectionPtr selection = CreateEmptySelection();
  loomc_target_selection_options_t second_target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      /*.structure_size=*/sizeof(second_target_options),
      /*.next=*/nullptr,
      /*.target_selection=*/selection.get(),
  };
  loomc_target_selection_options_t first_target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      /*.structure_size=*/sizeof(first_target_options),
      /*.next=*/&second_target_options,
      /*.target_selection=*/selection.get(),
  };
  loomc_pass_program_options_t pass_options =
      PassOptionsWithSelection(&first_target_options);

  loomc_pass_program_t* pass_program = nullptr;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_pass_program_create_empty(context.get(), &pass_options,
                                      loomc_allocator_system(), &pass_program));
  EXPECT_EQ(pass_program, nullptr);
}

TEST(TargetTest, RejectsTargetSelectionOptionsWithoutSelection) {
  ContextPtr context = CreateContext();
  loomc_target_selection_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_selection=*/nullptr,
  };
  loomc_pass_program_options_t pass_options =
      PassOptionsWithSelection(&target_options);

  loomc_pass_program_t* pass_program = nullptr;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_pass_program_create_empty(context.get(), &pass_options,
                                      loomc_allocator_system(), &pass_program));
  EXPECT_EQ(pass_program, nullptr);
}

TEST(TargetTest, ReusesSelectionAcrossCompileWorkspaces) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  TargetProfilePtr profile = CreateEmptyProfile(target_environment.get());
  TargetSelectionPtr selection = CreateSelectionFromProfile(profile.get());
  ContextPtr context = CreateSpirvContext(target_environment.get());
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program =
      CreateEmptyPassProgramWithSelection(context.get(), selection.get());

  loomc_target_selection_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_selection=*/selection.get(),
  };
  loomc_compile_options_t compile_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(compile_options),
      /*.next=*/&target_options,
      /*.module_name=*/loomc_make_cstring_view("jit_kernel"),
      /*.entry_symbol=*/loomc_make_cstring_view("@entry"),
      /*.artifact_flags=*/0,
      /*.config=*/{},
  };

  for (int i = 0; i < 2; ++i) {
    WorkspacePtr workspace = CreateWorkspace();
    ModulePtr module = CreateIdentityModule(context.get(), "entry");
    loomc_result_t* result = nullptr;
    loomc_status_t status = loomc_compile_module(
        compiler.get(), workspace.get(), pass_program.get(), module.get(),
        &compile_options, loomc_allocator_system(), &result);
    LOOMC_EXPECT_OK(status);
    ResultPtr result_ptr(result);
    ExpectSucceededResult(result_ptr.get());
  }
}

TEST(TargetTest, ReusesSelectionAcrossLinkWorkspaces) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  TargetProfilePtr profile = CreateEmptyProfile(target_environment.get());
  TargetSelectionPtr selection = CreateSelectionFromProfile(profile.get());
  ContextPtr context = CreateSpirvContext(target_environment.get());
  LinkerPtr linker = CreateLinker(context.get());
  LinkIndexPtr link_index = CreateSingleSourceLinkIndex(context.get());

  loomc_target_selection_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_selection=*/selection.get(),
  };
  loomc_link_options_t link_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LINK_OPTIONS,
      /*.structure_size=*/sizeof(link_options),
      /*.next=*/&target_options,
      /*.link_index=*/link_index.get(),
      /*.module_name=*/loomc_make_cstring_view("linked_jit_module"),
      /*.root_symbols=*/nullptr,
      /*.root_symbol_count=*/0,
      /*.flags=*/LOOMC_LINK_FLAG_INCLUDE_EXPORTED_ROOTS,
  };

  for (int i = 0; i < 2; ++i) {
    WorkspacePtr workspace = CreateWorkspace();
    loomc_module_t* module = nullptr;
    loomc_result_t* result = nullptr;
    loomc_status_t status = loomc_link_module(linker.get(), workspace.get(),
                                              &link_options, &module, &result);
    LOOMC_EXPECT_OK(status);
    ModulePtr module_ptr(module);
    ResultPtr result_ptr(result);
    ExpectSucceededResult(result_ptr.get());
    EXPECT_NE(module_ptr.get(), nullptr);
  }
}

}  // namespace
