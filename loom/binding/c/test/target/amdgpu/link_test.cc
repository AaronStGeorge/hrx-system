// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <tuple>

#include "iree/testing/gtest.h"
#include "target_test_util.h"

namespace loomc::testing::amdgpu {
namespace {

class AmdgpuLinkProfileTest
    : public ::testing::TestWithParam<std::tuple<bool, bool>> {};

TEST_P(AmdgpuLinkProfileTest,
       HomogeneousProfileMatchesExplicitKernelAssignments) {
  const auto [bytecode, wave64] = GetParam();
  TargetEnvironmentPtr environment = CreateAmdgpuTargetEnvironment();
  ContextPtr context = CreateAmdgpuContext(environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  TargetProfilePtr profile =
      CreateTargetProfile(environment.get(), wave64 ? "gfx942" : "gfx1151");
  SourcePtr source = CreateTextSource("kernels.loom", R"(
func.def @helper() {
  func.return
}
kernel.def @first() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  func.call @helper() : () -> ()
  kernel.return
}
kernel.def @second() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  func.call @helper() : () -> ()
  kernel.return
}
)");
  if (bytecode) {
    ModulePtr input =
        DeserializeModule(context.get(), workspace.get(), source.get());
    ASSERT_NE(input, nullptr);
    source = SerializeModule(input.get(), LOOMC_SOURCE_FORMAT_BYTECODE);
  }
  LinkIndexPtr index = CreateLinkIndex(context.get(), source.get());
  ASSERT_NE(index, nullptr);
  LinkerPtr linker = CreateLinker(context.get());
  loomc_link_options_t options = {
      .structure_size = sizeof(options),
      .link_index = index.get(),
      .mode = LOOMC_LINK_MODE_LINK,
      .flags = LOOMC_LINK_FLAG_INCLUDE_INPUT_EXPORTS,
      .target_profile = profile.get(),
  };
  loomc_module_t* raw_module = nullptr;
  loomc_result_t* raw_result = nullptr;
  LOOMC_ASSERT_OK(loomc_link_module(linker.get(), workspace.get(), &options,
                                    &raw_module, &raw_result));
  ModulePtr homogeneous(raw_module);
  ResultPtr result(raw_result);
  ExpectSucceededResult(result.get());
  ASSERT_NE(homogeneous, nullptr);

  const loomc_target_specialization_t assignments[] = {
      {
          .function_symbol = loomc_make_cstring_view("first"),
          .target_profile = profile.get(),
      },
      {
          .function_symbol = loomc_make_cstring_view("second"),
          .target_profile = profile.get(),
      },
  };
  const loomc_target_specialization_options_t explicit_targets = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      .structure_size = sizeof(explicit_targets),
      .specializations = assignments,
      .specialization_count = 2,
  };
  options.next = &explicit_targets;
  // Two target policies cannot compete for the same invocation.
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT,
                         loomc_link_module(linker.get(), workspace.get(),
                                           &options, &raw_module, &raw_result));
  EXPECT_EQ(raw_module, nullptr);
  EXPECT_EQ(raw_result, nullptr);

  options.target_profile = nullptr;
  LOOMC_ASSERT_OK(loomc_link_module(linker.get(), workspace.get(), &options,
                                    &raw_module, &raw_result));
  ModulePtr explicit_module(raw_module);
  result.reset(raw_result);
  ExpectSucceededResult(result.get());
  ASSERT_NE(explicit_module, nullptr);
  const std::string expected = SerializeModuleToText(explicit_module.get());
  EXPECT_EQ(SerializeModuleToText(homogeneous.get()), expected);

  // The linked module owns target IR after every invocation input is released.
  profile.reset();
  index.reset();
  linker.reset();
  source.reset();
  environment.reset();
  SourcePtr linked_bytecode =
      SerializeModule(homogeneous.get(), LOOMC_SOURCE_FORMAT_BYTECODE);
  homogeneous.reset();
  ModulePtr roundtrip =
      DeserializeModule(context.get(), workspace.get(), linked_bytecode.get());
  EXPECT_EQ(SerializeModuleToText(roundtrip.get()), expected);
}

TEST_P(AmdgpuLinkProfileTest,
       ProfileSpecializesOnlyDeclaredTargetsBeforeSelection) {
  const auto [bytecode, wave64] = GetParam();
  TargetEnvironmentPtr environment = CreateAmdgpuTargetEnvironment();
  ContextPtr context = CreateAmdgpuContext(environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  TargetProfilePtr profile =
      CreateTargetProfile(environment.get(), wave64 ? "gfx942" : "gfx1151");
  SourcePtr source = CreateTextSource("root_profile.loom", R"(
target.decl @device

amdgpu.target<gfx11-generic> @rdna
amdgpu.target<gfx9-4-generic> @cdna

template.decl @choose(%value: i32) -> (i32)

template.def<@choose> target(@rdna) @choose_rdna(%value: i32) -> (i32) {
  template.return %value : i32
}

template.def<@choose> target(@cdna) @choose_cdna(%value: i32) -> (i32) {
  template.return %value : i32
}

func.def target(@device) @select(%value: i32) -> (i32) {
  %result = template.apply<@choose>(%value) : (i32) -> (i32)
  func.return %result : i32
}

func.def public target(@device) @dispatch(%value: i32) -> (i32) {
  %result = func.call @select(%value) : (i32) -> (i32)
  func.return %result : i32
}

func.def public target(@rdna) @dispatch_rdna(%value: i32) -> (i32) {
  %result = template.apply<@choose>(%value) : (i32) -> (i32)
  func.return %result : i32
}

func.def public target(@cdna) @dispatch_cdna(%value: i32) -> (i32) {
  %result = template.apply<@choose>(%value) : (i32) -> (i32)
  func.return %result : i32
}
)");
  if (bytecode) {
    ModulePtr input =
        DeserializeModule(context.get(), workspace.get(), source.get());
    source = SerializeModule(input.get(), LOOMC_SOURCE_FORMAT_BYTECODE);
  }
  LinkIndexPtr index = CreateLinkIndex(context.get(), source.get());
  LinkerPtr linker = CreateLinker(context.get());
  loomc_string_view_t root = loomc_make_cstring_view("dispatch");
  loomc_link_options_t options = {
      .structure_size = sizeof(options),
      .link_index = index.get(),
      .module_name = loomc_make_cstring_view("selected_root"),
      .mode = LOOMC_LINK_MODE_LINK,
      .root_symbols = &root,
      .root_symbol_count = 1,
      .target_profile = profile.get(),
  };
  loomc_module_t* raw_module = nullptr;
  loomc_result_t* raw_result = nullptr;
  LOOMC_ASSERT_OK(loomc_link_module(linker.get(), workspace.get(), &options,
                                    &raw_module, &raw_result));
  ModulePtr module(raw_module);
  ResultPtr result(raw_result);
  ExpectSucceededResult(result.get());
  ASSERT_NE(module, nullptr);

  const std::string text = SerializeModuleToText(module.get());
  const char* selected = wave64 ? "@choose_cdna" : "@choose_rdna";
  const char* rejected = wave64 ? "@choose_rdna" : "@choose_cdna";
  EXPECT_NE(text.find(selected), std::string::npos) << text;
  EXPECT_EQ(text.find(rejected), std::string::npos) << text;

  // A concrete root keeps its authored context even when the homogeneous
  // profile describes the opposite target family.
  root = loomc_make_cstring_view(wave64 ? "dispatch_rdna" : "dispatch_cdna");
  raw_module = nullptr;
  raw_result = nullptr;
  LOOMC_ASSERT_OK(loomc_link_module(linker.get(), workspace.get(), &options,
                                    &raw_module, &raw_result));
  module.reset(raw_module);
  result.reset(raw_result);
  ExpectSucceededResult(result.get());
  const std::string concrete_text = SerializeModuleToText(module.get());
  EXPECT_NE(concrete_text.find(rejected), std::string::npos) << concrete_text;
  EXPECT_EQ(concrete_text.find(selected), std::string::npos) << concrete_text;
}

INSTANTIATE_TEST_SUITE_P(RepresentationsAndProfiles, AmdgpuLinkProfileTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool()));

}  // namespace
}  // namespace loomc::testing::amdgpu
