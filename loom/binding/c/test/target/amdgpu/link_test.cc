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
      .kernel_target_profile = profile.get(),
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

  options.kernel_target_profile = nullptr;
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

INSTANTIATE_TEST_SUITE_P(RepresentationsAndProfiles, AmdgpuLinkProfileTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool()));

}  // namespace
}  // namespace loomc::testing::amdgpu
