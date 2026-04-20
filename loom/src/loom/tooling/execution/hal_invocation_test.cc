// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal_invocation.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

TEST(HalInvocationTest, RequestInitializeDefaultsToSingleWorkgroup) {
  loom_run_hal_invocation_request_t request = {};
  loom_run_hal_invocation_request_initialize(&request);

  EXPECT_EQ(request.options.entry_point, 0u);
  EXPECT_EQ(request.options.workgroup_count[0], 1u);
  EXPECT_EQ(request.options.workgroup_count[1], 1u);
  EXPECT_EQ(request.options.workgroup_count[2], 1u);
}

TEST(HalInvocationTest, ResultOwnsOutputBuilder) {
  loom_run_hal_invocation_result_t result = {};
  loom_run_hal_invocation_result_initialize(iree_allocator_system(), &result);

  IREE_ASSERT_OK(iree_string_builder_append_cstring(&result.output, "hello"));
  EXPECT_TRUE(iree_string_view_equal(iree_string_builder_view(&result.output),
                                     IREE_SV("hello")));

  loom_run_hal_invocation_result_deinitialize(&result);
}

TEST(HalInvocationTest, RunRejectsTooManyBindingsBeforeDeviceUse) {
  loom_run_hal_runtime_t runtime = {};
  loom_run_hal_executable_t executable = {};
  loom_run_hal_invocation_request_t request = {};
  loom_run_hal_invocation_request_initialize(&request);
  request.runtime = &runtime;
  request.executable = &executable;
  request.bindings.count = LOOM_RUN_HAL_MAX_BINDING_COUNT + 1;

  loom_run_hal_invocation_result_t result = {};
  loom_run_hal_invocation_result_initialize(iree_allocator_system(), &result);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_run_hal_invocation_run(&request, iree_allocator_system(), &result));

  loom_run_hal_invocation_result_deinitialize(&result);
}

TEST(HalInvocationTest, RunRejectsMissingBindingStorageBeforeDeviceUse) {
  loom_run_hal_runtime_t runtime = {};
  loom_run_hal_executable_t executable = {};
  loom_run_hal_invocation_request_t request = {};
  loom_run_hal_invocation_request_initialize(&request);
  request.runtime = &runtime;
  request.executable = &executable;
  request.bindings.count = 1;

  loom_run_hal_invocation_result_t result = {};
  loom_run_hal_invocation_result_initialize(iree_allocator_system(), &result);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_run_hal_invocation_run(&request, iree_allocator_system(), &result));

  loom_run_hal_invocation_result_deinitialize(&result);
}

TEST(HalInvocationTest, RunRejectsExpectedBindingCountMismatchBeforeDeviceUse) {
  loom_run_hal_runtime_t runtime = {};
  loom_run_hal_executable_t executable = {};
  iree_string_view_t bindings[] = {IREE_SV("&4xi32")};
  const char binding_conventions[] = {'r'};
  iree_string_view_t expected_bindings[] = {IREE_SV("4xi32=0"),
                                            IREE_SV("i32=1")};
  const char expected_binding_conventions[] = {'r', 'r'};

  loom_run_hal_invocation_request_t request = {};
  loom_run_hal_invocation_request_initialize(&request);
  request.runtime = &runtime;
  request.executable = &executable;
  request.bindings = (loom_run_hal_binding_specs_t){
      .values = bindings,
      .conventions = binding_conventions,
      .count = IREE_ARRAYSIZE(bindings),
  };
  request.expected_bindings = (loom_run_hal_binding_specs_t){
      .values = expected_bindings,
      .conventions = expected_binding_conventions,
      .count = IREE_ARRAYSIZE(expected_bindings),
  };

  loom_run_hal_invocation_result_t result = {};
  loom_run_hal_invocation_result_initialize(iree_allocator_system(), &result);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_run_hal_invocation_run(&request, iree_allocator_system(), &result));

  loom_run_hal_invocation_result_deinitialize(&result);
}

TEST(HalInvocationTest, RunRequiresInitializedRuntime) {
  loom_run_hal_runtime_t runtime = {};
  loom_run_hal_executable_t executable = {};
  loom_run_hal_invocation_request_t request = {};
  loom_run_hal_invocation_request_initialize(&request);
  request.runtime = &runtime;
  request.executable = &executable;

  loom_run_hal_invocation_result_t result = {};
  loom_run_hal_invocation_result_initialize(iree_allocator_system(), &result);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_run_hal_invocation_run(&request, iree_allocator_system(), &result));

  loom_run_hal_invocation_result_deinitialize(&result);
}

}  // namespace
}  // namespace loom
