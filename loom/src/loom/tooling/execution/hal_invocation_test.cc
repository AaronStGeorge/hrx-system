// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal_invocation.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/api.h"

namespace loom {
namespace {

iree_vm_instance_t* hal_invocation_test_vm_instance = nullptr;

class HalInvocationTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    IREE_ASSERT_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                           iree_allocator_system(),
                                           &hal_invocation_test_vm_instance));
  }

  static void TearDownTestSuite() {
    iree_vm_instance_release(hal_invocation_test_vm_instance);
    hal_invocation_test_vm_instance = nullptr;
  }
};

TEST_F(HalInvocationTest, RequestInitializeDefaultsToSingleWorkgroup) {
  loom_run_hal_invocation_request_t request = {};
  loom_run_hal_invocation_request_initialize(&request);

  EXPECT_EQ(request.options.entry_point, 0u);
  EXPECT_EQ(request.options.workgroup_count[0], 1u);
  EXPECT_EQ(request.options.workgroup_count[1], 1u);
  EXPECT_EQ(request.options.workgroup_count[2], 1u);
}

TEST_F(HalInvocationTest, ResultOwnsOutputBuilder) {
  loom_run_hal_invocation_result_t result = {};
  loom_run_hal_invocation_result_initialize(iree_allocator_system(), &result);

  IREE_ASSERT_OK(iree_string_builder_append_cstring(&result.output, "hello"));
  EXPECT_TRUE(iree_string_view_equal(iree_string_builder_view(&result.output),
                                     IREE_SV("hello")));

  loom_run_hal_invocation_result_deinitialize(&result);
}

TEST_F(HalInvocationTest, RunRejectsTooManyBindingsBeforeDeviceUse) {
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

TEST_F(HalInvocationTest, RunRejectsMissingBindingStorageBeforeDeviceUse) {
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

TEST_F(HalInvocationTest,
       RunRejectsExpectedBindingCountMismatchBeforeDeviceUse) {
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

TEST_F(HalInvocationTest, RunRequiresInitializedRuntime) {
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

TEST_F(HalInvocationTest,
       RunPlanRejectsExpectedBindingCountMismatchBeforeDeviceUse) {
  loom_run_hal_runtime_t runtime = {};
  loom_run_hal_executable_t executable = {};
  loom_run_hal_invocation_plan_t plan = {};
  loom_run_hal_invocation_plan_initialize(&plan);

  const iree_vm_type_def_t value_type =
      iree_vm_make_value_type_def(IREE_VM_VALUE_TYPE_I32);
  IREE_ASSERT_OK(iree_vm_list_create(value_type, 2, iree_allocator_system(),
                                     &plan.bindings));
  IREE_ASSERT_OK(iree_vm_list_create(value_type, 2, iree_allocator_system(),
                                     &plan.expected_bindings));
  iree_vm_value_t value = iree_vm_value_make_i32(0);
  IREE_ASSERT_OK(iree_vm_list_push_value(plan.bindings, &value));
  IREE_ASSERT_OK(iree_vm_list_push_value(plan.expected_bindings, &value));
  IREE_ASSERT_OK(iree_vm_list_push_value(plan.expected_bindings, &value));

  loom_run_hal_invocation_result_t result = {};
  loom_run_hal_invocation_result_initialize(iree_allocator_system(), &result);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_run_hal_invocation_run_plan(&runtime, &executable, &plan,
                                       iree_allocator_system(), &result));

  loom_run_hal_invocation_result_deinitialize(&result);
  loom_run_hal_invocation_plan_deinitialize(&plan);
}

TEST_F(HalInvocationTest, PreparedCandidatePrepareRequiresInitializedRuntime) {
  loom_run_hal_runtime_t runtime = {};
  loom_run_hal_executable_t executable = {};
  loom_run_hal_prepared_candidate_t candidate = {};

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_run_hal_prepared_candidate_prepare(
                            &runtime, &executable, &candidate));

  loom_run_hal_prepared_candidate_deinitialize(&candidate);
}

TEST_F(HalInvocationTest, DispatchPlanRejectsMissingPreparedExecutable) {
  loom_run_hal_runtime_t runtime = {};
  loom_run_hal_prepared_candidate_t candidate = {};
  loom_run_hal_invocation_plan_t plan = {};
  loom_run_hal_invocation_plan_initialize(&plan);

  const iree_vm_type_def_t value_type =
      iree_vm_make_value_type_def(IREE_VM_VALUE_TYPE_I32);
  IREE_ASSERT_OK(iree_vm_list_create(value_type, 0, iree_allocator_system(),
                                     &plan.bindings));

  loom_run_hal_iteration_t iteration = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_run_hal_invocation_dispatch_plan(
          &runtime, &candidate, &plan, iree_allocator_system(), &iteration));

  loom_run_hal_iteration_deinitialize(&iteration);
  loom_run_hal_invocation_plan_deinitialize(&plan);
}

TEST_F(HalInvocationTest,
       DispatchPlanRejectsTargetLimitViolationBeforeDeviceUse) {
  static const loom_target_snapshot_t snapshot = {
      .name = IREE_SVL("test-snapshot"),
      .max_workgroup_count = {.x = 4, .y = 4, .z = 4},
  };
  static const loom_target_export_plan_t export_plan = {
      .name = IREE_SVL("test-export"),
      .abi_kind = LOOM_TARGET_ABI_HAL_KERNEL,
      .hal_kernel =
          {
              .binding_alignment = 16,
              .required_workgroup_size = {.x = 0, .y = 0, .z = 0},
          },
  };
  static const loom_target_bundle_t target_bundle = {
      .name = IREE_SVL("test-bundle"),
      .snapshot = &snapshot,
      .export_plan = &export_plan,
  };

  loom_run_hal_runtime_t runtime = {};
  loom_run_hal_prepared_candidate_t candidate = {
      .target_bundle = &target_bundle,
      .executable = reinterpret_cast<iree_hal_executable_t*>(1),
  };
  loom_run_hal_invocation_plan_t plan = {};
  loom_run_hal_invocation_plan_initialize(&plan);
  plan.options.workgroup_count[0] = 5;

  const iree_vm_type_def_t value_type =
      iree_vm_make_value_type_def(IREE_VM_VALUE_TYPE_I32);
  IREE_ASSERT_OK(iree_vm_list_create(value_type, 0, iree_allocator_system(),
                                     &plan.bindings));

  loom_run_hal_iteration_t iteration = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_run_hal_invocation_dispatch_plan(
          &runtime, &candidate, &plan, iree_allocator_system(), &iteration));

  candidate.executable = nullptr;
  loom_run_hal_iteration_deinitialize(&iteration);
  loom_run_hal_invocation_plan_deinitialize(&plan);
}

TEST_F(HalInvocationTest, CollectResultsRejectsMissingIterationBindings) {
  loom_run_hal_runtime_t runtime = {};
  loom_run_hal_invocation_plan_t plan = {};
  loom_run_hal_invocation_plan_initialize(&plan);

  const iree_vm_type_def_t value_type =
      iree_vm_make_value_type_def(IREE_VM_VALUE_TYPE_I32);
  IREE_ASSERT_OK(iree_vm_list_create(value_type, 0, iree_allocator_system(),
                                     &plan.bindings));

  loom_run_hal_iteration_t iteration = {};
  loom_run_hal_invocation_result_t result = {};
  loom_run_hal_invocation_result_initialize(iree_allocator_system(), &result);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_run_hal_invocation_collect_results(
          &runtime, &plan, &iteration, iree_allocator_system(), &result));

  loom_run_hal_invocation_result_deinitialize(&result);
  loom_run_hal_invocation_plan_deinitialize(&plan);
}

TEST_F(HalInvocationTest, CollectResultsRejectsIterationPlanBindingMismatch) {
  loom_run_hal_runtime_t runtime = {};
  loom_run_hal_invocation_plan_t plan = {};
  loom_run_hal_invocation_plan_initialize(&plan);
  loom_run_hal_iteration_t iteration = {};
  loom_run_hal_iteration_initialize(&iteration);

  const iree_vm_type_def_t value_type =
      iree_vm_make_value_type_def(IREE_VM_VALUE_TYPE_I32);
  IREE_ASSERT_OK(iree_vm_list_create(value_type, 1, iree_allocator_system(),
                                     &plan.bindings));
  IREE_ASSERT_OK(iree_vm_list_create(value_type, 0, iree_allocator_system(),
                                     &iteration.bindings));
  iree_vm_value_t value = iree_vm_value_make_i32(0);
  IREE_ASSERT_OK(iree_vm_list_push_value(plan.bindings, &value));

  loom_run_hal_invocation_result_t result = {};
  loom_run_hal_invocation_result_initialize(iree_allocator_system(), &result);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_run_hal_invocation_collect_results(
          &runtime, &plan, &iteration, iree_allocator_system(), &result));

  loom_run_hal_invocation_result_deinitialize(&result);
  loom_run_hal_iteration_deinitialize(&iteration);
  loom_run_hal_invocation_plan_deinitialize(&plan);
}

}  // namespace
}  // namespace loom
