// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/testbench_actual.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

using ::iree::testing::status::StatusIs;
using ::testing::HasSubstr;

class HalTestbenchActualTest : public ::testing::Test {};

static loom_testbench_value_t I32Value(int32_t value) {
  loom_testbench_value_t result = {};
  result.kind = LOOM_TESTBENCH_VALUE_KIND_SCALAR;
  result.scalar.kind = IREE_TOOLING_VALUE_KIND_I32;
  result.scalar.storage.i32 = value;
  return result;
}

static loom_testbench_value_t I64Value(int64_t value) {
  loom_testbench_value_t result = {};
  result.kind = LOOM_TESTBENCH_VALUE_KIND_SCALAR;
  result.scalar.kind = IREE_TOOLING_VALUE_KIND_I64;
  result.scalar.storage.i64 = value;
  return result;
}

static loom_testbench_value_t F64Value(double value) {
  loom_testbench_value_t result = {};
  result.kind = LOOM_TESTBENCH_VALUE_KIND_SCALAR;
  result.scalar.kind = IREE_TOOLING_VALUE_KIND_F64;
  result.scalar.storage.f64 = value;
  return result;
}

static const loom_run_hal_artifact_provider_t kFakeHalArtifactProvider = {
    /*.name=*/IREE_SVL("fake-hal"),
    /*.hal_driver_name=*/IREE_SVL("fake"),
    /*.target_family_name=*/IREE_SVL("fake-target"),
};

TEST_F(HalTestbenchActualTest, RequiresExplicitDeviceWhenHalProviderExists) {
  const loom_run_hal_artifact_provider_t* artifact_providers[] = {
      &kFakeHalArtifactProvider,
  };
  loom_run_hal_artifact_provider_registry_t registry = {};
  loom_run_hal_artifact_provider_registry_initialize_from_entries(
      artifact_providers, IREE_ARRAYSIZE(artifact_providers), &registry);

  loom_run_hal_testbench_context_t context = {};
  loom_run_hal_testbench_context_initialize(&registry, iree_allocator_system(),
                                            &context);

  iree::Status status = iree::internal::ConsumeForTest(
      loom_run_hal_testbench_context_ensure_runtime(&context));
  EXPECT_THAT(status, StatusIs(iree::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.ToString(), HasSubstr("explicit --device= URI"));
  EXPECT_THAT(status.ToString(), HasSubstr("fake-hal"));

  loom_run_hal_testbench_context_deinitialize(&context);
}

TEST_F(HalTestbenchActualTest, ScalarInputsPackDispatchConstantWords) {
  loom_testbench_value_t inputs[] = {
      I32Value(0x12345678),
      I64Value(static_cast<int64_t>(0x1122334455667788ull)),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      loom_type_scalar(LOOM_SCALAR_TYPE_I64),
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  loom_run_hal_binding_list_t bindings = {};

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_values(
      inputs, input_types, IREE_ARRAYSIZE(inputs), &options,
      iree_allocator_system(), &bindings));

  EXPECT_EQ(bindings.count, 0u);
  EXPECT_EQ(options.constant_count, 3u);
  EXPECT_EQ(options.constants[0], 0x12345678u);
  EXPECT_EQ(options.constants[1], 0x55667788u);
  EXPECT_EQ(options.constants[2], 0x11223344u);

  loom_run_hal_binding_list_deinitialize(&bindings);
}

TEST_F(HalTestbenchActualTest, F64InputsPackDispatchConstantWords) {
  loom_testbench_value_t inputs[] = {
      F64Value(1.0),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_F64),
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  loom_run_hal_binding_list_t bindings = {};

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_values(
      inputs, input_types, IREE_ARRAYSIZE(inputs), &options,
      iree_allocator_system(), &bindings));

  EXPECT_EQ(bindings.count, 0u);
  EXPECT_EQ(options.constant_count, 2u);
  EXPECT_EQ(options.constants[0], 0x00000000u);
  EXPECT_EQ(options.constants[1], 0x3ff00000u);

  loom_run_hal_binding_list_deinitialize(&bindings);
}

TEST_F(HalTestbenchActualTest, IndexInputPacksAsOneDispatchConstantWord) {
  loom_testbench_value_t inputs[] = {
      I64Value(3584),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  loom_run_hal_binding_list_t bindings = {};

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_values(
      inputs, input_types, IREE_ARRAYSIZE(inputs), &options,
      iree_allocator_system(), &bindings));

  EXPECT_EQ(bindings.count, 0u);
  EXPECT_EQ(options.constant_count, 1u);
  EXPECT_EQ(options.constants[0], 3584u);

  loom_run_hal_binding_list_deinitialize(&bindings);
}

TEST_F(HalTestbenchActualTest, OffsetInputPacksAsTwoDispatchConstantWords) {
  loom_testbench_value_t inputs[] = {
      I64Value(static_cast<int64_t>(0x1122334455667788ull)),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  loom_run_hal_binding_list_t bindings = {};

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_values(
      inputs, input_types, IREE_ARRAYSIZE(inputs), &options,
      iree_allocator_system(), &bindings));

  EXPECT_EQ(bindings.count, 0u);
  EXPECT_EQ(options.constant_count, 2u);
  EXPECT_EQ(options.constants[0], 0x55667788u);
  EXPECT_EQ(options.constants[1], 0x11223344u);

  loom_run_hal_binding_list_deinitialize(&bindings);
}

}  // namespace
}  // namespace loom
