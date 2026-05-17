// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/testbench_actual.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/api.h"

namespace loom {
namespace {

iree_vm_instance_t* hal_testbench_actual_test_vm_instance = nullptr;

class HalTestbenchActualTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    IREE_ASSERT_OK(iree_vm_instance_create(
        IREE_VM_TYPE_CAPACITY_DEFAULT, iree_allocator_system(),
        &hal_testbench_actual_test_vm_instance));
  }

  static void TearDownTestSuite() {
    iree_vm_instance_release(hal_testbench_actual_test_vm_instance);
    hal_testbench_actual_test_vm_instance = nullptr;
  }
};

TEST_F(HalTestbenchActualTest, ScalarInputsPackDispatchConstantWords) {
  iree_vm_variant_t inputs[] = {
      iree_vm_make_variant_value(iree_vm_value_make_i32(0x12345678)),
      iree_vm_make_variant_value(
          iree_vm_value_make_i64(static_cast<int64_t>(0x1122334455667788ull))),
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  iree_vm_list_t* bindings = nullptr;

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_variants(
      inputs, IREE_ARRAYSIZE(inputs), &options, iree_allocator_system(),
      &bindings));

  EXPECT_EQ(iree_vm_list_size(bindings), 0u);
  EXPECT_EQ(options.constant_count, 3u);
  EXPECT_EQ(options.constants[0], 0x12345678u);
  EXPECT_EQ(options.constants[1], 0x55667788u);
  EXPECT_EQ(options.constants[2], 0x11223344u);

  iree_vm_list_release(bindings);
}

TEST_F(HalTestbenchActualTest, F64InputsPackDispatchConstantWords) {
  iree_vm_variant_t inputs[] = {
      iree_vm_make_variant_value(iree_vm_value_make_f64(1.0)),
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  iree_vm_list_t* bindings = nullptr;

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_variants(
      inputs, IREE_ARRAYSIZE(inputs), &options, iree_allocator_system(),
      &bindings));

  EXPECT_EQ(iree_vm_list_size(bindings), 0u);
  EXPECT_EQ(options.constant_count, 2u);
  EXPECT_EQ(options.constants[0], 0x00000000u);
  EXPECT_EQ(options.constants[1], 0x3ff00000u);

  iree_vm_list_release(bindings);
}

}  // namespace
}  // namespace loom
