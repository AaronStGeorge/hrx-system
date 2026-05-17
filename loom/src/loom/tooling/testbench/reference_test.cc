// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/reference.h"

#include <cstring>
#include <vector>

#include "iree/base/internal/math.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/api.h"

namespace loom {
namespace {

template <typename T>
struct BufferContents {
  const T* values;
  iree_host_size_t count;
};

template <typename T>
static iree_status_t FillBufferView(iree_hal_buffer_mapping_t* mapping,
                                    void* user_data) {
  const BufferContents<T>* contents =
      static_cast<const BufferContents<T>*>(user_data);
  const iree_host_size_t byte_count =
      contents->count * sizeof(contents->values[0]);
  memcpy(mapping->contents.data, contents->values, byte_count);
  return iree_ok_status();
}

class ReferenceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                           host_allocator_, &vm_instance_));
    IREE_ASSERT_OK(iree_hal_module_register_all_types(vm_instance_));
    IREE_ASSERT_OK(
        iree_hal_allocator_create_heap(IREE_SV("testbench"), host_allocator_,
                                       host_allocator_, &device_allocator_));
  }

  void TearDown() override {
    iree_hal_allocator_release(device_allocator_);
    iree_vm_instance_release(vm_instance_);
  }

  iree_hal_buffer_params_t BufferParams() {
    return iree_hal_buffer_params_t{
        .usage = IREE_HAL_BUFFER_USAGE_DEFAULT |
                 IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
        .access = IREE_HAL_MEMORY_ACCESS_ALL,
        .type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
    };
  }

  template <typename T>
  iree_vm_variant_t MakeBufferView(std::vector<iree_hal_dim_t> shape,
                                   iree_hal_element_type_t element_type,
                                   const std::vector<T>& values) {
    BufferContents<T> contents = {
        .values = values.data(),
        .count = values.size(),
    };
    iree_hal_buffer_view_t* buffer_view = nullptr;
    IREE_CHECK_OK(iree_hal_buffer_view_generate_buffer(
        /*device=*/nullptr, device_allocator_, shape.size(), shape.data(),
        element_type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, BufferParams(),
        FillBufferView<T>, &contents, &buffer_view));
    iree_vm_ref_t buffer_view_ref = iree_hal_buffer_view_move_ref(buffer_view);
    return iree_vm_make_variant_ref_assign(buffer_view_ref);
  }

  void ExpectF32BufferView(const iree_vm_variant_t& variant,
                           std::vector<float> expected_values) {
    ASSERT_TRUE(iree_vm_variant_is_ref(variant));
    iree_hal_buffer_view_t* buffer_view = nullptr;
    IREE_ASSERT_OK(iree_hal_buffer_view_check_deref(variant.ref, &buffer_view));
    ASSERT_NE(buffer_view, nullptr);
    EXPECT_EQ(iree_hal_buffer_view_shape_rank(buffer_view), 2u);
    EXPECT_EQ(iree_hal_buffer_view_shape_dim(buffer_view, 0), 2);
    EXPECT_EQ(iree_hal_buffer_view_shape_dim(buffer_view, 1), 2);
    EXPECT_EQ(iree_hal_buffer_view_element_type(buffer_view),
              IREE_HAL_ELEMENT_TYPE_FLOAT_32);

    std::vector<float> actual_values(expected_values.size());
    IREE_ASSERT_OK(iree_hal_buffer_map_read(
        iree_hal_buffer_view_buffer(buffer_view), /*source_offset=*/0,
        actual_values.data(), actual_values.size() * sizeof(float)));
    EXPECT_EQ(actual_values, expected_values);
  }

  iree_allocator_t host_allocator_ = iree_allocator_system();
  iree_vm_instance_t* vm_instance_ = nullptr;
  iree_hal_allocator_t* device_allocator_ = nullptr;
};

TEST_F(ReferenceTest, ComputesF16MatmulWithF32Accumulator) {
  std::vector<uint16_t> lhs_values = {
      iree_math_f32_to_f16(1.0f), iree_math_f32_to_f16(2.0f),
      iree_math_f32_to_f16(3.0f), iree_math_f32_to_f16(4.0f),
      iree_math_f32_to_f16(5.0f), iree_math_f32_to_f16(6.0f),
  };
  std::vector<uint16_t> rhs_values = {
      iree_math_f32_to_f16(7.0f),  iree_math_f32_to_f16(8.0f),
      iree_math_f32_to_f16(9.0f),  iree_math_f32_to_f16(10.0f),
      iree_math_f32_to_f16(11.0f), iree_math_f32_to_f16(12.0f),
  };
  std::vector<float> init_values = {0.5f, 1.0f, 1.5f, 2.0f};

  iree_vm_variant_t inputs[3] = {
      MakeBufferView<uint16_t>({2, 3}, IREE_HAL_ELEMENT_TYPE_FLOAT_16,
                               lhs_values),
      MakeBufferView<uint16_t>({3, 2}, IREE_HAL_ELEMENT_TYPE_FLOAT_16,
                               rhs_values),
      MakeBufferView<float>({2, 2}, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                            init_values),
  };

  loom_testbench_reference_matmul_oracle_options_t options = {
      .device_allocator = device_allocator_,
      .result_buffer_params = BufferParams(),
      .host_allocator = host_allocator_,
  };
  loom_testbench_oracle_provider_t provider = {};
  loom_testbench_reference_matmul_oracle_provider_initialize(&options,
                                                             &provider);

  iree_vm_variant_t results[1] = {iree_vm_variant_empty()};
  loom_testbench_invocation_plan_t invocation = {};
  IREE_ASSERT_OK(provider.invoke.fn(provider.invoke.user_data, &invocation,
                                    IREE_ARRAYSIZE(inputs), inputs,
                                    IREE_ARRAYSIZE(results), results));
  ExpectF32BufferView(results[0], {58.5f, 65.0f, 140.5f, 156.0f});

  iree_vm_variant_reset(&results[0]);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(inputs); ++i) {
    iree_vm_variant_reset(&inputs[i]);
  }
}

}  // namespace
}  // namespace loom
