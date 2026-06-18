// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device_spec_builder.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

TEST(DeviceSpecTest, CreatesSpecFromParams) {
  iree_hal_allocator_t* allocator = NULL;
  IREE_ASSERT_OK(
      iree_hal_allocator_create_heap(IREE_SV("test"), iree_allocator_system(),
                                     iree_allocator_system(), &allocator));

  iree_hal_amdgpu_target_id_t target_id = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_target_id_parse(
      IREE_SV("gfx1100"), IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_ALLOW_ARCH_ONLY,
      &target_id));

  iree_hal_amdgpu_device_spec_physical_device_params_t physical_device = {
      /*.target_id=*/target_id,
      /*.uuid=*/{{0x11}},
      /*.pci=*/{/*.domain=*/0, /*.bus=*/3, /*.device=*/0, /*.function=*/0},
      /*.numa=*/{/*.node_id=*/1},
      /*.physical_ordinal=*/7,
      /*.queue_count=*/2,
      /*.compute_unit_count=*/40,
      /*.wavefront_size=*/32,
      /*.maximum_workgroup_local_memory_size=*/64 * 1024,
      /*.flags=*/IREE_HAL_AMDGPU_DEVICE_SPEC_PHYSICAL_DEVICE_FLAG_UUID |
          IREE_HAL_AMDGPU_DEVICE_SPEC_PHYSICAL_DEVICE_FLAG_PCI_ADDRESS,
  };
  iree_hal_amdgpu_device_spec_params_t params = {
      /*.logical_device_id=*/IREE_SV("amdgpu://0"),
      /*.display_name=*/IREE_SV("AMDGPU test device"),
      /*.timestamp_frequency_hz=*/1000000000ull,
      /*.physical_device_count=*/1,
      /*.physical_devices=*/&physical_device,
      /*.device_memory_capacity_bytes=*/64ull * 1024ull * 1024ull * 1024ull,
      /*.device_allocator=*/allocator,
      /*.sanitizer=*/{},
      /*.flags=*/IREE_HAL_AMDGPU_DEVICE_SPEC_PARAM_FLAG_DMABUF,
  };

  iree_hal_device_spec_t* device_spec = NULL;
  IREE_ASSERT_OK(iree_hal_amdgpu_device_spec_create(
      &params, iree_allocator_system(), &device_spec));

  const iree_hal_device_identity_spec_t* identity =
      iree_hal_device_spec_identity(device_spec);
  ASSERT_NE(identity, nullptr);
  EXPECT_TRUE(iree_string_view_equal(identity->driver_id, IREE_SV("amdgpu")));
  EXPECT_TRUE(iree_string_view_equal(identity->backend_id, IREE_SV("hsa")));
  ASSERT_EQ(identity->physical_device_count, 1);
  EXPECT_EQ(identity->physical_devices[0].physical_ordinal, 7);
  EXPECT_TRUE(
      iree_all_bits_set(identity->physical_devices[0].identity.flags,
                        IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_UUID |
                            IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_PCI_ADDRESS |
                            IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_NUMA_NODE));

  const iree_hal_device_queue_spec_t* queues =
      iree_hal_device_spec_queues(device_spec);
  ASSERT_NE(queues, nullptr);
  ASSERT_EQ(queues->family_count, 1);
  EXPECT_EQ(queues->families[0].queue_count, 2);
  EXPECT_EQ(queues->families[0].timestamp_frequency_hz, 1000000000ull);

  const iree_hal_device_dispatch_spec_t* dispatch =
      iree_hal_device_spec_dispatch(device_spec);
  ASSERT_NE(dispatch, nullptr);
  EXPECT_EQ(dispatch->subgroup.default_size, 32);
  EXPECT_EQ(dispatch->subgroup.supported_size_mask, 1ull << 32);
  EXPECT_EQ(dispatch->execution.unit_count, 40);
  EXPECT_EQ(dispatch->execution.maximum_workgroup_local_memory_size, 64 * 1024);
  EXPECT_EQ(dispatch->execution.maximum_workgroup_local_memory_size_optin,
            64 * 1024);

  const iree_hal_device_memory_spec_t* memory =
      iree_hal_device_spec_memory(device_spec);
  ASSERT_NE(memory, nullptr);
  ASSERT_EQ(memory->heap_count, 1);
  EXPECT_EQ(memory->heaps[0].capacity_bytes,
            64ull * 1024ull * 1024ull * 1024ull);
  EXPECT_FALSE(iree_all_bits_set(
      memory->heaps[0].flags, IREE_HAL_MEMORY_HEAP_SPEC_FLAG_CAPACITY_UNKNOWN));

  const iree_hal_device_executable_spec_t* executables =
      iree_hal_device_spec_executables(device_spec);
  ASSERT_NE(executables, nullptr);
  ASSERT_GE(executables->format_count, 1);
  EXPECT_TRUE(iree_string_view_equal(executables->formats[0].format,
                                     IREE_SV("gfx1100")));
  ASSERT_GE(executables->target_count, 1);
  EXPECT_TRUE(iree_string_view_equal(executables->targets[0].family,
                                     IREE_SV("amdgpu")));
  EXPECT_TRUE(iree_string_view_equal(executables->targets[0].processor,
                                     IREE_SV("gfx1100")));
  EXPECT_TRUE(iree_string_view_equal(executables->targets[0].loader_target,
                                     IREE_SV("gfx1100")));

  iree_hal_device_spec_release(device_spec);
  iree_hal_allocator_release(allocator);
}

}  // namespace
}  // namespace iree::hal::amdgpu
