// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/device_spec_builder.h"

#include "iree/hal/local/loaders/static_library_loader.h"
#include "iree/hal/memory/cpu_slab_provider.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal {
namespace {

TEST(LocalDeviceSpecBuilderTest, CapturesCommonLocalFacts) {
  iree_hal_executable_loader_t* loader = NULL;
  IREE_ASSERT_OK(iree_hal_static_library_loader_create(
      /*library_count=*/0, /*library_query_fns=*/NULL,
      iree_hal_executable_import_provider_null(), iree_allocator_system(),
      &loader));

  iree_hal_local_device_spec_params_t params = {
      /*.logical_device_id=*/IREE_SV("local0"),
      /*.display_name=*/IREE_SV("Local Device"),
      /*.driver_id=*/IREE_SV("local-test"),
      /*.backend_id=*/IREE_SV("local"),
      /*.queue_count=*/2,
      /*.default_queue_worker_count=*/8,
      /*.loader_count=*/1,
      /*.loaders=*/&loader,
  };
  iree_hal_device_spec_t* spec = NULL;
  IREE_ASSERT_OK(iree_hal_local_device_spec_create(
      &params, iree_allocator_system(), &spec));

  const iree_hal_device_identity_spec_t* identity =
      iree_hal_device_spec_identity(spec);
  ASSERT_NE(identity, nullptr);
  EXPECT_TRUE(
      iree_string_view_equal(identity->logical_device_id, IREE_SV("local0")));
  ASSERT_EQ(identity->physical_device_count, 1);

  const iree_hal_device_memory_spec_t* memory =
      iree_hal_device_spec_memory(spec);
  ASSERT_NE(memory, nullptr);
  ASSERT_EQ(memory->heap_count, 1);
  EXPECT_TRUE(iree_all_bits_set(
      memory->heaps[0].flags,
      IREE_HAL_MEMORY_HEAP_SPEC_FLAG_CAPACITY_UNKNOWN |
          IREE_HAL_MEMORY_HEAP_SPEC_FLAG_MAXIMUM_ALLOCATION_SIZE_UNKNOWN));
  ASSERT_EQ(memory->memory_type_count, 1);
  EXPECT_EQ(memory->memory_types[0].memory_type,
            IREE_HAL_CPU_SLAB_PROVIDER_MEMORY_TYPE);
  EXPECT_EQ(memory->memory_types[0].allowed_buffer_usage,
            IREE_HAL_CPU_SLAB_PROVIDER_BUFFER_USAGE);

  const iree_hal_device_queue_spec_t* queues =
      iree_hal_device_spec_queues(spec);
  ASSERT_NE(queues, nullptr);
  ASSERT_EQ(queues->family_count, 1);
  EXPECT_EQ(queues->families[0].queue_count, 2);

  const iree_hal_device_dispatch_spec_t* dispatch =
      iree_hal_device_spec_dispatch(spec);
  ASSERT_NE(dispatch, nullptr);
  EXPECT_EQ(dispatch->execution.unit_count, 8);
  EXPECT_EQ(dispatch->subgroup.default_size, 1);

  const iree_hal_device_executable_spec_t* executables =
      iree_hal_device_spec_executables(spec);
  ASSERT_NE(executables, nullptr);
  ASSERT_EQ(executables->format_count, 1);
  EXPECT_TRUE(iree_string_view_equal(executables->formats[0].format,
                                     IREE_SV("static")));

  iree_hal_device_spec_release(spec);
  iree_hal_executable_loader_release(loader);
}

}  // namespace
}  // namespace iree::hal
