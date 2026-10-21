// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>
#include <thread>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amd/gpu/cts/test_device.h"
#include "iree/hal/drivers/amd/gpu/image/hip_images.h"
#include "iree/hal/slab_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {
IREE_ASYNC_FIXED_FRONTIER_TYPE(CompletedFrontier, 2);

using iree::hal::amd::gpu::cts::GpuNativeTest;
using iree::hal::cts::Ref;

struct DispatchPlacement {
  // Number of queues participating in each workload.
  uint32_t queue_count;
  // Require real code uploads even when the native CPU aperture works.
  bool staged;
};

class GpuNativeDispatchTest
    : public GpuNativeTest,
      public ::testing::WithParamInterface<DispatchPlacement> {
 protected:
  void SetUp() override {
    if (GetParam().staged) {
      IREE_ASSERT_OK(memory_gate_.CreateFabric(&fabric_));
      memory_gate_.RequireCodeUpload();
    }
    GpuNativeTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    const auto* files = iree_hal_amd_gpu_hip_images_create();
    const char* selector = context_->target_info.gfx_ip.major == 11
                               ? "gfx11_generic"
                               : "gfx12_generic";
    for (size_t i = 0; i < iree_hal_amd_gpu_hip_images_size(); ++i) {
      if (std::strstr(files[i].name, selector)) {
        artifact_ = iree_make_const_byte_span(files[i].data, files[i].size);
      }
    }
    ASSERT_NE(artifact_.data, nullptr);
  }

  iree_status_t Load(iree_hal_executable_t** executable) {
    const auto* target =
        &iree_hal_device_spec_executables(iree_hal_device_spec(device_))
             ->targets[0];
    iree_hal_executable_load_params_t load;
    iree_hal_executable_load_params_initialize(&load);
    load.executable_data = artifact_;
    return iree_hal_executable_load(iree_hal_queue_family(queues_[0]), target,
                                    &load, executable);
  }

  // Native capability restriction; allocation and execution remain real.
  iree::hal::amd::gpu::cts::NativeMemoryGate memory_gate_;
  // Compiler-produced image matching the physical target.
  iree_const_byte_span_t artifact_ = {};
};

TEST_P(GpuNativeDispatchTest, ExecutableGlobalsPreservePlacementAndVisibility) {
  Create();
  if (HasFatalFailure()) {
    return;
  }
  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(Load(executable.out()));
  iree_hal_executable_global_t token;
  IREE_ASSERT_OK(iree_hal_executable_lookup_global_by_name(
      executable, IREE_SV("image_value"), &token));
  iree_hal_buffer_t* global = nullptr;
  IREE_ASSERT_OK(iree_hal_executable_global_buffer(executable, token, &global));
  iree_hal_executable_function_t function;
  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      executable, IREE_SV("image_increment"), &function));
  Ref<iree_hal_semaphore_t> timeline;
  CreateSemaphore(timeline.out());
  if (HasFatalFailure()) {
    return;
  }
  auto* semaphore = timeline.get();
  auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = config.workgroup_size[2] = 1;
  const auto binding = iree_hal_make_buffer_ref(global, 0, sizeof(uint32_t));
  const uint32_t increment = 9;
  EXPECT_NE(
      iree_hal_buffer_memory_type(global) & IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      0u);
  const bool mapped =
      iree_hal_buffer_memory_type(global) & IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  if (GetParam().staged) {
    EXPECT_FALSE(mapped);
  }
  const uint32_t values[] = {31, 47, 83, 109};
  uint64_t completed = 0;
  for (uint64_t i = 0; i < IREE_ARRAYSIZE(values); ++i) {
    auto* queue = queues_[i % GetParam().queue_count];
    if (mapped) {
      IREE_ASSERT_OK(
          iree_hal_buffer_map_write(global, 0, &values[i], sizeof(values[i])));
    } else {
      const iree_hal_transfer_operation_t update = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE,
          .update =
              {
                  .source_buffer = &values[i],
                  .target_buffer = global,
                  .length = sizeof(values[i]),
              },
      };
      uint64_t ready = completed + 1;
      IREE_ASSERT_OK(iree_hal_queue_transfer(
          queue, {completed ? 1u : 0u, &semaphore, &completed},
          {1, &semaphore, &ready}, 1, &update, nullptr));
      completed = ready;
    }
    uint64_t after = completed + 1;
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        queue, {completed ? 1u : 0u, &semaphore, &completed},
        {1, &semaphore, &after}, executable, function, config,
        iree_make_const_byte_span(&increment, sizeof(increment)), {1, &binding},
        nullptr, IREE_HAL_DISPATCH_FLAG_NONE));
    uint32_t observed = 0;
    if (mapped) {
      IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, after,
                                             iree_infinite_timeout(),
                                             IREE_ASYNC_WAIT_FLAG_NONE));
      IREE_ASSERT_OK(
          iree_hal_buffer_map_read(global, 0, &observed, sizeof(observed)));
      completed = after;
    } else {
      const iree_hal_transfer_operation_t download = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
          .download =
              {
                  .source_buffer = global,
                  .target = &observed,
                  .length = sizeof(observed),
              },
      };
      completed = after + 1;
      IREE_ASSERT_OK(iree_hal_queue_transfer(queue, {1, &semaphore, &after},
                                             {1, &semaphore, &completed}, 1,
                                             &download, nullptr));
      IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, completed,
                                             iree_infinite_timeout(),
                                             IREE_ASYNC_WAIT_FLAG_NONE));
    }
    EXPECT_EQ(observed, values[i] + increment);
  }
}

TEST_P(GpuNativeDispatchTest, DirectDispatchThroughExplicitPoolAndSealedGroup) {
  constexpr uint32_t kThreads = 4;
  constexpr uint32_t kSteps = 256;
  const uint32_t queue_count = GetParam().queue_count;
  options_.pm4.argument_capacity = 8;
  options_.pm4.retirement_capacity = 8;
  Create();
  if (HasFatalFailure()) {
    return;
  }

  const auto* family = iree_hal_queue_family(queues_[0]);
  const iree_hal_pool_family_access_t access = {
      .family = family,
      .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
      .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
  };
  const iree_hal_pool_scope_t scope = {
      .family_count = 1,
      .families = &access,
      .host =
          {
              .access = IREE_HAL_MEMORY_ACCESS_READ,
              .modes = IREE_HAL_MAPPING_MODE_SCOPED,
          },
  };
  iree_hal_slab_pool_options_t pool_options;
  iree_hal_slab_pool_options_initialize(&pool_options);
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, scope, &pool_options,
                                           iree_allocator_system(), &pool));
  iree_hal_buffer_t* output = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(kThreads * kSteps * 4, 4),
      iree_infinite_timeout(), &output));
  iree_hal_memory_scope_t program_scope;
  IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
      group_, {IREE_HAL_MEMORY_SITE_PROGRAM, family}, &program_scope));
  iree_hal_buffer_native_binding_slot_t address_slot;
  IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
      pool, program_scope, IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
      &address_slot));
  const uint64_t address =
      iree_hal_buffer_native_binding(output, address_slot).device_address;

  std::array<iree_hal_executable_t*, kThreads> executables = {};
  std::array<iree_hal_executable_function_t, kThreads> functions = {};
  std::array<iree_hal_semaphore_t*, kThreads> timelines = {};
  for (uint32_t t = 0; t < kThreads; ++t) {
    IREE_ASSERT_OK(Load(&executables[t]));
    IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
        executables[t], IREE_SV("image_increment"), &functions[t]));
    CreateSemaphore(&timelines[t]);
  }
  // An idle caller can discover failure before creating any capture. It still
  // completes the rejected operation through its output timeline and leaves
  // later independent dispatches runnable. Executing this dispatch by mistake
  // would also change image_increment's global and corrupt the checks below.
  iree_hal_semaphore_t* failed_input = nullptr;
  iree_hal_semaphore_t* failed_output = nullptr;
  CreateSemaphore(&failed_input);
  CreateSemaphore(&failed_output);
  iree_hal_semaphore_fail(
      failed_input, iree_make_status(IREE_STATUS_ABORTED, "producer failed"));
  uint64_t failed_value = 1;
  uint32_t increment = 1;
  auto failed_binding = iree_hal_make_buffer_ref(output, 0, sizeof(uint32_t));
  auto failed_config = iree_hal_make_static_dispatch_config(1, 1, 1);
  failed_config.workgroup_size[0] = 32;
  failed_config.workgroup_size[1] = failed_config.workgroup_size[2] = 1;
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      queues_[0], {1, &failed_input, &failed_value},
      {1, &failed_output, &failed_value}, executables[0], functions[0],
      failed_config, iree_make_const_byte_span(&increment, sizeof(increment)),
      {1, &failed_binding}, nullptr, 0));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        iree_hal_semaphore_wait(failed_output, failed_value,
                                                iree_infinite_timeout(),
                                                IREE_ASYNC_WAIT_FLAG_NONE));
  iree_hal_semaphore_release(failed_output);
  iree_hal_semaphore_release(failed_input);
  std::array<std::thread, kThreads> threads;
  for (uint32_t t = 0; t < kThreads; ++t) {
    threads[t] = std::thread([&, t] {
      auto submit = [&](uint32_t i) {
        struct {
          uint64_t output;
          uint32_t value;
        } arguments = {address + (t * kSteps + i) * 4, 1};
        uint64_t before = i, after = i + 1;
        auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
        config.workgroup_size[0] = 32;
        config.workgroup_size[1] = config.workgroup_size[2] = 1;
        auto binding =
            iree_hal_make_buffer_ref(output, (t * kSteps + i) * 4, 4);
        IREE_ASSERT_OK(iree_hal_queue_dispatch(
            queues_[i % queue_count], {i ? 1u : 0u, &timelines[t], &before},
            {1, &timelines[t], &after}, executables[t], functions[t], config,
            t & 1 ? iree_make_const_byte_span(&arguments, 12)
                  : iree_make_const_byte_span(&arguments.value, 4),
            t & 1 ? iree_hal_buffer_ref_list_empty()
                  : iree_hal_buffer_ref_list_t{1, &binding},
            nullptr,
            t & 1 ? IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS : 0));
        std::memset(&arguments, 0xCD, sizeof(arguments));
      };
      // A consumer cannot occupy the native FIFO until its producer has been
      // published, including when both operations target the same queue.
      for (uint32_t i = 0; i < kSteps; i += 2) {
        submit(i + 1);
        submit(i);
      }
      iree_hal_executable_release(executables[t]);
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  // Trim can overlap native execution and retirement without taking storage
  // away from publishers or peer queues retaining binary-signal readers.
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  for (auto* timeline : timelines) {
    IREE_ASSERT_OK(iree_hal_semaphore_wait(
        timeline, kSteps, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    CompletedFrontier frontier;
    const uint8_t count = iree_async_semaphore_query_frontier(
        reinterpret_cast<iree_async_semaphore_t*>(timeline),
        iree_async_fixed_frontier_as_frontier(&frontier), 2);
    ASSERT_EQ(count, queue_count);
    for (uint8_t i = 0; i < count; ++i) {
      EXPECT_TRUE(iree_async_frontier_tracker_query_epoch(
          tracker_, frontier.entries[i].axis, frontier.entries[i].epoch));
    }
    iree_hal_semaphore_release(timeline);
  }
  std::array<uint32_t, kThreads * kSteps> values;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(output, 0, values.data(), sizeof(values)));
  for (uint32_t t = 0; t < kThreads; ++t) {
    for (uint32_t i = 0; i < kSteps; ++i) {
      EXPECT_EQ(values[t * kSteps + i], 20 + i) << t << ":" << i;
    }
  }
  iree_hal_buffer_release(output);
  iree_hal_pool_release(pool);
}

INSTANTIATE_TEST_SUITE_P(QueuePlacement, GpuNativeDispatchTest,
                         ::testing::Values(DispatchPlacement{1, false},
                                           DispatchPlacement{2, false},
                                           DispatchPlacement{1, true}),
                         [](const auto& info) {
                           return std::string(info.param.queue_count == 1
                                                  ? "SameQueue"
                                                  : "CrossQueue") +
                                  (info.param.staged ? "StagedCode" : "");
                         });

}  // namespace
