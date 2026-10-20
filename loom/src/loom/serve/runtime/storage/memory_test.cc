// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/storage/memory.h"

#include <array>
#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/memory/tlsf_pool.h"
#include "iree/hal/testing/deferred_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/execution/execution.h"
#include "loom/serve/runtime/storage/virtual_pool.h"
#include "loom/serve/testing/device.h"
#include "loom/serve/testing/native_allocator.h"

namespace {

constexpr uint64_t kMemoryTestSlabSize = 2 * 1024 * 1024;

class MemoryTestEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    loom_serve_host_t* host = nullptr;
    iree_hal_driver_t* driver = nullptr;
    auto status =
        loom_serve_host_create(1, nullptr, iree_allocator_system(), &host);
    if (iree_status_is_ok(status)) {
      status = loom_serve_test_driver_acquire(host, &driver);
    }
    if (iree_status_is_ok(status)) {
      const loom_serve_device_options_t options = {
          .driver = driver,
          .uri = loom_serve_test_device()->device_uri,
          .backing = LOOM_SERVE_DEVICE_BACKING_ELASTIC,
          .weight_slab_size = kMemoryTestSlabSize,
          .state_slab_size = kMemoryTestSlabSize / 4,
          .memory_limit = 3 * kMemoryTestSlabSize,
      };
      status = loom_serve_device_group_allocate(host, 1, &options,
                                                native.allocator(), &group);
    }
    iree_hal_driver_release(driver);
    loom_serve_host_release(host);
    IREE_ASSERT_OK(status);
    IREE_ASSERT_OK(native.Observe(loom_serve_device_group_at(group, 0)));
  }

  void TearDown() override {
    IREE_EXPECT_OK(loom_serve_device_group_free(group));
    group = nullptr;
    native.AwaitReleased();
  }

  // Allocator dependency retained through all asynchronous device cleanup.
  loom_serve::testing::NativeAllocator native;
  // One native device shared by all cases; pools and reservations are per case.
  loom_serve_device_group_t* group = nullptr;
};

auto* const memory_test_environment = static_cast<MemoryTestEnvironment*>(
    ::testing::AddGlobalTestEnvironment(new MemoryTestEnvironment));

// Exercises mapping rejection after real queue allocation has been accepted.
// The storage owner uses only allocation, deallocation, and mapping operations.
class MappingQueue : public iree_hal_queue_t {
 public:
  ~MappingQueue() { Reset(); }

  void Initialize(iree_hal_queue_t* queue) {
    queue_ = queue;
    iree_hal_queue_retain(queue_);
    const iree_hal_queue_params_t params = {
        .priority = queue->priority,
        .features = queue->features,
        .execution_resources = queue->execution_resources,
    };
    static const iree_hal_queue_vtable_t vtable = {
        .destroy =
            [](iree_hal_queue_t* base) {
              auto* self = static_cast<MappingQueue*>(base);
              iree_hal_queue_release(self->queue_);
              self->queue_ = nullptr;
            },
        .alloca =
            [](iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
               iree_hal_semaphore_list_t signals, iree_hal_pool_t* pool,
               iree_host_size_t count,
               const iree_hal_pool_reservation_request_t* requests,
               iree_hal_buffer_t** buffers) {
              return iree_hal_queue_alloca(
                  static_cast<MappingQueue*>(base)->queue_, waits, signals,
                  pool, count, requests, buffers);
            },
        .dealloca =
            [](iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
               iree_hal_semaphore_list_t signals, iree_host_size_t count,
               iree_hal_buffer_t* const* buffers) {
              return iree_hal_queue_dealloca(
                  static_cast<MappingQueue*>(base)->queue_, waits, signals,
                  count, buffers);
            },
        .map =
            [](iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
               iree_hal_semaphore_list_t signals, iree_host_size_t count,
               const iree_hal_memory_mapping_t* mappings,
               iree_hal_queue_map_flags_t flags) {
              auto* self = static_cast<MappingQueue*>(base);
              if (self->fail_next_map) {
                self->fail_next_map = false;
                return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
              }
              return iree_hal_queue_map(self->queue_, waits, signals, count,
                                        mappings, flags);
            },
    };
    iree_hal_queue_initialize(iree_hal_queue_family(queue), &params, &vtable,
                              this);
  }

  void Reset() {
    if (queue_) {
      iree_hal_queue_release(this);
    }
  }

  // Rejects one mapping submission without changing its signal semaphores.
  bool fail_next_map = false;

 private:
  // Retained native queue used for all accepted operations.
  iree_hal_queue_t* queue_ = nullptr;
};

class MemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    group = memory_test_environment->group;
    device = loom_serve_device_group_at(group, 0);
    ASSERT_NO_FATAL_FAILURE(SetUpPool());
    IREE_ASSERT_OK(loom_serve_device_create_execution(device, &execution));
  }

  virtual void SetUpPool() { UseBacking(false, 3 * kSlabSize, nullptr); }

  void TearDown() override {
    if (deferred_backing) {
      deferred_backing->ResumeAcquisitions();
      deferred_backing->CompleteReturns();
    }
    iree_hal_buffer_release(workspace);
    if (virtual_pool) {
      const auto frontier = loom_serve_execution_frontier(execution);
      for (size_t i = 0; i < IREE_ARRAYSIZE(frontier.semaphores); ++i) {
        iree_status_ignore(loom_serve_virtual_pool_wait(
            virtual_pool, frontier.semaphores[i], frontier.values[i]));
      }
      AwaitVirtualPool();
    }
    if (execution) {
      IREE_ASSERT_OK(loom_serve_execution_drain(execution));
    }
    if (logical_pool) {
      // Raw queue captures can retire after timeline readiness.
      while (iree_atomic_ref_count_load(&logical_pool->resource.ref_count) >
             1) {
        loom_serve_workspace_statistics(loom_serve_device_workspace(device));
        std::this_thread::yield();
      }
      iree_hal_pool_release(logical_pool);
    }
    IREE_ASSERT_OK(loom_serve_virtual_pool_free(virtual_pool));
    IREE_ASSERT_OK(loom_serve_virtual_buffer_free(first));
    IREE_ASSERT_OK(loom_serve_virtual_buffer_free(second));
    IREE_ASSERT_OK(loom_serve_virtual_buffer_free(third));
    if (pool) {
      loom_serve_memory_pool_trim(pool, 0);
      // Intentional failure cases retain their original status while native
      // disposal still reaches its physical capacity boundary.
      iree_status_ignore(loom_serve_memory_pool_await(pool));
      const auto statistics = loom_serve_memory_pool_statistics(pool);
      EXPECT_EQ(statistics.committed_bytes, 0u);
      EXPECT_EQ(statistics.reserved_bytes, 0u);
    }
    loom_serve_memory_pool_free(owned_pool);
    iree_hal_pool_release(deferred_backing);
    mapping_queue.Reset();
    loom_serve_execution_release(execution);
    IREE_EXPECT_OK(loom_serve_test_device_trim_and_drain(device, 0));
  }

  void UseDeferredBacking(uint64_t limit_bytes = 3 * kSlabSize,
                          iree_hal_queue_t* queue = nullptr) {
    UseBacking(true, limit_bytes, queue);
  }

  void UseBacking(bool deferred, uint64_t limit_bytes,
                  iree_hal_queue_t* queue) {
    // Selection happens before this case allocates any reservations.
    loom_serve_memory_pool_free(owned_pool);
    owned_pool = nullptr;
    pool = nullptr;
    const auto host_allocator = native.allocator();
    const auto* dispatch =
        iree_hal_queue_family(loom_serve_device_dispatch_queue(device));
    if (!queue) {
      queue = loom_serve_device_transfer_queue(device);
    }
    const auto* transfer = iree_hal_queue_family(queue);
    iree_hal_pool_family_access_t families[] = {
        {
            .family = dispatch,
            .usage =
                IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
            .interfaces = UINT64_C(1)
                          << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE,
        },
        {
            .family = transfer,
            .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
            .interfaces = UINT64_C(1)
                          << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE,
        },
    };
    const iree_hal_pool_scope_t scope = {
        .family_count = dispatch == transfer ? 1u : 2u,
        .families = families,
    };
    auto* hal_group = loom_serve_device_group(device);
    iree_hal_pool_t* backing = nullptr;
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    IREE_ASSERT_OK(iree_hal_slab_pool_create(hal_group, scope, &options,
                                             host_allocator, &backing));
    if (deferred) {
      auto status = iree::hal::testing::DeferredBackingPool::Create(
          backing, &deferred_backing);
      iree_hal_pool_release(backing);
      IREE_ASSERT_OK(status);
      backing = deferred_backing;
      iree_hal_pool_retain(backing);
    }
    for (auto& family : families) {
      family.interfaces = 0;
    }
    iree_hal_virtual_buffer_capabilities_t capabilities;
    auto status = iree_hal_virtual_buffer_query_capabilities(
        hal_group, scope, host_allocator, &capabilities);
    loom_serve_memory_geometry_t geometries[2] = {};
    if (iree_status_is_ok(status)) {
      status = loom_serve_memory_pool_query_geometry(&capabilities, backing,
                                                     kSlabSize, &geometries[0]);
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_memory_pool_query_geometry(
          &capabilities, backing, deferred ? kSlabSize : kSlabSize / 4,
          &geometries[1]);
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_memory_pool_allocate(hal_group, scope, backing, queue,
                                               geometries, limit_bytes,
                                               host_allocator, &owned_pool);
    }
    iree_hal_pool_release(backing);
    IREE_ASSERT_OK(status);
    pool = owned_pool;
  }

  void RoundTrip(iree_hal_buffer_t* buffer, uint64_t offset, uint32_t value) {
    std::array<uint32_t, 64> observed = {};
    const iree_hal_transfer_operation_t operations[] = {
        {
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
            .fill =
                {
                    .target_buffer = buffer,
                    .target_offset = offset,
                    .length = sizeof(observed),
                    .pattern = &value,
                    .pattern_length = sizeof(value),
                },
        },
        {
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
            .download =
                {
                    .source_buffer = buffer,
                    .source_offset = offset,
                    .target = observed.data(),
                    .length = sizeof(observed),
                },
        },
    };
    uint64_t completion = 0;
    IREE_ASSERT_OK(
        loom_serve_execution_transfer(execution, 1, operations, &completion));
    IREE_ASSERT_OK(loom_serve_execution_transfer(execution, 1, operations + 1,
                                                 &completion));
    if (virtual_pool) {
      const auto frontier = loom_serve_execution_frontier(execution);
      IREE_ASSERT_OK(loom_serve_virtual_pool_wait(
          virtual_pool, frontier.semaphores[0], completion));
    } else {
      IREE_ASSERT_OK(loom_serve_execution_wait(execution, completion));
    }
    for (const auto word : observed) {
      EXPECT_EQ(word, value);
    }
  }

  void CreateVirtualPool() {
    IREE_ASSERT_OK(loom_serve_virtual_pool_allocate(
        pool, loom_serve_device_weight_geometry(device),
        iree_allocator_system(), &virtual_pool));
    const iree_hal_tlsf_pool_options_t options = {
        .tlsf_options = {.range_length = kSlabSize, .alignment = 256},
    };
    IREE_ASSERT_OK(iree_hal_tlsf_pool_create(
        loom_serve_virtual_pool_handle(virtual_pool), &options,
        iree_allocator_system(), &logical_pool));
  }

  void AwaitVirtualPool() {
    iree_notification_t notification;
    iree_notification_initialize(&notification);
    loom_serve_virtual_pool_set_observer(
        virtual_pool, {[](void* user_data) {
                         iree_notification_post(
                             static_cast<iree_notification_t*>(user_data),
                             IREE_ALL_WAITERS);
                       },
                       &notification});
    iree_status_t status = iree_ok_status();
    bool pending = true;
    while (pending && iree_status_is_ok(status)) {
      const auto token = iree_notification_prepare_wait(&notification);
      status = loom_serve_virtual_pool_advance(virtual_pool);
      pending = loom_serve_virtual_pool_is_pending(virtual_pool);
      if (pending && iree_status_is_ok(status)) {
        iree_notification_commit_wait(&notification, token, IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      } else {
        iree_notification_cancel_wait(&notification);
      }
    }
    loom_serve_virtual_pool_set_observer(virtual_pool, {});
    iree_notification_deinitialize(&notification);
    IREE_EXPECT_OK(status);
  }

  void CheckRetained(iree_hal_buffer_t* buffer, uint64_t offset,
                     uint32_t value) {
    std::array<uint32_t, 64> observed = {};
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download =
            {
                .source_buffer = buffer,
                .source_offset = offset,
                .target = observed.data(),
                .length = sizeof(observed),
            },
    };
    uint64_t completion = 0;
    IREE_ASSERT_OK(
        loom_serve_execution_transfer(execution, 1, &download, &completion));
    IREE_ASSERT_OK(loom_serve_execution_wait(execution, completion));
    for (const auto word : observed) {
      EXPECT_EQ(word, value);
    }
  }

  // Process-owned allocator dependency shared by the native device and pools.
  loom_serve::testing::NativeAllocator& native =
      memory_test_environment->native;
  // Optional mapping dependency outliving the private pool's native cleanup.
  MappingQueue mapping_queue;
  // Per-case physical owner, isolating statistics and terminal pool failures.
  loom_serve_memory_pool_t* owned_pool = nullptr;
  // Optional adapter retaining real parent tokens beyond logical completion.
  iree::hal::testing::DeferredBackingPool* deferred_backing = nullptr;
  // Coarse physical slabs, independent of logical KV block size.
  static constexpr uint64_t kSlabSize = kMemoryTestSlabSize;
  // Offset deliberately crossing the 32-bit byte-address boundary.
  static constexpr uint64_t kHighOffset = (uint64_t{1} << 32) + 512;
  // Borrowed process-owned group enclosing all native device resources.
  loom_serve_device_group_t* group = nullptr;
  // Borrowed device record within the application group.
  loom_serve_device_t* device = nullptr;
  // Owned model timeline for transfers and workspace allocations.
  loom_serve_execution_t* execution = nullptr;
  // One physical budget shared by the independent reservations.
  loom_serve_memory_pool_t* pool = nullptr;
  // Independent accounting groups competing for the same physical budget.
  std::array<loom_serve_memory_statistics_t, 3> statistics = {};
  // First model-like reservation, sparsely backed beyond four GiB.
  loom_serve_virtual_buffer_t* first = nullptr;
  // Independent reservation competing for the same physical budget.
  loom_serve_virtual_buffer_t* second = nullptr;
  // Concurrent third model sharing physical capacity with the second.
  loom_serve_virtual_buffer_t* third = nullptr;
  // Private command allocation retained until its queue release is submitted.
  iree_hal_buffer_t* workspace = nullptr;
  // Queue-facing source whose cold work advances on this test's owner.
  loom_serve_virtual_pool_t* virtual_pool = nullptr;
  // Persistent logical allocator above virtual_pool.
  iree_hal_pool_t* logical_pool = nullptr;
};

// These cases hold a native allocation worker while other device work proceeds.
// Task devices share that worker with execution, so their composition excludes
// this fixture instead of blocking the sole CPU worker needed for progress.
using DedicatedNativeWorkerMemoryTest = MemoryTest;

// Device-wide trim and group ownership use the real device-owned pool. Other
// cases isolate their statistics and intentional failures in private pools.
class DeviceMemoryTest : public MemoryTest {
 protected:
  void SetUpPool() override { pool = loom_serve_device_memory_pool(device); }
};

TEST_F(MemoryTest, GeometryRespectsFiniteBackingCapacity) {
  iree_hal_pool_family_access_t family = {
      .family = iree_hal_queue_family(loom_serve_device_transfer_queue(device)),
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
      .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE,
  };
  const iree_hal_pool_scope_t scope = {.family_count = 1, .families = &family};
  iree_hal_pool_t* source = nullptr;
  iree_hal_slab_pool_options_t source_options;
  iree_hal_slab_pool_options_initialize(&source_options);
  IREE_ASSERT_OK(iree_hal_slab_pool_create(loom_serve_device_group(device),
                                           scope, &source_options,
                                           iree_allocator_system(), &source));
  family.interfaces = 0;
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      loom_serve_device_group(device), scope, iree_allocator_system(),
      &capabilities));
  loom_serve_memory_geometry_t geometry;
  IREE_ASSERT_OK(loom_serve_memory_pool_query_geometry(&capabilities, source,
                                                       kSlabSize, &geometry));
  const iree_hal_fixed_block_pool_options_t options = {
      .block_size = geometry.slab_size,
      .blocks_per_slab = 1,
      .alignment = geometry.granularity,
  };
  iree_hal_pool_t* bounded_source = nullptr;
  IREE_ASSERT_OK(iree_hal_fixed_block_pool_create(
      source, &options, iree_allocator_system(), &bounded_source));
  iree_hal_pool_release(source);
  IREE_ASSERT_OK(loom_serve_memory_pool_query_geometry(
      &capabilities, bounded_source, kSlabSize, &geometry));
  EXPECT_EQ(geometry.slab_size, kSlabSize);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_serve_memory_pool_query_geometry(&capabilities, bounded_source,
                                            2 * kSlabSize, &geometry));
  EXPECT_EQ(geometry.slab_size, 0u);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_memory_pool_query_geometry(&capabilities, bounded_source,
                                            kSlabSize - 1, &geometry));
  IREE_ASSERT_OK(loom_serve_memory_pool_query_geometry(
      &capabilities, bounded_source, 0, &geometry));
  EXPECT_GT(geometry.slab_size, 0u);
  EXPECT_LE(geometry.slab_size, kSlabSize);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(bounded_source, &stats);
  EXPECT_EQ(stats.bytes_committed, 0u);
  iree_hal_pool_release(bounded_source);
}

TEST_F(MemoryTest, DetachedSlabsServeTwoModelsThenReturnToOriginalAddresses) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), 3 * kSlabSize, 256,
      &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), kSlabSize, 256,
      &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), 2 * kSlabSize, 256,
      &statistics[2], &third));
  auto* const original = loom_serve_virtual_buffer_handle(first);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 3 * kSlabSize));
  RoundTrip(original, 2 * kSlabSize, 0x12345678);
  // Warm swaps must not ask the native provider for another slab. Keep a
  // failure armed through both swaps, then consume it after an explicit trim.
  native.FailNextAllocation(IREE_ALLOCATOR_COMMAND_CALLOC,
                            /*native_only=*/true);
  loom_serve_virtual_buffer_begin_trim(first);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  auto backing = loom_serve_memory_pool_statistics(pool);
  EXPECT_EQ(statistics[0].committed_bytes, 0u);
  EXPECT_EQ(backing.committed_bytes, 3 * kSlabSize);
  EXPECT_EQ(backing.cached_bytes, 3 * kSlabSize);

  const loom_serve_memory_range_t ranges[] = {{second, 0, kSlabSize},
                                              {third, 0, 2 * kSlabSize}};
  bool admitted = false;
  IREE_ASSERT_OK(loom_serve_memory_pool_try_commit(pool, IREE_ARRAYSIZE(ranges),
                                                   ranges, &admitted));
  ASSERT_TRUE(admitted);
  RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x87654321);
  RoundTrip(loom_serve_virtual_buffer_handle(third), kSlabSize, 0xabcdef01);
  CheckRetained(loom_serve_virtual_buffer_handle(second), 0, 0x87654321);
  backing = loom_serve_memory_pool_statistics(pool);
  EXPECT_EQ(backing.committed_bytes, 3 * kSlabSize);
  EXPECT_EQ(backing.cached_bytes, 0u);

  loom_serve_virtual_buffer_begin_trim(second);
  loom_serve_virtual_buffer_begin_trim(third);
  loom_serve_virtual_buffer_t* detached[] = {second, third};
  loom_serve_memory_operation_t* operation = nullptr;
  IREE_ASSERT_OK(loom_serve_memory_pool_trim_async(
      pool, IREE_ARRAYSIZE(detached), detached, nullptr, 0, &operation));
  loom_serve_memory_operation_await(operation);
  IREE_ASSERT_OK(loom_serve_memory_operation_query_status(operation));
  loom_serve_memory_operation_free(operation);
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 3 * kSlabSize));
  EXPECT_EQ(loom_serve_virtual_buffer_handle(first), original);
  RoundTrip(original, 2 * kSlabSize, 0x10203040);
  backing = loom_serve_memory_pool_statistics(pool);
  EXPECT_EQ(backing.committed_bytes, 3 * kSlabSize);
  EXPECT_EQ(backing.cached_bytes, 0u);

  // Explicit physical trim cannot destroy the active model's mappings.
  loom_serve_memory_pool_trim(pool, 0);
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  CheckRetained(original, 2 * kSlabSize, 0x10203040);
  loom_serve_virtual_buffer_begin_trim(first);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
  loom_serve_memory_pool_trim(pool, 0);
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_virtual_buffer_commit(first, 0, kSlabSize));
}

TEST_F(MemoryTest, VirtualCommandPoolReusesRetainedSlabsAcrossTrimAndRegrowth) {
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking());
  const auto geometry = loom_serve_device_weight_geometry(device);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, geometry, 2 * kSlabSize, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, geometry, 2 * kSlabSize, 256, &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, geometry, kSlabSize, 256, &statistics[2], &third));
  auto* const scratch_address = loom_serve_virtual_buffer_handle(second);

  // Logical command allocation retains the virtual range, not its physical
  // slabs. The same TLSF instance survives every physical detach below.
  const iree_hal_tlsf_pool_options_t options = {
      .tlsf_options = {.alignment = 256},
  };
  iree_hal_pool_t* logical_handle = nullptr;
  IREE_ASSERT_OK(iree_hal_tlsf_pool_create_from_buffer(
      scratch_address, 0, IREE_HAL_WHOLE_BUFFER, &options,
      iree_allocator_system(), &logical_handle));
  const std::unique_ptr<iree_hal_pool_t, decltype(&iree_hal_pool_release)>
      logical(logical_handle, iree_hal_pool_release);

  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 2 * kSlabSize));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(third, 0, kSlabSize));
  ASSERT_NO_FATAL_FAILURE(
      RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x12345678));
  ASSERT_NO_FATAL_FAILURE(
      RoundTrip(loom_serve_virtual_buffer_handle(third), 0, 0x87654321));
  iree_hal_pool_stats_t baseline;
  iree_hal_pool_query_stats(deferred_backing, &baseline);

  // Live retained data occupies the entire physical budget. Preparing scratch
  // cannot take its backing or impair the independently usable warm range.
  const loom_serve_memory_range_t scratch = {second, 0, 2 * kSlabSize};
  bool admitted = true;
  IREE_ASSERT_OK(
      loom_serve_memory_pool_try_commit(pool, 1, &scratch, &admitted));
  ASSERT_FALSE(admitted);
  ASSERT_NO_FATAL_FAILURE(
      CheckRetained(loom_serve_virtual_buffer_handle(third), 0, 0x87654321));
  loom_serve_virtual_buffer_begin_trim(first);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));

  // Repeat actual queue allocation, device writes, downloads and deallocation
  // with differently sized command workspaces. Source slabs are prepared once;
  // neither warm execution nor cross-role reuse acquires another native slab.
  auto execute = [&] {
    const iree_hal_pool_reservation_request_t requests[] = {
        iree_hal_make_pool_reservation_request(3 * kSlabSize / 2, 256),
        iree_hal_make_pool_reservation_request(kSlabSize / 4, 256),
    };
    iree_hal_buffer_t* buffers[IREE_ARRAYSIZE(requests)] = {};
    IREE_ASSERT_OK(loom_serve_execution_alloca_from_pool(
        execution, logical.get(), IREE_ARRAYSIZE(requests), requests, buffers));
    std::unique_ptr<iree_hal_buffer_t, decltype(&iree_hal_buffer_release)>
        first_buffer(buffers[0], iree_hal_buffer_release);
    std::unique_ptr<iree_hal_buffer_t, decltype(&iree_hal_buffer_release)>
        second_buffer(buffers[1], iree_hal_buffer_release);
    ASSERT_NO_FATAL_FAILURE(RoundTrip(buffers[0], kSlabSize - 128, 0xabcdef01));
    ASSERT_NO_FATAL_FAILURE(RoundTrip(buffers[1], 0, 0x10203040));
    uint64_t completion = 0;
    for (auto* buffer : buffers) {
      IREE_ASSERT_OK(
          loom_serve_execution_dealloca(execution, buffer, &completion));
    }
    first_buffer.reset();
    second_buffer.reset();
    IREE_ASSERT_OK(loom_serve_execution_drain(execution));
    // Collect retired queue roots after joining their device uses.
    loom_serve_workspace_statistics(loom_serve_device_workspace(device));
  };
  for (int cycle = 0; cycle < 2; ++cycle) {
    IREE_ASSERT_OK(
        loom_serve_memory_pool_try_commit(pool, 1, &scratch, &admitted));
    ASSERT_TRUE(admitted);
    EXPECT_EQ(loom_serve_virtual_buffer_handle(second), scratch_address);
    ASSERT_NO_FATAL_FAILURE(execute());
    ASSERT_NO_FATAL_FAILURE(execute());
    auto memory = loom_serve_memory_pool_statistics(pool);
    EXPECT_EQ(memory.committed_bytes, 3 * kSlabSize);
    EXPECT_EQ(memory.peak_bytes, 3 * kSlabSize);

    loom_serve_virtual_buffer_begin_trim(second);
    IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(second));
    IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
    IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 2 * kSlabSize));
    ASSERT_NO_FATAL_FAILURE(RoundTrip(loom_serve_virtual_buffer_handle(first),
                                      kSlabSize, 0x11223344));
    ASSERT_NO_FATAL_FAILURE(
        CheckRetained(loom_serve_virtual_buffer_handle(third), 0, 0x87654321));
    loom_serve_virtual_buffer_begin_trim(first);
    IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
    IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
    iree_hal_pool_stats_t reused;
    iree_hal_pool_query_stats(deferred_backing, &reused);
    EXPECT_EQ(reused.reserve_count, baseline.reserve_count);
  }

  // A true physical trim does not invalidate virtual addresses or TLSF reuse
  // history. Regrowth acquires only the two slabs the next invocation needs.
  loom_serve_virtual_buffer_begin_trim(third);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(third));
  loom_serve_memory_pool_trim(pool, 0);
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes, 0u);
  IREE_ASSERT_OK(
      loom_serve_memory_pool_try_commit(pool, 1, &scratch, &admitted));
  ASSERT_TRUE(admitted);
  EXPECT_EQ(loom_serve_virtual_buffer_handle(second), scratch_address);
  ASSERT_NO_FATAL_FAILURE(execute());
  iree_hal_pool_stats_t regrown;
  iree_hal_pool_query_stats(deferred_backing, &regrown);
  EXPECT_EQ(regrown.reserve_count, baseline.reserve_count + 2);
}

TEST_F(MemoryTest, QueueDrivenVirtualBackingReusesRetainedSlabs) {
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking());
  const auto geometry = loom_serve_device_weight_geometry(device);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, geometry, 2 * kSlabSize, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 2 * kSlabSize));
  ASSERT_NO_FATAL_FAILURE(
      RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x13579bdf));
  loom_serve_virtual_buffer_begin_trim(first);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
  iree_hal_pool_stats_t baseline;
  iree_hal_pool_query_stats(deferred_backing, &baseline);
  ASSERT_NO_FATAL_FAILURE(CreateVirtualPool());
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_serve_virtual_pool_free(virtual_pool));
  const iree_hal_buffer_backing_facts_t* address_owner = nullptr;
  auto execute = [&] {
    // A dedicated logical extent still obtains two ordinary physical slabs.
    const auto request =
        iree_hal_make_pool_reservation_request(3 * kSlabSize / 2, 256);
    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(loom_serve_execution_alloca_from_pool(
        execution, logical_pool, 1, &request, &buffer));
    std::unique_ptr<iree_hal_buffer_t, decltype(&iree_hal_buffer_release)>
        allocation(buffer, iree_hal_buffer_release);
    ASSERT_NO_FATAL_FAILURE(RoundTrip(buffer, kSlabSize - 128, 0xabcdef01));
    const auto* backing = iree_hal_buffer_memory_view(buffer).backing;
    if (!address_owner) {
      address_owner = backing;
    }
    EXPECT_EQ(backing, address_owner);
    uint64_t completion = 0;
    IREE_ASSERT_OK(
        loom_serve_execution_dealloca(execution, buffer, &completion));
    allocation.reset();
    const auto frontier = loom_serve_execution_frontier(execution);
    IREE_ASSERT_OK(loom_serve_virtual_pool_wait(
        virtual_pool, frontier.semaphores[0], completion));
    IREE_ASSERT_OK(loom_serve_execution_drain(execution));
    while (true) {
      loom_serve_workspace_statistics(loom_serve_device_workspace(device));
      iree_hal_pool_stats_t logical;
      iree_hal_pool_query_stats(logical_pool, &logical);
      if (!logical.reservation_count) {
        break;
      }
      std::this_thread::yield();
    }
  };
  ASSERT_NO_FATAL_FAILURE(execute());
  ASSERT_NO_FATAL_FAILURE(execute());
  iree_hal_pool_stats_t reused;
  iree_hal_pool_query_stats(deferred_backing, &reused);
  EXPECT_EQ(reused.reserve_count, baseline.reserve_count);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            2 * kSlabSize);

  iree_hal_pool_trim(loom_serve_virtual_pool_handle(virtual_pool),
                     IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
  ASSERT_NO_FATAL_FAILURE(AwaitVirtualPool());
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 2 * kSlabSize));
  ASSERT_NO_FATAL_FAILURE(
      RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x2468ace0));
  iree_hal_pool_query_stats(deferred_backing, &reused);
  EXPECT_EQ(reused.reserve_count, baseline.reserve_count);
  loom_serve_virtual_buffer_begin_trim(first);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
  loom_serve_memory_pool_trim(pool, 0);
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes, 0u);
  ASSERT_NO_FATAL_FAILURE(execute());
  iree_hal_pool_query_stats(deferred_backing, &reused);
  EXPECT_EQ(reused.reserve_count, baseline.reserve_count + 2);
}

TEST_F(MemoryTest, VirtualBackingBatchDenialPreservesWarmPrefix) {
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking());
  ASSERT_NO_FATAL_FAILURE(CreateVirtualPool());
  auto* source = loom_serve_virtual_pool_handle(virtual_pool);
  struct Completion {
    // Number of accepted callbacks, including inline warm completion.
    size_t count = 0;
    // Capacity outcome from the most recent completion.
    iree_hal_pool_acquire_result_t result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
    // Owned failure, consumed by the test after completion.
    iree_status_t status = iree_ok_status();
  } completion;
  const iree_hal_pool_acquire_callback_t callback = {
      .fn =
          [](void* user_data, iree_status_t status,
             iree_hal_pool_acquire_result_t result) {
            auto* completion = static_cast<Completion*>(user_data);
            ++completion->count;
            completion->result = result;
            completion->status = status;
          },
      .user_data = &completion,
  };
  const iree_hal_pool_reservation_request_t requests[] = {
      iree_hal_make_pool_reservation_request(2 * kSlabSize, 256),
      iree_hal_make_pool_reservation_request(2 * kSlabSize, 256),
  };
  iree_hal_pool_reservation_t reservations[2] = {};
  iree_hal_pool_acquire_info_t infos[2] = {};
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      source, 1, requests, nullptr, reservations, infos, callback,
      IREE_HAL_POOL_RESERVE_FLAG_NONE));
  ASSERT_NO_FATAL_FAILURE(AwaitVirtualPool());
  IREE_ASSERT_OK(completion.status);
  const auto original = reservations[0];
  iree_hal_pool_reservation_view_t view;
  ASSERT_TRUE(
      iree_hal_pool_query_reservation_views(source, 1, reservations, &view));
  ASSERT_NO_FATAL_FAILURE(RoundTrip(view.buffer, 0, 0x12345678));
  iree_hal_pool_release_reservations(source, 1, reservations,
                                     infos[0].reuse_frontier);

  // The warm first range must roll back when the complete four-slab request
  // cannot fit the three-slab budget. Caller output slots stay untouched.
  reservations[0].block_handle = 17;
  reservations[1].block_handle = 29;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      source, 2, requests, nullptr, reservations, infos, callback,
      IREE_HAL_POOL_RESERVE_FLAG_NONE));
  ASSERT_NO_FATAL_FAILURE(AwaitVirtualPool());
  IREE_ASSERT_OK(completion.status);
  EXPECT_EQ(completion.result, IREE_HAL_POOL_ACQUIRE_OVER_BUDGET);
  EXPECT_EQ(reservations[0].block_handle, 17u);
  EXPECT_EQ(reservations[1].block_handle, 29u);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            2 * kSlabSize);

  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      source, 1, requests, nullptr, reservations, infos, callback,
      IREE_HAL_POOL_RESERVE_FLAG_NONE));
  // No owner pass is needed for a warm hit after rollback.
  EXPECT_EQ(completion.count, 3u);
  IREE_ASSERT_OK(completion.status);
  EXPECT_EQ(reservations[0].block_handle, original.block_handle);
  ASSERT_TRUE(
      iree_hal_pool_query_reservation_views(source, 1, reservations, &view));
  ASSERT_NO_FATAL_FAILURE(CheckRetained(view.buffer, 0, 0x12345678));
  iree_hal_buffer_t* owned = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_materialize_reservations(
      source, 1, requests, reservations,
      IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP, &owned));
  iree_hal_buffer_release(owned);
  iree_hal_pool_stats_t statistics;
  iree_hal_pool_query_stats(source, &statistics);
  EXPECT_EQ(statistics.reservation_count, 0u);
}

TEST_F(MemoryTest,
       VirtualBackingMappingFailureCompletesWithoutPublishingViews) {
  mapping_queue.Initialize(loom_serve_device_transfer_queue(device));
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking(3 * kSlabSize, &mapping_queue));
  ASSERT_NO_FATAL_FAILURE(CreateVirtualPool());
  mapping_queue.fail_next_map = true;
  struct Completion {
    // Accepted transaction has delivered its one terminal callback.
    bool complete = false;
    // Owned failure delivered independently of physical cleanup.
    iree_status_t status = iree_ok_status();
  } completion;
  const iree_hal_pool_acquire_callback_t callback = {
      .fn =
          [](void* user_data, iree_status_t status,
             iree_hal_pool_acquire_result_t result) {
            auto* completion = static_cast<Completion*>(user_data);
            completion->complete = true;
            completion->status = status;
          },
      .user_data = &completion,
  };
  const auto request = iree_hal_make_pool_reservation_request(kSlabSize, 256);
  iree_hal_pool_reservation_t reservation = {.block_handle = 17};
  iree_hal_pool_acquire_info_t info = {};
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      loom_serve_virtual_pool_handle(virtual_pool), 1, &request, nullptr,
      &reservation, &info, callback, IREE_HAL_POOL_RESERVE_FLAG_NONE));
  ASSERT_NO_FATAL_FAILURE(AwaitVirtualPool());
  ASSERT_TRUE(completion.complete);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, completion.status);
  EXPECT_EQ(reservation.block_handle, 17u);
  // The failed range still owns its physical storage. Checked outer cleanup,
  // not callback failure, is what permits reuse.
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes, kSlabSize);
}

TEST_F(MemoryTest, VirtualBackingTrimWaitsForActualHistoryCompletion) {
  ASSERT_NO_FATAL_FAILURE(CreateVirtualPool());
  auto* source = loom_serve_virtual_pool_handle(virtual_pool);
  struct Completion {
    // Accepted acquisitions deliver their terminal result on the owner.
    size_t count = 0;
    // Owned status consumed immediately after each completed acquisition.
    iree_status_t status = iree_ok_status();
  } completion;
  const iree_hal_pool_acquire_callback_t callback = {
      .fn =
          [](void* user_data, iree_status_t status,
             iree_hal_pool_acquire_result_t result) {
            auto* completion = static_cast<Completion*>(user_data);
            ++completion->count;
            completion->status = status;
          },
      .user_data = &completion,
  };
  const auto request = iree_hal_make_pool_reservation_request(kSlabSize, 256);
  iree_hal_pool_reservation_t reservation = {};
  iree_hal_pool_acquire_info_t info = {};
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      source, 1, &request, nullptr, &reservation, &info, callback,
      IREE_HAL_POOL_RESERVE_FLAG_NONE));
  ASSERT_NO_FATAL_FAILURE(AwaitVirtualPool());
  IREE_ASSERT_OK(completion.status);
  iree_hal_pool_reservation_view_t view;
  ASSERT_TRUE(
      iree_hal_pool_query_reservation_views(source, 1, &reservation, &view));
  ASSERT_NO_FATAL_FAILURE(RoundTrip(view.buffer, 0, 0x23571113));

  // An unpublished immutable join supplies a real pending completion point
  // without registering a colliding native queue axis or creating a thread.
  iree_async_frontier_timeline_t* timeline = nullptr;
  IREE_ASSERT_OK(iree_async_frontier_tracker_acquire_timeline(
      source->frontier_tracker, &timeline));
  std::unique_ptr<iree_async_frontier_timeline_t,
                  decltype(&iree_async_frontier_timeline_release)>
      timeline_owner(timeline, iree_async_frontier_timeline_release);
  iree_async_frontier_join_t* join = nullptr;
  IREE_ASSERT_OK(iree_async_frontier_join_allocate(timeline, &join));
  const auto point = iree_async_frontier_join_begin(join);
  std::unique_ptr<iree_async_frontier_join_t,
                  decltype(&iree_async_frontier_join_publish)>
      completion_gate(join, iree_async_frontier_join_publish);
  IREE_ASYNC_FIXED_FRONTIER_TYPE(Frontier, 1);
  const Frontier history = {.entry_count = 1, .entries = {point}};
  const auto* frontier = iree_async_fixed_frontier_as_const_frontier(&history);
  iree_hal_pool_release_reservations(source, 1, &reservation, frontier);
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      source, 1, &request, frontier, &reservation, &info, callback,
      IREE_HAL_POOL_RESERVE_FLAG_NONE));
  EXPECT_EQ(completion.count, 2u);
  IREE_ASSERT_OK(completion.status);
  EXPECT_EQ(info.result, IREE_HAL_POOL_ACQUIRE_OK);
  // Dominance licenses ordered logical reuse, not physical detach.
  iree_hal_pool_release_reservations(source, 1, &reservation,
                                     info.reuse_frontier);
  iree_hal_pool_trim(source, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
  IREE_ASSERT_OK(loom_serve_virtual_pool_advance(virtual_pool));
  EXPECT_TRUE(loom_serve_virtual_pool_is_pending(virtual_pool));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes, kSlabSize);
  ASSERT_NO_FATAL_FAILURE(CheckRetained(view.buffer, 0, 0x23571113));

  completion_gate.reset();
  timeline_owner.reset();
  ASSERT_NO_FATAL_FAILURE(AwaitVirtualPool());
  iree_hal_pool_stats_t detached;
  iree_hal_pool_query_stats(source, &detached);
  EXPECT_EQ(detached.bytes_committed, 0u);
  // The common cache still owns the physical slab, available to any role.
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).cached_bytes, kSlabSize);
}

TEST_F(MemoryTest, WeightTailReusesStateClassAcrossTwoCompleteSwaps) {
  const auto bulk = loom_serve_device_weight_geometry(device);
  const auto state = loom_serve_device_state_geometry(device);
  const uint64_t bytes = 2 * bulk.slab_size + state.slab_size;
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(pool, bulk, bytes, 256,
                                                    &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, state, state.slab_size, 256, &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, bulk, 2 * bulk.slab_size, 256, &statistics[2], &third));
  auto* const identity = loom_serve_virtual_buffer_handle(first);
  const auto missing = loom_serve_virtual_buffer_missing_demand(first);
  EXPECT_EQ(missing.cached_bytes[0], 2 * bulk.slab_size);
  EXPECT_EQ(missing.cached_bytes[1], state.slab_size);
  EXPECT_EQ(missing.uncached_bytes, 0u);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, bytes));
  native.FailNextAllocation(IREE_ALLOCATOR_COMMAND_CALLOC, true);
  for (uint32_t cycle = 0; cycle < 2; ++cycle) {
    RoundTrip(identity, 2 * bulk.slab_size, 100 + cycle);
    loom_serve_virtual_buffer_begin_trim(first);
    IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
    const loom_serve_memory_range_t ranges[] = {{second, 0, state.slab_size},
                                                {third, 0, 2 * bulk.slab_size}};
    bool admitted = false;
    IREE_ASSERT_OK(loom_serve_memory_pool_try_commit(
        pool, IREE_ARRAYSIZE(ranges), ranges, &admitted));
    ASSERT_TRUE(admitted);
    RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 200 + cycle);
    RoundTrip(loom_serve_virtual_buffer_handle(third), bulk.slab_size,
              300 + cycle);
    loom_serve_virtual_buffer_begin_trim(second);
    loom_serve_virtual_buffer_begin_trim(third);
    IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(second));
    IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(third));
    IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, bytes));
    EXPECT_EQ(loom_serve_virtual_buffer_handle(first), identity);
    EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes, bytes);
    const auto backed = loom_serve_virtual_buffer_missing_demand(first);
    EXPECT_EQ(backed.cached_bytes[0], 0u);
    EXPECT_EQ(backed.cached_bytes[1], 0u);
    EXPECT_EQ(backed.uncached_bytes, 0u);
  }
  // The failure remains armed through both warm cycles and is consumed only
  // after explicitly disposing the reusable native slabs.
  loom_serve_virtual_buffer_begin_trim(first);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
  loom_serve_memory_pool_trim(pool, 0);
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_virtual_buffer_commit(first, 0, bytes));
}

TEST_F(MemoryTest, PhysicalPeakSurvivesCachedSlabDisposal) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), kSlabSize, 256,
      &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_state_geometry(device), kSlabSize, 256,
      &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, kSlabSize));
  loom_serve_virtual_buffer_begin_trim(first);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(second, 0, kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);
  // Neither reservation ever owns more than one slab's worth of bytes, but
  // the native source owns both the live state class and the idle weight class.
  auto memory = loom_serve_memory_pool_statistics(pool);
  EXPECT_EQ(memory.committed_bytes, 2 * kSlabSize);
  EXPECT_EQ(memory.cached_bytes, kSlabSize);
  EXPECT_EQ(memory.peak_bytes, 2 * kSlabSize);
  loom_serve_memory_pool_trim(pool, 0);
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  memory = loom_serve_memory_pool_statistics(pool);
  EXPECT_EQ(memory.committed_bytes, kSlabSize);
  EXPECT_EQ(memory.cached_bytes, 0u);
  EXPECT_EQ(memory.peak_bytes, 2 * kSlabSize);
  CheckRetained(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);
}

TEST_F(DedicatedNativeWorkerMemoryTest,
       HeldGrowthAndUnrelatedReturnDoNotCreateOtherClassCredit) {
  const auto bulk = loom_serve_device_weight_geometry(device);
  const auto state = loom_serve_device_state_geometry(device);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(pool, bulk, kSlabSize, 256,
                                                    &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(pool, bulk, kSlabSize, 256,
                                                    &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, state, 2 * kSlabSize, 256, &statistics[2], &third));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, kSlabSize));
  loom_serve_memory_operation_t* operation = nullptr;
  {
    loom_serve::testing::ScopedNativeHold hold(native,
                                               IREE_ALLOCATOR_COMMAND_CALLOC);
    const loom_serve_memory_range_t range = {second, 0, kSlabSize};
    loom_serve_memory_admission_t commit_admission;
    IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
        pool, 1, &range, nullptr, 0, &commit_admission, &operation));
    ASSERT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
    hold.WaitEntered();
    // No consumers remain on first. Its checked detach returns a regular
    // slab while the independent cold acquisition still owns pending capacity.
    IREE_ASSERT_OK(loom_serve_virtual_buffer_free(first));
    first = nullptr;
    loom_serve_memory_admission_t admission;
    IREE_ASSERT_OK(loom_serve_memory_pool_prepare(
        pool, loom_serve_virtual_buffer_missing_demand(third), &admission));
    EXPECT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_DEFERRED);
    EXPECT_EQ(statistics[2].pending_bytes, 0u);
  }
  loom_serve_memory_operation_await(operation);
  IREE_ASSERT_OK(loom_serve_memory_operation_query_status(operation));
  loom_serve_memory_operation_free(operation);
  loom_serve_memory_admission_t admission;
  IREE_ASSERT_OK(loom_serve_memory_pool_prepare(
      pool, loom_serve_virtual_buffer_missing_demand(third), &admission));
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(third, 0, 2 * kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x11223344);
  RoundTrip(loom_serve_virtual_buffer_handle(third), kSlabSize, 0x55667788);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            3 * kSlabSize);
}

TEST_F(DedicatedNativeWorkerMemoryTest,
       IncompatibleIdleSlabsStayChargedUntilNativeDisposal) {
  const auto bulk = loom_serve_device_weight_geometry(device);
  const auto state = loom_serve_device_state_geometry(device);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, bulk, 3 * kSlabSize, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, state, 2 * kSlabSize, 256, &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 3 * kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);
  loom_serve_virtual_buffer_begin_trim(first);
  loom_serve_virtual_buffer_keep(first, 0, kSlabSize);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
  {
    loom_serve::testing::ScopedNativeHold hold(native,
                                               IREE_ALLOCATOR_COMMAND_FREE);
    loom_serve_memory_admission_t admission;
    IREE_ASSERT_OK(loom_serve_memory_pool_prepare(
        pool, loom_serve_virtual_buffer_missing_demand(second), &admission));
    EXPECT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_DEFERRED);
    hold.WaitEntered();
    const auto memory = loom_serve_memory_pool_statistics(pool);
    EXPECT_EQ(memory.committed_bytes, 3 * kSlabSize);
    EXPECT_EQ(memory.releasing_bytes, 2 * kSlabSize);
    EXPECT_EQ(memory.cached_bytes, 0u);
    CheckRetained(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);
  }
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(second, 0, 2 * kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(second), kSlabSize, 0x87654321);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            3 * kSlabSize);
}

TEST_F(DedicatedNativeWorkerMemoryTest,
       FailedAcquisitionPreservesAnAdmittedStreamingCohort) {
  const auto bulk = loom_serve_device_weight_geometry(device);
  const auto state = loom_serve_device_state_geometry(device);
  const uint64_t failed_bytes = 2 * state.slab_size;
  const uint64_t cohort_bytes = 6 * state.slab_size;
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, state, failed_bytes, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, bulk, bulk.slab_size, 256, &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, state, cohort_bytes, 256, &statistics[2], &third));
  std::array<loom_serve_memory_operation_t*, 4> operations = {};
  native.FailNextAllocation(IREE_ALLOCATOR_COMMAND_CALLOC, true);
  {
    loom_serve::testing::ScopedNativeHold hold(native,
                                               IREE_ALLOCATOR_COMMAND_CALLOC);
    const loom_serve_memory_range_t failed = {first, 0, failed_bytes};
    loom_serve_memory_admission_t commit_admission;
    IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
        pool, 1, &failed, nullptr, 0, &commit_admission, &operations[0]));
    ASSERT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
    hold.WaitEntered();
    const loom_serve_memory_range_t growth = {second, 0, bulk.slab_size};
    IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
        pool, 1, &growth, nullptr, 0, &commit_admission, &operations[1]));
    ASSERT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);

    // A complete cohort is admitted before its separate read windows commit
    // backing. Both unrelated operations still own accepted capacity here.
    loom_serve_memory_admission_t admission;
    IREE_ASSERT_OK(loom_serve_memory_pool_prepare(
        pool, loom_serve_virtual_buffer_missing_demand(third), &admission));
    ASSERT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
    hold.ReleaseAndArm(IREE_ALLOCATOR_COMMAND_CALLOC);
    hold.WaitEntered();
    loom_serve_memory_operation_await(operations[0]);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        loom_serve_memory_operation_query_status(operations[0]));
    loom_serve_memory_operation_free(operations[0]);
    operations[0] = nullptr;
    EXPECT_FALSE(loom_serve_memory_operation_is_complete(operations[1]));
    IREE_EXPECT_OK(loom_serve_memory_pool_query_status(pool));

    // Publishing the unrelated failure must neither revoke the cohort's
    // admission nor grant its still-unreconciled native charge to a retry.
    const loom_serve_memory_range_t windows[] = {
        {third, 0, state.slab_size},
        {third, state.slab_size, cohort_bytes - state.slab_size},
    };
    for (size_t i = 0; i < IREE_ARRAYSIZE(windows); ++i) {
      IREE_EXPECT_OK(loom_serve_memory_pool_commit_async(
          pool, 1, windows + i, nullptr, 0, &commit_admission,
          &operations[i + 2]));
      EXPECT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
    }
    IREE_ASSERT_OK(loom_serve_memory_pool_prepare(
        pool, loom_serve_virtual_buffer_missing_demand(first), &admission));
    EXPECT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_DEFERRED);
  }
  for (auto* operation : operations) {
    if (!operation) {
      continue;
    }
    loom_serve_memory_operation_await(operation);
    IREE_EXPECT_OK(loom_serve_memory_operation_query_status(operation));
    loom_serve_memory_operation_free(operation);
  }
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            bulk.slab_size + cohort_bytes);
  RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);
  RoundTrip(loom_serve_virtual_buffer_handle(third), state.slab_size,
            0x87654321);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, failed_bytes));
  RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x11223344);
  CheckRetained(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);
  CheckRetained(loom_serve_virtual_buffer_handle(third), state.slab_size,
                0x87654321);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            3 * kSlabSize);
}

TEST_F(DedicatedNativeWorkerMemoryTest,
       ExplicitTrimSettlesBeforeStreamingCohortAdmission) {
  const auto bulk = loom_serve_device_weight_geometry(device);
  const auto state = loom_serve_device_state_geometry(device);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, bulk, bulk.slab_size, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, state, state.slab_size, 256, &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, bulk, 2 * bulk.slab_size, 256, &statistics[2], &third));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, bulk.slab_size));
  loom_serve_virtual_buffer_begin_trim(first);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
  loom_serve_memory_operation_t* operation = nullptr;
  {
    loom_serve::testing::ScopedNativeHold hold(native,
                                               IREE_ALLOCATOR_COMMAND_CALLOC);
    const loom_serve_memory_range_t growth = {second, 0, state.slab_size};
    loom_serve_memory_admission_t commit_admission;
    IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
        pool, 1, &growth, nullptr, 0, &commit_admission, &operation));
    ASSERT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
    hold.WaitEntered();
    loom_serve_memory_pool_trim(pool, 0);
    // The cohort would fit, but its promise must not depend on cached backing
    // that an already-requested trim could destroy between streamed commits.
    loom_serve_memory_admission_t admission;
    IREE_ASSERT_OK(loom_serve_memory_pool_prepare(
        pool, loom_serve_virtual_buffer_missing_demand(third), &admission));
    EXPECT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_DEFERRED);
  }
  loom_serve_memory_operation_await(operation);
  IREE_ASSERT_OK(loom_serve_memory_operation_query_status(operation));
  loom_serve_memory_operation_free(operation);
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            state.slab_size);
  loom_serve_memory_admission_t admission;
  IREE_ASSERT_OK(loom_serve_memory_pool_prepare(
      pool, loom_serve_virtual_buffer_missing_demand(third), &admission));
  ASSERT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(third, 0, bulk.slab_size));
  IREE_ASSERT_OK(
      loom_serve_virtual_buffer_commit(third, bulk.slab_size, bulk.slab_size));
  RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x11223344);
  RoundTrip(loom_serve_virtual_buffer_handle(third), bulk.slab_size,
            0x55667788);
}

TEST_F(MemoryTest, MixedSlabsAndShortTailsShareExactNativeBudget) {
  const auto bulk = loom_serve_device_weight_geometry(device);
  const auto state = loom_serve_device_state_geometry(device);
  const uint64_t tail_bytes = state.slab_size;
  const uint64_t first_bytes = bulk.slab_size + tail_bytes;
  const uint64_t committed_bytes = first_bytes + state.slab_size;
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking(committed_bytes));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, bulk, first_bytes - 1, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, state, 2 * state.slab_size, 256, &statistics[1], &second));
  EXPECT_EQ(statistics[0].reserved_bytes, first_bytes);
  EXPECT_EQ(loom_serve_virtual_buffer_slab_count(first), 2u);

  // Duplicate and overlapping logical ranges charge each physical extent once.
  const loom_serve_memory_range_t ranges[] = {
      {first, bulk.slab_size - 4, 8},
      {second, 0, 4},
      {first, bulk.slab_size, tail_bytes - 1},
  };
  deferred_backing->defer = true;
  loom_serve_memory_admission_t commit_admission;
  loom_serve_memory_operation_t* operation = nullptr;
  IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
      pool, IREE_ARRAYSIZE(ranges), ranges, nullptr, 0, &commit_admission,
      &operation));
  ASSERT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
  deferred_backing->WaitForAcquisition();
  EXPECT_EQ(statistics[0].pending_bytes, first_bytes);
  EXPECT_EQ(statistics[1].pending_bytes, state.slab_size);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).pending_bytes,
            committed_bytes);
  deferred_backing->ResumeAcquisitions();
  loom_serve_memory_operation_await(operation);
  IREE_ASSERT_OK(loom_serve_memory_operation_query_status(operation));
  loom_serve_memory_operation_free(operation);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            committed_bytes);
  EXPECT_EQ(statistics[0].committed_bytes, first_bytes);
  EXPECT_EQ(statistics[1].committed_bytes, state.slab_size);
  RoundTrip(loom_serve_virtual_buffer_handle(first), bulk.slab_size,
            0x12345678);
  RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x87654321);

  const loom_serve_memory_range_t extra = {second, state.slab_size, 4};
  IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
      pool, 1, &extra, nullptr, 0, &commit_admission, &operation));
  EXPECT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_CAPACITY);
  EXPECT_EQ(operation, nullptr);

  // Return a full bulk slab and a state slab in one batch, retaining the tail.
  // Native return, not the logical trim, makes their capacity available again.
  deferred_backing->DeferReturns();
  loom_serve_virtual_buffer_begin_trim(first);
  loom_serve_virtual_buffer_keep(first, bulk.slab_size, tail_bytes);
  loom_serve_virtual_buffer_begin_trim(second);
  loom_serve_virtual_buffer_t* buffers[] = {first, second};
  IREE_ASSERT_OK(loom_serve_memory_pool_trim_async(
      pool, IREE_ARRAYSIZE(buffers), buffers, nullptr, 0, &operation));
  loom_serve_memory_operation_await(operation);
  IREE_ASSERT_OK(loom_serve_memory_operation_query_status(operation));
  loom_serve_memory_operation_free(operation);
  EXPECT_EQ(statistics[0].committed_bytes, tail_bytes);
  EXPECT_EQ(statistics[1].committed_bytes, 0u);
  loom_serve_memory_pool_trim(pool, 0);
  const auto retiring = loom_serve_memory_pool_statistics(pool);
  EXPECT_EQ(retiring.committed_bytes, committed_bytes);
  EXPECT_EQ(retiring.releasing_bytes, bulk.slab_size + state.slab_size);
  EXPECT_EQ(deferred_backing->PendingReturnCount(), 2u);
  IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
      pool, 1, &extra, nullptr, 0, &commit_admission, &operation));
  EXPECT_EQ(operation, nullptr);
  CheckRetained(loom_serve_virtual_buffer_handle(first), bulk.slab_size,
                0x12345678);
  // Retirement can finish before a caller handles the denied commitment. Its
  // original result must still request another admission turn: a fresh pending
  // query no longer distinguishes progress from permanent capacity exhaustion.
  deferred_backing->CompleteReturns();
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  EXPECT_FALSE(loom_serve_memory_pool_is_pending(pool));
  EXPECT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_DEFERRED);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            tail_bytes);
  bool admitted = false;
  IREE_ASSERT_OK(loom_serve_memory_pool_try_commit(pool, 1, &extra, &admitted));
  ASSERT_TRUE(admitted);
  RoundTrip(loom_serve_virtual_buffer_handle(second), state.slab_size,
            0xABCDEF01);
}

TEST_F(MemoryTest, PendingReleaseRetainsBudgetAndLeavesKeptSlabsUsable) {
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking());
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), 3 * kSlabSize, 256,
      &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), 2 * kSlabSize, 256,
      &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 3 * kSlabSize));
  ASSERT_NO_FATAL_FAILURE(
      RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0xA1B2C3D4));
  std::atomic<uint32_t> wakes = 0;
  loom_serve_memory_pool_set_observer(
      pool,
      [](void* user_data) {
        static_cast<std::atomic<uint32_t>*>(user_data)->fetch_add(1);
      },
      &wakes);
  loom_serve_memory_operation_t* operation = nullptr;
  iree_hal_semaphore_t* ready = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      loom_serve_device_handle(device), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &ready));
  {
    // Queue completion returns ownership to the source pool. Keep its real
    // parent tokens live until explicitly releasing them below, independently
    // of metadata destruction or the threads chosen by a native backend.
    deferred_backing->DeferReturns();
    loom_serve_virtual_buffer_begin_trim(first);
    loom_serve_virtual_buffer_keep(first, 0, kSlabSize);
    IREE_ASSERT_OK(loom_serve_memory_pool_trim_async(pool, 1, &first, ready, 1,
                                                     &operation));
    ASSERT_NE(operation, nullptr);
    loom_serve_memory_operation_await(operation);
    loom_serve_memory_pool_trim(pool, 0);
    EXPECT_EQ(deferred_backing->PendingReturnCount(), 2u);
    EXPECT_TRUE(loom_serve_memory_operation_is_complete(operation));
    const auto memory = loom_serve_memory_pool_statistics(pool);
    EXPECT_EQ(memory.committed_bytes, 3 * kSlabSize);
    EXPECT_EQ(memory.releasing_bytes, 2 * kSlabSize);
    EXPECT_EQ(statistics[0].releasing_bytes, 0u);
    loom_serve_memory_admission_t admission;
    IREE_ASSERT_OK(loom_serve_memory_pool_prepare(
        pool, loom_serve_virtual_buffer_missing_demand(second), &admission));
    EXPECT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_DEFERRED);
    ASSERT_NO_FATAL_FAILURE(
        CheckRetained(loom_serve_virtual_buffer_handle(first), 0, 0xA1B2C3D4));
    EXPECT_GE(wakes.load(), 1u);
    uint64_t value = 0;
    IREE_EXPECT_OK(iree_hal_semaphore_query(ready, &value));
    EXPECT_EQ(value, 1u);
    deferred_backing->CompleteReturns();
  }
  IREE_ASSERT_OK(iree_hal_semaphore_wait(ready, 1, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_TRUE(loom_serve_memory_operation_is_complete(operation));
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  const auto retired = loom_serve_memory_pool_statistics(pool);
  EXPECT_EQ(retired.pending_bytes, 0u);
  EXPECT_EQ(retired.releasing_bytes, 0u);
  EXPECT_EQ(retired.committed_bytes, kSlabSize);
  EXPECT_EQ(retired.released_bytes, 2 * kSlabSize);
  EXPECT_GE(wakes.load(), 1u);
  EXPECT_EQ(statistics[0].releasing_bytes, 0u);
  EXPECT_EQ(statistics[0].committed_bytes, kSlabSize);
  EXPECT_EQ(statistics[0].released_bytes, 2 * kSlabSize);
  loom_serve_memory_operation_free(operation);
  loom_serve_memory_pool_set_observer(pool, nullptr, nullptr);
  operation = nullptr;
  {
    loom_serve::testing::ScopedNativeHold gate(native,
                                               IREE_ALLOCATOR_COMMAND_CALLOC);
    IREE_ASSERT_OK(loom_serve_memory_pool_trim_async(pool, 1, &first, ready, 2,
                                                     &operation));
    EXPECT_EQ(operation, nullptr);
    uint64_t value = 0;
    IREE_EXPECT_OK(iree_hal_semaphore_query(ready, &value));
    EXPECT_EQ(value, 2u);
  }
  iree_hal_semaphore_release(ready);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(second, 0, 2 * kSlabSize));
  ASSERT_NO_FATAL_FAILURE(
      RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x55667788));
}

TEST_F(DedicatedNativeWorkerMemoryTest,
       PoolObserverDetachesFromAlreadyAcceptedNativeWork) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), kSlabSize, 256,
      &statistics[0], &first));
  std::atomic<uint32_t> wakes = 0;
  loom_serve_memory_pool_set_observer(
      pool,
      [](void* user_data) {
        static_cast<std::atomic<uint32_t>*>(user_data)->fetch_add(1);
      },
      &wakes);
  loom_serve_memory_operation_t* operation = nullptr;
  {
    loom_serve::testing::ScopedNativeHold gate(native,
                                               IREE_ALLOCATOR_COMMAND_CALLOC);
    const loom_serve_memory_range_t range = {first, 0, kSlabSize};
    loom_serve_memory_admission_t commit_admission;
    IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
        pool, 1, &range, nullptr, 0, &commit_admission, &operation));
    ASSERT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
    gate.WaitEntered();
    loom_serve_memory_pool_set_observer(pool, nullptr, nullptr);
  }
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  EXPECT_EQ(wakes.load(), 0u);
  loom_serve_memory_operation_free(operation);
}

TEST_F(DeviceMemoryTest, SparseGrowthTrimRegrowthAndSharedAdmission) {
  const auto before = loom_serve_memory_pool_statistics(pool);
  const auto* dispatch =
      iree_hal_queue_family(loom_serve_device_dispatch_queue(device));
  const auto* transfer =
      iree_hal_queue_family(loom_serve_device_transfer_queue(device));
  const iree_hal_pool_family_access_t families[] = {
      {
          .family = dispatch,
          .usage = IREE_HAL_BUFFER_USAGE_STORAGE |
                   IREE_HAL_BUFFER_USAGE_TRANSFER |
                   IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS,
      },
      {.family = transfer, .usage = IREE_HAL_BUFFER_USAGE_TRANSFER},
  };
  const iree_hal_pool_scope_t scope = {
      .family_count = dispatch == transfer ? 1u : 2u,
      .families = families,
  };
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      loom_serve_device_group(device), scope, iree_allocator_system(),
      &capabilities));
  if (capabilities.max_byte_length < kHighOffset + kSlabSize) {
    GTEST_SKIP() << "Native address interval limit is "
                 << capabilities.max_byte_length
                 << " bytes; this case requires offsets above four GiB";
  }
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), kHighOffset + kSlabSize,
      256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), 2 * kSlabSize, 256,
      &statistics[1], &second));
  auto* const identity = loom_serve_virtual_buffer_handle(first);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_serve_device_group_free(group));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes, 0u);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, kHighOffset, 256));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 512));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 256, 256));
  RoundTrip(identity, kHighOffset, 0x12345678);
  RoundTrip(identity, 0, 0xABCDEF01);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            2 * kSlabSize);

  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(second, 0, 256));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_serve_virtual_buffer_commit(second, kSlabSize, 256));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            3 * kSlabSize);
  loom_serve_virtual_buffer_begin_trim(first);
  loom_serve_virtual_buffer_keep(first, kHighOffset, 256);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).released_bytes,
            before.released_bytes + kSlabSize);
  EXPECT_EQ(loom_serve_virtual_buffer_handle(first), identity);
  CheckRetained(identity, kHighOffset, 0x12345678);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(second, kSlabSize, 256));
  RoundTrip(loom_serve_virtual_buffer_handle(second), kSlabSize, 0xDEADBEEF);
  loom_serve_virtual_buffer_begin_trim(second);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 512));
  EXPECT_EQ(loom_serve_virtual_buffer_handle(first), identity);
  RoundTrip(identity, 0, 0xCAFEBABE);
  CheckRetained(identity, kHighOffset, 0x12345678);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).peak_bytes, 3 * kSlabSize);
  EXPECT_EQ(statistics[0].committed_bytes, 2 * kSlabSize);
  EXPECT_EQ(statistics[1].committed_bytes, 0u);
  EXPECT_EQ(statistics[0].released_bytes, kSlabSize);
  EXPECT_EQ(statistics[1].released_bytes, 2 * kSlabSize);
  EXPECT_EQ(statistics[0].peak_bytes, 2 * kSlabSize);
  EXPECT_EQ(statistics[1].peak_bytes, 2 * kSlabSize);
}

TEST_F(MemoryTest, CohortAdmissionCountsUniqueSlabsAndLeavesDenialUntouched) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), 3 * kSlabSize, 256,
      &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), 2 * kSlabSize, 256,
      &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 256));
  RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);
  const loom_serve_memory_range_t ranges[] = {
      {first, 0, 256},
      {first, kSlabSize, 256},
      {first, kSlabSize + 128, 256},
      {second, kSlabSize - 128, 256},
  };
  bool admitted = true;
  IREE_ASSERT_OK(loom_serve_memory_pool_try_commit(pool, IREE_ARRAYSIZE(ranges),
                                                   ranges, &admitted));
  EXPECT_FALSE(admitted);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes, kSlabSize);
  EXPECT_EQ(statistics[1].committed_bytes, 0u);
  CheckRetained(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);
  IREE_ASSERT_OK(loom_serve_memory_pool_try_commit(pool, 3, ranges, &admitted));
  EXPECT_TRUE(admitted);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            2 * kSlabSize);
  RoundTrip(loom_serve_virtual_buffer_handle(first), kSlabSize, 0x87654321);
  IREE_ASSERT_OK(loom_serve_memory_pool_try_commit(pool, 3, ranges, &admitted));
  EXPECT_TRUE(admitted);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            2 * kSlabSize);
  CheckRetained(loom_serve_virtual_buffer_handle(first), kSlabSize, 0x87654321);
  const loom_serve_memory_range_t last = {second, 0, 256};
  IREE_ASSERT_OK(loom_serve_memory_pool_try_commit(pool, 1, &last, &admitted));
  EXPECT_TRUE(admitted);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            3 * kSlabSize);
}

TEST_F(MemoryTest, RecommittedSlabsPublishEveryWordOfTheNewGeneration) {
  constexpr size_t kGenerationCount = 4;
  constexpr uint64_t kOrigin = 2 * kSlabSize;
  constexpr size_t kWordsPerSlab = kSlabSize / sizeof(uint32_t);
  const std::array<loom_serve_virtual_buffer_t**, 2> destinations = {&first,
                                                                     &second};
  for (size_t i = 0; i < destinations.size(); ++i) {
    IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
        pool, loom_serve_device_weight_geometry(device),
        kOrigin + 4 * kSlabSize, 256, &statistics[i], destinations[i]));
  }
  const std::array<iree_hal_buffer_t*, 2> handles = {
      loom_serve_virtual_buffer_handle(first),
      loom_serve_virtual_buffer_handle(second)};
  // Prepare every generation before the first mapping turnover. Comparing or
  // filling a large host image between remap and device access would perturb
  // the publication boundary this case exercises.
  std::array<std::vector<uint32_t>, kGenerationCount> expected;
  for (size_t generation = 0; generation < kGenerationCount; ++generation) {
    expected[generation].resize(2 * kWordsPerSlab);
    for (size_t word = 0; word < expected[generation].size(); ++word) {
      expected[generation][word] =
          static_cast<uint32_t>((generation + 1) << 28) |
          static_cast<uint32_t>(word);
    }
  }
  std::vector<uint32_t> observed(2 * kWordsPerSlab);
  std::vector<uint32_t> captured(2 * kWordsPerSlab);
  uint64_t completion = 0;
  for (size_t generation = 0; generation < kGenerationCount; ++generation) {
    SCOPED_TRACE(generation);
    const uint64_t offset = kOrigin + (generation % 2) * 2 * kSlabSize;
    const std::array<loom_serve_memory_range_t, 2> ranges = {{
        {first, offset, kSlabSize},
        {second, offset, kSlabSize},
    }};
    std::array<iree_hal_transfer_operation_t, 2> uploads = {};
    std::array<iree_hal_transfer_operation_t, 2> downloads = {};
    for (size_t i = 0; i < handles.size(); ++i) {
      uploads[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
      uploads[i].upload = {expected[generation].data() + i * kWordsPerSlab,
                           handles[i], offset, kSlabSize};
      downloads[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
      downloads[i].download = {handles[i], offset,
                               observed.data() + i * kWordsPerSlab, kSlabSize};
    }
    bool admitted = false;
    IREE_ASSERT_OK(loom_serve_memory_pool_try_commit(pool, ranges.size(),
                                                     ranges.data(), &admitted));
    ASSERT_TRUE(admitted);
    IREE_ASSERT_OK(loom_serve_execution_transfer(execution, uploads.size(),
                                                 uploads.data(), &completion));
    IREE_ASSERT_OK(loom_serve_execution_transfer(
        execution, downloads.size(), downloads.data(), &completion));
    IREE_ASSERT_OK(loom_serve_execution_wait(execution, completion));
    ASSERT_EQ(observed, expected[generation]);
    if (generation) {
      ASSERT_EQ(captured, expected[generation - 1]);
    }
    // Capture and immediately relinquish both physical slabs, as in a cold
    // session suspension. The next iteration reuses the retained VA roots.
    for (size_t i = 0; i < handles.size(); ++i) {
      downloads[i].download.target = captured.data() + i * kWordsPerSlab;
    }
    IREE_ASSERT_OK(loom_serve_execution_transfer(
        execution, downloads.size(), downloads.data(), &completion));
    IREE_ASSERT_OK(loom_serve_execution_wait(execution, completion));
    for (auto* buffer : {first, second}) {
      loom_serve_virtual_buffer_begin_trim(buffer);
      IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(buffer));
    }
  }
  ASSERT_EQ(captured, expected.back());
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            2 * kSlabSize);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).cached_bytes,
            2 * kSlabSize);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).peak_bytes, 2 * kSlabSize);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).released_bytes,
            kGenerationCount * 2 * kSlabSize);
}

TEST_F(MemoryTest, FailedReclamationRetainsSharedFailureAndRemainingOwnership) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), kSlabSize, 256,
      &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), kSlabSize, 256,
      &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 256));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(second, 0, 256));
  RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);
  RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x87654321);
  loom_serve_memory_admission_t admission;
  IREE_ASSERT_OK(loom_serve_memory_pool_prepare(
      pool, {.cached_bytes = {4 * kSlabSize}}, &admission));
  EXPECT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_CAPACITY);
  IREE_EXPECT_OK(loom_serve_memory_pool_query_status(pool));

  // The dependency returns reusable backing before reporting a subsequent
  // failure. The requested capacity would now fit, but cannot be admitted.
  loom_serve_memory_pool_set_reclaimer(
      pool, {first, [](void* self, uint64_t target_bytes) -> iree_status_t {
               auto* buffer = static_cast<loom_serve_virtual_buffer_t*>(self);
               loom_serve_virtual_buffer_begin_trim(buffer);
               IREE_RETURN_IF_ERROR(loom_serve_virtual_buffer_trim(buffer));
               return iree_make_status(IREE_STATUS_UNAVAILABLE,
                                       "reclamation failed after release");
             }});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      loom_serve_memory_pool_prepare(pool, {.cached_bytes = {2 * kSlabSize}},
                                     &admission));
  EXPECT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_CAPACITY);
  EXPECT_EQ(statistics[0].committed_bytes, 0u);
  EXPECT_EQ(statistics[1].committed_bytes, kSlabSize);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).reserved_bytes,
            2 * kSlabSize);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            2 * kSlabSize);
  loom_serve_memory_pool_set_reclaimer(pool, {});
  for (int i = 0; i < 2; ++i) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                          loom_serve_memory_pool_query_status(pool));
  }
  // No new device work follows shared failure. TearDown drains accepted work
  // and releases the remaining ownership through the ordinary native path.
}

TEST_F(DeviceMemoryTest, DeviceTrimPreservesLiveWorkspaceAndMutableState) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), kSlabSize, 256,
      &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 256));
  RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0xABCDEF01);
  IREE_ASSERT_OK(
      loom_serve_execution_reserve_workspace(execution, kSlabSize, 256));
  const iree_hal_pool_reservation_request_t request = {
      .allocation_size = kSlabSize,
      .min_alignment = 256,
  };
  for (int iteration = 0; iteration < 2; ++iteration) {
    IREE_ASSERT_OK(
        loom_serve_execution_alloca(execution, 1, &request, &workspace));
    RoundTrip(workspace, 0, 0x12345678);
    const auto before = loom_serve_device_memory_statistics(device);
    EXPECT_EQ(before.retained.committed_bytes, kSlabSize);
    EXPECT_EQ(before.workspace.bytes_committed, kSlabSize);
    EXPECT_EQ(before.workspace.reservation_count, 1u);
    IREE_ASSERT_OK(loom_serve_test_device_trim_and_drain(device, 0));
    const auto live = loom_serve_device_memory_statistics(device);
    EXPECT_EQ(live.retained.committed_bytes, kSlabSize);
    EXPECT_EQ(live.workspace.bytes_committed, kSlabSize);
    EXPECT_EQ(live.workspace.reservation_count, 1u);
    // A real consumer still reaches the allocation after an unreachable trim.
    CheckRetained(workspace, 0, 0x12345678);
    RoundTrip(workspace, 0, 0x87654321);
    CheckRetained(loom_serve_virtual_buffer_handle(first), 0, 0xABCDEF01);
    uint64_t completion = 0;
    IREE_ASSERT_OK(
        loom_serve_execution_dealloca(execution, workspace, &completion));
    iree_hal_buffer_release(workspace);
    workspace = nullptr;
    IREE_ASSERT_OK(loom_serve_execution_drain(execution));
    IREE_ASSERT_OK(
        loom_serve_test_device_trim_and_drain(device, 2 * kSlabSize));
    EXPECT_EQ(
        loom_serve_device_memory_statistics(device).workspace.bytes_committed,
        kSlabSize);
    IREE_ASSERT_OK(loom_serve_test_device_trim_and_drain(device, kSlabSize));
    const auto idle = loom_serve_device_memory_statistics(device);
    EXPECT_EQ(idle.retained.committed_bytes, kSlabSize);
    EXPECT_EQ(idle.workspace.bytes_committed, 0u);
    EXPECT_EQ(idle.workspace.reservation_count, 0u);
    CheckRetained(loom_serve_virtual_buffer_handle(first), 0, 0xABCDEF01);
  }
}

TEST_F(DedicatedNativeWorkerMemoryTest,
       NativeAllocationPublicationDoesNotDoubleChargeAdmission) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), kSlabSize, 256,
      &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), 2 * kSlabSize, 256,
      &statistics[1], &second));
  std::array<loom_serve_memory_operation_t*, 2> operations = {};
  {
    // Select this acquisition before holding its publication. Earlier native
    // operations may still free metadata after their backing has retired.
    loom_serve::testing::ScopedNativeHold gate(native,
                                               IREE_ALLOCATOR_COMMAND_CALLOC);
    const loom_serve_memory_range_t first_range = {first, 0, kSlabSize};
    loom_serve_memory_admission_t commit_admission;
    IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
        pool, 1, &first_range, nullptr, 0, &commit_admission, &operations[0]));
    ASSERT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
    gate.WaitEntered();
    EXPECT_EQ(loom_serve_memory_pool_statistics(pool).peak_bytes, 0u);
    // The source pool frees its acquisition capture after accounting native
    // backing and before publishing completion to the queue. Transfer the hold
    // atomically so that this worker cannot pass the publication boundary.
    gate.ReleaseAndArm(IREE_ALLOCATOR_COMMAND_FREE);
    gate.WaitEntered();
    EXPECT_FALSE(loom_serve_memory_operation_is_complete(operations[0]));
    const auto pending = loom_serve_memory_pool_statistics(pool);
    EXPECT_EQ(pending.peak_bytes, kSlabSize);
    EXPECT_EQ(pending.pending_bytes, kSlabSize);
    EXPECT_EQ(pending.committed_bytes, 0u);
    EXPECT_EQ(pending.releasing_bytes, 0u);
    loom_serve_memory_admission_t admission;
    IREE_ASSERT_OK(loom_serve_memory_pool_prepare(
        pool, {.cached_bytes = {2 * kSlabSize}}, &admission));
    EXPECT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
    const loom_serve_memory_range_t second_range = {second, 0, 2 * kSlabSize};
    IREE_EXPECT_OK(loom_serve_memory_pool_commit_async(
        pool, 1, &second_range, nullptr, 0, &commit_admission, &operations[1]));
    EXPECT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
    EXPECT_NE(operations[1], nullptr);
    EXPECT_EQ(loom_serve_memory_pool_statistics(pool).pending_bytes,
              3 * kSlabSize);
  }
  for (auto* operation : operations) {
    loom_serve_memory_operation_await(operation);
    IREE_EXPECT_OK(loom_serve_memory_operation_query_status(operation));
    loom_serve_memory_operation_free(operation);
  }
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  const auto completed = loom_serve_memory_pool_statistics(pool);
  EXPECT_EQ(completed.pending_bytes, 0u);
  EXPECT_EQ(completed.committed_bytes, 3 * kSlabSize);
  EXPECT_EQ(completed.releasing_bytes, 0u);
  RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);
  RoundTrip(loom_serve_virtual_buffer_handle(second), kSlabSize, 0x87654321);
}

TEST_F(MemoryTest, PendingCommitChargesBudgetAndLeavesPublishedStorageUsable) {
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking());
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), 3 * kSlabSize, 256,
      &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), kSlabSize, 256,
      &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);
  iree_hal_semaphore_t* ready = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      loom_serve_device_handle(device), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &ready));
  loom_serve_memory_operation_t* operation = nullptr;
  {
    deferred_backing->defer = true;
    const loom_serve_memory_range_t ranges[] = {
        {first, kSlabSize, 2 * kSlabSize},
        {first, kSlabSize + 256, kSlabSize},
    };
    loom_serve_memory_admission_t commit_admission;
    IREE_EXPECT_OK(loom_serve_memory_pool_commit_async(
        pool, IREE_ARRAYSIZE(ranges), ranges, ready, 1, &commit_admission,
        &operation));
    EXPECT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
    deferred_backing->WaitForAcquisition();
    EXPECT_FALSE(loom_serve_memory_operation_is_complete(operation));
    const auto pending = loom_serve_memory_pool_statistics(pool);
    EXPECT_EQ(pending.committed_bytes, kSlabSize);
    EXPECT_EQ(pending.pending_bytes, 2 * kSlabSize);
    EXPECT_EQ(statistics[0].pending_bytes, 2 * kSlabSize);
    const loom_serve_memory_range_t other = {second, 0, kSlabSize};
    bool admitted = false;
    IREE_EXPECT_OK(
        loom_serve_memory_pool_try_commit(pool, 1, &other, &admitted));
    EXPECT_FALSE(admitted);
    // Native work already backed by a different range remains executable while
    // commitment is held. No source-file readiness or queue ordering is mocked.
    CheckRetained(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);
    deferred_backing->ResumeAcquisitions();
  }
  IREE_ASSERT_OK(iree_hal_semaphore_wait(ready, 1, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_TRUE(loom_serve_memory_operation_is_complete(operation));
  IREE_EXPECT_OK(loom_serve_memory_pool_query_status(pool));
  const auto completed = loom_serve_memory_pool_statistics(pool);
  EXPECT_EQ(completed.committed_bytes, 3 * kSlabSize);
  EXPECT_EQ(completed.pending_bytes, 0u);
  RoundTrip(loom_serve_virtual_buffer_handle(first), 2 * kSlabSize, 0x87654321);
  loom_serve_memory_operation_free(operation);
  operation = nullptr;
  // Already-backed requests signal readiness without another acquisition and
  // consume no additional budget.
  {
    deferred_backing->defer = true;
    const loom_serve_memory_range_t range = {first, kSlabSize, kSlabSize};
    loom_serve_memory_admission_t commit_admission;
    IREE_EXPECT_OK(loom_serve_memory_pool_commit_async(
        pool, 1, &range, ready, 2, &commit_admission, &operation));
    EXPECT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
    EXPECT_EQ(operation, nullptr);
    uint64_t value = 0;
    IREE_EXPECT_OK(iree_hal_semaphore_query(ready, &value));
    EXPECT_EQ(value, 2u);
    EXPECT_EQ(deferred_backing->pending.count, 0u);
    deferred_backing->ResumeAcquisitions();
  }
  iree_hal_semaphore_release(ready);
}

TEST_F(MemoryTest, FailedReadinessDoesNotReleaseNativeOperationOwnership) {
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking());
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), 2 * kSlabSize, 256,
      &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), kSlabSize, 256,
      &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(second, 0, kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);
  iree_hal_semaphore_t* ready = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      loom_serve_device_handle(device), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &ready));
  loom_serve_memory_operation_t* operation = nullptr;
  {
    deferred_backing->defer = true;
    const loom_serve_memory_range_t range = {first, 0, kSlabSize};
    loom_serve_memory_admission_t commit_admission;
    IREE_EXPECT_OK(loom_serve_memory_pool_commit_async(
        pool, 1, &range, ready, 1, &commit_admission, &operation));
    EXPECT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
    deferred_backing->WaitForAcquisition();
    // A failed timeline can be observed before an accepted native producer has
    // even started. It is not permission to destroy its reservation or job.
    iree_hal_semaphore_fail(
        ready, iree_make_status(IREE_STATUS_CANCELLED, "cancelled timeline"));
    uint64_t value = 0;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                          iree_hal_semaphore_query(ready, &value));
    EXPECT_FALSE(loom_serve_memory_operation_is_complete(operation));
    EXPECT_EQ(loom_serve_memory_pool_statistics(pool).pending_bytes, kSlabSize);
    CheckRetained(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);
    deferred_backing->ResumeAcquisitions();
  }
  loom_serve_memory_operation_await(operation);
  EXPECT_TRUE(loom_serve_memory_operation_is_complete(operation));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        loom_serve_memory_operation_query_status(operation));
  IREE_EXPECT_OK(loom_serve_memory_pool_query_status(pool));
  uint64_t value = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        iree_hal_semaphore_query(ready, &value));
  const auto memory = loom_serve_memory_pool_statistics(pool);
  EXPECT_EQ(memory.committed_bytes, 2 * kSlabSize);
  EXPECT_EQ(memory.pending_bytes, 0u);
  loom_serve_memory_operation_free(operation);
  iree_hal_semaphore_release(ready);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, kSlabSize, kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(first), kSlabSize, 0x87654321);
  CheckRetained(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);
}

TEST_F(MemoryTest, NativeAllocationFailurePreservesWarmWorkAndAllowsRetry) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), kSlabSize, 256,
      &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), kSlabSize, 256,
      &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);

  native.FailNextAllocation(IREE_ALLOCATOR_COMMAND_CALLOC,
                            /*native_only=*/true);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_virtual_buffer_commit(second, 0, kSlabSize));
  IREE_EXPECT_OK(loom_serve_memory_pool_query_status(pool));
  EXPECT_EQ(statistics[1].pending_bytes, 0u);
  EXPECT_EQ(statistics[1].committed_bytes, 0u);
  CheckRetained(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);

  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(second, 0, kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x87654321);
  CheckRetained(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);
}

TEST_F(MemoryTest, FailedMappingKeepsSourcesUntilCheckedDetach) {
  constexpr uint64_t kMappingBytes = 2 * kSlabSize;
  mapping_queue.Initialize(loom_serve_device_transfer_queue(device));
  ASSERT_NO_FATAL_FAILURE(
      UseDeferredBacking(kMappingBytes + kSlabSize, &mapping_queue));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), kMappingBytes, 256,
      &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_device_weight_geometry(device), kSlabSize, 256,
      &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(second, 0, kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);

  // Mapping rejection cannot retire the allocation already accepted by the
  // native queue. Keep it pending until the failed mapping has been observed.
  deferred_backing->defer = true;
  mapping_queue.fail_next_map = true;
  const loom_serve_memory_range_t range = {first, 0, kMappingBytes};
  loom_serve_memory_admission_t commit_admission;
  loom_serve_memory_operation_t* operation = nullptr;
  IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
      pool, 1, &range, nullptr, 0, &commit_admission, &operation));
  ASSERT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
  deferred_backing->WaitForAcquisition();
  EXPECT_FALSE(mapping_queue.fail_next_map);
  EXPECT_FALSE(loom_serve_memory_operation_is_complete(operation));
  deferred_backing->ResumeAcquisitions();
  loom_serve_memory_operation_await(operation);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_memory_operation_query_status(operation));
  loom_serve_memory_operation_free(operation);
  IREE_EXPECT_OK(loom_serve_memory_pool_query_status(pool));
  EXPECT_EQ(statistics[0].pending_bytes, 0u);
  EXPECT_EQ(statistics[0].committed_bytes, kMappingBytes);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_virtual_buffer_commit(first, 0, kSlabSize));
  CheckRetained(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);

  IREE_ASSERT_OK(loom_serve_virtual_buffer_free(first));
  first = nullptr;
  loom_serve_memory_pool_trim(pool, 0);
  IREE_EXPECT_OK(loom_serve_memory_pool_await(pool));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes, kSlabSize);
  CheckRetained(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);
}

}  // namespace
