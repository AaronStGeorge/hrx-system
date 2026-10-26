// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/aql/state.h"

void iree_hal_amd_gpu_aql_queue_close(iree_hal_queue_t* base) {
  iree_atomic_store(&((iree_hal_amd_gpu_aql_queue_t*)base)->closed, 1,
                    iree_memory_order_release);
}

bool iree_hal_amd_gpu_aql_queue_is_idle(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  iree_slim_mutex_lock(&queue->pending.completion_mutex);
  const bool idle =
      !iree_atomic_load(&queue->pending.host_owners,
                        iree_memory_order_acquire) &&
      !iree_atomic_load(&queue->growth.pending_count,
                        iree_memory_order_acquire) &&
      !iree_hal_amd_gpu_aql_scratch_is_pending(&queue->scratch) &&
      iree_hal_amd_gpu_publisher_is_idle(&queue->publisher) &&
      iree_atomic_load(&queue->producer.count, iree_memory_order_acquire) ==
          iree_atomic_load(&queue->completed.count, iree_memory_order_acquire);
  iree_slim_mutex_unlock(&queue->pending.completion_mutex);
  return idle;
}

void iree_hal_amd_gpu_aql_queue_assign_frontier(
    iree_hal_queue_t* base, iree_async_frontier_tracker_t* tracker,
    iree_async_axis_t axis) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  queue->frontier.tracker = tracker;
  queue->frontier.axis = axis;
}

void iree_hal_amd_gpu_aql_queue_set_release(iree_hal_queue_t* base,
                                            bool (*release)(void*,
                                                            iree_hal_queue_t*),
                                            void* user_data) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  queue->release.fn = release;
  queue->release.user_data = user_data;
}

uint64_t iree_hal_amd_gpu_aql_queue_completed_epoch(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  return (uint64_t)iree_atomic_load(&queue->completed.count,
                                    iree_memory_order_acquire);
}

amdf_status_t iree_hal_amd_gpu_aql_queue_shutdown(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  IREE_ASSERT(iree_hal_amd_gpu_aql_queue_is_idle(base),
              "native shutdown requires joined accepted work");
  amdf_status_t status = iree_hal_amd_gpu_user_queue_deinitialize(
      &queue->storage, queue->producer.ring.reserved);
  if (!amdf_status_is_ok(status)) {
    queue->context->cleanup_blocked = true;
    return status;
  }
  if (queue->scratch.growth) {
    iree_hal_amd_gpu_aql_scratch_deinitialize(&queue->scratch);
  }
  iree_atomic_store(&queue->options.signals->waiting, 0,
                    iree_memory_order_release);
  iree_hal_amd_gpu_signal_cache_trim(queue->context, 0, queue->options.signals);
  iree_hal_amd_gpu_argument_storage_t* arguments = queue->producer.arguments;
  if (arguments) {
    arguments->next = queue->producer.retired_arguments;
  } else {
    arguments = queue->producer.retired_arguments;
  }
  queue->producer.arguments = queue->producer.retired_arguments = NULL;
  while (arguments) {
    iree_hal_amd_gpu_argument_storage_t* next = arguments->next;
    iree_hal_amd_gpu_argument_storage_free(queue->context, arguments);
    arguments = next;
  }
  return AMDF_STATUS_OK;
}

void iree_hal_amd_gpu_aql_queue_destroy(iree_hal_queue_t* base) {
  iree_hal_amd_gpu_aql_queue_t* queue = (void*)base;
  if (queue->release.fn && !queue->release.fn(queue->release.user_data, base)) {
    return;
  }
  IREE_ASSERT(!queue->storage.queue,
              "parent must join native ownership before final queue release");
  iree_hal_amd_gpu_publisher_deinitialize(&queue->publisher);
  iree_hal_amd_gpu_host_wait_set_deinitialize(&queue->pending.waits);
  iree_slim_mutex_deinitialize(&queue->pending.completion_mutex);
  iree_async_semaphore_wait_context_deinitialize(&queue->wait_context);
  iree_hal_amd_gpu_capture_pool_deinitialize(&queue->captures);
  uint64_t* frontier = queue->producer.frontier;
  if (frontier != (void*)(queue->issued + queue->options.retirement_capacity)) {
    iree_allocator_free(queue->host_allocator, frontier);
  }
  iree_allocator_free_aligned(queue->host_allocator, queue);
}
