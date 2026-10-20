// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_STORAGE_VIRTUAL_POOL_H_
#define LOOM_SERVE_RUNTIME_STORAGE_VIRTUAL_POOL_H_

#include "loom/serve/runtime/host/wake.h"
#include "loom/serve/runtime/storage/memory.h"

#ifdef __cplusplus
extern "C" {
#endif

// Prepared virtual ranges backing a logical allocator. Physical storage belongs
// exclusively to memory_pool; this pool never acquires native slabs itself.
// Range addresses survive physical trim. Queue callbacks may acquire/release
// ranges concurrently, while advance/free share memory_pool's serialized
// owner. Cold callbacks publish requests and wake that owner; ready reuse
// requires neither owner progress nor allocation.
typedef struct loom_serve_virtual_pool_t loom_serve_virtual_pool_t;

// Reserves one initial address range to qualify the immutable prepared-storage
// contract. No physical storage is committed. The physical owner outlives this
// pool. geometry defines physical slabs, not logical reservation sizes.
iree_status_t loom_serve_virtual_pool_allocate(
    loom_serve_memory_pool_t* memory_pool,
    loom_serve_memory_geometry_t geometry, iree_allocator_t host_allocator,
    loom_serve_virtual_pool_t** out_pool);

// Borrowed HAL interface for TLSF or another logical allocator.
iree_hal_pool_t* loom_serve_virtual_pool_handle(
    loom_serve_virtual_pool_t* pool);

// Installs the owner's nonblocking wake. Detachment joins callback access.
// A blocking outer caller may instead use wait below. Neither callback may
// invoke advance, free, or another operation on the physical owner.
void loom_serve_virtual_pool_set_observer(loom_serve_virtual_pool_t* pool,
                                          loom_serve_wake_callback_t observer);

// Advances accepted cold allocations and physical trims without waiting.
// Allocation errors complete their original HAL requests. An unowned trim
// failure propagates here and preserves the range until outer teardown.
iree_status_t loom_serve_virtual_pool_advance(loom_serve_virtual_pool_t* pool);

// Whether cold requests, mapping operations, or requested trims need owner
// progress. Active reservations and idle, retained backing are not pending.
bool loom_serve_virtual_pool_is_pending(loom_serve_virtual_pool_t* pool);

// Blocking outer-owner wait that also advances cold backing. Shared task
// schedulers use advance and their normal semaphore observation instead.
// The caller is the physical owner and excludes concurrent owner operations.
iree_status_t loom_serve_virtual_pool_wait(loom_serve_virtual_pool_t* pool,
                                           iree_hal_semaphore_t* semaphore,
                                           uint64_t value);

// Outer teardown after all clients and accepted callbacks have retired.
// Releases backing and addresses; cleanup failure retains ownership for retry.
// On success consumes this pool's initial HAL reference. NULL is accepted.
iree_status_t loom_serve_virtual_pool_free(loom_serve_virtual_pool_t* pool);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_STORAGE_VIRTUAL_POOL_H_
