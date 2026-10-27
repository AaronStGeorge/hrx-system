// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_DEVICE_POOL_H_
#define LOOM_SERVE_RUNTIME_DEVICE_POOL_H_

#include "loom/serve/runtime/execution/workspace.h"
#include "loom/serve/runtime/parameter/residency.h"

#ifdef __cplusplus
extern "C" {
#endif

// Explicit storage, admission, and reclamation owner. Consumers share policy
// by borrowing this object, independently of their execution device. The HAL
// group and allocation queue outlive the pool and all of its consumers.
typedef struct loom_serve_pool_t loom_serve_pool_t;

typedef struct loom_serve_pool_options_t {
  // Stable virtual roots with independently reclaimable physical backing.
  bool elastic;
  // Preferred parameter and workspace slab extent, in bytes.
  iree_device_size_t weight_slab_size;
  // Preferred sparse-state slab extent, in bytes.
  iree_device_size_t state_slab_size;
  // Serving-managed physical allowance; zero leaves it unbounded.
  uint64_t memory_limit;
  // Fixed workspace sanitizer policy, qualified by the HAL child allocator.
  iree_hal_asan_pool_options_t asan;
} loom_serve_pool_options_t;

// Qualifies the complete simultaneous access scope before publishing an owner.
// Creates metadata and native sources, not model payload. All mutable pool and
// residency operations use one serialized host owner. Physical placement and
// execution ordering are not inferred from the selected allocation queue.
iree_status_t loom_serve_pool_allocate(iree_hal_device_group_t* group,
                                       iree_hal_pool_scope_t scope,
                                       iree_hal_queue_t* allocation_queue,
                                       const loom_serve_pool_options_t* options,
                                       iree_allocator_t host_allocator,
                                       loom_serve_pool_t** out_pool);

// True while registered models, parameter banks, retained output, or elastic
// reservations still borrow the owner. The pool's own idle workspace ranges
// are excluded. NULL is unused. Called on the serialized owner.
bool loom_serve_pool_is_in_use(const loom_serve_pool_t* pool);

// Releases a retired owner. Live consumers or failed native cleanup
// preserve remaining ownership for the caller. NULL is permitted. All model,
// execution, and result consumers retire before the caller destroys the pool.
iree_status_t loom_serve_pool_free(loom_serve_pool_t* pool);

// Borrowed resources outliving every model, execution, and output using them.
iree_hal_pool_t* loom_serve_pool_storage(const loom_serve_pool_t* pool);
loom_serve_workspace_t* loom_serve_pool_workspace(
    const loom_serve_pool_t* pool);
loom_serve_memory_pool_t* loom_serve_pool_memory(const loom_serve_pool_t* pool);
loom_serve_memory_geometry_t loom_serve_pool_weight_geometry(
    const loom_serve_pool_t* pool);
loom_serve_memory_geometry_t loom_serve_pool_state_geometry(
    const loom_serve_pool_t* pool);
loom_serve_residency_cache_t* loom_serve_pool_residency_cache(
    const loom_serve_pool_t* pool);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_DEVICE_POOL_H_
