// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_QUEUE_SCOPE_H_
#define IREE_HAL_DRIVERS_AMDGPU_QUEUE_SCOPE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Immutable identity for one logical AMDGPU host queue.
//
// Executable-load paths use this cold-path metadata to publish queue-specific
// globals without adding per-dispatch host bookkeeping. Device code can derive
// a queue-local packet slot from its implicit dispatch pointer using
// |aql_ring_base|, |aql_ring_mask|, and optional queue-owned TSAN state.
typedef struct iree_hal_amdgpu_queue_scope_t {
  // One-bit HAL queue affinity selecting this queue.
  iree_hal_queue_affinity_t queue_affinity;

  // Flattened logical queue ordinal in the owning HAL device.
  iree_host_size_t queue_ordinal;

  // Physical GPU device ordinal owning this queue.
  iree_host_size_t physical_device_ordinal;

  // Queue ordinal relative to |physical_device_ordinal|.
  iree_host_size_t physical_queue_ordinal;

  // Device-visible base address of the HSA AQL packet ring.
  uint64_t aql_ring_base;

  // Power-of-two packet-ring slot mask.
  uint64_t aql_ring_mask;

  // Device-visible iree_hal_amdgpu_tsan_queue_state_t pointer, or zero.
  uint64_t tsan_queue_state_base;
} iree_hal_amdgpu_queue_scope_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_QUEUE_SCOPE_H_
