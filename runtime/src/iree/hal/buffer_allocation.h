// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_BUFFER_ALLOCATION_H_
#define IREE_HAL_BUFFER_ALLOCATION_H_

#include "iree/hal/buffer.h"
#include "iree/hal/pool.h"
#include "iree/hal/profile_sink.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Identity of one logical allocation, independent of which queue releases it.
typedef struct iree_hal_buffer_allocation_profile_t {
  // Nonzero process-wide allocation identifier, stable through decommit.
  uint64_t id;
  // Origin device's capture session, or zero when no session was captured.
  // Only the origin device may use this value to filter its own events.
  iree_hal_profile_session_id_t session_id;
} iree_hal_buffer_allocation_profile_t;

// Returns a new process-wide allocation identity. Producers use the same
// namespace for native storage and queue allocation epochs so that captures
// spanning devices can join allocation and release events unambiguously.
IREE_API_EXPORT uint64_t iree_hal_buffer_allocation_next_id(void);

// Queue allocation lifecycle supplied by a buffer root. This interface owns
// logical pool reservations; it does not acquire ownership of externally
// imported storage. The source pool outlives the buffer and all transactions.
// Queue resource sets retain roots through completion, and semaphore edges
// order allocation, use, and decommit independently of submission order.
struct iree_hal_buffer_allocation_vtable_t {
  // Returns the immutable identity captured before the root is published.
  iree_hal_buffer_allocation_profile_t(IREE_API_PTR* profile)(
      iree_hal_buffer_t* buffer);
  // Captures the allocation epoch, even before a reservation is attached.
  // Returns the borrowed source pool. Duplicate capture fails without changing
  // ownership. Success must be paired with abort or reservation transfer.
  iree_status_t(IREE_API_PTR* begin_dealloca)(iree_hal_buffer_t* buffer,
                                              iree_hal_pool_t** out_pool);
  // Rolls back a capture that has not transferred its reservation or published
  // a completion action. The epoch can then be captured again.
  void(IREE_API_PTR* abort_dealloca)(iree_hal_buffer_t* buffer);
  // Transfers the captured reservation to the caller after allocation has
  // attached it. The source pool remains borrowed. This precedes decommit;
  // release may carry a pending death frontier that orders future reuse.
  void(IREE_API_PTR* take_dealloca_reservation)(
      iree_hal_buffer_t* buffer, iree_hal_pool_t** out_pool,
      iree_hal_pool_reservation_t* out_reservation);
  // Releases staged/committed backing after all accesses complete, before
  // publishing deallocation completion. Safe when no backing was attached.
  void(IREE_API_PTR* decommit)(iree_hal_buffer_t* buffer);
};

// Returns the root's static allocation interface, or NULL for ordinary buffers.
// The public queue boundary validates root identity, this capability, and
// family grants once. The operations below consume that validated root
// directly.
static inline const iree_hal_buffer_allocation_vtable_t*
iree_hal_buffer_allocation_vtable(const iree_hal_buffer_t* buffer) {
  return ((const iree_hal_buffer_vtable_t*)buffer->resource.vtable)->allocation;
}

// Returns the stable allocation identity and origin capture session.
IREE_API_EXPORT iree_hal_buffer_allocation_profile_t
iree_hal_buffer_allocation_profile(iree_hal_buffer_t* buffer);

// Captures the epoch and returns its borrowed pool; may precede commitment.
IREE_API_EXPORT iree_status_t iree_hal_buffer_allocation_begin_dealloca(
    iree_hal_buffer_t* buffer, iree_hal_pool_t** out_pool);

// Restores an epoch whose deallocation capture was not submitted.
IREE_API_EXPORT void iree_hal_buffer_allocation_abort_dealloca(
    iree_hal_buffer_t* buffer);

// Transfers one captured reservation after allocation and before decommit.
IREE_API_EXPORT void iree_hal_buffer_allocation_take_dealloca_reservation(
    iree_hal_buffer_t* buffer, iree_hal_pool_t** out_pool,
    iree_hal_pool_reservation_t* out_reservation);

// Releases backing at an ordered completion or allocation rollback boundary.
IREE_API_EXPORT void iree_hal_buffer_allocation_decommit(
    iree_hal_buffer_t* buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_BUFFER_ALLOCATION_H_
