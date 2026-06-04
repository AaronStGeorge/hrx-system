// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HIP_RCCL_CHANNEL_H_
#define IREE_HAL_DRIVERS_HIP_RCCL_CHANNEL_H_

#include <string.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/hip/api.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"
#include "iree/hal/drivers/hip/rccl_dynamic_symbols.h"
#include "iree/hal/utils/collective_batch.h"
#include "iree/hal/utils/stream_tracing.h"

// Returns true if |id| is all zeros indicating an empty ID.
static inline bool iree_hal_hip_nccl_id_is_empty(
    const iree_hal_hip_nccl_id_t* id) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(id->data); ++i) {
    if (id->data[i] != 0) return false;
  }
  return true;
}

#if IREE_HAL_HIP_ENABLE_RCCL

// Gets a unique ID to bootstrap a new communicator. It calls ncclGetUniqueId
// under the hood.
iree_status_t iree_hal_hip_nccl_get_unique_id(
    const iree_hal_hip_nccl_dynamic_symbols_t* symbols,
    iree_hal_hip_nccl_id_t* out_id);

// Creates a IREE HAL channel using the given NCCL |id|, |rank|, and |count|.
// It calls ncclCommInitRankConfig under the hood.
iree_status_t iree_hal_hip_nccl_channel_create(
    const iree_hal_hip_dynamic_symbols_t* hip_symbols,
    const iree_hal_hip_nccl_dynamic_symbols_t* nccl_symbols,
    const iree_hal_hip_nccl_id_t* id, int rank, int count,
    iree_allocator_t host_allocator, iree_hal_channel_t** out_channel);

// Performs a non-blocking submission of |batch| to |stream|.
// The backing storage of |batch| is dropped immediately but all resources
// referenced will be retained by the parent command buffer for its lifetime.
// Note that operations in the batch may apply to different channels.
iree_status_t iree_hal_hip_nccl_submit_batch(
    const iree_hal_hip_nccl_dynamic_symbols_t* nccl_symbols,
    iree_hal_stream_tracing_context_t* tracing_context,
    iree_hal_stream_tracing_context_event_list_t* tracing_event_list,
    const iree_hal_collective_batch_t* batch, hipStream_t stream);

#else

static inline iree_status_t iree_hal_hip_nccl_get_unique_id(
    const iree_hal_hip_nccl_dynamic_symbols_t* symbols,
    iree_hal_hip_nccl_id_t* out_id) {
  IREE_ASSERT_ARGUMENT(symbols);
  IREE_ASSERT_ARGUMENT(out_id);
  memset(out_id, 0, sizeof(*out_id));
  return iree_hal_hip_nccl_dynamic_symbols_unavailable_status();
}

static inline iree_status_t iree_hal_hip_nccl_channel_create(
    const iree_hal_hip_dynamic_symbols_t* hip_symbols,
    const iree_hal_hip_nccl_dynamic_symbols_t* nccl_symbols,
    const iree_hal_hip_nccl_id_t* id, int rank, int count,
    iree_allocator_t host_allocator, iree_hal_channel_t** out_channel) {
  IREE_ASSERT_ARGUMENT(hip_symbols);
  IREE_ASSERT_ARGUMENT(nccl_symbols);
  IREE_ASSERT_ARGUMENT(id);
  IREE_ASSERT_ARGUMENT(out_channel);
  *out_channel = NULL;
  return iree_hal_hip_nccl_dynamic_symbols_unavailable_status();
}

static inline iree_status_t iree_hal_hip_nccl_submit_batch(
    const iree_hal_hip_nccl_dynamic_symbols_t* nccl_symbols,
    iree_hal_stream_tracing_context_t* tracing_context,
    iree_hal_stream_tracing_context_event_list_t* tracing_event_list,
    const iree_hal_collective_batch_t* batch, hipStream_t stream) {
  IREE_ASSERT_ARGUMENT(nccl_symbols);
  IREE_ASSERT_ARGUMENT(batch);
  return iree_hal_hip_nccl_dynamic_symbols_unavailable_status();
}

#endif  // IREE_HAL_HIP_ENABLE_RCCL

#endif  // IREE_HAL_DRIVERS_HIP_RCCL_CHANNEL_H_
