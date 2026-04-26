// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-neutral launch-shape contracts.
//
// Target snapshots describe capability envelopes supplied by the hosting layer
// or target preset. Function/export contracts may optionally select a concrete
// launch shape. This layer validates those dense records without probing the
// current host device and without inventing fallback launch sizes.

#ifndef LOOM_TARGET_LAUNCH_H_
#define LOOM_TARGET_LAUNCH_H_

#include "iree/base/api.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when no concrete workgroup size has been selected.
bool loom_target_workgroup_size_is_empty(
    const loom_target_workgroup_size_t* size);

// Returns true when all workgroup dimensions are non-zero.
bool loom_target_workgroup_size_is_concrete(
    const loom_target_workgroup_size_t* size);

// Returns true when some but not all workgroup dimensions are non-zero.
bool loom_target_workgroup_size_is_partial(
    const loom_target_workgroup_size_t* size);

// Multiplies workgroup dimensions using 64-bit arithmetic.
uint64_t loom_target_workgroup_size_flat_product(
    const loom_target_workgroup_size_t* size);

// Validates HAL kernel launch facts against the target capability envelope.
//
// An empty required workgroup size is valid and means launch selection remains
// dynamic/unresolved. A concrete required workgroup size must fit any
// per-dimension and flat target limits present in |snapshot|. A partial
// required size is malformed.
iree_status_t loom_target_validate_hal_kernel_launch(
    const loom_target_snapshot_t* snapshot,
    const loom_target_hal_kernel_abi_t* hal_kernel);

// Requires a concrete HAL kernel workgroup size for a consumer that cannot
// represent dynamic launch selection.
iree_status_t loom_target_require_concrete_hal_kernel_launch(
    const loom_target_hal_kernel_abi_t* hal_kernel,
    iree_string_view_t consumer_name);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_LAUNCH_H_
