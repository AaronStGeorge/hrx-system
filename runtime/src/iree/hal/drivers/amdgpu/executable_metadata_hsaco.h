// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_EXECUTABLE_METADATA_HSACO_H_
#define IREE_HAL_DRIVERS_AMDGPU_EXECUTABLE_METADATA_HSACO_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/executable_metadata.h"
#include "iree/hal/drivers/amdgpu/util/hsaco_metadata.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Calculates common executable metadata storage counts for |hsaco_metadata|.
iree_status_t iree_hal_amdgpu_executable_metadata_calculate_hsaco_counts(
    const iree_hal_amdgpu_hsaco_metadata_t* hsaco_metadata,
    iree_hal_amdgpu_executable_metadata_counts_t* out_counts);

// Populates |metadata| from decoded HSACO metadata.
//
// |metadata| must have been allocated with counts returned from
// iree_hal_amdgpu_executable_metadata_calculate_hsaco_counts. String views
// populated into |metadata| borrow from |hsaco_metadata| and are valid for as
// long as |code_object_data| remains alive.
iree_status_t iree_hal_amdgpu_executable_metadata_populate_from_hsaco(
    const iree_hal_amdgpu_hsaco_metadata_t* hsaco_metadata,
    iree_const_byte_span_t code_object_data,
    iree_hal_amdgpu_executable_metadata_t* metadata);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_EXECUTABLE_METADATA_HSACO_H_
