// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_DEVICE_SPEC_BUILDER_H_
#define IREE_HAL_DRIVERS_VULKAN_DEVICE_SPEC_BUILDER_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/vulkan/device_spec.h"
#include "iree/hal/utils/device_spec_builder.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Encodes |spec| and adds it as a Vulkan driver-local facet to |builder|.
IREE_API_EXPORT iree_status_t iree_hal_vulkan_device_spec_builder_add_facet(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_vulkan_device_spec_t* spec);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_DEVICE_SPEC_BUILDER_H_
