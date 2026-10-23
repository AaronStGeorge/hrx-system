// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_CTS_TESTDATA_H_
#define IREE_HAL_DRIVERS_AMD_GPU_CTS_TESTDATA_H_

#include <vector>

#include "amdf/gpu.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/hal/drivers/amd/gpu/target/identity.h"

namespace iree::hal::amd::gpu::cts {

// Selects a compatible compiled target, preferring exact over generic code.
// Returns a borrowed entry in |targets|, or nullptr when none is compatible.
const iree::hal::cts::ExecutableTarget* SelectExecutableTarget(
    const std::vector<iree::hal::cts::ExecutableTarget>& targets,
    const iree_hal_amd_gpu_target_identity_t& device_identity);

// Resolves the native physical identity and looks up a logical program in a
// linked CTS collection. The returned bytes borrow immutable embedded storage.
// A missing compatible image or program is a configuration error, not a skip.
iree_status_t LookupExecutableData(
    const char* collection, const amdf_gpu_endpoint_info_t& endpoint_info,
    iree_string_view_t file_name, iree_const_byte_span_t* out_data);

}  // namespace iree::hal::amd::gpu::cts

#endif  // IREE_HAL_DRIVERS_AMD_GPU_CTS_TESTDATA_H_
