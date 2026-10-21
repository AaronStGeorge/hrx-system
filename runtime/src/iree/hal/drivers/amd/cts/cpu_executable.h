// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_CTS_CPU_EXECUTABLE_H_
#define IREE_HAL_DRIVERS_AMD_CTS_CPU_EXECUTABLE_H_

#include "iree/hal/api.h"

namespace iree::hal::amd::cts {

// Creates a Task executable that adds its uint32 constant to one element per
// workgroup in its single binding. The caller owns the returned reference.
iree_hal_executable_t* CreateCpuAddExecutable(
    const iree_hal_queue_family_t* family);

}  // namespace iree::hal::amd::cts

#endif  // IREE_HAL_DRIVERS_AMD_CTS_CPU_EXECUTABLE_H_
