// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/abi/asan.h"
#include "iree/hal/drivers/amdgpu/abi/feedback.h"
#include "iree/hal/drivers/amdgpu/device/support/asan.h"

extern volatile iree_hal_amdgpu_asan_config_t iree_asan_config;
extern volatile iree_hal_amdgpu_feedback_config_t iree_feedback_config;

void __asan_store4(uint64_t address) {
  uint64_t shadow_address =
      iree_asan_config.shadow_base +
      ((address - iree_asan_config.application_window_base) >>
       iree_asan_config.shadow_scale_shift);
  (void)iree_hal_amdgpu_asan_report_access(
      &iree_feedback_config, IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_WRITE, address, 4,
      0x4153414E53544F34ull, shadow_address, 0);
}
