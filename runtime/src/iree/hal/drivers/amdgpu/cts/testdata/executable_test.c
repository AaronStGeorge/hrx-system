// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/abi/asan.h"
#include "iree/hal/drivers/amdgpu/abi/feedback.h"
#include "iree/hal/drivers/amdgpu/abi/tsan.h"
#include "iree/hal/drivers/amdgpu/device/support/asan.h"
#include "iree/hal/drivers/amdgpu/device/support/kernel.h"

[[gnu::visibility("protected"), gnu::used]]
volatile uint64_t executable_test_global = 0x0123456789ABCDEFull;

[[gnu::visibility("protected"), gnu::used]]
volatile iree_hal_amdgpu_asan_config_t iree_asan_config = {0};

[[gnu::visibility("protected"), gnu::used]]
volatile iree_hal_amdgpu_feedback_config_t iree_feedback_config = {0};

[[gnu::visibility("protected"), gnu::used]]
volatile iree_hal_amdgpu_tsan_config_t iree_tsan_config = {0};

IREE_AMDGPU_ATTRIBUTE_KERNEL void export0(uint64_t* lhs, uint64_t* rhs,
                                          uint32_t c0, uint32_t c1) {
  if (c0 == c1) {
    lhs[0] = executable_test_global;
  } else if (c0 == 0x4153414Eu && c1 == 0x43464721u) {
    lhs[0] = iree_asan_config.record_length;
    lhs[1] = iree_asan_config.flags;
    lhs[2] = iree_asan_config.shadow_base;
    lhs[3] = iree_asan_config.shadow_size;
    lhs[4] = iree_asan_config.shadow_slab_size;
  } else if (c0 == 0x4644424Bu && c1 == 0x43464721u) {
    lhs[0] = iree_feedback_config.record_length;
    lhs[1] = iree_feedback_config.flags;
    lhs[2] = iree_feedback_config.channel_base;
    lhs[3] = iree_feedback_config.notify_signal.handle;
    lhs[4] = iree_feedback_config.source_context;
  } else if (c0 == 0x5453414Eu && c1 == 0x43464721u) {
    lhs[0] = iree_tsan_config.record_length;
    lhs[1] = iree_tsan_config.flags;
    lhs[2] = iree_tsan_config.shadow_base;
    lhs[3] = iree_tsan_config.shadow_size;
    lhs[4] = iree_tsan_config.workgroup_shadow_stride;
    lhs[5] = iree_tsan_config.workgroup_capacity;
    lhs[6] = iree_tsan_config.shadow_entry_size;
    lhs[7] = iree_tsan_config.memory_granule_shift;
  } else if (c0 == 0x4153414Eu && c1 == 0x52505421u) {
    lhs[0] = iree_hal_amdgpu_asan_report_access(
                 &iree_feedback_config, IREE_HAL_AMDGPU_ASAN_ACCESS_KIND_WRITE,
                 0x123456789ABCDEFull, 16, 0xC0DEFACEu, 0x56789ABCDEFull, 0xF0u)
                 ? 1
                 : 0;
  } else {
    lhs[0] = rhs[0];
  }
}
