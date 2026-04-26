// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared AMDGPU lowering vocabulary used by plans and generated tables.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_KINDS_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_KINDS_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_amdgpu_memory_address_form_e {
  LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT = 0,
  LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO = 1,
  LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_2ADDR = 2,
  LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR = 3,
  LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID = 4,
  LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT = 5,
} loom_amdgpu_memory_address_form_t;

typedef enum loom_amdgpu_atomic_operation_kind_e {
  LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE = 0,
  LOOM_AMDGPU_ATOMIC_OPERATION_RMW = 1,
  LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG = 2,
} loom_amdgpu_atomic_operation_kind_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_KINDS_H_
