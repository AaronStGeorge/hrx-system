// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_INTERNAL_CPU_X86_64_H_
#define IREE_BASE_INTERNAL_CPU_X86_64_H_

#include "iree/base/cpu_data.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Register outputs from one CPUID leaf and subleaf.
typedef struct iree_cpu_x86_64_registers_t {
  // EAX output.
  uint32_t eax;
  // EBX output.
  uint32_t ebx;
  // ECX output.
  uint32_t ecx;
  // EDX output.
  uint32_t edx;
} iree_cpu_x86_64_registers_t;

// Hardware and OS facts used to admit executable x86 features. Unsupported
// CPUID leaves are zero. Hardware-supported XSTATE components alone never
// establish that a feature can be used by the process.
typedef struct iree_cpu_x86_64_capabilities_t {
  // Maximum basic leaf and processor vendor ID from CPUID(0, 0).
  iree_cpu_x86_64_registers_t leaf0;
  // Basic instruction features from CPUID(1, 0).
  iree_cpu_x86_64_registers_t leaf1;
  // Structured instruction features from CPUID(7, 0).
  iree_cpu_x86_64_registers_t leaf7_0;
  // Additional structured instruction features from CPUID(7, 1).
  iree_cpu_x86_64_registers_t leaf7_1;
  // Extended instruction features from CPUID(0x80000001, 0).
  iree_cpu_x86_64_registers_t extended_leaf1;
  // XCR0-enabled components, including OS-guaranteed lazy state support.
  uint64_t enabled_xstate;
  // Process permissions for dynamically enabled XSTATE components. Zero when
  // the platform cannot establish process-wide permission.
  uint64_t permitted_xstate;
} iree_cpu_x86_64_capabilities_t;

// Converts instruction capabilities and OS state into IREE_CPU_DATA0_X86_64_*
// feature bits. This does not query or modify hardware or OS state.
uint64_t iree_cpu_x86_64_decode_features(
    const iree_cpu_x86_64_capabilities_t* capabilities);

// Converts architectural processor identity and usable instruction features
// into stable x86-64 CPU data fields. This does not query hardware or OS state.
void iree_cpu_x86_64_decode_data(
    const iree_cpu_x86_64_capabilities_t* capabilities,
    uint64_t out_fields[IREE_CPU_DATA_FIELD_COUNT]);

#if defined(IREE_ARCH_X86_64)
// Queries exact processor identity and features usable by this process without
// enabling optional state or requesting permission. Optional state permissions
// must be established before querying; their absence leaves the corresponding
// feature bits unset.
void iree_cpu_x86_64_query_data(uint64_t out_fields[IREE_CPU_DATA_FIELD_COUNT]);
#endif  // IREE_ARCH_X86_64

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_INTERNAL_CPU_X86_64_H_
