// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AIE2P F32 packet physical-representation policy.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_F32_REPRESENTATION_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_F32_REPRESENTATION_H_

#include "loom/codegen/low/lower/representation_observer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_aie2p_f32_representation_e {
  // Sixteen F32 lanes carried by two 256-bit vector registers.
  LOOM_AIE2P_F32_REPRESENTATION_VECTOR = 1,
  // Sixteen F32 lanes carried by one low accumulator quarter.
  LOOM_AIE2P_F32_REPRESENTATION_ACCUMULATOR = 2,
} loom_aie2p_f32_representation_t;

// Returns true when |source_type| is one native sixteen-lane F32 packet.
bool loom_aie2p_f32_packet_type(loom_type_t source_type);

// Function-local F32 packet representation selection.
extern const loom_low_lower_source_plan_observer_t
    loom_aie2p_f32_representation_observer;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_F32_REPRESENTATION_H_
