// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU native assembly-fragment emission from target-low packet sidecars.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_ASSEMBLY_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_ASSEMBLY_H_

#include "iree/base/api.h"
#include "iree/base/string_builder.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/schedule.h"

#ifdef __cplusplus
extern "C" {
#endif

// Emits an AMDGPU assembly fragment for one scheduled and allocated AMDGPU
// low.func.def. The fragment assumes exact physical-register inputs and
// outputs; it does not emit kernel metadata, PAL metadata, or an ELF code
// object envelope. Values must be physically allocated and unspilled.
iree_status_t loom_amdgpu_emit_assembly_fragment(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_allocation_sidecar_t* allocation,
    iree_string_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_ASSEMBLY_H_
