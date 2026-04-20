// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU HAL resource materialization for target-low kernels.
//
// low.abi.resource records describe the HAL-facing binding ABI and low.resource
// imports the target-low resource value. This layer lowers each low.resource
// into the AMDGPU packet-level sequence that loads the raw binding pointer from
// kernargs and constructs the four-SGPR buffer resource consumed by MUBUF ops.

#ifndef LOOM_TARGET_ARCH_AMDGPU_HAL_RESOURCE_MATERIALIZATION_H_
#define LOOM_TARGET_ARCH_AMDGPU_HAL_RESOURCE_MATERIALIZATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_hal_resource_materialization_result_t {
  // Number of low.resource ops expanded into packet-level low IR.
  iree_host_size_t materialized_resource_count;
  // True when the pass inserted the kernarg segment pointer live-in.
  bool inserted_kernarg_segment_ptr_live_in;
} loom_amdgpu_hal_resource_materialization_result_t;

// Expands AMDGPU HAL low.resource imports in |function_op|.
//
// The expansion inserts or reuses a
// low.live_in<amdgpu.kernarg_segment_ptr> value, loads one 64-bit binding
// pointer from the kernarg slot assigned by low.abi.resource, materializes the
// range and flags words, and replaces low.resource with a low.concat producing
// reg<amdgpu.sgpr x4>.
iree_status_t loom_amdgpu_hal_resource_materialize(
    loom_module_t* module, loom_op_t* function_op,
    const loom_target_bundle_t* target_bundle,
    loom_amdgpu_hal_resource_materialization_result_t* out_result,
    iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_HAL_RESOURCE_MATERIALIZATION_H_
