// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent cleanup for descriptor-backed low functions.
//
// Cleanup is the mutating boundary between low lowering and backend sidecar
// construction. It removes dead structural value ops and descriptor packets
// whose target descriptor explicitly marks them dead-removable, so scheduling
// and allocation only observe packets that can affect the emitted program.

#ifndef LOOM_CODEGEN_LOW_CLEANUP_H_
#define LOOM_CODEGEN_LOW_CLEANUP_H_

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Removes dead-removable target-low ops from one low.func.def.
//
// |descriptor_registry| selects the descriptor set named by the low function's
// target profile. Descriptor-backed low.op/low.const packets are removable only
// when their descriptor carries LOOM_LOW_DESCRIPTOR_FLAG_DEAD_REMOVABLE and all
// SSA results are unused. The pass arena is transient and comes from the
// module's arena block pool.
iree_status_t loom_low_cleanup_function(
    loom_module_t* module, loom_op_t* low_func_op,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t emitter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_CLEANUP_H_
