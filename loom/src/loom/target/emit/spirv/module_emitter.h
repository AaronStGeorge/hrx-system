// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V low-function binary emission.
//
// This stage consumes already-selected target-low IR. It does not perform
// source legality, source-to-low lowering, target specialization, scheduling,
// or allocation. The mutable module argument is used only for the
// function-local value-domain scratch map.

#ifndef LOOM_TARGET_EMIT_SPIRV_MODULE_EMITTER_H_
#define LOOM_TARGET_EMIT_SPIRV_MODULE_EMITTER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/target/emit/spirv/module_builder.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Emits one target-low function body as a standalone SPIR-V module.
//
// |low_function_op| must be a low.func.def or low.kernel.def with a
// target-bound body. The output module owns allocator-backed word storage and
// must be deinitialized by the caller.
iree_status_t loom_spirv_emit_low_function_module(
    loom_module_t* module, loom_op_t* low_function_op,
    const loom_low_descriptor_registry_t* descriptor_registry,
    loom_target_selection_t target_selection,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_allocator_t* scratch_arena,
    loom_spirv_module_binary_t* out_module, iree_allocator_t allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_SPIRV_MODULE_EMITTER_H_
