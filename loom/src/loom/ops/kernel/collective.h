// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_OPS_KERNEL_COLLECTIVE_H_
#define LOOM_OPS_KERNEL_COLLECTIVE_H_

#include "loom/ir/facts.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the execution scope containing every participant of a kernel
// collective. NONE means this operation has no scoped collective contract;
// it does not relax an independently declared Convergent trait. Memory space
// and memory ordering are independent of the execution scope.
loom_value_fact_uniform_scope_t loom_kernel_collective_scope(
    const loom_op_t* op);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_KERNEL_COLLECTIVE_H_
