// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Bridges Loom vector dot ops to CPU packed-dot target contracts.

#ifndef LOOM_TARGET_CPU_PACKED_DOT_VECTOR_H_
#define LOOM_TARGET_CPU_PACKED_DOT_VECTOR_H_

#include "loom/ir/module.h"
#include "loom/target/cpu/packed_dot_contract.h"

#ifdef __cplusplus
extern "C" {
#endif

// Infers the target-independent packed-dot match request represented by |op|.
//
// The helper accepts vector.dot2f, vector.dot4i, and vector.dot8i4 when their
// register shapes are fully static and representable by the CPU descriptor
// contract. Target feature bits and optional family restrictions remain caller
// policy; the inferred request leaves those fields clear for the selected CPU
// lowering profile to populate before calling loom_cpu_packed_dot_select.
//
// Returns false for non-packed-dot ops, dynamic shapes, malformed IR, and
// vector dot forms that have no CPU packed-dot descriptor payload vocabulary.
bool loom_cpu_packed_dot_match_request_from_vector_op(
    const loom_module_t* module, const loom_op_t* op,
    loom_cpu_packed_dot_match_request_t* out_request);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_CPU_PACKED_DOT_VECTOR_H_
