// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-neutral bundle rows for authored LLVMIR projection records.

#ifndef LOOM_OPS_LLVMIR_TARGET_RECORDS_H_
#define LOOM_OPS_LLVMIR_TARGET_RECORDS_H_

#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selector-indexed target bundle rows used by llvmir.target records. The rows
// contain only common target facts; LLVM triple, data layout, CPU, and feature
// strings stay on the llvmir.target op itself.
extern const loom_target_bundle_table_t loom_llvmir_projection_target_bundles;

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_LLVMIR_TARGET_RECORDS_H_
