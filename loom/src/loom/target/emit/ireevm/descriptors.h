// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Hand-authored low descriptor sets for the IREE VM target.
//
// These descriptors are intentionally small. They prove the shared target-low
// table ABI against an immediately useful VM-shaped target before native CPU or
// GPU descriptor importers exist.

#ifndef LOOM_TARGET_EMIT_IREEVM_DESCRIPTORS_H_
#define LOOM_TARGET_EMIT_IREEVM_DESCRIPTORS_H_

#include "loom/codegen/low/descriptors.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the built-in core IREE VM descriptor set.
const loom_low_descriptor_set_t* loom_ireevm_core_descriptor_set(void);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_EMIT_IREEVM_DESCRIPTORS_H_
