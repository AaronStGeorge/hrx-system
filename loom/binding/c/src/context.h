// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_CONTEXT_STORAGE_H_
#define LOOMC_CONTEXT_STORAGE_H_

#include "loom/ir/context.h"
#include "loomc/context.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the Loom context owned by the public context handle.
LOOMC_API_PRIVATE loom_context_t* loomc_context_loom_context(
    loomc_context_t* context);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_CONTEXT_STORAGE_H_
