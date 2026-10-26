// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_CONFIGURATION_ENTRY_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_CONFIGURATION_ENTRY_H_

#include "loom/ir/ir.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Realizes a typed whole-array configuration kernel using the dense external
// binding table declared by its configuration program.
iree_status_t loom_aie2p_configuration_entry_realize(loom_pass_t* pass,
                                                     loom_module_t* module,
                                                     loom_func_like_t function);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_CONFIGURATION_ENTRY_H_
