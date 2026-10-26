// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_OPS_BFP_OPERAND_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_OPS_BFP_OPERAND_H_

#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns true when |type| is an opaque native BFP16EBS8 matrix operand.
bool loom_aie2p_bfp16ebs8_operand_type_isa(const loom_module_t* module,
                                           loom_type_t type);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_OPS_BFP_OPERAND_H_
