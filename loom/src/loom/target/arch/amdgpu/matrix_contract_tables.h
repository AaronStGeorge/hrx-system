// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU matrix contract descriptor tables.

#ifndef LOOM_TARGET_AMDGPU_MATRIX_CONTRACT_TABLES_H_
#define LOOM_TARGET_AMDGPU_MATRIX_CONTRACT_TABLES_H_

#include "loom/target/arch/amdgpu/matrix_contract_types.h"

#ifdef __cplusplus
extern "C" {
#endif

extern const loom_amdgpu_matrix_contract_descriptor_t
    kLoomAmdgpuMatrixContractDescriptors[];
extern const iree_host_size_t kLoomAmdgpuMatrixContractDescriptorCount;

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_AMDGPU_MATRIX_CONTRACT_TABLES_H_
