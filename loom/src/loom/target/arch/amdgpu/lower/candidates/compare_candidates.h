// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU source-to-low compare descriptor candidate rows.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_CANDIDATES_COMPARE_CANDIDATES_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_CANDIDATES_COMPARE_CANDIDATES_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_compare_descriptor_candidate_t {
  // Source compare op kind handled by this row.
  loom_op_kind_t op_kind;
  // Source compare predicate value matched by this row.
  uint8_t predicate;
  // Dense AMDGPU descriptor ref selected when present in the descriptor set.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
  // Dense AMDGPU descriptor ref selected when the left-hand source is inline.
  loom_amdgpu_descriptor_ref_t src0_inline_descriptor_ref;
  // Dense AMDGPU descriptor ref selected when the right-hand source is inline.
  loom_amdgpu_descriptor_ref_t src1_inline_descriptor_ref;
} loom_amdgpu_compare_descriptor_candidate_t;

extern const loom_amdgpu_compare_descriptor_candidate_t
    kLoomAmdgpuCompareDescriptorCandidates[];
extern const iree_host_size_t kLoomAmdgpuCompareDescriptorCandidateCount;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_CANDIDATES_COMPARE_CANDIDATES_H_
