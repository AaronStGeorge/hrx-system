// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU source-to-low atomic descriptor candidate rows.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_CANDIDATES_ATOMIC_CANDIDATES_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_CANDIDATES_ATOMIC_CANDIDATES_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/target/arch/amdgpu/lower/kinds.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_AMDGPU_ATOMIC_KIND_NONE UINT8_MAX

typedef enum loom_amdgpu_atomic_value_kind_e {
  LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32 = 0,
  LOOM_AMDGPU_ATOMIC_VALUE_KIND_F32 = 1,
  LOOM_AMDGPU_ATOMIC_VALUE_KIND_PACKED_F16 = 2,
  LOOM_AMDGPU_ATOMIC_VALUE_KIND_PACKED_BF16 = 3,
} loom_amdgpu_atomic_value_kind_t;

typedef struct loom_amdgpu_atomic_descriptor_candidate_t {
  // Source memory space matched by this row.
  loom_value_fact_memory_space_t memory_space;
  // Target addressing form emitted by this row.
  loom_amdgpu_memory_address_form_t address_form;
  // Source atomic operation form matched by this row.
  loom_amdgpu_atomic_operation_kind_t operation_kind;
  // Source atomic arithmetic kind matched by this row.
  uint8_t atomic_kind;
  // Source scalar value type required by this row.
  loom_amdgpu_atomic_value_kind_t value_kind;
  // Dense AMDGPU descriptor ref selected when present in the descriptor set.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
} loom_amdgpu_atomic_descriptor_candidate_t;

extern const loom_amdgpu_atomic_descriptor_candidate_t
    kLoomAmdgpuAtomicDescriptorCandidates[];
extern const iree_host_size_t kLoomAmdgpuAtomicDescriptorCandidateCount;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_CANDIDATES_ATOMIC_CANDIDATES_H_
