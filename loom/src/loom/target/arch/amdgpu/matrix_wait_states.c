// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/matrix_wait_states.h"

#define LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(profile_, use_, pass_, cycles_) \
  {                                                                        \
      .profile = profile_,                                                 \
      .use = use_,                                                         \
      .pass_count = pass_,                                                 \
      .cycle_count = cycles_,                                              \
  }

static const loom_amdgpu_matrix_wait_result_row_t
    kAmdgpuMatrixWaitResultRows[] = {
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_NON_MATRIX, 2, 5),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_NON_MATRIX, 4, 7),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_NON_MATRIX, 8, 11),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_NON_MATRIX, 16, 19),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_EXACT, 2, 2),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_EXACT, 4, 0),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_EXACT, 8, 0),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_EXACT, 16, 0),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_OVERLAP, 2, 3),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_OVERLAP, 4, 5),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_OVERLAP, 8, 9),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_OVERLAP, 16, 17),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRC_AB, 2, 5),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRC_AB, 4, 7),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRC_AB, 8, 11),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRC_AB, 16, 19),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_NON_MATRIX, 2, 5),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_NON_MATRIX, 4, 8),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_NON_MATRIX, 8, 12),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_NON_MATRIX, 16, 20),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_EXACT, 2, 2),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_EXACT, 4, 0),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_EXACT, 8, 0),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_EXACT, 16, 0),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_OVERLAP, 2, 3),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_OVERLAP, 4, 5),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_OVERLAP, 8, 9),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_OVERLAP, 16, 17),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRC_AB, 2, 5),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRC_AB, 4, 8),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRC_AB, 8, 12),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRC_AB, 16, 20),
};

#undef LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW

bool loom_amdgpu_matrix_wait_profile_from_feature_profile(
    loom_amdgpu_matrix_feature_profile_t feature_profile,
    loom_amdgpu_matrix_wait_profile_t* out_profile) {
  *out_profile = LOOM_AMDGPU_MATRIX_WAIT_PROFILE_UNKNOWN;
  switch (feature_profile) {
    case LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX908:
    case LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX90A:
    case LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940:
      *out_profile = LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950;
      return true;
    case LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950:
      *out_profile = LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950;
      return true;
    case LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_NONE:
    case LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11:
    case LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12:
    case LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250:
    default:
      return false;
  }
}

iree_host_size_t loom_amdgpu_matrix_wait_result_row_count(void) {
  return IREE_ARRAYSIZE(kAmdgpuMatrixWaitResultRows);
}

const loom_amdgpu_matrix_wait_result_row_t*
loom_amdgpu_matrix_wait_result_row_at(iree_host_size_t index) {
  if (index >= IREE_ARRAYSIZE(kAmdgpuMatrixWaitResultRows)) {
    return NULL;
  }
  return &kAmdgpuMatrixWaitResultRows[index];
}

const loom_amdgpu_matrix_wait_result_row_t* loom_amdgpu_matrix_wait_result_find(
    loom_amdgpu_matrix_wait_profile_t profile, uint16_t pass_count,
    loom_amdgpu_matrix_wait_result_use_t use) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kAmdgpuMatrixWaitResultRows);
       ++i) {
    const loom_amdgpu_matrix_wait_result_row_t* row =
        &kAmdgpuMatrixWaitResultRows[i];
    if (row->profile == profile && row->pass_count == pass_count &&
        row->use == use) {
      return row;
    }
  }
  return NULL;
}

bool loom_amdgpu_matrix_wait_result_cycle_count(
    loom_amdgpu_matrix_wait_profile_t profile, uint16_t pass_count,
    loom_amdgpu_matrix_wait_result_use_t use, uint16_t* out_cycle_count) {
  *out_cycle_count = 0;
  const loom_amdgpu_matrix_wait_result_row_t* row =
      loom_amdgpu_matrix_wait_result_find(profile, pass_count, use);
  if (row == NULL) {
    return false;
  }
  *out_cycle_count = row->cycle_count;
  return true;
}
