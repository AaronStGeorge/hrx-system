// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/coverage.h"

#define LOOM_AMDGPU_NATIVE_LOW_PHASES                                          \
  (LOOM_TARGET_COVERAGE_PHASE_DESCRIPTOR |                                     \
   LOOM_TARGET_COVERAGE_PHASE_LOW_PARSE |                                      \
   LOOM_TARGET_COVERAGE_PHASE_SOURCE_LOWER |                                   \
   LOOM_TARGET_COVERAGE_PHASE_SCHEDULE | LOOM_TARGET_COVERAGE_PHASE_ALLOCATE | \
   LOOM_TARGET_COVERAGE_PHASE_TEXT_EMIT |                                      \
   LOOM_TARGET_COVERAGE_PHASE_BINARY_ENCODE |                                  \
   LOOM_TARGET_COVERAGE_PHASE_ARTIFACT_WRAP)

#define LOOM_AMDGPU_EXECUTABLE_LOW_PHASES                               \
  (LOOM_AMDGPU_NATIVE_LOW_PHASES | LOOM_TARGET_COVERAGE_PHASE_INSPECT | \
   LOOM_TARGET_COVERAGE_PHASE_RUN)

#define LOOM_AMDGPU_MATRIX_DESCRIPTOR_PHASES                                   \
  (LOOM_TARGET_COVERAGE_PHASE_DESCRIPTOR |                                     \
   LOOM_TARGET_COVERAGE_PHASE_LOW_PARSE |                                      \
   LOOM_TARGET_COVERAGE_PHASE_SCHEDULE | LOOM_TARGET_COVERAGE_PHASE_ALLOCATE | \
   LOOM_TARGET_COVERAGE_PHASE_TEXT_EMIT)

static const loom_target_coverage_row_t kLoomAmdgpuTargetCoverageRows[] = {
    {
        .target_key = IREE_SVL("amdgpu-gfx11"),
        .descriptor_set_key = IREE_SVL("amdgpu.gfx11.core"),
        .category = IREE_SVL("vector.arithmetic"),
        .semantic_tag = IREE_SVL("i32-f32-vgpr"),
        .expected_phases = LOOM_AMDGPU_EXECUTABLE_LOW_PHASES,
        .supported_phases = LOOM_AMDGPU_NATIVE_LOW_PHASES,
        .gap_key = IREE_SVL("validation-run-provider-optional"),
    },
    {
        .target_key = IREE_SVL("amdgpu-gfx11"),
        .descriptor_set_key = IREE_SVL("amdgpu.gfx11.core"),
        .category = IREE_SVL("memory.global"),
        .semantic_tag = IREE_SVL("buffer-b128-load-store"),
        .expected_phases = LOOM_AMDGPU_EXECUTABLE_LOW_PHASES,
        .supported_phases = LOOM_AMDGPU_NATIVE_LOW_PHASES,
        .gap_key = IREE_SVL("validation-run-provider-optional"),
    },
    {
        .target_key = IREE_SVL("amdgpu-gfx11"),
        .descriptor_set_key = IREE_SVL("amdgpu.gfx11.core"),
        .category = IREE_SVL("matrix.contract"),
        .semantic_tag = IREE_SVL("wmma-mfma"),
        .expected_phases = LOOM_AMDGPU_NATIVE_LOW_PHASES,
        .supported_phases = LOOM_AMDGPU_MATRIX_DESCRIPTOR_PHASES,
        .gap_key = IREE_SVL("source-lower-matrix-encoding-missing"),
    },
    {
        .target_key = IREE_SVL("amdgpu-gfx12"),
        .descriptor_set_key = IREE_SVL("amdgpu.gfx12.core"),
        .category = IREE_SVL("matrix.contract"),
        .semantic_tag = IREE_SVL("swmmac"),
        .expected_phases = LOOM_AMDGPU_NATIVE_LOW_PHASES,
        .supported_phases = LOOM_AMDGPU_MATRIX_DESCRIPTOR_PHASES,
        .gap_key = IREE_SVL("source-lower-matrix-encoding-missing"),
    },
    {
        .target_key = IREE_SVL("amdgpu"),
        .descriptor_set_key = IREE_SVL("amdgpu.gfx11.core"),
        .category = IREE_SVL("artifact"),
        .semantic_tag = IREE_SVL("hsaco"),
        .expected_phases = LOOM_TARGET_COVERAGE_PHASE_ARTIFACT_WRAP |
                           LOOM_TARGET_COVERAGE_PHASE_INSPECT |
                           LOOM_TARGET_COVERAGE_PHASE_RUN,
        .supported_phases = LOOM_TARGET_COVERAGE_PHASE_ARTIFACT_WRAP,
        .gap_key = IREE_SVL("validation-run-provider-optional"),
    },
};

const loom_target_coverage_provider_t loom_amdgpu_target_coverage_provider = {
    .name = IREE_SVL("amdgpu"),
    .rows = kLoomAmdgpuTargetCoverageRows,
    .row_count = IREE_ARRAYSIZE(kLoomAmdgpuTargetCoverageRows),
};
