// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/coverage.h"

#define LOOM_X86_TEXT_LOW_PHASES                                               \
  (LOOM_TARGET_COVERAGE_PHASE_DESCRIPTOR |                                     \
   LOOM_TARGET_COVERAGE_PHASE_LOW_PARSE |                                      \
   LOOM_TARGET_COVERAGE_PHASE_SOURCE_LOWER |                                   \
   LOOM_TARGET_COVERAGE_PHASE_SCHEDULE | LOOM_TARGET_COVERAGE_PHASE_ALLOCATE | \
   LOOM_TARGET_COVERAGE_PHASE_TEXT_EMIT)

#define LOOM_X86_EXECUTABLE_LOW_PHASES                                   \
  (LOOM_X86_TEXT_LOW_PHASES | LOOM_TARGET_COVERAGE_PHASE_BINARY_ENCODE | \
   LOOM_TARGET_COVERAGE_PHASE_ARTIFACT_WRAP |                            \
   LOOM_TARGET_COVERAGE_PHASE_INSPECT | LOOM_TARGET_COVERAGE_PHASE_RUN)

static const loom_target_coverage_row_t kLoomX86TargetCoverageRows[] = {
    {
        .target_key = IREE_SVL("x86-avx512"),
        .descriptor_set_key = IREE_SVL("x86.avx512.core"),
        .category = IREE_SVL("vector.arithmetic"),
        .semantic_tag = IREE_SVL("avx512.i32-f32"),
        .expected_phases = LOOM_X86_EXECUTABLE_LOW_PHASES,
        .supported_phases = LOOM_X86_TEXT_LOW_PHASES,
        .gap_key = IREE_SVL("cpu-object-abi-missing"),
    },
    {
        .target_key = IREE_SVL("x86-packed-dot"),
        .descriptor_set_key = IREE_SVL("x86.packed_dot.core"),
        .category = IREE_SVL("vector.dot"),
        .semantic_tag = IREE_SVL("vnni-bf16-f16"),
        .expected_phases = LOOM_X86_EXECUTABLE_LOW_PHASES,
        .supported_phases = LOOM_X86_TEXT_LOW_PHASES,
        .gap_key = IREE_SVL("cpu-object-abi-missing"),
    },
    {
        .target_key = IREE_SVL("x86"),
        .descriptor_set_key = IREE_SVL("x86.avx512.core"),
        .category = IREE_SVL("artifact"),
        .semantic_tag = IREE_SVL("elf-object"),
        .expected_phases = LOOM_TARGET_COVERAGE_PHASE_BINARY_ENCODE |
                           LOOM_TARGET_COVERAGE_PHASE_ARTIFACT_WRAP |
                           LOOM_TARGET_COVERAGE_PHASE_INSPECT |
                           LOOM_TARGET_COVERAGE_PHASE_RUN,
        .supported_phases = LOOM_TARGET_COVERAGE_PHASE_NONE,
        .gap_key = IREE_SVL("cpu-object-abi-missing"),
    },
};

const loom_target_coverage_provider_t loom_x86_target_coverage_provider = {
    .name = IREE_SVL("x86"),
    .rows = kLoomX86TargetCoverageRows,
    .row_count = IREE_ARRAYSIZE(kLoomX86TargetCoverageRows),
};
