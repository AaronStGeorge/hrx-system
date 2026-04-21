// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/wasm/coverage.h"

#define LOOM_WASM_MODULE_LOW_PHASES                                            \
  (LOOM_TARGET_COVERAGE_PHASE_DESCRIPTOR |                                     \
   LOOM_TARGET_COVERAGE_PHASE_LOW_PARSE |                                      \
   LOOM_TARGET_COVERAGE_PHASE_SOURCE_LOWER |                                   \
   LOOM_TARGET_COVERAGE_PHASE_SCHEDULE | LOOM_TARGET_COVERAGE_PHASE_ALLOCATE | \
   LOOM_TARGET_COVERAGE_PHASE_BINARY_ENCODE |                                  \
   LOOM_TARGET_COVERAGE_PHASE_ARTIFACT_WRAP)

#define LOOM_WASM_EXECUTABLE_LOW_PHASES                               \
  (LOOM_WASM_MODULE_LOW_PHASES | LOOM_TARGET_COVERAGE_PHASE_INSPECT | \
   LOOM_TARGET_COVERAGE_PHASE_RUN)

static const loom_target_coverage_row_t kLoomWasmTargetCoverageRows[] = {
    {
        .target_key = IREE_SVL("wasm-simd128"),
        .descriptor_set_key = IREE_SVL("wasm.core.simd128"),
        .category = IREE_SVL("vector.arithmetic"),
        .semantic_tag = IREE_SVL("simd128.i32-f32"),
        .expected_phases = LOOM_WASM_EXECUTABLE_LOW_PHASES,
        .supported_phases = LOOM_WASM_MODULE_LOW_PHASES,
        .gap_key = IREE_SVL("validation-run-adapter-missing"),
    },
    {
        .target_key = IREE_SVL("wasm-simd128"),
        .descriptor_set_key = IREE_SVL("wasm.core.simd128"),
        .category = IREE_SVL("memory.linear"),
        .semantic_tag = IREE_SVL("v128.load-store"),
        .expected_phases = LOOM_WASM_EXECUTABLE_LOW_PHASES,
        .supported_phases = LOOM_WASM_MODULE_LOW_PHASES,
        .gap_key = IREE_SVL("validation-run-adapter-missing"),
    },
    {
        .target_key = IREE_SVL("wasm-simd128"),
        .descriptor_set_key = IREE_SVL("wasm.core.simd128"),
        .category = IREE_SVL("artifact"),
        .semantic_tag = IREE_SVL("wasm-module"),
        .expected_phases = LOOM_TARGET_COVERAGE_PHASE_BINARY_ENCODE |
                           LOOM_TARGET_COVERAGE_PHASE_ARTIFACT_WRAP |
                           LOOM_TARGET_COVERAGE_PHASE_INSPECT |
                           LOOM_TARGET_COVERAGE_PHASE_RUN,
        .supported_phases = LOOM_TARGET_COVERAGE_PHASE_BINARY_ENCODE |
                            LOOM_TARGET_COVERAGE_PHASE_ARTIFACT_WRAP,
        .gap_key = IREE_SVL("validation-run-adapter-missing"),
    },
};

const loom_target_coverage_provider_t loom_wasm_target_coverage_provider = {
    .name = IREE_SVL("wasm"),
    .rows = kLoomWasmTargetCoverageRows,
    .row_count = IREE_ARRAYSIZE(kLoomWasmTargetCoverageRows),
};
