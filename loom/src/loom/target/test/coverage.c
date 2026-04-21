// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/test/coverage.h"

static const loom_target_coverage_row_t kLoomTestTargetCoverageRows[] = {
    {
        .target_key = IREE_SVL("test-low"),
        .descriptor_set_key = IREE_SVL("test.low.core"),
        .category = IREE_SVL("vector.arithmetic"),
        .semantic_tag = IREE_SVL("i32.add"),
        .expected_phases = LOOM_TARGET_COVERAGE_PHASE_DESCRIPTOR |
                           LOOM_TARGET_COVERAGE_PHASE_LOW_PARSE |
                           LOOM_TARGET_COVERAGE_PHASE_SCHEDULE |
                           LOOM_TARGET_COVERAGE_PHASE_ALLOCATE |
                           LOOM_TARGET_COVERAGE_PHASE_TEXT_EMIT,
        .supported_phases = LOOM_TARGET_COVERAGE_PHASE_DESCRIPTOR |
                            LOOM_TARGET_COVERAGE_PHASE_LOW_PARSE |
                            LOOM_TARGET_COVERAGE_PHASE_SCHEDULE |
                            LOOM_TARGET_COVERAGE_PHASE_ALLOCATE |
                            LOOM_TARGET_COVERAGE_PHASE_TEXT_EMIT,
    },
    {
        .target_key = IREE_SVL("test-low"),
        .descriptor_set_key = IREE_SVL("test.low.core"),
        .category = IREE_SVL("artifact"),
        .semantic_tag = IREE_SVL("elf.object"),
        .expected_phases = LOOM_TARGET_COVERAGE_PHASE_BINARY_ENCODE |
                           LOOM_TARGET_COVERAGE_PHASE_ARTIFACT_WRAP |
                           LOOM_TARGET_COVERAGE_PHASE_RUN,
        .supported_phases = LOOM_TARGET_COVERAGE_PHASE_NONE,
        .gap_key = IREE_SVL("test-target-only"),
    },
};

const loom_target_coverage_provider_t loom_test_target_coverage_provider = {
    .name = IREE_SVL("test"),
    .rows = kLoomTestTargetCoverageRows,
    .row_count = IREE_ARRAYSIZE(kLoomTestTargetCoverageRows),
};
