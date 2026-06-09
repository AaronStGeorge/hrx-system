// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compile-report adapters for target-low emission frames.

#ifndef LOOM_TARGET_COMPILE_REPORT_LOW_H_
#define LOOM_TARGET_COMPILE_REPORT_LOW_H_

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/lower/lower.h"
#include "loom/target/compile_report.h"

#ifdef __cplusplus
extern "C" {
#endif

// Records schedule, allocation, pressure-row, and spill-row summaries for one
// packetized low function. Optional detail rows are copied only when |report|
// has caller-owned row storage configured.
iree_status_t loom_target_compile_report_record_low_emission_frame(
    loom_target_compile_report_t* report,
    const loom_low_emission_frame_t* frame);

// Records source-to-low selection and emission summaries for one lowered
// source function. Optional detail rows are copied only when |report| has
// caller-owned source-low row storage configured.
void loom_target_compile_report_record_low_lowering(
    loom_target_compile_report_t* report,
    const loom_low_lower_result_t* lower_result);

// Allocates transient source-to-low row storage sized to the remaining row
// capacity in |report|. Returns empty storage when reports or detail rows are
// disabled. Rows must be recorded into |report| before |arena| is reset.
iree_status_t loom_target_compile_report_allocate_low_lowering_rows(
    const loom_target_compile_report_t* report, iree_arena_allocator_t* arena,
    loom_low_lower_report_storage_t* out_storage);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_COMPILE_REPORT_LOW_H_
