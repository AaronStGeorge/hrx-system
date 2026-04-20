// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compile-report adapters for target-low packetization sidecars.

#ifndef LOOM_TARGET_COMPILE_REPORT_LOW_H_
#define LOOM_TARGET_COMPILE_REPORT_LOW_H_

#include "loom/codegen/low/packetization.h"
#include "loom/target/compile_report.h"

#ifdef __cplusplus
extern "C" {
#endif

// Records schedule, allocation, pressure-row, and spill-row summaries for one
// packetized low function. Optional detail rows are copied only when |report|
// has caller-owned row storage configured.
void loom_target_compile_report_record_low_packetization(
    loom_target_compile_report_t* report,
    const loom_low_packetization_t* packetization);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_COMPILE_REPORT_LOW_H_
