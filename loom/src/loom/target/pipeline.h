// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target lowering pass-pipeline construction.
//
// These helpers build ordinary pass IR for target lowering. They do not run
// pass programs, select artifacts, packetize low functions, or emit object
// bytes. A compile driver owns execution of the resulting pass.pipeline.

#ifndef LOOM_TARGET_PIPELINE_H_
#define LOOM_TARGET_PIPELINE_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/pass/environment.h"
#include "loom/target/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_pipeline_options_t {
  // Maximum source-to-low diagnostics. Zero uses source-to-low's default.
  uint32_t source_to_low_max_errors;
} loom_target_pipeline_options_t;

// Builds a module-root pipeline that lowers source/kernel IR to prepared
// target-low IR.
//
// The produced pipeline is target-neutral at the driver level:
// - source/kernel normalization and cleanup run on functions;
// - source-to-low runs as the shared module pass;
// - target providers contribute materialization/preparation pass IR;
// - common low packetization preparation runs on functions.
//
// |pass_environment| must contain the capabilities required by the produced
// passes when the pipeline is later verified or executed.
iree_status_t loom_target_pipeline_build_to_prepared_low(
    loom_module_t* pipeline_module, iree_string_view_t name,
    const loom_target_pipeline_options_t* options,
    const loom_target_environment_t* target_environment,
    loom_pass_environment_t pass_environment, loom_op_t** out_pipeline_op);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_PIPELINE_H_
