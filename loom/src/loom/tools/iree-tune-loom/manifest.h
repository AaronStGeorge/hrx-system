// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Artifact bundle manifest emission for iree-tune-loom.

#ifndef LOOM_TOOLS_IREE_TUNE_LOOM_MANIFEST_H_
#define LOOM_TOOLS_IREE_TUNE_LOOM_MANIFEST_H_

#include "iree/base/api.h"
#include "loom/tools/iree-tune-loom/model.h"

#ifdef __cplusplus
extern "C" {
#endif

// Appends the process command line as a JSON array.
iree_status_t iree_tune_loom_append_command_line_json(
    int argc, char** argv, iree_string_builder_t* output);

// Writes manifest.json for an enabled artifact bundle.
iree_status_t iree_tune_loom_write_artifact_bundle_manifest(
    const iree_tune_loom_artifact_bundle_t* bundle,
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_hal_context_t* hal_context, iree_string_view_t source,
    iree_string_view_t command_line_json, bool dry_run,
    iree_tune_loom_shape_specialization_mode_t shape_specialization_mode,
    iree_allocator_t allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_TUNE_LOOM_MANIFEST_H_
