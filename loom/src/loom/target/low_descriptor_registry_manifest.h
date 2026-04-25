// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_LOW_DESCRIPTOR_REGISTRY_MANIFEST_H_
#define LOOM_TARGET_LOW_DESCRIPTOR_REGISTRY_MANIFEST_H_

#include "loom/target/low_descriptor_registry.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Appends a compact JSON manifest for the linked target-low registry package to
// |builder|. The manifest is a diagnostic/tooling format that lists linked
// descriptor-set summaries and target bundle selections without embedding full
// instruction descriptor rows.
iree_status_t loom_target_low_descriptor_registry_format_manifest_json(
    const loom_target_low_descriptor_registry_t* registry,
    loom_low_descriptor_requirement_flags_t requirements,
    iree_string_builder_t* builder);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // LOOM_TARGET_LOW_DESCRIPTOR_REGISTRY_MANIFEST_H_
