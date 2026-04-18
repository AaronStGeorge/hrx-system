// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target preset registry.
//
// This is the target-neutral lookup layer shared by command-line-style target
// selection and target.preset IR expansion. It deliberately knows only about
// target record payloads, not IR mutation or backend descriptor tables.

#ifndef LOOM_TARGET_PRESET_REGISTRY_H_
#define LOOM_TARGET_PRESET_REGISTRY_H_

#include "iree/base/api.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_preset_registry_t {
  // Borrowed target-bundle preset pointers linked into this registry package.
  const loom_target_bundle_t* const* target_bundles;
  // Number of valid target-bundle preset pointers in |target_bundles|.
  iree_host_size_t target_bundle_count;
} loom_target_preset_registry_t;

// Looks up a target preset bundle by exact key in |registry|. Empty keys are
// rejected because target selection must be explicit and reproducible.
iree_status_t loom_target_preset_registry_lookup_bundle(
    const loom_target_preset_registry_t* registry, iree_string_view_t key,
    const loom_target_bundle_t** out_bundle);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_PRESET_REGISTRY_H_
