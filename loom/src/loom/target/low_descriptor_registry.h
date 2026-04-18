// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Linked low descriptor registry for selected Loom target packages.
//
// This package is the build-level join point for descriptor sets that a tool or
// compiler binary elects to link. The target-independent codegen/low substrate
// remains free of architecture and emitter dependencies; callers that want the
// bundled registry depend on this library instead of recreating descriptor-set
// arrays locally.

#ifndef LOOM_TARGET_LOW_DESCRIPTOR_REGISTRY_H_
#define LOOM_TARGET_LOW_DESCRIPTOR_REGISTRY_H_

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_TARGET_LOW_DESCRIPTOR_REGISTRY_MAX_DESCRIPTOR_SETS 8

typedef struct loom_target_low_descriptor_registry_t {
  // Borrowed descriptor-set pointers linked into this target registry package.
  const loom_low_descriptor_set_t*
      descriptor_sets[LOOM_TARGET_LOW_DESCRIPTOR_REGISTRY_MAX_DESCRIPTOR_SETS];
  // Registry view over descriptor_sets.
  loom_low_descriptor_registry_t registry;
} loom_target_low_descriptor_registry_t;

// Initializes the descriptor-set registry selected by the current target
// package build.
void loom_target_low_descriptor_registry_initialize(
    loom_target_low_descriptor_registry_t* out_registry);

// Looks up a descriptor set by key in the selected target registry.
iree_status_t loom_target_low_descriptor_set_lookup(
    iree_string_view_t key,
    const loom_low_descriptor_set_t** out_descriptor_set);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_LOW_DESCRIPTOR_REGISTRY_H_
