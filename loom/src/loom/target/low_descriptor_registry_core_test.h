// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Explicit core test-low registry package.
//
// This package intentionally links only the synthetic test-low descriptor set
// and preset bundle. Core compiler tests and developer tools use it when they
// need a tiny deterministic target package instead of real backend tables.
// Production tools that need real backend descriptor sets should use
// backend-specific registry packages instead.

#ifndef LOOM_TARGET_LOW_DESCRIPTOR_REGISTRY_CORE_TEST_H_
#define LOOM_TARGET_LOW_DESCRIPTOR_REGISTRY_CORE_TEST_H_

#include "loom/target/low_descriptor_registry.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initializes the core test-low descriptor-set registry.
void loom_target_core_test_low_descriptor_registry_initialize(
    loom_target_low_descriptor_registry_t* out_registry);

// Looks up a descriptor set by key in the core test-low registry, or returns
// NULL when no descriptor set matches.
const loom_low_descriptor_set_t*
loom_target_core_test_low_descriptor_set_lookup(iree_string_view_t key);

// Looks up a target-low preset bundle by key in the core test-low registry.
iree_status_t loom_target_core_test_low_bundle_lookup(
    iree_string_view_t key, const loom_target_bundle_t** out_bundle);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_LOW_DESCRIPTOR_REGISTRY_CORE_TEST_H_
