// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Low descriptor packages linked into loom-check.
//
// The descriptor-local low verifier remains target-package agnostic; this file
// is the loom-check tool's explicit selection of descriptor sets that tests may
// reference.

#ifndef LOOM_TOOLS_LOOM_CHECK_LOW_TARGETS_H_
#define LOOM_TOOLS_LOOM_CHECK_LOW_TARGETS_H_

#include "iree/base/api.h"
#include "loom/codegen/low/target_binding.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_CHECK_LOW_MAX_DESCRIPTOR_SETS 2

typedef struct loom_check_low_descriptor_registry_t {
  // Borrowed descriptor-set pointers linked into the loom-check binary.
  const loom_low_descriptor_set_t*
      descriptor_sets[LOOM_CHECK_LOW_MAX_DESCRIPTOR_SETS];
  // Registry view over descriptor_sets.
  loom_low_descriptor_registry_t registry;
} loom_check_low_descriptor_registry_t;

// Initializes the descriptor-set registry available to loom-check tests.
void loom_check_low_descriptor_registry_initialize(
    loom_check_low_descriptor_registry_t* out_registry);

// Looks up a descriptor set by key in the loom-check registry.
iree_status_t loom_check_low_descriptor_set_lookup(
    iree_string_view_t key,
    const loom_low_descriptor_set_t** out_descriptor_set);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_LOW_TARGETS_H_
