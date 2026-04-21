// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Pass wrappers for source-to-target-low lowering.

#ifndef LOOM_CODEGEN_LOW_LOWER_PASS_H_
#define LOOM_CODEGEN_LOW_LOWER_PASS_H_

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/lower.h"
#include "loom/codegen/low/pass_requirements.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

const loom_pass_info_t* loom_low_source_to_low_pass_info(void);

typedef struct loom_low_source_to_low_pass_config_t {
  // Selected target-low descriptor registry used to verify descriptor choices.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Selected source-to-target-low lowering policy registry.
  const loom_low_lower_policy_registry_t* policy_registry;
  // Optional target-specific legality providers forwarded to source legality.
  loom_target_low_legality_provider_list_t legality_provider_list;
} loom_low_source_to_low_pass_config_t;

// Returns true if |config| satisfies a requirement declared by the
// source-to-low pass descriptor.
bool loom_low_source_to_low_pass_config_satisfies_requirement(
    const loom_low_source_to_low_pass_config_t* config,
    iree_string_view_t requirement);

iree_status_t loom_low_source_to_low_create(loom_pass_t* pass,
                                            iree_string_view_t options);

iree_status_t loom_low_source_to_low_run(loom_pass_t* pass,
                                         loom_module_t* module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_PASS_H_
