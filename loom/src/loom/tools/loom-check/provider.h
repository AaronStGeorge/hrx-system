// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-provider composition for loom-check binaries.
//
// Target packages expose loom_check_provider_t records and the loom-check
// binary links the selected provider set. The provider layer assembles those
// records into the loom_check_environment_t consumed by the shared CLI runner.

#ifndef LOOM_TOOLS_LOOM_CHECK_PROVIDER_H_
#define LOOM_TOOLS_LOOM_CHECK_PROVIDER_H_

#include "iree/base/api.h"
#include "loom/ir/context.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/target/low_legality.h"
#include "loom/target/low_packet_diagnostics.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_lower_policy_registry_t
    loom_low_lower_policy_registry_t;
typedef struct loom_check_emit_provider_t loom_check_emit_provider_t;
typedef struct loom_check_requirement_provider_t
    loom_check_requirement_provider_t;

// Registers target-owned dialects contributed by a provider.
typedef iree_status_t (*loom_check_provider_context_registration_fn_t)(
    loom_context_t* context);

// Initializes a target-low descriptor registry package. Registry tables are
// linked into the runner binary and do not allocate.
typedef void (*loom_check_low_descriptor_registry_initializer_t)(
    loom_target_low_descriptor_registry_t* out_registry);

// Initializes a source-to-target-low lowering policy registry package. Registry
// tables are linked into the runner binary and do not allocate.
typedef void (*loom_check_low_lower_policy_registry_initializer_t)(
    loom_low_lower_policy_registry_t* out_registry);

// Target-owned contribution linked into a loom-check environment.
typedef struct loom_check_provider_t {
  // Stable provider name used in diagnostics and help text.
  iree_string_view_t name;
  // Optional function that registers target-owned dialects with the
  // loom-check context.
  loom_check_provider_context_registration_fn_t register_context;
  // Optional function that initializes a target-low descriptor registry
  // package.
  loom_check_low_descriptor_registry_initializer_t
      initialize_low_descriptor_registry;
  // Optional function that initializes a source-to-low lowering policy package.
  loom_check_low_lower_policy_registry_initializer_t
      initialize_low_lower_policy_registry;
  // Optional target-low source legality providers contributed by this provider.
  loom_target_low_legality_provider_list_t low_legality_provider_list;
  // Optional target-low packet diagnostic providers contributed by this
  // provider.
  loom_target_low_packet_diagnostic_provider_list_t
      low_packet_diagnostic_provider_list;
  // Optional emit provider table contributed by this provider.
  const loom_check_emit_provider_t* const* emit_providers;
  // Number of entries in |emit_providers|.
  iree_host_size_t emit_provider_count;
  // Optional requirement provider table contributed by this provider.
  const loom_check_requirement_provider_t* const* requirement_providers;
  // Number of entries in |requirement_providers|.
  iree_host_size_t requirement_provider_count;
} loom_check_provider_t;

// Static provider table linked into a loom-check binary or embedding.
typedef struct loom_check_provider_set_t {
  // Provider contribution table.
  const loom_check_provider_t* const* providers;
  // Number of entries in |providers|.
  iree_host_size_t provider_count;
} loom_check_provider_set_t;

// Runs loom-check using production dialects plus |provider_set|'s target
// contributions.
int loom_check_provider_main(int argc, char** argv,
                             const loom_check_provider_set_t* provider_set);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_PROVIDER_H_
