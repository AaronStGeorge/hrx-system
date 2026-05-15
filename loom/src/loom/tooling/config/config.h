// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Config materialization helpers shared by Loom command-line tools and
// programmatic compiler entry points.

#ifndef LOOM_TOOLING_CONFIG_CONFIG_H_
#define LOOM_TOOLING_CONFIG_CONFIG_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// One caller-provided config binding. The key is the symbol name with or
// without the textual '@' sigil. The value is parsed according to the matched
// config.decl/config.def result type.
typedef struct loom_tooling_config_binding_t {
  // Config symbol name, without ownership transfer.
  iree_string_view_t key;
  // Textual config value, without ownership transfer.
  iree_string_view_t value;
} loom_tooling_config_binding_t;

// Materialization policy flags.
enum loom_tooling_config_materialize_flag_bits_e {
  // Reject bindings whose keys do not name config symbols in the module.
  // Programmatic callers compiling many modules with a shared config map can
  // leave this unset so non-sensitive bindings are ignored.
  LOOM_TOOLING_CONFIG_MATERIALIZE_REQUIRE_MATCHES = 1u << 0,
};
typedef uint32_t loom_tooling_config_materialize_flags_t;

// Options for materializing config values into a module.
typedef struct loom_tooling_config_materialize_options_t {
  // Policy flags controlling typo handling and module sensitivity.
  loom_tooling_config_materialize_flags_t flags;
  // Borrowed array of caller-provided bindings.
  const loom_tooling_config_binding_t* bindings;
  // Number of entries in |bindings|.
  iree_host_size_t binding_count;
} loom_tooling_config_materialize_options_t;

// Summary of a materialization run.
typedef struct loom_tooling_config_materialize_result_t {
  // Number of bindings that replaced config symbols with config.def ops.
  iree_host_size_t materialized_count;
  // Number of bindings ignored because the module has no matching config
  // symbol and REQUIRE_MATCHES was unset.
  iree_host_size_t ignored_count;
} loom_tooling_config_materialize_result_t;

// Initializes options to a safe default: no bindings and non-strict matching.
void loom_tooling_config_materialize_options_initialize(
    loom_tooling_config_materialize_options_t* out_options);

// Parses a single `key=value` command-line assignment into borrowed views.
//
// The split happens at the first '='. Both sides are trimmed. A leading '@' on
// the key is accepted and removed so command-line spelling matches IR spelling
// without forcing shell users to quote sigils unnecessarily.
iree_status_t loom_tooling_config_parse_assignment(
    iree_string_view_t assignment, loom_tooling_config_binding_t* out_binding);

// Replaces matching config.decl/config.def symbol ops with config.def ops whose
// initializer attributes are parsed from |options->bindings|. Bindings without
// matching config symbols are ignored unless REQUIRE_MATCHES is set.
//
// This is intentionally a direct module operation rather than a pass. Tooling
// should call it immediately after loading and, when requested, initially
// verifying a module. The resulting IR then behaves exactly like linked or
// user-authored config.def IR for verifiers, dependency analysis, and passes.
iree_status_t loom_tooling_config_materialize_module(
    loom_module_t* module,
    const loom_tooling_config_materialize_options_t* options,
    iree_arena_block_pool_t* block_pool,
    loom_tooling_config_materialize_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_CONFIG_CONFIG_H_
