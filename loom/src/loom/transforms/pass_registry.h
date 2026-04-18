// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Pass descriptor registry.
//
// The registry is the shared dispatch surface between pass-pipeline tools and
// concrete pass implementations. The pass manager remains a small execution
// engine; registries decide which pass descriptors are linked into a given tool
// or compiler build.

#ifndef LOOM_TRANSFORMS_PASS_REGISTRY_H_
#define LOOM_TRANSFORMS_PASS_REGISTRY_H_

#include "iree/base/api.h"
#include "loom/transforms/pass.h"

#ifdef __cplusplus
extern "C" {
#endif

// Bitset of loom_pass_descriptor_flag_bits_e values.
typedef uint32_t loom_pass_descriptor_flags_t;

// Bitset of loom_pass_option_schema_flag_bits_e values.
typedef uint32_t loom_pass_option_schema_flags_t;

// Returns static pass metadata for one descriptor.
typedef const loom_pass_info_t* (*loom_pass_info_fn_t)(void);

enum loom_pass_descriptor_flag_bits_e {
  // Descriptor is known to the registry but unavailable in this build/config.
  LOOM_PASS_DESCRIPTOR_UNAVAILABLE = 1u << 0,
};

enum loom_pass_option_schema_flag_bits_e {
  // Option must be present in the pass option dictionary.
  LOOM_PASS_OPTION_SCHEMA_REQUIRED = 1u << 0,
};

// Kind-specific validation applied to one textual pass option.
typedef enum loom_pass_option_schema_kind_e {
  // Opaque non-empty string validated by the pass create callback.
  LOOM_PASS_OPTION_SCHEMA_STRING = 0,
  // Unsigned 32-bit integer validated against an inclusive range.
  LOOM_PASS_OPTION_SCHEMA_UINT32 = 1,
  // String selected from a descriptor-owned value table.
  LOOM_PASS_OPTION_SCHEMA_ENUM = 2,
} loom_pass_option_schema_kind_t;

// One allowed value for an enum pass option.
typedef struct loom_pass_option_enum_value_t {
  // Accepted textual value.
  iree_string_view_t value;
} loom_pass_option_enum_value_t;

// Typed schema for one named pass option.
typedef struct loom_pass_option_schema_t {
  // Option name matching a loom_pass_info_t option_defs entry.
  iree_string_view_t name;
  // Option value kind.
  loom_pass_option_schema_kind_t kind;
  // Option presence and validation flags.
  loom_pass_option_schema_flags_t flags;
  // Inclusive lower bound when |kind| is LOOM_PASS_OPTION_SCHEMA_UINT32.
  uint32_t minimum_uint32;
  // Inclusive upper bound when |kind| is LOOM_PASS_OPTION_SCHEMA_UINT32.
  uint32_t maximum_uint32;
  // Sorted allowed values when |kind| is LOOM_PASS_OPTION_SCHEMA_ENUM.
  const loom_pass_option_enum_value_t* enum_values;
  // Number of entries in |enum_values|.
  uint16_t enum_value_count;
} loom_pass_option_schema_t;

// Static descriptor for one pass implementation.
typedef struct loom_pass_descriptor_t {
  // Canonical pass key used by textual pipelines and future pass.run ops.
  iree_string_view_t key;
  // Returns static pass metadata.
  loom_pass_info_fn_t info;
  union {
    // Module-pass callback when info->kind is LOOM_PASS_MODULE.
    loom_module_pass_fn_t module_run;
    // Function-pass callback when info->kind is LOOM_PASS_FUNCTION.
    loom_function_pass_fn_t function_run;
  };
  // Optional pass instance creation callback.
  loom_pass_create_fn_t create;
  // Optional pass instance destruction callback.
  loom_pass_destroy_fn_t destroy;
  // Descriptor availability flags.
  loom_pass_descriptor_flags_t flags;
  // Explanation used when the descriptor is unavailable.
  iree_string_view_t unavailable_reason;
  // Typed option schemas matching the pass info option definitions.
  const loom_pass_option_schema_t* option_schema;
  // Number of entries in |option_schema|.
  uint16_t option_schema_count;
} loom_pass_descriptor_t;

// A sorted registry of pass descriptors.
typedef struct loom_pass_registry_t {
  // Descriptors sorted by canonical key.
  const loom_pass_descriptor_t* descriptors;
  // Number of descriptors in |descriptors|.
  iree_host_size_t descriptor_count;
} loom_pass_registry_t;

// Returns the builtin registry containing the pass implementations linked into
// the standard Loom compiler/tool build.
const loom_pass_registry_t* loom_pass_builtin_registry(void);

// Verifies registry ordering, key uniqueness, and descriptor shape.
iree_status_t loom_pass_registry_verify(const loom_pass_registry_t* registry);

// Looks up |key| by exact canonical spelling. Returns OK with
// |*out_descriptor| NULL when the key is unknown.
iree_status_t loom_pass_registry_lookup(
    const loom_pass_registry_t* registry, iree_string_view_t key,
    const loom_pass_descriptor_t** out_descriptor);

// Returns true when |descriptor| can be instantiated in the current build.
bool loom_pass_descriptor_is_available(
    const loom_pass_descriptor_t* descriptor);

// Validates textual option assignments against |descriptor|'s schema.
iree_status_t loom_pass_descriptor_validate_options(
    const loom_pass_descriptor_t* descriptor, iree_string_view_t options);

// Adds |descriptor| to |manager| with caller-owned option text and borrowed
// per-entry user data.
iree_status_t loom_pass_manager_add_descriptor(
    loom_pass_manager_t* manager, const loom_pass_descriptor_t* descriptor,
    iree_string_view_t options, void* user_data);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_PASS_REGISTRY_H_
