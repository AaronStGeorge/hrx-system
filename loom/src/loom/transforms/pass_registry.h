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

// Static requirement declared by a pass descriptor.
typedef struct loom_pass_requirement_def_t {
  // Stable requirement key.
  iree_string_view_t key;
  // Human-readable description of the requirement.
  iree_string_view_t description;
} loom_pass_requirement_def_t;

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
  // Requirement metadata needed before the pass can run.
  const loom_pass_requirement_def_t* requirement_defs;
  // Number of entries in |requirement_defs|.
  uint16_t requirement_count;
} loom_pass_descriptor_t;

// Decoded value for one pass option schema entry.
typedef struct loom_pass_decoded_option_t {
  // Option schema describing this value.
  const loom_pass_option_schema_t* schema;
  // True when the option was present in the source option dictionary.
  bool present;
  union {
    // String value when schema->kind is LOOM_PASS_OPTION_SCHEMA_STRING.
    iree_string_view_t string_value;
    // Integer value when schema->kind is LOOM_PASS_OPTION_SCHEMA_UINT32.
    uint32_t uint32_value;
    // Index into schema->enum_values when kind is LOOM_PASS_OPTION_SCHEMA_ENUM.
    uint16_t enum_value_index;
  };
} loom_pass_decoded_option_t;

// Immutable decoded option set for one pass descriptor.
typedef struct loom_pass_decoded_options_t {
  // Descriptor that owns the option schema.
  const loom_pass_descriptor_t* descriptor;
  // Arena-owned decoded options indexed by descriptor option schema order.
  const loom_pass_decoded_option_t* options;
  // Number of entries in |options|.
  uint16_t option_count;
} loom_pass_decoded_options_t;

// A sorted registry of pass descriptors.
typedef struct loom_pass_registry_t {
  // Descriptors sorted by canonical key.
  const loom_pass_descriptor_t* descriptors;
  // Number of descriptors in |descriptors|.
  iree_host_size_t descriptor_count;
} loom_pass_registry_t;

// Resolved descriptor entry from a textual pass pipeline.
typedef struct loom_pass_pipeline_descriptor_entry_t {
  // Parsed textual pipeline entry.
  loom_pass_pipeline_entry_spec_t spec;
  // Resolved pass descriptor.
  const loom_pass_descriptor_t* descriptor;
  // Zero-based index in the textual pipeline.
  iree_host_size_t pipeline_index;
} loom_pass_pipeline_descriptor_entry_t;

// Provides optional per-entry user data while loading a textual pipeline.
typedef iree_status_t (*loom_pass_pipeline_configure_fn_t)(
    void* user_data, const loom_pass_pipeline_descriptor_entry_t* entry,
    void** out_pass_user_data);

// Callback invoked for each resolved pipeline entry while loading a textual
// pass pipeline.
typedef struct loom_pass_pipeline_configure_callback_t {
  // Optional configure function.
  loom_pass_pipeline_configure_fn_t fn;
  // Opaque caller data passed to |fn|.
  void* user_data;
} loom_pass_pipeline_configure_callback_t;

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

// Decodes textual option assignments against |descriptor|'s schema into
// immutable arena-owned storage. String values reference the original option
// text and must not outlive it.
iree_status_t loom_pass_descriptor_decode_options(
    const loom_pass_descriptor_t* descriptor, iree_string_view_t options,
    iree_arena_allocator_t* arena, loom_pass_decoded_options_t* out_options);

// Adds |descriptor| to |manager| with caller-owned option text and borrowed
// per-entry user data.
iree_status_t loom_pass_manager_add_descriptor(
    loom_pass_manager_t* manager, const loom_pass_descriptor_t* descriptor,
    iree_string_view_t options, void* user_data);

// Parses |pipeline|, resolves descriptors through |registry|, and appends the
// resulting pass entries to |manager|. |configure| may attach per-entry user
// data after descriptor lookup and before validation/addition.
iree_status_t loom_pass_manager_add_pipeline(
    loom_pass_manager_t* manager, const loom_pass_registry_t* registry,
    iree_string_view_t pipeline,
    loom_pass_pipeline_configure_callback_t configure);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_PASS_REGISTRY_H_
