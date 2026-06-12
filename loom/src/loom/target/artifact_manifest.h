// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-neutral manifest model for emitted Loom artifacts.

#ifndef LOOM_TARGET_ARTIFACT_MANIFEST_H_
#define LOOM_TARGET_ARTIFACT_MANIFEST_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// Manifest schema version emitted by loom_target_artifact_manifest_format_json.
#define LOOM_TARGET_ARTIFACT_MANIFEST_SCHEMA_VERSION 1u

typedef enum loom_target_artifact_manifest_mode_e {
  // Does not format an artifact manifest.
  LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE = 0,
  // Formats stable artifact, target, function, global, and summary ABI facts.
  LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY = 1,
  // Adds detail arrays such as function parameters and global references.
  LOOM_TARGET_ARTIFACT_MANIFEST_MODE_DETAILS = 2,
  // Requests analysis-capable formatting; currently includes detail arrays.
  LOOM_TARGET_ARTIFACT_MANIFEST_MODE_ANALYSIS = 3,
} loom_target_artifact_manifest_mode_t;

typedef struct loom_target_artifact_manifest_format_options_t {
  // Selected manifest detail mode.
  loom_target_artifact_manifest_mode_t mode;
} loom_target_artifact_manifest_format_options_t;

// Initializes formatting options with manifest output disabled.
void loom_target_artifact_manifest_format_options_initialize(
    loom_target_artifact_manifest_format_options_t* out_options);

// Parses "", "none", "summary", "details", or "analysis" into a mode.
iree_status_t loom_target_artifact_manifest_mode_parse(
    iree_string_view_t value, loom_target_artifact_manifest_mode_t* out_mode);

// Returns the stable JSON spelling for |mode|.
iree_string_view_t loom_target_artifact_manifest_mode_name(
    loom_target_artifact_manifest_mode_t mode);

typedef uint32_t loom_target_artifact_manifest_artifact_flags_t;

enum loom_target_artifact_manifest_artifact_flag_bits_e {
  // Indicates that byte_length carries the emitted artifact byte length.
  LOOM_TARGET_ARTIFACT_MANIFEST_ARTIFACT_FLAG_BYTE_LENGTH = 1u << 0,
};

typedef struct loom_target_artifact_manifest_artifact_t {
  // Stable artifact format name, such as "elf" or "spirv-binary".
  iree_string_view_t format;
  // Optional artifact name supplied by the emitter or packager.
  iree_string_view_t name;
  // Bitfield selecting optional artifact facts that are present.
  loom_target_artifact_manifest_artifact_flags_t flags;
  // Emitted artifact byte length when BYTE_LENGTH is set.
  uint64_t byte_length;
} loom_target_artifact_manifest_artifact_t;

typedef struct loom_target_artifact_manifest_target_t {
  // Manifest-local target name referenced by function/global target arrays.
  iree_string_view_t name;
  // Optional target family, such as "amdgpu", "spirv", "wasm", or "x86".
  iree_string_view_t family;
  // Optional processor or architecture spelling, such as "gfx1100".
  iree_string_view_t processor;
  // Optional target triple spelling.
  iree_string_view_t triple;
  // Optional target profile spelling.
  iree_string_view_t profile;
  // Optional code-object target spelling.
  iree_string_view_t code_object_target;
  // Optional target feature names.
  const iree_string_view_t* feature_names;
  // Number of target feature names.
  iree_host_size_t feature_name_count;
} loom_target_artifact_manifest_target_t;

typedef enum loom_target_artifact_manifest_parameter_kind_e {
  // Parameter kind was not classified by the collector.
  LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_KIND_UNKNOWN = 0,
  // Plain by-value scalar or aggregate parameter.
  LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_KIND_VALUE = 1,
  // Pointer-like parameter whose address space is represented by the target.
  LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_KIND_POINTER = 2,
  // Resource binding parameter.
  LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_KIND_BINDING = 3,
  // Constant or specialization parameter.
  LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_KIND_CONSTANT = 4,
} loom_target_artifact_manifest_parameter_kind_t;

typedef uint32_t loom_target_artifact_manifest_parameter_flags_t;

enum loom_target_artifact_manifest_parameter_flag_bits_e {
  // Indicates that index carries the parameter index.
  LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_INDEX = 1u << 0,
  // Indicates that byte_offset carries the ABI byte offset.
  LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_BYTE_OFFSET = 1u << 1,
  // Indicates that byte_length carries the ABI byte length.
  LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_BYTE_LENGTH = 1u << 2,
  // Indicates that byte_alignment carries the ABI byte alignment.
  LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_BYTE_ALIGNMENT = 1u << 3,
};

typedef struct loom_target_artifact_manifest_parameter_t {
  // Optional source or ABI parameter name.
  iree_string_view_t name;
  // Stable parameter category.
  loom_target_artifact_manifest_parameter_kind_t kind;
  // Optional type spelling, when the collector has a stable source type.
  iree_string_view_t type;
  // Bitfield selecting optional parameter facts that are present.
  loom_target_artifact_manifest_parameter_flags_t flags;
  // Parameter index when INDEX is set.
  uint32_t index;
  // ABI byte offset when BYTE_OFFSET is set.
  uint64_t byte_offset;
  // ABI byte length when BYTE_LENGTH is set.
  uint64_t byte_length;
  // ABI byte alignment when BYTE_ALIGNMENT is set.
  uint64_t byte_alignment;
} loom_target_artifact_manifest_parameter_t;

typedef uint32_t loom_target_artifact_manifest_interface_flags_t;

enum loom_target_artifact_manifest_interface_flag_bits_e {
  // Indicates that parameter_count carries the logical parameter count.
  LOOM_TARGET_ARTIFACT_MANIFEST_INTERFACE_FLAG_PARAMETER_COUNT = 1u << 0,
  // Indicates that binding_count carries the resource binding count.
  LOOM_TARGET_ARTIFACT_MANIFEST_INTERFACE_FLAG_BINDING_COUNT = 1u << 1,
  // Indicates that constant_byte_length carries packed constant bytes.
  LOOM_TARGET_ARTIFACT_MANIFEST_INTERFACE_FLAG_CONSTANT_BYTE_LENGTH = 1u << 2,
};

typedef struct loom_target_artifact_manifest_interface_t {
  // Bitfield selecting optional interface facts that are present.
  loom_target_artifact_manifest_interface_flags_t flags;
  // Logical parameter count when PARAMETER_COUNT is set.
  uint32_t parameter_count;
  // Resource binding count when BINDING_COUNT is set.
  uint32_t binding_count;
  // Packed constant byte length when CONSTANT_BYTE_LENGTH is set.
  uint64_t constant_byte_length;
  // Optional parameter detail rows emitted in details and analysis modes.
  const loom_target_artifact_manifest_parameter_t* parameters;
  // Number of parameter detail rows.
  iree_host_size_t parameter_detail_count;
} loom_target_artifact_manifest_interface_t;

typedef uint32_t loom_target_artifact_manifest_execution_flags_t;

enum loom_target_artifact_manifest_execution_flag_bits_e {
  // Indicates that workgroup_size carries a static x/y/z workgroup size.
  LOOM_TARGET_ARTIFACT_MANIFEST_EXECUTION_FLAG_WORKGROUP_SIZE = 1u << 0,
  // Indicates that subgroup_size carries a static subgroup or wavefront size.
  LOOM_TARGET_ARTIFACT_MANIFEST_EXECUTION_FLAG_SUBGROUP_SIZE = 1u << 1,
};

typedef struct loom_target_artifact_manifest_execution_t {
  // Bitfield selecting optional execution facts that are present.
  loom_target_artifact_manifest_execution_flags_t flags;
  // Static workgroup size when WORKGROUP_SIZE is set.
  uint32_t workgroup_size[3];
  // Static subgroup or wavefront size when SUBGROUP_SIZE is set.
  uint32_t subgroup_size;
} loom_target_artifact_manifest_execution_t;

typedef struct loom_target_artifact_manifest_function_t {
  // Artifact function name used by loaders and direct JIT hosts.
  iree_string_view_t name;
  // Optional authored source name, omitted when it matches name.
  iree_string_view_t source_name;
  // Target names that this function was compiled for.
  const iree_string_view_t* target_names;
  // Number of target name references.
  iree_host_size_t target_name_count;
  // Function interface facts.
  loom_target_artifact_manifest_interface_t interface;
  // Function execution facts.
  loom_target_artifact_manifest_execution_t execution;
  // Global names referenced by this function.
  const iree_string_view_t* used_global_names;
  // Number of global name references.
  iree_host_size_t used_global_name_count;
} loom_target_artifact_manifest_function_t;

typedef uint32_t loom_target_artifact_manifest_global_flags_t;

enum loom_target_artifact_manifest_global_flag_bits_e {
  // Indicates that byte_length carries the global storage byte length.
  LOOM_TARGET_ARTIFACT_MANIFEST_GLOBAL_FLAG_BYTE_LENGTH = 1u << 0,
  // Indicates that byte_alignment carries the global storage byte alignment.
  LOOM_TARGET_ARTIFACT_MANIFEST_GLOBAL_FLAG_BYTE_ALIGNMENT = 1u << 1,
};

typedef struct loom_target_artifact_manifest_global_t {
  // Artifact global name used by loaders and direct JIT hosts.
  iree_string_view_t name;
  // Optional authored source name, omitted when it matches name.
  iree_string_view_t source_name;
  // Optional type or storage class spelling.
  iree_string_view_t type;
  // Target names that this global was compiled for.
  const iree_string_view_t* target_names;
  // Number of target name references.
  iree_host_size_t target_name_count;
  // Bitfield selecting optional global facts that are present.
  loom_target_artifact_manifest_global_flags_t flags;
  // Global storage byte length when BYTE_LENGTH is set.
  uint64_t byte_length;
  // Global storage byte alignment when BYTE_ALIGNMENT is set.
  uint64_t byte_alignment;
} loom_target_artifact_manifest_global_t;

typedef struct loom_target_artifact_manifest_t {
  // Emitted artifact identity and size facts.
  loom_target_artifact_manifest_artifact_t artifact;
  // Manifest-local target records.
  const loom_target_artifact_manifest_target_t* targets;
  // Number of manifest-local target records.
  iree_host_size_t target_count;
  // Artifact function records.
  const loom_target_artifact_manifest_function_t* functions;
  // Number of artifact function records.
  iree_host_size_t function_count;
  // Artifact global records.
  const loom_target_artifact_manifest_global_t* globals;
  // Number of artifact global records.
  iree_host_size_t global_count;
} loom_target_artifact_manifest_t;

// Formats |manifest| as one structured JSON object into |stream|.
//
// NONE mode writes nothing. SUMMARY mode writes target-neutral artifact,
// function, global, interface, and execution facts. DETAILS and ANALYSIS modes
// additionally write detail arrays currently present in the model. Analysis
// mode is accepted now so API/flag surfaces can share one spelling while report
// providers remain separately gated and disabled by default.
iree_status_t loom_target_artifact_manifest_format_json(
    const loom_target_artifact_manifest_t* manifest,
    const loom_target_artifact_manifest_format_options_t* options,
    loom_output_stream_t* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARTIFACT_MANIFEST_H_
