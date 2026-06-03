// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_MODULE_H_
#define LOOMC_MODULE_H_

#include <stdio.h>

#include "loomc/context.h"
#include "loomc/result.h"
#include "loomc/source.h"

/// @file
/// Opaque in-memory Loom modules.
///
/// Modules are the typed handoff between parsing, linking, optimization, and
/// compilation. They let embedders compose API operations without printing and
/// reparsing `.loom` text on the hot path. Serialization is the boundary that
/// turns a module back into text or bytecode bytes for storage, display, cache
/// keys, or handoff to systems that consume sources. Deserialization is the
/// inverse boundary that turns text or bytecode bytes into an owned mutable
/// module handle.
///
/// @par Example
/// Serialize a linked module back to an immutable source handle:
///
/// @code{.c}
/// loomc_module_serialize_options_t options = {
///     .type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
///     .structure_size = sizeof(loomc_module_serialize_options_t),
///     .format = LOOMC_SOURCE_FORMAT_BYTECODE,
///     .identifier = loomc_make_cstring_view("linked.loombc"),
/// };
/// loomc_source_t* serialized = NULL;
/// loomc_status_t status = loomc_module_serialize_to_source(
///     module, &options, loomc_allocator_system(), &serialized);
/// if (!loomc_status_is_ok(status)) {
///   return status;
/// }
///
/// // Reuse serialized as ordinary source input, cache it, or inspect its
/// bytes.
///
/// loomc_source_release(serialized);
/// @endcode
///
/// @par Example
/// Deserialize a source handle into a module and inspect the result:
///
/// @code{.c}
/// loomc_module_t* module = NULL;
/// loomc_result_t* result = NULL;
/// loomc_status_t status = loomc_module_deserialize_from_source(
///     context, serialized, NULL, loomc_allocator_system(), &module, &result);
/// if (!loomc_status_is_ok(status)) {
///   return status;
/// }
/// if (!loomc_result_succeeded(result)) {
///   // Diagnostics are available through loomc_result_diagnostic_at.
/// }
///
/// loomc_result_release(result);
/// loomc_module_release(module);
/// @endcode

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque parsed, linked, optimized, or lowered module.
///
/// @thread_safety
/// Retain/release operations are safe from multiple threads. Module contents
/// are not internally synchronized. Read-only operations such as serialization
/// may run concurrently with other read-only operations when the caller
/// guarantees that no mutating module operation is active. Mutating operations
/// such as compiling a module require exclusive access to that module handle.
typedef struct loomc_module_t loomc_module_t;

/// Text presentation policy used when serializing `.loom` text.
typedef enum loomc_module_text_presentation_e {
  /// Prefer target-low assembly syntax when the module selects one descriptor
  /// set unambiguously; otherwise print canonical text.
  LOOMC_MODULE_TEXT_PRESENTATION_DEFAULT = 0,

  /// Force canonical text with descriptor-backed target-low operations printed
  /// as ordinary `low.op<...>` operations.
  LOOMC_MODULE_TEXT_PRESENTATION_GENERIC = 1,

  /// Force descriptor-backed target-low assembly syntax. Serialization fails
  /// when no descriptor set is provided or can be selected unambiguously.
  LOOMC_MODULE_TEXT_PRESENTATION_LOW_ASM = 2,
} loomc_module_text_presentation_t;

/// Module serialization options.
///
/// Callers zero-initialize this descriptor, set `type` to
/// `LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS`, set `structure_size` to
/// `sizeof(loomc_module_serialize_options_t)`, and fill the requested fields.
typedef struct loomc_module_serialize_options_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS`
  /// when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Extension chain for future serialization options.
  const void* next;

  /// Output source format. Unknown selects textual `.loom`.
  loomc_source_format_t format;

  /// Identifier to attach to a returned source. Empty uses a format-specific
  /// default. Path and `FILE*` serialization do not interpret this field.
  loomc_string_view_t identifier;

  /// Presentation policy for textual `.loom` output.
  loomc_module_text_presentation_t text_presentation;

  /// Target-low descriptor-set key used by low assembly presentation.
  ///
  /// Empty lets the serializer infer a single descriptor set from target-low
  /// functions in the module. Non-empty requests low assembly presentation
  /// unless `text_presentation` is `LOOMC_MODULE_TEXT_PRESENTATION_GENERIC`,
  /// which is rejected as contradictory.
  loomc_string_view_t low_asm_descriptor_set_key;
} loomc_module_serialize_options_t;

/// Module deserialization options.
///
/// Callers zero-initialize this descriptor, set `type` to
/// `LOOMC_STRUCTURE_TYPE_MODULE_DESERIALIZE_OPTIONS`, set `structure_size` to
/// `sizeof(loomc_module_deserialize_options_t)`, and fill the requested fields.
typedef struct loomc_module_deserialize_options_t {
  /// Structure type. Must be
  /// `LOOMC_STRUCTURE_TYPE_MODULE_DESERIALIZE_OPTIONS` when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Extension chain for future deserialization options.
  const void* next;

  /// Input source format. Unknown uses the source format when available, then
  /// falls back to bytecode magic detection and textual `.loom`.
  loomc_source_format_t format;

  /// Identifier used for diagnostics and module provenance. Empty uses the
  /// input source identifier, filesystem path, or a format-specific default.
  loomc_string_view_t identifier;

  /// Persistent module arena block size in bytes. Zero selects the default.
  loomc_host_size_t block_size;
} loomc_module_deserialize_options_t;

/// Retains an opaque module for another owner.
///
/// @param module Module to retain.
///
/// @thread_safety
/// Retain/release operations are intended to be safe from multiple threads.
LOOMC_API_EXPORT void loomc_module_retain(loomc_module_t* module);

/// Releases an opaque module from one owner.
///
/// @param module Module to release. Passing `NULL` is allowed.
///
/// @thread_safety
/// Retain/release operations are intended to be safe from multiple threads. The
/// module is destroyed when the final reference is released.
LOOMC_API_EXPORT void loomc_module_release(loomc_module_t* module);

/// Serializes a module into an immutable source handle.
///
/// @param module Module to serialize.
/// @param options Serialization options. `NULL` selects textual `.loom`.
/// @param allocator Host allocator used for source-owned storage.
/// @param out_source Receives one retained source on success.
/// @return OK when serialization succeeded.
///
/// @ownership
/// The caller owns the returned source and releases it with
/// `loomc_source_release`. Serialized bytes are owned by that source and remain
/// valid until the source is released.
///
/// @thread_safety
/// Serialization is read-only with respect to `module`. Concurrent
/// serialization of the same module is valid when the caller guarantees that no
/// mutating operation is active.
LOOMC_API_EXPORT loomc_status_t loomc_module_serialize_to_source(
    const loomc_module_t* module,
    const loomc_module_serialize_options_t* options,
    loomc_allocator_t allocator, loomc_source_t** out_source);

/// Serializes a module to an open C `FILE*`.
///
/// @param module Module to serialize.
/// @param options Serialization options. `NULL` selects textual `.loom`.
/// @param file Open writable file handle, such as `stdout`.
/// @return OK when serialization succeeded.
///
/// @ownership
/// The caller retains ownership of `file`; this function does not close it.
LOOMC_API_EXPORT loomc_status_t loomc_module_serialize_to_file(
    const loomc_module_t* module,
    const loomc_module_serialize_options_t* options, FILE* file);

/// Serializes a module to a filesystem path.
///
/// @param module Module to serialize.
/// @param options Serialization options. `NULL` selects textual `.loom`.
/// @param path Output file path. The path is a borrowed byte view and does not
/// need to be NUL-terminated.
/// @param allocator Host allocator used for transient file path/stream storage.
/// @return OK when serialization succeeded.
LOOMC_API_EXPORT loomc_status_t loomc_module_serialize_to_path(
    const loomc_module_t* module,
    const loomc_module_serialize_options_t* options, loomc_string_view_t path,
    loomc_allocator_t allocator);

/// Deserializes an immutable source handle into an owned mutable module.
///
/// @param context Context used to resolve Loom dialect and bytecode metadata.
/// @param source Source bytes to deserialize.
/// @param options Deserialization options. `NULL` infers format from `source`.
/// @param allocator Host allocator used for module-owned storage.
/// @param out_module Receives a retained module when the result succeeds.
/// Receives `NULL` when the result fails.
/// @param out_result Receives a retained result for the operation.
/// @return OK when deserialization ran to a result. Non-OK statuses represent
/// API misuse or infrastructure failures before a result could be produced.
///
/// @ownership
/// The caller owns `out_module` when non-NULL and releases it with
/// `loomc_module_release`. The caller always owns `out_result` on an OK return
/// and releases it with `loomc_result_release`.
///
/// @lifetime
/// Returned modules and results do not borrow from `source`. Diagnostics retain
/// `source` when they reference it. Later module operations may mutate the
/// returned module in place.
LOOMC_API_EXPORT loomc_status_t loomc_module_deserialize_from_source(
    loomc_context_t* context, const loomc_source_t* source,
    const loomc_module_deserialize_options_t* options,
    loomc_allocator_t allocator, loomc_module_t** out_module,
    loomc_result_t** out_result);

/// Deserializes bytes from an open C `FILE*` into a module.
///
/// @param context Context used to resolve Loom dialect and bytecode metadata.
/// @param file Open readable file handle. Bytes are read from the current file
/// position to EOF.
/// @param options Deserialization options. `NULL` infers format from bytes.
/// @param allocator Host allocator used for module-owned storage and temporary
/// input bytes.
/// @param out_module Receives a retained module when the result succeeds.
/// Receives `NULL` when the result fails.
/// @param out_result Receives a retained result for the operation.
/// @return OK when deserialization ran to a result. Non-OK statuses represent
/// API misuse, file read failures, or infrastructure failures before a result
/// could be produced.
///
/// @ownership
/// The caller retains ownership of `file`; this function does not close it.
LOOMC_API_EXPORT loomc_status_t loomc_module_deserialize_from_file(
    loomc_context_t* context, FILE* file,
    const loomc_module_deserialize_options_t* options,
    loomc_allocator_t allocator, loomc_module_t** out_module,
    loomc_result_t** out_result);

/// Deserializes bytes from a filesystem path into a module.
///
/// @param context Context used to resolve Loom dialect and bytecode metadata.
/// @param path Input file path. The path is a borrowed byte view and does not
/// need to be NUL-terminated.
/// @param options Deserialization options. `NULL` infers format from bytes.
/// @param allocator Host allocator used for module-owned storage and temporary
/// input bytes.
/// @param out_module Receives a retained module when the result succeeds.
/// Receives `NULL` when the result fails.
/// @param out_result Receives a retained result for the operation.
/// @return OK when deserialization ran to a result. Non-OK statuses represent
/// API misuse, file read failures, or infrastructure failures before a result
/// could be produced.
LOOMC_API_EXPORT loomc_status_t loomc_module_deserialize_from_path(
    loomc_context_t* context, loomc_string_view_t path,
    const loomc_module_deserialize_options_t* options,
    loomc_allocator_t allocator, loomc_module_t** out_module,
    loomc_result_t** out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_MODULE_H_
